//! Streaming GGUF → bgz17 indexer.
//!
//! Reads a GGUF model file tensor-by-tensor (seek, not load-all),
//! projects each weight matrix to Base17 via golden-step averaging,
//! writes compressed output. Peak RAM = one tensor + pipeline buffers.
//!
//! ```text
//! GGUF file (GB)
//!   → read header (tensor directory, offsets)
//!   → for each tensor:
//!       seek to offset → dequant to f32 slice
//!       classify layer type (Attention/FFN/Conv2D/Norm)
//!       reshape: rows × cols (Attention/FFN) or filters × kernel_dim (Conv2D)
//!       golden-step project each row → Base17 (34 bytes)
//!       write CompressedTensor { name, shape, base17_rows }
//!       drop f32 slice (RAM freed)
//! ```
//!
//! Supports: F32, F16, BF16, Q8_0, Q4_0, Q4_K (via gguf.rs dequant).

use super::bgz17_bridge::Base17;
use super::gguf::{self, GgufFile, TensorInfo, GgmlType};
use std::io::{Read, Seek, Write};

// ============================================================================
// Layer classification
// ============================================================================

/// What kind of layer a tensor belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Attention Q/K/V/O projection: [hidden, hidden] or [hidden, head_dim].
    Attention,
    /// Feed-forward: [hidden, intermediate] or [intermediate, hidden].
    FeedForward,
    /// Conv2D kernel: [out_ch, in_ch, kH, kW] → treat as [out_ch, in_ch*kH*kW].
    Conv2D,
    /// Layer/Group/RMS norm: small, keep as-is (not worth compressing).
    Norm,
    /// Embedding table: [vocab, hidden].
    Embedding,
    /// Unknown or too small to bother.
    Skip,
}

/// Classify a tensor by its name (llama.cpp / HuggingFace naming conventions).
pub fn classify_tensor(name: &str, dims: &[u64]) -> LayerType {
    let ndim = dims.len();
    let total: u64 = dims.iter().product();

    // Skip tiny tensors (norms, biases)
    if total < 1024 {
        return LayerType::Skip;
    }

    // Norm layers
    if name.contains("norm") || name.contains("ln_") || name.contains("layer_norm") {
        return LayerType::Norm;
    }

    // Embedding
    if name.contains("embed") || name.contains("token_embd") || name.contains("wte") || name.contains("wpe") {
        return LayerType::Embedding;
    }

    // Conv2D: 4D tensor [out_ch, in_ch, kH, kW]
    if ndim == 4 {
        return LayerType::Conv2D;
    }

    // Attention projections
    if name.contains("attn") || name.contains("self_attn")
        || name.contains("attn_q") || name.contains("attn_k")
        || name.contains("attn_v") || name.contains("attn_output")
        || name.contains("q_proj") || name.contains("k_proj")
        || name.contains("v_proj") || name.contains("o_proj")
    {
        return LayerType::Attention;
    }

    // Feed-forward
    if name.contains("ffn") || name.contains("mlp") || name.contains("fc1")
        || name.contains("fc2") || name.contains("gate") || name.contains("up_proj")
        || name.contains("down_proj") || name.contains("w1") || name.contains("w2")
        || name.contains("w3")
    {
        return LayerType::FeedForward;
    }

    // 2D matrix we can't classify — compress anyway
    if ndim == 2 && total >= 4096 {
        return LayerType::Attention; // treat as generic weight matrix
    }

    LayerType::Skip
}

// ============================================================================
// Golden-step projection: f32 row → Base17
// ============================================================================

const BASE_DIM: usize = 17;
/// Golden-step = round(17 / φ) = round(17 / 1.618) = 11. gcd(11,17)=1 → visits all residues.
const GOLDEN_STEP: usize = 11;
const FP_SCALE: f64 = 256.0;

/// Golden-step position table (compile-time).
const GOLDEN_POS: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

/// Project a single f32 row vector to Base17 via golden-step octave averaging.
///
/// This is the f32 analog of `Base17::encode(&[i8])` — same golden-step
/// traversal, but operating on float weights instead of binary accumulators.
pub fn project_row_to_base17(row: &[f32]) -> Base17 {
    let d = row.len();
    let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];

    for octave in 0..n_octaves {
        for bi in 0..BASE_DIM {
            let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
            if dim < d {
                sum[bi] += row[dim] as f64;
                count[bi] += 1;
            }
        }
    }

    let mut dims = [0i16; BASE_DIM];
    for i in 0..BASE_DIM {
        if count[i] > 0 {
            let mean = sum[i] / count[i] as f64;
            dims[i] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    Base17 { dims }
}

// ============================================================================
// BF16-direct optimizations: skip f32 intermediate, strided octave sampling
// ============================================================================

/// Halftone-dropped golden positions: keep every other step (9 of 17).
/// Well-distributed across 0..16; max gap = 3. Odd bins interpolated.
const HALFTONE_POS: [u8; 9] = {
    let mut t = [0u8; 9];
    let mut i = 0;
    let mut j = 0;
    while i < BASE_DIM {
        if i % 2 == 0 {
            t[j] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
            j += 1;
        }
        i += 1;
    }
    t
};

/// Which of the 17 Base17 bins each halftone position maps to.
const HALFTONE_TO_BIN: [u8; 9] = [0, 2, 4, 6, 8, 10, 12, 14, 16];

/// Convert one BF16 u16 to f64. Zero allocation.
#[inline(always)]
fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// Project a BF16 row directly to Base17. No f32 Vec allocated.
///
/// Same golden-step octave averaging as project_row_to_base17(),
/// but reads u16 BF16 values and converts inline to f64 accumulator.
/// Memory: 17 × f64 accumulators = 136 bytes stack.
pub fn project_row_bf16_direct(row: &[u16]) -> Base17 {
    let d = row.len();
    let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];

    for octave in 0..n_octaves {
        for bi in 0..BASE_DIM {
            let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
            if dim < d {
                sum[bi] += bf16_to_f64(row[dim]);
                count[bi] += 1;
            }
        }
    }

    let mut dims = [0i16; BASE_DIM];
    for i in 0..BASE_DIM {
        if count[i] > 0 {
            let mean = sum[i] / count[i] as f64;
            dims[i] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    Base17 { dims }
}

/// Project a BF16 row with octave stride and halftone dropping.
///
/// For a 5120-element row at stride=16:
///   302 octaves / 16 = 19 sampled × 9 halftone = 171 BF16→f64 conversions
///   vs 5120 in the full path (97% reduction).
/// Odd bins interpolated from neighbors.
pub fn project_row_bf16_strided(row: &[u16], octave_stride: usize) -> Base17 {
    let d = row.len();
    let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;

    let mut half_sum = [0.0f64; 9];
    let mut half_count = [0u32; 9];

    let mut octave = 0;
    while octave < n_octaves {
        for hi in 0..9 {
            let dim = octave * BASE_DIM + HALFTONE_POS[hi] as usize;
            if dim < d {
                half_sum[hi] += bf16_to_f64(row[dim]);
                half_count[hi] += 1;
            }
        }
        octave += octave_stride;
    }

    let mut dims = [0i16; BASE_DIM];

    // Even bins: direct from halftone samples
    for hi in 0..9 {
        let bin = HALFTONE_TO_BIN[hi] as usize;
        if half_count[hi] > 0 {
            let mean = half_sum[hi] / half_count[hi] as f64;
            dims[bin] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
        }
    }

    // Odd bins: interpolate from neighbors (circular)
    for odd in (1..BASE_DIM).step_by(2) {
        let left = dims[odd - 1] as i32;
        let right = dims[(odd + 1) % BASE_DIM] as i32;
        dims[odd] = ((left + right) / 2) as i16;
    }

    Base17 { dims }
}

/// Read a BF16 tensor as raw u16 values. NO f32 conversion.
/// `buf` is reusable — caller allocates once, passes to every tensor.
pub fn read_tensor_bf16_raw<R: Read + Seek>(
    reader: &mut R,
    gguf_file: &gguf::GgufFile,
    tensor: &gguf::TensorInfo,
    buf: &mut Vec<u16>,
) -> Result<usize, String> {
    let abs_offset = gguf_file.tensor_data_offset + tensor.offset;
    reader.seek(std::io::SeekFrom::Start(abs_offset)).map_err(|e| e.to_string())?;

    let n_elements = tensor.element_count() as usize;
    if buf.len() < n_elements {
        buf.resize(n_elements, 0);
    }

    // SAFETY: u16 and [u8; 2] have the same layout on little-endian (x86/ARM).
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2)
    };
    reader.read_exact(byte_slice).map_err(|e| e.to_string())?;

    Ok(n_elements)
}

/// Helper: tensor dimensions → (rows, cols) without needing data.
fn tensor_to_rows_dims(dims: &[u64], layer_type: &LayerType) -> (usize, usize) {
    match layer_type {
        LayerType::Conv2D if dims.len() == 4 => {
            (dims[0] as usize, (dims[1] * dims[2] * dims[3]) as usize)
        }
        _ if dims.len() >= 2 => {
            let rows = dims[0] as usize;
            let cols: usize = dims[1..].iter().map(|&d| d as usize).product();
            (rows, cols)
        }
        _ => {
            let total: usize = dims.iter().map(|&d| d as usize).product();
            (1, total)
        }
    }
}

/// Helper: LayerType → stats array index.
fn layer_type_index(lt: &LayerType) -> usize {
    match lt {
        LayerType::Attention => 0,
        LayerType::FeedForward => 1,
        LayerType::Conv2D => 2,
        LayerType::Norm => 3,
        LayerType::Embedding => 4,
        LayerType::Skip => 5,
    }
}

/// Stream-index a BF16 GGUF with all optimizations.
///
/// vs stream_index_gguf():
///   - No f32 Vec allocation (saves 283 MB per tensor)
///   - Reusable u16 buffer (one alloc for entire shard)
///   - Strided octave projection (97% fewer conversions when stride>1)
///   - Direct BF16→f64 inline conversion
///
/// Falls back to f32 path for non-BF16 dtypes.
pub fn stream_index_gguf_bf16<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    octave_stride: usize,
    callback: Option<&dyn Fn(&str, &LayerType, usize, usize)>,
) -> Result<IndexStats, String> {
    let gguf_header = gguf::read_gguf_header(reader)?;
    let mut stats = IndexStats::default();
    stats.tensors_total = gguf_header.tensors.len();

    writer.write_all(b"BGZ7").map_err(|e| e.to_string())?;
    writer.write_all(&(gguf_header.tensors.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;

    // ONE reusable buffer — grows to largest tensor, never shrinks
    let mut bf16_buf: Vec<u16> = Vec::new();

    for tensor in &gguf_header.tensors {
        let layer_type = classify_tensor(&tensor.name, &tensor.dimensions);

        if matches!(layer_type, LayerType::Skip | LayerType::Norm) {
            stats.tensors_skipped += 1;
            continue;
        }

        let is_bf16 = matches!(tensor.dtype, gguf::GgmlType::BF16);

        if is_bf16 {
            // FAST PATH: BF16 direct — no f32 intermediate
            let n_elements = read_tensor_bf16_raw(reader, &gguf_header, tensor, &mut bf16_buf)?;
            let (n_rows, n_cols) = tensor_to_rows_dims(&tensor.dimensions, &layer_type);

            let mut rows = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let start = r * n_cols;
                let end = (start + n_cols).min(n_elements);
                let row_slice = &bf16_buf[start..end];
                let b17 = if octave_stride > 1 {
                    project_row_bf16_strided(row_slice, octave_stride)
                } else {
                    project_row_bf16_direct(row_slice)
                };
                rows.push(b17);
            }

            let orig_bytes = (n_rows * n_cols * 4) as u64;
            let comp_bytes = (rows.len() * Base17::BYTE_SIZE) as u64;

            let ct = CompressedTensor {
                name: tensor.name.clone(),
                layer_type: layer_type.clone(),
                original_shape: tensor.dimensions.clone(),
                n_rows,
                n_cols,
                rows,
            };
            ct.write_to(writer)?;

            let lt_idx = layer_type_index(&layer_type);
            stats.by_type[lt_idx].0 += 1;
            stats.by_type[lt_idx].1 += orig_bytes;
            stats.by_type[lt_idx].2 += comp_bytes;
            stats.original_bytes += orig_bytes;
            stats.compressed_bytes += comp_bytes;
            stats.tensors_indexed += 1;

            let peak = n_elements as u64 * 2;
            if peak > stats.peak_tensor_bytes { stats.peak_tensor_bytes = peak; }

            if let Some(cb) = callback {
                cb(&tensor.name, &layer_type, orig_bytes as usize, comp_bytes as usize);
            }
        } else {
            // FALLBACK: non-BF16 — use original f32 path
            let data = gguf::read_tensor_f32(reader, &gguf_header, tensor)?;
            let tensor_bytes = data.len() as u64 * 4;
            if tensor_bytes > stats.peak_tensor_bytes {
                stats.peak_tensor_bytes = tensor_bytes;
            }

            let (n_rows, n_cols) = tensor_to_rows(&data, &tensor.dimensions, &layer_type);
            let mut rows = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let start = r * n_cols;
                let end = (start + n_cols).min(data.len());
                rows.push(project_row_to_base17(&data[start..end]));
            }

            let orig_bytes = (n_rows * n_cols * 4) as u64;
            let comp_bytes = (rows.len() * Base17::BYTE_SIZE) as u64;

            let ct = CompressedTensor {
                name: tensor.name.clone(),
                layer_type: layer_type.clone(),
                original_shape: tensor.dimensions.clone(),
                n_rows,
                n_cols,
                rows,
            };
            ct.write_to(writer)?;

            let lt_idx = layer_type_index(&layer_type);
            stats.by_type[lt_idx].0 += 1;
            stats.by_type[lt_idx].1 += orig_bytes;
            stats.by_type[lt_idx].2 += comp_bytes;
            stats.original_bytes += orig_bytes;
            stats.compressed_bytes += comp_bytes;
            stats.tensors_indexed += 1;

            if let Some(cb) = callback {
                cb(&tensor.name, &layer_type, orig_bytes as usize, comp_bytes as usize);
            }
        }
    }

    Ok(stats)
}

// ============================================================================
// Compressed tensor output
// ============================================================================

/// One compressed tensor: name + per-row Base17 projections.
#[derive(Clone, Debug)]
pub struct CompressedTensor {
    pub name: String,
    pub layer_type: LayerType,
    pub original_shape: Vec<u64>,
    /// Number of rows (vectors) in the matrix.
    pub n_rows: usize,
    /// Number of columns (dimension of each vector) before projection.
    pub n_cols: usize,
    /// Base17 projection per row. Length = n_rows.
    pub rows: Vec<Base17>,
}

impl CompressedTensor {
    /// Total compressed size in bytes.
    pub fn compressed_bytes(&self) -> usize {
        self.rows.len() * Base17::BYTE_SIZE
    }

    /// Original size in bytes (f32).
    pub fn original_bytes(&self) -> usize {
        self.n_rows * self.n_cols * 4
    }

    /// Compression ratio.
    pub fn ratio(&self) -> f64 {
        if self.compressed_bytes() == 0 { return 0.0; }
        self.original_bytes() as f64 / self.compressed_bytes() as f64
    }

    /// Serialize to bytes: [name_len:u32][name][layer_type:u8][n_rows:u32][n_cols:u32][base17 × n_rows]
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), String> {
        let name_bytes = self.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        w.write_all(name_bytes).map_err(|e| e.to_string())?;

        let lt_byte: u8 = match self.layer_type {
            LayerType::Attention => 0,
            LayerType::FeedForward => 1,
            LayerType::Conv2D => 2,
            LayerType::Norm => 3,
            LayerType::Embedding => 4,
            LayerType::Skip => 5,
        };
        w.write_all(&[lt_byte]).map_err(|e| e.to_string())?;
        w.write_all(&(self.n_rows as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        w.write_all(&(self.n_cols as u32).to_le_bytes()).map_err(|e| e.to_string())?;

        for b17 in &self.rows {
            w.write_all(&b17.to_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

// ============================================================================
// Reshape helpers
// ============================================================================

/// Reshape a flat f32 tensor into rows × cols based on layer type.
///
/// - Attention/FFN/Embedding: dims = [rows, cols] → rows vectors of cols dimensions.
/// - Conv2D: dims = [out_ch, in_ch, kH, kW] → out_ch vectors of (in_ch * kH * kW) dims.
/// - Norm/Skip: returned as single row.
fn tensor_to_rows(data: &[f32], dims: &[u64], layer_type: &LayerType) -> (usize, usize) {
    match layer_type {
        LayerType::Conv2D if dims.len() == 4 => {
            let out_ch = dims[0] as usize;
            let kernel_dim = (dims[1] * dims[2] * dims[3]) as usize;
            (out_ch, kernel_dim)
        }
        _ if dims.len() >= 2 => {
            let rows = dims[0] as usize;
            let cols: usize = dims[1..].iter().map(|&d| d as usize).product();
            (rows, cols)
        }
        _ => {
            (1, data.len())
        }
    }
}

// ============================================================================
// Streaming indexer
// ============================================================================

/// Statistics from one indexing run.
#[derive(Clone, Debug, Default)]
pub struct IndexStats {
    pub tensors_total: usize,
    pub tensors_indexed: usize,
    pub tensors_skipped: usize,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub peak_tensor_bytes: u64,
    pub by_type: [(usize, u64, u64); 6], // per LayerType: (count, orig_bytes, comp_bytes)
}

impl IndexStats {
    pub fn overall_ratio(&self) -> f64 {
        if self.compressed_bytes == 0 { return 0.0; }
        self.original_bytes as f64 / self.compressed_bytes as f64
    }
}

/// Stream-index a GGUF file: read header, process each tensor, write compressed output.
///
/// Peak RAM = largest single tensor as f32 + pipeline overhead.
/// For Llama 4 Scout: largest expert = 5120 × 13824 × 4 = ~270 MB.
/// Total RAM: ~300 MB regardless of model size.
pub fn stream_index_gguf<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    callback: Option<&dyn Fn(&str, &LayerType, usize, usize)>,
) -> Result<IndexStats, String> {
    let gguf = gguf::read_gguf_header(reader)?;
    let mut stats = IndexStats::default();
    stats.tensors_total = gguf.tensors.len();

    // Write file header: magic + tensor count
    writer.write_all(b"BGZ7").map_err(|e| e.to_string())?;
    writer.write_all(&(gguf.tensors.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;

    for tensor in &gguf.tensors {
        let layer_type = classify_tensor(&tensor.name, &tensor.dimensions);

        // Skip norms and tiny tensors
        if matches!(layer_type, LayerType::Skip | LayerType::Norm) {
            stats.tensors_skipped += 1;
            continue;
        }

        // Read tensor data as f32 (dequantizing if needed)
        let data = gguf::read_tensor_f32(reader, &gguf, tensor)?;

        let tensor_bytes = data.len() as u64 * 4;
        if tensor_bytes > stats.peak_tensor_bytes {
            stats.peak_tensor_bytes = tensor_bytes;
        }

        // Reshape into row vectors
        let (n_rows, n_cols) = tensor_to_rows(&data, &tensor.dimensions, &layer_type);

        // Project each row to Base17
        let mut rows = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let start = r * n_cols;
            let end = (start + n_cols).min(data.len());
            let row_slice = &data[start..end];
            rows.push(project_row_to_base17(row_slice));
        }

        let ct = CompressedTensor {
            name: tensor.name.clone(),
            layer_type: layer_type.clone(),
            original_shape: tensor.dimensions.clone(),
            n_rows,
            n_cols,
            rows,
        };

        // Update stats
        let orig = ct.original_bytes() as u64;
        let comp = ct.compressed_bytes() as u64;
        stats.tensors_indexed += 1;
        stats.original_bytes += orig;
        stats.compressed_bytes += comp;

        let lt_idx = match &ct.layer_type {
            LayerType::Attention => 0,
            LayerType::FeedForward => 1,
            LayerType::Conv2D => 2,
            LayerType::Norm => 3,
            LayerType::Embedding => 4,
            LayerType::Skip => 5,
        };
        stats.by_type[lt_idx].0 += 1;
        stats.by_type[lt_idx].1 += orig;
        stats.by_type[lt_idx].2 += comp;

        if let Some(cb) = callback {
            cb(&ct.name, &ct.layer_type, ct.original_bytes(), ct.compressed_bytes());
        }

        // Write compressed tensor
        ct.write_to(writer)?;

        // data dropped here — RAM freed for next tensor
    }

    Ok(stats)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_classify_attention() {
        assert_eq!(classify_tensor("blk.0.attn_q.weight", &[4096, 4096]), LayerType::Attention);
        assert_eq!(classify_tensor("blk.0.attn_k.weight", &[4096, 1024]), LayerType::Attention);
        assert_eq!(classify_tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]), LayerType::Attention);
    }

    #[test]
    fn test_classify_ffn() {
        assert_eq!(classify_tensor("blk.0.ffn_gate.weight", &[4096, 11008]), LayerType::FeedForward);
        assert_eq!(classify_tensor("blk.0.ffn_up.weight", &[4096, 11008]), LayerType::FeedForward);
        assert_eq!(classify_tensor("model.layers.0.mlp.gate_proj.weight", &[4096, 11008]), LayerType::FeedForward);
    }

    #[test]
    fn test_classify_conv2d() {
        assert_eq!(classify_tensor("unet.conv1.weight", &[512, 512, 3, 3]), LayerType::Conv2D);
    }

    #[test]
    fn test_classify_norm() {
        assert_eq!(classify_tensor("blk.0.attn_norm.weight", &[4096]), LayerType::Norm);
    }

    #[test]
    fn test_classify_embedding() {
        assert_eq!(classify_tensor("token_embd.weight", &[32000, 4096]), LayerType::Embedding);
    }

    #[test]
    fn test_classify_skip_small() {
        assert_eq!(classify_tensor("some.bias", &[128]), LayerType::Skip);
    }

    #[test]
    fn test_project_row_basic() {
        // Constant row → all dims should be the same
        let row = vec![1.0f32; 4096];
        let b17 = project_row_to_base17(&row);
        // Mean of 1.0 scaled by 256 = 256
        for &d in &b17.dims {
            assert_eq!(d, 256);
        }
    }

    #[test]
    fn test_project_row_zero() {
        let row = vec![0.0f32; 4096];
        let b17 = project_row_to_base17(&row);
        assert_eq!(b17, Base17::zero());
    }

    #[test]
    fn test_project_row_preserves_ordering() {
        // Two rows that differ → their Base17 L1 should be > 0
        let row_a = vec![1.0f32; 4096];
        let mut row_b = vec![1.0f32; 4096];
        row_b[0] = 100.0;
        row_b[1] = -100.0;

        let a = project_row_to_base17(&row_a);
        let b = project_row_to_base17(&row_b);
        assert!(a.l1(&b) > 0, "different rows should have nonzero L1");
    }

    #[test]
    fn test_project_small_row() {
        // Row smaller than 17 dims — should still work
        let row = vec![2.0f32; 8];
        let b17 = project_row_to_base17(&row);
        // Some dims will have count=0 and stay 0
        let nonzero = b17.dims.iter().filter(|&&d| d != 0).count();
        assert!(nonzero > 0 && nonzero <= 8);
    }

    #[test]
    fn test_conv2d_reshape() {
        // Conv2D [512, 512, 3, 3] → 512 rows of 4608
        let dims = vec![512u64, 512, 3, 3];
        let (rows, cols) = tensor_to_rows(&[], &dims, &LayerType::Conv2D);
        assert_eq!(rows, 512);
        assert_eq!(cols, 4608);
    }

    #[test]
    fn test_attention_reshape() {
        let dims = vec![4096u64, 4096];
        let (rows, cols) = tensor_to_rows(&[], &dims, &LayerType::Attention);
        assert_eq!(rows, 4096);
        assert_eq!(cols, 4096);
    }

    #[test]
    fn test_compressed_tensor_ratio() {
        let ct = CompressedTensor {
            name: "test".into(),
            layer_type: LayerType::Attention,
            original_shape: vec![4096, 4096],
            n_rows: 4096,
            n_cols: 4096,
            rows: vec![Base17::zero(); 4096],
        };
        assert_eq!(ct.original_bytes(), 4096 * 4096 * 4); // 64 MB
        assert_eq!(ct.compressed_bytes(), 4096 * 34); // 136 KB
        let ratio = ct.ratio();
        assert!(ratio > 480.0 && ratio < 490.0, "ratio={}", ratio); // ~482x
    }

    #[test]
    fn test_stream_index_synthetic_gguf() {
        // Build a minimal GGUF in memory with 2 tensors
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

        // Tensor 1: attention weight [64, 64] F32
        let t1_name = "blk.0.attn_q.weight";
        buf.extend_from_slice(&(t1_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t1_name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndims
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Tensor 2: norm (small, should be skipped)
        let t2_name = "blk.0.attn_norm.weight";
        buf.extend_from_slice(&(t2_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t2_name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndims
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        let t2_offset = 64 * 64 * 4; // after tensor 1
        buf.extend_from_slice(&(t2_offset as u64).to_le_bytes());

        // Pad to alignment (32 bytes)
        while buf.len() % 32 != 0 { buf.push(0); }

        // Tensor 1 data: 64×64 f32
        for i in 0..(64 * 64) {
            buf.extend_from_slice(&((i as f32) * 0.001).to_le_bytes());
        }

        // Tensor 2 data: 64 f32
        for i in 0..64 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let mut reader = Cursor::new(&buf);
        let mut output = Vec::new();

        let stats = stream_index_gguf(&mut reader, &mut output, None).unwrap();

        assert_eq!(stats.tensors_total, 2);
        assert_eq!(stats.tensors_indexed, 1); // attention
        assert_eq!(stats.tensors_skipped, 1); // norm
        assert!(stats.compressed_bytes > 0);
        assert!(stats.original_bytes > stats.compressed_bytes);
        assert!(output.len() > 8); // magic + at least one tensor

        // Verify output magic
        assert_eq!(&output[0..4], b"BGZ7");
    }

    #[test]
    #[ignore] // Requires /tmp/openchat/openchat-3.5-0106.Q8_0.gguf
    fn test_stream_index_openchat_q8() {
        use std::io::{BufReader, BufWriter};

        let path = "/tmp/openchat/openchat-3.5-0106.Q8_0.gguf";
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => { eprintln!("SKIP: {} not found", path); return; }
        };
        let input_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut reader = BufReader::new(file);

        let out_path = "/tmp/openchat/openchat-3.5-0106.bgz7";
        let out = std::fs::File::create(out_path).expect("create output");
        let mut writer = BufWriter::new(out);

        let stats = stream_index_gguf(
            &mut reader,
            &mut writer,
            Some(&|name, layer_type, orig, comp| {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:50} {:12?} {:>10} → {:>8} ({:.0}×)",
                    name, layer_type, orig, comp, ratio);
            }),
        ).expect("stream_index_gguf");

        drop(writer);
        let out_size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);

        eprintln!();
        eprintln!("=== OpenChat 3.5 Q8_0 → bgz17 Results ===");
        eprintln!("  Input:  {:.2} GB ({})", input_size as f64 / 1e9, path);
        eprintln!("  Output: {:.2} MB ({})", out_size as f64 / 1e6, out_path);
        eprintln!("  Tensors: {} total, {} indexed, {} skipped",
            stats.tensors_total, stats.tensors_indexed, stats.tensors_skipped);
        eprintln!("  Original (f32): {:.2} MB", stats.original_bytes as f64 / 1e6);
        eprintln!("  Compressed:     {:.2} MB", stats.compressed_bytes as f64 / 1e6);
        eprintln!("  Overall ratio:  {:.1}×", stats.overall_ratio());
        eprintln!("  Peak tensor:    {:.2} MB", stats.peak_tensor_bytes as f64 / 1e6);
        eprintln!();

        let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
        for (i, name) in type_names.iter().enumerate() {
            let (count, orig, comp) = stats.by_type[i];
            if count > 0 {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:<12} {:>3} tensors: {:>10.2} MB → {:>8.2} MB ({:.1}×)",
                    name, count, orig as f64 / 1e6, comp as f64 / 1e6, ratio);
            }
        }

        assert!(stats.tensors_indexed > 0, "should index at least some tensors");
        assert!(stats.overall_ratio() > 10.0, "ratio should be significant: {:.1}", stats.overall_ratio());
    }

    #[test]
    #[ignore] // Streams from HuggingFace — requires network + time
    fn test_stream_index_llama4_scout_from_hf() {
        use super::super::http_reader::{HttpRangeReader, resolve_hf_url};
        use std::io::BufWriter;

        let repo = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF";
        let filename = "Llama-4-Scout-17B-16E-Instruct-UD-IQ1_S.gguf";

        eprintln!("Resolving {} / {} ...", repo, filename);
        let (url, size) = match resolve_hf_url(repo, filename) {
            Ok(r) => r,
            Err(e) => { eprintln!("SKIP: {}", e); return; }
        };
        eprintln!("  URL resolved, size: {:.2} GB", size as f64 / 1e9);

        let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024); // 16 MB chunks

        let out_path = "/tmp/llama4_scout.bgz7";
        let out = std::fs::File::create(out_path).expect("create output");
        let mut writer = BufWriter::new(out);

        eprintln!("Streaming index...");
        let stats = stream_index_gguf(
            &mut reader,
            &mut writer,
            Some(&|name, layer_type, orig, comp| {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:60} {:12?} {:>12} → {:>8} ({:.0}×)",
                    name, layer_type, orig, comp, ratio);
            }),
        ).expect("stream_index_gguf");

        drop(writer);
        let out_size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);

        eprintln!();
        eprintln!("=== Llama 4 Scout → bgz17 (streamed from HF) ===");
        eprintln!("  Source:     {:.2} GB ({})", size as f64 / 1e9, filename);
        eprintln!("  Output:     {:.2} MB ({})", out_size as f64 / 1e6, out_path);
        eprintln!("  Downloaded: {:.2} GB", reader.bytes_downloaded() as f64 / 1e9);
        eprintln!("  Tensors:    {} indexed, {} skipped",
            stats.tensors_indexed, stats.tensors_skipped);
        eprintln!("  Original (f32): {:.2} GB", stats.original_bytes as f64 / 1e9);
        eprintln!("  Compressed:     {:.2} MB", stats.compressed_bytes as f64 / 1e6);
        eprintln!("  Ratio:          {:.1}×", stats.overall_ratio());
        eprintln!("  Peak tensor:    {:.2} MB", stats.peak_tensor_bytes as f64 / 1e6);

        let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
        for (i, name) in type_names.iter().enumerate() {
            let (count, orig, comp) = stats.by_type[i];
            if count > 0 {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:<12} {:>3} tensors: {:>10.2} GB → {:>8.2} MB ({:.1}×)",
                    name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
            }
        }

        assert!(stats.tensors_indexed > 0);
    }

    /// Run one shard of Llama 4 Scout BF16 through the streaming indexer.
    /// Returns the output path on success.
    fn run_llama4_shard(shard: u32) -> Option<(String, IndexStats)> {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let repo = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF";
        let filename = format!(
            "BF16/Llama-4-Scout-17B-16E-Instruct-BF16-{:05}-of-00005.gguf", shard
        );
        // Shards are ~18-44 GB each; use conservative 44 GB estimate
        let size: u64 = 44_000_000_000;

        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
        eprintln!("Streaming shard {}/5: {}", shard, filename);

        let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);

        let out_path = format!("/tmp/llama4_scout_shard{}.bgz7", shard);
        let out = std::fs::File::create(&out_path).expect("create output");
        let mut writer = BufWriter::new(out);

        let stats = stream_index_gguf(
            &mut reader,
            &mut writer,
            Some(&|name, layer_type, orig, comp| {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:60} {:12?} {:>12} → {:>8} ({:.0}×)",
                    name, layer_type, orig, comp, ratio);
            }),
        ).expect("stream_index_gguf");

        drop(writer);
        let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

        eprintln!();
        eprintln!("=== Llama 4 Scout BF16 Shard {}/5 → bgz17 ===", shard);
        eprintln!("  Output:     {:.2} MB ({})", out_size as f64 / 1e6, out_path);
        eprintln!("  Downloaded: {:.2} GB", reader.bytes_downloaded() as f64 / 1e9);
        eprintln!("  Tensors:    {} indexed, {} skipped",
            stats.tensors_indexed, stats.tensors_skipped);
        eprintln!("  Original (f32): {:.2} GB", stats.original_bytes as f64 / 1e9);
        eprintln!("  Compressed:     {:.2} MB", stats.compressed_bytes as f64 / 1e6);
        eprintln!("  Ratio:          {:.1}×", stats.overall_ratio());
        eprintln!("  Peak tensor:    {:.2} MB", stats.peak_tensor_bytes as f64 / 1e6);

        let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
        for (i, name) in type_names.iter().enumerate() {
            let (count, orig, comp) = stats.by_type[i];
            if count > 0 {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:<12} {:>3} tensors: {:>10.2} GB → {:>8.2} MB ({:.1}×)",
                    name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
            }
        }

        assert!(stats.tensors_indexed > 0);
        Some((out_path, stats))
    }

    #[test]
    #[ignore]
    fn test_stream_index_llama4_bf16_shard1() { run_llama4_shard(1); }
    #[test]
    #[ignore]
    fn test_stream_index_llama4_bf16_shard2() { run_llama4_shard(2); }
    #[test]
    #[ignore]
    fn test_stream_index_llama4_bf16_shard3() { run_llama4_shard(3); }
    #[test]
    #[ignore]
    fn test_stream_index_llama4_bf16_shard4() { run_llama4_shard(4); }
    #[test]
    #[ignore]
    fn test_stream_index_llama4_bf16_shard5() { run_llama4_shard(5); }

    // ── BF16-direct optimization tests ──

    #[test]
    fn test_halftone_positions_coverage() {
        let positions: Vec<u8> = HALFTONE_POS.to_vec();
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 3, 5, 6, 8, 10, 13, 15]);
    }

    #[test]
    fn test_bf16_to_f64_accuracy() {
        assert_eq!(bf16_to_f64(0x3F80), 1.0);
        assert_eq!(bf16_to_f64(0x0000), 0.0);
        assert_eq!(bf16_to_f64(0xBF80), -1.0);
        let v = bf16_to_f64(0x4049);
        assert!((v - 3.140625).abs() < 0.01);
    }

    #[test]
    fn test_strided_vs_full_agreement() {
        // Constant BF16 row → stride shouldn't matter
        let row: Vec<u16> = vec![0x3F80; 5120]; // all 1.0
        let full = project_row_bf16_direct(&row);
        let strided = project_row_bf16_strided(&row, 16);

        for i in 0..17 {
            let diff = (full.dims[i] as i32 - strided.dims[i] as i32).abs();
            assert!(diff <= 1, "bin {} differs by {}: full={}, strided={}",
                i, diff, full.dims[i], strided.dims[i]);
        }
    }

    #[test]
    fn test_bf16_direct_matches_f32_path() {
        // Same data in BF16 and f32 should produce identical Base17
        let f32_row: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
        let bf16_row: Vec<u16> = f32_row.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();

        let from_f32 = project_row_to_base17(&f32_row);
        let from_bf16 = project_row_bf16_direct(&bf16_row);

        // BF16 truncates mantissa, so allow ±1 tolerance per dim
        for i in 0..17 {
            let diff = (from_f32.dims[i] as i32 - from_bf16.dims[i] as i32).abs();
            assert!(diff <= 2, "bin {} differs by {}", i, diff);
        }
    }
}
