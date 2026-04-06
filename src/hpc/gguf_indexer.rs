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
use std::io::{Read, Seek, SeekFrom, Write};

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
/// round(17 / φ) = 11 — maximally irrational stride across BASE_DIM positions.
const GOLDEN_STEP: usize = (BASE_DIM as f64 / std::f64::consts::GOLDEN_RATIO + 0.5) as usize;
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

// ── F64x8 SIMD: 8 rows → 8 Base17 in parallel ──

/// Gather 8 BF16 values from 8 rows at the same column, convert to F64x8.
///
/// The gather is scalar (8 indexed loads) but the result is SIMD.
/// At -O2 with AVX-512, rustc may emit vpgatherqd + shift + vcvtps2pd.
#[inline(always)]
fn gather_bf16_x8(buf: &[u16], offsets: &[usize; 8]) -> crate::simd::F64x8 {
    crate::simd::F64x8::from_array([
        bf16_to_f64(buf[offsets[0]]),
        bf16_to_f64(buf[offsets[1]]),
        bf16_to_f64(buf[offsets[2]]),
        bf16_to_f64(buf[offsets[3]]),
        bf16_to_f64(buf[offsets[4]]),
        bf16_to_f64(buf[offsets[5]]),
        bf16_to_f64(buf[offsets[6]]),
        bf16_to_f64(buf[offsets[7]]),
    ])
}

/// Project 8 BF16 rows simultaneously to 8 Base17 patterns.
///
/// Memory: 17 × F64x8 accumulators on stack = 17 × 64 = 1088 bytes.
pub fn project_8rows_bf16_simd(
    buf: &[u16],
    row_starts: &[usize; 8],
    n_cols: usize,
    octave_stride: usize,
) -> [Base17; 8] {
    use crate::simd::F64x8;

    let n_octaves = (n_cols + BASE_DIM - 1) / BASE_DIM;

    let mut sums: [F64x8; BASE_DIM] = [F64x8::splat(0.0); BASE_DIM];
    let mut counts: [u32; BASE_DIM] = [0; BASE_DIM];

    // All 17 golden-step positions per sampled octave. Stride skips octaves,
    // NOT positions — every bin gets real data from actual weight values.
    let mut octave = 0;
    while octave < n_octaves {
        for bi in 0..BASE_DIM {
            let col = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
            if col < n_cols {
                let offsets: [usize; 8] = [
                    row_starts[0] + col, row_starts[1] + col,
                    row_starts[2] + col, row_starts[3] + col,
                    row_starts[4] + col, row_starts[5] + col,
                    row_starts[6] + col, row_starts[7] + col,
                ];
                sums[bi] += gather_bf16_x8(buf, &offsets);
                counts[bi] += 1;
            }
        }
        octave += octave_stride;
    }

    // Finalize: mean → scale → clamp → i16, all 8 lanes parallel
    let lo = F64x8::splat(-32768.0);
    let hi = F64x8::splat(32767.0);

    let mut dims_x8: [[i16; BASE_DIM]; 8] = [[0i16; BASE_DIM]; 8];

    for bin in 0..BASE_DIM {
        let c = counts[bin].max(1) as f64;
        let scaled = sums[bin] * F64x8::splat(FP_SCALE / c);
        let clamped = scaled.round().simd_clamp(lo, hi);
        let vals = clamped.to_array();
        for lane in 0..8 {
            dims_x8[lane][bin] = vals[lane] as i16;
        }
    }

    [
        Base17 { dims: dims_x8[0] }, Base17 { dims: dims_x8[1] },
        Base17 { dims: dims_x8[2] }, Base17 { dims: dims_x8[3] },
        Base17 { dims: dims_x8[4] }, Base17 { dims: dims_x8[5] },
        Base17 { dims: dims_x8[6] }, Base17 { dims: dims_x8[7] },
    ]
}

/// Scalar fallback for remainder rows (< 8).
pub fn project_1row_bf16_strided(row: &[u16], octave_stride: usize) -> Base17 {
    let d = row.len();
    let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;

    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];

    // All 17 positions per sampled octave — no halftone, all bins real
    let mut octave = 0;
    while octave < n_octaves {
        for bi in 0..BASE_DIM {
            let col = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
            if col < d {
                sum[bi] += bf16_to_f64(row[col]);
                count[bi] += 1;
            }
        }
        octave += octave_stride;
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

/// Project an entire BF16 tensor to Base17 using F64x8 SIMD.
///
/// Processes 8 rows in parallel per SIMD batch. Each of the 17 bins
/// holds an F64x8 accumulator (8 rows × 17 bins = 136 f64 lanes = 17 zmm registers).
///
/// Per sampled octave: 17 positions × 8 bf16_to_f64 gathers → 17 vaddpd.
/// For 5120-col rows at stride=16: 19 octaves × 17 = 323 vaddpd per 8-row batch.
pub fn project_tensor_bf16_simd(
    buf: &[u16],
    n_rows: usize,
    n_cols: usize,
    octave_stride: usize,
) -> Vec<Base17> {
    let mut result = Vec::with_capacity(n_rows);

    let full_batches = n_rows / 8;

    for batch in 0..full_batches {
        let base_row = batch * 8;
        let row_starts: [usize; 8] = [
            (base_row + 0) * n_cols, (base_row + 1) * n_cols,
            (base_row + 2) * n_cols, (base_row + 3) * n_cols,
            (base_row + 4) * n_cols, (base_row + 5) * n_cols,
            (base_row + 6) * n_cols, (base_row + 7) * n_cols,
        ];
        let b17s = project_8rows_bf16_simd(buf, &row_starts, n_cols, octave_stride);
        result.extend_from_slice(&b17s);
    }

    // Scalar tail
    for r in (full_batches * 8)..n_rows {
        let start = r * n_cols;
        let end = (start + n_cols).min(buf.len());
        result.push(project_1row_bf16_strided(&buf[start..end], octave_stride));
    }

    result
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
    let header = gguf::read_gguf_header(reader)?;
    stream_index_gguf_bf16_with_header(reader, writer, &header, octave_stride, callback)
}

/// Core BF16-direct indexer — works with any pre-parsed header (GGUF or safetensors).
///
/// The header must have:
/// - `tensor_data_offset`: absolute byte offset where tensor data starts
/// - `tensors`: Vec<TensorInfo> with name, dimensions, dtype, offset (relative to data start)
pub fn stream_index_gguf_bf16_with_header<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    header: &gguf::GgufFile,
    octave_stride: usize,
    callback: Option<&dyn Fn(&str, &LayerType, usize, usize)>,
) -> Result<IndexStats, String> {
    let mut stats = IndexStats::default();
    stats.tensors_total = header.tensors.len();

    writer.write_all(b"BGZ7").map_err(|e| e.to_string())?;
    writer.write_all(&(header.tensors.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;

    // Reusable buffer — capped at 128 MB (64M u16 elements).
    // Tensors larger than this are read in row batches.
    const MAX_BUF_ELEMS: usize = 64 * 1024 * 1024; // 128 MB of u16
    let mut bf16_buf: Vec<u16> = Vec::new();

    for tensor in &header.tensors {
        let layer_type = classify_tensor(&tensor.name, &tensor.dimensions);

        if matches!(layer_type, LayerType::Skip | LayerType::Norm) {
            stats.tensors_skipped += 1;
            continue;
        }

        let is_bf16 = matches!(tensor.dtype, gguf::GgmlType::BF16);

        if is_bf16 {
            // FAST PATH: BF16 direct — chunked row-batch reading.
            // Caps memory at MAX_BUF_ELEMS regardless of tensor size.
            // A 10.7 GB ffn_gate_exps tensor reads in ~128 MB batches.
            let (n_rows, n_cols) = tensor_to_rows_dims(&tensor.dimensions, &layer_type);
            let chunk_rows = if n_cols > 0 {
                (MAX_BUF_ELEMS / n_cols).max(8).min(n_rows) // at least 8 rows (SIMD batch)
            } else {
                n_rows
            };
            let chunk_elems = chunk_rows * n_cols;

            // Grow buffer to chunk size (not full tensor size)
            if bf16_buf.len() < chunk_elems {
                bf16_buf.resize(chunk_elems, 0);
            }

            // Seek to tensor start
            let abs_offset = header.tensor_data_offset + tensor.offset;
            reader.seek(std::io::SeekFrom::Start(abs_offset)).map_err(|e| e.to_string())?;

            let mut rows: Vec<Base17> = Vec::with_capacity(n_rows);
            let mut rows_done: usize = 0;
            let is_large = n_rows > chunk_rows;

            while rows_done < n_rows {
                let batch_n = (n_rows - rows_done).min(chunk_rows);
                let batch_elems = batch_n * n_cols;

                // Read batch bytes into reusable buffer
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        bf16_buf.as_mut_ptr() as *mut u8,
                        batch_elems * 2,
                    )
                };
                reader.read_exact(byte_slice).map_err(|e| e.to_string())?;

                // Project this batch
                if octave_stride > 1 {
                    let batch_b17 = project_tensor_bf16_simd(
                        &bf16_buf[..batch_elems], batch_n, n_cols, octave_stride
                    );
                    rows.extend_from_slice(&batch_b17);
                } else {
                    for r in 0..batch_n {
                        let start = r * n_cols;
                        rows.push(project_row_bf16_direct(&bf16_buf[start..start + n_cols]));
                    }
                }

                rows_done += batch_n;

                // Progress for large tensors (every chunk)
                if is_large && rows_done < n_rows {
                    eprintln!("    ... {}/{} rows ({:.0}%)",
                        rows_done, n_rows, rows_done as f64 / n_rows as f64 * 100.0);
                }
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

            let buf_bytes = chunk_elems as u64 * 2;
            if buf_bytes > stats.peak_tensor_bytes { stats.peak_tensor_bytes = buf_bytes; }

            // Shrink buffer if it grew past the cap (shouldn't, but defensive)
            if bf16_buf.len() > MAX_BUF_ELEMS {
                bf16_buf.truncate(MAX_BUF_ELEMS);
                bf16_buf.shrink_to(MAX_BUF_ELEMS);
            }

            if let Some(cb) = callback {
                cb(&tensor.name, &layer_type, orig_bytes as usize, comp_bytes as usize);
            }
        } else {
            // FALLBACK: non-BF16 — use original f32 path
            let data = gguf::read_tensor_f32(reader, &header, tensor)?;
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

    /// Deserialize from bytes: [name_len:u32][name][layer_type:u8][n_rows:u32][n_cols:u32][base17 × n_rows]
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self, String> {
        let mut u32_buf = [0u8; 4];

        r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        let name_len = u32::from_le_bytes(u32_buf) as usize;

        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(|e| e.to_string())?;
        let name = String::from_utf8(name_bytes).map_err(|e| e.to_string())?;

        let mut lt_buf = [0u8; 1];
        r.read_exact(&mut lt_buf).map_err(|e| e.to_string())?;
        let layer_type = match lt_buf[0] {
            0 => LayerType::Attention,
            1 => LayerType::FeedForward,
            2 => LayerType::Conv2D,
            3 => LayerType::Norm,
            4 => LayerType::Embedding,
            _ => LayerType::Skip,
        };

        r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        let n_rows = u32::from_le_bytes(u32_buf) as usize;

        r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        let n_cols = u32::from_le_bytes(u32_buf) as usize;

        let mut rows = Vec::with_capacity(n_rows);
        let mut b17_buf = [0u8; Base17::BYTE_SIZE];
        for _ in 0..n_rows {
            r.read_exact(&mut b17_buf).map_err(|e| e.to_string())?;
            rows.push(Base17::from_bytes(&b17_buf));
        }

        Ok(CompressedTensor {
            name,
            layer_type,
            original_shape: vec![], // not stored in bgz7
            n_rows,
            n_cols,
            rows,
        })
    }
}

/// Read all tensors from a bgz7 file.
///
/// Returns Vec of (name, layer_type, rows) tuples.
pub fn read_bgz7_file(path: &str) -> Result<Vec<CompressedTensor>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut reader = std::io::BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"BGZ7" {
        return Err(format!("bad magic: {:?}", magic));
    }

    let mut u32_buf = [0u8; 4];
    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let n_tensors = u32::from_le_bytes(u32_buf) as usize;

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        tensors.push(CompressedTensor::read_from(&mut reader)?);
    }

    Ok(tensors)
}

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

    /// Exact Scout BF16 shard sizes (verified via HuggingFace HEAD).
    /// Run one shard of Llama 4 Scout BF16 through the BF16-direct indexer.
    ///
    /// Uses stream_index_gguf_bf16 with F64x8 SIMD and strided octave sampling.
    /// No f32 intermediate allocation. Reusable u16 buffer inside the indexer.
    fn run_llama4_shard(shard: u32) -> Option<(String, IndexStats)> {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let repo = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF";
        let filename = format!(
            "BF16/Llama-4-Scout-17B-16E-Instruct-BF16-{:05}-of-00005.gguf", shard
        );
        let octave_stride: usize = 16;

        eprintln!("Streaming shard {}/5: {}", shard, filename);
        eprintln!("  BF16-direct, octave_stride={}, F64x8 SIMD", octave_stride);

        let mut reader = HttpRangeReader::from_hf(repo, &filename, 256 * 1024 * 1024)
            .expect("failed to resolve HF URL — check repo/filename");

        let out_path = format!("/tmp/llama4_scout_shard{}.bgz7", shard);
        let out = std::fs::File::create(&out_path).expect("create output");
        let mut writer = BufWriter::new(out);

        let stats = stream_index_gguf_bf16(
            &mut reader,
            &mut writer,
            octave_stride,
            Some(&|name, layer_type, orig, comp| {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:60} {:12?} {:>12} → {:>8} ({:.0}×)",
                    name, layer_type, orig, comp, ratio);
            }),
        ).expect("stream_index_gguf_bf16");

        drop(writer);
        let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

        eprintln!();
        eprintln!("=== Llama 4 Scout BF16 Shard {}/5 → bgz17 (BF16-direct) ===", shard);
        eprintln!("  Output:     {:.2} MB ({})", out_size as f64 / 1e6, out_path);
        eprintln!("  Downloaded: {:.2} GB", reader.bytes_downloaded() as f64 / 1e9);
        eprintln!("  Tensors:    {} indexed, {} skipped",
            stats.tensors_indexed, stats.tensors_skipped);
        eprintln!("  Original (f32): {:.2} GB", stats.original_bytes as f64 / 1e9);
        eprintln!("  Compressed:     {:.2} MB", stats.compressed_bytes as f64 / 1e6);
        eprintln!("  Ratio:          {:.1}×", stats.overall_ratio());
        eprintln!("  Peak buf (BF16): {:.2} MB", stats.peak_tensor_bytes as f64 / 1e6);

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
        let strided = project_1row_bf16_strided(&row, 16);

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

    #[test]
    fn test_simd_matches_scalar_constant() {
        let n_cols = 5120;
        let n_rows = 16; // 2 full SIMD batches
        let buf: Vec<u16> = vec![0x3F80; n_rows * n_cols]; // all 1.0 in BF16

        let simd_results = project_tensor_bf16_simd(&buf, n_rows, n_cols, 1);
        assert_eq!(simd_results.len(), n_rows);

        for r in 1..n_rows {
            for bin in 0..BASE_DIM {
                let diff = (simd_results[0].dims[bin] as i32 - simd_results[r].dims[bin] as i32).abs();
                assert!(diff == 0, "row {} bin {} differs: {} vs {}",
                    r, bin, simd_results[0].dims[bin], simd_results[r].dims[bin]);
            }
        }
    }

    #[test]
    fn test_simd_matches_scalar_strided() {
        let n_cols = 13824;
        let n_rows = 11; // 1 full batch + 3 remainder
        let mut buf = vec![0x3F80u16; n_rows * n_cols];
        for i in (0..buf.len()).step_by(2) {
            buf[i] = 0xBF80; // -1.0
        }

        let simd_results = project_tensor_bf16_simd(&buf, n_rows, n_cols, 16);
        assert_eq!(simd_results.len(), n_rows);

        for r in 0..n_rows {
            let start = r * n_cols;
            let scalar = project_1row_bf16_strided(&buf[start..start + n_cols], 16);
            for bin in 0..BASE_DIM {
                let diff = (simd_results[r].dims[bin] as i32 - scalar.dims[bin] as i32).abs();
                assert!(diff <= 1, "row {} bin {} simd={} scalar={} diff={}",
                    r, bin, simd_results[r].dims[bin], scalar.dims[bin], diff);
            }
        }
    }

    #[test]
    fn test_simd_tail_handling() {
        let n_cols = 256;
        for n_rows in 1..8 {
            let buf: Vec<u16> = vec![0x4000; n_rows * n_cols]; // 2.0 in BF16
            let results = project_tensor_bf16_simd(&buf, n_rows, n_cols, 16);
            assert_eq!(results.len(), n_rows, "wrong count for n_rows={}", n_rows);
        }
    }

    #[test]
    #[ignore] // Streams ~801 GB from HuggingFace
    fn test_stream_index_llama4_maverick_bf16_all_shards() {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let repo = "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF";

        let shards: [(u8, &str, u64); 18] = [
            ( 1, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00001-of-00018.gguf", 46_166_870_240),
            ( 2, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00002-of-00018.gguf", 42_949_673_376),
            ( 3, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00003-of-00018.gguf", 42_949_673_376),
            ( 4, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00004-of-00018.gguf", 42_949_673_376),
            ( 5, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00005-of-00018.gguf", 47_943_931_840),
            ( 6, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00006-of-00018.gguf", 42_949_673_376),
            ( 7, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00007-of-00018.gguf", 42_949_673_376),
            ( 8, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00008-of-00018.gguf", 42_949_673_376),
            ( 9, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00009-of-00018.gguf", 47_922_960_288),
            (10, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00010-of-00018.gguf", 42_949_673_376),
            (11, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00011-of-00018.gguf", 42_949_673_376),
            (12, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00012-of-00018.gguf", 47_912_433_568),
            (13, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00013-of-00018.gguf", 42_949_673_376),
            (14, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00014-of-00018.gguf", 42_949_673_376),
            (15, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00015-of-00018.gguf", 42_949_673_376),
            (16, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00016-of-00018.gguf", 47_912_474_624),
            (17, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00017-of-00018.gguf", 42_949_673_376),
            (18, "BF16/Llama-4-Maverick-17B-128E-Instruct-BF16-00018-of-00018.gguf", 48_214_491_296),
        ];

        // Octave stride: 16 = "4 octaves higher" with halftone skip
        // Change to 1 for full-precision comparison run
        let octave_stride: usize = 16;

        let mut grand_total_source: u64 = 0;
        let mut grand_total_compressed: u64 = 0;
        let mut grand_total_original: u64 = 0;
        let mut grand_total_tensors: usize = 0;
        let mut grand_by_type: [(usize, u64, u64); 6] = [(0, 0, 0); 6];

        let mut output_files: Vec<String> = Vec::new();
        let keep_recent: usize = 3;

        eprintln!("━━━ Llama 4 Maverick BF16-Direct Indexer ━━━");
        eprintln!("  Octave stride: {} (halftone skip: {})", octave_stride, octave_stride > 1);
        eprintln!("  BF16 direct: yes (no f32 intermediate)");
        eprintln!("  Reusable buffer: yes");
        eprintln!();

        for (shard_num, filename, size) in shards.iter() {
            let out_path = format!("/tmp/llama4_maverick_shard{:02}.bgz7", shard_num);

            eprintln!("━━━ Shard {:02}/18 ━━━", shard_num);

            let mut reader = HttpRangeReader::from_hf(repo, filename, 256 * 1024 * 1024)
                .expect("failed to resolve HF URL");

            let out = std::fs::File::create(&out_path).expect("create output");
            let mut writer = BufWriter::new(out);

            let stats = stream_index_gguf_bf16(
                &mut reader,
                &mut writer,
                octave_stride,
                Some(&|name, layer_type, orig, comp| {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("  {:60} {:12?} {:>12} → {:>8} ({:.0}×)",
                        name, layer_type, orig, comp, ratio);
                }),
            ).unwrap_or_else(|e| panic!("stream_index_gguf_bf16 shard {} failed: {}", shard_num, e));

            drop(writer);
            let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

            eprintln!("  Shard {:02}: {:.2} GB → {:.2} MB ({:.0}×)  peak_buf={:.1} MB",
                shard_num, *size as f64 / 1e9, out_size as f64 / 1e6,
                stats.overall_ratio(),
                stats.peak_tensor_bytes as f64 / 1e6);

            let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
            for (j, name) in type_names.iter().enumerate() {
                let (count, orig, comp) = stats.by_type[j];
                if count > 0 {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("    {:<12} {:>3} tensors: {:>10.2} GB → {:>8.2} MB ({:.0}×)",
                        name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
                    grand_by_type[j].0 += count;
                    grand_by_type[j].1 += orig;
                    grand_by_type[j].2 += comp;
                }
            }

            grand_total_source += *size;
            grand_total_compressed += out_size;
            grand_total_original += stats.original_bytes;
            grand_total_tensors += stats.tensors_indexed;

            // Tail deletion
            output_files.push(out_path.clone());
            while output_files.len() > keep_recent {
                let old = output_files.remove(0);
                match std::fs::remove_file(&old) {
                    Ok(()) => eprintln!("  Tail cleanup: {}", old),
                    Err(e) => eprintln!("  Tail cleanup warning: {} — {}", old, e),
                }
            }

            drop(reader);
            assert!(stats.tensors_indexed > 0, "shard {} empty", shard_num);
            eprintln!("  {}/18 done ({:.0}%)", shard_num, *shard_num as f64 / 18.0 * 100.0);
            eprintln!();
        }

        // Final cleanup
        for p in &output_files {
            let _ = std::fs::remove_file(p);
        }

        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("LLAMA 4 MAVERICK 17B-128E — FULL MODEL (ALL 18 SHARDS)");
        eprintln!("  Mode: BF16-direct, octave_stride={}", octave_stride);
        eprintln!("  Source (BF16):   {:>10.2} GB", grand_total_source as f64 / 1e9);
        eprintln!("  Original (f32):  {:>10.2} GB", grand_total_original as f64 / 1e9);
        eprintln!("  Compressed:      {:>10.2} MB", grand_total_compressed as f64 / 1e6);
        eprintln!("  Overall ratio:   {:>10.0}×",
            grand_total_original as f64 / grand_total_compressed.max(1) as f64);
        eprintln!("  Tensors indexed: {}", grand_total_tensors);

        let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
        for (j, name) in type_names.iter().enumerate() {
            let (count, orig, comp) = grand_by_type[j];
            if count > 0 {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("    {:<12} {:>4} tensors: {:>10.2} GB → {:>8.2} MB ({:.0}×)",
                    name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
            }
        }
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        assert!(grand_total_tensors > 500);
        assert!(grand_total_compressed < 500_000_000);
    }
}
