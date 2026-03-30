//! Minimal GGUF reader — extracts f32 weight tensors from GGUF model files.
//!
//! Supports: F32, F16, BF16, Q8_0 dequantization.
//! Purpose: load one attention head's Q/K/V weights for bgz-tensor benchmarking.
//!
//! # Format
//!
//! ```text
//! [magic:4][version:4][tensor_count:8][metadata_count:8]
//! [metadata_kv × metadata_count]
//! [tensor_info × tensor_count]
//! [padding to alignment]
//! [tensor_data]
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// GGUF magic number: "GGUF" in little-endian.
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32

/// Tensor data type in GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    F64 = 28,
    BF16 = 30,
    Unknown = 255,
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            28 => Self::F64,
            30 => Self::BF16,
            _ => Self::Unknown,
        }
    }
}

impl GgmlType {
    /// Bytes per element for unquantized types. Returns None for quantized types.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::F64 => Some(8),
            _ => None, // Quantized types have block structure
        }
    }

    /// Block size for quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            _ => 1, // Unquantized: 1 element per "block"
        }
    }

    /// Bytes per block for quantized types.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::Q4_0 => 18,    // 2 (scale) + 32/2 (nibbles) = 18
            Self::Q4_1 => 20,    // 2 (scale) + 2 (min) + 32/2 = 20
            Self::Q8_0 => 34,    // 2 (scale) + 32 (int8s) = 34
            Self::Q4_K => 144,   // Complex block structure
            _ => 0,
        }
    }
}

/// Info about one tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64, // relative to tensor data start
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }
}

/// Parsed GGUF header + tensor directory.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, String>, // simplified: all values as strings
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64, // absolute file offset where tensor data starts
    pub alignment: u64,
}

/// Read a GGUF file header and tensor directory.
pub fn read_gguf_header<R: Read + Seek>(reader: &mut R) -> Result<GgufFile, String> {
    // Magic
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        return Err(format!("Not a GGUF file: magic={:#x}, expected={:#x}", magic, GGUF_MAGIC));
    }

    // Version
    let version = read_u32(reader)?;
    if version < 2 || version > 3 {
        return Err(format!("Unsupported GGUF version: {}", version));
    }

    // Counts
    let tensor_count = read_u64(reader)?;
    let metadata_count = read_u64(reader)?;

    // Metadata KV pairs (simplified: read keys, skip complex values)
    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)?;
        let value = read_metadata_value(reader, value_type)?;
        metadata.insert(key, value);
    }

    // Alignment
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32);

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = read_u32(reader)? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(read_u64(reader)?);
        }
        let dtype = GgmlType::from(read_u32(reader)?);
        let offset = read_u64(reader)?;
        tensors.push(TensorInfo { name, dimensions, dtype, offset });
    }

    // Compute tensor data start: current position, aligned up
    let current_pos = reader.stream_position().map_err(|e| e.to_string())?;
    let tensor_data_offset = (current_pos + alignment - 1) / alignment * alignment;

    Ok(GgufFile {
        version,
        metadata,
        tensors,
        tensor_data_offset,
        alignment,
    })
}

/// Read one tensor's data as f32 (dequantizing if needed).
pub fn read_tensor_f32<R: Read + Seek>(
    reader: &mut R,
    gguf: &GgufFile,
    tensor: &TensorInfo,
) -> Result<Vec<f32>, String> {
    let abs_offset = gguf.tensor_data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| e.to_string())?;

    let n_elements = tensor.element_count() as usize;

    match tensor.dtype {
        GgmlType::F32 => {
            let mut buf = vec![0u8; n_elements * 4];
            reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GgmlType::F16 => {
            let mut buf = vec![0u8; n_elements * 2];
            reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect())
        }
        GgmlType::BF16 => {
            let mut buf = vec![0u8; n_elements * 2];
            reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect())
        }
        GgmlType::Q8_0 => {
            dequantize_q8_0(reader, n_elements)
        }
        GgmlType::Q4_0 => {
            dequantize_q4_0(reader, n_elements)
        }
        GgmlType::Q4_K => {
            dequantize_q4_k(reader, n_elements)
        }
        other => Err(format!("Unsupported dtype for dequantization: {:?}", other)),
    }
}

/// Find a tensor by name pattern (e.g., "blk.0.attn_q.weight").
pub fn find_tensor<'a>(gguf: &'a GgufFile, pattern: &str) -> Option<&'a TensorInfo> {
    gguf.tensors.iter().find(|t| t.name.contains(pattern))
}

/// List all tensor names and shapes.
pub fn list_tensors(gguf: &GgufFile) -> Vec<(String, Vec<u64>, GgmlType)> {
    gguf.tensors.iter()
        .map(|t| (t.name.clone(), t.dimensions.clone(), t.dtype))
        .collect()
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64(r)? as usize;
    if len > 65536 {
        return Err(format!("String too long: {} bytes", len));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

fn read_metadata_value<R: Read + Seek>(r: &mut R, value_type: u32) -> Result<String, String> {
    match value_type {
        0 => { let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok(b[0].to_string()) } // u8
        1 => { let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok((b[0] as i8).to_string()) } // i8
        2 => { let mut b = [0u8; 2]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok(u16::from_le_bytes(b).to_string()) } // u16
        3 => { let mut b = [0u8; 2]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok(i16::from_le_bytes(b).to_string()) } // i16
        4 => Ok(read_u32(r)?.to_string()), // u32
        5 => { let v = read_u32(r)?; Ok((v as i32).to_string()) } // i32
        6 => { let mut b = [0u8; 4]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok(f32::from_le_bytes(b).to_string()) } // f32
        7 => { let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok((b[0] != 0).to_string()) } // bool
        8 => read_string(r), // string
        9 => { // array
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            // Skip array elements (we don't need them for tensor loading)
            for _ in 0..count {
                let _ = read_metadata_value(r, elem_type)?;
            }
            Ok(format!("[array of {} × type {}]", count, elem_type))
        }
        10 => Ok(read_u64(r)?.to_string()), // u64
        11 => { let v = read_u64(r)?; Ok((v as i64).to_string()) } // i64
        12 => { let mut b = [0u8; 8]; r.read_exact(&mut b).map_err(|e| e.to_string())?; Ok(f64::from_le_bytes(b).to_string()) } // f64
        _ => Err(format!("Unknown metadata value type: {}", value_type)),
    }
}

/// Dequantize Q8_0: each block = 2 bytes scale (f16) + 32 bytes int8.
fn dequantize_q8_0<R: Read>(r: &mut R, n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 32;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(n_elements);

    for _ in 0..n_blocks {
        // Read scale as f16
        let mut scale_buf = [0u8; 2];
        r.read_exact(&mut scale_buf).map_err(|e| e.to_string())?;
        let scale = f16_to_f32(u16::from_le_bytes(scale_buf));

        // Read 32 int8 values
        let mut quants = [0u8; 32];
        r.read_exact(&mut quants).map_err(|e| e.to_string())?;

        for &q in &quants {
            result.push((q as i8) as f32 * scale);
        }
    }

    result.truncate(n_elements);
    Ok(result)
}

/// Dequantize Q4_0: each block = 2 bytes scale (f16) + 16 bytes (32 nibbles).
fn dequantize_q4_0<R: Read>(r: &mut R, n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 32;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(n_elements);

    for _ in 0..n_blocks {
        let mut scale_buf = [0u8; 2];
        r.read_exact(&mut scale_buf).map_err(|e| e.to_string())?;
        let scale = f16_to_f32(u16::from_le_bytes(scale_buf));

        let mut nibbles = [0u8; 16];
        r.read_exact(&mut nibbles).map_err(|e| e.to_string())?;

        for &byte in &nibbles {
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            result.push(lo as f32 * scale);
            result.push(hi as f32 * scale);
        }
    }

    result.truncate(n_elements);
    Ok(result)
}

/// Dequantize Q4_K: super-blocks of 256 elements.
///
/// Q4_K block layout (144 bytes for 256 elements):
/// - 2 bytes: d (f16 scale)
/// - 2 bytes: dmin (f16 min)
/// - 12 bytes: scales (6-bit per sub-block, packed)
/// - 128 bytes: 256 4-bit quantized values (nibbles)
fn dequantize_q4_k<R: Read>(r: &mut R, n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 256;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut result = Vec::with_capacity(n_elements);

    for _ in 0..n_blocks {
        // Read d and dmin (f16)
        let mut d_buf = [0u8; 2];
        let mut dmin_buf = [0u8; 2];
        r.read_exact(&mut d_buf).map_err(|e| e.to_string())?;
        r.read_exact(&mut dmin_buf).map_err(|e| e.to_string())?;
        let d = f16_to_f32(u16::from_le_bytes(d_buf));
        let dmin = f16_to_f32(u16::from_le_bytes(dmin_buf));

        // Read scales (12 bytes = 8 sub-block scales + 8 sub-block mins, 6-bit packed)
        let mut scales_raw = [0u8; 12];
        r.read_exact(&mut scales_raw).map_err(|e| e.to_string())?;

        // Decode 8 scale/min pairs from 12 bytes (6 bits each)
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = scales_raw[i] & 0x3F;
            mn[i] = scales_raw[i + 4] & 0x3F;
            sc[i + 4] = ((scales_raw[i + 8] & 0x0F) << 2) | (scales_raw[i] >> 6);
            mn[i + 4] = ((scales_raw[i + 8] >> 4) << 2) | (scales_raw[i + 4] >> 6);
        }

        // Read 128 bytes of nibbles (256 4-bit values)
        let mut nibbles = [0u8; 128];
        r.read_exact(&mut nibbles).map_err(|e| e.to_string())?;

        // Dequantize: each sub-block of 32 elements
        for j in 0..8 {
            let sub_d = d * sc[j] as f32;
            let sub_m = dmin * mn[j] as f32;
            let nib_offset = j * 16;
            for k in 0..16 {
                let byte = nibbles[nib_offset + k];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                result.push(lo * sub_d - sub_m);
                result.push(hi * sub_d - sub_m);
            }
        }
    }

    result.truncate(n_elements);
    Ok(result)
}

/// Convert f16 bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31); // ±0
        }
        // Subnormal f16 → normal f32
        let mut m = mantissa;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        // f16 bias=15, f32 bias=127. Subnormal f16 has implicit exponent 1-15=-14.
        // After normalizing mantissa (e shifts), f32 exponent = 127 + (1-15) + e = 113 + e.
        // Minimum e = -10 (mantissa 0x001), giving f32_exp = 103. Always valid.
        let f32_exp = (113 + e) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }
    if exp == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        return f32::from_bits(f32_bits);
    }
    // Normal
    let f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    f32::from_bits(f32_bits)
}

/// Convert BF16 bit pattern to f32 (just shift left 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&metadata_count.to_le_bytes());
        buf
    }

    fn append_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn append_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], dtype: u32, offset: u64) {
        append_string(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let mut buf = make_gguf_header(1, 0);
        append_tensor_info(&mut buf, "test.weight", &[4, 4], 0, 0); // F32, offset 0

        // Pad to alignment (32 bytes)
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Tensor data: 16 f32 values
        for i in 0..16u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let mut cursor = Cursor::new(&buf);
        let gguf = read_gguf_header(&mut cursor).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "test.weight");
        assert_eq!(gguf.tensors[0].dimensions, vec![4, 4]);
        assert_eq!(gguf.tensors[0].dtype, GgmlType::F32);

        let data = read_tensor_f32(&mut cursor, &gguf, &gguf.tensors[0]).unwrap();
        assert_eq!(data.len(), 16);
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[15] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_conversion() {
        // f16 for 1.0: sign=0, exp=15, mantissa=0 → 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-4);
        // f16 for 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // f16 for -1.0: 0xBC00
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_bf16_conversion() {
        // bf16 for 1.0: 0x3F80 (upper 16 bits of f32 1.0)
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_q8_0_dequantize() {
        // Build a Q8_0 block: f16 scale + 32 int8 values
        let scale: f32 = 0.5;
        let scale_f16_bits: u16 = 0x3800; // f16 for 0.5
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }

        let mut cursor = Cursor::new(&block);
        let result = dequantize_q8_0(&mut cursor, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 1e-4); // 0 * 0.5 = 0
        assert!((result[1] - 0.5).abs() < 1e-4); // 1 * 0.5 = 0.5
        assert!((result[10] - 5.0).abs() < 1e-4); // 10 * 0.5 = 5.0
    }

    #[test]
    fn test_list_tensors() {
        let mut buf = make_gguf_header(2, 0);
        append_tensor_info(&mut buf, "blk.0.attn_q.weight", &[4096, 4096], 8, 0);
        append_tensor_info(&mut buf, "blk.0.attn_k.weight", &[4096, 1024], 8, 4096 * 4096 * 34 / 32);

        while buf.len() % 32 != 0 { buf.push(0); }

        let mut cursor = Cursor::new(&buf);
        let gguf = read_gguf_header(&mut cursor).unwrap();
        let tensors = list_tensors(&gguf);
        assert_eq!(tensors.len(), 2);
        assert!(tensors[0].0.contains("attn_q"));
        assert!(tensors[1].0.contains("attn_k"));
    }
}
