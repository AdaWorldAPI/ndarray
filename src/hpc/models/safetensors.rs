//! Generic safetensors file loader.
//!
//! Shared between GPT-2, Stable Diffusion, BERT — any model stored
//! in HuggingFace safetensors format.
//!
//! Format: `[header_size:u64_le][header_json][tensor_data]`

use std::collections::HashMap;

/// Tensor metadata from safetensors header.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    /// Byte offset into the data section.
    pub offset: usize,
    /// Byte size of the tensor data.
    pub size: usize,
}

/// Parsed safetensors file: header metadata + raw bytes.
pub struct SafeTensorsFile {
    /// Raw file bytes (header + data).
    data: Vec<u8>,
    /// Byte offset where tensor data begins.
    data_start: usize,
    /// Tensor name → metadata.
    pub tensors: HashMap<String, TensorMeta>,
}

impl SafeTensorsFile {
    /// Load and parse a safetensors file.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        Self::from_bytes(data)
    }

    /// Parse from in-memory bytes (for embedded weights or tests).
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("file too small for safetensors header".into());
        }

        let header_size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]) as usize;

        if 8 + header_size > data.len() {
            return Err(format!("header_size {} exceeds file len {}", header_size, data.len()));
        }

        let header_json = std::str::from_utf8(&data[8..8 + header_size])
            .map_err(|e| format!("invalid UTF-8 in header: {}", e))?;

        let data_start = 8 + header_size;
        let tensors = parse_header(header_json)?;

        Ok(Self { data, data_start, tensors })
    }

    /// Read a tensor as Vec<f32> (little-endian F32).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self.tensors.get(name)
            .ok_or_else(|| format!("missing tensor: {}", name))?;
        let start = self.data_start + meta.offset;
        let end = start + meta.size;
        if end > self.data.len() {
            return Err(format!("tensor {} [{}, {}) exceeds file len {}", name, start, end, self.data.len()));
        }
        Ok(self.data[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read a tensor as Vec<f16> stored as raw u16 (for F16 tensors).
    pub fn read_f16_raw(&self, name: &str) -> Result<Vec<u16>, String> {
        let meta = self.tensors.get(name)
            .ok_or_else(|| format!("missing tensor: {}", name))?;
        let start = self.data_start + meta.offset;
        let end = start + meta.size;
        if end > self.data.len() {
            return Err(format!("tensor {} exceeds file", name));
        }
        Ok(self.data[start..end]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Total data size in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len() - self.data_start
    }
}

/// Transpose a [rows, cols] row-major matrix to [cols, rows].
/// Used by all models to pre-transpose weights for SIMD-contiguous matmul.
pub fn transpose_matrix(data: &mut Vec<f32>, rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols);
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    *data = transposed;
}

/// Parse safetensors JSON header to tensor metadata.
fn parse_header(json: &str) -> Result<HashMap<String, TensorMeta>, String> {
    let mut tensors = HashMap::new();
    let mut pos = 0;

    while let Some(key_start) = json[pos..].find('"') {
        let key_start = pos + key_start + 1;
        let key_end = match json[key_start..].find('"') {
            Some(e) => key_start + e,
            None => break,
        };
        let key = &json[key_start..key_end];
        pos = key_end + 1;

        // Skip __metadata__
        if key == "__metadata__" {
            if let Some(end) = json[pos..].find('}') {
                pos += end + 1;
            }
            continue;
        }

        // Find data_offsets
        if let Some(offsets_start) = json[pos..].find("data_offsets") {
            let search_start = pos + offsets_start;
            if let Some(bracket_start) = json[search_start..].find('[') {
                let arr_start = search_start + bracket_start + 1;
                if let Some(bracket_end) = json[arr_start..].find(']') {
                    let arr = &json[arr_start..arr_start + bracket_end];
                    let nums: Vec<usize> = arr.split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if nums.len() == 2 {
                        tensors.insert(key.to_string(), TensorMeta {
                            offset: nums[0],
                            size: nums[1] - nums[0],
                        });
                    }
                }
            }
        }

        if let Some(brace) = json[pos..].find('}') {
            pos += brace + 1;
        }
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_basic() {
        let json = r#"{"tensor_a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}, "tensor_b": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}}"#;
        let tensors = parse_header(json).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["tensor_a"].offset, 0);
        assert_eq!(tensors["tensor_a"].size, 16);
        assert_eq!(tensors["tensor_b"].offset, 16);
        assert_eq!(tensors["tensor_b"].size, 8);
    }

    #[test]
    fn test_parse_header_with_metadata() {
        let json = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}"#;
        let tensors = parse_header(json).unwrap();
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("w"));
    }

    #[test]
    fn test_transpose_matrix() {
        let mut m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        transpose_matrix(&mut m, 2, 3);
        // Expected 3×2: [1,4,2,5,3,6]
        assert_eq!(m, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_safetensors_from_bytes_too_small() {
        let err = SafeTensorsFile::from_bytes(vec![0; 4]);
        assert!(err.is_err());
    }
}
