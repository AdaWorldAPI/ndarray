//! GPT-2 weight loading from safetensors format.
//!
//! Loads ALL weights needed for inference, not just embeddings.
//! Each tensor stored as contiguous f32 arrays for SIMD access.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// GPT-2 model configuration.
pub const VOCAB_SIZE: usize = 50257;
pub const EMBED_DIM: usize = 768;
pub const NUM_LAYERS: usize = 12;
pub const NUM_HEADS: usize = 12;
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 64
pub const MLP_DIM: usize = 3072; // 4 × EMBED_DIM
pub const MAX_SEQ_LEN: usize = 1024;

/// All weights for one transformer layer.
#[derive(Clone)]
pub struct LayerWeights {
    /// Attention layer norm: weight [768] + bias [768]
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    /// Combined Q/K/V projection: [768, 2304] + bias [2304]
    pub attn_qkv_weight: Vec<f32>,
    pub attn_qkv_bias: Vec<f32>,
    /// Output projection: [768, 768] + bias [768]
    pub attn_out_weight: Vec<f32>,
    pub attn_out_bias: Vec<f32>,
    /// MLP layer norm: weight [768] + bias [768]
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    /// MLP fc: [768, 3072] + bias [3072]
    pub mlp_fc_weight: Vec<f32>,
    pub mlp_fc_bias: Vec<f32>,
    /// MLP proj: [3072, 768] + bias [768]
    pub mlp_proj_weight: Vec<f32>,
    pub mlp_proj_bias: Vec<f32>,
}

/// Complete GPT-2 model weights.
#[derive(Clone)]
pub struct Gpt2Weights {
    /// Token embedding: [50257, 768]
    pub wte: Vec<f32>,
    /// Position embedding: [1024, 768]
    pub wpe: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<LayerWeights>,
    /// Final layer norm
    pub ln_f_weight: Vec<f32>,
    pub ln_f_bias: Vec<f32>,
}

impl Gpt2Weights {
    /// Load from a safetensors file via our memory-mapped reader.
    ///
    /// This reads ALL weights needed for inference (~500MB f32).
    /// For production: weights would be quantized or compiled to AttentionTable.
    pub fn from_safetensors(path: &std::path::Path) -> Result<Self, String> {
        // Safetensors format: [header_size:u64_le][header_json][tensor_data]
        let file = std::fs::read(path).map_err(|e| e.to_string())?;

        let header_size = u64::from_le_bytes([
            file[0], file[1], file[2], file[3],
            file[4], file[5], file[6], file[7],
        ]) as usize;

        let header_json = std::str::from_utf8(&file[8..8 + header_size])
            .map_err(|e| e.to_string())?;

        // Parse tensor metadata from JSON header
        let data_start = 8 + header_size;
        let tensors = parse_safetensors_header(header_json)?;

        let read_tensor = |name: &str| -> Result<Vec<f32>, String> {
            let info = tensors.get(name)
                .ok_or_else(|| format!("Missing tensor: {}", name))?;
            let start = data_start + info.offset;
            let end = start + info.size;
            if end > file.len() {
                return Err(format!("Tensor {} extends beyond file", name));
            }
            Ok(file[start..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        let wte = read_tensor("wte.weight")?;
        let wpe = read_tensor("wpe.weight")?;
        let ln_f_weight = read_tensor("ln_f.weight")?;
        let ln_f_bias = read_tensor("ln_f.bias")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let prefix = format!("h.{}", i);
            layers.push(LayerWeights {
                ln1_weight: read_tensor(&format!("{}.ln_1.weight", prefix))?,
                ln1_bias: read_tensor(&format!("{}.ln_1.bias", prefix))?,
                attn_qkv_weight: read_tensor(&format!("{}.attn.c_attn.weight", prefix))?,
                attn_qkv_bias: read_tensor(&format!("{}.attn.c_attn.bias", prefix))?,
                attn_out_weight: read_tensor(&format!("{}.attn.c_proj.weight", prefix))?,
                attn_out_bias: read_tensor(&format!("{}.attn.c_proj.bias", prefix))?,
                ln2_weight: read_tensor(&format!("{}.ln_2.weight", prefix))?,
                ln2_bias: read_tensor(&format!("{}.ln_2.bias", prefix))?,
                mlp_fc_weight: read_tensor(&format!("{}.mlp.c_fc.weight", prefix))?,
                mlp_fc_bias: read_tensor(&format!("{}.mlp.c_fc.bias", prefix))?,
                mlp_proj_weight: read_tensor(&format!("{}.mlp.c_proj.weight", prefix))?,
                mlp_proj_bias: read_tensor(&format!("{}.mlp.c_proj.bias", prefix))?,
            });
        }

        let mut weights = Gpt2Weights {
            wte, wpe, layers, ln_f_weight, ln_f_bias,
        };
        weights.transpose_weights_for_simd();
        Ok(weights)
    }

    /// Transpose all weight matrices from [in_dim, out_dim] to [out_dim, in_dim].
    /// After this, matmul can read weight rows contiguously for F32x16 SIMD.
    fn transpose_weights_for_simd(&mut self) {
        for layer in &mut self.layers {
            transpose_matrix(&mut layer.attn_qkv_weight, EMBED_DIM, 3 * EMBED_DIM);
            transpose_matrix(&mut layer.attn_out_weight, EMBED_DIM, EMBED_DIM);
            transpose_matrix(&mut layer.mlp_fc_weight, EMBED_DIM, MLP_DIM);
            transpose_matrix(&mut layer.mlp_proj_weight, MLP_DIM, EMBED_DIM);
        }
    }
}

/// Transpose a [rows, cols] matrix in-place to [cols, rows].
fn transpose_matrix(data: &mut Vec<f32>, rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols);
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    *data = transposed;
}

/// Tensor metadata from safetensors header.
struct TensorMeta {
    offset: usize,
    size: usize,
}

/// Parse safetensors JSON header to get tensor offsets and sizes.
fn parse_safetensors_header(json: &str) -> Result<HashMap<String, TensorMeta>, String> {
    // Minimal JSON parser for safetensors header format:
    // { "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }, ... }
    let mut tensors = HashMap::new();

    // Find each tensor entry
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

        // Advance past the closing brace of this tensor's value
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
    fn test_config_consistency() {
        assert_eq!(EMBED_DIM, NUM_HEADS * HEAD_DIM);
        assert_eq!(MLP_DIM, 4 * EMBED_DIM);
        assert_eq!(VOCAB_SIZE, 50257);
    }
}
