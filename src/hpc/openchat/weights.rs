//! OpenChat 3.5 / Mistral-7B weight loading from GGUF format.
//!
//! Uses `crate::hpc::gguf` for dequantization (Q4_K_M, Q8_0, F16).
//! No weights stored in the binary — loaded at runtime from user-provided GGUF.
//!
//! GGUF tensor naming convention (llama.cpp style):
//! ```text
//! token_embd.weight         → [32000, 4096]
//! blk.{i}.attn_q.weight     → [4096, 4096]
//! blk.{i}.attn_k.weight     → [1024, 4096]  (GQA: 8 KV heads × 128D)
//! blk.{i}.attn_v.weight     → [1024, 4096]
//! blk.{i}.attn_output.weight → [4096, 4096]
//! blk.{i}.attn_norm.weight  → [4096]         (RMSNorm, no bias)
//! blk.{i}.ffn_gate.weight   → [14336, 4096]  (SiLU gate)
//! blk.{i}.ffn_up.weight     → [14336, 4096]  (up projection)
//! blk.{i}.ffn_down.weight   → [4096, 14336]  (down projection)
//! blk.{i}.ffn_norm.weight   → [4096]         (RMSNorm)
//! output_norm.weight        → [4096]
//! output.weight             → [32000, 4096]   (or tied to token_embd)
//! ```

use crate::hpc::gguf::{self, GgufFile};
use crate::hpc::models::safetensors::transpose_matrix;

/// Mistral-7B / OpenChat 3.5 configuration.
pub const VOCAB_SIZE: usize = 32000;
pub const EMBED_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = EMBED_DIM / NUM_Q_HEADS; // 128
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;   // 1024
pub const MLP_DIM: usize = 14336; // Mistral uses 14336 (not 4× embed)
pub const MAX_SEQ_LEN: usize = 8192; // Mistral supports 8K context (32K with sliding window)
pub const ROPE_THETA: f32 = 10000.0;
pub const RMS_EPS: f32 = 1e-5;
pub const GQA_RATIO: usize = NUM_Q_HEADS / NUM_KV_HEADS; // 4

/// Weights for one Mistral transformer layer.
#[derive(Clone)]
pub struct MistralLayerWeights {
    /// Attention RMSNorm weight [4096] (no bias).
    pub attn_norm: Vec<f32>,
    /// Q projection: [4096, 4096] → pre-transposed to [4096, 4096].
    pub attn_q: Vec<f32>,
    /// K projection: [1024, 4096] → pre-transposed to [1024, 4096].
    pub attn_k: Vec<f32>,
    /// V projection: [1024, 4096] → pre-transposed to [1024, 4096].
    pub attn_v: Vec<f32>,
    /// Output projection: [4096, 4096] → pre-transposed.
    pub attn_output: Vec<f32>,
    /// FFN RMSNorm weight [4096].
    pub ffn_norm: Vec<f32>,
    /// Gate projection (SiLU): [14336, 4096] → pre-transposed.
    pub ffn_gate: Vec<f32>,
    /// Up projection: [14336, 4096] → pre-transposed.
    pub ffn_up: Vec<f32>,
    /// Down projection: [4096, 14336] → pre-transposed.
    pub ffn_down: Vec<f32>,
}

/// Complete OpenChat/Mistral-7B model weights.
#[derive(Clone)]
pub struct OpenChatWeights {
    /// Token embedding: [32000, 4096].
    pub token_embd: Vec<f32>,
    /// Transformer layers.
    pub layers: Vec<MistralLayerWeights>,
    /// Final RMSNorm weight [4096].
    pub output_norm: Vec<f32>,
    /// Output projection (lm_head): [32000, 4096].
    /// May be tied to token_embd (same data).
    pub output: Vec<f32>,
}

impl OpenChatWeights {
    /// Load from a GGUF file (e.g., openchat_3.5.Q4_K_M.gguf).
    ///
    /// Dequantizes all tensors to f32 on load. For Q4_K_M (~4.4GB GGUF),
    /// the f32 model will use ~28GB RAM. For Q8_0 (~7.7GB GGUF), ~28GB.
    ///
    /// Pre-transposes weight matrices for SIMD-contiguous `matmul_vec`.
    pub fn from_gguf(path: &std::path::Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("open {}: {}", path.display(), e))?;
        let header = gguf::read_gguf_header(&mut file)?;

        let mut read = |name: &str| -> Result<Vec<f32>, String> {
            let tensor = gguf::find_tensor(&header, name)
                .ok_or_else(|| format!("missing tensor: {}", name))?;
            gguf::read_tensor_f32(&mut file, &header, tensor)
        };

        let token_embd = read("token_embd.weight")?;
        let output_norm = read("output_norm.weight")?;

        // Output may be tied to token_embd
        let output = if header.tensors.iter().any(|t| t.name == "output.weight") {
            read("output.weight")?
        } else {
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let mut attn_q = read(&format!("blk.{}.attn_q.weight", i))?;
            let mut attn_k = read(&format!("blk.{}.attn_k.weight", i))?;
            let mut attn_v = read(&format!("blk.{}.attn_v.weight", i))?;
            let mut attn_output = read(&format!("blk.{}.attn_output.weight", i))?;
            let mut ffn_gate = read(&format!("blk.{}.ffn_gate.weight", i))?;
            let mut ffn_up = read(&format!("blk.{}.ffn_up.weight", i))?;
            let mut ffn_down = read(&format!("blk.{}.ffn_down.weight", i))?;

            // Pre-transpose for SIMD-contiguous matmul
            transpose_matrix(&mut attn_q, EMBED_DIM, EMBED_DIM);
            transpose_matrix(&mut attn_k, EMBED_DIM, KV_DIM);
            transpose_matrix(&mut attn_v, EMBED_DIM, KV_DIM);
            transpose_matrix(&mut attn_output, EMBED_DIM, EMBED_DIM);
            transpose_matrix(&mut ffn_gate, EMBED_DIM, MLP_DIM);
            transpose_matrix(&mut ffn_up, EMBED_DIM, MLP_DIM);
            transpose_matrix(&mut ffn_down, MLP_DIM, EMBED_DIM);

            layers.push(MistralLayerWeights {
                attn_norm: read(&format!("blk.{}.attn_norm.weight", i))?,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm: read(&format!("blk.{}.ffn_norm.weight", i))?,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(OpenChatWeights {
            token_embd,
            layers,
            output_norm,
            output,
        })
    }
}

/// OpenChat 3.5 chat template tokens.
pub mod chat_template {
    /// Beginning of text.
    pub const BOS_TOKEN_ID: u32 = 1;
    /// End of text.
    pub const EOS_TOKEN_ID: u32 = 2;
    /// OpenChat uses "GPT4 Correct User:" / "GPT4 Correct Assistant:" markers.
    /// These are tokenized sequences, not single tokens.
    /// The prefix for user messages (approximate token IDs — actual depends on tokenizer).
    pub const USER_PREFIX: &str = "GPT4 Correct User: ";
    pub const ASSISTANT_PREFIX: &str = "GPT4 Correct Assistant:";
    /// End-of-turn token (used to separate turns in OpenChat).
    pub const EOT_TOKEN: &str = "<|end_of_turn|>";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_consistency() {
        assert_eq!(EMBED_DIM, NUM_Q_HEADS * HEAD_DIM);
        assert_eq!(KV_DIM, NUM_KV_HEADS * HEAD_DIM);
        assert_eq!(GQA_RATIO, 4);
        assert_eq!(HEAD_DIM, 128);
    }

    #[test]
    fn test_kv_dim() {
        // GQA: 8 KV heads × 128D = 1024
        assert_eq!(KV_DIM, 1024);
    }

    #[test]
    fn test_chat_template() {
        assert!(chat_template::USER_PREFIX.contains("User"));
        assert!(chat_template::ASSISTANT_PREFIX.contains("Assistant"));
    }
}
