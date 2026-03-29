//! CLIP text encoder — transforms text tokens into conditioning embeddings.
//!
//! Same transformer architecture as GPT-2 but:
//! - 77 max sequence length (not 1024)
//! - No causal mask (bidirectional attention)
//! - Output is the full sequence, not just last token
//!
//! All ops via `crate::hpc::models::layers` (shared F32x16 SIMD).

use crate::hpc::models::layers;

/// CLIP text encoder configuration.
pub const CLIP_VOCAB_SIZE: usize = 49408;
pub const CLIP_EMBED_DIM: usize = 768;
pub const CLIP_NUM_LAYERS: usize = 12;
pub const CLIP_NUM_HEADS: usize = 12;
pub const CLIP_HEAD_DIM: usize = CLIP_EMBED_DIM / CLIP_NUM_HEADS;
pub const CLIP_MAX_SEQ: usize = 77;
pub const CLIP_MLP_DIM: usize = 3072;

/// Weights for one CLIP transformer layer.
#[derive(Clone)]
pub struct ClipLayerWeights {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub attn_qkv_weight: Vec<f32>,
    pub attn_qkv_bias: Vec<f32>,
    pub attn_out_weight: Vec<f32>,
    pub attn_out_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    pub mlp_fc_weight: Vec<f32>,
    pub mlp_fc_bias: Vec<f32>,
    pub mlp_proj_weight: Vec<f32>,
    pub mlp_proj_bias: Vec<f32>,
}

/// Complete CLIP text encoder weights.
#[derive(Clone)]
pub struct ClipWeights {
    pub token_embedding: Vec<f32>,    // [49408, 768]
    pub position_embedding: Vec<f32>, // [77, 768]
    pub layers: Vec<ClipLayerWeights>,
    pub ln_final_weight: Vec<f32>,
    pub ln_final_bias: Vec<f32>,
}

/// CLIP text encoder.
pub struct ClipEncoder {
    weights: ClipWeights,
}

impl ClipEncoder {
    pub fn new(weights: ClipWeights) -> Self {
        Self { weights }
    }

    /// Encode token IDs → embeddings [seq_len, 768].
    ///
    /// Uses bidirectional attention (no causal mask).
    /// Returns the full sequence of hidden states.
    pub fn encode(&self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len().min(CLIP_MAX_SEQ);
        let mut hidden = vec![0.0f32; seq_len * CLIP_EMBED_DIM];

        // Token + position embedding
        for (t, &token_id) in tokens.iter().take(seq_len).enumerate() {
            let tok_off = token_id as usize * CLIP_EMBED_DIM;
            let pos_off = t * CLIP_EMBED_DIM;
            let hid_off = t * CLIP_EMBED_DIM;
            for d in 0..CLIP_EMBED_DIM {
                hidden[hid_off + d] =
                    self.weights.token_embedding[tok_off + d]
                    + self.weights.position_embedding[pos_off + d];
            }
        }

        // Transformer layers
        for layer in &self.weights.layers {
            self.transformer_layer(layer, &mut hidden, seq_len);
        }

        // Final layer norm (per-position)
        for t in 0..seq_len {
            let off = t * CLIP_EMBED_DIM;
            layers::layer_norm(
                &mut hidden[off..off + CLIP_EMBED_DIM],
                &self.weights.ln_final_weight,
                &self.weights.ln_final_bias,
            );
        }

        hidden
    }

    /// One transformer layer (bidirectional self-attention + MLP).
    fn transformer_layer(
        &self,
        layer: &ClipLayerWeights,
        hidden: &mut [f32],
        seq_len: usize,
    ) {
        // Process each position through attention + MLP
        // For the scaffold: simplified single-token path.
        // Full implementation would do batched multi-head attention.
        for t in 0..seq_len {
            let off = t * CLIP_EMBED_DIM;
            let mut normed = hidden[off..off + CLIP_EMBED_DIM].to_vec();
            layers::layer_norm(&mut normed, &layer.ln1_weight, &layer.ln1_bias);

            // Self-attention (simplified: each position attends to itself for scaffold)
            let mut attn_out = vec![0.0f32; CLIP_EMBED_DIM];
            layers::matmul_vec(
                &normed, &layer.attn_out_weight, &layer.attn_out_bias,
                &mut attn_out, CLIP_EMBED_DIM, CLIP_EMBED_DIM,
            );

            // Residual
            for d in 0..CLIP_EMBED_DIM {
                hidden[off + d] += attn_out[d];
            }

            // MLP
            let mut normed2 = hidden[off..off + CLIP_EMBED_DIM].to_vec();
            layers::layer_norm(&mut normed2, &layer.ln2_weight, &layer.ln2_bias);

            let mut fc_out = vec![0.0f32; CLIP_MLP_DIM];
            layers::matmul_vec(&normed2, &layer.mlp_fc_weight, &layer.mlp_fc_bias, &mut fc_out, CLIP_EMBED_DIM, CLIP_MLP_DIM);
            layers::gelu(&mut fc_out);

            let mut proj_out = vec![0.0f32; CLIP_EMBED_DIM];
            layers::matmul_vec(&fc_out, &layer.mlp_proj_weight, &layer.mlp_proj_bias, &mut proj_out, CLIP_MLP_DIM, CLIP_EMBED_DIM);

            // Residual
            for d in 0..CLIP_EMBED_DIM {
                hidden[off + d] += proj_out[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_config() {
        assert_eq!(CLIP_EMBED_DIM, CLIP_NUM_HEADS * CLIP_HEAD_DIM);
        assert_eq!(CLIP_MLP_DIM, 4 * CLIP_EMBED_DIM);
    }

    #[test]
    fn test_clip_encode_shape() {
        let weights = ClipWeights {
            token_embedding: vec![0.0; CLIP_VOCAB_SIZE * CLIP_EMBED_DIM],
            position_embedding: vec![0.0; CLIP_MAX_SEQ * CLIP_EMBED_DIM],
            layers: Vec::new(), // no layers = just embedding + final LN
            ln_final_weight: vec![1.0; CLIP_EMBED_DIM],
            ln_final_bias: vec![0.0; CLIP_EMBED_DIM],
        };
        let enc = ClipEncoder::new(weights);
        let out = enc.encode(&[0, 1, 2]);
        assert_eq!(out.len(), 3 * CLIP_EMBED_DIM);
    }
}
