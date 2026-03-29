//! Stable Diffusion weight loading from safetensors.
//!
//! Uses the shared `models::safetensors` loader.
//! SD 1.5 has ~860M params across CLIP + UNet + VAE.
//!
//! No weights are stored in this crate — they're loaded at runtime
//! from user-provided safetensors files (disk space conscious).

use crate::hpc::models::safetensors::{SafeTensorsFile, transpose_matrix};
use super::clip::*;

/// Load CLIP text encoder weights from a safetensors file.
///
/// Expected tensor names follow HuggingFace diffusers convention:
/// `text_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.weight`
pub fn load_clip_weights(file: &SafeTensorsFile) -> Result<ClipWeights, String> {
    let token_embedding = file.read_f32("text_model.embeddings.token_embedding.weight")?;
    let position_embedding = file.read_f32("text_model.embeddings.position_embedding.weight")?;
    let ln_final_weight = file.read_f32("text_model.final_layer_norm.weight")?;
    let ln_final_bias = file.read_f32("text_model.final_layer_norm.bias")?;

    let mut layers = Vec::with_capacity(CLIP_NUM_LAYERS);
    for i in 0..CLIP_NUM_LAYERS {
        let prefix = format!("text_model.encoder.layers.{}", i);

        // CLIP stores Q/K/V separately — we concatenate to match GPT-2 pattern
        let q_weight = file.read_f32(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_weight = file.read_f32(&format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_weight = file.read_f32(&format!("{}.self_attn.v_proj.weight", prefix))?;
        let q_bias = file.read_f32(&format!("{}.self_attn.q_proj.bias", prefix))?;
        let k_bias = file.read_f32(&format!("{}.self_attn.k_proj.bias", prefix))?;
        let v_bias = file.read_f32(&format!("{}.self_attn.v_proj.bias", prefix))?;

        // Concatenate Q/K/V into combined [768, 2304]
        let mut attn_qkv_weight = Vec::with_capacity(q_weight.len() * 3);
        attn_qkv_weight.extend_from_slice(&q_weight);
        attn_qkv_weight.extend_from_slice(&k_weight);
        attn_qkv_weight.extend_from_slice(&v_weight);

        let mut attn_qkv_bias = Vec::with_capacity(q_bias.len() * 3);
        attn_qkv_bias.extend_from_slice(&q_bias);
        attn_qkv_bias.extend_from_slice(&k_bias);
        attn_qkv_bias.extend_from_slice(&v_bias);

        let mut attn_out_weight = file.read_f32(&format!("{}.self_attn.out_proj.weight", prefix))?;
        let attn_out_bias = file.read_f32(&format!("{}.self_attn.out_proj.bias", prefix))?;

        let mut mlp_fc_weight = file.read_f32(&format!("{}.mlp.fc1.weight", prefix))?;
        let mlp_fc_bias = file.read_f32(&format!("{}.mlp.fc1.bias", prefix))?;
        let mut mlp_proj_weight = file.read_f32(&format!("{}.mlp.fc2.weight", prefix))?;
        let mlp_proj_bias = file.read_f32(&format!("{}.mlp.fc2.bias", prefix))?;

        // Pre-transpose for SIMD-contiguous access
        transpose_matrix(&mut attn_out_weight, CLIP_EMBED_DIM, CLIP_EMBED_DIM);
        transpose_matrix(&mut mlp_fc_weight, CLIP_EMBED_DIM, CLIP_MLP_DIM);
        transpose_matrix(&mut mlp_proj_weight, CLIP_MLP_DIM, CLIP_EMBED_DIM);

        layers.push(ClipLayerWeights {
            ln1_weight: file.read_f32(&format!("{}.layer_norm1.weight", prefix))?,
            ln1_bias: file.read_f32(&format!("{}.layer_norm1.bias", prefix))?,
            attn_qkv_weight,
            attn_qkv_bias,
            attn_out_weight,
            attn_out_bias,
            ln2_weight: file.read_f32(&format!("{}.layer_norm2.weight", prefix))?,
            ln2_bias: file.read_f32(&format!("{}.layer_norm2.bias", prefix))?,
            mlp_fc_weight,
            mlp_fc_bias,
            mlp_proj_weight,
            mlp_proj_bias,
        });
    }

    Ok(ClipWeights {
        token_embedding,
        position_embedding,
        layers,
        ln_final_weight,
        ln_final_bias,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_weight_names() {
        // Just verify the naming convention compiles
        let prefix = "text_model.encoder.layers.0";
        let name = format!("{}.self_attn.q_proj.weight", prefix);
        assert!(name.contains("q_proj"));
    }
}
