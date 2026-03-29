//! UNet denoiser — iterative denoising via cross-attention + ResBlocks.
//!
//! The core of Stable Diffusion: takes noisy latent + text conditioning,
//! predicts noise to remove. Runs N times per image (20-50 steps).
//!
//! # SD-specific ops (not shared with GPT-2):
//! - Conv2D (spatial convolution)
//! - GroupNorm (via `models::layers::group_norm`)
//! - SiLU activation (via `models::layers::silu`)
//! - Cross-attention (text embeddings condition denoising)
//! - Timestep embedding (sinusoidal positional encoding for diffusion step)

use crate::hpc::models::layers;

/// UNet configuration for SD 1.5.
pub const LATENT_CHANNELS: usize = 4;
pub const LATENT_SIZE: usize = 64; // 512/8 = 64 (VAE downscale)
pub const MODEL_CHANNELS: usize = 320;
pub const NUM_RES_BLOCKS: usize = 2;
pub const ATTENTION_RESOLUTIONS: &[usize] = &[4, 2, 1]; // at 16×16, 32×32, 64×64
pub const CHANNEL_MULT: &[usize] = &[1, 2, 4, 4]; // 320, 640, 1280, 1280
pub const NUM_HEADS: usize = 8;
pub const CONTEXT_DIM: usize = 768; // CLIP output dim

/// Timestep embedding via sinusoidal encoding.
pub fn timestep_embedding(timestep: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut emb = vec![0.0f32; dim];
    let log_base = -(10000.0f32.ln()) / (half as f32 - 1.0);

    for i in 0..half {
        let freq = (log_base * i as f32).exp();
        let angle = timestep * freq;
        emb[i] = angle.cos();
        emb[half + i] = angle.sin();
    }
    emb
}

/// Depthwise Conv2D (3×3, padding=1, stride=1).
///
/// Operates on [channels, height, width] layout.
/// Minimal implementation — no dilation, no groups beyond depthwise.
pub fn conv2d_3x3(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_channels * h * w];

    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for oh in 0..h {
                for ow in 0..w {
                    let mut sum = 0.0f32;
                    for kh in 0..3usize {
                        for kw in 0..3usize {
                            let ih = oh as isize + kh as isize - 1;
                            let iw = ow as isize + kw as isize - 1;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ic * h * w + ih as usize * w + iw as usize;
                                let w_idx = oc * in_channels * 9 + ic * 9 + kh * 3 + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    output[oc * h * w + oh * w + ow] += sum;
                }
            }
        }
        // Add bias
        let bias_val = bias[oc];
        for i in 0..(h * w) {
            output[oc * h * w + i] += bias_val;
        }
    }
    output
}

/// ResBlock: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip.
pub struct ResBlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub channels: usize,
    pub h: usize,
    pub w: usize,
}

impl ResBlockWeights {
    /// Forward pass through a ResBlock.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let c = self.channels;
        let h = self.h;
        let w = self.w;

        // GroupNorm → SiLU → Conv
        let mut x = input.to_vec();
        layers::group_norm(&mut x, 32.min(c), &self.norm1_weight, &self.norm1_bias);
        layers::silu(&mut x);
        let x = conv2d_3x3(&x, &self.conv1_weight, &self.conv1_bias, c, c, h, w);

        // GroupNorm → SiLU → Conv
        let mut x = x;
        layers::group_norm(&mut x, 32.min(c), &self.norm2_weight, &self.norm2_bias);
        layers::silu(&mut x);
        let mut x = conv2d_3x3(&x, &self.conv2_weight, &self.conv2_bias, c, c, h, w);

        // Skip connection
        for i in 0..x.len() {
            x[i] += input[i];
        }
        x
    }
}

/// Predict noise given noisy latent + text conditioning + timestep.
///
/// This is the scaffold — full implementation would chain:
/// down_blocks → mid_block → up_blocks with skip connections.
pub fn predict_noise(
    noisy_latent: &[f32],
    text_embeddings: &[f32],
    timestep: f32,
) -> Vec<f32> {
    let _t_emb = timestep_embedding(timestep, MODEL_CHANNELS);
    // Scaffold: return zero noise prediction (actual UNet weights needed)
    vec![0.0f32; noisy_latent.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_embedding_shape() {
        let emb = timestep_embedding(500.0, 320);
        assert_eq!(emb.len(), 320);
    }

    #[test]
    fn test_timestep_embedding_varies() {
        let e1 = timestep_embedding(100.0, 64);
        let e2 = timestep_embedding(900.0, 64);
        // Different timesteps should give different embeddings
        let diff: f32 = e1.iter().zip(&e2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.1, "different timesteps should differ");
    }

    #[test]
    fn test_conv2d_bias_only() {
        // Zero weights, non-zero bias
        let input = vec![0.0f32; 1 * 4 * 4]; // 1ch, 4×4
        let weight = vec![0.0f32; 1 * 1 * 9]; // 1→1, 3×3
        let bias = vec![2.0f32];
        let out = conv2d_3x3(&input, &weight, &bias, 1, 1, 4, 4);
        assert_eq!(out.len(), 16);
        assert!((out[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_predict_noise_shape() {
        let latent = vec![0.0f32; LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE];
        let text = vec![0.0f32; 77 * 768];
        let noise = predict_noise(&latent, &text, 500.0);
        assert_eq!(noise.len(), latent.len());
    }
}
