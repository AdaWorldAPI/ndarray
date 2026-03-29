//! VAE decoder — latent space [4, 64, 64] → RGB pixels [3, 512, 512].
//!
//! The final stage: takes denoised latents and decodes to a visible image.
//! Uses Conv2D + GroupNorm + SiLU (same as UNet but simpler architecture).

use super::unet::conv2d_3x3;
use crate::hpc::models::layers;

/// VAE configuration.
pub const VAE_LATENT_CHANNELS: usize = 4;
pub const VAE_OUT_CHANNELS: usize = 3; // RGB
pub const VAE_SCALE_FACTOR: usize = 8; // latent is 8× smaller than output

/// VAE decoder weights (simplified).
#[derive(Clone)]
pub struct VaeDecoderWeights {
    /// Post-quantization conv: [4, mid_ch, 3, 3]
    pub post_quant_conv_weight: Vec<f32>,
    pub post_quant_conv_bias: Vec<f32>,
    pub mid_channels: usize,
    /// Final conv to RGB: [mid_ch, 3, 3, 3]
    pub final_conv_weight: Vec<f32>,
    pub final_conv_bias: Vec<f32>,
}

/// Decode latent tensor to RGB image.
///
/// Input: `[4, h, w]` latent (scaled by 1/0.18215).
/// Output: `[3, h*8, w*8]` RGB pixels in [0, 1].
pub fn decode(latent: &[f32], h: usize, w: usize) -> Vec<f32> {
    let out_h = h * VAE_SCALE_FACTOR;
    let out_w = w * VAE_SCALE_FACTOR;

    // Scale latent
    let mut scaled: Vec<f32> = latent.iter().map(|&x| x / 0.18215).collect();

    // Nearest-neighbor upsample (scaffold — actual VAE uses learned upsampling)
    let mut upsampled = vec![0.0f32; VAE_OUT_CHANNELS * out_h * out_w];
    for c in 0..VAE_OUT_CHANNELS.min(VAE_LATENT_CHANNELS) {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let ih = oh / VAE_SCALE_FACTOR;
                let iw = ow / VAE_SCALE_FACTOR;
                upsampled[c * out_h * out_w + oh * out_w + ow] =
                    scaled[c * h * w + ih * w + iw];
            }
        }
    }

    // Clamp to [0, 1]
    for v in &mut upsampled {
        *v = v.clamp(0.0, 1.0);
    }

    upsampled
}

/// Convert [C, H, W] float tensor to [H, W, C] u8 RGB.
pub fn to_rgb_u8(tensor: &[f32], channels: usize, h: usize, w: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; h * w * channels];
    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let val = tensor[c * h * w + y * w + x];
                rgb[(y * w + x) * channels + c] = (val * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_shape() {
        let latent = vec![0.5f32; 4 * 8 * 8]; // 4ch, 8×8
        let output = decode(&latent, 8, 8);
        assert_eq!(output.len(), 3 * 64 * 64); // 3ch, 64×64
    }

    #[test]
    fn test_decode_clamped() {
        let latent = vec![100.0f32; 4 * 4 * 4];
        let output = decode(&latent, 4, 4);
        for &v in &output {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_to_rgb_u8() {
        let tensor = vec![0.0f32, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]; // 3ch, 1×3
        let rgb = to_rgb_u8(&tensor, 3, 1, 3);
        assert_eq!(rgb.len(), 9);
        assert_eq!(rgb[0], 0);   // R of pixel 0
        assert_eq!(rgb[1], 0);   // G of pixel 0
        assert_eq!(rgb[2], 0);   // B of pixel 0
    }

    #[test]
    fn test_to_rgb_u8_clamp() {
        let tensor = vec![-1.0f32, 2.0, 0.5]; // 3ch, 1×1
        let rgb = to_rgb_u8(&tensor, 3, 1, 1);
        assert_eq!(rgb[0], 0);   // clamped from -1
        assert_eq!(rgb[1], 255); // clamped from 2
        assert_eq!(rgb[2], 127); // 0.5 * 255 = 127.5 → 127
    }
}
