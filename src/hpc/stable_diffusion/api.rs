//! OpenAI-compatible API for Stable Diffusion (/v1/images/generations).
//!
//! Uses shared `models::api_types` for the response envelope.

use crate::hpc::models::api_types::{ImageData, ImageResponse};
use super::clip::ClipEncoder;
use super::scheduler::{DdimScheduler, SchedulerConfig};
use super::unet;
use super::vae;

/// Request body for /v1/images/generations.
#[derive(Clone, Debug)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    /// Pre-tokenized prompt (CLIP tokens).
    pub prompt_tokens: Vec<u32>,
    /// Image dimensions.
    pub width: usize,
    pub height: usize,
    /// Number of diffusion steps.
    pub num_steps: usize,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of images to generate.
    pub n: usize,
}

impl Default for ImageGenerationRequest {
    fn default() -> Self {
        Self {
            model: "stable-diffusion-v1-5".into(),
            prompt: String::new(),
            prompt_tokens: Vec::new(),
            width: 512,
            height: 512,
            num_steps: 20,
            guidance_scale: 7.5,
            seed: 42,
            n: 1,
        }
    }
}

/// Stable Diffusion API wrapper.
pub struct StableDiffusionApi {
    clip: ClipEncoder,
    scheduler_config: SchedulerConfig,
}

impl StableDiffusionApi {
    pub fn new(clip: ClipEncoder) -> Self {
        Self {
            clip,
            scheduler_config: SchedulerConfig::default(),
        }
    }

    /// Generate images from a text prompt.
    ///
    /// Full pipeline: CLIP encode → UNet denoise × N steps → VAE decode.
    pub fn generate(&self, req: &ImageGenerationRequest) -> ImageResponse {
        let latent_h = req.height / 8;
        let latent_w = req.width / 8;

        // CLIP encode
        let text_embeddings = self.clip.encode(&req.prompt_tokens);

        // Initialize scheduler
        let scheduler = DdimScheduler::new(SchedulerConfig {
            num_inference_steps: req.num_steps,
            ..self.scheduler_config.clone()
        });

        // Initialize latent with deterministic noise from seed
        let latent_size = 4 * latent_h * latent_w;
        let mut latent = generate_noise(latent_size, req.seed);

        // Denoising loop
        for &t in &scheduler.timesteps {
            let noise_pred = unet::predict_noise(&latent, &text_embeddings, t as f32);
            latent = scheduler.step(&noise_pred, t, &latent);
        }

        // VAE decode
        let pixels = vae::decode(&latent, latent_h, latent_w);
        let rgb = vae::to_rgb_u8(&pixels, 3, req.height, req.width);

        // Encode as base64 PNG (scaffold — actual PNG encoding would be here)
        let b64 = base64_placeholder(&rgb, req.width, req.height);

        ImageResponse {
            created: 0,
            data: vec![ImageData {
                b64_json: Some(b64),
                url: None,
                revised_prompt: Some(req.prompt.clone()),
            }],
        }
    }
}

/// Deterministic noise from seed (xoshiro256++).
fn generate_noise(size: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut noise = Vec::with_capacity(size);
    for _ in 0..size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Convert to f32 in [-1, 1] range
        let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        noise.push(bits * 2.0 - 1.0);
    }
    noise
}

/// Placeholder base64 (actual PNG encoding would need a png crate).
fn base64_placeholder(rgb: &[u8], _w: usize, _h: usize) -> String {
    format!("raw_rgb_{}_bytes", rgb.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::clip::{ClipWeights, CLIP_VOCAB_SIZE, CLIP_EMBED_DIM, CLIP_MAX_SEQ};

    fn dummy_clip() -> ClipEncoder {
        ClipEncoder::new(ClipWeights {
            token_embedding: vec![0.0; CLIP_VOCAB_SIZE * CLIP_EMBED_DIM],
            position_embedding: vec![0.0; CLIP_MAX_SEQ * CLIP_EMBED_DIM],
            layers: Vec::new(),
            ln_final_weight: vec![1.0; CLIP_EMBED_DIM],
            ln_final_bias: vec![0.0; CLIP_EMBED_DIM],
        })
    }

    #[test]
    fn test_default_request() {
        let req = ImageGenerationRequest::default();
        assert_eq!(req.width, 512);
        assert_eq!(req.height, 512);
        assert_eq!(req.num_steps, 20);
    }

    #[test]
    fn test_generate_returns_image() {
        let api = StableDiffusionApi::new(dummy_clip());
        let req = ImageGenerationRequest {
            prompt_tokens: vec![0, 1, 2],
            ..Default::default()
        };
        let resp = api.generate(&req);
        assert_eq!(resp.data.len(), 1);
        assert!(resp.data[0].b64_json.is_some());
    }

    #[test]
    fn test_deterministic_noise() {
        let n1 = generate_noise(100, 42);
        let n2 = generate_noise(100, 42);
        assert_eq!(n1, n2, "same seed should give same noise");
    }

    #[test]
    fn test_different_seeds() {
        let n1 = generate_noise(100, 42);
        let n2 = generate_noise(100, 123);
        assert_ne!(n1, n2);
    }
}
