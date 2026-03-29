//! Stable Diffusion API — wraps the pipeline with OpenAI-compatible types.
//!
//! Endpoint: `/v1/images/generations`

use crate::hpc::models::api_types::*;
use super::clip::ClipEncoder;
use super::scheduler::{DdimScheduler, SchedulerConfig};
use super::unet;
use super::vae;

/// Stable Diffusion API wrapper.
pub struct StableDiffusionApi {
    clip: ClipEncoder,
    scheduler_config: SchedulerConfig,
}

impl StableDiffusionApi {
    pub fn new(clip: ClipEncoder) -> Self {
        Self { clip, scheduler_config: SchedulerConfig::default() }
    }

    /// `/v1/images/generations`
    pub fn generate(&self, req: &ImageGenerationRequest) -> ImageResponse {
        let (w, h) = req.dimensions();
        let latent_h = h / 8;
        let latent_w = w / 8;
        let n = req.n.unwrap_or(1);
        let seed = req.seed.unwrap_or(42);
        let prompt_tokens = req.prompt_tokens.as_deref().unwrap_or(&[]);

        let text_embeddings = self.clip.encode(prompt_tokens);

        let scheduler = DdimScheduler::new(SchedulerConfig {
            num_inference_steps: 20,
            ..self.scheduler_config.clone()
        });

        let mut images = Vec::with_capacity(n);
        for img_idx in 0..n {
            let latent_size = 4 * latent_h * latent_w;
            let mut latent = generate_noise(latent_size, seed.wrapping_add(img_idx as u64));

            for &t in &scheduler.timesteps {
                let noise_pred = unet::predict_noise(&latent, &text_embeddings, t as f32);
                latent = scheduler.step(&noise_pred, t, &latent);
            }

            let pixels = vae::decode(&latent, latent_h, latent_w);
            let rgb = vae::to_rgb_u8(&pixels, 3, h, w);
            let b64 = base64_placeholder(&rgb);

            images.push(ImageData {
                b64_json: Some(b64),
                url: None,
                revised_prompt: Some(req.prompt.clone()),
            });
        }

        ImageResponse { created: 0, data: images }
    }

    /// `/v1/models/{id}`
    pub fn model_info() -> Model {
        Model::new("stable-diffusion-v1-5", "stabilityai", 0)
    }
}

fn generate_noise(size: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut noise = Vec::with_capacity(size);
    for _ in 0..size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        noise.push(bits * 2.0 - 1.0);
    }
    noise
}

fn base64_placeholder(rgb: &[u8]) -> String {
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
        assert_eq!(req.dimensions(), (512, 512));
        assert_eq!(req.n, Some(1));
    }

    #[test]
    fn test_generate_returns_image() {
        let api = StableDiffusionApi::new(dummy_clip());
        let req = ImageGenerationRequest {
            prompt: "a cat".into(),
            prompt_tokens: Some(vec![0, 1, 2]),
            ..Default::default()
        };
        let resp = api.generate(&req);
        assert_eq!(resp.data.len(), 1);
        assert!(resp.data[0].b64_json.is_some());
        assert_eq!(resp.data[0].revised_prompt.as_deref(), Some("a cat"));
    }

    #[test]
    fn test_generate_multiple() {
        let api = StableDiffusionApi::new(dummy_clip());
        let req = ImageGenerationRequest {
            prompt: "test".into(),
            prompt_tokens: Some(vec![0]),
            n: Some(3),
            ..Default::default()
        };
        let resp = api.generate(&req);
        assert_eq!(resp.data.len(), 3);
    }

    #[test]
    fn test_deterministic() {
        let n1 = generate_noise(100, 42);
        let n2 = generate_noise(100, 42);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_model_info() {
        let m = StableDiffusionApi::model_info();
        assert_eq!(m.id, "stable-diffusion-v1-5");
        assert_eq!(m.object, "model");
    }
}
