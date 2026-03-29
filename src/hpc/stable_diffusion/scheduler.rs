//! Noise scheduler — DDPM/DDIM diffusion scheduling.
//!
//! Controls the denoising process: how much noise to add/remove at each step.
//! The scheduler is model-agnostic — it just manages the noise schedule.

/// Scheduler configuration.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Total training timesteps (typically 1000).
    pub num_train_timesteps: usize,
    /// Beta schedule start.
    pub beta_start: f32,
    /// Beta schedule end.
    pub beta_end: f32,
    /// Number of inference steps (20-50 typical).
    pub num_inference_steps: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            num_inference_steps: 20,
        }
    }
}

/// DDIM scheduler (Denoising Diffusion Implicit Models).
///
/// Faster than DDPM — can skip steps. 20 steps ≈ 50 DDPM steps quality.
pub struct DdimScheduler {
    config: SchedulerConfig,
    /// Precomputed alpha cumulative products.
    alphas_cumprod: Vec<f32>,
    /// Timesteps for inference (evenly spaced subset of training timesteps).
    pub timesteps: Vec<usize>,
}

impl DdimScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let n = config.num_train_timesteps;

        // Linear beta schedule
        let betas: Vec<f32> = (0..n)
            .map(|i| {
                config.beta_start + (config.beta_end - config.beta_start) * i as f32 / (n - 1) as f32
            })
            .collect();

        // Alphas = 1 - beta
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Cumulative product of alphas
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut prod = 1.0f32;
        for &a in &alphas {
            prod *= a;
            alphas_cumprod.push(prod);
        }

        // Evenly spaced timesteps for inference
        let step_size = n / config.num_inference_steps;
        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| i * step_size)
            .collect();

        Self { config, alphas_cumprod, timesteps }
    }

    /// Single denoising step: given model noise prediction, compute x_{t-1} from x_t.
    ///
    /// Returns the updated (less noisy) latent.
    pub fn step(&self, model_output: &[f32], timestep: usize, sample: &[f32]) -> Vec<f32> {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_prod_t.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt();

        // Predict x_0 from noise prediction
        // x_0 = (x_t - sqrt(1 - alpha) * noise) / sqrt(alpha)
        let inv_sqrt_alpha = 1.0 / sqrt_alpha;

        let mut result = Vec::with_capacity(sample.len());
        for i in 0..sample.len() {
            let pred_x0 = (sample[i] - sqrt_one_minus_alpha * model_output[i]) * inv_sqrt_alpha;
            // For DDIM with eta=0 (deterministic): x_{t-1} directly from x_0
            result.push(pred_x0);
        }

        result
    }

    /// Add noise to a clean sample for a given timestep.
    pub fn add_noise(&self, original: &[f32], noise: &[f32], timestep: usize) -> Vec<f32> {
        let alpha = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha.sqrt();
        let sqrt_one_minus = (1.0 - alpha).sqrt();

        original.iter().zip(noise).map(|(&x, &n)| {
            sqrt_alpha * x + sqrt_one_minus * n
        }).collect()
    }

    /// Number of inference steps.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.num_train_timesteps, 1000);
        assert_eq!(cfg.num_inference_steps, 20);
    }

    #[test]
    fn test_ddim_timesteps() {
        let sched = DdimScheduler::new(SchedulerConfig::default());
        assert_eq!(sched.timesteps.len(), 20);
        // First timestep should be near 1000, last near 0
        assert!(sched.timesteps[0] > sched.timesteps[19]);
    }

    #[test]
    fn test_alphas_cumprod_monotonic() {
        let sched = DdimScheduler::new(SchedulerConfig::default());
        // Alphas cumprod should be monotonically decreasing
        for i in 1..sched.alphas_cumprod.len() {
            assert!(sched.alphas_cumprod[i] <= sched.alphas_cumprod[i - 1]);
        }
    }

    #[test]
    fn test_add_noise_identity_at_t0() {
        let sched = DdimScheduler::new(SchedulerConfig::default());
        let original = vec![1.0, 2.0, 3.0];
        let noise = vec![0.5, 0.5, 0.5];
        let noisy = sched.add_noise(&original, &noise, 0);
        // At t=0, alpha≈1, so noisy ≈ original
        for (o, n) in original.iter().zip(&noisy) {
            assert!((o - n).abs() < 0.1);
        }
    }

    #[test]
    fn test_step_denoises() {
        let sched = DdimScheduler::new(SchedulerConfig::default());
        let sample = vec![0.5f32; 4];
        let noise_pred = vec![0.1f32; 4];
        let result = sched.step(&noise_pred, 500, &sample);
        assert_eq!(result.len(), 4);
        // Result should be different from input
        assert!((result[0] - sample[0]).abs() > 0.001);
    }
}
