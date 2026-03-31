//! #16 Prompt Scaffold Optimization — evolutionary style parameter tuning.
//! Science: Hansen & Ostermeier (2001) CMA-ES, Sutton & Barto (2018).

pub struct FieldModulation {
    pub resonance_threshold: f64,
    pub fan_out: usize,
    pub depth_bias: f64,
    pub breadth_bias: f64,
    pub noise_tolerance: f64,
    pub speed_bias: f64,
    pub exploration: f64,
}

impl Default for FieldModulation {
    fn default() -> Self {
        Self { resonance_threshold: 0.7, fan_out: 6, depth_bias: 0.5, breadth_bias: 0.5, noise_tolerance: 0.3, speed_bias: 0.5, exploration: 0.3 }
    }
}

pub fn evolve_style(parent: &FieldModulation, task_reward: f32, mutation_rate: f32, seed: u64) -> FieldModulation {
    let noise = |seed_offset: u64| -> f64 {
        let mut s = seed.wrapping_add(seed_offset);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as i32) as f64 / (i32::MAX as f64)
    };
    let mr = mutation_rate as f64 * (1.0 - task_reward as f64);
    FieldModulation {
        resonance_threshold: (parent.resonance_threshold + noise(1) * mr).clamp(0.0, 1.0),
        fan_out: ((parent.fan_out as f64 + noise(2) * mr * 5.0).round() as usize).clamp(1, 20),
        depth_bias: (parent.depth_bias + noise(3) * mr).clamp(0.0, 1.0),
        breadth_bias: (parent.breadth_bias + noise(4) * mr).clamp(0.0, 1.0),
        noise_tolerance: (parent.noise_tolerance + noise(5) * mr).clamp(0.0, 1.0),
        speed_bias: (parent.speed_bias + noise(6) * mr).clamp(0.0, 1.0),
        exploration: (parent.exploration + noise(7) * mr).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_evolve_high_reward() {
        let parent = FieldModulation::default();
        let child = evolve_style(&parent, 0.95, 0.1, 42);
        // High reward -> small mutations
        assert!((child.resonance_threshold - parent.resonance_threshold).abs() < 0.1);
    }
    #[test]
    fn test_evolve_low_reward() {
        let parent = FieldModulation::default();
        let child = evolve_style(&parent, 0.1, 0.5, 42);
        // Low reward -> bigger mutations (but still bounded)
        assert!(child.resonance_threshold >= 0.0 && child.resonance_threshold <= 1.0);
    }
}
