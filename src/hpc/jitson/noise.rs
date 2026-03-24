//! Noise parameter types for JIT compilation of noise functions.
//!
//! Per-octave frequency and amplitude scales are baked into compiled
//! functions as immediate values, avoiding per-sample parameter loads.
//! These types are always available (no Cranelift dependency).

/// Noise octave parameters — compiled as JIT immediates.
///
/// Per-octave frequency and amplitude scales are baked into the compiled
/// function as immediate values, avoiding per-sample parameter loads.
#[derive(Debug, Clone)]
pub struct NoiseParams {
    /// Per-octave: (frequency_scale, amplitude_scale)
    pub octaves: Vec<(f64, f64)>,
    /// Lacunarity: frequency multiplier per octave
    pub lacunarity: f64,
    /// Persistence: amplitude multiplier per octave
    pub persistence: f64,
}

impl NoiseParams {
    /// Create standard Perlin noise parameters.
    pub fn perlin(num_octaves: usize, lacunarity: f64, persistence: f64) -> Self {
        let mut octaves = Vec::with_capacity(num_octaves);
        let mut freq = 1.0;
        let mut amp = 1.0;
        for _ in 0..num_octaves {
            octaves.push((freq, amp));
            freq *= lacunarity;
            amp *= persistence;
        }
        Self { octaves, lacunarity, persistence }
    }

    /// Number of octaves.
    pub fn num_octaves(&self) -> usize {
        self.octaves.len()
    }

    /// Total amplitude sum (for normalization).
    pub fn amplitude_sum(&self) -> f64 {
        self.octaves.iter().map(|(_, a)| a.abs()).sum()
    }

    /// Evaluate noise at a point using scalar octave accumulation.
    /// This is the reference implementation that JIT-compiled code must match.
    pub fn evaluate_reference(&self, x: f64, y: f64, z: f64, base_noise: fn(f64, f64, f64) -> f64) -> f64 {
        let mut value = 0.0;
        for &(freq, amp) in &self.octaves {
            value += amp * base_noise(x * freq, y * freq, z * freq);
        }
        value
    }
}

/// Gradient vectors for 3D Perlin noise (12 edges of a cube).
pub const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

/// Simple hash-based 3D noise (deterministic, not cryptographic).
pub fn simple_noise_3d(x: f64, y: f64, z: f64) -> f64 {
    // Simple value noise for testing
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let iz = z.floor() as i64;
    let hash = (ix.wrapping_mul(73856093) ^ iy.wrapping_mul(19349663) ^ iz.wrapping_mul(83492791)) as u64;
    // Map to [-1, 1]
    (hash % 1000) as f64 / 500.0 - 1.0
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn test_noise_params_perlin() {
        let params = NoiseParams::perlin(4, 2.0, 0.5);
        assert_eq!(params.num_octaves(), 4);
        assert!((params.octaves[0].0 - 1.0).abs() < 1e-10);
        assert!((params.octaves[1].0 - 2.0).abs() < 1e-10);
        assert!((params.octaves[1].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_noise_evaluate_deterministic() {
        let params = NoiseParams::perlin(4, 2.0, 0.5);
        let v1 = params.evaluate_reference(1.0, 2.0, 3.0, simple_noise_3d);
        let v2 = params.evaluate_reference(1.0, 2.0, 3.0, simple_noise_3d);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_amplitude_sum() {
        let params = NoiseParams::perlin(4, 2.0, 0.5);
        let sum = params.amplitude_sum();
        // 1.0 + 0.5 + 0.25 + 0.125 = 1.875
        assert!((sum - 1.875).abs() < 1e-10);
    }
}
