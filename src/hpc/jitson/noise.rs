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

/// Precomputed noise configuration for JIT compilation.
///
/// All per-octave parameters are flattened into arrays for direct
/// embedding as immediates in generated code. This is the "shopping list"
/// that tells the Cranelift backend what immediates to bake.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::jitson::noise::{NoiseParams, CompiledNoiseConfig, simple_noise_3d};
///
/// let params = NoiseParams::perlin(4, 2.0, 0.5);
/// let config = CompiledNoiseConfig::from_params(&params, 42);
/// let v1 = params.evaluate_reference(1.0, 2.0, 3.0, simple_noise_3d);
/// let v2 = config.evaluate(1.0, 2.0, 3.0, simple_noise_3d);
/// assert!((v1 - v2).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct CompiledNoiseConfig {
    /// Per-octave frequency scale (one per octave).
    pub frequencies: Vec<f64>,
    /// Per-octave amplitude scale (one per octave).
    pub amplitudes: Vec<f64>,
    /// Per-octave seed perturbation offset.
    pub seed_offsets: Vec<u64>,
    /// Normalization factor: `1.0 / amplitude_sum`.
    pub normalization: f64,
}

impl CompiledNoiseConfig {
    /// Build a compiled config from noise parameters and a seed.
    ///
    /// Each octave gets a unique seed offset derived from the base seed
    /// so that different octaves sample from different noise gradients.
    pub fn from_params(params: &NoiseParams, seed: u64) -> Self {
        let mut frequencies = Vec::with_capacity(params.num_octaves());
        let mut amplitudes = Vec::with_capacity(params.num_octaves());
        let mut seed_offsets = Vec::with_capacity(params.num_octaves());

        for (i, &(freq, amp)) in params.octaves.iter().enumerate() {
            frequencies.push(freq);
            amplitudes.push(amp);
            // Simple hash: seed XOR (octave_index * golden ratio constant)
            seed_offsets.push(seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        }

        let amp_sum = params.amplitude_sum();
        let normalization = if amp_sum > 0.0 { 1.0 / amp_sum } else { 1.0 };

        Self { frequencies, amplitudes, seed_offsets, normalization }
    }

    /// Evaluate using the compiled config (reference, matches what JIT would produce).
    pub fn evaluate(&self, x: f64, y: f64, z: f64, base_noise: fn(f64, f64, f64) -> f64) -> f64 {
        let mut value = 0.0;
        for i in 0..self.frequencies.len() {
            let freq = self.frequencies[i];
            value += self.amplitudes[i] * base_noise(x * freq, y * freq, z * freq);
        }
        value
    }

    /// Evaluate and normalize to [-1, 1] range.
    pub fn evaluate_normalized(
        &self,
        x: f64,
        y: f64,
        z: f64,
        base_noise: fn(f64, f64, f64) -> f64,
    ) -> f64 {
        self.evaluate(x, y, z, base_noise) * self.normalization
    }

    /// Number of octaves.
    pub fn num_octaves(&self) -> usize {
        self.frequencies.len()
    }
}

/// Baked biome parameters for JIT terrain fill.
///
/// Combines biome-specific constants (heights, block types) with noise
/// parameters, creating a self-contained recipe that can be compiled into
/// a native terrain fill loop.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::jitson::noise::{TerrainFillParams, NoiseParams, simple_noise_3d};
///
/// let params = TerrainFillParams {
///     base_height: 64,
///     height_variation: 8.0,
///     surface_block: 1,    // grass
///     subsurface_block: 3, // dirt
///     fill_block: 4,       // stone
///     biome_noise: NoiseParams::perlin(4, 2.0, 0.5),
/// };
/// let section = params.fill_section_reference(4, 42, simple_noise_3d);
/// assert_eq!(section.len(), 4096); // 16^3
/// ```
#[derive(Debug, Clone)]
pub struct TerrainFillParams {
    /// Base terrain height in blocks (Y coordinate).
    pub base_height: i32,
    /// Maximum height variation from noise (blocks).
    pub height_variation: f64,
    /// Block state ID for the surface layer (e.g., grass).
    pub surface_block: u16,
    /// Block state ID for subsurface layers (e.g., dirt, 3 blocks deep).
    pub subsurface_block: u16,
    /// Block state ID for the fill (e.g., stone).
    pub fill_block: u16,
    /// Noise parameters for terrain height variation.
    pub biome_noise: NoiseParams,
}

impl TerrainFillParams {
    /// Reference terrain fill: for each (x, z) column in a 16x16 section,
    /// compute height from noise, then fill block states top-down.
    ///
    /// Output: 16 * 16 * 16 = 4096 block state IDs (Y-major ordering:
    /// index = y * 256 + z * 16 + x).
    ///
    /// Block state ID 0 = air.
    pub fn fill_section_reference(
        &self,
        section_y: i32,
        seed: u64,
        base_noise: fn(f64, f64, f64) -> f64,
    ) -> Vec<u16> {
        let mut blocks = vec![0u16; 4096]; // all air initially
        let section_base_y = section_y * 16;

        for z in 0..16 {
            for x in 0..16 {
                // Compute terrain height for this column using noise
                let nx = x as f64 / 16.0 + (seed as f64 * 0.001);
                let nz = z as f64 / 16.0 + (seed as f64 * 0.0013);
                let noise_val = self.biome_noise.evaluate_reference(nx, 0.0, nz, base_noise);
                let height = self.base_height + (noise_val * self.height_variation) as i32;

                for y in 0..16 {
                    let world_y = section_base_y + y;
                    let idx = (y as usize) * 256 + (z as usize) * 16 + (x as usize);

                    if world_y > height {
                        // Air (already 0)
                    } else if world_y == height {
                        blocks[idx] = self.surface_block;
                    } else if world_y >= height - 3 {
                        blocks[idx] = self.subsurface_block;
                    } else {
                        blocks[idx] = self.fill_block;
                    }
                }
            }
        }

        blocks
    }
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

    // ---------- CompiledNoiseConfig ----------

    #[test]
    fn test_compiled_noise_matches_reference() {
        let params = NoiseParams::perlin(4, 2.0, 0.5);
        let config = CompiledNoiseConfig::from_params(&params, 42);

        for i in 0..10 {
            let x = i as f64 * 0.7;
            let y = i as f64 * 0.3;
            let z = i as f64 * 1.1;
            let ref_val = params.evaluate_reference(x, y, z, simple_noise_3d);
            let compiled_val = config.evaluate(x, y, z, simple_noise_3d);
            assert!(
                (ref_val - compiled_val).abs() < 1e-10,
                "mismatch at ({x},{y},{z}): ref={ref_val} compiled={compiled_val}"
            );
        }
    }

    #[test]
    fn test_compiled_noise_num_octaves() {
        let params = NoiseParams::perlin(6, 2.0, 0.5);
        let config = CompiledNoiseConfig::from_params(&params, 0);
        assert_eq!(config.num_octaves(), 6);
        assert_eq!(config.frequencies.len(), 6);
        assert_eq!(config.amplitudes.len(), 6);
        assert_eq!(config.seed_offsets.len(), 6);
    }

    #[test]
    fn test_compiled_noise_normalization() {
        let params = NoiseParams::perlin(4, 2.0, 0.5);
        let config = CompiledNoiseConfig::from_params(&params, 0);
        // normalization = 1.0 / 1.875
        assert!((config.normalization - 1.0 / 1.875).abs() < 1e-10);
    }

    #[test]
    fn test_compiled_noise_seed_offsets_unique() {
        let config = CompiledNoiseConfig::from_params(&NoiseParams::perlin(8, 2.0, 0.5), 42);
        // All seed offsets should be unique
        let mut seen = std::collections::HashSet::new();
        for &offset in &config.seed_offsets {
            assert!(seen.insert(offset), "duplicate seed offset");
        }
    }

    // ---------- TerrainFillParams ----------

    #[test]
    fn test_terrain_fill_section_size() {
        let params = TerrainFillParams {
            base_height: 64,
            height_variation: 8.0,
            surface_block: 1,
            subsurface_block: 3,
            fill_block: 4,
            biome_noise: NoiseParams::perlin(4, 2.0, 0.5),
        };
        let section = params.fill_section_reference(4, 42, simple_noise_3d);
        assert_eq!(section.len(), 4096);
    }

    #[test]
    fn test_terrain_fill_deterministic() {
        let params = TerrainFillParams {
            base_height: 64,
            height_variation: 8.0,
            surface_block: 1,
            subsurface_block: 3,
            fill_block: 4,
            biome_noise: NoiseParams::perlin(4, 2.0, 0.5),
        };
        let s1 = params.fill_section_reference(4, 42, simple_noise_3d);
        let s2 = params.fill_section_reference(4, 42, simple_noise_3d);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_terrain_fill_has_blocks() {
        let params = TerrainFillParams {
            base_height: 64,
            height_variation: 4.0,
            surface_block: 1,
            subsurface_block: 3,
            fill_block: 4,
            biome_noise: NoiseParams::perlin(2, 2.0, 0.5),
        };
        // Section at y=3 (blocks 48-63) should have mostly solid blocks
        // since base_height is 64
        let section = params.fill_section_reference(3, 42, simple_noise_3d);
        let non_air = section.iter().filter(|&&b| b != 0).count();
        assert!(non_air > 0, "section below terrain should have non-air blocks");
    }

    #[test]
    fn test_terrain_fill_above_ground_is_air() {
        let params = TerrainFillParams {
            base_height: 32,
            height_variation: 2.0,
            surface_block: 1,
            subsurface_block: 3,
            fill_block: 4,
            biome_noise: NoiseParams::perlin(2, 2.0, 0.5),
        };
        // Section at y=10 (blocks 160-175) should be all air since
        // base_height + variation is far below
        let section = params.fill_section_reference(10, 42, simple_noise_3d);
        let non_air = section.iter().filter(|&&b| b != 0).count();
        assert_eq!(non_air, 0, "section well above terrain should be all air");
    }

    #[test]
    fn test_terrain_fill_block_types() {
        let params = TerrainFillParams {
            base_height: 64,
            height_variation: 0.0, // flat terrain at y=64
            surface_block: 10,
            subsurface_block: 20,
            fill_block: 30,
            biome_noise: NoiseParams::perlin(1, 1.0, 1.0),
        };
        // Section at y=3 (blocks 48-63), flat terrain at y=64
        let section = params.fill_section_reference(3, 0, simple_noise_3d);
        // With zero variation + simple noise, all columns should have same height
        // Check that we see surface, subsurface, and fill blocks
        let has_fill = section.iter().any(|&b| b == 30);
        assert!(has_fill, "should have fill blocks below surface");
    }
}
