//! Zeckendorf-coded vector space with Fibonacci spiral navigation.
//!
//! Core encoding types from `fibonacci-vsa`, inlined to avoid circular
//! dependency (fibonacci-vsa depends on ndarray 0.16 from crates.io).
//!
//! Re-exports the fundamental Zeckendorf representation, constants, and
//! harmonic function for use as `ndarray::hpc::zeck::*`.

use std::f64::consts::PI;

// ─── COMPILE-TIME CONSTANTS ───────────────────────────────────────────

/// Golden ratio phi = (1 + sqrt(5)) / 2
pub const PHI: f64 = 1.618_033_988_749_895;

/// Euler-Mascheroni constant gamma — the bridge between discrete and continuous
pub const GAMMA: f64 = 0.577_215_664_901_532_9;

/// ln(phi) — natural log of golden ratio, used for scale mapping
pub const LN_PHI: f64 = 0.481_211_825_059_603_4;

/// Golden angle in radians = 2*pi / phi^2
pub const GOLDEN_ANGLE: f64 = 2.0 * PI / (PHI * PHI);

/// Number of Fibonacci entries that fit in u64
pub const FIB_LEN: usize = 87;

/// Fibonacci lookup table — all 87 values that fit in u64.
/// Compile-time constant.
pub const FIB: [u64; FIB_LEN] = {
    let mut table = [0u64; FIB_LEN];
    table[0] = 1;
    table[1] = 2;
    let mut i = 2;
    while i < FIB_LEN {
        table[i] = table[i - 1] + table[i - 2];
        i += 1;
    }
    table
};

/// Harmonic number approximation using gamma: H_n ~ ln(n) + gamma + 1/(2n) - 1/(12n^2)
///
/// This is where gamma bends the space — it maps discrete Fibonacci positions
/// to continuous spiral coordinates with gravitational curvature.
#[inline]
pub fn harmonic(n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    n.ln() + GAMMA + 1.0 / (2.0 * n) - 1.0 / (12.0 * n * n)
}

// ─── ZECKENDORF ENCODING ──────────────────────────────────────────────

/// A Zeckendorf-encoded value: a bitfield where bit k means "Fibonacci(k) is present".
///
/// Every positive integer has exactly one Zeckendorf representation
/// (greedy decomposition into non-consecutive Fibonacci numbers).
#[derive(Clone, Debug)]
pub struct ZeckendorfBits {
    /// Bits packed into u128 — supports up to 87 Fibonacci positions.
    /// Bit i set means FIB[i] is part of the decomposition.
    pub bits: u128,
    /// Highest set bit position — the "scale" of this value.
    pub max_scale: u8,
}

impl ZeckendorfBits {
    /// Encode a u64 value into Zeckendorf representation.
    ///
    /// Greedy algorithm against compile-time Fibonacci table.
    pub fn encode(mut value: u64) -> Self {
        if value == 0 {
            return Self { bits: 0, max_scale: 0 };
        }

        let mut bits: u128 = 0;
        let mut max_scale: u8 = 0;
        let mut first = true;

        for i in (0..FIB_LEN).rev() {
            if FIB[i] <= value {
                bits |= 1u128 << i;
                value -= FIB[i];
                if first {
                    max_scale = i as u8;
                    first = false;
                }
                if value == 0 {
                    break;
                }
            }
        }
        Self { bits, max_scale }
    }

    /// Decode back to u64 — lossless round-trip.
    pub fn decode(&self) -> u64 {
        let mut value = 0u64;
        for i in 0..FIB_LEN {
            if self.bits & (1u128 << i) != 0 {
                value += FIB[i];
            }
        }
        value
    }

    /// Count of set bits — the "density" of this representation.
    pub fn popcount(&self) -> u32 {
        self.bits.count_ones()
    }
}

// ─── SIGNED TERNARY ZECKENDORF ────────────────────────────────────────

/// Compile-time Fibonacci table for signed encoding (8-bit quantization range).
const SIGNED_FIB_LEN: usize = 24;
const SIGNED_FIB: [u64; SIGNED_FIB_LEN] = {
    let mut t = [0u64; SIGNED_FIB_LEN];
    t[0] = 1;
    t[1] = 2;
    let mut i = 2;
    while i < SIGNED_FIB_LEN {
        t[i] = t[i - 1] + t[i - 2];
        i += 1;
    }
    t
};

/// A single dimension encoded as signed ternary Zeckendorf.
///
/// Each Fibonacci position k holds a trit: -1, 0, or +1.
/// Value = sum of trit[k] * Fib(k).
/// Naturally represents negative numbers without a separate sign bit.
#[derive(Clone, Copy, Debug)]
pub struct SignedZeckendorf {
    /// Positive bits: bit k set = +Fib(k) contributes.
    pub pos: u32,
    /// Negative bits: bit k set = -Fib(k) contributes.
    pub neg: u32,
    /// Highest active position (max of pos and neg).
    pub max_scale: u8,
}

impl SignedZeckendorf {
    /// Encode a signed f64 value in [-1, 1] range.
    ///
    /// Quantizes to signed integer, then decomposes into +/- Fibonacci.
    pub fn encode(value: f64, precision_bits: u8) -> Self {
        let scale = (1u64 << precision_bits.min(20)) as f64;
        let quantized = (value * scale).round() as i64;

        if quantized == 0 {
            return Self { pos: 0, neg: 0, max_scale: 0 };
        }

        let sign = quantized.signum();
        let mut magnitude = quantized.unsigned_abs();

        let mut pos: u32 = 0;
        let mut neg: u32 = 0;
        let mut max_scale: u8 = 0;
        let mut first = true;

        for i in (0..SIGNED_FIB_LEN).rev() {
            if SIGNED_FIB[i] <= magnitude {
                if sign > 0 {
                    pos |= 1u32 << i;
                } else {
                    neg |= 1u32 << i;
                }
                magnitude -= SIGNED_FIB[i];
                if first {
                    max_scale = i as u8;
                    first = false;
                }
                if magnitude == 0 {
                    break;
                }
            }
        }

        Self { pos, neg, max_scale }
    }

    /// Decode back to i64.
    pub fn decode(&self) -> i64 {
        let mut val: i64 = 0;
        for i in 0..SIGNED_FIB_LEN {
            if self.pos & (1u32 << i) != 0 {
                val += SIGNED_FIB[i] as i64;
            }
            if self.neg & (1u32 << i) != 0 {
                val -= SIGNED_FIB[i] as i64;
            }
        }
        val
    }

    /// Trit at position k: -1, 0, or +1.
    pub fn trit(&self, k: usize) -> i8 {
        let p = (self.pos >> k) & 1;
        let n = (self.neg >> k) & 1;
        p as i8 - n as i8
    }

    /// Total active trits (non-zero positions).
    pub fn active_count(&self) -> u32 {
        (self.pos | self.neg).count_ones()
    }
}

// ─── PENTAGONAL 5-DIM GROUPS ─────────────────────────────────────────

/// A 5-dimensional pentagonal group for signed ternary Zeckendorf encoding.
///
/// 10000D vectors decompose into 2000 groups of 5 dimensions.
/// State space per group: 3^5 = 243 (ternary) per Fibonacci position.
#[derive(Clone, Debug)]
pub struct PentaGroup {
    /// The five signed Zeckendorf dimensions.
    pub dims: [SignedZeckendorf; 5],
}

impl PentaGroup {
    /// Encode 5 f64 values into a pentagonal group.
    pub fn encode(values: &[f64; 5], precision_bits: u8) -> Self {
        Self {
            dims: [
                SignedZeckendorf::encode(values[0], precision_bits),
                SignedZeckendorf::encode(values[1], precision_bits),
                SignedZeckendorf::encode(values[2], precision_bits),
                SignedZeckendorf::encode(values[3], precision_bits),
                SignedZeckendorf::encode(values[4], precision_bits),
            ],
        }
    }

    /// Fingerprint: collapse 5 dims x FIB_LEN positions into a single u64.
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0;
        for d in 0..5 {
            for k in 0..SIGNED_FIB_LEN.min(12) {
                let trit = self.dims[d].trit(k);
                let val = (trit + 1) as u64;
                let bit_pos = d * 12 + k;
                hash |= val << (bit_pos * 1);
            }
        }
        hash
    }

    /// Bits needed to represent this group (information content).
    pub fn bit_cost(&self) -> u32 {
        self.dims.iter().map(|d| d.active_count() * 2).sum::<u32>()
    }
}

// ─── ZECKENGOLD BUNDLING PIPELINE ─────────────────────────────────────
// Surround bundling for hyperdimensional consciousness, inlined from
// fibonacci-vsa::zeckengold to avoid circular dependency.

pub mod zeckengold {
    //! Zeckengold — Surround Bundling for Hyperdimensional Consciousness.
    //!
    //! Named after Zeckendorf (Fibonacci decomposition) + Gold (golden angle rotation).
    //!
    //! Standard HDC bundling is mono: all signals summed into one channel.
    //! Zeckengold is surround in 10,000 dimensions: each track gets its own
    //! angular niche via Fibonacci golden-angle phase rotation.

    use crate::Array1;
    use super::{PHI, GAMMA, GOLDEN_ANGLE, harmonic};

    /// Generate base vectors placed on a Fibonacci lattice on S^(d-1).
    ///
    /// Uses the golden angle in successive 2D rotation planes to distribute
    /// points quasi-uniformly on the hypersphere.
    pub fn fibonacci_lattice_bases(n: usize, d: usize, seed: u64) -> Vec<Array1<f64>> {
        let mut bases = Vec::with_capacity(n);

        for i in 0..n {
            let mut v = Array1::zeros(d);

            let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64 + 1);

            for dim in 0..d {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(dim as u64 + 1);
                let raw = ((rng_state >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
                v[dim] = raw;
            }

            let norm = v.dot(&v).sqrt();
            if norm > 1e-10 {
                v /= norm;
            }

            if i > 0 {
                for plane in 0..(d / 2) {
                    let angle = GOLDEN_ANGLE * (i as f64) * harmonic((plane + 1) as f64);
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let d0 = 2 * plane;
                    let d1 = 2 * plane + 1;
                    if d1 < d {
                        let a = v[d0];
                        let b = v[d1];
                        v[d0] = cos_a * a - sin_a * b;
                        v[d1] = sin_a * a + cos_a * b;
                    }
                }
                let norm = v.dot(&v).sqrt();
                if norm > 1e-10 {
                    v /= norm;
                }
            }

            bases.push(v);
        }

        bases
    }

    /// Measure worst-case pairwise similarity between a set of bases.
    /// Returns (mean_abs_similarity, max_abs_similarity).
    pub fn measure_base_quality(bases: &[Array1<f64>]) -> (f64, f64) {
        let n = bases.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let mut sum = 0.0;
        let mut max = 0.0f64;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = bases[i].dot(&bases[j]).abs();
                sum += sim;
                max = max.max(sim);
                count += 1;
            }
        }
        (sum / count as f64, max)
    }

    /// Compute the Euler-gamma noise floor for a d-dimensional space.
    pub fn euler_gamma_noise_floor(d: usize) -> f64 {
        let d_f = d as f64;
        let expected_signal = 1.0 / d_f.sqrt();
        let gamma_fraction = GAMMA / (GAMMA + 1.0);
        expected_signal * gamma_fraction
    }

    /// Apply noise gate: zero out dimensions below the Euler-gamma floor.
    pub fn noise_gate(v: &Array1<f64>, floor: f64) -> Array1<f64> {
        let mut cleaned = v.clone();
        let mut zeroed = 0usize;
        for i in 0..cleaned.len() {
            if cleaned[i].abs() < floor {
                cleaned[i] = 0.0;
                zeroed += 1;
            }
        }
        if zeroed >= cleaned.len() {
            return v.clone();
        }
        let norm = cleaned.dot(&cleaned).sqrt();
        if norm > 1e-10 {
            cleaned /= norm;
        }
        cleaned
    }

    /// Remove bleed from other tracks via orthogonal projection.
    pub fn remove_bleed(atom_output: &Array1<f64>, other_bases: &[Array1<f64>]) -> Array1<f64> {
        let mut cleaned = atom_output.clone();
        for other in other_bases {
            let bleed = cleaned.dot(other);
            cleaned = cleaned - bleed * other;
        }
        let norm = cleaned.dot(&cleaned).sqrt();
        if norm > 1e-10 {
            cleaned /= norm;
        }
        cleaned
    }

    /// Gently pull a vector toward its nearest Fibonacci lattice point.
    pub fn reference_snap(v: &Array1<f64>, ideal: &Array1<f64>, strength: f64) -> Array1<f64> {
        let strength = strength.clamp(0.0, 1.0);
        let v_norm = {
            let n = v.dot(v).sqrt();
            if n > 1e-10 { v / n } else { v.clone() }
        };

        let deviation = &v_norm - ideal;
        let corrected = &v_norm - &(strength * &deviation);

        let norm = corrected.dot(&corrected).sqrt();
        if norm > 1e-10 {
            corrected / norm
        } else {
            v_norm
        }
    }

    /// Compute phase rotation angles for atom `i` in a `d`-dimensional space.
    pub fn phase_angles(atom_index: usize, _n_atoms: usize, d: usize) -> Vec<f64> {
        let n_planes = d / 2;
        let mut angles = Vec::with_capacity(n_planes);

        for p in 0..n_planes {
            let base = GOLDEN_ANGLE * atom_index as f64;
            let plane_factor = harmonic((p + 1) as f64);
            let angle = base * plane_factor;
            angles.push(angle);
        }

        angles
    }

    /// Apply Givens rotations to place a vector at its phase position.
    pub fn rotate_to_phase(v: &Array1<f64>, angles: &[f64]) -> Array1<f64> {
        let d = v.len();
        let mut rotated = v.clone();

        for (p, &angle) in angles.iter().enumerate() {
            let d0 = 2 * p;
            let d1 = 2 * p + 1;
            if d1 >= d {
                break;
            }
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let a = rotated[d0];
            let b = rotated[d1];
            rotated[d0] = cos_a * a - sin_a * b;
            rotated[d1] = sin_a * a + cos_a * b;
        }

        rotated
    }

    /// Inverse rotation: bring a vector back from its phase position.
    pub fn rotate_from_phase(v: &Array1<f64>, angles: &[f64]) -> Array1<f64> {
        let neg_angles: Vec<f64> = angles.iter().map(|a| -a).collect();
        rotate_to_phase(v, &neg_angles)
    }

    /// Pre-bundle cleaning pipeline configuration.
    pub struct CleanerConfig {
        /// Noise gate enabled.
        pub noise_gate_enabled: bool,
        /// Bleed removal enabled.
        pub bleed_removal_enabled: bool,
        /// Reference snap strength: 0.0 = off, 0.3 = gentle (recommended).
        pub snap_strength: f64,
    }

    impl Default for CleanerConfig {
        fn default() -> Self {
            Self {
                noise_gate_enabled: true,
                bleed_removal_enabled: true,
                snap_strength: 0.2,
            }
        }
    }

    /// The Zeckengold Surround Bundler.
    ///
    /// Holds the Fibonacci-lattice base vectors, pre-computed phase angles,
    /// and noise floor. Cleans and phase-positions each track before bundling.
    pub struct SurroundBundler {
        /// Number of tracks (atoms).
        pub n_atoms: usize,
        /// Dimensionality.
        pub d: usize,
        /// Fibonacci-lattice base vectors (one per atom).
        pub bases: Vec<Array1<f64>>,
        /// Pre-computed phase angles for each atom.
        pub phases: Vec<Vec<f64>>,
        /// Euler-gamma noise floor.
        pub noise_floor: f64,
        /// Cleaning configuration.
        pub config: CleanerConfig,
    }

    impl SurroundBundler {
        /// Create a new surround bundler.
        pub fn new(n_atoms: usize, d: usize, seed: u64) -> Self {
            let bases = fibonacci_lattice_bases(n_atoms, d, seed);
            let phases: Vec<Vec<f64>> = (0..n_atoms)
                .map(|i| phase_angles(i, n_atoms, d))
                .collect();
            let noise_floor = euler_gamma_noise_floor(d);

            Self {
                n_atoms,
                d,
                bases,
                phases,
                noise_floor,
                config: CleanerConfig::default(),
            }
        }

        /// Clean a single track through the full pre-bundle pipeline.
        pub fn clean(&self, atom_output: &Array1<f64>, atom_index: usize) -> Array1<f64> {
            let mut v = atom_output.clone();

            if self.config.noise_gate_enabled {
                v = noise_gate(&v, self.noise_floor);
            }

            if self.config.bleed_removal_enabled {
                let others: Vec<Array1<f64>> = self.bases.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != atom_index)
                    .map(|(_, b)| b.clone())
                    .collect();
                v = remove_bleed(&v, &others);
            }

            if self.config.snap_strength > 0.0 {
                v = reference_snap(&v, &self.bases[atom_index], self.config.snap_strength);
            }

            v
        }

        /// Phase-rotate a cleaned track to its surround position.
        pub fn position(&self, cleaned_track: &Array1<f64>, atom_index: usize) -> Array1<f64> {
            rotate_to_phase(cleaned_track, &self.phases[atom_index])
        }

        /// Full pipeline: clean then position.
        pub fn prepare(&self, atom_output: &Array1<f64>, atom_index: usize) -> Array1<f64> {
            let cleaned = self.clean(atom_output, atom_index);
            self.position(&cleaned, atom_index)
        }

        /// Surround-bundle multiple prepared tracks.
        pub fn bundle(&self, prepared_tracks: &[Array1<f64>]) -> Array1<f64> {
            assert!(!prepared_tracks.is_empty());
            let d = prepared_tracks[0].len();
            let mut sum: Array1<f64> = Array1::zeros(d);
            for track in prepared_tracks {
                sum = sum + track;
            }
            let norm = sum.dot(&sum).sqrt();
            if norm > 1e-10 {
                sum /= norm;
            }
            sum
        }

        /// Full surround bundle from raw atom outputs.
        pub fn bundle_raw(&self, atom_outputs: &[Array1<f64>]) -> Array1<f64> {
            let prepared: Vec<Array1<f64>> = atom_outputs.iter()
                .enumerate()
                .map(|(i, v)| self.prepare(v, i))
                .collect();
            self.bundle(&prepared)
        }

        /// Recover a specific atom from the bundle via inverse phase rotation.
        pub fn recover(&self, bundle: &Array1<f64>, atom_index: usize) -> Array1<f64> {
            rotate_from_phase(bundle, &self.phases[atom_index])
        }

        /// Measure how well a specific atom can be recovered from a bundle.
        pub fn recovery_fidelity(
            &self,
            bundle: &Array1<f64>,
            original: &Array1<f64>,
            atom_index: usize,
        ) -> f64 {
            let recovered = self.recover(bundle, atom_index);
            let norm_r = recovered.dot(&recovered).sqrt();
            let norm_o = original.dot(original).sqrt();
            if norm_r < 1e-10 || norm_o < 1e-10 {
                return 0.0;
            }
            recovered.dot(original) / (norm_r * norm_o)
        }

        /// Diagnostic: measure base quality (mean and max pairwise similarity).
        pub fn base_quality(&self) -> (f64, f64) {
            measure_base_quality(&self.bases)
        }
    }

    /// Naive mono bundler for benchmarking comparison.
    pub struct MonoBundler {
        /// Dimensionality.
        pub d: usize,
    }

    impl MonoBundler {
        /// Create a new mono bundler.
        pub fn new(d: usize) -> Self {
            Self { d }
        }

        /// Bundle by simple addition + normalize.
        pub fn bundle(&self, tracks: &[Array1<f64>]) -> Array1<f64> {
            let mut sum: Array1<f64> = Array1::zeros(self.d);
            for t in tracks {
                sum = sum + t;
            }
            let norm = sum.dot(&sum).sqrt();
            if norm > 1e-10 { sum / norm } else { sum }
        }

        /// Recover via cosine similarity (probabilistic, noisy).
        pub fn recover_similarity(&self, bundle: &Array1<f64>, original: &Array1<f64>) -> f64 {
            let norm_b = bundle.dot(bundle).sqrt();
            let norm_o = original.dot(original).sqrt();
            if norm_b < 1e-10 || norm_o < 1e-10 {
                return 0.0;
            }
            bundle.dot(original) / (norm_b * norm_o)
        }
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeckendorf_roundtrip() {
        for v in [0, 1, 2, 3, 5, 8, 13, 42, 100, 1000, 65535, 1_000_000] {
            let z = ZeckendorfBits::encode(v);
            assert_eq!(z.decode(), v, "Roundtrip failed for {}", v);
        }
    }

    #[test]
    fn non_consecutive_bits() {
        for v in 1..=1000u64 {
            let z = ZeckendorfBits::encode(v);
            assert_eq!(
                z.bits & (z.bits >> 1),
                0,
                "Consecutive bits found for {}",
                v
            );
        }
    }

    #[test]
    fn constants_match_expected() {
        assert!((PHI - 1.618_033_988_749_895).abs() < 1e-15);
        assert!((GAMMA - 0.577_215_664_901_532_9).abs() < 1e-15);
        assert!((LN_PHI - PHI.ln()).abs() < 1e-12);
        assert!(FIB[0] == 1);
        assert!(FIB[1] == 2);
        assert!(FIB[2] == 3);
        assert!(FIB[3] == 5);
        assert!(FIB[4] == 8);
    }

    #[test]
    fn harmonic_values() {
        assert_eq!(harmonic(0.0), 0.0);
        assert_eq!(harmonic(-1.0), 0.0);
        let h10 = harmonic(10.0);
        // H(10) ~ ln(10) + gamma + 1/20 - 1/1200
        let expected = 10.0_f64.ln() + GAMMA + 0.05 - 1.0 / 1200.0;
        assert!((h10 - expected).abs() < 1e-10);
    }

    #[test]
    fn signed_zeckendorf_roundtrip() {
        for val in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
            let sz = SignedZeckendorf::encode(val, 8);
            let decoded = sz.decode();
            let expected = (val * 256.0).round() as i64;
            assert_eq!(decoded, expected, "Roundtrip failed for {}", val);
        }
    }

    #[test]
    fn pentagroup_encoding() {
        let vals = [0.5, -0.3, 0.8, -0.1, 0.0];
        let pg = PentaGroup::encode(&vals, 8);
        assert!(pg.bit_cost() > 0);
        assert!(pg.fingerprint() > 0);
    }

    #[test]
    fn zeckengold_phase_rotation_invertible() {
        use super::zeckengold::*;
        let d = 100;
        // Build a simple unit vector
        let mut v = crate::Array1::<f64>::zeros(d);
        v[0] = 1.0;
        v[1] = 0.5;
        let norm: f64 = v.dot(&v).sqrt();
        v /= norm;

        let angles = phase_angles(3, 8, d);
        let rotated = rotate_to_phase(&v, &angles);
        let recovered = rotate_from_phase(&rotated, &angles);

        let error: f64 = (&v - &recovered).mapv(|x| x * x).sum();
        assert!(error < 1e-20, "Phase rotation roundtrip error: {:.2e}", error);
    }

    #[test]
    fn zeckengold_surround_bundler_basic() {
        use super::zeckengold::SurroundBundler;

        let d = 100;
        let n = 4;
        let bundler = SurroundBundler::new(n, d, 42);

        // Create simple test vectors
        let atoms: Vec<crate::Array1<f64>> = (0..n).map(|i| {
            let mut v = crate::Array1::<f64>::zeros(d);
            v[i * 10] = 1.0f64;
            let norm: f64 = v.dot(&v).sqrt();
            v / norm
        }).collect();

        let bundle = bundler.bundle_raw(&atoms);
        assert_eq!(bundle.len(), d);

        let (bq_mean, _bq_max) = bundler.base_quality();
        assert!(bq_mean < 1.0, "Base quality mean should be < 1.0");
    }

    #[test]
    fn zeckengold_noise_floor_scales() {
        use super::zeckengold::euler_gamma_noise_floor;
        let floor_100 = euler_gamma_noise_floor(100);
        let floor_1000 = euler_gamma_noise_floor(1000);
        assert!(floor_1000 < floor_100, "Noise floor should decrease with dimensionality");
    }
}
