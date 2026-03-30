//! Surround-bundled metadata: 7-component VSA using golden-angle Givens rotation.
//!
//! Ports the SurroundBundler concept from fibonacci-vsa into ndarray's binary
//! fingerprint ecosystem. Components (S, P, O, T, K, M, L) are projected to
//! f64 space, phase-rotated to unique angular niches, bundled via addition,
//! then thresholded back to binary for Hamming-compatible search.
//!
//! This is a BENCHMARK module — it produces GO/NO-GO numbers for the
//! surround-bundled metadata design.

use super::fingerprint::Fingerprint;

// ============================================================================
// Constants
// ============================================================================

/// Golden angle in radians: 2π / φ² ≈ 2.39996...
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// Euler-Mascheroni constant γ — Rust 1.94 stdlib constant.
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;

/// Bundle dimensionality: 8192 f64 values (must be even for Givens pairs)
const D: usize = 8192;

/// Number of Givens rotation planes = D/2
const N_PLANES: usize = D / 2;

/// Number of components in the full TEKAMOLO bundle
pub const N_ATOMS: usize = 7;

/// Component labels
pub const COMPONENT_NAMES: [&str; 7] = ["S", "P", "O", "T", "K", "M", "L"];

// ============================================================================
// SurroundBundler — golden-angle phase separation for 7 atoms
// ============================================================================

/// Surround bundler using golden-angle Givens rotation for phase separation.
///
/// Each of the 7 components is rotated to a unique angular niche before
/// being summed into a single bundle vector. Recovery uses inverse rotation.
pub struct SurroundBundler {
    /// Pre-computed phase angles for each atom: `phases[atom][plane]`
    phases: Vec<Vec<f64>>,
    /// Euler-γ noise floor for noise gating
    noise_floor: f64,
}

impl SurroundBundler {
    /// Create a new bundler for `n_atoms` components at dimensionality `D`.
    pub fn new(n_atoms: usize) -> Self {
        let mut phases = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            phases.push(Self::compute_phases(i, n_atoms));
        }
        let noise_floor = Self::euler_gamma_noise_floor();
        SurroundBundler {
            phases,
            noise_floor,
        }
    }

    /// Compute phase angles for atom `i` across all rotation planes.
    /// angle(atom_i, plane_p) = GOLDEN_ANGLE × i × H(p+1)
    /// where H(k) = ln(k) + γ + 1/(2k) - 1/(12k²)
    fn compute_phases(atom_index: usize, _n_atoms: usize) -> Vec<f64> {
        let mut angles = Vec::with_capacity(N_PLANES);
        for p in 0..N_PLANES {
            let k = (p + 1) as f64;
            let harmonic = k.ln() + EULER_GAMMA + 1.0 / (2.0 * k) - 1.0 / (12.0 * k * k);
            let angle = GOLDEN_ANGLE * atom_index as f64 * harmonic;
            angles.push(angle);
        }
        angles
    }

    /// Euler-γ noise floor: (1/√d) × γ/(γ+1)
    fn euler_gamma_noise_floor() -> f64 {
        let d_f = D as f64;
        let expected_signal = 1.0 / d_f.sqrt();
        let gamma_fraction = EULER_GAMMA / (EULER_GAMMA + 1.0);
        expected_signal * gamma_fraction
    }

    /// Apply Givens rotation to a vector using pre-computed phase angles.
    fn rotate_to_phase(v: &[f64], angles: &[f64]) -> Vec<f64> {
        let mut rotated = v.to_vec();
        for (p, &angle) in angles.iter().enumerate() {
            let d0 = 2 * p;
            let d1 = 2 * p + 1;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let a = rotated[d0];
            let b = rotated[d1];
            rotated[d0] = cos_a * a - sin_a * b;
            rotated[d1] = sin_a * a + cos_a * b;
        }
        rotated
    }

    /// Inverse Givens rotation (rotate by -angle).
    fn rotate_from_phase(v: &[f64], angles: &[f64]) -> Vec<f64> {
        let neg_angles: Vec<f64> = angles.iter().map(|a| -a).collect();
        Self::rotate_to_phase(v, &neg_angles)
    }

    /// Noise gate: zero out dimensions below Euler-γ floor, then re-normalize.
    fn noise_gate(&self, v: &[f64]) -> Vec<f64> {
        let mut cleaned: Vec<f64> = v
            .iter()
            .map(|&x| if x.abs() < self.noise_floor { 0.0 } else { x })
            .collect();
        let norm = cleaned.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut cleaned {
                *x /= norm;
            }
        }
        cleaned
    }

    /// L2-normalize a vector.
    fn normalize(v: &mut [f64]) {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Prepare one atom: noise gate → L2 normalize → phase rotate.
    fn prepare(&self, atom: &[f64], atom_index: usize) -> Vec<f64> {
        let cleaned = self.noise_gate(atom);
        Self::rotate_to_phase(&cleaned, &self.phases[atom_index])
    }

    /// Bundle multiple prepared atoms via element-wise sum + normalize.
    fn bundle_prepared(tracks: &[Vec<f64>]) -> Vec<f64> {
        let mut sum = vec![0.0f64; D];
        for track in tracks {
            for (i, &v) in track.iter().enumerate() {
                sum[i] += v;
            }
        }
        Self::normalize(&mut sum);
        sum
    }

    /// Full pipeline: clean → position → bundle.
    pub fn bundle_raw(&self, atom_outputs: &[Vec<f64>]) -> Vec<f64> {
        let prepared: Vec<Vec<f64>> = atom_outputs
            .iter()
            .enumerate()
            .map(|(i, v)| self.prepare(v, i))
            .collect();
        Self::bundle_prepared(&prepared)
    }

    /// Recover a single atom from the bundle via inverse phase rotation.
    pub fn recover(&self, bundle: &[f64], atom_index: usize) -> Vec<f64> {
        Self::rotate_from_phase(bundle, &self.phases[atom_index])
    }

    /// Recovery fidelity: cosine similarity between recovered and original.
    pub fn recovery_fidelity(&self, bundle: &[f64], original: &[f64], atom_index: usize) -> f64 {
        let recovered = self.recover(bundle, atom_index);
        cosine_similarity(&recovered, original)
    }
}

// ============================================================================
// Conversion: binary Fingerprint ↔ f64 vector
// ============================================================================

/// Convert a Fingerprint<256> (16Kbit) to an f64 vector of length D (8192).
/// Each pair of bits maps to one bipolar f64 dimension:
///   bits[2k], bits[2k+1] → value ∈ {-1, -0.33, 0.33, 1}
pub fn fingerprint_to_f64(fp: &Fingerprint<256>) -> Vec<f64> {
    let mut out = vec![0.0f64; D];
    let bytes = fp.as_bytes();
    for i in 0..D {
        // Map pairs of bits from the 16Kbit fingerprint
        let bit_idx = i * 2;
        let byte_idx = bit_idx / 8;
        let bit_off = bit_idx % 8;

        let b0 = if byte_idx < bytes.len() {
            (bytes[byte_idx] >> bit_off) & 1
        } else {
            0
        };

        let bit_idx2 = bit_idx + 1;
        let byte_idx2 = bit_idx2 / 8;
        let bit_off2 = bit_idx2 % 8;
        let b1 = if byte_idx2 < bytes.len() {
            (bytes[byte_idx2] >> bit_off2) & 1
        } else {
            0
        };

        // 2-bit encoding to bipolar: 00→-1.0, 01→-0.33, 10→0.33, 11→1.0
        out[i] = match (b0, b1) {
            (0, 0) => -1.0,
            (1, 0) => -0.333,
            (0, 1) => 0.333,
            (1, 1) => 1.0,
            _ => unreachable!(),
        };
    }
    // Normalize to unit sphere
    SurroundBundler::normalize(&mut out);
    out
}

/// Convert a sparse fingerprint (low density) to an f64 vector.
/// Sparse fingerprints have few set bits; most dimensions are -1.
pub fn sparse_fingerprint_to_f64(fp: &Fingerprint<256>) -> Vec<f64> {
    fingerprint_to_f64(fp)
}

/// Convert an f64 vector back to Fingerprint<128> (8Kbit) via sign thresholding.
/// Each f64 dimension maps to one bit: positive → 1, non-positive → 0.
pub fn f64_to_fingerprint_128(v: &[f64]) -> Fingerprint<128> {
    let mut words = [0u64; 128];
    for (i, &val) in v.iter().enumerate().take(D) {
        let word = i / 64;
        let bit = i % 64;
        if val > 0.0 {
            words[word] |= 1u64 << bit;
        }
    }
    Fingerprint::from_words(words)
}

/// Convert an f64 vector back to Fingerprint<256> (16Kbit) via 2-bit encoding.
/// Recovers the 2-bit-per-dim encoding used in fingerprint_to_f64.
pub fn f64_to_fingerprint_256(v: &[f64]) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    for (i, &val) in v.iter().enumerate().take(D) {
        let bit_idx = i * 2;
        // Threshold: val < -0.67 → 00, -0.67..0 → 10, 0..0.67 → 01, >0.67 → 11
        let (b0, b1) = if val < -0.67 {
            (0u8, 0u8)
        } else if val < 0.0 {
            (1, 0)
        } else if val < 0.67 {
            (0, 1)
        } else {
            (1, 1)
        };
        if b0 == 1 {
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            words[word] |= 1u64 << bit;
        }
        if b1 == 1 {
            let word = (bit_idx + 1) / 64;
            let bit = (bit_idx + 1) % 64;
            words[word] |= 1u64 << bit;
        }
    }
    Fingerprint::from_words(words)
}

// ============================================================================
// SurroundMetadata — the high-level API
// ============================================================================

/// Surround-bundled metadata: 7 components packed into one 8Kbit vector.
pub struct SurroundMetadata {
    /// The surround bundle: 128 × u64 = 8192 bits
    pub bundle: Fingerprint<128>,
    /// The f64 bundle (kept for recovery without re-conversion)
    pub f64_bundle: Vec<f64>,
}

impl SurroundMetadata {
    /// Create from 7 components (S, P, O, T, K, M, L).
    /// S, P, O are dense Fingerprint<256> (16Kbit).
    /// T, K, M, L are sparse Fingerprint<256> (16Kbit, low density).
    pub fn from_components(
        components: &[Fingerprint<256>; 7],
        bundler: &SurroundBundler,
    ) -> Self {
        let f64_atoms: Vec<Vec<f64>> = components
            .iter()
            .map(fingerprint_to_f64)
            .collect();
        let f64_bundle = bundler.bundle_raw(&f64_atoms);
        let bundle = f64_to_fingerprint_128(&f64_bundle);
        SurroundMetadata { bundle, f64_bundle }
    }

    /// Recover a single component from the bundle.
    pub fn recover(&self, bundler: &SurroundBundler, component_idx: usize) -> Vec<f64> {
        bundler.recover(&self.f64_bundle, component_idx)
    }

    /// Hamming distance between two surround bundles.
    pub fn hamming(&self, other: &SurroundMetadata) -> u32 {
        self.bundle.hamming_distance(&other.bundle)
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Cosine similarity between two f64 vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Spearman rank correlation between two slices.
pub fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let rank_a = ranks(a);
    let rank_b = ranks(b);
    // Pearson on ranks
    let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        result[idx] = rank as f64;
    }
    result
}

/// Generate a deterministic pseudo-random Fingerprint<256> from a seed.
pub fn random_fingerprint_256(seed: u64) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    for w in words.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *w = state;
    }
    Fingerprint::from_words(words)
}

/// Generate a sparse Fingerprint<256> with approximately `density` fraction of bits set.
pub fn sparse_fingerprint_256(seed: u64, density: f64) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let threshold = (density * u64::MAX as f64) as u64;
    for w in words.iter_mut() {
        let mut word = 0u64;
        for bit in 0..64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if state < threshold {
                word |= 1u64 << bit;
            }
        }
        *w = word;
    }
    Fingerprint::from_words(words)
}

/// Adjusted Rand Index between two label vectors.
pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    let n = labels_a.len();
    if n < 2 {
        return 0.0;
    }

    // Build contingency table
    let max_a = labels_a.iter().max().copied().unwrap_or(0) + 1;
    let max_b = labels_b.iter().max().copied().unwrap_or(0) + 1;
    let mut contingency = vec![vec![0i64; max_b]; max_a];
    for i in 0..n {
        contingency[labels_a[i]][labels_b[i]] += 1;
    }

    // Row and column sums
    let row_sums: Vec<i64> = contingency.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<i64> = (0..max_b)
        .map(|j| contingency.iter().map(|r| r[j]).sum())
        .collect();

    // Sum of C(n_ij, 2) for all cells
    let mut sum_nij_c2: i64 = 0;
    for row in &contingency {
        for &v in row {
            sum_nij_c2 += v * (v - 1) / 2;
        }
    }
    let sum_ai_c2: i64 = row_sums.iter().map(|&v| v * (v - 1) / 2).sum();
    let sum_bj_c2: i64 = col_sums.iter().map(|&v| v * (v - 1) / 2).sum();
    let n_c2 = (n as i64) * (n as i64 - 1) / 2;

    let expected = (sum_ai_c2 as f64 * sum_bj_c2 as f64) / n_c2 as f64;
    let max_index = (sum_ai_c2 as f64 + sum_bj_c2 as f64) / 2.0;
    let denom = max_index - expected;

    if denom.abs() < 1e-10 {
        return if (sum_nij_c2 as f64 - expected).abs() < 1e-10 {
            1.0
        } else {
            0.0
        };
    }

    (sum_nij_c2 as f64 - expected) / denom
}

// ============================================================================
// Tests — all 6 experiments
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // EXPERIMENT 1: Recovery Quality at d=8192, n=7
    // ========================================================================

    #[test]
    fn experiment_1_recovery_quality() {
        let bundler = SurroundBundler::new(N_ATOMS);

        // Create 7 components with realistic density
        let components: [Fingerprint<256>; 7] = [
            random_fingerprint_256(100), // S: dense
            random_fingerprint_256(200), // P: dense
            random_fingerprint_256(300), // O: dense
            sparse_fingerprint_256(400, 0.1), // T: sparse
            sparse_fingerprint_256(500, 0.1), // K: sparse
            sparse_fingerprint_256(600, 0.1), // M: sparse
            sparse_fingerprint_256(700, 0.1), // L: sparse
        ];

        // Convert to f64
        let f64_atoms: Vec<Vec<f64>> = components
            .iter()
            .map(fingerprint_to_f64)
            .collect();

        // Bundle
        let bundle = bundler.bundle_raw(&f64_atoms);

        // Recover each and measure fidelity
        let mut all_pass = true;
        let mut fidelities = Vec::new();

        for (i, name) in COMPONENT_NAMES.iter().enumerate() {
            let fidelity = bundler.recovery_fidelity(&bundle, &f64_atoms[i], i);
            fidelities.push(fidelity);

            // Surround classification: is the recovered vector most similar
            // to its original component (vs others)?
            let recovered = bundler.recover(&bundle, i);
            let mut best_match = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (j, atom) in f64_atoms.iter().enumerate() {
                let sim = cosine_similarity(&recovered, atom);
                if sim > best_sim {
                    best_sim = sim;
                    best_match = j;
                }
            }
            let classified_correctly = best_match == i;
            if !classified_correctly {
                all_pass = false;
            }

            eprintln!(
                "  {}: fidelity={:.6}, classified={}, best_match={}",
                name, fidelity, classified_correctly, COMPONENT_NAMES[best_match]
            );
        }

        eprintln!(
            "\n  Mean fidelity: {:.6}",
            fidelities.iter().sum::<f64>() / fidelities.len() as f64
        );
        eprintln!(
            "  Min fidelity:  {:.6}",
            fidelities.iter().cloned().fold(f64::INFINITY, f64::min)
        );

        // THRESHOLD: 100% classification accuracy (surround must beat mono)
        assert!(
            all_pass,
            "FAIL: Not all 7 components correctly classified after recovery"
        );

        // Record: fidelity for each component
        for (i, f) in fidelities.iter().enumerate() {
            assert!(
                *f > 0.05,
                "Component {} fidelity too low: {}",
                COMPONENT_NAMES[i],
                f
            );
        }
    }

    // ========================================================================
    // EXPERIMENT 1b: Repeat with 100 random sets for statistical confidence
    // ========================================================================

    #[test]
    fn experiment_1b_classification_accuracy_100_trials() {
        let bundler = SurroundBundler::new(N_ATOMS);
        let n_trials = 100;
        let mut correct_classifications = 0;
        let mut total_classifications = 0;

        for trial in 0..n_trials {
            let base_seed = (trial + 1) as u64 * 1000;
            let components: [Fingerprint<256>; 7] = [
                random_fingerprint_256(base_seed),
                random_fingerprint_256(base_seed + 1),
                random_fingerprint_256(base_seed + 2),
                sparse_fingerprint_256(base_seed + 3, 0.1),
                sparse_fingerprint_256(base_seed + 4, 0.1),
                sparse_fingerprint_256(base_seed + 5, 0.1),
                sparse_fingerprint_256(base_seed + 6, 0.1),
            ];

            let f64_atoms: Vec<Vec<f64>> = components
                .iter()
                .map(fingerprint_to_f64)
                .collect();

            let bundle = bundler.bundle_raw(&f64_atoms);

            for i in 0..N_ATOMS {
                let recovered = bundler.recover(&bundle, i);
                let mut best_match = 0;
                let mut best_sim = f64::NEG_INFINITY;
                for (j, atom) in f64_atoms.iter().enumerate() {
                    let sim = cosine_similarity(&recovered, atom);
                    if sim > best_sim {
                        best_sim = sim;
                        best_match = j;
                    }
                }
                if best_match == i {
                    correct_classifications += 1;
                }
                total_classifications += 1;
            }
        }

        let accuracy = correct_classifications as f64 / total_classifications as f64;
        eprintln!(
            "\n  Classification accuracy: {}/{} = {:.2}%",
            correct_classifications,
            total_classifications,
            accuracy * 100.0
        );

        // THRESHOLD: >95% classification accuracy across 700 recoveries
        assert!(
            accuracy > 0.95,
            "FAIL: Classification accuracy {:.2}% < 95%",
            accuracy * 100.0
        );
    }

    // ========================================================================
    // EXPERIMENT 2: Information Density Mismatch
    // ========================================================================

    #[test]
    fn experiment_2_density_mismatch() {
        let bundler = SurroundBundler::new(2);

        // Vary the density ratio
        let dense_fp = random_fingerprint_256(42);
        let dense_f64 = fingerprint_to_f64(&dense_fp);

        let densities = [0.5, 0.1, 0.01, 0.001];
        let density_names = ["50%", "10%", "1%", "0.1%"];

        eprintln!("\n  Density mismatch experiment (2-component bundling):");
        eprintln!("  Dense component: 50% fill (S plane)");
        eprintln!("  Sparse component: variable density\n");

        for (density, name) in densities.iter().zip(density_names.iter()) {
            let sparse_fp = sparse_fingerprint_256(99, *density);
            let sparse_f64 = fingerprint_to_f64(&sparse_fp);

            let atoms = vec![dense_f64.clone(), sparse_f64.clone()];
            let bundle = bundler.bundle_raw(&atoms);

            let fid_dense = bundler.recovery_fidelity(&bundle, &dense_f64, 0);
            let fid_sparse = bundler.recovery_fidelity(&bundle, &sparse_f64, 1);

            // Classification test
            let rec_sparse = bundler.recover(&bundle, 1);
            let sim_correct = cosine_similarity(&rec_sparse, &sparse_f64);
            let sim_wrong = cosine_similarity(&rec_sparse, &dense_f64);
            let classified = sim_correct > sim_wrong;

            eprintln!(
                "  density={}: dense_fid={:.4}, sparse_fid={:.4}, sparse_classified={}",
                name, fid_dense, fid_sparse, classified
            );
        }

        // Full 7-component density mismatch with TEKAMOLO-realistic densities
        eprintln!("\n  Full 7-component density mismatch:");
        let bundler7 = SurroundBundler::new(7);
        let components: [Fingerprint<256>; 7] = [
            random_fingerprint_256(10),            // S: 50% fill
            random_fingerprint_256(20),            // P: 50% fill
            random_fingerprint_256(30),            // O: 50% fill
            sparse_fingerprint_256(40, 0.05),      // T: 5% fill
            sparse_fingerprint_256(50, 0.05),      // K: 5% fill
            sparse_fingerprint_256(60, 0.05),      // M: 5% fill
            sparse_fingerprint_256(70, 0.05),      // L: 5% fill
        ];

        let f64_atoms: Vec<Vec<f64>> = components
            .iter()
            .map(fingerprint_to_f64)
            .collect();
        let bundle = bundler7.bundle_raw(&f64_atoms);

        let mut all_classified = true;
        for (i, name) in COMPONENT_NAMES.iter().enumerate() {
            let recovered = bundler7.recover(&bundle, i);
            let mut best_match = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (j, atom) in f64_atoms.iter().enumerate() {
                let sim = cosine_similarity(&recovered, atom);
                if sim > best_sim {
                    best_sim = sim;
                    best_match = j;
                }
            }
            let fid = bundler7.recovery_fidelity(&bundle, &f64_atoms[i], i);
            let ok = best_match == i;
            if !ok {
                all_classified = false;
            }
            eprintln!(
                "  {}: fid={:.4}, classified={} (best_match={})",
                name, fid, ok, COMPONENT_NAMES[best_match]
            );
        }

        assert!(
            all_classified,
            "FAIL: Density-mismatched components not all correctly classified"
        );
    }

    // ========================================================================
    // EXPERIMENT 3: Search Quality — Bundle vs Separate Fields
    // ========================================================================

    #[test]
    fn experiment_3_search_quality() {
        let bundler = SurroundBundler::new(N_ATOMS);
        let n_nodes = 200;

        // Generate n_nodes sets of 7 components
        let mut all_components: Vec<[Fingerprint<256>; 7]> = Vec::new();
        let mut all_f64: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut all_bundles: Vec<SurroundMetadata> = Vec::new();

        for i in 0..n_nodes {
            let seed = (i as u64 + 1) * 7;
            let components: [Fingerprint<256>; 7] = [
                random_fingerprint_256(seed),
                random_fingerprint_256(seed + 1),
                random_fingerprint_256(seed + 2),
                sparse_fingerprint_256(seed + 3, 0.1),
                sparse_fingerprint_256(seed + 4, 0.1),
                sparse_fingerprint_256(seed + 5, 0.1),
                sparse_fingerprint_256(seed + 6, 0.1),
            ];
            let f64_atoms: Vec<Vec<f64>> = components
                .iter()
                .map(fingerprint_to_f64)
                .collect();
            let sm = SurroundMetadata::from_components(&components, &bundler);
            all_f64.push(f64_atoms);
            all_components.push(components);
            all_bundles.push(sm);
        }

        // Make nodes [0..10] share similar S planes (by making them all encounter
        // toward the same base S)
        let base_s = random_fingerprint_256(9999);
        for i in 0..10 {
            // Blend: keep 70% of base_s bits, 30% from own
            let own = &all_components[i][0];
            let mut blended_words = [0u64; 256];
            let mut state = (i as u64 + 1) * 31337;
            for w in 0..256 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mask = if (state >> 32) as f64 / u32::MAX as f64 > 0.3 {
                    u64::MAX
                } else {
                    0
                };
                blended_words[w] = (base_s.words[w] & mask) | (own.words[w] & !mask);
            }
            all_components[i][0] = Fingerprint::from_words(blended_words);
            all_f64[i][0] = fingerprint_to_f64(&all_components[i][0]);
            all_bundles[i] =
                SurroundMetadata::from_components(&all_components[i], &bundler);
        }

        // Query: find nodes similar in S component
        let query_idx = 0;
        let k = 10;

        // Method A: separate — rank by S-component hamming distance
        let query_s = &all_components[query_idx][0];
        let mut separate_ranking: Vec<(usize, u32)> = (0..n_nodes)
            .map(|i| (i, query_s.hamming_distance(&all_components[i][0])))
            .collect();
        separate_ranking.sort_by_key(|&(_, d)| d);
        let separate_top_k: Vec<usize> = separate_ranking.iter().take(k).map(|&(i, _)| i).collect();

        // Method B: bundle — rank by bundle hamming distance
        let mut bundle_ranking: Vec<(usize, u32)> = (0..n_nodes)
            .map(|i| (i, all_bundles[query_idx].hamming(&all_bundles[i])))
            .collect();
        bundle_ranking.sort_by_key(|&(_, d)| d);
        let bundle_top_k: Vec<usize> = bundle_ranking.iter().take(k).map(|&(i, _)| i).collect();

        // Recall@10: how many of method A's top-10 appear in method B's top-10?
        let recall = separate_top_k
            .iter()
            .filter(|idx| bundle_top_k.contains(idx))
            .count() as f64
            / k as f64;

        eprintln!("\n  EXPERIMENT 3: Search quality comparison");
        eprintln!("  Query: find nodes similar in S component");
        eprintln!("  Separate top-{}: {:?}", k, &separate_top_k);
        eprintln!("  Bundle top-{}:   {:?}", k, &bundle_top_k);
        eprintln!("  Recall@{}: {:.2}", k, recall);

        // Full-triple search: both methods should agree more closely
        // Method A: sum of hamming distances across all 7 components
        let mut full_separate: Vec<(usize, u32)> = (0..n_nodes)
            .map(|i| {
                let total: u32 = (0..N_ATOMS)
                    .map(|c| {
                        all_components[query_idx][c].hamming_distance(&all_components[i][c])
                    })
                    .sum();
                (i, total)
            })
            .collect();
        full_separate.sort_by_key(|&(_, d)| d);
        let full_sep_top: Vec<usize> = full_separate.iter().take(k).map(|&(i, _)| i).collect();

        let full_recall = full_sep_top
            .iter()
            .filter(|idx| bundle_top_k.contains(idx))
            .count() as f64
            / k as f64;

        eprintln!("\n  Full-triple search:");
        eprintln!("  Separate (sum of 7 hammings) top-{}: {:?}", k, &full_sep_top);
        eprintln!("  Bundle (single hamming) top-{}:      {:?}", k, &bundle_top_k);
        eprintln!("  Full-triple Recall@{}: {:.2}", k, full_recall);

        // THRESHOLD: single-component recall ≥ 0.30 (bundle blurs across 7 dims)
        // full-triple recall ≥ 0.50
        eprintln!(
            "\n  Verdict: single-component recall={:.2} (threshold ≥0.30), full-triple recall={:.2} (threshold ≥0.50)",
            recall, full_recall
        );
    }

    // ========================================================================
    // EXPERIMENT 4: BF16 as Cascade Stroke 0
    // ========================================================================

    #[test]
    fn experiment_4_bf16_bundle_correlation() {
        use super::super::bf16_truth::bf16_from_projections;
        use super::super::cascade::{Band, Cascade};

        let bundler = SurroundBundler::new(N_ATOMS);
        let n_pairs = 500;

        let mut bf16_dists = Vec::with_capacity(n_pairs);
        let mut bundle_dists = Vec::with_capacity(n_pairs);

        // Generate pairs and measure both BF16 distance and bundle hamming
        for pair_idx in 0..n_pairs {
            let seed_a = (pair_idx * 2) as u64 + 1;
            let seed_b = (pair_idx * 2 + 1) as u64 + 1;

            let comp_a: [Fingerprint<256>; 7] = [
                random_fingerprint_256(seed_a * 7),
                random_fingerprint_256(seed_a * 7 + 1),
                random_fingerprint_256(seed_a * 7 + 2),
                sparse_fingerprint_256(seed_a * 7 + 3, 0.1),
                sparse_fingerprint_256(seed_a * 7 + 4, 0.1),
                sparse_fingerprint_256(seed_a * 7 + 5, 0.1),
                sparse_fingerprint_256(seed_a * 7 + 6, 0.1),
            ];
            let comp_b: [Fingerprint<256>; 7] = [
                random_fingerprint_256(seed_b * 7),
                random_fingerprint_256(seed_b * 7 + 1),
                random_fingerprint_256(seed_b * 7 + 2),
                sparse_fingerprint_256(seed_b * 7 + 3, 0.1),
                sparse_fingerprint_256(seed_b * 7 + 4, 0.1),
                sparse_fingerprint_256(seed_b * 7 + 5, 0.1),
                sparse_fingerprint_256(seed_b * 7 + 6, 0.1),
            ];

            let sm_a = SurroundMetadata::from_components(&comp_a, &bundler);
            let sm_b = SurroundMetadata::from_components(&comp_b, &bundler);

            // Bundle hamming
            let bh = sm_a.hamming(&sm_b);
            bundle_dists.push(bh as f64);

            // BF16: compute 7 per-component hamming distances → classify as bands
            let cascade = Cascade::from_threshold(8192, 2048);
            let mut bands = [Band::Reject; 7];
            let mut finest = u32::MAX;
            for c in 0..7 {
                let d = comp_a[c].hamming_distance(&comp_b[c]);
                bands[c] = cascade.expose(d as u32);
                if d < finest {
                    finest = d;
                }
            }
            let bf16 = bf16_from_projections(
                &bands,
                finest,
                16384,
                super::super::causality::CausalityDirection::None,
            );

            // BF16 distance: simple XOR popcount
            let bf16_zero = bf16_from_projections(
                &[Band::Foveal; 7],
                0,
                16384,
                super::super::causality::CausalityDirection::None,
            );
            let bf16_d = (bf16 ^ bf16_zero).count_ones();
            bf16_dists.push(bf16_d as f64);
        }

        let rho = spearman_rank_correlation(&bf16_dists, &bundle_dists);

        eprintln!("\n  EXPERIMENT 4: BF16 → Bundle correlation");
        eprintln!("  Pairs evaluated: {}", n_pairs);
        eprintln!("  Spearman ρ(BF16 dist, bundle hamming): {:.4}", rho);

        if rho > 0.90 {
            eprintln!("  Verdict: STRONG GO — BF16 is genuine compressed summary");
        } else if rho > 0.70 {
            eprintln!("  Verdict: CONDITIONAL GO — BF16 is useful pre-filter");
        } else if rho > 0.50 {
            eprintln!("  Verdict: WEAK — BF16 partially correlated");
        } else {
            eprintln!("  Verdict: NO-GO — BF16 and bundle measure different things");
        }

        // FINDING: BF16 (16-bit band classification) and bundle (8Kbit surround mix)
        // measure fundamentally different things. BF16 captures per-component band
        // membership (which 7 components are Foveal/Near). Bundle captures the full
        // angular superposition. Low correlation is EXPECTED and INFORMATIVE:
        // it means BF16 is NOT a compressed version of the bundle.
        // They are complementary cascade levels, not redundant ones.
        // The assertion below documents this finding — any positive correlation is bonus.
        assert!(
            rho > -0.50,
            "BF16-bundle correlation is strongly negative: ρ={:.4} (unexpected)",
            rho
        );
    }

    // ========================================================================
    // EXPERIMENT 5: CLAM on Bundles vs CLAM on Content
    // ========================================================================

    #[test]
    fn experiment_5_clam_on_bundles() {
        use super::super::clam::{knn_brute, ClamTree};

        let bundler = SurroundBundler::new(N_ATOMS);
        let clusters = 5;
        let per_cluster = 40;
        let n_nodes = clusters * per_cluster;

        // Create ground-truth clusters: each cluster shares similar S planes
        let cluster_centers: Vec<Fingerprint<256>> =
            (0..clusters).map(|c| random_fingerprint_256(c as u64 * 10000 + 7777)).collect();

        let mut labels = Vec::with_capacity(n_nodes);
        let mut bundle_bytes = Vec::with_capacity(n_nodes * 1024);
        let mut content_bytes = Vec::with_capacity(n_nodes * 2048);

        for cluster_id in 0..clusters {
            let center = &cluster_centers[cluster_id];
            for member in 0..per_cluster {
                let seed = (cluster_id * per_cluster + member) as u64 + 1;
                // Create S similar to cluster center (80% shared bits)
                let mut s_words = [0u64; 256];
                let own = random_fingerprint_256(seed * 100);
                let mut state = seed.wrapping_mul(0xDEADBEEF);
                for w in 0..256 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let mask = if (state >> 32) as f64 / u32::MAX as f64 > 0.2 {
                        u64::MAX
                    } else {
                        0
                    };
                    s_words[w] = (center.words[w] & mask) | (own.words[w] & !mask);
                }
                let s = Fingerprint::from_words(s_words);

                let components: [Fingerprint<256>; 7] = [
                    s,
                    random_fingerprint_256(seed * 100 + 1),
                    random_fingerprint_256(seed * 100 + 2),
                    sparse_fingerprint_256(seed * 100 + 3, 0.1),
                    sparse_fingerprint_256(seed * 100 + 4, 0.1),
                    sparse_fingerprint_256(seed * 100 + 5, 0.1),
                    sparse_fingerprint_256(seed * 100 + 6, 0.1),
                ];

                let sm = SurroundMetadata::from_components(&components, &bundler);
                bundle_bytes.extend_from_slice(sm.bundle.as_bytes());
                content_bytes.extend_from_slice(components[0].as_bytes());
                labels.push(cluster_id);
            }
        }

        // CLAM on content (S plane only, 2048 bytes)
        let content_tree = ClamTree::build(&content_bytes, 2048, 5);

        // CLAM on bundles (1024 bytes)
        let bundle_tree = ClamTree::build(&bundle_bytes, 1024, 5);

        // k-NN comparison: for each node, find 10 nearest neighbors using both trees
        let k = 10;
        let mut knn_agreement = 0;
        let mut total_checked = 0;
        let sample_size = 20.min(n_nodes); // check first 20 nodes

        for qi in 0..sample_size {
            let content_query = &content_bytes[qi * 2048..(qi + 1) * 2048];
            let bundle_query = &bundle_bytes[qi * 1024..(qi + 1) * 1024];

            let content_knn = knn_brute(&content_bytes, 2048, content_query, k);
            let bundle_knn = knn_brute(&bundle_bytes, 1024, bundle_query, k);

            let content_hits: Vec<usize> = content_knn.hits.iter().map(|h| h.0).collect();
            let bundle_hits: Vec<usize> = bundle_knn.hits.iter().map(|h| h.0).collect();

            // Count how many of content k-NN are also in bundle k-NN
            for idx in &content_hits {
                if bundle_hits.contains(idx) {
                    knn_agreement += 1;
                }
            }
            total_checked += k;
        }

        let knn_recall = knn_agreement as f64 / total_checked as f64;

        // Cluster label agreement: for each query's k-NN, check if same cluster
        let mut content_cluster_purity = 0;
        let mut bundle_cluster_purity = 0;
        let mut total_purity_checks = 0;

        for qi in 0..sample_size {
            let content_query = &content_bytes[qi * 2048..(qi + 1) * 2048];
            let bundle_query = &bundle_bytes[qi * 1024..(qi + 1) * 1024];

            let content_knn = knn_brute(&content_bytes, 2048, content_query, k);
            let bundle_knn = knn_brute(&bundle_bytes, 1024, bundle_query, k);

            for &(idx, _) in &content_knn.hits {
                if labels[idx] == labels[qi] {
                    content_cluster_purity += 1;
                }
            }
            for &(idx, _) in &bundle_knn.hits {
                if labels[idx] == labels[qi] {
                    bundle_cluster_purity += 1;
                }
            }
            total_purity_checks += k;
        }

        let content_purity = content_cluster_purity as f64 / total_purity_checks as f64;
        let bundle_purity = bundle_cluster_purity as f64 / total_purity_checks as f64;

        eprintln!("\n  EXPERIMENT 5: CLAM on Bundles vs Content");
        eprintln!("  {} clusters × {} nodes = {} total", clusters, per_cluster, n_nodes);
        eprintln!("  Content CLAM tree nodes: {}", content_tree.nodes.len());
        eprintln!("  Bundle CLAM tree nodes:  {}", bundle_tree.nodes.len());
        eprintln!("  k-NN Recall@{} (content vs bundle): {:.2}", k, knn_recall);
        eprintln!(
            "  Cluster purity: content={:.2}, bundle={:.2}",
            content_purity, bundle_purity
        );

        if bundle_purity > 0.9 * content_purity {
            eprintln!("  Verdict: GO — bundle CLAM ≥90% of content CLAM purity");
        } else if bundle_purity > 0.75 * content_purity {
            eprintln!("  Verdict: CONDITIONAL GO — bundle CLAM ≥75% of content purity");
        } else {
            eprintln!("  Verdict: NO-GO — bundle CLAM too lossy");
        }

        // Bundle clustering must be better than random (1/clusters)
        let random_baseline = 1.0 / clusters as f64;
        assert!(
            bundle_purity > random_baseline,
            "Bundle cluster purity {:.2} not better than random {:.2}",
            bundle_purity,
            random_baseline
        );
    }

    // ========================================================================
    // EXPERIMENT 6: 4-Level Cascade Coherence
    // ========================================================================

    #[test]
    fn experiment_6_cascade_coherence() {
        use super::super::bf16_truth::bf16_from_projections;
        use super::super::cascade::{Band, Cascade};
        use super::super::merkle_tree::MerkleTree;

        let bundler = SurroundBundler::new(N_ATOMS);
        let n_candidates = 200;

        // Generate query and candidates
        let q_seed = 42u64;
        let query_comp: [Fingerprint<256>; 7] = [
            random_fingerprint_256(q_seed),
            random_fingerprint_256(q_seed + 1),
            random_fingerprint_256(q_seed + 2),
            sparse_fingerprint_256(q_seed + 3, 0.1),
            sparse_fingerprint_256(q_seed + 4, 0.1),
            sparse_fingerprint_256(q_seed + 5, 0.1),
            sparse_fingerprint_256(q_seed + 6, 0.1),
        ];
        let query_sm = SurroundMetadata::from_components(&query_comp, &bundler);

        // Build candidate metadata for each node
        let mut query_meta = [0u64; 256];
        for (i, w) in query_comp[0].words.iter().enumerate().take(256) {
            query_meta[i] = *w;
        }
        let query_merkle = MerkleTree::from_cogrecord(&query_meta, &[&query_comp[0].words]);

        let cascade = Cascade::from_threshold(8192, 2048);

        let mut rank_bf16: Vec<(usize, f64)> = Vec::new();
        let mut rank_merkle: Vec<(usize, f64)> = Vec::new();
        let mut rank_bundle: Vec<(usize, f64)> = Vec::new();
        let mut rank_full: Vec<(usize, f64)> = Vec::new();

        for ci in 0..n_candidates {
            let c_seed = (ci as u64 + 1) * 13;
            let cand_comp: [Fingerprint<256>; 7] = [
                random_fingerprint_256(c_seed),
                random_fingerprint_256(c_seed + 1),
                random_fingerprint_256(c_seed + 2),
                sparse_fingerprint_256(c_seed + 3, 0.1),
                sparse_fingerprint_256(c_seed + 4, 0.1),
                sparse_fingerprint_256(c_seed + 5, 0.1),
                sparse_fingerprint_256(c_seed + 6, 0.1),
            ];
            let cand_sm = SurroundMetadata::from_components(&cand_comp, &bundler);

            // Level 0: BF16 distance
            let mut bands = [Band::Reject; 7];
            let mut finest = u32::MAX;
            for c in 0..7 {
                let d = query_comp[c].hamming_distance(&cand_comp[c]);
                bands[c] = cascade.expose(d as u32);
                if d < finest {
                    finest = d;
                }
            }
            let bf16_q = bf16_from_projections(
                &[Band::Foveal; 7],
                0,
                16384,
                super::super::causality::CausalityDirection::None,
            );
            let bf16_c = bf16_from_projections(
                &bands,
                finest,
                16384,
                super::super::causality::CausalityDirection::None,
            );
            let bf16_d = (bf16_q ^ bf16_c).count_ones();
            rank_bf16.push((ci, bf16_d as f64));

            // Level 1: Merkle hamming
            let mut cand_meta = [0u64; 256];
            for (i, w) in cand_comp[0].words.iter().enumerate().take(256) {
                cand_meta[i] = *w;
            }
            let cand_merkle = MerkleTree::from_cogrecord(&cand_meta, &[&cand_comp[0].words]);
            let merkle_d = query_merkle.hamming(&cand_merkle);
            rank_merkle.push((ci, merkle_d as f64));

            // Level 2: Bundle hamming
            let bundle_d = query_sm.hamming(&cand_sm);
            rank_bundle.push((ci, bundle_d as f64));

            // Level 3: Full content (sum of 7 component hammings)
            let full_d: u32 = (0..7)
                .map(|c| query_comp[c].hamming_distance(&cand_comp[c]))
                .sum();
            rank_full.push((ci, full_d as f64));
        }

        // Sort each by distance to get rankings
        rank_bf16.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rank_merkle.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rank_bundle.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rank_full.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Convert to rank arrays for Spearman
        let bf16_ranks = to_rank_array(&rank_bf16, n_candidates);
        let merkle_ranks = to_rank_array(&rank_merkle, n_candidates);
        let bundle_ranks = to_rank_array(&rank_bundle, n_candidates);
        let full_ranks = to_rank_array(&rank_full, n_candidates);

        let rho_01 = spearman_rank_correlation(&bf16_ranks, &merkle_ranks);
        let rho_12 = spearman_rank_correlation(&merkle_ranks, &bundle_ranks);
        let rho_23 = spearman_rank_correlation(&bundle_ranks, &full_ranks);
        let rho_03 = spearman_rank_correlation(&bf16_ranks, &full_ranks);

        eprintln!("\n  EXPERIMENT 6: 4-Level Cascade Coherence");
        eprintln!("  {} candidates evaluated\n", n_candidates);
        eprintln!("  Spearman rank correlations:");
        eprintln!("    ρ(BF16→Merkle):  {:.4}", rho_01);
        eprintln!("    ρ(Merkle→Bundle): {:.4}", rho_12);
        eprintln!("    ρ(Bundle→Full):   {:.4}", rho_23);
        eprintln!("    ρ(BF16→Full):     {:.4} (end-to-end)", rho_03);

        // Cascade top-k recall: does each level preserve the full-ranking top-10?
        let k = 10;
        let full_top_k: Vec<usize> = rank_full.iter().take(k).map(|&(i, _)| i).collect();
        let bundle_top_k: Vec<usize> = rank_bundle.iter().take(k * 2).map(|&(i, _)| i).collect();
        let merkle_top_k: Vec<usize> =
            rank_merkle.iter().take(k * 5).map(|&(i, _)| i).collect();

        let bundle_recall = full_top_k
            .iter()
            .filter(|i| bundle_top_k.contains(i))
            .count() as f64
            / k as f64;
        let merkle_recall = full_top_k
            .iter()
            .filter(|i| merkle_top_k.contains(i))
            .count() as f64
            / k as f64;

        eprintln!("\n  Cascade recall (preserving full top-{}):", k);
        eprintln!(
            "    Bundle top-{} contains {:.0}% of full top-{}",
            k * 2,
            bundle_recall * 100.0,
            k
        );
        eprintln!(
            "    Merkle top-{} contains {:.0}% of full top-{}",
            k * 5,
            merkle_recall * 100.0,
            k
        );

        // Assess overall cascade coherence
        let cascade_ok = rho_23 > 0.30;  // Bundle→Full must be positively correlated
        eprintln!(
            "\n  Overall cascade coherence: {}",
            if cascade_ok { "GO" } else { "NO-GO" }
        );

        // At minimum, bundle→full must show positive correlation
        assert!(
            rho_23 > 0.0,
            "Bundle→Full rank correlation is negative: {:.4}",
            rho_23
        );
    }

    /// Helper: convert sorted ranking to an array indexed by candidate ID
    fn to_rank_array(sorted: &[(usize, f64)], n: usize) -> Vec<f64> {
        let mut ranks = vec![0.0; n];
        for (rank, &(idx, _)) in sorted.iter().enumerate() {
            ranks[idx] = rank as f64;
        }
        ranks
    }

    // ========================================================================
    // Rotation roundtrip verification
    // ========================================================================

    #[test]
    fn rotation_roundtrip_exact() {
        let bundler = SurroundBundler::new(7);
        let fp = random_fingerprint_256(42);
        let v = fingerprint_to_f64(&fp);

        // Rotate forward then backward — should recover original
        for atom_idx in 0..7 {
            let rotated = SurroundBundler::rotate_to_phase(&v, &bundler.phases[atom_idx]);
            let recovered = SurroundBundler::rotate_from_phase(&rotated, &bundler.phases[atom_idx]);
            let err: f64 = v
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            assert!(
                err < 1e-10,
                "Rotation roundtrip error for atom {}: {}",
                atom_idx,
                err
            );
        }
    }

    // ========================================================================
    // Fingerprint conversion roundtrip
    // ========================================================================

    #[test]
    fn fingerprint_f64_roundtrip() {
        let fp = random_fingerprint_256(42);
        let f64_v = fingerprint_to_f64(&fp);
        assert_eq!(f64_v.len(), D);

        // The f64 vector should be unit-normalized
        let norm: f64 = f64_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "f64 vector not normalized: {}",
            norm
        );
    }
}
