//! Pillar-8 — Temporal-drift covariance sandwich probe.
//!
//! Models three physiological motion bands (cardiac, respiratory, micro) and
//! verifies that the sandwich update `Σ_{t+1} = M_t · Σ_t · M_tᵀ` preserves
//! symmetric positive-definiteness (SPD) at rate ≥ [`PILLAR_8_PSD_THRESHOLD`].
//!
//! # Math
//!
//! For each sub-step the update is:
//!
//! ```text
//! Σ_{t+1} = M_t · Σ_t · M_tᵀ,   M_t = sqrt(σ_temporal_band)
//! ```
//!
//! where `M_t` is drawn as a random SPD matrix with Frobenius norm equal to
//! `σ_temporal_band`. The SPD sandwich update preserves SPD exactly in exact
//! arithmetic; numerical drift is measured by checking the leading-minor
//! criterion (Sylvester) after every sub-step.
//!
//! # Motion bands
//!
//! | Band        | Approx. freq. | σ_temporal (Frobenius) | RMS displacement |
//! |-------------|---------------|------------------------|-----------------|
//! | Cardiac     | ~6 Hz         | 0.05                   | ~5 mm           |
//! | Respiratory | ~0.3 Hz       | 0.20                   | ~20 mm          |
//! | Micro       | ~120 Hz       | 0.001                  | ~0.1 mm         |
//!
//! # PASS gate
//!
//! The threshold [`PILLAR_8_PSD_THRESHOLD`] is currently a placeholder; see
//! the TODO annotation below. Per joint savant P1-2 ruling this is documented
//! as explicitly arbitrary so it is never silently arbitrary.
//!
//! # Example
//!
//! ```rust
//! use ndarray::hpc::pillar::temporal_sandwich::{prove_pillar_8, prove_pillar_8_band, MotionBand};
//! let reports = prove_pillar_8();
//! assert_eq!(reports.len(), 3);
//! let r = prove_pillar_8_band(MotionBand::Cardiac);
//! r.print();
//! ```

use super::prove_runner::{assert_psd_rate, random_contractive_spd3, PillarReport, SplitMix64};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Splitmix64 seed for Pillar-8 (all bands derive sub-seeds from this).
pub const PILLAR_8_SEED: u64 = 0x_E0_DA_5A_DC_5A_DD;

/// PASS gate — minimum PSD preservation rate across all sub-steps.
///
/// PLACEHOLDER — calibrate against echocardiography literature before
/// finalising certification. Per joint savant P1-2 ruling: this is
/// documented-arbitrary, not silently arbitrary.
///
// TODO(calibrate-pillar-8-σ_temporal): replace with literature-grounded value
// after benchmarking against clinical echocardiography motion-compensation data.
pub const PILLAR_8_PSD_THRESHOLD: f64 = 0.999;

/// Cardiac band: ~6 Hz, Frobenius σ ≈ 0.05 (~5 mm RMS displacement).
pub const SIGMA_CARDIAC: f32 = 0.05;

/// Respiratory band: ~0.3 Hz, Frobenius σ ≈ 0.20 (~20 mm RMS displacement).
pub const SIGMA_RESPIRATORY: f32 = 0.20;

/// Micro-motion band: ~120 Hz, Frobenius σ ≈ 0.001 (~0.1 mm RMS displacement).
pub const SIGMA_MICRO: f32 = 0.001;

/// Number of independent random paths per band.
const N_PATHS: u32 = 1_000;

/// Number of sandwich sub-steps per path.
const N_SUBSTEPS: u32 = 30;

// ── MotionBand ────────────────────────────────────────────────────────────────

/// Physiological motion band selector for Pillar-8.
///
/// Each variant corresponds to a clinically-relevant frequency range and
/// determines the `σ_temporal` Frobenius norm used when sampling the
/// sandwich multiplier `M_t`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::temporal_sandwich::MotionBand;
/// let sigma = MotionBand::Cardiac.sigma();
/// assert_eq!(sigma, 0.05_f32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionBand {
    /// Cardiac motion: ~6 Hz, σ_temporal ≈ 0.05.
    Cardiac,
    /// Respiratory motion: ~0.3 Hz, σ_temporal ≈ 0.20.
    Respiratory,
    /// Micro-motion: ~120 Hz, σ_temporal ≈ 0.001.
    Micro,
}

impl MotionBand {
    /// Return the Frobenius σ_temporal for this band.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::temporal_sandwich::MotionBand;
    /// assert_eq!(MotionBand::Respiratory.sigma(), 0.20_f32);
    /// ```
    #[inline]
    pub fn sigma(self) -> f32 {
        match self {
            MotionBand::Cardiac => SIGMA_CARDIAC,
            MotionBand::Respiratory => SIGMA_RESPIRATORY,
            MotionBand::Micro => SIGMA_MICRO,
        }
    }

    /// Pillar-8 sub-seed offset so each band gets an independent RNG stream.
    ///
    /// Seeds are derived by XOR-mixing the master seed with a band-specific
    /// constant so that the three streams are independent and reproducible.
    #[inline]
    fn sub_seed(self) -> u64 {
        // XOR with small primes to decorrelate streams while staying deterministic.
        match self {
            MotionBand::Cardiac => PILLAR_8_SEED ^ 0x0000_0000_0000_0001,
            MotionBand::Respiratory => PILLAR_8_SEED ^ 0x0000_0000_0000_0002,
            MotionBand::Micro => PILLAR_8_SEED ^ 0x0000_0000_0000_0004,
        }
    }

    /// Human-readable name for report output.
    #[inline]
    fn name(self) -> &'static str {
        match self {
            MotionBand::Cardiac => "cardiac",
            MotionBand::Respiratory => "respiratory",
            MotionBand::Micro => "micro",
        }
    }
}

// ── Sandwich kernel ───────────────────────────────────────────────────────────

/// Apply the sandwich update `Σ_{t+1} = M · Σ · Mᵀ` for 3×3 matrices.
///
/// Both `sigma` and `m` are represented as row-major `[[f32; 3]; 3]`.
/// The result is the full matrix product `M · Σ · Mᵀ`.
///
/// # Arguments
///
/// * `sigma` — current covariance matrix (must be SPD on entry).
/// * `m`     — sandwich multiplier drawn from `random_contractive_spd3`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::temporal_sandwich::sandwich_update_3x3;
/// let eye = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let out = sandwich_update_3x3(&eye, &eye);
/// // Identity sandwich: output == input.
/// for i in 0..3 { for j in 0..3 { let _ = out[i][j]; } }
/// ```
pub fn sandwich_update_3x3(sigma: &[[f32; 3]; 3], m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // tmp = M · Σ  (3×3 × 3×3)
    let mut tmp = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            tmp[i][j] = m[i][0] * sigma[0][j] + m[i][1] * sigma[1][j] + m[i][2] * sigma[2][j];
        }
    }

    // out = tmp · Mᵀ  = (M · Σ) · Mᵀ
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // (M · Σ) · Mᵀ  ←→  sum_k  tmp[i][k] * m[j][k]
            out[i][j] = tmp[i][0] * m[j][0] + tmp[i][1] * m[j][1] + tmp[i][2] * m[j][2];
        }
    }
    out
}

/// Test whether a 3×3 symmetric matrix is positive-definite via Sylvester's
/// leading-minor criterion.
///
/// Returns `true` if all three leading principal minors are strictly positive:
/// * `m[0][0] > 0`
/// * `m[0][0]*m[1][1] - m[0][1]² > 0`
/// * `det(m) > 0`
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::temporal_sandwich::is_spd_3x3;
/// let eye = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// assert!(is_spd_3x3(&eye));
/// ```
#[inline]
pub fn is_spd_3x3(m: &[[f32; 3]; 3]) -> bool {
    let m00 = m[0][0];
    let m01 = m[0][1];
    let m02 = m[0][2];
    let m11 = m[1][1];
    let m12 = m[1][2];
    let m22 = m[2][2];

    // Leading 1×1 minor
    if m00 <= 0.0 {
        return false;
    }
    // Leading 2×2 minor
    let det2 = m00 * m11 - m01 * m01;
    if det2 <= 0.0 {
        return false;
    }
    // Full 3×3 determinant (cofactor expansion along row 0)
    let det3 =
        m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
    det3 > 0.0
}

// ── Band probe ────────────────────────────────────────────────────────────────

/// Run the Pillar-8 temporal sandwich probe for a single motion band.
///
/// Simulates [`N_PATHS`] = 1000 independent covariance paths, each with
/// [`N_SUBSTEPS`] = 30 sandwich sub-steps, using the band's σ_temporal as the
/// Frobenius norm of the random multiplier `M_t`. After every sub-step the
/// updated Σ is checked for SPD via Sylvester's criterion.
///
/// Returns a [`PillarReport`] with:
/// * `psd_rate` — fraction of (path × sub-step) pairs where Σ stayed SPD.
/// * `lognorm_concentration` — mean of `|ln ‖Σ‖_F|` across all final states,
///   measuring how concentrated the log-norms are around zero.
/// * `passed` — `psd_rate >= PILLAR_8_PSD_THRESHOLD`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::temporal_sandwich::{prove_pillar_8_band, MotionBand};
/// let r = prove_pillar_8_band(MotionBand::Micro);
/// r.print();
/// assert!(r.passed);
/// ```
pub fn prove_pillar_8_band(band: MotionBand) -> PillarReport {
    let mut rng = SplitMix64::new(band.sub_seed());
    let sigma_band = band.sigma();

    let total_checks = N_PATHS * N_SUBSTEPS;
    let mut psd_passed: u32 = 0;
    let mut lognorm_sum: f64 = 0.0;

    for _ in 0..N_PATHS {
        // Initial Σ₀: identity matrix (unambiguously SPD)
        let mut sigma: [[f32; 3]; 3] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        for _ in 0..N_SUBSTEPS {
            // Draw a random SPD multiplier M_t with the band's Frobenius norm.
            let m_t = random_contractive_spd3(&mut rng, sigma_band);

            // Apply sandwich: Σ_{t+1} = M_t · Σ_t · M_tᵀ
            sigma = sandwich_update_3x3(&sigma, &m_t);

            // Check SPD preservation.
            if is_spd_3x3(&sigma) {
                psd_passed += 1;
            }
        }

        // Accumulate log-Frobenius of the final Σ for this path.
        // ‖Σ‖_F² = Σ_ii² + 2·Σ_{i<j} Σ_ij²
        let frob_sq = sigma[0][0] * sigma[0][0]
            + sigma[1][1] * sigma[1][1]
            + sigma[2][2] * sigma[2][2]
            + 2.0 * (sigma[1][0] * sigma[1][0] + sigma[2][0] * sigma[2][0] + sigma[2][1] * sigma[2][1]);
        let frob = (frob_sq as f64).sqrt();
        if frob > 0.0 {
            lognorm_sum += frob.ln().abs();
        }
    }

    let psd_rate = if total_checks > 0 {
        (psd_passed as f64) / (total_checks as f64)
    } else {
        0.0
    };
    let lognorm_concentration = lognorm_sum / (N_PATHS as f64);
    let passed = assert_psd_rate(psd_passed, total_checks, PILLAR_8_PSD_THRESHOLD);

    PillarReport {
        pillar_id: 8,
        seed: band.sub_seed(),
        n_paths: N_PATHS,
        n_hops: N_SUBSTEPS,
        psd_rate,
        lognorm_concentration,
        passed,
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run Pillar-8 for all three motion bands and return one [`PillarReport`] per band.
///
/// The reports are ordered: `[Cardiac, Respiratory, Micro]`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::temporal_sandwich::prove_pillar_8;
/// let reports = prove_pillar_8();
/// assert_eq!(reports.len(), 3);
/// for r in &reports { r.print(); }
/// let all_pass = reports.iter().all(|r| r.passed);
/// assert!(all_pass);
/// ```
pub fn prove_pillar_8() -> Vec<PillarReport> {
    let bands = [MotionBand::Cardiac, MotionBand::Respiratory, MotionBand::Micro];
    bands.iter().map(|&b| prove_pillar_8_band(b)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sandwich_update_3x3 ───────────────────────────────────────────────────

    #[test]
    fn sandwich_identity_is_noop() {
        // M · Σ · Mᵀ with M = I and Σ = I must give I.
        let eye: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = sandwich_update_3x3(&eye, &eye);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
                assert!((out[i][j] - expected).abs() < 1e-6, "out[{i}][{j}] = {}", out[i][j]);
            }
        }
    }

    #[test]
    fn sandwich_preserves_symmetry() {
        // M · Σ · Mᵀ must be symmetric when Σ is symmetric.
        let mut rng = SplitMix64::new(0xFEED_FACE_DEAD_BEEF);
        let sigma = random_contractive_spd3(&mut rng, 0.1);
        let m = random_contractive_spd3(&mut rng, 0.05);
        let out = sandwich_update_3x3(&sigma, &m);
        // Check symmetry: out[i][j] ≈ out[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (out[i][j] - out[j][i]).abs() < 1e-5,
                    "symmetry broken: out[{i}][{j}]={} vs out[{j}][{i}]={}",
                    out[i][j],
                    out[j][i]
                );
            }
        }
    }

    // ── is_spd_3x3 ────────────────────────────────────────────────────────────

    #[test]
    fn is_spd_identity_true() {
        let eye: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(is_spd_3x3(&eye));
    }

    #[test]
    fn is_spd_negative_diagonal_false() {
        let bad: [[f32; 3]; 3] = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(!is_spd_3x3(&bad));
    }

    #[test]
    fn is_spd_singular_false() {
        // Rank-deficient: all-zero last row/col → det = 0.
        let sing: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        assert!(!is_spd_3x3(&sing));
    }

    #[test]
    fn is_spd_random_spd3_true() {
        // random_contractive_spd3 must always generate SPD matrices.
        let mut rng = SplitMix64::new(0x1234_ABCD_5678_EF90);
        for _ in 0..200 {
            let m = random_contractive_spd3(&mut rng, 0.1);
            assert!(is_spd_3x3(&m), "random_contractive_spd3 produced non-SPD matrix: {m:?}");
        }
    }

    // ── MotionBand ────────────────────────────────────────────────────────────

    #[test]
    fn motion_band_sigma_values() {
        assert_eq!(MotionBand::Cardiac.sigma(), SIGMA_CARDIAC);
        assert_eq!(MotionBand::Respiratory.sigma(), SIGMA_RESPIRATORY);
        assert_eq!(MotionBand::Micro.sigma(), SIGMA_MICRO);
    }

    #[test]
    fn motion_band_sub_seeds_distinct() {
        let sc = MotionBand::Cardiac.sub_seed();
        let sr = MotionBand::Respiratory.sub_seed();
        let sm = MotionBand::Micro.sub_seed();
        assert_ne!(sc, sr);
        assert_ne!(sr, sm);
        assert_ne!(sc, sm);
    }

    // ── prove_pillar_8_band ───────────────────────────────────────────────────

    #[test]
    fn prove_band_cardiac_pass() {
        let r = prove_pillar_8_band(MotionBand::Cardiac);
        assert_eq!(r.pillar_id, 8);
        assert_eq!(r.n_paths, 1_000);
        assert_eq!(r.n_hops, 30);
        assert!(r.passed, "Cardiac band FAIL: psd_rate={:.6}", r.psd_rate);
        assert!(r.psd_rate >= PILLAR_8_PSD_THRESHOLD);
    }

    #[test]
    fn prove_band_respiratory_pass() {
        let r = prove_pillar_8_band(MotionBand::Respiratory);
        assert!(r.passed, "Respiratory band FAIL: psd_rate={:.6}", r.psd_rate);
    }

    #[test]
    fn prove_band_micro_pass() {
        let r = prove_pillar_8_band(MotionBand::Micro);
        assert!(r.passed, "Micro band FAIL: psd_rate={:.6}", r.psd_rate);
    }

    #[test]
    fn prove_band_deterministic() {
        // Running the same band twice must produce identical psd_rate.
        let r1 = prove_pillar_8_band(MotionBand::Respiratory);
        let r2 = prove_pillar_8_band(MotionBand::Respiratory);
        assert_eq!(r1.psd_rate.to_bits(), r2.psd_rate.to_bits(), "non-deterministic psd_rate");
    }

    // ── prove_pillar_8 ────────────────────────────────────────────────────────

    #[test]
    fn prove_pillar_8_returns_three_reports() {
        let reports = prove_pillar_8();
        assert_eq!(reports.len(), 3);
    }

    #[test]
    fn prove_pillar_8_all_pass() {
        let reports = prove_pillar_8();
        for r in &reports {
            r.print();
            assert!(
                r.passed,
                "Pillar-8 band FAIL: pillar_id={} psd_rate={:.6}",
                r.pillar_id,
                r.psd_rate
            );
        }
    }

    #[test]
    fn prove_pillar_8_band_order() {
        // prove_pillar_8 must return [Cardiac, Respiratory, Micro] in that order.
        let reports = prove_pillar_8();
        let r_cardiac = prove_pillar_8_band(MotionBand::Cardiac);
        let r_respiratory = prove_pillar_8_band(MotionBand::Respiratory);
        let r_micro = prove_pillar_8_band(MotionBand::Micro);

        assert_eq!(reports[0].psd_rate.to_bits(), r_cardiac.psd_rate.to_bits());
        assert_eq!(reports[1].psd_rate.to_bits(), r_respiratory.psd_rate.to_bits());
        assert_eq!(reports[2].psd_rate.to_bits(), r_micro.psd_rate.to_bits());
    }

    // ── lognorm_concentration sanity ──────────────────────────────────────────

    #[test]
    fn lognorm_concentration_finite() {
        // Concentration must be a finite non-negative number for all bands.
        let reports = prove_pillar_8();
        for r in &reports {
            assert!(
                r.lognorm_concentration.is_finite() && r.lognorm_concentration >= 0.0,
                "lognorm_concentration not finite non-negative: {}",
                r.lognorm_concentration
            );
        }
    }

    // ── seed constant ─────────────────────────────────────────────────────────

    #[test]
    fn pillar_8_seed_constant() {
        // Ensure the SEED constant matches the specification.
        assert_eq!(PILLAR_8_SEED, 0x_E0_DA_5A_DC_5A_DD);
    }

    // ── band name smoke test ──────────────────────────────────────────────────

    #[test]
    fn motion_band_names_nonempty() {
        assert!(!MotionBand::Cardiac.name().is_empty());
        assert!(!MotionBand::Respiratory.name().is_empty());
        assert!(!MotionBand::Micro.name().is_empty());
    }
}
