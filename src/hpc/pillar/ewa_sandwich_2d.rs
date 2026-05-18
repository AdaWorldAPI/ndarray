//! Pillar-6: 2D EWA-sandwich certification probe.
//!
//! Certifies that the exponentially-weighted average (EWA) covariance update
//! rule for 2×2 matrices preserves symmetric positive-definiteness (SPD) at
//! rate ≥ 0.999 across 1 000 independent paths of 10 hops each.
//!
//! # EWA sandwich update
//!
//! Given a current covariance Σ ∈ SPD(2) and a contractive step matrix
//! M ∈ SPD(2) with ‖M‖_F = σ_step = 0.2, the update is:
//!
//! ```text
//!   Σ_{n+1} = M · Σ_n · Mᵀ
//! ```
//!
//! Because M is symmetric (M = Mᵀ), this simplifies to:
//!
//! ```text
//!   Σ_{n+1} = M · Σ_n · M
//! ```
//!
//! A product of the form A·B·Aᵀ with B SPD and A any non-singular matrix is
//! always SPD; since M is SPD (hence non-singular), the update preserves SPD
//! exactly in exact arithmetic. Floating-point rounding can break this near
//! near-singular inputs; the probe verifies empirical rate ≥ 0.999.
//!
//! # Log-norm Frobenius concentration
//!
//! After each hop the Frobenius norm of the new Σ is recorded.  The probe
//! reports `lognorm_concentration = std(log ‖Σ‖_F) / |mean(log ‖Σ‖_F)|`
//! (relative standard deviation in log space), a dimensionless measure of
//! how tightly the cascade concentrates around its geometric mean.
//!
//! # PASS criteria (Pillar-6)
//!
//! | Criterion | Threshold |
//! |-----------|-----------|
//! | PSD rate across 1 000 paths × 10 hops | ≥ 0.999 |
//!
//! # Seed
//!
//! `PILLAR_6_SEED = 0x_DA_5A_DC_5A_DD`

use crate::hpc::linalg::Spd2;
use crate::hpc::pillar::prove_runner::{assert_psd_rate, random_contractive_spd2, PillarReport, SplitMix64};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Deterministic RNG seed for Pillar-6 — anchored in the design doc.
pub const PILLAR_6_SEED: u64 = 0x_DA_5A_DC_5A_DD;

/// PSD preservation rate threshold for Pillar-6 PASS.
pub const PILLAR_6_PSD_THRESHOLD: f64 = 0.999;

/// Frobenius norm of each random step matrix M; controls contractivity.
const SIGMA_STEP: f32 = 0.2;

/// Number of independent random paths per probe run.
const N_PATHS: u32 = 1_000;

/// Number of EWA-sandwich hops per path.
const N_HOPS: u32 = 10;

/// SPD-check epsilon — eigenvalue must exceed this to count as PD.
const SPD_EPS: f32 = 1e-9;

// ── 2×2 sandwich kernel ───────────────────────────────────────────────────────

/// Compute the EWA sandwich update `Σ_new = M · Σ · Mᵀ` for 2×2 SPD matrices.
///
/// Both `m` and `sigma` are symmetric, so `Mᵀ = M` and the formula reduces to
/// `M · Σ · M`. The result is also symmetric and (in exact arithmetic) SPD
/// whenever M is non-singular and Σ is SPD.
///
/// The computation is fully inlined via the explicit entries:
///
/// ```text
///   Let T = M · Σ  (2×2 dense product, using symmetry of Σ):
///     T[0][0] = m11·σ11 + m12·σ12
///     T[0][1] = m11·σ12 + m12·σ22
///     T[1][0] = m12·σ11 + m22·σ12
///     T[1][1] = m12·σ12 + m22·σ22
///
///   Then Σ_new = T · M (multiply T on the right by M):
///     σ11_new = T[0][0]·m11 + T[0][1]·m12
///     σ12_new = T[0][0]·m12 + T[0][1]·m22
///     σ22_new = T[1][0]·m12 + T[1][1]·m22
/// ```
///
/// # Arguments
///
/// * `m`     – contractive SPD step matrix M.
/// * `sigma` – current covariance Σ_n.
///
/// # Returns
///
/// Updated covariance Σ_{n+1} = M · Σ_n · M.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::ewa_sandwich_2d::ewa_sandwich_step_2d;
/// use ndarray::hpc::linalg::Spd2;
/// let m = Spd2::new(0.1, 0.0, 0.1);
/// let sigma = Spd2::I;
/// let sigma_new = ewa_sandwich_step_2d(&m, &sigma);
/// // sigma_new ≈ 0.01 * I (contractive scaling)
/// assert!((sigma_new.a11 - 0.01).abs() < 1e-7);
/// assert!((sigma_new.a22 - 0.01).abs() < 1e-7);
/// assert!(sigma_new.a12.abs() < 1e-7);
/// ```
#[inline]
pub fn ewa_sandwich_step_2d(m: &Spd2, sigma: &Spd2) -> Spd2 {
    let m11 = m.a11;
    let m12 = m.a12;
    let m22 = m.a22;

    let s11 = sigma.a11;
    let s12 = sigma.a12;
    let s22 = sigma.a22;

    // T = M · Σ  (using Σ symmetric: Σ[1][0] = Σ[0][1] = s12)
    let t00 = m11 * s11 + m12 * s12;
    let t01 = m11 * s12 + m12 * s22;
    let t10 = m12 * s11 + m22 * s12;
    let t11 = m12 * s12 + m22 * s22;

    // Σ_new = T · M  (M symmetric: M[1][0] = M[0][1] = m12)
    let s11_new = t00 * m11 + t01 * m12;
    let s12_new = t00 * m12 + t01 * m22;
    let s22_new = t10 * m12 + t11 * m22;

    Spd2::new(s11_new, s12_new, s22_new)
}

// ── Probe ─────────────────────────────────────────────────────────────────────

/// Run the Pillar-6 certification probe.
///
/// Simulates 1 000 independent EWA-cascade paths of 10 hops each, starting
/// from Σ₀ = I (the 2×2 identity covariance). At each hop a fresh random
/// contractive SPD matrix M is drawn from `random_contractive_spd2` with
/// ‖M‖_F = 0.2 and applied as:
///
/// ```text
///   Σ_{n+1} = M · Σ_n · M
/// ```
///
/// After each hop the probe checks whether Σ_{n+1} is SPD (via Sylvester
/// criterion with eps = 1e-9) and records ‖Σ_{n+1}‖_F in log space.
///
/// PASS if psd_rate ≥ [`PILLAR_6_PSD_THRESHOLD`] = 0.999.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::ewa_sandwich_2d::prove_pillar_6;
/// let report = prove_pillar_6();
/// report.print();
/// assert!(report.passed, "Pillar-6 must PASS");
/// ```
pub fn prove_pillar_6() -> PillarReport {
    let mut rng = SplitMix64::new(PILLAR_6_SEED);

    let total_hops = N_PATHS * N_HOPS;
    let mut psd_passed: u32 = 0;

    // Log-norm Frobenius tracking (two-pass Welford for numerical stability).
    // We accumulate: n, mean_ln, M2_ln (for variance of log-norms).
    let mut ln_count: u32 = 0;
    let mut ln_mean: f64 = 0.0;
    let mut ln_m2: f64 = 0.0;

    for _ in 0..N_PATHS {
        // Each path starts from the identity covariance.
        let mut sigma = Spd2::I;

        for _ in 0..N_HOPS {
            // Draw a random contractive SPD step matrix.
            let m_arr = random_contractive_spd2(&mut rng, SIGMA_STEP);
            let m = Spd2::new(m_arr[0][0], m_arr[0][1], m_arr[1][1]);

            // Apply sandwich update.
            sigma = ewa_sandwich_step_2d(&m, &sigma);

            // PSD check.
            if sigma.is_symmetric_pd(SPD_EPS) {
                psd_passed += 1;
            }

            // Track log Frobenius norm (Welford online mean/variance).
            let frob = sigma.frobenius_sq().sqrt() as f64;
            if frob > 0.0 {
                let ln_frob = frob.ln();
                ln_count += 1;
                let delta = ln_frob - ln_mean;
                ln_mean += delta / ln_count as f64;
                let delta2 = ln_frob - ln_mean;
                ln_m2 += delta * delta2;
            }
        }
    }

    let psd_rate = psd_passed as f64 / total_hops as f64;

    // lognorm_concentration = std(log ‖Σ‖_F) / |mean(log ‖Σ‖_F)|
    let lognorm_concentration = if ln_count >= 2 && ln_mean.abs() > 1e-15 {
        let variance = ln_m2 / (ln_count - 1) as f64;
        variance.sqrt() / ln_mean.abs()
    } else {
        0.0
    };

    let passed = assert_psd_rate(psd_passed, total_hops, PILLAR_6_PSD_THRESHOLD);

    PillarReport {
        pillar_id: 6,
        seed: PILLAR_6_SEED,
        n_paths: N_PATHS,
        n_hops: N_HOPS,
        psd_rate,
        lognorm_concentration,
        passed,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sandwich kernel unit tests ────────────────────────────────────────────

    #[test]
    fn sandwich_identity_m_identity_sigma_gives_identity() {
        // M = I, Σ = I → Σ_new = I · I · I = I
        let sigma_new = ewa_sandwich_step_2d(&Spd2::I, &Spd2::I);
        assert!((sigma_new.a11 - 1.0).abs() < 1e-7, "a11 = {}", sigma_new.a11);
        assert!(sigma_new.a12.abs() < 1e-7, "a12 = {}", sigma_new.a12);
        assert!((sigma_new.a22 - 1.0).abs() < 1e-7, "a22 = {}", sigma_new.a22);
    }

    #[test]
    fn sandwich_scalar_m_scales_sigma() {
        // M = c·I, Σ = I → Σ_new = c²·I
        let c = 0.5_f32;
        let m = Spd2::new(c, 0.0, c);
        let sigma_new = ewa_sandwich_step_2d(&m, &Spd2::I);
        let c2 = c * c;
        assert!((sigma_new.a11 - c2).abs() < 1e-7, "a11 expected {c2}, got {}", sigma_new.a11);
        assert!(sigma_new.a12.abs() < 1e-7, "a12 = {}", sigma_new.a12);
        assert!((sigma_new.a22 - c2).abs() < 1e-7, "a22 expected {c2}, got {}", sigma_new.a22);
    }

    #[test]
    fn sandwich_result_is_spd_for_spd_inputs() {
        // For any SPD M and Σ, the sandwich M·Σ·M must remain SPD.
        let m = Spd2::new(0.3, 0.05, 0.4);
        let sigma = Spd2::new(2.0, 0.1, 3.0);
        assert!(m.is_symmetric_pd(1e-9), "test fixture m not SPD");
        assert!(sigma.is_symmetric_pd(1e-9), "test fixture sigma not SPD");
        let sigma_new = ewa_sandwich_step_2d(&m, &sigma);
        assert!(
            sigma_new.is_symmetric_pd(1e-9),
            "sandwich result not SPD: a11={} a12={} a22={}",
            sigma_new.a11,
            sigma_new.a12,
            sigma_new.a22
        );
    }

    #[test]
    fn sandwich_diagonal_m_cross_check() {
        // M = diag(p, q), Σ = [[a, b], [b, c]]
        // M·Σ·M = [[p²a, pqb], [pqb, q²c]]
        let p = 0.3_f32;
        let q = 0.7_f32;
        let a = 2.0_f32;
        let b = 0.4_f32;
        let c = 3.0_f32;
        let m = Spd2::new(p, 0.0, q);
        let sigma = Spd2::new(a, b, c);
        let sigma_new = ewa_sandwich_step_2d(&m, &sigma);
        let exp11 = p * p * a;
        let exp12 = p * q * b;
        let exp22 = q * q * c;
        assert!((sigma_new.a11 - exp11).abs() < 1e-6, "a11: expected {exp11}, got {}", sigma_new.a11);
        assert!((sigma_new.a12 - exp12).abs() < 1e-6, "a12: expected {exp12}, got {}", sigma_new.a12);
        assert!((sigma_new.a22 - exp22).abs() < 1e-6, "a22: expected {exp22}, got {}", sigma_new.a22);
    }

    #[test]
    fn sandwich_contractivity_reduces_frobenius_norm() {
        // With ‖M‖_F = SIGMA_STEP < 1 and Σ = I, after 10 hops Frobenius
        // norm must be strictly smaller than 1.0.
        let mut rng = SplitMix64::new(0xCAFE_BEEF_1234_5678);
        let mut sigma = Spd2::I;
        for _ in 0..10 {
            let m_arr = random_contractive_spd2(&mut rng, SIGMA_STEP);
            let m = Spd2::new(m_arr[0][0], m_arr[0][1], m_arr[1][1]);
            sigma = ewa_sandwich_step_2d(&m, &sigma);
        }
        let frob = sigma.frobenius_sq().sqrt();
        assert!(frob < 1.0, "Frobenius norm {frob:.6} should be < 1.0 after 10 contractive hops");
    }

    // ── full prove_pillar_6 certification ─────────────────────────────────────

    #[test]
    fn prove_pillar_6_passes() {
        let report = prove_pillar_6();
        report.print();
        assert!(report.passed, "Pillar-6 FAIL: psd_rate={:.6} threshold={}", report.psd_rate, PILLAR_6_PSD_THRESHOLD);
    }

    #[test]
    fn prove_pillar_6_psd_rate_at_least_threshold() {
        let report = prove_pillar_6();
        assert!(
            report.psd_rate >= PILLAR_6_PSD_THRESHOLD,
            "psd_rate={:.6} below threshold={}",
            report.psd_rate,
            PILLAR_6_PSD_THRESHOLD
        );
    }

    #[test]
    fn prove_pillar_6_seed_matches_constant() {
        let report = prove_pillar_6();
        assert_eq!(report.seed, PILLAR_6_SEED, "seed mismatch");
    }

    #[test]
    fn prove_pillar_6_path_and_hop_counts_correct() {
        let report = prove_pillar_6();
        assert_eq!(report.n_paths, N_PATHS, "n_paths mismatch");
        assert_eq!(report.n_hops, N_HOPS, "n_hops mismatch");
    }

    #[test]
    fn prove_pillar_6_is_deterministic() {
        // Two calls with the same seed must produce identical reports.
        let r1 = prove_pillar_6();
        let r2 = prove_pillar_6();
        assert_eq!(r1.psd_rate.to_bits(), r2.psd_rate.to_bits(), "psd_rate not deterministic");
        assert_eq!(
            r1.lognorm_concentration.to_bits(),
            r2.lognorm_concentration.to_bits(),
            "lognorm_concentration not deterministic"
        );
        assert_eq!(r1.passed, r2.passed, "passed flag not deterministic");
    }
}
