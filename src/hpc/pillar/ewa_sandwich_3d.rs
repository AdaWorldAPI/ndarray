//! Pillar-7 — 3D EWA-sandwich certification probe.
//!
//! Certifies that the 3D EWA (Elliptical Weighted Average) sandwich update
//!
//! ```text
//!     Σ_{n+1} = M · Σ_n · Mᵀ
//! ```
//!
//! preserves symmetric positive-definiteness (SPD) across 1 000 random paths
//! × 10 hops, where `M = sqrt(step_Σ_3d)` is a contractive `Spd3` drawn
//! from the shared `prove_runner::random_contractive_spd3` helper.
//!
//! # Relationship to Pillar-6
//!
//! Pillar-7 is the direct 3D twin of the Pillar-6 2D probe. Where Pillar-6
//! certifies `Σ' = M · Σ · Mᵀ` for 2×2 SPD matrices, Pillar-7 does the same
//! for 3×3 SPD matrices using `crate::hpc::splat3d::spd3::sandwich` and the
//! Smith-1961 closed-form eigendecomposition in `eig_sym::eig_sym_3`.
//!
//! # PASS criteria
//!
//! * PSD rate ≥ 0.999 over 1 000 paths × 10 hops (10 000 SPD checks).
//! * Log-norm Frobenius KS-Thm-1 concentration: reported but not gated.
//!
//! # Parity gate
//!
//! `eig_sym_3` (PR-X10 A4) must produce eigenvalues bit-equivalent to
//! `splat3d::Spd3::eig` (Smith-1961 in `splat3d/spd3.rs`). The
//! `parity_eig_sym_3_vs_spd3_eig` test enforces max abs error < 1e-5 over
//! 100 random SPD3 matrices drawn with the Pillar-7 SEED.
//!
//! # Feature gate
//!
//! Compiled under `#[cfg(feature = "pillar")]`. The parity gate test also
//! requires `#[cfg(feature = "splat3d")]`.

use crate::hpc::linalg::eig_sym::eig_sym_3;
use crate::hpc::linalg::Spd3;
use crate::hpc::pillar::prove_runner::{assert_psd_rate, random_contractive_spd3, PillarReport, SplitMix64};
use crate::hpc::splat3d::spd3::sandwich;

// ── Public constants ──────────────────────────────────────────────────────────

/// SEED for the Pillar-7 splitmix64 RNG. All runs are SEED-anchored for
/// cross-platform reproducibility and deterministic audit trails.
pub const PILLAR_7_SEED: u64 = 0x_EDA_5A_DC_5A_DD;

/// Minimum PSD preservation rate for the Pillar-7 probe to PASS.
pub const PILLAR_7_PSD_THRESHOLD: f64 = 0.999;

// ── Sandwich kernel ───────────────────────────────────────────────────────────

/// Convert a raw 3×3 array (from `random_contractive_spd3`) to an `Spd3`.
///
/// Reads only the upper triangle (the helper guarantees symmetry) and wraps
/// it in the `Spd3` repr.
#[inline]
fn array_to_spd3(m: [[f32; 3]; 3]) -> Spd3 {
    Spd3::new(m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2])
}

/// Compute the EWA sandwich `M · Σ · Mᵀ` using `splat3d::spd3::sandwich`.
///
/// `m` is the step matrix (square root of a contractive SPD3).
/// `sigma` is the current 3×3 SPD covariance.
///
/// The `sandwich` function from `splat3d::spd3` computes `M · N · Mᵀ`
/// (since M is symmetric, `Mᵀ = M`), averaging the two off-diagonal
/// halves to suppress f32 asymmetry. Output is always symmetric.
#[inline]
pub fn ewa_sandwich_3d(m: &Spd3, sigma: &Spd3) -> Spd3 {
    sandwich(m, sigma)
}

/// Check whether `sigma` is symmetric positive-definite via eigenvalue check.
///
/// Uses `eig_sym_3` (Smith-1961 closed-form, same algorithm as `Spd3::eig`)
/// and asserts that the smallest eigenvalue `λ₃ > 0`. This is numerically
/// robust even when the matrix determinant underflows to zero in f32 (which
/// can happen when all eigenvalues are tiny but positive, e.g. after many
/// contractive sandwich steps). The eigenvalue computation works in terms of
/// the trace and off-diagonal structure, which remain well-conditioned.
#[inline]
fn is_psd_eig(s: &Spd3) -> bool {
    let rows = s.to_rows();
    let (_, _, l3, _) = eig_sym_3(&rows);
    l3 > 0.0
}

/// Frobenius norm of log(Σ) — used for the KS-Thm-1 concentration metric.
///
/// Computes the matrix logarithm via spectral lift (Smith-1961 eigenvalues),
/// then returns `‖log(Σ)‖_F`. Eigenvalues are clamped to a small positive
/// ε before `ln` so the output stays finite under f32 cancellation noise.
#[inline]
fn log_frob(s: &Spd3) -> f32 {
    let rows = s.to_rows();
    let (l1, l2, l3, v) = eig_sym_3(&rows);
    let eps = 1e-30_f32;
    let ln1 = l1.max(eps).ln();
    let ln2 = l2.max(eps).ln();
    let ln3 = l3.max(eps).ln();
    // ‖V · diag(ln λ) · Vᵀ‖_F = sqrt(ln1² + ln2² + ln3²) since V is orthonormal.
    (ln1 * ln1 + ln2 * ln2 + ln3 * ln3).sqrt()
}

// ── Main probe ────────────────────────────────────────────────────────────────

/// Run the Pillar-7 3D EWA-sandwich certification probe.
///
/// Simulates 1 000 random paths × 10 cascade hops. On each hop the covariance
/// is updated by the sandwich `Σ_{n+1} = M · Σ_n · Mᵀ` where `M =
/// sqrt(step_Σ_3d)` is a random contractive `Spd3` drawn from the shared
/// harness. PSD is checked after every hop via Sylvester's criterion.
///
/// # PASS criteria
///
/// * `psd_rate ≥ PILLAR_7_PSD_THRESHOLD` (= 0.999)
///
/// # Determinism
///
/// The RNG is seeded with `PILLAR_7_SEED` so every run on every platform
/// produces the identical path sequence and the identical `PillarReport`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::ewa_sandwich_3d::prove_pillar_7;
/// let report = prove_pillar_7();
/// report.print();
/// assert!(report.passed);
/// ```
pub fn prove_pillar_7() -> PillarReport {
    const N_PATHS: u32 = 1_000;
    const N_HOPS: u32 = 10;
    const SIGMA_STEP: f32 = 0.1; // contractive step — keeps Σ bounded

    let mut rng = SplitMix64::new(PILLAR_7_SEED);

    let mut psd_passed: u32 = 0;
    let total_hops: u32 = N_PATHS * N_HOPS;

    // Log-norm Frobenius accumulator for KS-Thm-1 concentration.
    let mut log_frob_sum: f64 = 0.0;
    let mut log_frob_sum_sq: f64 = 0.0;
    let mut log_frob_count: u32 = 0;

    for _path in 0..N_PATHS {
        // Each path starts from the 3×3 identity covariance.
        let mut sigma = Spd3::I;

        for _hop in 0..N_HOPS {
            // Draw a random contractive SPD3 step matrix and take its sqrt.
            let step_raw = random_contractive_spd3(&mut rng, SIGMA_STEP);
            let step_spd = array_to_spd3(step_raw);
            let m = step_spd.sqrt();

            // Apply the EWA sandwich: Σ_{n+1} = M · Σ_n · Mᵀ.
            sigma = ewa_sandwich_3d(&m, &sigma);

            // PSD check via Sylvester's criterion.
            if is_psd_eig(&sigma) {
                psd_passed += 1;

                // Accumulate log-norm Frobenius for concentration metric.
                let lf = log_frob(&sigma) as f64;
                log_frob_sum += lf;
                log_frob_sum_sq += lf * lf;
                log_frob_count += 1;
            }
        }
    }

    let psd_rate = (psd_passed as f64) / (total_hops as f64);
    let passed = assert_psd_rate(psd_passed, total_hops, PILLAR_7_PSD_THRESHOLD);

    // KS-Thm-1 concentration: sample std-dev of log-norm Frobenius values.
    // A well-concentrated distribution has small variance (near 0).
    let lognorm_concentration = if log_frob_count > 1 {
        let mean = log_frob_sum / log_frob_count as f64;
        let var = log_frob_sum_sq / log_frob_count as f64 - mean * mean;
        var.max(0.0).sqrt()
    } else {
        0.0
    };

    PillarReport {
        pillar_id: 7,
        seed: PILLAR_7_SEED,
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

    // ── Smoke test: prove_pillar_7 must pass ──────────────────────────────────

    #[test]
    fn prove_pillar_7_passes() {
        let report = prove_pillar_7();
        report.print();
        assert!(
            report.passed,
            "Pillar-7 FAILED: psd_rate={:.6} < threshold={:.6}",
            report.psd_rate, PILLAR_7_PSD_THRESHOLD
        );
        assert_eq!(report.pillar_id, 7);
        assert_eq!(report.seed, PILLAR_7_SEED);
        assert_eq!(report.n_paths, 1_000);
        assert_eq!(report.n_hops, 10);
    }

    // ── PSD rate meets the 0.999 threshold ───────────────────────────────────

    #[test]
    fn psd_rate_at_or_above_threshold() {
        let report = prove_pillar_7();
        assert!(
            report.psd_rate >= PILLAR_7_PSD_THRESHOLD,
            "psd_rate={:.6} below threshold={:.6}",
            report.psd_rate,
            PILLAR_7_PSD_THRESHOLD
        );
    }

    // ── Determinism: two runs with the same seed produce identical results ─────

    #[test]
    fn prove_pillar_7_is_deterministic() {
        let r1 = prove_pillar_7();
        let r2 = prove_pillar_7();
        assert_eq!(r1.psd_rate.to_bits(), r2.psd_rate.to_bits(), "psd_rate not deterministic");
        assert_eq!(
            r1.lognorm_concentration.to_bits(),
            r2.lognorm_concentration.to_bits(),
            "lognorm_concentration not deterministic"
        );
    }

    // ── Identity step: sandwich with identity M should leave Σ unchanged ──────

    #[test]
    fn sandwich_with_identity_is_noop() {
        let sigma = Spd3::new(2.0, 0.5, 0.3, 3.0, 0.4, 1.5);
        let result = ewa_sandwich_3d(&Spd3::I, &sigma);
        // M = I → Σ' = I · Σ · I = Σ; check within f32 rounding (1e-5).
        assert!((result.a11 - sigma.a11).abs() < 1e-5, "a11 mismatch");
        assert!((result.a12 - sigma.a12).abs() < 1e-5, "a12 mismatch");
        assert!((result.a13 - sigma.a13).abs() < 1e-5, "a13 mismatch");
        assert!((result.a22 - sigma.a22).abs() < 1e-5, "a22 mismatch");
        assert!((result.a23 - sigma.a23).abs() < 1e-5, "a23 mismatch");
        assert!((result.a33 - sigma.a33).abs() < 1e-5, "a33 mismatch");
    }

    // ── SPD input + SPD step → SPD output ────────────────────────────────────

    #[test]
    fn sandwich_preserves_spd_for_random_inputs() {
        let mut rng = SplitMix64::new(PILLAR_7_SEED ^ 0xDEAD_BEEF);
        for trial in 0..200 {
            let step_raw = random_contractive_spd3(&mut rng, 0.1);
            let step_spd = array_to_spd3(step_raw);
            let m = step_spd.sqrt();
            let sigma_raw = random_contractive_spd3(&mut rng, 1.0);
            let sigma = array_to_spd3(sigma_raw);
            let result = ewa_sandwich_3d(&m, &sigma);
            assert!(is_psd_eig(&result), "trial {trial}: sandwich output not SPD: {:?}", result);
        }
    }

    // ── eig_sym_3 parity gate vs Spd3::eig (Smith-1961 bit-equivalence) ──────
    //
    // Critical: both `eig_sym_3` (PR-X10 A4, `linalg::eig_sym`) and
    // `splat3d::Spd3::eig` implement the same Smith-1961 algorithm. The
    // parity gate verifies their eigenvalues agree within 1e-5 over 100
    // random SPD3 matrices drawn with the Pillar-7 SEED, confirming that
    // calling the algorithm from either location produces the same result.

    #[cfg(feature = "splat3d")]
    #[test]
    fn parity_eig_sym_3_vs_spd3_eig() {
        let mut rng = SplitMix64::new(PILLAR_7_SEED);
        let mut max_err = 0.0f32;

        for trial in 0..100 {
            // Draw a random SPD3 via the contractive helper (always SPD).
            let raw = random_contractive_spd3(&mut rng, 1.0);
            let spd = array_to_spd3(raw);

            // Reference: Spd3::eig (Smith-1961 in splat3d/spd3.rs).
            let (ref_l1, ref_l2, ref_l3, _ref_v) = spd.eig();

            // Under test: eig_sym_3 (Smith-1961 in linalg/eig_sym.rs).
            let rows = spd.to_rows();
            let (l1, l2, l3, _v) = eig_sym_3(&rows);

            let err = (l1 - ref_l1)
                .abs()
                .max((l2 - ref_l2).abs())
                .max((l3 - ref_l3).abs());
            max_err = max_err.max(err);

            assert!(
                err < 1e-5,
                "trial {trial}: eigenvalue parity error = {err:.2e} (want < 1e-5)\n  \
                 Spd3::eig  = ({ref_l1}, {ref_l2}, {ref_l3})\n  \
                 eig_sym_3  = ({l1}, {l2}, {l3})"
            );
        }
        // Global summary so the log shows the worst-case error.
        assert!(max_err < 1e-5, "Smith-1961 parity gate: max_err = {max_err:.2e} over 100 trials (want < 1e-5)");
    }

    // ── array_to_spd3 correctly reads the upper triangle ─────────────────────

    #[test]
    fn array_to_spd3_upper_triangle() {
        let m = [[1.0f32, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];
        let s = array_to_spd3(m);
        assert_eq!(s.a11, 1.0);
        assert_eq!(s.a12, 2.0);
        assert_eq!(s.a13, 3.0);
        assert_eq!(s.a22, 4.0);
        assert_eq!(s.a23, 5.0);
        assert_eq!(s.a33, 6.0);
    }

    // ── is_psd_eig rejects non-SPD matrices ────────────────────────────

    #[test]
    fn psd_eig_rejects_indefinite() {
        // A matrix with a negative diagonal is clearly not SPD.
        let s = Spd3::new(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(!is_psd_eig(&s), "Should reject matrix with negative a11");
    }

    #[test]
    fn psd_eig_accepts_identity() {
        assert!(is_psd_eig(&Spd3::I), "Identity must be SPD");
    }

    // ── log_frob is finite for SPD inputs ────────────────────────────────────

    #[test]
    fn log_frob_finite_for_spd() {
        let mut rng = SplitMix64::new(0xABCD_EF01_2345_6789);
        for _ in 0..50 {
            let raw = random_contractive_spd3(&mut rng, 1.0);
            let spd = array_to_spd3(raw);
            let lf = log_frob(&spd);
            assert!(lf.is_finite(), "log_frob returned non-finite: {lf}");
        }
    }
}
