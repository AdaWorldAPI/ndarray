#![allow(missing_docs)]
//! Pillar-12 — Anisotropic-splat construction invariance certification.
//!
//! Substrate-tier pillar: certifies that the construction
//!
//! ```text
//!   Σ  =  R(q) · diag(s²) · R(q)ᵀ
//! ```
//!
//! consumed by `hpc::splat3d::spd3::Spd3::from_scale_quat` and by the
//! cognitive splat carrier yields a covariance whose spectral functions
//! match their closed-form values *exactly* (within f32 rounding), and
//! that Σ is symmetric positive-definite for every positive scale and
//! every unit quaternion in the production sample space.
//!
//! # The four invariants
//!
//! For any orthogonal rotation `R` and any positive diagonal scaling
//! `diag(s²)`, the resulting `Σ` satisfies:
//!
//! ```text
//!   trace(Σ)        =  s₁² + s₂² + s₃²
//!   det(Σ)          =  (s₁ · s₂ · s₃)²
//!   ‖Σ‖_F²          =  s₁⁴ + s₂⁴ + s₃⁴
//!   Σ ≻ 0          (SPD — three leading principal minors positive)
//! ```
//!
//! The first three are *rotation-invariant* — they depend on the
//! scales alone, not on the quaternion. The fourth is the
//! positive-definiteness that the EWA-sandwich pipeline (Pillar-7) and
//! the cognitive cascade both rely on at every covariance push-forward.
//!
//! # Why the substrate needs this
//!
//! 3D Gaussian splatting (Kerbl 2023) compresses each splat into a
//! `(mean, scale, quaternion)` triple and reconstructs Σ on the hot
//! path; the reconstruction is the place where the substrate's claims
//! about "splats have the shape we said they do" actually have to hold.
//! Two failure modes the cognitive cascade cannot see otherwise:
//!
//! 1. *Non-unit quaternion accumulation*. The Shoemake constructor
//!    yields unit quaternions to within rounding, but downstream
//!    arithmetic (interpolation, drift over plasticity steps, codec
//!    encode/decode round-trips) can drift the norm away from 1. A
//!    non-unit `q` produces a non-orthogonal `R`, which produces a Σ
//!    that *looks like* an SPD matrix (Sylvester still passes for
//!    small drift) but whose trace, determinant, and Frobenius norm
//!    no longer match the scale invariants. The cognitive cascade
//!    silently mis-weights such splats.
//! 2. *Scale-axis swap or sign flip*. A code path that accidentally
//!    interprets `(s₁, s₂, s₃)` as `(s₂, s₁, s₃)` will still produce
//!    valid Σ shapes — same set of eigenvalues — and pass
//!    Sylvester. The trace and determinant remain unchanged
//!    (symmetric functions of the scale vector). But the
//!    *correspondence* between the input scale tuple and the output
//!    covariance is wrong, and downstream code that consumes
//!    `(scale, Σ)` jointly (the EWA sandwich, the SH coefficient
//!    expansion) will mis-compute. The Frobenius identity
//!    `‖Σ‖_F² = s₁⁴ + s₂⁴ + s₃⁴` *is* symmetric in the scales, so it
//!    won't catch this specifically — but the per-eigenvalue check
//!    below will. (Eigenvalues are not implemented in this first
//!    pass; we leave that to a follow-up pillar revision.)
//!
//! Pillar-12 pins the three closed-form identities and the SPD
//! property to a measurable failure rate over 4096 synthetic splats.
//!
//! # Probe design (synthetic only)
//!
//! 1. Generate `N_SPLATS = 4096` synthetic `(scale, quaternion)` pairs
//!    via `SplitMix64`:
//!    - per-axis scales `s_i` lognormal with median `SCALE_MEDIAN`
//!      and log-width `SCALE_LOG_WIDTH` (matches the Kerbl 2023
//!      anisotropy regime),
//!    - quaternions Shoemake-uniform on `S³`.
//! 2. For each pair construct `Σ = R(q) · diag(s²) · R(q)ᵀ`.
//! 3. Check four invariants:
//!    - *Sylvester SPD*: `Σ₀₀ > 0`, `det(Σ[0:2]) > 0`, `det(Σ) > 0`.
//!    - *Trace*: `|trace(Σ) − (s₁² + s₂² + s₃²)| / (s₁² + s₂² + s₃²) ≤ ε`.
//!    - *Determinant*: `|det(Σ) − (s₁ s₂ s₃)²| / (s₁ s₂ s₃)² ≤ ε`.
//!    - *Frobenius*: `| ‖Σ‖_F² − (s₁⁴ + s₂⁴ + s₃⁴) | / (s₁⁴ + s₂⁴ + s₃⁴) ≤ ε`.
//!
//! All four checks run in pure f32 — the float-precision budget `ε` is
//! chosen so the test passes for an exact f32 implementation but fails
//! visibly for any non-f32-precision drift.
//!
//! # Cross-verification
//!
//! The closed-form invariants (trace, determinant, Frobenius norm) are
//! verifiable by hand on small examples and reproducible in numpy or
//! scipy with single-precision arithmetic. The Σ construction here is
//! independently re-derived rather than imported from
//! `crate::hpc::splat3d::spd3` — that decoupling is exactly what makes
//! this a *certification* pillar rather than a unit test.
//!
//! # Pass criteria
//!
//! - `psd_rate ≥ PASS_FRACTION` (default 1.0): every splat must pass
//!   Sylvester. There is no statistical noise in this check; any
//!   failure indicates a construction bug.
//! - `lognorm_concentration ≤ INVARIANT_BUDGET` (default 1e-3):
//!   `log(1 + max relative invariant error)` across all three
//!   rotation invariants over all splats. The 1e-3 budget is three
//!   orders of magnitude above expected f32 round-off, leaving room
//!   for the worst-case quaternion catastrophic cancellation while
//!   still catching real bugs.
//!
//! # References
//!
//! * Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
//!   *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
//!   ACM TOG 42(4). (The `(mean, scale, quaternion)` splat
//!   representation whose reconstruction Σ = R·diag(s²)·Rᵀ this
//!   pillar certifies.)
//! * Shoemake, K. (1992). *Uniform Random Rotations.* Graphics
//!   Gems III. (The S³-uniform quaternion sampler.)
//! * Higham, N. J. (2002). *Accuracy and Stability of Numerical
//!   Algorithms*, 2nd ed., SIAM. (The f32 precision budget the
//!   invariants are tested against.)
//!
//! # SEED
//!
//! `PILLAR_12_SEED = 0x_C12A_550A_D0DD`

use crate::hpc::pillar::prove_runner::{PillarReport, SplitMix64};

// ── Constants ────────────────────────────────────────────────────────────────

/// Deterministic seed for all Pillar-12 RNG streams.
pub const PILLAR_12_SEED: u64 = 0x_C12A_550A_D0DD;

/// Number of synthetic splats tested.
const N_SPLATS: u32 = 4096;

/// Log-median scale (one standard deviation along each principal axis,
/// pre-rotation). Picked so the average splat covers ~1/16 of the unit
/// cube edge, matching the Kerbl 2023 anisotropy regime.
const SCALE_MEDIAN: f32 = 0.125;

/// Log-width of the scale lognormal distribution. 0.35 spans roughly
/// an octave at 95% confidence.
const SCALE_LOG_WIDTH: f32 = 0.35;

/// Minimum acceptable SPD pass rate. There is no statistical noise here:
/// `R · diag(s²) · Rᵀ` is SPD for any positive `s`, so any failure is a
/// construction bug.
const PASS_FRACTION: f64 = 1.0;

/// Budget on `log(1 + max relative invariant error)`. Three orders of
/// magnitude above expected f32 round-off (~1e-6).
const INVARIANT_BUDGET: f64 = 1e-3;

// ── Sampling ────────────────────────────────────────────────────────────────

/// Sample a positive scale via lognormal `exp(ln(SCALE_MEDIAN) + W · ξ)`.
#[inline]
fn sample_scale_axis(rng: &mut SplitMix64) -> f32 {
    let z = rng.next_normal_f32();
    (SCALE_MEDIAN.ln() + SCALE_LOG_WIDTH * z).exp()
}

/// Sample a unit quaternion uniformly on S³ (Shoemake 1992).
#[inline]
fn sample_unit_quaternion(rng: &mut SplitMix64) -> [f32; 4] {
    let u1 = rng.next_f32();
    let u2 = rng.next_f32();
    let u3 = rng.next_f32();
    let s1 = (1.0_f32 - u1).sqrt();
    let s2 = u1.sqrt();
    let two_pi = core::f32::consts::TAU;
    [
        s1 * (two_pi * u2).sin(),
        s1 * (two_pi * u2).cos(),
        s2 * (two_pi * u3).sin(),
        s2 * (two_pi * u3).cos(),
    ]
}

// ── Covariance construction (independently re-derived) ──────────────────────

/// Build the upper triangle of `Σ = R(q) · diag(s²) · R(q)ᵀ` in row-major
/// order: `[Σ₀₀, Σ₀₁, Σ₀₂, Σ₁₁, Σ₁₂, Σ₂₂]`.
///
/// Quaternion convention: `q = (w, x, y, z)`. Mirrors
/// `crate::hpc::splat3d::spd3::Spd3::from_scale_quat` — the duplication
/// is the point: any drift between the production code and this
/// re-derivation surfaces as a pillar failure.
fn covariance_from_scale_quat(scale: [f32; 3], quat: [f32; 4]) -> [f32; 6] {
    let [w, qx, qy, qz] = quat;
    let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let r01 = 2.0 * (qx * qy - qz * w);
    let r02 = 2.0 * (qx * qz + qy * w);
    let r10 = 2.0 * (qx * qy + qz * w);
    let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let r12 = 2.0 * (qy * qz - qx * w);
    let r20 = 2.0 * (qx * qz - qy * w);
    let r21 = 2.0 * (qy * qz + qx * w);
    let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

    let s0sq = scale[0] * scale[0];
    let s1sq = scale[1] * scale[1];
    let s2sq = scale[2] * scale[2];

    let sigma_00 = r00 * r00 * s0sq + r01 * r01 * s1sq + r02 * r02 * s2sq;
    let sigma_01 = r00 * r10 * s0sq + r01 * r11 * s1sq + r02 * r12 * s2sq;
    let sigma_02 = r00 * r20 * s0sq + r01 * r21 * s1sq + r02 * r22 * s2sq;
    let sigma_11 = r10 * r10 * s0sq + r11 * r11 * s1sq + r12 * r12 * s2sq;
    let sigma_12 = r10 * r20 * s0sq + r11 * r21 * s1sq + r12 * r22 * s2sq;
    let sigma_22 = r20 * r20 * s0sq + r21 * r21 * s1sq + r22 * r22 * s2sq;

    [sigma_00, sigma_01, sigma_02, sigma_11, sigma_12, sigma_22]
}

// ── Spectral functions ──────────────────────────────────────────────────────

/// Trace of a 3×3 symmetric matrix given upper-triangle.
#[inline]
fn trace_sym3(ut: [f32; 6]) -> f32 {
    ut[0] + ut[3] + ut[5]
}

/// Determinant of a 3×3 symmetric matrix given upper-triangle.
#[inline]
fn det_sym3(ut: [f32; 6]) -> f32 {
    let [a, b, c, d, e, f] = ut;
    a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
}

/// Squared Frobenius norm of a 3×3 symmetric matrix given upper-triangle.
/// `‖M‖_F² = M₀₀² + M₁₁² + M₂₂² + 2·(M₀₁² + M₀₂² + M₁₂²)`.
#[inline]
fn frob_sq_sym3(ut: [f32; 6]) -> f32 {
    let [a, b, c, d, e, f] = ut;
    a * a + d * d + f * f + 2.0 * (b * b + c * c + e * e)
}

/// Sylvester check: all three leading principal minors positive.
#[inline]
fn is_spd_sylvester(ut: [f32; 6]) -> bool {
    let [a, b, c, d, e, f] = ut;
    if a <= 0.0 {
        return false;
    }
    let det2 = a * d - b * b;
    if det2 <= 0.0 {
        return false;
    }
    let det3 = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    det3 > 0.0
}

// ── Public probe entry point ────────────────────────────────────────────────

/// Run the Pillar-12 splat-construction invariance certification probe.
///
/// Generates `N_SPLATS = 4096` synthetic `(scale, quaternion)` pairs
/// from the canonical seed, constructs `Σ = R(q) · diag(s²) · R(q)ᵀ`
/// for each, and verifies four invariants:
///
/// - Sylvester SPD on every Σ.
/// - `trace(Σ) = s₁² + s₂² + s₃²` (rotation invariant).
/// - `det(Σ) = (s₁ s₂ s₃)²` (rotation invariant).
/// - `‖Σ‖_F² = s₁⁴ + s₂⁴ + s₃⁴` (rotation invariant).
///
/// # Returns
///
/// A [`PillarReport`] with semantics:
/// - `n_paths` = `N_SPLATS` (splats tested)
/// - `n_hops` = `3` (invariants checked per splat: trace, det, Frobenius)
/// - `psd_rate` = fraction of splats passing Sylvester SPD.
/// - `lognorm_concentration` = `log(1 + max relative invariant error)`.
/// - `passed` = `psd_rate ≥ PASS_FRACTION ∧ lognorm_concentration ≤ INVARIANT_BUDGET`.
pub fn prove_pillar_12() -> PillarReport {
    let mut rng = SplitMix64::new(PILLAR_12_SEED);

    let mut spd_pass = 0u32;
    let mut max_rel_err = 0.0_f32;

    for _ in 0..N_SPLATS {
        let s = [sample_scale_axis(&mut rng), sample_scale_axis(&mut rng), sample_scale_axis(&mut rng)];
        let q = sample_unit_quaternion(&mut rng);
        let sigma = covariance_from_scale_quat(s, q);

        // Sylvester.
        if is_spd_sylvester(sigma) {
            spd_pass += 1;
        }

        // Rotation invariants — predicted from scales alone.
        let pred_trace = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let pred_det = (s[0] * s[1] * s[2]).powi(2);
        let pred_frob_sq = s[0].powi(4) + s[1].powi(4) + s[2].powi(4);

        let actual_trace = trace_sym3(sigma);
        let actual_det = det_sym3(sigma);
        let actual_frob_sq = frob_sq_sym3(sigma);

        // Relative errors. Guard against pred = 0 (impossible at lognormal
        // scales but defensive).
        let trace_err = if pred_trace > f32::MIN_POSITIVE {
            (actual_trace - pred_trace).abs() / pred_trace
        } else {
            0.0
        };
        let det_err = if pred_det > f32::MIN_POSITIVE {
            (actual_det - pred_det).abs() / pred_det
        } else {
            0.0
        };
        let frob_err = if pred_frob_sq > f32::MIN_POSITIVE {
            (actual_frob_sq - pred_frob_sq).abs() / pred_frob_sq
        } else {
            0.0
        };

        if trace_err > max_rel_err {
            max_rel_err = trace_err;
        }
        if det_err > max_rel_err {
            max_rel_err = det_err;
        }
        if frob_err > max_rel_err {
            max_rel_err = frob_err;
        }
    }

    let psd_rate = (spd_pass as f64) / (N_SPLATS as f64);
    let lognorm_concentration = (1.0 + max_rel_err as f64).ln();
    let passed = psd_rate >= PASS_FRACTION && lognorm_concentration <= INVARIANT_BUDGET;

    PillarReport {
        pillar_id: 12,
        seed: PILLAR_12_SEED,
        n_paths: N_SPLATS,
        n_hops: 3,
        psd_rate,
        lognorm_concentration,
        passed,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_12_seed_matches_spec() {
        assert_eq!(PILLAR_12_SEED, 0x_C12A_550A_D0DD);
    }

    #[test]
    fn quaternion_is_unit() {
        let mut rng = SplitMix64::new(PILLAR_12_SEED);
        for _ in 0..100 {
            let q = sample_unit_quaternion(&mut rng);
            let n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            assert!((n - 1.0).abs() < 1e-5, "non-unit quaternion: norm={n}");
        }
    }

    #[test]
    fn trace_invariant_holds_for_identity_rotation() {
        // q = (1, 0, 0, 0) → R = I → Σ = diag(s²).
        let s = [0.1_f32, 0.2, 0.3];
        let q = [1.0_f32, 0.0, 0.0, 0.0];
        let sigma = covariance_from_scale_quat(s, q);
        let predicted = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let actual = trace_sym3(sigma);
        assert!((actual - predicted).abs() < 1e-6, "trace {actual} vs {predicted}");
    }

    #[test]
    fn det_invariant_holds_for_identity_rotation() {
        let s = [0.1_f32, 0.2, 0.3];
        let q = [1.0_f32, 0.0, 0.0, 0.0];
        let sigma = covariance_from_scale_quat(s, q);
        let predicted = (s[0] * s[1] * s[2]).powi(2);
        let actual = det_sym3(sigma);
        assert!((actual - predicted).abs() < 1e-9, "det {actual} vs {predicted}");
    }

    #[test]
    fn frob_invariant_holds_for_identity_rotation() {
        let s = [0.1_f32, 0.2, 0.3];
        let q = [1.0_f32, 0.0, 0.0, 0.0];
        let sigma = covariance_from_scale_quat(s, q);
        let predicted = s[0].powi(4) + s[1].powi(4) + s[2].powi(4);
        let actual = frob_sq_sym3(sigma);
        assert!((actual - predicted).abs() < 1e-9, "frob² {actual} vs {predicted}");
    }

    #[test]
    fn frob_invariant_holds_under_rotation() {
        // Same scales, different quaternions → Frobenius norm² unchanged.
        let s = [0.1_f32, 0.2, 0.3];
        let mut rng = SplitMix64::new(PILLAR_12_SEED);
        let predicted = s[0].powi(4) + s[1].powi(4) + s[2].powi(4);
        for _ in 0..32 {
            let q = sample_unit_quaternion(&mut rng);
            let sigma = covariance_from_scale_quat(s, q);
            let actual = frob_sq_sym3(sigma);
            let rel_err = (actual - predicted).abs() / predicted;
            assert!(rel_err < 1e-5, "rel_err {rel_err} for q={q:?}");
        }
    }

    #[test]
    fn is_spd_sylvester_accepts_identity() {
        // Σ = I (positive scales 1, identity rotation) is SPD.
        let sigma = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0];
        assert!(is_spd_sylvester(sigma));
    }

    #[test]
    fn is_spd_sylvester_rejects_negative_diagonal() {
        let sigma = [-1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0];
        assert!(!is_spd_sylvester(sigma));
    }

    #[test]
    fn prove_pillar_12_passes() {
        let report = prove_pillar_12();
        report.print();
        assert!(
            report.passed,
            "Pillar-12 FAILED: psd_rate={:.6} lognorm={:.6}",
            report.psd_rate, report.lognorm_concentration
        );
    }

    #[test]
    fn prove_pillar_12_seed_anchored() {
        let r1 = prove_pillar_12();
        let r2 = prove_pillar_12();
        assert_eq!(r1.passed, r2.passed);
        assert!((r1.psd_rate - r2.psd_rate).abs() < 1e-12);
        assert!((r1.lognorm_concentration - r2.lognorm_concentration).abs() < 1e-12);
    }

    /// Drift-detection: the pillar's `covariance_from_scale_quat`
    /// independently re-derives `Σ = R(q) · diag(s²) · R(q)ᵀ`. The
    /// production code path at `crate::hpc::splat3d::spd3::Spd3::from_scale_quat`
    /// is the substrate the pillar is *defending*. This test runs both
    /// on the same SplitMix64-seeded sample of 256 `(scale, quat)` pairs
    /// and asserts agreement to within `1e-5` per upper-triangle entry.
    ///
    /// Any divergence ≥ ε indicates one of two failure modes:
    ///   (a) production drifted from the canonical quaternion-rotation
    ///       formula (the pillar definition wins by design — fix the
    ///       production code), or
    ///   (b) the pillar itself drifted (audit `covariance_from_scale_quat`
    ///       against Kerbl 2023 Eq. 3 before changing).
    ///
    /// This is the *coupling* the per-pillar docstring promises:
    /// production and pillar share no code, but they share a CI gate
    /// that compares them point-for-point.
    #[test]
    fn pillar_12_matches_production_spd3_from_scale_quat() {
        use crate::hpc::splat3d::spd3::Spd3;

        const N: u32 = 256;
        let mut rng = SplitMix64::new(PILLAR_12_SEED);
        let mut max_abs_err: f32 = 0.0;

        for _ in 0..N {
            let s = [sample_scale_axis(&mut rng), sample_scale_axis(&mut rng), sample_scale_axis(&mut rng)];
            let q = sample_unit_quaternion(&mut rng);

            let pillar = covariance_from_scale_quat(s, q);
            let prod = Spd3::from_scale_quat(s, q);
            let prod_ut = [prod.a11, prod.a12, prod.a13, prod.a22, prod.a23, prod.a33];

            for (i, (&p, &pr)) in pillar.iter().zip(prod_ut.iter()).enumerate() {
                let err = (p - pr).abs();
                if err > max_abs_err {
                    max_abs_err = err;
                }
                assert!(
                    err < 1e-5,
                    "Pillar/Spd3 drift at lane {i}: pillar={p:.7} prod={pr:.7} err={err:.2e} s={s:?} quat={q:?}"
                );
            }
        }

        eprintln!("Pillar 12 ↔ Spd3::from_scale_quat agreement: max_abs_err={max_abs_err:.3e} over {N} pairs");
    }
}
