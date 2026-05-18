//! Degree-3 real spherical harmonics evaluator for 3D Gaussian Splatting.
//!
//! # Mathematical claim
//!
//! Given a unit view direction `d = (x, y, z)` and 16 SH coefficients per
//! channel, evaluates the dot product of the degree-0..3 real spherical
//! harmonic basis with the coefficient vector, adds the Inria +0.5 DC offset,
//! and clamps to [0, 1]. Implements the convention from:
//!
//!   Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Novel View
//!   Synthesis", SIGGRAPH 2023 — Appendix A, SH evaluation.
//!
//! # Basis functions (per-channel, real SH, Condon-Shortley convention)
//!
//! ```text
//! Y_00             = SH_C0                               (degree 0, 1 term)
//! Y_1-1  Y_10  Y_11 = -SH_C1·y, SH_C1·z, -SH_C1·x      (degree 1, 3 terms)
//! Y_2-2..Y_22        SH_C2[0..4]×polynomial              (degree 2, 5 terms)
//! Y_3-3..Y_33        SH_C3[0..6]×polynomial              (degree 3, 7 terms)
//! ```
//!
//! # Storage layout
//!
//! For a single gaussian's 48-float SH block:
//! - Channel 0 (R): `sh[0..16]`
//! - Channel 1 (G): `sh[16..32]`
//! - Channel 2 (B): `sh[32..48]`
//!
//! For the batched 16-gaussian `sh_block` (`sh_eval_deg3_x16`):
//! - Gaussian g, channel c, basis k: `sh_block[g*48 + c*16 + k]`

use crate::simd::F32x16;

// ════════════════════════════════════════════════════════════════════════════
// SH basis constants (Inria / Wikipedia "Table of spherical harmonics")
// ════════════════════════════════════════════════════════════════════════════

/// Number of basis functions per channel for degree-3 SH (bands 0..=3).
pub const SH_BASIS_PER_CHANNEL: usize = 16;

/// Degree-0 normalization: 1 / (2 √π).
const SH_C0: f32 = 0.28209479177387814;

/// Degree-1 normalization: √(3 / 4π).
const SH_C1: f32 = 0.4886025119029199;

/// Degree-2 normalization constants (5 terms).
const SH_C2: [f32; 5] = [
    1.0925484305920792,   // √(15/π)/2
    -1.0925484305920792,  // -√(15/π)/2
    0.31539156525252005,  // √(5/π)/4
    -1.0925484305920792,  // -√(15/π)/2
    0.5462742152960396,   // √(15/π)/4
];

/// Degree-3 normalization constants (7 terms).
const SH_C3: [f32; 7] = [
    -0.5900435899266435,  // -√(35/(2π))/4
    2.890611442640554,    // √(105/π)/2
    -0.4570457994644658,  // -√(21/(2π))/4
    0.3731763325901154,   // √(7/π)/4
    -0.4570457994644658,  // -√(21/(2π))/4
    1.445305721320277,    // √(105/π)/4
    -0.5900435899266435,  // -√(35/(2π))/4
];

// ════════════════════════════════════════════════════════════════════════════
// Scalar single-gaussian evaluator
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate degree-3 real SH for a single gaussian at unit view direction `d`.
///
/// Returns linear RGB in [0, 1] (clamped, with Inria +0.5 DC offset).
///
/// # Inputs
/// - `sh`: at least 48 floats, layout: R=`sh[0..16]`, G=`sh[16..32]`,
///   B=`sh[32..48]`.
/// - `d`: unit-norm direction from gaussian center to camera. The caller
///   ensures normalization; this function does NOT re-normalize.
///
/// # Panics
/// In debug builds, panics if `sh.len() < 48`.
#[inline]
pub fn sh_eval_deg3(sh: &[f32], d: [f32; 3]) -> [f32; 3] {
    debug_assert!(sh.len() >= 48, "sh slice must have at least 48 elements");

    let [x, y, z] = d;

    // Precompute frequently-used products.
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // Degree-3 polynomial terms.
    let p3_neg3 = y * (3.0 * xx - yy);   // Y_3-3
    let p3_neg2 = xy * z;                  // Y_3-2
    let p3_neg1 = y * (4.0 * zz - xx - yy); // Y_3-1
    let p3_0    = z * (2.0 * zz - 3.0 * xx - 3.0 * yy); // Y_30
    let p3_pos1 = x * (4.0 * zz - xx - yy); // Y_31
    let p3_pos2 = z * (xx - yy);           // Y_32
    let p3_pos3 = x * (xx - 3.0 * yy);    // Y_33

    let mut rgb = [0.0f32; 3];

    for c in 0..3 {
        // Indexing into the channel's 16-element block.
        // SAFETY: guaranteed by the debug_assert above; in release we rely on
        // the caller's contract that sh.len() >= 48.
        let base = c * 16;

        // We use get_unchecked-equivalent via direct indexing — the compiler
        // can elide bounds checks after the debug_assert in release.
        let s = |k: usize| sh[base + k];

        let v = SH_C0 * s(0)
            // degree 1
            + (-SH_C1 * y) * s(1)
            + ( SH_C1 * z) * s(2)
            + (-SH_C1 * x) * s(3)
            // degree 2
            + SH_C2[0] * xy          * s(4)
            + SH_C2[1] * yz          * s(5)
            + SH_C2[2] * (2.0 * zz - xx - yy) * s(6)
            + SH_C2[3] * xz          * s(7)
            + SH_C2[4] * (xx - yy)   * s(8)
            // degree 3
            + SH_C3[0] * p3_neg3 * s(9)
            + SH_C3[1] * p3_neg2 * s(10)
            + SH_C3[2] * p3_neg1 * s(11)
            + SH_C3[3] * p3_0    * s(12)
            + SH_C3[4] * p3_pos1 * s(13)
            + SH_C3[5] * p3_pos2 * s(14)
            + SH_C3[6] * p3_pos3 * s(15);

        rgb[c] = (v + 0.5).clamp(0.0, 1.0);
    }

    rgb
}

// ════════════════════════════════════════════════════════════════════════════
// SIMD batched 16-gaussian evaluator
// ════════════════════════════════════════════════════════════════════════════

/// Batched SH eval: 16 gaussians at 16 view directions.
///
/// `sh_block`: `&[f32]` of length `>= 16 * 48 = 768`, laid out as
/// `[gaussian_0_sh[48], gaussian_1_sh[48], ..., gaussian_15_sh[48]]`.
///
/// `dirs`: one unit view direction per gaussian, `[[f32; 3]; 16]`.
///
/// `out`: per-gaussian RGB destination, `[[f32; 3]; 16]`.
///
/// Uses `F32x16` to evaluate all 16 gaussians' dot-products in lockstep.
/// For each of the 16 basis functions and 3 channels, a lane-wise multiply-add
/// accumulates `basis_k(d[g]) * sh_coeff[g][c][k]` across all 16 gaussians
/// simultaneously. On AVX-512 each inner iteration is a single `vfmadd`
/// instruction operating on all 16 lanes.
#[inline]
pub fn sh_eval_deg3_x16(
    sh_block: &[f32],
    dirs: &[[f32; 3]; 16],
    out: &mut [[f32; 3]; 16],
) {
    debug_assert!(sh_block.len() >= 16 * 48, "sh_block must have at least 768 elements");

    // Step 1: Evaluate the 16 basis values for each of the 16 gaussians.
    // basis[k][g] = k-th basis function evaluated at gaussian g's direction.
    let mut basis = [[0.0f32; 16]; 16];

    for g in 0..16 {
        let [x, y, z] = dirs[g];
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        basis[0][g]  = SH_C0;
        basis[1][g]  = -SH_C1 * y;
        basis[2][g]  =  SH_C1 * z;
        basis[3][g]  = -SH_C1 * x;
        basis[4][g]  = SH_C2[0] * xy;
        basis[5][g]  = SH_C2[1] * yz;
        basis[6][g]  = SH_C2[2] * (2.0 * zz - xx - yy);
        basis[7][g]  = SH_C2[3] * xz;
        basis[8][g]  = SH_C2[4] * (xx - yy);
        basis[9][g]  = SH_C3[0] * (y * (3.0 * xx - yy));
        basis[10][g] = SH_C3[1] * (xy * z);
        basis[11][g] = SH_C3[2] * (y * (4.0 * zz - xx - yy));
        basis[12][g] = SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
        basis[13][g] = SH_C3[4] * (x * (4.0 * zz - xx - yy));
        basis[14][g] = SH_C3[5] * (z * (xx - yy));
        basis[15][g] = SH_C3[6] * (x * (xx - 3.0 * yy));
    }

    // Step 2: For each channel, accumulate dot products across basis functions.
    // acc_c[lane g] = sum_k( basis[k][g] * sh_block[g*48 + c*16 + k] )
    let zero = F32x16::splat(0.0);
    let half = F32x16::splat(0.5);
    let lo   = F32x16::splat(0.0);
    let hi   = F32x16::splat(1.0);

    for c in 0..3 {
        let mut acc = zero;

        for k in 0..16 {
            // Gather basis_k values across 16 gaussians into one SIMD vector.
            let basis_vec = F32x16::from_array(basis[k]);

            // Gather the k-th SH coefficient for channel c, across 16 gaussians.
            let mut coeff_arr = [0.0f32; 16];
            for g in 0..16 {
                coeff_arr[g] = sh_block[g * 48 + c * 16 + k];
            }
            let coeff_vec = F32x16::from_array(coeff_arr);

            // acc += basis_vec * coeff_vec  (lane-wise multiply-add)
            acc = basis_vec.mul_add(coeff_vec, acc);
        }

        // Apply Inria +0.5 offset and clamp to [0, 1].
        let result = (acc + half).simd_clamp(lo, hi);
        let result_arr = result.to_array();

        // Scatter results back to AoS output.
        for g in 0..16 {
            out[g][c] = result_arr[g];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for float comparisons.
    const EPS: f32 = 1e-6;

    fn make_zero_sh() -> Vec<f32> {
        vec![0.0f32; 48]
    }

    // ── Test 1 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_deg0_returns_constant_offset_color() {
        // Degree-0 basis is rotation-invariant: only s[0] contributes,
        // and the result is the same regardless of view direction.
        let s0 = 0.7_f32;
        let expected = (SH_C0 * s0 + 0.5).clamp(0.0, 1.0);

        for c in 0..3 {
            let mut sh = make_zero_sh();
            sh[c * 16] = s0; // s[0] for channel c

            let d1 = [1.0_f32, 0.0, 0.0];
            let d2 = [0.0_f32, 1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt()];

            let rgb1 = sh_eval_deg3(&sh, d1);
            let rgb2 = sh_eval_deg3(&sh, d2);

            assert!(
                (rgb1[c] - expected).abs() < EPS,
                "channel {c} dir1: got {}, expected {expected}", rgb1[c]
            );
            assert!(
                (rgb2[c] - expected).abs() < EPS,
                "channel {c} dir2: got {}, expected {expected}", rgb2[c]
            );

            // Other channels should be clamped to 0.5 (zero coefficients).
            for other_c in 0..3 {
                if other_c != c {
                    assert!(
                        (rgb1[other_c] - 0.5).abs() < EPS,
                        "channel {other_c} should be 0.5 when c={c}"
                    );
                }
            }
        }
    }

    // ── Test 2 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_with_zero_coeffs_returns_half() {
        let sh = make_zero_sh();
        let dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0_f32 / 3.0_f32.sqrt(), 1.0_f32 / 3.0_f32.sqrt(), 1.0_f32 / 3.0_f32.sqrt()],
        ];
        for d in dirs {
            let rgb = sh_eval_deg3(&sh, d);
            for c in 0..3 {
                assert!(
                    (rgb[c] - 0.5).abs() < EPS,
                    "zero coeffs at dir {d:?}: channel {c} = {}, expected 0.5", rgb[c]
                );
            }
        }
    }

    // ── Test 3 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_view_dependent_changes_with_dir() {
        // s[1] = 1.0 for channel R: basis is -SH_C1 * y.
        // At (0,0,1): y=0 → v = 0 → rgb[0] = 0.5
        // At (0,1,0): y=1 → v = -SH_C1 → rgb[0] = clamp(0.5 - SH_C1, 0, 1)
        let mut sh = make_zero_sh();
        sh[1] = 1.0; // s[1] for channel R

        let rgb_z = sh_eval_deg3(&sh, [0.0, 0.0, 1.0]);
        let rgb_y = sh_eval_deg3(&sh, [0.0, 1.0, 0.0]);

        assert!(
            (rgb_z[0] - 0.5).abs() < EPS,
            "at (0,0,1): expected 0.5, got {}", rgb_z[0]
        );

        let expected_y = (0.5 + (-SH_C1)).clamp(0.0, 1.0);
        assert!(
            (rgb_y[0] - expected_y).abs() < EPS,
            "at (0,1,0): expected {expected_y}, got {}", rgb_y[0]
        );

        // The two outputs should differ.
        assert!(
            (rgb_z[0] - rgb_y[0]).abs() > 1e-4,
            "outputs should differ between directions"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_clamps_to_unit_interval() {
        // Large positive coefficient → clamp to 1.0
        let mut sh_pos = make_zero_sh();
        for c in 0..3 {
            sh_pos[c * 16] = 100.0;
        }
        let rgb_pos = sh_eval_deg3(&sh_pos, [1.0, 0.0, 0.0]);
        for c in 0..3 {
            assert_eq!(rgb_pos[c], 1.0, "channel {c} should clamp to 1.0");
        }

        // Large negative coefficient → clamp to 0.0
        let mut sh_neg = make_zero_sh();
        for c in 0..3 {
            sh_neg[c * 16] = -100.0;
        }
        let rgb_neg = sh_eval_deg3(&sh_neg, [1.0, 0.0, 0.0]);
        for c in 0..3 {
            assert_eq!(rgb_neg[c], 0.0, "channel {c} should clamp to 0.0");
        }
    }

    // ── Test 5 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_x16_matches_scalar_loop() {
        // Generate 16 × 48 deterministic SH coefficients via xorshift32.
        fn xorshift32(state: &mut u32) -> f32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            // Map to [-1, 1] for plausible SH coefficient range.
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }

        let mut rng = 0xDEAD_BEEF_u32;
        let mut sh_block = [0.0f32; 768];
        for v in sh_block.iter_mut() {
            *v = xorshift32(&mut rng);
        }

        // Generate 16 unit directions.
        let mut dirs = [[0.0f32; 3]; 16];
        for g in 0..16 {
            let a = xorshift32(&mut rng);
            let b = xorshift32(&mut rng);
            let c = xorshift32(&mut rng);
            let len = (a * a + b * b + c * c).sqrt().max(1e-8);
            dirs[g] = [a / len, b / len, c / len];
        }

        // Batched SIMD eval.
        let mut out_simd = [[0.0f32; 3]; 16];
        sh_eval_deg3_x16(&sh_block, &dirs, &mut out_simd);

        // Scalar reference loop.
        for g in 0..16 {
            let rgb_scalar = sh_eval_deg3(&sh_block[g * 48..], dirs[g]);
            for c in 0..3 {
                let delta = (out_simd[g][c] - rgb_scalar[c]).abs();
                assert!(
                    delta < 5e-5,
                    "gaussian {g} channel {c}: SIMD={} scalar={} delta={delta}",
                    out_simd[g][c], rgb_scalar[c]
                );
            }
        }
    }

    // ── Test 6 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_eval_x16_with_all_same_input_is_constant() {
        // All 16 gaussians have identical SH and identical direction.
        let mut sh_single = make_zero_sh();
        sh_single[0]  = 0.3;   // R s[0]
        sh_single[16] = 0.1;   // G s[0]
        sh_single[32] = -0.2;  // B s[0]
        sh_single[1]  = 0.5;   // R s[1]

        let mut sh_block = [0.0f32; 768];
        for g in 0..16 {
            sh_block[g * 48..g * 48 + 48].copy_from_slice(&sh_single);
        }

        let dir = [1.0_f32 / 3.0_f32.sqrt(), 1.0_f32 / 3.0_f32.sqrt(), 1.0_f32 / 3.0_f32.sqrt()];
        let dirs = [dir; 16];

        let mut out = [[0.0f32; 3]; 16];
        sh_eval_deg3_x16(&sh_block, &dirs, &mut out);

        let first = out[0];
        for g in 1..16 {
            for c in 0..3 {
                assert!(
                    (out[g][c] - first[c]).abs() < 1e-6,
                    "gaussian {g} channel {c}: {}, expected {}", out[g][c], first[c]
                );
            }
        }
    }

    // ── Test 7 ────────────────────────────────────────────────────────────
    #[test]
    fn sh_constants_match_normalization() {
        // For normalized real SH, ∫ Y_00² dΩ = 1 over the unit sphere.
        // Y_00 = SH_C0 (constant), ∫ dΩ = 4π.
        // So SH_C0² * 4π ≈ 1.
        let val = 4.0 * std::f32::consts::PI * SH_C0 * SH_C0;
        assert!(
            (val - 1.0).abs() < 1e-6,
            "SH_C0 normalization: 4π·SH_C0² = {val}, expected ≈1.0"
        );
    }

    // ── Test 8 — analytical ground truth at d=(0,0,1) ─────────────────────
    //
    // PP-13 PR 2 finding (promoted per the "biggest residual risk" rule
    // from PR 1): Tests 1-7 all compare scalar vs SIMD or check
    // degenerate inputs. A wrong SH constant (sign flip or magnitude
    // error) would affect scalar AND SIMD identically and pass every
    // other test. This test pins individual basis-function outputs to
    // analytical ground truth values at a known direction, so any
    // constant regression triggers immediately.
    //
    // At d = (0, 0, 1): x=0, y=0, z=1. Most cross-product basis terms
    // vanish; the non-zero ones are exactly:
    //   k = 0  (Y_00)                          : SH_C0
    //   k = 2  (Y_10  = SH_C1 · z)             : SH_C1
    //   k = 6  (Y_20  = SH_C2[2] · (2z² − x² − y²))  : SH_C2[2] · 2
    //   k = 12 (Y_30  = SH_C3[3] · z(2z² − 3x² − 3y²)) : SH_C3[3] · 2
    // All other 12 basis functions evaluate to zero.
    #[test]
    fn sh_eval_analytical_ground_truth_at_positive_z() {
        let d = [0.0f32, 0.0, 1.0];
        let expected_basis = [
            (0usize, SH_C0),
            (2, SH_C1),
            (6, SH_C2[2] * 2.0),
            (12, SH_C3[3] * 2.0),
        ];

        for &(k, expected_basis_val) in &expected_basis {
            // Single non-zero coefficient on channel R (lane k), value 1.0.
            // Channels G and B all-zero → should return exactly 0.5.
            let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
            sh[k] = 1.0;
            let rgb = sh_eval_deg3(&sh, d);

            let expected_r = (expected_basis_val + 0.5).clamp(0.0, 1.0);
            assert!(
                (rgb[0] - expected_r).abs() < 1e-5,
                "basis k={k}: expected R = clamp({expected_basis_val} + 0.5) = {expected_r}, got {}",
                rgb[0]
            );
            assert!(
                (rgb[1] - 0.5).abs() < 1e-6,
                "basis k={k}: G should be 0.5 (no coeffs), got {}",
                rgb[1]
            );
            assert!(
                (rgb[2] - 0.5).abs() < 1e-6,
                "basis k={k}: B should be 0.5 (no coeffs), got {}",
                rgb[2]
            );
        }

        // Negative case: every basis function that SHOULD evaluate to
        // zero at this direction (all the y- and x-bearing terms).
        let zero_basis_indices = [1usize, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15];
        for &k in &zero_basis_indices {
            let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
            sh[k] = 1.0;
            let rgb = sh_eval_deg3(&sh, d);
            assert!(
                (rgb[0] - 0.5).abs() < 1e-6,
                "basis k={k}: should vanish at d=(0,0,1), got R = {}",
                rgb[0]
            );
        }
    }
}

// Tests need SH_COEFFS_PER_GAUSSIAN from the sibling `gaussian` module.
// Importing in a cfg(test) block rather than the main module body keeps
// the production SH code self-contained (sh.rs only depends on
// `crate::simd`, never on `gaussian.rs`).
#[cfg(test)]
use crate::hpc::splat3d::gaussian::SH_COEFFS_PER_GAUSSIAN;
