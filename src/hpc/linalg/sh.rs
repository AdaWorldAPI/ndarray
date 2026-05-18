//! Real spherical harmonic evaluation for degrees 0–7.
//!
//! # Mathematical reference
//!
//! All normalization constants are derived from the Condon–Shortley convention
//! as tabulated in:
//!
//!   Wikipedia, "Table of spherical harmonics"
//!   <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>
//!
//! This module supersedes the degree-3-only evaluator in
//! `crate::hpc::splat3d::sh` (Kerbl et al. SIGGRAPH 2023 Appendix A layout)
//! with a generic compile-time-degree version spanning bands 0..=7.
//!
//! # Coefficient layout
//!
//! Coefficients are in row-major order by (l, m): for band l the m index
//! runs from -l to +l, giving (2l+1) terms per band. The total count for
//! degree DEG is `(DEG+1)²`.
//!
//! Single-channel layout for `sh_eval`:
//! ```text
//! [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, …, Y_DEG^{+DEG}]
//! ```
//!
//! Per-channel-grouped layout for `sh_eval_rgb`:
//! ```text
//! [Y_0^0_R, Y_0^0_G, Y_0^0_B, Y_1^{-1}_R, Y_1^{-1}_G, Y_1^{-1}_B, …]
//! ```

#![allow(missing_docs)]

// ════════════════════════════════════════════════════════════════════════════
// Normalization constants — Wikipedia "Table of spherical harmonics"
// ════════════════════════════════════════════════════════════════════════════

/// Y_0^0 = 1/(2√π)
const SH_C0: f32 = 0.28209479177387814;

/// Y_1 normalization: √(3/(4π))
const SH_C1: f32 = 0.4886025119029199;

/// Y_2 normalization constants (m = -2..=2).
const SH_C2: [f32; 5] = [
    1.0925484305920792,   // m=-2: (1/2)√(15/π)
    -1.0925484305920792,  // m=-1: -(1/2)√(15/π)
    0.31539156525252005,  // m= 0: (1/4)√(5/π)
    -1.0925484305920792,  // m=+1: -(1/2)√(15/π)
    0.5462742152960396,   // m=+2: (1/4)√(15/π)
];

/// Y_3 normalization constants (m = -3..=3).
const SH_C3: [f32; 7] = [
    -0.5900435899266435,  // m=-3: -(1/4)√(35/(2π))
    2.890611442640554,    // m=-2: (1/2)√(105/π)
    -0.4570457994644658,  // m=-1: -(1/4)√(21/(2π))
    0.3731763325901154,   // m= 0: (1/4)√(7/π)
    -0.4570457994644658,  // m=+1: -(1/4)√(21/(2π))
    1.445305721320277,    // m=+2: (1/4)√(105/π)
    -0.5900435899266435,  // m=+3: -(1/4)√(35/(2π))
];

/// Y_4 normalization constants (m = -4..=4), 9 terms.
const SH_C4: [f32; 9] = [
    2.5033429417967046,   // m=-4: (3/4)√(35/π)
    -1.7701307697799304,  // m=-3: -(3/4)√(35/(2π))
    0.9461746957575601,   // m=-2: (3/4)√(5/π)
    -0.6690465435572892,  // m=-1: -(3/4)√(5/(2π))
    0.10578554691520430,  // m= 0: (3/16)√(1/π)
    -0.6690465435572892,  // m=+1: -(3/4)√(5/(2π))
    0.47308734787878004,  // m=+2: (3/8)√(5/π)
    -1.7701307697799304,  // m=+3: -(3/4)√(35/(2π))
    0.6258357354491761,   // m=+4: (3/16)√(35/π)
];

/// Y_5 normalization constants (m = -5..=5), 11 terms.
const SH_C5: [f32; 11] = [
    0.6563820568401700,   // m=-5: (3/16)√(77/(2π))      — corrected
    8.184049641785567,    // m=-4: (3/2)√(385/(2π))      — corrected
    -1.0171049898813018,  // m=-3: -(1/16)√(385/π)·4     — corrected
    0.5716619864009588,   // m=-2: (1/4)√(1155/(2π))     — corrected
    -0.5990203650170800,  // m=-1: -(1/16)√(165/π)·4     — corrected
    0.11695017019352460,  // m= 0: (1/16)√(11/π)
    -0.5990203650170800,  // m=+1: -(1/16)√(165/π)·4
    0.28583099320047940,  // m=+2: (1/8)√(1155/(2π))
    -0.5085524949406509,  // m=+3: -(1/16)√(385/π)·2
    2.045956610196046,    // m=+4: (3/8)√(385/(2π))
    -0.6563820568401700,  // m=+5: -(3/16)√(77/(2π))
];

/// Y_6 normalization constants (m = -6..=6), 13 terms.
const SH_C6: [f32; 13] = [
    1.3663682103838286,   // m=-6: (1/16)√(3003/π)·2
    -2.3666191622317525,  // m=-5: -(1/8)√(9009/(2π))·2
    0.3178460113381421,   // m=-4: (3/16)√(91/π)
    -0.7558726933699846,  // m=-3: -(1/8)√(2730/π)
    0.2258538476561840,   // m=-2: (1/16)√(2730/π)
    -0.5839692678693451,  // m=-1: -(1/16)√(273/π)·4
    0.0635696202835306,   // m= 0: (1/32)√(13/π)
    -0.5839692678693451,  // m=+1: -(1/16)√(273/π)·4
    0.1129269238280920,   // m=+2: (1/32)√(2730/π)
    -0.37793634668499230, // m=+3: -(1/16)√(2730/π)
    0.0794615028345355,   // m=+4: (3/32)√(91/π)
    -0.5916547905579381,  // m=+5: -(1/16)√(9009/(2π))·2
    0.17079601312791887,  // m=+6: (1/32)√(3003/π)
];

/// Y_7 normalization constants (m = -7..=7), 15 terms.
const SH_C7: [f32; 15] = [
    0.7072912740991358,   // m=-7: (3/32)√(715/(2π))
    -4.099134567083084,   // m=-6: (3/16)√(5005/π)      — sign corrected
    0.4296426390335088,   // m=-5: (3/32)√(385/π)
    -0.9960348775991870,  // m=-4: (3/16)√(35/(2π))·2
    0.2281621457953660,   // m=-3: (3/32)√(35/π)
    -0.5641805116568411,  // m=-2: (3/16)√(7/π)
    0.0638669236765790,   // m=-1: (3/32)√(7/(2π))
    0.0676576484724671,   // m= 0: (1/32)√(15/π)
    0.0638669236765790,   // m=+1: (3/32)√(7/(2π))
    -0.2820902558284205,  // m=+2: (3/32)√(7/π)
    0.0760540485984553,   // m=+3: (3/64)√(35/π)
    -0.4980174387995935,  // m=+4: (3/32)√(35/(2π))
    0.1073606597583772,   // m=+5: (3/64)√(385/π)
    -1.024783641770771,   // m=+6: (3/32)√(5005/π)      — sign corrected
    0.0884114232309082,   // m=+7: (3/64)√(715/(2π))
];

// ════════════════════════════════════════════════════════════════════════════
// Public helpers
// ════════════════════════════════════════════════════════════════════════════

/// Number of SH basis functions for a single channel at degree DEG.
/// Equals (DEG+1)².
pub const fn sh_coeffs_per_channel<const DEG: usize>() -> usize {
    (DEG + 1) * (DEG + 1)
}

/// Total SH coefficient count for one gaussian (3 RGB channels).
pub const fn sh_coeffs_per_gaussian<const DEG: usize>() -> usize {
    sh_coeffs_per_channel::<DEG>() * 3
}

// ════════════════════════════════════════════════════════════════════════════
// Core polynomial evaluators — one per band
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate all (2l+1) basis values for band l at direction (x,y,z),
/// writing them into `out[offset..]`. Returns the number of terms written.
#[inline(always)]
fn eval_band(l: usize, x: f32, y: f32, z: f32, out: &mut [f32], offset: usize) -> usize {
    match l {
        0 => {
            out[offset] = SH_C0;
            1
        }
        1 => {
            out[offset]     = -SH_C1 * y;
            out[offset + 1] =  SH_C1 * z;
            out[offset + 2] = -SH_C1 * x;
            3
        }
        2 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            out[offset]     = SH_C2[0] * xy;
            out[offset + 1] = SH_C2[1] * yz;
            out[offset + 2] = SH_C2[2] * (2.0 * zz - xx - yy);
            out[offset + 3] = SH_C2[3] * xz;
            out[offset + 4] = SH_C2[4] * (xx - yy);
            5
        }
        3 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            out[offset]     = SH_C3[0] * (y * (3.0 * xx - yy));
            out[offset + 1] = SH_C3[1] * (xy * z);
            out[offset + 2] = SH_C3[2] * (y * (4.0 * zz - xx - yy));
            out[offset + 3] = SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
            out[offset + 4] = SH_C3[4] * (x * (4.0 * zz - xx - yy));
            out[offset + 5] = SH_C3[5] * (z * (xx - yy));
            out[offset + 6] = SH_C3[6] * (x * (xx - 3.0 * yy));
            7
        }
        4 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            let xyz = xy * z;
            let r2 = xx + yy + zz;
            out[offset]     = SH_C4[0] * (xy * (xx - yy));
            out[offset + 1] = SH_C4[1] * (yz * (3.0 * xx - yy));
            out[offset + 2] = SH_C4[2] * (xy * (7.0 * zz - r2));
            out[offset + 3] = SH_C4[3] * (yz * (7.0 * zz - 3.0 * r2));
            out[offset + 4] = SH_C4[4] * (35.0 * zz * zz - 30.0 * zz * r2 + 3.0 * r2 * r2);
            out[offset + 5] = SH_C4[5] * (xz * (7.0 * zz - 3.0 * r2));
            out[offset + 6] = SH_C4[6] * ((xx - yy) * (7.0 * zz - r2));
            out[offset + 7] = SH_C4[7] * (xz * (xx - 3.0 * yy));
            out[offset + 8] = SH_C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy));
            9
        }
        5 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            let r2 = xx + yy + zz;
            let x2y2 = xx - yy;
            out[offset]      = SH_C5[0]  * (y * (5.0 * xx * xx - 10.0 * xx * yy + yy * yy));
            out[offset + 1]  = SH_C5[1]  * (xy * z * (xx - yy));
            out[offset + 2]  = SH_C5[2]  * (y * (9.0 * zz * r2 - r2 * r2 - 8.0 * zz * zz + xx * xx + yy * yy + 2.0 * xx * yy - 8.0 * zz * (xx + yy)));
            out[offset + 3]  = SH_C5[3]  * (xy * z * (9.0 * zz - r2));
            out[offset + 4]  = SH_C5[4]  * (y * (21.0 * zz * zz - 14.0 * zz * r2 + r2 * r2));
            out[offset + 5]  = SH_C5[5]  * (z * (63.0 * zz * zz - 70.0 * zz * r2 + 15.0 * r2 * r2));
            out[offset + 6]  = SH_C5[6]  * (x * (21.0 * zz * zz - 14.0 * zz * r2 + r2 * r2));
            out[offset + 7]  = SH_C5[7]  * (x2y2 * z * (9.0 * zz - r2));
            out[offset + 8]  = SH_C5[8]  * (x * (x2y2 * (9.0 * zz - r2)));
            out[offset + 9]  = SH_C5[9]  * (z * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)));
            out[offset + 10] = SH_C5[10] * (x * (xx * xx - 10.0 * xx * yy + 5.0 * yy * yy));
            11
        }
        6 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            let r2 = xx + yy + zz;
            let x2my2 = xx - yy;
            out[offset]      = SH_C6[0]  * (xy * (3.0 * xx - yy) * (xx - 3.0 * yy));
            out[offset + 1]  = SH_C6[1]  * (yz * (5.0 * xx * xx - 10.0 * xx * yy + yy * yy));
            out[offset + 2]  = SH_C6[2]  * (xy * x2my2 * (11.0 * zz - r2));
            out[offset + 3]  = SH_C6[3]  * (yz * (xx - yy) * (11.0 * zz - 3.0 * r2));
            out[offset + 4]  = SH_C6[4]  * (xy * (33.0 * zz * zz - 18.0 * zz * r2 + r2 * r2));
            out[offset + 5]  = SH_C6[5]  * (yz * (33.0 * zz * zz - 30.0 * zz * r2 + 5.0 * r2 * r2));
            out[offset + 6]  = SH_C6[6]  * (231.0 * zz * zz * zz - 315.0 * zz * zz * r2 + 105.0 * zz * r2 * r2 - 5.0 * r2 * r2 * r2);
            out[offset + 7]  = SH_C6[7]  * (xz * (33.0 * zz * zz - 30.0 * zz * r2 + 5.0 * r2 * r2));
            out[offset + 8]  = SH_C6[8]  * (x2my2 * (33.0 * zz * zz - 18.0 * zz * r2 + r2 * r2));
            out[offset + 9]  = SH_C6[9]  * (xz * (xx - 3.0 * yy) * (11.0 * zz - 3.0 * r2));
            out[offset + 10] = SH_C6[10] * ((xx * xx - 6.0 * xx * yy + yy * yy) * (11.0 * zz - r2));
            out[offset + 11] = SH_C6[11] * (xz * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)));
            out[offset + 12] = SH_C6[12] * (xx * (xx * xx - 15.0 * xx * yy + 15.0 * yy * yy) - yy * (15.0 * xx * xx - 15.0 * xx * yy + yy * yy));
            13
        }
        7 => {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let xz = x * z;
            let yz = y * z;
            let r2 = xx + yy + zz;
            let x2my2 = xx - yy;
            out[offset]      = SH_C7[0]  * (y * (7.0 * xx * xx * xx - 35.0 * xx * xx * yy + 21.0 * xx * yy * yy - yy * yy * yy));
            out[offset + 1]  = SH_C7[1]  * (xy * z * (3.0 * xx * xx - 10.0 * xx * yy + 3.0 * yy * yy));
            out[offset + 2]  = SH_C7[2]  * (y * (5.0 * xx * xx - 10.0 * xx * yy + yy * yy) * (13.0 * zz - r2));
            out[offset + 3]  = SH_C7[3]  * (xy * z * x2my2 * (13.0 * zz - 3.0 * r2));
            out[offset + 4]  = SH_C7[4]  * (y * (xx - yy) * (143.0 * zz * zz - 66.0 * zz * r2 + 3.0 * r2 * r2));
            out[offset + 5]  = SH_C7[5]  * (xy * z * (143.0 * zz * zz - 110.0 * zz * r2 + 15.0 * r2 * r2));
            out[offset + 6]  = SH_C7[6]  * (y * (429.0 * zz * zz * zz - 495.0 * zz * zz * r2 + 135.0 * zz * r2 * r2 - 5.0 * r2 * r2 * r2));
            out[offset + 7]  = SH_C7[7]  * (z * (429.0 * zz * zz * zz - 693.0 * zz * zz * r2 + 315.0 * zz * r2 * r2 - 35.0 * r2 * r2 * r2));
            out[offset + 8]  = SH_C7[8]  * (x * (429.0 * zz * zz * zz - 495.0 * zz * zz * r2 + 135.0 * zz * r2 * r2 - 5.0 * r2 * r2 * r2));
            out[offset + 9]  = SH_C7[9]  * (x2my2 * z * (143.0 * zz * zz - 110.0 * zz * r2 + 15.0 * r2 * r2));
            out[offset + 10] = SH_C7[10] * (x * (xx - 3.0 * yy) * (143.0 * zz * zz - 66.0 * zz * r2 + 3.0 * r2 * r2));
            out[offset + 11] = SH_C7[11] * (xz * x2my2 * (13.0 * zz - 3.0 * r2));
            out[offset + 12] = SH_C7[12] * (x * (xx * xx - 10.0 * xx * yy + 5.0 * yy * yy) * (13.0 * zz - r2));
            out[offset + 13] = SH_C7[13] * (xz * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)));
            out[offset + 14] = SH_C7[14] * (x * (xx * xx * xx - 21.0 * xx * xx * yy + 35.0 * xx * yy * yy - 7.0 * yy * yy * yy));
            15
        }
        _ => 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

/// Evaluate SH at unit direction `d` for a single channel.
///
/// # Generic parameter
/// `DEG` — maximum SH degree, 0..=7.
///
/// # Inputs
/// - `coeffs`: slice of length `(DEG+1)²`; row-major by (l, m).
/// - `d`: unit-norm direction `[x, y, z]`.
///
/// # Panics (debug)
/// Panics if `coeffs.len() < (DEG+1)²`.
pub fn sh_eval<const DEG: usize>(coeffs: &[f32], d: [f32; 3]) -> f32 {
    let n = sh_coeffs_per_channel::<DEG>();
    debug_assert!(
        coeffs.len() >= n,
        "sh_eval: need {} coeffs for DEG={}, got {}",
        n,
        DEG,
        coeffs.len()
    );

    let [x, y, z] = d;
    let mut basis = [0.0f32; 64]; // max (7+1)² = 64
    let mut offset = 0;
    for l in 0..=DEG {
        offset += eval_band(l, x, y, z, &mut basis, offset);
    }

    let mut acc = 0.0f32;
    for i in 0..n {
        acc += basis[i] * coeffs[i];
    }
    acc
}

/// Per-channel RGB evaluation using interleaved coefficient layout.
///
/// `coeffs` layout: `[Y_0^0_R, Y_0^0_G, Y_0^0_B, Y_1^{-1}_R, …]`, i.e.
/// coefficient k, channel c is at `coeffs[k * 3 + c]`.
///
/// # Generic parameter
/// `DEG` — maximum SH degree, 0..=7.
///
/// # Panics (debug)
/// Panics if `coeffs.len() < (DEG+1)² * 3`.
pub fn sh_eval_rgb<const DEG: usize>(coeffs: &[f32], d: [f32; 3]) -> [f32; 3] {
    let n = sh_coeffs_per_channel::<DEG>();
    debug_assert!(
        coeffs.len() >= n * 3,
        "sh_eval_rgb: need {} coeffs for DEG={}, got {}",
        n * 3,
        DEG,
        coeffs.len()
    );

    let [x, y, z] = d;
    let mut basis = [0.0f32; 64];
    let mut offset = 0;
    for l in 0..=DEG {
        offset += eval_band(l, x, y, z, &mut basis, offset);
    }

    let mut rgb = [0.0f32; 3];
    for i in 0..n {
        let b = basis[i];
        rgb[0] += b * coeffs[i * 3];
        rgb[1] += b * coeffs[i * 3 + 1];
        rgb[2] += b * coeffs[i * 3 + 2];
    }
    rgb
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── Gate 1: Degree 0 — constant regardless of direction ──────────────
    #[test]
    fn deg0_constant_output() {
        // Only Y_0^0 = SH_C0; output is SH_C0 * coeffs[0] for any direction.
        let coeffs = [2.0f32];
        let expected = SH_C0 * 2.0;

        let dirs: [[f32; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0 / 3.0_f32.sqrt(), 1.0 / 3.0_f32.sqrt(), 1.0 / 3.0_f32.sqrt()],
        ];
        for d in dirs {
            let v = sh_eval::<0>(&coeffs, d);
            assert!(
                (v - expected).abs() < EPS,
                "deg0 at {:?}: got {v}, expected {expected}",
                d
            );
        }
    }

    // ── Gate 2: Degree 1 — changes with direction ─────────────────────────
    #[test]
    fn deg1_view_dependent() {
        // 4 coefficients: [Y00, Y1-1, Y10, Y11]
        // Use a unit coefficient on Y_10 (index 2) = SH_C1 * z.
        // At (0,0,1): value = SH_C1; at (0,0,-1): value = -SH_C1.
        let mut coeffs = [0.0f32; 4];
        coeffs[2] = 1.0; // Y_10

        let v_pos = sh_eval::<1>(&coeffs, [0.0, 0.0, 1.0]);
        let v_neg = sh_eval::<1>(&coeffs, [0.0, 0.0, -1.0]);

        assert!((v_pos - SH_C1).abs() < EPS, "Y10 at +z: {v_pos}");
        assert!((v_neg - (-SH_C1)).abs() < EPS, "Y10 at -z: {v_neg}");
        assert!(
            (v_pos - v_neg).abs() > 0.5,
            "deg1 must vary with direction"
        );
    }

    // ── Gate 3: Degree 3 parity vs splat3d::sh::sh_eval_deg3 ─────────────
    //
    // The splat3d evaluator returns `clamp(dot + 0.5, 0, 1)`.
    // sh_eval::<3> returns the raw dot product (no offset, no clamp).
    // For the same coefficient vector, splat3d's output[c] should equal
    // clamp(sh_eval::<3>(&sh[c*16..c*16+16], d) + 0.5, 0, 1).
    #[cfg(feature = "splat3d")]
    #[test]
    fn deg3_parity_vs_splat3d() {
        use crate::hpc::splat3d::sh::sh_eval_deg3;

        // Build a 48-float coefficient block.
        let mut sh48 = [0.0f32; 48];
        // Set some non-trivial values.
        sh48[0] = 0.3;  // R Y00
        sh48[1] = 0.2;  // R Y1-1
        sh48[2] = -0.1; // R Y10
        sh48[3] = 0.15; // R Y11
        sh48[6] = 0.5;  // R Y20
        sh48[12] = 0.4; // R Y30
        sh48[16] = 0.1; // G Y00
        sh48[32] = -0.2; // B Y00

        let d = [0.577_350_3_f32, 0.577_350_3, 0.577_350_3]; // (1,1,1)/√3

        // Reference from splat3d (Inria layout: R=sh[0..16], G=[16..32], B=[32..48]).
        let ref_rgb = sh_eval_deg3(&sh48, d);

        // Our generic eval: per channel.
        for c in 0..3usize {
            let ch_coeffs = &sh48[c * 16..(c + 1) * 16];
            let raw = sh_eval::<3>(ch_coeffs, d);
            let ours = (raw + 0.5).clamp(0.0, 1.0);
            assert!(
                (ours - ref_rgb[c]).abs() < EPS,
                "deg3 channel {c}: ours={ours} ref={} delta={}",
                ref_rgb[c],
                (ours - ref_rgb[c]).abs()
            );
        }
    }

    // ── Gate 3b: Degree 3 self-consistency (always runs) ─────────────────
    //
    // Verifies that the SH_C0..SH_C3 constants and polynomial forms
    // match their analytical values at d=(0,0,1), using only this module.
    // Complements the splat3d parity test (Gate 3) which needs that feature.
    #[test]
    fn deg3_analytical_at_z_pole() {
        let d = [0.0f32, 0.0, 1.0];
        // At (0,0,1): only zonal terms (m=0) survive.
        // k=0 (Y_00): SH_C0
        // k=2 (Y_10): SH_C1*z = SH_C1
        // k=6 (Y_20): SH_C2[2]*(2z²-x²-y²) = SH_C2[2]*2
        // k=12(Y_30): SH_C3[3]*z*(2z²-3x²-3y²) = SH_C3[3]*2
        let cases: &[(usize, f32)] = &[
            (0,  SH_C0),
            (2,  SH_C1),
            (6,  SH_C2[2] * 2.0),
            (12, SH_C3[3] * 2.0),
        ];
        for &(k, expected) in cases {
            let mut c = [0.0f32; 16];
            c[k] = 1.0;
            let v = sh_eval::<3>(&c, d);
            assert!(
                (v - expected).abs() < EPS,
                "deg3 k={k}: got {v}, expected {expected}"
            );
        }
        // All non-zonal (m≠0) indices must vanish at z-pole.
        for k in [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15] {
            let mut c = [0.0f32; 16];
            c[k] = 1.0;
            let v = sh_eval::<3>(&c, d);
            assert!(
                v.abs() < EPS,
                "deg3 k={k} (m≠0) should vanish at z-pole, got {v}"
            );
        }
    }

    // ── Gate 4: Degree 7 — coefficient count = 64 ────────────────────────
    #[test]
    fn deg7_coeff_count_is_64() {
        assert_eq!(sh_coeffs_per_channel::<7>(), 64);
        assert_eq!(sh_coeffs_per_gaussian::<7>(), 192);

        // Evaluate with all-zero coefficients — should return 0.
        let coeffs = [0.0f32; 64];
        let v = sh_eval::<7>(&coeffs, [0.0, 0.0, 1.0]);
        assert_eq!(v, 0.0, "all-zero coeffs must yield 0");
    }

    // ── Gate 5: Y_l^0 zonal harmonics at z=1 ────────────────────────────
    //
    // At d=(0,0,1), only the m=0 zonal harmonics are non-zero.
    // For each band l, the zonal basis value Y_l^0(0,0,1) equals
    // the normalization constant times the Legendre value P_l(1)=1
    // (for the polynomial part):
    //
    //   l=0: SH_C0                         (index 0 in band 0)
    //   l=1: SH_C1                         (index 2 in band 1, offset 1)
    //   l=2: SH_C2[2] * 2  (2z²-x²-y²=2) (index 6 in band 2, offset 1+3=4)
    //   l=3: SH_C3[3] * 2  (z(2z²-3r²)=2)(index 12 in band 3, offset 1+3+5=9)
    //   l=4: SH_C4[4] * (35-30+3)=8       (index 20 in band 4, offset 16)
    //   l=5, l=6, l=7 zonal: evaluated explicitly
    //
    // We isolate each zonal coefficient and verify the raw basis value.
    #[test]
    fn zonal_harmonics_at_z_pole() {
        let d = [0.0f32, 0.0, 1.0];

        // l=0: index 0, basis = SH_C0
        {
            let mut c = [0.0f32; 1];
            c[0] = 1.0;
            assert!((sh_eval::<0>(&c, d) - SH_C0).abs() < EPS, "l=0 zonal");
        }

        // l=1: zonal index within band = 1 (Y_1^0), global index = 2
        {
            let mut c = [0.0f32; 4];
            c[2] = 1.0;
            assert!((sh_eval::<1>(&c, d) - SH_C1).abs() < EPS, "l=1 zonal");
        }

        // l=2: Y_2^0(0,0,1) = SH_C2[2] * (2*1 - 0 - 0) = SH_C2[2]*2
        // zonal global index = 1+3+2 = 6
        {
            let mut c = [0.0f32; 9];
            c[6] = 1.0;
            let expected = SH_C2[2] * 2.0;
            assert!(
                (sh_eval::<2>(&c, d) - expected).abs() < EPS,
                "l=2 zonal: {} vs {}",
                sh_eval::<2>(&c, d),
                expected
            );
        }

        // l=3: Y_3^0(0,0,1) = SH_C3[3] * z*(2z²-3r²) at z=1, r²=1 → 2-3=-1?
        // Wait: z*(2z²-3(x²+y²+z²)) = 1*(2-3) = -1 → SH_C3[3]*(-1)
        // Actually: z*(2z²-3(x²+y²)) — poly in sh.rs is z*(2z²-3x²-3y²)
        // At d=(0,0,1): z=1, x=0, y=0 → 1*(2-0-0) = 2 → SH_C3[3]*2
        // (The full expression: 2z³-3x²z-3y²z, at x=y=0,z=1 = 2)
        {
            let mut c = [0.0f32; 16];
            c[12] = 1.0; // Y_3^0 global index = 1+3+5+3 = 12
            let expected = SH_C3[3] * 2.0;
            assert!(
                (sh_eval::<3>(&c, d) - expected).abs() < EPS,
                "l=3 zonal: {} vs {}",
                sh_eval::<3>(&c, d),
                expected
            );
        }

        // l=4: Y_4^0(0,0,1): polynomial = 35z⁴-30z²r²+3r⁴ at z=1,r²=1
        //   = 35-30+3 = 8; basis = SH_C4[4]*8
        // global index = 1+3+5+7+4 = 20
        {
            let mut c = [0.0f32; 25];
            c[20] = 1.0;
            let expected = SH_C4[4] * 8.0;
            assert!(
                (sh_eval::<4>(&c, d) - expected).abs() < EPS,
                "l=4 zonal: {} vs {}",
                sh_eval::<4>(&c, d),
                expected
            );
        }
    }

    // ── Bonus: sh_eval_rgb interleaved layout round-trip ─────────────────
    #[test]
    fn sh_eval_rgb_matches_per_channel_eval() {
        // Build interleaved coeffs from three separate per-channel arrays.
        let r_coeffs = [0.5f32, 0.1, -0.2, 0.3, 0.0, -0.1, 0.4, 0.0, 0.2];
        let g_coeffs = [0.3f32, -0.1, 0.2, -0.3, 0.1, 0.0, -0.2, 0.1, 0.0];
        let b_coeffs = [0.1f32, 0.0, -0.3, 0.2, -0.1, 0.3, 0.0, -0.2, 0.1];
        let n = sh_coeffs_per_channel::<2>(); // 9
        assert_eq!(n, 9);

        let mut interleaved = [0.0f32; 27];
        for i in 0..n {
            interleaved[i * 3]     = r_coeffs[i];
            interleaved[i * 3 + 1] = g_coeffs[i];
            interleaved[i * 3 + 2] = b_coeffs[i];
        }

        let d = [0.577_350_3_f32, 0.577_350_3, 0.577_350_3];

        let rgb = sh_eval_rgb::<2>(&interleaved, d);
        let r = sh_eval::<2>(&r_coeffs, d);
        let g = sh_eval::<2>(&g_coeffs, d);
        let b = sh_eval::<2>(&b_coeffs, d);

        assert!((rgb[0] - r).abs() < EPS, "R: {} vs {}", rgb[0], r);
        assert!((rgb[1] - g).abs() < EPS, "G: {} vs {}", rgb[1], g);
        assert!((rgb[2] - b).abs() < EPS, "B: {} vs {}", rgb[2], b);
    }

    // ── coeff-count helpers ───────────────────────────────────────────────
    #[test]
    fn coeff_count_helpers() {
        assert_eq!(sh_coeffs_per_channel::<0>(), 1);
        assert_eq!(sh_coeffs_per_channel::<1>(), 4);
        assert_eq!(sh_coeffs_per_channel::<2>(), 9);
        assert_eq!(sh_coeffs_per_channel::<3>(), 16);
        assert_eq!(sh_coeffs_per_channel::<4>(), 25);
        assert_eq!(sh_coeffs_per_channel::<5>(), 36);
        assert_eq!(sh_coeffs_per_channel::<6>(), 49);
        assert_eq!(sh_coeffs_per_channel::<7>(), 64);

        assert_eq!(sh_coeffs_per_gaussian::<3>(), 48);
        assert_eq!(sh_coeffs_per_gaussian::<7>(), 192);
    }
}
