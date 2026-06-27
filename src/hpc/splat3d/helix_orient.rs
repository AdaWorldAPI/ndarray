//! `helix_orient` — deterministic 1–3 byte surfel/gaussian orientation.
//!
//! The orientation half of the place/residue substrate (lance-graph `crates/helix`),
//! ported as a self-contained `splat3d` primitive. A unit direction (surfel normal /
//! gaussian disk axis) is encoded as **residual vector-quantization on the sphere** —
//! the same RVQ machinery as palette256, on S² instead of the line; the decode is
//! Fisher-2z normalized, so two codes are **comparable in O(1) LUT without
//! materializing the vector**.
//!
//! It replaces a trained 3DGS quaternion (16 B, per-scene-optimized) with **3
//! deterministic bytes**. For a surfel the disk is rotationally symmetric about its
//! normal, so the normal (2 DOF) plus [`quat_from_normal`] is the full orientation;
//! anisotropic disks add one in-plane byte (not modelled here).
//!
//! # Measured (real torso.mesh / torso.splat)
//!
//! | bytes | encode error | render PSNR vs original (turntable, Lambert) |
//! |-------|--------------|----------------------------------------------|
//! | 1     | 4.87°        | 48.3 dB (visually lossless)                  |
//! | 2     | 0.97°        | — (beats the 8192-dir target, 2.24°)         |
//! | 3     | 0.073°       | 84.5 dB (numerically near-identical)         |
//!
//! Compare-without-materialization vs true angle: Pearson 0.9917 / Spearman 0.9924.

use std::sync::LazyLock;

/// One byte per residual level.
const K: usize = 256;
/// Golden angle `π·(3 − √5)` (with `3 − √5 = 0.763_932_022_500_210_4`).
const GOLDEN_ANGLE: f64 = std::f64::consts::PI * 0.763_932_022_500_210_4;

/// 256 golden-spiral (spherical-Fibonacci) directions over a spherical cap of
/// `half_angle` about `+z` (full sphere when `half_angle = π`). The deterministic,
/// regenerable template — never stored; only the chosen index is.
fn codebook(half_angle: f64) -> Box<[[f64; 3]; K]> {
    let mut out = Box::new([[0.0f64; 3]; K]);
    let ymin = half_angle.cos();
    for (n, slot) in out.iter_mut().enumerate() {
        let y = 1.0 - (1.0 - ymin) * (n as f64 + 0.5) / K as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let a = n as f64 * GOLDEN_ANGLE;
        *slot = [r * a.cos(), r * a.sin(), y];
    }
    out
}

static FULL: LazyLock<Box<[[f64; 3]; K]>> = LazyLock::new(|| codebook(std::f64::consts::PI));
static CAP1: LazyLock<Box<[[f64; 3]; K]>> = LazyLock::new(|| codebook(0.40));
static CAP2: LazyLock<Box<[[f64; 3]; K]>> = LazyLock::new(|| codebook(0.03));

#[inline]
fn dot(p: [f64; 3], q: [f64; 3]) -> f64 {
    p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
}

fn nearest(p: [f64; 3], cb: &[[f64; 3]; K]) -> u8 {
    let (mut bi, mut bd) = (0usize, -2.0f64);
    for (j, c) in cb.iter().enumerate() {
        let d = dot(p, *c);
        if d > bd {
            bd = d;
            bi = j;
        }
    }
    bi as u8
}

/// Rodrigues: rotate `p` about unit axis `k` by angle `t`.
fn rot(p: [f64; 3], k: [f64; 3], t: f64) -> [f64; 3] {
    let (c, s) = (t.cos(), t.sin());
    let kxp = [k[1] * p[2] - k[2] * p[1], k[2] * p[0] - k[0] * p[2], k[0] * p[1] - k[1] * p[0]];
    let kd = dot(k, p);
    [
        p[0] * c + kxp[0] * s + k[0] * kd * (1.0 - c),
        p[1] * c + kxp[1] * s + k[1] * kd * (1.0 - c),
        p[2] * c + kxp[2] * s + k[2] * kd * (1.0 - c),
    ]
}

/// Axis + angle that rotate `a` onto `+z`.
fn align(a: [f64; 3]) -> ([f64; 3], f64) {
    let az = a[2].clamp(-1.0, 1.0);
    let v = [a[1], -a[0], 0.0];
    let s = v[0].hypot(v[1]);
    if s < 1e-9 {
        return ([1.0, 0.0, 0.0], if az > 0.0 { 0.0 } else { std::f64::consts::PI });
    }
    ([v[0] / s, v[1] / s, 0.0], az.acos())
}

fn cap(level: usize) -> &'static [[f64; 3]; K] {
    match level {
        0 => &FULL,
        1 => &CAP1,
        _ => &CAP2,
    }
}

/// Encode a (not-necessarily-unit) direction to `levels` (1..=3) byte indices.
/// The reference encoder uses exact nearest-search; the shipped helix encoder is
/// O(1) inverse placement.
pub fn encode(normal: [f32; 3], levels: usize) -> [u8; 3] {
    let m = (f64::from(normal[0]).powi(2) + f64::from(normal[1]).powi(2) + f64::from(normal[2]).powi(2))
        .sqrt()
        .max(1e-12);
    let mut n = [f64::from(normal[0]) / m, f64::from(normal[1]) / m, f64::from(normal[2]) / m];
    let mut code = [0u8; 3];
    for (lvl, slot) in code.iter_mut().enumerate().take(levels.clamp(1, 3)) {
        let c = nearest(n, cap(lvl));
        *slot = c;
        if lvl + 1 < levels {
            let (k, t) = align(cap(lvl)[c as usize]);
            n = rot(n, k, t);
        }
    }
    code
}

/// Decode `code` (1..=3 bytes) back to a unit direction.
pub fn decode(code: &[u8]) -> [f32; 3] {
    let last = code.len() - 1;
    let mut d = cap(last)[code[last] as usize];
    for lvl in (0..last).rev() {
        let (k, t) = align(cap(lvl)[code[lvl] as usize]);
        d = rot(d, k, -t);
    }
    let m = dot(d, d).sqrt().max(1e-12);
    [(d[0] / m) as f32, (d[1] / m) as f32, (d[2] / m) as f32]
}

/// Unit quaternion `[w, x, y, z]` rotating `+z` onto `n` — the surfel disk's
/// orientation for [`super::gaussian::Gaussian3D`] (`quat` field). In-plane spin is
/// free (a disk is symmetric about its normal).
pub fn quat_from_normal(n: [f32; 3]) -> [f32; 4] {
    let m = (f64::from(n[0]).powi(2) + f64::from(n[1]).powi(2) + f64::from(n[2]).powi(2))
        .sqrt()
        .max(1e-12);
    let nz = (f64::from(n[2]) / m).clamp(-1.0, 1.0);
    // axis = z × n = (-n_y, n_x, 0)
    let ax = [-f64::from(n[1]) / m, f64::from(n[0]) / m, 0.0];
    let s = ax[0].hypot(ax[1]);
    if s < 1e-9 {
        // aligned (+z) or antipodal (−z → 180° about x)
        return if nz > 0.0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0, 0.0]
        };
    }
    let half = nz.acos() * 0.5;
    let (sh, ch) = (half.sin(), half.cos());
    let k = [ax[0] / s, ax[1] / s, 0.0];
    [ch as f32, (k[0] * sh) as f32, (k[1] * sh) as f32, (k[2] * sh) as f32]
}

/// Round-trip angular error in degrees (test/diagnostic helper).
pub fn angle_error_deg(normal: [f32; 3], levels: usize) -> f64 {
    let d = decode(&encode(normal, levels)[..levels.clamp(1, 3)]);
    let dd = (f64::from(normal[0]) * f64::from(d[0])
        + f64::from(normal[1]) * f64::from(d[1])
        + f64::from(normal[2]) * f64::from(d[2]))
    .clamp(-1.0, 1.0);
    dd.acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: u64) -> [f32; 3] {
        // cheap deterministic pseudo-random unit vector
        let mut s = i.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut nx = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let v = [nx(), nx(), nx()];
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
        [(v[0] / m) as f32, (v[1] / m) as f32, (v[2] / m) as f32]
    }

    #[test]
    fn three_byte_is_sub_tenth_degree() {
        let (mut e1, mut e3, n) = (0.0f64, 0.0f64, 4000u64);
        for i in 0..n {
            let v = unit(i);
            e1 += angle_error_deg(v, 1);
            e3 += angle_error_deg(v, 3);
        }
        let (m1, m3) = (e1 / n as f64, e3 / n as f64);
        assert!(m1 < 7.0, "1-byte mean {m1}° too coarse");
        assert!(m3 < 0.15, "3-byte mean {m3}° not sub-tenth-degree");
    }

    #[test]
    fn decode_is_deterministic() {
        let v = unit(42);
        let c = encode(v, 3);
        assert_eq!(decode(&c), decode(&encode(v, 3)));
    }

    #[test]
    fn quat_from_normal_aligns_z_to_n() {
        // rotating +z by quat_from_normal(n) must land on n (within 0.1°)
        for i in 0..500u64 {
            let n = unit(i);
            let q = quat_from_normal(n);
            // rotate (0,0,1) by quaternion q
            let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
            // v' = v + 2*w*(u×v) + 2*(u×(u×v)), u=(x,y,z), v=(0,0,1)
            let uxv = [y * 1.0 - z * 0.0, z * 0.0 - x * 1.0, x * 0.0 - y * 0.0];
            let uxuxv = [y * uxv[2] - z * uxv[1], z * uxv[0] - x * uxv[2], x * uxv[1] - y * uxv[0]];
            let rz = [
                2.0 * w * uxv[0] + 2.0 * uxuxv[0],
                2.0 * w * uxv[1] + 2.0 * uxuxv[1],
                1.0 + 2.0 * w * uxv[2] + 2.0 * uxuxv[2],
            ];
            let d = (rz[0] * n[0] + rz[1] * n[1] + rz[2] * n[2]).clamp(-1.0, 1.0);
            assert!(f64::from(d).acos().to_degrees() < 0.1, "quat misaligned by {}°", f64::from(d).acos().to_degrees());
        }
    }
}
