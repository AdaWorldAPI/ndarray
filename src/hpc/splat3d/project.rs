//! EWA projection kernel — world-space 3D gaussians → screen-space 2D conics.
//!
//! # Mathematical claim (Zwicker 2001 / Kerbl 2023, Appendix A)
//!
//! For a 3D gaussian with world-space covariance Σ_world and a pinhole camera
//! with view matrix V (world → camera) and perspective Jacobian J ∈ ℝ^{2×3},
//! the Elliptical-Weighted-Average (EWA) projection gives screen-space
//! covariance:
//!
//! ```text
//!   W     = V[0:3, 0:3]                  (rotation/scale part of view)
//!   Σ_cam = W · Σ_world · Wᵀ            (3×3, world → camera)
//!   Σ_img = J · Σ_cam · Jᵀ              (2×2, camera → image)
//!   J     = [[ fx/z,  0,   -fx·x/z²  ],
//!            [  0,   fy/z, -fy·y/z²  ]]  (linearised perspective)
//! ```
//!
//! The 2D conic (inverse of Σ_img) is what the rasterizer feeds into its
//! α-blend kernel. A half-pixel anti-aliasing dilation (+0.3 on diagonals)
//! is applied before inversion following Kerbl 2023.
//!
//! # SIMD strategy
//!
//! The conic + depth + radius math runs through `F32x16` (16 gaussians/step).
//! SH evaluation stays scalar: 16 distinct view directions defeat the SH SIMD
//! batch (unique basis tables per direction), and the rasterizer — not the
//! projector — is the SH bottleneck.

use super::gaussian::{GaussianBatch, SH_COEFFS_PER_GAUSSIAN};
use super::sh::sh_eval_deg3;
use super::spd3::Spd3;
use crate::simd::F32x16;

// ════════════════════════════════════════════════════════════════════════════
// Padding helper (mirrors gaussian.rs)
// ════════════════════════════════════════════════════════════════════════════

#[inline]
const fn pad_to_lanes(n: usize, lanes: usize) -> usize {
    (n + lanes - 1) / lanes * lanes
}

// ════════════════════════════════════════════════════════════════════════════
// Camera
// ════════════════════════════════════════════════════════════════════════════

/// Pinhole camera with a row-major 4×4 view matrix.
///
/// `view` transforms world-space homogeneous points into camera space
/// (camera looks down +Z in camera space — i.e. μ_cam.z > 0 is in front).
///
/// `#[repr(C, align(64))]` — the struct fits one 64-byte cache line
/// (4×4×4 = 64 bytes for `view`, plus 9 more f32 = 36 bytes, plus padding).
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct Camera {
    /// Row-major 4×4 view matrix: world → camera.
    pub view: [[f32; 4]; 4],
    /// Focal lengths in pixels.
    pub fx: f32,
    pub fy: f32,
    /// Principal point in pixels.
    pub cx: f32,
    pub cy: f32,
    /// Near and far depth clip planes (camera-space Z).
    pub near: f32,
    pub far: f32,
    /// Image dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// World-space camera origin (for view-direction computation).
    pub position: [f32; 3],
}

impl Camera {
    /// Identity camera at origin looking down +Z, no perspective skew.
    ///
    /// `fx = fy = max(width, height)` so the projected pixel scale is sane;
    /// principal point at image centre; `near = 0.01`, `far = 1000.0`.
    pub fn identity_at_origin(width: u32, height: u32) -> Self {
        let f = width.max(height) as f32;
        Self {
            view: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            fx: f,
            fy: f,
            cx: width as f32 * 0.5,
            cy: height as f32 * 0.5,
            near: 0.01,
            far: 1000.0,
            width,
            height,
            position: [0.0, 0.0, 0.0],
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ProjectedBatch
// ════════════════════════════════════════════════════════════════════════════

/// Per-gaussian projection output. SoA layout, padded to `PREFERRED_F32_LANES`.
///
/// Each `Vec` has length `capacity`. Active slots are `0..len`; slots
/// `len..capacity` are zero-initialised padding.
#[derive(Debug, Clone)]
pub struct ProjectedBatch {
    /// Number of active projected gaussians.
    pub len: usize,
    /// Padded capacity (multiple of `PREFERRED_F32_LANES`).
    pub capacity: usize,
    /// Screen-space X coordinate (pixels).
    pub screen_x: Vec<f32>,
    /// Screen-space Y coordinate (pixels).
    pub screen_y: Vec<f32>,
    /// Camera-space depth (μ_cam.z).
    pub depth: Vec<f32>,
    /// 2D conic coefficient A = inv-cov[0][0].
    pub conic_a: Vec<f32>,
    /// 2D conic coefficient B = inv-cov[0][1].
    pub conic_b: Vec<f32>,
    /// 2D conic coefficient C = inv-cov[1][1].
    pub conic_c: Vec<f32>,
    /// 3σ screen-space bounding radius (pixels).
    pub radius: Vec<f32>,
    /// View-dependent red channel (from SH eval, clamped to [0, 1]).
    pub color_r: Vec<f32>,
    /// View-dependent green channel.
    pub color_g: Vec<f32>,
    /// View-dependent blue channel.
    pub color_b: Vec<f32>,
    /// Opacity (copied from `GaussianBatch`).
    pub opacity: Vec<f32>,
    /// Visibility flag: `1` = visible, `0` = culled
    /// (depth clip / off-screen / degenerate conic).
    pub valid: Vec<u8>,
}

/// The SIMD chunk width — always 16, regardless of the native SIMD tier.
/// `project_chunk_x16` processes exactly 16 gaussians per call via a
/// staging buffer, so `ProjectedBatch` and the logical walk in
/// `project_batch` are padded to this constant, not `PREFERRED_F32_LANES`.
const CHUNK_WIDTH: usize = 16;

impl ProjectedBatch {
    /// Allocate output batch with capacity for `n` gaussians (rounded up
    /// to `CHUNK_WIDTH = 16`). All buffers zero-initialised.
    pub fn with_capacity(n: usize) -> Self {
        let capacity = pad_to_lanes(n.max(1), CHUNK_WIDTH);
        Self {
            len: 0,
            capacity,
            screen_x: vec![0.0; capacity],
            screen_y: vec![0.0; capacity],
            depth: vec![0.0; capacity],
            conic_a: vec![0.0; capacity],
            conic_b: vec![0.0; capacity],
            conic_c: vec![0.0; capacity],
            radius: vec![0.0; capacity],
            color_r: vec![0.0; capacity],
            color_g: vec![0.0; capacity],
            color_b: vec![0.0; capacity],
            opacity: vec![0.0; capacity],
            valid: vec![0u8; capacity],
        }
    }

    /// Reset to empty without deallocating. Zeros the `valid` slice so
    /// any previously-written slots are no longer considered visible.
    pub fn clear(&mut self) {
        self.len = 0;
        for v in self.valid.iter_mut() {
            *v = 0;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Private math helpers
// ════════════════════════════════════════════════════════════════════════════

/// W · Σ_world · Wᵀ where W is an arbitrary (asymmetric) 3×3 matrix.
///
/// W is row-major: `w[row][col]`.
/// Σ_world is a symmetric SPD stored as `Spd3` (upper triangle).
///
/// NOT the same as `Spd3::sandwich` (which only handles symmetric M).
#[inline]
fn sandwich_3x3_asym(w: &[[f32; 3]; 3], sigma: &Spd3) -> Spd3 {
    // Expand Σ to full 3×3 (symmetric):
    let s = sigma.to_rows();

    // T = W · Σ  (3×3 × 3×3 → 3×3)
    let mut t = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = w[i][0] * s[0][j] + w[i][1] * s[1][j] + w[i][2] * s[2][j];
        }
    }

    // Result = T · Wᵀ  (3×3 × 3×3 → 3×3, upper triangle only)
    // (T · Wᵀ)[i][j] = T[i][0]*W[j][0] + T[i][1]*W[j][1] + T[i][2]*W[j][2]
    let a11 = t[0][0] * w[0][0] + t[0][1] * w[0][1] + t[0][2] * w[0][2];
    let a12 = t[0][0] * w[1][0] + t[0][1] * w[1][1] + t[0][2] * w[1][2];
    let a13 = t[0][0] * w[2][0] + t[0][1] * w[2][1] + t[0][2] * w[2][2];
    let a22 = t[1][0] * w[1][0] + t[1][1] * w[1][1] + t[1][2] * w[1][2];
    let a23 = t[1][0] * w[2][0] + t[1][1] * w[2][1] + t[1][2] * w[2][2];
    let a33 = t[2][0] * w[2][0] + t[2][1] * w[2][1] + t[2][2] * w[2][2];

    Spd3::new(a11, a12, a13, a22, a23, a33)
}

/// J · Σ_cam · Jᵀ where J ∈ ℝ^{2×3} is the perspective Jacobian.
///
/// Returns the upper triangle of the 2×2 symmetric Σ_img as `(a, b, c)`:
/// ```text
///   Σ_img = [[ a,  b ],
///            [ b,  c ]]
/// ```
/// where `a = Σ[0][0]`, `b = Σ[0][1]`, `c = Σ[1][1]`.
#[inline]
fn sandwich_2x3(j: &[[f32; 3]; 2], sigma_cam: &Spd3) -> (f32, f32, f32) {
    // Expand Σ_cam:
    let s = sigma_cam.to_rows();

    // T = J · Σ_cam  (2×3 × 3×3 → 2×3)
    let mut t = [[0.0f32; 3]; 2];
    for i in 0..2 {
        for k in 0..3 {
            t[i][k] = j[i][0] * s[0][k] + j[i][1] * s[1][k] + j[i][2] * s[2][k];
        }
    }

    // Σ_img = T · Jᵀ  (2×3 × 3×2 → 2×2, upper triangle)
    // Σ_img[i][j] = T[i][0]*J[j][0] + T[i][1]*J[j][1] + T[i][2]*J[j][2]
    let a = t[0][0] * j[0][0] + t[0][1] * j[0][1] + t[0][2] * j[0][2];
    let b = t[0][0] * j[1][0] + t[0][1] * j[1][1] + t[0][2] * j[1][2];
    let c = t[1][0] * j[1][0] + t[1][1] * j[1][1] + t[1][2] * j[1][2];

    (a, b, c)
}

// ════════════════════════════════════════════════════════════════════════════
// Scalar single-gaussian kernel (used internally and for tests)
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// SIMD inner loop: 16 gaussians per step
// ════════════════════════════════════════════════════════════════════════════

/// Staging buffer for one 16-wide chunk. Filled by `project_batch` from the
/// source `GaussianBatch` SoA channels; zero-padded beyond active data.
struct Chunk16 {
    mean_x: [f32; 16],
    mean_y: [f32; 16],
    mean_z: [f32; 16],
    quat_w: [f32; 16],
    quat_x: [f32; 16],
    quat_y: [f32; 16],
    quat_z: [f32; 16],
    scale_x: [f32; 16],
    scale_y: [f32; 16],
    scale_z: [f32; 16],
    opacity: [f32; 16],
    // SH: 16 gaussians × 48 coefficients each = 768 floats.
    sh: [f32; 16 * SH_COEFFS_PER_GAUSSIAN],
}

impl Chunk16 {
    fn zeros() -> Self {
        Self {
            mean_x: [0.0; 16],
            mean_y: [0.0; 16],
            mean_z: [0.0; 16],
            quat_w: [0.0; 16],
            quat_x: [0.0; 16],
            quat_y: [0.0; 16],
            quat_z: [0.0; 16],
            scale_x: [0.0; 16],
            scale_y: [0.0; 16],
            scale_z: [0.0; 16],
            opacity: [0.0; 16],
            sh: [0.0; 16 * SH_COEFFS_PER_GAUSSIAN],
        }
    }

    /// Fill from `gaussians[start..start+count]` (count ≤ 16).
    fn fill_from(gaussians: &GaussianBatch, start: usize, count: usize) -> Self {
        let mut c = Self::zeros();
        for k in 0..count {
            let i = start + k;
            c.mean_x[k] = gaussians.mean_x[i];
            c.mean_y[k] = gaussians.mean_y[i];
            c.mean_z[k] = gaussians.mean_z[i];
            c.quat_w[k] = gaussians.quat_w[i];
            c.quat_x[k] = gaussians.quat_x[i];
            c.quat_y[k] = gaussians.quat_y[i];
            c.quat_z[k] = gaussians.quat_z[i];
            c.scale_x[k] = gaussians.scale_x[i];
            c.scale_y[k] = gaussians.scale_y[i];
            c.scale_z[k] = gaussians.scale_z[i];
            c.opacity[k] = gaussians.opacity[i];
            let src_base = i * SH_COEFFS_PER_GAUSSIAN;
            let dst_base = k * SH_COEFFS_PER_GAUSSIAN;
            c.sh[dst_base..dst_base + SH_COEFFS_PER_GAUSSIAN]
                .copy_from_slice(&gaussians.sh[src_base..src_base + SH_COEFFS_PER_GAUSSIAN]);
        }
        c
    }
}

/// Project 16 gaussians from a pre-staged `Chunk16` using F32x16 SIMD for the
/// conic / depth / radius math. SH eval stays scalar (unique view direction
/// per gaussian).
///
/// `start` is the original batch offset (used to write into `out` and mask
/// against `gaussians.len`). `count` is how many of the 16 lanes are active
/// (lanes `count..16` are zero-padded and forced `valid = 0`).
fn project_chunk_x16(
    chunk: &Chunk16, gaussians_len: usize, start: usize, count: usize, camera: &Camera, out: &mut ProjectedBatch,
) {
    // ── 1. Load SoA mean lanes ───────────────────────────────────────────
    let mx = F32x16::from_slice(&chunk.mean_x);
    let my = F32x16::from_slice(&chunk.mean_y);
    let mz = F32x16::from_slice(&chunk.mean_z);

    // ── 2. μ_cam = V · (mx, my, mz, 1)ᵀ ────────────────────────────────
    let v = &camera.view;
    let v00 = F32x16::splat(v[0][0]);
    let v01 = F32x16::splat(v[0][1]);
    let v02 = F32x16::splat(v[0][2]);
    let v03 = F32x16::splat(v[0][3]);
    let v10 = F32x16::splat(v[1][0]);
    let v11 = F32x16::splat(v[1][1]);
    let v12 = F32x16::splat(v[1][2]);
    let v13 = F32x16::splat(v[1][3]);
    let v20 = F32x16::splat(v[2][0]);
    let v21 = F32x16::splat(v[2][1]);
    let v22 = F32x16::splat(v[2][2]);
    let v23 = F32x16::splat(v[2][3]);

    let cam_x = v00 * mx + v01 * my + v02 * mz + v03;
    let cam_y = v10 * mx + v11 * my + v12 * mz + v13;
    let cam_z = v20 * mx + v21 * my + v22 * mz + v23;

    // ── 3. Depth clip mask ───────────────────────────────────────────────
    let near = F32x16::splat(camera.near);
    let far = F32x16::splat(camera.far);
    // visible = cam_z >= near && cam_z <= far
    let depth_ok_ge = cam_z.simd_ge(near);
    let depth_ok_le = cam_z.simd_le(far);

    // ── 4. Perspective projection ─────────────────────────────────────────
    let one = F32x16::splat(1.0);
    let z_inv = one / cam_z;
    let fx = F32x16::splat(camera.fx);
    let fy = F32x16::splat(camera.fy);
    let cx = F32x16::splat(camera.cx);
    let cy = F32x16::splat(camera.cy);
    let sx = fx * cam_x * z_inv + cx;
    let sy = fy * cam_y * z_inv + cy;

    // ── 5. Reconstruct covariance + compute Σ_cam + Σ_img ─────────────────
    // W = upper-left 3×3 of view matrix (same for all 16 gaussians).
    let w00 = v[0][0];
    let w01 = v[0][1];
    let w02 = v[0][2];
    let w10 = v[1][0];
    let w11 = v[1][1];
    let w12 = v[1][2];
    let w20 = v[2][0];
    let w21 = v[2][1];
    let w22 = v[2][2];

    // Load quaternion and scale for 16 gaussians.
    let qw = F32x16::from_slice(&chunk.quat_w);
    let qx = F32x16::from_slice(&chunk.quat_x);
    let qy = F32x16::from_slice(&chunk.quat_y);
    let qz = F32x16::from_slice(&chunk.quat_z);
    let sc_x = F32x16::from_slice(&chunk.scale_x);
    let sc_y = F32x16::from_slice(&chunk.scale_y);
    let sc_z = F32x16::from_slice(&chunk.scale_z);

    // Quaternion → rotation matrix (mirrors gaussian.rs covariance_x16).
    let two = F32x16::splat(2.0);
    let xx = qx * qx;
    let yy = qy * qy;
    let zz = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    let r00 = one - two * (yy + zz);
    let r01 = two * (xy - wz);
    let r02 = two * (xz + wy);
    let r10 = two * (xy + wz);
    let r11 = one - two * (xx + zz);
    let r12 = two * (yz - wx);
    let r20 = two * (xz - wy);
    let r21 = two * (yz + wx);
    let r22 = one - two * (xx + yy);

    // s² = scale²
    let s0 = sc_x * sc_x;
    let s1 = sc_y * sc_y;
    let s2 = sc_z * sc_z;

    // M = R · diag(s²): scale column k by sₖ²
    let m00 = r00 * s0;
    let m01 = r01 * s1;
    let m02 = r02 * s2;
    let m10 = r10 * s0;
    let m11 = r11 * s1;
    let m12 = r12 * s2;
    let m20 = r20 * s0;
    let m21 = r21 * s1;
    let m22 = r22 * s2;

    // Σ_world upper triangle = M · Rᵀ
    let sw11 = m00 * r00 + m01 * r01 + m02 * r02;
    let sw12 = m00 * r10 + m01 * r11 + m02 * r12;
    let sw13 = m00 * r20 + m01 * r21 + m02 * r22;
    let sw22 = m10 * r10 + m11 * r11 + m12 * r12;
    let sw23 = m10 * r20 + m11 * r21 + m12 * r22;
    let sw33 = m20 * r20 + m21 * r21 + m22 * r22;

    // Σ_cam = W · Σ_world · Wᵀ  — SIMD lanes, scalar W entries
    // T = W · Σ_world  (each T[i][j] = sum_k W[i][k] * sw[k][j])
    // Σ_world full (using symmetry: sw[j][k] = sw[k][j]):
    //   sw[0] = [sw11, sw12, sw13]
    //   sw[1] = [sw12, sw22, sw23]
    //   sw[2] = [sw13, sw23, sw33]
    let w00s = F32x16::splat(w00);
    let w01s = F32x16::splat(w01);
    let w02s = F32x16::splat(w02);
    let w10s = F32x16::splat(w10);
    let w11s = F32x16::splat(w11);
    let w12s = F32x16::splat(w12);
    let w20s = F32x16::splat(w20);
    let w21s = F32x16::splat(w21);
    let w22s = F32x16::splat(w22);

    // T[0][j] = W[0][0]*sw[0][j] + W[0][1]*sw[1][j] + W[0][2]*sw[2][j]
    let t00 = w00s * sw11 + w01s * sw12 + w02s * sw13;
    let t01 = w00s * sw12 + w01s * sw22 + w02s * sw23;
    let t02 = w00s * sw13 + w01s * sw23 + w02s * sw33;

    let t10 = w10s * sw11 + w11s * sw12 + w12s * sw13;
    let t11 = w10s * sw12 + w11s * sw22 + w12s * sw23;
    let t12 = w10s * sw13 + w11s * sw23 + w12s * sw33;

    let t20 = w20s * sw11 + w21s * sw12 + w22s * sw13;
    let t21 = w20s * sw12 + w21s * sw22 + w22s * sw23;
    let t22 = w20s * sw13 + w21s * sw23 + w22s * sw33;

    // Σ_cam[i][j] = T[i][0]*W[j][0] + T[i][1]*W[j][1] + T[i][2]*W[j][2]
    // upper triangle: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    let sc11 = t00 * w00s + t01 * w01s + t02 * w02s;
    let sc12 = t00 * w10s + t01 * w11s + t02 * w12s;
    let sc13 = t00 * w20s + t01 * w21s + t02 * w22s;
    let sc22 = t10 * w10s + t11 * w11s + t12 * w12s;
    let sc23 = t10 * w20s + t11 * w21s + t12 * w22s;
    let sc33 = t20 * w20s + t21 * w21s + t22 * w22s;

    // Σ_img = J · Σ_cam · Jᵀ
    // J = [[ fx*z_inv, 0, -fx*cx_cam*z_inv2 ],
    //      [ 0, fy*z_inv, -fy*cy_cam*z_inv2 ]]
    let z_inv2 = z_inv * z_inv;
    let j00 = fx * z_inv;
    let j02 = fx * cam_x * (F32x16::splat(-1.0)) * z_inv2; // -fx*cam_x/z²
    let j11 = fy * z_inv;
    let j12 = fy * cam_y * (F32x16::splat(-1.0)) * z_inv2; // -fy*cam_y/z²
                                                           // j01=0, j10=0

    // T_img = J · Σ_cam  (2×3 × 3×3 → 2×3)
    // T_img[0][k] = J[0][0]*Σ[0][k] + J[0][2]*Σ[2][k]  (j01=0)
    // T_img[1][k] = J[1][1]*Σ[1][k] + J[1][2]*Σ[2][k]  (j10=0)
    // Σ_cam (full, using symmetry):
    //   col 0: sc11, sc12, sc13
    //   col 1: sc12, sc22, sc23
    //   col 2: sc13, sc23, sc33
    let ti00 = j00 * sc11 + j02 * sc13;
    let ti01 = j00 * sc12 + j02 * sc23;
    let ti02 = j00 * sc13 + j02 * sc33;

    let ti10 = j11 * sc12 + j12 * sc13;
    let ti11 = j11 * sc22 + j12 * sc23;
    let ti12 = j11 * sc23 + j12 * sc33;

    // Σ_img = T_img · Jᵀ  (2×3 × 3×2 → 2×2 upper triangle)
    // Σ_img[0][0] = T_img[0][0]*J[0][0] + T_img[0][2]*J[0][2]  (J[0][1]=0)
    // Σ_img[0][1] = T_img[0][0]*J[1][0] + T_img[0][1]*J[1][1] + T_img[0][2]*J[1][2]
    //             = T_img[0][1]*J[1][1] + T_img[0][2]*J[1][2]  (J[1][0]=0)
    // Σ_img[1][1] = T_img[1][1]*J[1][1] + T_img[1][2]*J[1][2]  (J[1][0]=0)
    let mut sig_a = ti00 * j00 + ti02 * j02;
    let sig_b = ti01 * j11 + ti02 * j12;
    let mut sig_c = ti11 * j11 + ti12 * j12;

    // Step 6: ½-pixel dilation.
    let dil = F32x16::splat(0.3);
    sig_a = sig_a + dil;
    sig_c = sig_c + dil;

    // Step 7: 2D conic.
    let det = sig_a * sig_c - sig_b * sig_b;
    let eps = F32x16::splat(1e-12);
    let det_ok = det.simd_gt(eps);
    let inv_det = one / det;
    let conic_a = inv_det * sig_c;
    let conic_b = F32x16::splat(0.0) - inv_det * sig_b;
    let conic_c = inv_det * sig_a;

    // Step 8: 3σ radius.
    let half = F32x16::splat(0.5);
    let three = F32x16::splat(3.0);
    let mid = half * (sig_a + sig_c);
    let d_disc = mid * mid - det;
    let lambda_max = mid + d_disc.simd_max(F32x16::splat(0.0)).sqrt();
    let radius = three * lambda_max.sqrt();

    // On-screen AABB cull (scalar per-lane: unpack then check).
    let mut sx_arr = [0.0f32; 16];
    let mut sy_arr = [0.0f32; 16];
    let mut rad_arr = [0.0f32; 16];
    sx.copy_to_slice(&mut sx_arr);
    sy.copy_to_slice(&mut sy_arr);
    radius.copy_to_slice(&mut rad_arr);

    let w_f = camera.width as f32;
    let h_f = camera.height as f32;

    // Gather scalar results for writeback.
    let mut depth_arr = [0.0f32; 16];
    let mut ca_arr = [0.0f32; 16];
    let mut cb_arr = [0.0f32; 16];
    let mut cc_arr = [0.0f32; 16];
    cam_z.copy_to_slice(&mut depth_arr);
    conic_a.copy_to_slice(&mut ca_arr);
    conic_b.copy_to_slice(&mut cb_arr);
    conic_c.copy_to_slice(&mut cc_arr);

    // Unpack depth_ok masks.
    let mut depth_ok_ge_arr = [0.0f32; 16];
    let mut depth_ok_le_arr = [0.0f32; 16];
    let mut det_ok_arr = [0.0f32; 16];
    // Select trick: mask selects 1.0 (true) or 0.0 (false).
    depth_ok_ge
        .select(F32x16::splat(1.0), F32x16::splat(0.0))
        .copy_to_slice(&mut depth_ok_ge_arr);
    depth_ok_le
        .select(F32x16::splat(1.0), F32x16::splat(0.0))
        .copy_to_slice(&mut depth_ok_le_arr);
    det_ok
        .select(F32x16::splat(1.0), F32x16::splat(0.0))
        .copy_to_slice(&mut det_ok_arr);

    for k in 0..16 {
        let idx = start + k;
        out.valid[idx] = 0;

        // Lane beyond active data — skip.
        if k >= count || idx >= gaussians_len {
            continue;
        }

        // Depth clip.
        if depth_ok_ge_arr[k] == 0.0 || depth_ok_le_arr[k] == 0.0 {
            continue;
        }

        // Degenerate conic.
        if det_ok_arr[k] == 0.0 {
            continue;
        }

        let r = rad_arr[k];
        let sxk = sx_arr[k];
        let syk = sy_arr[k];

        // On-screen AABB.
        if sxk + r < 0.0 || sxk - r >= w_f {
            continue;
        }
        if syk + r < 0.0 || syk - r >= h_f {
            continue;
        }

        // View direction → SH eval (scalar, using chunk's staged data).
        let mx_k = chunk.mean_x[k];
        let my_k = chunk.mean_y[k];
        let mz_k = chunk.mean_z[k];
        let dx = mx_k - camera.position[0];
        let dy = my_k - camera.position[1];
        let dz = mz_k - camera.position[2];
        let len_inv = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
        let dir = [dx * len_inv, dy * len_inv, dz * len_inv];

        let sh_base = k * SH_COEFFS_PER_GAUSSIAN;
        let sh_slice = &chunk.sh[sh_base..sh_base + SH_COEFFS_PER_GAUSSIAN];
        let [col_r, col_g, col_b] = sh_eval_deg3(sh_slice, dir);

        out.screen_x[idx] = sxk;
        out.screen_y[idx] = syk;
        out.depth[idx] = depth_arr[k];
        out.conic_a[idx] = ca_arr[k];
        out.conic_b[idx] = cb_arr[k];
        out.conic_c[idx] = cc_arr[k];
        out.radius[idx] = r;
        out.color_r[idx] = col_r;
        out.color_g[idx] = col_g;
        out.color_b[idx] = col_b;
        out.opacity[idx] = chunk.opacity[k];
        out.valid[idx] = 1;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public driver
// ════════════════════════════════════════════════════════════════════════════

/// Project all gaussians in `gaussians` into `out`, resetting `out` first.
///
/// Walks the input in 16-wide logical chunks using a staging buffer that is
/// always padded to exactly 16 slots, calling `project_chunk_x16` for each.
/// Trailing pad slots (indices `gaussians.len..capacity`) are never marked
/// `valid = 1`. After the call `out.len == gaussians.len`.
///
/// The output `out` is resized to hold at least as many slots as `gaussians`
/// has active gaussians (padded to 16). The caller must pre-size `out` to
/// at least `gaussians.len` before calling.
///
/// # Panics
/// Panics if `out.capacity < gaussians.len` (caller must pre-size).
pub fn project_batch(gaussians: &GaussianBatch, camera: &Camera, out: &mut ProjectedBatch) {
    // out is padded to CHUNK_WIDTH (16); each chunk writes to
    // out[start..start+16] so we need at least one chunk per 16 gaussians.
    let needed = pad_to_lanes(gaussians.len.max(1), CHUNK_WIDTH);
    assert!(
        out.capacity >= needed,
        "project_batch: out.capacity ({}) < needed ({needed}) for gaussians.len ({})",
        out.capacity,
        gaussians.len,
    );

    out.clear();
    out.len = gaussians.len;

    if gaussians.len == 0 {
        return;
    }

    let mut start = 0;
    while start < gaussians.len {
        let count = (gaussians.len - start).min(CHUNK_WIDTH);
        let chunk = Chunk16::fill_from(gaussians, start, count);
        project_chunk_x16(&chunk, gaussians.len, start, count, camera, out);
        start += CHUNK_WIDTH;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::gaussian::{Gaussian3D, GaussianBatch, SH_COEFFS_PER_GAUSSIAN};
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Build a minimal GaussianBatch with one gaussian at `mean`, identity
    /// rotation, given scale, zero SH, and opacity 1.
    fn single_gaussian(
        mean: [f32; 3], scale: [f32; 3], sh_override: Option<[f32; SH_COEFFS_PER_GAUSSIAN]>,
    ) -> GaussianBatch {
        let mut b = GaussianBatch::with_capacity(1);
        let mut g = Gaussian3D::unit();
        g.mean = mean;
        g.scale = scale;
        g.quat = [1.0, 0.0, 0.0, 0.0];
        g.opacity = 1.0;
        if let Some(sh) = sh_override {
            g.sh = sh;
        }
        b.push(g);
        b
    }

    /// Scalar reference for `project_batch` — used in x16-vs-scalar parity test.
    fn project_one_scalar(
        gaussians: &GaussianBatch, i: usize, camera: &Camera,
    ) -> Option<(f32, f32, f32, f32, f32, f32, f32)> {
        let mx = gaussians.mean_x[i];
        let my = gaussians.mean_y[i];
        let mz = gaussians.mean_z[i];
        let v = &camera.view;
        let cam_x = v[0][0] * mx + v[0][1] * my + v[0][2] * mz + v[0][3];
        let cam_y = v[1][0] * mx + v[1][1] * my + v[1][2] * mz + v[1][3];
        let cam_z = v[2][0] * mx + v[2][1] * my + v[2][2] * mz + v[2][3];
        if cam_z < camera.near || cam_z > camera.far {
            return None;
        }
        let z_inv = 1.0 / cam_z;
        let sx = camera.fx * cam_x * z_inv + camera.cx;
        let sy = camera.fy * cam_y * z_inv + camera.cy;
        let z_inv2 = z_inv * z_inv;
        let j: [[f32; 3]; 2] = [
            [camera.fx * z_inv, 0.0, -camera.fx * cam_x * z_inv2],
            [0.0, camera.fy * z_inv, -camera.fy * cam_y * z_inv2],
        ];
        let w: [[f32; 3]; 3] = [[v[0][0], v[0][1], v[0][2]], [v[1][0], v[1][1], v[1][2]], [v[2][0], v[2][1], v[2][2]]];
        let sigma_world = Spd3::from_scale_quat(
            [gaussians.scale_x[i], gaussians.scale_y[i], gaussians.scale_z[i]],
            [gaussians.quat_w[i], gaussians.quat_x[i], gaussians.quat_y[i], gaussians.quat_z[i]],
        );
        let sigma_cam = sandwich_3x3_asym(&w, &sigma_world);
        let (mut sig_a, mut sig_b, mut sig_c) = sandwich_2x3(&j, &sigma_cam);
        sig_a += 0.3;
        sig_c += 0.3;
        let det = sig_a * sig_c - sig_b * sig_b;
        if det <= 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let conic_a = inv_det * sig_c;
        let conic_b = -inv_det * sig_b;
        let conic_c = inv_det * sig_a;
        let mid = 0.5 * (sig_a + sig_c);
        let d_disc = mid * mid - det;
        let lambda_max = mid + d_disc.max(0.0).sqrt();
        let radius = 3.0 * lambda_max.sqrt();
        let w_f = camera.width as f32;
        let h_f = camera.height as f32;
        if sx + radius < 0.0 || sx - radius >= w_f {
            return None;
        }
        if sy + radius < 0.0 || sy - radius >= h_f {
            return None;
        }
        Some((sx, sy, cam_z, conic_a, conic_b, conic_c, radius))
    }

    // ── Test 1 ──────────────────────────────────────────────────────────────

    #[test]
    fn camera_identity_at_origin_sane_defaults() {
        let cam = Camera::identity_at_origin(512, 400);
        assert_eq!(cam.width, 512);
        assert_eq!(cam.height, 400);
        assert!(approx(cam.fx, 512.0, 1e-6), "fx={}", cam.fx);
        assert!(approx(cam.fy, 512.0, 1e-6), "fy={}", cam.fy);
        assert!(approx(cam.cx, 256.0, 1e-6), "cx={}", cam.cx);
        assert!(approx(cam.cy, 200.0, 1e-6), "cy={}", cam.cy);
        assert!(approx(cam.near, 0.01, 1e-6), "near={}", cam.near);
        assert!(approx(cam.far, 1000.0, 1e-6), "far={}", cam.far);
        // position at origin
        assert_eq!(cam.position, [0.0, 0.0, 0.0]);
    }

    // ── Test 2 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_origin_gaussian_at_depth_1_lands_at_screen_center() {
        let cam = Camera::identity_at_origin(512, 512);
        let gaussians = single_gaussian([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 1, "gaussian should be visible");
        assert!(approx(out.screen_x[0], 256.0, 1.0), "screen_x={}", out.screen_x[0]);
        assert!(approx(out.screen_y[0], 256.0, 1.0), "screen_y={}", out.screen_y[0]);
        assert!(approx(out.depth[0], 1.0, 1e-4), "depth={}", out.depth[0]);
    }

    // ── Test 3 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_culls_behind_near_plane() {
        let cam = Camera::identity_at_origin(512, 512);
        // near = 0.01, put gaussian at z = -1 (behind camera)
        let gaussians = single_gaussian([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 0, "behind near plane should be culled");
    }

    // ── Test 4 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_culls_beyond_far_plane() {
        let mut cam = Camera::identity_at_origin(512, 512);
        cam.far = 1000.0;
        let gaussians = single_gaussian([0.0, 0.0, 2000.0], [1.0, 1.0, 1.0], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 0, "beyond far plane should be culled");
    }

    // ── Test 5 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_culls_off_screen() {
        // 64×64 image, gaussian at (100, 0, 1) — far off screen
        let cam = Camera::identity_at_origin(64, 64);
        let gaussians = single_gaussian([100.0, 0.0, 1.0], [0.01, 0.01, 0.01], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 0, "off-screen gaussian should be culled");
    }

    // ── Test 6 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_conic_is_positive_definite_for_isotropic_gaussian() {
        let cam = Camera::identity_at_origin(512, 512);
        let gaussians = single_gaussian([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 1, "should be visible");
        let a = out.conic_a[0];
        let b = out.conic_b[0];
        let c = out.conic_c[0];
        assert!(a > 0.0, "conic_a must be > 0, got {a}");
        assert!(c > 0.0, "conic_c must be > 0, got {c}");
        assert!(a * c - b * b > 0.0, "conic must be SPD: a*c - b² = {}", a * c - b * b);
    }

    // ── Test 7 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_chunk_x16_matches_scalar_loop() {
        // Build 32 distinct gaussians with small positive scales at varying depths.
        let mut batch = GaussianBatch::with_capacity(32);
        let mut state = 0xDEAD_BEEFu32;
        let mut rng = |s: &mut u32| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f32) / (u32::MAX as f32)
        };
        for i in 0..32 {
            let mut g = Gaussian3D::unit();
            g.mean = [rng(&mut state) * 2.0 - 1.0, rng(&mut state) * 2.0 - 1.0, 1.0 + rng(&mut state) * 5.0];
            g.scale = [0.1 + rng(&mut state) * 0.4; 3];
            // vary i to distinguish gaussians
            g.scale[0] += i as f32 * 0.01;
            g.quat = [1.0, 0.0, 0.0, 0.0];
            g.opacity = rng(&mut state);
            batch.push(g);
        }
        let cam = Camera::identity_at_origin(512, 512);
        let mut out = ProjectedBatch::with_capacity(batch.capacity);
        project_batch(&batch, &cam, &mut out);

        for i in 0..32 {
            let scalar = project_one_scalar(&batch, i, &cam);
            match scalar {
                None => {
                    assert_eq!(out.valid[i], 0, "lane {i}: SIMD says valid but scalar says culled");
                }
                Some((sx, sy, depth, ca, cb, cc, rad)) => {
                    assert_eq!(out.valid[i], 1, "lane {i}: SIMD culled but scalar says visible");
                    let tol = 1e-3;
                    assert!(
                        approx(out.screen_x[i], sx, tol),
                        "lane {i} screen_x: simd={} scalar={sx}",
                        out.screen_x[i]
                    );
                    assert!(
                        approx(out.screen_y[i], sy, tol),
                        "lane {i} screen_y: simd={} scalar={sy}",
                        out.screen_y[i]
                    );
                    assert!(approx(out.depth[i], depth, tol), "lane {i} depth: simd={} scalar={depth}", out.depth[i]);
                    assert!(approx(out.conic_a[i], ca, tol), "lane {i} conic_a: simd={} scalar={ca}", out.conic_a[i]);
                    assert!(approx(out.conic_b[i], cb, tol), "lane {i} conic_b: simd={} scalar={cb}", out.conic_b[i]);
                    assert!(approx(out.conic_c[i], cc, tol), "lane {i} conic_c: simd={} scalar={cc}", out.conic_c[i]);
                    assert!(approx(out.radius[i], rad, tol), "lane {i} radius: simd={} scalar={rad}", out.radius[i]);
                }
            }
        }
    }

    // ── Test 8 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_radius_scales_with_covariance_magnitude() {
        let cam = Camera::identity_at_origin(1024, 1024);
        let g1 = single_gaussian([0.0, 0.0, 2.0], [1.0, 1.0, 1.0], None);
        let g2 = single_gaussian([0.0, 0.0, 2.0], [2.0, 2.0, 2.0], None);

        let mut out1 = ProjectedBatch::with_capacity(g1.capacity);
        let mut out2 = ProjectedBatch::with_capacity(g2.capacity);
        project_batch(&g1, &cam, &mut out1);
        project_batch(&g2, &cam, &mut out2);

        assert_eq!(out1.valid[0], 1, "g1 should be visible");
        assert_eq!(out2.valid[0], 1, "g2 should be visible");

        let r1 = out1.radius[0];
        let r2 = out2.radius[0];
        // Covariance scales as s², so σ scales as s → radius ≈ 2× for 2× scale.
        // We check within 20% tolerance.
        let ratio = r2 / r1;
        assert!(approx(ratio, 2.0, 0.3), "radius ratio should be ~2, got {ratio} (r1={r1}, r2={r2})");
    }

    // ── Test 9 ──────────────────────────────────────────────────────────────

    #[test]
    fn project_view_direction_normalized() {
        // DC-only SH: sh[0]=1.0 → R channel gets SH_C0 * 1.0 + 0.5
        // (the Inria +0.5 offset from sh_eval_deg3)
        const SH_C0: f32 = 0.28209479177387814;
        let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
        sh[0] = 1.0; // R channel DC coefficient
        let cam = Camera::identity_at_origin(512, 512);
        let gaussians = single_gaussian([0.0, 0.0, 5.0], [1.0, 1.0, 1.0], Some(sh));
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 1, "should be visible");
        // R = clamp(SH_C0 * 1.0 + 0.5, 0, 1)
        let expected_r = (SH_C0 + 0.5).clamp(0.0, 1.0);
        assert!(approx(out.color_r[0], expected_r, 1e-5), "R color: got {}, expected {expected_r}", out.color_r[0]);
        // G channel: all-zero SH → 0.5
        assert!(approx(out.color_g[0], 0.5, 1e-5), "G should be 0.5, got {}", out.color_g[0]);
        // B channel: all-zero SH → 0.5
        assert!(approx(out.color_b[0], 0.5, 1e-5), "B should be 0.5, got {}", out.color_b[0]);
    }

    // ── Test 10 ─────────────────────────────────────────────────────────────

    #[test]
    fn project_clear_resets_len_and_valid() {
        let cam = Camera::identity_at_origin(512, 512);
        let gaussians = single_gaussian([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.len, 1);
        assert_eq!(out.valid[0], 1);
        out.clear();
        assert_eq!(out.len, 0, "clear should set len=0");
        for (i, &v) in out.valid.iter().enumerate() {
            assert_eq!(v, 0, "valid[{i}] should be 0 after clear");
        }
    }

    // ── Test 11 — analytical ground truth for W·Σ·Wᵀ with non-identity W ───
    //
    // PP-13 PR 3 P0.1 (promoted): Tests 2–10 all use `Camera::identity_at_origin`,
    // which has W=I₃ in the upper-left 3×3 of the view matrix. The W·Σ·Wᵀ
    // sandwich is therefore trivially Σ for every test — a sign error in
    // the SIMD `sc12/sc13/sc23` accumulators (the asymmetric 3×3 cross
    // terms in `project_chunk_x16`) would produce wrong projected ellipses
    // for any rotated camera while passing all other tests.
    //
    // Setup: 90° rotation about +Y in the view matrix, gaussian at world
    // (-5, 0, 0) so its camera-frame position is R_y(90°)·(-5,0,0)ᵀ = (0,0,5)
    // — i.e. directly in front of the camera at depth 5. Σ_world =
    // diag(4, 1, 0.25) from scale = [2, 1, 0.5] with identity quat.
    //
    // Analytical Σ_cam = R_y(90°) · diag(4, 1, 0.25) · R_y(90°)ᵀ
    //                  = diag(0.25, 1, 4)
    // (axes permuted by the rotation — the X-scale of Σ_world ends up on
    // the Z-axis of Σ_cam and vice versa).
    //
    // J at μ_cam=(0,0,5):
    //   J = [[fx/5,  0,  0],
    //        [ 0,  fy/5, 0]]
    //   (the -fx·x/z² and -fy·y/z² terms vanish because cam_x = cam_y = 0)
    //
    // J · Σ_cam · Jᵀ = diag((fx/5)²·0.25, (fy/5)²·1)
    //                = [(fx²/100, 0), (0, fy²/25)]
    //
    // With fx = fy = 512: Σ_img = [(2621.44, 0), (0, 10485.76)] pre-dilation.
    // Add 0.3 to each diagonal: Σ_img = [(2621.74, 0), (0, 10486.06)].
    //
    // Conic = inv(Σ_img):
    //   det = 2621.74 · 10486.06 ≈ 2.749e7
    //   conic_a =  10486.06 / det ≈ 3.81e-4
    //   conic_b = 0
    //   conic_c =  2621.74  / det ≈ 9.54e-5
    //
    // A transpose error in the SIMD sandwich (e.g. swapping `t00*w10s` for
    // `t10*w00s`) would produce wrong sig_a/sig_c values that this test
    // would fail.
    #[test]
    fn project_non_identity_view_rotation_matches_analytical() {
        // R_y(90°): [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]] with cos=0, sin=1.
        let view = [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let fx = 512.0_f32;
        let fy = 512.0_f32;
        let cx = 256.0_f32;
        let cy = 256.0_f32;
        let cam = Camera {
            view,
            fx,
            fy,
            cx,
            cy,
            near: 0.01,
            far: 1000.0,
            width: 512,
            height: 512,
            position: [0.0, 0.0, 0.0],
        };
        // Gaussian at world (-5, 0, 0) — camera-frame position (0, 0, 5).
        // scale = [2, 1, 0.5] → Σ_world = diag(4, 1, 0.25).
        let gaussians = single_gaussian([-5.0, 0.0, 0.0], [2.0, 1.0, 0.5], None);
        let mut out = ProjectedBatch::with_capacity(gaussians.capacity);
        project_batch(&gaussians, &cam, &mut out);
        assert_eq!(out.valid[0], 1, "should be visible after 90° Y rotation");

        // Screen center (μ_cam_xy = 0).
        assert!((out.screen_x[0] - cx).abs() < 1e-3, "screen_x = {}, expected cx = {cx}", out.screen_x[0]);
        assert!((out.screen_y[0] - cy).abs() < 1e-3, "screen_y = {}, expected cy = {cy}", out.screen_y[0]);
        // Depth = camera-frame z = 5.
        assert!((out.depth[0] - 5.0).abs() < 1e-4, "depth = {}, expected 5.0", out.depth[0]);

        // Σ_img after AA dilation: [[fx²·0.25/25 + 0.3, 0], [0, fy²·1/25 + 0.3]].
        // Note: J at z=5 ⇒ (fx/5)²·0.25 = fx²/100, and (fy/5)²·1 = fy²/25.
        let sig_a_expected = fx * fx / 100.0 + 0.3;
        let sig_c_expected = fy * fy / 25.0 + 0.3;
        let det = sig_a_expected * sig_c_expected;
        let conic_a_expected = sig_c_expected / det;
        let conic_b_expected = 0.0;
        let conic_c_expected = sig_a_expected / det;

        // Relative tolerance 1e-3 — the SIMD path through three matrix
        // products (W·Σ, ·Wᵀ, J·Σ_cam·Jᵀ) accumulates ~1e-4 absolute.
        assert!(
            (out.conic_a[0] - conic_a_expected).abs() < 1e-6,
            "conic_a = {}, expected {conic_a_expected}",
            out.conic_a[0]
        );
        assert!(
            (out.conic_b[0] - conic_b_expected).abs() < 1e-6,
            "conic_b = {}, expected {conic_b_expected} (Σ_cam is axis-aligned → b=0)",
            out.conic_b[0]
        );
        assert!(
            (out.conic_c[0] - conic_c_expected).abs() < 1e-6,
            "conic_c = {}, expected {conic_c_expected}",
            out.conic_c[0]
        );

        // Radius = 3 · sqrt(λ_max(Σ_img)). λ_max = max(sig_a, sig_c) since
        // off-diagonal is 0. sig_c is the larger.
        let radius_expected = 3.0 * sig_c_expected.sqrt();
        assert!(
            (out.radius[0] - radius_expected).abs() < 1e-3,
            "radius = {}, expected {radius_expected}",
            out.radius[0]
        );
    }

    // ── Test 12 — partial-chunk lane masking (PP-13 PR 3 P1 promoted) ──────
    //
    // Confirms the `k >= count || idx >= gaussians.len` lane guard in
    // `project_chunk_x16` correctly marks trailing padded lanes as
    // invalid when `gaussians.len` is not a multiple of 16.
    #[test]
    fn project_partial_chunk_masks_padded_lanes() {
        for n in [1usize, 7, 15, 17, 23, 31] {
            let mut batch = GaussianBatch::with_capacity(n);
            for _ in 0..n {
                batch.push(Gaussian3D {
                    mean: [0.0, 0.0, 1.0],
                    scale: [0.1, 0.1, 0.1],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    opacity: 0.5,
                    sh: [0.0; SH_COEFFS_PER_GAUSSIAN],
                });
            }
            let cam = Camera::identity_at_origin(512, 512);
            let mut out = ProjectedBatch::with_capacity(batch.capacity);
            project_batch(&batch, &cam, &mut out);
            // First `n` should be valid; remaining `capacity - n` must
            // all be `valid=0`.
            for i in 0..n {
                assert_eq!(out.valid[i], 1, "n={n}: slot {i} (< len) should be valid");
            }
            for i in n..out.capacity {
                assert_eq!(out.valid[i], 0, "n={n}: padded slot {i} (>= len) must be invalid");
            }
        }
    }
}
