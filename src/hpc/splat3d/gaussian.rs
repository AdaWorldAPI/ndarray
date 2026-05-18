//! Structure-of-Arrays batch storage for 3D Gaussian Splatting.
//!
//! # Layout
//!
//! Each field is a separate `Vec<f32>` (SoA) padded to `PREFERRED_F32_LANES`
//! so SIMD passes never hit a scalar tail. The batch holds:
//!
//! ```text
//!   12 scalar channels × capacity f32s  (mean xyz, scale xyz, quat wxyz, opacity)
//!   48 SH coefficients × capacity f32s  (degree-3 RGB: 3 × 16)
//! ```
//!
//! # SIMD covariance
//!
//! `covariance_x16(start, out)` batches 16 Σ = R · diag(s²) · Rᵀ computations
//! via `crate::simd::F32x16`, mirroring the scalar formula in
//! `Spd3::from_scale_quat` lane-by-lane. See that function for the
//! derivation of the rotation matrix and the Σ upper-triangle.

use crate::simd::{F32x16, PREFERRED_F32_LANES};
use super::spd3::Spd3;

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/// SH degree (3 = 16 coefficients per RGB channel = 48 total per gaussian).
pub const SH_DEGREE: usize = 3;
/// 16 SH basis functions per channel for degree-3.
pub const SH_COEFFS_PER_CHANNEL: usize = (SH_DEGREE + 1) * (SH_DEGREE + 1);
/// 48 floats per gaussian total (3 channels × 16 coeffs).
pub const SH_COEFFS_PER_GAUSSIAN: usize = SH_COEFFS_PER_CHANNEL * 3;

// ════════════════════════════════════════════════════════════════════════════
// Padding helper (self-contained — pad_to_lanes in renderer.rs is not pub)
// ════════════════════════════════════════════════════════════════════════════

/// Round `n` up to the nearest multiple of `lanes`.
#[inline]
const fn pad_to_lanes(n: usize, lanes: usize) -> usize {
    (n + lanes - 1) / lanes * lanes
}

// ════════════════════════════════════════════════════════════════════════════
// Gaussian3D — convenience AoS input shape
// ════════════════════════════════════════════════════════════════════════════

/// Single 3D gaussian — the test/demo input shape. Not used in the hot path;
/// the rasterizer reads from `GaussianBatch` directly.
pub struct Gaussian3D {
    /// World-space mean position [x, y, z].
    pub mean: [f32; 3],
    /// Anisotropic scale standard deviations [sx, sy, sz].
    pub scale: [f32; 3],
    /// Unit quaternion [w, x, y, z].
    pub quat: [f32; 4],
    /// Opacity in [0, 1].
    pub opacity: f32,
    /// Degree-3 SH coefficients (48 floats). Layout:
    /// `sh[ch * 16 + basis_idx]` for ch in 0..3.
    pub sh: [f32; SH_COEFFS_PER_GAUSSIAN],
}

impl Gaussian3D {
    /// Identity-rotation gaussian at the origin, isotropic unit scale,
    /// fully opaque, SH coefficients all zero. Useful as a test stub
    /// before `push`.
    pub fn unit() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            quat: [1.0, 0.0, 0.0, 0.0],
            opacity: 1.0,
            sh: [0.0; SH_COEFFS_PER_GAUSSIAN],
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GaussianBatch — SoA storage
// ════════════════════════════════════════════════════════════════════════════

/// Structure-of-Arrays batch of 3D gaussians, padded to `PREFERRED_F32_LANES`
/// so SIMD passes never encounter a scalar tail.
///
/// Mirror of `hpc::renderer::RenderFrame` but for gaussian splat fields.
pub struct GaussianBatch {
    /// Active gaussian count (≤ capacity).
    pub len: usize,
    /// Padded capacity (multiple of PREFERRED_F32_LANES).
    pub capacity: usize,
    /// Position (length = capacity each).
    pub mean_x: Vec<f32>,
    pub mean_y: Vec<f32>,
    pub mean_z: Vec<f32>,
    /// Anisotropic standard-deviation scale (length = capacity each).
    pub scale_x: Vec<f32>,
    pub scale_y: Vec<f32>,
    pub scale_z: Vec<f32>,
    /// Rotation quaternion (w, x, y, z), unit norm (length = capacity each).
    pub quat_w: Vec<f32>,
    pub quat_x: Vec<f32>,
    pub quat_y: Vec<f32>,
    pub quat_z: Vec<f32>,
    /// Opacity in [0, 1] (length = capacity).
    pub opacity: Vec<f32>,
    /// SH coefficients (length = SH_COEFFS_PER_GAUSSIAN * capacity).
    /// Layout: gaussian-major, channel-major within:
    ///   sh[i * 48 + ch * 16 + basis_idx]
    pub sh: Vec<f32>,
}

impl GaussianBatch {
    /// Allocate empty batch with capacity for `n` gaussians (rounded up
    /// to PREFERRED_F32_LANES). All buffers zero-initialized.
    pub fn with_capacity(n: usize) -> Self {
        let capacity = pad_to_lanes(n.max(1), PREFERRED_F32_LANES);
        Self {
            len: 0,
            capacity,
            mean_x:  vec![0.0; capacity],
            mean_y:  vec![0.0; capacity],
            mean_z:  vec![0.0; capacity],
            scale_x: vec![0.0; capacity],
            scale_y: vec![0.0; capacity],
            scale_z: vec![0.0; capacity],
            quat_w:  vec![0.0; capacity],
            quat_x:  vec![0.0; capacity],
            quat_y:  vec![0.0; capacity],
            quat_z:  vec![0.0; capacity],
            opacity: vec![0.0; capacity],
            sh:      vec![0.0; SH_COEFFS_PER_GAUSSIAN * capacity],
        }
    }

    /// Reset to empty (`len = 0`) without deallocating. Trailing slots
    /// already zero from `with_capacity`; new pushes overwrite.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Push one gaussian into the next slot. Panics if `len == capacity`.
    /// Callers in tight loops should use `with_capacity` to pre-size.
    pub fn push(&mut self, g: Gaussian3D) {
        assert!(
            self.len < self.capacity,
            "GaussianBatch::push: len == capacity ({})",
            self.capacity
        );
        let i = self.len;
        self.mean_x[i]  = g.mean[0];
        self.mean_y[i]  = g.mean[1];
        self.mean_z[i]  = g.mean[2];
        self.scale_x[i] = g.scale[0];
        self.scale_y[i] = g.scale[1];
        self.scale_z[i] = g.scale[2];
        self.quat_w[i]  = g.quat[0];
        self.quat_x[i]  = g.quat[1];
        self.quat_y[i]  = g.quat[2];
        self.quat_z[i]  = g.quat[3];
        self.opacity[i] = g.opacity;
        let sh_base = i * SH_COEFFS_PER_GAUSSIAN;
        self.sh[sh_base..sh_base + SH_COEFFS_PER_GAUSSIAN]
            .copy_from_slice(&g.sh);
        self.len += 1;
    }

    /// Reconstruct the i-th gaussian's covariance from scale + quat via
    /// `Spd3::from_scale_quat`. Panics if `i >= len`.
    pub fn covariance(&self, i: usize) -> Spd3 {
        assert!(i < self.len, "covariance: index {i} >= len {}", self.len);
        let scale = [self.scale_x[i], self.scale_y[i], self.scale_z[i]];
        let quat  = [self.quat_w[i],  self.quat_x[i],  self.quat_y[i],  self.quat_z[i]];
        Spd3::from_scale_quat(scale, quat)
    }

    /// Batched covariance reconstruction: 16 gaussians at indices
    /// `[start, start + 16)`. Writes into `out`. Panics if
    /// `start + 16 > self.capacity`.
    ///
    /// Uses `crate::simd::F32x16` to SIMD-batch the quat→rotation
    /// cross products and the Σ = R · diag(s²) · Rᵀ product.
    /// Output is AoS `[Spd3; 16]`.
    ///
    /// # Precondition on padded lanes
    ///
    /// The bound is on `capacity`, NOT `len`, so the SIMD pad allows
    /// any 16-aligned block read at the cost of correctness for slots
    /// `>= len`. Padded slots have `scale = [0, 0, 0]` and
    /// `quat = [0, 0, 0, 0]` (degenerate zero-norm quaternion), which
    /// the closed-form Σ = R · diag(s²) · Rᵀ collapses to the zero
    /// matrix — **non-SPD** and unsafe to feed into a downstream
    /// inverse / sandwich. Callers walking the batch in 16-wide
    /// chunks (e.g. PR 3's `project_batch`) must mask the trailing
    /// `(capacity - len)` lanes of the final chunk before consuming
    /// `out`. The `valid` mask carried by `ProjectedBatch` (PR 3) is
    /// the canonical place for that bookkeeping.
    pub fn covariance_x16(&self, start: usize, out: &mut [Spd3; 16]) {
        assert!(
            start + 16 <= self.capacity,
            "covariance_x16: start ({start}) + 16 > capacity ({})",
            self.capacity
        );

        // ── 1. Load 7 SoA channels into F32x16 lanes ────────────────────
        let qw = F32x16::from_slice(&self.quat_w[start..start + 16]);
        let qx = F32x16::from_slice(&self.quat_x[start..start + 16]);
        let qy = F32x16::from_slice(&self.quat_y[start..start + 16]);
        let qz = F32x16::from_slice(&self.quat_z[start..start + 16]);
        let sx = F32x16::from_slice(&self.scale_x[start..start + 16]);
        let sy = F32x16::from_slice(&self.scale_y[start..start + 16]);
        let sz = F32x16::from_slice(&self.scale_z[start..start + 16]);

        // ── 2. Intermediate quaternion products (mirror from_scale_quat) ─
        let two = F32x16::splat(2.0);
        let one = F32x16::splat(1.0);

        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        // Rotation matrix (row-major):
        //   R = [[r00, r01, r02],
        //        [r10, r11, r12],
        //        [r20, r21, r22]]
        let r00 = one - two * (yy + zz);
        let r01 = two * (xy - wz);
        let r02 = two * (xz + wy);
        let r10 = two * (xy + wz);
        let r11 = one - two * (xx + zz);
        let r12 = two * (yz - wx);
        let r20 = two * (xz - wy);
        let r21 = two * (yz + wx);
        let r22 = one - two * (xx + yy);

        // ── 3. s² = scale squared ────────────────────────────────────────
        let s0 = sx * sx;
        let s1 = sy * sy;
        let s2 = sz * sz;

        // ── 4. M = R · diag(s²): scale column k by sₖ² ─────────────────
        let m00 = r00 * s0; let m01 = r01 * s1; let m02 = r02 * s2;
        let m10 = r10 * s0; let m11 = r11 * s1; let m12 = r12 * s2;
        let m20 = r20 * s0; let m21 = r21 * s1; let m22 = r22 * s2;

        // ── 5. Σ = M · Rᵀ — upper triangle ──────────────────────────────
        let a11 = m00 * r00 + m01 * r01 + m02 * r02;
        let a12 = m00 * r10 + m01 * r11 + m02 * r12;
        let a13 = m00 * r20 + m01 * r21 + m02 * r22;
        let a22 = m10 * r10 + m11 * r11 + m12 * r12;
        let a23 = m10 * r20 + m11 * r21 + m12 * r22;
        let a33 = m20 * r20 + m21 * r21 + m22 * r22;

        // ── 6. Scatter SoA → AoS [Spd3; 16] ────────────────────────────
        let mut buf_a11 = [0.0f32; 16];
        let mut buf_a12 = [0.0f32; 16];
        let mut buf_a13 = [0.0f32; 16];
        let mut buf_a22 = [0.0f32; 16];
        let mut buf_a23 = [0.0f32; 16];
        let mut buf_a33 = [0.0f32; 16];
        a11.copy_to_slice(&mut buf_a11);
        a12.copy_to_slice(&mut buf_a12);
        a13.copy_to_slice(&mut buf_a13);
        a22.copy_to_slice(&mut buf_a22);
        a23.copy_to_slice(&mut buf_a23);
        a33.copy_to_slice(&mut buf_a33);
        for k in 0..16 {
            out[k] = Spd3::new(
                buf_a11[k], buf_a12[k], buf_a13[k],
                buf_a22[k], buf_a23[k], buf_a33[k],
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn approx_spd3(a: Spd3, b: Spd3, tol: f32) -> bool {
        approx(a.a11, b.a11, tol)
            && approx(a.a12, b.a12, tol)
            && approx(a.a13, b.a13, tol)
            && approx(a.a22, b.a22, tol)
            && approx(a.a23, b.a23, tol)
            && approx(a.a33, b.a33, tol)
    }

    // Deterministic xorshift32.
    fn rng_u32(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    fn rng_f32(state: &mut u32) -> f32 {
        rng_u32(state) as f32 / u32::MAX as f32
    }

    /// Build a normalized quaternion from 4 random floats.
    fn rng_quat(state: &mut u32) -> [f32; 4] {
        let mut q = [
            -1.0 + 2.0 * rng_f32(state),
            -1.0 + 2.0 * rng_f32(state),
            -1.0 + 2.0 * rng_f32(state),
            -1.0 + 2.0 * rng_f32(state),
        ];
        let n = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
        for v in &mut q { *v /= n; }
        q
    }

    fn rng_scale(state: &mut u32) -> [f32; 3] {
        [
            0.2 + 1.8 * rng_f32(state),
            0.2 + 1.8 * rng_f32(state),
            0.2 + 1.8 * rng_f32(state),
        ]
    }

    // ── Test 1 ──────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_batch_with_capacity_pads_lanes() {
        for n in [1usize, 7, 15, 16, 17, 100] {
            let b = GaussianBatch::with_capacity(n);
            let expected = pad_to_lanes(n.max(1), PREFERRED_F32_LANES);
            assert_eq!(b.capacity, expected, "n={n}: capacity mismatch");
            assert_eq!(b.len, 0);
            assert_eq!(b.mean_x.len(),  expected, "n={n}: mean_x len");
            assert_eq!(b.mean_y.len(),  expected, "n={n}: mean_y len");
            assert_eq!(b.mean_z.len(),  expected, "n={n}: mean_z len");
            assert_eq!(b.scale_x.len(), expected, "n={n}: scale_x len");
            assert_eq!(b.scale_y.len(), expected, "n={n}: scale_y len");
            assert_eq!(b.scale_z.len(), expected, "n={n}: scale_z len");
            assert_eq!(b.quat_w.len(),  expected, "n={n}: quat_w len");
            assert_eq!(b.quat_x.len(),  expected, "n={n}: quat_x len");
            assert_eq!(b.quat_y.len(),  expected, "n={n}: quat_y len");
            assert_eq!(b.quat_z.len(),  expected, "n={n}: quat_z len");
            assert_eq!(b.opacity.len(), expected, "n={n}: opacity len");
            assert_eq!(b.sh.len(), SH_COEFFS_PER_GAUSSIAN * expected, "n={n}: sh len");
        }
    }

    // ── Test 2 ──────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_batch_push_preserves_alignment() {
        let n = 4;
        let mut b = GaussianBatch::with_capacity(n);
        let cap = b.capacity;
        for i in 0..n {
            let mut g = Gaussian3D::unit();
            g.mean[0] = i as f32 + 1.0;
            g.opacity = (i as f32 + 1.0) * 0.1;
            b.push(g);
        }
        assert_eq!(b.len, n);
        // All pushed slots populated.
        for i in 0..n {
            assert!(b.mean_x[i] != 0.0, "slot {i} mean_x should be non-zero");
            assert!(b.opacity[i] != 0.0, "slot {i} opacity should be non-zero");
        }
        // Slots after len still zero (padding).
        for i in n..cap {
            assert_eq!(b.mean_x[i], 0.0, "pad slot {i} mean_x not zero");
            assert_eq!(b.opacity[i], 0.0, "pad slot {i} opacity not zero");
        }
    }

    // ── Test 3 ──────────────────────────────────────────────────────────────

    #[test]
    #[should_panic]
    fn gaussian_batch_push_panics_at_capacity() {
        let mut b = GaussianBatch::with_capacity(1);
        // Fill to capacity.
        for _ in 0..b.capacity {
            b.push(Gaussian3D::unit());
        }
        // This push must panic.
        b.push(Gaussian3D::unit());
    }

    // ── Test 4 ──────────────────────────────────────────────────────────────

    #[test]
    fn covariance_from_unit_quat_is_diag_of_scale_squared() {
        let mut b = GaussianBatch::with_capacity(1);
        let mut g = Gaussian3D::unit();
        g.scale = [2.0, 1.5, 0.8];
        g.quat  = [1.0, 0.0, 0.0, 0.0]; // identity rotation
        b.push(g);
        let cov = b.covariance(0);
        // Σ = diag(s²) = diag(4.0, 2.25, 0.64)
        assert!(approx(cov.a11, 4.0,  1e-6), "a11={}", cov.a11);
        assert!(approx(cov.a22, 2.25, 1e-6), "a22={}", cov.a22);
        assert!(approx(cov.a33, 0.64, 1e-6), "a33={}", cov.a33);
        assert!(approx(cov.a12, 0.0,  1e-6), "a12={}", cov.a12);
        assert!(approx(cov.a13, 0.0,  1e-6), "a13={}", cov.a13);
        assert!(approx(cov.a23, 0.0,  1e-6), "a23={}", cov.a23);
    }

    // ── Test 5 ──────────────────────────────────────────────────────────────

    #[test]
    fn covariance_with_90deg_y_rotation_matches_spd3() {
        // 90° about Y: quat = (cos 45°, 0, sin 45°, 0)
        let h = (0.5f32).sqrt();
        let scale = [2.0f32, 1.5, 0.8];
        let quat  = [h, 0.0, h, 0.0];
        let mut b = GaussianBatch::with_capacity(1);
        let mut g = Gaussian3D::unit();
        g.scale = scale;
        g.quat  = quat;
        b.push(g);
        let got      = b.covariance(0);
        let expected = Spd3::from_scale_quat(scale, quat);
        assert!(
            approx_spd3(got, expected, 1e-5),
            "got={got:?} expected={expected:?}"
        );
    }

    // ── Test 6 ──────────────────────────────────────────────────────────────

    #[test]
    fn covariance_x16_matches_scalar_loop() {
        let mut state = 0xC0FFEE_u32;
        let mut b = GaussianBatch::with_capacity(16);
        for _ in 0..16 {
            let mut g = Gaussian3D::unit();
            g.scale = rng_scale(&mut state);
            g.quat  = rng_quat(&mut state);
            b.push(g);
        }
        let mut simd_out = [Spd3::ZERO; 16];
        b.covariance_x16(0, &mut simd_out);
        for i in 0..16 {
            let scalar = b.covariance(i);
            assert!(
                approx_spd3(simd_out[i], scalar, 1e-4),
                "lane {i}: simd={:?} scalar={:?}",
                simd_out[i],
                scalar,
            );
        }
    }

    // ── Test 7 ──────────────────────────────────────────────────────────────

    #[test]
    fn clear_resets_len_preserves_capacity() {
        let mut b = GaussianBatch::with_capacity(8);
        let cap = b.capacity;
        for _ in 0..4 {
            b.push(Gaussian3D::unit());
        }
        assert_eq!(b.len, 4);
        b.clear();
        assert_eq!(b.len, 0);
        assert_eq!(b.capacity, cap);
    }

    // ── Test 8 ──────────────────────────────────────────────────────────────

    #[test]
    fn gaussian3d_unit_constructor() {
        let g = Gaussian3D::unit();
        assert_eq!(g.mean,    [0.0, 0.0, 0.0]);
        assert_eq!(g.scale,   [1.0, 1.0, 1.0]);
        assert_eq!(g.quat,    [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.opacity, 1.0);
        assert_eq!(g.sh,      [0.0; SH_COEFFS_PER_GAUSSIAN]);
    }

    // ── Test 9 — covariance_x16 with start > 0 (PP-13 PR2 P1 promoted) ─────
    //
    // The existing covariance_x16_matches_scalar_loop test fires with
    // start=0. An off-by-one in the SoA slice arithmetic
    // (`self.quat_w[start..start+16]`) would still pass start=0 since
    // any constant offset of 0 collapses to identity. Walk a non-zero
    // start so each input index `start + k` differs from lane index `k`.
    #[test]
    fn covariance_x16_with_nonzero_start_matches_scalar() {
        let mut state = 0xACE0_C0DEu32;
        let mut batch = GaussianBatch::with_capacity(48);
        for _ in 0..32 {
            batch.push(sample_gaussian3d(&mut state));
        }
        let start = 16; // walk past the first SIMD block
        let mut out_simd = [Spd3::ZERO; 16];
        batch.covariance_x16(start, &mut out_simd);
        for k in 0..16 {
            let scalar = batch.covariance(start + k);
            // 1e-4 absolute matches PR 1's sandwich_x16 tolerance; the
            // SoA-transpose-then-recombine pipeline accumulates the
            // same evaluation-order noise.
            assert!(
                approx_spd3(out_simd[k], scalar, 1e-4),
                "lane k={k} (index {}): simd={:?}, scalar={:?}",
                start + k, out_simd[k], scalar,
            );
        }
    }

    // ── Test 10 — SH round-trip through SoA (PP-13 PR2 P1 promoted) ─────────
    //
    // Verifies the bridge between `push` (writes the 48-float SH block
    // at `self.sh[i*48..]`) and `sh::sh_eval_deg3` (reads at the same
    // offset). If `SH_COEFFS_PER_GAUSSIAN` were misdefined, or `push`'s
    // SH copy used the wrong base, the resulting RGB would silently
    // drift from the analytical expectation. Uses Test 8 from sh.rs's
    // suite as the analytical reference (basis k=0 → SH_C0 + 0.5).
    #[test]
    fn push_then_sh_eval_round_trips_through_soa() {
        use super::super::sh::sh_eval_deg3;
        let mut g = Gaussian3D::unit();
        // Non-zero SH coefficient for channel R, basis k=0 (Y_00 — the
        // direction-invariant DC term). Channels G/B all-zero → 0.5.
        g.sh[0] = 1.0;
        // Also set a coefficient at the LAST slot to verify the full
        // 48-float span survives the SoA copy.
        g.sh[47] = 0.5;
        let mut batch = GaussianBatch::with_capacity(16);
        // Push 5 unit gaussians first, then ours at index 5, so the SoA
        // offset arithmetic isn't trivial.
        for _ in 0..5 {
            batch.push(Gaussian3D::unit());
        }
        batch.push(g);
        // Pull the SH slice for gaussian 5 directly out of the batch and
        // run sh_eval at d=(0,0,1) (where Y_00 dominates).
        let base = 5 * SH_COEFFS_PER_GAUSSIAN;
        let sh_slice = &batch.sh[base..base + SH_COEFFS_PER_GAUSSIAN];
        // Sanity-check the SoA contents: indices 0 and 47 survived; the
        // 46 in between are zero (this is also a fence-post check on
        // the push SH-copy bounds).
        assert!(
            (sh_slice[0] - 1.0).abs() < 1e-7,
            "SoA sh[0] for gaussian 5 = {}, expected 1.0", sh_slice[0]
        );
        assert!(
            (sh_slice[47] - 0.5).abs() < 1e-7,
            "SoA sh[47] for gaussian 5 = {}, expected 0.5", sh_slice[47]
        );
        for k in 1..47 {
            assert!(
                sh_slice[k].abs() < 1e-7,
                "SoA sh[{k}] for gaussian 5 = {}, expected 0", sh_slice[k]
            );
        }
        // And the round-trip evaluation must reflect that DC coefficient.
        let rgb = sh_eval_deg3(sh_slice, [0.0, 0.0, 1.0]);
        // sh.rs SH_C0 ≈ 0.282; with the +0.5 Inria offset → 0.782.
        assert!(
            (rgb[0] - 0.7820948).abs() < 1e-5,
            "R channel via SoA: got {}, want ≈ {} (SH_C0 + 0.5)", rgb[0], 0.7820948
        );
        // G channel = 0.5 (all-zero coeffs).
        // B channel: sh[47] = 0.5 is the *last* B coefficient (basis k=15
        // = Y_3,3 = -SH_C3[6] · x(x²-3y²)). At d=(0,0,1) x=0 so this
        // basis vanishes → B = 0.5.
        assert!(
            (rgb[1] - 0.5).abs() < 1e-6,
            "G channel: got {}, want 0.5", rgb[1]
        );
        assert!(
            (rgb[2] - 0.5).abs() < 1e-6,
            "B channel (sh[47] basis vanishes at d=(0,0,1)): got {}, want 0.5", rgb[2]
        );
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Build a random Gaussian3D — reuses Worker C's existing `rng_*`
    /// helpers so the test module stays single-sourced.
    fn sample_gaussian3d(state: &mut u32) -> Gaussian3D {
        Gaussian3D {
            mean: [rng_f32(state), rng_f32(state), rng_f32(state)],
            scale: rng_scale(state),
            quat: rng_quat(state),
            opacity: rng_f32(state),
            sh: [0.0; SH_COEFFS_PER_GAUSSIAN],
        }
    }
}
