//! Quaternion algebra — `Quat` carrier and operations.
//!
//! # Overview
//!
//! [`Quat`] is a unit-quaternion carrier stored as `(w, x, y, z)` with the
//! scalar part first.  All operations assume the quaternion is unit-length
//! where it matters (rotation, `slerp`, `from_mat`).  Call [`Quat::normalize`]
//! after accumulating small floating-point errors.
//!
//! # Feature gate
//!
//! This module is compiled only when `--features linalg` is passed.  It is
//! declared as `pub mod quat` in `crate::hpc::linalg::mod`.
//!
//! # No SIMD / no unsafe
//!
//! Per the PR-X10 hard rules, this file contains **no** SIMD intrinsics and
//! **no** `unsafe` blocks.  The batched helper [`quat_mul_x16`] loops over
//! plain scalar code; SIMD acceleration belongs in a future sprint.

use super::matrix::Mat3;

// ════════════════════════════════════════════════════════════════════════════
// Quat — the carrier
// ════════════════════════════════════════════════════════════════════════════

/// Unit quaternion: `w + xi + yj + zk`.
///
/// The scalar part `w` comes first so the in-memory layout matches the
/// `[w, x, y, z]` convention used by most real-time-graphics APIs.
///
/// `#[repr(C, align(16))]` places all four `f32` fields on a single 16-byte
/// SIMD word, ready for future SSE/NEON four-wide loads.
///
/// # Invariant
///
/// Methods that produce or consume *rotations* (`from_axis_angle`, `to_mat`,
/// `rotate_vec`, `slerp`) assume `self` is unit-length.  Use [`Quat::normalize`]
/// to restore the invariant after accumulated floating-point drift.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Quat;
/// let q = Quat::I;
/// assert_eq!(q.w, 1.0);
/// assert_eq!(q.norm_sq(), 1.0);
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Quat {
    /// Scalar part.
    pub w: f32,
    /// i-component.
    pub x: f32,
    /// j-component.
    pub y: f32,
    /// k-component.
    pub z: f32,
}

impl PartialEq for Quat {
    fn eq(&self, other: &Self) -> bool {
        self.w == other.w && self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Quat {
    // ── Constants ────────────────────────────────────────────────────────

    /// The identity quaternion: no rotation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// let q = Quat::I;
    /// let v = [1.0_f32, 2.0, 3.0];
    /// let rv = q.rotate_vec(v);
    /// assert!((rv[0] - v[0]).abs() < 1e-6);
    /// assert!((rv[1] - v[1]).abs() < 1e-6);
    /// assert!((rv[2] - v[2]).abs() < 1e-6);
    /// ```
    pub const I: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    // ── Constructors ─────────────────────────────────────────────────────

    /// Build a unit quaternion from an axis and an angle (radians).
    ///
    /// `axis` need not be pre-normalised; this function normalises it
    /// internally.  If `axis` is the zero vector the identity quaternion
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_2;
    /// let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    /// let v = q.rotate_vec([1.0, 0.0, 0.0]);
    /// assert!((v[0] - 0.0).abs() < 1e-6, "x: {}", v[0]);
    /// assert!((v[1] - 1.0).abs() < 1e-6, "y: {}", v[1]);
    /// ```
    #[inline]
    pub fn from_axis_angle(axis: [f32; 3], radians: f32) -> Self {
        let len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if len_sq < f32::EPSILON {
            return Self::I;
        }
        let inv_len = len_sq.sqrt().recip();
        let ax = axis[0] * inv_len;
        let ay = axis[1] * inv_len;
        let az = axis[2] * inv_len;
        let half = radians * 0.5;
        let s = half.sin();
        Self {
            w: half.cos(),
            x: ax * s,
            y: ay * s,
            z: az * s,
        }
    }

    /// Extract a unit quaternion from a rotation matrix using Shepperd's
    /// method with sign tracking.
    ///
    /// The method selects the numerically largest component first to avoid
    /// division by a near-zero quantity, then recovers the remaining three
    /// components.  This is stable for any proper rotation matrix (det = +1).
    ///
    /// **Precondition**: `r` must be a proper rotation matrix.  Passing a
    /// general matrix produces an unspecified result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 0.7);
    /// let m = q.to_mat();
    /// let q2 = Quat::from_mat(&m);
    /// // q and q2 represent the same rotation (may differ by global sign).
    /// let v = [1.0_f32, 0.0, 0.0];
    /// let v1 = q.rotate_vec(v);
    /// let v2 = q2.rotate_vec(v);
    /// assert!((v1[0] - v2[0]).abs() < 1e-5);
    /// assert!((v1[1] - v2[1]).abs() < 1e-5);
    /// assert!((v1[2] - v2[2]).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn from_mat(r: &Mat3) -> Self {
        // Elements of the 3×3 rotation matrix.
        let m00 = r.get(0, 0);
        let m01 = r.get(0, 1);
        let m02 = r.get(0, 2);
        let m10 = r.get(1, 0);
        let m11 = r.get(1, 1);
        let m12 = r.get(1, 2);
        let m20 = r.get(2, 0);
        let m21 = r.get(2, 1);
        let m22 = r.get(2, 2);

        // trace = 4w² - 1  ⟹  4w² = trace + 1
        let trace = m00 + m11 + m22;

        // Compute all four squared magnitudes.
        let w2 = (1.0 + trace) * 0.25;
        let x2 = (1.0 + m00 - m11 - m22) * 0.25;
        let y2 = (1.0 - m00 + m11 - m22) * 0.25;
        let z2 = (1.0 - m00 - m11 + m22) * 0.25;

        // Clamp against numerical noise before taking sqrt.
        let w2 = w2.max(0.0);
        let x2 = x2.max(0.0);
        let y2 = y2.max(0.0);
        let z2 = z2.max(0.0);

        // Shepperd: pick the component with largest magnitude as the pivot.
        if w2 >= x2 && w2 >= y2 && w2 >= z2 {
            let w = w2.sqrt();
            let w4 = 4.0 * w;
            Self {
                w,
                x: (m21 - m12) / w4,
                y: (m02 - m20) / w4,
                z: (m10 - m01) / w4,
            }
        } else if x2 >= y2 && x2 >= z2 {
            let x = x2.sqrt();
            let x4 = 4.0 * x;
            Self {
                w: (m21 - m12) / x4,
                x,
                y: (m01 + m10) / x4,
                z: (m02 + m20) / x4,
            }
        } else if y2 >= z2 {
            let y = y2.sqrt();
            let y4 = 4.0 * y;
            Self {
                w: (m02 - m20) / y4,
                x: (m01 + m10) / y4,
                y,
                z: (m12 + m21) / y4,
            }
        } else {
            let z = z2.sqrt();
            let z4 = 4.0 * z;
            Self {
                w: (m10 - m01) / z4,
                x: (m02 + m20) / z4,
                y: (m12 + m21) / z4,
                z,
            }
        }
    }

    // ── Conversions ──────────────────────────────────────────────────────

    /// Convert this unit quaternion to a 3×3 rotation matrix.
    ///
    /// The output is column-major in memory convention (following the
    /// `Mat3 = MatN<3>` row-major storage: element `(i, j)` is row i, col j).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// let m = Quat::I.to_mat();
    /// // Identity quaternion → identity matrix.
    /// assert!((m.get(0, 0) - 1.0).abs() < 1e-7);
    /// assert!((m.get(1, 1) - 1.0).abs() < 1e-7);
    /// assert!((m.get(2, 2) - 1.0).abs() < 1e-7);
    /// assert!(m.get(0, 1).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn to_mat(&self) -> Mat3 {
        let Self { w, x, y, z } = *self;
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        Mat3::from_array([
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ])
    }

    // ── Algebraic operations ─────────────────────────────────────────────

    /// Conjugate: `(w, -x, -y, -z)`.
    ///
    /// For a unit quaternion this equals the inverse.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_4;
    /// let q = Quat::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_4);
    /// let qc = q.conjugate();
    /// assert_eq!(qc.w, q.w);
    /// assert_eq!(qc.x, -q.x);
    /// ```
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Squared norm: `w² + x² + y² + z²`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_3;
    /// let q = Quat::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_3);
    /// assert!((q.norm_sq() - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn norm_sq(&self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Dot product with `other`: `self.w·other.w + self.x·other.x + …`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// let q = Quat::I;
    /// assert!((q.dot(&q) - 1.0).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Inverse: `conjugate / norm²`.
    ///
    /// Returns the identity if the quaternion is degenerate (norm² < ε).
    ///
    /// For a unit quaternion this is the same as [`conjugate`](Self::conjugate).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_4;
    /// let q = Quat::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_4);
    /// let qi = q.inverse();
    /// let prod = q.mul(&qi);
    /// assert!((prod.w - 1.0).abs() < 1e-6);
    /// assert!(prod.x.abs() < 1e-6);
    /// assert!(prod.y.abs() < 1e-6);
    /// assert!(prod.z.abs() < 1e-6);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let ns = self.norm_sq();
        if ns < f32::EPSILON {
            return Self::I;
        }
        let inv = ns.recip();
        Self {
            w: self.w * inv,
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
        }
    }

    /// Normalise to unit length.
    ///
    /// Returns the identity if the quaternion is degenerate (norm < ε).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// let q = Quat { w: 2.0, x: 0.0, y: 0.0, z: 0.0 };
    /// let n = q.normalize();
    /// assert!((n.w - 1.0).abs() < 1e-7);
    /// assert!((n.norm_sq() - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn normalize(&self) -> Self {
        let ns = self.norm_sq();
        if ns < f32::EPSILON {
            return Self::I;
        }
        let inv = ns.sqrt().recip();
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    /// Hamilton product `self ⊗ other`.
    ///
    /// **Does not** preserve unit-length automatically; call
    /// [`normalize`](Self::normalize) periodically to prevent drift.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_2;
    /// // Two 90° rotations about z compose to 180° about z.
    /// let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    /// let q2 = q.mul(&q);
    /// let v = q2.rotate_vec([1.0, 0.0, 0.0]);
    /// assert!((v[0] - (-1.0)).abs() < 1e-5, "x={}", v[0]);
    /// assert!(v[1].abs() < 1e-5, "y={}", v[1]);
    /// ```
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let (w1, x1, y1, z1) = (self.w, self.x, self.y, self.z);
        let (w2, x2, y2, z2) = (other.w, other.x, other.y, other.z);
        Self {
            w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        }
    }

    /// Rotate a 3-vector by this unit quaternion: `q ⊗ [0,v] ⊗ q*`.
    ///
    /// Uses the optimised 15-multiply form (Rodrigues / Fuster 2009).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_2;
    /// // 90° around z: [1,0,0] → [0,1,0].
    /// let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    /// let v = q.rotate_vec([1.0, 0.0, 0.0]);
    /// assert!((v[0] - 0.0).abs() < 1e-6);
    /// assert!((v[1] - 1.0).abs() < 1e-6);
    /// assert!(v[2].abs() < 1e-6);
    /// ```
    #[inline]
    pub fn rotate_vec(&self, v: [f32; 3]) -> [f32; 3] {
        // Optimised form: t = 2 * cross(q.xyz, v); result = v + q.w*t + cross(q.xyz, t)
        let (qx, qy, qz) = (self.x, self.y, self.z);
        let (vx, vy, vz) = (v[0], v[1], v[2]);

        // t = 2 * (q.xyz × v)
        let tx = 2.0 * (qy * vz - qz * vy);
        let ty = 2.0 * (qz * vx - qx * vz);
        let tz = 2.0 * (qx * vy - qy * vx);

        // result = v + w*t + q.xyz × t
        [
            vx + self.w * tx + qy * tz - qz * ty,
            vy + self.w * ty + qz * tx - qx * tz,
            vz + self.w * tz + qx * ty - qy * tx,
        ]
    }

    /// Spherical linear interpolation from `self` to `other` by parameter `t ∈ [0, 1]`.
    ///
    /// Chooses the shortest arc (negates `other` if `dot < 0`).  Falls back
    /// to normalised linear interpolation (nlerp) when the two quaternions
    /// are very close (|θ| < ε) to avoid dividing by a near-zero sine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Quat;
    /// use core::f32::consts::FRAC_PI_2;
    /// let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    /// // t=0 → identity, t=1 → q.
    /// let s0 = Quat::I.slerp(&q, 0.0);
    /// let s1 = Quat::I.slerp(&q, 1.0);
    /// assert!((s0.w - Quat::I.w).abs() < 1e-6);
    /// assert!((s1.w - q.w).abs() < 1e-6);
    /// assert!((s1.z - q.z).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let mut dot = self.dot(other);

        // Choose the shorter arc.
        let other_sign;
        if dot < 0.0 {
            dot = -dot;
            other_sign = -1.0f32;
        } else {
            other_sign = 1.0f32;
        }

        // Clamp to valid acos domain.
        let dot = dot.min(1.0);

        const SLERP_THRESHOLD: f32 = 0.9995;
        if dot > SLERP_THRESHOLD {
            // Quaternions nearly parallel — use nlerp to avoid div-by-zero.
            let scale0 = 1.0 - t;
            let scale1 = t * other_sign;
            return Self {
                w: self.w * scale0 + other.w * scale1,
                x: self.x * scale0 + other.x * scale1,
                y: self.y * scale0 + other.y * scale1,
                z: self.z * scale0 + other.z * scale1,
            }
            .normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let scale0 = ((1.0 - t) * theta).sin() / sin_theta;
        let scale1 = (t * theta).sin() / sin_theta * other_sign;

        Self {
            w: self.w * scale0 + other.w * scale1,
            x: self.x * scale0 + other.x * scale1,
            y: self.y * scale0 + other.y * scale1,
            z: self.z * scale0 + other.z * scale1,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Batched helpers
// ════════════════════════════════════════════════════════════════════════════

/// Batched 16-wide quaternion multiply: `out[i] = a[i] ⊗ b[i]`.
///
/// Designed for the splat3d backward pass which updates 16 quaternion
/// parameters per gradient step.  The loop body is identical to
/// [`Quat::mul`]; the fixed-width array signature lets the compiler
/// unroll and auto-vectorise on x86-64 AVX and aarch64 NEON.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::{Quat, quat_mul_x16};
/// let a = [Quat::I; 16];
/// let b = [Quat::I; 16];
/// let mut out = [Quat::I; 16];
/// quat_mul_x16(&a, &b, &mut out);
/// for q in &out {
///     assert!((q.w - 1.0).abs() < 1e-7);
/// }
/// ```
#[inline]
pub fn quat_mul_x16(a: &[Quat; 16], b: &[Quat; 16], out: &mut [Quat; 16]) {
    for i in 0..16 {
        out[i] = a[i].mul(&b[i]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Helper: approximately equal vectors.
    fn vec_approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    // Helper: approximately equal quaternions (up to global sign).
    fn quat_same_rotation(a: Quat, b: Quat, eps: f32) -> bool {
        let dot = a.dot(&b).abs(); // abs: same rotation may differ by sign
        (dot - 1.0).abs() < eps
    }

    // ── Identity round-trip ──────────────────────────────────────────────

    #[test]
    fn identity_rotate_vec_is_identity() {
        let v = [1.0_f32, 2.0, 3.0];
        let rv = Quat::I.rotate_vec(v);
        assert!(vec_approx_eq(rv, v, 1e-6), "I·v should equal v, got {rv:?}");
    }

    #[test]
    fn identity_norm_sq_is_one() {
        assert!((Quat::I.norm_sq() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn identity_to_mat_is_identity_mat() {
        let m = Quat::I.to_mat();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((m.get(i, j) - expected).abs() < 1e-6, "I.to_mat()[{i}][{j}] = {} ≠ {expected}", m.get(i, j));
            }
        }
    }

    // ── axis_angle(z, π/2) rotates [1,0,0] → ~[0,1,0] ─────────────────

    #[test]
    fn axis_angle_z_pi_half_rotates_x_to_y() {
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        let v = q.rotate_vec([1.0, 0.0, 0.0]);
        assert!(vec_approx_eq(v, [0.0, 1.0, 0.0], 1e-5), "got {v:?}");
    }

    // ── slerp boundary conditions ────────────────────────────────────────

    #[test]
    fn slerp_at_t0_is_identity() {
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        let s = Quat::I.slerp(&q, 0.0);
        assert!(quat_same_rotation(s, Quat::I, 1e-5), "slerp(I, q, 0) should be I");
    }

    #[test]
    fn slerp_at_t1_is_other() {
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        let s = Quat::I.slerp(&q, 1.0);
        assert!(quat_same_rotation(s, q, 1e-5), "slerp(I, q, 1) should be q");
    }

    #[test]
    fn slerp_midpoint_half_angle() {
        // Interpolating between I and a 90° rotation at t=0.5 should give ~45°.
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        let s = Quat::I.slerp(&q, 0.5);
        let q_half = Quat::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_4);
        assert!(quat_same_rotation(s, q_half, 1e-5));
    }

    // ── conjugate·self = norm_sq ─────────────────────────────────────────

    #[test]
    fn conjugate_mul_self_equals_norm_sq() {
        let q = Quat::from_axis_angle([1.0, 1.0, 0.0], 1.23);
        // q* ⊗ q should be purely scalar, equal to norm².
        let prod = q.conjugate().mul(&q);
        let ns = q.norm_sq();
        assert!((prod.w - ns).abs() < 1e-5, "w mismatch: {} vs {}", prod.w, ns);
        assert!(prod.x.abs() < 1e-5, "x should be ~0, got {}", prod.x);
        assert!(prod.y.abs() < 1e-5, "y should be ~0, got {}", prod.y);
        assert!(prod.z.abs() < 1e-5, "z should be ~0, got {}", prod.z);
    }

    // ── mul associates with axis_angle composition ────────────────────────

    #[test]
    fn mul_associates_with_axis_angle_composition() {
        // Rotating by α then β about the same axis equals rotating by α+β.
        let alpha = PI / 3.0;
        let beta = PI / 5.0;
        let axis = [0.0, 1.0, 0.0_f32];
        let qa = Quat::from_axis_angle(axis, alpha);
        let qb = Quat::from_axis_angle(axis, beta);
        let qab = Quat::from_axis_angle(axis, alpha + beta);

        let q_mul = qa.mul(&qb).normalize();
        assert!(quat_same_rotation(q_mul, qab, 1e-5), "qa⊗qb ≠ q(α+β): {q_mul:?} vs {qab:?}");
    }

    // ── from_mat / to_mat round-trip ─────────────────────────────────────

    #[test]
    fn to_mat_from_mat_round_trip() {
        let q = Quat::from_axis_angle([1.0, 2.0, 3.0], 0.9).normalize();
        let m = q.to_mat();
        let q2 = Quat::from_mat(&m);
        assert!(quat_same_rotation(q, q2, 1e-4), "round-trip failed: {q:?} vs {q2:?}");
    }

    // ── normalize ────────────────────────────────────────────────────────

    #[test]
    fn normalize_makes_unit() {
        let q = Quat {
            w: 3.0,
            x: 1.0,
            y: 2.0,
            z: 2.0,
        };
        let n = q.normalize();
        assert!((n.norm_sq() - 1.0).abs() < 1e-6);
    }

    // ── inverse ──────────────────────────────────────────────────────────

    #[test]
    fn inverse_gives_identity_product() {
        let q = Quat::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_4);
        let qi = q.inverse();
        let prod = q.mul(&qi);
        assert!((prod.w - 1.0).abs() < 1e-5);
        assert!(prod.x.abs() < 1e-5);
        assert!(prod.y.abs() < 1e-5);
        assert!(prod.z.abs() < 1e-5);
    }

    // ── quat_mul_x16 ─────────────────────────────────────────────────────

    #[test]
    fn quat_mul_x16_identity() {
        let a = [Quat::I; 16];
        let b = [Quat::I; 16];
        let mut out = [Quat {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 16];
        quat_mul_x16(&a, &b, &mut out);
        for (i, q) in out.iter().enumerate() {
            assert!((q.w - 1.0).abs() < 1e-7, "out[{i}].w = {}", q.w);
        }
    }

    #[test]
    fn quat_mul_x16_matches_scalar() {
        let mut a = [Quat::I; 16];
        let mut b = [Quat::I; 16];
        for i in 0..16 {
            a[i] = Quat::from_axis_angle([1.0, 0.0, 0.0], (i as f32) * 0.1);
            b[i] = Quat::from_axis_angle([0.0, 1.0, 0.0], (i as f32) * 0.05);
        }
        let mut out_batch = [Quat::I; 16];
        quat_mul_x16(&a, &b, &mut out_batch);
        for i in 0..16 {
            let expected = a[i].mul(&b[i]);
            assert!(quat_same_rotation(out_batch[i], expected, 1e-6), "mismatch at i={i}");
        }
    }
}
