//! Slice-level elementwise ops built on the polyfill SIMD types.
//!
//! Every function uses `crate::simd::F32x16` (or the appropriate type),
//! which is already dispatched: AVX-512 → AVX2 → NEON → scalar.
//! These ops inherit that dispatch — no platform-specific code here.
//!
//! Re-exported flat through `ndarray::simd::add_f32`, etc.

use crate::simd::{F32x16, F64x8};

// ═══════════════════════════════════════════════════════════════════
// f32 binary ops (out-of-place)
// ═══════════════════════════════════════════════════════════════════

/// Elementwise add: `out[i] = a[i] + b[i]`.
pub fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x + y, |x, y| x + y)
}

/// Elementwise subtract: `out[i] = a[i] - b[i]`.
pub fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x - y, |x, y| x - y)
}

/// Elementwise multiply: `out[i] = a[i] * b[i]`.
pub fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x * y, |x, y| x * y)
}

/// Elementwise divide: `out[i] = a[i] / b[i]`.
pub fn div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x / y, |x, y| x / y)
}

// ═══════════════════════════════════════════════════════════════════
// f32 inplace ops
// ═══════════════════════════════════════════════════════════════════

/// Inplace add: `dst[i] += src[i]`.
pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d + s, |d, s| *d += s)
}

/// Inplace subtract: `dst[i] -= src[i]`.
pub fn sub_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d - s, |d, s| *d -= s)
}

/// Inplace multiply: `dst[i] *= src[i]`.
pub fn mul_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d * s, |d, s| *d *= s)
}

/// Inplace divide: `dst[i] /= src[i]`.
pub fn div_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d / s, |d, s| *d /= s)
}

// ═══════════════════════════════════════════════════════════════════
// f32 scalar ops
// ═══════════════════════════════════════════════════════════════════

/// Scalar multiply: `out[i] = a[i] * scalar`.
pub fn scale_f32(a: &[f32], scalar: f32) -> Vec<f32> {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) * s).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] * scalar;
        i += 1;
    }
    out
}

/// Scalar add: `out[i] = a[i] + scalar`.
pub fn add_scalar_f32(a: &[f32], scalar: f32) -> Vec<f32> {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) + s).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] + scalar;
        i += 1;
    }
    out
}

/// Inplace scalar multiply: `a[i] *= scalar`.
pub fn scale_f32_inplace(a: &mut [f32], scalar: f32) {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) * s).copy_to_slice(&mut a[i..]);
        i += 16;
    }
    while i < n {
        a[i] *= scalar;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// f64 binary ops
// ═══════════════════════════════════════════════════════════════════

/// Elementwise add f64: `out[i] = a[i] + b[i]`.
pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    binary_f64(a, b, |x, y| x + y, |x, y| x + y)
}

/// Elementwise multiply f64: `out[i] = a[i] * b[i]`.
pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    binary_f64(a, b, |x, y| x * y, |x, y| x * y)
}

/// Inplace add f64: `dst[i] += src[i]`.
pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
    inplace_f64(dst, src, |d, s| d + s, |d, s| *d += s)
}

// ═══════════════════════════════════════════════════════════════════
// Internal dispatch helpers
// ═══════════════════════════════════════════════════════════════════

#[inline]
fn binary_f32(
    a: &[f32], b: &[f32], simd_op: impl Fn(F32x16, F32x16) -> F32x16, scalar_op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        simd_op(F32x16::from_slice(&a[i..]), F32x16::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = scalar_op(a[i], b[i]);
        i += 1;
    }
    out
}

#[inline]
fn inplace_f32(
    dst: &mut [f32], src: &[f32], simd_op: impl Fn(F32x16, F32x16) -> F32x16, scalar_op: impl Fn(&mut f32, f32),
) {
    let n = dst.len().min(src.len());
    let mut i = 0;
    while i + 16 <= n {
        simd_op(F32x16::from_slice(&dst[i..]), F32x16::from_slice(&src[i..])).copy_to_slice(&mut dst[i..]);
        i += 16;
    }
    while i < n {
        scalar_op(&mut dst[i], src[i]);
        i += 1;
    }
}

#[inline]
fn binary_f64(
    a: &[f64], b: &[f64], simd_op: impl Fn(F64x8, F64x8) -> F64x8, scalar_op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f64; n];
    let mut i = 0;
    while i + 8 <= n {
        simd_op(F64x8::from_slice(&a[i..]), F64x8::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i < n {
        out[i] = scalar_op(a[i], b[i]);
        i += 1;
    }
    out
}

#[inline]
fn inplace_f64(
    dst: &mut [f64], src: &[f64], simd_op: impl Fn(F64x8, F64x8) -> F64x8, scalar_op: impl Fn(&mut f64, f64),
) {
    let n = dst.len().min(src.len());
    let mut i = 0;
    while i + 8 <= n {
        simd_op(F64x8::from_slice(&dst[i..]), F64x8::from_slice(&src[i..])).copy_to_slice(&mut dst[i..]);
        i += 8;
    }
    while i < n {
        scalar_op(&mut dst[i], src[i]);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// WS-4: elementwise pow / min / max slice ops
// ═══════════════════════════════════════════════════════════════════

/// Elementwise power: `out[i] = a[i].powf(b[i])`.
///
/// No SIMD `powf` is available in the polyfill layer; each element is
/// computed with scalar `f32::powf`.  The output length is
/// `a.len().min(b.len())`.
///
/// # Example
/// ```
/// use ndarray::simd::pow_f32;
/// let out = pow_f32(&[2.0, 3.0, 4.0], &[3.0, 2.0, 0.5]);
/// assert!((out[0] - 8.0).abs() < 1e-5);
/// assert!((out[1] - 9.0).abs() < 1e-5);
/// assert!((out[2] - 2.0).abs() < 1e-5);
/// ```
// TODO(ws4): vectorize via exp(b * ln(a)) once hpc::vml exposes simd exp/ln
pub fn pow_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = a[i].powf(b[i]);
    }
    out
}

/// Elementwise minimum: `out[i] = a[i].min(b[i])`.
///
/// NaN semantics: follows `f32::min` — a NaN argument does not "win" the
/// minimum; the non-NaN value is returned.  If both are NaN the result is
/// NaN.  The output length is `a.len().min(b.len())`.
///
/// SIMD chunks of 16 f32 values are processed with `F32x16::simd_min`;
/// the remaining tail elements are processed with scalar `f32::min`.
///
/// # Example
/// ```
/// use ndarray::simd::min_f32;
/// let out = min_f32(&[1.0, 5.0, 3.0], &[4.0, 2.0, 3.0]);
/// assert!((out[0] - 1.0).abs() < 1e-6);
/// assert!((out[1] - 2.0).abs() < 1e-6);
/// assert!((out[2] - 3.0).abs() < 1e-6);
/// ```
pub fn min_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x.simd_min(y), |x, y| x.min(y))
}

/// Elementwise maximum: `out[i] = a[i].max(b[i])`.
///
/// NaN semantics: follows `f32::max` — a NaN argument does not "win" the
/// maximum; the non-NaN value is returned.  If both are NaN the result is
/// NaN.  The output length is `a.len().min(b.len())`.
///
/// SIMD chunks of 16 f32 values are processed with `F32x16::simd_max`;
/// the remaining tail elements are processed with scalar `f32::max`.
///
/// # Example
/// ```
/// use ndarray::simd::max_f32;
/// let out = max_f32(&[1.0, 5.0, 3.0], &[4.0, 2.0, 3.0]);
/// assert!((out[0] - 4.0).abs() < 1e-6);
/// assert!((out[1] - 5.0).abs() < 1e-6);
/// assert!((out[2] - 3.0).abs() < 1e-6);
/// ```
pub fn max_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x.simd_max(y), |x, y| x.max(y))
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32_aligned() {
        let a = vec![1.0f32; 32];
        let b = vec![2.0f32; 32];
        let c = add_f32(&a, &b);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn add_f32_misaligned_tail() {
        let a = vec![1.0f32; 33];
        let b = vec![2.0f32; 33];
        let c = add_f32(&a, &b);
        assert_eq!(c.len(), 33);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn mul_f32_inplace_works() {
        let mut dst = vec![2.0f32; 17];
        let src = vec![3.0f32; 17];
        mul_f32_inplace(&mut dst, &src);
        assert!(dst.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn scale_f32_works() {
        let a = vec![4.0f32; 35];
        let b = scale_f32(&a, 0.5);
        assert!(b.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn scale_f32_inplace_works() {
        let mut a = vec![10.0f32; 19];
        scale_f32_inplace(&mut a, 0.1);
        assert!(a.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn add_scalar_f32_works() {
        let a = vec![1.0f32; 20];
        let b = add_scalar_f32(&a, 99.0);
        assert!(b.iter().all(|&v| (v - 100.0).abs() < 1e-6));
    }

    #[test]
    fn sub_f32_works() {
        let c = sub_f32(&[5.0; 3], &[2.0; 3]);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn div_f32_works() {
        let c = div_f32(&[6.0; 4], &[3.0; 4]);
        assert!(c.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn add_f64_works() {
        let c = add_f64(&[1.0f64; 17], &[2.0f64; 17]);
        assert_eq!(c.len(), 17);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-12));
    }

    #[test]
    fn empty_slices() {
        assert!(add_f32(&[], &[]).is_empty());
        assert!(mul_f32(&[], &[]).is_empty());
        assert!(scale_f32(&[], 2.0).is_empty());
    }

    #[test]
    fn mismatched_lengths_takes_min() {
        let c = add_f32(&[1.0; 10], &[2.0; 5]);
        assert_eq!(c.len(), 5);
    }

    // ── pow_f32 ───────────────────────────────────────────────────────────

    #[test]
    fn pow_f32_empty() {
        assert!(pow_f32(&[], &[]).is_empty());
    }

    #[test]
    fn pow_f32_short() {
        // fewer than PREFERRED_F32_LANES (16) elements — scalar tail only
        let a = [2.0f32, 3.0, 4.0];
        let b = [3.0f32, 2.0, 0.5];
        let out = pow_f32(&a, &b);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 8.0).abs() < 1e-5, "2^3");
        assert!((out[1] - 9.0).abs() < 1e-5, "3^2");
        assert!((out[2] - 2.0).abs() < 1e-5, "4^0.5");
    }

    #[test]
    fn pow_f32_aligned() {
        // exactly 32 elements (2 × 16-lane chunks)
        let a = vec![2.0f32; 32];
        let b = vec![4.0f32; 32];
        let out = pow_f32(&a, &b);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5), "2^4=16");
    }

    #[test]
    fn pow_f32_tail() {
        // 33 elements — 2 full SIMD chunks + 1 tail element
        let a = vec![3.0f32; 33];
        let b = vec![2.0f32; 33];
        let out = pow_f32(&a, &b);
        assert_eq!(out.len(), 33);
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-5), "3^2=9");
    }

    // ── min_f32 ───────────────────────────────────────────────────────────

    #[test]
    fn min_f32_empty() {
        assert!(min_f32(&[], &[]).is_empty());
    }

    #[test]
    fn min_f32_short() {
        let out = min_f32(&[1.0f32, 5.0, 3.0], &[4.0, 2.0, 6.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn min_f32_aligned() {
        // 32 elements
        let a = vec![3.0f32; 32];
        let b = vec![5.0f32; 32];
        let out = min_f32(&a, &b);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn min_f32_tail() {
        // 35 elements — 2 full SIMD chunks + 3 tail
        let mut a = vec![1.0f32; 35];
        let mut b = vec![2.0f32; 35];
        a[34] = 9.0;
        b[34] = 7.0;
        let out = min_f32(&a, &b);
        assert_eq!(out.len(), 35);
        assert!((out[34] - 7.0).abs() < 1e-6, "tail element");
        assert!(out[..34].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn min_f32_nan() {
        // NaN should not "win" — non-NaN value is returned
        let a = [f32::NAN, 2.0f32];
        let b = [1.0f32, f32::NAN];
        let out = min_f32(&a, &b);
        // f32::min(NaN, 1.0) == 1.0
        assert!((out[0] - 1.0).abs() < 1e-6, "NaN on left");
        // f32::min(2.0, NaN) == 2.0
        assert!((out[1] - 2.0).abs() < 1e-6, "NaN on right");
    }

    // ── max_f32 ───────────────────────────────────────────────────────────

    #[test]
    fn max_f32_empty() {
        assert!(max_f32(&[], &[]).is_empty());
    }

    #[test]
    fn max_f32_short() {
        let out = max_f32(&[1.0f32, 5.0, 3.0], &[4.0, 2.0, 6.0]);
        assert_eq!(out, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn max_f32_aligned() {
        // 32 elements
        let a = vec![7.0f32; 32];
        let b = vec![5.0f32; 32];
        let out = max_f32(&a, &b);
        assert!(out.iter().all(|&v| (v - 7.0).abs() < 1e-6));
    }

    #[test]
    fn max_f32_tail() {
        // 33 elements — 2 full SIMD chunks + 1 tail
        let mut a = vec![3.0f32; 33];
        let mut b = vec![4.0f32; 33];
        a[32] = 10.0;
        b[32] = 1.0;
        let out = max_f32(&a, &b);
        assert_eq!(out.len(), 33);
        assert!((out[32] - 10.0).abs() < 1e-6, "tail element");
        assert!(out[..32].iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    #[test]
    fn max_f32_nan() {
        // NaN should not "win" — non-NaN value is returned
        let a = [f32::NAN, 2.0f32];
        let b = [1.0f32, f32::NAN];
        let out = max_f32(&a, &b);
        // f32::max(NaN, 1.0) == 1.0
        assert!((out[0] - 1.0).abs() < 1e-6, "NaN on left");
        // f32::max(2.0, NaN) == 2.0
        assert!((out[1] - 2.0).abs() < 1e-6, "NaN on right");
    }
}
