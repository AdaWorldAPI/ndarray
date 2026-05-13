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
}
