//! VML: Vectorized Math Library.
//!
//! Element-wise transcendental and arithmetic operations on slices.
//! SIMD-accelerated via `F32x16` compat types where possible.
//! Pure Rust implementations; MKL-accelerated versions behind `intel-mkl` feature gate.

use crate::backend::simd_compat::{simd_exp_f32, F32x16};

/// Element-wise exp: out[i] = exp(x[i])
///
/// Processes 16 elements at a time via SIMD polynomial approximation.
pub fn vsexp(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let v = F32x16::from_slice(&x[i..]);
        simd_exp_f32(v).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    // Scalar tail
    while i < n {
        out[i] = x[i].exp();
        i += 1;
    }
}

/// Element-wise exp (f64).
pub fn vdexp(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.exp();
    }
}

/// Element-wise natural log: out[i] = ln(x[i])
pub fn vsln(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.ln();
    }
}

/// Element-wise natural log (f64).
pub fn vdln(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.ln();
    }
}

/// Element-wise sqrt: out[i] = sqrt(x[i])
///
/// SIMD-accelerated via F32x16::sqrt().
pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        F32x16::from_slice(&x[i..]).sqrt().copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = x[i].sqrt();
        i += 1;
    }
}

/// Element-wise sqrt (f64).
pub fn vdsqrt(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.sqrt();
    }
}

/// Element-wise abs: out[i] = |x[i]|
///
/// SIMD-accelerated via F32x16::abs().
pub fn vsabs(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        F32x16::from_slice(&x[i..]).abs().copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = x[i].abs();
        i += 1;
    }
}

/// Element-wise abs (f64).
pub fn vdabs(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.abs();
    }
}

/// Element-wise add: out[i] = a[i] + b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) + F32x16::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Element-wise mul: out[i] = a[i] * b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) * F32x16::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

/// Element-wise div: out[i] = a[i] / b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsdiv(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) / F32x16::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] / b[i];
        i += 1;
    }
}

/// Element-wise sin: out[i] = sin(x[i])
pub fn vssin(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.sin();
    }
}

/// Element-wise cos: out[i] = cos(x[i])
pub fn vscos(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.cos();
    }
}

/// Element-wise pow: out[i] = a[i] ^ b[i]
pub fn vspow(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &av), &bv) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = av.powf(bv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsexp() {
        let x = [0.0f32, 1.0, 2.0];
        let mut out = [0.0f32; 3];
        vsexp(&x, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_vsln() {
        let x = [1.0f32, std::f32::consts::E];
        let mut out = [0.0f32; 2];
        vsln(&x, &mut out);
        assert!((out[0]).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vssqrt() {
        let x = [4.0f32, 9.0, 16.0];
        let mut out = [0.0f32; 3];
        vssqrt(&x, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_vsadd() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        vsadd(&a, &b, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vssin() {
        let x = [0.0f32, core::f32::consts::FRAC_PI_2];
        let mut out = [0.0f32; 2];
        vssin(&x, &mut out);
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }
}
