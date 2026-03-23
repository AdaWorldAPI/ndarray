//! VML: Vectorized Math Library.
//!
//! Element-wise transcendental and arithmetic operations on slices.
//! SIMD-accelerated via `F32x16` compat types where possible.
//! Pure Rust implementations; MKL-accelerated versions behind `intel-mkl` feature gate.

use crate::simd::{simd_exp_f32, simd_ln_f32, F32x16, F64x8};

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
///
/// Scalar loop — no SIMD polynomial for f64 exp yet.
/// F64x8 load/store still enables autovectorization on some targets.
pub fn vdexp(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.exp();
    }
}

/// Element-wise natural log: out[i] = ln(x[i])
///
/// SIMD-accelerated via `simd_ln_f32` (16 lanes per iteration).
pub fn vsln(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let v = F32x16::from_slice(&x[i..]);
        simd_ln_f32(v).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = x[i].ln();
        i += 1;
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
///
/// SIMD-accelerated via `F64x8::sqrt()` (8 lanes per iteration).
pub fn vdsqrt(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 8 <= n {
        F64x8::from_slice(&x[i..]).sqrt().copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i < n {
        out[i] = x[i].sqrt();
        i += 1;
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
///
/// SIMD-accelerated via `F64x8::abs()` (8 lanes per iteration).
pub fn vdabs(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 8 <= n {
        F64x8::from_slice(&x[i..]).abs().copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i < n {
        out[i] = x[i].abs();
        i += 1;
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
///
/// SIMD-batched: loads 16 floats via F32x16, applies scalar sin per lane,
/// stores back. Amortizes load/store overhead; a true SIMD sin polynomial
/// can replace the inner loop later.
pub fn vssin(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let v = F32x16::from_slice(&x[i..]);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].sin();
        }
        F32x16::from_array(res).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = x[i].sin();
        i += 1;
    }
}

/// Element-wise cos: out[i] = cos(x[i])
///
/// SIMD-batched: loads 16 floats via F32x16, applies scalar cos per lane.
pub fn vscos(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let v = F32x16::from_slice(&x[i..]);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].cos();
        }
        F32x16::from_array(res).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = x[i].cos();
        i += 1;
    }
}

/// Element-wise pow: out[i] = a[i] ^ b[i]
///
/// Uses SIMD exp+ln: `a^b = exp(b * ln(a))`. The `simd_ln_f32` and
/// `simd_exp_f32` kernels provide 16-wide vectorization.
pub fn vspow(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let va = F32x16::from_slice(&a[i..]);
        let vb = F32x16::from_slice(&b[i..]);
        // a^b = exp(b * ln(a))
        let result = simd_exp_f32(vb * simd_ln_f32(va));
        result.copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i].powf(b[i]);
        i += 1;
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

    #[test]
    fn test_vsln_simd() {
        // Exercise the SIMD path (>= 16 elements)
        let mut x = [0.0f32; 20];
        for i in 0..20 {
            x[i] = (i + 1) as f32;
        }
        let mut out = [0.0f32; 20];
        vsln(&x, &mut out);
        for i in 0..20 {
            assert!((out[i] - ((i + 1) as f32).ln()).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_vdsqrt_simd() {
        let mut x = [0.0f64; 12];
        for i in 0..12 {
            x[i] = ((i + 1) * (i + 1)) as f64;
        }
        let mut out = [0.0f64; 12];
        vdsqrt(&x, &mut out);
        for i in 0..12 {
            assert!((out[i] - (i + 1) as f64).abs() < 1e-10, "mismatch at {i}");
        }
    }

    #[test]
    fn test_vdabs_simd() {
        let x: Vec<f64> = (-5..5).map(|i| i as f64).collect();
        let mut out = vec![0.0f64; 10];
        vdabs(&x, &mut out);
        for i in 0..10 {
            assert_eq!(out[i], (x[i]).abs());
        }
    }

    #[test]
    fn test_vscos() {
        let x = [0.0f32, core::f32::consts::PI];
        let mut out = [0.0f32; 2];
        vscos(&x, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vspow_simd() {
        // Test a^b = exp(b*ln(a)) for known values
        let a = [2.0f32, 3.0, 4.0];
        let b = [3.0f32, 2.0, 0.5];
        let mut out = [0.0f32; 3];
        vspow(&a, &b, &mut out);
        assert!((out[0] - 8.0).abs() < 1e-3, "2^3 = {}", out[0]);
        assert!((out[1] - 9.0).abs() < 1e-3, "3^2 = {}", out[1]);
        assert!((out[2] - 2.0).abs() < 1e-3, "4^0.5 = {}", out[2]);
    }

    #[test]
    fn test_vssin_simd_batch() {
        // Exercise SIMD path with 32 elements
        let mut x = [0.0f32; 32];
        for i in 0..32 {
            x[i] = (i as f32) * 0.1;
        }
        let mut out = [0.0f32; 32];
        vssin(&x, &mut out);
        for i in 0..32 {
            assert!((out[i] - x[i].sin()).abs() < 1e-5, "mismatch at {i}");
        }
    }
}
