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
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        simd_exp_f32(F32x16::from_slice(x_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.exp();
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
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        simd_ln_f32(F32x16::from_slice(x_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
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
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        F32x16::from_slice(x_chunk).sqrt().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sqrt();
    }
}

/// Element-wise sqrt (f64).
///
/// SIMD-accelerated via `F64x8::sqrt()` (8 lanes per iteration).
pub fn vdsqrt(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        F64x8::from_slice(x_chunk).sqrt().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 8;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sqrt();
    }
}

/// Element-wise abs: out[i] = |x[i]|
///
/// SIMD-accelerated via F32x16::abs().
pub fn vsabs(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        F32x16::from_slice(x_chunk).abs().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.abs();
    }
}

/// Element-wise abs (f64).
///
/// SIMD-accelerated via `F64x8::abs()` (8 lanes per iteration).
pub fn vdabs(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        F64x8::from_slice(x_chunk).abs().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 8;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.abs();
    }
}

/// Element-wise add: out[i] = a[i] + b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(out.chunks_exact_mut(16)) {
        (F32x16::from_slice(a_chunk) + F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..].iter().zip(b[tail_start..].iter()).zip(out[tail_start..].iter_mut()) {
        *o = av + bv;
    }
}

/// Element-wise mul: out[i] = a[i] * b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(out.chunks_exact_mut(16)) {
        (F32x16::from_slice(a_chunk) * F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..].iter().zip(b[tail_start..].iter()).zip(out[tail_start..].iter_mut()) {
        *o = av * bv;
    }
}

/// Element-wise div: out[i] = a[i] / b[i]
///
/// SIMD-accelerated via F32x16 operator overload.
pub fn vsdiv(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(out.chunks_exact_mut(16)) {
        (F32x16::from_slice(a_chunk) / F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..].iter().zip(b[tail_start..].iter()).zip(out[tail_start..].iter_mut()) {
        *o = av / bv;
    }
}

/// Element-wise sin: out[i] = sin(x[i])
///
/// SIMD-batched: loads 16 floats via F32x16, applies scalar sin per lane,
/// stores back. Amortizes load/store overhead; a true SIMD sin polynomial
/// can replace the inner loop later.
pub fn vssin(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let v = F32x16::from_slice(x_chunk);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].sin();
        }
        F32x16::from_array(res).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sin();
    }
}

/// Element-wise cos: out[i] = cos(x[i])
///
/// SIMD-batched: loads 16 floats via F32x16, applies scalar cos per lane.
pub fn vscos(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let v = F32x16::from_slice(x_chunk);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].cos();
        }
        F32x16::from_array(res).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.cos();
    }
}

/// Element-wise pow: out[i] = a[i] ^ b[i]
///
/// Uses SIMD exp+ln: `a^b = exp(b * ln(a))`. The `simd_ln_f32` and
/// `simd_exp_f32` kernels provide 16-wide vectorization.
///
/// **Domain restriction**: the SIMD path requires `a[i] > 0`. Negative bases
/// produce NaN (since `ln(negative)` is undefined), unlike scalar `powf` which
/// handles some negative-base cases. The scalar tail (len < 16) uses `powf`.
pub fn vspow(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(out.chunks_exact_mut(16)) {
        let va = F32x16::from_slice(a_chunk);
        let vb = F32x16::from_slice(b_chunk);
        // a^b = exp(b * ln(a))
        simd_exp_f32(vb * simd_ln_f32(va)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..].iter().zip(b[tail_start..].iter()).zip(out[tail_start..].iter_mut()) {
        *o = av.powf(bv);
    }
}

/// Element-wise tanh: out[i] = tanh(x[i])
///
/// Uses the identity: tanh(x) = 2·sigmoid(2x) - 1
/// which reuses our SIMD sigmoid (F32x16 polynomial).
pub fn vstanh(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    let two = F32x16::splat(2.0);
    let one = F32x16::splat(1.0);
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        // tanh(x) = 2·sigmoid(2x) - 1
        let two_x = v * two;
        // sigmoid(z) = 1/(1+exp(-z))
        let neg_two_x = F32x16::splat(0.0) - two_x;
        let exp_neg = simd_exp_f32(neg_two_x);
        let sigmoid = one / (exp_neg + one);
        let tanh_v = sigmoid.mul_add(two, F32x16::splat(-1.0));
        tanh_v.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].tanh();
    }
}

/// Element-wise floor: out[i] = floor(x[i])
///
/// Uses F32x16 hardware VRNDSCALEPS (AVX-512) or equivalent.
pub fn vsfloor(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        v.floor().copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].floor();
    }
}

/// Element-wise ceil: out[i] = ceil(x[i])
pub fn vsceil(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        // ceil = -floor(-x)
        let neg = F32x16::splat(0.0) - v;
        let floored = neg.floor();
        let ceiled = F32x16::splat(0.0) - floored;
        ceiled.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].ceil();
    }
}

/// Element-wise round (ties to even): out[i] = round(x[i])
pub fn vsround(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        v.round().copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].round_ties_even();
    }
}

/// Element-wise negate: out[i] = -x[i]
pub fn vsneg(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    let zero = F32x16::splat(0.0);
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        (zero - v).copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = -x[i];
    }
}

/// Element-wise trunc: out[i] = trunc(x[i]) (round toward zero)
pub fn vstrunc(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    // trunc = sign(x) * floor(abs(x))
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        let abs_v = v.abs();
        let floored = abs_v.floor();
        // Restore sign: if original was negative, negate the result
        // Use: trunc(x) = floor(x) if x >= 0, ceil(x) if x < 0
        // Simpler: just floor(abs(x)) * sign(x)
        // We can do sign via: x / abs(x), but that's NaN for 0.
        // Instead: if x >= 0, result = floor(abs(x)), else -floor(abs(x))
        let zero = F32x16::splat(0.0);
        let mask = v.simd_lt(zero); // true where x < 0
        let pos_result = floored;
        let neg_result = zero - floored;
        let result = mask.select(neg_result, pos_result);
        result.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].trunc();
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
    #[test]
    fn test_f64_golden_step_hydration_cost() {
        use std::f64::consts;
        // Rust 1.94: std::f64::consts::PHI and GAMMA are stable.
        // On 1.93: define manually.
        #[allow(dead_code)]
        const PHI: f64 = 1.618033988749894848204586834365638118;

        // Simulate: 4096D f64 vector → Base17-style projection → hydration back
        let d = 4096usize;
        let base_dim = 17usize;
        let golden_step = 11usize;

        // Generate "weight" data (deterministic, mimics Gaussian distribution)
        let weights: Vec<f64> = (0..d)
            .map(|i| ((i as f64 * 0.7 + 13.0).sin() * 2.5))
            .collect();

        // ── ENCODING: f64[4096] → f64[17] (golden-step projection) ──
        let encode_start = std::time::Instant::now();
        let n_octaves = (d + base_dim - 1) / base_dim;
        let mut sum = [0.0f64; 17];
        let mut count = [0u32; 17];
        for octave in 0..n_octaves {
            for bi in 0..base_dim {
                let dim = octave * base_dim + ((bi * golden_step) % base_dim);
                if dim < d {
                    sum[bi] += weights[dim];
                    count[bi] += 1;
                }
            }
        }
        let mut coefficients_f64 = [0.0f64; 17];
        for i in 0..base_dim {
            if count[i] > 0 {
                coefficients_f64[i] = sum[i] / count[i] as f64;
            }
        }
        let encode_time = encode_start.elapsed();

        // ── QUANTIZE: f64[17] → i16[17] (what Base17 stores) ──
        let fp_scale = 1000.0;
        let coefficients_i16: Vec<i16> = coefficients_f64.iter()
            .map(|&v| (v * fp_scale).round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        // ── HYDRATION: i16[17] → f64[4096] (reconstruct from golden-step basis) ──
        let hydrate_start = std::time::Instant::now();
        let mut reconstructed = vec![0.0f64; d];
        for octave in 0..n_octaves {
            for bi in 0..base_dim {
                let dim = octave * base_dim + ((bi * golden_step) % base_dim);
                if dim < d {
                    reconstructed[dim] = coefficients_i16[bi] as f64 / fp_scale;
                }
            }
        }
        let hydrate_time = hydrate_start.elapsed();

        // ── MEASURE: reconstruction quality ──
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_orig = 0.0f64;
        for i in 0..d {
            let err = weights[i] - reconstructed[i];
            sum_sq_err += err * err;
            sum_sq_orig += weights[i] * weights[i];
        }
        let relative_error = (sum_sq_err / sum_sq_orig).sqrt();

        // ── REPORT ──
        eprintln!("=== F64 Golden-Step Hydration Cost ===");
        eprintln!("  Input:       f64[{}] = {} bytes", d, d * 8);
        eprintln!("  Encoded:     i16[17] = 34 bytes");
        eprintln!("  Compression: {}×", (d * 8) / 34);
        eprintln!("  Encode time: {:?}", encode_time);
        eprintln!("  Hydrate time: {:?}", hydrate_time);
        eprintln!("  Relative error: {:.6}", relative_error);
        eprintln!("  Reconstruction quality: {:.4}%", (1.0 - relative_error) * 100.0);

        // The surface area cost IS just the encode + hydrate.
        // The middle (i16 distance, SimilarityTable lookup) is O(1) regardless.
        // For f64 tensors: the ONLY extra cost vs f32 tensors is the
        // f64→f64 accumulation in the projection (instead of f32→f64).
        // That's ~0 extra cost because the projection already uses f64 sums.

        assert!(encode_time.as_micros() < 100, "encoding should be < 100μs");
        assert!(hydrate_time.as_micros() < 100, "hydration should be < 100μs");
    }
    #[test]
    fn test_golden_step_vs_random_projection_rho() {
        // Compare: golden-step 17D projection vs random 17D projection
        // on synthetic weight-like data (approximate Gaussian).
        // Measures Spearman ρ of pairwise distances.
        
        let d = 256; // weight vector dimension (small for test speed)
        let n = 50;  // number of vectors to compare
        let base_dim = 17;
        let golden_step = 11;
        
        // Generate weight-like vectors (deterministic, Gaussian-ish)
        let vectors: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 97 + j * 31) as f64 * 0.001).sin() * 2.0)
                    .collect()
            })
            .collect();
        
        // Ground truth: pairwise L2 distances in full d-D space
        let mut gt_distances = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f64 = vectors[i].iter().zip(&vectors[j])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                gt_distances.push(dist);
            }
        }
        
        // Golden-step projection: project each vector to 17D
        let golden_projected: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let n_octaves = (d + base_dim - 1) / base_dim;
                let mut sum = vec![0.0f64; base_dim];
                let mut count = vec![0u32; base_dim];
                for octave in 0..n_octaves {
                    for bi in 0..base_dim {
                        let dim = octave * base_dim + ((bi * golden_step) % base_dim);
                        if dim < d {
                            sum[bi] += v[dim];
                            count[bi] += 1;
                        }
                    }
                }
                sum.iter().zip(&count)
                    .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
                    .collect()
            })
            .collect();
        
        // Random projection: use a deterministic "random" 17×d matrix
        let random_matrix: Vec<Vec<f64>> = (0..base_dim)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7919 + j * 104729) as f64 * 0.00001).sin())
                    .collect()
            })
            .collect();
        
        let random_projected: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                random_matrix.iter()
                    .map(|row| {
                        row.iter().zip(v).map(|(r, x)| r * x).sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        
        // Compute pairwise distances in both projected spaces
        let golden_distances: Vec<f64> = {
            let mut dists = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let dist: f64 = golden_projected[i].iter().zip(&golden_projected[j])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    dists.push(dist);
                }
            }
            dists
        };
        
        let random_distances: Vec<f64> = {
            let mut dists = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let dist: f64 = random_projected[i].iter().zip(&random_projected[j])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    dists.push(dist);
                }
            }
            dists
        };
        
        // Compute Spearman ρ: rank correlation between GT and projected distances
        fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
            let n = a.len();
            let rank_a = ranks(a);
            let rank_b = ranks(b);
            let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
            let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;
            let mut cov = 0.0f64;
            let mut var_a = 0.0f64;
            let mut var_b = 0.0f64;
            for i in 0..n {
                let da = rank_a[i] - mean_a;
                let db = rank_b[i] - mean_b;
                cov += da * db;
                var_a += da * da;
                var_b += db * db;
            }
            if var_a < 1e-10 || var_b < 1e-10 { return 0.0; }
            cov / (var_a * var_b).sqrt()
        }
        
        fn ranks(values: &[f64]) -> Vec<f64> {
            let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut result = vec![0.0; values.len()];
            for (rank, (idx, _)) in indexed.into_iter().enumerate() {
                result[idx] = rank as f64;
            }
            result
        }
        
        let rho_golden = spearman_rho(&gt_distances, &golden_distances);
        let rho_random = spearman_rho(&gt_distances, &random_distances);
        
        eprintln!("=== Projection Quality (Spearman ρ) ===");
        eprintln!("  Golden-step 17D: ρ = {:.4}", rho_golden);
        eprintln!("  Random 17D:      ρ = {:.4}", rho_random);
        eprintln!("  Δ (golden - random): {:.4}", rho_golden - rho_random);
        
        // Both should preserve SOME ranking (ρ > 0)
        assert!(rho_golden > 0.0, "golden-step ρ should be positive");
        assert!(rho_random > 0.0, "random ρ should be positive");
        // The interesting question: is golden better than random?
        // We don't assert this — we just measure it.
        // If Δ ≈ 0 → golden step is the 52°N problem.
        // If Δ > 0.05 → golden step captures real structure.
    }
    #[test]
    #[ignore] // Requires /tmp/tiny_imagenet_200.json (run with --include-ignored)
    fn test_bgz17_on_tiny_imagenet() {
        // Load real image feature vectors from tiny-imagenet (binary format).
        // Generate with: python3 script that saves [d:u32][n:u32][f32 × d × n]
        let bytes = match std::fs::read("/tmp/tiny_imagenet_200.bin") {
            Ok(b) => b,
            Err(_) => {
                eprintln!("SKIP: /tmp/tiny_imagenet_200.bin not found");
                return;
            }
        };

        let d = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

        let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
        let float_data = &bytes[8..];
        for i in 0..n {
            let v: Vec<f64> = (0..d)
                .map(|j| {
                    let off = (i * d + j) * 4;
                    f32::from_le_bytes([float_data[off], float_data[off+1], float_data[off+2], float_data[off+3]]) as f64
                })
                .collect();
            vectors.push(v);
        }
        
        let n = vectors.len();
        eprintln!("Loaded {} vectors of dim {} from tiny-imagenet", n, d);
        assert!(n >= 50, "Need at least 50 vectors");
        
        // Use first 100 for speed
        let n = n.min(100);
        let vectors = &vectors[..n];
        
        let base_dim = 17;
        let golden_step = 11;

        // Ground truth: pairwise L2 distances
        let mut gt_distances = Vec::new();
        for i in 0..n {
            for j in (i+1)..n {
                let dist: f64 = vectors[i].iter().zip(&vectors[j])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                gt_distances.push(dist);
            }
        }

        // Golden-step projection
        let golden_projected: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let n_octaves = (d + base_dim - 1) / base_dim;
                let mut sum = vec![0.0f64; base_dim];
                let mut count = vec![0u32; base_dim];
                for octave in 0..n_octaves {
                    for bi in 0..base_dim {
                        let dim = octave * base_dim + ((bi * golden_step) % base_dim);
                        if dim < d { sum[bi] += v[dim]; count[bi] += 1; }
                    }
                }
                sum.iter().zip(&count).map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
            })
            .collect();

        // Random projection
        let random_matrix: Vec<Vec<f64>> = (0..base_dim)
            .map(|i| (0..d).map(|j| ((i * 7919 + j * 104729) as f64 * 0.00001).sin()).collect())
            .collect();
        let random_projected: Vec<Vec<f64>> = vectors.iter()
            .map(|v| random_matrix.iter().map(|row| row.iter().zip(v).map(|(r, x)| r * x).sum::<f64>()).collect())
            .collect();

        // Simple mean projection (average every 17 consecutive dims)
        let mean_projected: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                (0..base_dim).map(|bi| {
                    let chunk: Vec<f64> = (bi..d).step_by(base_dim).map(|i| v[i]).collect();
                    if chunk.is_empty() { 0.0 } else { chunk.iter().sum::<f64>() / chunk.len() as f64 }
                }).collect()
            })
            .collect();

        // Compute projected distances
        fn pairwise_l2(proj: &[Vec<f64>]) -> Vec<f64> {
            let n = proj.len();
            let mut dists = Vec::new();
            for i in 0..n { for j in (i+1)..n {
                let d: f64 = proj[i].iter().zip(&proj[j]).map(|(a,b)| (a-b)*(a-b)).sum::<f64>().sqrt();
                dists.push(d);
            }}
            dists
        }

        let golden_dists = pairwise_l2(&golden_projected);
        let random_dists = pairwise_l2(&random_projected);
        let mean_dists = pairwise_l2(&mean_projected);

        // Spearman ρ
        fn spearman(a: &[f64], b: &[f64]) -> f64 {
            fn ranks(v: &[f64]) -> Vec<f64> {
                let mut idx: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
                idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut r = vec![0.0; v.len()];
                for (rank, (i, _)) in idx.into_iter().enumerate() { r[i] = rank as f64; }
                r
            }
            let ra = ranks(a); let rb = ranks(b);
            let n = a.len() as f64;
            let ma: f64 = ra.iter().sum::<f64>() / n;
            let mb: f64 = rb.iter().sum::<f64>() / n;
            let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
            for i in 0..a.len() {
                let (da, db) = (ra[i] - ma, rb[i] - mb);
                cov += da * db; va += da * da; vb += db * db;
            }
            if va < 1e-10 || vb < 1e-10 { 0.0 } else { cov / (va * vb).sqrt() }
        }

        let rho_golden = spearman(&gt_distances, &golden_dists);
        let rho_random = spearman(&gt_distances, &random_dists);
        let rho_mean = spearman(&gt_distances, &mean_dists);

        eprintln!("=== bgz17 on Tiny ImageNet (real pixel data) ===");
        eprintln!("  Golden-step 17D: ρ = {:.4}", rho_golden);
        eprintln!("  Random 17D:      ρ = {:.4}", rho_random);
        eprintln!("  Mean-stride 17D: ρ = {:.4}", rho_mean);
        eprintln!("  Δ golden-random: {:.4}", rho_golden - rho_random);
        eprintln!("  Δ golden-mean:   {:.4}", rho_golden - rho_mean);
        eprintln!();
        if (rho_golden - rho_random).abs() < 0.02 {
            eprintln!("  VERDICT: Golden-step ≈ random on pixel data (52°N problem)");
            eprintln!("  bgz17's value is NOT in the projection axes");
            eprintln!("  bgz17's value IS in the distance table + cascade infrastructure");
        } else if rho_golden > rho_random + 0.02 {
            eprintln!("  VERDICT: Golden-step > random! The Fibonacci structure captures something.");
        } else {
            eprintln!("  VERDICT: Random > golden-step. Golden-step is WORSE for this data.");
        }

        // Basic sanity: golden-step should preserve reasonable ranking
        assert!(rho_golden > 0.3, "golden ρ too low: {}", rho_golden);
        // Random projection CAN be very low on structured data — that's expected
        assert!(rho_random > -0.5, "random ρ impossibly low: {}", rho_random);
    }
    #[test]
    #[ignore]
    fn test_photography_grid_vs_golden_step() {
        // Compare: 1/3 grid line sampling vs golden-step on tiny-imagenet
        let bytes = match std::fs::read("/tmp/tiny_imagenet_200.bin") {
            Ok(b) => b,
            Err(_) => { eprintln!("SKIP: /tmp/tiny_imagenet_200.bin not found"); return; }
        };

        let d = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize; // 12288
        let n_total = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let float_data = &bytes[8..];

        // Load vectors as 64×64×3 images
        let img_w = 64usize;
        let img_h = 64usize;
        let channels = 3usize;
        let n = n_total.min(100);

        let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f64> = (0..d)
                .map(|j| {
                    let off = (i * d + j) * 4;
                    f32::from_le_bytes([float_data[off], float_data[off+1], float_data[off+2], float_data[off+3]]) as f64
                })
                .collect();
            vectors.push(v);
        }

        // Helper: extract pixel at (row, col, channel) from flat vector
        let pixel = |v: &[f64], r: usize, c: usize, ch: usize| -> f64 {
            v[r * img_w * channels + c * channels + ch]
        };

        // ── Projection 1: Golden-step 17D (baseline) ──
        let base_dim = 17;
        let golden_step = 11;
        let golden_proj: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let n_octaves = (d + base_dim - 1) / base_dim;
                let mut sum = vec![0.0f64; base_dim];
                let mut count = vec![0u32; base_dim];
                for octave in 0..n_octaves {
                    for bi in 0..base_dim {
                        let dim = octave * base_dim + ((bi * golden_step) % base_dim);
                        if dim < d { sum[bi] += v[dim]; count[bi] += 1; }
                    }
                }
                sum.iter().zip(&count).map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
            })
            .collect();

        // ── Projection 2: 1/3 + 2/3 grid lines (4 lines × 64 × 3 = 768D) ──
        let grid_lines_proj: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let mut features = Vec::with_capacity(768);
                // Horizontal lines at row 1/3 and 2/3
                let r1 = img_h / 3;      // row 21
                let r2 = 2 * img_h / 3;  // row 43
                for &r in &[r1, r2] {
                    for c in 0..img_w {
                        for ch in 0..channels {
                            features.push(pixel(v, r, c, ch));
                        }
                    }
                }
                // Vertical lines at col 1/3 and 2/3
                let c1 = img_w / 3;
                let c2 = 2 * img_w / 3;
                for &c in &[c1, c2] {
                    for r in 0..img_h {
                        for ch in 0..channels {
                            features.push(pixel(v, r, c, ch));
                        }
                    }
                }
                features
            })
            .collect();

        // ── Projection 3: 1/2 + 1/3 + 2/3 grid (6 lines × 64 × 3 = 1152D) ──
        let full_grid_proj: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let mut features = Vec::with_capacity(1152);
                // Horizontal: 1/3, 1/2, 2/3
                for &r in &[img_h / 3, img_h / 2, 2 * img_h / 3] {
                    for c in 0..img_w {
                        for ch in 0..channels { features.push(pixel(v, r, c, ch)); }
                    }
                }
                // Vertical: 1/3, 1/2, 2/3
                for &c in &[img_w / 3, img_w / 2, 2 * img_w / 3] {
                    for r in 0..img_h {
                        for ch in 0..channels { features.push(pixel(v, r, c, ch)); }
                    }
                }
                features
            })
            .collect();

        // ── Projection 4: 4 intersection points only (4 × 3 = 12D) ──
        let intersections_proj: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let mut features = Vec::with_capacity(12);
                for &r in &[img_h / 3, 2 * img_h / 3] {
                    for &c in &[img_w / 3, 2 * img_w / 3] {
                        for ch in 0..channels { features.push(pixel(v, r, c, ch)); }
                    }
                }
                features
            })
            .collect();

        // ── Ground truth pairwise distances ──
        let mut gt_dists = Vec::new();
        for i in 0..n { for j in (i+1)..n {
            let d: f64 = vectors[i].iter().zip(&vectors[j]).map(|(a,b)| (a-b)*(a-b)).sum::<f64>().sqrt();
            gt_dists.push(d);
        }}

        fn pairwise_l2(proj: &[Vec<f64>]) -> Vec<f64> {
            let n = proj.len();
            let mut d = Vec::new();
            for i in 0..n { for j in (i+1)..n {
                let dist: f64 = proj[i].iter().zip(&proj[j]).map(|(a,b)| (a-b)*(a-b)).sum::<f64>().sqrt();
                d.push(dist);
            }}
            d
        }

        fn spearman(a: &[f64], b: &[f64]) -> f64 {
            fn ranks(v: &[f64]) -> Vec<f64> {
                let mut idx: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
                idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut r = vec![0.0; v.len()];
                for (rank, (i, _)) in idx.into_iter().enumerate() { r[i] = rank as f64; }
                r
            }
            let (ra, rb) = (ranks(a), ranks(b));
            let n = a.len() as f64;
            let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
            let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
            for i in 0..a.len() {
                let (da, db) = (ra[i] - ma, rb[i] - mb);
                cov += da * db; va += da * da; vb += db * db;
            }
            if va < 1e-10 || vb < 1e-10 { 0.0 } else { cov / (va * vb).sqrt() }
        }

        let rho_golden = spearman(&gt_dists, &pairwise_l2(&golden_proj));
        let rho_grid4 = spearman(&gt_dists, &pairwise_l2(&grid_lines_proj));
        let rho_grid6 = spearman(&gt_dists, &pairwise_l2(&full_grid_proj));
        let rho_intersect = spearman(&gt_dists, &pairwise_l2(&intersections_proj));

        eprintln!("=== Photography Grid vs Golden-Step on Tiny ImageNet ===");
        eprintln!("  Golden-step 17D:        ρ = {:.4}  (34 bytes)", rho_golden);
        eprintln!("  4 intersections 12D:    ρ = {:.4}  (24 bytes)", rho_intersect);
        eprintln!("  4 grid lines 768D:      ρ = {:.4}  (1,536 bytes)", rho_grid4);
        eprintln!("  6 grid lines 1152D:     ρ = {:.4}  (2,304 bytes)", rho_grid6);
        eprintln!();
        eprintln!("  Bytes vs ρ efficiency:");
        eprintln!("  Golden:  {:.4} ρ / 34 bytes  = {:.4} ρ/byte", rho_golden, rho_golden / 34.0);
        eprintln!("  4-grid:  {:.4} ρ / 1536 bytes = {:.6} ρ/byte", rho_grid4, rho_grid4 / 1536.0);
        eprintln!("  6-grid:  {:.4} ρ / 2304 bytes = {:.6} ρ/byte", rho_grid6, rho_grid6 / 2304.0);
        eprintln!("  Points:  {:.4} ρ / 24 bytes  = {:.4} ρ/byte", rho_intersect, rho_intersect / 24.0);

        assert!(rho_golden > 0.3);
        assert!(rho_grid4 > rho_golden, "grid lines should beat golden-step on raw pixels");
    }
    #[test]
    #[ignore]
    fn test_heel_hip_archetype_bundling() {
        // Build HEEL (class) and HIP (within-class) archetypes from tiny-imagenet.
        // Test: can we identify which class an image belongs to via bundle similarity?
        let bytes = match std::fs::read("/tmp/tiny_imagenet_labeled.bin") {
            Ok(b) => b,
            Err(_) => { eprintln!("SKIP: /tmp/tiny_imagenet_labeled.bin not found"); return; }
        };

        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let d = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let n_classes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        // Read labels
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 4;
            labels.push(u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as usize);
        }

        // Read pixel vectors
        let pixel_start = 12 + n * 4;
        let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f64> = (0..d)
                .map(|j| {
                    let off = pixel_start + (i * d + j) * 4;
                    f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as f64
                })
                .collect();
            vectors.push(v);
        }

        eprintln!("Loaded {} images, {} classes, {}D", n, n_classes, d);

        // ── Extract grid-line features (1/3 + 2/3, 768D per image) ──
        let img_w = 64usize;
        let img_h = 64usize;
        let ch = 3usize;
        let pixel = |v: &[f64], r: usize, c: usize, channel: usize| -> f64 {
            v[r * img_w * ch + c * ch + channel]
        };

        let features: Vec<Vec<f64>> = vectors.iter()
            .map(|v| {
                let mut f = Vec::with_capacity(768);
                for &r in &[img_h / 3, 2 * img_h / 3] {
                    for c in 0..img_w { for channel in 0..ch { f.push(pixel(v, r, c, channel)); } }
                }
                for &c in &[img_w / 3, 2 * img_w / 3] {
                    for r in 0..img_h { for channel in 0..ch { f.push(pixel(v, r, c, channel)); } }
                }
                f
            })
            .collect();

        let feat_d = features[0].len();

        // ── Build HEEL archetypes: mean feature vector per class ──
        let mut heel_archetypes: Vec<Vec<f64>> = vec![vec![0.0; feat_d]; n_classes];
        let mut class_counts = vec![0usize; n_classes];
        for (i, &label) in labels.iter().enumerate() {
            for j in 0..feat_d {
                heel_archetypes[label][j] += features[i][j];
            }
            class_counts[label] += 1;
        }
        for c in 0..n_classes {
            if class_counts[c] > 0 {
                for j in 0..feat_d {
                    heel_archetypes[c][j] /= class_counts[c] as f64;
                }
            }
        }

        // ── Test: classify each image by nearest HEEL archetype ──
        let mut correct = 0usize;
        let mut total = 0usize;
        for (i, &true_label) in labels.iter().enumerate() {
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                if class_counts[c] == 0 { continue; }
                let dist: f64 = features[i].iter().zip(&heel_archetypes[c])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_class = c;
                }
            }
            if best_class == true_label {
                correct += 1;
            }
            total += 1;
        }
        let accuracy = correct as f64 / total as f64;

        // ── Test: classify via golden-step compressed archetypes (34 bytes each) ──
        let base_dim = 17;
        let golden_step = 11;

        let compress = |v: &[f64]| -> Vec<f64> {
            let fd = v.len();
            let n_oct = (fd + base_dim - 1) / base_dim;
            let mut sum = vec![0.0f64; base_dim];
            let mut cnt = vec![0u32; base_dim];
            for oct in 0..n_oct {
                for bi in 0..base_dim {
                    let dim = oct * base_dim + ((bi * golden_step) % base_dim);
                    if dim < fd { sum[bi] += v[dim]; cnt[bi] += 1; }
                }
            }
            sum.iter().zip(&cnt).map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
        };

        let compressed_archetypes: Vec<Vec<f64>> = heel_archetypes.iter().map(|a| compress(a)).collect();
        let compressed_features: Vec<Vec<f64>> = features.iter().map(|f| compress(f)).collect();

        let mut correct_compressed = 0usize;
        for (i, &true_label) in labels.iter().enumerate() {
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                if class_counts[c] == 0 { continue; }
                let dist: f64 = compressed_features[i].iter().zip(&compressed_archetypes[c])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist { best_dist = dist; best_class = c; }
            }
            if best_class == true_label { correct_compressed += 1; }
        }
        let accuracy_compressed = correct_compressed as f64 / total as f64;

        // ── CHAODA: find outliers (images far from ALL archetypes) ──
        let mut max_distances: Vec<(usize, f64)> = Vec::new();
        for (i, _) in labels.iter().enumerate() {
            let mut min_dist = f64::MAX;
            for c in 0..n_classes {
                if class_counts[c] == 0 { continue; }
                let dist: f64 = features[i].iter().zip(&heel_archetypes[c])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                if dist < min_dist { min_dist = dist; }
            }
            max_distances.push((i, min_dist));
        }
        max_distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let outlier_count = max_distances.iter().take(10).count();

        eprintln!("=== HEEL Archetype Classification (Tiny ImageNet) ===");
        eprintln!("  Classes: {}, Images: {}", n_classes, total);
        eprintln!("  Grid-line features (768D):");
        eprintln!("    HEEL accuracy:           {:.1}% ({}/{})", accuracy * 100.0, correct, total);
        eprintln!("  Golden-step compressed (17D = 34 bytes):");
        eprintln!("    Compressed accuracy:     {:.1}% ({}/{})", accuracy_compressed * 100.0, correct_compressed, total);
        eprintln!("  Accuracy loss from compression: {:.1}%", (accuracy - accuracy_compressed) * 100.0);
        eprintln!("  Top-10 outliers (CHAODA candidates):");
        for (idx, dist) in max_distances.iter().take(5) {
            eprintln!("    image {} (class {}): dist={:.4}", idx, labels[*idx], dist);
        }
        eprintln!("  Random baseline (1/{}): {:.1}%", n_classes, 100.0 / n_classes as f64);

        assert!(accuracy > 1.0 / n_classes as f64, "should beat random");
        assert!(accuracy_compressed > 1.0 / n_classes as f64, "compressed should beat random too");
    }
    #[test]
    #[ignore]
    fn test_hip_multi_object_detection() {
        // HIP bundles for multi-object detection:
        // Given an image, detect if it contains features of MULTIPLE classes
        // by unbinding one class archetype and checking residual against others.
        //
        // Bird/fence scenario: if unbind(image, bird) correlates with fence → both present.
        
        let bytes = match std::fs::read("/tmp/tiny_imagenet_labeled.bin") {
            Ok(b) => b,
            Err(_) => { eprintln!("SKIP: /tmp/tiny_imagenet_labeled.bin not found"); return; }
        };

        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let d = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let n_classes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 4;
            labels.push(u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as usize);
        }

        let pixel_start = 12 + n * 4;
        let img_w = 64usize;
        let img_h = 64usize;
        let ch = 3usize;

        // Extract grid-line features (768D)
        let features: Vec<Vec<f64>> = (0..n).map(|i| {
            let v_start = pixel_start + i * d * 4;
            let pixel = |r: usize, c: usize, channel: usize| -> f64 {
                let off = v_start + (r * img_w * ch + c * ch + channel) * 4;
                f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as f64
            };
            let mut f = Vec::with_capacity(768);
            for &r in &[img_h / 3, 2 * img_h / 3] {
                for c in 0..img_w { for channel in 0..ch { f.push(pixel(r, c, channel)); } }
            }
            for &c in &[img_w / 3, 2 * img_w / 3] {
                for r in 0..img_h { for channel in 0..ch { f.push(pixel(r, c, channel)); } }
            }
            f
        }).collect();

        let feat_d = features[0].len();

        // ── Build HEEL archetypes per class ──
        let mut archetypes: Vec<Vec<f64>> = vec![vec![0.0; feat_d]; n_classes];
        let mut counts = vec![0usize; n_classes];
        for (i, &label) in labels.iter().enumerate() {
            for j in 0..feat_d { archetypes[label][j] += features[i][j]; }
            counts[label] += 1;
        }
        for c in 0..n_classes {
            if counts[c] > 0 {
                for j in 0..feat_d { archetypes[c][j] /= counts[c] as f64; }
            }
        }

        // ── Cosine similarity helper ──
        let cosine = |a: &[f64], b: &[f64]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if mag_a < 1e-10 || mag_b < 1e-10 { 0.0 } else { dot / (mag_a * mag_b) }
        };

        // ── HIP: within-class variance (how spread is each class?) ──
        let mut hip_variance = vec![0.0f64; n_classes];
        for (i, &label) in labels.iter().enumerate() {
            let dist: f64 = features[i].iter().zip(&archetypes[label])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            hip_variance[label] += dist;
        }
        for c in 0..n_classes {
            if counts[c] > 0 { hip_variance[c] /= counts[c] as f64; }
        }

        // ── Multi-object simulation: "subtract" one class, check residual ──
        // For each image, compute: residual = image_features - nearest_archetype
        // Then check: does the residual correlate with ANY other archetype?
        // High correlation → multi-object (or class confusion at the boundary)

        let mut multi_object_candidates = Vec::new();
        for (i, &true_label) in labels.iter().enumerate() {
            // Subtract the true class archetype (simulates "removing" the primary object)
            let residual: Vec<f64> = features[i].iter().zip(&archetypes[true_label])
                .map(|(a, b)| a - b)
                .collect();

            // Check residual against all OTHER class archetypes
            let mut best_other_class = 0;
            let mut best_other_sim = f64::NEG_INFINITY;
            for c in 0..n_classes {
                if c == true_label || counts[c] == 0 { continue; }
                let sim = cosine(&residual, &archetypes[c]);
                if sim > best_other_sim {
                    best_other_sim = sim;
                    best_other_class = c;
                }
            }

            if best_other_sim > 0.3 {
                multi_object_candidates.push((i, true_label, best_other_class, best_other_sim));
            }
        }

        // ── BRANCH: intersection features between class pairs ──
        // For the top multi-object candidates, the residual IS the intersection
        // features — what's left after removing the primary class IS the secondary class.
        let mut pair_counts: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for &(_, primary, secondary, _) in &multi_object_candidates {
            let key = if primary < secondary { (primary, secondary) } else { (secondary, primary) };
            *pair_counts.entry(key).or_insert(0) += 1;
        }

        // ── CHAODA: outliers are images that don't fit ANY archetype well ──
        // (far from primary AND residual doesn't match secondary)
        let mut outliers = Vec::new();
        for (i, &true_label) in labels.iter().enumerate() {
            let primary_dist: f64 = features[i].iter().zip(&archetypes[true_label])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            
            // If far from own class AND not detected as multi-object
            let is_multi = multi_object_candidates.iter().any(|&(idx, _, _, _)| idx == i);
            if primary_dist > hip_variance[true_label] * 2.0 && !is_multi {
                outliers.push((i, true_label, primary_dist));
            }
        }

        eprintln!("=== HIP Multi-Object Detection ===");
        eprintln!("  Images: {}, Classes: {}", n, n_classes);
        eprintln!("  Multi-object candidates (residual sim > 0.3): {}", multi_object_candidates.len());
        eprintln!("  Top class-pair intersections (BRANCH traversals):");
        let mut pairs: Vec<_> = pair_counts.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1));
        for ((c1, c2), count) in pairs.iter().take(5) {
            eprintln!("    class {} × class {}: {} images share features", c1, c2, count);
        }
        eprintln!("  CHAODA outliers (far from all archetypes): {}", outliers.len());
        for (idx, label, dist) in outliers.iter().take(3) {
            eprintln!("    image {} (class {}): dist={:.3} (>{:.3} threshold)",
                idx, label, dist, hip_variance[*label] * 2.0);
        }
        eprintln!("  Per-class HIP spread (intra-class variance):");
        for c in 0..n_classes {
            if counts[c] > 0 {
                eprintln!("    class {}: variance={:.3}, count={}", c, hip_variance[c], counts[c]);
            }
        }

        assert!(multi_object_candidates.len() > 0, "should find some multi-object candidates");
    }
    #[test]
    #[ignore]
    fn test_centroid_focus_classification() {
        // Centroid-focused classification:
        // 1. Find energy centroid around each 1/3 intersection
        // 2. Extract detailed patch at centroid
        // 3. Classify patch → more precise than whole-image archetype
        
        let bytes = match std::fs::read("/tmp/tiny_imagenet_labeled.bin") {
            Ok(b) => b,
            Err(_) => { eprintln!("SKIP: /tmp/tiny_imagenet_labeled.bin not found"); return; }
        };

        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let d = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let n_classes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 4;
            labels.push(u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as usize);
        }

        let pixel_start = 12 + n * 4;
        let img_w = 64usize;
        let img_h = 64usize;
        let ch = 3usize;

        // ── Helper: read pixel from binary ──
        let pixel = |img_idx: usize, r: usize, c: usize, channel: usize| -> f64 {
            let off = pixel_start + (img_idx * d + r * img_w * ch + c * ch + channel) * 4;
            f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as f64
        };

        // ── Helper: luminance at (r,c) ──
        let luma = |img_idx: usize, r: usize, c: usize| -> f64 {
            0.299 * pixel(img_idx, r, c, 0) + 0.587 * pixel(img_idx, r, c, 1) + 0.114 * pixel(img_idx, r, c, 2)
        };

        // ── Step 1: For each image, find energy centroid around each 1/3 intersection ──
        let patch_radius = 8usize; // 16×16 patch around each intersection
        let intersections = [(img_h/3, img_w/3), (img_h/3, 2*img_w/3),
                             (2*img_h/3, img_w/3), (2*img_h/3, 2*img_w/3)];

        struct FocusPoint {
            centroid_r: f64,
            centroid_c: f64,
            energy: f64,
        }

        let n_use = n.min(200);

        // For each image, find the highest-energy intersection and its centroid
        let mut focus_features: Vec<Vec<f64>> = Vec::with_capacity(n_use);

        for img_idx in 0..n_use {
            let mut best_focus = FocusPoint { centroid_r: 32.0, centroid_c: 32.0, energy: 0.0 };

            for &(ir, ic) in &intersections {
                // Compute energy centroid within patch
                let mut total_energy = 0.0f64;
                let mut weighted_r = 0.0f64;
                let mut weighted_c = 0.0f64;

                let r_min = ir.saturating_sub(patch_radius);
                let r_max = (ir + patch_radius).min(img_h);
                let c_min = ic.saturating_sub(patch_radius);
                let c_max = (ic + patch_radius).min(img_w);

                for r in r_min..r_max {
                    for c in c_min..c_max {
                        let e = luma(img_idx, r, c);
                        // Use gradient magnitude as energy (edges = objects)
                        let grad = if r > 0 && r < img_h-1 && c > 0 && c < img_w-1 {
                            let dx = luma(img_idx, r, c+1) - luma(img_idx, r, c-1);
                            let dy = luma(img_idx, r+1, c) - luma(img_idx, r-1, c);
                            (dx * dx + dy * dy).sqrt()
                        } else {
                            0.0
                        };
                        total_energy += grad;
                        weighted_r += r as f64 * grad;
                        weighted_c += c as f64 * grad;
                    }
                }

                if total_energy > best_focus.energy {
                    best_focus = FocusPoint {
                        centroid_r: if total_energy > 0.0 { weighted_r / total_energy } else { ir as f64 },
                        centroid_c: if total_energy > 0.0 { weighted_c / total_energy } else { ic as f64 },
                        energy: total_energy,
                    };
                }
            }

            // ── Step 2: Extract detailed patch at centroid (12×12 = 144 pixels × 3ch = 432D) ──
            let focus_radius = 6usize;
            let cr = best_focus.centroid_r.round() as usize;
            let cc = best_focus.centroid_c.round() as usize;
            let r_min = cr.saturating_sub(focus_radius);
            let r_max = (cr + focus_radius).min(img_h);
            let c_min = cc.saturating_sub(focus_radius);
            let c_max = (cc + focus_radius).min(img_w);

            let mut patch_features = Vec::with_capacity(432);
            for r in r_min..r_max {
                for c in c_min..c_max {
                    for channel in 0..ch {
                        patch_features.push(pixel(img_idx, r, c, channel));
                    }
                }
            }
            // Pad to fixed size if patch was clipped by image boundary
            patch_features.resize(432, 0.0);
            focus_features.push(patch_features);
        }

        let feat_d = 432;

        // ── Step 3: Build archetypes from centroid patches ──
        let mut focus_archetypes: Vec<Vec<f64>> = vec![vec![0.0; feat_d]; n_classes];
        let mut counts = vec![0usize; n_classes];
        for (i, &label) in labels[..n_use].iter().enumerate() {
            for j in 0..feat_d { focus_archetypes[label][j] += focus_features[i][j]; }
            counts[label] += 1;
        }
        for c in 0..n_classes {
            if counts[c] > 0 { for j in 0..feat_d { focus_archetypes[c][j] /= counts[c] as f64; } }
        }

        // ── Step 4: Classify by nearest centroid-patch archetype ──
        let mut correct_focus = 0usize;
        for (i, &true_label) in labels[..n_use].iter().enumerate() {
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                if counts[c] == 0 { continue; }
                let dist: f64 = focus_features[i].iter().zip(&focus_archetypes[c])
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
                if dist < best_dist { best_dist = dist; best_class = c; }
            }
            if best_class == true_label { correct_focus += 1; }
        }
        let accuracy_focus = correct_focus as f64 / n_use as f64;

        // ── Step 5: Compress centroid patches via golden-step ──
        let base_dim = 17;
        let golden_step = 11;
        let compress = |v: &[f64]| -> Vec<f64> {
            let fd = v.len();
            let n_oct = (fd + base_dim - 1) / base_dim;
            let mut sum = vec![0.0f64; base_dim];
            let mut cnt = vec![0u32; base_dim];
            for oct in 0..n_oct {
                for bi in 0..base_dim {
                    let dim = oct * base_dim + ((bi * golden_step) % base_dim);
                    if dim < fd { sum[bi] += v[dim]; cnt[bi] += 1; }
                }
            }
            sum.iter().zip(&cnt).map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
        };

        let compressed_arch: Vec<Vec<f64>> = focus_archetypes.iter().map(|a| compress(a)).collect();
        let compressed_feat: Vec<Vec<f64>> = focus_features.iter().map(|f| compress(f)).collect();

        let mut correct_compressed = 0usize;
        for (i, &true_label) in labels[..n_use].iter().enumerate() {
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                if counts[c] == 0 { continue; }
                let dist: f64 = compressed_feat[i].iter().zip(&compressed_arch[c])
                    .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
                if dist < best_dist { best_dist = dist; best_class = c; }
            }
            if best_class == true_label { correct_compressed += 1; }
        }
        let accuracy_compressed = correct_compressed as f64 / n_use as f64;

        eprintln!("=== Centroid-Focus Classification ===");
        eprintln!("  Images: {}, Classes: {}", n_use, n_classes);
        eprintln!();
        eprintln!("  Centroid patch 432D:     {:.1}% ({}/{})", accuracy_focus * 100.0, correct_focus, n_use);
        eprintln!("  Compressed 17D (34B):    {:.1}% ({}/{})", accuracy_compressed * 100.0, correct_compressed, n_use);
        eprintln!("  Random baseline:         {:.1}%", 100.0 / n_classes as f64);
        eprintln!();
        eprintln!("  Compare to grid-line approaches:");
        eprintln!("    Grid 768D:     29.8%  (1536 bytes)");
        eprintln!("    Grid→17D:      14.2%  (34 bytes)");
        eprintln!("    Focus 432D:    {:.1}%  (864 bytes) ← centroid sweet spot", accuracy_focus * 100.0);
        eprintln!("    Focus→17D:     {:.1}%  (34 bytes) ← compressed focus", accuracy_compressed * 100.0);

        assert!(accuracy_focus > 1.0 / n_classes as f64, "should beat random");
    }
    #[test]
    #[ignore]
    fn test_multi_scan_nars_revision() {
        // Multiple scan strategies with NARS evidence revision.
        // Each scan is independent evidence. Revision increases confidence.
        // Stop when confidence > threshold (elevation cascade).
        
        let bytes = match std::fs::read("/tmp/tiny_imagenet_labeled.bin") {
            Ok(b) => b,
            Err(_) => { eprintln!("SKIP: /tmp/tiny_imagenet_labeled.bin not found"); return; }
        };

        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let d = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let n_classes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 4;
            labels.push(u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as usize);
        }
        let pixel_start = 12 + n * 4;
        let img_w = 64usize; let img_h = 64usize; let ch = 3usize;
        let n_use = n.min(200);

        let pixel = |img: usize, r: usize, c: usize, channel: usize| -> f64 {
            let off = pixel_start + (img * d + r * img_w * ch + c * ch + channel) * 4;
            f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) as f64
        };
        let luma = |img: usize, r: usize, c: usize| -> f64 {
            0.299 * pixel(img, r, c, 0) + 0.587 * pixel(img, r, c, 1) + 0.114 * pixel(img, r, c, 2)
        };

        // ── NARS revision ──
        fn nars_revision(f1: f64, c1: f64, f2: f64, c2: f64) -> (f64, f64) {
            let w1 = c1 / (1.0 - c1 + 1e-9);
            let w2 = c2 / (1.0 - c2 + 1e-9);
            let w = w1 + w2;
            let f = (w1 * f1 + w2 * f2) / (w + 1e-9);
            let c = w / (w + 1.0);
            (f.clamp(0.0, 1.0), c.clamp(0.0, 0.999))
        }

        // ── Scan strategy: extract features from a region ──
        fn extract_patch(pixel_fn: &dyn Fn(usize, usize, usize) -> f64,
                         r_center: usize, c_center: usize, radius: usize,
                         img_h: usize, img_w: usize, ch: usize) -> Vec<f64> {
            let mut f = Vec::new();
            let r0 = r_center.saturating_sub(radius);
            let r1 = (r_center + radius).min(img_h);
            let c0 = c_center.saturating_sub(radius);
            let c1 = (c_center + radius).min(img_w);
            for r in r0..r1 { for c in c0..c1 { for channel in 0..ch {
                f.push(pixel_fn(r, c, channel));
            }}}
            f
        }

        // ── Build archetypes for each scan strategy ──
        // Strategy 1: NW intersection (1/3, 1/3), 8×8 patch
        // Strategy 2: NE intersection (1/3, 2/3), 8×8 patch
        // Strategy 3: SW intersection (2/3, 1/3), 8×8 patch
        // Strategy 4: SE intersection (2/3, 2/3), 8×8 patch
        // Strategy 5: center crop, 12×12 patch
        // Strategy 6: horizontal midline
        // Strategy 7: vertical midline

        struct ScanStrategy {
            name: &'static str,
            extract: Box<dyn Fn(usize) -> Vec<f64>>,
        }

        // Build per-class archetypes for each strategy, then score
        let intersections = [
            ("NW patch", img_h/3, img_w/3, 4usize),
            ("NE patch", img_h/3, 2*img_w/3, 4),
            ("SW patch", 2*img_h/3, img_w/3, 4),
            ("SE patch", 2*img_h/3, 2*img_w/3, 4),
            ("Center",   img_h/2, img_w/2, 6),
        ];

        // For each strategy, build archetypes and classify
        let mut strategy_scores: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_use]; // per-image: [(class, similarity)]

        for &(name, cr, cc, radius) in &intersections {
            // Extract features for all images
            let features: Vec<Vec<f64>> = (0..n_use).map(|img| {
                let p = |r: usize, c: usize, channel: usize| pixel(img, r, c, channel);
                extract_patch(&p, cr, cc, radius, img_h, img_w, ch)
            }).collect();

            if features[0].is_empty() { continue; }
            let fd = features[0].len();

            // Build archetypes
            let mut arch = vec![vec![0.0; fd]; n_classes];
            let mut cnt = vec![0usize; n_classes];
            for (i, &l) in labels[..n_use].iter().enumerate() {
                for j in 0..fd { arch[l][j] += features[i][j]; }
                cnt[l] += 1;
            }
            for c in 0..n_classes {
                if cnt[c] > 0 { for j in 0..fd { arch[c][j] /= cnt[c] as f64; } }
            }

            // Score each image
            for i in 0..n_use {
                let mut best_c = 0;
                let mut best_sim = f64::NEG_INFINITY;
                for c in 0..n_classes {
                    if cnt[c] == 0 { continue; }
                    let dist: f64 = features[i].iter().zip(&arch[c])
                        .map(|(a, b)| (a-b)*(a-b)).sum::<f64>().sqrt();
                    let sim = 1.0 / (1.0 + dist); // convert distance to similarity
                    if sim > best_sim { best_sim = sim; best_c = c; }
                }
                strategy_scores[i].push((best_c, best_sim));
            }
        }

        // ── Multi-scan NARS revision: combine all strategy votes ──
        let mut correct_single = vec![0usize; intersections.len()]; // per-strategy accuracy
        let mut correct_revised = 0usize;

        for i in 0..n_use {
            let true_label = labels[i];

            // Single strategy accuracies
            for (s, &(pred_class, _)) in strategy_scores[i].iter().enumerate() {
                if pred_class == true_label { correct_single[s] += 1; }
            }

            // NARS revision: accumulate weighted evidence across all strategies.
            // Each scan contributes its similarity as evidence weight for the class it detected.
            // Confidence grows with number of agreeing scans (NARS: more evidence = more confident).
            let mut class_evidence: Vec<f64> = vec![0.0; n_classes]; // total similarity weight
            let mut class_votes: Vec<u32> = vec![0; n_classes];      // vote count

            for &(pred_class, similarity) in &strategy_scores[i] {
                class_evidence[pred_class] += similarity;
                class_votes[pred_class] += 1;
            }

            // Pick class with highest accumulated evidence.
            // NARS interpretation: frequency = avg similarity, confidence = vote proportion.
            let total_scans = strategy_scores[i].len() as f64;
            let mut best_c = 0;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..n_classes {
                if class_votes[c] == 0 { continue; }
                let avg_sim = class_evidence[c] / class_votes[c] as f64;
                let vote_frac = class_votes[c] as f64 / total_scans;
                // Combined: how similar (frequency) × how many agree (confidence)
                let score = avg_sim * vote_frac;
                if score > best_score { best_score = score; best_c = c; }
            }
            if best_c == true_label { correct_revised += 1; }
        }

        let revised_accuracy = correct_revised as f64 / n_use as f64;

        eprintln!("=== Multi-Scan NARS Revision ===");
        eprintln!("  {} images, {} classes, {} scan strategies", n_use, n_classes, intersections.len());
        eprintln!();
        eprintln!("  Per-strategy accuracy:");
        for (s, &(name, _, _, _)) in intersections.iter().enumerate() {
            let acc = correct_single[s] as f64 / n_use as f64;
            eprintln!("    {}: {:.1}% ({}/{})", name, acc * 100.0, correct_single[s], n_use);
        }
        eprintln!();
        eprintln!("  NARS-revised (all strategies combined): {:.1}% ({}/{})",
            revised_accuracy * 100.0, correct_revised, n_use);
        eprintln!("  Random baseline: {:.1}%", 100.0 / n_classes as f64);
        eprintln!();
        let best_single = correct_single.iter().max().unwrap();
        let best_single_acc = *best_single as f64 / n_use as f64;
        let improvement = revised_accuracy - best_single_acc;
        eprintln!("  Improvement over best single scan: {:.1}%", improvement * 100.0);
        eprintln!("  This is NARS evidence accumulation — each scan adds confidence.");

        assert!(revised_accuracy > best_single_acc,
            "NARS revision should improve over best single: {:.1}% vs {:.1}%",
            revised_accuracy * 100.0, best_single_acc * 100.0);
    }
}
