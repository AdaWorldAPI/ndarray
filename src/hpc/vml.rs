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
}
