//! FFT: Forward and inverse Fast Fourier Transform.
//!
//! Pure Rust Cooley-Tukey radix-2 implementation.
//! MKL-accelerated version available behind `intel-mkl` feature gate.

// FFT operates on raw slices; no ndarray imports needed.

/// Forward FFT on interleaved complex f32 data.
///
/// Input/output format: [re0, im0, re1, im1, ...]
/// Length n must be a power of 2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::fft::fft_f32;
///
/// let mut data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // DC impulse
/// fft_f32(&mut data, 4);
/// // All bins should be (1, 0)
/// assert!((data[0] - 1.0).abs() < 1e-5); // bin 0 real
/// ```
pub fn fft_f32(data: &mut [f32], n: usize) {
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");
    assert!(data.len() >= 2 * n, "Data must have at least 2*n elements");

    // Bit-reversal permutation
    bit_reverse_f32(data, n);

    // Cooley-Tukey butterfly
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = -2.0 * core::f32::consts::PI / size as f32;
        for k in (0..n).step_by(size) {
            for j in 0..half {
                let w_re = (angle * j as f32).cos();
                let w_im = (angle * j as f32).sin();
                let t_re = w_re * data[2 * (k + j + half)] - w_im * data[2 * (k + j + half) + 1];
                let t_im = w_re * data[2 * (k + j + half) + 1] + w_im * data[2 * (k + j + half)];
                let u_re = data[2 * (k + j)];
                let u_im = data[2 * (k + j) + 1];
                data[2 * (k + j)] = u_re + t_re;
                data[2 * (k + j) + 1] = u_im + t_im;
                data[2 * (k + j + half)] = u_re - t_re;
                data[2 * (k + j + half) + 1] = u_im - t_im;
            }
        }
        size *= 2;
    }
}

/// Inverse FFT on interleaved complex f32 data.
pub fn ifft_f32(data: &mut [f32], n: usize) {
    assert!(n.is_power_of_two());
    assert!(data.len() >= 2 * n);

    // Conjugate
    for i in 0..n {
        data[2 * i + 1] = -data[2 * i + 1];
    }

    // Forward FFT
    fft_f32(data, n);

    // Conjugate and scale
    let scale = 1.0 / n as f32;
    for i in 0..n {
        data[2 * i] *= scale;
        data[2 * i + 1] *= -scale;
    }
}

/// Forward FFT on interleaved complex f64 data.
pub fn fft_f64(data: &mut [f64], n: usize) {
    assert!(n.is_power_of_two());
    assert!(data.len() >= 2 * n);

    bit_reverse_f64(data, n);

    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = -2.0 * core::f64::consts::PI / size as f64;
        for k in (0..n).step_by(size) {
            for j in 0..half {
                let w_re = (angle * j as f64).cos();
                let w_im = (angle * j as f64).sin();
                let t_re = w_re * data[2 * (k + j + half)] - w_im * data[2 * (k + j + half) + 1];
                let t_im = w_re * data[2 * (k + j + half) + 1] + w_im * data[2 * (k + j + half)];
                let u_re = data[2 * (k + j)];
                let u_im = data[2 * (k + j) + 1];
                data[2 * (k + j)] = u_re + t_re;
                data[2 * (k + j) + 1] = u_im + t_im;
                data[2 * (k + j + half)] = u_re - t_re;
                data[2 * (k + j + half) + 1] = u_im - t_im;
            }
        }
        size *= 2;
    }
}

/// Inverse FFT on interleaved complex f64 data.
pub fn ifft_f64(data: &mut [f64], n: usize) {
    assert!(n.is_power_of_two());
    assert!(data.len() >= 2 * n);

    for i in 0..n {
        data[2 * i + 1] = -data[2 * i + 1];
    }
    fft_f64(data, n);
    let scale = 1.0 / n as f64;
    for i in 0..n {
        data[2 * i] *= scale;
        data[2 * i + 1] *= -scale;
    }
}

/// Real-to-complex FFT (f32): input is n real values, output is n/2+1 complex pairs.
///
/// Returns interleaved complex output: [re0, im0, re1, im1, ..., re_{n/2}, im_{n/2}]
pub fn rfft_f32(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of 2");

    // Pack real data as complex (zero imaginary)
    let mut complex = vec![0.0f32; 2 * n];
    for (i, &v) in input.iter().enumerate() {
        complex[2 * i] = v;
    }

    fft_f32(&mut complex, n);

    // Return first n/2+1 complex pairs
    let out_len = n / 2 + 1;
    complex[..2 * out_len].to_vec()
}

// ── Walsh-Hadamard Transform ──────────────────────────────────────
//
// The WHT is to quantization codecs what FFT is to signal processing:
// an O(n log n) orthogonal rotation that spreads energy uniformly
// across all coefficients. Unlike SVD (data-adaptive, O(n²k) training),
// the Hadamard rotation is deterministic, free, and self-inverse.
//
// Used by the HadCascade codec (bgz-tensor) for residual rotation
// before i4/i2 quantization. ICC 1.0000 on real model weights.

/// In-place Walsh-Hadamard Transform (normalized, self-inverse).
///
/// `data` length must be a power of 2. After transform, `||WHT(x)|| = ||x||`
/// (energy-preserving). Applying WHT twice returns the original vector.
///
/// SIMD: uses F32x16 butterfly for blocks ≥ 16 elements.
///
/// # Example
///
/// ```
/// use ndarray::hpc::fft::wht_f32;
///
/// let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
/// wht_f32(&mut x);
/// assert!((x[0] - 0.5).abs() < 1e-6); // 1/sqrt(4) * 1 = 0.5
///
/// // Self-inverse: WHT(WHT(x)) = x
/// wht_f32(&mut x);
/// assert!((x[0] - 1.0).abs() < 1e-5);
/// ```
pub fn wht_f32(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "WHT length must be a power of 2");

    let mut h = 1;
    while h < n {
        if h >= 16 {
            wht_butterfly_simd(data, n, h);
        } else {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let x = data[j];
                    let y = data[j + h];
                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
        }
        h *= 2;
    }

    let norm = 1.0 / (n as f32).sqrt();
    let mut i = 0;
    while i + 16 <= n {
        use crate::simd::F32x16;
        let v = F32x16::from_slice(&data[i..]);
        let scaled = v * F32x16::splat(norm);
        scaled.copy_to_slice(&mut data[i..i + 16]);
        i += 16;
    }
    while i < n {
        data[i] *= norm;
        i += 1;
    }
}

/// WHT butterfly for one level, SIMD-accelerated for h ≥ 16.
fn wht_butterfly_simd(data: &mut [f32], n: usize, h: usize) {
    use crate::simd::F32x16;
    for i in (0..n).step_by(h * 2) {
        let mut j = 0;
        while j + 16 <= h {
            let a = F32x16::from_slice(&data[i + j..]);
            let b = F32x16::from_slice(&data[i + j + h..]);
            let sum = a + b;
            let diff = a - b;
            sum.copy_to_slice(&mut data[i + j..i + j + 16]);
            diff.copy_to_slice(&mut data[i + j + h..i + j + h + 16]);
            j += 16;
        }
        while j < h {
            let x = data[i + j];
            let y = data[i + j + h];
            data[i + j] = x + y;
            data[i + j + h] = x - y;
            j += 1;
        }
    }
}

/// Convenience: WHT on a new vector (non-mutating).
pub fn wht_f32_new(input: &[f32]) -> Vec<f32> {
    let mut out = input.to_vec();
    wht_f32(&mut out);
    out
}

// ── Helpers ────────────────────────────────────────────────────────

fn bit_reverse_f32(data: &mut [f32], n: usize) {
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

fn bit_reverse_f64(data: &mut [f64], n: usize) {
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_dc_impulse() {
        // DC impulse: [1+0i, 0, 0, 0] → all bins = 1+0i
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        fft_f32(&mut data, 4);
        for i in 0..4 {
            assert!((data[2 * i] - 1.0).abs() < 1e-5, "bin {} real", i);
            assert!(data[2 * i + 1].abs() < 1e-5, "bin {} imag", i);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let original = vec![1.0f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let mut data = original.clone();
        fft_f64(&mut data, 4);
        ifft_f64(&mut data, 4);
        for i in 0..4 {
            assert!((data[2 * i] - original[2 * i]).abs() < 1e-10);
            assert!((data[2 * i + 1] - original[2 * i + 1]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wht_self_inverse() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        wht_f32(&mut data);
        wht_f32(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "self-inverse: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_wht_energy_preservation() {
        let mut data = vec![1.0f32, -2.0, 3.0, -4.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        wht_f32(&mut data);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4,
            "energy: {} vs {}", norm_before, norm_after);
    }

    #[test]
    fn test_wht_large_simd() {
        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.618).sin()).collect();
        let original = data.clone();
        wht_f32(&mut data);
        // Norm preservation at 1024-d (hits SIMD path)
        let n_orig: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_wht: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n_orig - n_wht).abs() / n_orig < 1e-4,
            "SIMD WHT norm: {} vs {}", n_orig, n_wht);
        // Self-inverse
        wht_f32(&mut data);
        let max_err = original.iter().zip(data.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "SIMD self-inverse max_err: {}", max_err);
    }

    #[test]
    fn test_rfft() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = rfft_f32(&input);
        // n/2+1 = 3 complex pairs = 6 floats
        assert_eq!(output.len(), 6);
        // DC component: sum = 10
        assert!((output[0] - 10.0).abs() < 1e-4);
    }
}
