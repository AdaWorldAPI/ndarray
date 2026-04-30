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

/// Walsh-Hadamard Transform (in-place, unnormalized).
///
/// Standard butterfly algorithm; companion to [`fft_f32`]. The transform is
/// its own inverse up to scaling — running `wht_f32` twice on the same data
/// multiplies every element by `n`. Divide by `n` after a round-trip if a
/// normalized inverse is desired.
///
/// `data.len()` must be a power of two.
///
/// # Example
///
/// ```
/// use ndarray::hpc::fft::wht_f32;
///
/// let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
/// wht_f32(&mut data);
/// // DC impulse → all bins equal 1.0
/// for &v in &data { assert!((v - 1.0).abs() < 1e-6); }
/// ```
pub fn wht_f32(data: &mut [f32]) {
    let n = data.len();
    assert!(
        n.is_power_of_two(),
        "WHT requires power-of-two length, got {n}"
    );
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
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
    fn test_rfft() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = rfft_f32(&input);
        // n/2+1 = 3 complex pairs = 6 floats
        assert_eq!(output.len(), 6);
        // DC component: sum = 10
        assert!((output[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_wht_zeros() {
        let mut data = vec![0.0f32; 8];
        wht_f32(&mut data);
        for &v in &data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_wht_dc_impulse() {
        // [1, 0, 0, 0, ...] → all-ones
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0;
        wht_f32(&mut data);
        for &v in &data {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_wht_round_trip() {
        // Running WHT twice scales by n; dividing by n recovers the input.
        let original = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let mut data = original.clone();
        wht_f32(&mut data);
        wht_f32(&mut data);
        let n = original.len() as f32;
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a / n - b).abs() < 1e-5, "round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wht_known_pair() {
        // WHT of [1, 1] = [2, 0]; WHT of [1, -1] = [0, 2]
        let mut a = vec![1.0f32, 1.0];
        wht_f32(&mut a);
        assert!((a[0] - 2.0).abs() < 1e-6);
        assert!(a[1].abs() < 1e-6);

        let mut b = vec![1.0f32, -1.0];
        wht_f32(&mut b);
        assert!(b[0].abs() < 1e-6);
        assert!((b[1] - 2.0).abs() < 1e-6);
    }
}
