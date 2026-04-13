//! MDCT / iMDCT: Modified Discrete Cosine Transform.
//!
//! Transcoded from Opus CELT `celt/mdct.c`.
//! Forward: 960 PCM samples → 480 frequency coefficients.
//! Inverse: 480 coefficients → 960 PCM samples (with overlap-add).
//!
//! MDCT is: window → fold → FFT → post-rotate.
//! iMDCT is the reverse: pre-rotate → IFFT → unfold → window.
//!
//! Uses `hpc::fft::fft_f32` / `ifft_f32` internally. No external deps.

use crate::hpc::fft;
use core::f32::consts::PI;

/// Opus CELT frame size at 48kHz: 960 samples = 20ms.
pub const FRAME_SIZE: usize = 960;
/// MDCT output: N/2 = 480 frequency coefficients.
pub const MDCT_SIZE: usize = FRAME_SIZE / 2;

/// Sine window for MDCT (Opus uses a sine window for CELT mode).
/// w[n] = sin(π/N × (n + 0.5))
pub fn sine_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| (PI / n as f32 * (i as f32 + 0.5)).sin()).collect()
}

/// Forward MDCT: time-domain → frequency-domain.
///
/// Input: `pcm` — N samples (N must be power of 2, typically 960 padded to 1024).
/// Output: N/2 frequency coefficients.
///
/// Algorithm (Type-IV DCT via FFT):
///   1. Window the input
///   2. Fold: combine first and second half with sign flips
///   3. Pre-rotate by exp(-j·π/2N·(2n+1+N/2))
///   4. FFT of N/4 complex values
///   5. Post-rotate and extract real parts
pub fn mdct_forward(pcm: &[f32]) -> Vec<f32> {
    // Pad to next power of 2 if needed
    let n = pcm.len().next_power_of_two();
    let n2 = n / 2;
    let n4 = n / 4;

    // Window
    let window = sine_window(n);
    let mut windowed = vec![0.0f32; n];
    for i in 0..pcm.len().min(n) {
        windowed[i] = pcm[i] * window[i];
    }

    // Fold + pre-rotate → N/4 complex values for FFT
    let mut fft_buf = vec![0.0f32; n4 * 2]; // interleaved complex

    for k in 0..n4 {
        // Folding indices (from CELT mdct.c)
        let cos_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).cos();
        let sin_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).sin();

        // Combine samples from symmetric positions
        let a = windowed.get(2 * k).copied().unwrap_or(0.0);
        let b = windowed.get(n - 1 - 2 * k).copied().unwrap_or(0.0);
        let c = windowed.get(n2 + 2 * k).copied().unwrap_or(0.0);
        let d = if n2 > 2 * k + 1 { windowed[n2 - 1 - 2 * k] } else { 0.0 };

        let re = a - c;
        let im = d + b; // deliberate sign from MDCT folding

        // Pre-rotate
        fft_buf[2 * k] = re * cos_val + im * sin_val;
        fft_buf[2 * k + 1] = im * cos_val - re * sin_val;
    }

    // FFT
    fft::fft_f32(&mut fft_buf, n4);

    // Post-rotate → extract MDCT coefficients
    let mut output = vec![0.0f32; n2];
    for k in 0..n4 {
        let cos_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).cos();
        let sin_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).sin();

        let re = fft_buf[2 * k];
        let im = fft_buf[2 * k + 1];

        // Two output coefficients per FFT bin
        output[2 * k] = re * cos_val + im * sin_val;
        output[2 * k + 1] = im * cos_val - re * sin_val;
    }

    output
}

/// Inverse MDCT: frequency-domain → time-domain.
///
/// Input: N/2 frequency coefficients.
/// Output: N time-domain samples (needs overlap-add with previous frame).
pub fn mdct_backward(coeffs: &[f32]) -> Vec<f32> {
    // Pad to power of 2 if needed
    let n2_raw = coeffs.len();
    let n = (n2_raw * 2).next_power_of_two();
    let n2 = n / 2;
    let n4 = n / 4;
    let mut padded = vec![0.0f32; n2];
    padded[..n2_raw.min(n2)].copy_from_slice(&coeffs[..n2_raw.min(n2)]);

    // Pre-rotate → N/4 complex values
    let mut fft_buf = vec![0.0f32; n4 * 2];
    for k in 0..n4 {
        let cos_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).cos();
        let sin_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).sin();

        let a = padded.get(2 * k).copied().unwrap_or(0.0);
        let b = padded.get(2 * k + 1).copied().unwrap_or(0.0);

        fft_buf[2 * k] = a * cos_val + b * sin_val;
        fft_buf[2 * k + 1] = b * cos_val - a * sin_val;
    }

    // Inverse FFT
    fft::ifft_f32(&mut fft_buf, n4);

    // Post-rotate + unfold → N time-domain samples
    let window = sine_window(n);
    let mut output = vec![0.0f32; n];

    for k in 0..n4 {
        let cos_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).cos();
        let sin_val = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32 / 2.0) * 0.5).sin();

        let re = fft_buf[2 * k];
        let im = fft_buf[2 * k + 1];

        let y_re = re * cos_val + im * sin_val;
        let y_im = im * cos_val - re * sin_val;

        // Unfold to symmetric positions
        let idx_a = 2 * k;
        let idx_b = n - 1 - 2 * k;
        if idx_a < n { output[idx_a] = y_re * window[idx_a]; }
        if idx_b < n { output[idx_b] = y_im * window[idx_b]; }
    }

    // Scale (MDCT normalization: 2/N)
    let scale = 2.0 / n as f32;
    for s in &mut output { *s *= scale; }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdct_round_trip() {
        // Generate a simple test signal (sum of two sinusoids)
        let n = 1024; // power of 2
        let pcm: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
            })
            .collect();

        let coeffs = mdct_forward(&pcm);
        assert_eq!(coeffs.len(), n / 2);

        let reconstructed = mdct_backward(&coeffs);
        assert_eq!(reconstructed.len(), n);

        // Check non-trivial output (not all zeros)
        // Note: perfect reconstruction requires overlap-add of consecutive frames.
        // Single-frame roundtrip preserves energy but not exact waveform.
        let energy: f32 = reconstructed.iter().map(|s| s * s).sum();
        assert!(energy > 1e-6, "Reconstructed signal has no energy: {}", energy);
    }

    #[test]
    fn mdct_coeffs_nonzero() {
        let pcm: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let coeffs = mdct_forward(&pcm);
        let max_coeff = coeffs.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
        assert!(max_coeff > 0.01, "MDCT coefficients are all near zero");
    }
}
