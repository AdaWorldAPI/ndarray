//! Mel filterbank — transcoded from Whisper's audio preprocessing.
//!
//! 80-channel mel spectrogram at 16kHz, matching Whisper's frontend:
//!   PCM 16kHz → STFT (400-sample window, 160-sample hop) → mel filterbank → log scale
//!
//! The mel scale maps linear frequencies to perceptual pitch:
//!   mel(f) = 2595 × log₁₀(1 + f/700)
//!
//! Key insight stolen from Whisper: the mel spectrogram IS the phoneme
//! fingerprint space. Each 80-dim mel frame can be compressed to a
//! 6-byte CAM fingerprint for HHTL cascade search.
//!
//! Zero external dependencies — uses `hpc::fft` internally.

use crate::hpc::fft;
use core::f32::consts::PI;

/// Number of mel channels (Whisper default).
pub const N_MELS: usize = 80;
/// STFT window size (400 samples = 25ms at 16kHz).
pub const STFT_WINDOW: usize = 400;
/// STFT hop size (160 samples = 10ms at 16kHz).
pub const STFT_HOP: usize = 160;
/// Sample rate for mel computation (Whisper operates at 16kHz).
pub const MEL_SAMPLE_RATE: usize = 16000;
/// FFT size (next power of 2 from STFT_WINDOW).
pub const FFT_SIZE: usize = 512;
/// Number of FFT bins used: FFT_SIZE/2 + 1.
pub const N_FFT_BINS: usize = FFT_SIZE / 2 + 1;

/// Convert frequency in Hz to mel scale.
/// Whisper uses the Slaney formula: mel = 2595 × log₁₀(1 + f/700)
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Precomputed mel filterbank matrix: [N_MELS × N_FFT_BINS].
///
/// Row-major: `filters[mel * N_FFT_BINS + bin]` = weight for mel channel `mel`
/// at FFT bin `bin`. Each row is a triangular filter centered at the mel-spaced
/// frequency.
///
/// Build once, reuse for every frame. 80 × 257 × 4 bytes = ~82 KB.
pub fn build_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut filters = vec![0.0f32; n_mels * n_bins];

    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 points evenly spaced in mel domain
    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
        .collect();

    // Convert mel points back to Hz, then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points.iter()
        .map(|&h| h * n_fft as f32 / sample_rate as f32)
        .collect();

    // Build triangular filters
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for bin in 0..n_bins {
            let b = bin as f32;
            let weight = if b >= left && b < center {
                // Rising slope
                (b - left) / (center - left).max(1e-10)
            } else if b >= center && b <= right {
                // Falling slope
                (right - b) / (right - center).max(1e-10)
            } else {
                0.0
            };
            filters[m * n_bins + bin] = weight;
        }
    }

    filters
}

/// Hann window for STFT.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos())).collect()
}

/// Compute magnitude spectrogram via STFT.
///
/// Input: mono f32 PCM at 16kHz.
/// Output: `[n_frames × n_bins]` magnitude values (row-major).
///
/// Uses `hpc::fft` internally. Window = Hann, hop = 160 samples.
pub fn stft_magnitude(pcm: &[f32], window_size: usize, hop_size: usize) -> Vec<f32> {
    let n_fft = window_size.next_power_of_two();
    let n_bins = n_fft / 2 + 1;
    let window = hann_window(window_size);

    let n_frames = if pcm.len() >= window_size {
        (pcm.len() - window_size) / hop_size + 1
    } else {
        0
    };

    let mut magnitudes = Vec::with_capacity(n_frames * n_bins);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;

        // Apply window, then pack as interleaved [re, im, re, im, ...]
        let mut data = vec![0.0f32; 2 * n_fft];
        for i in 0..window_size.min(pcm.len() - start) {
            data[2 * i] = pcm[start + i] * window[i]; // real
            // imaginary stays 0
        }

        // FFT (interleaved complex: data[2*k] = re, data[2*k+1] = im)
        fft::fft_f32(&mut data, n_fft);

        // Magnitude: |X[k]| = sqrt(re² + im²)
        for bin in 0..n_bins {
            let re = data[2 * bin];
            let im = data[2 * bin + 1];
            let mag = (re * re + im * im).sqrt();
            magnitudes.push(mag);
        }
    }

    magnitudes
}

/// Compute 80-channel log mel spectrogram (Whisper frontend).
///
/// Input: mono f32 PCM at 16kHz.
/// Output: `[n_frames × N_MELS]` log-mel values (row-major).
///
/// Pipeline: PCM → STFT magnitude → mel filterbank → log scale.
pub fn log_mel_spectrogram(pcm: &[f32]) -> Vec<f32> {
    let n_bins = FFT_SIZE / 2 + 1;

    // Build mel filters (could be cached, but 82KB is cheap)
    let filters = build_mel_filters(MEL_SAMPLE_RATE, FFT_SIZE, N_MELS);

    // STFT magnitude
    let mag = stft_magnitude(pcm, STFT_WINDOW, STFT_HOP);
    let n_frames = mag.len() / n_bins;

    // Apply mel filterbank + log scale
    let mut log_mel = Vec::with_capacity(n_frames * N_MELS);

    for frame in 0..n_frames {
        for mel in 0..N_MELS {
            let mut energy = 0.0f32;
            for bin in 0..n_bins {
                energy += filters[mel * n_bins + bin] * mag[frame * n_bins + bin];
            }
            // Log scale with floor (Whisper uses max(energy, 1e-10))
            let log_e = energy.max(1e-10).ln();
            log_mel.push(log_e);
        }
    }

    log_mel
}

/// Compress an 80-dim mel frame to BF16 (160 bytes → useful for distance).
pub fn mel_frame_to_bf16(frame: &[f32]) -> [u16; N_MELS] {
    let mut bf16 = [0u16; N_MELS];
    for i in 0..N_MELS.min(frame.len()) {
        let bits = frame[i].to_bits();
        let lsb = (bits >> 16) & 1;
        let biased = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
        bf16[i] = (biased >> 16) as u16;
    }
    bf16
}

/// L1 distance between two BF16 mel frames (for HHTL cascade).
pub fn mel_l1_bf16(a: &[u16; N_MELS], b: &[u16; N_MELS]) -> f32 {
    let mut d = 0.0f32;
    for i in 0..N_MELS {
        let va = f32::from_bits((a[i] as u32) << 16);
        let vb = f32::from_bits((b[i] as u32) << 16);
        d += (va - vb).abs();
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_hz_roundtrip() {
        for &f in &[440.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(f);
            let back = mel_to_hz(mel);
            assert!((f - back).abs() < 0.01, "Roundtrip failed: {} → {} → {}", f, mel, back);
        }
    }

    #[test]
    fn mel_scale_monotonic() {
        let m1 = hz_to_mel(100.0);
        let m2 = hz_to_mel(1000.0);
        let m3 = hz_to_mel(8000.0);
        assert!(m1 < m2 && m2 < m3);
        // Higher frequencies are compressed in mel scale
        assert!((m2 - m1) > (m3 - m2) * 0.3);
    }

    #[test]
    fn build_filters_shape() {
        let filters = build_mel_filters(MEL_SAMPLE_RATE, FFT_SIZE, N_MELS);
        assert_eq!(filters.len(), N_MELS * N_FFT_BINS);
        // Each mel channel should have some nonzero weights
        for mel in 0..N_MELS {
            let row_sum: f32 = (0..N_FFT_BINS)
                .map(|bin| filters[mel * N_FFT_BINS + bin])
                .sum();
            assert!(row_sum > 0.0, "Mel channel {} has no energy", mel);
        }
    }

    #[test]
    fn log_mel_440hz_sine() {
        // 440Hz sine at 16kHz, 1 second
        let n_samples = MEL_SAMPLE_RATE;
        let pcm: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / MEL_SAMPLE_RATE as f32).sin())
            .collect();

        let log_mel = log_mel_spectrogram(&pcm);
        let n_frames = log_mel.len() / N_MELS;
        assert!(n_frames > 0, "Should produce at least one frame");

        // The mel channel containing 440Hz should have high energy
        // 440Hz ≈ mel channel ~14 (depends on exact mel spacing)
        let frame0 = &log_mel[0..N_MELS];
        let max_mel = frame0.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // Peak should be in low-to-mid range (440Hz is low)
        assert!(max_mel.0 < 30, "440Hz peak at mel {}, expected < 30", max_mel.0);
    }

    #[test]
    fn mel_bf16_roundtrip() {
        let frame: Vec<f32> = (0..N_MELS).map(|i| (i as f32 * 0.1) - 4.0).collect();
        let bf16 = mel_frame_to_bf16(&frame);
        for i in 0..N_MELS {
            let recovered = f32::from_bits((bf16[i] as u32) << 16);
            let err = (frame[i] - recovered).abs();
            assert!(err < 0.1, "BF16 error at mel {}: {:.4} vs {:.4}", i, frame[i], recovered);
        }
    }
}
