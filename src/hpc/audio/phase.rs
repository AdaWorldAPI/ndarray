//! Phase shift dynamics — measuring what amplitude alone misses.
//!
//! Amplitude tells you WHAT frequencies are present.
//! Phase tells you HOW they relate to each other in time.
//!
//! Phase coherence between harmonics:
//!   High coherence → voiced sound (vowels, singing, resonance)
//!   Low coherence → noise (consonants, breath, static)
//!   Phase locked → natural voice
//!   Phase random → synthetic/robotic
//!
//! Phase gradient across frames:
//!   Steady phase → sustained note (singing, humming)
//!   Rotating phase → vibrato, tremolo
//!   Phase discontinuity → attack, plosive, glottal stop
//!
//! Maps to QPL dims:
//!   Phase coherence → coherence (dim 9) + clarity (dim 4)
//!   Phase gradient → velocity (dim 7) + integration (dim 16)
//!   Phase stability → groundedness (dim 14)
//!   Phase entropy → entropy (dim 8)
//!
//! Uses the same STFT from mel.rs but keeps phase info instead of
//! discarding it (which is what magnitude spectrograms do).

use crate::hpc::fft;
use core::f32::consts::PI;
use super::bands;

/// Phase coherence between adjacent harmonics within one frame.
///
/// Measures how "locked" the harmonics are to each other.
/// Natural voice: harmonics are phase-locked (coherence ≈ 1.0).
/// Noise: random phase relationships (coherence ≈ 0.0).
///
/// Returns per-band coherence values [0.0, 1.0].
pub fn band_phase_coherence(
    real: &[f32],
    imag: &[f32],
) -> [f32; bands::N_BANDS] {
    let mut coherence = [0.0f32; bands::N_BANDS];

    for band in 0..bands::N_BANDS {
        let lo = bands::CELT_BANDS_48K[band];
        let hi = bands::CELT_BANDS_48K[band + 1].min(real.len().min(imag.len()));
        if hi <= lo + 1 { continue; }

        // Phase differences between adjacent bins within this band
        let mut cos_sum = 0.0f64;
        let mut sin_sum = 0.0f64;
        let mut count = 0u32;

        for i in lo..(hi - 1) {
            if i >= real.len() || i + 1 >= real.len() { break; }
            let phase_i = imag[i].atan2(real[i]);
            let phase_next = imag[i + 1].atan2(real[i + 1]);
            let diff = phase_next - phase_i;
            cos_sum += diff.cos() as f64;
            sin_sum += diff.sin() as f64;
            count += 1;
        }

        if count > 0 {
            // Resultant length of unit vectors (circular mean)
            let r = ((cos_sum * cos_sum + sin_sum * sin_sum).sqrt()) / count as f64;
            coherence[band] = r.min(1.0) as f32;
        }
    }

    coherence
}

/// Phase gradient between two consecutive frames.
///
/// Measures how much phase rotates between frames at each band.
/// Steady gradient → sustained pitch (the gradient IS the frequency).
/// Changing gradient → pitch modulation (vibrato, portamento).
/// Zero gradient → DC or silence.
///
/// Returns per-band gradient in radians/frame.
pub fn phase_gradient(
    prev_real: &[f32], prev_imag: &[f32],
    curr_real: &[f32], curr_imag: &[f32],
) -> [f32; bands::N_BANDS] {
    let mut gradient = [0.0f32; bands::N_BANDS];

    for band in 0..bands::N_BANDS {
        let lo = bands::CELT_BANDS_48K[band];
        let hi = bands::CELT_BANDS_48K[band + 1]
            .min(prev_real.len())
            .min(curr_real.len());
        if hi <= lo { continue; }

        let mut total_diff = 0.0f64;
        let mut count = 0u32;

        for i in lo..hi {
            if i >= prev_real.len() || i >= curr_real.len() { break; }
            let prev_phase = prev_imag[i].atan2(prev_real[i]);
            let curr_phase = curr_imag[i].atan2(curr_real[i]);
            // Unwrap phase difference to [-π, π]
            let mut diff = curr_phase - prev_phase;
            while diff > PI { diff -= 2.0 * PI; }
            while diff < -PI { diff += 2.0 * PI; }
            total_diff += diff.abs() as f64;
            count += 1;
        }

        if count > 0 {
            gradient[band] = (total_diff / count as f64) as f32;
        }
    }

    gradient
}

/// Compact phase descriptor: 4 bytes capturing the essential phase dynamics.
///
/// byte 0: overall coherence (0=noise, 255=perfectly locked harmonics)
/// byte 1: gradient magnitude (0=static, 255=rapid phase rotation)
/// byte 2: coherence entropy (0=uniform coherence, 255=mixed voiced/unvoiced)
/// byte 3: gradient stability (0=steady pitch, 255=rapidly changing pitch)
///
/// These 4 bytes complement AudioFrame's PVQ summary:
///   PVQ summary = amplitude shape (WHAT)
///   Phase descriptor = temporal relationship (HOW)
///
/// Together: complete nonverbal vocal characterization in 52 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseDescriptor {
    pub bytes: [u8; 4],
}

impl PhaseDescriptor {
    /// Build from band coherence and gradient.
    pub fn from_bands(coherence: &[f32; bands::N_BANDS], gradient: &[f32; bands::N_BANDS]) -> Self {
        // Overall coherence: weighted mean (weight mid-bands more — voice formants)
        let mut coh_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for i in 0..bands::N_BANDS {
            let w = if (4..=14).contains(&i) { 2.0 } else { 1.0 }; // voice range weight
            coh_sum += coherence[i] * w;
            weight_sum += w;
        }
        let mean_coherence = coh_sum / weight_sum.max(1.0);

        // Gradient magnitude: RMS of per-band gradients
        let grad_rms = (gradient.iter().map(|g| g * g).sum::<f32>() / bands::N_BANDS as f32).sqrt();

        // Coherence entropy: are some bands voiced and others not?
        let mut coh_entropy = 0.0f32;
        let coh_total: f32 = coherence.iter().sum::<f32>().max(1e-10);
        for &c in coherence {
            if c > 1e-10 {
                let p = c / coh_total;
                coh_entropy -= p * p.ln();
            }
        }
        let max_entropy = (bands::N_BANDS as f32).ln();
        let norm_coh_entropy = coh_entropy / max_entropy;

        // Gradient stability: std dev of gradients (high = changing pitch)
        let grad_mean = gradient.iter().sum::<f32>() / bands::N_BANDS as f32;
        let grad_var = gradient.iter()
            .map(|g| (g - grad_mean) * (g - grad_mean))
            .sum::<f32>() / bands::N_BANDS as f32;
        let grad_std = grad_var.sqrt();

        PhaseDescriptor {
            bytes: [
                (mean_coherence * 255.0).clamp(0.0, 255.0) as u8,
                (grad_rms * 255.0 / PI).clamp(0.0, 255.0) as u8,
                (norm_coh_entropy * 255.0).clamp(0.0, 255.0) as u8,
                (grad_std * 255.0 / PI).clamp(0.0, 255.0) as u8,
            ],
        }
    }

    /// Map phase descriptor to QPL dims it informs.
    ///
    /// Returns (coherence→dim9, clarity→dim4, velocity→dim7,
    ///          entropy→dim8, groundedness→dim14).
    pub fn to_qualia_dims(&self) -> [(usize, f32); 5] {
        let coherence = self.bytes[0] as f32 / 255.0;
        let gradient = self.bytes[1] as f32 / 255.0;
        let coh_entropy = self.bytes[2] as f32 / 255.0;
        let stability = 1.0 - self.bytes[3] as f32 / 255.0;

        [
            (9,  coherence),    // coherence: phase-locked = unified
            (4,  coherence),    // clarity: locked harmonics = clear
            (7,  gradient),     // velocity: phase rotation = movement
            (8,  coh_entropy),  // entropy: mixed voiced/unvoiced
            (14, stability),    // groundedness: steady pitch = rooted
        ]
    }

    /// Is this a voiced frame? (coherence > threshold)
    pub fn is_voiced(&self) -> bool {
        self.bytes[0] > 128 // > 50% coherence
    }

    /// Is this an attack/plosive? (low coherence + high gradient)
    pub fn is_attack(&self) -> bool {
        self.bytes[0] < 64 && self.bytes[1] > 128
    }
}

/// STFT with phase preservation.
///
/// Returns (magnitude_per_frame, real_per_frame, imag_per_frame).
/// Each frame has n_fft/2+1 bins.
pub fn stft_with_phase(
    pcm: &[f32],
    window_size: usize,
    hop_size: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_fft = window_size.next_power_of_two();
    let n_bins = n_fft / 2 + 1;
    let window: Vec<f32> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / window_size as f32).cos()))
        .collect();

    let n_frames = if pcm.len() >= window_size {
        (pcm.len() - window_size) / hop_size + 1
    } else {
        0
    };

    let mut mags = Vec::with_capacity(n_frames);
    let mut reals = Vec::with_capacity(n_frames);
    let mut imags = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let mut data = vec![0.0f32; 2 * n_fft];
        for i in 0..window_size.min(pcm.len() - start) {
            data[2 * i] = pcm[start + i] * window[i];
        }

        fft::fft_f32(&mut data, n_fft);

        let mut mag = Vec::with_capacity(n_bins);
        let mut real = Vec::with_capacity(n_bins);
        let mut imag = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let re = data[2 * bin];
            let im = data[2 * bin + 1];
            mag.push((re * re + im * im).sqrt());
            real.push(re);
            imag.push(im);
        }

        mags.push(mag);
        reals.push(real);
        imags.push(imag);
    }

    (mags, reals, imags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_has_high_coherence() {
        // Pure 440Hz sine → all energy in one bin → high coherence
        let n = 1024;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        let (_mags, reals, imags) = stft_with_phase(&pcm, 512, 256);
        if reals.is_empty() { return; }

        let coh = band_phase_coherence(&reals[0], &imags[0]);
        // At least one band should have high coherence (the one with 440Hz)
        let max_coh = coh.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_coh > 0.3, "Pure sine should have coherent band: max={}", max_coh);
    }

    #[test]
    fn noise_has_low_coherence() {
        // White noise → random phases → low coherence
        let n = 1024;
        let mut rng = 0x12345678u64;
        let pcm: Vec<f32> = (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }).collect();

        let (_mags, reals, imags) = stft_with_phase(&pcm, 512, 256);
        if reals.is_empty() { return; }

        let coh = band_phase_coherence(&reals[0], &imags[0]);
        let mean_coh: f32 = coh.iter().sum::<f32>() / bands::N_BANDS as f32;
        // Noise should have lower mean coherence than pure tone
        assert!(mean_coh < 0.8, "Noise should have moderate-low coherence: mean={}", mean_coh);
    }

    #[test]
    fn phase_descriptor_voiced_detection() {
        let voiced_coh = [0.9f32; bands::N_BANDS];
        let steady_grad = [0.1f32; bands::N_BANDS];
        let desc = PhaseDescriptor::from_bands(&voiced_coh, &steady_grad);
        assert!(desc.is_voiced(), "High coherence should be voiced");
        assert!(!desc.is_attack(), "Steady should not be attack");
    }

    #[test]
    fn phase_descriptor_attack_detection() {
        let noise_coh = [0.1f32; bands::N_BANDS];
        let high_grad = [2.0f32; bands::N_BANDS];
        let desc = PhaseDescriptor::from_bands(&noise_coh, &high_grad);
        assert!(!desc.is_voiced(), "Low coherence should not be voiced");
        assert!(desc.is_attack(), "Low coherence + high gradient = attack");
    }

    #[test]
    fn phase_to_qualia_dims_valid() {
        let desc = PhaseDescriptor { bytes: [200, 50, 100, 30] };
        let dims = desc.to_qualia_dims();
        for (dim_idx, value) in dims {
            assert!(dim_idx < 17, "Invalid dim index: {}", dim_idx);
            assert!(value >= 0.0 && value <= 1.0, "Dim {} value out of range: {}", dim_idx, value);
        }
    }
}
