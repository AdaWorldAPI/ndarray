//! AudioFrame: 48-byte codec for one frame of audio.
//!
//! The complete encode/decode pipeline:
//!   encode: PCM → MDCT → band energies (gain) + PVQ (shape) → AudioFrame
//!   decode: AudioFrame → band energies × PVQ shape → iMDCT → PCM
//!
//! One AudioFrame = one graph node in lance-graph. 48 bytes = CAM-compatible.

use super::mdct;
use super::bands;
use super::pvq;

/// One audio frame: 42 bytes gain + 6 bytes shape = 48 bytes.
///
/// Maps to SPO:
///   Subject = spectral (WHAT frequencies) → band energies
///   Predicate = temporal (WHEN they happen) → PVQ summary bytes 2-3
///   Object = harmonic (HOW they ring) → PVQ summary bytes 4-5
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioFrame {
    /// 21 band energies as BF16 (42 bytes). The gain component.
    pub band_energies: [u16; bands::N_BANDS],
    /// PVQ shape fingerprint (6 bytes). HEEL/HIP/TWIG levels.
    pub pvq_summary: [u8; 6],
}

impl AudioFrame {
    /// Total byte size: 42 (energies) + 6 (pvq) = 48.
    pub const BYTE_SIZE: usize = bands::N_BANDS * 2 + 6;

    /// Encode one frame of PCM audio.
    ///
    /// `pcm`: mono f32 samples (padded to power of 2 internally).
    /// `pvq_k`: PVQ pulse budget per band (higher = better quality, more bits).
    pub fn encode(pcm: &[f32], pvq_k: u32) -> Self {
        // MDCT: time → frequency
        let coeffs = mdct::mdct_forward(pcm);

        // Band energies (gain)
        let energies = bands::band_energies(&coeffs);
        let bf16_energies = bands::energies_to_bf16(&energies);

        // Normalize bands (remove gain, keep shape)
        let shape = bands::normalize_bands(&coeffs, &energies);

        // PVQ encode the shape of the first (most important) band
        // For production: encode all 21 bands. For the POC: just first band's summary.
        let first_band_end = bands::CELT_BANDS_48K[1].min(shape.len());
        let pulses = pvq::pvq_encode(&shape[..first_band_end], pvq_k);
        let summary = pvq::pvq_summary(&pulses);

        AudioFrame {
            band_energies: bf16_energies,
            pvq_summary: summary,
        }
    }

    /// Decode: reconstruct PCM from AudioFrame + optional full PVQ data.
    ///
    /// Without PVQ data: uses band energies only (coarse reconstruction).
    /// The PVQ summary gives the HHTL routing info, not the full shape.
    /// For full quality: pass the per-band PVQ pulse vectors.
    pub fn decode_coarse(&self) -> Vec<f32> {
        let energies = bands::bf16_to_energies(&self.band_energies);

        // Synthesize a simple spectral envelope from band energies
        // Each band gets a flat spectrum at its energy level
        let n2 = bands::CELT_BANDS_48K[bands::N_BANDS].min(480);
        let mut coeffs = vec![0.0f32; n2];
        for band in 0..bands::N_BANDS {
            let lo = bands::CELT_BANDS_48K[band];
            let hi = bands::CELT_BANDS_48K[band + 1].min(n2);
            let n_bins = (hi - lo).max(1);
            let per_bin = energies[band] / (n_bins as f32).sqrt();
            for i in lo..hi {
                // Alternate signs for a more natural-sounding shape
                let sign = if (i - lo) % 2 == 0 { 1.0 } else { -1.0 };
                coeffs[i] = per_bin * sign;
            }
        }

        // iMDCT: frequency → time
        mdct::mdct_backward(&coeffs)
    }

    /// Serialize to 48 bytes.
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut bytes = [0u8; Self::BYTE_SIZE];
        for i in 0..bands::N_BANDS {
            let b = self.band_energies[i].to_le_bytes();
            bytes[i * 2] = b[0];
            bytes[i * 2 + 1] = b[1];
        }
        bytes[42..48].copy_from_slice(&self.pvq_summary);
        bytes
    }

    /// Deserialize from 48 bytes.
    pub fn from_bytes(bytes: &[u8; Self::BYTE_SIZE]) -> Self {
        let mut band_energies = [0u16; bands::N_BANDS];
        for i in 0..bands::N_BANDS {
            band_energies[i] = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        }
        let mut pvq_summary = [0u8; 6];
        pvq_summary.copy_from_slice(&bytes[42..48]);
        AudioFrame { band_energies, pvq_summary }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn frame_48_bytes() {
        assert_eq!(AudioFrame::BYTE_SIZE, 48);
    }

    #[test]
    fn encode_decode_nonzero() {
        // 440Hz sine at 48kHz, 1024 samples
        let pcm: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        let frame = AudioFrame::encode(&pcm, 8);

        // Band energies should be nonzero (at least the band containing 440Hz)
        let total_energy: f32 = frame.band_energies.iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .sum();
        assert!(total_energy > 0.01, "Encoded frame has no energy: {}", total_energy);

        // Decode
        let decoded = frame.decode_coarse();
        assert!(!decoded.is_empty());
        let decoded_energy: f32 = decoded.iter().map(|s| s * s).sum();
        assert!(decoded_energy > 1e-10, "Decoded has no energy: {}", decoded_energy);
    }

    #[test]
    fn serialize_roundtrip() {
        let frame = AudioFrame {
            band_energies: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            pvq_summary: [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
        };
        let bytes = frame.to_bytes();
        let recovered = AudioFrame::from_bytes(&bytes);
        assert_eq!(frame, recovered);
    }
}
