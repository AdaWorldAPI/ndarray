//! Opus CELT band energy computation.
//!
//! 21 quasi-Bark critical bands at 48kHz. Each band's energy is the
//! gain component of gain-shape quantization. The normalized coefficients
//! (after dividing by band energy) are the shape component → PVQ.
//!
//! Band boundaries from Opus `celt/modes.c` eBands48.

/// Opus CELT band boundaries at 48kHz, 960-sample frames (480 MDCT bins).
/// 22 boundaries define 21 bands. Bin index = frequency / (48000 / 960).
/// Band 0: bins 0-3 (~0-200 Hz), Band 20: bins 400-480 (~20-24 kHz).
pub const CELT_BANDS_48K: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 52, 60, 68, 80, 96,
    112, 136, 160, 200, 256, 480,
];

/// Number of critical bands.
pub const N_BANDS: usize = 21;

/// Compute band energies from MDCT coefficients.
///
/// Returns 21 f32 energies (sqrt of sum-of-squares per band).
/// These are the "gain" in gain-shape quantization.
pub fn band_energies(coeffs: &[f32]) -> [f32; N_BANDS] {
    let mut energies = [0.0f32; N_BANDS];
    for band in 0..N_BANDS {
        let lo = CELT_BANDS_48K[band];
        let hi = CELT_BANDS_48K[band + 1].min(coeffs.len());
        let mut sum_sq = 0.0f32;
        for i in lo..hi {
            if i < coeffs.len() {
                sum_sq += coeffs[i] * coeffs[i];
            }
        }
        energies[band] = sum_sq.sqrt();
    }
    energies
}

/// Normalize MDCT coefficients by band energy (produce unit-energy shape).
///
/// After normalization, each band has unit energy. The shape encodes
/// the spectral tilt within the band. PVQ quantizes this shape.
pub fn normalize_bands(coeffs: &[f32], energies: &[f32; N_BANDS]) -> Vec<f32> {
    let mut normalized = coeffs.to_vec();
    for band in 0..N_BANDS {
        let lo = CELT_BANDS_48K[band];
        let hi = CELT_BANDS_48K[band + 1].min(normalized.len());
        let e = energies[band].max(1e-10);
        for i in lo..hi {
            if i < normalized.len() {
                normalized[i] /= e;
            }
        }
    }
    normalized
}

/// Denormalize: multiply shape coefficients by band energies.
///
/// Inverse of normalize_bands. Used in the decoder path:
/// PVQ-decoded shape × band energies → MDCT coefficients → iMDCT → PCM.
pub fn denormalize_bands(shape: &[f32], energies: &[f32; N_BANDS]) -> Vec<f32> {
    let mut coeffs = shape.to_vec();
    for band in 0..N_BANDS {
        let lo = CELT_BANDS_48K[band];
        let hi = CELT_BANDS_48K[band + 1].min(coeffs.len());
        let e = energies[band];
        for i in lo..hi {
            if i < coeffs.len() {
                coeffs[i] *= e;
            }
        }
    }
    coeffs
}

/// Pack band energies to BF16 (21 × 2 bytes = 42 bytes).
pub fn energies_to_bf16(energies: &[f32; N_BANDS]) -> [u16; N_BANDS] {
    let mut bf16 = [0u16; N_BANDS];
    for i in 0..N_BANDS {
        let bits = energies[i].to_bits();
        let lsb = (bits >> 16) & 1;
        let biased = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
        bf16[i] = (biased >> 16) as u16;
    }
    bf16
}

/// Unpack BF16 band energies to f32.
pub fn bf16_to_energies(bf16: &[u16; N_BANDS]) -> [f32; N_BANDS] {
    let mut energies = [0.0f32; N_BANDS];
    for i in 0..N_BANDS {
        energies[i] = f32::from_bits((bf16[i] as u32) << 16);
    }
    energies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_count() {
        assert_eq!(CELT_BANDS_48K.len(), N_BANDS + 1);
    }

    #[test]
    fn band_energies_nonzero() {
        let coeffs: Vec<f32> = (0..480).map(|i| (i as f32 * 0.05).sin()).collect();
        let e = band_energies(&coeffs);
        let total: f32 = e.iter().sum();
        assert!(total > 0.1, "Total band energy too low: {}", total);
    }

    #[test]
    fn normalize_denormalize_roundtrip() {
        let coeffs: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
        let e = band_energies(&coeffs);
        let shape = normalize_bands(&coeffs, &e);
        let recovered = denormalize_bands(&shape, &e);

        for (orig, rec) in coeffs.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.01,
                "Roundtrip mismatch: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn bf16_energy_roundtrip() {
        let e = [1.0, 0.5, 2.0, 0.001, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bf16 = energies_to_bf16(&e);
        let recovered = bf16_to_energies(&bf16);
        for i in 0..5 {
            let err = (e[i] - recovered[i]).abs() / e[i].max(1e-6);
            assert!(err < 0.02, "BF16 roundtrip error for band {}: {:.4}", i, err);
        }
    }
}
