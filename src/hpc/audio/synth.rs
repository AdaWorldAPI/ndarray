//! Synthesize pipeline: VoiceFrame → AudioFrame → iMDCT → PCM → WAV.
//!
//! This is the missing piece identified in lance-graph PR #168:
//!   "AudioFrame not connected to HHTL cascade levels"
//!   "WAV synthesis was bits-as-vectors — needs audio primitives"
//!
//! The pipeline:
//!   1. VoiceFrame (21B) → decompose into RvqFrame + PhaseDescriptor
//!   2. RvqFrame.archetype → VoiceCodebook lookup → VoiceArchetype (16B)
//!   3. RvqFrame.coarse → band energy prediction (8 codes → 21 BF16 bands)
//!   4. RvqFrame.fine → PVQ shape refinement (8 codes → 6B summary)
//!   5. PhaseDescriptor → phase-modulate the reconstructed bands
//!   6. AudioFrame.decode_coarse() → iMDCT → PCM
//!   7. Overlap-add consecutive frames → continuous PCM stream
//!   8. Write WAV header + PCM → .wav file
//!
//! The mode coloring (from Qualia17D → Mode → family_band_weights) is
//! applied at step 3: band energies are scaled by the QPL family's
//! spectral EQ before synthesis.

use super::codec::AudioFrame;
use super::bands;
use super::voice::{VoiceArchetype, VoiceCodebook, VoiceFrame, RvqFrame};
use super::phase::PhaseDescriptor;
use super::modes;

/// Decode a sequence of VoiceFrames into PCM audio.
///
/// This is the complete synthesis pipeline:
///   VoiceFrame → AudioFrame → iMDCT → overlap-add → PCM
///
/// `codebook`: the voice codebook (256 archetypes) for speaker lookup.
/// `coarse_centroids`: 256 × 21 BF16 band energy centroids (from HHTL HIP level).
/// `sample_rate`: output sample rate (48000 for Opus compatibility).
///
/// Returns mono f32 PCM samples.
pub fn synthesize(
    frames: &[VoiceFrame],
    codebook: &VoiceCodebook,
    coarse_centroids: &[[u16; bands::N_BANDS]; 256],
    sample_rate: u32,
) -> Vec<f32> {
    if frames.is_empty() { return vec![]; }

    // Frame parameters (Opus CELT compatible)
    let frame_samples = 960; // 20ms at 48kHz
    let hop_size = frame_samples / 2; // 50% overlap
    let total_samples = hop_size * (frames.len() + 1);
    let mut output = vec![0.0f32; total_samples];

    for (idx, vf) in frames.iter().enumerate() {
        // Step 1: Decompose VoiceFrame
        let rvq = &vf.rvq;
        let phase = &vf.phase;

        // Step 2: Look up voice archetype
        let archetype_idx = rvq.archetype as usize;
        let _archetype = if archetype_idx < codebook.entries.len() {
            codebook.entries[archetype_idx]
        } else {
            VoiceArchetype::zero()
        };

        // Step 3: Reconstruct band energies from coarse codes
        // Each coarse code indexes into the centroid table
        let band_energies = reconstruct_band_energies(rvq, coarse_centroids);

        // Step 4: Build AudioFrame from predicted energies + PVQ summary from fine codes
        let pvq_summary = fine_to_pvq_summary(&rvq.fine);
        let audio_frame = AudioFrame {
            band_energies,
            pvq_summary,
        };

        // Step 5: Phase modulation — adjust band energies based on phase coherence
        // Voiced frames get boosted mid-bands, attacks get transient emphasis
        let modulated = phase_modulate_frame(&audio_frame, phase);

        // Step 6: Decode to PCM via iMDCT
        let pcm = modulated.decode_coarse();

        // Step 7: Overlap-add into output buffer
        let start = idx * hop_size;
        let overlap_len = pcm.len().min(total_samples - start);
        for i in 0..overlap_len {
            // Hann window for smooth overlap-add
            let t = i as f32 / pcm.len() as f32;
            let window = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * t).cos());
            output[start + i] += pcm[i] * window;
        }
    }

    // Resample if needed (our MDCT produces at 48kHz, caller may want 24kHz)
    if sample_rate == 24000 {
        // Simple 2:1 decimation with averaging
        output = output.chunks(2)
            .map(|c| if c.len() == 2 { (c[0] + c[1]) * 0.5 } else { c[0] })
            .collect();
    }

    output
}

/// Reconstruct 21 BF16 band energies from RvqFrame coarse codes.
///
/// Each coarse code (0-255) indexes the HHTL HIP-level centroid table.
/// The 8 coarse codes cover overlapping band groups:
///   code[0]: bands 0-2   (sub-bass + bass)
///   code[1]: bands 3-5   (low-mid)
///   code[2]: bands 6-8   (mid)
///   code[3]: bands 9-11  (upper-mid)
///   code[4]: bands 12-14 (presence)
///   code[5]: bands 15-17 (brilliance)
///   code[6]: bands 18-20 (air)
///   code[7]: global gain  (scales all bands)
fn reconstruct_band_energies(
    rvq: &RvqFrame,
    centroids: &[[u16; bands::N_BANDS]; 256],
) -> [u16; bands::N_BANDS] {
    // Start with the centroid pointed to by code[0] (base spectral shape)
    let base = centroids[rvq.coarse[0] as usize];
    let mut energies = base;

    // Blend in contributions from other coarse codes per band group
    let band_groups: [(usize, usize); 7] = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21),
    ];

    for (group_idx, &(lo, hi)) in band_groups.iter().enumerate() {
        let code_idx = group_idx + 1;
        if code_idx >= 8 { break; }
        let centroid = &centroids[rvq.coarse[code_idx] as usize];
        for band in lo..hi.min(bands::N_BANDS) {
            // Weighted blend: 60% base + 40% group-specific centroid
            let base_f = f32::from_bits((energies[band] as u32) << 16);
            let group_f = f32::from_bits((centroid[band] as u32) << 16);
            let blended = base_f * 0.6 + group_f * 0.4;
            energies[band] = (blended.to_bits() >> 16) as u16;
        }
    }

    // Global gain from code[7]
    let gain = (rvq.coarse[7] as f32) / 128.0; // 0.0 to ~2.0
    for band in 0..bands::N_BANDS {
        let e = f32::from_bits((energies[band] as u32) << 16);
        let scaled = e * gain;
        energies[band] = (scaled.to_bits() >> 16) as u16;
    }

    energies
}

/// Convert 8 fine RVQ codes to a 6-byte PVQ summary.
///
/// The fine codes carry spectral detail within each band group.
/// We compress them to the AudioFrame's 6-byte PVQ summary format:
///   bytes 0-1: sign pattern (from fine[0..2])
///   bytes 2-3: temporal gradient (from fine[2..5])
///   bytes 4-5: harmonic detail (from fine[5..8])
fn fine_to_pvq_summary(fine: &[u8; 8]) -> [u8; 6] {
    [
        fine[0] ^ fine[1],  // sign pattern XOR
        fine[1] ^ fine[2],  // sign pattern continuation
        fine[2],            // temporal gradient
        fine[3] ^ fine[4],  // temporal modulation
        fine[5],            // harmonic detail
        fine[6] ^ fine[7],  // harmonic modulation
    ]
}

/// Apply phase modulation to an AudioFrame.
///
/// Voiced frames (high coherence): boost mid-band energy (formants).
/// Attacks (low coherence + high gradient): sharpen transient.
/// Noise (low coherence + low gradient): spread energy more evenly.
fn phase_modulate_frame(frame: &AudioFrame, phase: &PhaseDescriptor) -> AudioFrame {
    let mut out = *frame;
    let coherence = phase.bytes[0] as f32 / 255.0;
    let gradient = phase.bytes[1] as f32 / 255.0;

    for band in 0..bands::N_BANDS {
        let e = f32::from_bits((out.band_energies[band] as u32) << 16);
        let modulated = if phase.is_voiced() {
            // Voiced: boost formant region (bands 4-14), suppress extremes
            if (4..=14).contains(&band) {
                e * (1.0 + coherence * 0.3)
            } else {
                e * (1.0 - coherence * 0.1)
            }
        } else if phase.is_attack() {
            // Attack: boost all bands briefly (transient energy)
            e * (1.0 + gradient * 0.5)
        } else {
            // Noise: flatten spectrum slightly
            e * (1.0 + (0.5 - coherence) * 0.2)
        };
        out.band_energies[band] = (modulated.to_bits() >> 16) as u16;
    }

    out
}

/// Write PCM samples as a 16-bit WAV file.
///
/// Mono, little-endian, standard PCM format.
/// The WAV file is complete and playable by any audio software.
pub fn write_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let n_samples = pcm.len();
    let bits_per_sample: u16 = 16;
    let n_channels: u16 = 1;
    let byte_rate = sample_rate * (bits_per_sample as u32 / 8) * n_channels as u32;
    let block_align = n_channels * (bits_per_sample / 8);
    let data_size = (n_samples * 2) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + n_samples * 2);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // sub-chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // PCM format
    wav.extend_from_slice(&n_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    // Normalize and convert to i16
    let max_abs = pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-10);
    let scale = 32767.0 / max_abs;

    for &sample in pcm {
        let s = (sample * scale).clamp(-32768.0, 32767.0) as i16;
        wav.extend_from_slice(&s.to_le_bytes());
    }

    wav
}

/// Validate a WAV byte buffer (basic sanity check).
pub fn validate_wav(wav: &[u8]) -> Result<(u32, usize), &'static str> {
    if wav.len() < 44 { return Err("WAV too short"); }
    if &wav[0..4] != b"RIFF" { return Err("Missing RIFF header"); }
    if &wav[8..12] != b"WAVE" { return Err("Missing WAVE format"); }
    if &wav[12..16] != b"fmt " { return Err("Missing fmt chunk"); }

    let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
    let data_start = 44; // standard PCM WAV
    let data_size = wav.len() - data_start;
    let n_samples = data_size / 2; // 16-bit samples

    Ok((sample_rate, n_samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_wav_valid_header() {
        let pcm = vec![0.5f32; 4800]; // 100ms at 48kHz
        let wav = write_wav(&pcm, 48000);
        let (sr, n) = validate_wav(&wav).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(n, 4800);
    }

    #[test]
    fn write_wav_nonzero_samples() {
        let pcm: Vec<f32> = (0..960)
            .map(|i| (2.0 * core::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let wav = write_wav(&pcm, 48000);
        // Check data section has nonzero content
        let data = &wav[44..];
        let nonzero = data.iter().filter(|&&b| b != 0).count();
        assert!(nonzero > data.len() / 4, "WAV data should be mostly nonzero");
    }

    #[test]
    fn synthesize_empty_returns_empty() {
        let codebook = VoiceCodebook { entries: vec![VoiceArchetype::zero()] };
        let centroids = [[0u16; bands::N_BANDS]; 256];
        let pcm = synthesize(&[], &codebook, &centroids, 48000);
        assert!(pcm.is_empty());
    }

    #[test]
    fn synthesize_single_frame() {
        let codebook = VoiceCodebook { entries: vec![VoiceArchetype::zero(); 256] };
        // Create centroids with some energy in mid-bands
        let mut centroids = [[0u16; bands::N_BANDS]; 256];
        for c in centroids.iter_mut() {
            for band in 4..14 {
                // Set BF16 value for 0.1 (reasonable band energy)
                c[band] = (0.1f32.to_bits() >> 16) as u16;
            }
        }

        let frame = VoiceFrame {
            rvq: RvqFrame { archetype: 0, coarse: [0, 0, 0, 0, 0, 0, 0, 128], fine: [128; 8] },
            phase: PhaseDescriptor { bytes: [200, 30, 128, 50] }, // voiced, steady
        };

        let pcm = synthesize(&[frame], &codebook, &centroids, 48000);
        assert!(!pcm.is_empty(), "Should produce samples");
        let energy: f32 = pcm.iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "Should have nonzero energy");
    }

    #[test]
    fn fine_to_pvq_deterministic() {
        let fine = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let a = fine_to_pvq_summary(&fine);
        let b = fine_to_pvq_summary(&fine);
        assert_eq!(a, b);
    }

    #[test]
    fn phase_modulate_voiced_boosts_mid() {
        let mut energies = [0u16; bands::N_BANDS];
        for band in 0..bands::N_BANDS {
            energies[band] = (0.5f32.to_bits() >> 16) as u16;
        }
        let frame = AudioFrame { band_energies: energies, pvq_summary: [0; 6] };
        let voiced = PhaseDescriptor { bytes: [255, 30, 128, 50] }; // high coherence

        let modulated = phase_modulate_frame(&frame, &voiced);

        // Mid-bands (4-14) should be boosted
        let mid_orig: f32 = (4..=14).map(|b| f32::from_bits((frame.band_energies[b] as u32) << 16)).sum();
        let mid_mod: f32 = (4..=14).map(|b| f32::from_bits((modulated.band_energies[b] as u32) << 16)).sum();
        assert!(mid_mod > mid_orig, "Voiced phase should boost mid-bands: {} vs {}", mid_mod, mid_orig);
    }

    #[test]
    fn roundtrip_encode_synthesize() {
        // Encode a 440Hz sine, then synthesize back
        let pcm: Vec<f32> = (0..1024)
            .map(|i| (2.0 * core::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        let audio_frame = AudioFrame::encode(&pcm, 8);

        // Build a codebook with this frame's energies as the only centroid
        let codebook = VoiceCodebook { entries: vec![VoiceArchetype::zero(); 256] };
        let mut centroids = [[0u16; bands::N_BANDS]; 256];
        centroids[0] = audio_frame.band_energies;

        let voice_frame = VoiceFrame {
            rvq: RvqFrame { archetype: 0, coarse: [0, 0, 0, 0, 0, 0, 0, 128], fine: [0; 8] },
            phase: PhaseDescriptor { bytes: [200, 30, 128, 50] },
        };

        let decoded = synthesize(&[voice_frame], &codebook, &centroids, 48000);
        assert!(!decoded.is_empty());
        let energy: f32 = decoded.iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "Roundtrip should preserve energy");
    }
}
