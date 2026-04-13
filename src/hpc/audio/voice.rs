//! VoiceArchetype — transcoded from Bark's 3-stage RVQ hierarchy.
//!
//! Bark's 3-stage pipeline (semantic GPT-2 → coarse GPT-2 → fine model)
//! maps directly to HHTL cascade levels:
//!
//!   HEEL: VoiceArchetype (16 i8 channels — voice identity qualia)
//!   HIP:  spectral envelope (21 BF16 band energies from Opus bands)
//!   TWIG: PVQ fine detail (6-byte harmonic signature)
//!   LEAF: full iMDCT → PCM waveform
//!
//! ElevenLabs insight: voice cloning = archetype embedding.
//! A 16-channel i8 vector captures speaker identity:
//!   channels 0-3: pitch register (bass/tenor/alto/soprano)
//!   channels 4-7: resonance (chest/head/nasal/breathy)
//!   channels 8-11: articulation (crisp/smooth/rough/whisper)
//!   channels 12-15: prosody (flat/dynamic/staccato/legato)
//!
//! Total: 16 bytes per voice identity. Fits in one SIMD lane.

/// Number of voice archetype channels.
pub const N_VOICE_CHANNELS: usize = 16;

/// VoiceArchetype: 16 i8 channels capturing voice identity.
///
/// Maps to Bark's semantic tokens (Stage 1): the coarse "what kind of voice"
/// decision, before any spectral detail. L1 distance between archetypes
/// predicts voice similarity.
///
/// The 16 channels correspond to perceptual voice qualia:
///   Pitch register, resonance, articulation, prosody.
///
/// Compression: 16 bytes (vs Bark's 1024-dim semantic token embedding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceArchetype {
    pub channels: [i8; N_VOICE_CHANNELS],
}

impl VoiceArchetype {
    pub const BYTE_SIZE: usize = N_VOICE_CHANNELS;

    /// Zero archetype (neutral voice).
    pub fn zero() -> Self {
        VoiceArchetype { channels: [0i8; N_VOICE_CHANNELS] }
    }

    /// L1 distance between two archetypes.
    #[inline]
    pub fn l1(&self, other: &VoiceArchetype) -> u32 {
        let mut d = 0u32;
        for i in 0..N_VOICE_CHANNELS {
            d += (self.channels[i] as i32 - other.channels[i] as i32).unsigned_abs();
        }
        d
    }

    /// Cosine similarity (for voice matching).
    pub fn cosine(&self, other: &VoiceArchetype) -> f64 {
        let mut dot = 0i64;
        let mut na = 0i64;
        let mut nb = 0i64;
        for i in 0..N_VOICE_CHANNELS {
            let a = self.channels[i] as i64;
            let b = other.channels[i] as i64;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let denom = ((na as f64) * (nb as f64)).sqrt();
        if denom < 1e-12 { 0.0 } else { dot as f64 / denom }
    }

    /// Extract archetype from raw embedding by quantizing to 16 channels.
    ///
    /// Takes a high-dimensional embedding (e.g., Bark's 1024-dim semantic token
    /// or ElevenLabs' speaker embedding) and compresses to 16 i8 channels
    /// via strided sampling + quantization.
    ///
    /// The stride determines which embedding dimensions map to which channels:
    ///   dim[0], dim[stride], dim[2*stride], ... → channels 0..15
    pub fn from_embedding(embedding: &[f32], stride: usize) -> Self {
        let mut channels = [0i8; N_VOICE_CHANNELS];

        // Find scale factor for quantization to i8 range
        let max_abs = embedding.iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-10);
        let scale = 127.0 / max_abs;

        for ch in 0..N_VOICE_CHANNELS {
            let dim = ch * stride.max(1);
            if dim < embedding.len() {
                channels[ch] = (embedding[dim] * scale).clamp(-128.0, 127.0) as i8;
            }
        }

        VoiceArchetype { channels }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; N_VOICE_CHANNELS] {
        let mut bytes = [0u8; N_VOICE_CHANNELS];
        for i in 0..N_VOICE_CHANNELS {
            bytes[i] = self.channels[i] as u8;
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; N_VOICE_CHANNELS]) -> Self {
        let mut channels = [0i8; N_VOICE_CHANNELS];
        for i in 0..N_VOICE_CHANNELS {
            channels[i] = bytes[i] as i8;
        }
        VoiceArchetype { channels }
    }

    /// Pitch register (channels 0-3 magnitude).
    pub fn pitch_energy(&self) -> u32 {
        (0..4).map(|i| self.channels[i].unsigned_abs() as u32).sum()
    }

    /// Resonance quality (channels 4-7 magnitude).
    pub fn resonance_energy(&self) -> u32 {
        (4..8).map(|i| self.channels[i].unsigned_abs() as u32).sum()
    }

    /// Articulation quality (channels 8-11 magnitude).
    pub fn articulation_energy(&self) -> u32 {
        (8..12).map(|i| self.channels[i].unsigned_abs() as u32).sum()
    }

    /// Prosody quality (channels 12-15 magnitude).
    pub fn prosody_energy(&self) -> u32 {
        (12..16).map(|i| self.channels[i].unsigned_abs() as u32).sum()
    }

    /// Modulate archetype with phase dynamics.
    ///
    /// Phase coherence sharpens articulation channels (8-11).
    /// Phase gradient boosts prosody channels (12-15).
    /// This is the bridge: amplitude identity (archetype) + temporal
    /// dynamics (phase) = complete voice characterization.
    ///
    /// The phase descriptor IS relative pressure within — it modulates
    /// the archetype's channels proportionally, not by overwriting.
    pub fn modulate_with_phase(&self, phase: &super::phase::PhaseDescriptor) -> Self {
        let mut out = *self;

        // Phase coherence → sharpen articulation (high coherence = crisp)
        let coherence = phase.bytes[0] as i16; // 0-255
        for i in 8..12 {
            // Scale articulation channels toward their sign direction
            let sign = if out.channels[i] >= 0 { 1i16 } else { -1 };
            let boost = sign * (coherence - 128) / 8; // ±16 max
            out.channels[i] = (out.channels[i] as i16 + boost).clamp(-127, 127) as i8;
        }

        // Phase gradient → boost prosody dynamics (high gradient = dynamic)
        let gradient = phase.bytes[1] as i16;
        for i in 12..16 {
            let sign = if out.channels[i] >= 0 { 1i16 } else { -1 };
            let boost = sign * (gradient - 128) / 8;
            out.channels[i] = (out.channels[i] as i16 + boost).clamp(-127, 127) as i8;
        }

        out
    }
}

/// VoiceCodebook: collection of voice archetypes for HHTL routing.
///
/// Maps to Bark Stage 1: the set of "voice types" the system knows about.
/// Each voice in the codebook is a prototype speaker pattern.
/// New speakers are matched to nearest archetype via L1 distance.
///
/// For a 256-entry codebook: 256 × 16 bytes = 4 KB.
#[derive(Clone, Debug)]
pub struct VoiceCodebook {
    pub entries: Vec<VoiceArchetype>,
}

impl VoiceCodebook {
    /// Build from raw embeddings (e.g., from Bark speaker prompts).
    pub fn build(embeddings: &[Vec<f32>], stride: usize) -> Self {
        let entries: Vec<VoiceArchetype> = embeddings.iter()
            .map(|e| VoiceArchetype::from_embedding(e, stride))
            .collect();
        VoiceCodebook { entries }
    }

    /// Find nearest archetype.
    pub fn nearest(&self, query: &VoiceArchetype) -> (u8, u32) {
        let mut best_idx = 0u8;
        let mut best_dist = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let d = query.l1(entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i as u8;
            }
        }
        (best_idx, best_dist)
    }

    /// Build 256 × 256 distance table for HHTL cascade.
    ///
    /// Returns a flat `[k × k]` u16 table (same format as AttentionTable).
    pub fn build_distance_table(&self) -> Vec<u16> {
        let k = self.entries.len();
        let mut table = vec![0u16; k * k];
        for i in 0..k {
            for j in (i + 1)..k {
                let d = self.entries[i].l1(&self.entries[j]);
                // Scale to u16: max L1 for 16 i8 channels = 16 × 255 = 4080
                let scaled = ((d as u32 * 65535) / 4080).min(65535) as u16;
                table[i * k + j] = scaled;
                table[j * k + i] = scaled;
            }
        }
        table
    }

    /// Byte size.
    pub fn byte_size(&self) -> usize {
        self.entries.len() * VoiceArchetype::BYTE_SIZE
    }
}

/// RVQ code frame: Bark's 3-stage output compressed to HHTL levels.
///
/// Stage 1 (semantic) → HEEL: voice archetype index (1 byte)
/// Stage 2 (coarse)  → HIP: 8 coarse spectral codes (8 bytes)
/// Stage 3 (fine)    → TWIG: 8 fine detail codes (8 bytes)
///
/// Total: 17 bytes per frame (vs Bark's ~128 bytes per frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvqFrame {
    /// HEEL: voice archetype index (0-255).
    pub archetype: u8,
    /// HIP: coarse spectral codes (8 codebook indices).
    pub coarse: [u8; 8],
    /// TWIG: fine detail codes (8 codebook indices).
    pub fine: [u8; 8],
}

impl RvqFrame {
    pub const BYTE_SIZE: usize = 17;

    /// Serialize to 17 bytes.
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut bytes = [0u8; Self::BYTE_SIZE];
        bytes[0] = self.archetype;
        bytes[1..9].copy_from_slice(&self.coarse);
        bytes[9..17].copy_from_slice(&self.fine);
        bytes
    }

    /// Deserialize from 17 bytes.
    pub fn from_bytes(bytes: &[u8; Self::BYTE_SIZE]) -> Self {
        let mut coarse = [0u8; 8];
        let mut fine = [0u8; 8];
        coarse.copy_from_slice(&bytes[1..9]);
        fine.copy_from_slice(&bytes[9..17]);
        RvqFrame { archetype: bytes[0], coarse, fine }
    }

    /// HEEL check: same voice archetype?
    #[inline]
    pub fn same_voice(&self, other: &RvqFrame) -> bool {
        self.archetype == other.archetype
    }

    /// HIP distance: L1 over coarse codes.
    pub fn coarse_l1(&self, other: &RvqFrame) -> u32 {
        let mut d = 0u32;
        for i in 0..8 {
            d += (self.coarse[i] as i32 - other.coarse[i] as i32).unsigned_abs();
        }
        d
    }
}

/// Complete voice frame: RVQ codes + phase dynamics.
///
/// The full 21-byte nonverbal unit:
///   RvqFrame (17B): WHAT the voice is doing (identity + spectral + detail)
///   PhaseDescriptor (4B): HOW the harmonics relate in time
///
/// This is the minimum viable unit for lossless nonverbal transmission.
/// AudioFrame (48B) + PhaseDescriptor (4B) = 52B is the analysis frame.
/// VoiceFrame (21B) is the compressed synthesis frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoiceFrame {
    pub rvq: RvqFrame,
    pub phase: super::phase::PhaseDescriptor,
}

impl VoiceFrame {
    pub const BYTE_SIZE: usize = RvqFrame::BYTE_SIZE + 4; // 21 bytes

    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut bytes = [0u8; Self::BYTE_SIZE];
        bytes[..17].copy_from_slice(&self.rvq.to_bytes());
        bytes[17..21].copy_from_slice(&self.phase.bytes);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; Self::BYTE_SIZE]) -> Self {
        let mut rvq_bytes = [0u8; 17];
        rvq_bytes.copy_from_slice(&bytes[..17]);
        let mut phase_bytes = [0u8; 4];
        phase_bytes.copy_from_slice(&bytes[17..21]);
        VoiceFrame {
            rvq: RvqFrame::from_bytes(&rvq_bytes),
            phase: super::phase::PhaseDescriptor { bytes: phase_bytes },
        }
    }

    /// Is this a voiced frame? (delegates to phase)
    pub fn is_voiced(&self) -> bool {
        self.phase.is_voiced()
    }

    /// Is this an attack/plosive? (delegates to phase)
    pub fn is_attack(&self) -> bool {
        self.phase.is_attack()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archetype_self_distance_zero() {
        let a = VoiceArchetype { channels: [10, -20, 30, -40, 50, -60, 70, -80,
                                             90, -100, 110, -120, 5, -15, 25, -35] };
        assert_eq!(a.l1(&a), 0);
    }

    #[test]
    fn archetype_self_cosine_one() {
        let a = VoiceArchetype { channels: [10, -20, 30, -40, 50, -60, 70, -80,
                                             1, 2, 3, 4, 5, 6, 7, 8] };
        let c = a.cosine(&a);
        assert!((c - 1.0).abs() < 1e-10, "Self cosine should be 1.0: {}", c);
    }

    #[test]
    fn archetype_from_embedding() {
        let emb: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) - 51.2).collect();
        let arch = VoiceArchetype::from_embedding(&emb, 64);
        // Should be nonzero
        let mag: u32 = arch.channels.iter().map(|&c| c.unsigned_abs() as u32).sum();
        assert!(mag > 0, "Archetype should be nonzero");
    }

    #[test]
    fn archetype_serialize_roundtrip() {
        let a = VoiceArchetype { channels: [1, -2, 3, -4, 5, -6, 7, -8,
                                             9, -10, 11, -12, 13, -14, 15, -16] };
        let bytes = a.to_bytes();
        let recovered = VoiceArchetype::from_bytes(&bytes);
        assert_eq!(a, recovered);
    }

    #[test]
    fn codebook_nearest() {
        let entries = vec![
            VoiceArchetype { channels: [100; 16] },
            VoiceArchetype { channels: [-100; 16] },
            VoiceArchetype { channels: [0; 16] },
        ];
        let cb = VoiceCodebook { entries };
        let query = VoiceArchetype { channels: [90; 16] };
        let (idx, dist) = cb.nearest(&query);
        assert_eq!(idx, 0, "Should match first entry");
        assert!(dist < 200, "Should be close: {}", dist);
    }

    #[test]
    fn rvq_frame_roundtrip() {
        let frame = RvqFrame {
            archetype: 42,
            coarse: [1, 2, 3, 4, 5, 6, 7, 8],
            fine: [10, 20, 30, 40, 50, 60, 70, 80],
        };
        let bytes = frame.to_bytes();
        let recovered = RvqFrame::from_bytes(&bytes);
        assert_eq!(frame, recovered);
    }

    #[test]
    fn phase_modulation_changes_articulation() {
        let base = VoiceArchetype { channels: [0, 0, 0, 0, 0, 0, 0, 0,
                                                50, 50, 50, 50, 0, 0, 0, 0] };
        // High coherence → should boost articulation channels
        let high_coh = super::super::phase::PhaseDescriptor { bytes: [255, 128, 128, 128] };
        let modulated = base.modulate_with_phase(&high_coh);

        // Articulation channels (8-11) should be boosted
        let base_art: i32 = (8..12).map(|i| base.channels[i].unsigned_abs() as i32).sum();
        let mod_art: i32 = (8..12).map(|i| modulated.channels[i].unsigned_abs() as i32).sum();
        assert!(mod_art >= base_art, "High coherence should boost articulation: {} vs {}", mod_art, base_art);
    }

    #[test]
    fn voice_frame_roundtrip() {
        let frame = VoiceFrame {
            rvq: RvqFrame { archetype: 7, coarse: [1; 8], fine: [2; 8] },
            phase: super::super::phase::PhaseDescriptor { bytes: [200, 50, 100, 30] },
        };
        let bytes = frame.to_bytes();
        assert_eq!(bytes.len(), VoiceFrame::BYTE_SIZE);
        let recovered = VoiceFrame::from_bytes(&bytes);
        assert_eq!(frame, recovered);
    }

    #[test]
    fn voice_frame_size() {
        assert_eq!(VoiceFrame::BYTE_SIZE, 21, "VoiceFrame should be 21 bytes (17 RVQ + 4 phase)");
    }

    #[test]
    fn distance_table_symmetric() {
        let entries = vec![
            VoiceArchetype { channels: [10; 16] },
            VoiceArchetype { channels: [-10; 16] },
            VoiceArchetype { channels: [50; 16] },
        ];
        let cb = VoiceCodebook { entries };
        let table = cb.build_distance_table();
        let k = 3;
        for i in 0..k {
            for j in 0..k {
                assert_eq!(table[i * k + j], table[j * k + i],
                    "Distance table not symmetric at ({}, {})", i, j);
            }
        }
    }
}
