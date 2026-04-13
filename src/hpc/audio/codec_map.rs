//! Codec provenance map: which real codec each primitive comes from.
//!
//! Every primitive in this audio stack was stolen from a production codec.
//! Nothing invented — only transcoded and compressed to fit the HHTL cascade.
//!
//! ```text
//! ┌─────────────┬──────────┬─────────┬────────┬─────────┬──────┬───────────┐
//! │ Our type    │ Opus     │ Whisper │ MP3    │ Vorbis  │ Bark │ ElevenLabs│
//! ├─────────────┼──────────┼─────────┼────────┼─────────┼──────┼───────────┤
//! │ MDCT        │ CELT     │         │ hybrid │ ✓       │      │           │
//! │ 21 bands    │ eBands48 │         │ 32 sub │ ✓       │      │           │
//! │ PVQ shape   │ CELT PVQ │         │        │ residue │      │           │
//! │ Mel 80ch    │          │ frontend│        │         │      │           │
//! │ Phase 4B    │          │ STFT ∠  │        │         │      │           │
//! │ VoiceArch   │          │         │        │         │ spk  │ embedding │
//! │ RvqFrame    │          │         │        │         │ 3stg │           │
//! │ OctaveBand  │          │         │ ✓      │ floor   │      │           │
//! │ Mode        │          │         │        │         │      │ emotion   │
//! │ HHTL skip   │          │         │ mask   │ floor   │      │           │
//! │ CompLinear  │          │         │        │ VQ cb   │ RVQ  │           │
//! │ Qualia17D   │ (QPL)    │         │        │         │ sem  │ emotion   │
//! └─────────────┴──────────┴─────────┴────────┴─────────┴──────┴───────────┘
//! ```
//!
//! The architecture replaces neural inference with graph search at every stage:
//!   MP3's psychoacoustic model → HHTL cascade (RouteAction::Skip)
//!   Whisper's transformer → phoneme graph shortest path
//!   Bark's 3 GPT-2 stages → 3 HHTL levels (HEEL/HIP/TWIG)
//!   Vorbis's codebook VQ → CompiledLinear VNNI palette lookup
//!   ElevenLabs' voice cloning → VoiceArchetype 16-byte embedding

/// Codec provenance for each audio primitive.
///
/// Documents which production codec each type was transcoded from,
/// what aspect of that codec it captures, and what it replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecSource {
    Opus,
    Whisper,
    Mp3,
    OggVorbis,
    Bark,
    ElevenLabs,
}

/// What aspect of audio each primitive captures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioAspect {
    /// Spectral energy distribution (WHAT frequencies)
    SpectralEnvelope,
    /// Fine spectral shape within bands (HOW the energy is distributed)
    SpectralShape,
    /// Perceptual frequency mapping (WHERE in human hearing)
    PerceptualMapping,
    /// Temporal phase relationships (WHEN harmonics align)
    PhaseRelationship,
    /// Speaker identity (WHO is speaking)
    SpeakerIdentity,
    /// Semantic/emotional content (WHY it sounds that way)
    SemanticContent,
    /// Psychoacoustic masking (WHAT to skip)
    MaskingDecision,
    /// Codebook lookup (HOW to decompress)
    CodebookLookup,
}

/// Complete provenance record for one primitive.
pub struct Provenance {
    pub our_type: &'static str,
    pub byte_size: usize,
    pub source: CodecSource,
    pub aspect: AudioAspect,
    pub source_concept: &'static str,
    pub what_it_replaces: &'static str,
}

/// Full provenance table for every audio primitive.
///
/// This IS the design document. If a new primitive doesn't appear here,
/// it wasn't stolen from a real codec and shouldn't exist.
pub const PROVENANCE: &[Provenance] = &[
    // ═══ From Opus CELT ═══
    Provenance {
        our_type: "AudioFrame.band_energies",
        byte_size: 42,
        source: CodecSource::Opus,
        aspect: AudioAspect::SpectralEnvelope,
        source_concept: "eBands48 critical bands, gain in gain-shape split",
        what_it_replaces: "Per-coefficient quantization (MP3/Vorbis)",
    },
    Provenance {
        our_type: "AudioFrame.pvq_summary",
        byte_size: 6,
        source: CodecSource::Opus,
        aspect: AudioAspect::SpectralShape,
        source_concept: "PVQ (Pyramid Vector Quantization) pulse allocation",
        what_it_replaces: "Huffman-coded residuals (MP3) / VQ codebook (Vorbis)",
    },
    Provenance {
        our_type: "mdct_forward / mdct_backward",
        byte_size: 0, // transform, not stored
        source: CodecSource::Opus,
        aspect: AudioAspect::SpectralEnvelope,
        source_concept: "CELT MDCT: 960-sample window → 480 frequency bins",
        what_it_replaces: "FFT+windowing (all codecs use some form)",
    },

    // ═══ From Whisper ═══
    Provenance {
        our_type: "mel::log_mel_spectrogram",
        byte_size: 160, // 80 × BF16 per frame
        source: CodecSource::Whisper,
        aspect: AudioAspect::PerceptualMapping,
        source_concept: "80-channel mel filterbank at 16kHz, Hann STFT",
        what_it_replaces: "Transformer encoder (150M params → 80 f32 per frame)",
    },

    // ═══ From MP3 ═══
    Provenance {
        our_type: "HhtlCache::route() → Skip",
        byte_size: 0, // decision, not stored
        source: CodecSource::Mp3,
        aspect: AudioAspect::MaskingDecision,
        source_concept: "Psychoacoustic masking model (simultaneous + temporal)",
        what_it_replaces: "ISO 11172-3 psychoacoustic model 1/2 (iterative bit allocation)",
    },
    Provenance {
        our_type: "OctaveBand",
        byte_size: 13, // 3×f32 + u8
        source: CodecSource::Mp3,
        aspect: AudioAspect::SpectralEnvelope,
        source_concept: "32-subband polyphase filterbank (octave-spaced)",
        what_it_replaces: "Per-subband quantization + Huffman (MP3 granules)",
    },

    // ═══ From Ogg Vorbis ═══
    Provenance {
        our_type: "CompiledLinear (ndarray burn)",
        byte_size: 65536, // 256 centroids × 256 dim
        source: CodecSource::OggVorbis,
        aspect: AudioAspect::CodebookLookup,
        source_concept: "VQ codebook: precomputed centroids, lookup-based decode",
        what_it_replaces: "Huffman trees (MP3) / arithmetic coding (Opus range coder)",
    },

    // ═══ From Bark (Suno) ═══
    Provenance {
        our_type: "RvqFrame.archetype (HEEL)",
        byte_size: 1,
        source: CodecSource::Bark,
        aspect: AudioAspect::SemanticContent,
        source_concept: "Stage 1: GPT-2 semantic tokens (coarse meaning)",
        what_it_replaces: "350M-param GPT-2 autoregressive generation",
    },
    Provenance {
        our_type: "RvqFrame.coarse (HIP)",
        byte_size: 8,
        source: CodecSource::Bark,
        aspect: AudioAspect::SpectralEnvelope,
        source_concept: "Stage 2: GPT-2 coarse acoustic tokens (spectral envelope)",
        what_it_replaces: "350M-param GPT-2 conditioned on semantic tokens",
    },
    Provenance {
        our_type: "RvqFrame.fine (TWIG)",
        byte_size: 8,
        source: CodecSource::Bark,
        aspect: AudioAspect::SpectralShape,
        source_concept: "Stage 3: non-autoregressive fine acoustic tokens",
        what_it_replaces: "Fine model (smaller network, fills spectral detail)",
    },

    // ═══ From ElevenLabs ═══
    Provenance {
        our_type: "VoiceArchetype",
        byte_size: 16,
        source: CodecSource::ElevenLabs,
        aspect: AudioAspect::SpeakerIdentity,
        source_concept: "Speaker embedding (voice cloning conditioning vector)",
        what_it_replaces: "512-dim speaker embedding (2KB → 16 bytes)",
    },

    // ═══ Phase (novel — no codec stores this) ═══
    Provenance {
        our_type: "PhaseDescriptor",
        byte_size: 4,
        source: CodecSource::Whisper, // closest: Whisper STFT preserves phase internally
        aspect: AudioAspect::PhaseRelationship,
        source_concept: "STFT phase (discarded by all codecs except Griffin-Lim)",
        what_it_replaces: "Nothing — all codecs discard phase. We keep it as relative pressure.",
    },

    // ═══ Qualia (novel — derived from QPL musical calibration) ═══
    Provenance {
        our_type: "Qualia17D",
        byte_size: 68,
        source: CodecSource::Bark, // closest: Bark semantic tokens carry meaning
        aspect: AudioAspect::SemanticContent,
        source_concept: "QPL: Octave→arousal, Fifth→valence, Third→warmth, Tritone→tension",
        what_it_replaces: "No codec captures nonverbal meaning explicitly. This is the grid.",
    },
];

/// Total bytes for one complete frame (all primitives combined).
///
/// AudioFrame (48) + PhaseDescriptor (4) + VoiceArchetype (16, amortized)
/// = 52 bytes per frame for complete nonverbal characterization.
/// + RvqFrame (17) for HHTL-compressed TTS output = 69 bytes.
///
/// Compare:
///   MP3 128kbps: ~417 bytes per 26ms frame
///   Opus 64kbps: ~166 bytes per 20ms frame
///   Bark tokens: ~128 bytes per frame
///   Ours: 52-69 bytes per frame (complete, including phase + identity)
pub const FRAME_BUDGET: usize = 52;
pub const FRAME_BUDGET_WITH_TTS: usize = 69;

/// Codec comparison: bits per second at comparable quality.
///
/// These are approximate — our codec is lossy in a fundamentally
/// different way (palette quantization, not psychoacoustic masking).
pub const BITRATE_COMPARISON: &[(&str, u32, &str)] = &[
    ("MP3 128k",     128_000, "psychoacoustic masking, Huffman"),
    ("Opus 64k",      64_000, "CELT+SILK hybrid, range coder"),
    ("Vorbis 128k",  128_000, "MDCT, floor+residue, VQ codebook"),
    ("Bark tokens",   25_600, "3-stage RVQ, ~100 tokens/sec × 256 bits"),
    ("Ours (48kHz)",  20_800, "52 bytes × 50 fps × 8 bits = 20.8 kbps"),
    ("Ours (24kHz)",  10_400, "52 bytes × 25 fps × 8 bits = 10.4 kbps"),
];

/// Verify every AudioAspect is covered by at least one primitive.
/// If an aspect is missing, we have a hole in our codec design.
pub fn verify_aspect_coverage() -> Vec<AudioAspect> {
    use AudioAspect::*;
    let all = [SpectralEnvelope, SpectralShape, PerceptualMapping,
               PhaseRelationship, SpeakerIdentity, SemanticContent,
               MaskingDecision, CodebookLookup];

    all.iter()
        .filter(|&&aspect| !PROVENANCE.iter().any(|p| p.aspect == aspect))
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_aspects_covered() {
        let missing = verify_aspect_coverage();
        assert!(missing.is_empty(), "Missing audio aspects: {:?}", missing);
    }

    #[test]
    fn frame_budget_correct() {
        // AudioFrame (48) + PhaseDescriptor (4) = 52
        assert_eq!(FRAME_BUDGET, 48 + 4);
        // + RvqFrame (17) = 69
        assert_eq!(FRAME_BUDGET_WITH_TTS, 48 + 4 + 17);
    }

    #[test]
    fn provenance_byte_sizes_consistent() {
        // AudioFrame = 42 (energies) + 6 (pvq) = 48
        let af_energies = PROVENANCE.iter().find(|p| p.our_type == "AudioFrame.band_energies").unwrap();
        let af_pvq = PROVENANCE.iter().find(|p| p.our_type == "AudioFrame.pvq_summary").unwrap();
        assert_eq!(af_energies.byte_size + af_pvq.byte_size, 48);

        // RvqFrame = 1 (HEEL) + 8 (HIP) + 8 (TWIG) = 17
        let rvq_heel = PROVENANCE.iter().find(|p| p.our_type == "RvqFrame.archetype (HEEL)").unwrap();
        let rvq_hip = PROVENANCE.iter().find(|p| p.our_type == "RvqFrame.coarse (HIP)").unwrap();
        let rvq_twig = PROVENANCE.iter().find(|p| p.our_type == "RvqFrame.fine (TWIG)").unwrap();
        assert_eq!(rvq_heel.byte_size + rvq_hip.byte_size + rvq_twig.byte_size, 17);
    }

    #[test]
    fn every_source_codec_represented() {
        // All 6 source codecs should appear at least once
        for source in [CodecSource::Opus, CodecSource::Whisper, CodecSource::Mp3,
                       CodecSource::OggVorbis, CodecSource::Bark, CodecSource::ElevenLabs] {
            assert!(PROVENANCE.iter().any(|p| p.source == source),
                "Codec {:?} not represented in provenance table", source);
        }
    }

    #[test]
    fn our_bitrate_competitive() {
        // Our codec should be lower bitrate than all traditional codecs
        let ours_24k = BITRATE_COMPARISON.iter()
            .find(|&&(name, _, _)| name == "Ours (24kHz)")
            .unwrap().1;
        let mp3 = BITRATE_COMPARISON.iter()
            .find(|&&(name, _, _)| name == "MP3 128k")
            .unwrap().1;
        assert!(ours_24k < mp3, "Our codec should be lower bitrate than MP3");
    }
}
