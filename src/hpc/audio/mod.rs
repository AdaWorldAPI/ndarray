//! Audio primitives transcoded from Opus CELT, Whisper, and Bark.
//!
//! Steals the best ideas from each:
//!   Opus  — MDCT + PVQ gain-shape split + CELT critical bands
//!   Whisper — 80-channel mel filterbank (perceptual frequency mapping)
//!   Bark  — 3-stage RVQ hierarchy (semantic→coarse→fine → HHTL levels)
//!   ElevenLabs — voice cloning as archetype embedding (16 i8 channels)
//!
//! AudioFrame (48 bytes) from Opus is the storage format.
//! Mel spectrogram from Whisper is the recognition format.
//! VoiceArchetype (16 bytes) from Bark/ElevenLabs is the speaker identity.
//! RvqFrame (17 bytes) is the compressed TTS output.
//!
//! Zero external dependencies — uses `hpc::fft` internally.

pub mod mdct;
pub mod bands;
pub mod pvq;
pub mod codec;
pub mod mel;
pub mod voice;
pub mod modes;
pub mod phase;
