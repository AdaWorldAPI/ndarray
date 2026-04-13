//! Audio primitives transcoded from Opus CELT.
//!
//! MDCT, band energy extraction, PVQ, and AudioFrame for the
//! HHTL cascade → waveform synthesis pipeline.
//!
//! Zero external dependencies — uses `hpc::fft` internally.

pub mod mdct;
pub mod bands;
pub mod pvq;
pub mod codec;
