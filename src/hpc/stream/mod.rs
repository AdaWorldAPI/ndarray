//! Vertical streaming structs for the SoA columns.
//! Per cognitive-substrate-convergence-v1.md §5 L-20.
//!
//! Sprint-12 scope (W-F4/5/6): `QualiaStream` + `InferenceStream` +
//! `SplatFieldStream` forward-iterator scaffolds. Sprint-13 (D-CSV-17):
//! `par_*` rayon variants wired behind `#[cfg(feature = "rayon")]`.

pub mod inference;
pub mod qualia;
pub mod splat_field;

pub use inference::{InferenceRow, InferenceStream};
pub use qualia::{QualiaI4Row, QualiaStream};
pub use splat_field::{SplatField, SplatFieldStream};

#[cfg(feature = "rayon")]
pub use inference::par_inference_stream;
#[cfg(feature = "rayon")]
pub use qualia::par_qualia_stream;
#[cfg(feature = "rayon")]
pub use splat_field::par_splat_field_stream;
