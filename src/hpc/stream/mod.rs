//! Vertical streaming structs for the SoA columns.
//! Per cognitive-substrate-convergence-v1.md §5 L-20.
//!
//! Sprint-12 scope (W-F4/5/6): `QualiaStream` + `InferenceStream` +
//! `SplatFieldStream` forward-iterator scaffolds. Sprint-13+:
//! `par_*` rayon variants once rayon is wired into the ndarray
//! feature gate.

pub mod inference;
pub mod qualia;
pub mod splat_field;

pub use inference::{InferenceRow, InferenceStream};
pub use qualia::{QualiaI4Row, QualiaStream};
pub use splat_field::{SplatField, SplatFieldStream};
