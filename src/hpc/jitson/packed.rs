//! PackedDatabase: stroke-aligned layout for streaming cascade search.
//!
//! Re-exports from [`crate::hpc::packed`] — the canonical implementation lives
//! there alongside other HPC modules. This re-export makes it accessible from
//! the `jitson` module namespace for users who import jitson as a unit.
//!
//! See [`crate::hpc::packed`] for full documentation.

pub use crate::hpc::packed::*;
