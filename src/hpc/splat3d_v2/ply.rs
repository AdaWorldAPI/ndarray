//! Minimal PLY reader v2 for the 3DGS canonical scene format.
//!
//! Placeholder module re-exporting v1 symbols.  The v2 PLY reader will
//! extend the Inria format loader to populate `SplatCell<D>` fields
//! (INT4 thinking-style, qualia) once A3/A4 workers define those types.
//!
//! Per A1 brief § "File moves": v1 `splat3d/` does not list `ply.rs`
//! explicitly in the moves table (the table covers tile, frame, gaussian,
//! project, raster, sh, spd3, mod), but `ply.rs` is a peer in v1's
//! `mod.rs` and is included here for completeness as a passthrough stub.
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//!   § "File moves"
//! - Inria PLY format: `splat3d/ply.rs` header (Kerbl 2023 §3.2)
//! - `SplatCell<D>` extension: A3/A4 workers (G1/G2 deliverables)

//// TODO(A3/A4): extend read_ply_v2 to populate SplatCell<D> INT4 fields
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§G2 INT4×N packed dot

pub use crate::hpc::splat3d::{read_ply, PlyError};
