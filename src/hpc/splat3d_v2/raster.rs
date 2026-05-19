//! Per-tile rasterizer v2 — depth-sorted alpha-blend with `F32x16` pixel rows.
//!
//! Placeholder module re-exporting v1 symbols.  The v2 rasterizer will add:
//! - NARS truth-revision composition path (G3, A5 deliverable) behind
//!   the `splat4d-nars-compose` feature flag (off by default per Forbidden
//!   Constraint #5 of the master prompt).
//! - B-Compose bundle swap: `hreduce_sum_f32x16` (alpha) ↔
//!   `revise_truth_f32x16` (NARS), same lane width, same latency class.
//! - `T_SATURATION_EPS` parameterised so the same outer loop serves both
//!   interpretations (done-criteria #5 from master prompt).
//!
//! Per A1 brief § "File moves": `splat3d/raster.rs` → `splat3d_v2/raster.rs`.
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - G3 NARS truth-revision kernel: A5 worker
//! - G4 fast_exp_x16 precision audit: A5 worker
//! - B-Compose: A5 bundle (hreduce_sum_f32x16 / revise_truth_f32x16)
//! - Forbidden Constraint #5: no NARS activation by default (W7 closure swap)

//// TODO(A5): add rasterize_tile_v2 with B-Compose closure-swappable path
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§Forbidden constraints #5
//// TODO(A5): gate NARS path behind splat4d-nars-compose feature flag
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§Done criteria #5

pub use crate::hpc::splat3d::{rasterize_frame, rasterize_tile, T_SATURATION_EPS};
