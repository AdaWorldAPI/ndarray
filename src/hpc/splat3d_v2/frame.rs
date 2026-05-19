//! `SplatFrameV2` / `SplatRendererV2` — v2 double-buffered driver.
//!
//! Placeholder module re-exporting v1 symbols until the per-module v2
//! refactor lands.  When the v2 frame pipeline is implemented (A6 railway
//! smoke deployment) this module will be replaced with a v2-native
//! `SplatFrameV2` that owns a `SplatBinnerV2` instead of v1's `TileBinning`.
//!
//! Per A1 brief § "File moves": `splat3d/frame.rs` → `splat3d_v2/frame.rs`.
//! The full implementation is A6's deliverable (`frame_pipeline` wired to
//! HLS + axum + Prom metrics).
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - Master prompt § "Worker spawn shape": A6 deps on A1 + A5 only.

//// TODO(A6): replace pub-use passthrough with v2-native SplatFrameV2 and
//// SplatRendererV2 owning BlockedGrid<SplatBinList,1,1>
//// — see 01-a1-tileinstance-v2-brief.md:§File moves

pub use crate::hpc::splat3d::{SplatFrame, SplatRenderer};
