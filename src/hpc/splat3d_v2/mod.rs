//! CPU-SIMD 3D Gaussian Splatting v2 — typed cascade on
//! `BlockedGrid<SplatBinList, 1, 1>` (PR-X3 substrate).
//!
//! # Relation to siblings
//!
//! - **`crate::hpc::splat3d`** — v1 pipeline (bespoke 16×16 binner,
//!   hand-rolled CSR). **Kept unchanged until W7 closure swap.** Both
//!   v1 and v2 compile and pass tests simultaneously per the side-by-side
//!   decision (`pr-x4-design.md` § Q1).
//! - **`crate::hpc::splat4d`** — the 4-tier cognitive cascade that consumes
//!   `splat3d_v2` as its L1 projection substrate. See `splat4d::cascade`
//!   for `CascadeAddr` and the full tier scheme.
//!
//! # Migration status
//!
//! PR-X4 A1 (W4-W5). Workers A2-A6 spawn after A1 lands. This module
//! declares the v2 tree; per-module implementations land in their own
//! sprint workers.
//!
//! # SIMD bundle consumption (A1 scope)
//!
//! Per A1 brief § "SIMD bundles — B-Splat + B-Interleave-Transpose":
//! - **B-Splat**: `splat_f32x16`, `splat_i32x16` — broadcast a Gaussian
//!   centre across 16 tile lanes during the bin step.
//! - **B-Interleave-Transpose**: `interleave_f32x16 ∘ transpose_inplace`
//!   — row-major splat3d ↔ lane-major splat4d boundary primitive.
//!
//! PR-X4 must NOT reach past these bundles into raw `std::arch::*`
//! intrinsics — breaks SG2 p95 ≤ 20 ms. Missing bundles are proposed
//! against `vertical-simd-consumer-contract.md` before use.
//!
//! # Feature flags
//!
//! The module is exported from `lib.rs` under `#[cfg(feature = "splat3d-v2")]`
//! (added in the thematic commit after all A1-A6 workers complete, not here).
//! `splat4d-nars-compose` (default off) gates the NARS-revision composition
//! path; alpha-compositing is always the default per Forbidden Constraint #5.
//!
//! # Cross-references
//!
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - Master prompt: `.claude/knowledge/hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md`
//! - PR-X3 substrate: `.claude/knowledge/pr-x3-cognitive-grid-design.md`
//! - PR-X10 A12b gate: `linalg::hilbert::hilbert3d_{en,de}code` — L4 Hilbert-3D
//!   bug must be fixed before `CascadeAddr` worker (A2) may spawn.

//// TODO(A1): remove this allow once all submodules have real implementations
//// — see 01-a1-tileinstance-v2-brief.md:§Exit criteria
#![allow(dead_code)]

// ── Sub-modules ──────────────────────────────────────────────────────────────

pub mod tile;
pub mod frame;
pub mod gaussian;
pub mod project;
pub mod raster;
pub mod sh;
pub mod spd3;
pub mod ply;

// ── Re-exports from tile (the only A1-owned public surface) ──────────────────

pub use tile::{SplatBinList, SplatBinnerV2, TileInstance, TILE_SIZE_V2};

// ── Passthrough re-exports from v1 for modules not yet refactored ─────────────
//
// These re-exports let consumers import from `splat3d_v2` for all symbols;
// the per-module refactor workers (A2-A6) will replace each `pub use …::*`
// with v2-native exports one module at a time.

pub use crate::hpc::splat3d::{
    Gaussian3D, GaussianBatch, PlyError,
    SH_COEFFS_PER_CHANNEL, SH_COEFFS_PER_GAUSSIAN, SH_DEGREE,
    Camera, ProjectedBatch,
    T_SATURATION_EPS,
    SH_BASIS_PER_CHANNEL,
    Spd3,
};
