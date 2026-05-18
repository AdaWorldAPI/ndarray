//! Generic 2-D block-padded grid for the cognitive shader stack.
//!
//! `BlockedGrid<T, BR, BC>` pads storage to (BR, BC) base-block boundaries
//! and exposes both logical-cell accessors and flat padded-storage slices.
//! Higher tiers (L2 / L3 / L4) are expressed as iteration patterns over the
//! padded base storage — not as extra padding — and are implemented across
//! submodules below.
//!
//! No SIMD primitives, no `#[target_feature]`, no distance metrics. See PR-X3 design doc.
//!
//! Submodule ownership (one file per sprint worker — see
//! `.claude/knowledge/pr-x3-cognitive-grid-design.md` §"Worker decomposition"):
//! - `base`         (worker A1) — `BlockedGrid`, `GridBlock`, `GridBlockMut`, accessors
//! - `iter`         (worker A2) — `BaseBlockIter`, `BaseBlockIterMut`, `blocks_base*`
//! - `super_block`  (worker A3) — `GridSuperBlock`, `GridSuperBlockMut`, `TierBlockIter`, `blocks_tier`
//! - `compute`      (worker A4) — `map_base`, `map_tier`, `bulk_apply_base`, `bulk_apply_tier`
//! - `aliases`      (worker A5) — `ShaderMantissaGrid`, `AmxBf16Grid`, … and L1-L4 alias impls
//! - `grid_struct_macro` (worker B) — `blocked_grid_struct!` SoA-of-grids macro

mod base;
mod iter;
mod super_block;
mod compute;
mod aliases;

pub use aliases::{
    AmxBf16Grid, AmxInt8Grid, HalfSquareU64, ShaderMantissaGrid, SquareF64Stack8, StripF32Stack2, StripF32Stack4,
};
pub use base::{BlockedGrid, GridBlock, GridBlockMut};
pub use iter::{BaseBlockIter, BaseBlockIterMut};
pub use super_block::{GridSuperBlock, GridSuperBlockMut, TierBlockIter, TierBlockIterMut};
// (compute has no re-exports — only adds impls on existing types)
