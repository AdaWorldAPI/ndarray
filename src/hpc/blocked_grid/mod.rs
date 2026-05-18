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
//!
//! # Zero-dimension compile-fail guard
//!
//! Instantiating `BlockedGrid` with a zero block dimension is a **compile-time
//! error** (the `const { assert!(BR > 0 && BC > 0, …) }` in `new_with_pad` fires
//! before any code is generated).
//!
//! ```compile_fail
//! use ndarray::hpc::blocked_grid::BlockedGrid;
//! // BR = 0 is rejected at compile time — this must not compile.
//! let _ = BlockedGrid::<u64, 0, 64>::new_with_pad(10, 10, 0);
//! ```
//!
//! # Canonical compose pattern
//!
//! The idiomatic way to derive a new grid from an existing one:
//!
//! ```
//! use ndarray::hpc::blocked_grid::{BlockedGrid, ShaderMantissaGrid};
//!
//! // Build a 128×128 ShaderMantissaGrid (= BlockedGrid<u64, 64, 64>).
//! // padded_rows() == padded_cols() == 128, so blocks_l1() yields 4 blocks (2×2).
//! let input: ShaderMantissaGrid = BlockedGrid::new(128, 128);
//! assert_eq!(input.blocks_l1().count(), 4); // 2 block-rows × 2 block-cols
//!
//! // map_l1: derive a transformed grid without mutating input (data-flow Rule #3).
//! let output = input.map_l1::<u64, _>(|inp, outp| {
//!     for r in 0..64 {
//!         let in_row  = inp.row(r);
//!         let out_row = outp.row_mut(r);
//!         for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
//!             *dst = src.wrapping_add(1);
//!         }
//!     }
//! });
//!
//! // Rule #3 demo: input is unchanged — all cells remain u64::default() == 0.
//! assert!(input.as_padded_slice().iter().all(|&v| v == 0u64));
//!
//! // Output: every cell is input (0) + 1 = 1.
//! assert!(output.as_padded_slice().iter().all(|&v| v == 1u64));
//!
//! // blocks_l1 iterates the 4 base blocks; blocks_l2 requires ≥256×256.
//! let coords: Vec<_> = output.blocks_l1()
//!     .map(|b| (b.block_row(), b.block_col()))
//!     .collect();
//! assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
//! ```

mod base;
mod iter;
mod super_block;
mod compute;
mod aliases;
#[cfg(test)]
mod tests;

pub use aliases::{
    AmxBf16Grid, AmxInt8Grid, HalfSquareU64, ShaderMantissaGrid, SquareF64Stack8, StripF32Stack2, StripF32Stack4,
};
pub use base::{BlockedGrid, GridBlock, GridBlockMut};
pub use iter::{BaseBlockIter, BaseBlockIterMut};
pub use super_block::{GridSuperBlock, GridSuperBlockMut, TierBlockIter, TierBlockIterMut};
// (compute has no re-exports — only adds impls on existing types)
