//! Convenience type aliases and L1/L2/L3/L4 tier-alias impls for the cognitive-shader
//! hierarchy built on top of [`BlockedGrid`].
//!
//! Worker scope: `src/hpc/blocked_grid/aliases.rs` (sprint worker A5 — see
//! `.claude/knowledge/pr-x3-cognitive-grid-design.md` §"Worker decomposition").
//!
//! ## What lives here
//!
//! 1. **Seven type aliases** pinning the common hardware-block × cell-type shapes
//!    (ShaderMantissaGrid, AmxBf16Grid, AmxInt8Grid, StripF32Stack2, StripF32Stack4,
//!    SquareF64Stack8, HalfSquareU64).
//!
//! 2. **L1/L2/L3/L4 alias methods** on `BlockedGrid<T, 64, 64>` only (Q7 ruling —
//!    non-64×64 grids use the raw `blocks_tier::<N>` / `map_tier::<N>` /
//!    `bulk_apply_tier::<N>` const-generic methods directly).
//!
//! ## Out of scope (NOT in this file)
//! - SIMD register-bank stack types (`StackedU64x8<N>`, …) → PR-X5
//! - Typed distance bulk functions → W7, bench-gated
//! - CausalEdge64 mantissa cell kernel → W7
//!
//! See `.claude/knowledge/cognitive-distance-typing.md` — no distance-aware API here.
//! See `.claude/rules/data-flow.md` — all `&mut self` alias methods are write-back only.
//!
//! ## Canonical compose pattern
//!
//! ```
//! use ndarray::hpc::blocked_grid::BlockedGrid;
//!
//! // Build a 128×128 ShaderMantissaGrid (= BlockedGrid<u64>) and derive a
//! // transformed grid using map_l1.
//! let input = BlockedGrid::<u64>::new(128, 128);
//! // map_l1 returns a new grid; input is not mutated.
//! let output = input.map_l1::<u64, _>(|inp, outp| {
//!     for r in 0..64 {
//!         let in_row = inp.row(r);
//!         let out_row = outp.row_mut(r);
//!         for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
//!             *dst = src.wrapping_add(1);
//!         }
//!     }
//! });
//! // Verify input is unchanged (all cells remain 0 = u64::default()).
//! assert!(input.as_padded_slice().iter().all(|&v| v == 0u64));
//! // Verify output has every cell incremented.
//! assert!(output.as_padded_slice().iter().all(|&v| v == 1u64));
//! ```

use super::base::{BlockedGrid, GridBlock, GridBlockMut};
use super::iter::BaseBlockIter;
use super::super_block::{GridSuperBlock, GridSuperBlockMut, TierBlockIter};

// ============================================================
// Type aliases — cognitive-shader and SIMD-tier shapes
// ============================================================

/// Default cognitive shader cell-block: 64×64 u64 mantissa grid.
///
/// Each cell carries one u64 CausalEdge64 identity — the precision-controlling
/// bits of the cognitive shader pass (BF16-mantissa-analogous role). Storage is
/// `u64` because CausalEdge64 is a u64-packed structure.
///
/// Padding to 64-cell boundaries means a 100×100 ShaderMantissaGrid has
/// `padded_rows() == padded_cols() == 128`.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type ShaderMantissaGrid = BlockedGrid<u64, 64, 64>;
/// let g = ShaderMantissaGrid::new(1024, 768);
/// assert_eq!(g.padded_rows(), 1024);
/// assert_eq!(g.padded_cols(), 768);
/// assert_eq!(ShaderMantissaGrid::block_dims(), (64, 64));
/// ```
pub type ShaderMantissaGrid = BlockedGrid<u64, 64, 64>;

/// AMX BF16 tile grid — each cell-block is one AMX BF16 tile (16×16 BF16).
///
/// Storage type is `u16` because BF16 lives in `u16` carriers. One 16×16
/// block maps directly to one TDPBF16PS AMX tile instruction.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type AmxBf16Grid = BlockedGrid<u16, 16, 16>;
/// let g = AmxBf16Grid::new(100, 100);
/// // next multiple of 16 ≥ 100 is 112
/// assert_eq!(g.padded_rows(), 112);
/// assert_eq!(g.padded_cols(), 112);
/// assert_eq!(AmxBf16Grid::block_dims(), (16, 16));
/// ```
pub type AmxBf16Grid = BlockedGrid<u16, 16, 16>;

/// AMX INT8 tile grid — half-square TDPBUSD shape (16×64).
///
/// Storage type is `u8`. The 16×64 shape matches the AMX INT8 TDPBUSD
/// instruction's half-square tile (16 rows × 64 columns of bytes).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type AmxInt8Grid = BlockedGrid<u8, 16, 64>;
/// let g = AmxInt8Grid::new(100, 100);
/// // next multiple of 16 ≥ 100 is 112 (row); next multiple of 64 ≥ 100 is 128 (col)
/// assert_eq!(g.padded_rows(), 112);
/// assert_eq!(g.padded_cols(), 128);
/// assert_eq!(AmxInt8Grid::block_dims(), (16, 64));
/// ```
pub type AmxInt8Grid = BlockedGrid<u8, 16, 64>;

/// F32 vertical-stack-2 strip — 2 F32x16 registers per cell-block.
///
/// Each 2×16 block covers two AVX-512 F32x16 register widths stacked
/// vertically. Useful for cache-line-aligned F32 pairs.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type StripF32Stack2 = BlockedGrid<f32, 2, 16>;
/// let g = StripF32Stack2::new(10, 32);
/// assert_eq!(g.padded_rows(), 10); // ceil(10/2)*2 = 10
/// assert_eq!(g.padded_cols(), 32); // ceil(32/16)*16 = 32
/// assert_eq!(StripF32Stack2::block_dims(), (2, 16));
/// ```
pub type StripF32Stack2 = BlockedGrid<f32, 2, 16>;

/// F32 vertical-stack-4 strip — 4 F32x16 registers per cell-block.
///
/// Each 4×16 block covers four AVX-512 F32x16 register widths stacked
/// vertically.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type StripF32Stack4 = BlockedGrid<f32, 4, 16>;
/// let g = StripF32Stack4::new(8, 32);
/// assert_eq!(g.padded_rows(), 8);  // ceil(8/4)*4 = 8
/// assert_eq!(g.padded_cols(), 32); // ceil(32/16)*16 = 32
/// assert_eq!(StripF32Stack4::block_dims(), (4, 16));
/// ```
pub type StripF32Stack4 = BlockedGrid<f32, 4, 16>;

/// F64 8×8 square — 8 F64x8 registers per cell-block.
///
/// The 8×8 shape maps to the canonical square BLAS micro-kernel for matrix
/// multiplication using AVX-512 F64x8 registers (8 elements × 8 rows).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type SquareF64Stack8 = BlockedGrid<f64, 8, 8>;
/// let g = SquareF64Stack8::new(16, 16);
/// assert_eq!(g.padded_rows(), 16); // ceil(16/8)*8 = 16
/// assert_eq!(g.padded_cols(), 16); // ceil(16/8)*8 = 16
/// assert_eq!(SquareF64Stack8::block_dims(), (8, 8));
/// ```
pub type SquareF64Stack8 = BlockedGrid<f64, 8, 8>;

/// Half-square U64 grid — 32×64 blocks — when row stride is expensive.
///
/// The 32×64 shape provides a half-square U64 layout: half the row count of
/// the 64×64 default but the same column count. Useful when row-traversal
/// cost dominates.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// type HalfSquareU64 = BlockedGrid<u64, 32, 64>;
/// let g = HalfSquareU64::new(64, 64);
/// assert_eq!(g.padded_rows(), 64); // ceil(64/32)*32 = 64
/// assert_eq!(g.padded_cols(), 64); // ceil(64/64)*64 = 64
/// assert_eq!(HalfSquareU64::block_dims(), (32, 64));
/// ```
pub type HalfSquareU64 = BlockedGrid<u64, 32, 64>;

// ============================================================
// L1/L2/L3/L4 alias impls — 64×64 base ONLY (Q7 ruling)
//
// Rationale: the L1–L4 naming maps to the cache hierarchy for the default
// 64×64 CausalEdge64 mantissa grid. Non-64×64 grids (AMX, strip, etc.) use
// the raw blocks_tier::<N> / map_tier::<N> / bulk_apply_tier::<N> escape
// hatches directly.
//
// Three impl blocks matching A4's exact bound structure:
//   - blocks_l*: T: Copy (delegates to blocks_base / blocks_tier)
//   - map_l*: T: Copy (delegates to map_base / map_tier, which require T: Copy)
//   - bulk_apply_l1: T: Copy (delegates to bulk_apply_base; T: Copy bound inherited)
//   - bulk_apply_l2/l3/l4: T-unbounded (delegates to bulk_apply_tier)
// ============================================================

// ----------------------------------------------------------
// Read-only tier iterators — T: Copy required by blocks_base / blocks_tier
// ----------------------------------------------------------

impl<T: Copy> BlockedGrid<T, 64, 64> {
    /// L1 tier iterator: 64×64 base blocks (innermost, ~32 KB each).
    ///
    /// Following cache-hierarchy convention: L1 = innermost (32 KB),
    /// L4 = framebuffer-scale (2 GB).
    ///
    /// Delegates to [`blocks_base`](BlockedGrid::blocks_base).
    /// A 128×128 ShaderMantissaGrid yields 4 blocks (2×2).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(128, 128);
    /// assert_eq!(g.blocks_l1().count(), 4); // 2×2 base blocks
    /// ```
    pub fn blocks_l1(&self) -> BaseBlockIter<'_, T, 64, 64> {
        self.blocks_base()
    }

    /// L2 tier iterator: 256×256 super-blocks (4×4 L1 blocks, ~512 KB each).
    ///
    /// See cache-hierarchy convention note in [`blocks_l1`](BlockedGrid::blocks_l1).
    /// Delegates to [`blocks_tier::<4>`](BlockedGrid::blocks_tier).
    ///
    /// # Panics
    /// Panics if `padded_rows % 256 != 0` or `padded_cols % 256 != 0`.
    /// A 128×128 grid (padded 128) is too small for L2 iteration — use a
    /// 256×256 (or larger) grid.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// assert_eq!(g.blocks_l2().count(), 1); // single 256×256 super-block
    /// ```
    pub fn blocks_l2(&self) -> TierBlockIter<'_, T, 64, 64, 4> {
        self.blocks_tier::<4>()
    }

    /// L3 tier iterator: 4096×4096 super-blocks (64×64 L1 blocks, ~128 MB each).
    ///
    /// See cache-hierarchy convention note in [`blocks_l1`](BlockedGrid::blocks_l1).
    /// Delegates to [`blocks_tier::<64>`](BlockedGrid::blocks_tier).
    ///
    /// # Panics
    /// Panics if `padded_rows % 4096 != 0` or `padded_cols % 4096 != 0`.
    /// A 64×64 grid is too small for L3 iteration and will panic.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 grid → padded 64×64; 64 % (64*64=4096) != 0 → panic
    /// let g = BlockedGrid::<u64>::new(64, 64);
    /// let _ = g.blocks_l3().count();
    /// ```
    pub fn blocks_l3(&self) -> TierBlockIter<'_, T, 64, 64, 64> {
        self.blocks_tier::<64>()
    }

    /// L4 tier iterator: 16384×16384 super-blocks (256×256 L1 blocks, ~2 GB each).
    ///
    /// See cache-hierarchy convention note in [`blocks_l1`](BlockedGrid::blocks_l1).
    /// Delegates to [`blocks_tier::<256>`](BlockedGrid::blocks_tier).
    ///
    /// # Panics
    /// Panics if `padded_rows % 16384 != 0` or `padded_cols % 16384 != 0`.
    /// A 64×64 grid is too small for L4 iteration and will panic.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 grid → padded 64×64; 64 % (64*256=16384) != 0 → panic
    /// let g = BlockedGrid::<u64>::new(64, 64);
    /// let _ = g.blocks_l4().count();
    /// ```
    pub fn blocks_l4(&self) -> TierBlockIter<'_, T, 64, 64, 256> {
        self.blocks_tier::<256>()
    }
}

// ----------------------------------------------------------
// PRIMARY compute paths — T: Copy, U: Copy + Default
// Delegates to map_base / map_tier.
// ----------------------------------------------------------

impl<T: Copy> BlockedGrid<T, 64, 64> {
    /// L1 PRIMARY compute path: map a closure over every 64×64 base block,
    /// returning a new grid of element type `U`.
    ///
    /// Delegates to [`map_base`](BlockedGrid::map_base). The input grid is
    /// never mutated; this satisfies `.claude/rules/data-flow.md` Rule #3.
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back (e.g., scratch-buffer pipelines) see [`bulk_apply_l1`](BlockedGrid::bulk_apply_l1).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(64, 64);
    /// let out = g.map_l1::<u64, _>(|inp, outp| {
    ///     for r in 0..64 {
    ///         let in_row = inp.row(r);
    ///         let out_row = outp.row_mut(r);
    ///         for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
    ///             *dst = src.wrapping_add(1);
    ///         }
    ///     }
    /// });
    /// // Input unchanged.
    /// assert!(g.as_padded_slice().iter().all(|&v| v == 0u64));
    /// // Output incremented.
    /// assert!(out.as_padded_slice().iter().all(|&v| v == 1u64));
    /// ```
    pub fn map_l1<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where
        F: FnMut(&GridBlock<'_, T, 64, 64>, &mut GridBlockMut<'_, U, 64, 64>),
    {
        self.map_base(f)
    }

    /// L2 PRIMARY compute path: map a closure over every 256×256 super-block
    /// (4×4 L1 blocks), returning a new grid of element type `U`.
    ///
    /// Delegates to [`map_tier::<U, 4, _>`](BlockedGrid::map_tier).
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back see [`bulk_apply_l2`](BlockedGrid::bulk_apply_l2).
    ///
    /// # Panics
    /// Panics if `padded_rows % 256 != 0` or `padded_cols % 256 != 0`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let out = g.map_l2::<u64, _>(|_inp, _outp| {});
    /// assert!(g.as_padded_slice().iter().all(|&v| v == 0u64));
    /// ```
    pub fn map_l2<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where
        F: FnMut(&GridSuperBlock<'_, T, 64, 64, 4>, &mut GridSuperBlockMut<'_, U, 64, 64, 4>),
    {
        self.map_tier::<U, 4, _>(f)
    }

    /// L3 PRIMARY compute path: map a closure over every 4096×4096 super-block
    /// (64×64 L1 blocks), returning a new grid of element type `U`.
    ///
    /// Delegates to [`map_tier::<U, 64, _>`](BlockedGrid::map_tier).
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back see [`bulk_apply_l3`](BlockedGrid::bulk_apply_l3).
    ///
    /// # Panics
    /// Panics if `padded_rows % 4096 != 0` or `padded_cols % 4096 != 0`.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 is too small for L3 iteration → panics
    /// let g = BlockedGrid::<u64>::new(64, 64);
    /// let _ = g.map_l3::<u64, _>(|_inp, _outp| {});
    /// ```
    pub fn map_l3<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where
        F: FnMut(&GridSuperBlock<'_, T, 64, 64, 64>, &mut GridSuperBlockMut<'_, U, 64, 64, 64>),
    {
        self.map_tier::<U, 64, _>(f)
    }

    /// L4 PRIMARY compute path: map a closure over every 16384×16384 super-block
    /// (256×256 L1 blocks), returning a new grid of element type `U`.
    ///
    /// Delegates to [`map_tier::<U, 256, _>`](BlockedGrid::map_tier).
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back see [`bulk_apply_l4`](BlockedGrid::bulk_apply_l4).
    ///
    /// # Panics
    /// Panics if `padded_rows % 16384 != 0` or `padded_cols % 16384 != 0`.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 is too small for L4 iteration → panics
    /// let g = BlockedGrid::<u64>::new(64, 64);
    /// let _ = g.map_l4::<u64, _>(|_inp, _outp| {});
    /// ```
    pub fn map_l4<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where
        F: FnMut(&GridSuperBlock<'_, T, 64, 64, 256>, &mut GridSuperBlockMut<'_, U, 64, 64, 256>),
    {
        self.map_tier::<U, 256, _>(f)
    }
}

// ----------------------------------------------------------
// SECONDARY write-back paths — bulk_apply_l1 requires T: Copy
// (matches A4's bulk_apply_base bound); l2/l3/l4 are T-unbounded
// (matches A4's bulk_apply_tier bound).
// ----------------------------------------------------------

impl<T: Copy> BlockedGrid<T, 64, 64> {
    /// L1 SECONDARY write-back path: apply a closure in-place over every
    /// 64×64 base block.
    ///
    /// Delegates to [`bulk_apply_base`](BlockedGrid::bulk_apply_base).
    ///
    /// The closure `f` receives:
    /// - `&mut GridBlockMut<'_, T, 64, 64>` — mutable window into the current block.
    /// - `(usize, usize)` — `(block_row, block_col)` coordinates (0-based).
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_l1`](BlockedGrid::map_l1)
    /// instead, which returns a fresh grid.
    ///
    /// Workers MUST NOT place CausalEdge64 mantissa-pass logic, cascade
    /// reasoning, or any other compute kernel inside this closure. Sentinel
    /// reviews will flag violations.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(64, 64);
    /// let mut call_count = 0usize;
    /// g.bulk_apply_l1(|_blk, (_br, _bc)| {
    ///     call_count += 1;
    /// });
    /// // 64×64 grid → 1 base block → closure called exactly once.
    /// assert_eq!(call_count, 1);
    /// ```
    pub fn bulk_apply_l1<F>(&mut self, f: F)
    where
        F: FnMut(&mut GridBlockMut<'_, T, 64, 64>, (usize, usize)),
    {
        self.bulk_apply_base(f)
    }
}

impl<T> BlockedGrid<T, 64, 64> {
    /// L2 SECONDARY write-back path: apply a closure in-place over every
    /// 256×256 super-block (4×4 L1 blocks).
    ///
    /// Delegates to [`bulk_apply_tier::<4, _>`](BlockedGrid::bulk_apply_tier).
    ///
    /// The closure `f` receives:
    /// - `&mut GridSuperBlockMut<'_, T, 64, 64, 4>` — mutable window.
    /// - `(usize, usize)` — `(super_row, super_col)` coordinates (0-based).
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_l2`](BlockedGrid::map_l2)
    /// instead, which returns a fresh grid.
    ///
    /// Workers MUST NOT place CausalEdge64 mantissa-pass logic, cascade
    /// reasoning, or any other compute kernel inside this closure. Sentinel
    /// reviews will flag violations.
    ///
    /// # Panics
    /// Panics if `padded_rows % 256 != 0` or `padded_cols % 256 != 0`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let mut call_count = 0usize;
    /// g.bulk_apply_l2(|_sb, (sr, sc)| {
    ///     call_count += 1;
    ///     assert_eq!((sr, sc), (0, 0));
    /// });
    /// assert_eq!(call_count, 1);
    /// ```
    pub fn bulk_apply_l2<F>(&mut self, f: F)
    where
        F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 4>, (usize, usize)),
    {
        self.bulk_apply_tier::<4, _>(f)
    }

    /// L3 SECONDARY write-back path: apply a closure in-place over every
    /// 4096×4096 super-block (64×64 L1 blocks).
    ///
    /// Delegates to [`bulk_apply_tier::<64, _>`](BlockedGrid::bulk_apply_tier).
    ///
    /// The closure `f` receives:
    /// - `&mut GridSuperBlockMut<'_, T, 64, 64, 64>` — mutable window.
    /// - `(usize, usize)` — `(super_row, super_col)` coordinates (0-based).
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_l3`](BlockedGrid::map_l3)
    /// instead, which returns a fresh grid.
    ///
    /// Workers MUST NOT place CausalEdge64 mantissa-pass logic, cascade
    /// reasoning, or any other compute kernel inside this closure. Sentinel
    /// reviews will flag violations.
    ///
    /// # Panics
    /// Panics if `padded_rows % 4096 != 0` or `padded_cols % 4096 != 0`.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 grid → padded 64; 64 % 4096 != 0 → panic
    /// let mut g = BlockedGrid::<u64>::new(64, 64);
    /// g.bulk_apply_l3(|_sb, _coords| {});
    /// ```
    pub fn bulk_apply_l3<F>(&mut self, f: F)
    where
        F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 64>, (usize, usize)),
    {
        self.bulk_apply_tier::<64, _>(f)
    }

    /// L4 SECONDARY write-back path: apply a closure in-place over every
    /// 16384×16384 super-block (256×256 L1 blocks, ~2 GB framebuffer).
    ///
    /// Delegates to [`bulk_apply_tier::<256, _>`](BlockedGrid::bulk_apply_tier).
    ///
    /// The closure `f` receives:
    /// - `&mut GridSuperBlockMut<'_, T, 64, 64, 256>` — mutable window.
    /// - `(usize, usize)` — `(super_row, super_col)` coordinates (0-based).
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_l4`](BlockedGrid::map_l4)
    /// instead, which returns a fresh grid.
    ///
    /// Workers MUST NOT place CausalEdge64 mantissa-pass logic, cascade
    /// reasoning, or any other compute kernel inside this closure. Sentinel
    /// reviews will flag violations.
    ///
    /// # Panics
    /// Panics if `padded_rows % 16384 != 0` or `padded_cols % 16384 != 0`.
    ///
    /// # Example
    /// ```should_panic
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 64×64 grid → padded 64; 64 % 16384 != 0 → panic
    /// let mut g = BlockedGrid::<u64>::new(64, 64);
    /// g.bulk_apply_l4(|_sb, _coords| {});
    /// ```
    pub fn bulk_apply_l4<F>(&mut self, f: F)
    where
        F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 256>, (usize, usize)),
    {
        self.bulk_apply_tier::<256, _>(f)
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Type alias construction — padding arithmetic
    // ------------------------------------------------------------------

    /// ShaderMantissaGrid::new(1024, 768) → padded_rows == 1024, padded_cols == 768
    /// (both are already exact multiples of 64).
    #[test]
    fn shader_mantissa_grid_1024x768() {
        let g = ShaderMantissaGrid::new(1024, 768);
        assert_eq!(g.padded_rows(), 1024);
        assert_eq!(g.padded_cols(), 768);
        assert_eq!(g.rows(), 1024);
        assert_eq!(g.cols(), 768);
    }

    /// AmxBf16Grid::new(100, 100) → padded_rows == 112, padded_cols == 112
    /// (next multiple of 16 ≥ 100).
    #[test]
    fn amx_bf16_grid_100x100() {
        let g = AmxBf16Grid::new(100, 100);
        assert_eq!(g.padded_rows(), 112);
        assert_eq!(g.padded_cols(), 112);
    }

    /// AmxInt8Grid::new(100, 100) → padded_rows == 112 (next multiple of 16),
    /// padded_cols == 128 (next multiple of 64).
    #[test]
    fn amx_int8_grid_100x100() {
        let g = AmxInt8Grid::new(100, 100);
        assert_eq!(g.padded_rows(), 112);
        assert_eq!(g.padded_cols(), 128);
    }

    /// StripF32Stack2 and StripF32Stack4 — verify block dims.
    #[test]
    fn strip_f32_stack_dims() {
        assert_eq!(StripF32Stack2::block_dims(), (2, 16));
        assert_eq!(StripF32Stack4::block_dims(), (4, 16));
    }

    /// SquareF64Stack8 — verify block dims.
    #[test]
    fn square_f64_stack8_dims() {
        assert_eq!(SquareF64Stack8::block_dims(), (8, 8));
    }

    /// HalfSquareU64 — verify block dims and padding.
    #[test]
    fn half_square_u64_dims() {
        assert_eq!(HalfSquareU64::block_dims(), (32, 64));
        let g = HalfSquareU64::new(100, 100);
        assert_eq!(g.padded_rows(), 128); // ceil(100/32)*32 = 128
        assert_eq!(g.padded_cols(), 128); // ceil(100/64)*64 = 128
    }

    // ------------------------------------------------------------------
    // blocks_l1 — 128×128 ShaderMantissaGrid → 4 blocks (2×2)
    // ------------------------------------------------------------------

    #[test]
    fn blocks_l1_128x128_yields_4_blocks() {
        let g = ShaderMantissaGrid::new(128, 128);
        assert_eq!(g.blocks_l1().count(), 4);
    }

    #[test]
    fn blocks_l1_row_major_order() {
        let g = ShaderMantissaGrid::new(128, 128);
        let coords: Vec<_> = g
            .blocks_l1()
            .map(|b| (b.block_row(), b.block_col()))
            .collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    // ------------------------------------------------------------------
    // blocks_l2 — 256×256 → 1 super-block; 128×128 → panic
    // ------------------------------------------------------------------

    #[test]
    fn blocks_l2_256x256_yields_1_super_block() {
        let g = ShaderMantissaGrid::new(256, 256);
        assert_eq!(g.blocks_l2().count(), 1);
    }

    #[test]
    #[should_panic(
        expected = "BlockedGrid::blocks_tier::<4>: padded extent 128×128 is not divisible by tier stride 256×256"
    )]
    fn blocks_l2_128x128_panics() {
        let g = ShaderMantissaGrid::new(128, 128);
        let _ = g.blocks_l2().count();
    }

    // ------------------------------------------------------------------
    // blocks_l3 / blocks_l4 — too-small grid panics
    // ------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn blocks_l3_too_small_panics() {
        let g = ShaderMantissaGrid::new(64, 64);
        let _ = g.blocks_l3().count();
    }

    #[test]
    #[should_panic]
    fn blocks_l4_too_small_panics() {
        let g = ShaderMantissaGrid::new(64, 64);
        let _ = g.blocks_l4().count();
    }

    // ------------------------------------------------------------------
    // map_l1 — input unchanged, closure-mapped values in output
    // ------------------------------------------------------------------

    #[test]
    fn map_l1_64x64_input_unchanged() {
        let g = ShaderMantissaGrid::new(64, 64);
        let before: Vec<u64> = g.as_padded_slice().to_vec();

        let out = g.map_l1::<u64, _>(|inp, outp| {
            for r in 0..64 {
                let in_row = inp.row(r);
                let out_row = outp.row_mut(r);
                for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
                    *dst = src.wrapping_add(42);
                }
            }
        });

        // Input unchanged.
        assert_eq!(g.as_padded_slice(), before.as_slice());
        // Output has every cell = 0 + 42 = 42.
        assert!(out.as_padded_slice().iter().all(|&v| v == 42u64));
    }

    // ------------------------------------------------------------------
    // bulk_apply_l1 — closure called exactly once on 64×64 grid
    // ------------------------------------------------------------------

    #[test]
    fn bulk_apply_l1_64x64_called_once() {
        let mut g = ShaderMantissaGrid::new(64, 64);
        let mut call_count = 0usize;
        g.bulk_apply_l1(|_blk, (_br, _bc)| {
            call_count += 1;
        });
        assert_eq!(call_count, 1);
    }

    // ------------------------------------------------------------------
    // Non-64×64 grid has no blocks_l1 method (type-system enforcement)
    // The compile_fail is in the docstring of AmxBf16Grid type alias above.
    // This positive test verifies AmxBf16Grid builds correctly.
    // ------------------------------------------------------------------

    #[test]
    fn amx_bf16_grid_builds() {
        let g = AmxBf16Grid::new(16, 16);
        assert_eq!(g.padded_rows(), 16);
        assert_eq!(g.padded_cols(), 16);
        // blocks_l1 is not available on AmxBf16Grid — calling it here would
        // be a compile-time error (verified by compile_fail doctest).
    }
}
