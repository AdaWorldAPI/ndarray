//! Compute and write-back methods for [`BlockedGrid`].
//!
//! Worker scope: `src/hpc/blocked_grid/compute.rs` (sprint worker A4 — see
//! `.claude/knowledge/pr-x3-cognitive-grid-design.md` §"Worker decomposition").
//!
//! Provides two impl blocks on [`BlockedGrid<T, BR, BC>`]:
//!
//! - **PRIMARY compute paths** (`map_base`, `map_tier`): immutable `self`, returns a
//!   fresh [`BlockedGrid`]. Satisfies `.claude/rules/data-flow.md` Rule #3
//!   ("No `&mut self` during computation. Ever.").
//! - **SECONDARY write-back paths** (`bulk_apply_base`, `bulk_apply_tier`): `&mut self`,
//!   gated write-back only. Every `&mut self` method carries the mandatory
//!   `# Data-flow rule` docstring section citing the rule path verbatim.
//!
//! ## Out of scope (NOT in this file)
//! - SIMD register-bank stack types (`StackedU64x8<N>`, …) → PR-X5
//! - Typed distance bulk functions → W7, bench-gated
//! - CausalEdge64 mantissa cell kernel → W7
//!
//! See `.claude/knowledge/cognitive-distance-typing.md` — no distance API here.
//! See `.claude/rules/data-flow.md` — all `&mut self` paths are write-back only.

use super::base::{BlockedGrid, GridBlock, GridBlockMut};
use super::super_block::{GridSuperBlock, GridSuperBlockMut};

// ============================================================
// GridBlock row / rows accessors
//
// `row()` and `rows()` are specified in the PR-X3 design doc for `GridBlock`
// but were not included in A1's base.rs. Added here in the sibling `compute`
// module using the `data_slice()` / `padded_cols_stride()` helpers that A4
// added to base.rs (the minimal touch needed; justified in the commit message).
// ============================================================

impl<'a, T, const BR: usize, const BC: usize> GridBlock<'a, T, BR, BC> {
    /// Borrow row `r` of this block as a contiguous `&[T]` of length `BC`.
    ///
    /// Block storage uses the parent grid's stride (`padded_cols`), so each
    /// row within the block is contiguous in memory.
    ///
    /// # Panics
    /// Panics in debug builds if `r >= BR`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// g.set(0, 2, 42);
    /// let blk = GridBlock::from_grid(&g, 0, 0);
    /// assert_eq!(blk.row(0)[2], 42);
    /// assert_eq!(blk.row(0).len(), 4);
    /// ```
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < BR, "row {} out of block range {}", r, BR);
        let stride = self.padded_cols_stride();
        let start = r * stride;
        &self.data_slice()[start..start + BC]
    }

    /// Iterator over the `BR` rows of this block, each as `&[T]` of length `BC`.
    ///
    /// Yields rows in ascending order (0 first, `BR - 1` last).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// let blk = GridBlock::from_grid(&g, 0, 0);
    /// assert_eq!(blk.rows().count(), 4);
    /// for row in blk.rows() {
    ///     assert_eq!(row.len(), 4);
    /// }
    /// ```
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        (0..BR).map(move |r| self.row(r))
    }
}

// ============================================================
// PRIMARY compute paths — immutable self, returns a new grid
//
// These satisfy `.claude/rules/data-flow.md` Rule #3:
// "No `&mut self` during computation. Ever."
// The closure receives a read-only view of the input block and a mutable
// view of the OUTPUT block in the freshly-allocated result grid, so the
// input is never mutated.
// ============================================================

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Map a closure over every base block, producing a new grid with element
    /// type `U`.
    ///
    /// The closure `f` receives:
    /// - `&GridBlock<'_, T, BR, BC>` — read-only window into the input block.
    /// - `&mut GridBlockMut<'_, U, BR, BC>` — mutable window into the
    ///   corresponding block of the freshly-allocated output grid.
    ///
    /// The input grid is **never mutated**; `map_base` satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant.
    ///
    /// The output grid is initialised to `U::default()` before the closure
    /// runs, so any cell not written by the closure retains the default value.
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back (e.g., scratch-buffer pipelines) see [`bulk_apply_base`].
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// // Map a 100×100 u64 grid to a new grid where each cell is input + 1.
    /// let g = BlockedGrid::<u64>::new(100, 100);
    /// let out = g.map_base::<u64, _>(|inp, outp| {
    ///     for r in 0..64 {
    ///         let in_row = inp.row(r);
    ///         let out_row = outp.row_mut(r);
    ///         for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
    ///             *dst = src.wrapping_add(1);
    ///         }
    ///     }
    /// });
    /// // Input is unchanged.
    /// assert!(g.as_padded_slice().iter().all(|&v| v == 0));
    /// // Output has every cell incremented by 1.
    /// assert_eq!(out.as_padded_slice()[0], 1);
    /// ```
    ///
    /// [`bulk_apply_base`]: BlockedGrid::bulk_apply_base
    pub fn map_base<U: Copy + Default, F>(&self, mut f: F) -> BlockedGrid<U, BR, BC>
    where
        F: FnMut(&GridBlock<'_, T, BR, BC>, &mut GridBlockMut<'_, U, BR, BC>),
    {
        let mut out = BlockedGrid::<U, BR, BC>::new(self.rows(), self.cols());
        let n_block_rows = if BR == 0 { 0 } else { self.padded_rows() / BR };
        let n_block_cols = if BC == 0 { 0 } else { self.padded_cols() / BC };
        for br in 0..n_block_rows {
            for bc in 0..n_block_cols {
                let inp = GridBlock::from_grid(self, br, bc);
                let mut outp = GridBlockMut::from_grid(&mut out, br, bc);
                f(&inp, &mut outp);
            }
        }
        out
    }

    /// Map a closure over every `N×N` super-block, producing a new grid with
    /// element type `U`.
    ///
    /// The closure `f` receives:
    /// - `&GridSuperBlock<'_, T, BR, BC, N>` — read-only window into the input
    ///   super-block.
    /// - `&mut GridSuperBlockMut<'_, U, BR, BC, N>` — mutable window into the
    ///   corresponding super-block of the freshly-allocated output grid.
    ///
    /// Same data-flow invariant as [`map_base`]: the input grid is never
    /// mutated.
    ///
    /// # Panics
    /// Delegates to [`blocks_tier`](BlockedGrid::blocks_tier) and
    /// [`blocks_tier_mut`](BlockedGrid::blocks_tier_mut); panics with a clear
    /// message if `padded_rows % (BR * N) != 0` or
    /// `padded_cols % (BC * N) != 0`.
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back use [`bulk_apply_tier`] instead.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// // Map a 256×256 u64 grid at tier N=4; verify input unchanged.
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let out = g.map_tier::<u64, 4, _>(|inp, _outp| {
    ///     // inp is the one 256×256 super-block; we leave outp at default.
    ///     let _ = inp.super_row();
    /// });
    /// assert!(g.as_padded_slice().iter().all(|&v| v == 0));
    /// assert!(out.as_padded_slice().iter().all(|&v| v == 0));
    /// ```
    ///
    /// [`map_base`]: BlockedGrid::map_base
    /// [`bulk_apply_tier`]: BlockedGrid::bulk_apply_tier
    pub fn map_tier<U: Copy + Default, const N: usize, F>(&self, mut f: F) -> BlockedGrid<U, BR, BC>
    where
        F: FnMut(&GridSuperBlock<'_, T, BR, BC, N>, &mut GridSuperBlockMut<'_, U, BR, BC, N>),
    {
        let mut out = BlockedGrid::<U, BR, BC>::new(self.rows(), self.cols());
        // Pair up input and output super-block iterators in lockstep.
        // Both iterators traverse in the same row-major order with identical
        // super-row/super-col coordinates, so zip is correct.
        let mut out_iter = out.blocks_tier_mut::<N>();
        for inp_sb in self.blocks_tier::<N>() {
            let mut out_sb = out_iter
                .next()
                .expect("input and output tier iterators must have the same length");
            f(&inp_sb, &mut out_sb);
        }
        out
    }
}

// ============================================================
// SECONDARY write-back paths — &mut self, gated mutation only
//
// bulk_apply_base lives in the T: Copy impl block (same as map_base)
// because blocks_base_mut requires T: Copy.
// bulk_apply_tier lives in a separate T-unbounded impl block because
// blocks_tier_mut does not require T: Copy.
// ============================================================

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Apply a closure in-place over every base block.
    ///
    /// The closure `f` receives:
    /// - `&mut GridBlockMut<'_, T, BR, BC>` — mutable window into the current
    ///   block.
    /// - `(usize, usize)` — `(block_row, block_col)` coordinates of the block
    ///   within the super-block grid (0-based, row-major).
    ///
    /// Blocks are visited in row-major order.
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_base`] instead, which
    /// returns a fresh grid.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// // Stamp each block with (block_row * 1000 + block_col).
    /// let mut g = BlockedGrid::<u64>::new(100, 100);
    /// g.bulk_apply_base(|blk, (br, bc)| {
    ///     let stamp = (br * 1000 + bc) as u64;
    ///     blk.row_mut(0)[0] = stamp;
    /// });
    /// // Block (0,0) → cell (0,0) = 0*1000+0 = 0
    /// assert_eq!(g.get(0, 0), 0);
    /// // Block (0,1) → cell (0,64) = 0*1000+1 = 1
    /// assert_eq!(g.get(0, 64), 1);
    /// // Block (1,0) → cell (64,0) = 1*1000+0 = 1000
    /// assert_eq!(g.get(64, 0), 1000);
    /// ```
    ///
    /// [`map_base`]: BlockedGrid::map_base
    pub fn bulk_apply_base<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut GridBlockMut<'_, T, BR, BC>, (usize, usize)),
    {
        for mut blk in self.blocks_base_mut() {
            let coords = (blk.block_row(), blk.block_col());
            f(&mut blk, coords);
        }
    }
}

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Apply a closure in-place over every `N×N` super-block.
    ///
    /// The closure `f` receives:
    /// - `&mut GridSuperBlockMut<'_, T, BR, BC, N>` — mutable window into the
    ///   current super-block.
    /// - `(usize, usize)` — `(super_row, super_col)` coordinates of the
    ///   super-block within the super-block grid (0-based, row-major).
    ///
    /// Super-blocks are visited in row-major order.
    ///
    /// # Panics
    /// Panics if `padded_rows % (BR * N) != 0` or `padded_cols % (BC * N) != 0`
    /// (inherited from [`blocks_tier_mut`](BlockedGrid::blocks_tier_mut)).
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_tier`] instead, which
    /// returns a fresh grid.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// // 256×256 grid: exactly one N=4 super-block.
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let mut call_count = 0usize;
    /// g.bulk_apply_tier::<4, _>(|sb, (sr, sc)| {
    ///     call_count += 1;
    ///     assert_eq!((sr, sc), (0, 0)); // single super-block at origin
    ///     // write-back: stamp super-block origin
    ///     let _ = sb.super_row();
    /// });
    /// assert_eq!(call_count, 1);
    /// ```
    ///
    /// [`map_tier`]: BlockedGrid::map_tier
    pub fn bulk_apply_tier<const N: usize, F>(&mut self, mut f: F)
    where
        F: FnMut(&mut GridSuperBlockMut<'_, T, BR, BC, N>, (usize, usize)),
    {
        for mut sb in self.blocks_tier_mut::<N>() {
            let coords = (sb.super_row(), sb.super_col());
            f(&mut sb, coords);
        }
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::super::base::{BlockedGrid, GridBlock};

    // ------------------------------------------------------------------
    // map_base — input-unchanged invariant
    // ------------------------------------------------------------------

    /// map_base on a 100×100 u64 grid: every cell in the output is input+1,
    /// AND the input is completely unchanged after the call.
    #[test]
    fn map_base_input_unchanged_u64() {
        let g = BlockedGrid::<u64>::new(100, 100);
        // Take a snapshot of the input's padded storage.
        let before: Vec<u64> = g.as_padded_slice().to_vec();

        let out = g.map_base::<u64, _>(|inp, outp| {
            for r in 0..64 {
                let in_row = inp.row(r);
                let out_row = outp.row_mut(r);
                for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
                    *dst = src.wrapping_add(1);
                }
            }
        });

        // Input is unchanged.
        assert_eq!(g.as_padded_slice(), before.as_slice());
        // Output has every cell equal to 0u64.wrapping_add(1) = 1.
        assert!(out.as_padded_slice().iter().all(|&v| v == 1));
    }

    /// map_base allows U != T: BlockedGrid<u64> → BlockedGrid<u32>.
    #[test]
    fn map_base_type_conversion_u64_to_u32() {
        let mut g = BlockedGrid::<u64, 64, 64>::new(64, 64);
        // Fill with values that truncate predictably.
        g.set(0, 0, 0xDEAD_BEEF_0000_0001_u64);
        g.set(0, 1, 0x0000_0000_CAFE_BABE_u64);

        let out = g.map_base::<u32, _>(|inp, outp| {
            for r in 0..64 {
                let in_row = inp.row(r);
                let out_row = outp.row_mut(r);
                for (dst, &src) in out_row.iter_mut().zip(in_row.iter()) {
                    *dst = src as u32; // truncate to u32
                }
            }
        });

        // Input unchanged.
        assert_eq!(g.get(0, 0), 0xDEAD_BEEF_0000_0001_u64);
        // Output is truncated u32.
        assert_eq!(out.get(0, 0), 0x0000_0001_u32);
        assert_eq!(out.get(0, 1), 0xCAFE_BABE_u32);
    }

    // ------------------------------------------------------------------
    // map_tier — input-unchanged invariant
    // ------------------------------------------------------------------

    /// map_tier::<4> on a 256×256 grid: input is unchanged after the call.
    #[test]
    fn map_tier_input_unchanged_256x256() {
        let mut g = BlockedGrid::<u64>::new(256, 256);
        // Fill with a non-zero pattern so we can distinguish from default.
        for v in g.as_padded_slice_mut().iter_mut() {
            *v = 0xA5A5_A5A5_A5A5_A5A5_u64;
        }
        let before: Vec<u64> = g.as_padded_slice().to_vec();

        let _out = g.map_tier::<u64, 4, _>(|_inp, _outp| {
            // Closure does nothing; output stays at U::default() = 0.
        });

        // Input is unchanged.
        assert_eq!(g.as_padded_slice(), before.as_slice());
    }

    // ------------------------------------------------------------------
    // map_base — empty grid
    // ------------------------------------------------------------------

    /// map_base on an empty grid (0×0) returns an empty grid; closure is never
    /// called.
    #[test]
    fn map_base_empty_grid_closure_not_called() {
        let g = BlockedGrid::<u64>::new(0, 0);
        let mut called = 0usize;
        let out = g.map_base::<u64, _>(|_inp, _outp| {
            called += 1;
        });
        assert_eq!(called, 0, "closure must not be called for empty grid");
        assert_eq!(out.rows(), 0);
        assert_eq!(out.cols(), 0);
        assert_eq!(out.as_padded_slice().len(), 0);
    }

    // ------------------------------------------------------------------
    // bulk_apply_base — write-back correctness
    // ------------------------------------------------------------------

    /// bulk_apply_base sets every block's first cell to
    /// (block_row * 1000 + block_col); subsequent reads see the writes.
    #[test]
    fn bulk_apply_base_write_back_correct() {
        let mut g = BlockedGrid::<u64>::new(100, 100);
        g.bulk_apply_base(|blk, (br, bc)| {
            let stamp = (br * 1000 + bc) as u64;
            blk.row_mut(0)[0] = stamp;
        });

        // 100×100 → 128×128 padded → 2×2 = 4 blocks (64×64 blocks).
        // block (0,0): row_origin=0, col_origin=0
        assert_eq!(g.get(0, 0), 0 * 1000 + 0); // block (0,0)
                                               // block (0,1): row_origin=0, col_origin=64
        assert_eq!(g.get(0, 64), 0 * 1000 + 1); // block (0,1)
                                                // block (1,0): row_origin=64, col_origin=0
        assert_eq!(g.get(64, 0), 1 * 1000 + 0); // block (1,0)
                                                // block (1,1): row_origin=64, col_origin=64
        assert_eq!(g.get(64, 64), 1 * 1000 + 1); // block (1,1)
    }

    /// bulk_apply_base iterates all blocks in row-major order.
    /// 100×100 grid with 64×64 blocks → 2×2 = 4 blocks; closure called 4 times.
    #[test]
    fn bulk_apply_base_closure_called_four_times() {
        let mut g = BlockedGrid::<u64>::new(100, 100);
        let mut call_count = 0usize;
        let mut coords_seen = Vec::new();

        g.bulk_apply_base(|_blk, (br, bc)| {
            call_count += 1;
            coords_seen.push((br, bc));
        });

        assert_eq!(call_count, 4, "expected 4 blocks (2×2)");
        // Row-major order: (0,0), (0,1), (1,0), (1,1).
        assert_eq!(coords_seen, vec![(0, 0), (0, 1), (1, 0), (1, 1)], "blocks must be visited in row-major order");
    }

    // ------------------------------------------------------------------
    // bulk_apply_tier — write-back correctness
    // ------------------------------------------------------------------

    /// bulk_apply_tier::<4> on a 256×256 grid → closure called exactly once.
    #[test]
    fn bulk_apply_tier4_256x256_called_once() {
        let mut g = BlockedGrid::<u64>::new(256, 256);
        let mut call_count = 0usize;

        g.bulk_apply_tier::<4, _>(|_sb, (sr, sc)| {
            call_count += 1;
            assert_eq!((sr, sc), (0, 0), "only one super-block at origin");
        });

        assert_eq!(call_count, 1, "expected exactly one N=4 super-block");
    }

    /// bulk_apply_tier::<4> on a 128×128 grid → panics (128 % 256 != 0).
    #[test]
    #[should_panic(
        expected = "BlockedGrid::blocks_tier::<4>: padded extent 128×128 is not divisible by tier stride 256×256"
    )]
    fn bulk_apply_tier4_128x128_panics() {
        let mut g = BlockedGrid::<u64>::new(128, 128);
        g.bulk_apply_tier::<4, _>(|_sb, _coords| {});
    }

    // ------------------------------------------------------------------
    // GridBlock::row / rows accessors
    // ------------------------------------------------------------------

    /// GridBlock::row returns the correct contiguous slice.
    #[test]
    fn grid_block_row_returns_correct_slice() {
        let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
        // Set cell (1, 2) — block (0,0), local row=1, col=2.
        g.set(1, 2, 99);
        let blk = GridBlock::from_grid(&g, 0, 0);
        let row = blk.row(1);
        assert_eq!(row.len(), 4);
        assert_eq!(row[2], 99);
    }

    /// GridBlock::rows yields BR rows each of length BC.
    #[test]
    fn grid_block_rows_yields_br_rows() {
        let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
        let blk = GridBlock::from_grid(&g, 0, 0);
        let rows: Vec<&[u8]> = blk.rows().collect();
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert_eq!(row.len(), 4);
        }
    }
}
