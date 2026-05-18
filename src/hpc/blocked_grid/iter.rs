//! Base-block iterators for [`BlockedGrid`].
//!
//! Provides [`BaseBlockIter`] and [`BaseBlockIterMut`] — row-major iterators
//! over the BR×BC base blocks of a [`BlockedGrid`].  Both types are returned
//! by the companion methods [`BlockedGrid::blocks_base`] and
//! [`BlockedGrid::blocks_base_mut`] added via the `impl` blocks at the bottom
//! of this file.
//!
//! No SIMD primitives, no `#[target_feature]`, no distance metrics.
//! Layout-only per the PR-X3 design doc
//! (`.claude/knowledge/pr-x3-cognitive-grid-design.md`).
//!
//! ## Distance-typing guardrail
//! This module is layout-only per `.claude/knowledge/cognitive-distance-typing.md`.
//! No distance-aware API is added here; see W7 for typed distance bulk fns.
//!
//! ## Data-flow rule
//! `blocks_base_mut` is a **write-back** helper per `.claude/rules/data-flow.md`
//! Rule #3.  For computation that derives a new grid from an input grid use
//! `map_base` (worker A4, PRIMARY compute path) instead.

use super::base::{BlockedGrid, GridBlock, GridBlockMut};
use std::marker::PhantomData;

// ============================================================
// BaseBlockIter — read-only row-major iterator
// ============================================================

/// Row-major iterator over the BR×BC base blocks of a [`BlockedGrid`].
///
/// Yields one [`GridBlock`] per `(block_row, block_col)` pair in row-major
/// order: `(0,0), (0,1), …, (0, n_bc-1), (1,0), …`.
///
/// Constructed via [`BlockedGrid::blocks_base`].
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
/// let blocks: Vec<_> = g.blocks_base().collect();
/// assert_eq!(blocks.len(), 4); // 2×2 blocks
/// assert_eq!(blocks[0].block_row(), 0);
/// assert_eq!(blocks[0].block_col(), 0);
/// assert_eq!(blocks[1].block_col(), 1);
/// assert_eq!(blocks[2].block_row(), 1);
/// ```
pub struct BaseBlockIter<'a, T, const BR: usize, const BC: usize> {
    grid: &'a BlockedGrid<T, BR, BC>,
    block_row: usize,
    block_col: usize,
    n_block_rows: usize,
    n_block_cols: usize,
}

impl<'a, T: Copy, const BR: usize, const BC: usize> Iterator for BaseBlockIter<'a, T, BR, BC> {
    type Item = GridBlock<'a, T, BR, BC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_row >= self.n_block_rows {
            return None;
        }
        let br = self.block_row;
        let bc = self.block_col;

        // Advance in row-major order.
        self.block_col += 1;
        if self.block_col >= self.n_block_cols {
            self.block_col = 0;
            self.block_row += 1;
        }

        Some(GridBlock::from_grid(self.grid, br, bc))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.block_row >= self.n_block_rows {
            0
        } else {
            let total = self.n_block_rows * self.n_block_cols;
            let consumed = self.block_row * self.n_block_cols + self.block_col;
            total - consumed
        };
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy, const BR: usize, const BC: usize> ExactSizeIterator for BaseBlockIter<'a, T, BR, BC> {}

// ============================================================
// BaseBlockIterMut — mutable row-major iterator
// ============================================================

/// Row-major mutable iterator over the BR×BC base blocks of a [`BlockedGrid`].
///
/// Yields one [`GridBlockMut`] per `(block_row, block_col)` pair in row-major
/// order.  Each yielded block holds a mutable window into the parent grid's
/// storage.
///
/// Constructed via [`BlockedGrid::blocks_base_mut`].
///
/// # Data-flow rule
/// This is a **write-back** helper per `.claude/rules/data-flow.md` Rule #3.
/// For computation that derives a new grid from an input grid use `map_base`
/// (PRIMARY compute path) instead.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
/// for mut blk in g.blocks_base_mut() {
///     // write-back: fill the first cell of each block with the block index.
///     let idx = blk.block_row() * 2 + blk.block_col();
///     blk.row_mut(0)[0] = idx as u8;
/// }
/// assert_eq!(g.get(0, 0), 0);
/// assert_eq!(g.get(0, 4), 1);
/// assert_eq!(g.get(4, 0), 2);
/// assert_eq!(g.get(4, 4), 3);
/// ```
pub struct BaseBlockIterMut<'a, T, const BR: usize, const BC: usize> {
    /// Raw pointer to the grid so we can re-borrow it mutably for each block.
    ///
    /// # Safety invariant
    /// `ptr` is valid for the lifetime `'a` and no other reference to the grid
    /// coexists with any yielded `GridBlockMut`.  This is guaranteed by the
    /// borrow checker: `blocks_base_mut` takes `&'a mut BlockedGrid`, moves it
    /// into this struct, and the struct itself is not `Copy`/`Clone`, so at
    /// most one `BaseBlockIterMut` can exist for a given grid at a time.
    ptr: *mut BlockedGrid<T, BR, BC>,
    block_row: usize,
    block_col: usize,
    n_block_rows: usize,
    n_block_cols: usize,
    _marker: PhantomData<&'a mut BlockedGrid<T, BR, BC>>,
}

// SAFETY: `BaseBlockIterMut` is the sole owner of the `&'a mut BlockedGrid`
// borrow (encoded as a raw pointer for lending-iterator ergonomics).  It is
// safe to send across threads if `T: Send`, and sharing references is safe if
// `T: Sync`.
unsafe impl<'a, T: Send, const BR: usize, const BC: usize> Send for BaseBlockIterMut<'a, T, BR, BC> {}
unsafe impl<'a, T: Sync, const BR: usize, const BC: usize> Sync for BaseBlockIterMut<'a, T, BR, BC> {}

impl<'a, T: Copy, const BR: usize, const BC: usize> Iterator for BaseBlockIterMut<'a, T, BR, BC> {
    type Item = GridBlockMut<'a, T, BR, BC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_row >= self.n_block_rows {
            return None;
        }
        let br = self.block_row;
        let bc = self.block_col;

        // Advance in row-major order before yielding, so the borrow below does
        // not alias with a future call to `next`.
        self.block_col += 1;
        if self.block_col >= self.n_block_cols {
            self.block_col = 0;
            self.block_row += 1;
        }

        // SAFETY:
        // 1. `self.ptr` was created from a valid `&'a mut BlockedGrid` in
        //    `blocks_base_mut`; the grid is live for `'a`.
        // 2. We advance the index *before* re-borrowing, so no two alive
        //    `GridBlockMut` values overlap in the grid's storage.
        // 3. `GridBlockMut::from_grid` slices into disjoint regions of the
        //    flat storage for distinct (br, bc) pairs (each block covers
        //    `[row_origin..row_origin+BR) × [col_origin..col_origin+BC)` in
        //    the padded grid, which are non-overlapping by construction).
        let grid: &'a mut BlockedGrid<T, BR, BC> = unsafe { &mut *self.ptr };
        Some(GridBlockMut::from_grid(grid, br, bc))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.block_row >= self.n_block_rows {
            0
        } else {
            let total = self.n_block_rows * self.n_block_cols;
            let consumed = self.block_row * self.n_block_cols + self.block_col;
            total - consumed
        };
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy, const BR: usize, const BC: usize> ExactSizeIterator for BaseBlockIterMut<'a, T, BR, BC> {}

// ============================================================
// BlockedGrid methods: blocks_base / blocks_base_mut
// ============================================================

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Iterator over BR×BC base blocks, yielding one [`GridBlock`] per
    /// `(block_row, block_col)` pair in row-major order.
    ///
    /// The iterator implements [`ExactSizeIterator`], so `.len()` is available
    /// before consuming any items.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// let g = BlockedGrid::<u64>::new(100, 100);
    /// // 100×100 grid with 64×64 blocks → 2×2 = 4 base blocks.
    /// assert_eq!(g.blocks_base().len(), 4);
    ///
    /// let mut blocks = g.blocks_base();
    /// let b = blocks.next().unwrap();
    /// assert_eq!((b.block_row(), b.block_col()), (0, 0));
    /// assert_eq!((b.row_origin(), b.col_origin()), (0, 0));
    ///
    /// let b = blocks.next().unwrap();
    /// assert_eq!((b.block_row(), b.block_col()), (0, 1));
    /// assert_eq!((b.row_origin(), b.col_origin()), (0, 64));
    /// ```
    pub fn blocks_base(&self) -> BaseBlockIter<'_, T, BR, BC> {
        let n_block_rows = if BR == 0 { 0 } else { self.padded_rows() / BR };
        let n_block_cols = if BC == 0 { 0 } else { self.padded_cols() / BC };
        BaseBlockIter {
            grid: self,
            block_row: 0,
            block_col: 0,
            n_block_rows,
            n_block_cols,
        }
    }

    /// Mutable iterator over BR×BC base blocks, yielding one [`GridBlockMut`]
    /// per `(block_row, block_col)` pair in row-major order.
    ///
    /// The iterator implements [`ExactSizeIterator`].
    ///
    /// # Data-flow rule
    /// This is a **write-back** operation per `.claude/rules/data-flow.md`
    /// Rule #3.  The closure receives a mutable window into each block and is
    /// expected to perform **write-back** operations only (e.g., scratch-buffer
    /// fill, gated XOR).  For computation that derives a new grid from the
    /// input use `map_base` (PRIMARY compute path — worker A4) which returns a
    /// fresh grid and does not mutate the input.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    ///
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// // Write-back: stamp each block with its linear index.
    /// for mut blk in g.blocks_base_mut() {
    ///     let stamp = (blk.block_row() * 2 + blk.block_col()) as u8;
    ///     blk.row_mut(0)[0] = stamp;
    /// }
    /// assert_eq!(g.get(0, 0), 0);
    /// assert_eq!(g.get(0, 4), 1);
    /// assert_eq!(g.get(4, 0), 2);
    /// assert_eq!(g.get(4, 4), 3);
    /// ```
    pub fn blocks_base_mut(&mut self) -> BaseBlockIterMut<'_, T, BR, BC> {
        let n_block_rows = if BR == 0 { 0 } else { self.padded_rows() / BR };
        let n_block_cols = if BC == 0 { 0 } else { self.padded_cols() / BC };
        BaseBlockIterMut {
            ptr: self as *mut BlockedGrid<T, BR, BC>,
            block_row: 0,
            block_col: 0,
            n_block_rows,
            n_block_cols,
            _marker: PhantomData,
        }
    }
}

// ============================================================
// GridBlockMut row accessor
// ============================================================
//
// `row_mut` on `GridBlockMut` is needed by the iterator doctests and unit
// tests (and by downstream workers A4/A6).  It was not included in A1's
// base.rs.  It is added here in the sibling `iter` module; it is `pub` so
// consumers can use it via `crate::hpc::blocked_grid::GridBlockMut::row_mut`.
// The coordinator may elect to move it to base.rs in a later pass.

impl<'a, T, const BR: usize, const BC: usize> GridBlockMut<'a, T, BR, BC> {
    /// Mutable borrow of row `r` of this block as a contiguous `&mut [T]` of
    /// length BC.
    ///
    /// # Panics
    /// Panics in debug builds if `r >= BR`.
    ///
    /// # Data-flow rule
    /// Direct row mutation is a **write-back** operation per
    /// `.claude/rules/data-flow.md` Rule #3 ("No `&mut self` during
    /// computation. Ever."). The closure body that calls `row_mut` must be
    /// performing gated write-back only (single-target XOR, BUNDLE majority
    /// merge, scratch-buffer fill). For COMPUTE paths use
    /// [`BlockedGrid::map_base`] which returns a fresh grid and never
    /// mutates `self`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// let mut blk = GridBlockMut::from_grid(&mut g, 0, 0);
    /// blk.row_mut(0)[0] = 42;
    /// drop(blk);
    /// assert_eq!(g.get(0, 0), 42);
    /// ```
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < BR, "row {} out of block range {}", r, BR);
        let stride = self.padded_cols();
        let start = r * stride;
        &mut self.data_mut()[start..start + BC]
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::super::base::{BlockedGrid, GridBlockMut};

    // ----------------------------------------------------------
    // blocks_base — basic count and order
    // ----------------------------------------------------------

    #[test]
    fn blocks_base_100x100_yields_4_blocks() {
        let g = BlockedGrid::<u64>::new(100, 100);
        assert_eq!(g.blocks_base().count(), 4);
    }

    #[test]
    fn blocks_base_row_major_order() {
        let g = BlockedGrid::<u64>::new(100, 100);
        let coords: Vec<_> = g
            .blocks_base()
            .map(|b| (b.block_row(), b.block_col()))
            .collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn blocks_base_origins_match_block_dims() {
        let g = BlockedGrid::<u64>::new(100, 100);
        for blk in g.blocks_base() {
            assert_eq!(blk.row_origin(), blk.block_row() * 64);
            assert_eq!(blk.col_origin(), blk.block_col() * 64);
        }
    }

    #[test]
    fn blocks_base_64x64_single_block() {
        let g = BlockedGrid::<u64>::new(64, 64);
        let blocks: Vec<_> = g.blocks_base().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_row(), 0);
        assert_eq!(blocks[0].block_col(), 0);
    }

    #[test]
    fn blocks_base_empty_grid_yields_zero() {
        let g = BlockedGrid::<u64>::new(0, 0);
        assert_eq!(g.blocks_base().count(), 0);
    }

    // ----------------------------------------------------------
    // ExactSizeIterator
    // ----------------------------------------------------------

    #[test]
    fn exact_size_iter_len_before_next() {
        let g = BlockedGrid::<u64>::new(100, 100);
        let iter = g.blocks_base();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn exact_size_iter_len_decrements() {
        let g = BlockedGrid::<u64>::new(100, 100);
        let mut iter = g.blocks_base();
        assert_eq!(iter.len(), 4);
        iter.next();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    // ----------------------------------------------------------
    // blocks_base_mut — mutation is visible via blocks_base
    // ----------------------------------------------------------

    #[test]
    fn blocks_base_mut_mutation_visible() {
        let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
        for mut blk in g.blocks_base_mut() {
            let stamp = (blk.block_row() * 2 + blk.block_col()) as u8;
            blk.row_mut(0)[0] = stamp;
        }
        // Four 4×4 blocks in an 8×8 padded grid.
        assert_eq!(g.get(0, 0), 0); // block (0,0)
        assert_eq!(g.get(0, 4), 1); // block (0,1)
        assert_eq!(g.get(4, 0), 2); // block (1,0)
        assert_eq!(g.get(4, 4), 3); // block (1,1)
    }

    #[test]
    fn blocks_base_mut_exact_size() {
        let mut g = BlockedGrid::<u64>::new(100, 100);
        let iter = g.blocks_base_mut();
        assert_eq!(iter.len(), 4);
    }

    // ----------------------------------------------------------
    // Half-square shape: BlockedGrid<u8, 16, 64>
    // ----------------------------------------------------------

    #[test]
    fn half_square_16x64_yields_4_blocks() {
        // new(32, 128) → padded 32×128 → 2×2 = 4 blocks of shape 16×64
        let g = BlockedGrid::<u8, 16, 64>::new(32, 128);
        let blocks: Vec<_> = g.blocks_base().collect();
        assert_eq!(blocks.len(), 4);
        // Row-major order
        let coords: Vec<_> = blocks
            .iter()
            .map(|b| (b.block_row(), b.block_col()))
            .collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        // Origins
        assert_eq!(blocks[0].row_origin(), 0);
        assert_eq!(blocks[0].col_origin(), 0);
        assert_eq!(blocks[1].row_origin(), 0);
        assert_eq!(blocks[1].col_origin(), 64);
        assert_eq!(blocks[2].row_origin(), 16);
        assert_eq!(blocks[2].col_origin(), 0);
        assert_eq!(blocks[3].row_origin(), 16);
        assert_eq!(blocks[3].col_origin(), 64);
    }

    // ----------------------------------------------------------
    // GridBlockMut::row_mut — basic accessor
    // ----------------------------------------------------------

    #[test]
    fn row_mut_accessor_write_read() {
        let mut g = BlockedGrid::<u32, 4, 4>::new(8, 8);
        {
            let mut blk = GridBlockMut::from_grid(&mut g, 1, 1);
            blk.row_mut(0)[0] = 0xABCD;
            blk.row_mut(0)[3] = 0x1234;
        }
        // block (1,1) → row_origin = 4, col_origin = 4
        assert_eq!(g.get(4, 4), 0xABCD);
        assert_eq!(g.get(4, 7), 0x1234);
    }
}
