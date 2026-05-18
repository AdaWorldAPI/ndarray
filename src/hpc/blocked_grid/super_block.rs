//! Super-block and tier-iterator types for [`BlockedGrid`].
//!
//! Worker scope: `src/hpc/blocked_grid/super_block.rs` (sprint worker A3 — see
//! `.claude/knowledge/pr-x3-cognitive-grid-design.md` §"Worker decomposition").
//!
//! Provides `N×N` super-blocks over base `BR×BC` blocks, enabling the
//! L2/L3/L4 cache-hierarchy iteration pattern described in the PR-X3 design doc.
//!
//! ## Out of scope (NOT in this file)
//! - SIMD register-bank stack types (`StackedU64x8<N>`, …) → PR-X5
//! - Typed distance bulk functions → W7, bench-gated
//! - CausalEdge64 mantissa cell kernel → W7
//!
//! See `.claude/knowledge/cognitive-distance-typing.md` — no distance API here.
//! See `.claude/rules/data-flow.md` — all `&mut self` paths are write-back only.

use std::marker::PhantomData;

use super::base::{BlockedGrid, GridBlock, GridBlockMut};

// ============================================================
// GridSuperBlock — read-only N×N super-block view
// ============================================================

/// Read-only N×N super-block view: a window of N×N base `BR×BC` blocks.
///
/// Produced by [`TierBlockIter`] (via [`BlockedGrid::blocks_tier`]).  Each
/// super-block covers `N * BR` rows and `N * BC` columns of the parent grid.
///
/// Carries `PhantomData<&'a T>` for lifetime variance (Q3 ruling, PR-X3 design doc).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let g = BlockedGrid::<u64>::new(256, 256);
/// let sb = g.blocks_tier::<4>().next().unwrap();
/// assert_eq!(sb.super_row(), 0);
/// assert_eq!(sb.super_col(), 0);
/// assert_eq!(sb.row_origin(), 0);
/// assert_eq!(sb.col_origin(), 0);
/// assert_eq!(sb.base_blocks().count(), 16); // 4×4
/// ```
pub struct GridSuperBlock<'a, T, const BR: usize, const BC: usize, const N: usize> {
    super_row: usize,
    super_col: usize,
    row_origin: usize, // = super_row * N * BR
    col_origin: usize, // = super_col * N * BC
    padded_cols: usize,
    /// Points to cell (row_origin, 0) in the parent grid's flat storage.
    /// Length = N * BR * padded_cols (covers all rows of the super-block).
    data: &'a [T],
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> GridSuperBlock<'a, T, BR, BC, N> {
    /// Super-block row index (0-based within the super-block grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let sb = g.blocks_tier::<2>().nth(2).unwrap(); // row 1, col 0
    /// assert_eq!(sb.super_row(), 1);
    /// ```
    pub fn super_row(&self) -> usize {
        self.super_row
    }

    /// Super-block column index (0-based within the super-block grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let sb = g.blocks_tier::<2>().nth(1).unwrap(); // row 0, col 1
    /// assert_eq!(sb.super_col(), 1);
    /// ```
    pub fn super_col(&self) -> usize {
        self.super_col
    }

    /// Row index in the parent grid of the first row of this super-block.
    ///
    /// Equals `super_row * N * BR`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64, 64, 64>::new(256, 256);
    /// let sb = g.blocks_tier::<2>().nth(2).unwrap(); // super_row=1
    /// assert_eq!(sb.row_origin(), 128); // 1 * 2 * 64
    /// ```
    pub fn row_origin(&self) -> usize {
        self.row_origin
    }

    /// Column index in the parent grid of the first column of this super-block.
    ///
    /// Equals `super_col * N * BC`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64, 64, 64>::new(256, 256);
    /// let sb = g.blocks_tier::<2>().nth(1).unwrap(); // super_col=1
    /// assert_eq!(sb.col_origin(), 128); // 1 * 2 * 64
    /// ```
    pub fn col_origin(&self) -> usize {
        self.col_origin
    }

    /// Iterate the N×N base `BR×BC` blocks inside this super-block in
    /// row-major order (inner loop: base-block column).
    ///
    /// Yields `N * N` items.  Each [`GridBlock`] knows its absolute
    /// `block_row`, `block_col`, `row_origin`, and `col_origin` within the
    /// parent grid.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let sb = g.blocks_tier::<4>().next().unwrap();
    /// let blks: Vec<_> = sb.base_blocks().collect();
    /// assert_eq!(blks.len(), 16); // 4×4
    /// // First block is at (block_row=0, block_col=0)
    /// assert_eq!(blks[0].block_row(), 0);
    /// assert_eq!(blks[0].block_col(), 0);
    /// // Last block is at (block_row=3, block_col=3)
    /// assert_eq!(blks[15].block_row(), 3);
    /// assert_eq!(blks[15].block_col(), 3);
    /// ```
    pub fn base_blocks(&self) -> impl Iterator<Item = GridBlock<'_, T, BR, BC>> {
        let base_block_row_start = self.super_row * N; // absolute base-block row
        let base_block_col_start = self.super_col * N; // absolute base-block col
        let padded_cols = self.padded_cols;
        let data: &'a [T] = self.data;
        let row_origin = self.row_origin;

        (0..N).flat_map(move |local_br| {
            (0..N).map(move |local_bc| {
                let abs_block_row = base_block_row_start + local_br;
                let abs_block_col = base_block_col_start + local_bc;
                let abs_row_origin = row_origin + local_br * BR;
                let abs_col_origin = base_block_col_start * BC + local_bc * BC;

                // Offset within `data` (which starts at (row_origin, 0)) to
                // reach cell (abs_row_origin, abs_col_origin).
                let start = local_br * BR * padded_cols + abs_col_origin;
                let end = if BR == 0 {
                    start
                } else {
                    start + (BR - 1) * padded_cols + BC
                };
                let end = end.min(data.len());

                GridBlock::from_raw(
                    &data[start..end],
                    abs_block_row,
                    abs_block_col,
                    abs_row_origin,
                    abs_col_origin,
                    padded_cols,
                )
            })
        })
    }
}

// ============================================================
// GridSuperBlockMut — mutable N×N super-block view
// ============================================================

/// Mutable N×N super-block view: a window of N×N base `BR×BC` blocks.
///
/// Produced by [`TierBlockIterMut`] (via [`BlockedGrid::blocks_tier_mut`]).
///
/// Carries `PhantomData<&'a mut T>` for lifetime variance (Q3 ruling).
///
/// # Data-flow rule
/// This is a **write-back** type per `.claude/rules/data-flow.md` Rule #3.
/// For COMPUTE paths (reading input, deriving a new grid) use `map_tier`
/// (worker A4, PRIMARY path) which returns a fresh grid and never mutates
/// the input. Use `GridSuperBlockMut` only for gated write-back operations
/// (scratch-buffer fill, single-target XOR, BUNDLE majority merge).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let mut g = BlockedGrid::<u64>::new(256, 256);
/// for mut sb in g.blocks_tier_mut::<2>() {
///     // write-back: fill the super-block with a sentinel
///     for blk in sb.base_blocks_mut() {
///         // blk is a GridBlockMut — call row_mut / cell_mut on it
///         let _ = blk.block_row(); // demonstration
///     }
/// }
/// ```
pub struct GridSuperBlockMut<'a, T, const BR: usize, const BC: usize, const N: usize> {
    super_row: usize,
    super_col: usize,
    row_origin: usize,
    col_origin: usize,
    padded_cols: usize,
    /// Pointer into the parent grid's flat storage at cell (row_origin, 0).
    /// Length = N * BR * padded_cols.
    data: *mut T,
    data_len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> GridSuperBlockMut<'a, T, BR, BC, N> {
    /// Super-block row index (0-based within the super-block grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let sb = g.blocks_tier_mut::<2>().nth(2).unwrap();
    /// assert_eq!(sb.super_row(), 1);
    /// ```
    pub fn super_row(&self) -> usize {
        self.super_row
    }

    /// Super-block column index (0-based within the super-block grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let sb = g.blocks_tier_mut::<2>().nth(1).unwrap();
    /// assert_eq!(sb.super_col(), 1);
    /// ```
    pub fn super_col(&self) -> usize {
        self.super_col
    }

    /// Row index in the parent grid of the first row of this super-block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64, 64, 64>::new(256, 256);
    /// let sb = g.blocks_tier_mut::<2>().nth(2).unwrap();
    /// assert_eq!(sb.row_origin(), 128);
    /// ```
    pub fn row_origin(&self) -> usize {
        self.row_origin
    }

    /// Column index in the parent grid of the first column of this super-block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64, 64, 64>::new(256, 256);
    /// let sb = g.blocks_tier_mut::<2>().nth(1).unwrap();
    /// assert_eq!(sb.col_origin(), 128);
    /// ```
    pub fn col_origin(&self) -> usize {
        self.col_origin
    }

    /// Iterate the N×N mutable base `BR×BC` block views inside this super-block
    /// in row-major order.
    ///
    /// Each [`GridBlockMut`] provides mutable access to its `BR×BC` cell region.
    /// Blocks are non-overlapping by construction; the iterator yields `N * N` items.
    ///
    /// # Data-flow rule
    /// Write-back only — see [`GridSuperBlockMut`] type-level note.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let mut sb = g.blocks_tier_mut::<2>().next().unwrap();
    /// assert_eq!(sb.base_blocks_mut().count(), 4); // 2×2
    /// ```
    pub fn base_blocks_mut(&mut self) -> impl Iterator<Item = GridBlockMut<'_, T, BR, BC>> {
        let base_block_row_start = self.super_row * N;
        let base_block_col_start = self.super_col * N;
        let padded_cols = self.padded_cols;
        let row_origin = self.row_origin;
        let data_ptr = self.data;
        let data_len = self.data_len;
        let col_origin_abs = base_block_col_start * BC;

        (0..N).flat_map(move |local_br| {
            (0..N).map(move |local_bc| {
                let abs_block_row = base_block_row_start + local_br;
                let abs_block_col = base_block_col_start + local_bc;
                let abs_row_origin = row_origin + local_br * BR;
                let abs_col_origin = col_origin_abs + local_bc * BC;

                let start = local_br * BR * padded_cols + abs_col_origin;
                let end = if BR == 0 {
                    start
                } else {
                    start + (BR - 1) * padded_cols + BC
                };
                let end = end.min(data_len);

                // SAFETY: Each (local_br, local_bc) pair accesses a
                // non-overlapping sub-region of the super-block's data buffer.
                // Base blocks within the N×N super-block occupy disjoint
                // row ranges (local_br selects which BR-row group) and disjoint
                // column positions within each row (local_bc selects which BC
                // column group).  The iterator yields them one at a time, so
                // the caller cannot hold two simultaneous mutable borrows to
                // the same data.  `data_ptr` is valid for `data_len` elements
                // — guaranteed by `TierBlockIterMut` which derives it from the
                // grid's `Vec<T>`.
                let slice = unsafe { std::slice::from_raw_parts_mut(data_ptr.add(start), end - start) };

                GridBlockMut::from_raw(slice, abs_block_row, abs_block_col, abs_row_origin, abs_col_origin, padded_cols)
            })
        })
    }
}

// SAFETY: `GridSuperBlockMut` holds a raw pointer that is exclusively owned
// (derived from `&'a mut [T]` via `TierBlockIterMut`). Sending across threads
// requires T: Send; sharing immutably requires T: Sync.
unsafe impl<'a, T: Send, const BR: usize, const BC: usize, const N: usize> Send
    for GridSuperBlockMut<'a, T, BR, BC, N>
{
}
unsafe impl<'a, T: Sync, const BR: usize, const BC: usize, const N: usize> Sync
    for GridSuperBlockMut<'a, T, BR, BC, N>
{
}

// ============================================================
// TierBlockIter — read-only super-block iterator
// ============================================================

/// Iterator over N×N super-blocks in row-major order.
///
/// Produced by [`BlockedGrid::blocks_tier`]. Yields one
/// [`GridSuperBlock`] per `(super_row, super_col)` pair.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let g = BlockedGrid::<u64>::new(256, 256);
/// let sbs: Vec<_> = g.blocks_tier::<4>().collect();
/// assert_eq!(sbs.len(), 1); // 256×256 / (4*64 × 4*64) = 1×1
/// ```
pub struct TierBlockIter<'a, T, const BR: usize, const BC: usize, const N: usize> {
    grid: &'a BlockedGrid<T, BR, BC>,
    super_row: usize,
    super_col: usize,
    n_super_rows: usize,
    n_super_cols: usize,
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> Iterator for TierBlockIter<'a, T, BR, BC, N> {
    type Item = GridSuperBlock<'a, T, BR, BC, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.super_row >= self.n_super_rows {
            return None;
        }

        let sr = self.super_row;
        let sc = self.super_col;

        // Advance position (row-major)
        self.super_col += 1;
        if self.super_col >= self.n_super_cols {
            self.super_col = 0;
            self.super_row += 1;
        }

        let row_origin = sr * N * BR;
        let col_origin = sc * N * BC;
        let padded_cols = self.grid.padded_cols();

        // Slice from (row_origin, 0) covering N*BR full rows.
        let start = row_origin * padded_cols;
        let row_count = N * BR;
        let end = (start + row_count * padded_cols).min(self.grid.as_padded_slice().len());
        let data = &self.grid.as_padded_slice()[start..end];

        Some(GridSuperBlock {
            super_row: sr,
            super_col: sc,
            row_origin,
            col_origin,
            padded_cols,
            data,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.super_row >= self.n_super_rows {
            0
        } else {
            let rows_done = self.super_row * self.n_super_cols + self.super_col;
            self.n_super_rows * self.n_super_cols - rows_done
        };
        (remaining, Some(remaining))
    }
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> ExactSizeIterator for TierBlockIter<'a, T, BR, BC, N> {}

// ============================================================
// TierBlockIterMut — mutable super-block iterator
// ============================================================

/// Mutable iterator over N×N super-blocks in row-major order.
///
/// Produced by [`BlockedGrid::blocks_tier_mut`]. Yields one
/// [`GridSuperBlockMut`] per `(super_row, super_col)` pair.
///
/// # Data-flow rule
/// Write-back only per `.claude/rules/data-flow.md` Rule #3. For COMPUTE
/// paths use `map_tier` (worker A4).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let mut g = BlockedGrid::<u64>::new(256, 256);
/// let count = g.blocks_tier_mut::<2>().count();
/// assert_eq!(count, 4); // 2×2 super-blocks
/// ```
pub struct TierBlockIterMut<'a, T, const BR: usize, const BC: usize, const N: usize> {
    data: *mut T,
    data_len: usize,
    padded_cols: usize,
    super_row: usize,
    super_col: usize,
    n_super_rows: usize,
    n_super_cols: usize,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: `TierBlockIterMut` owns exclusive access to the underlying data
// (derived from `&'a mut BlockedGrid<T, BR, BC>`).
unsafe impl<'a, T: Send, const BR: usize, const BC: usize, const N: usize> Send for TierBlockIterMut<'a, T, BR, BC, N> {}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> Iterator for TierBlockIterMut<'a, T, BR, BC, N> {
    type Item = GridSuperBlockMut<'a, T, BR, BC, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.super_row >= self.n_super_rows {
            return None;
        }

        let sr = self.super_row;
        let sc = self.super_col;

        // Advance position (row-major)
        self.super_col += 1;
        if self.super_col >= self.n_super_cols {
            self.super_col = 0;
            self.super_row += 1;
        }

        let row_origin = sr * N * BR;
        let col_origin = sc * N * BC;
        let padded_cols = self.padded_cols;

        // Offset to (row_origin, 0) in the flat buffer.
        let start = row_origin * padded_cols;
        let row_count = N * BR;
        let end = (start + row_count * padded_cols).min(self.data_len);
        let len = end - start;

        Some(GridSuperBlockMut {
            super_row: sr,
            super_col: sc,
            row_origin,
            col_origin,
            padded_cols,
            // SAFETY: `start` is within `[0, data_len)` (guaranteed by the
            // divisibility check in `blocks_tier_mut`); `len` elements from
            // `data.add(start)` are within the original allocation.  Each call
            // to `next()` advances past the rows consumed by the returned
            // super-block (disjoint from all future yields), so no two live
            // `GridSuperBlockMut` items alias the same memory.
            data: unsafe { self.data.add(start) },
            data_len: len,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.super_row >= self.n_super_rows {
            0
        } else {
            let rows_done = self.super_row * self.n_super_cols + self.super_col;
            self.n_super_rows * self.n_super_cols - rows_done
        };
        (remaining, Some(remaining))
    }
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> ExactSizeIterator for TierBlockIterMut<'a, T, BR, BC, N> {}

// ============================================================
// BlockedGrid::blocks_tier / blocks_tier_mut
// ============================================================

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Iterator over super-blocks of `N` base-blocks per side (N×N grid of
    /// base blocks), in row-major order.
    ///
    /// Each super-block spans `N * BR` rows and `N * BC` columns of the parent
    /// grid.  Yields one [`GridSuperBlock`] per `(super_row, super_col)` pair.
    ///
    /// # Panics
    /// Panics if `padded_rows % (BR * N) != 0` or `padded_cols % (BC * N) != 0`.
    /// Pick a grid size whose padded extents are divisible by the tier stride, or
    /// use a smaller `N`.
    ///
    /// # Example — 256×256 grid, N=4 (L2 super-blocks)
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// let sbs: Vec<_> = g.blocks_tier::<4>().collect();
    /// assert_eq!(sbs.len(), 1); // (256/256) × (256/256)
    /// // The single super-block contains 4×4 = 16 base blocks.
    /// assert_eq!(sbs[0].base_blocks().count(), 16);
    /// ```
    ///
    /// # Example — 256×256 grid, N=2
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(256, 256);
    /// assert_eq!(g.blocks_tier::<2>().count(), 4); // 2×2 super-blocks
    /// ```
    ///
    /// # Example — empty grid
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(0, 0);
    /// assert_eq!(g.blocks_tier::<4>().count(), 0); // no super-blocks
    /// ```
    pub fn blocks_tier<const N: usize>(&self) -> TierBlockIter<'_, T, BR, BC, N> {
        let padded_rows = self.padded_rows();
        let padded_cols = self.padded_cols();

        // Empty grid: 0 % anything == 0, so no panic.
        if padded_rows == 0 || padded_cols == 0 {
            return TierBlockIter {
                grid: self,
                super_row: 0,
                super_col: 0,
                n_super_rows: 0,
                n_super_cols: 0,
            };
        }

        let tier_row_stride = BR * N;
        let tier_col_stride = BC * N;

        assert!(
            padded_rows % tier_row_stride == 0 && padded_cols % tier_col_stride == 0,
            "BlockedGrid::blocks_tier::<{N}>: padded extent {padded_rows}×{padded_cols} \
             is not divisible by tier stride {tier_row_stride}×{tier_col_stride}"
        );

        TierBlockIter {
            grid: self,
            super_row: 0,
            super_col: 0,
            n_super_rows: padded_rows / tier_row_stride,
            n_super_cols: padded_cols / tier_col_stride,
        }
    }

    /// Mutable iterator over super-blocks of `N` base-blocks per side.
    ///
    /// Yields one [`GridSuperBlockMut`] per `(super_row, super_col)` pair in
    /// row-major order.  Each item provides mutable access to its N×N base blocks
    /// via [`GridSuperBlockMut::base_blocks_mut`].
    ///
    /// # Data-flow rule
    /// Write-back only per `.claude/rules/data-flow.md` Rule #3.
    /// For COMPUTE paths (read-derive-new-grid) use `map_tier` (worker A4)
    /// which returns a fresh [`BlockedGrid`] and never mutates the input.
    ///
    /// # Panics
    /// Same divisibility condition as [`blocks_tier`](BlockedGrid::blocks_tier).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(256, 256);
    /// let count = g.blocks_tier_mut::<2>().count();
    /// assert_eq!(count, 4);
    /// ```
    pub fn blocks_tier_mut<const N: usize>(&mut self) -> TierBlockIterMut<'_, T, BR, BC, N> {
        let padded_rows = self.padded_rows();
        let padded_cols = self.padded_cols();

        if padded_rows == 0 || padded_cols == 0 {
            return TierBlockIterMut {
                data: self.as_padded_slice_mut().as_mut_ptr(),
                data_len: 0,
                padded_cols,
                super_row: 0,
                super_col: 0,
                n_super_rows: 0,
                n_super_cols: 0,
                _marker: PhantomData,
            };
        }

        let tier_row_stride = BR * N;
        let tier_col_stride = BC * N;

        assert!(
            padded_rows % tier_row_stride == 0 && padded_cols % tier_col_stride == 0,
            "BlockedGrid::blocks_tier::<{N}>: padded extent {padded_rows}×{padded_cols} \
             is not divisible by tier stride {tier_row_stride}×{tier_col_stride}"
        );

        let n_super_rows = padded_rows / tier_row_stride;
        let n_super_cols = padded_cols / tier_col_stride;
        let slice = self.as_padded_slice_mut();
        let data_len = slice.len();
        let data = slice.as_mut_ptr();

        TierBlockIterMut {
            data,
            data_len,
            padded_cols,
            super_row: 0,
            super_col: 0,
            n_super_rows,
            n_super_cols,
            _marker: PhantomData,
        }
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // blocks_tier count checks
    // ------------------------------------------------------------------

    /// 256×256 grid with N=4 → 1×1 super-block (one 256×256 super-block).
    #[test]
    fn tier4_256x256_yields_one_super_block() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let count = g.blocks_tier::<4>().count();
        assert_eq!(count, 1);
    }

    /// 256×256 grid with N=2 → 2×2 = 4 super-blocks.
    #[test]
    fn tier2_256x256_yields_four_super_blocks() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let count = g.blocks_tier::<2>().count();
        assert_eq!(count, 4);
    }

    /// N=1 is the degenerate case: each super-block wraps exactly one base block.
    /// Count must equal the number of base blocks.
    #[test]
    fn tier1_degenerate_matches_base_block_count() {
        let g = BlockedGrid::<u64>::new(256, 256);
        // 256/64 = 4 base-block rows × 4 base-block cols = 16 base blocks
        let tier1_count = g.blocks_tier::<1>().count();
        assert_eq!(tier1_count, 16);
    }

    /// Empty grid (0×0) → 0 super-blocks, no panic.
    #[test]
    fn tier4_empty_grid_yields_zero() {
        let g = BlockedGrid::<u64>::new(0, 0);
        assert_eq!(g.blocks_tier::<4>().count(), 0);
    }

    // ------------------------------------------------------------------
    // base_blocks inner count
    // ------------------------------------------------------------------

    /// Each super-block from N=4 contains N*N = 16 base blocks.
    #[test]
    fn tier4_super_block_contains_16_base_blocks() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let sb = g.blocks_tier::<4>().next().unwrap();
        assert_eq!(sb.base_blocks().count(), 16);
    }

    /// Each super-block from N=2 contains 4 base blocks.
    #[test]
    fn tier2_super_block_contains_4_base_blocks() {
        let g = BlockedGrid::<u64>::new(256, 256);
        for sb in g.blocks_tier::<2>() {
            assert_eq!(sb.base_blocks().count(), 4);
        }
    }

    /// N=1 degenerate case: each super-block contains exactly 1 base block.
    #[test]
    fn tier1_super_block_contains_1_base_block() {
        let g = BlockedGrid::<u64>::new(256, 256);
        for sb in g.blocks_tier::<1>() {
            assert_eq!(sb.base_blocks().count(), 1);
        }
    }

    // ------------------------------------------------------------------
    // base_blocks row-major order
    // ------------------------------------------------------------------

    /// base_blocks() yields blocks in row-major order.
    #[test]
    fn tier4_base_blocks_row_major_order() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let sb = g.blocks_tier::<4>().next().unwrap();
        let blks: Vec<_> = sb.base_blocks().collect();
        assert_eq!(blks.len(), 16);
        // Row-major: (br=0,bc=0),(br=0,bc=1),...,(br=0,bc=3),(br=1,bc=0),...
        for (i, blk) in blks.iter().enumerate() {
            let expected_br = i / 4;
            let expected_bc = i % 4;
            assert_eq!(blk.block_row(), expected_br, "index {i}");
            assert_eq!(blk.block_col(), expected_bc, "index {i}");
        }
    }

    /// Row-major order for 2×2 super-block grid.
    #[test]
    fn tier2_super_blocks_row_major_order() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let sbs: Vec<_> = g.blocks_tier::<2>().collect();
        assert_eq!(sbs[0].super_row(), 0);
        assert_eq!(sbs[0].super_col(), 0);
        assert_eq!(sbs[1].super_row(), 0);
        assert_eq!(sbs[1].super_col(), 1);
        assert_eq!(sbs[2].super_row(), 1);
        assert_eq!(sbs[2].super_col(), 0);
        assert_eq!(sbs[3].super_row(), 1);
        assert_eq!(sbs[3].super_col(), 1);
    }

    // ------------------------------------------------------------------
    // Origin coordinates
    // ------------------------------------------------------------------

    /// Verify row_origin / col_origin of super-blocks.
    #[test]
    fn tier2_origins() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let sbs: Vec<_> = g.blocks_tier::<2>().collect();
        // super_row=0, super_col=0 → origin (0, 0)
        assert_eq!(sbs[0].row_origin(), 0);
        assert_eq!(sbs[0].col_origin(), 0);
        // super_row=0, super_col=1 → origin (0, 2*64=128)
        assert_eq!(sbs[1].row_origin(), 0);
        assert_eq!(sbs[1].col_origin(), 128);
        // super_row=1, super_col=0 → origin (2*64=128, 0)
        assert_eq!(sbs[2].row_origin(), 128);
        assert_eq!(sbs[2].col_origin(), 0);
        // super_row=1, super_col=1 → origin (128, 128)
        assert_eq!(sbs[3].row_origin(), 128);
        assert_eq!(sbs[3].col_origin(), 128);
    }

    // ------------------------------------------------------------------
    // Panic on invalid divisibility
    // ------------------------------------------------------------------

    /// 128×128 padded grid with N=4: 128 % (64*4=256) != 0 → panic.
    #[test]
    #[should_panic(
        expected = "BlockedGrid::blocks_tier::<4>: padded extent 128×128 is not divisible by tier stride 256×256"
    )]
    fn tier4_128x128_panics() {
        let g = BlockedGrid::<u64>::new(128, 128);
        let _ = g.blocks_tier::<4>().count(); // trigger panic
    }

    // ------------------------------------------------------------------
    // Mutable iterator — mutation visibility
    // ------------------------------------------------------------------

    /// blocks_tier_mut::<2> — write via mutable super-block, read back.
    #[test]
    fn tier_mut_2_mutation_visible() {
        let mut g = BlockedGrid::<u64>::new(256, 256);
        // Write a distinct sentinel into each super-block's first cell.
        // Super-block (sr, sc) → sentinel = (sr*2 + sc + 1) as u64
        for mut sb in g.blocks_tier_mut::<2>() {
            let sr = sb.super_row();
            let sc = sb.super_col();
            let sentinel = (sr * 2 + sc + 1) as u64;
            // Write via base_blocks_mut: set first cell of first base block.
            for mut blk in sb.base_blocks_mut() {
                if blk.block_row() == sr * 2 && blk.block_col() == sc * 2 {
                    // Use base_blocks_mut raw data — access via data_mut()
                    let d = blk.data_mut();
                    if !d.is_empty() {
                        d[0] = sentinel;
                    }
                    break;
                }
            }
        }
        // Read back: each super-block's (sr, sc) origin cell should equal sentinel.
        for sb in g.blocks_tier::<2>() {
            let sr = sb.super_row();
            let sc = sb.super_col();
            let expected = (sr * 2 + sc + 1) as u64;
            // row_origin and col_origin give us the logical coordinates.
            let row = sb.row_origin();
            let col = sb.col_origin();
            assert_eq!(g.get(row, col), expected, "super_block ({sr},{sc}) sentinel mismatch");
        }
    }

    /// count via mutable iterator.
    #[test]
    fn tier_mut_count() {
        let mut g = BlockedGrid::<u64>::new(256, 256);
        assert_eq!(g.blocks_tier_mut::<4>().count(), 1);
        assert_eq!(g.blocks_tier_mut::<2>().count(), 4);
        assert_eq!(g.blocks_tier_mut::<1>().count(), 16);
    }

    /// Empty grid — mutable iterator yields zero, no panic.
    #[test]
    fn tier_mut_empty_no_panic() {
        let mut g = BlockedGrid::<u64>::new(0, 0);
        assert_eq!(g.blocks_tier_mut::<4>().count(), 0);
    }

    // ------------------------------------------------------------------
    // size_hint / ExactSizeIterator
    // ------------------------------------------------------------------

    #[test]
    fn tier_size_hint_exact() {
        let g = BlockedGrid::<u64>::new(256, 256);
        let iter = g.blocks_tier::<2>();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.len(), 4);
    }
}
