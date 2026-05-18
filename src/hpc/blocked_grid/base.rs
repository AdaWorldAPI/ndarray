use std::marker::PhantomData;

// ============================================================
// BlockedGrid — the primary type
// ============================================================

/// Generic block-padded 2-D grid. Storage is row-major, padded to
/// (BR, BC) base-block boundaries on both axes. Higher tiers (L2 / L3 / L4
/// for the 64×64 default) are expressed as iteration patterns, not as
/// extra padding.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::BlockedGrid;
/// let g = BlockedGrid::<u64>::new(100, 100);
/// assert_eq!(g.padded_rows(), 128);
/// assert_eq!(g.padded_cols(), 128);
/// ```
pub struct BlockedGrid<T, const BLK_ROW: usize = 64, const BLK_COL: usize = 64> {
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    data: Vec<T>,
}

// ============================================================
// Constructors
// ============================================================

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Create a grid sized to (rows, cols), with storage padded to the (BR, BC)
    /// block boundary, with all cells — including padding — initialised to
    /// `pad_value`. The `T: Copy` bound is the only constraint on `T`; no
    /// `Default` is required.
    ///
    /// # Panics (compile-time)
    /// Instantiating `BlockedGrid::<T, 0, BC>` or `BlockedGrid::<T, BR, 0>` is
    /// a compile-time error: the const assert `BR > 0 && BC > 0` fires before
    /// any code is generated.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64, 64, 64>::new_with_pad(100, 100, 0xDEAD_BEEF);
    /// assert_eq!(g.padded_rows(), 128);
    /// assert_eq!(g.padded_cols(), 128);
    /// assert!(g.as_padded_slice().iter().all(|&v| v == 0xDEAD_BEEF));
    /// ```
    pub fn new_with_pad(rows: usize, cols: usize, pad_value: T) -> Self {
        const { assert!(BR > 0 && BC > 0, "BlockedGrid: block dims must be > 0") };
        // If either logical dimension is zero the grid is empty.
        let padded_rows = if rows == 0 || cols == 0 {
            0
        } else {
            rows.div_ceil(BR) * BR
        };
        let padded_cols = if rows == 0 || cols == 0 {
            0
        } else {
            cols.div_ceil(BC) * BC
        };
        let data = vec![pad_value; padded_rows * padded_cols];
        Self {
            rows,
            cols,
            padded_rows,
            padded_cols,
            data,
        }
    }
}

impl<T: Copy + Default, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Create a grid initialised to `T::default()`. Convenience wrapper over
    /// [`new_with_pad`](BlockedGrid::new_with_pad) for types where the default
    /// value is the natural padding fill (e.g. `u64` → `0` = causally-null edge).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64>::new(100, 100);
    /// assert_eq!(g.rows(), 100);
    /// assert_eq!(g.cols(), 100);
    /// assert_eq!(g.padded_rows(), 128);
    /// assert_eq!(g.padded_cols(), 128);
    /// assert_eq!(g.as_padded_slice().len(), 128 * 128);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::new_with_pad(rows, cols, T::default())
    }
}

// ============================================================
// Dimension accessors (no T bound required)
// ============================================================

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Return the logical row count (as passed to the constructor).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8>::new(10, 20);
    /// assert_eq!(g.rows(), 10);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the logical column count (as passed to the constructor).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8>::new(10, 20);
    /// assert_eq!(g.cols(), 20);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Return the padded row count (`ceil(rows / BR) * BR`).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8, 64, 64>::new(100, 100);
    /// assert_eq!(g.padded_rows(), 128);
    /// ```
    pub fn padded_rows(&self) -> usize {
        self.padded_rows
    }

    /// Return the padded column count (`ceil(cols / BC) * BC`).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8, 64, 64>::new(100, 100);
    /// assert_eq!(g.padded_cols(), 128);
    /// ```
    pub fn padded_cols(&self) -> usize {
        self.padded_cols
    }

    /// Return the base block dimensions `(BR, BC)`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// assert_eq!(BlockedGrid::<u64, 16, 64>::block_dims(), (16, 64));
    /// assert_eq!(BlockedGrid::<u64>::block_dims(), (64, 64));
    /// ```
    pub fn block_dims() -> (usize, usize) {
        (BR, BC)
    }

    /// Logical `(row, col)` → flat index into the padded storage vector.
    ///
    /// The index satisfies `flat = row * padded_cols + col`. Asserts (in debug
    /// builds) that `(row, col)` lies within the **logical** extent, not just
    /// the padded extent — padding cells are unreachable via this method.
    ///
    /// Use [`as_padded_slice`](BlockedGrid::as_padded_slice) with this index to
    /// read a logical cell: `slice[grid.idx(r, c)]`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// // 100×100 grid with 64×64 blocks → padded_cols = 128
    /// let g = BlockedGrid::<u64>::new(100, 100);
    /// // Row 3, col 5 → flat index = 3 * 128 + 5 = 389
    /// assert_eq!(g.idx(3, 5), 3 * 128 + 5);
    /// ```
    pub fn idx(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows, "row {} out of logical range {}", row, self.rows);
        debug_assert!(col < self.cols, "col {} out of logical range {}", col, self.cols);
        row * self.padded_cols + col
    }
}

// ============================================================
// Cell accessors (T: Copy)
// ============================================================

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Read the cell at logical `(row, col)`.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(100, 100);
    /// g.set(50, 50, 0xCAFE);
    /// assert_eq!(g.get(50, 50), 0xCAFE);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[self.idx(row, col)]
    }

    /// Write `v` to the cell at logical `(row, col)`.
    ///
    /// # Data-flow rule
    /// This is a **write-back** operation per `.claude/rules/data-flow.md`
    /// Rule #3. Use this only for constructing or filling a grid before
    /// computation. For per-block transformation use `map_base` (PRIMARY
    /// compute path — worker A4) which returns a new grid and does not mutate
    /// the input.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u64>::new(10, 10);
    /// g.set(3, 7, 42);
    /// assert_eq!(g.get(3, 7), 42);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, v: T) {
        let i = self.idx(row, col);
        self.data[i] = v;
    }

    /// Borrow the full padded storage as a flat slice. Useful for SIMD-stage
    /// closures that walk the storage as a 1-D vector at the BR×BC base tier.
    ///
    /// # Footgun
    /// The returned slice **includes padding cells** at the right and bottom
    /// of the logical extent. The slice length is `padded_rows() * padded_cols()`,
    /// not `rows() * cols()`. Cells at indices that map outside the logical
    /// (rows, cols) box are padding cells (default-initialized via [`new`] or
    /// set explicitly via [`new_with_pad`]).
    ///
    /// To compute a logical-cell flat index correctly, use [`idx`]:
    /// `as_padded_slice()[grid.idx(r, c)]`. NEVER index the slice as
    /// `r * cols() + c` — that ignores stride and reads the wrong cell.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8, 64, 64>::new(100, 100);
    /// assert_eq!(g.as_padded_slice().len(), 128 * 128); // padded extent
    /// ```
    ///
    /// [`new`]: BlockedGrid::new
    /// [`new_with_pad`]: BlockedGrid::new_with_pad
    /// [`idx`]: BlockedGrid::idx
    pub fn as_padded_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable variant — see [`as_padded_slice`](BlockedGrid::as_padded_slice) footgun note.
    ///
    /// # Footgun
    /// The returned slice **includes padding cells** at the right and bottom
    /// of the logical extent. The slice length is `padded_rows() * padded_cols()`,
    /// not `rows() * cols()`. Cells at indices that map outside the logical
    /// (rows, cols) box are padding cells (default-initialized via [`new`] or
    /// set explicitly via [`new_with_pad`]).
    ///
    /// To compute a logical-cell flat index correctly, use [`idx`]:
    /// `as_padded_slice_mut()[grid.idx(r, c)]`. NEVER index the slice as
    /// `r * cols() + c` — that ignores stride and reads the wrong cell.
    ///
    /// # Data-flow rule
    /// Direct slice mutation is a **write-back** operation per
    /// `.claude/rules/data-flow.md` Rule #3. For per-block transformation use
    /// `map_base` (PRIMARY compute path — worker A4) which returns a new grid
    /// and does not mutate the input.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let mut g = BlockedGrid::<u8, 64, 64>::new(100, 100);
    /// assert_eq!(g.as_padded_slice_mut().len(), 128 * 128);
    /// ```
    pub fn as_padded_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// ============================================================
// Block view types
// ============================================================

/// Read-only base-block window into a [`BlockedGrid`].
///
/// Carries an explicit `PhantomData<&'a T>` for lifetime variance (idiomatic
/// Rust 2024 — Q3 ruling from the PR-X3 design doc).
///
/// Note: `data` points into the parent grid's storage with stride `padded_cols`,
/// so each logical row of the block occupies `padded_cols` elements in memory
/// even though the block is only `BC` cells wide.
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
/// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
/// let blk = GridBlock::from_grid(&g, 0, 0);
/// assert_eq!(blk.block_row(), 0);
/// assert_eq!(blk.block_col(), 0);
/// assert_eq!(blk.row_origin(), 0);
/// assert_eq!(blk.col_origin(), 0);
/// ```
pub struct GridBlock<'a, T, const BR: usize, const BC: usize> {
    block_row: usize,
    block_col: usize,
    row_origin: usize,
    col_origin: usize,
    padded_cols: usize,
    /// Points to the element at `(row_origin, col_origin)` in the parent grid's
    /// flat storage. Length is `BR * padded_cols` (covers all BR rows including
    /// inter-row stride gaps).
    data: &'a [T],
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const BR: usize, const BC: usize> GridBlock<'a, T, BR, BC> {
    /// Construct a `GridBlock` from a grid reference and block coordinates.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// let blk = GridBlock::from_grid(&g, 1, 1);
    /// assert_eq!(blk.block_row(), 1);
    /// assert_eq!(blk.row_origin(), 4);
    /// assert_eq!(blk.col_origin(), 4);
    /// ```
    pub fn from_grid(grid: &'a BlockedGrid<T, BR, BC>, block_row: usize, block_col: usize) -> Self
    where
        T: Copy,
    {
        let row_origin = block_row * BR;
        let col_origin = block_col * BC;
        let start = row_origin * grid.padded_cols + col_origin;
        // The slice covers from the block origin to the end of the last block row
        // (stride-wise). The full allocation is always available.
        let end = if BR == 0 {
            start
        } else {
            start + (BR - 1) * grid.padded_cols + BC
        };
        let end = end.min(grid.data.len());
        Self {
            block_row,
            block_col,
            row_origin,
            col_origin,
            padded_cols: grid.padded_cols,
            data: &grid.data[start..end],
            _marker: PhantomData,
        }
    }

    /// Base-block row index (0-based within the grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlock::from_grid(&g, 1, 0).block_row(), 1);
    /// ```
    pub fn block_row(&self) -> usize {
        self.block_row
    }

    /// Base-block column index (0-based within the grid).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlock::from_grid(&g, 0, 1).block_col(), 1);
    /// ```
    pub fn block_col(&self) -> usize {
        self.block_col
    }

    /// Row index in the parent grid of the first row of this block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlock::from_grid(&g, 2, 0).row_origin(), 8);
    /// ```
    pub fn row_origin(&self) -> usize {
        self.row_origin
    }

    /// Column index in the parent grid of the first column of this block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlock};
    /// let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlock::from_grid(&g, 0, 2).col_origin(), 8);
    /// ```
    pub fn col_origin(&self) -> usize {
        self.col_origin
    }
}

/// Mutable base-block window into a [`BlockedGrid`].
///
/// Carries an explicit `PhantomData<&'a mut T>` for lifetime variance (Q3 ruling).
///
/// # Example
/// ```
/// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
/// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
/// let blk = GridBlockMut::from_grid(&mut g, 0, 0);
/// assert_eq!(blk.block_row(), 0);
/// assert_eq!(blk.block_col(), 0);
/// ```
pub struct GridBlockMut<'a, T, const BR: usize, const BC: usize> {
    block_row: usize,
    block_col: usize,
    row_origin: usize,
    col_origin: usize,
    padded_cols: usize,
    /// Points to the element at `(row_origin, col_origin)` in the parent grid's
    /// flat storage. Length is `BR * padded_cols` minus the trailing gap
    /// (covers the last block row up to the final cell).
    data: &'a mut [T],
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const BR: usize, const BC: usize> GridBlockMut<'a, T, BR, BC> {
    /// Construct a `GridBlockMut` from a mutable grid reference and block coordinates.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// let blk = GridBlockMut::from_grid(&mut g, 1, 1);
    /// assert_eq!(blk.block_row(), 1);
    /// assert_eq!(blk.row_origin(), 4);
    /// ```
    pub fn from_grid(grid: &'a mut BlockedGrid<T, BR, BC>, block_row: usize, block_col: usize) -> Self
    where
        T: Copy,
    {
        let row_origin = block_row * BR;
        let col_origin = block_col * BC;
        let start = row_origin * grid.padded_cols + col_origin;
        let end = if BR == 0 {
            start
        } else {
            start + (BR - 1) * grid.padded_cols + BC
        };
        let end = end.min(grid.data.len());
        let padded_cols = grid.padded_cols;
        Self {
            block_row,
            block_col,
            row_origin,
            col_origin,
            padded_cols,
            data: &mut grid.data[start..end],
            _marker: PhantomData,
        }
    }

    /// Base-block row index.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlockMut::from_grid(&mut g, 1, 0).block_row(), 1);
    /// ```
    pub fn block_row(&self) -> usize {
        self.block_row
    }

    /// Base-block column index.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlockMut::from_grid(&mut g, 0, 1).block_col(), 1);
    /// ```
    pub fn block_col(&self) -> usize {
        self.block_col
    }

    /// Row index in the parent grid of the first row of this block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlockMut::from_grid(&mut g, 2, 0).row_origin(), 8);
    /// ```
    pub fn row_origin(&self) -> usize {
        self.row_origin
    }

    /// Column index in the parent grid of the first column of this block.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::{BlockedGrid, GridBlockMut};
    /// let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
    /// assert_eq!(GridBlockMut::from_grid(&mut g, 0, 2).col_origin(), 8);
    /// ```
    pub fn col_origin(&self) -> usize {
        self.col_origin
    }

    /// Access internal data slice (for use by future iterator workers).
    #[doc(hidden)]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Access padded_cols stride (for use by future iterator workers).
    #[doc(hidden)]
    pub fn padded_cols(&self) -> usize {
        self.padded_cols
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper type that has no Default impl — tests T: Copy + !Default path.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct NoDefault(u32);

    // --------------------------------------------------------
    // new / new_with_pad — padding arithmetic
    // --------------------------------------------------------

    #[test]
    fn new_zero_zero() {
        let g = BlockedGrid::<u64>::new(0, 0);
        assert_eq!(g.padded_rows(), 0);
        assert_eq!(g.padded_cols(), 0);
        assert!(g.data.is_empty());
    }

    #[test]
    fn new_zero_rows() {
        // rows=0 → empty grid regardless of cols
        let g = BlockedGrid::<u64>::new(0, 100);
        assert_eq!(g.rows(), 0);
        assert_eq!(g.cols(), 100);
        assert_eq!(g.padded_rows(), 0);
        assert_eq!(g.padded_cols(), 0);
        assert!(g.data.is_empty());
    }

    #[test]
    fn new_zero_cols() {
        // cols=0 → empty grid regardless of rows
        let g = BlockedGrid::<u64>::new(100, 0);
        assert_eq!(g.rows(), 100);
        assert_eq!(g.cols(), 0);
        assert_eq!(g.padded_rows(), 0);
        assert_eq!(g.padded_cols(), 0);
        assert!(g.data.is_empty());
    }

    #[test]
    fn new_100_100_default_64x64() {
        let g = BlockedGrid::<u64>::new(100, 100);
        assert_eq!(g.padded_rows(), 128);
        assert_eq!(g.padded_cols(), 128);
        assert_eq!(g.data.len(), 128 * 128);
    }

    #[test]
    fn new_64_64_single_block() {
        let g = BlockedGrid::<u64>::new(64, 64);
        assert_eq!(g.padded_rows(), 64);
        assert_eq!(g.padded_cols(), 64);
        assert_eq!(g.data.len(), 64 * 64);
    }

    #[test]
    fn new_256_256_four_blocks() {
        let g = BlockedGrid::<u64>::new(256, 256);
        assert_eq!(g.padded_rows(), 256);
        assert_eq!(g.padded_cols(), 256);
        assert_eq!(g.data.len(), 256 * 256);
    }

    #[test]
    fn new_with_pad_fills_all_cells() {
        let g = BlockedGrid::<u64, 64, 64>::new_with_pad(8, 8, 0xDEAD_BEEF_u64);
        // 8 rows / 8 cols round up to one 64×64 block
        assert_eq!(g.padded_rows(), 64);
        assert_eq!(g.padded_cols(), 64);
        assert!(g.as_padded_slice().iter().all(|&v| v == 0xDEAD_BEEF_u64));
    }

    #[test]
    fn new_with_pad_no_default_type() {
        // NoDefault has no Default impl — verifies T: Copy only constraint
        let g = BlockedGrid::<NoDefault, 64, 64>::new_with_pad(10, 10, NoDefault(42));
        assert_eq!(g.padded_rows(), 64);
        assert_eq!(g.padded_cols(), 64);
        assert!(g.as_padded_slice().iter().all(|&v| v == NoDefault(42)));
    }

    // --------------------------------------------------------
    // block_dims
    // --------------------------------------------------------

    #[test]
    fn block_dims_default() {
        assert_eq!(BlockedGrid::<u64>::block_dims(), (64, 64));
    }

    #[test]
    fn block_dims_custom() {
        assert_eq!(BlockedGrid::<u8, 16, 64>::block_dims(), (16, 64));
        assert_eq!(BlockedGrid::<f32, 1, 16>::block_dims(), (1, 16));
    }

    // --------------------------------------------------------
    // Half-square and single-strip shapes
    // --------------------------------------------------------

    #[test]
    fn half_square_shape() {
        let g = BlockedGrid::<u8, 16, 64>::new(20, 100);
        assert_eq!(g.padded_rows(), 32); // ceil(20/16)*16 = 32
        assert_eq!(g.padded_cols(), 128); // ceil(100/64)*64 = 128
    }

    #[test]
    fn single_strip_shape() {
        let g = BlockedGrid::<f32, 1, 16>::new(10, 30);
        assert_eq!(g.padded_rows(), 10); // ceil(10/1)*1 = 10
        assert_eq!(g.padded_cols(), 32); // ceil(30/16)*16 = 32
    }

    // --------------------------------------------------------
    // idx
    // --------------------------------------------------------

    #[test]
    fn idx_formula() {
        // 100×100 grid with default 64×64 blocks → padded_cols = 128
        let g = BlockedGrid::<u64>::new(100, 100);
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(1, 0), 128);
        assert_eq!(g.idx(3, 5), 3 * 128 + 5);
        assert_eq!(g.idx(99, 99), 99 * 128 + 99);
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn idx_out_of_range_row() {
        let g = BlockedGrid::<u64>::new(10, 10);
        let _ = g.idx(10, 0); // row == logical rows → out of range
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn idx_out_of_range_col() {
        let g = BlockedGrid::<u64>::new(10, 10);
        let _ = g.idx(0, 10); // col == logical cols → out of range
    }

    // --------------------------------------------------------
    // get / set round-trip
    // --------------------------------------------------------

    #[test]
    fn get_set_round_trip() {
        let mut g = BlockedGrid::<u64>::new(100, 100);
        g.set(50, 50, 0xCAFE);
        assert_eq!(g.get(50, 50), 0xCAFE);
    }

    #[test]
    fn get_set_multiple_cells() {
        let mut g = BlockedGrid::<u32>::new(64, 64);
        for r in 0..64u32 {
            for c in 0..64u32 {
                g.set(r as usize, c as usize, r * 64 + c);
            }
        }
        for r in 0..64u32 {
            for c in 0..64u32 {
                assert_eq!(g.get(r as usize, c as usize), r * 64 + c);
            }
        }
    }

    // --------------------------------------------------------
    // as_padded_slice
    // --------------------------------------------------------

    #[test]
    fn padded_slice_len() {
        let g = BlockedGrid::<u64>::new(100, 100);
        assert_eq!(g.as_padded_slice().len(), 128 * 128);
    }

    #[test]
    fn padded_slice_mut_len() {
        let mut g = BlockedGrid::<u64>::new(100, 100);
        assert_eq!(g.as_padded_slice_mut().len(), 128 * 128);
    }

    #[test]
    fn padded_slice_default_zero() {
        let g = BlockedGrid::<u64>::new(100, 100);
        assert!(g.as_padded_slice().iter().all(|&v| v == 0));
    }

    // --------------------------------------------------------
    // GridBlock / GridBlockMut construction
    // --------------------------------------------------------

    #[test]
    fn grid_block_fields() {
        let g = BlockedGrid::<u8, 4, 4>::new(8, 8);
        let blk = GridBlock::from_grid(&g, 1, 1);
        assert_eq!(blk.block_row(), 1);
        assert_eq!(blk.block_col(), 1);
        assert_eq!(blk.row_origin(), 4);
        assert_eq!(blk.col_origin(), 4);
    }

    #[test]
    fn grid_block_mut_fields() {
        let mut g = BlockedGrid::<u8, 4, 4>::new(8, 8);
        let blk = GridBlockMut::from_grid(&mut g, 1, 1);
        assert_eq!(blk.block_row(), 1);
        assert_eq!(blk.block_col(), 1);
        assert_eq!(blk.row_origin(), 4);
        assert_eq!(blk.col_origin(), 4);
    }

    // --------------------------------------------------------
    // Accessor consistency
    // --------------------------------------------------------

    #[test]
    fn accessor_consistency() {
        let g = BlockedGrid::<u64, 8, 8>::new(20, 30);
        assert_eq!(g.rows(), 20);
        assert_eq!(g.cols(), 30);
        assert_eq!(g.padded_rows(), 24); // ceil(20/8)*8
        assert_eq!(g.padded_cols(), 32); // ceil(30/8)*8
        assert_eq!(g.as_padded_slice().len(), 24 * 32);
    }
}
