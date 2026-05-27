//! `GridStorage` — the polymorphic substrate trait (PR-X9 A1).
//!
//! Both the dense [`BlockedGrid`] (PR-X3) and the lazy
//! [`LazyBlockedGrid`](super::lazy::LazyBlockedGrid) (PR-X9) implement
//! `GridStorage`, so cascade consumers can be written once against
//! `S: GridStorage<BR, BC>` and pick storage at the call site.
//!
//! # u64-concrete (v1)
//!
//! Cognitive cells are `CausalEdge64` (a `u64`), so the trait is concrete
//! over `u64` rather than generic `T`. The PR-X9 design's generic-`T`
//! `GridStorage<T>` (with a GAT block iterator + `T: Into<u64>` gather) is
//! deferred: the basin/δ/escape encoding is intrinsically u64, and a
//! u64-concrete trait avoids the GAT + trait-bound machinery for no v1
//! benefit.
//!
//! No `#[target_feature]`, no per-arch intrinsics, no distance API
//! (layering rule). `gather_u64x8` is the SIMD seam — PR-X5's typed
//! register banks consume its output.

use crate::hpc::blocked_grid::BlockedGrid;

/// A block-padded 2-D `u64` grid backend, dense or lazy.
pub trait GridStorage<const BR: usize, const BC: usize> {
    /// Logical row count.
    fn rows(&self) -> usize;
    /// Logical column count.
    fn cols(&self) -> usize;
    /// Padded row count (`ceil(rows / BR) * BR`).
    fn padded_rows(&self) -> usize;
    /// Padded column count (`ceil(cols / BC) * BC`).
    fn padded_cols(&self) -> usize;

    /// Read one logical cell. Lazy storage materializes it on demand from
    /// the codebook + perturbation.
    fn get_cell(&self, row: usize, col: usize) -> u64;

    /// Gather 8 consecutive cells (one AVX-512 / SVE register width) from
    /// `(row, col)`. Cells at or past the logical extent (padding) read as
    /// `0`. Requires `col + 8 <= padded_cols()`.
    fn gather_u64x8(&self, row: usize, col: usize) -> [u64; 8];

    /// Materialize the whole grid as a dense [`BlockedGrid`] — the escape
    /// hatch for tests / dense-vs-lazy parity gates. O(cells); never call
    /// on a hot path for lazy storage.
    fn materialize_dense(&self) -> BlockedGrid<u64, BR, BC>;
}

impl<const BR: usize, const BC: usize> GridStorage<BR, BC> for BlockedGrid<u64, BR, BC> {
    fn rows(&self) -> usize {
        BlockedGrid::rows(self)
    }
    fn cols(&self) -> usize {
        BlockedGrid::cols(self)
    }
    fn padded_rows(&self) -> usize {
        BlockedGrid::padded_rows(self)
    }
    fn padded_cols(&self) -> usize {
        BlockedGrid::padded_cols(self)
    }
    fn get_cell(&self, row: usize, col: usize) -> u64 {
        self.get(row, col)
    }
    fn gather_u64x8(&self, row: usize, col: usize) -> [u64; 8] {
        // Read straight from padded storage so a gather that crosses into
        // block padding (col in [cols, padded_cols)) returns the padding
        // fill rather than tripping the logical-bounds debug_assert in `get`.
        let stride = self.padded_cols();
        let slice = self.as_padded_slice();
        let base = row * stride + col;
        core::array::from_fn(|i| slice.get(base + i).copied().unwrap_or(0))
    }
    fn materialize_dense(&self) -> BlockedGrid<u64, BR, BC> {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_grid_implements_storage() {
        let mut g = BlockedGrid::<u64, 64, 64>::new(64, 64);
        g.set(3, 5, 0xABCD);
        let s: &dyn GridStorage<64, 64> = &g;
        assert_eq!(s.rows(), 64);
        assert_eq!(s.cols(), 64);
        assert_eq!(s.get_cell(3, 5), 0xABCD);
    }

    #[test]
    fn dense_gather_matches_individual_gets() {
        let mut g = BlockedGrid::<u64, 64, 64>::new(64, 64);
        for c in 0..8 {
            g.set(2, c, (c as u64) * 100 + 1);
        }
        let gathered = g.gather_u64x8(2, 0);
        for c in 0..8 {
            assert_eq!(gathered[c], g.get(2, c));
        }
    }

    #[test]
    fn dense_materialize_dense_is_equal_copy() {
        let mut g = BlockedGrid::<u64, 64, 64>::new(70, 70);
        g.set(10, 10, 42);
        let d = GridStorage::materialize_dense(&g);
        assert_eq!(d.as_padded_slice(), g.as_padded_slice());
    }
}
