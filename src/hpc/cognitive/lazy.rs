//! `LazyBlockedGrid` — basin-relative lazy grid storage (PR-X9 A4/A5).
//!
//! Stores each `CausalEdge64` cell as `(basin_idx, mode)` relative to the
//! shared [`CamCodebook`] instead of a dense `u64`. The encoding is the
//! x265-shaped skip/merge/delta/escape taxonomy ([`CellMode`]), applied to
//! a **lossless XOR-relative** model rather than the codec's lossy
//! arithmetic δ:
//!
//! | Mode   | Condition (`xor = value ^ basin.edge`) | Decode                         |
//! |--------|-----------------------------------------|--------------------------------|
//! | Skip   | `xor == 0`                              | `basin.edge`                   |
//! | Delta  | `0 < xor <= 0xFF`                       | `basin.edge ^ (δ byte)`        |
//! | Merge  | δ byte equals an already-coded N/W      | `basin.edge ^ neighbour δ byte`|
//! |        | Delta neighbour on the same basin       |                                |
//! | Escape | `xor > 0xFF`                            | full `u64` from the escape map |
//!
//! Because the δ byte *is* the low-byte XOR and Escape keeps the full
//! value, `decode_to_dense ∘ encode_from_dense` is **bit-exact** — there is
//! no quantization loss, so the codec's lossy arithmetic `rdo_select` (a
//! different, arithmetic distortion metric) does not apply; only the
//! [`CellMode`]/[`MergeDir`] taxonomy is shared.
//!
//! # Why not generic δ / `rdo_select`
//!
//! Cognitive edges are compared by XOR-Hamming (`nearest_basin` is
//! XOR-min), not arithmetic distance. Reusing `codec::rdo_select` (which
//! scores `|δ − δ̂|`) would impose the wrong distortion model. v1 therefore
//! does an exact XOR-relative mode decision and shares only the mode enum.
//!
//! # Merge is backward-only (N / W)
//!
//! A single row-major encode pass can only inherit from already-coded
//! neighbours, so Merge looks North and West (not E/S). Decode reads the
//! same backward neighbour, so reconstruction needs no forward references.

use super::sparse::SparseGrid;
use super::storage::GridStorage;
use crate::hpc::blocked_grid::BlockedGrid;
use crate::hpc::codec::{CellMode, MergeDir};
use crate::hpc::ogit_bridge::CamCodebook;

/// Index of the codebook atom whose `edge` is closest (min XOR) to
/// `value`. `None` for an empty codebook. Mirrors
/// `ogit_bridge::CognitiveBridge::nearest_basin` (kept inline so the lazy
/// grid borrows only the codebook, not the whole bridge).
#[inline]
fn nearest_basin(codebook: &CamCodebook, value: u64) -> Option<u16> {
    let atoms = &codebook.atoms;
    if atoms.is_empty() {
        return None;
    }
    // basin_idx is u16; the codebook is capped at 4096 atoms (CamCodebook::
    // MAX_ATOMS), so `best_idx as u16` below cannot truncate.
    debug_assert!(atoms.len() <= u16::MAX as usize + 1, "codebook exceeds u16 basin-index range");
    let mut best_idx = 0usize;
    let mut best_dist = value ^ atoms[0].edge;
    for (i, atom) in atoms.iter().enumerate().skip(1) {
        let dist = value ^ atom.edge;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    Some(best_idx as u16)
}

/// Lazy basin-relative grid over `BR × BC` blocks. Borrows the shared
/// immutable [`CamCodebook`]; owns only the per-grid [`SparseGrid`].
///
/// No `derive(Debug)`: the borrowed `CamCodebook` is not `Debug`.
#[derive(Clone)]
pub struct LazyBlockedGrid<'a, const BR: usize = 64, const BC: usize = 64> {
    codebook: &'a CamCodebook,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    sparse: SparseGrid,
}

impl<'a, const BR: usize, const BC: usize> LazyBlockedGrid<'a, BR, BC> {
    /// Encode a dense `u64` grid into lazy basin-relative storage against
    /// `codebook`. Bit-exact: [`decode_to_dense`](Self::decode_to_dense)
    /// recovers the original.
    pub fn encode_from_dense(dense: &BlockedGrid<u64, BR, BC>, codebook: &'a CamCodebook) -> Self {
        let rows = dense.rows();
        let cols = dense.cols();
        let mut sparse = SparseGrid::new(rows, cols);

        for r in 0..rows {
            for c in 0..cols {
                let ci = sparse.cell_index(r, c);
                let value = dense.get(r, c);
                let Some(basin) = nearest_basin(codebook, value) else {
                    // Empty codebook: nothing to be relative to → escape.
                    sparse.set_escape(ci, 0, value);
                    continue;
                };
                let edge = codebook.atoms[basin as usize].edge;
                let xor = value ^ edge;
                if xor == 0 {
                    sparse.set_skip(ci, basin);
                } else if xor <= 0xFF {
                    let byte = xor as u8;
                    if let Some(dir) = Self::merge_dir(&sparse, ci, r, c, cols, basin, byte) {
                        sparse.set_merge(ci, basin, dir as u8);
                    } else {
                        sparse.set_delta(ci, basin, byte);
                    }
                } else {
                    sparse.set_escape(ci, basin, value);
                }
            }
        }

        Self {
            codebook,
            rows,
            cols,
            padded_rows: dense.padded_rows(),
            padded_cols: dense.padded_cols(),
            sparse,
        }
    }

    /// A backward (North/West) Delta neighbour on the same basin carrying
    /// the same δ byte, if one exists — the Merge candidate.
    #[inline]
    fn merge_dir(
        sparse: &SparseGrid, ci: usize, r: usize, c: usize, cols: usize, basin: u16, byte: u8,
    ) -> Option<MergeDir> {
        if r > 0 {
            let n = ci - cols;
            if sparse.mode(n) == CellMode::Delta && sparse.basin(n) == basin && sparse.aux(n) == byte {
                return Some(MergeDir::North);
            }
        }
        if c > 0 {
            let w = ci - 1;
            if sparse.mode(w) == CellMode::Delta && sparse.basin(w) == basin && sparse.aux(w) == byte {
                return Some(MergeDir::West);
            }
        }
        None
    }

    /// Reconstruct the `u64` value of logical cell `ci`.
    #[inline]
    fn reconstruct(&self, ci: usize) -> u64 {
        let basin = self.sparse.basin(ci);
        let edge = self
            .codebook
            .atoms
            .get(basin as usize)
            .map(|a| a.edge)
            .unwrap_or(0);
        match self.sparse.mode(ci) {
            CellMode::Skip => edge,
            CellMode::Delta => edge ^ self.sparse.aux(ci) as u64,
            CellMode::Merge => {
                // The Merge neighbour is a same-basin Delta cell, so its
                // basin edge equals `edge`; XOR with its δ byte recovers
                // the value. Backward-only dirs ⇒ the neighbour is already
                // reconstructable.
                // The encoder only ever emits North or West (backward-only).
                // Assert that invariant; treat any other code as West so a
                // corrupt tag can't silently pick a wrong direction unnoticed
                // in debug builds.
                let nb = if self.sparse.aux(ci) == MergeDir::North as u8 {
                    ci - self.cols
                } else {
                    debug_assert_eq!(self.sparse.aux(ci), MergeDir::West as u8, "Merge dir must be North or West");
                    ci - 1
                };
                edge ^ self.sparse.aux(nb) as u64
            }
            CellMode::Escape => self.sparse.escape(ci).unwrap_or(edge),
        }
    }

    /// Decode the whole grid back to a dense `BlockedGrid<u64>`.
    pub fn decode_to_dense(&self) -> BlockedGrid<u64, BR, BC> {
        let mut out = BlockedGrid::<u64, BR, BC>::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let ci = self.sparse.cell_index(r, c);
                out.set(r, c, self.reconstruct(ci));
            }
        }
        out
    }

    /// Approximate heap footprint of the per-grid sparse storage in bytes
    /// (excludes the shared codebook). For the compression-ratio gate.
    pub fn byte_size(&self) -> usize {
        self.sparse.byte_size()
    }

    /// Number of `Escape` cells.
    pub fn escape_count(&self) -> usize {
        self.sparse.escape_count()
    }
}

impl<const BR: usize, const BC: usize> GridStorage<BR, BC> for LazyBlockedGrid<'_, BR, BC> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn padded_rows(&self) -> usize {
        self.padded_rows
    }
    fn padded_cols(&self) -> usize {
        self.padded_cols
    }
    fn get_cell(&self, row: usize, col: usize) -> u64 {
        if row >= self.rows || col >= self.cols {
            return 0; // padding
        }
        self.reconstruct(self.sparse.cell_index(row, col))
    }
    fn gather_u64x8(&self, row: usize, col: usize) -> [u64; 8] {
        core::array::from_fn(|i| self.get_cell(row, col + i))
    }
    fn materialize_dense(&self) -> BlockedGrid<u64, BR, BC> {
        self.decode_to_dense()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::ogit_bridge::CognitiveBridge;

    fn codebook() -> std::sync::Arc<CamCodebook> {
        CognitiveBridge::load_embedded()
            .expect("embedded cognitive TTL must load")
            .codebook
    }

    #[test]
    fn roundtrip_is_bit_exact_on_mixed_grid() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        let mut dense = BlockedGrid::<u64, 64, 64>::new(4, 4);
        // A mix of skip (exact), delta (low-byte off), escape (high-byte off).
        dense.set(0, 0, e0); // skip
        dense.set(0, 1, e0 ^ 0x07); // delta
        dense.set(0, 2, e0 ^ 0x10_0000); // escape (xor > 0xFF)
        dense.set(1, 0, 0xFFFF_FFFF_FFFF_FFFF); // escape outlier
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        let back = lazy.decode_to_dense();
        assert_eq!(back.as_padded_slice(), dense.as_padded_slice());
    }

    #[test]
    fn skip_dominates_when_every_cell_matches_a_basin() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        let mut dense = BlockedGrid::<u64, 64, 64>::new(8, 8);
        for r in 0..8 {
            for c in 0..8 {
                dense.set(r, c, e0);
            }
        }
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        assert_eq!(lazy.escape_count(), 0);
        // Every cell is Skip (xor 0) → decodes back exactly.
        assert_eq!(lazy.decode_to_dense().as_padded_slice(), dense.as_padded_slice());
    }

    #[test]
    fn adjacent_equal_deltas_become_merge() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        let mut dense = BlockedGrid::<u64, 64, 64>::new(1, 2);
        dense.set(0, 0, e0 ^ 0x07);
        dense.set(0, 1, e0 ^ 0x07); // same basin + same δ byte as West neighbour
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        // cell (0,0) = Delta, cell (0,1) = Merge(West).
        assert_eq!(lazy.sparse.mode(0), CellMode::Delta);
        assert_eq!(lazy.sparse.mode(1), CellMode::Merge);
        assert_eq!(lazy.sparse.aux(1), MergeDir::West as u8);
        assert_eq!(lazy.decode_to_dense().as_padded_slice(), dense.as_padded_slice());
    }

    #[test]
    fn outlier_is_escape_and_lossless() {
        let cb = codebook();
        let mut dense = BlockedGrid::<u64, 64, 64>::new(2, 2);
        dense.set(0, 0, 0x0123_4567_89AB_CDEF);
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        assert!(lazy.escape_count() >= 1);
        assert_eq!(lazy.get_cell(0, 0), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn gather_matches_individual_get_cell() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        let mut dense = BlockedGrid::<u64, 64, 64>::new(4, 16);
        for c in 0..16 {
            dense.set(2, c, e0 ^ (c as u64 & 0xFF));
        }
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        let g = lazy.gather_u64x8(2, 0);
        for i in 0..8 {
            assert_eq!(g[i], lazy.get_cell(2, i));
            assert_eq!(g[i], dense.get(2, i));
        }
    }

    #[test]
    fn gather_past_logical_extent_reads_zero() {
        let cb = codebook();
        let dense = BlockedGrid::<u64, 64, 64>::new(1, 3); // cols 0..3 logical
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        let g = lazy.gather_u64x8(0, 0);
        // cols 3..8 are past the logical extent → 0.
        for i in 3..8 {
            assert_eq!(g[i], 0);
        }
    }

    #[test]
    fn coherent_grid_compresses_below_dense() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        // 64×64 coherent grid: mostly skip + a few deltas.
        let mut dense = BlockedGrid::<u64, 64, 64>::new(64, 64);
        for r in 0..64 {
            for c in 0..64 {
                let v = if (r + c) % 16 == 0 { e0 ^ 0x03 } else { e0 };
                dense.set(r, c, v);
            }
        }
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        let dense_bytes = 64 * 64 * core::mem::size_of::<u64>();
        assert!(
            lazy.byte_size() < dense_bytes,
            "lazy {} bytes should be < dense {} bytes",
            lazy.byte_size(),
            dense_bytes
        );
        // And still bit-exact.
        assert_eq!(lazy.decode_to_dense().as_padded_slice(), dense.as_padded_slice());
    }

    #[test]
    fn materialize_dense_via_trait_equals_decode() {
        let cb = codebook();
        let e0 = cb.atoms[0].edge;
        let mut dense = BlockedGrid::<u64, 64, 64>::new(4, 4);
        dense.set(1, 1, e0 ^ 0x09);
        let lazy = LazyBlockedGrid::<64, 64>::encode_from_dense(&dense, &cb);
        let m = GridStorage::materialize_dense(&lazy);
        assert_eq!(m.as_padded_slice(), dense.as_padded_slice());
    }
}
