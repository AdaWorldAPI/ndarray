//! Columnar sparse storage for a lazy grid (PR-X9 A3).
//!
//! [`SparseGrid`] holds one logical-cell-indexed (row-major) set of
//! columns: the per-cell `basin_idx`, a 2-bit-packed mode tag, an 8-bit
//! `aux` byte (the δ byte for Delta cells, or the [`MergeDir`] discriminant
//! for Merge cells), and an escape map for the rare full-`u64` outliers.
//!
//! The 2-bit mode packing ([`ModeBits`]) is the design's per-cell storage
//! win: 4 cells per byte. Per-cell footprint ≈ `basin_idx` (2 B) + mode
//! (0.25 B) + `aux` (1 B) = 3.25 B vs 8 B dense ≈ 2.5× before escape.

use std::collections::HashMap;

use crate::hpc::codec::CellMode;

/// Mode discriminant → [`CellMode`]. The 2-bit codes match the codec's
/// on-wire taxonomy (`Skip=00`, `Merge=01`, `Delta=10`, `Escape=11`).
#[inline]
pub(super) fn mode_from_bits(bits: u8) -> CellMode {
    match bits & 0b11 {
        0b00 => CellMode::Skip,
        0b01 => CellMode::Merge,
        0b10 => CellMode::Delta,
        _ => CellMode::Escape,
    }
}

// ════════════════════════════════════════════════════════════════════
// ModeBits — 2-bit-packed mode tags
// ════════════════════════════════════════════════════════════════════

/// `len` mode tags packed 4-per-byte (2 bits each).
#[derive(Debug, Clone)]
pub struct ModeBits {
    bits: Vec<u8>,
    len: usize,
}

impl ModeBits {
    /// All-`Skip` (`0b00`) tags for `len` cells.
    pub fn new(len: usize) -> Self {
        Self {
            bits: vec![0u8; len.div_ceil(4)],
            len,
        }
    }

    /// Number of tags.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no tags.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Backing byte length (`ceil(len / 4)`) — used for size accounting.
    pub fn byte_len(&self) -> usize {
        self.bits.len()
    }

    /// Read the 2-bit tag at `i`.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        debug_assert!(i < self.len, "ModeBits::get index {i} out of range {}", self.len);
        (self.bits[i / 4] >> ((i % 4) * 2)) & 0b11
    }

    /// Write the 2-bit tag at `i` (low 2 bits of `m`).
    #[inline]
    pub fn set(&mut self, i: usize, m: u8) {
        debug_assert!(i < self.len, "ModeBits::set index {i} out of range {}", self.len);
        let shift = (i % 4) * 2;
        let byte = &mut self.bits[i / 4];
        *byte = (*byte & !(0b11 << shift)) | ((m & 0b11) << shift);
    }
}

// ════════════════════════════════════════════════════════════════════
// SparseGrid
// ════════════════════════════════════════════════════════════════════

/// Columnar per-cell storage backing a [`LazyBlockedGrid`](super::lazy::LazyBlockedGrid).
/// Indexed row-major by `cell = row * n_cols + col`.
#[derive(Debug, Clone)]
pub struct SparseGrid {
    n_rows: usize,
    n_cols: usize,
    /// Per-cell basin index into the codebook.
    basin_idx: Vec<u16>,
    /// Per-cell 2-bit mode tag.
    modes: ModeBits,
    /// Per-cell auxiliary byte: the δ byte for `Delta`, the `MergeDir`
    /// discriminant for `Merge`, `0` otherwise.
    aux: Vec<u8>,
    /// Full `u64` value for `Escape` cells, keyed by `usize` cell index
    /// (not `u32` — a u32 key would collide for grids with > 2³² cells).
    escape: HashMap<usize, u64>,
}

impl SparseGrid {
    /// Allocate an all-`Skip`, all-basin-0 sparse grid for `n_rows × n_cols`
    /// logical cells.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        let n_cells = n_rows * n_cols;
        Self {
            n_rows,
            n_cols,
            basin_idx: vec![0u16; n_cells],
            modes: ModeBits::new(n_cells),
            aux: vec![0u8; n_cells],
            escape: HashMap::new(),
        }
    }

    /// Logical dimensions `(rows, cols)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Row-major cell index for `(row, col)`.
    #[inline]
    pub fn cell_index(&self, row: usize, col: usize) -> usize {
        row * self.n_cols + col
    }

    /// Basin index of cell `ci`.
    #[inline]
    pub fn basin(&self, ci: usize) -> u16 {
        self.basin_idx[ci]
    }

    /// Mode of cell `ci`.
    #[inline]
    pub fn mode(&self, ci: usize) -> CellMode {
        mode_from_bits(self.modes.get(ci))
    }

    /// Auxiliary byte of cell `ci` (δ byte or merge-dir discriminant).
    #[inline]
    pub fn aux(&self, ci: usize) -> u8 {
        self.aux[ci]
    }

    /// Escape value of cell `ci`, if it is an `Escape` cell.
    #[inline]
    pub fn escape(&self, ci: usize) -> Option<u64> {
        self.escape.get(&ci).copied()
    }

    /// Set cell `ci` to `Skip` with the given basin.
    pub fn set_skip(&mut self, ci: usize, basin: u16) {
        self.basin_idx[ci] = basin;
        self.modes.set(ci, CellMode::Skip as u8);
        self.aux[ci] = 0;
    }

    /// Set cell `ci` to `Delta` with the given basin and δ byte.
    pub fn set_delta(&mut self, ci: usize, basin: u16, delta: u8) {
        self.basin_idx[ci] = basin;
        self.modes.set(ci, CellMode::Delta as u8);
        self.aux[ci] = delta;
    }

    /// Set cell `ci` to `Merge` with the given basin and direction code.
    pub fn set_merge(&mut self, ci: usize, basin: u16, dir: u8) {
        self.basin_idx[ci] = basin;
        self.modes.set(ci, CellMode::Merge as u8);
        self.aux[ci] = dir;
    }

    /// Set cell `ci` to `Escape`, storing the full value.
    pub fn set_escape(&mut self, ci: usize, basin: u16, value: u64) {
        self.basin_idx[ci] = basin;
        self.modes.set(ci, CellMode::Escape as u8);
        self.aux[ci] = 0;
        self.escape.insert(ci, value);
    }

    /// Approximate heap footprint in bytes (the columns + escape map).
    /// Used by the compression-ratio test.
    pub fn byte_size(&self) -> usize {
        self.basin_idx.len() * core::mem::size_of::<u16>()
            + self.modes.byte_len()
            + self.aux.len()
            + self.escape.len() * (core::mem::size_of::<usize>() + core::mem::size_of::<u64>())
    }

    /// Number of `Escape` cells (for tests / diagnostics).
    pub fn escape_count(&self) -> usize {
        self.escape.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_bits_roundtrip_all_codes() {
        let mut m = ModeBits::new(10);
        for i in 0..10 {
            m.set(i, (i % 4) as u8);
        }
        for i in 0..10 {
            assert_eq!(m.get(i), (i % 4) as u8);
        }
    }

    #[test]
    fn mode_bits_set_does_not_disturb_neighbours() {
        let mut m = ModeBits::new(4); // all 4 share one byte
        m.set(0, 0b11);
        m.set(1, 0b10);
        m.set(2, 0b01);
        m.set(3, 0b00);
        assert_eq!(m.get(0), 0b11);
        assert_eq!(m.get(1), 0b10);
        assert_eq!(m.get(2), 0b01);
        assert_eq!(m.get(3), 0b00);
        // Overwrite cell 1; others unchanged.
        m.set(1, 0b00);
        assert_eq!(m.get(0), 0b11);
        assert_eq!(m.get(1), 0b00);
        assert_eq!(m.get(2), 0b01);
    }

    #[test]
    fn mode_bits_byte_len_is_ceil_div_4() {
        assert_eq!(ModeBits::new(0).byte_len(), 0);
        assert_eq!(ModeBits::new(1).byte_len(), 1);
        assert_eq!(ModeBits::new(4).byte_len(), 1);
        assert_eq!(ModeBits::new(5).byte_len(), 2);
    }

    #[test]
    fn sparse_grid_setters_and_getters() {
        let mut g = SparseGrid::new(2, 3); // 6 cells
        let ci = g.cell_index(1, 2);
        assert_eq!(ci, 5);
        g.set_delta(ci, 7, 0x2A);
        assert_eq!(g.mode(ci), CellMode::Delta);
        assert_eq!(g.basin(ci), 7);
        assert_eq!(g.aux(ci), 0x2A);
        assert_eq!(g.escape(ci), None);

        g.set_escape(0, 3, 0xDEAD_BEEF);
        assert_eq!(g.mode(0), CellMode::Escape);
        assert_eq!(g.escape(0), Some(0xDEAD_BEEF));
        assert_eq!(g.escape_count(), 1);
    }

    #[test]
    fn fresh_grid_is_all_skip_basin_zero() {
        let g = SparseGrid::new(4, 4);
        for ci in 0..16 {
            assert_eq!(g.mode(ci), CellMode::Skip);
            assert_eq!(g.basin(ci), 0);
        }
    }
}
