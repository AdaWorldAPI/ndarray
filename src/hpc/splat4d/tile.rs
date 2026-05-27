//! Tier-aware tile binning carrier + accessors (PR-X4 A1).
//!
//! [`TileBinning`] is the multi-resolution generalization of the bespoke
//! `splat3d::tile::TileBinning`: instead of one flat 16×16 pixel grid, it
//! bins each splat into the L1–L4 tiles its 3σ footprint covers, where
//! tier `k`'s tile pitch is `TIER_STRIDES[k-1] · (BR×BC)` cells — matching
//! the `BlockedGrid` `blocks_l1`/`l2`/`l3`/`l4` strides `[1, 4, 64, 256]`.
//!
//! This file owns the carrier struct + O(1) accessors. The binning
//! constructors (`from_projected_l1` / `from_projected_cascade`) live in
//! [`super::bin`]; composition lives in [`super::compose`].
//!
//! # Storage
//!
//! All `(tile, splat)` bindings for all tiers live in one flat
//! `instances` vector, sorted by `(tier, block_row, block_col,
//! confidence DESC)`. Per-tier prefix-sum arrays index into it so
//! `tile_instances(tier, br, bc)` is an O(1) slice into a
//! confidence-descending run.

use super::splat::Splat;

/// Base-block stride per tile side at each tier (L1..L4), matching the
/// `BlockedGrid` tier hierarchy: L1 = 1 base block, L2 = 4×4, L3 = 64×64,
/// L4 = 256×256 base blocks per super-tile side.
pub const TIER_STRIDES: [u32; 4] = [1, 4, 64, 256];

/// Number of cascade tiers.
pub const N_TIERS: usize = 4;

// ════════════════════════════════════════════════════════════════════
// TileInstance
// ════════════════════════════════════════════════════════════════════

/// One `(tile, splat)` binding emitted during binning.
///
/// `tier` is 1..=4 (L1..L4); `(block_row, block_col)` identifies the tile
/// at that tier; `splat_id` indexes the caller's splat slice; `confidence`
/// is the splat's NARS sort key (replicated here so sorting touches only
/// this 16-byte struct, not the larger `Splat`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileInstance {
    /// Cascade tier: 1 = L1, 2 = L2, 3 = L3, 4 = L4.
    pub tier: u8,
    /// Tile row at this tier.
    pub block_row: u16,
    /// Tile column at this tier.
    pub block_col: u16,
    /// Index into the caller's splat slice.
    pub splat_id: u32,
    /// Splat confidence (sort key; highest composited first).
    pub confidence: f32,
}

// ════════════════════════════════════════════════════════════════════
// TileBinning
// ════════════════════════════════════════════════════════════════════

/// Tier-aware tile binning over a framebuffer, generic over tile shape
/// (`BR×BC`, default 64×64 to match the `BlockedGrid` L1 cache fit).
///
/// Build via [`from_projected_l1`](TileBinning::from_projected_l1) (L1
/// only) or [`from_projected_cascade`](TileBinning::from_projected_cascade)
/// (all four tiers) in [`super::bin`]; query via
/// [`tile_instances`](TileBinning::tile_instances) /
/// [`l1_bins`](TileBinning::l1_bins).
#[derive(Debug, Clone)]
pub struct TileBinning<const BR: usize = 64, const BC: usize = 64> {
    /// All `(tile, splat)` bindings, sorted `(tier, block_row, block_col,
    /// confidence DESC)`.
    pub(super) instances: Vec<TileInstance>,
    /// Per-tier prefix sums into `instances` (global offsets). Entry
    /// `tier_prefix[t][lin]` is the global start index of tier-`(t+1)`
    /// tile `lin`'s run; `tier_prefix[t]` has length `n_rows·n_cols + 1`.
    pub(super) tier_prefix: [Vec<u32>; N_TIERS],
    /// `(n_block_rows, n_block_cols)` per tier.
    pub(super) tier_dims: [(u32, u32); N_TIERS],
    /// Framebuffer logical dimensions in cells.
    pub(super) fb_rows: u32,
    /// Framebuffer logical dimensions in cells.
    pub(super) fb_cols: u32,
}

impl<const BR: usize, const BC: usize> TileBinning<BR, BC> {
    /// Framebuffer dimensions `(rows, cols)` in cells.
    pub fn framebuffer_dims(&self) -> (u32, u32) {
        (self.fb_rows, self.fb_cols)
    }

    /// Tile-grid dimensions `(n_block_rows, n_block_cols)` at `tier`
    /// (1..=4). Returns `(0, 0)` for an out-of-range tier.
    pub fn tier_dims(&self, tier: u8) -> (u32, u32) {
        match (tier as usize).checked_sub(1) {
            Some(t) if t < N_TIERS => self.tier_dims[t],
            _ => (0, 0),
        }
    }

    /// Total number of `(tile, splat)` bindings across all tiers.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether any bindings were emitted.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// O(1) slice of the bindings for one tile, in confidence-descending
    /// order. Returns `&[]` for an out-of-range tier or tile coordinate.
    ///
    /// `tier ∈ 1..=4`.
    pub fn tile_instances(&self, tier: u8, block_row: u16, block_col: u16) -> &[TileInstance] {
        let Some(t) = (tier as usize).checked_sub(1) else {
            return &[];
        };
        if t >= N_TIERS {
            return &[];
        }
        let (n_rows, n_cols) = self.tier_dims[t];
        if u32::from(block_row) >= n_rows || u32::from(block_col) >= n_cols {
            return &[];
        }
        let lin = block_row as usize * n_cols as usize + block_col as usize;
        let prefix = &self.tier_prefix[t];
        // `prefix` has length n_rows*n_cols + 1, so lin+1 is in range.
        let start = prefix[lin] as usize;
        let end = prefix[lin + 1] as usize;
        &self.instances[start..end]
    }

    /// Iterate every L1 tile as `(block_row, block_col, &[TileInstance])`
    /// in row-major order, suitable for `BlockedGrid::map_l1`-style
    /// consumption. Empty tiles yield an empty slice.
    pub fn l1_bins(&self) -> impl Iterator<Item = (u16, u16, &[TileInstance])> + '_ {
        let (n_rows, n_cols) = self.tier_dims[0];
        (0..n_rows).flat_map(move |r| {
            (0..n_cols).map(move |c| {
                let br = r as u16;
                let bc = c as u16;
                (br, bc, self.tile_instances(1, br, bc))
            })
        })
    }
}

// Compile-time assurance that `TileInstance` packs into one 16-byte slot
// (tier:1 + pad + block_row:2 + block_col:2 + splat_id:4 + confidence:4).
const _: () = assert!(core::mem::size_of::<TileInstance>() == 16);

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::splat4d::{SplatCell, SplatCovariance};

    fn splat2(row: f32, col: f32, sigma2: f32, conf: f32) -> Splat<2> {
        Splat::<2>::new([row, col], SplatCovariance::isotropic(sigma2), SplatCell::default(), conf)
    }

    #[test]
    fn tile_instance_is_16_bytes() {
        assert_eq!(core::mem::size_of::<TileInstance>(), 16);
    }

    #[test]
    fn empty_binning_has_no_instances() {
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[], (128, 128));
        assert!(b.is_empty());
        assert_eq!(b.tile_instances(1, 0, 0), &[]);
        assert_eq!(b.framebuffer_dims(), (128, 128));
    }

    #[test]
    fn out_of_range_tier_and_tile_return_empty() {
        let splats = [splat2(10.0, 10.0, 1.0, 1.0)];
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&splats, (128, 128));
        assert_eq!(b.tile_instances(0, 0, 0), &[]); // tier 0 invalid
        assert_eq!(b.tile_instances(5, 0, 0), &[]); // tier 5 invalid
        assert_eq!(b.tile_instances(1, 99, 0), &[]); // tile row out of range
    }

    #[test]
    fn l1_bins_visits_every_tile_in_row_major() {
        // 128×128 fb, 64×64 tiles → 2×2 = 4 L1 tiles.
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[], (128, 128));
        let coords: Vec<(u16, u16)> = b.l1_bins().map(|(r, c, _)| (r, c)).collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }
}
