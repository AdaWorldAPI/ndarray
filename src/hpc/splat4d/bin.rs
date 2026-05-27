//! Tile-binning constructors for [`TileBinning`] (PR-X4 A2).
//!
//! Each splat is inserted into every tile (at the requested tiers) whose
//! cell-space tile box intersects the splat's 3σ AABB. Tier `k`'s tile
//! pitch is `TIER_STRIDES[k-1] · (BR×BC)` cells, so coarser tiers have
//! fewer, larger tiles — a multi-resolution tile grid over the same
//! framebuffer. After collection the bindings are sorted
//! `(tier, block_row, block_col, confidence DESC)` and indexed by per-tier
//! prefix sums.
//!
//! Per-tier covariance *broadening* (the Gaussian-pyramid effect) is a
//! composition-time concern ([`super::compose`]), not a binning-extent
//! one: binning uses the splat's physical 3σ footprint at every tier.

use super::splat::Splat;
use super::tile::{TileBinning, TileInstance, N_TIERS, TIER_STRIDES};

/// Inclusive `[lo, hi]` cell range covered by `[min_f, max_f]` clamped to
/// `[0, dim)`. `None` if the interval lies fully outside the grid.
#[inline]
pub(super) fn cell_range(min_f: f32, max_f: f32, dim: u32) -> Option<(u32, u32)> {
    if dim == 0 || max_f < 0.0 || min_f > (dim - 1) as f32 {
        return None;
    }
    let lo = min_f.max(0.0).floor() as u32;
    let hi = max_f.min((dim - 1) as f32).floor() as u32;
    if lo > hi {
        return None;
    }
    Some((lo, hi))
}

impl<const BR: usize, const BC: usize> TileBinning<BR, BC> {
    /// Bin `splats` into L1 tiles only over a `framebuffer` of `(rows,
    /// cols)` cells. Backward-compatible single-tier path.
    ///
    /// Requires `D ≥ 2` (axes 0 and 1 are the cell-space row/col).
    ///
    /// ```
    /// use ndarray::hpc::splat4d::{Splat, SplatCovariance, SplatCell, TileBinning};
    /// let s = Splat::<2>::new([50.0, 50.0], SplatCovariance::isotropic(4.0), SplatCell::default(), 1.0);
    /// let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[s], (128, 128));
    /// // Centre (50,50), 3σ=6 → spans rows/cols [44,56], all inside tile (0,0).
    /// assert_eq!(bin.tile_instances(1, 0, 0).len(), 1);
    /// assert_eq!(bin.tile_instances(1, 0, 1).len(), 0);
    /// ```
    pub fn from_projected_l1<const D: usize>(splats: &[Splat<D>], framebuffer: (u32, u32)) -> Self {
        Self::build::<D>(splats, framebuffer, &[1])
    }

    /// Bin `splats` into all four tiers (L1..L4) over `framebuffer`.
    /// The same splat appears in each tier's overlapping tiles.
    ///
    /// Requires `D ≥ 2`.
    pub fn from_projected_cascade<const D: usize>(splats: &[Splat<D>], framebuffer: (u32, u32)) -> Self {
        Self::build::<D>(splats, framebuffer, &[1, 2, 3, 4])
    }

    /// Shared binning core. `tiers` lists the 1-indexed tiers to emit.
    fn build<const D: usize>(splats: &[Splat<D>], framebuffer: (u32, u32), tiers: &[u8]) -> Self {
        const { assert!(D >= 2, "TileBinning: binning needs D >= 2 (row=axis 0, col=axis 1)") };
        let (fb_rows, fb_cols) = framebuffer;

        // Per-tier tile-grid dimensions.
        let mut tier_dims = [(0u32, 0u32); N_TIERS];
        if fb_rows != 0 && fb_cols != 0 {
            for (t, dims) in tier_dims.iter_mut().enumerate() {
                let pitch_r = TIER_STRIDES[t] * BR as u32;
                let pitch_c = TIER_STRIDES[t] * BC as u32;
                *dims = (fb_rows.div_ceil(pitch_r), fb_cols.div_ceil(pitch_c));
            }
        }

        // Emit one TileInstance per (requested tier, overlapping tile, splat).
        let mut instances: Vec<TileInstance> = Vec::new();
        for &tier in tiers {
            let t = tier as usize - 1;
            let pitch_r = TIER_STRIDES[t] * BR as u32;
            let pitch_c = TIER_STRIDES[t] * BC as u32;
            for (id, s) in splats.iter().enumerate() {
                let (mn, mx) = s.aabb_3sigma();
                let Some((r_lo, r_hi)) = cell_range(mn[0], mx[0], fb_rows) else {
                    continue;
                };
                let Some((c_lo, c_hi)) = cell_range(mn[1], mx[1], fb_cols) else {
                    continue;
                };
                let (tr_lo, tr_hi) = (r_lo / pitch_r, r_hi / pitch_r);
                let (tc_lo, tc_hi) = (c_lo / pitch_c, c_hi / pitch_c);
                // Tile indices are stored as u16 on TileInstance. The cap is
                // ~u16::MAX tiles/side = fb side > 65535·BR (≈ 4.19M cells for
                // BR=64) — orders of magnitude past any allocatable grid and
                // far past the L4 = 16384 cognitive-cascade extent. The
                // debug_assert makes the invariant explicit rather than letting
                // an `as u16` silently wrap into the wrong bin.
                debug_assert!(
                    tr_hi <= u16::MAX as u32 && tc_hi <= u16::MAX as u32,
                    "splat4d: tile index exceeds u16 (framebuffer too large for the binner)"
                );
                for br in tr_lo..=tr_hi {
                    for bc in tc_lo..=tc_hi {
                        instances.push(TileInstance {
                            tier,
                            block_row: br as u16,
                            block_col: bc as u16,
                            splat_id: id as u32,
                            confidence: s.confidence,
                        });
                    }
                }
            }
        }

        // Sort (tier, block_row, block_col, confidence DESC). Ordering by
        // (block_row, block_col) is monotone with the linear tile index, so
        // the per-tier prefix build below sees a lin-ascending run.
        instances.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then(a.block_row.cmp(&b.block_row))
                .then(a.block_col.cmp(&b.block_col))
                .then(b.confidence.total_cmp(&a.confidence))
        });

        // Per-tier prefix sums as GLOBAL offsets into `instances`.
        let mut tier_prefix: [Vec<u32>; N_TIERS] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut cursor = 0usize;
        for (t, prefix_slot) in tier_prefix.iter_mut().enumerate() {
            let (n_rows, n_cols) = tier_dims[t];
            let n_cells = n_rows as usize * n_cols as usize;
            let tier_start = cursor;
            let mut counts = vec![0u32; n_cells];
            while cursor < instances.len() && instances[cursor].tier as usize == t + 1 {
                let inst = instances[cursor];
                let lin = inst.block_row as usize * n_cols as usize + inst.block_col as usize;
                counts[lin] += 1;
                cursor += 1;
            }
            let mut prefix = vec![0u32; n_cells + 1];
            let mut acc = tier_start as u32;
            for (l, &cnt) in counts.iter().enumerate() {
                prefix[l] = acc;
                acc += cnt;
            }
            prefix[n_cells] = acc;
            *prefix_slot = prefix;
        }

        Self {
            instances,
            tier_prefix,
            tier_dims,
            fb_rows,
            fb_cols,
        }
    }
}

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
    fn single_splat_lands_in_one_l1_tile() {
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[splat2(50.0, 50.0, 4.0, 1.0)], (128, 128));
        assert_eq!(b.tile_instances(1, 0, 0).len(), 1);
        assert_eq!(b.tile_instances(1, 0, 1).len(), 0);
        assert_eq!(b.tile_instances(1, 1, 0).len(), 0);
    }

    #[test]
    fn splat_spanning_column_boundary_lands_in_both_tiles() {
        // Centre at col 64 (the tile-0/tile-1 boundary), 3σ = 6 → cols [58, 70]
        // straddles tiles (0,0) and (0,1).
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[splat2(50.0, 64.0, 4.0, 1.0)], (128, 128));
        assert_eq!(b.tile_instances(1, 0, 0).len(), 1);
        assert_eq!(b.tile_instances(1, 0, 1).len(), 1);
    }

    #[test]
    fn splat_fully_outside_framebuffer_is_dropped() {
        // Centre at (-100, -100), 3σ small → AABB entirely negative.
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[splat2(-100.0, -100.0, 1.0, 1.0)], (128, 128));
        assert!(b.is_empty());
    }

    #[test]
    fn tile_instances_sorted_by_confidence_desc() {
        // Three splats in the same tile (0,0) with ascending confidence;
        // the bin must return them highest-first.
        let splats = [splat2(10.0, 10.0, 1.0, 0.2), splat2(12.0, 12.0, 1.0, 0.9), splat2(14.0, 14.0, 1.0, 0.5)];
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&splats, (128, 128));
        let confs: Vec<f32> = b
            .tile_instances(1, 0, 0)
            .iter()
            .map(|i| i.confidence)
            .collect();
        assert_eq!(confs, vec![0.9, 0.5, 0.2]);
    }

    #[test]
    fn cascade_bins_same_splat_into_every_tier() {
        // A 16384×16384 fb has all four tiers (L1 pitch 64, L4 pitch 16384).
        // The splat centre is cell 100 on both axes; at tier k it lands in
        // tile (100 / pitch_k): L1 pitch 64 → tile 1, coarser tiers → tile 0.
        let b = TileBinning::<64, 64>::from_projected_cascade::<2>(&[splat2(100.0, 100.0, 4.0, 1.0)], (16384, 16384));
        for (tier, stride) in [(1u8, 1u32), (2, 4), (3, 64), (4, 256)] {
            let t = (100 / (stride * 64)) as u16;
            assert_eq!(b.tile_instances(tier, t, t).len(), 1, "tier {tier} should hold the splat at tile ({t},{t})");
        }
    }

    #[test]
    fn cascade_tier_dims_follow_strides() {
        let b = TileBinning::<64, 64>::from_projected_cascade::<2>(&[], (16384, 16384));
        assert_eq!(b.tier_dims(1), (256, 256)); // 16384 / 64
        assert_eq!(b.tier_dims(2), (64, 64)); // / 256
        assert_eq!(b.tier_dims(3), (4, 4)); // / 4096
        assert_eq!(b.tier_dims(4), (1, 1)); // / 16384
    }

    #[test]
    fn l1_only_leaves_higher_tiers_empty() {
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[splat2(50.0, 50.0, 4.0, 1.0)], (256, 256));
        assert_eq!(b.tile_instances(1, 0, 0).len(), 1);
        assert_eq!(b.tile_instances(2, 0, 0).len(), 0);
    }

    #[test]
    fn total_instance_count_matches_tiles_touched() {
        // Splat spanning a 2×2 block of L1 tiles → 4 instances.
        // Centre (64,64), 3σ=12 → rows/cols [52,76] → tiles row{0,1}×col{0,1}.
        let b = TileBinning::<64, 64>::from_projected_l1::<2>(&[splat2(64.0, 64.0, 16.0, 1.0)], (256, 256));
        assert_eq!(b.len(), 4);
    }
}
