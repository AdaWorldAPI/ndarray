//! Splat composition onto `BlockedGrid` + the L1–L4 pyramid (PR-X4 A4).
//!
//! [`TileBinning::compose_l1`] rasterizes each tile's splats (confidence
//! DESC) into a fresh [`BlockedGrid`] via a caller-supplied **blend**
//! closure — the cognitive-shader spacetime evolution operator at the
//! finest scale. [`TileBinning::compose_cascade`] produces the full
//! [`SplatPyramid`]: L1 plus three successively down-pooled levels.
//!
//! # Data-flow rule (`.claude/rules/data-flow.md` Rule #3)
//!
//! Both methods are **builders**: they construct and return new grids and
//! never mutate the binning or the splat slice. The `set` calls during
//! construction are the builder/constructor exemption, not a compute-path
//! `&mut self`.
//!
//! # Blend vs pool
//!
//! - `blend: Fn(SplatCell, T) -> T` is the per-cell splat operator (v1:
//!   the caller supplies alpha-compositing or any accumulation; W7 swaps
//!   in NARS truth-revision — same signature).
//! - `pool: Fn(&[T]) -> T` down-samples a level to the next coarser one.
//!   The design doc left the pyramid down-sample op implicit; making it an
//!   explicit closure keeps composition generic over `T` (no `T: Add` /
//!   `T: From<f32>` requirement) and deterministic.
//!
//! # Distance-typing guardrail
//!
//! No `enum BlendMode` / distance dispatch — the closure boundary *is* the
//! dispatch (`.claude/knowledge/cognitive-distance-typing.md`).

use super::bin::cell_range;
use super::splat::{Splat, SplatCell};
use super::tile::TileBinning;
use crate::hpc::blocked_grid::BlockedGrid;

/// Per-side down-sample factors L1→L2→L3→L4, matching the tier strides
/// `[1, 4, 64, 256]` (successive ratios `4, 16, 4`).
const PYRAMID_FACTORS: [usize; 3] = [4, 16, 4];

/// Four-level Gaussian pyramid from [`TileBinning::compose_cascade`].
/// `l1` is the finest (full framebuffer); each coarser level is the
/// previous one down-pooled by the corresponding [`PYRAMID_FACTORS`] entry.
///
/// No `derive(Clone/Debug)`: `BlockedGrid`'s own `Clone`/`Debug` require
/// `T: Copy + Default`, which a blanket derive wouldn't add.
pub struct SplatPyramid<T, const BR: usize = 64, const BC: usize = 64> {
    /// Finest level — full framebuffer composite.
    pub l1: BlockedGrid<T, BR, BC>,
    /// L1 down-pooled 4× per side.
    pub l2: BlockedGrid<T, BR, BC>,
    /// L2 down-pooled 16× per side.
    pub l3: BlockedGrid<T, BR, BC>,
    /// L3 down-pooled 4× per side.
    pub l4: BlockedGrid<T, BR, BC>,
}

impl<const BR: usize, const BC: usize> TileBinning<BR, BC> {
    /// Compose all L1 splats into a fresh grid sized to the framebuffer.
    ///
    /// For each L1 tile, its splats are applied highest-confidence-first;
    /// each splat's 3σ footprint cells (clipped to the tile and the
    /// framebuffer) are updated by `blend(splat.cell, accumulator)`.
    ///
    /// ```
    /// use ndarray::hpc::splat4d::{Splat, SplatCovariance, SplatCell, TileBinning};
    /// // One splat at (10,10), 3σ=3 → a 7×7 footprint of cells [7,13]².
    /// let s = Splat::<2>::new([10.0, 10.0], SplatCovariance::isotropic(1.0), SplatCell::default(), 1.0);
    /// let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[s], (64, 64));
    /// // Count how many cells the splat touched, via a u32 "+1" blend.
    /// let grid = bin.compose_l1::<2, u32, _>(&[s], |_cell, acc| acc + 1);
    /// let touched = grid.as_padded_slice().iter().filter(|&&v| v == 1).count();
    /// assert_eq!(touched, 7 * 7);
    /// ```
    pub fn compose_l1<const D: usize, T, F>(&self, splats: &[Splat<D>], blend: F) -> BlockedGrid<T, BR, BC>
    where
        T: Copy + Default,
        F: Fn(SplatCell, T) -> T,
    {
        let mut grid = BlockedGrid::<T, BR, BC>::new(self.fb_rows as usize, self.fb_cols as usize);
        if self.fb_rows == 0 || self.fb_cols == 0 {
            return grid;
        }
        for (br, bc, insts) in self.l1_bins() {
            // This tile's logical cell extent (clipped to the framebuffer).
            let tile_r0 = br as usize * BR;
            let tile_c0 = bc as usize * BC;
            let tile_r1 = (tile_r0 + BR).min(self.fb_rows as usize);
            let tile_c1 = (tile_c0 + BC).min(self.fb_cols as usize);
            for inst in insts {
                let s = &splats[inst.splat_id as usize];
                let (mn, mx) = s.aabb_3sigma();
                let Some((sr_lo, sr_hi)) = cell_range(mn[0], mx[0], self.fb_rows) else {
                    continue;
                };
                let Some((sc_lo, sc_hi)) = cell_range(mn[1], mx[1], self.fb_cols) else {
                    continue;
                };
                // Intersect the splat footprint with this tile.
                let r0 = (sr_lo as usize).max(tile_r0);
                let r1 = (sr_hi as usize + 1).min(tile_r1);
                let c0 = (sc_lo as usize).max(tile_c0);
                let c1 = (sc_hi as usize + 1).min(tile_c1);
                for r in r0..r1 {
                    for c in c0..c1 {
                        let acc = grid.get(r, c);
                        grid.set(r, c, blend(s.cell, acc));
                    }
                }
            }
        }
        grid
    }

    /// Compose the full L1–L4 pyramid. `blend` builds L1 from the splats;
    /// `pool` down-samples each level to the next (factors `4, 16, 4`).
    ///
    /// ```
    /// use ndarray::hpc::splat4d::{Splat, SplatCovariance, SplatCell, TileBinning};
    /// let s = Splat::<2>::new([100.0, 100.0], SplatCovariance::isotropic(9.0), SplatCell::default(), 1.0);
    /// let bin = TileBinning::<64, 64>::from_projected_cascade::<2>(&[s], (256, 256));
    /// let pyr = bin.compose_cascade::<2, u32, _, _>(
    ///     &[s],
    ///     |_cell, acc| acc + 1,                                   // blend: count hits
    ///     |cells| cells.iter().copied().max().unwrap_or(0),       // pool: max
    /// );
    /// assert_eq!(pyr.l1.rows(), 256);
    /// assert_eq!(pyr.l2.rows(), 64);  // 256 / 4
    /// assert_eq!(pyr.l3.rows(), 4);   // 64 / 16
    /// assert_eq!(pyr.l4.rows(), 1);   // 4 / 4
    /// ```
    pub fn compose_cascade<const D: usize, T, FB, FP>(
        &self, splats: &[Splat<D>], blend: FB, pool: FP,
    ) -> SplatPyramid<T, BR, BC>
    where
        T: Copy + Default,
        FB: Fn(SplatCell, T) -> T,
        FP: Fn(&[T]) -> T,
    {
        let l1 = self.compose_l1::<D, T, _>(splats, blend);
        let l2 = downsample(&l1, PYRAMID_FACTORS[0], &pool);
        let l3 = downsample(&l2, PYRAMID_FACTORS[1], &pool);
        let l4 = downsample(&l3, PYRAMID_FACTORS[2], &pool);
        SplatPyramid { l1, l2, l3, l4 }
    }
}

/// Down-sample `grid` by `factor` per side, pooling each `factor × factor`
/// input block to one output cell via `pool`. Output dims are
/// `ceil(rows / factor) × ceil(cols / factor)`.
fn downsample<T, FP, const BR: usize, const BC: usize>(
    grid: &BlockedGrid<T, BR, BC>, factor: usize, pool: &FP,
) -> BlockedGrid<T, BR, BC>
where
    T: Copy + Default,
    FP: Fn(&[T]) -> T,
{
    let in_rows = grid.rows();
    let in_cols = grid.cols();
    let out_rows = in_rows.div_ceil(factor);
    let out_cols = in_cols.div_ceil(factor);
    let mut out = BlockedGrid::<T, BR, BC>::new(out_rows, out_cols);
    if out_rows == 0 || out_cols == 0 {
        return out;
    }
    let mut scratch: Vec<T> = Vec::with_capacity(factor * factor);
    for r in 0..out_rows {
        for c in 0..out_cols {
            scratch.clear();
            let ir1 = ((r + 1) * factor).min(in_rows);
            let ic1 = ((c + 1) * factor).min(in_cols);
            for ir in (r * factor)..ir1 {
                for ic in (c * factor)..ic1 {
                    scratch.push(grid.get(ir, ic));
                }
            }
            // `r < out_rows = ceil(in_rows/factor)` ⇒ `r*factor < in_rows`,
            // so `scratch` is always non-empty here.
            out.set(r, c, pool(&scratch));
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::splat4d::{SplatCell, SplatCovariance};

    fn splat2(row: f32, col: f32, sigma2: f32, conf: f32, vocab: u16) -> Splat<2> {
        Splat::<2>::new([row, col], SplatCovariance::isotropic(sigma2), SplatCell::new(0, [0; 16], [0; 8], vocab), conf)
    }

    #[test]
    fn compose_l1_empty_is_default_grid() {
        let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[], (64, 64));
        let grid = bin.compose_l1::<2, u32, _>(&[], |_c, acc| acc + 1);
        assert!(grid.as_padded_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn compose_l1_counts_footprint_cells() {
        // 3σ=3 → 7×7 footprint, fully inside a 64×64 grid.
        let s = splat2(10.0, 10.0, 1.0, 1.0, 0);
        let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[s], (64, 64));
        let grid = bin.compose_l1::<2, u32, _>(&[s], |_c, acc| acc + 1);
        let touched = grid.as_padded_slice().iter().filter(|&&v| v == 1).count();
        assert_eq!(touched, 49);
        // Centre cell touched exactly once.
        assert_eq!(grid.get(10, 10), 1);
    }

    #[test]
    fn compose_l1_blend_sees_cell_payload() {
        // Blend writes the splat's vocab into each covered cell.
        let s = splat2(5.0, 5.0, 1.0, 1.0, 0x2A);
        let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[s], (64, 64));
        let grid = bin.compose_l1::<2, u16, _>(&[s], |cell, _acc| cell.vocab);
        assert_eq!(grid.get(5, 5), 0x2A);
    }

    #[test]
    fn compose_l1_no_op_blend_leaves_default() {
        let s = splat2(5.0, 5.0, 1.0, 1.0, 7);
        let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&[s], (64, 64));
        let grid = bin.compose_l1::<2, u32, _>(&[s], |_c, acc| acc);
        assert!(grid.as_padded_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn compose_l1_highest_confidence_composited_first() {
        // Two overlapping splats in one tile; blend records the LAST writer
        // (acc-overwrite). Confidence-DESC order means the LOWER-confidence
        // splat writes last and therefore wins the final value.
        let hi = splat2(10.0, 10.0, 1.0, 0.9, 100);
        let lo = splat2(10.0, 10.0, 1.0, 0.1, 200);
        let splats = [hi, lo];
        let bin = TileBinning::<64, 64>::from_projected_l1::<2>(&splats, (64, 64));
        let grid = bin.compose_l1::<2, u16, _>(&splats, |cell, _acc| cell.vocab);
        // hi (0.9) composited first, lo (0.1) second → final = lo.vocab.
        assert_eq!(grid.get(10, 10), 200);
    }

    #[test]
    fn compose_cascade_l1_matches_standalone_compose_l1() {
        let s = splat2(50.0, 50.0, 9.0, 1.0, 0);
        let bin = TileBinning::<64, 64>::from_projected_cascade::<2>(&[s], (256, 256));
        let standalone = bin.compose_l1::<2, u32, _>(&[s], |_c, acc| acc + 1);
        let pyr = bin.compose_cascade::<2, u32, _, _>(
            &[s],
            |_c, acc| acc + 1,
            |cells| cells.iter().copied().max().unwrap_or(0),
        );
        assert_eq!(pyr.l1.as_padded_slice(), standalone.as_padded_slice());
    }

    #[test]
    fn compose_cascade_level_dims_follow_factors() {
        let bin = TileBinning::<64, 64>::from_projected_cascade::<2>(&[], (256, 256));
        let pyr =
            bin.compose_cascade::<2, u32, _, _>(&[], |_c, acc| acc, |cells| cells.iter().copied().max().unwrap_or(0));
        assert_eq!((pyr.l1.rows(), pyr.l1.cols()), (256, 256));
        assert_eq!((pyr.l2.rows(), pyr.l2.cols()), (64, 64)); // /4
        assert_eq!((pyr.l3.rows(), pyr.l3.cols()), (4, 4)); // /16
        assert_eq!((pyr.l4.rows(), pyr.l4.cols()), (1, 1)); // /4
    }

    #[test]
    fn compose_cascade_max_pool_propagates_hit() {
        // A single splat hit at L1 → max-pool keeps a 1 at every coarser level.
        let s = splat2(50.0, 50.0, 4.0, 1.0, 0);
        let bin = TileBinning::<64, 64>::from_projected_cascade::<2>(&[s], (256, 256));
        let pyr = bin.compose_cascade::<2, u32, _, _>(
            &[s],
            |_c, acc| acc + 1,
            |cells| cells.iter().copied().max().unwrap_or(0),
        );
        assert_eq!(pyr.l1.get(50, 50), 1);
        // Max-pool down to L4 (a single cell) preserves the hit.
        assert_eq!(pyr.l4.get(0, 0), 1);
    }
}
