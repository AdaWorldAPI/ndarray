//! 16×16 tile binner — bridge between [`ProjectedBatch`] (PR 3) and the
//! per-tile rasterizer (PR 5).
//!
//! # Mathematical claim
//!
//! For each visible projected gaussian (`valid[i] == 1`), the 3σ
//! screen-space bounding circle has radius `r = projected.radius[i]`.
//! Its AABB in pixel space is `[cx − r, cx + r] × [cy − r, cy + r]`.
//! Every 16×16 tile whose pixel extent overlaps that AABB receives one
//! [`TileInstance`] binding.
//!
//! # Depth sort invariant
//!
//! [`TileInstance::depth_bits`] stores `depth.to_bits()` — the raw IEEE-754
//! bit pattern of a **positive** f32. PR 3 guarantees `depth > 0` for every
//! valid gaussian (near-clip is `> 0`), so positive f32 values sort
//! identically as u32 bit patterns. The packed u64 key
//! `(tile_id as u64) << 32 | (depth_bits as u64)` therefore sorts instances
//! tile-major and depth-ascending within each tile, which is the
//! front-to-back order the alpha-blend in PR 5 requires.
//!
//! # Algorithm
//!
//! 1. Compute tile grid dimensions (ceil-div of image dimensions by
//!    [`TILE_SIZE`]).
//! 2. Pass 1 — count: for each visible gaussian, compute the tile AABB and
//!    accumulate the total number of (tile, gaussian) pairs.
//! 3. Allocate the instance `Vec` with exact capacity.
//! 4. Pass 2 — emit: walk each visible gaussian's tile AABB and push one
//!    [`TileInstance`] per touched tile.
//! 5. Sort the instance list by packed u64 key (tile_id major, depth
//!    ascending within tile) using `slice::sort_unstable_by_key`.
//! 6. Build the prefix-sum [`TileBinning::tile_offsets`] table (length
//!    `n_tiles + 1`) via a single scan of the sorted instance list.

use super::project::{Camera, ProjectedBatch};

// ════════════════════════════════════════════════════════════════════════════
// Constants + core types
// ════════════════════════════════════════════════════════════════════════════

/// Pixel side length of one tile.
pub const TILE_SIZE: u32 = 16;

/// One (tile, gaussian) binding emitted during binning.
///
/// Layout: `#[repr(C, align(16))]` — 16 bytes per instance, so 4 instances
/// fit one 64-byte cache line. Fields must not be reordered.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileInstance {
    /// Linear tile index: `tile_y * tile_cols + tile_x`.
    pub tile_id: u32,
    /// Index of the gaussian within [`ProjectedBatch`].
    pub gaussian_id: u32,
    /// Raw IEEE-754 bit pattern of `projected.depth[gaussian_id]`.
    ///
    /// Positive f32 values are monotonically ordered by their u32 bit
    /// pattern (IEEE-754 guarantee), so sorting by this field gives
    /// depth-ascending order. PR 3 guarantees `depth > 0`.
    pub depth_bits: u32,
    /// Padding to reach 16 bytes; always zero.
    pub _pad: u32,
}

/// Output of binning: sorted instance list + per-tile prefix-sum index.
///
/// Use [`TileBinning::from_projected`] to construct, and
/// [`TileBinning::tile_instances`] for O(1) per-tile slice access.
pub struct TileBinning {
    /// Number of tiles along the image X axis (ceil-div of width by [`TILE_SIZE`]).
    pub tile_cols: u32,
    /// Number of tiles along the image Y axis (ceil-div of height by [`TILE_SIZE`]).
    pub tile_rows: u32,
    /// All (tile, gaussian) instances, sorted by
    /// `(tile_id << 32) | depth_bits` — tile_id major, depth ascending
    /// within each tile.
    pub instances: Vec<TileInstance>,
    /// Prefix-sum offset table; length = `tile_cols * tile_rows + 1`.
    ///
    /// Tile `t` owns `instances[tile_offsets[t]..tile_offsets[t+1]]`.
    /// Empty tiles have `tile_offsets[t] == tile_offsets[t+1]`.
    pub tile_offsets: Vec<u32>,
}

// ════════════════════════════════════════════════════════════════════════════
// Ceil-div helper
// ════════════════════════════════════════════════════════════════════════════

#[inline]
const fn ceil_div(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

// ════════════════════════════════════════════════════════════════════════════
// TileBinning implementation
// ════════════════════════════════════════════════════════════════════════════

impl TileBinning {
    /// Bin all visible gaussians into the 16×16 tile grid.
    ///
    /// Only gaussians with `projected.valid[i] == 1` are processed.
    /// Each such gaussian contributes one [`TileInstance`] for every
    /// 16×16 tile overlapped by its 3σ screen-space bounding circle.
    pub fn from_projected(projected: &ProjectedBatch, camera: &Camera) -> Self {
        let tile_cols = ceil_div(camera.width, TILE_SIZE);
        let tile_rows = ceil_div(camera.height, TILE_SIZE);
        let n_tiles = (tile_cols * tile_rows) as usize;

        // ── Pass 1: count total instances ────────────────────────────────
        let mut total: usize = 0;
        for i in 0..projected.len {
            if projected.valid[i] == 0 {
                continue;
            }
            let (tx_min, tx_max, ty_min, ty_max) =
                tile_aabb(projected, i, tile_cols, tile_rows);
            let w = tx_max.saturating_sub(tx_min) as usize;
            let h = ty_max.saturating_sub(ty_min) as usize;
            total += w * h;
        }

        // ── Pass 2: emit instances ────────────────────────────────────────
        let mut instances: Vec<TileInstance> = Vec::with_capacity(total);
        for i in 0..projected.len {
            if projected.valid[i] == 0 {
                continue;
            }
            // The IEEE-754 positive-f32 → u32 sort trick requires
            // `depth > 0`. PR 3's near cull guarantees this for any
            // `valid == 1` slot; the debug_assert catches a caller
            // that violates the precondition (which would silently
            // produce wrong sort order in release builds).
            debug_assert!(
                projected.depth[i] > 0.0 && projected.depth[i].is_finite(),
                "tile binning requires positive finite depth for valid gaussians \
                 (got {} at slot {i}); PR 3's near cull must filter these out",
                projected.depth[i]
            );
            let depth_bits = projected.depth[i].to_bits();
            let (tx_min, tx_max, ty_min, ty_max) =
                tile_aabb(projected, i, tile_cols, tile_rows);
            for ty in ty_min..ty_max {
                for tx in tx_min..tx_max {
                    instances.push(TileInstance {
                        tile_id: ty * tile_cols + tx,
                        gaussian_id: i as u32,
                        depth_bits,
                        _pad: 0,
                    });
                }
            }
        }

        // ── Sort by packed u64 key: tile_id major, depth ascending ────────
        instances.sort_unstable_by_key(|inst| {
            ((inst.tile_id as u64) << 32) | (inst.depth_bits as u64)
        });

        // ── Build prefix-sum offset table ─────────────────────────────────
        let mut tile_offsets: Vec<u32> = vec![0u32; n_tiles + 1];
        for (idx, inst) in instances.iter().enumerate() {
            // +1 so that after the final pass tile_offsets[t+1] holds end
            // We will convert to proper prefix-sum below.
            let t = inst.tile_id as usize;
            // Use tile_offsets[t+1] as a count first, then prefix-sum.
            tile_offsets[t + 1] += 1;
        }
        // Convert counts to prefix sums
        for t in 0..n_tiles {
            tile_offsets[t + 1] += tile_offsets[t];
        }

        Self {
            tile_cols,
            tile_rows,
            instances,
            tile_offsets,
        }
    }

    /// Return the sorted slice of instances for tile `(tile_x, tile_y)`,
    /// in front-to-back depth order (the contract PR 5's rasterizer
    /// alpha-blend expects).
    ///
    /// # Returns
    ///
    /// - **Empty slice** if the tile contains no visible gaussians.
    /// - **Empty slice** for out-of-range coordinates
    ///   (`tile_x >= tile_cols` or `tile_y >= tile_rows`). Silent —
    ///   no panic, no `Result`. Callers iterating the full grid in
    ///   nested loops with their own bounds don't pay a branch cost.
    ///   PR 5's per-tile driver iterates `0..tile_rows × 0..tile_cols`
    ///   so the out-of-range path is defensive only.
    pub fn tile_instances(&self, tile_x: u32, tile_y: u32) -> &[TileInstance] {
        if tile_x >= self.tile_cols || tile_y >= self.tile_rows {
            return &[];
        }
        let t = (tile_y * self.tile_cols + tile_x) as usize;
        let start = self.tile_offsets[t] as usize;
        let end = self.tile_offsets[t + 1] as usize;
        &self.instances[start..end]
    }

    /// Total number of (tile, gaussian) instance pairs across all tiles.
    pub fn total_instances(&self) -> usize {
        self.instances.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Private helper — tile AABB for gaussian i
// ════════════════════════════════════════════════════════════════════════════

/// Compute the clamped tile-space AABB `(tx_min, tx_max, ty_min, ty_max)`
/// for gaussian `i`. Ranges are half-open `[min, max)`. If the AABB is
/// entirely outside the grid, `tx_max <= tx_min` or `ty_max <= ty_min`
/// (caller checks with `saturating_sub` → 0 width/height → no tiles emitted).
#[inline]
fn tile_aabb(
    projected: &ProjectedBatch,
    i: usize,
    tile_cols: u32,
    tile_rows: u32,
) -> (u32, u32, u32, u32) {
    let cx = projected.screen_x[i];
    let cy = projected.screen_y[i];
    let r  = projected.radius[i];

    // Pixel-space extent, then convert to tile coordinates.
    let px_min = cx - r;
    let px_max = cx + r;
    let py_min = cy - r;
    let py_max = cy + r;

    // Tile coordinates: lowest tile is `floor(px_min / ts)`. Highest
    // tile is `floor(px_max / ts)`; the exclusive upper bound is then
    // `floor(px_max / ts) + 1`.
    //
    // The naive `ceil(px_max / ts)` would under-count by ONE TILE when
    // `px_max` is an exact multiple of `TILE_SIZE` (so `ceil == floor`).
    // Example: cx=88, r=8 → px_max=96. ceil(96/16) = 6, range [_, 6).
    // But pixel 96 sits in tile 6 (floor(96/16) = 6), so tile 6 must
    // be in the binning — under the ceil formula it is missed,
    // producing a one-pixel rendering seam on every gaussian whose
    // 3σ edge lands on a tile boundary (PP-13 PR 4 P0 finding).
    // Using `floor + 1` is monotonic and includes the boundary tile.
    let ts = TILE_SIZE as f32;
    let tx_min_f = (px_min / ts).floor();
    let tx_max_f = (px_max / ts).floor() + 1.0;
    let ty_min_f = (py_min / ts).floor();
    let ty_max_f = (py_max / ts).floor() + 1.0;

    // Clamp to valid tile range [0, tile_cols] / [0, tile_rows].
    // Using saturating cast: negative floats → 0 (via max 0.0 before cast).
    let tx_min = (tx_min_f.max(0.0) as u32).min(tile_cols);
    let tx_max = (tx_max_f.max(0.0) as u32).min(tile_cols);
    let ty_min = (ty_min_f.max(0.0) as u32).min(tile_rows);
    let ty_max = (ty_max_f.max(0.0) as u32).min(tile_rows);

    (tx_min, tx_max, ty_min, ty_max)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::project::{Camera, ProjectedBatch};

    /// Build a minimal `ProjectedBatch` from a list of
    /// `(screen_x, screen_y, radius, depth)` tuples, all valid.
    /// The optional `valid_flags` vec overrides the default (all 1).
    fn make_projected(
        gaussians: &[(f32, f32, f32, f32)],
        valid_flags: Option<&[u8]>,
    ) -> ProjectedBatch {
        let n = gaussians.len();
        let mut p = ProjectedBatch::with_capacity(n.max(1));
        p.len = n;
        for (i, &(sx, sy, r, d)) in gaussians.iter().enumerate() {
            p.screen_x[i] = sx;
            p.screen_y[i] = sy;
            p.radius[i]   = r;
            p.depth[i]    = d;
            p.valid[i]    = valid_flags.map(|f| f[i]).unwrap_or(1);
        }
        p
    }

    // ── Test 1 ───────────────────────────────────────────────────────────────

    #[test]
    fn tile_size_is_16() {
        assert_eq!(TILE_SIZE, 16);
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────

    #[test]
    fn tile_grid_dims_match_image_ceildiv() {
        let camera = Camera::identity_at_origin(1920, 1080);
        let projected = ProjectedBatch::with_capacity(1); // empty (len=0)
        let binning = TileBinning::from_projected(&projected, &camera);

        assert_eq!(binning.tile_cols, 120);  // ceil(1920/16)
        assert_eq!(binning.tile_rows, 68);   // ceil(1080/16)
        assert_eq!(binning.instances.len(), 0);
        assert_eq!(binning.tile_offsets.len(), 120 * 68 + 1);
        assert!(binning.tile_offsets.iter().all(|&o| o == 0));
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────

    #[test]
    fn single_gaussian_on_tile_boundary_touches_one_tile() {
        // screen_x=8, screen_y=8, radius=4 → AABB [4,12]×[4,12] → tile (0,0)
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(8.0, 8.0, 4.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);

        assert_eq!(binning.tile_instances(0, 0).len(), 1,
            "tile (0,0) should have 1 instance");

        // All other tiles must be empty.
        for ty in 0..binning.tile_rows {
            for tx in 0..binning.tile_cols {
                if tx == 0 && ty == 0 { continue; }
                assert_eq!(
                    binning.tile_instances(tx, ty).len(), 0,
                    "tile ({tx},{ty}) should be empty"
                );
            }
        }
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────

    #[test]
    fn large_gaussian_touches_multiple_tiles() {
        // screen_x=256, screen_y=256, radius=50
        // pixel AABB: [206,306]×[206,306]
        // tile x: floor(206/16)=12 ..= ceil(306/16)=20  → 12..20 (width 8)
        // tile y: 12..20 (height 8) → 8×8 = 64 tiles? Let's compute:
        // floor(206/16) = floor(12.875) = 12
        // ceil(306/16)  = ceil(19.125)  = 20
        // range [12,20) = 8 tiles wide, 8 tiles tall → 64 instances
        // But task says 7×7=49. Let me re-read: AABB [206,306] covers tiles
        // x ∈ [206/16, 306/16] = [12.875, 19.125]
        // floor(12.875)=12, ceil(19.125)=20, so tx in [12,20) = 8 tiles
        // Similarly ty in [12,20) = 8 tiles → 64 instances.
        // The task spec says "tiles x ∈ [12, 19]" which looks like inclusive,
        // i.e. 8 tiles. Let's verify with actual computation: 8×8=64.
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(256.0, 256.0, 50.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);

        // Count expected tiles:
        // px_min=206, px_max=306
        // tx_min=floor(206/16)=12, tx_max=ceil(306/16)=ceil(19.125)=20
        // 8 tiles wide, 8 tiles tall → 64 total
        let expected_count = 8 * 8_usize; // 64
        assert_eq!(binning.instances.len(), expected_count,
            "expected {expected_count} instances for 50-radius gaussian");

        // Build set of covered tiles from instances
        use std::collections::HashSet;
        let tile_cols = binning.tile_cols;
        let covered: HashSet<(u32, u32)> = binning.instances.iter()
            .map(|inst| (inst.tile_id % tile_cols, inst.tile_id / tile_cols))
            .collect();

        // All tiles in [12..20) × [12..20) must be covered
        for ty in 12u32..20 {
            for tx in 12u32..20 {
                assert!(covered.contains(&(tx, ty)),
                    "tile ({tx},{ty}) should be covered");
            }
        }
        assert_eq!(covered.len(), expected_count);
    }

    // ── Test 5 ───────────────────────────────────────────────────────────────

    #[test]
    fn depth_sorted_within_tile() {
        // 3 gaussians all fully inside tile (5,5):
        // tile (5,5) covers pixels [80,96)×[80,96), centre at 88.
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(
            &[
                (88.0, 88.0, 4.0, 3.0),  // gaussian 0, depth 3
                (88.0, 88.0, 4.0, 1.0),  // gaussian 1, depth 1
                (88.0, 88.0, 4.0, 2.0),  // gaussian 2, depth 2
            ],
            None,
        );
        let binning = TileBinning::from_projected(&projected, &camera);

        let slice = binning.tile_instances(5, 5);
        assert_eq!(slice.len(), 3);

        // depth_bits must be in ascending order
        assert_eq!(slice[0].depth_bits, 1.0_f32.to_bits());
        assert_eq!(slice[1].depth_bits, 2.0_f32.to_bits());
        assert_eq!(slice[2].depth_bits, 3.0_f32.to_bits());
    }

    // ── Test 6 ───────────────────────────────────────────────────────────────

    #[test]
    fn empty_tile_returns_empty_slice() {
        // Push 1 gaussian into tile (5,5) only — tile (0,0) must be empty.
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(88.0, 88.0, 4.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);

        assert_eq!(binning.tile_instances(0, 0).len(), 0);

        // Offsets consistency: everything before tile (5,5) should be 0
        let tile_55 = 5 * binning.tile_cols + 5;
        assert_eq!(binning.tile_offsets[0], 0);
        assert_eq!(
            binning.tile_offsets[0],
            binning.tile_offsets[tile_55 as usize],
            "no instances should land before tile (5,5)"
        );
    }

    // ── Test 7 ───────────────────────────────────────────────────────────────

    #[test]
    fn culled_gaussians_not_binned() {
        let camera = Camera::identity_at_origin(512, 512);
        // gaussian 0: valid=0 (culled), gaussian 1: valid=1
        let projected = make_projected(
            &[
                (88.0, 88.0, 4.0, 1.0),  // gaussian 0 — will be culled
                (88.0, 88.0, 4.0, 2.0),  // gaussian 1 — valid
            ],
            Some(&[0, 1]),
        );
        let binning = TileBinning::from_projected(&projected, &camera);

        // Only gaussian_id=1 should appear
        assert!(binning.instances.iter().all(|inst| inst.gaussian_id == 1),
            "only gaussian 1 (valid) should be in the instances");

        // At least 1 instance emitted for gaussian 1
        let count_g1 = binning.instances.len();
        assert!(count_g1 > 0, "gaussian 1 should produce at least 1 instance");
    }

    // ── Test 8 ───────────────────────────────────────────────────────────────

    #[test]
    fn aabb_clamped_to_grid_boundaries() {
        // screen_x=0, screen_y=0, radius=100 on 512×512
        // pixel AABB: [-100,100]×[-100,100]
        // after clamping to [0,512]: [0,100]×[0,100]
        // tile x: [0, ceil(100/16)) = [0, 7) = 7 tiles wide
        // tile y: [0, 7) = 7 tiles tall → 7×7 = 49 tiles
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(0.0, 0.0, 100.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);

        // ceil(100/16) = ceil(6.25) = 7
        let expected = 7 * 7_usize;
        assert_eq!(binning.instances.len(), expected,
            "clamped AABB should give 7×7=49 tiles");

        // All instances should have tile coordinates in [0..7)×[0..7)
        let tile_cols = binning.tile_cols;
        for inst in &binning.instances {
            let tx = inst.tile_id % tile_cols;
            let ty = inst.tile_id / tile_cols;
            assert!(tx < 7 && ty < 7,
                "tile ({tx},{ty}) is outside expected [0..7)×[0..7)");
        }
    }

    // ── Test 9 ───────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_outside_image_not_binned() {
        // screen_x=1000, screen_y=1000, radius=50 on 512×512
        // pixel AABB: [950,1050]×[950,1050] — entirely outside [0,512]
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(1000.0, 1000.0, 50.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);

        assert_eq!(binning.instances.len(), 0,
            "off-screen gaussian should produce zero instances");
    }

    // ── Test 10 ──────────────────────────────────────────────────────────────

    #[test]
    fn tile_offsets_monotonically_non_decreasing() {
        // Build 50 gaussians scattered across a 1024×1024 image
        let camera = Camera::identity_at_origin(1024, 1024);
        let gaussians: Vec<(f32, f32, f32, f32)> = (0..50)
            .map(|i| {
                let x = (i as f32) * 20.0 + 10.0;
                let y = (i as f32) * 15.0 + 8.0;
                (x, y, 12.0, i as f32 + 1.0)
            })
            .collect();
        let projected = make_projected(&gaussians, None);
        let binning = TileBinning::from_projected(&projected, &camera);

        let n_tiles = (binning.tile_cols * binning.tile_rows) as usize;
        assert_eq!(binning.tile_offsets.len(), n_tiles + 1);

        // Monotonically non-decreasing
        for t in 0..n_tiles {
            assert!(
                binning.tile_offsets[t] <= binning.tile_offsets[t + 1],
                "tile_offsets[{t}]={} > tile_offsets[{}]={}",
                binning.tile_offsets[t], t + 1, binning.tile_offsets[t + 1]
            );
        }

        // All offsets ≤ instances.len()
        let inst_len = binning.instances.len() as u32;
        assert!(
            binning.tile_offsets.iter().all(|&o| o <= inst_len),
            "some offset exceeds instances.len()"
        );
    }

    // ── Test 11 — exact-tile-boundary edge case (PP-13 PR4 P0 promoted) ────
    //
    // When the 3σ pixel extent `px_max = cx + r` is an EXACT multiple
    // of TILE_SIZE, the old `ceil(px_max/16)` formula under-counted
    // by one tile: a gaussian whose right edge lands at pixel 96.0
    // sits in tile 6 (floor(96/16) = 6), but ceil gave the exclusive
    // upper bound as 6 → tile 6 was missed in the binning → PR 5
    // would render a one-pixel seam along the tile boundary for
    // every gaussian that happens to hit this case.
    //
    // The `floor + 1` fix is monotonic across the boundary AND
    // backwards-compatible with the existing tests (which all use
    // non-multiple-of-16 px_max values). This test pins the corner
    // case explicitly so a future "optimization" doesn't regress.
    #[test]
    fn gaussian_edge_on_exact_tile_boundary_includes_the_boundary_tile() {
        // cx = 88, r = 8 → px range [80.0, 96.0]. px_min = 80 = 5·16,
        // px_max = 96 = 6·16. Tile range:
        //   tx_min = floor(80/16) = 5
        //   tx_max = floor(96/16) + 1 = 7  (exclusive)
        // Covered tiles: {5, 6}. Two tiles per axis, so 2×2 = 4 instances.
        let camera = Camera::identity_at_origin(512, 512);
        let projected = make_projected(&[(88.0, 88.0, 8.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);
        assert_eq!(
            binning.instances.len(), 4,
            "exact-boundary gaussian: expected 4 instances (tiles {{5,6}}²), got {}",
            binning.instances.len()
        );
        // Tile 5 (left-of-boundary) AND tile 6 (right-of-boundary) must
        // both be covered. Pre-fix, tile 6 was missing.
        assert_eq!(binning.tile_instances(5, 5).len(), 1, "tile (5,5) missing");
        assert_eq!(binning.tile_instances(5, 6).len(), 1, "tile (5,6) missing");
        assert_eq!(binning.tile_instances(6, 5).len(), 1, "tile (6,5) missing");
        assert_eq!(
            binning.tile_instances(6, 6).len(), 1,
            "tile (6,6) MISSING — the regression PP-13 caught: \
             px_max = 6·16 = 96, ceil(96/16) = 6 (under-count by one tile)"
        );
    }

    // ── Test 12 — sub-TILE_SIZE image (PP-13 P1: sub-tile grid coverage) ───
    //
    // For an image smaller than TILE_SIZE, the grid is exactly 1×1.
    // Ceil-div math: tile_cols = ceil(8/16) = 1.
    #[test]
    fn sub_tile_size_image_has_single_tile_grid() {
        let camera = Camera::identity_at_origin(8, 8);
        let projected = make_projected(&[(4.0, 4.0, 2.0, 1.0)], None);
        let binning = TileBinning::from_projected(&projected, &camera);
        assert_eq!(binning.tile_cols, 1, "tile_cols for 8px image");
        assert_eq!(binning.tile_rows, 1, "tile_rows for 8px image");
        assert_eq!(binning.tile_offsets.len(), 2, "1 tile + sentinel");
        assert_eq!(binning.instances.len(), 1, "single gaussian → 1 instance");
        assert_eq!(binning.tile_instances(0, 0).len(), 1);
    }

    // ── Test 13 — tile_offsets sentinel invariant (PP-13 P1 promoted) ──────
    //
    // `tile_offsets[n_tiles] == instances.len()`. PR 5 relies on this
    // as the closing bracket so every tile's slice is uniformly
    // `instances[offsets[t]..offsets[t+1]]` without bounds branching.
    #[test]
    fn tile_offsets_sentinel_equals_instances_len() {
        let camera = Camera::identity_at_origin(256, 256);
        let gaussians: Vec<(f32, f32, f32, f32)> = (0..20)
            .map(|i| ((i as f32) * 11.0 + 5.0, (i as f32) * 9.0 + 7.0, 8.0, i as f32 + 1.0))
            .collect();
        let projected = make_projected(&gaussians, None);
        let binning = TileBinning::from_projected(&projected, &camera);
        let n_tiles = (binning.tile_cols * binning.tile_rows) as usize;
        let sentinel = *binning.tile_offsets.last().expect("offsets always have sentinel");
        let actual_count = binning.instances.len() as u32;
        assert_eq!(
            sentinel, actual_count,
            "sentinel offsets[{}] = {sentinel}, instances.len() = {actual_count} — mismatch breaks the PR 5 slice bracket invariant",
            n_tiles,
        );
    }
}
