//! TileInstance v2 + `BlockedGrid<SplatBinList, 1, 1>` binner skeleton.
//!
//! # Role in the cascade
//!
//! This module is the structural heart of PR-X4 A1.  It replaces the v1
//! bespoke `Vec<TileInstance> + Vec<u32>` hand-rolled CSR with a typed
//! `BlockedGrid<SplatBinList, 1, 1>` from PR-X3.  The `<1, 1>` block
//! params mean **1×1 cells per substrate block** — each tile is its own
//! atomic block.  Cascade-tier striding (4×4 at L2, 16×16 at L3, 4×4 at
//! L4) belongs to A2; A1 emits `tier == 1` only.
//!
//! # Verbatim struct
//!
//! Per A1 brief § "Verbatim struct" (pre-sprint lines 95-103 of
//! `01-a1-tileinstance-v2-brief.md`):
//!
//! ```
//! #[repr(C, align(16))]
//! pub struct TileInstance {
//!     pub tier: u8,          // 1 = L1, 2 = L2, 3 = L3, 4 = L4
//!     pub _pad: [u8; 3],
//!     pub block_row: u16,
//!     pub block_col: u16,
//!     pub gaussian_id: u32,
//!     pub confidence: f32,
//! }
//! ```
//!
//! # Confidence key
//!
//! `confidence = 1.0 / (depth + EPS)` so highest-first sort recovers
//! front-to-back order under the new key, matching the v1 depth-ascending
//! sort.  A2 will populate L2-L4 tiers with NARS-confidence values;
//! A1 populates only `tier == 1`.
//!
//! # PP-13 PR4 P0 boundary-tile fix
//!
//! Per A1 brief § "BlockedGrid migration": the `floor + 1` boundary-tile
//! fix at `splat3d/tile.rs:241-243` MUST be preserved verbatim.
//! `tile_aabb_v2` carries this identical logic.  A regression here silently
//! breaks SG3.
//!
//! # SIMD bundles consumed by this module
//!
//! Per A1 brief § "SIMD bundles — B-Splat + B-Interleave-Transpose" and
//! master prompt § "Forbidden constraints" #2:
//! - **B-Splat** (`splat_f32x16`, `splat_i32x16`): broadcast a Gaussian
//!   centre across 16 tile lanes during the bin step.
//! - **B-Interleave-Transpose** (`interleave_f32x16 ∘ transpose_inplace`):
//!   row-major splat3d ↔ lane-major splat4d boundary primitive.
//!
//! No raw `std::arch::*` intrinsics here — all SIMD through `crate::simd::*`.

use crate::hpc::blocked_grid::BlockedGrid;
use crate::hpc::splat3d::project::{Camera, ProjectedBatch};
use crate::simd::F32x16;

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/// Pixel side-length of one L1 tile in v2.
/// Matches v1's `TILE_SIZE` — both must agree for the v1↔v2 parity test.
///
/// Per A1 brief § "Parity tests" #1: the emitted tile coordinates must
/// correspond to `v1::tile_id` via `block_row * tile_cols + block_col`.
pub const TILE_SIZE_V2: u32 = 16;

/// Small epsilon added to depth before reciprocal to avoid div-by-zero when
/// computing `confidence = 1.0 / (depth + DEPTH_EPS)`.
///
/// Value matches the near-plane guarantee from `splat3d::project` (depth > 0
/// for any valid gaussian), so DEPTH_EPS is a safety margin only.
pub const DEPTH_EPS: f32 = 1e-6;

// ════════════════════════════════════════════════════════════════════════════
// TileInstance v2 — verbatim struct per A1 brief § "Verbatim struct"
// ════════════════════════════════════════════════════════════════════════════

/// One (tile, gaussian) binding emitted during v2 binning.
///
/// Per A1 brief § "Verbatim struct" (pre-sprint lines 95-103).
///
/// Layout: `#[repr(C, align(16))]` — 16 bytes per instance, so 4 instances
/// fit one 64-byte cache line.  Fields must not be reordered; the packed-u64
/// radix sort key uses `(tier as u64) << 48 | (block_row as u64) << 32 |
/// (block_col as u64) << 16 | confidence_bits` (A2 finalises the key).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileInstance {
    /// Cascade tier: 1 = L1, 2 = L2, 3 = L3, 4 = L4.
    /// A1 emits `tier == 1` only; A2 populates L2-L4.
    pub tier: u8,
    /// Alignment padding — always zero.
    pub _pad: [u8; 3],
    /// Block row index of the tile within the tier grid.
    pub block_row: u16,
    /// Block column index of the tile within the tier grid.
    pub block_col: u16,
    /// Index of the Gaussian within [`ProjectedBatch`].
    pub gaussian_id: u32,
    /// Sort key (highest-first).  For graphics: `1.0 / (depth + DEPTH_EPS)`.
    /// For NARS cognitive path: the NARS confidence value (A5 deliverable).
    /// Per A1 brief § "Verbatim struct": replaces v1's `depth_bits`.
    pub confidence: f32,
}

// ════════════════════════════════════════════════════════════════════════════
// SplatBinList — per-block payload type
// ════════════════════════════════════════════════════════════════════════════

/// Per-tile bin payload in `BlockedGrid<SplatBinList, 1, 1>`.
///
/// Replaces v1's `Vec<TileInstance> + Vec<u32> prefix` hand-rolled CSR.
/// The `<1, 1>` block params ensure one `SplatBinList` per tile.
///
/// Per A1 brief § "BlockedGrid migration": this is
/// `SmallVec<[TileInstance; 8]>` or equivalent.  Using a plain `Vec` here
/// as the skeleton placeholder; the implementation sprint may switch to
/// `SmallVec` (stack-inline for the common ≤8 Gaussians-per-tile case)
/// without breaking the public API.
//// TODO(A1): consider SmallVec<[TileInstance; 8]> for ≤8 inline — see 01-a1-tileinstance-v2-brief.md:§BlockedGrid migration
#[derive(Clone, Debug, Default)]
pub struct SplatBinList {
    /// Sorted (confidence-descending) list of instances that fall on this tile.
    pub instances: Vec<TileInstance>,
}

impl SplatBinList {
    /// Construct an empty bin.
    #[inline]
    pub fn new() -> Self {
        Self { instances: Vec::new() }
    }

    /// Push one binding; caller sorts later.
    #[inline]
    pub fn push(&mut self, inst: TileInstance) {
        self.instances.push(inst);
    }

    /// Sort by confidence descending (highest-confidence first).
    /// Called once after all gaussians are binned.
    #[inline]
    pub fn sort_descending(&mut self) {
        self.instances.sort_unstable_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SplatBinnerV2 — BlockedGrid<SplatBinList, 1, 1> binner skeleton
// ════════════════════════════════════════════════════════════════════════════

/// Output of v2 binning: `BlockedGrid<SplatBinList, 1, 1>` replacing v1's
/// hand-rolled CSR.
///
/// Construction via [`SplatBinnerV2::from_projected`], which replicates the
/// two-pass count+emit pattern from v1 (`TileBinning::from_projected`) on the
/// typed substrate.  The prefix-sum CSR is replaced by `BlockedGrid` per-block
/// storage; iteration uses `BlockedGrid::blocks_base()`.
///
/// Per A1 brief § "BlockedGrid migration": `BlockedGrid::<SplatBinList, 1, 1>
/// ::new_with_pad(rows, cols, SplatBinList::new())` is the constructor;
/// callers traverse via `grid.blocks_base()`.
pub struct SplatBinnerV2 {
    /// Number of tile-columns (L1 grid width).  `ceil(image_width / TILE_SIZE_V2)`.
    pub tile_cols: u32,
    /// Number of tile-rows (L1 grid height).  `ceil(image_height / TILE_SIZE_V2)`.
    pub tile_rows: u32,
    /// Typed block grid: one `SplatBinList` per (row, col) tile.
    ///
    /// Block params `<1, 1>`: each tile is its own atomic block.  Per A1 brief
    /// § "BlockedGrid migration": "cascade-tier striding belongs to A2."
    pub grid: BlockedGrid<SplatBinList, 1, 1>,
}

impl SplatBinnerV2 {
    /// Bin all visible gaussians into the L1 16×16 tile grid.
    ///
    /// Replicates the two-pass count+emit shape of v1 (`TileBinning::from_projected`)
    /// on the `BlockedGrid<SplatBinList, 1, 1>` substrate.  The packed-u64 radix
    /// sort in v1 is preserved logically; per-tile sorting is done via
    /// `SplatBinList::sort_descending`.
    ///
    /// Per A1 brief § "BlockedGrid migration":
    /// > The PP-13 PR4 P0 boundary-tile fix (`floor + 1` instead of `ceil`)
    /// > at `splat3d/tile.rs:241-243` MUST be preserved verbatim in the v2 port.
    ///
    /// # Consumes B-Splat
    ///
    /// Consumes **B-Splat** (`splat_f32x16`, `splat_i32x16`) to broadcast a
    /// Gaussian centre across 16 tile lanes during the bin step.
    /// Per master prompt § "SIMD bundles" table.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if any valid gaussian has non-positive or
    /// non-finite depth (matching v1's `debug_assert!` at `tile.rs:132-138`).
    pub fn from_projected(projected: &ProjectedBatch, camera: &Camera) -> Self {
        todo!("A1: implement two-pass bin on BlockedGrid<SplatBinList,1,1> — \
               see 01-a1-tileinstance-v2-brief.md:§BlockedGrid migration")
    }

    /// Return the `SplatBinList` for tile `(tile_x, tile_y)`, or an empty
    /// default if coordinates are out of range.
    ///
    /// Matches v1's `TileBinning::tile_instances` semantics — silent empty
    /// slice for out-of-range, no panic.  Callers iterating the full grid
    /// should use `self.grid.blocks_base()` instead (avoids the bounds check).
    pub fn tile_bin(&self, tile_x: u32, tile_y: u32) -> &SplatBinList {
        todo!("A1: delegate to BlockedGrid cell accessor — \
               see 01-a1-tileinstance-v2-brief.md:§BlockedGrid migration")
    }

    /// Total instance count across all tiles (diagnostic / test helper).
    pub fn total_instances(&self) -> usize {
        todo!("A1: sum SplatBinList::instances.len() across grid.blocks_base()")
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Private helper — tile AABB (verbatim port of v1 `tile_aabb`)
// ════════════════════════════════════════════════════════════════════════════

/// Compute the clamped tile-space AABB `(tx_min, tx_max, ty_min, ty_max)`
/// for gaussian `i`.  Ranges are half-open `[min, max)`.
///
/// **Verbatim port of `splat3d/tile.rs::tile_aabb` (lines 216-253).**
///
/// The `floor + 1` formula for `tx_max`/`ty_max` is the PP-13 PR4 P0 fix.
/// Per A1 brief § "BlockedGrid migration":
/// > This MUST be preserved verbatim in the v2 port — a regression here
/// > silently breaks SG3.
///
/// Consumes **B-Splat** implicitly: the calling `from_projected` loop
/// broadcasts Gaussian positions via `splat_f32x16` before calling this
/// function per-gaussian.
#[inline]
fn tile_aabb_v2(
    projected: &ProjectedBatch,
    i: usize,
    tile_cols: u32,
    tile_rows: u32,
) -> (u32, u32, u32, u32) {
    todo!("A1: verbatim port of splat3d/tile.rs:tile_aabb with TILE_SIZE_V2 — \
           PP-13 PR4 P0 floor+1 fix must be preserved exactly; \
           see 01-a1-tileinstance-v2-brief.md:§BlockedGrid migration")
}

// ════════════════════════════════════════════════════════════════════════════
// B-Interleave-Transpose boundary primitive (skeleton)
// ════════════════════════════════════════════════════════════════════════════

/// Convert a row-major `SplatBinList` (splat3d v1/v2 layout) into a
/// lane-major `F32x16` block suitable for splat4d cascade consumption.
///
/// Per A1 brief § "SIMD bundles — B-Splat + B-Interleave-Transpose":
/// A1 IS the v1↔v2 boundary, so this is its primary tool.
///
/// # Consumes B-Interleave-Transpose
///
/// Consumes **B-Interleave-Transpose** (`interleave_f32x16 ∘ transpose_inplace`):
/// row-major splat3d ↔ lane-major splat4d.  Per master prompt § "SIMD bundles"
/// table.
///
/// Must NOT reach past this bundle into raw `std::arch::*` intrinsics.
/// Per Forbidden Constraint #2 of the master prompt.
//// TODO(A1): wire interleave_f32x16 ∘ transpose_inplace from crate::simd — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§SIMD bundles table
pub fn interleave_transpose_bin_list(bin: &SplatBinList) -> F32x16 {
    todo!("A1: apply B-Interleave-Transpose (interleave_f32x16 ∘ transpose_inplace) \
           to convert row-major SplatBinList into lane-major F32x16 for splat4d; \
           see 01-a1-tileinstance-v2-brief.md:§SIMD bundles and \
           hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§B-Interleave-Transpose")
}

// ════════════════════════════════════════════════════════════════════════════
// Parity test helpers (skeleton)
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::splat3d::project::{Camera, ProjectedBatch};

    /// Parity test fixture seed per A1 brief § "Parity tests" #1.
    ///
    /// Feed both v1 and v2 binners the same 16-Gaussian fixture (seed
    /// `0xA1_B1_NA_RY`) and assert emitted `TileInstance` streams agree on
    /// `(block_row/block_col ↔ tile_id, gaussian_id, sort order)`.
    //// TODO(A1): implement v1-vs-v2 binner parity — seed 0xA1_B1_NA_RY — see 01-a1-tileinstance-v2-brief.md:§Parity tests
    #[test]
    #[ignore = "skeleton — implement in A1 sprint"]
    fn v1_v2_binner_parity_16_gaussian_fixture() {
        todo!("A1 parity test: 16-Gaussian fixture, seed 0xA1_B1_NA_RY; \
               assert v1 tile_id == v2 block_row*tile_cols+block_col, \
               gaussian_id matches, sort order under depth/confidence transform matches")
    }

    /// Boundary-tile coverage regression: PP-13 PR4 P0 case.
    ///
    /// Per A1 brief § "Parity tests" #2: 3σ-ellipse straddling a tile boundary
    /// must produce identical per-tile splat counts in v2 as v1.
    //// TODO(A1): implement boundary-tile coverage regression PP-13 PR4 P0 — see 01-a1-tileinstance-v2-brief.md:§Parity tests
    #[test]
    #[ignore = "skeleton — implement in A1 sprint"]
    fn boundary_tile_coverage_regression_pp13_pr4_p0() {
        todo!("A1 boundary-tile regression: 3σ-ellipse on tile boundary; \
               v2 must match v1 per-tile splat counts exactly; \
               the floor+1 fix in tile_aabb_v2 is the load-bearing line")
    }

    /// `TileInstance` size/alignment check — 16 bytes, 16-byte aligned.
    #[test]
    fn tile_instance_v2_layout() {
        assert_eq!(std::mem::size_of::<TileInstance>(), 16);
        assert_eq!(std::mem::align_of::<TileInstance>(), 16);
    }

    /// `TileInstance::tier` is the first byte (offset 0).
    #[test]
    fn tile_instance_v2_tier_at_offset_zero() {
        let inst = TileInstance {
            tier: 0xAB,
            _pad: [0; 3],
            block_row: 0,
            block_col: 0,
            gaussian_id: 0,
            confidence: 0.0,
        };
        // SAFETY: TileInstance is #[repr(C, align(16))]; byte 0 is `tier`.
        let bytes: [u8; 16] = unsafe { std::mem::transmute(inst) };
        assert_eq!(bytes[0], 0xAB, "tier must be at byte offset 0");
    }
}
