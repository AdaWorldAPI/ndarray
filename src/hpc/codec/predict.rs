//! Intra-prediction mode decision (PR-X12 A3, intra path).
//!
//! Encoder-side kernel: given a cell value, its nearest-basin index +
//! delta, and the four cardinal neighbour `LeafCu`s, choose the best
//! [`CellMode`] for the cell and emit the corresponding [`LeafCu`].
//!
//! This is the **mode-decision** kernel, not the inverse-projection
//! reconstruction. Decoder-side reconstruction is the inverse of this
//! decision tree and is folded into PR-X12 A6 RDO + A8 stream
//! interpretation; A3 ships only the encoder direction.
//!
//! # The decision tree
//!
//! For one cell at (row, col) inside a CTU:
//!
//! ```text
//!   ┌─────────────────────────────────────────────────┐
//!   │ delta == 0  ?         → Skip(basin_idx)         │
//!   └─────────────────────────────────────────────────┘
//!                  │ no
//!                  ▼
//!   ┌─────────────────────────────────────────────────┐
//!   │ any same-dir neighbour has Delta-mode with the  │
//!   │ SAME basin_idx AND SAME |delta| (sign-tolerant) │
//!   │                       → Merge(basin_idx, dir)   │
//!   └─────────────────────────────────────────────────┘
//!                  │ no candidate
//!                  ▼
//!   ┌─────────────────────────────────────────────────┐
//!   │ |delta| fits in i8 (≤ 127)?                     │
//!   │                       → Delta(basin_idx, δ_u8)  │
//!   └─────────────────────────────────────────────────┘
//!                  │ overflow
//!                  ▼
//!   ┌─────────────────────────────────────────────────┐
//!   │                       → Escape(basin_idx, idx)  │
//!   │      (caller appends raw u64 to escape vector)  │
//!   └─────────────────────────────────────────────────┘
//! ```
//!
//! The order is **Skip → Merge → Delta → Escape** because the wire
//! cost is monotonically increasing in the same order (2 → 3 → 3 → 6
//! bytes per [`packed_byte_len`](super::mode::packed_byte_len)). The
//! decision picks the cheapest mode that fits.
//!
//! # What A3-intra does NOT do
//!
//! - **Inter prediction** (parent-tier neighbours from the
//!   `BlockedGrid` L2/L3 cascade) — deferred to PR-X12 A3 follow-up.
//! - **Rate-distortion optimisation** — A3-intra picks by exact match
//!   only. Soft mode-switching with λ-RDO is PR-X12 A6.
//! - **Transform / quantisation** — A3-intra works on already-decoded
//!   integer deltas. The transform path (DCT-II for delta residuals)
//!   is PR-X12 A4.
//! - **SIMD-batched CTU sweep** — scalar reference today. The
//!   `F32x16`-batched form (16 cells per inner loop via
//!   `crate::simd_soa::MultiLaneColumn`) is a follow-up after the
//!   reference + reconstruction parity test pin the math.

use super::ctu::{CellMode, LeafCu, MergeDir};
use super::mode::BASIN_NONE;

// ════════════════════════════════════════════════════════════════════
// Inputs to the encoder mode decision
// ════════════════════════════════════════════════════════════════════

/// Per-cell context the encoder needs to choose a mode.
///
/// Built by the caller from the CTU's basin lookup + the per-cell
/// neighbour table. The encoder does not own the basin codebook or the
/// escape vector; it returns an `Escape(basin_idx, escape_idx)` leaf
/// and lets the caller push the original cell value into the per-frame
/// escape vector at `escape_idx`.
///
/// # Fields
///
/// - `basin_idx`: nearest basin's index in the per-frame codebook,
///   already resolved by the caller (typically via
///   `ogit_bridge::nearest_basin`). Must be `<= MAX_BASIN_IDX`
///   (12 bits) per [`super::mode::pack_header`]. The encoder does not
///   re-validate.
/// - `delta_i32`: signed delta from the basin's u8-quantised
///   representation of the cell. The encoder branches on `|delta|`
///   to decide between Delta (fits in i8) and Escape (overflows).
///   `i32` width avoids overflow when the caller computes
///   `cell_value - basin_value` for two u8 inputs.
/// - `neighbours`: NEWS (in [`MergeDir`] discriminant order:
///   `North=0, East=1, West=2, South=3`) optional neighbour leaves.
///   `None` for boundary cells; the Merge candidate scan skips `None`
///   entries.
///
/// ```text
///   slot 0 → MergeDir::North   (discr 0)
///   slot 1 → MergeDir::East    (discr 1)
///   slot 2 → MergeDir::West    (discr 2)
///   slot 3 → MergeDir::South   (discr 3)
/// ```
///
/// ```
/// use ndarray::hpc::codec::{IntraContext, LeafCu};
/// let north = LeafCu::delta(5, 17);
/// let ctx = IntraContext {
///     basin_idx: 5,
///     delta_i32: 17,
///     neighbours: [Some(&north), None, None, None],
/// };
/// assert_eq!(ctx.basin_idx, 5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct IntraContext<'a> {
    /// Pre-resolved basin index (12-bit max).
    pub basin_idx: u16,
    /// Signed delta from basin → cell, in the basin's u8 quantisation
    /// space.
    pub delta_i32: i32,
    /// NEWS neighbour leaves, indexed by [`MergeDir`] discriminant
    /// (`North=0, East=1, West=2, South=3`).
    pub neighbours: [Option<&'a LeafCu>; 4],
}

/// Configuration for the intra-prediction decision.
///
/// Reserved for future expansion (Merge tolerance, RDO knobs in A6).
/// Empty today; constructed via [`Default`] so additions don't break
/// callers.
///
/// ```
/// use ndarray::hpc::codec::IntraConfig;
/// let cfg = IntraConfig::default();
/// // No tunables yet — call sites stay future-compatible.
/// let _ = cfg;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct IntraConfig {
    // Reserved. Future fields land here without breaking the signature.
    _reserved: (),
}

// ════════════════════════════════════════════════════════════════════
// The decision kernel
// ════════════════════════════════════════════════════════════════════

/// Encoder-side intra-prediction. Returns the cheapest [`LeafCu`]
/// representation of the cell described by `ctx`.
///
/// See the module docs for the decision tree (Skip → Merge → Delta →
/// Escape) and the rationale (monotone wire cost).
///
/// `escape_next` is a write-cursor into the caller's per-frame escape
/// vector. When the decision falls through to Escape, the kernel reads
/// the cursor, emits a leaf referencing that idx, and post-increments
/// the cursor so subsequent cells in the same batch get fresh,
/// non-colliding idxs. Pass `None` to disable lossless Escape — the
/// kernel then clamps `δ` to i8 range and emits a `Delta` leaf whose
/// reconstruction is **not bit-exact** (caller must accept the loss).
///
/// # Examples
///
/// Skip when the cell is exactly the basin:
///
/// ```
/// use ndarray::hpc::codec::predict::{predict_intra, IntraContext, IntraConfig};
/// use ndarray::hpc::codec::CellMode;
/// let ctx = IntraContext {
///     basin_idx: 42,
///     delta_i32: 0,
///     neighbours: [None; 4],
/// };
/// let leaf = predict_intra(&ctx, &IntraConfig::default(), None);
/// assert_eq!(leaf.mode, CellMode::Skip);
/// assert_eq!(leaf.basin_idx, 42);
/// ```
///
/// Delta when no Merge candidate exists but |δ| fits in i8:
///
/// ```
/// use ndarray::hpc::codec::predict::{predict_intra, IntraContext, IntraConfig};
/// use ndarray::hpc::codec::CellMode;
/// let ctx = IntraContext {
///     basin_idx: 42,
///     delta_i32: 17,
///     neighbours: [None; 4],
/// };
/// let leaf = predict_intra(&ctx, &IntraConfig::default(), None);
/// assert_eq!(leaf.mode, CellMode::Delta);
/// assert_eq!(leaf.delta, Some(17));
/// ```
///
/// Escape with an allocator — repeated calls bump the cursor:
///
/// ```
/// use ndarray::hpc::codec::predict::{predict_intra, IntraContext, IntraConfig};
/// use ndarray::hpc::codec::CellMode;
/// let mut next = 7u32;
/// let ctx = IntraContext { basin_idx: 1, delta_i32: 1000, neighbours: [None; 4] };
/// let a = predict_intra(&ctx, &IntraConfig::default(), Some(&mut next));
/// let b = predict_intra(&ctx, &IntraConfig::default(), Some(&mut next));
/// assert_eq!(a.escape_idx, Some(7));
/// assert_eq!(b.escape_idx, Some(8));
/// assert_eq!(next, 9);
/// assert_eq!(a.mode, CellMode::Escape);
/// ```
pub fn predict_intra(ctx: &IntraContext, _cfg: &IntraConfig, escape_next: Option<&mut u32>) -> LeafCu {
    // ── 1. Skip ──────────────────────────────────────────────────────
    if ctx.delta_i32 == 0 {
        return LeafCu::skip(ctx.basin_idx);
    }

    // i8-fit gates both Merge and Delta. Out-of-range δ must skip
    // Merge entirely — wrapping `200_i32 as u8` aliases to `0xC8`,
    // which could spuriously match a neighbour whose byte equals
    // `0xC8` (i8 = -56), producing a leaf the decoder reconstructs as
    // -56 instead of 200.
    let fits_i8 = (-128..=127).contains(&ctx.delta_i32);
    let our_delta_u8 = ctx.delta_i32 as u8; // wrapping cast matches A2 pack

    // ── 2. Merge ─────────────────────────────────────────────────────
    //
    // A neighbour is a Merge candidate iff:
    //   (a) its mode is Delta (Skip / Merge / Escape neighbours carry
    //       no reusable delta to inherit from)
    //   (b) its basin_idx matches ours (Merge inheritance implicitly
    //       points at the SAME basin — different basins mean a
    //       different reference frame)
    //   (c) its δ exactly matches our δ as a u8 (sign-tolerant via
    //       wrapping cast; matches the A2 pack format where Delta
    //       stores a raw u8 byte without a sign bit)
    //
    // We scan NEWS in discriminant order (N=0, E=1, W=2, S=3) and
    // pick the first match.
    // Multiple matches all collapse to the same coded leaf, so the
    // first-hit policy is order-deterministic without affecting
    // bitstream length.
    if fits_i8 {
        for (i, nb_slot) in ctx.neighbours.iter().enumerate() {
            let Some(nb) = nb_slot else { continue };
            if nb.mode != CellMode::Delta {
                continue;
            }
            if nb.basin_idx != ctx.basin_idx {
                continue;
            }
            if nb.delta != Some(our_delta_u8) {
                continue;
            }
            let dir = merge_dir_from_index(i);
            return LeafCu::merge(ctx.basin_idx, dir);
        }
    }

    // ── 3. Delta ─────────────────────────────────────────────────────
    //
    // i8 range is [-128, 127]. We pack as raw u8 (wrapping cast) so
    // the encoder's reconstruction must read the byte back as i8 to
    // recover the sign. This matches how `LeafCu::delta` stores it and
    // how `super::mode::pack_leaf` writes it.
    if fits_i8 {
        return LeafCu::delta(ctx.basin_idx, our_delta_u8);
    }

    // ── 4. Escape ────────────────────────────────────────────────────
    //
    // |δ| doesn't fit in i8. The cursor `escape_next` is a write-pointer
    // into the caller's per-frame escape vector; we read it, emit a
    // leaf referencing that idx, and post-increment so subsequent
    // overflow cells in the batch don't collide on the same vector
    // slot. If the caller didn't provide an allocator, we fall back to
    // a saturated Delta (lossy: reconstruction is NOT bit-exact, but
    // never panicking) so a misconfigured encoder still produces a
    // valid bytestream. The lossy leaf's `mode` is `CellMode::Delta`
    // even though its semantic value overflowed i8 — by contract the
    // caller has acknowledged the precision loss.
    match escape_next {
        Some(next) => {
            let idx = *next;
            *next = next.wrapping_add(1);
            LeafCu::escape(ctx.basin_idx, idx)
        }
        None => {
            let clamped = ctx.delta_i32.clamp(-128, 127) as u8;
            LeafCu::delta(ctx.basin_idx, clamped)
        }
    }
}

#[inline]
fn merge_dir_from_index(i: usize) -> MergeDir {
    match i {
        0 => MergeDir::North,
        1 => MergeDir::East,
        2 => MergeDir::West,
        _ => MergeDir::South,
    }
}

/// Sanity-check sentinel: returns `true` iff the resolved basin index
/// is the "no basin" marker. Encoders that compute basins lazily can
/// short-circuit Skip/Merge/Delta and emit Escape directly when this
/// fires.
///
/// ```
/// use ndarray::hpc::codec::{is_no_basin, BASIN_NONE};
/// assert!(is_no_basin(BASIN_NONE));
/// assert!(!is_no_basin(0));
/// ```
#[inline]
pub fn is_no_basin(basin_idx: u16) -> bool {
    basin_idx == BASIN_NONE
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_neighbours<'a>(basin: u16, delta: i32, neighbours: [Option<&'a LeafCu>; 4]) -> IntraContext<'a> {
        IntraContext {
            basin_idx: basin,
            delta_i32: delta,
            neighbours,
        }
    }

    #[test]
    fn skip_when_delta_is_zero() {
        let leaf = predict_intra(&ctx_with_neighbours(100, 0, [None; 4]), &IntraConfig::default(), None);
        assert_eq!(leaf, LeafCu::skip(100));
    }

    #[test]
    fn skip_preferred_over_neighbour_match() {
        // δ=0 trumps everything else, even a perfect Merge candidate.
        let nb = LeafCu::delta(100, 0);
        let neighbours = [Some(&nb), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 0, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Skip);
    }

    #[test]
    fn delta_in_i8_range() {
        for d in [-128i32, -1, 1, 127] {
            let leaf = predict_intra(&ctx_with_neighbours(100, d, [None; 4]), &IntraConfig::default(), None);
            assert_eq!(leaf.mode, CellMode::Delta);
            assert_eq!(leaf.delta, Some(d as u8));
        }
    }

    #[test]
    fn merge_when_neighbour_delta_matches_basin_and_value() {
        // Northern neighbour: Delta-mode, same basin, same δ as us.
        let nb_north = LeafCu::delta(100, 17);
        let neighbours = [Some(&nb_north), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Merge);
        assert_eq!(leaf.merge_dir, Some(MergeDir::North));
        assert_eq!(leaf.basin_idx, 100);
    }

    #[test]
    fn merge_skipped_when_basin_differs() {
        // Same δ but different basin → cannot Merge (different
        // reference frame). Falls through to Delta.
        let nb_north = LeafCu::delta(99, 17);
        let neighbours = [Some(&nb_north), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Delta);
    }

    #[test]
    fn merge_skipped_when_neighbour_mode_is_not_delta() {
        // Skip / Merge / Escape neighbours carry no inheritable δ.
        let nb_skip = LeafCu::skip(100);
        let nb_merge = LeafCu::merge(100, MergeDir::North);
        let nb_esc = LeafCu::escape(100, 0);
        let neighbours = [Some(&nb_skip), Some(&nb_merge), None, Some(&nb_esc)];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Delta);
    }

    #[test]
    fn merge_picks_first_hit_in_news_order() {
        // Both N and E qualify; encoder must pick N (lower index).
        let nb_match = LeafCu::delta(100, 17);
        let neighbours = [Some(&nb_match), Some(&nb_match), None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.merge_dir, Some(MergeDir::North));
    }

    #[test]
    fn merge_slot_2_maps_to_west_and_slot_3_to_south() {
        // Slot-3 South coverage gap noted in review. Verify the
        // discriminant order (N=0, E=1, W=2, S=3) is reflected at
        // the merge_dir output, not just NEWS-by-convention.
        let nb = LeafCu::delta(100, 17);

        let only_west = [None, None, Some(&nb), None];
        let leaf_w = predict_intra(&ctx_with_neighbours(100, 17, only_west), &IntraConfig::default(), None);
        assert_eq!(leaf_w.merge_dir, Some(MergeDir::West));

        let only_south = [None, None, None, Some(&nb)];
        let leaf_s = predict_intra(&ctx_with_neighbours(100, 17, only_south), &IntraConfig::default(), None);
        assert_eq!(leaf_s.merge_dir, Some(MergeDir::South));
    }

    #[test]
    fn merge_negative_delta_via_wrapping_cast() {
        // δ = -17 packs to 0xEF (= 239 as u8). Neighbour stored as
        // u8 = 0xEF MUST match — the cast must be wrapping, not
        // saturating.
        let nb_match = LeafCu::delta(100, (-17_i32) as u8);
        let neighbours = [None, Some(&nb_match), None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, -17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Merge);
        assert_eq!(leaf.merge_dir, Some(MergeDir::East));
    }

    #[test]
    fn escape_when_delta_overflows_i8_and_allocator_present() {
        let mut next = 42u32;
        let leaf = predict_intra(&ctx_with_neighbours(100, 1000, [None; 4]), &IntraConfig::default(), Some(&mut next));
        assert_eq!(leaf.mode, CellMode::Escape);
        assert_eq!(leaf.escape_idx, Some(42));
        assert_eq!(leaf.basin_idx, 100);
        // Cursor advanced so the next Escape gets a fresh idx.
        assert_eq!(next, 43);
    }

    #[test]
    fn escape_allocator_advances_across_batched_calls() {
        // Regression: two consecutive Escape decisions must not
        // collide on the same vector slot. With a `&mut u32` cursor
        // the kernel post-increments, so cell A sees idx N and
        // cell B sees idx N+1.
        let mut next = 5u32;
        let a = predict_intra(&ctx_with_neighbours(7, 999, [None; 4]), &IntraConfig::default(), Some(&mut next));
        let b = predict_intra(&ctx_with_neighbours(7, -999, [None; 4]), &IntraConfig::default(), Some(&mut next));
        assert_eq!(a.escape_idx, Some(5));
        assert_eq!(b.escape_idx, Some(6));
        assert_eq!(next, 7);
        assert_ne!(a.escape_idx, b.escape_idx);
    }

    #[test]
    fn escape_lossy_fallback_when_no_allocator() {
        // Without an escape_next_idx, the encoder clamps to i8 range.
        // The result is a valid LeafCu but the reconstruction won't
        // be bit-exact.
        let leaf = predict_intra(&ctx_with_neighbours(100, 1000, [None; 4]), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Delta);
        assert_eq!(leaf.delta, Some(127));
    }

    #[test]
    fn escape_lossy_fallback_negative_overflow() {
        let leaf = predict_intra(&ctx_with_neighbours(100, -1000, [None; 4]), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Delta);
        assert_eq!(leaf.delta, Some((-128_i32) as u8));
    }

    #[test]
    fn pack_then_unpack_chained_through_intra_decision() {
        // End-to-end: encoder picks Merge for one cell. The packed
        // representation must round-trip via A2's pack/unpack with
        // bit-exact fidelity.
        use super::super::mode::{pack_leaf, unpack_leaf};
        let nb = LeafCu::delta(100, 17);
        let neighbours = [None, Some(&nb), None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default(), None);
        assert_eq!(leaf.mode, CellMode::Merge);

        let mut buf = [0u8; 6];
        let n = pack_leaf(&leaf, &mut buf).unwrap();
        let (decoded, consumed) = unpack_leaf(&buf).unwrap();
        assert_eq!(n, consumed);
        assert_eq!(decoded, leaf);
    }

    #[test]
    fn is_no_basin_sentinel_round_trip() {
        assert!(is_no_basin(BASIN_NONE));
        assert!(!is_no_basin(0));
        assert!(!is_no_basin(100));
    }

    #[test]
    fn overflow_delta_does_not_alias_to_merge() {
        // Regression for the wrapping-cast Merge alias bug:
        // δ = 200 (overflows i8) must NOT match a neighbour whose
        // u8 byte equals (200 as u8) = 0xC8 (= -56 in i8). The
        // encoder must take the Escape path (or, here, the lossy
        // clamp fallback because no allocator is wired).
        let nb_alias = LeafCu::delta(100, 0xC8);
        let neighbours = [Some(&nb_alias), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 200, neighbours), &IntraConfig::default(), None);
        assert_ne!(leaf.mode, CellMode::Merge, "overflow δ must not Merge");
        // With no allocator the encoder clamps to +127 (lossy Delta).
        assert_eq!(leaf.mode, CellMode::Delta);
        assert_eq!(leaf.delta, Some(127));
    }

    #[test]
    fn overflow_delta_with_allocator_takes_escape() {
        let nb_alias = LeafCu::delta(100, 0xC8);
        let neighbours = [Some(&nb_alias), None, None, None];
        let mut next = 7u32;
        let leaf = predict_intra(&ctx_with_neighbours(100, 200, neighbours), &IntraConfig::default(), Some(&mut next));
        assert_eq!(leaf.mode, CellMode::Escape);
        assert_eq!(leaf.escape_idx, Some(7));
        assert_eq!(next, 8);
    }

    #[test]
    fn pack_leaf_accepts_mode_sized_buffers() {
        // Regression for the P2 6-byte-minimum bug: Skip should pack
        // into a 2-byte buffer, Merge/Delta into a 3-byte buffer.
        use super::super::mode::{pack_leaf, packed_byte_len};
        let skip = LeafCu::skip(10);
        let mut buf2 = [0u8; 2];
        assert_eq!(pack_leaf(&skip, &mut buf2), Some(2));
        assert_eq!(packed_byte_len(CellMode::Skip), 2);

        let delta = LeafCu::delta(10, 7);
        let mut buf3 = [0u8; 3];
        assert_eq!(pack_leaf(&delta, &mut buf3), Some(3));

        // Escape still needs 6 bytes; a 3-byte buffer is rejected.
        let esc = LeafCu::escape(10, 99);
        let mut buf3b = [0u8; 3];
        assert_eq!(pack_leaf(&esc, &mut buf3b), None);
    }
}
