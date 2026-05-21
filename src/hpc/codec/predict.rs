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
/// - `neighbours`: NESW (in [`MergeDir`] discriminant order) optional
///   neighbour leaves. `None` for boundary cells; the Merge candidate
///   scan skips `None` entries.
#[derive(Debug, Clone, Copy)]
pub struct IntraContext<'a> {
    /// Pre-resolved basin index (12-bit max).
    pub basin_idx: u16,
    /// Signed delta from basin → cell, in the basin's u8 quantisation
    /// space.
    pub delta_i32: i32,
    /// NESW neighbour leaves, indexed by [`MergeDir`] discriminant.
    pub neighbours: [Option<&'a LeafCu>; 4],
}

/// Configuration for the intra-prediction decision.
///
/// Today a single field; the field exists so the API can grow
/// (Merge tolerance, RDO knobs in A6) without a signature break.
#[derive(Debug, Clone, Copy)]
pub struct IntraConfig {
    /// Future allocator for the encoder's escape vector — returns the
    /// next index to write. `None` disables Escape mode (the encoder
    /// will fall back to Delta-with-truncation, which **loses
    /// precision** but never panics; callers wanting lossless coding
    /// must provide a real allocator).
    ///
    /// Stateless API today: encoder calls `escape_next_idx` once per
    /// Escape decision. The caller is responsible for actually
    /// appending the u64 cell value into the escape vector at the
    /// returned index — this kernel doesn't see the cell value.
    pub escape_next_idx: Option<u32>,
}

impl Default for IntraConfig {
    fn default() -> Self {
        Self { escape_next_idx: None }
    }
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
/// # Examples
///
/// Skip when the cell is exactly the basin:
///
/// ```
/// use ndarray::hpc::codec::predict::{predict_intra, IntraContext, IntraConfig};
/// use ndarray::hpc::codec::{CellMode, LeafCu};
/// let ctx = IntraContext {
///     basin_idx: 42,
///     delta_i32: 0,
///     neighbours: [None; 4],
/// };
/// let leaf = predict_intra(&ctx, &IntraConfig::default());
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
/// let leaf = predict_intra(&ctx, &IntraConfig::default());
/// assert_eq!(leaf.mode, CellMode::Delta);
/// assert_eq!(leaf.delta, Some(17));
/// ```
pub fn predict_intra(ctx: &IntraContext, cfg: &IntraConfig) -> LeafCu {
    // ── 1. Skip ──────────────────────────────────────────────────────
    if ctx.delta_i32 == 0 {
        return LeafCu::skip(ctx.basin_idx);
    }

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
    // We scan NESW in discriminant order and pick the first match.
    // Multiple matches all collapse to the same coded leaf, so the
    // first-hit policy is order-deterministic without affecting
    // bitstream length.
    let our_delta_u8 = ctx.delta_i32 as u8; // wrapping cast matches A2 pack
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

    // ── 3. Delta ─────────────────────────────────────────────────────
    //
    // i8 range is [-128, 127]. We pack as raw u8 (wrapping cast) so
    // the encoder's reconstruction must read the byte back as i8 to
    // recover the sign. This matches how `LeafCu::delta` stores it and
    // how `super::mode::pack_leaf` writes it.
    if (-128..=127).contains(&ctx.delta_i32) {
        return LeafCu::delta(ctx.basin_idx, our_delta_u8);
    }

    // ── 4. Escape ────────────────────────────────────────────────────
    //
    // |δ| doesn't fit in i8. Caller must own the per-frame escape
    // vector and provide the next-write index; we return a leaf that
    // references it. If the caller didn't provide an allocator, we
    // fall back to a saturated Delta (lossy but never panicking) so
    // a misconfigured encoder still produces a valid bytestream.
    match cfg.escape_next_idx {
        Some(idx) => LeafCu::escape(ctx.basin_idx, idx),
        None => {
            // Lossy fallback: clamp to i8 range. Caller is responsible
            // for noticing that the reconstruction won't be bit-exact.
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
        let leaf = predict_intra(&ctx_with_neighbours(100, 0, [None; 4]), &IntraConfig::default());
        assert_eq!(leaf, LeafCu::skip(100));
    }

    #[test]
    fn skip_preferred_over_neighbour_match() {
        // δ=0 trumps everything else, even a perfect Merge candidate.
        let nb = LeafCu::delta(100, 0);
        let neighbours = [Some(&nb), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 0, neighbours), &IntraConfig::default());
        assert_eq!(leaf.mode, CellMode::Skip);
    }

    #[test]
    fn delta_in_i8_range() {
        for d in [-128i32, -1, 1, 127] {
            let leaf = predict_intra(&ctx_with_neighbours(100, d, [None; 4]), &IntraConfig::default());
            assert_eq!(leaf.mode, CellMode::Delta);
            assert_eq!(leaf.delta, Some(d as u8));
        }
    }

    #[test]
    fn merge_when_neighbour_delta_matches_basin_and_value() {
        // Northern neighbour: Delta-mode, same basin, same δ as us.
        let nb_north = LeafCu::delta(100, 17);
        let neighbours = [Some(&nb_north), None, None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default());
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
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default());
        assert_eq!(leaf.mode, CellMode::Delta);
    }

    #[test]
    fn merge_skipped_when_neighbour_mode_is_not_delta() {
        // Skip / Merge / Escape neighbours carry no inheritable δ.
        let nb_skip = LeafCu::skip(100);
        let nb_merge = LeafCu::merge(100, MergeDir::North);
        let nb_esc = LeafCu::escape(100, 0);
        let neighbours = [Some(&nb_skip), Some(&nb_merge), None, Some(&nb_esc)];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default());
        assert_eq!(leaf.mode, CellMode::Delta);
    }

    #[test]
    fn merge_picks_first_hit_in_nesw_order() {
        // Both N and E qualify; encoder must pick N (lower index).
        let nb_match = LeafCu::delta(100, 17);
        let neighbours = [Some(&nb_match), Some(&nb_match), None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default());
        assert_eq!(leaf.merge_dir, Some(MergeDir::North));
    }

    #[test]
    fn merge_negative_delta_via_wrapping_cast() {
        // δ = -17 packs to 0xEF (= 239 as u8). Neighbour stored as
        // u8 = 0xEF MUST match — the cast must be wrapping, not
        // saturating.
        let nb_match = LeafCu::delta(100, (-17_i32) as u8);
        let neighbours = [None, Some(&nb_match), None, None];
        let leaf = predict_intra(&ctx_with_neighbours(100, -17, neighbours), &IntraConfig::default());
        assert_eq!(leaf.mode, CellMode::Merge);
        assert_eq!(leaf.merge_dir, Some(MergeDir::East));
    }

    #[test]
    fn escape_when_delta_overflows_i8_and_allocator_present() {
        let cfg = IntraConfig {
            escape_next_idx: Some(42),
        };
        let leaf = predict_intra(&ctx_with_neighbours(100, 1000, [None; 4]), &cfg);
        assert_eq!(leaf.mode, CellMode::Escape);
        assert_eq!(leaf.escape_idx, Some(42));
        assert_eq!(leaf.basin_idx, 100);
    }

    #[test]
    fn escape_lossy_fallback_when_no_allocator() {
        // Without an escape_next_idx, the encoder clamps to i8 range.
        // The result is a valid LeafCu but the reconstruction won't
        // be bit-exact.
        let leaf = predict_intra(&ctx_with_neighbours(100, 1000, [None; 4]), &IntraConfig::default());
        assert_eq!(leaf.mode, CellMode::Delta);
        assert_eq!(leaf.delta, Some(127));
    }

    #[test]
    fn escape_lossy_fallback_negative_overflow() {
        let leaf = predict_intra(&ctx_with_neighbours(100, -1000, [None; 4]), &IntraConfig::default());
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
        let leaf = predict_intra(&ctx_with_neighbours(100, 17, neighbours), &IntraConfig::default());
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
}
