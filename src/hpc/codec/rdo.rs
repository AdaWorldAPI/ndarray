//! λ-rate-distortion mode selection (PR-X12 A6).
//!
//! [`super::predict`] makes a *hard* mode decision: Skip iff δ = 0,
//! Merge iff a neighbour matches exactly, else Delta/Escape. RDO is the
//! *soft* generalization — for each cell it scores all four modes by
//!
//! ```text
//!   cost = rate + λ · distortion
//! ```
//!
//! and picks the minimum. At λ = 0 it minimizes pure rate (always the
//! cheapest mode that exists); as λ → ∞ it minimizes pure distortion
//! (lossless Escape when available, else the tightest Delta). This is
//! the lever that lets high-confidence cells spend bits for fidelity and
//! low-confidence cells tolerate lossy compression — the design's
//! "λ calibrated via NARS confidence".
//!
//! # Integer λ (no float)
//!
//! The design doc sketches `lambda: f32`. This implementation uses a
//! **fixed-point** λ instead — [`RdoConfig::lambda_q8`] is λ × 256 as a
//! `u32`, and the cost is computed in `u64`:
//!
//! ```text
//!   cost_q8 = (rate_bytes << 8) + lambda_q8 · distortion
//! ```
//!
//! This keeps mode decisions deterministic and bit-identical across
//! platforms (no float rounding divergence), consistent with the
//! substrate's no-float discipline. `f32` λ is a cross-platform
//! reproducibility hazard the codec doesn't need: distortion is an
//! integer delta magnitude and rate is an integer byte count, so the
//! whole R + λD lattice is exactly representable in fixed point.
//!
//! # Rate model
//!
//! Rate is the leaf's on-wire byte length
//! ([`super::mode::packed_byte_len`]): Skip = 2, Merge = 3, Delta = 3,
//! Escape = 6. This is the *pre-entropy* size — the bytes a leaf
//! actually occupies before the A7 rANS pass folds the mode tag down to
//! `-log2(p)` bits. An entropy-aware rate (mode tag weighted by its
//! frequency) is a documented follow-on; the pre-entropy byte rate is
//! exact, deterministic, and never under-counts the payload.
//!
//! # Distortion model
//!
//! Distortion is the absolute error of the reconstructed δ against the
//! true δ, both in the basin's u8-quantization space (matching
//! [`super::predict::IntraContext`]'s `delta_i32`):
//!
//! | Mode   | Reconstructed δ            | Distortion              |
//! |--------|----------------------------|-------------------------|
//! | Skip   | 0 (basin exactly)          | `|δ|`                   |
//! | Merge  | neighbour's δ (as `i8`)    | `|δ − δ_neighbour|`     |
//! | Delta  | `clamp(δ, −128, 127)`      | `|δ − clamp(δ)|` (0 in range) |
//! | Escape | δ (full value preserved)   | 0 (lossless)            |
//!
//! Escape requires a caller-supplied escape-vector cursor (same contract
//! as [`super::predict::predict_intra`]); without one, Escape is not a
//! candidate and the selector falls back to the best lossy mode.

use super::ctu::{CellMode, LeafCu, MergeDir};
use super::mode::packed_byte_len;

// ════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════

/// RDO tuning. `lambda_q8` is the rate-distortion tradeoff λ in Q8 fixed
/// point (λ × 256): `0` minimizes rate, large values minimize distortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdoConfig {
    /// λ × 256. Cost = `(rate_bytes << 8) + lambda_q8 · distortion`.
    pub lambda_q8: u32,
}

impl RdoConfig {
    /// λ = 0 — pure rate minimization (always the cheapest feasible mode,
    /// regardless of fidelity loss).
    pub const RATE_ONLY: Self = Self { lambda_q8: 0 };

    /// A λ large enough that any non-zero distortion outweighs the 6-byte
    /// Escape rate, i.e. effectively lossless mode selection. `1 << 20`
    /// (λ = 4096) dominates the maximum 6-byte rate (`6 << 8 = 1536`) for
    /// any distortion ≥ 1.
    pub const LOSSLESS: Self = Self { lambda_q8: 1 << 20 };

    /// Construct from a fixed-point λ.
    ///
    /// ```
    /// use ndarray::hpc::codec::rdo::RdoConfig;
    /// // λ = 2.5 → lambda_q8 = 640
    /// let cfg = RdoConfig::from_lambda_q8(640);
    /// assert_eq!(cfg.lambda_q8, 640);
    /// ```
    pub const fn from_lambda_q8(lambda_q8: u32) -> Self {
        Self { lambda_q8 }
    }
}

impl Default for RdoConfig {
    /// A middle λ (= 16.0) biased toward fidelity but not strictly
    /// lossless — Skip/Merge a cell only when the distortion is small.
    fn default() -> Self {
        Self { lambda_q8: 16 << 8 }
    }
}

// ════════════════════════════════════════════════════════════════════
// Context + result
// ════════════════════════════════════════════════════════════════════

/// Per-cell context for the RDO decision. Mirrors
/// [`super::predict::IntraContext`]: a pre-resolved basin index, the
/// signed δ from basin to cell (in u8-quantization space), and the four
/// NEWS neighbour leaves (indexed by [`MergeDir`] discriminant:
/// `North=0, East=1, West=2, South=3`).
#[derive(Debug, Clone, Copy)]
pub struct RdoContext<'a> {
    /// Pre-resolved basin index (12-bit max, not re-validated here).
    pub basin_idx: u16,
    /// Signed δ from basin → cell, in the basin's u8 quantization space.
    pub delta_i32: i32,
    /// NEWS neighbour leaves, `None` at block boundaries.
    pub neighbours: [Option<&'a LeafCu>; 4],
}

/// The selected mode plus its scored cost and reconstruction distortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdoChoice {
    /// The chosen leaf, ready to pack via [`super::mode::pack_leaf`].
    pub leaf: LeafCu,
    /// `(rate_bytes << 8) + lambda_q8 · distortion` for the chosen mode.
    pub cost_q8: u64,
    /// Reconstruction error |δ − δ̂| in u8-quantization units (0 = exact).
    pub distortion: u32,
}

// ════════════════════════════════════════════════════════════════════
// The selector
// ════════════════════════════════════════════════════════════════════

/// Score all feasible modes and return the minimum-cost
/// [`RdoChoice`].
///
/// Feasible modes are scored in the order Skip → Merge → Delta → Escape;
/// ties are resolved in favour of the earlier (lower-rate) mode via a
/// strict `<` comparison. Skip and Delta are always feasible; Merge is
/// feasible only when a NEWS neighbour is a `Delta`-mode leaf on the same
/// basin; Escape is feasible only when `escape_next` is supplied.
///
/// When Escape wins, the caller's escape-vector cursor is post-incremented
/// (the chosen leaf references the pre-increment value) so batched calls
/// don't collide — identical to [`super::predict::predict_intra`]. The
/// cursor is left untouched if any other mode wins.
///
/// # Examples
///
/// λ = 0 picks the cheapest mode regardless of distortion — Skip (2
/// bytes) even when δ ≠ 0:
///
/// ```
/// use ndarray::hpc::codec::rdo::{rdo_select, RdoConfig, RdoContext};
/// use ndarray::hpc::codec::CellMode;
/// let ctx = RdoContext { basin_idx: 7, delta_i32: 40, neighbours: [None; 4] };
/// let choice = rdo_select(&ctx, &RdoConfig::RATE_ONLY, None);
/// assert_eq!(choice.leaf.mode, CellMode::Skip);
/// assert_eq!(choice.distortion, 40); // lossy: reconstructs as basin
/// ```
///
/// A lossless λ keeps the same in-range δ exact via Delta (0 distortion):
///
/// ```
/// use ndarray::hpc::codec::rdo::{rdo_select, RdoConfig, RdoContext};
/// use ndarray::hpc::codec::CellMode;
/// let ctx = RdoContext { basin_idx: 7, delta_i32: 40, neighbours: [None; 4] };
/// let choice = rdo_select(&ctx, &RdoConfig::LOSSLESS, None);
/// assert_eq!(choice.leaf.mode, CellMode::Delta);
/// assert_eq!(choice.distortion, 0);
/// ```
pub fn rdo_select(ctx: &RdoContext, cfg: &RdoConfig, escape_next: Option<&mut u32>) -> RdoChoice {
    let lambda = cfg.lambda_q8 as u64;

    // ── Skip (always feasible) ───────────────────────────────────────
    // Reconstructs the cell as the basin exactly; distortion = |δ|.
    let mut best = Candidate::new(LeafCu::skip(ctx.basin_idx), ctx.delta_i32.unsigned_abs(), lambda);

    // ── Merge (feasible if a Delta-mode same-basin neighbour exists) ──
    // The reconstructed δ is the neighbour's stored δ read back as i8.
    // Among qualifying neighbours, pick the one with the smallest
    // distortion (closest δ); ties keep the earlier NEWS slot.
    if let Some((dir, nb_dist)) = best_merge_neighbour(ctx) {
        consider(&mut best, Candidate::new(LeafCu::merge(ctx.basin_idx, dir), nb_dist, lambda));
    }

    // ── Delta (always feasible) ──────────────────────────────────────
    // δ clamped into i8 range; distortion = |δ − clamp(δ)| (0 in range).
    let clamped = ctx.delta_i32.clamp(-128, 127);
    let delta_dist = ctx.delta_i32.abs_diff(clamped);
    consider(&mut best, Candidate::new(LeafCu::delta(ctx.basin_idx, clamped as u8), delta_dist, lambda));

    // ── Escape (feasible only with a cursor) ─────────────────────────
    // Lossless: the full cell value is preserved in the escape vector, so
    // distortion = 0. Build the candidate against the *current* cursor
    // value, but only advance the cursor (post-increment) if Escape wins —
    // a losing Escape leaves the caller's allocator untouched.
    if let Some(next) = escape_next {
        let cand = Candidate::new(LeafCu::escape(ctx.basin_idx, *next), 0, lambda);
        if cand.cost < best.cost {
            *next = next.wrapping_add(1);
            best = cand;
        }
    }

    RdoChoice {
        leaf: best.leaf,
        cost_q8: best.cost,
        distortion: best.distortion,
    }
}

// ════════════════════════════════════════════════════════════════════
// Internals
// ════════════════════════════════════════════════════════════════════

/// A scored mode candidate during selection. `cost` is computed at
/// construction from the leaf's wire rate, the distortion, and λ.
struct Candidate {
    leaf: LeafCu,
    distortion: u32,
    cost: u64,
}

impl Candidate {
    /// Score `leaf` (rate derived from its mode) at the given distortion
    /// and fixed-point λ.
    #[inline]
    fn new(leaf: LeafCu, distortion: u32, lambda_q8: u64) -> Self {
        let rate = packed_byte_len(leaf.mode) as u64;
        Self {
            leaf,
            distortion,
            cost: cost_q8(rate, distortion, lambda_q8),
        }
    }
}

/// `(rate << 8) + λ_q8 · distortion`, saturating so an extreme λ can't
/// wrap. `u64` headroom: rate ≤ 6, distortion ≤ ~2³¹, λ ≤ 2³² → the
/// product is ≤ 2⁶³, well inside `u64`, but saturating keeps it total.
#[inline]
fn cost_q8(rate_bytes: u64, distortion: u32, lambda_q8: u64) -> u64 {
    (rate_bytes << 8).saturating_add(lambda_q8.saturating_mul(distortion as u64))
}

/// Replace `best` with `cand` if `cand` is strictly cheaper. Strict `<`
/// keeps the earlier (lower-rate) mode on ties, giving deterministic
/// selection.
#[inline]
fn consider(best: &mut Candidate, cand: Candidate) {
    if cand.cost < best.cost {
        *best = cand;
    }
}

/// Find the NEWS neighbour that yields the smallest Merge distortion.
///
/// A neighbour qualifies iff it is a `Delta`-mode leaf on the same basin
/// (Merge inherits a δ, and only across the same reference basin). The
/// reconstructed δ is the neighbour's stored byte read back as `i8`.
/// Returns `(dir, distortion)` for the best qualifying neighbour, or
/// `None` if none qualify.
fn best_merge_neighbour(ctx: &RdoContext) -> Option<(MergeDir, u32)> {
    let mut best: Option<(MergeDir, u32)> = None;
    for (i, slot) in ctx.neighbours.iter().enumerate() {
        let Some(nb) = slot else { continue };
        if nb.mode != CellMode::Delta || nb.basin_idx != ctx.basin_idx {
            continue;
        }
        let Some(nb_delta) = nb.delta else { continue };
        let recon = (nb_delta as i8) as i32;
        let dist = ctx.delta_i32.abs_diff(recon);
        let dir = merge_dir_from_index(i);
        match best {
            Some((_, best_dist)) if best_dist <= dist => {}
            _ => best = Some((dir, dist)),
        }
    }
    best
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

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(basin: u16, delta: i32, neighbours: [Option<&'a LeafCu>; 4]) -> RdoContext<'a> {
        RdoContext {
            basin_idx: basin,
            delta_i32: delta,
            neighbours,
        }
    }

    #[test]
    fn skip_when_delta_zero_at_any_lambda() {
        for cfg in [RdoConfig::RATE_ONLY, RdoConfig::default(), RdoConfig::LOSSLESS] {
            let choice = rdo_select(&ctx(7, 0, [None; 4]), &cfg, None);
            assert_eq!(choice.leaf.mode, CellMode::Skip);
            assert_eq!(choice.distortion, 0);
        }
    }

    #[test]
    fn rate_only_prefers_skip_even_when_lossy() {
        let choice = rdo_select(&ctx(7, 50, [None; 4]), &RdoConfig::RATE_ONLY, None);
        assert_eq!(choice.leaf.mode, CellMode::Skip);
        assert_eq!(choice.distortion, 50);
        // λ = 0 → cost is pure rate = 2 bytes << 8.
        assert_eq!(choice.cost_q8, 2 << 8);
    }

    #[test]
    fn lossless_lambda_keeps_in_range_delta_exact() {
        let choice = rdo_select(&ctx(7, 40, [None; 4]), &RdoConfig::LOSSLESS, None);
        assert_eq!(choice.leaf.mode, CellMode::Delta);
        assert_eq!(choice.leaf.delta, Some(40));
        assert_eq!(choice.distortion, 0);
    }

    #[test]
    fn negative_in_range_delta_packs_wrapping() {
        let choice = rdo_select(&ctx(7, -40, [None; 4]), &RdoConfig::LOSSLESS, None);
        assert_eq!(choice.leaf.mode, CellMode::Delta);
        assert_eq!(choice.leaf.delta, Some((-40_i32) as u8));
        assert_eq!(choice.distortion, 0);
    }

    #[test]
    fn out_of_range_delta_chooses_escape_when_lossless_and_allocator() {
        let mut next = 3u32;
        let choice = rdo_select(&ctx(7, 1000, [None; 4]), &RdoConfig::LOSSLESS, Some(&mut next));
        assert_eq!(choice.leaf.mode, CellMode::Escape);
        assert_eq!(choice.leaf.escape_idx, Some(3));
        assert_eq!(choice.distortion, 0);
        assert_eq!(next, 4, "cursor advances only when Escape wins");
    }

    #[test]
    fn escape_cursor_untouched_when_escape_loses() {
        // δ = 0 → Skip wins (distortion 0, cheapest rate); the cursor must
        // not advance even though an allocator was supplied.
        let mut next = 9u32;
        let choice = rdo_select(&ctx(7, 0, [None; 4]), &RdoConfig::LOSSLESS, Some(&mut next));
        assert_eq!(choice.leaf.mode, CellMode::Skip);
        assert_eq!(next, 9, "cursor must not advance when Escape loses");
    }

    #[test]
    fn out_of_range_delta_without_allocator_falls_back_to_clamped_delta() {
        // No escape cursor → Escape infeasible; the best lossless-ish
        // option is a clamped Delta (lossy).
        let choice = rdo_select(&ctx(7, 1000, [None; 4]), &RdoConfig::LOSSLESS, None);
        assert_eq!(choice.leaf.mode, CellMode::Delta);
        assert_eq!(choice.leaf.delta, Some(127));
        assert_eq!(choice.distortion, 1000 - 127);
    }

    #[test]
    fn merge_chosen_when_neighbour_matches_and_lambda_rewards_it() {
        // Northern neighbour is Delta(basin=7, δ=20). True δ = 20, so both
        // Merge and Delta reconstruct exactly (distortion 0) at 3 bytes —
        // an exact cost tie. Skip (2 bytes) has distortion 20 and loses at
        // lossless λ. Merge is scored before Delta, so it wins the tie.
        let nb = LeafCu::delta(7, 20);
        let neighbours = [Some(&nb), None, None, None];
        let choice = rdo_select(&ctx(7, 20, neighbours), &RdoConfig::LOSSLESS, None);
        assert_eq!(choice.leaf.mode, CellMode::Merge);
        assert_eq!(choice.leaf.merge_dir, Some(MergeDir::North));
        assert_eq!(choice.distortion, 0);
    }

    #[test]
    fn merge_picks_closest_neighbour() {
        // Among qualifying neighbours, `best_merge_neighbour` must pick the
        // closest δ. An exact-match Merge ties Delta (both 3 bytes, dist 0)
        // and wins on scoring order; a non-exact neighbour never wins
        // (Delta represents any in-range δ optimally), so the selector must
        // surface the EXACT neighbour, not the off-by-5 one.
        let east = LeafCu::delta(7, 20); // exact: |20 − 20| = 0
        let west = LeafCu::delta(7, 25); // off:   |20 − 25| = 5
        let neighbours = [None, Some(&east), Some(&west), None];
        let choice = rdo_select(&ctx(7, 20, neighbours), &RdoConfig::default(), None);
        assert_eq!(choice.leaf.mode, CellMode::Merge);
        assert_eq!(choice.leaf.merge_dir, Some(MergeDir::East));
        assert_eq!(choice.distortion, 0);
    }

    #[test]
    fn merge_rejected_when_basin_differs() {
        let nb = LeafCu::delta(99, 20); // different basin
        let neighbours = [Some(&nb), None, None, None];
        let choice = rdo_select(&ctx(7, 20, neighbours), &RdoConfig::LOSSLESS, None);
        // Can't Merge across basins → exact Delta(20) instead.
        assert_eq!(choice.leaf.mode, CellMode::Delta);
        assert_eq!(choice.leaf.delta, Some(20));
    }

    #[test]
    fn merge_rejected_when_neighbour_not_delta_mode() {
        let nb_skip = LeafCu::skip(7);
        let nb_merge = LeafCu::merge(7, MergeDir::North);
        let neighbours = [Some(&nb_skip), Some(&nb_merge), None, None];
        let choice = rdo_select(&ctx(7, 20, neighbours), &RdoConfig::LOSSLESS, None);
        assert_eq!(choice.leaf.mode, CellMode::Delta);
    }

    #[test]
    fn cost_is_rate_plus_lambda_distortion() {
        // λ_q8 = 256 (λ = 1.0), δ = 10, no neighbours, no escape.
        // Skip:  rate 2, dist 10 → (2<<8) + 256*10 = 512 + 2560 = 3072
        // Delta: rate 3, dist 0  → (3<<8) + 0       = 768
        // Delta wins.
        let cfg = RdoConfig::from_lambda_q8(256);
        let choice = rdo_select(&ctx(7, 10, [None; 4]), &cfg, None);
        assert_eq!(choice.leaf.mode, CellMode::Delta);
        assert_eq!(choice.cost_q8, 768);
    }

    #[test]
    fn chosen_leaf_round_trips_through_mode_pack() {
        use super::super::mode::{pack_leaf, unpack_leaf};
        let nb = LeafCu::delta(7, 20);
        let neighbours = [None, Some(&nb), None, None];
        let choice = rdo_select(&ctx(7, 20, neighbours), &RdoConfig::LOSSLESS, None);
        let mut buf = [0u8; 6];
        let n = pack_leaf(&choice.leaf, &mut buf).unwrap();
        let (decoded, consumed) = unpack_leaf(&buf).unwrap();
        assert_eq!(n, consumed);
        assert_eq!(decoded, choice.leaf);
    }

    #[test]
    fn candidate_new_matches_free_cost_fn() {
        // `Candidate::new` derives rate from the leaf mode (Delta = 3) and
        // must agree with the free `cost_q8` used for the Escape compare.
        let c = Candidate::new(LeafCu::delta(7, 5), 7, 256);
        assert_eq!(c.cost, cost_q8(3, 7, 256));
    }
}
