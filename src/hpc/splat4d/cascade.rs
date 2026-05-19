//! `splat4d::cascade` — 4-nibble Hilbert-3D cascade addressing.
//!
//! # Layout
//!
//! `CascadeAddr(u16)` packs four tier nibbles into a single `u16`:
//!
//! ```text
//! bits 15-12 │ bits 11-8 │ bits 7-4 │ bits 3-0
//!   L4 nibble   L3 nibble   L2 nibble   L1 nibble
//! ```
//!
//! Each nibble selects one of 16 children at tier L1..L4, giving 16 children
//! per parent. `parent()` masks off the L4 nibble; `children()` enumerates
//! all 16 nibble values at the L4 slot.
//!
//! # Gate — PR-X10 A12b L4 Hilbert-3D fix
//!
//! **The L4 addressing path is blocked until A12b lands.**
//!
//! Verbatim bug from `pp13-brutally-honest-tester-verdict.md` § P0-4:
//! `hilbert3d_encode([15,15,15], 4)` returns **2925**, expected **4095**.
//! At level=4 the encode is not a bijection on `[0, 4096)`. Until A12b ships
//! the `NEXT_STATE`/`H_TO_XYZ` re-derivation from Hamilton 2006 Table 2 plus
//! `round_trip_level4_exhaustive`, calling `from_position_l4` returns
//! `Err(NotReadyL4)`. L1-L3 paths are exercisable today.
//!
//! **Do NOT re-introduce a bespoke Hilbert-3D here** (forbidden constraint #1
//! from the PR-X4 pre-sprint prompt). All Hilbert encoding is delegated to
//! `crate::hpc::linalg::hilbert::{hilbert3d_encode, hilbert3d_decode}`.
//!
//! # SIMD bundle consumed
//!
//! - **B-Cascade-Permute** (`shuffle_lanes_4x4 ∘ transpose_16x16`): the
//!   cross-tier rotation L_k → L_{k+1}. A2 must not reach past this bundle
//!   into raw `std::arch::*` intrinsics. If the bundle primitive is absent
//!   from `ndarray::simd`, file a pre-PR-X4 gating issue against
//!   `vertical-simd-consumer-contract.md` before implementing.
//!
//! # AABB quantisation convention
//!
//! At level `k`, each axis has `2^k` cells. Quantisation:
//!
//! ```text
//! q = floor((p - bbox.min) / bbox.size * (1 << level))
//! ```
//!
//! clamped to `[0, (1 << level) - 1]` per axis. The 4-nibble layout covers
//! L1 (2 cells/axis), L2 (4), L3 (8), L4 (16). L4's 12-bit Hilbert range
//! fits within 3 of the 4 cascade-addr nibbles; nibbles are assembled per-tier
//! if `hilbert3d_encode` returns a monolithic per-call index rather than a
//! packed cascade representation (flag this discrepancy with the A12b author
//! at spawn time per `02-a2-cascadeaddr-brief.md`).
//!
//! # Cross-references
//!
//! - `pr-x4-planning/02-a2-cascadeaddr-brief.md` — full A2 worker brief
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Cascade addressing"
//! - `pp13-brutally-honest-tester-verdict.md` § P0-4 — L4 bug
//! - `crate::hpc::linalg::hilbert` — the Hilbert-3D encode/decode A2 consumes

use crate::hpc::aabb::Aabb;

// ════════════════════════════════════════════════════════════════════════════
// Error type — L4 gate sentinel
// ════════════════════════════════════════════════════════════════════════════

/// Error returned by [`CascadeAddr::from_position_l4`] until PR-X10 A12b's
/// `round_trip_level4_exhaustive` test passes on master.
///
/// `hilbert3d_encode([15,15,15], 4)` currently returns 2925, not 4095;
/// the L4 encode is not a bijection. Ship L1-L3 first.
///
/// Gate: A12b must land `hilbert3d_encode([15,15,15], 4) == 4095` and
/// the exhaustive `round_trip_level4_exhaustive` test (4096 cells × 4 µs ≈
/// 16 ms) before this error is removed.
///
/// Reference: `pp13-brutally-honest-tester-verdict.md` § P0-4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotReadyL4;

impl core::fmt::Display for NotReadyL4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            "L4 cascade addressing requires PR-X10 A12b hilbert3d_encode([15,15,15],4)==4095 \
             (currently returns 2925); round_trip_level4_exhaustive must pass on master first",
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CascadeAddr
// ════════════════════════════════════════════════════════════════════════════

/// 4-nibble Hilbert-3D cascade address for the (4×4)^4 splat cascade.
///
/// The `u16` payload packs one nibble per tier:
///
/// ```text
/// bits 15-12 = L4 nibble
/// bits 11- 8 = L3 nibble
/// bits  7- 4 = L2 nibble
/// bits  3- 0 = L1 nibble
/// ```
///
/// Each nibble selects one of 16 children at its tier. `parent()` clears the
/// L4 nibble; `children()` enumerates all 16 L4-slot values for this parent.
///
/// # Copy semantics
///
/// `CascadeAddr` is `Copy` — 2 bytes, stack-allocated, zero heap.
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::hpc::splat4d::cascade::CascadeAddr;
/// use ndarray::hpc::aabb::Aabb;
///
/// let bbox = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
/// let addr = CascadeAddr::from_position([0.5, 0.5, 0.5], bbox, 3);
/// let center = addr.to_position_center(bbox);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CascadeAddr(pub u16);

impl CascadeAddr {
    // ── Nibble accessors ─────────────────────────────────────────────────────

    /// Extract the nibble at tier `l` (0 = L1 .. 3 = L4).
    ///
    /// Returns the 4-bit value at nibble position `l` from the low end.
    /// `l == 0` extracts bits 3-0; `l == 3` extracts bits 15-12.
    ///
    /// # Panics
    ///
    /// Panics in debug if `l > 3`.
    #[inline]
    pub fn level(&self, l: u8) -> u8 {
        debug_assert!(l <= 3, "level nibble index must be 0..=3, got {l}");
        ((self.0 >> (l * 4)) & 0xF) as u8
    }

    // ── Hierarchy navigation ─────────────────────────────────────────────────

    /// Return the parent address by clearing the L4 nibble (bits 15-12).
    ///
    /// The parent is the address at tier L3 that contains this cell. As per
    /// the 4-nibble layout, masking off bits 15-12 gives the L3 super-block
    /// address that all 16 L4 children of the same parent share.
    ///
    /// Invariant: `addr.children()[i].parent() == addr` for all `i` in 0..16.
    #[inline]
    pub fn parent(&self) -> CascadeAddr {
        CascadeAddr(self.0 & !0xF000)
    }

    /// Enumerate all 16 children by filling the L4 nibble with 0..=15.
    ///
    /// The returned array contains every `CascadeAddr` whose `parent()` is
    /// `self`. Children are ordered by nibble value (0-indexed). Used by the
    /// cascade streaming kernel to walk L4 sub-cells of an L3 super-block.
    ///
    /// Invariant: for all `i` in 0..16, `self.children()[i].parent() == self`.
    #[inline]
    pub fn children(&self) -> [CascadeAddr; 16] {
        //// TODO(A2): implement via B-Cascade-Permute once the bundle lands in
        //// `ndarray::simd::shuffle_lanes_4x4`. For now use scalar fill. If
        //// the bundle is absent, file a pre-PR-X4 gating issue against
        //// `vertical-simd-consumer-contract.md`.
        //// Ref: `02-a2-cascadeaddr-brief.md` § "SIMD bundle — B-Cascade-Permute"
        todo!("CascadeAddr::children — fill 16 L4 nibbles via B-Cascade-Permute")
    }

    // ── Position ↔ address ───────────────────────────────────────────────────

    /// Encode a world-space position `p` into a `CascadeAddr` at tier `level`
    /// (1..=3 supported today; level=4 requires A12b — use
    /// [`from_position_l4`](CascadeAddr::from_position_l4) for the gated path).
    ///
    /// # Quantisation
    ///
    /// For each axis:
    ///
    /// ```text
    /// q_axis = floor((p_axis - bbox.min[axis]) / bbox_size[axis] * (1 << level))
    ///          clamped to [0, (1 << level) - 1]
    /// ```
    ///
    /// The quantised 3-tuple is passed to
    /// `crate::hpc::linalg::hilbert::hilbert3d_encode(q, level)`.
    ///
    /// # Gate note
    ///
    /// Level=4 internally calls `hilbert3d_encode` with level=4 which is
    /// currently broken (returns 2925 for `[15,15,15]`). This function
    /// delegates to [`from_position_l4`](CascadeAddr::from_position_l4) for
    /// level=4 and propagates `Err(NotReadyL4)` as a panic if called with
    /// `level == 4` here. Prefer the `_l4` variant for gated callers.
    ///
    /// # Panics
    ///
    /// Panics if `level == 0` or `level > 4`.
    ///
    /// # References
    ///
    /// - `02-a2-cascadeaddr-brief.md` § "AABB quantisation convention"
    /// - `crate::hpc::linalg::hilbert` (PR-X10 A12b deliverable)
    pub fn from_position(p: [f32; 3], bbox: Aabb, level: u8) -> CascadeAddr {
        //// TODO(A2): quantise p per-axis using the AABB convention above, then
        //// call `crate::hpc::linalg::hilbert::hilbert3d_encode(q, level) as u16`.
        //// For level=4, this function must panic with a NotReadyL4 message until
        //// A12b's `round_trip_level4_exhaustive` passes on master.
        //// Note: if hilbert3d_encode returns a monolithic per-call index rather
        //// than a packed cascade, call once per tier and assemble nibbles.
        //// Ref: `02-a2-cascadeaddr-brief.md` § "AABB quantisation convention"
        //// Ref: B-Cascade-Permute bundle for the cross-tier rotation
        let _ = (p, bbox, level);
        todo!("CascadeAddr::from_position — quantise + hilbert3d_encode(q, level)")
    }

    /// L4-gated version of `from_position`. Returns `Err(NotReadyL4)` until
    /// PR-X10 A12b's `hilbert3d_encode([15,15,15], 4) == 4095` passes on master.
    ///
    /// Callers that need L4 addressing should use this variant so the gate
    /// is explicit and the `NotReadyL4` sentinel propagates cleanly through
    /// the pipeline rather than panicking.
    ///
    /// # Gate condition
    ///
    /// Remove the `Err(NotReadyL4)` short-circuit and delegate to
    /// `from_position(p, bbox, 4)` once:
    ///
    /// 1. `hilbert3d_encode([15,15,15], 4) == 4095` on master (A12b fix).
    /// 2. `round_trip_level4_exhaustive` test (4096 cells) passes green.
    ///
    /// Reference: `pp13-brutally-honest-tester-verdict.md` § P0-4.
    ///
    /// Reference: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md`
    ///            § "Forbidden constraints" #1.
    pub fn from_position_l4(p: [f32; 3], bbox: Aabb) -> Result<CascadeAddr, NotReadyL4> {
        //// TODO(A2 — gate): remove Err(NotReadyL4) return once PR-X10 A12b lands
        //// `hilbert3d_encode([15,15,15], 4) == 4095` and
        //// `round_trip_level4_exhaustive` passes on master. See
        //// `pp13-brutally-honest-tester-verdict.md` § P0-4 for root cause.
        //// Do NOT re-derive or fork hilbert3d_encode here (forbidden #1).
        let _ = (p, bbox);
        Err(NotReadyL4)
    }

    /// Decode a `CascadeAddr` to the world-space center of its grid cell.
    ///
    /// Inverse of `from_position`: decodes via
    /// `crate::hpc::linalg::hilbert::hilbert3d_decode`, then maps the
    /// quantised cell index back to the cell center in world space:
    ///
    /// ```text
    /// center_axis = bbox.min[axis] + (q_axis + 0.5) / (1 << level) * bbox_size[axis]
    /// ```
    ///
    /// # Gate note
    ///
    /// At level=4, `hilbert3d_decode` depends on the same table as `encode`;
    /// decode results are currently unreliable at L4 until A12b lands.
    ///
    /// # References
    ///
    /// - `02-a2-cascadeaddr-brief.md` § "AABB quantisation convention"
    /// - `crate::hpc::linalg::hilbert` (PR-X10 A12b deliverable)
    pub fn to_position_center(&self, bbox: Aabb, level: u8) -> [f32; 3] {
        //// TODO(A2): call `crate::hpc::linalg::hilbert::hilbert3d_decode(self.0 as u32, level)`,
        //// then map per-axis: center[ax] = bbox.min[ax] + (q[ax] + 0.5) / (1 << level) * size[ax].
        //// Ref: `02-a2-cascadeaddr-brief.md` § "AABB quantisation convention"
        let _ = (bbox, level);
        todo!("CascadeAddr::to_position_center — hilbert3d_decode + cell-center mapping")
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// L1-nibble accessor smoke test — no Hilbert dep needed.
    #[test]
    fn level_nibble_extracts_correct_bits() {
        // Build a known address: nibbles = [0xA, 0xB, 0xC, 0xD] (L1..L4).
        let addr = CascadeAddr(0xDCBA_u16);
        assert_eq!(addr.level(0), 0xA, "L1 nibble");
        assert_eq!(addr.level(1), 0xB, "L2 nibble");
        assert_eq!(addr.level(2), 0xC, "L3 nibble");
        assert_eq!(addr.level(3), 0xD, "L4 nibble");
    }

    /// `parent()` clears bits 15-12, leaving the L1-L3 nibbles intact.
    #[test]
    fn parent_clears_l4_nibble() {
        let addr = CascadeAddr(0xDCBA_u16);
        let parent = addr.parent();
        assert_eq!(parent.0, 0x0CBA, "parent must zero L4 nibble");
        assert_eq!(parent.level(0), 0xA);
        assert_eq!(parent.level(1), 0xB);
        assert_eq!(parent.level(2), 0xC);
        assert_eq!(parent.level(3), 0x0, "L4 nibble must be zero after parent()");
    }

    /// `from_position_l4` must return `Err(NotReadyL4)` unconditionally
    /// until A12b's level-4 round-trip test passes on master.
    #[test]
    fn from_position_l4_returns_not_ready() {
        let bbox = Aabb::new([0.0; 3], [1.0; 3]);
        let result = CascadeAddr::from_position_l4([0.5, 0.5, 0.5], bbox);
        assert_eq!(result, Err(NotReadyL4), "L4 path must be gated until A12b lands");
    }

    //// TODO(A2): add exhaustive level=1..3 round-trip tests once from_position
    //// and to_position_center are implemented. Verify:
    ////   for all q in [0, 2^level)^3:
    ////     decode(encode(q)) == q
    //// Ref: `02-a2-cascadeaddr-brief.md` § "Tests"

    //// TODO(A2): add parent/children round-trip exhaustive test once children()
    //// is implemented:
    ////   for any addr, addr.children()[i].parent() == addr for all i in 0..16
    //// Ref: `02-a2-cascadeaddr-brief.md` § "Tests"
}
