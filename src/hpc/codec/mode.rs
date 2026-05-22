//! Mode bit-pack / unpack helpers (PR-X12 A2).
//!
//! Compact wire-friendly representation of the [`CellMode`] +
//! [`MergeDir`] + [`LeafCu`] fields from [`super::ctu`]. The functions
//! here are the inverse of each other and pack into the smallest
//! integer width that fits, leaving the per-mode payload (`delta`,
//! `escape_idx`) for callers to append/consume as raw bytes.
//!
//! # Header layout — `pack_header` / `unpack_header`
//!
//! Each leaf has a fixed 16-bit header followed by a variable-width
//! tail. The header packs the most-frequently-accessed fields so a
//! decoder can route on a single `u16` load:
//!
//! ```text
//!     bit  15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
//!         ┌──┬──┬──┬──┬──────────────────────┐
//!         │ 0│ 0│M1│M0│      basin_idx (12)  │   ← 16-bit header
//!         └──┴──┴──┴──┴──────────────────────┘
//!         │  │  │  │  └────────────────────── basin_idx (bits 0..=11)
//!         │  │  └──┴────────────────────────── 2-bit mode (bits 12..=13)
//!         └──┴──────────────────────────────── reserved high bits 14,15
//! ```
//!
//! Bits 14-15 are reserved at zero; the impl is
//! `(mode_bits << 12) | basin_bits`, so mode lives at bits 12-13 and
//! basin at bits 0-11. A future encoder can repurpose bits 14-15
//! (e.g., for a per-leaf `merge_dir` overlap) without disturbing
//! existing decoders that mask bits 14-15 off.
//!
//! # Per-mode tail width
//!
//! | Mode   | Header | Tail bytes               | Total |
//! |--------|--------|--------------------------|-------|
//! | Skip   | 2      | 0                            | 2     |
//! | Merge  | 2      | 1 (low 2 bits = `MergeDir`)  | 3     |
//! | Delta  | 2      | 1 (`u8` perturbation)        | 3     |
//! | Escape | 2      | 4 (`u32` escape_idx, LE)     | 6     |
//!
//! The Merge tail is a full byte even though only its low 2 bits carry
//! `MergeDir` — high 6 bits are reserved and masked off on read.
//!
//! The compact pack writes header (LE) then the per-mode tail. The
//! `escape_idx` width is the worst case; a future A7 rANS pass can
//! shrink it via per-frame frequency tables — A2 stays format-stable.
//!
//! # What A2 does NOT do
//!
//! - **Bytestream framing** (frame headers, CTU markers) — lives in
//!   PR-X12 A8 `stream.rs`.
//! - **Entropy coding** (rANS) — lives in PR-X12 A7 `ans.rs`. A2's
//!   output is the input to A7.
//! - **Per-frame escape vector** — caller maintains it; A2 packs the
//!   `escape_idx` referencing into the leaf header.

use super::ctu::{CellMode, LeafCu, MergeDir};

// ════════════════════════════════════════════════════════════════════
// Header pack / unpack (16-bit)
// ════════════════════════════════════════════════════════════════════

// ── Design invariant ────────────────────────────────────────────────
//
// The 12-bit basin field addresses exactly one of 4096 real Leaves
// (HHTL ontology, see `src/hpc/ogit_bridge/assets/cognitive/entities/Leaf.ttl`:
// 16 Hips × 16 Twigs × 16 Leaves per Heel, every Leaf carries a real
// `basinSignature`). Do NOT reserve a value as a "no basin" /
// "not yet decided" sentinel.
//
// Rationale: that's authoring-time uncertainty (encoder mid-decision)
// leaking into wire-format ontological state. Keep optionality in the
// type, not in a magic value:
//
//   - encoder scratch state → `Option<u16>` (Some = committed, None = TBD)
//   - persisted / wire-format → plain `u16`, every value real
//
// The doubt must collapse to Some(basin) before the leaf is packed.
// Once a leaf reaches the wire it has a basin, period.

/// Maximum encodable `basin_idx`. Equal to `(1 << 12) - 1 = 4095`,
/// the full 12-bit range.
///
/// The codebook follows the HHTL ontology (`Heel > Hip > Twig > Leaf`,
/// see the entity TTLs under `src/hpc/ogit_bridge/assets/cognitive/entities/`):
/// `16 Hips × 16 Twigs × 16 Leaves = 4096 Leaves per Heel`. Every value
/// `0..=MAX_BASIN_IDX` addresses one Leaf, each carrying a real
/// `basinSignature`.
///
/// ```
/// use ndarray::hpc::codec::MAX_BASIN_IDX;
/// assert_eq!(MAX_BASIN_IDX, (1 << 12) - 1);
/// assert_eq!(MAX_BASIN_IDX, 4095);
/// ```
pub const MAX_BASIN_IDX: u16 = (1 << 12) - 1; // 4095

/// Private: 12-bit mask for the basin field of the packed header.
const BASIN_FIELD_MASK: u16 = 0x0FFF;

/// Pack `(mode, basin_idx)` into a 16-bit header.
///
/// `basin_idx` must be `<= MAX_BASIN_IDX` (12 bits). Higher bits are
/// silently truncated; the encoder should clamp before calling.
///
/// ```
/// use ndarray::hpc::codec::mode::{pack_header, unpack_header};
/// use ndarray::hpc::codec::CellMode;
/// let h = pack_header(CellMode::Delta, 1234);
/// assert_eq!(unpack_header(h), (CellMode::Delta, 1234));
/// ```
#[inline]
pub fn pack_header(mode: CellMode, basin_idx: u16) -> u16 {
    let mode_bits = (mode as u16) & 0b11;
    let basin_bits = basin_idx & BASIN_FIELD_MASK;
    (mode_bits << 12) | basin_bits
}

/// Unpack a 16-bit header into `(mode, basin_idx)`.
///
/// The 2-bit mode field always decodes (all 4 variants are valid).
/// `basin_idx` is the 12-bit lower field, exactly as packed.
///
/// ```
/// use ndarray::hpc::codec::{pack_header, unpack_header, CellMode};
/// let h = pack_header(CellMode::Escape, 7);
/// assert_eq!(unpack_header(h), (CellMode::Escape, 7));
/// ```
#[inline]
pub fn unpack_header(packed: u16) -> (CellMode, u16) {
    let mode_bits = ((packed >> 12) & 0b11) as u8;
    let basin_idx = packed & BASIN_FIELD_MASK;
    let mode = match mode_bits {
        0b00 => CellMode::Skip,
        0b01 => CellMode::Merge,
        0b10 => CellMode::Delta,
        _ => CellMode::Escape,
    };
    (mode, basin_idx)
}

// ════════════════════════════════════════════════════════════════════
// MergeDir 2-bit pack / unpack
// ════════════════════════════════════════════════════════════════════

/// Pack a [`MergeDir`] into the lower 2 bits of a `u8`.
///
/// ```
/// use ndarray::hpc::codec::{pack_merge_dir, MergeDir};
/// assert_eq!(pack_merge_dir(MergeDir::East), 1);
/// ```
#[inline]
pub fn pack_merge_dir(dir: MergeDir) -> u8 {
    dir as u8
}

/// Unpack the lower 2 bits of a `u8` into a [`MergeDir`].
///
/// All four 2-bit values map to a valid `MergeDir`; bits 2-7 are
/// ignored.
///
/// ```
/// use ndarray::hpc::codec::{pack_merge_dir, unpack_merge_dir, MergeDir};
/// for d in [MergeDir::North, MergeDir::East, MergeDir::West, MergeDir::South] {
///     assert_eq!(unpack_merge_dir(pack_merge_dir(d)), d);
/// }
/// ```
#[inline]
pub fn unpack_merge_dir(byte: u8) -> MergeDir {
    match byte & 0b11 {
        0 => MergeDir::North,
        1 => MergeDir::East,
        2 => MergeDir::West,
        _ => MergeDir::South,
    }
}

// ════════════════════════════════════════════════════════════════════
// Whole-leaf pack / unpack
// ════════════════════════════════════════════════════════════════════

/// Compact pack: writes header (2 bytes, LE) + per-mode tail into
/// `out`. Returns the number of bytes written.
///
/// The buffer must have at least 6 bytes of space (the Escape-mode
/// worst case) — callers iterating CTUs typically pre-allocate
/// `6 * cell_count` and trim afterwards.
///
/// Returns `None` in two cases:
/// - `out.len() < packed_byte_len(leaf.mode)` (insufficient capacity for
///   the *mode's* width — Skip needs 2, Merge/Delta need 3, Escape needs 6).
/// - `leaf` is structurally malformed for its mode: `Merge` without a
///   `merge_dir`, `Delta` without a `delta`, or `Escape` without an
///   `escape_idx`. The `LeafCu::merge` / `delta` / `escape` constructors
///   enforce these invariants; only struct-literal callers bypassing the
///   constructors can hit this case. Pack is therefore bijective on the
///   well-formed `LeafCu` subset.
///
/// Format:
/// - Bytes 0-1: header (`pack_header(mode, basin_idx)`, LE)
/// - Bytes 2..: per-mode tail (see module docs)
///
/// ```
/// use ndarray::hpc::codec::mode::{pack_leaf, unpack_leaf};
/// use ndarray::hpc::codec::LeafCu;
/// let leaf = LeafCu::delta(42, 0x7F);
/// let mut buf = [0u8; 6];
/// let n = pack_leaf(&leaf, &mut buf).unwrap();
/// assert_eq!(n, 3);
/// let (decoded, consumed) = unpack_leaf(&buf).unwrap();
/// assert_eq!(decoded, leaf);
/// assert_eq!(consumed, 3);
/// ```
pub fn pack_leaf(leaf: &LeafCu, out: &mut [u8]) -> Option<usize> {
    let required = packed_byte_len(leaf.mode);
    if out.len() < required {
        return None;
    }
    let header = pack_header(leaf.mode, leaf.basin_idx);
    out[..2].copy_from_slice(&header.to_le_bytes());
    // Per-mode tail. `?` rejects malformed `LeafCu`s (e.g. a hand-built
    // `LeafCu { mode: Merge, merge_dir: None, .. }`) with `None` rather
    // than silently rewriting them into a different valid leaf. The
    // `LeafCu::merge/delta/escape` constructors enforce the invariants;
    // only struct-literal callers bypassing those constructors hit
    // these short-circuits.
    let tail_len = match leaf.mode {
        CellMode::Skip => 0,
        CellMode::Merge => {
            out[2] = pack_merge_dir(leaf.merge_dir?);
            1
        }
        CellMode::Delta => {
            out[2] = leaf.delta?;
            1
        }
        CellMode::Escape => {
            let idx = leaf.escape_idx?;
            out[2..6].copy_from_slice(&idx.to_le_bytes());
            4
        }
    };
    Some(2 + tail_len)
}

/// Compact unpack: reads header + per-mode tail from `buf`. Returns
/// `(leaf, bytes_consumed)`.
///
/// Returns `None` if the buffer is shorter than the per-mode width
/// (2 for Skip, 3 for Merge/Delta, 6 for Escape).
///
/// ```
/// use ndarray::hpc::codec::{pack_leaf, unpack_leaf, LeafCu, MergeDir};
/// let leaf = LeafCu::merge(7, MergeDir::West);
/// let mut buf = [0u8; 3];
/// pack_leaf(&leaf, &mut buf).unwrap();
/// let (decoded, n) = unpack_leaf(&buf).unwrap();
/// assert_eq!(decoded, leaf);
/// assert_eq!(n, 3);
/// ```
pub fn unpack_leaf(buf: &[u8]) -> Option<(LeafCu, usize)> {
    if buf.len() < 2 {
        return None;
    }
    let header = u16::from_le_bytes([buf[0], buf[1]]);
    let (mode, basin_idx) = unpack_header(header);
    let (leaf, consumed) = match mode {
        CellMode::Skip => (LeafCu::skip(basin_idx), 2),
        CellMode::Merge => {
            if buf.len() < 3 {
                return None;
            }
            (LeafCu::merge(basin_idx, unpack_merge_dir(buf[2])), 3)
        }
        CellMode::Delta => {
            if buf.len() < 3 {
                return None;
            }
            (LeafCu::delta(basin_idx, buf[2]), 3)
        }
        CellMode::Escape => {
            if buf.len() < 6 {
                return None;
            }
            let idx = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
            (LeafCu::escape(basin_idx, idx), 6)
        }
    };
    Some((leaf, consumed))
}

/// Byte cost of packing a leaf in this mode. Useful for pre-sizing
/// a buffer without packing first.
///
/// ```
/// use ndarray::hpc::codec::{packed_byte_len, CellMode};
/// assert_eq!(packed_byte_len(CellMode::Skip), 2);
/// assert_eq!(packed_byte_len(CellMode::Merge), 3);
/// assert_eq!(packed_byte_len(CellMode::Delta), 3);
/// assert_eq!(packed_byte_len(CellMode::Escape), 6);
/// ```
#[inline]
pub const fn packed_byte_len(mode: CellMode) -> usize {
    match mode {
        CellMode::Skip => 2,
        CellMode::Merge => 3,
        CellMode::Delta => 3,
        CellMode::Escape => 6,
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_all_modes_and_basin_extents() {
        for mode in [CellMode::Skip, CellMode::Merge, CellMode::Delta, CellMode::Escape] {
            for basin in [0u16, 1, 42, 1234, MAX_BASIN_IDX] {
                let h = pack_header(mode, basin);
                assert_eq!(unpack_header(h), (mode, basin), "mode={mode:?}, basin={basin}");
            }
        }
    }

    #[test]
    fn header_truncates_oversize_basin_idx() {
        // basin_idx = 4096 doesn't fit in 12 bits; the high bit gets
        // dropped, giving back basin=0.
        let h = pack_header(CellMode::Skip, 4096);
        let (_, basin) = unpack_header(h);
        assert_eq!(basin, 0);
    }

    #[test]
    fn merge_dir_roundtrip_all_four() {
        for dir in [MergeDir::North, MergeDir::East, MergeDir::West, MergeDir::South] {
            let b = pack_merge_dir(dir);
            assert_eq!(unpack_merge_dir(b), dir);
        }
    }

    #[test]
    fn merge_dir_ignores_high_bits() {
        // High bits 2-7 are reserved; unpack should mask them out.
        assert_eq!(unpack_merge_dir(0b1111_1100), MergeDir::North);
        assert_eq!(unpack_merge_dir(0b1111_1101), MergeDir::East);
    }

    #[test]
    fn leaf_pack_skip_is_2_bytes() {
        let leaf = LeafCu::skip(100);
        let mut buf = [0xAAu8; 6];
        let n = pack_leaf(&leaf, &mut buf).unwrap();
        assert_eq!(n, 2);
        // Bytes 2-5 untouched.
        assert_eq!(&buf[2..], &[0xAA, 0xAA, 0xAA, 0xAA]);
    }

    #[test]
    fn leaf_pack_merge_is_3_bytes() {
        let leaf = LeafCu::merge(100, MergeDir::East);
        let mut buf = [0u8; 6];
        let n = pack_leaf(&leaf, &mut buf).unwrap();
        assert_eq!(n, 3);
        let (decoded, consumed) = unpack_leaf(&buf).unwrap();
        assert_eq!(decoded, leaf);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn leaf_pack_delta_is_3_bytes() {
        let leaf = LeafCu::delta(100, 0xCC);
        let mut buf = [0u8; 6];
        let n = pack_leaf(&leaf, &mut buf).unwrap();
        assert_eq!(n, 3);
        let (decoded, consumed) = unpack_leaf(&buf).unwrap();
        assert_eq!(decoded, leaf);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn leaf_pack_escape_is_6_bytes() {
        let leaf = LeafCu::escape(100, 0xDEAD_BEEF);
        let mut buf = [0u8; 6];
        let n = pack_leaf(&leaf, &mut buf).unwrap();
        assert_eq!(n, 6);
        let (decoded, consumed) = unpack_leaf(&buf).unwrap();
        assert_eq!(decoded, leaf);
        assert_eq!(consumed, 6);
    }

    #[test]
    fn leaf_pack_rejects_short_buffer() {
        let leaf = LeafCu::escape(100, 0xDEAD_BEEF);
        let mut buf = [0u8; 5]; // 1 short of Escape's worst case
        assert!(pack_leaf(&leaf, &mut buf).is_none());
    }

    #[test]
    fn leaf_pack_rejects_malformed_merge_without_dir() {
        // Bypass `LeafCu::merge` constructor and hand-build a leaf with
        // mode = Merge but merge_dir = None. The previous unwrap_or(North)
        // behavior would silently coerce this into a valid leaf — now we
        // reject with None instead.
        let malformed = LeafCu {
            mode: CellMode::Merge,
            basin_idx: 10,
            delta: None,
            merge_dir: None,
            escape_idx: None,
        };
        let mut buf = [0u8; 6];
        assert!(pack_leaf(&malformed, &mut buf).is_none());
    }

    #[test]
    fn leaf_pack_rejects_malformed_delta_without_value() {
        let malformed = LeafCu {
            mode: CellMode::Delta,
            basin_idx: 10,
            delta: None,
            merge_dir: None,
            escape_idx: None,
        };
        let mut buf = [0u8; 6];
        assert!(pack_leaf(&malformed, &mut buf).is_none());
    }

    #[test]
    fn leaf_pack_rejects_malformed_escape_without_idx() {
        let malformed = LeafCu {
            mode: CellMode::Escape,
            basin_idx: 10,
            delta: None,
            merge_dir: None,
            escape_idx: None,
        };
        let mut buf = [0u8; 6];
        assert!(pack_leaf(&malformed, &mut buf).is_none());
    }

    #[test]
    fn max_basin_idx_fills_full_12bit_range() {
        // 16 Hips × 16 Twigs × 16 Leaves = 4096 Leaves per Heel
        // (HHTL ontology, see Leaf.ttl). Every value 0..=MAX_BASIN_IDX
        // addresses one Leaf.
        assert_eq!(MAX_BASIN_IDX, (1 << 12) - 1);
        assert_eq!(MAX_BASIN_IDX, 4095);
    }

    #[test]
    fn header_round_trips_max_basin_idx() {
        let h = pack_header(CellMode::Skip, MAX_BASIN_IDX);
        assert_eq!(unpack_header(h), (CellMode::Skip, MAX_BASIN_IDX));
    }

    #[test]
    fn leaf_unpack_rejects_short_buffer() {
        // Header says Escape but only 2 bytes follow → not enough.
        let mut buf = [0u8; 3];
        let header = pack_header(CellMode::Escape, 50);
        buf[..2].copy_from_slice(&header.to_le_bytes());
        assert!(unpack_leaf(&buf).is_none());
    }

    #[test]
    fn packed_byte_len_matches_pack_output() {
        let cases = [
            (LeafCu::skip(10), CellMode::Skip),
            (LeafCu::merge(10, MergeDir::West), CellMode::Merge),
            (LeafCu::delta(10, 7), CellMode::Delta),
            (LeafCu::escape(10, 99), CellMode::Escape),
        ];
        for (leaf, mode) in cases {
            let mut buf = [0u8; 6];
            let n = pack_leaf(&leaf, &mut buf).unwrap();
            assert_eq!(n, packed_byte_len(mode));
        }
    }

    #[test]
    fn stream_pack_then_unpack_roundtrips_mixed_leaves() {
        // Encode a sequence of mixed-mode leaves into one buffer,
        // decode in order, assert exact equality of all 8.
        let leaves = [
            LeafCu::skip(0),
            LeafCu::delta(1, 0xAB),
            LeafCu::merge(2, MergeDir::North),
            LeafCu::escape(3, 0xDEAD_BEEF),
            LeafCu::skip(MAX_BASIN_IDX),
            LeafCu::delta(MAX_BASIN_IDX, 0xFF),
            LeafCu::merge(MAX_BASIN_IDX, MergeDir::South),
            LeafCu::escape(MAX_BASIN_IDX, u32::MAX),
        ];
        // Worst case: 8 × 6 bytes = 48
        let mut buf = vec![0u8; 48];
        let mut offset = 0;
        for leaf in &leaves {
            let n = pack_leaf(leaf, &mut buf[offset..]).unwrap();
            offset += n;
        }
        let total_written = offset;
        // Decode in order.
        let mut decoded = Vec::with_capacity(8);
        let mut read = 0;
        while read < total_written {
            let (leaf, n) = unpack_leaf(&buf[read..]).unwrap();
            decoded.push(leaf);
            read += n;
        }
        assert_eq!(decoded.len(), 8);
        assert_eq!(&decoded[..], &leaves[..]);
        assert_eq!(read, total_written);
    }
}
