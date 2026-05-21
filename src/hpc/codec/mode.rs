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
//!     MSB                                                    LSB
//!     ┌──┬──┬──────────────────────────────┐
//!     │M0│M1│         basin_idx (12)        │   ← 16-bit header
//!     └──┴──┴──────────────────────────────┘
//!     │  │  └─ basin_idx is the only payload field always present
//!     └──┴──── 2-bit mode discriminant (CellMode::as_u8())
//!     (top 2 bits)
//! ```
//!
//! The remaining 2 bits at the top of the second byte are reserved for
//! the encoder's future `merge_dir` overlap when the mode is `Merge`;
//! a separate `pack_mode_dir` helper keeps `Merge`'s direction in a
//! single byte alongside `Skip`/`Delta`/`Escape`'s mode tag.
//!
//! # Per-mode tail width
//!
//! | Mode   | Header | Tail bytes               | Total |
//! |--------|--------|--------------------------|-------|
//! | Skip   | 2      | 0                        | 2     |
//! | Merge  | 2      | 1 (`MergeDir` 2-bit)     | 3     |
//! | Delta  | 2      | 1 (`u8` perturbation)    | 3     |
//! | Escape | 2      | 4 (`u32` escape_idx, LE) | 6     |
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

/// Maximum encodable `basin_idx`. Stored in the lower 12 bits of the
/// header; values >= this constant overflow the header field.
pub const MAX_BASIN_IDX: u16 = (1 << 12) - 1; // 4095

/// Tag inside the per-frame basin codebook for "no basin assigned"
/// (encoder-side sentinel during mode decision).
pub const BASIN_NONE: u16 = MAX_BASIN_IDX;

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
    let basin_bits = basin_idx & MAX_BASIN_IDX;
    (mode_bits << 12) | basin_bits
}

/// Unpack a 16-bit header into `(mode, basin_idx)`.
///
/// The 2-bit mode field always decodes (all 4 variants are valid).
/// `basin_idx` is the 12-bit lower field, exactly as packed.
#[inline]
pub fn unpack_header(packed: u16) -> (CellMode, u16) {
    let mode_bits = ((packed >> 12) & 0b11) as u8;
    let basin_idx = packed & MAX_BASIN_IDX;
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
#[inline]
pub fn pack_merge_dir(dir: MergeDir) -> u8 {
    dir as u8
}

/// Unpack the lower 2 bits of a `u8` into a [`MergeDir`].
///
/// All four 2-bit values map to a valid `MergeDir`; bits 2-7 are
/// ignored.
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
/// Returns `None` if `out.len() < 6` (insufficient capacity).
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
    if out.len() < 6 {
        return None;
    }
    let header = pack_header(leaf.mode, leaf.basin_idx);
    out[..2].copy_from_slice(&header.to_le_bytes());
    let tail_len = match leaf.mode {
        CellMode::Skip => 0,
        CellMode::Merge => {
            // Caller guarantees `merge_dir.is_some()` for `Merge` mode
            // (LeafCu::merge constructor enforces this). Fall back to
            // North if the invariant is violated, to keep encoder
            // robustness — the decoder will still produce a valid leaf.
            out[2] = pack_merge_dir(leaf.merge_dir.unwrap_or(MergeDir::North));
            1
        }
        CellMode::Delta => {
            out[2] = leaf.delta.unwrap_or(0);
            1
        }
        CellMode::Escape => {
            let idx = leaf.escape_idx.unwrap_or(0);
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
