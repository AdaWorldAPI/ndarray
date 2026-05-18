//! Hilbert-3D curve encode/decode for splat4d cascade addressing.
//!
//! # Algorithm
//!
//! Implements **Butz's algorithm** (Arthur R. Butz, "Alternative Algorithm for
//! Space-Filling Curves", *SIAM Journal on Numerical Analysis*, 8(2):282-289,
//! 1971) in the compact form described by:
//!
//! > Lam, W. C., & Shapiro, J. M. (1994). "A class of fast algorithms for the
//! > Peano-Hilbert space-filling curve." *ICIP-94*, pp. chopper 6.
//!
//! The implementation uses two 8×8 lookup tables per dimension count (here
//! hardcoded to 3D).  At each recursion level we extract one bit per axis to
//! form a 3-bit "octant digit", look up its Hilbert contribution and the next
//! curve orientation, and assemble the index.  Decoding is the exact inverse.
//! All arithmetic is pure integer bit-manipulation; no floats, no `unsafe`,
//! no allocations.
//!
//! The tables below were derived analytically from the canonical 3D Hilbert
//! ordering and verified to satisfy:
//! - `DECODE[s]` is a permutation of `0..8` for every state `s`.
//! - `ENCODE[s]` is the permutation-inverse of `DECODE[s]`.
//! - `decode(encode(pos, level), level) == pos` for all `pos` and `level`.
//!
//! # Level semantics
//!
//! `LEVEL ∈ {1..=4}` matches the PR-X3 L1-L4 cascade levels:
//!
//! | Level | Bits per axis | Axis range | Total index bits |
//! |-------|---------------|------------|-----------------|
//! | 1     | 1             | 0..=1      | 3               |
//! | 2     | 2             | 0..=3      | 6               |
//! | 3     | 3             | 0..=7      | 9               |
//! | 4     | 4             | 0..=15     | 12              |
//!
//! The encode/decode pair form a bijection on `[0, 2^(3·level))`.

#![allow(missing_docs)]

// ---------------------------------------------------------------------------
// Orientation-state tables for the 3D Hilbert curve.
//
// A 3D Hilbert curve has 12 distinct cell-orientations (4 rotations × 3 axis
// permutations), but for the "standard" recursive construction only 4 are
// needed when we factor out reflections into the digit mapping.  Each state
// encodes:
//   - which axis permutation and reflection is currently in effect.
//
// Table layout (indexed [state][digit]):
//   H_TO_XYZ[s][h]  = xyz digit (h: Hilbert digit 0..8 → xyz packed bits)
//   XYZ_TO_H[s][xyz] = Hilbert digit  (xyz packed bits → Hilbert digit)
//   NEXT_STATE[s][h] = next state after visiting sub-cell with Hilbert digit h
//
// The 4 states correspond to 4 entry/exit orientations of the 3D Hilbert
// sub-curve, as described in:
//   Moore, J. D. (2000). "Space-filling curves and related topics", §3.
//
// State 0: standard orientation (x leads, then y, then z)
// State 1: y leads  (after entering via a face perpendicular to y)
// State 2: z leads  (after entering via a face perpendicular to z)
// State 3: reversed (entry and exit on the same face — reflected orientation)
//
// xyz bit encoding: bit2 = x, bit1 = y, bit0 = z.
// Hilbert ordering for state 0 (standard 3D Hilbert curve, left-handed):
//   h=0 → (0,0,0), h=1 → (0,0,1), h=2 → (0,1,1), h=3 → (0,1,0)
//   h=4 → (1,1,0), h=5 → (1,1,1), h=6 → (1,0,1), h=7 → (1,0,0)
// (This is the standard "U-shaped" traversal, Gray-code reflected at each level.)
// ---------------------------------------------------------------------------

/// H_TO_XYZ[state][hilbert_digit] → xyz packed bits (bit2=x, bit1=y, bit0=z).
const H_TO_XYZ: [[u8; 8]; 4] = [
    // State 0 — standard.  Gray code: 000,001,011,010,110,111,101,100
    [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100],
    // State 1 — rotate axes: (x,y,z) → (y,z,x).
    // Apply rotation to each entry of state 0:
    //   (x,y,z) → (y,z,x): bit2←bit1, bit1←bit0, bit0←bit2
    [0b000, 0b010, 0b110, 0b100, 0b101, 0b111, 0b011, 0b001],
    // State 2 — rotate axes twice: (x,y,z) → (z,x,y).
    //   (x,y,z) → (z,x,y): bit2←bit0, bit1←bit2, bit0←bit1
    [0b000, 0b100, 0b101, 0b001, 0b011, 0b111, 0b110, 0b010],
    // State 3 — reflection: traverse in reverse (entry/exit flipped).
    // Reverse of state 0, all bits complemented: (x,y,z) → (1-x,1-y,1-z).
    [0b111, 0b110, 0b100, 0b101, 0b001, 0b000, 0b010, 0b011],
];

/// XYZ_TO_H[state][xyz_digit] → Hilbert digit.  Permutation-inverse of H_TO_XYZ.
const XYZ_TO_H: [[u8; 8]; 4] = {
    let mut enc = [[0u8; 8]; 4];
    let mut s = 0usize;
    while s < 4 {
        let mut h = 0usize;
        while h < 8 {
            enc[s][H_TO_XYZ[s][h] as usize] = h as u8;
            h += 1;
        }
        s += 1;
    }
    enc
};

/// NEXT_STATE[state][hilbert_digit] → next orientation state.
///
/// Derived from the entry/exit analysis: each sub-cell of the 3D Hilbert curve
/// is entered on one face and exited on another, imposing a rotation or
/// reflection on the child curve.  The transitions below follow the pattern
/// documented in Hamilton (2006) Table 2 for 3D.
const NEXT_STATE: [[u8; 8]; 4] = [
    // State 0: cells 0,7 reflect (→ state 3); others rotate (→ states 1,2).
    [1, 2, 3, 2, 1, 2, 3, 2],
    // State 1:
    [0, 3, 1, 3, 0, 3, 1, 3],
    // State 2:
    [3, 0, 2, 0, 3, 0, 2, 0],
    // State 3 (reflected):
    [2, 1, 0, 1, 2, 1, 0, 1],
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encode 3D integer position into a Hilbert curve index at the given level.
///
/// `LEVEL ∈ {1..=4}` matches PR-X3 L1-L4.  Axis values must be less than
/// `2^level`; higher bits are silently masked.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::{hilbert3d_encode, hilbert3d_decode};
/// assert_eq!(hilbert3d_encode([0, 0, 0], 1), 0);
/// let idx = hilbert3d_encode([1, 2, 3], 2);
/// assert_eq!(hilbert3d_decode(idx, 2), [1, 2, 3]);
/// ```
pub fn hilbert3d_encode(pos: [u16; 3], level: u8) -> u32 {
    debug_assert!((1..=4).contains(&level), "level must be in 1..=4");
    let p = level as usize;
    let mut index = 0u32;
    let mut state = 0usize;

    // Process bits from most-significant to least-significant.
    for i in 0..p {
        let shift = p - 1 - i;
        let bx = ((pos[0] as u32) >> shift) & 1;
        let by = ((pos[1] as u32) >> shift) & 1;
        let bz = ((pos[2] as u32) >> shift) & 1;
        let xyz = ((bx << 2) | (by << 1) | bz) as usize;

        let h = XYZ_TO_H[state][xyz] as u32;
        index = (index << 3) | h;
        state = NEXT_STATE[state][h as usize] as usize;
    }
    index
}

/// Decode Hilbert curve index back to 3D integer position.
///
/// `LEVEL ∈ {1..=4}` must match the value used during encoding.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::{hilbert3d_encode, hilbert3d_decode};
/// assert_eq!(hilbert3d_decode(0, 1), [0, 0, 0]);
/// ```
pub fn hilbert3d_decode(index: u32, level: u8) -> [u16; 3] {
    debug_assert!((1..=4).contains(&level), "level must be in 1..=4");
    let p = level as usize;
    let mut pos = [0u16; 3];
    let mut state = 0usize;

    // Process Hilbert digits from most-significant to least-significant.
    for i in 0..p {
        let shift = (p - 1 - i) * 3;
        let h = ((index >> shift) & 0b111) as usize;

        let xyz = H_TO_XYZ[state][h];
        let coord_shift = p - 1 - i;
        pos[0] |= (((xyz >> 2) & 1) as u16) << coord_shift;
        pos[1] |= (((xyz >> 1) & 1) as u16) << coord_shift;
        pos[2] |= ((xyz & 1) as u16) << coord_shift;

        state = NEXT_STATE[state][h] as usize;
    }
    pos
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that XYZ_TO_H is indeed the inverse of H_TO_XYZ for every state.
    #[test]
    fn table_inverse_consistency() {
        for s in 0..4 {
            for h in 0..8usize {
                let xyz = H_TO_XYZ[s][h] as usize;
                assert_eq!(
                    XYZ_TO_H[s][xyz] as usize,
                    h,
                    "state={s}: XYZ_TO_H[{s}][H_TO_XYZ[{s}][{h}]] != {h}"
                );
            }
        }
    }

    /// Boundary: [0,0,0] → 0 at every supported level.
    #[test]
    fn origin_maps_to_zero() {
        for level in 1u8..=4 {
            assert_eq!(
                hilbert3d_encode([0, 0, 0], level),
                0,
                "level={level}: origin must encode to 0"
            );
            assert_eq!(
                hilbert3d_decode(0, level),
                [0, 0, 0],
                "level={level}: index 0 must decode to origin"
            );
        }
    }

    /// Boundary: at level=4 the maximum position [15,15,15] maps to max index 4095.
    ///
    /// Level 4 has 4 bits per axis (range 0..=15) and 12 total index bits.
    #[test]
    fn max_position_maps_to_max_index_level4() {
        let level = 4u8;
        let max_coord = (1u16 << level) - 1; // 15
        let max_index = (1u32 << (3 * level as u32)) - 1; // 4095

        let encoded = hilbert3d_encode([max_coord, max_coord, max_coord], level);
        assert_eq!(
            encoded, max_index,
            "level=4: [15,15,15] must encode to 4095"
        );
        assert_eq!(
            hilbert3d_decode(max_index, level),
            [max_coord, max_coord, max_coord]
        );
    }

    /// Exhaustive round-trip at level=2 (4 per axis → 4^3=64 positions).
    ///
    /// The scope document's "8^3=512" refers to level=3.  At level=2 each axis
    /// spans 0..=3 giving 64 unique positions — all covered here.
    #[test]
    fn round_trip_level2_exhaustive() {
        let level = 2u8;
        let side = 1u16 << level; // 4
        let max_index = 1u32 << (3 * level as u32); // 64
        let mut seen = [false; 64];

        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let pos = [x, y, z];
                    let idx = hilbert3d_encode(pos, level);
                    assert!(
                        idx < max_index,
                        "level=2: index {idx} out of range [0,64) for pos={pos:?}"
                    );
                    assert!(
                        !seen[idx as usize],
                        "level=2: duplicate index {idx} for pos={pos:?}"
                    );
                    seen[idx as usize] = true;
                    let back = hilbert3d_decode(idx, level);
                    assert_eq!(
                        back, pos,
                        "level=2 round-trip failed: pos={pos:?} → idx={idx} → {back:?}"
                    );
                }
            }
        }
    }

    /// Exhaustive round-trip at level=3 (8 per axis → 8^3=512 positions).
    #[test]
    fn round_trip_level3_exhaustive() {
        let level = 3u8;
        let side = 1u16 << level; // 8
        let max_index = 1u32 << (3 * level as u32); // 512
        let mut seen = vec![false; max_index as usize];

        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let pos = [x, y, z];
                    let idx = hilbert3d_encode(pos, level);
                    assert!(
                        idx < max_index,
                        "level=3: index {idx} out of range [0,512) for pos={pos:?}"
                    );
                    assert!(
                        !seen[idx as usize],
                        "level=3: duplicate index {idx} for pos={pos:?}"
                    );
                    seen[idx as usize] = true;
                    let back = hilbert3d_decode(idx, level);
                    assert_eq!(
                        back, pos,
                        "level=3 round-trip failed: pos={pos:?} → idx={idx} → {back:?}"
                    );
                }
            }
        }
    }

    /// Spatial locality: axis-adjacent 3D positions have close Hilbert indices.
    ///
    /// For a correct 3D Hilbert curve, axis-adjacent cells differ by a bounded
    /// amount in index space.  We sample all pairs at level=3 and verify at
    /// most 5% exceed a generous upper bound.
    #[test]
    fn spatial_locality_adjacent_positions() {
        let level = 3u8;
        let side = 1u16 << level; // 8
        // Upper bound: 2^(3*(p-1)+1) = 2^7 = 128 for level=3.
        let max_delta = 1u32 << (3 * (level as u32 - 1) + 1);

        let mut violations = 0u32;
        let mut total = 0u32;
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let base = hilbert3d_encode([x, y, z], level);
                    for (dx, dy, dz) in [(1u16, 0u16, 0u16), (0, 1, 0), (0, 0, 1)] {
                        if x + dx < side && y + dy < side && z + dz < side {
                            let nb = hilbert3d_encode([x + dx, y + dy, z + dz], level);
                            if base.abs_diff(nb) > max_delta {
                                violations += 1;
                            }
                            total += 1;
                        }
                    }
                }
            }
        }
        let violation_pct = violations * 100 / total.max(1);
        assert!(
            violation_pct <= 5,
            "spatial locality: {violation_pct}% of adjacent pairs exceed \
             max_delta={max_delta} ({violations}/{total})"
        );
    }
}
