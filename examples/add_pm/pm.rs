//! `add_pm`: Codec9 / DR512 `P × M` lanes added in place, 64 at a time.
//!
//! A lane is a magnitude `M` (one byte of a [`U8x64`]) and a presence bit `P`
//! (one bit of a `u64`). The DR512 law (`.claude/knowledge/dr512-epistemic-pm-invariant.md`):
//! `P=0, M=0` is never touched; `P=0, M>0` is `-M`; `P=1` is `+M`.
//!
//! The sum is sign-magnitude addition, with no decode to a signed integer:
//!
//! - an untouched lane is the identity (`untouched ⊕ x = x`), and the result
//!   is touched if either input was. This needs no blend: the never-touched
//!   code is a zero magnitude, so the identity is the addition itself;
//! - same polarity: `M = M_a + M_b`, `P = P_a`; the lanes where that sum
//!   passes 255 are returned in the carry mask, with `M` saturated at 255;
//! - opposite polarity: `M = max - min`, `P` from the larger magnitude;
//! - a touched zero result is `+0` (`P=1`), never the never-touched code.
//!
//! The carry mask is the exact-escape signal Codec9's own encoder lacks (it
//! saturates silently): a caller applies the exact sum of those lanes
//! through a wider path. Off the carry lanes the result is exact.
//!
//! Subtraction is the same addition of the polarity-negated operand
//! ([`neg_pm`], [`sub_pm`]). XOR governs polarity; magnitudes stay unsigned.
use ndarray::simd::U8x64;

/// 64 `P × M` lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pm {
    pub mag: U8x64,
    pub pos: u64,
}

impl Pm {
    /// Every lane never touched.
    pub fn untouched() -> Self {
        Pm {
            mag: U8x64::splat(0),
            pos: 0,
        }
    }
}

/// Lanes that hold a value (touched): `P=1`, or `M != 0`.
#[inline(always)]
pub fn touched(x: Pm) -> u64 {
    x.pos | !x.mag.cmpeq_mask(U8x64::splat(0))
}

/// `a ⊕ b` lane-wise, and the lanes whose exact sum is outside ±255.
#[inline(always)]
pub fn add_pm(a: Pm, b: Pm) -> (Pm, u64) {
    let zero = U8x64::splat(0);
    let (ta, tb) = (touched(a), touched(b));
    let same = !(a.pos ^ b.pos);
    // same polarity: saturating sum; carry where M_a > 255 - M_b
    let sum = a.mag.saturating_add(b.mag);
    let room = U8x64::splat(255).saturating_sub(b.mag);
    let carry = same & ta & tb & a.mag.cmpgt_mask(room);
    // opposite polarity: |M_a - M_b|, polarity of the larger magnitude
    let hi = a.mag.simd_max(b.mag);
    let lo = a.mag.simd_min(b.mag);
    let diff = hi.saturating_sub(lo);
    let a_larger = a.mag.cmpgt_mask(b.mag);
    let p_opp = (a_larger & a.pos) | (!a_larger & b.pos);
    let mag = U8x64::mask_blend(same, diff, sum);
    let pos = (same & a.pos) | (!same & p_opp);
    // A zero result is +0 when either input was touched, and stays the
    // never-touched code when neither was. No identity blend is needed: read
    // as sign-magnitude, the never-touched code is a zero magnitude, so
    // `untouched ⊕ x = x` falls out of the addition itself; only this zero
    // normalization has to know about presence.
    let pos = pos | (mag.cmpeq_mask(zero) & (ta | tb));
    (Pm { mag, pos }, carry)
}

/// `-b` lane-wise: the polarity bit flips on lanes with a magnitude.
///
/// `P_¬b = P_b ⊕ (M_b ≠ 0)`. A zero magnitude keeps its polarity, so the
/// never-touched code (`P=0, M=0`) stays never touched and `+0` stays `+0`:
/// `-0` is never produced, and remains only the never-touched sentinel.
#[inline(always)]
pub fn neg_pm(b: Pm) -> Pm {
    let nonzero = !b.mag.cmpeq_mask(U8x64::splat(0));
    Pm {
        mag: b.mag,
        pos: b.pos ^ nonzero,
    }
}

/// `a ⊖ b` lane-wise: `a ⊕ (-b)`, with `add_pm`'s carry contract. There is
/// no separate subtractor.
#[inline(always)]
pub fn sub_pm(a: Pm, b: Pm) -> (Pm, u64) {
    add_pm(a, neg_pm(b))
}

// ------------------------------------------------------------------ scalar reference
// Decode to i32 and back: the path `add_pm` avoids, and the oracle it is checked by.

/// One lane's value: `None` = never touched.
pub fn decode_lane(m: u8, p: bool) -> Option<i32> {
    match (p, m) {
        (false, 0) => None,
        (false, m) => Some(-i32::from(m)),
        (true, m) => Some(i32::from(m)),
    }
}

/// A touched value in ±255 to its lane; zero is `+0`.
pub fn encode_lane(v: i32) -> (u8, bool) {
    assert!((-255..=255).contains(&v), "out of range: {v}");
    (v.unsigned_abs() as u8, v >= 0)
}

/// The reference sum of one lane: value (or never touched) and overflow.
pub fn add_ref(a: Option<i32>, b: Option<i32>) -> (Option<i32>, bool) {
    match (a, b) {
        (None, x) | (x, None) => (x, false),
        (Some(x), Some(y)) => (Some(x + y), (x + y).abs() > 255),
    }
}

/// The reference difference of one lane: `a - b`, never touched only when
/// both are, and overflow outside ±255.
pub fn sub_ref(a: Option<i32>, b: Option<i32>) -> (Option<i32>, bool) {
    add_ref(a, b.map(|y| -y))
}

/// `add_pm` by decode → i32 → encode, 64 lanes. Carry lanes are saturated.
pub fn add_decoded(a: Pm, b: Pm) -> (Pm, u64) {
    let (am, bm) = (a.mag.to_array(), b.mag.to_array());
    let mut mag = [0u8; 64];
    let (mut pos, mut carry) = (0u64, 0u64);
    for i in 0..64 {
        let x = decode_lane(am[i], a.pos >> i & 1 == 1);
        let y = decode_lane(bm[i], b.pos >> i & 1 == 1);
        let (s, over) = add_ref(x, y);
        if let Some(v) = s {
            let (m, p) = encode_lane(v.clamp(-255, 255));
            mag[i] = m;
            pos |= u64::from(p) << i;
        }
        carry |= u64::from(over) << i;
    }
    (
        Pm {
            mag: U8x64::from_array(mag),
            pos,
        },
        carry,
    )
}
