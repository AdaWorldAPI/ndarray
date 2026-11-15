//! Signed `P × M` lanes added and subtracted in place, 64 at a time.
//!
//! A lane is a magnitude `M` (one byte of a [`U8x64`]) and a polarity bit `P`
//! (one bit of a `u64`), under the DR512 law: `P=0, M=0` is never touched,
//! `P=0, M>0` is `-M`, `P=1` is `+M` (so `P=1, M=0` is the touched `+0`).
//!
//! - [`add_pm`]: sign-magnitude addition. Same polarity adds the magnitudes;
//!   opposite polarity takes the difference and the larger one's polarity. A
//!   zero result is `+0` when either input was touched, and stays never
//!   touched when neither was.
//! - [`neg_pm`]: flips the polarity on lanes with a magnitude,
//!   `P_¬b = P_b ⊕ (M_b ≠ 0)`, so the never-touched code and `+0` are fixed
//!   points and `-0` is never produced.
//! - [`sub_pm`]: `add_pm(a, neg_pm(b))`. There is no separate subtractor.
//!
//! **The carry contract.** A lane whose exact result is outside ±255 is a
//! carry lane. Its byte is saturated and is not the result. [`PmSum`] does
//! not hand its lanes out as a plain [`Pm`]: [`PmSum::lane`] reports a carry
//! lane as [`PmLane::Escaped`], and the raw bytes come only from
//! [`PmSum::into_saturated_parts`], whose name says what they are. The caller
//! applies the exact value of each carry lane through a wider path.
//!
//! Proven in `tests/add_pm.rs` (every 512 × 512 codeword pair against an
//! `i32` oracle, for both operations) on the AVX-512, AVX2, AVX, x86-64-v2,
//! NEON and wasm32 realizations. Experimental, see [`super`].
use crate::simd::U8x64;

/// 64 `P × M` lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pm {
    /// Magnitude per lane.
    pub mag: U8x64,
    /// Polarity per lane; with a zero magnitude, the presence bit.
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

/// Lanes that hold a value: `P=1`, or `M != 0`.
#[inline(always)]
pub fn touched(x: Pm) -> u64 {
    x.pos | !x.mag.cmpeq_mask(U8x64::splat(0))
}

/// One lane of a [`PmSum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmLane {
    /// Neither input touched this lane.
    NeverTouched,
    /// The exact result, in `-255..=255`.
    Exact(i16),
    /// The exact result is outside ±255; the lane holds no value.
    Escaped,
}

/// The result of [`add_pm`] or [`sub_pm`]: 64 lanes and the carry mask.
#[must_use = "carry lanes hold no value; read them through `lane` or handle `carry`"]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PmSum {
    lanes: Pm,
    carry: u64,
}

impl PmSum {
    /// Lanes whose exact result is outside ±255.
    pub fn carry(&self) -> u64 {
        self.carry
    }

    /// Lane `i` (`i < 64`).
    pub fn lane(&self, i: usize) -> PmLane {
        assert!(i < 64, "lane {i} out of range");
        if self.carry >> i & 1 == 1 {
            return PmLane::Escaped;
        }
        let m = i16::from(self.lanes.mag.to_array()[i]);
        match (self.lanes.pos >> i & 1 == 1, m) {
            (false, 0) => PmLane::NeverTouched,
            (false, m) => PmLane::Exact(-m),
            (true, m) => PmLane::Exact(m),
        }
    }

    /// The lanes as a [`Pm`], when no lane carried.
    pub fn exact(&self) -> Option<Pm> {
        (self.carry == 0).then_some(self.lanes)
    }

    /// The raw lanes and the carry mask. On a carry lane the byte is
    /// saturated at 255 with the inputs' polarity: it is not the result, and
    /// the caller must replace it with the exact value.
    pub fn into_saturated_parts(self) -> (Pm, u64) {
        (self.lanes, self.carry)
    }
}

/// `a ⊕ b` lane-wise. See the module doc for the carry contract.
#[inline(always)]
pub fn add_pm(a: Pm, b: Pm) -> PmSum {
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
    // A zero result is +0 when either input was touched. The never-touched
    // code is a zero magnitude, so `untouched ⊕ x = x` needs no blend; only
    // this normalization reads presence.
    let pos = pos | (mag.cmpeq_mask(zero) & (ta | tb));
    PmSum {
        lanes: Pm { mag, pos },
        carry,
    }
}

/// `-b` lane-wise: polarity flipped where the magnitude is non-zero.
#[inline(always)]
pub fn neg_pm(b: Pm) -> Pm {
    let nonzero = !b.mag.cmpeq_mask(U8x64::splat(0));
    Pm {
        mag: b.mag,
        pos: b.pos ^ nonzero,
    }
}

/// `a ⊖ b` lane-wise: `add_pm(a, neg_pm(b))`.
#[inline(always)]
pub fn sub_pm(a: Pm, b: Pm) -> PmSum {
    add_pm(a, neg_pm(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(vals: &[Option<i16>]) -> Pm {
        let mut mag = [0u8; 64];
        let mut pos = 0u64;
        for (i, v) in vals.iter().enumerate() {
            if let Some(v) = *v {
                mag[i] = v.unsigned_abs() as u8;
                pos |= u64::from(v >= 0) << i;
            }
        }
        Pm {
            mag: U8x64::from_array(mag),
            pos,
        }
    }

    #[test]
    fn lane_reports_never_touched_exact_and_escaped() {
        let a = block(&[None, Some(-5), Some(200), Some(0)]);
        let b = block(&[None, Some(5), Some(100), Some(0)]);
        let s = add_pm(a, b);
        assert_eq!(s.lane(0), PmLane::NeverTouched);
        assert_eq!(s.lane(1), PmLane::Exact(0), "-5 + +5 = +0");
        assert_eq!(s.lane(2), PmLane::Escaped, "300 has no lane value");
        assert_eq!(s.lane(3), PmLane::Exact(0), "+0 + +0 = +0");
        assert_eq!(s.carry(), 1 << 2);
        assert_eq!(s.exact(), None, "a carried sum is not exact");
        let (raw, carry) = s.into_saturated_parts();
        assert_eq!((raw.mag.to_array()[2], carry), (255, 1 << 2));
    }

    #[test]
    fn exact_is_some_without_carry() {
        let a = block(&[Some(10), Some(-20)]);
        let s = sub_pm(a, a);
        let e = s.exact().expect("no carry");
        assert_eq!(touched(e), 0b11, "x - x is a touched +0");
        assert_eq!(e.pos & 0b11, 0b11);
    }
}
