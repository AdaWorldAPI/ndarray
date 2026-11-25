//! D-PM-STATS probes over `P × M` blocks (`abakus::pm`): statistical folds,
//! signed comparison and selection, and sign-partitioned products.
//!
//! Research only. Everything is composed from existing `ndarray::simd` words;
//! no new kernel, API or layout. Correctness lives in `tests/pm_stats.rs`.
//!
//! The observation the folds rest on: an untouched lane has `M = 0`, so it
//! adds nothing to a sum of magnitudes, of signed values, of squares or of
//! products. Only the count `N` needs `touched`; nothing else needs a validity
//! filter.

use ndarray::hpc::abakus::pm::{touched, Pm};
use ndarray::simd::{U16x32, U8x64};

/// Sum of the magnitudes in `mask`'s lanes (`vpsadbw` on AVX-512).
#[inline(always)]
pub fn msum(mask: u64, m: U8x64) -> u64 {
    U8x64::mask_blend(mask, U8x64::splat(0), m).sum_bytes_u64()
}

/// `Σ (a_i · b_i)` over 64 lanes, exact: products are at most 255² and fit a
/// `u16` lane; each product is split into its two bytes so the byte sums stay
/// within `sum_bytes_u64`'s range.
#[inline(always)]
pub fn prod_sum(a: U8x64, b: U8x64) -> u64 {
    let lo = U16x32::from_u8x64_lo(a).mullo(U16x32::from_u8x64_lo(b));
    let hi = U16x32::from_u8x64_hi(a).mullo(U16x32::from_u8x64_hi(b));
    let (lo_h, hi_h) = (lo.shr(8), hi.shr(8));
    let (lo_l, hi_l) = (lo - lo_h.shl(8), hi - hi_h.shl(8));
    let low = lo_l.pack_saturate_u8(hi_l).sum_bytes_u64();
    let high = lo_h.pack_saturate_u8(hi_h).sum_bytes_u64();
    low + (high << 8)
}

// ---------- Probe 1: sufficient statistics ----------

/// Which statistics a caller demands.
#[derive(Clone, Copy, Debug)]
pub struct Demand {
    pub n: bool,
    pub s: bool,
    pub q: bool,
    pub a: bool,
}

impl Demand {
    pub const ALL: Demand = Demand { n: true, s: true, q: true, a: true };
    pub const SUM: Demand = Demand { n: false, s: true, q: false, a: false };
}

/// `N` selected touched values, `S = Σx`, `Q = Σx²`, `A = Σ|x|`. Fields not
/// demanded stay 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub n: u64,
    pub s: i64,
    pub q: u64,
    pub a: u64,
}

/// Arm A: positive and negative magnitudes reduced separately.
pub fn sum_split(xs: &[Pm], sel: &[u64]) -> i64 {
    let mut s = 0i64;
    for (x, &m) in xs.iter().zip(sel) {
        s += msum(m & x.pos, x.mag) as i64 - msum(m & !x.pos, x.mag) as i64;
    }
    s
}

/// Arm B: `S = Σ M − 2 Σ M_negative`.
pub fn sum_total_minus_twice_neg(xs: &[Pm], sel: &[u64]) -> i64 {
    let mut s = 0i64;
    for (x, &m) in xs.iter().zip(sel) {
        s += msum(m, x.mag) as i64 - 2 * msum(m & !x.pos, x.mag) as i64;
    }
    s
}

/// Arm C: one fold computing only the demanded statistics. `A` is shared by
/// `S` (arm B's form) when both are demanded.
pub fn fold(xs: &[Pm], sel: &[u64], d: Demand) -> Stats {
    let mut st = Stats::default();
    for (x, &m) in xs.iter().zip(sel) {
        if d.n {
            st.n += u64::from((m & touched(*x)).count_ones());
        }
        if d.s || d.a {
            let a = msum(m, x.mag);
            if d.a {
                st.a += a;
            }
            if d.s {
                st.s += a as i64 - 2 * msum(m & !x.pos, x.mag) as i64;
            }
        }
        if d.q {
            let mb = U8x64::mask_blend(m, U8x64::splat(0), x.mag);
            st.q += prod_sum(mb, mb);
        }
    }
    st
}

/// Population variance as an exact rational `(N·Q − S², N²)`; `None` for an
/// empty population. Exact in `i128` for any population this probe folds.
pub fn variance_parts(st: Stats) -> Option<(i128, i128)> {
    if st.n == 0 {
        return None;
    }
    let n = i128::from(st.n);
    Some((n * i128::from(st.q) - i128::from(st.s) * i128::from(st.s), n * n))
}

// ---------- Probe 2: signed comparison and selection ----------

/// A three-valued comparison: lanes where it holds, and lanes it cannot
/// decide because an operand is untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cmp {
    pub is: u64,
    pub unknown: u64,
}

/// `a < b`: two unsigned magnitude compares, then mask logic on polarity.
#[inline(always)]
pub fn lt(a: Pm, b: Pm) -> Cmp {
    let known = touched(a) & touched(b);
    let gab = a.mag.cmpgt_mask(b.mag);
    let gba = b.mag.cmpgt_mask(a.mag);
    let r = (!a.pos & b.pos) | (a.pos & b.pos & gba) | (!a.pos & !b.pos & gab);
    Cmp { is: r & known, unknown: !known }
}

/// Numerical `a = b`. `+0` has one code, so on touched lanes this equals
/// codeword identity; two untouched lanes are unknown, not equal.
#[inline(always)]
pub fn eq(a: Pm, b: Pm) -> Cmp {
    let known = touched(a) & touched(b);
    let r = !(a.pos ^ b.pos) & a.mag.cmpeq_mask(b.mag);
    Cmp { is: r & known, unknown: !known }
}

#[inline(always)]
pub fn le(a: Pm, b: Pm) -> Cmp {
    let (l, e) = (lt(a, b), eq(a, b));
    Cmp { is: l.is | e.is, unknown: l.unknown }
}

/// `x < c` for a constant: one magnitude compare plus `touched`, or for
/// `c = 0` a single `cmpeq` (`x < 0` is exactly `!P ∧ M ≠ 0`).
#[inline(always)]
pub fn lt_const(x: Pm, c: i16) -> Cmp {
    let t = touched(x);
    let r = if c > 0 {
        !x.pos | (x.pos & U8x64::splat(c as u8).cmpgt_mask(x.mag))
    } else if c == 0 {
        !x.pos
    } else {
        !x.pos & x.mag.cmpgt_mask(U8x64::splat((-c) as u8))
    };
    Cmp { is: r & t, unknown: !t }
}

/// The smaller of `a` and `b`, untouched where either is untouched.
#[inline(always)]
pub fn min(a: Pm, b: Pm) -> Pm {
    pick(lt(a, b), a, b)
}

/// The larger of `a` and `b`, untouched where either is untouched.
#[inline(always)]
pub fn max(a: Pm, b: Pm) -> Pm {
    pick(lt(b, a), a, b)
}

/// `a` where `c.is`, else `b`; untouched where `c.unknown`.
#[inline(always)]
fn pick(c: Cmp, a: Pm, b: Pm) -> Pm {
    let mag = U8x64::mask_blend(c.is, b.mag, a.mag);
    let mag = U8x64::mask_blend(c.unknown, mag, U8x64::splat(0));
    let pos = ((c.is & a.pos) | (!c.is & b.pos)) & !c.unknown;
    Pm { mag, pos }
}

// ---------- Probe 3: sign-partitioned products ----------

/// `Σ a·b` over the selected lanes, split by product polarity:
/// `Σ_same Ma·Mb − Σ_opposite Ma·Mb`.
pub fn dot_split(a: &[Pm], b: &[Pm], sel: &[u64]) -> i64 {
    let mut d = 0i64;
    for ((x, y), &m) in a.iter().zip(b).zip(sel) {
        let same = m & !(x.pos ^ y.pos);
        let opp = m & (x.pos ^ y.pos);
        let z = U8x64::splat(0);
        d += prod_sum(U8x64::mask_blend(same, z, x.mag), y.mag) as i64
            - prod_sum(U8x64::mask_blend(opp, z, x.mag), y.mag) as i64;
    }
    d
}

/// The same as `Σ_all − 2 Σ_opposite`, mirroring probe 1's arm B.
pub fn dot_total_minus_twice_opp(a: &[Pm], b: &[Pm], sel: &[u64]) -> i64 {
    let mut d = 0i64;
    for ((x, y), &m) in a.iter().zip(b).zip(sel) {
        let opp = m & (x.pos ^ y.pos);
        let z = U8x64::splat(0);
        d += prod_sum(U8x64::mask_blend(m, z, x.mag), y.mag) as i64
            - 2 * prod_sum(U8x64::mask_blend(opp, z, x.mag), y.mag) as i64;
    }
    d
}

// ---------- shared helpers for the oracle and baselines ----------

/// Codeword `c` in `0..512`: magnitude `c & 255`, polarity bit 8.
pub fn value_of(c: u16) -> Option<i32> {
    let (m, p) = (i32::from(c & 0xff), c >> 8 & 1 == 1);
    match (p, m) {
        (false, 0) => None,
        (true, m) => Some(m),
        (false, m) => Some(-m),
    }
}

pub fn block_of(codes: &[u16]) -> Pm {
    let mut mag = [0u8; 64];
    let mut pos = 0u64;
    for (i, &c) in codes.iter().enumerate() {
        mag[i] = (c & 0xff) as u8;
        pos |= u64::from(c >> 8 & 1) << i;
    }
    Pm { mag: U8x64::from_array(mag), pos }
}

pub fn lane(x: Pm, i: usize) -> Option<i32> {
    let m = i32::from(x.mag.to_array()[i]);
    match (x.pos >> i & 1 == 1, m) {
        (false, 0) => None,
        (true, m) => Some(m),
        (false, m) => Some(-m),
    }
}
