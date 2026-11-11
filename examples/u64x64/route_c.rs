//! D-U64X64-LLVM-1 route C ("SIMD SoA").
//!
//! A stock/cost plane (one category, 64 rows) is read in place as two
//! `U16x32` halves (rows 0..32 and 32..64), or one `U8x64` for u8 planes.
//! Row masks are 64-bit words; the halves are joined by `join`. No shuffles,
//! no i8, no allocation inside a kernel. Scalar loops appear only where the
//! facade has no op (listed in the report that accompanies this file):
//! narrowing u16 -> u8, widening u16 -> u32 for O4's final recombination,
//! the `RowStatus` enum writes, and per-group `U64x8` array reads.
//!
//! Disable-run anchors are marked `// C<n>:` (plan §6).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::harness::Class;
use crate::{Entry, Kernel};
use ndarray::simd::{U16x32, U64x8, U8x64};

/// A 64-row plane as two `U16x32` halves.
pub(crate) type P16 = [U16x32; 2];

/// Join the two 32-row halves of a row mask.
#[inline(always)]
pub(crate) fn join(lo: u32, hi: u32) -> u64 {
    // C1: the high half must be shifted into rows 32..64.
    let m = lo as u64 | (hi as u64) << 32;
    m
}

/// Carry out of the wrapping add `low + p = sum`, as 0 or 1 per lane.
#[inline(always)]
fn carry_after(low: U16x32, p: U16x32, sum: U16x32) -> U16x32 {
    ((low & p) | ((low | p) & !sum)).shr(15)
}

#[inline(always)]
pub(crate) fn load16(plane: &[u16; 64]) -> P16 {
    [U16x32::from_slice(&plane[..32]), U16x32::from_slice(&plane[32..])]
}

#[inline(always)]
pub(crate) fn widen_u8(v: U8x64) -> P16 {
    [U16x32::from_u8x64_lo(v), U16x32::from_u8x64_hi(v)]
}

/// Row mask -> 0xFFFF / 0x0000 per u16 lane.
#[inline(always)]
pub(crate) fn lane16(m: u64) -> P16 {
    let b = U8x64::mask_blend(m, U8x64::splat(0), U8x64::splat(0xFF));
    let lo = U16x32::from_u8x64_lo(b);
    let hi = U16x32::from_u8x64_hi(b);
    [lo | lo.shl(8), hi | hi.shl(8)]
}

/// Row mask -> 1 / 0 per u16 lane.
#[inline(always)]
fn ones16(m: u64) -> P16 {
    widen_u8(U8x64::mask_blend(m, U8x64::splat(0), U8x64::splat(1)))
}

// ---- O1 ----

/// O1: `row_cnt` is a vertical byte count of the masked words; `plane_cnt`
/// comes from `U64x8::popcnt`.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o1_c(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    let lv = U64x8::splat(live);
    let zero = U8x64::splat(0);
    let one = U8x64::splat(1);
    let mut acc = U8x64::splat(0);
    let mut total = 0u64;
    for g in 0..8 {
        let x = U64x8::from_slice(&a[g * 8..g * 8 + 8]) & U64x8::from_slice(&b[g * 8..g * 8 + 8]) & lv;
        let pc = x.popcnt();
        total += pc.reduce_sum();
        let pca = pc.to_array();
        let xa = x.to_array();
        for i in 0..8 {
            out.plane_cnt[g * 8 + i] = pca[i] as u8;
            acc = acc + U8x64::mask_blend(xa[i], zero, one);
        }
    }
    acc.copy_to_slice(&mut out.row_cnt);
    out.total = total as u16;
}

// ---- O2 ----

/// O2: `ge = !(t > v)` per demanded plane, unsigned; `t == 65536` never.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_c(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    out.ge = [0; 64];
    let mut all = u64::MAX;
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let tk = t[k];
        let ge = if tk > 65535 {
            0
        } else {
            let tv = U16x32::splat(tk as u16);
            let v = load16(&bound.src.stock[k]);
            let lt = join(tv.cmpgt_mask(v[0]), tv.cmpgt_mask(v[1]));
            !lt
        };
        out.ge[k] = ge;
        all &= ge;
    }
    out.all_ge = all;
}

// ---- O3 ----

/// O3: absent slots become 0xFFFF before the min; `arg` updates only on a
/// strictly smaller value or the first present slot, so ties keep the
/// smaller k. `have` tracks presence because a present 0xFFFF is
/// indistinguishable from the substitute.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_c(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    let src = bound.src;
    let ff = U16x32::splat(0xFFFF);
    let mut m: P16 = [ff, ff];
    let mut arg = U8x64::splat(0xFF);
    let mut have = 0u64;
    let mut unknown = 0u64;
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let p = src.present[k];
        unknown |= !p;
        let pm = lane16(p);
        let v = load16(&src.stock[k]);
        // C3: absent slots are replaced by 0xFFFF (v | !pm).
        let vs: P16 = [v[0] | !pm[0], v[1] | !pm[1]];
        // C2: strictly smaller only (m > v'); ties keep the earlier k.
        let lt = join(m[0].cmpgt_mask(vs[0]), m[1].cmpgt_mask(vs[1]));
        let upd = lt | (p & !have);
        arg = U8x64::mask_blend(upd, arg, U8x64::splat(k as u8));
        have |= p;
        m = [m[0].simd_min(vs[0]), m[1].simd_min(vs[1])];
    }
    let hm = lane16(have);
    (m[0] & hm[0]).copy_to_slice(&mut out.min[..32]);
    (m[1] & hm[1]).copy_to_slice(&mut out.min[32..]);
    arg.copy_to_slice(&mut out.arg);
    out.none = !have;
    out.unknown = unknown;
}

// ---- O4 / Col16 ----

/// Place-value column pair: `low + 2^16 * high` per lane.
#[derive(Clone, Copy)]
struct Col16 {
    low: U16x32,
    high: U16x32,
}

impl Col16 {
    #[inline(always)]
    fn zero() -> Self {
        Col16 {
            low: U16x32::zero(),
            high: U16x32::zero(),
        }
    }

    /// Add products `p` (each below 2^16).
    #[inline(always)]
    fn add(&mut self, p: U16x32) {
        let sum = self.low + p;
        self.high += carry_after(self.low, p, sum); // C4: carry into `high`.
        self.low = sum;
    }
}

/// O4: `sum = lo_acc + (hi_acc << 8)` where `lo_acc` / `hi_acc` accumulate
/// `mullo(cost, w & 0xFF)` and `mullo(cost, w >> 8)` (each <= 255*255).
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_c(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    let mut lo = [Col16::zero(), Col16::zero()];
    let mut hi = [Col16::zero(), Col16::zero()];
    for k in 0..64 {
        let wk = w[k];
        if wk == 0 {
            continue;
        }
        let c = widen_u8(U8x64::from_slice(&cost[k]));
        let wl = U16x32::splat(wk & 0xFF);
        let wh = U16x32::splat(wk >> 8);
        for h in 0..2 {
            lo[h].add(c[h].mullo(wl));
            if wk >> 8 != 0 {
                hi[h].add(c[h].mullo(wh)); // C6: high-byte accumulator.
            }
        }
    }
    let mut total = 0u64;
    for h in 0..2 {
        let ll = lo[h].low.to_array();
        let lh = lo[h].high.to_array();
        let hl = hi[h].low.to_array();
        let hh = hi[h].high.to_array();
        for i in 0..32 {
            let r = h * 32 + i;
            let a = (ll[i] as u32).wrapping_add((lh[i] as u32) << 16);
            let b = (hl[i] as u32).wrapping_add((hh[i] as u32) << 16);
            let s = a.wrapping_add(b << 8);
            out.sum[r] = s;
            if (live >> r) & 1 == 1 {
                total += s as u64;
            }
        }
    }
    out.total = total;
}

// ---- O4a ----

/// O4a: `z = x + y`, `ovf` = carry out of each lane.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4a_c(x: &ValueTile<u16>, y: &ValueTile<u16>, out: &mut O4aOut) {
    let one = U16x32::splat(1);
    for k in 0..64 {
        let xs = load16(&x[k]);
        let ys = load16(&y[k]);
        let s0 = xs[0] + ys[0];
        let s1 = xs[1] + ys[1];
        s0.copy_to_slice(&mut out.z[k][..32]);
        s1.copy_to_slice(&mut out.z[k][32..]);
        out.ovf[k] = join(carry_after(xs[0], ys[0], s0).cmpeq_mask(one), carry_after(xs[1], ys[1], s1).cmpeq_mask(one));
    }
}

// ---- O5 ----

/// O5: one transactional Abakus update. Both halves are computed first; the
/// counters are written only when no lane carries out.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o5_c(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let b = u.b as u32;
    let cnt = load16(&counters[k]);
    let ones = ones16(u.m);
    let inc: P16 = [ones[0].shl(b), ones[1].shl(b)];
    let sum: P16 = [cnt[0] + inc[0], cnt[1] + inc[1]];
    let one = U16x32::splat(1);
    let lanes = join(
        carry_after(cnt[0], inc[0], sum[0]).cmpeq_mask(one),
        carry_after(cnt[1], inc[1], sum[1]).cmpeq_mask(one),
    );
    if lanes != 0 {
        return Err(Refusal::Overflow { lanes });
    }
    sum[0].copy_to_slice(&mut counters[k][..32]);
    sum[1].copy_to_slice(&mut counters[k][32..]);
    Ok(())
}

// ---- O6 ----

/// Restoring quotient `min(floor(s / d), 255)` over bits 7..0. `d <= 255`
/// and `cand <= 255`, so `cand * d` never wraps. A quotient of 256 or more
/// saturates at 255, which is never equal to a cap (<= 128).
#[inline(always)]
pub(crate) fn quot8(s: &P16, d: U16x32) -> P16 {
    let mut q: P16 = [U16x32::zero(), U16x32::zero()];
    for b in (0..8).rev() {
        let bit = U16x32::splat(1u16 << b);
        let p0 = (q[0] | bit).mullo(d);
        let p1 = (q[1] | bit).mullo(d);
        let over = join(p0.cmpgt_mask(s[0]), p1.cmpgt_mask(s[1]));
        // C5: keep the bit where cand * d <= s (no borrow).
        let keep = lane16(!over);
        q = [q[0] | (bit & keep[0]), q[1] | (bit & keep[1])];
    }
    q
}

/// O6: per demanded k the exact quotient, then a running min with `limit`
/// kept on the first k that reaches the final cap (strictly smaller, or the
/// first tie, updates).
#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_c(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let d = &params.recipe.d;
    let (list, n) = params.recipe.demanded_list();

    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let qmax = U16x32::splat(QMAX as u16);
    let mut cap: P16 = [staff_raw[0].simd_min(qmax), staff_raw[1].simd_min(qmax)];

    let mut missing = 0u64;
    let mut first_miss = U8x64::splat(0);
    let mut lim = U8x64::splat(0);
    let mut limset = 0u64;

    for i in 0..n {
        let k = list[i] as usize;
        let p = src.present[k];
        let newm = !p & !missing;
        first_miss = U8x64::mask_blend(newm, first_miss, U8x64::splat(k as u8));
        missing |= !p;

        let s = load16(&src.stock[k]);
        let q = quot8(&s, U16x32::splat(d[k]));
        let lt = join(cap[0].cmpgt_mask(q[0]), cap[1].cmpgt_mask(q[1]));
        let eq = join(cap[0].cmpeq_mask(q[0]), cap[1].cmpeq_mask(q[1]));
        // C2: update only on strictly smaller, or the first tie; later ties keep the earlier k.
        let upd = lt | (eq & !limset);
        lim = U8x64::mask_blend(upd, lim, U8x64::splat(k as u8));
        limset |= upd;
        cap = [cap[0].simd_min(q[0]), cap[1].simd_min(q[1])];
    }

    let known = src.gate & !missing;
    let eqs = join(cap[0].cmpeq_mask(staff_raw[0]), cap[1].cmpeq_mask(staff_raw[1]));
    let mut limv = U8x64::mask_blend(!limset, lim, U8x64::splat(CEILING));
    limv = U8x64::mask_blend(!limset & eqs, limv, U8x64::splat(STAFF));
    limv = U8x64::mask_blend(!known, limv, U8x64::splat(0));
    let km = lane16(known);
    let capv: P16 = [cap[0] & km[0], cap[1] & km[1]];

    let capa = [capv[0].to_array(), capv[1].to_array()];
    let la = limv.to_array();
    let fm = first_miss.to_array();
    for r in 0..64 {
        out.cap[r] = capa[r >> 5][r & 31] as u8;
        out.limit[r] = la[r];
        out.status[r] = if (src.gate >> r) & 1 == 0 {
            RowStatus::Denied
        } else if (missing >> r) & 1 == 1 {
            RowStatus::Unknown { k: fm[r] }
        } else {
            RowStatus::Known
        };
    }
    out.known = known;
    out.unknown = src.gate & missing;
    out.denied = !src.gate;
}

/// O6 with the F7 power-of-two selection per demanded row; otherwise identical to `u64x64_o6_c`.
/// Original doc: per demanded k the exact quotient, then a running min with `limit`
/// kept on the first k that reaches the final cap (strictly smaller, or the
/// first tie, updates).
#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_c_pow2(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let d = &params.recipe.d;
    let (list, n) = params.recipe.demanded_list();

    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let qmax = U16x32::splat(QMAX as u16);
    let mut cap: P16 = [staff_raw[0].simd_min(qmax), staff_raw[1].simd_min(qmax)];

    let mut missing = 0u64;
    let mut first_miss = U8x64::splat(0);
    let mut lim = U8x64::splat(0);
    let mut limset = 0u64;

    for i in 0..n {
        let k = list[i] as usize;
        let p = src.present[k];
        let newm = !p & !missing;
        first_miss = U8x64::mask_blend(newm, first_miss, U8x64::splat(k as u8));
        missing |= !p;

        let s = load16(&src.stock[k]);
        let dk = d[k];
        // F7: only a power of two takes the shift path.
        let q = if dk.is_power_of_two() {
            let p = dk.trailing_zeros();
            let m = U16x32::splat(255);
            [s[0].shr(p).simd_min(m), s[1].shr(p).simd_min(m)]
        } else {
            quot8(&s, U16x32::splat(dk))
        };
        let lt = join(cap[0].cmpgt_mask(q[0]), cap[1].cmpgt_mask(q[1]));
        let eq = join(cap[0].cmpeq_mask(q[0]), cap[1].cmpeq_mask(q[1]));
        // C2: update only on strictly smaller, or the first tie; later ties keep the earlier k.
        let upd = lt | (eq & !limset);
        lim = U8x64::mask_blend(upd, lim, U8x64::splat(k as u8));
        limset |= upd;
        cap = [cap[0].simd_min(q[0]), cap[1].simd_min(q[1])];
    }

    let known = src.gate & !missing;
    let eqs = join(cap[0].cmpeq_mask(staff_raw[0]), cap[1].cmpeq_mask(staff_raw[1]));
    let mut limv = U8x64::mask_blend(!limset, lim, U8x64::splat(CEILING));
    limv = U8x64::mask_blend(!limset & eqs, limv, U8x64::splat(STAFF));
    limv = U8x64::mask_blend(!known, limv, U8x64::splat(0));
    let km = lane16(known);
    let capv: P16 = [cap[0] & km[0], cap[1] & km[1]];

    let capa = [capv[0].to_array(), capv[1].to_array()];
    let la = limv.to_array();
    let fm = first_miss.to_array();
    for r in 0..64 {
        out.cap[r] = capa[r >> 5][r & 31] as u8;
        out.limit[r] = la[r];
        out.status[r] = if (src.gate >> r) & 1 == 0 {
            RowStatus::Denied
        } else if (missing >> r) & 1 == 1 {
            RowStatus::Unknown { k: fm[r] }
        } else {
            RowStatus::Known
        };
    }
    out.known = known;
    out.unknown = src.gate & missing;
    out.denied = !src.gate;
}

// ---- entries ----

pub fn entries() -> Vec<Entry> {
    let e = |op, kernel| Entry {
        op,
        route: "c",
        variant: "",
        class: Class::Simd,
        kernel,
    };
    vec![
        e("O1", Kernel::O1(u64x64_o1_c)),
        e("O2", Kernel::O2(u64x64_o2_c)),
        e("O3", Kernel::O3(u64x64_o3_c)),
        e("O4", Kernel::O4(u64x64_o4_c)),
        e("O4a", Kernel::O4a(u64x64_o4a_c)),
        e("O5", Kernel::O5(u64x64_o5_c)),
        e("O6", Kernel::O6(u64x64_o6_c)),
        Entry {
            op: "O6",
            route: "c",
            variant: "pow2",
            class: Class::Simd,
            kernel: Kernel::O6(u64x64_o6_c_pow2),
        },
    ]
}
