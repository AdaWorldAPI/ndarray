//! D-U64X64-LLVM-1 route B ("bit-sliced").
//!
//! A 64-row value plane (one category, 64 lanes of u16 or u8) is held as bit
//! planes: plane `j` is a `u64` whose bit `r` is bit `j` of row `r`'s value.
//! All arithmetic is `u64` word logic across the 64 rows at once: masks,
//! `full_add_mask` / `half_add_mask` ripples, MSB-first comparators and
//! restoring division by a constant. Lanes are read back only where an `Out`
//! struct needs lane values, with `U8x64::mask_blend` and the `U16x32`
//! widening (no shuffles, no i8, no allocation inside a kernel).
//!
//! Public API (route E reuses these on cached planes): `planes_u16`,
//! `planes_u8`, `ge_const`, `o2_on_planes`, `o6_on_planes`.
//!
//! O6 reads only `stock`, `present`, `gate` and `staff`; the `Out` struct has
//! no cost field, so cost is not read at all (`network_cost` is the oracle's
//! post-processing of the `Out`).
//!
//! Disable-run anchors are marked `// B<n>:` (plan §6).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::harness::Class;
use crate::{Entry, Kernel};
use ndarray::simd::{full_add_mask, half_add_mask, U16x32, U8x64};

// ---- plane construction and readout ----

/// Bit planes of a u16 plane: plane `j` bit `r` = bit `j` of `plane[r]`.
#[inline(always)]
pub fn planes_u16(plane: &[u16; 64]) -> [u64; 16] {
    let lo = U16x32::from_slice(&plane[..32]);
    let hi = U16x32::from_slice(&plane[32..]);
    let mut p = [0u64; 16];
    for j in 0..16 {
        let m = U16x32::splat(1u16 << j);
        p[j] = (lo & m).cmpeq_mask(m) as u64 | ((hi & m).cmpeq_mask(m) as u64) << 32;
    }
    p
}

/// Bit planes of a u8 plane: plane `j` bit `r` = bit `j` of `plane[r]`.
#[inline(always)]
pub fn planes_u8(plane: &[u8; 64]) -> [u64; 8] {
    let v = U8x64::from_slice(plane);
    let mut p = [0u64; 8];
    for j in 0..8 {
        let m = U8x64::splat(1u8 << j);
        p[j] = (v & m).cmpeq_mask(m);
    }
    p
}

/// Up to 8 bit planes back to u8 lanes (plane `j` supplies bit `j`).
#[inline(always)]
fn lanes8(p: &[u64]) -> U8x64 {
    let zero = U8x64::splat(0);
    let mut v = zero;
    for j in 0..p.len() {
        v = v | U8x64::mask_blend(p[j], zero, U8x64::splat(1u8 << j));
    }
    v
}

/// 16 bit planes back to u16 lanes: rows 0..32 and rows 32..64.
#[inline(always)]
fn lanes16(p: &[u64; 16]) -> [U16x32; 2] {
    let lo = lanes8(&p[..8]);
    let hi = lanes8(&p[8..]);
    [
        U16x32::from_u8x64_lo(lo) | U16x32::from_u8x64_lo(hi).shl(8),
        U16x32::from_u8x64_hi(lo) | U16x32::from_u8x64_hi(hi).shl(8),
    ]
}

/// All-ones when bit `i` of `k` is set, else zero.
#[inline(always)]
fn kbit(k: usize, i: usize) -> u64 {
    if (k >> i) & 1 == 1 {
        u64::MAX
    } else {
        0
    }
}

// ---- comparators and constant arithmetic ----

/// Rows whose value is `>= t`, unsigned. `t >= 65536` gives 0; `t == 0` gives
/// all rows.
#[inline(always)]
pub fn ge_const(planes: &[u64; 16], t: u32) -> u64 {
    if t > 65535 {
        return 0;
    }
    let mut gt = 0u64;
    let mut eq = !0u64;
    for j in (0..16).rev() {
        if (t >> j) & 1 == 1 {
            eq &= planes[j];
        } else {
            gt |= eq & planes[j];
            eq &= !planes[j];
        }
    }
    let ge = gt | eq; // B3: the `eq` term makes this `>=`; without it the comparator is `>`.
    ge
}

/// `(lt, eq)`: rows where `q < c` and `q == c`, both 8-plane unsigned numbers.
#[inline(always)]
fn cmp8(q: &[u64; 8], c: &[u64; 8]) -> (u64, u64) {
    let mut lt = 0u64;
    let mut eq = !0u64;
    for j in (0..8).rev() {
        lt |= eq & !q[j] & c[j];
        eq &= !(q[j] ^ c[j]);
    }
    (lt, eq)
}

/// `rem -= c` on the rows of `m` (the others unchanged); the caller
/// guarantees `rem >= c` on those rows, so no row underflows.
#[inline(always)]
fn sub_const_masked(rem: &mut [u64; 16], c: u32, m: u64) {
    let mut borrow = 0u64;
    for j in 0..16 {
        let x = rem[j];
        let y = if (c >> j) & 1 == 1 { m } else { 0 };
        rem[j] = x ^ y ^ borrow;
        borrow = (!x & y) | (!(x ^ y) & borrow);
    }
}

// ---- O1 ----

/// O1: a CSA vertical counter over the 64 words `A[k] & B[k] & live`.
/// Pairs of words enter column 0 through a full adder; the carry ripples up
/// the other six columns with half adders (64 <= 2^7 - 1, so none escapes).
/// `plane_cnt` is a scalar `count_ones` of each masked word.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o1_b(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    let mut c = [0u64; 7];
    let mut total = 0u32;
    for k in 0..64 {
        let w = a[k] & b[k] & live;
        let n = w.count_ones();
        out.plane_cnt[k] = n as u8;
        total += n;
    }
    for pair in 0..32 {
        let wa = a[2 * pair] & b[2 * pair] & live;
        let wb = a[2 * pair + 1] & b[2 * pair + 1] & live;
        let (s, cy) = full_add_mask(c[0], wa, wb);
        c[0] = s;
        let mut carry = cy; // B1: CSA carry into the next column.
        for j in 1..7 {
            let (s, c2) = half_add_mask(c[j], carry);
            c[j] = s;
            carry = c2;
        }
    }
    lanes8(&c).copy_to_slice(&mut out.row_cnt);
    out.total = total as u16;
}

// ---- O2 ----

/// Shared O2 core: `planes_of(k)` supplies the bit planes of category `k`.
#[inline(always)]
fn o2_core<F: Fn(usize) -> [u64; 16]>(t: &[u32; 64], demanded: u64, out: &mut O2Out, planes_of: F) {
    out.ge = [0; 64];
    let mut all = u64::MAX;
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let ge = ge_const(&planes_of(k), t[k]);
        out.ge[k] = ge;
        all &= ge;
    }
    out.all_ge = all;
}

/// O2 on prebuilt stock planes; only demanded categories are visited, the
/// other `ge[k]` are 0 and `all_ge` is all ones for an empty demand.
pub fn o2_on_planes(stock: &[[u64; 16]; 64], t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    o2_core(t, demanded, out, |k| stock[k]);
}

/// O2: builds each demanded category's planes per call (unprepared B), then
/// the same comparator path as `o2_on_planes`.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_b(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    o2_core(t, demanded, out, |k| planes_u16(&bound.src.stock[k]));
}

// ---- O3 ----

/// O3: a running min held as 16 planes, `arg` as 8 planes and a `have` mask.
/// Each demanded `k` (ascending) replaces the min on the rows where the slot
/// is present and either nothing was seen yet or the value is strictly
/// smaller, so the smallest `k` wins ties. Absent slots are never taken.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_b(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    let src = bound.src;
    let mut m = [0u64; 16];
    let mut arg = [0u64; 8];
    let mut have = 0u64;
    let mut unknown = 0u64;
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let valid = src.present[k];
        unknown |= !valid;
        let v = planes_u16(&src.stock[k]);
        // lt: v < m, MSB first.
        let mut lt = 0u64;
        let mut eq = !0u64;
        for j in (0..16).rev() {
            lt |= eq & !v[j] & m[j];
            eq &= !(v[j] ^ m[j]);
        }
        let take = valid & (!have | lt);
        for j in 0..16 {
            m[j] = (take & v[j]) | (!take & m[j]);
        }
        for i in 0..8 {
            arg[i] = (take & kbit(k, i)) | (!take & arg[i]);
        }
        have |= valid;
    }
    let none = !have;
    for i in 0..8 {
        arg[i] |= none; // none rows read 0xFF
    }
    let mv = lanes16(&m); // m stays 0 on none rows
    mv[0].copy_to_slice(&mut out.min[..32]);
    mv[1].copy_to_slice(&mut out.min[32..]);
    lanes8(&arg).copy_to_slice(&mut out.arg);
    out.none = none;
    out.unknown = unknown;
}

// ---- O4 ----

/// O4: a 32-plane accumulator. For every set bit `b` of `w_k` the 8 planes of
/// `C[k]` are added at significance `b`. The maximum sum is
/// `255 * 65535 * 64 < 2^32`, so 32 planes never overflow.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_b(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    let mut acc = [0u64; 32];
    for k in 0..64 {
        let wk = w[k];
        if wk == 0 {
            continue;
        }
        let c = planes_u8(&cost[k]);
        for b in 0..16 {
            if (wk >> b) & 1 == 0 {
                continue;
            }
            let mut carry = 0u64;
            for i in 0..8 {
                let (s, cy) = full_add_mask(acc[b + i], c[i], carry);
                acc[b + i] = s;
                carry = cy;
            }
            for j in (b + 8)..32 {
                if carry == 0 {
                    break;
                }
                let (s, cy) = half_add_mask(acc[j], carry);
                acc[j] = s;
                carry = cy;
            }
        }
    }
    // total over live rows, per plane.
    let mut total = 0u64;
    for j in 0..32 {
        total += ((acc[j] & live).count_ones() as u64) << j;
    }
    out.total = total;
    // facade gap: no u16 -> u32 widening, so the two halves are joined per row.
    let mut lo_p = [0u64; 16];
    let mut hi_p = [0u64; 16];
    lo_p.copy_from_slice(&acc[..16]);
    hi_p.copy_from_slice(&acc[16..]);
    let lo = lanes16(&lo_p);
    let hi = lanes16(&hi_p);
    for h in 0..2 {
        let la = lo[h].to_array();
        let ha = hi[h].to_array();
        for i in 0..32 {
            out.sum[h * 32 + i] = la[i] as u32 | (ha[i] as u32) << 16;
        }
    }
}

// ---- O4a ----

/// O4a: a 16-plane ripple of `X[k] + Y[k]` per category; the carry out of
/// plane 15 is `ovf[k]`.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4a_b(x: &ValueTile<u16>, y: &ValueTile<u16>, out: &mut O4aOut) {
    for k in 0..64 {
        let px = planes_u16(&x[k]);
        let py = planes_u16(&y[k]);
        let mut z = [0u64; 16];
        let mut carry = 0u64;
        for j in 0..16 {
            let (s, cy) = full_add_mask(px[j], py[j], carry);
            z[j] = s;
            carry = cy;
        }
        let zl = lanes16(&z);
        zl[0].copy_to_slice(&mut out.z[k][..32]);
        zl[1].copy_to_slice(&mut out.z[k][32..]);
        out.ovf[k] = carry;
    }
}

// ---- O5 ----

/// O5: one transactional Abakus update. The counters of category `k` are
/// planed into a temporary, `2^b` is added on the lanes of `m` by a ripple
/// from plane `b` (carry-in `m`), and the counters are written back only if
/// no lane carries out of plane 15.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o5_b(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let mut p = planes_u16(&counters[k]);
    let mut carry = u.m;
    for j in (u.b as usize)..16 {
        let (s, cy) = half_add_mask(p[j], carry);
        p[j] = s;
        carry = cy;
    }
    if carry != 0 {
        // B2: refuse and leave `counters` untouched (the write-back below never runs).
        return Err(Refusal::Overflow { lanes: carry });
    }
    let l = lanes16(&p);
    l[0].copy_to_slice(&mut counters[k][..32]);
    l[1].copy_to_slice(&mut counters[k][32..]);
    Ok(())
}

// ---- O6 ----

/// Shared O6 core. Per demanded `k`: the quotient `min(floor(S/d_k), 128)` by
/// bitsliced restoring division (rows with `S >= 128 d` are 128 directly, the
/// rest need bits 6..0), a running min over 8 planes against the staff cap,
/// and `limit` kept as 7 planes (k in bits 0..5; `STAFF` = 64, `CEILING` = 65).
/// `limit` updates on a strictly smaller value, or on the first exact tie
/// with the staff cap; a later tie keeps the earlier k. A clipped 128 counts
/// as a tie only when the real quotient is exactly 128 (`S < 129 d`).
fn o6_core<F: Fn(usize) -> [u64; 16]>(bound: &Bound<'_>, params: &Params, out: &mut O6Out, planes_of: F) {
    let src = bound.src;
    let d = &params.recipe.d;
    let (list, n) = params.recipe.demanded_list();

    // staff_raw = staff * u, as 16 planes (shift-add over the set bits of u).
    let st = planes_u8(&src.staff);
    let mut sr = [0u64; 16];
    for b in 0..8 {
        if (params.u >> b) & 1 == 0 {
            continue;
        }
        let mut carry = 0u64;
        for i in 0..8 {
            let (s, cy) = full_add_mask(sr[b + i], st[i], carry);
            sr[b + i] = s;
            carry = cy;
        }
        for j in (b + 8)..16 {
            let (s, cy) = half_add_mask(sr[j], carry);
            sr[j] = s;
            carry = cy;
        }
    }
    let big = ge_const(&sr, 128);
    // Rows where staff_raw <= 128: the capped value equals staff_raw.
    let stf_exact = !ge_const(&sr, 129);
    let mut cap = [0u64; 8];
    for j in 0..7 {
        cap[j] = sr[j] & !big;
    }
    cap[7] = big;

    let mut lim = [0u64; 7];
    let mut limset = 0u64;
    let mut fm = [0u64; 6];
    let mut missing = 0u64;

    for i in 0..n {
        let k = list[i] as usize;
        let p = src.present[k];
        let newm = !p & !missing;
        for b in 0..6 {
            fm[b] = (newm & kbit(k, b)) | (!newm & fm[b]);
        }
        missing |= !p;

        let sp = planes_of(k);
        let dk = d[k] as u32;
        let big_q = ge_const(&sp, 128 * dk);
        let exceeds = ge_const(&sp, 129 * dk);
        let mut rem = sp;
        let mut q = [0u64; 8];
        for b in (0..7).rev() {
            let c = dk << b;
            let ge = ge_const(&rem, c);
            sub_const_masked(&mut rem, c, ge);
            q[b] = ge & !big_q;
        }
        q[7] = big_q;

        let (lt, eq) = cmp8(&q, &cap);
        let upd = lt | (eq & !exceeds & !limset);
        for j in 0..8 {
            cap[j] = (lt & q[j]) | (!lt & cap[j]);
        }
        for b in 0..6 {
            lim[b] = (upd & kbit(k, b)) | (!upd & lim[b]);
        }
        limset |= upd;
    }

    let known = src.gate & !missing;
    // No category matched the final cap: STAFF (64) if the staff cap is exact, else CEILING (65).
    let none_lim = !limset;
    lim[6] = none_lim;
    lim[0] |= none_lim & !stf_exact;
    for b in 0..7 {
        lim[b] &= known;
    }
    for j in 0..8 {
        cap[j] &= known;
    }

    lanes8(&cap).copy_to_slice(&mut out.cap);
    lanes8(&lim).copy_to_slice(&mut out.limit);
    let fma = lanes8(&fm).to_array();
    // The `RowStatus` enum has no lane form, so it is written per row.
    for r in 0..64 {
        out.status[r] = if (src.gate >> r) & 1 == 0 {
            RowStatus::Denied
        } else if (missing >> r) & 1 == 1 {
            RowStatus::Unknown { k: fma[r] }
        } else {
            RowStatus::Known
        };
    }
    out.known = known;
    out.unknown = src.gate & missing;
    out.denied = !src.gate;
}

/// O6 on prebuilt stock planes (`stock[k]` for each demanded `k`). Cost is not
/// read: `O6Out` carries no cost field.
pub fn o6_on_planes(stock: &[[u64; 16]; 64], bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    o6_core(bound, params, out, |k| stock[k]);
}

/// O6: builds each demanded category's stock planes per call.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_b(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    o6_core(bound, params, out, |k| planes_u16(&bound.src.stock[k]));
}

// ---- entries ----

pub fn entries() -> Vec<Entry> {
    let e = |op, kernel| Entry {
        op,
        route: "b",
        variant: "",
        class: Class::Bitslice,
        kernel,
    };
    vec![
        e("O1", Kernel::O1(u64x64_o1_b)),
        e("O2", Kernel::O2(u64x64_o2_b)),
        e("O3", Kernel::O3(u64x64_o3_b)),
        e("O4", Kernel::O4(u64x64_o4_b)),
        e("O4a", Kernel::O4a(u64x64_o4a_b)),
        e("O5", Kernel::O5(u64x64_o5_b)),
        e("O6", Kernel::O6(u64x64_o6_b)),
    ]
}
