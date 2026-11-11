//! D-U64X64-LLVM-1 route A ("native") and the A0 baseline.
//!
//! Route A is plain runtime-dimension Rust over the borrowed planes: integer
//! loops, `u64::count_ones`, `/` for O6's floor division, and early skips over
//! non-demanded categories (the demanded set is walked bit by bit, so bytes
//! read scale with |D|). No SIMD intrinsics, no shuffles, no i8, no allocation
//! inside a kernel.
//!
//! A0 (plan §3, last row) is the existing mask-RISC baseline for O2 and O6:
//! each demanded stock plane is widened into a stack `[i32; 64]` per call and
//! thresholded with `ndarray::simd::ge_i32_to_mask`, combined with
//! `ndarray::simd::mask_and`. O6 is the Foundry layer-cake: one CanCook(q)
//! mask per q in 1..=128, cap_r = the vertical count of masks with bit r set.
//! `limit` and the status words are derived scalarly per row from the final
//! cap, exactly as the oracle does.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::harness::Class;
use crate::{Entry, Kernel};
use ndarray::simd::{ge_i32_to_mask, mask_and};

// ---- O1 ----

/// O1: `plane_cnt[k] = popcnt(A[k] & B[k] & live)`; `row_cnt` accumulated by
/// walking the set bits of each masked word.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o1_a(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    out.row_cnt = [0; 64];
    let mut total = 0u32;
    for k in 0..64 {
        let x = a[k] & b[k] & live;
        let n = x.count_ones();
        out.plane_cnt[k] = n as u8;
        total += n;
        let mut w = x;
        while w != 0 {
            let r = w.trailing_zeros() as usize;
            w &= w - 1;
            out.row_cnt[r] += 1;
        }
    }
    out.total = total as u16;
}

// ---- O2 ----

/// O2: unsigned threshold per demanded plane; non-demanded planes give 0.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_a(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    out.ge = [0; 64];
    let mut all = u64::MAX;
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let tk = t[k];
        let plane = &bound.src.stock[k];
        let mut ge = 0u64;
        for r in 0..64 {
            ge |= ((plane[r] as u32 >= tk) as u64) << r;
        }
        out.ge[k] = ge;
        all &= ge;
    }
    out.all_ge = all;
}

// ---- O3 ----

/// O3: presence-aware min/argmin per row over demanded categories, strict `<`
/// so ties keep the smallest k. Absent slots are never read as values.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_a(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    let src = bound.src;
    let mut none = 0u64;
    let mut unknown = 0u64;
    for r in 0..64 {
        let mut have = false;
        let mut best = 0u16;
        let mut arg = 0u8;
        let mut dm = demanded;
        while dm != 0 {
            let k = dm.trailing_zeros() as usize;
            dm &= dm - 1;
            if (src.present[k] >> r) & 1 == 0 {
                unknown |= 1u64 << r;
                continue;
            }
            let v = src.stock[k][r];
            if !have || v < best {
                have = true;
                best = v;
                arg = k as u8;
            }
        }
        if have {
            out.min[r] = best;
            out.arg[r] = arg;
        } else {
            none |= 1u64 << r;
            out.min[r] = 0;
            out.arg[r] = 0xFF;
        }
    }
    out.none = none;
    out.unknown = unknown;
}

// ---- O4 ----

/// O4: weighted row sum; categories with `w_k == 0` are skipped.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_a(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    out.sum = [0; 64];
    for k in 0..64 {
        let wk = w[k] as u32;
        if wk == 0 {
            continue;
        }
        let plane = &cost[k];
        for r in 0..64 {
            out.sum[r] += wk * plane[r] as u32;
        }
    }
    let mut total = 0u64;
    let mut l = live;
    while l != 0 {
        let r = l.trailing_zeros() as usize;
        l &= l - 1;
        total += out.sum[r] as u64;
    }
    out.total = total;
}

// ---- O4a ----

/// O4a: checked add, wrapping sum plus carry-out bit per element.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o4a_a(x: &ValueTile<u16>, y: &ValueTile<u16>, out: &mut O4aOut) {
    for k in 0..64 {
        let mut ovf = 0u64;
        for r in 0..64 {
            let (z, c) = x[k][r].overflowing_add(y[k][r]);
            out.z[k][r] = z;
            ovf |= (c as u64) << r;
        }
        out.ovf[k] = ovf;
    }
}

// ---- O5 ----

/// O5: one transactional Abakus update. Check every lane of `m` first; commit
/// only when none overflows.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o5_a(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let inc = 1u32 << u.b;
    let mut lanes = 0u64;
    let mut w = u.m;
    while w != 0 {
        let r = w.trailing_zeros() as usize;
        w &= w - 1;
        if counters[k][r] as u32 + inc >= 65536 {
            lanes |= 1u64 << r;
        }
    }
    if lanes != 0 {
        return Err(Refusal::Overflow { lanes });
    }
    let mut w = u.m;
    while w != 0 {
        let r = w.trailing_zeros() as usize;
        w &= w - 1;
        counters[k][r] = (counters[k][r] as u32 + inc) as u16;
    }
    Ok(())
}

// ---- O6 ----

/// O6: CanCook capacity with runtime `/`, min, and an ordered `limit` tracked
/// in the same pass (first k attaining the final cap; a category ties ahead of
/// STAFF and CEILING).
#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_a(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let d = &params.recipe.d;
    let (list, n) = params.recipe.demanded_list();
    let mut known = 0u64;
    let mut unknown = 0u64;
    let mut denied = 0u64;
    for r in 0..64 {
        out.cap[r] = 0;
        out.limit[r] = 0;
        if (src.gate >> r) & 1 == 0 {
            out.status[r] = RowStatus::Denied;
            denied |= 1u64 << r;
            continue;
        }
        let mut missing: Option<u8> = None;
        for i in 0..n {
            let k = list[i] as usize;
            if (src.present[k] >> r) & 1 == 0 {
                missing = Some(k as u8);
                break;
            }
        }
        if let Some(k) = missing {
            out.status[r] = RowStatus::Unknown { k };
            unknown |= 1u64 << r;
            continue;
        }
        let staff_cap = src.staff[r] as u32 * params.u as u32;
        let mut cap = staff_cap.min(QMAX);
        let mut lim: Option<u8> = None;
        for i in 0..n {
            let k = list[i] as usize;
            let q = src.stock[k][r] as u32 / d[k] as u32;
            if q < cap {
                cap = q;
                lim = Some(k as u8);
            } else if q == cap && lim.is_none() {
                lim = Some(k as u8);
            }
        }
        out.status[r] = RowStatus::Known;
        out.cap[r] = cap as u8;
        out.limit[r] = match lim {
            Some(k) => k,
            None if staff_cap == cap => STAFF,
            None => CEILING,
        };
        known |= 1u64 << r;
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

// ---- A0: existing mask-RISC primitives over widened i32 planes ----

/// A0 O2: per demanded plane, widen `u16 -> i32` into a stack buffer, threshold
/// with `ge_i32_to_mask`, accumulate `all_ge` with `mask_and`.
#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_a0(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    out.ge = [0; 64];
    let mut all = [u64::MAX];
    let mut dm = demanded;
    while dm != 0 {
        let k = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let mut buf = [0i32; 64];
        for r in 0..64 {
            buf[r] = bound.src.stock[k][r] as i32;
        }
        let mut w = [0u64];
        ge_i32_to_mask(&buf, t[k] as i32, &mut w);
        out.ge[k] = w[0];
        let mut acc = [0u64];
        mask_and(&all, &w, &mut acc);
        all = acc;
    }
    out.all_ge = all[0];
}

/// A0 O6: the Foundry layer-cake. For q in 1..=128 build the CanCook(q) mask
/// (known rows, staff*u >= q, and `stock >= d_k*q` for every demanded k) with
/// `ge_i32_to_mask` and `mask_and`; cap_r is the number of q whose mask has
/// bit r set. Status words and `limit` are derived scalarly per row from the
/// final cap, as the oracle does (`stock < (cap+1)*d` is the multiply form of
/// `stock / d == cap` once `stock >= cap*d` is known).
#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_a0(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let d = &params.recipe.d;
    let (list, n) = params.recipe.demanded_list();

    // Known rows: gate open and every demanded category present.
    let mut known = src.gate;
    for i in 0..n {
        known &= src.present[list[i] as usize];
    }

    let mut staffcap = [0i32; 64];
    for r in 0..64 {
        staffcap[r] = src.staff[r] as i32 * params.u as i32;
    }

    let mut cook = [0u64; QMAX as usize];
    for q in 1..=QMAX as usize {
        let mut w = [0u64];
        ge_i32_to_mask(&staffcap, q as i32, &mut w);
        cook[q - 1] = known & w[0];
    }

    for i in 0..n {
        let k = list[i] as usize;
        let dk = d[k] as i32;
        let mut buf = [0i32; 64];
        for r in 0..64 {
            buf[r] = src.stock[k][r] as i32;
        }
        for q in 1..=QMAX as usize {
            let mut w = [0u64];
            ge_i32_to_mask(&buf, dk * q as i32, &mut w);
            let mut acc = [0u64];
            mask_and(&[cook[q - 1]], &w, &mut acc);
            cook[q - 1] = acc[0];
        }
    }

    // Vertical count: cap_r = number of q with CanCook(q) at row r.
    let mut cap = [0u8; 64];
    for q in 0..QMAX as usize {
        let m = cook[q];
        for r in 0..64 {
            cap[r] += ((m >> r) & 1) as u8;
        }
    }

    let mut unknown = 0u64;
    let mut denied = 0u64;
    for r in 0..64 {
        out.cap[r] = 0;
        out.limit[r] = 0;
        if (src.gate >> r) & 1 == 0 {
            out.status[r] = RowStatus::Denied;
            denied |= 1u64 << r;
            continue;
        }
        let mut missing: Option<u8> = None;
        for i in 0..n {
            let k = list[i] as usize;
            if (src.present[k] >> r) & 1 == 0 {
                missing = Some(k as u8);
                break;
            }
        }
        if let Some(k) = missing {
            out.status[r] = RowStatus::Unknown { k };
            unknown |= 1u64 << r;
            continue;
        }
        let c = cap[r] as u32;
        let mut lim: Option<u8> = None;
        for i in 0..n {
            let k = list[i] as usize;
            if (src.stock[k][r] as u32) < (c + 1) * d[k] as u32 {
                lim = Some(k as u8);
                break;
            }
        }
        let staff_cap = src.staff[r] as u32 * params.u as u32;
        out.status[r] = RowStatus::Known;
        out.cap[r] = cap[r];
        out.limit[r] = match lim {
            Some(k) => k,
            None if staff_cap == c => STAFF,
            None => CEILING,
        };
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

// ---- entries ----

pub fn entries() -> Vec<Entry> {
    let e = |op, route, kernel| Entry {
        op,
        route,
        variant: "",
        class: Class::Scalar,
        kernel,
    };
    vec![
        e("O1", "a", Kernel::O1(u64x64_o1_a)),
        e("O2", "a", Kernel::O2(u64x64_o2_a)),
        e("O3", "a", Kernel::O3(u64x64_o3_a)),
        e("O4", "a", Kernel::O4(u64x64_o4_a)),
        e("O4a", "a", Kernel::O4a(u64x64_o4a_a)),
        e("O5", "a", Kernel::O5(u64x64_o5_a)),
        e("O6", "a", Kernel::O6(u64x64_o6_a)),
        e("O2", "a0", Kernel::O2(u64x64_o2_a0)),
        e("O6", "a0", Kernel::O6(u64x64_o6_a0)),
    ]
}
