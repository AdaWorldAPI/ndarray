//! D-U64X64-LLVM-1 route D: const-generic specialization.
//!
//! Route D asks what LLVM does with whole-tile code when it can see the
//! dimensions (`const R, K`, both 64 here) and, for `const_sparse` /
//! `const_dense`, the recipe itself. Everything is plain safe Rust: fixed-size
//! arrays, const trip counts, no ndarray SIMD, no intrinsics, no asm, no
//! shuffles, no i8, no allocation in a kernel. Whether LLVM vectorizes any of
//! it is measured later from the IR/asm.
//!
//! Variants (each its own monomorphic `#[no_mangle] #[inline(never)]` symbol):
//! - `loop`: const-dim loops over all categories. The demanded mask is applied
//!   as an early `continue` per category (`if (D >> k) & 1 == 0 { continue }`),
//!   not as a branch-free select, so bytes read scale with |D| like route A.
//!   O6 keeps the runtime `/` (LLVM may not strength-reduce it).
//! - `u8`: rows processed in groups of 8 via `as_chunks::<8>()` / `[T; 8]`
//!   lane arrays, with the per-lane work written as selects.
//! - `full`: `macro_rules!` unrolls the 64-category loop into 64 copies
//!   (O2 and O6 only).
//! - `const_sparse` / `const_dense` (O6 only, op string "O6c"): the recipe is a
//!   compile-time constant. These exist only for the matching recipe and are
//!   labelled compile-time-recipe-only; the orchestrator wires a dedicated
//!   "O6c" check that feeds only the matching recipe.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::harness::Class;
use crate::{Entry, Kernel};

/// Compile-time recipe for the `const_*` O6 variants. This shadows the
/// fixture's runtime `Recipe` struct inside this module on purpose; the
/// runtime one is only reached through `params.recipe`.
pub trait Recipe {
    const ID: u64;
    const D: [u16; 64];
}

/// The Foundry store-09 recipe: d_3 = 1, d_17 = 2, d_40 = 1, all else 0.
pub struct Sparse3;

impl Recipe for Sparse3 {
    const ID: u64 = 0x5350_4152_5345_0003;
    const D: [u16; 64] = {
        let mut d = [0u16; 64];
        d[3] = 1;
        d[17] = 2;
        d[40] = 1;
        d
    };
}

/// Dense recipe: d_k = 1 + (k % 3) for every category.
pub struct Dense;

impl Recipe for Dense {
    const ID: u64 = 0x4445_4e53_4500_0064;
    const D: [u16; 64] = {
        let mut d = [0u16; 64];
        let mut k = 0;
        while k < 64 {
            d[k] = 1 + (k % 3) as u16;
            k += 1;
        }
        d
    };
}

/// True when the runtime recipe equals the compile-time one `R`.
pub fn recipe_matches<R: Recipe>(params: &Params) -> bool {
    params.recipe.d == R::D
}

/// Unrolls `$f!(k)` for k = 0..=63, each with a literal k.
macro_rules! unroll64 {
    ($f:ident) => {
        $f!(0); $f!(1); $f!(2); $f!(3); $f!(4); $f!(5); $f!(6); $f!(7);
        $f!(8); $f!(9); $f!(10); $f!(11); $f!(12); $f!(13); $f!(14); $f!(15);
        $f!(16); $f!(17); $f!(18); $f!(19); $f!(20); $f!(21); $f!(22); $f!(23);
        $f!(24); $f!(25); $f!(26); $f!(27); $f!(28); $f!(29); $f!(30); $f!(31);
        $f!(32); $f!(33); $f!(34); $f!(35); $f!(36); $f!(37); $f!(38); $f!(39);
        $f!(40); $f!(41); $f!(42); $f!(43); $f!(44); $f!(45); $f!(46); $f!(47);
        $f!(48); $f!(49); $f!(50); $f!(51); $f!(52); $f!(53); $f!(54); $f!(55);
        $f!(56); $f!(57); $f!(58); $f!(59); $f!(60); $f!(61); $f!(62); $f!(63);
    };
}

// ---- O1 ----

#[inline(always)]
fn o1_loop<const R: usize, const K: usize>(
    a: &[u64; K],
    b: &[u64; K],
    live: u64,
    out: &mut O1Out,
) {
    let mut acc = [0u8; R];
    let mut total = 0u32;
    for k in 0..K {
        let x = a[k] & b[k] & live;
        let n = x.count_ones();
        out.plane_cnt[k] = n as u8;
        total += n;
        for r in 0..R {
            acc[r] += ((x >> r) & 1) as u8;
        }
    }
    for r in 0..R {
        out.row_cnt[r] = acc[r];
    }
    out.total = total as u16;
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o1_d_loop(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    o1_loop::<64, 64>(a, b, live, out)
}

// ---- O2 ----

#[inline(always)]
fn o2_loop<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    t: &[u32; K],
    demanded: u64,
    out: &mut O2Out,
) {
    let mut all = u64::MAX;
    for k in 0..K {
        if (demanded >> k) & 1 == 0 {
            out.ge[k] = 0;
            continue;
        }
        let tk = t[k];
        let plane = &stock[k];
        let mut ge = 0u64;
        for r in 0..R {
            ge |= ((plane[r] as u32 >= tk) as u64) << r;
        }
        out.ge[k] = ge;
        all &= ge;
    }
    out.all_ge = all;
}

#[inline(always)]
fn o2_u8<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    t: &[u32; K],
    demanded: u64,
    out: &mut O2Out,
) {
    let mut all = u64::MAX;
    for k in 0..K {
        if (demanded >> k) & 1 == 0 {
            out.ge[k] = 0;
            continue;
        }
        let tk = t[k];
        let (chunks, _) = stock[k].as_slice().as_chunks::<8>();
        let mut ge = 0u64;
        for g in 0..R / 8 {
            let v = chunks[g];
            let mut byte = 0u8;
            for j in 0..8 {
                byte |= ((v[j] as u32 >= tk) as u8) << j;
            }
            ge |= (byte as u64) << (8 * g);
        }
        out.ge[k] = ge;
        all &= ge;
    }
    out.all_ge = all;
}

/// Full 64-category unroll; the per-plane row loop stays a loop.
#[inline(always)]
fn o2_full(stock: &ValueTile<u16>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    let mut all = u64::MAX;
    macro_rules! step {
        ($k:literal) => {{
            if (demanded >> $k) & 1 == 0 {
                out.ge[$k] = 0;
            } else {
                let tk = t[$k];
                let plane = &stock[$k];
                let mut ge = 0u64;
                for r in 0..64 {
                    ge |= ((plane[r] as u32 >= tk) as u64) << r;
                }
                out.ge[$k] = ge;
                all &= ge;
            }
        }};
    }
    unroll64!(step);
    out.all_ge = all;
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_d_loop(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    o2_loop::<64, 64>(&bound.src.stock, t, demanded, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_d_u8(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    o2_u8::<64, 64>(&bound.src.stock, t, demanded, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_d_full(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    o2_full(&bound.src.stock, t, demanded, out)
}

// ---- O3 ----

#[inline(always)]
fn o3_loop<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    present: &[u64; K],
    demanded: u64,
    out: &mut O3Out,
) {
    let mut none = 0u64;
    let mut unknown = 0u64;
    for r in 0..R {
        let mut have = false;
        let mut best = 0u16;
        let mut arg = 0u8;
        for k in 0..K {
            if (demanded >> k) & 1 == 0 {
                continue;
            }
            if (present[k] >> r) & 1 == 0 {
                unknown |= 1u64 << r;
                continue;
            }
            let v = stock[k][r];
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

#[inline(always)]
fn o3_u8<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    present: &[u64; K],
    demanded: u64,
    out: &mut O3Out,
) {
    let mut none = 0u64;
    let mut unknown = 0u64;
    for g in 0..R / 8 {
        let base = g * 8;
        let mut have = [false; 8];
        let mut best = [0u16; 8];
        let mut arg = [0u8; 8];
        let mut unk = [false; 8];
        for k in 0..K {
            if (demanded >> k) & 1 == 0 {
                continue;
            }
            let p = (present[k] >> base) as u32;
            let (chunks, _) = stock[k].as_slice().as_chunks::<8>();
            let v = chunks[g];
            for j in 0..8 {
                let pres = (p >> j) & 1 == 1;
                let take = pres && (!have[j] || v[j] < best[j]);
                best[j] = if take { v[j] } else { best[j] };
                arg[j] = if take { k as u8 } else { arg[j] };
                have[j] |= pres;
                unk[j] |= !pres;
            }
        }
        for j in 0..8 {
            let r = base + j;
            unknown |= (unk[j] as u64) << r;
            none |= ((!have[j]) as u64) << r;
            out.min[r] = if have[j] { best[j] } else { 0 };
            out.arg[r] = if have[j] { arg[j] } else { 0xFF };
        }
    }
    out.none = none;
    out.unknown = unknown;
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_d_loop(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    o3_loop::<64, 64>(&bound.src.stock, &bound.src.present, demanded, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_d_u8(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    o3_u8::<64, 64>(&bound.src.stock, &bound.src.present, demanded, out)
}

// ---- O4 ----

#[inline(always)]
fn o4_loop<const R: usize, const K: usize>(
    cost: &[[u8; R]; K],
    w: &[u16; K],
    live: u64,
    out: &mut O4Out,
) {
    let mut sum = [0u32; R];
    for k in 0..K {
        let wk = w[k] as u32;
        if wk == 0 {
            continue;
        }
        let plane = &cost[k];
        for r in 0..R {
            sum[r] = sum[r].wrapping_add(wk.wrapping_mul(plane[r] as u32));
        }
    }
    let mut total = 0u64;
    for r in 0..R {
        out.sum[r] = sum[r];
        total += (sum[r] as u64) & ((live >> r) & 1).wrapping_neg();
    }
    out.total = total;
}

#[inline(always)]
fn o4_u8<const R: usize, const K: usize>(
    cost: &[[u8; R]; K],
    w: &[u16; K],
    live: u64,
    out: &mut O4Out,
) {
    let mut total = 0u64;
    for g in 0..R / 8 {
        let mut acc = [0u32; 8];
        for k in 0..K {
            let wk = w[k] as u32;
            let (chunks, _) = cost[k].as_slice().as_chunks::<8>();
            let c = chunks[g];
            for j in 0..8 {
                acc[j] = acc[j].wrapping_add(wk.wrapping_mul(c[j] as u32));
            }
        }
        for j in 0..8 {
            let r = g * 8 + j;
            out.sum[r] = acc[j];
            total += (acc[j] as u64) & ((live >> r) & 1).wrapping_neg();
        }
    }
    out.total = total;
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_d_loop(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    o4_loop::<64, 64>(cost, w, live, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_d_u8(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    o4_u8::<64, 64>(cost, w, live, out)
}

// ---- O4a ----

#[inline(always)]
fn o4a_loop<const R: usize, const K: usize>(
    x: &[[u16; R]; K],
    y: &[[u16; R]; K],
    out: &mut O4aOut,
) {
    for k in 0..K {
        let mut ovf = 0u64;
        for r in 0..R {
            let (z, c) = x[k][r].overflowing_add(y[k][r]);
            out.z[k][r] = z;
            ovf |= (c as u64) << r;
        }
        out.ovf[k] = ovf;
    }
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o4a_d_loop(x: &ValueTile<u16>, y: &ValueTile<u16>, out: &mut O4aOut) {
    o4a_loop::<64, 64>(x, y, out)
}

// ---- O5 ----

#[inline(always)]
fn o5_loop<const R: usize, const K: usize>(
    counters: &mut [[u16; R]; K],
    u: O5Update,
) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let inc = 1u32 << u.b;
    let mut lanes = 0u64;
    for r in 0..R {
        let on = (u.m >> r) & 1 == 1;
        let ovf = on && counters[k][r] as u32 + inc >= 65536;
        lanes |= (ovf as u64) << r;
    }
    if lanes != 0 {
        return Err(Refusal::Overflow { lanes });
    }
    let inc16 = inc as u16;
    for r in 0..R {
        let mask = (((u.m >> r) & 1) as u16).wrapping_neg();
        counters[k][r] = counters[k][r].wrapping_add(inc16 & mask);
    }
    Ok(())
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o5_d_loop(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    o5_loop::<64, 64>(counters, u)
}

// ---- O6 ----

/// Shared tail of one Known row: cap from the staff/ceiling bound, then the
/// ordered `limit`.
#[inline(always)]
fn o6_limit(lim: Option<u8>, staff_cap: u32, cap: u32) -> u8 {
    match lim {
        Some(k) => k,
        None if staff_cap == cap => STAFF,
        None => CEILING,
    }
}

/// Row-outer loop with early `continue` over categories; `d` is whatever the
/// caller passes (runtime recipe for `loop`, a constant for `const_*`).
#[inline(always)]
fn o6_loop<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    present: &[u64; K],
    gate: u64,
    staff: &[u8; R],
    u: u8,
    d: &[u16; K],
    out: &mut O6Out,
) {
    let mut known = 0u64;
    let mut unknown = 0u64;
    let mut denied = 0u64;
    for r in 0..R {
        out.cap[r] = 0;
        out.limit[r] = 0;
        if (gate >> r) & 1 == 0 {
            out.status[r] = RowStatus::Denied;
            denied |= 1u64 << r;
            continue;
        }
        let mut missing: Option<u8> = None;
        for k in 0..K {
            if d[k] > 0 && (present[k] >> r) & 1 == 0 {
                missing = Some(k as u8);
                break;
            }
        }
        if let Some(k) = missing {
            out.status[r] = RowStatus::Unknown { k };
            unknown |= 1u64 << r;
            continue;
        }
        let staff_cap = staff[r] as u32 * u as u32;
        let mut cap = staff_cap.min(QMAX);
        let mut lim: Option<u8> = None;
        for k in 0..K {
            if d[k] == 0 {
                continue;
            }
            let q = stock[k][r] as u32 / d[k] as u32;
            if q < cap {
                cap = q;
                lim = Some(k as u8);
            } else if q == cap && lim.is_none() {
                lim = Some(k as u8);
            }
        }
        out.status[r] = RowStatus::Known;
        out.cap[r] = cap as u8;
        out.limit[r] = o6_limit(lim, staff_cap, cap);
        known |= 1u64 << r;
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

/// Row groups of 8 lanes; category-outer inside a group, lane work as selects.
#[inline(always)]
fn o6_u8<const R: usize, const K: usize>(
    stock: &[[u16; R]; K],
    present: &[u64; K],
    gate: u64,
    staff: &[u8; R],
    u: u8,
    d: &[u16; K],
    out: &mut O6Out,
) {
    let mut known = 0u64;
    let mut unknown = 0u64;
    let mut denied = 0u64;
    for g in 0..R / 8 {
        let base = g * 8;
        let mut staff_cap = [0u32; 8];
        let mut cap = [0u32; 8];
        let mut lim = [0u8; 8];
        let mut has = [false; 8];
        let mut miss = [0xFFu8; 8];
        for j in 0..8 {
            staff_cap[j] = staff[base + j] as u32 * u as u32;
            cap[j] = staff_cap[j].min(QMAX);
        }
        for k in 0..K {
            let dk = d[k] as u32;
            if dk == 0 {
                continue;
            }
            let p = (present[k] >> base) as u32;
            let (chunks, _) = stock[k].as_slice().as_chunks::<8>();
            let v = chunks[g];
            for j in 0..8 {
                let pres = (p >> j) & 1 == 1;
                let q = v[j] as u32 / dk;
                let lt = pres && q < cap[j];
                let eq = pres && q == cap[j] && !has[j];
                cap[j] = if lt { q } else { cap[j] };
                lim[j] = if lt || eq { k as u8 } else { lim[j] };
                has[j] |= lt || eq;
                miss[j] = if !pres && miss[j] == 0xFF { k as u8 } else { miss[j] };
            }
        }
        for j in 0..8 {
            let r = base + j;
            out.cap[r] = 0;
            out.limit[r] = 0;
            if (gate >> r) & 1 == 0 {
                out.status[r] = RowStatus::Denied;
                denied |= 1u64 << r;
            } else if miss[j] != 0xFF {
                out.status[r] = RowStatus::Unknown { k: miss[j] };
                unknown |= 1u64 << r;
            } else {
                let l = if has[j] { Some(lim[j]) } else { None };
                out.status[r] = RowStatus::Known;
                out.cap[r] = cap[j] as u8;
                out.limit[r] = o6_limit(l, staff_cap[j], cap[j]);
                known |= 1u64 << r;
            }
        }
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

/// Full 64-category unroll of both the missing scan and the cap scan.
#[inline(always)]
fn o6_full(
    stock: &ValueTile<u16>,
    present: &BitTile,
    gate: u64,
    staff: &[u8; 64],
    u: u8,
    d: &[u16; 64],
    out: &mut O6Out,
) {
    let mut known = 0u64;
    let mut unknown = 0u64;
    let mut denied = 0u64;
    for r in 0..64 {
        out.cap[r] = 0;
        out.limit[r] = 0;
        if (gate >> r) & 1 == 0 {
            out.status[r] = RowStatus::Denied;
            denied |= 1u64 << r;
            continue;
        }
        let mut missing = 0xFFu8;
        macro_rules! miss_step {
            ($k:literal) => {
                if missing == 0xFF && d[$k] > 0 && (present[$k] >> r) & 1 == 0 {
                    missing = $k;
                }
            };
        }
        unroll64!(miss_step);
        if missing != 0xFF {
            out.status[r] = RowStatus::Unknown { k: missing };
            unknown |= 1u64 << r;
            continue;
        }
        let staff_cap = staff[r] as u32 * u as u32;
        let mut cap = staff_cap.min(QMAX);
        let mut lim: Option<u8> = None;
        macro_rules! cap_step {
            ($k:literal) => {
                if d[$k] > 0 {
                    let q = stock[$k][r] as u32 / d[$k] as u32;
                    if q < cap {
                        cap = q;
                        lim = Some($k);
                    } else if q == cap && lim.is_none() {
                        lim = Some($k);
                    }
                }
            };
        }
        unroll64!(cap_step);
        out.status[r] = RowStatus::Known;
        out.cap[r] = cap as u8;
        out.limit[r] = o6_limit(lim, staff_cap, cap);
        known |= 1u64 << r;
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_d_loop(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let s = bound.src;
    o6_loop::<64, 64>(&s.stock, &s.present, s.gate, &s.staff, params.u, &params.recipe.d, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_d_u8(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let s = bound.src;
    o6_u8::<64, 64>(&s.stock, &s.present, s.gate, &s.staff, params.u, &params.recipe.d, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_d_full(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let s = bound.src;
    o6_full(&s.stock, &s.present, s.gate, &s.staff, params.u, &params.recipe.d, out)
}

/// Compile-time recipe: `R::D` replaces the runtime `d`, so LLVM can drop
/// undemanded categories and fold the divisions by constants.
#[inline(always)]
fn o6_const<R: Recipe>(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let s = bound.src;
    o6_loop::<64, 64>(&s.stock, &s.present, s.gate, &s.staff, params.u, &R::D, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_d_const_sparse(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    assert!(params.recipe.d == Sparse3::D, "recipe mismatch");
    o6_const::<Sparse3>(bound, params, out)
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_d_const_dense(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    assert!(params.recipe.d == Dense::D, "recipe mismatch");
    o6_const::<Dense>(bound, params, out)
}

// ---- entries ----

/// Every entry is registered as `Class::Scalar`; the orchestrator reclassifies
/// from the asm (D is SIMD only if its hot loop has more vector than scalar
/// data ops). The `const_*` entries use op "O6c" and must only be checked and
/// benchmarked with the matching recipe (`recipe_matches::<Sparse3>` /
/// `recipe_matches::<Dense>`).
pub fn entries() -> Vec<Entry> {
    let e = |op, variant, kernel| Entry {
        op,
        route: "d",
        variant,
        class: Class::Scalar,
        kernel,
    };
    vec![
        e("O1", "loop", Kernel::O1(u64x64_o1_d_loop)),
        e("O2", "loop", Kernel::O2(u64x64_o2_d_loop)),
        e("O2", "u8", Kernel::O2(u64x64_o2_d_u8)),
        e("O2", "full", Kernel::O2(u64x64_o2_d_full)),
        e("O3", "loop", Kernel::O3(u64x64_o3_d_loop)),
        e("O3", "u8", Kernel::O3(u64x64_o3_d_u8)),
        e("O4", "loop", Kernel::O4(u64x64_o4_d_loop)),
        e("O4", "u8", Kernel::O4(u64x64_o4_d_u8)),
        e("O4a", "loop", Kernel::O4a(u64x64_o4a_d_loop)),
        e("O5", "loop", Kernel::O5(u64x64_o5_d_loop)),
        e("O6", "loop", Kernel::O6(u64x64_o6_d_loop)),
        e("O6", "u8", Kernel::O6(u64x64_o6_d_u8)),
        e("O6", "full", Kernel::O6(u64x64_o6_d_full)),
        e("O6c", "const_sparse", Kernel::O6(u64x64_o6_d_const_sparse)),
        e("O6c", "const_dense", Kernel::O6(u64x64_o6_d_const_dense)),
    ]
}
