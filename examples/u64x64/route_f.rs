//! D-U64X64-LLVM-1 route F ("shader grid").
//!
//! The 64x64 is an X,Y grid: Y = category k (the demanded rows of the
//! source's own `ValueTile<u16>`, read in place), X = row r (64 lanes). A
//! shader is a per-cell function of `(y, x-row, uniforms)`; one generic
//! dispatcher maps it over the demanded Y in ascending order and folds the
//! result. Nothing is built from source values. Route F reuses route C's
//! per-cell arithmetic on purpose, so the measured difference is the
//! contract alone.
//!
//! Disable-run anchors are marked `// F<n>:`.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::harness::Class;
use crate::route_c::{self, join, lane16, quot8, widen_u8, P16};
use crate::{Entry, Kernel};
use ndarray::simd::{U16x32, U8x64};

/// One Y row of the grid: 64 X cells as two `U16x32` halves.
pub(crate) type Row = P16;

/// A per-cell function folded over the demanded Y rows.
pub(crate) trait Shader {
    type Acc: Copy;
    fn init(&self) -> Self::Acc;
    /// `row` is grid[y], loaded in place. Returns the folded accumulator.
    fn cell(&self, y: usize, row: Row, acc: Self::Acc) -> Self::Acc;
}

/// Maps `s` over every set bit of `ys` in ascending y, reading `grid[y]`
/// in place.
#[inline(always)]
pub(crate) fn dispatch<S: Shader>(grid: &ValueTile<u16>, mut ys: u64, s: &S) -> S::Acc {
    let mut acc = s.init();
    while ys != 0 {
        let y = ys.trailing_zeros() as usize;
        ys &= ys - 1; // F1: every demanded y is visited exactly once.
        acc = s.cell(y, route_c::load16(&grid[y]), acc);
    }
    acc
}

/// A shader that also writes one output word per visited y.
pub(crate) trait ShaderOut: Shader {
    fn out(&self, y: usize, row: Row, acc: Self::Acc) -> (Self::Acc, u64);
}

#[inline(always)]
pub(crate) fn dispatch_out<S: ShaderOut>(grid: &ValueTile<u16>, mut ys: u64, s: &S, out: &mut [u64; 64]) -> S::Acc {
    let mut acc = s.init();
    while ys != 0 {
        let y = ys.trailing_zeros() as usize;
        ys &= ys - 1; // F1
        let (a, w) = s.out(y, route_c::load16(&grid[y]), acc);
        acc = a;
        out[y] = w;
    }
    acc
}

// ---- O2 ----

/// O2 shader: `ge = !(t > v)` per demanded plane; `t > 65535` never.
struct O2Sh<'a> {
    t: &'a [u32; 64],
}

impl Shader for O2Sh<'_> {
    type Acc = u64;
    #[inline(always)]
    fn init(&self) -> u64 {
        u64::MAX
    }
    #[inline(always)]
    fn cell(&self, y: usize, row: Row, acc: u64) -> u64 {
        self.out(y, row, acc).0
    }
}

impl ShaderOut for O2Sh<'_> {
    #[inline(always)]
    fn out(&self, y: usize, row: Row, acc: u64) -> (u64, u64) {
        let tk = self.t[y];
        let ge = if tk > 65535 {
            0
        } else {
            let tv = U16x32::splat(tk as u16);
            let lt = join(tv.cmpgt_mask(row[0]), tv.cmpgt_mask(row[1]));
            !lt
        };
        (acc & ge, ge)
    }
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_f(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    out.ge = [0; 64];
    let sh = O2Sh { t };
    out.all_ge = dispatch_out(&bound.src.stock, demanded, &sh, &mut out.ge);
}

// ---- O3 ----

/// O3 shader: the running min / arg / presence fold of `u64x64_o3_c`.
struct O3Sh<'a> {
    src: &'a Source,
}

impl Shader for O3Sh<'_> {
    type Acc = (P16, U8x64, u64, u64);
    #[inline(always)]
    fn init(&self) -> Self::Acc {
        let ff = U16x32::splat(0xFFFF);
        ([ff, ff], U8x64::splat(0xFF), 0u64, 0u64)
    }
    #[inline(always)]
    fn cell(&self, y: usize, v: Row, acc: Self::Acc) -> Self::Acc {
        let (m, arg, have, unknown) = acc;
        let p = self.src.present[y];
        let unknown = unknown | !p;
        let pm = lane16(p);
        let vs: P16 = [v[0] | !pm[0], v[1] | !pm[1]];
        let lt = join(m[0].cmpgt_mask(vs[0]), m[1].cmpgt_mask(vs[1]));
        let upd = lt | (p & !have);
        let arg = U8x64::mask_blend(upd, arg, U8x64::splat(y as u8));
        let have = have | p;
        let m = [m[0].simd_min(vs[0]), m[1].simd_min(vs[1])];
        (m, arg, have, unknown)
    }
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_f(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    let src = bound.src;
    let sh = O3Sh { src };
    let (m, arg, have, unknown) = dispatch(&src.stock, demanded, &sh);
    let hm = lane16(have);
    (m[0] & hm[0]).copy_to_slice(&mut out.min[..32]);
    (m[1] & hm[1]).copy_to_slice(&mut out.min[32..]);
    arg.copy_to_slice(&mut out.arg);
    out.none = !have;
    out.unknown = unknown;
}

// ---- O6 ----

/// O6 shader: the per-k quotient, running cap and limit fold of
/// `u64x64_o6_c`. `cap0` is the initial cap (the X-uniform staff cap).
struct O6Sh<'a> {
    src: &'a Source,
    d: &'a [u16; 64],
    staff_raw: P16,
    cap0: P16,
}

impl Shader for O6Sh<'_> {
    type Acc = (P16, u64, U8x64, U8x64, u64);
    #[inline(always)]
    fn init(&self) -> Self::Acc {
        (self.cap0, 0u64, U8x64::splat(0), U8x64::splat(0), 0u64)
    }
    #[inline(always)]
    fn cell(&self, y: usize, s: Row, acc: Self::Acc) -> Self::Acc {
        let (cap, missing, first_miss, lim, limset) = acc;
        let p = self.src.present[y];
        let newm = !p & !missing;
        let first_miss = U8x64::mask_blend(newm, first_miss, U8x64::splat(y as u8));
        let missing = missing | !p;

        let q = quot8(&s, U16x32::splat(self.d[y]));
        let lt = join(cap[0].cmpgt_mask(q[0]), cap[1].cmpgt_mask(q[1]));
        let eq = join(cap[0].cmpeq_mask(q[0]), cap[1].cmpeq_mask(q[1]));
        let upd = lt | (eq & !limset);
        let lim = U8x64::mask_blend(upd, lim, U8x64::splat(y as u8));
        let limset = limset | upd;
        let cap = [cap[0].simd_min(q[0]), cap[1].simd_min(q[1])];
        (cap, missing, first_miss, lim, limset)
    }
}

/// O6 shader with the per-y quotient chosen from the Y uniform `d[y]`:
/// a power of two takes `min(s >> p, 255)`, anything else takes `quot8`.
/// Every other step is identical to `O6Sh`.
struct O6CoordSh<'a> {
    src: &'a Source,
    d: &'a [u16; 64],
    staff_raw: P16,
    cap0: P16,
}

impl Shader for O6CoordSh<'_> {
    type Acc = (P16, u64, U8x64, U8x64, u64);
    #[inline(always)]
    fn init(&self) -> Self::Acc {
        (self.cap0, 0u64, U8x64::splat(0), U8x64::splat(0), 0u64)
    }
    #[inline(always)]
    fn cell(&self, y: usize, s: Row, acc: Self::Acc) -> Self::Acc {
        let (cap, missing, first_miss, lim, limset) = acc;
        let p = self.src.present[y];
        let newm = !p & !missing;
        let first_miss = U8x64::mask_blend(newm, first_miss, U8x64::splat(y as u8));
        let missing = missing | !p;

        let dy = self.d[y];
        let q = if dy.is_power_of_two() {
            // F3: only a power of two takes the shift path.
            let sh = dy.trailing_zeros();
            let m = U16x32::splat(255);
            [s[0].shr(sh).simd_min(m), s[1].shr(sh).simd_min(m)]
        } else {
            quot8(&s, U16x32::splat(dy))
        };
        let lt = join(cap[0].cmpgt_mask(q[0]), cap[1].cmpgt_mask(q[1]));
        let eq = join(cap[0].cmpeq_mask(q[0]), cap[1].cmpeq_mask(q[1]));
        let upd = lt | (eq & !limset);
        let lim = U8x64::mask_blend(upd, lim, U8x64::splat(y as u8));
        let limset = limset | upd;
        let cap = [cap[0].simd_min(q[0]), cap[1].simd_min(q[1])];
        (cap, missing, first_miss, lim, limset)
    }
}

/// Finish shared by both O6 kernels (identical to `u64x64_o6_c`).
#[inline(always)]
fn o6_finish(src: &Source, staff_raw: P16, acc: (P16, u64, U8x64, U8x64, u64), out: &mut O6Out) {
    let (cap, missing, first_miss, lim, limset) = acc;
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

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_f(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let qmax = U16x32::splat(QMAX as u16);
    let cap0: P16 = [staff_raw[0].simd_min(qmax), staff_raw[1].simd_min(qmax)];
    let sh = O6Sh {
        src,
        d: &params.recipe.d,
        staff_raw,
        cap0,
    };
    let acc = dispatch(&src.stock, params.recipe.demanded(), &sh);
    o6_finish(src, staff_raw, acc, out);
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_f_lut(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    // F2: the table entry is min(s*u, QMAX), not s*u.
    let mut cap8 = [0u8; 256];
    for s in 0..256usize {
        cap8[s] = (s as u32 * params.u as u32).min(QMAX) as u8;
    }
    let mut c8 = [0u8; 64];
    ndarray::simd::lut_u8(&src.staff, &cap8, &mut c8);
    let cap0 = widen_u8(U8x64::from_slice(&c8));
    let sh = O6Sh {
        src,
        d: &params.recipe.d,
        staff_raw,
        cap0,
    };
    let acc = dispatch(&src.stock, params.recipe.demanded(), &sh);
    o6_finish(src, staff_raw, acc, out);
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_f_coord(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let qmax = U16x32::splat(QMAX as u16);
    let cap0: P16 = [staff_raw[0].simd_min(qmax), staff_raw[1].simd_min(qmax)];
    let sh = O6CoordSh {
        src,
        d: &params.recipe.d,
        staff_raw,
        cap0,
    };
    let acc = dispatch(&src.stock, params.recipe.demanded(), &sh);
    o6_finish(src, staff_raw, acc, out);
}

// ---- entries ----

pub fn entries() -> Vec<Entry> {
    let e = |op, variant, kernel| Entry {
        op,
        route: "f",
        variant,
        class: Class::Simd,
        kernel,
    };
    vec![
        e("O2", "", Kernel::O2(u64x64_o2_f)),
        e("O3", "", Kernel::O3(u64x64_o3_f)),
        e("O6", "", Kernel::O6(u64x64_o6_f)),
        e("O6", "lut", Kernel::O6(u64x64_o6_f_lut)),
        e("O6", "coord", Kernel::O6(u64x64_o6_f_coord)),
    ]
}
