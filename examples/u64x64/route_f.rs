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
use ndarray::simd::{U16x32, U64x8, U8x64};

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

// ---- separated cell evaluation (D-U64X64-SHADER-F-0) ----
//
// Each op is a `cell` pass that writes a caller-owned grid and a `reduce`
// pass that folds it. Arithmetic is copied from route C; only the split
// is new.

/// Copy of route C's private `carry_after`: carry out of `low + p = sum`.
#[inline(always)]
fn carry_after(low: U16x32, p: U16x32, sum: U16x32) -> U16x32 {
    ((low & p) | ((low | p) & !sum)).shr(15)
}

/// Copy of route C's private `ones16`: row mask -> 1 / 0 per u16 lane.
#[inline(always)]
fn ones16(m: u64) -> P16 {
    widen_u8(U8x64::mask_blend(m, U8x64::splat(0), U8x64::splat(1)))
}

/// Copy of route C's private `Col16`: `low + 2^16 * high` per lane.
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

    #[inline(always)]
    fn add(&mut self, p: U16x32) {
        let sum = self.low + p;
        self.high += carry_after(self.low, p, sum);
        self.low = sum;
    }
}

// ---- O1 ----

/// O1 cell: `I[y] = A[y] & B[y] & live`.
#[inline(always)]
pub(crate) fn fs_o1_cell(a: &BitTile, b: &BitTile, live: u64, grid: &mut BitTile) {
    let lv = U64x8::splat(live);
    for g in 0..8 {
        let x = U64x8::from_slice(&a[g * 8..g * 8 + 8]) & U64x8::from_slice(&b[g * 8..g * 8 + 8]) & lv;
        x.copy_to_slice(&mut grid[g * 8..g * 8 + 8]);
    }
}

/// O1 reduction: per-y popcount, per-x vertical count, total.
#[inline(always)]
pub(crate) fn fs_o1_reduce(grid: &BitTile, out: &mut O1Out) {
    let zero = U8x64::splat(0);
    let one = U8x64::splat(1);
    let mut acc = U8x64::splat(0);
    let mut total = 0u64;
    for g in 0..8 {
        let x = U64x8::from_slice(&grid[g * 8..g * 8 + 8]);
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

#[no_mangle]
#[inline(never)]
pub fn u64x64_o1_fs(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    let mut grid: BitTile = [0; 64];
    fs_o1_cell(a, b, live, &mut grid);
    fs_o1_reduce(&grid, out);
}

// ---- O2 ----

/// O2 cell: `ge[y] = !(t[y] > S[y])` for demanded y; `t > 65535` never.
/// Rows not demanded are not written.
#[inline(always)]
pub(crate) fn fs_o2_cell(bound: &Bound<'_>, t: &[u32; 64], dem: u64, ge: &mut [u64; 64]) {
    let mut dm = dem;
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let tk = t[y];
        ge[y] = if tk > 65535 {
            0
        } else {
            let tv = U16x32::splat(tk as u16);
            let v = route_c::load16(&bound.src.stock[y]);
            !join(tv.cmpgt_mask(v[0]), tv.cmpgt_mask(v[1]))
        };
    }
}

/// O2 reduction: AND over the demanded rows; `u64::MAX` when none.
#[inline(always)]
pub(crate) fn fs_o2_reduce(ge: &[u64; 64], dem: u64) -> u64 {
    let mut all = u64::MAX;
    let mut dm = dem;
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        all &= ge[y];
    }
    all
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o2_fs(bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    out.ge = [0; 64];
    fs_o2_cell(bound, t, demanded, &mut out.ge);
    out.all_ge = fs_o2_reduce(&out.ge, demanded);
}

// ---- O3 ----

/// O3 cell: `grid[y] = S[y] | !lane16(P[y])` for demanded y only.
#[inline(always)]
pub(crate) fn fs_o3_cell(bound: &Bound<'_>, dem: u64, grid: &mut ValueTile<u16>) {
    let src = bound.src;
    let mut dm = dem;
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let pm = lane16(src.present[y]);
        let v = route_c::load16(&src.stock[y]);
        let vs: P16 = [v[0] | !pm[0], v[1] | !pm[1]];
        vs[0].copy_to_slice(&mut grid[y][..32]);
        vs[1].copy_to_slice(&mut grid[y][32..]);
    }
}

/// O3 reduction: running min / arg over the cell grid. `have` and `unknown`
/// read the presence words again; that is the stated dependency.
#[inline(always)]
pub(crate) fn fs_o3_reduce(grid: &ValueTile<u16>, bound: &Bound<'_>, dem: u64, out: &mut O3Out) {
    let src = bound.src;
    let ff = U16x32::splat(0xFFFF);
    let mut m: P16 = [ff, ff];
    let mut arg = U8x64::splat(0xFF);
    let mut have = 0u64;
    let mut unknown = 0u64;
    let mut dm = dem;
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let p = src.present[y];
        unknown |= !p;
        let vs = route_c::load16(&grid[y]);
        // F5: only a strictly smaller value moves arg.
        let lt = join(m[0].cmpgt_mask(vs[0]), m[1].cmpgt_mask(vs[1]));
        let upd = lt | (p & !have);
        arg = U8x64::mask_blend(upd, arg, U8x64::splat(y as u8));
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

#[no_mangle]
#[inline(never)]
pub fn u64x64_o3_fs(bound: &Bound<'_>, demanded: u64, out: &mut O3Out) {
    let mut grid: ValueTile<u16> = [[0; 64]; 64];
    fs_o3_cell(bound, demanded, &mut grid);
    fs_o3_reduce(&grid, bound, demanded, out);
}

// ---- O4 ----

/// O4 cell: for each y with `w[y] != 0`, `lo[y] = C[y] * (w & 0xFF)` and, when
/// `w >> 8 != 0`, `hi[y] = C[y] * (w >> 8)` (each product <= 255*255).
#[inline(always)]
pub(crate) fn fs_o4_cell(cost: &ValueTile<u8>, w: &[u16; 64], lo: &mut ValueTile<u16>, hi: &mut ValueTile<u16>) {
    for y in 0..64 {
        let wk = w[y];
        if wk == 0 {
            continue;
        }
        let c = widen_u8(U8x64::from_slice(&cost[y]));
        let wl = U16x32::splat(wk & 0xFF);
        c[0].mullo(wl).copy_to_slice(&mut lo[y][..32]);
        c[1].mullo(wl).copy_to_slice(&mut lo[y][32..]);
        if wk >> 8 != 0 {
            let wh = U16x32::splat(wk >> 8);
            c[0].mullo(wh).copy_to_slice(&mut hi[y][..32]);
            c[1].mullo(wh).copy_to_slice(&mut hi[y][32..]);
        }
    }
}

/// O4 reduction: `Col16` carry sums over y, then the u32 finish.
#[inline(always)]
pub(crate) fn fs_o4_reduce(lo: &ValueTile<u16>, hi: &ValueTile<u16>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    let mut lc = [Col16::zero(), Col16::zero()];
    let mut hc = [Col16::zero(), Col16::zero()];
    for y in 0..64 {
        let wk = w[y];
        if wk == 0 {
            continue;
        }
        let l = route_c::load16(&lo[y]);
        for h in 0..2 {
            lc[h].add(l[h]);
        }
        if wk >> 8 != 0 {
            let u = route_c::load16(&hi[y]);
            for h in 0..2 {
                hc[h].add(u[h]);
            }
        }
    }
    let mut total = 0u64;
    for h in 0..2 {
        let ll = lc[h].low.to_array();
        let lh = lc[h].high.to_array();
        let hl = hc[h].low.to_array();
        let hh = hc[h].high.to_array();
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

#[no_mangle]
#[inline(never)]
pub fn u64x64_o4_fs(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    let mut lo: ValueTile<u16> = [[0; 64]; 64];
    let mut hi: ValueTile<u16> = [[0; 64]; 64];
    fs_o4_cell(cost, w, &mut lo, &mut hi);
    fs_o4_reduce(&lo, &hi, w, live, out);
}

// ---- O5 ----

/// O5 cell: `sum = row + (ones(m) << b)`; returns the carry-lane mask.
#[inline(always)]
pub(crate) fn fs_o5_cell(row: &[u16; 64], b: u32, m: u64, sum: &mut [u16; 64]) -> u64 {
    let cnt = route_c::load16(row);
    let ones = ones16(m);
    let inc: P16 = [ones[0].shl(b), ones[1].shl(b)];
    let s: P16 = [cnt[0] + inc[0], cnt[1] + inc[1]];
    let one = U16x32::splat(1);
    let lanes =
        join(carry_after(cnt[0], inc[0], s[0]).cmpeq_mask(one), carry_after(cnt[1], inc[1], s[1]).cmpeq_mask(one));
    s[0].copy_to_slice(&mut sum[..32]);
    s[1].copy_to_slice(&mut sum[32..]);
    lanes
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o5_fs(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let mut sum = [0u16; 64];
    let lanes = fs_o5_cell(&counters[k], u.b as u32, u.m, &mut sum);
    if lanes != 0 {
        return Err(Refusal::Overflow { lanes });
    }
    // F6: write back only if no lane carries.
    counters[k] = sum;
    Ok(())
}

// ---- O6 ----

/// Initial cap `min(staff * u, QMAX)` by C's `mullo` + `simd_min`.
#[inline(always)]
pub(crate) fn fs_o6_cap0(bound: &Bound<'_>, params: &Params) -> P16 {
    let staff = widen_u8(U8x64::from_slice(&bound.src.staff));
    let uv = U16x32::splat(params.u as u16);
    let raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let qmax = U16x32::splat(QMAX as u16);
    [raw[0].simd_min(qmax), raw[1].simd_min(qmax)]
}

/// O6 cell: `grid[y] = quot8(S[y], d[y])` for demanded y only.
#[inline(always)]
pub(crate) fn fs_o6_cell(bound: &Bound<'_>, params: &Params, grid: &mut ValueTile<u16>) {
    let src = bound.src;
    let mut dm = params.recipe.demanded();
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let s = route_c::load16(&src.stock[y]);
        let q = quot8(&s, U16x32::splat(params.recipe.d[y]));
        q[0].copy_to_slice(&mut grid[y][..32]);
        q[1].copy_to_slice(&mut grid[y][32..]);
    }
}

/// O6 cell with the per-y class: a power-of-two `d` takes `min(s >> p, 255)`,
/// anything else takes `quot8` (as `O6CoordSh`).
#[inline(always)]
pub(crate) fn fs_o6_cell_coord(bound: &Bound<'_>, params: &Params, grid: &mut ValueTile<u16>) {
    let src = bound.src;
    let mut dm = params.recipe.demanded();
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let s = route_c::load16(&src.stock[y]);
        let dy = params.recipe.d[y];
        let q = if dy.is_power_of_two() {
            // F3: only a power of two takes the shift path.
            let sh = dy.trailing_zeros();
            let m = U16x32::splat(255);
            [s[0].shr(sh).simd_min(m), s[1].shr(sh).simd_min(m)]
        } else {
            quot8(&s, U16x32::splat(dy))
        };
        q[0].copy_to_slice(&mut grid[y][..32]);
        q[1].copy_to_slice(&mut grid[y][32..]);
    }
}

/// O6 reduction: running cap with the limit tie rule, presence fold, finish.
/// `staff_raw` is recomputed here for the STAFF test.
#[inline(always)]
pub(crate) fn fs_o6_reduce(grid: &ValueTile<u16>, bound: &Bound<'_>, params: &Params, cap0: P16, out: &mut O6Out) {
    let src = bound.src;
    let staff = widen_u8(U8x64::from_slice(&src.staff));
    let uv = U16x32::splat(params.u as u16);
    let staff_raw: P16 = [staff[0].mullo(uv), staff[1].mullo(uv)];
    let mut cap = cap0;
    let mut missing = 0u64;
    let mut first_miss = U8x64::splat(0);
    let mut lim = U8x64::splat(0);
    let mut limset = 0u64;
    let mut dm = params.recipe.demanded();
    while dm != 0 {
        let y = dm.trailing_zeros() as usize;
        dm &= dm - 1;
        let p = src.present[y];
        let newm = !p & !missing;
        first_miss = U8x64::mask_blend(newm, first_miss, U8x64::splat(y as u8));
        missing |= !p;
        let q = route_c::load16(&grid[y]);
        let lt = join(cap[0].cmpgt_mask(q[0]), cap[1].cmpgt_mask(q[1]));
        let eq = join(cap[0].cmpeq_mask(q[0]), cap[1].cmpeq_mask(q[1]));
        let upd = lt | (eq & !limset);
        lim = U8x64::mask_blend(upd, lim, U8x64::splat(y as u8));
        limset |= upd;
        cap = [cap[0].simd_min(q[0]), cap[1].simd_min(q[1])];
    }
    o6_finish(src, staff_raw, (cap, missing, first_miss, lim, limset), out);
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_fs(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let mut grid: ValueTile<u16> = [[0; 64]; 64];
    fs_o6_cell(bound, params, &mut grid);
    let cap0 = fs_o6_cap0(bound, params);
    fs_o6_reduce(&grid, bound, params, cap0, out);
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_fs_coord(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let mut grid: ValueTile<u16> = [[0; 64]; 64];
    fs_o6_cell_coord(bound, params, &mut grid);
    let cap0 = fs_o6_cap0(bound, params);
    fs_o6_reduce(&grid, bound, params, cap0, out);
}

/// A table of `min(s * u, QMAX)` for every staff byte, prepared for one `u`.
pub(crate) struct CapLut {
    u: u32,
    table: [u8; 256],
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LutRefusal {
    ParamMismatch { prepared: u32, asked: u32 },
}

impl CapLut {
    pub(crate) fn prepare(u: u32) -> CapLut {
        let mut table = [0u8; 256];
        for s in 0..256usize {
            table[s] = (s as u32 * u).min(QMAX) as u8;
        }
        CapLut { u, table }
    }

    pub(crate) fn cap0(&self, staff: &[u8; 64], u: u32) -> Result<P16, LutRefusal> {
        // F4: a table prepared for one u is never used for another.
        if u != self.u {
            return Err(LutRefusal::ParamMismatch {
                prepared: self.u,
                asked: u,
            });
        }
        let mut c8 = [0u8; 64];
        ndarray::simd::lut_u8(staff, &self.table, &mut c8);
        Ok(widen_u8(U8x64::from_slice(&c8)))
    }
}

#[no_mangle]
#[inline(never)]
pub fn u64x64_o6_fs_lut(bound: &Bound<'_>, params: &Params, out: &mut O6Out) {
    let mut grid: ValueTile<u16> = [[0; 64]; 64];
    fs_o6_cell(bound, params, &mut grid);
    let lut = CapLut::prepare(params.u as u32);
    let cap0 = lut
        .cap0(&bound.src.staff, params.u as u32)
        .expect("lut prepared for this u");
    fs_o6_reduce(&grid, bound, params, cap0, out);
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn cap_lut_equals_mullo_min_for_every_u_and_staff() {
        for u in 1..=255u32 {
            let lut = CapLut::prepare(u);
            for s in 0..256u32 {
                assert_eq!(lut.table[s as usize] as u32, (s * u).min(128), "u={u} s={s}");
            }
        }
    }

    #[test]
    fn cap_lut_prepared_for_one_u_refuses_another() {
        let lut = CapLut::prepare(3);
        assert_eq!(lut.cap0(&[1; 64], 4).err(), Some(LutRefusal::ParamMismatch { prepared: 3, asked: 4 }));
        assert!(lut.cap0(&[1; 64], 3).is_ok());
    }

    const SENT: u16 = 0xA5A5;
    const DEM: [usize; 3] = [3, 17, 40];

    fn dem_mask() -> u64 {
        DEM.iter().fold(0u64, |m, &y| m | (1u64 << y))
    }

    #[test]
    fn split_cell_writes_only_demanded_rows() {
        let src = random_source(11);
        let bound = bind(&src, &Expect::of(&src)).expect("source binds");
        let recipe = Recipe::from_pairs(&[(3, 2), (17, 3), (40, 1)]).unwrap();
        let params = Params::new(recipe, 2, 10, [0; 64]).unwrap();
        assert_eq!(recipe.demanded(), dem_mask());

        let mut grid: ValueTile<u16> = [[SENT; 64]; 64];
        fs_o6_cell(&bound, &params, &mut grid);
        for y in 0..64 {
            if DEM.contains(&y) {
                for r in 0..64 {
                    let want = (src.stock[y][r] / recipe.d[y]).min(255);
                    assert_eq!(grid[y][r], want, "o6 y={y} r={r}");
                }
            } else {
                assert!(grid[y].iter().all(|&v| v == SENT), "o6 row {y} was written");
            }
        }

        let mut grid: ValueTile<u16> = [[SENT; 64]; 64];
        fs_o3_cell(&bound, dem_mask(), &mut grid);
        for y in 0..64 {
            if DEM.contains(&y) {
                for r in 0..64 {
                    let absent = (src.present[y] >> r) & 1 == 0;
                    let want = if absent { 0xFFFF } else { src.stock[y][r] };
                    assert_eq!(grid[y][r], want, "o3 y={y} r={r}");
                }
            } else {
                assert!(grid[y].iter().all(|&v| v == SENT), "o3 row {y} was written");
            }
        }
    }
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
        e("O1", "split", Kernel::O1(u64x64_o1_fs)),
        e("O2", "split", Kernel::O2(u64x64_o2_fs)),
        e("O3", "split", Kernel::O3(u64x64_o3_fs)),
        e("O4", "split", Kernel::O4(u64x64_o4_fs)),
        e("O5", "split", Kernel::O5(u64x64_o5_fs)),
        e("O6", "split", Kernel::O6(u64x64_o6_fs)),
        e("O6", "split_coord", Kernel::O6(u64x64_o6_fs_coord)),
        e("O6", "split_lut", Kernel::O6(u64x64_o6_fs_lut)),
    ]
}
