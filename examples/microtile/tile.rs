//! D-SIMD-MICRO-MACRO-1, step 1: one 128-bit microtile carrying several
//! isolated arithmetic fields, in four realizations that are checked against
//! one unpacked oracle.
//!
//! Layouts (`Packing`):
//! - `U24x5`: five 24-bit fields in bits 0..120, 8 metadata bits 120..128.
//! - `U9x14`: fourteen 9-bit fields in bits 0..126, 2 spare bits 126..128.
//!
//! Field `j` occupies bits `j*W .. j*W+W`. Its top bit is the sign bit when
//! the field is read as `I24`/`I9`. Bits outside the fields (`META`) are never
//! touched by arithmetic: every result carries the LEFT operand's metadata.
//!
//! Realizations:
//! - `u128` (`add`, `sub`, ...): the whole tile in one 128-bit integer. This is
//!   the "one register" form; rustc lowers it to a two-word add/adc pair.
//! - `soa` (`Soa*`): 8 tiles at a time over `ndarray::simd::U64x8`, with the
//!   tile split into a low and a high 64-bit plane. Fields that straddle bit 64
//!   (`U24` field 2, `U9` field 7) get their carry/borrow from an explicit
//!   64-bit carry-out, computed with lane-wise logic only.
//! - `widened`: field-major `u32` lanes over `ndarray::simd::U32x16`.
//! - `planes`: bit planes over 64 tiles, added with `ndarray::simd::full_add_mask`.
//!
//! Lab fixture only: no public type, no production caller.

use ndarray::simd::{full_add_mask, U32x16, U64x8};

/// A packed layout of `N` fields of width `W` in 128 bits.
pub trait Packing {
    const W: u32;
    const N: u32;
    /// One field's value mask, `2^W - 1`.
    const VMASK: u128 = (1u128 << Self::W) - 1;
    /// All field bits.
    const FIELDS: u128 = {
        let mut m = 0u128;
        let mut j = 0;
        while j < Self::N {
            m |= Self::VMASK << (j * Self::W);
            j += 1;
        }
        m
    };
    /// The top (sign) bit of every field.
    const HIGH: u128 = {
        let mut m = 0u128;
        let mut j = 0;
        while j < Self::N {
            m |= 1u128 << (j * Self::W + Self::W - 1);
            j += 1;
        }
        m
    };
    /// Field bits below each field's top bit.
    const LOW: u128 = Self::FIELDS & !Self::HIGH;
    /// Bits outside the fields: metadata (`U24x5`) or spare (`U9x14`).
    const META: u128 = !Self::FIELDS;
}

/// Five 24-bit fields + 8 metadata bits.
pub struct U24x5;
impl Packing for U24x5 {
    const W: u32 = 24;
    const N: u32 = 5;
}

/// Fourteen 9-bit fields + 2 spare bits.
pub struct U9x14;
impl Packing for U9x14 {
    const W: u32 = 9;
    const N: u32 = 14;
}

// ── Oracle: unpacked per-field arithmetic ────────────────────────────────

/// Field `j` as an unsigned value.
pub fn get<P: Packing>(t: u128, j: u32) -> u32 {
    ((t >> (j * P::W)) & P::VMASK) as u32
}

/// Field `j` sign-extended (`I24` / `I9`).
pub fn get_signed<P: Packing>(t: u128, j: u32) -> i32 {
    let s = 32 - P::W;
    ((get::<P>(t, j) << s) as i32) >> s
}

/// Build a tile from field values (each masked to `W` bits) and metadata.
pub fn pack<P: Packing>(fields: &[u32], meta: u128) -> u128 {
    assert_eq!(fields.len(), P::N as usize);
    let mut t = meta & P::META;
    for (j, &f) in fields.iter().enumerate() {
        t |= (f as u128 & P::VMASK) << (j as u32 * P::W);
    }
    t
}

/// Oracle add: per-field wrapping sum, per-field unsigned carry-out and
/// signed overflow, metadata from `a`.
pub fn oracle_add<P: Packing>(a: u128, b: u128) -> (u128, u32, u32) {
    let m = (1u64 << P::W) - 1;
    let (mut fields, mut carry, mut ovf) = (Vec::new(), 0u32, 0u32);
    for j in 0..P::N {
        let s = get::<P>(a, j) as u64 + get::<P>(b, j) as u64;
        fields.push((s & m) as u32);
        carry |= ((s > m) as u32) << j;
        let ss = get_signed::<P>(a, j) as i64 + get_signed::<P>(b, j) as i64;
        let lim = 1i64 << (P::W - 1);
        ovf |= ((ss >= lim || ss < -lim) as u32) << j;
    }
    (pack::<P>(&fields, a), carry, ovf)
}

/// Oracle sub: per-field wrapping difference and unsigned borrow-out.
pub fn oracle_sub<P: Packing>(a: u128, b: u128) -> (u128, u32) {
    let m = (1u64 << P::W) - 1;
    let (mut fields, mut borrow) = (Vec::new(), 0u32);
    for j in 0..P::N {
        let (x, y) = (get::<P>(a, j) as u64, get::<P>(b, j) as u64);
        fields.push((x.wrapping_sub(y) & m) as u32);
        borrow |= ((x < y) as u32) << j;
    }
    (pack::<P>(&fields, a), borrow)
}

/// Oracle stencil inside one tile: `f[j-1] + 2 f[j] + f[j+1]` mod `2^W`,
/// zero beyond the first and last field, metadata kept.
pub fn oracle_stencil<P: Packing>(t: u128) -> u128 {
    let n = P::N as i64;
    let f = |j: i64| {
        if j < 0 || j >= n {
            0u64
        } else {
            get::<P>(t, j as u32) as u64
        }
    };
    let m = (1u64 << P::W) - 1;
    let fields: Vec<u32> = (0..n)
        .map(|j| ((f(j - 1) + 2 * f(j) + f(j + 1)) & m) as u32)
        .collect();
    pack::<P>(&fields, t)
}

/// Oracle fold: exact per-field sums, unsigned and signed.
pub fn oracle_fold<P: Packing>(tiles: &[u128]) -> (Vec<u64>, Vec<i64>) {
    let mut u = vec![0u64; P::N as usize];
    let mut s = vec![0i64; P::N as usize];
    for &t in tiles {
        for j in 0..P::N {
            u[j as usize] += get::<P>(t, j) as u64;
            s[j as usize] += get_signed::<P>(t, j) as i64;
        }
    }
    (u, s)
}

/// Collect a per-field flag word (bit `j` = field `j`) from top-bit positions.
pub fn flags<P: Packing>(at_high: u128) -> u32 {
    (0..P::N).fold(0, |acc, j| acc | ((at_high >> (j * P::W + P::W - 1) & 1) as u32) << j)
}

// ── Realization 1: one u128 ──────────────────────────────────────────────

/// Field-isolated wrapping add, metadata from `a`. Carries never leave a
/// field: the top bit of every addend is cleared, so a field's carry lands
/// in its own top bit, which is then fixed by XOR.
#[inline]
pub fn add<P: Packing>(a: u128, b: u128) -> u128 {
    let s = (a & P::LOW) + (b & P::LOW);
    ((s ^ ((a ^ b) & P::HIGH)) & P::FIELDS) | (a & P::META)
}

/// Per-field unsigned carry-out of `a + b` given its result `r`, at top-bit
/// positions: the majority of `aH`, `bH` and the carry into `H`.
#[inline]
pub fn carry_out<P: Packing>(a: u128, b: u128, r: u128) -> u128 {
    ((a & b) | ((a | b) & !r)) & P::HIGH
}

/// Per-field signed overflow of `a + b` (I24/I9), at top-bit positions.
#[inline]
pub fn signed_overflow<P: Packing>(a: u128, b: u128, r: u128) -> u128 {
    (!(a ^ b) & (a ^ r)) & P::HIGH
}

/// Field-isolated wrapping sub, metadata from `a`. Setting each field's top
/// bit in `a` stops every borrow at its own field.
#[inline]
pub fn sub<P: Packing>(a: u128, b: u128) -> u128 {
    let d = (a | P::HIGH) - (b & P::LOW);
    ((d ^ ((a ^ !b) & P::HIGH)) & P::FIELDS) | (a & P::META)
}

/// Per-field unsigned borrow-out of `a - b` given its result `r`.
#[inline]
pub fn borrow_out<P: Packing>(a: u128, b: u128, r: u128) -> u128 {
    ((!a & b) | (!(a ^ b) & r)) & P::HIGH
}

/// Intra-tile stencil `f[j-1] + 2 f[j] + f[j+1]`: neighbours are moved by a
/// whole-tile shift of `W` bits and masked back to the field bits.
#[inline]
pub fn stencil<P: Packing>(t: u128) -> u128 {
    let f = t & P::FIELDS;
    // f[j-1] into field j. The last field lands outside the fields; `add`
    // masks its right operand to the field bits, so no mask is needed here.
    let left = f << P::W;
    let right = f >> P::W; // f[j+1] into field j
    let two = add::<P>(f, f);
    let r = add::<P>(add::<P>(two, left), right);
    (r & P::FIELDS) | (t & P::META)
}

// ── Realization 2: SoA low/high planes over U64x8 ────────────────────────

/// The `u128` constants split into low and high 64-bit words.
const fn lo(m: u128) -> u64 {
    m as u64
}
const fn hi(m: u128) -> u64 {
    (m >> 64) as u64
}

/// Carry out of bit 63 of `x + y = s`, as 0 or 1 per lane.
#[inline]
fn carry63(x: U64x8, y: U64x8, s: U64x8) -> U64x8 {
    ((x & y) | ((x | y) & !s)) >> U64x8::splat(63)
}

/// Borrow out of bit 63 of `x - y = d`, as 0 or 1 per lane.
#[inline]
fn borrow63(x: U64x8, y: U64x8, d: U64x8) -> U64x8 {
    ((!x & y) | ((!x | y) & d)) >> U64x8::splat(63)
}

/// Field-isolated add of 8 tiles. Returns the result planes and the carry-out
/// planes at top-bit positions.
#[inline]
pub fn soa_add<P: Packing>(alo: U64x8, ahi: U64x8, blo: U64x8, bhi: U64x8) -> [U64x8; 4] {
    let (ll, lh) = (U64x8::splat(lo(P::LOW)), U64x8::splat(hi(P::LOW)));
    let (hl, hh) = (U64x8::splat(lo(P::HIGH)), U64x8::splat(hi(P::HIGH)));
    let (fl, fh) = (U64x8::splat(lo(P::FIELDS)), U64x8::splat(hi(P::FIELDS)));
    let (ml, mh) = (U64x8::splat(lo(P::META)), U64x8::splat(hi(P::META)));
    let (xl, yl) = (alo & ll, blo & ll);
    let sl = xl + yl;
    let sh = (ahi & lh) + (bhi & lh) + carry63(xl, yl, sl);
    let rl = ((sl ^ ((alo ^ blo) & hl)) & fl) | (alo & ml);
    let rh = ((sh ^ ((ahi ^ bhi) & hh)) & fh) | (ahi & mh);
    let cl = ((alo & blo) | ((alo | blo) & !rl)) & hl;
    let ch = ((ahi & bhi) | ((ahi | bhi) & !rh)) & hh;
    [rl, rh, cl, ch]
}

/// Field-isolated sub of 8 tiles. Returns the result planes.
#[inline]
pub fn soa_sub<P: Packing>(alo: U64x8, ahi: U64x8, blo: U64x8, bhi: U64x8) -> [U64x8; 2] {
    let (ll, lh) = (U64x8::splat(lo(P::LOW)), U64x8::splat(hi(P::LOW)));
    let (hl, hh) = (U64x8::splat(lo(P::HIGH)), U64x8::splat(hi(P::HIGH)));
    let (fl, fh) = (U64x8::splat(lo(P::FIELDS)), U64x8::splat(hi(P::FIELDS)));
    let (ml, mh) = (U64x8::splat(lo(P::META)), U64x8::splat(hi(P::META)));
    let (xl, yl) = (alo | hl, blo & ll);
    let dl = xl - yl;
    let dh = (ahi | hh) - (bhi & lh) - borrow63(xl, yl, dl);
    let rl = ((dl ^ ((alo ^ !blo) & hl)) & fl) | (alo & ml);
    let rh = ((dh ^ ((ahi ^ !bhi) & hh)) & fh) | (ahi & mh);
    [rl, rh]
}

/// Borrowed SoA tiles: tile `t` is `lo[t] | hi[t] << 64`.
pub struct Soa<'a> {
    pub lo: &'a [u64],
    pub hi: &'a [u64],
}

impl Soa<'_> {
    pub fn len(&self) -> usize {
        self.lo.len()
    }
    pub fn tile(&self, t: usize) -> u128 {
        self.lo[t] as u128 | (self.hi[t] as u128) << 64
    }
}

/// Split tiles into owned SoA planes (setup only, not timed).
pub fn split(tiles: &[u128]) -> (Vec<u64>, Vec<u64>) {
    (tiles.iter().map(|&t| t as u64).collect(), tiles.iter().map(|&t| (t >> 64) as u64).collect())
}

/// `out = a + b` field-wise over whole slices; the tail uses the u128 form.
pub fn soa_add_slices<P: Packing>(a: &Soa, b: &Soa, out_lo: &mut [u64], out_hi: &mut [u64]) {
    let n = a.len();
    let body = n - n % 8;
    for i in (0..body).step_by(8) {
        let [rl, rh, _, _] = soa_add::<P>(
            U64x8::from_slice(&a.lo[i..]),
            U64x8::from_slice(&a.hi[i..]),
            U64x8::from_slice(&b.lo[i..]),
            U64x8::from_slice(&b.hi[i..]),
        );
        rl.copy_to_slice(&mut out_lo[i..]);
        rh.copy_to_slice(&mut out_hi[i..]);
    }
    for t in body..n {
        let r = add::<P>(a.tile(t), b.tile(t));
        out_lo[t] = r as u64;
        out_hi[t] = (r >> 64) as u64;
    }
}

/// `out = a - b` field-wise over whole slices.
pub fn soa_sub_slices<P: Packing>(a: &Soa, b: &Soa, out_lo: &mut [u64], out_hi: &mut [u64]) {
    let n = a.len();
    let body = n - n % 8;
    for i in (0..body).step_by(8) {
        let [rl, rh] = soa_sub::<P>(
            U64x8::from_slice(&a.lo[i..]),
            U64x8::from_slice(&a.hi[i..]),
            U64x8::from_slice(&b.lo[i..]),
            U64x8::from_slice(&b.hi[i..]),
        );
        rl.copy_to_slice(&mut out_lo[i..]);
        rh.copy_to_slice(&mut out_hi[i..]);
    }
    for t in body..n {
        let r = sub::<P>(a.tile(t), b.tile(t));
        out_lo[t] = r as u64;
        out_hi[t] = (r >> 64) as u64;
    }
}

/// Exact per-field fold over SoA tiles: unsigned sums and signed sums.
///
/// The signed sum uses `sum(sext f) = sum(f) - 2^W * #(top bit set)`, so both
/// come from one pass: one masked field read and one top-bit read per field.
pub fn soa_fold<P: Packing>(a: &Soa) -> (Vec<u64>, Vec<i64>) {
    let (n, nf) = (a.len(), P::N as usize);
    let body = n - n % 8;
    let vm = U64x8::splat(P::VMASK as u64);
    let one = U64x8::splat(1);
    let top = U64x8::splat((P::W - 1) as u64);
    // One pass over the tiles, one accumulator pair per field.
    let mut acc = [U64x8::splat(0); 16];
    let mut cnt = [U64x8::splat(0); 16];
    for i in (0..body).step_by(8) {
        let l = U64x8::from_slice(&a.lo[i..]);
        let h = U64x8::from_slice(&a.hi[i..]);
        for j in 0..nf {
            let off = j as u32 * P::W;
            let v = if off + P::W <= 64 {
                l >> U64x8::splat(off as u64)
            } else if off >= 64 {
                h >> U64x8::splat((off - 64) as u64)
            } else {
                (l >> U64x8::splat(off as u64)) | (h << U64x8::splat((64 - off) as u64))
            } & vm;
            acc[j] += v;
            cnt[j] += (v >> top) & one;
        }
    }
    let mut u: Vec<u64> = acc[..nf].iter().map(|x| x.reduce_sum()).collect();
    let mut neg: Vec<u64> = cnt[..nf].iter().map(|x| x.reduce_sum()).collect();
    for t in body..n {
        for j in 0..P::N {
            let f = get::<P>(a.tile(t), j);
            u[j as usize] += f as u64;
            neg[j as usize] += (f >> (P::W - 1)) as u64;
        }
    }
    let s = u
        .iter()
        .zip(&neg)
        .map(|(&s, &c)| s as i64 - ((c as i64) << P::W))
        .collect();
    (u, s)
}

// ── Realization 3: widened u32 lanes over U32x16 ─────────────────────────

/// Load 16 lanes. `U32x16::from_slice` is not on every realization (NEON
/// and wasm lack it), `from_array` is.
#[inline]
fn load16(s: &[u32]) -> U32x16 {
    U32x16::from_array(s[..16].try_into().expect("16 lanes"))
}

/// Field-major widened layout: `fields[j][t]` = field `j` of tile `t`.
pub fn widen<P: Packing>(tiles: &[u128]) -> Vec<Vec<u32>> {
    (0..P::N)
        .map(|j| tiles.iter().map(|&t| get::<P>(t, j)).collect())
        .collect()
}

/// `out = (a + b) mod 2^W` per lane; lengths must be a multiple of 16.
///
/// The `mod 2^W` is two shifts, not a mask: on NEON and wasm `U32x16` has
/// only `+`, `^` and the shifts, so `&` would not compile there.
pub fn widened_add<P: Packing>(a: &[u32], b: &[u32], out: &mut [u32]) {
    let k = U32x16::splat(32 - P::W);
    for i in (0..a.len()).step_by(16) {
        let s = (load16(&a[i..]) + load16(&b[i..])) << k >> k;
        s.to_array()
            .iter()
            .zip(&mut out[i..i + 16])
            .for_each(|(&v, o)| *o = v);
    }
}

/// Exact unsigned fold of one widened field. A `u32` lane can absorb
/// `floor((2^32 - 1) / (2^W - 1))` values before it overflows, so the lanes are
/// flushed to `u64` after that many steps.
pub fn widened_fold<P: Packing>(a: &[u32]) -> u64 {
    let block = (u32::MAX / (P::VMASK as u32)) as usize;
    let (mut total, mut acc, mut steps) = (0u64, U32x16::splat(0), 0usize);
    for i in (0..a.len()).step_by(16) {
        // Not `+=`: `U32x16` has no `AddAssign` on NEON or wasm.
        #[allow(clippy::assign_op_pattern)]
        {
            acc = acc + load16(&a[i..]);
        }
        steps += 1;
        if steps == block {
            total += acc.to_array().iter().map(|&x| x as u64).sum::<u64>();
            acc = U32x16::splat(0);
            steps = 0;
        }
    }
    total + acc.to_array().iter().map(|&x| x as u64).sum::<u64>()
}

// ── Realization 4: bit planes over 64 tiles ──────────────────────────────

/// `planes[j][blk * W + k]`: bit `k` of field `j` for the 64 tiles of block
/// `blk`, one bit per tile. A block's `W` planes are contiguous.
pub fn to_planes<P: Packing>(tiles: &[u128]) -> Vec<Vec<u64>> {
    let (w, blocks) = (P::W as usize, tiles.len().div_ceil(64));
    let mut p = vec![vec![0u64; blocks * w]; P::N as usize];
    for (t, &tile) in tiles.iter().enumerate() {
        for j in 0..P::N {
            let f = get::<P>(tile, j);
            for k in 0..w {
                p[j as usize][t / 64 * w + k] |= ((f >> k & 1) as u64) << (t % 64);
            }
        }
    }
    p
}

/// Ripple add of two `W`-plane fields over one 64-tile block; returns the
/// carry-out mask.
#[inline]
pub fn planes_add(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let mut c = 0;
    for k in 0..a.len() {
        let (s, c2) = full_add_mask(a[k], b[k], c);
        out[k] = s;
        c = c2;
    }
    c
}

/// Exact fold of one plane-stored field: `sum = Σ_k 2^k popcount(plane_k)`;
/// the signed sum gives the top plane the weight `-2^(W-1)`.
pub fn planes_fold(planes: &[u64], w: usize) -> (u64, i64) {
    let mut pop = [0u64; 32];
    for blk in planes.chunks_exact(w) {
        for (k, &x) in blk.iter().enumerate() {
            pop[k] += x.count_ones() as u64;
        }
    }
    let mut u = 0u64;
    let mut s = 0i64;
    for (k, &c) in pop[..w].iter().enumerate() {
        u += c << k;
        s += if k + 1 == w {
            -((c as i64) << k)
        } else {
            (c as i64) << k
        };
    }
    (u, s)
}
