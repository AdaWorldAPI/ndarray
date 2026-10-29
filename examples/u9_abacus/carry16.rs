//! D-U9-CARRY16: candidate A' (`U16x32::mullo` products, sign routed by the
//! polarity XOR) with its `f32` detour replaced by a 16-bit place-value
//! column. Each tally lane keeps `low` (wrapping `u16`) and `high` (count of
//! 2^16 units): the represented value is `low + 2^16 * high`. A product
//! `p < 2^16` is added to `low` with a wrapping add; when the add crosses
//! 2^16, one unit moves into `high`. Nothing is widened before it must be.
//!
//! Two ways to find that carry, both plain bit logic on `U16x32` lanes
//! (no masks, no compares into mask registers, no shuffles):
//!
//! * react ([`carry_after`]): add first, then read the carry out of the
//!   finished sum, `msb((low & p) | ((low | p) & !sum))`;
//! * anticipate ([`carry_before`]): before adding, the carry is set exactly
//!   when `p` exceeds the headroom `!low = 65535 - low`, which is the borrow
//!   out of `!low - p`: `msb((low & p) | (!(!low ^ p) & (!low - p)))`.

#![allow(dead_code)]

use crate::kernels::route_a;
use crate::kernels::Block;
use ndarray::simd::{U16x32, U8x64};

/// Carry out of the wrapping add `low + p = sum`, as 0 or 1 per lane.
#[inline(always)]
pub fn carry_after(low: U16x32, p: U16x32, sum: U16x32) -> U16x32 {
    ((low & p) | ((low | p) & !sum)).shr(15)
}

/// Carry out of `low + p`, as 0 or 1 per lane, computed from the operands
/// before the add: 1 exactly when `p > !low`, the borrow out of `!low - p`.
#[inline(always)]
pub fn carry_before(low: U16x32, p: U16x32) -> U16x32 {
    let head = !low;
    let d = head - p;
    ((low & p) | (!(head ^ p) & d)).shr(15)
}

/// One place-value column pair: `low + 2^16 * high` per lane.
#[derive(Clone, Copy)]
pub struct Col16 {
    pub low: U16x32,
    pub high: U16x32,
}

impl Col16 {
    pub fn zero() -> Self {
        Self {
            low: U16x32::splat(0),
            high: U16x32::splat(0),
        }
    }

    /// Adds products `p` (each below 2^16). `ANTICIPATE` picks the carry rule.
    #[inline(always)]
    pub fn add<const ANTICIPATE: bool>(&mut self, p: U16x32) {
        if ANTICIPATE {
            let c = carry_before(self.low, p);
            self.low += p;
            self.high += c;
        } else {
            let sum = self.low + p;
            self.high += carry_after(self.low, p, sum);
            self.low = sum;
        }
    }

    /// Sum of all 32 lanes in `i64`, read lane by lane from a stored array.
    pub fn total(&self) -> i64 {
        let (lo, hi) = (core::hint::black_box(self.low.to_array()), core::hint::black_box(self.high.to_array()));
        let mut t = 0i64;
        for (&l, &h) in lo.iter().zip(&hi) {
            t += i64::from(l) + (i64::from(h) << 16);
        }
        t
    }
}

/// Blocks before `flush`: each block adds one product per lane and tally, so
/// `high` grows by at most 1 per block and stays below 2^16 for 65_535 blocks.
pub const D_FLUSH_BLOCKS: usize = 65_535;

/// Candidate D: positive and negative tallies, each for the low and the high
/// 32 lanes of a block, folded into `i64` at most every `D_FLUSH_BLOCKS`.
pub struct AccD<const ANTICIPATE: bool> {
    col: [[Col16; 2]; 2],
    blocks: usize,
    total: i64,
}

impl<const ANTICIPATE: bool> AccD<ANTICIPATE> {
    pub fn new() -> Self {
        Self {
            col: [[Col16::zero(); 2]; 2],
            blocks: 0,
            total: 0,
        }
    }

    pub fn flush(&mut self) {
        let side = |c: &[Col16; 2]| c[0].total() + c[1].total();
        self.total += side(&self.col[0]) - side(&self.col[1]);
        self.col = [[Col16::zero(); 2]; 2];
        self.blocks = 0;
    }

    /// The arithmetic alone, on magnitudes already routed by sign.
    #[inline(always)]
    pub fn mac(&mut self, a_pos: U8x64, a_neg: U8x64, b_lo: U16x32, b_hi: U16x32) {
        let products = [
            [U16x32::from_u8x64_lo(a_pos).mullo(b_lo), U16x32::from_u8x64_hi(a_pos).mullo(b_hi)],
            [U16x32::from_u8x64_lo(a_neg).mullo(b_lo), U16x32::from_u8x64_hi(a_neg).mullo(b_hi)],
        ];
        for (cols, ps) in self.col.iter_mut().zip(products) {
            for (c, p) in cols.iter_mut().zip(ps) {
                c.add::<ANTICIPATE>(p);
            }
        }
        self.blocks += 1;
        if self.blocks == D_FLUSH_BLOCKS {
            self.flush();
        }
    }

    /// One block from the shared representation: route `|a|` as A does,
    /// widen `|b|`, multiply, add with carry.
    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (a_pos, a_neg) = route_a(a, b);
        self.mac(a_pos, a_neg, U16x32::from_u8x64_lo(b.mag), U16x32::from_u8x64_hi(b.mag));
    }

    /// One index block against a query whose magnitudes were widened once.
    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryD) {
        let negative = x.positive ^ q.positive;
        let zero = U8x64::splat(0);
        let x_pos = U8x64::mask_blend(negative, x.mag, zero);
        let x_neg = U8x64::mask_blend(!negative, x.mag, zero);
        self.mac(x_pos, x_neg, q.lo, q.hi);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

impl<const ANTICIPATE: bool> Default for AccD<ANTICIPATE> {
    fn default() -> Self {
        Self::new()
    }
}

/// A query block prepared once for a scan: magnitudes widened to `u16`.
pub struct QueryD {
    pub lo: U16x32,
    pub hi: U16x32,
    pub positive: u64,
}

impl QueryD {
    pub fn new(q: Block) -> Self {
        Self {
            lo: U16x32::from_u8x64_lo(q.mag),
            hi: U16x32::from_u8x64_hi(q.mag),
            positive: q.positive,
        }
    }
}
