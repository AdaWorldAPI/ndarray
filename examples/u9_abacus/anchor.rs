//! D-U9-ANCHOR: candidate E. B's split of `|b|` at the power of two 128,
//! `|b| = low7 + 128 * bit7`, with the anchor term taken as a shift instead
//! of a multiplication:
//!
//! `a * |b| = a * low7 + (sum of a over the lanes where bit 7 of b is set) << 7`
//!
//! `low7` still needs a real byte multiply; the anchor term needs only a masked
//! sum of `a`, weighted by 128 once, at flush. B spends a full byte-dot against
//! `-128` on it. Three ways to form the masked sum from existing facade words:
//!
//! * E4 ([`AccE4`]): byte-dot of `a` against all-ones bytes, masked by
//!   `sign mask AND bit-7 plane`, one per sign: still four byte-dots, but the
//!   second operand is a constant;
//! * E3 ([`AccE3`]): one byte-dot of `a` against a `+1 / -1` byte vector
//!   (built from the sign plane), masked by the bit-7 plane: three byte-dots;
//! * Esum ([`AccESum`]): `U8x64::mask_blend` + `sum_bytes_u64` per sign, which
//!   reduces horizontally on every call;
//! * E5 ([`AccE5`]): `U64x8::sum8_u8_add_masked` (#390) per sign into two
//!   persistent vertical accumulators, reduced only at flush.
//!
//! The polarity rule is unchanged: `negative = plane_a XOR plane_b`.

#![allow(dead_code)]

use crate::kernels::{bits_as_i8, lanes_i64, Block, B_FLUSH_BLOCKS};
use ndarray::simd::{I32x16, I8x64, U64x8, U8x64};

/// Lanes whose magnitude has bit 7 set, as a plane.
#[inline(always)]
pub fn bit7(mag: U8x64) -> u64 {
    let top = U8x64::splat(0x80);
    (mag & top).cmpeq_mask(top)
}

/// The prepared `b` side: `low7` as signed bytes and the bit-7 plane.
#[inline(always)]
pub fn split_anchor(b: U8x64) -> (I8x64, u64) {
    (bits_as_i8(b & U8x64::splat(0x7F)), bit7(b))
}

/// E4: four byte-dots, the two anchor dots against constant all-ones bytes.
/// Lanes: `[pos x low7, neg x low7, pos anchor, neg anchor]`. Per block a lane
/// moves by at most `4 * 255 * 127` (low7) or `4 * 255` (anchor), so
/// `B_FLUSH_BLOCKS` keeps every lane inside `i32`.
pub struct AccE4 {
    acc: [I32x16; 4],
    blocks: usize,
    total: i64,
}

impl AccE4 {
    pub fn new() -> Self {
        Self {
            acc: [I32x16::splat(0); 4],
            blocks: 0,
            total: 0,
        }
    }

    pub fn flush(&mut self) {
        let r = self.acc.map(lanes_i64);
        self.total += (r[0] - r[1]) + ((r[2] - r[3]) << 7);
        self.acc = [I32x16::splat(0); 4];
        self.blocks = 0;
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: u64) {
        let ones = I8x64::splat(1);
        let (p, n) = (!negative, negative);
        self.acc[0] = self.acc[0].dot4_u8_i8_add_masked(a, p, low7);
        self.acc[1] = self.acc[1].dot4_u8_i8_add_masked(a, n, low7);
        self.acc[2] = self.acc[2].dot4_u8_i8_add_masked(a, p & top, ones);
        self.acc[3] = self.acc[3].dot4_u8_i8_add_masked(a, n & top, ones);
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (low7, top) = split_anchor(b.mag);
        self.mac(a.mag, a.positive ^ b.positive, low7, top);
    }

    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryE) {
        self.mac(x.mag, x.positive ^ q.positive, q.low7, q.top);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

/// `+1` where the product is positive, `-1` where it is negative, as signed
/// bytes. Built by `U8x64::mask_blend` from the sign plane; a sign, not a
/// negated magnitude.
#[inline(always)]
pub fn sign_bytes(negative: u64) -> I8x64 {
    bits_as_i8(U8x64::mask_blend(negative, U8x64::splat(1), U8x64::splat(0xFF)))
}

/// E3: three byte-dots, the anchor in one signed dot.
/// Lanes: `[pos x low7, neg x low7, signed anchor]`.
pub struct AccE3 {
    acc: [I32x16; 3],
    blocks: usize,
    total: i64,
}

impl AccE3 {
    pub fn new() -> Self {
        Self {
            acc: [I32x16::splat(0); 3],
            blocks: 0,
            total: 0,
        }
    }

    pub fn flush(&mut self) {
        let r = self.acc.map(lanes_i64);
        self.total += (r[0] - r[1]) + (r[2] << 7);
        self.acc = [I32x16::splat(0); 3];
        self.blocks = 0;
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: u64) {
        self.acc[0] = self.acc[0].dot4_u8_i8_add_masked(a, !negative, low7);
        self.acc[1] = self.acc[1].dot4_u8_i8_add_masked(a, negative, low7);
        self.acc[2] = self.acc[2].dot4_u8_i8_add_masked(a, top, sign_bytes(negative));
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (low7, top) = split_anchor(b.mag);
        self.mac(a.mag, a.positive ^ b.positive, low7, top);
    }

    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryE) {
        self.mac(x.mag, x.positive ^ q.positive, q.low7, q.top);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

/// Esum: the two `low7` byte-dots, and the anchor sums as masked byte sums
/// reduced to scalars on every block (`sum_bytes_u64`).
pub struct AccESum {
    acc: [I32x16; 2],
    anchor: i64,
    blocks: usize,
    total: i64,
}

impl AccESum {
    pub fn new() -> Self {
        Self {
            acc: [I32x16::splat(0); 2],
            anchor: 0,
            blocks: 0,
            total: 0,
        }
    }

    pub fn flush(&mut self) {
        let r = self.acc.map(lanes_i64);
        self.total += (r[0] - r[1]) + (self.anchor << 7);
        self.acc = [I32x16::splat(0); 2];
        self.anchor = 0;
        self.blocks = 0;
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: u64) {
        let zero = U8x64::splat(0);
        self.acc[0] = self.acc[0].dot4_u8_i8_add_masked(a, !negative, low7);
        self.acc[1] = self.acc[1].dot4_u8_i8_add_masked(a, negative, low7);
        let pos = U8x64::mask_blend(!negative & top, zero, a).sum_bytes_u64();
        let neg = U8x64::mask_blend(negative & top, zero, a).sum_bytes_u64();
        self.anchor += pos as i64 - neg as i64;
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (low7, top) = split_anchor(b.mag);
        self.mac(a.mag, a.positive ^ b.positive, low7, top);
    }

    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryE) {
        self.mac(x.mag, x.positive ^ q.positive, q.low7, q.top);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

/// E5: the two `low7` byte-dots, and the anchor sums as two persistent
/// vertical masked byte sums (`U64x8::sum8_u8_add_masked`), reduced only at
/// flush.
///
/// Bounds per block: a `low7` lane gains at most `4 * 255 * 127 = 129_540`,
/// so after `B_FLUSH_BLOCKS` = 16_384 blocks it holds at most 2_122_383_360,
/// below `i32::MAX`. An anchor lane gains at most `8 * 255 = 2_040`, at most
/// 33_423_360 per flush; eight lanes sum to at most 267_386_880 and `<< 7`
/// to 3.4e10, inside `i64`. The `u64` lanes cannot overflow before the flush.
pub struct AccE5 {
    low7: [I32x16; 2],
    anchor: [U64x8; 2],
    blocks: usize,
    total: i64,
    /// Number of anchor reductions so far; one per flush, never per block.
    pub reductions: usize,
}

impl AccE5 {
    pub fn new() -> Self {
        Self {
            low7: [I32x16::splat(0); 2],
            anchor: [U64x8::splat(0); 2],
            blocks: 0,
            total: 0,
            reductions: 0,
        }
    }

    pub fn flush(&mut self) {
        let r = self.low7.map(lanes_i64);
        let anchor = self.anchor[0].reduce_sum() as i64 - self.anchor[1].reduce_sum() as i64;
        self.total += (r[0] - r[1]) + (anchor << 7);
        self.low7 = [I32x16::splat(0); 2];
        self.anchor = [U64x8::splat(0); 2];
        self.blocks = 0;
        self.reductions += 1;
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: u64) {
        self.low7[0] = self.low7[0].dot4_u8_i8_add_masked(a, !negative, low7);
        self.low7[1] = self.low7[1].dot4_u8_i8_add_masked(a, negative, low7);
        self.anchor[0] = self.anchor[0].sum8_u8_add_masked(a, !negative & top);
        self.anchor[1] = self.anchor[1].sum8_u8_add_masked(a, negative & top);
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (low7, top) = split_anchor(b.mag);
        self.mac(a.mag, a.positive ^ b.positive, low7, top);
    }

    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryE) {
        self.mac(x.mag, x.positive ^ q.positive, q.low7, q.top);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

/// A query prepared once: `low7` bytes, bit-7 plane, polarity.
pub struct QueryE {
    pub low7: I8x64,
    pub top: u64,
    pub positive: u64,
}

impl QueryE {
    pub fn new(q: Block) -> Self {
        let (low7, top) = split_anchor(q.mag);
        Self {
            low7,
            top,
            positive: q.positive,
        }
    }
}
