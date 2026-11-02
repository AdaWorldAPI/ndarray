//! A2, the exposure antenna: one prepared input, several output shapes.
//!
//! Fixture: the U9 block of #388 (64 `u8` magnitudes + one polarity plane per
//! operand). One prepared input feeds three output shapes:
//!
//! * masked byte-dot → 16 `i32` accumulators (B, and E5's `low7` half);
//! * masked SAD → 8 `u64` accumulators (E5's anchor);
//! * the exact signed fold → one `i64`.
//!
//! Candidates (E0 and E1 are the existing, unchanged kernels):
//!
//! * E0 = B (`kernels::AccB`), E1 = E5 (`anchor::AccE5`);
//! * E2 ([`AccE2`]): the sign-routed operand is prepared ONCE as two
//!   zero-masked vectors `a_pos = a | !negative`, `a_neg = a | negative`. The
//!   `low7` dots then run unmasked on them, and the anchor narrows them by the
//!   bit-7 plane. Reuse law: `maskz(maskz(a, s), t) == maskz(a, s & t)`, so
//!   `sum8(a_pos, top) == sum8(a, !negative & top)` and
//!   `dot(a_pos, w) == dot_masked(a, !negative, w)`. Exact; proven per byte in
//!   the tests.
//! * E3 ([`AccDual`]): two consumers fused in one pass over the same prepared
//!   input: B's exact signed dot (i32 lanes, i64 total) and a second,
//!   independent reduction, the sum of `|a|` over the positive-product lanes
//!   (u64 lanes). They share the load of `a` and the sign plane; their
//!   accumulator types and flush laws stay separate. Compared against the two
//!   consumers run as two passes.

#![allow(dead_code)]

use crate::anchor::QueryE;
use crate::kernels::{lanes_i64, AccB, Block, B_FLUSH_BLOCKS};
use ndarray::simd::{I32x16, I8x64, U64x8, U8x64};

/// E2: the sign-routed operand prepared once, narrowed for the anchor.
pub struct AccE2 {
    low7: [I32x16; 2],
    anchor: [U64x8; 2],
    blocks: usize,
    total: i64,
}

impl AccE2 {
    pub fn new() -> Self {
        Self {
            low7: [I32x16::splat(0); 2],
            anchor: [U64x8::splat(0); 2],
            blocks: 0,
            total: 0,
        }
    }

    pub fn flush(&mut self) {
        let r = self.low7.map(lanes_i64);
        let anchor = self.anchor[0].reduce_sum() as i64 - self.anchor[1].reduce_sum() as i64;
        self.total += (r[0] - r[1]) + (anchor << 7);
        self.low7 = [I32x16::splat(0); 2];
        self.anchor = [U64x8::splat(0); 2];
        self.blocks = 0;
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: u64) {
        let zero = U8x64::splat(0);
        // The prepared exposure: `a` routed by sign, once per block.
        let a_pos = U8x64::mask_blend(!negative, zero, a);
        let a_neg = U8x64::mask_blend(negative, zero, a);
        self.low7[0] = self.low7[0].dot4_u8_i8_add(a_pos, low7);
        self.low7[1] = self.low7[1].dot4_u8_i8_add(a_neg, low7);
        // The anchor narrows the SAME prepared vectors by bit 7.
        self.anchor[0] = self.anchor[0].sum8_u8_add_masked(a_pos, top);
        self.anchor[1] = self.anchor[1].sum8_u8_add_masked(a_neg, top);
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
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

/// The second consumer alone: the sum of `|a|` over lanes whose product is
/// positive, in `u64` lanes (never flushed: 8 * 255 per block per lane).
pub struct AccPosMag {
    acc: U64x8,
}

impl AccPosMag {
    pub fn new() -> Self {
        Self { acc: U64x8::splat(0) }
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64) {
        self.acc = self.acc.sum8_u8_add_masked(a, !negative);
    }

    pub fn finish(self) -> u64 {
        self.acc.reduce_sum()
    }
}

/// E3: B and the positive-magnitude sum fused in one pass. Each keeps its own
/// accumulator type and reduction law; only the input and the sign plane are
/// shared.
pub struct AccDual {
    pub b: AccB,
    pub pos_mag: AccPosMag,
}

impl AccDual {
    pub fn new() -> Self {
        Self {
            b: AccB::new(),
            pos_mag: AccPosMag::new(),
        }
    }

    #[inline(always)]
    pub fn mac(&mut self, a: U8x64, negative: u64, low7: I8x64, top: I8x64) {
        self.b.mac(a, negative, low7, top);
        self.pos_mag.mac(a, negative);
    }

    pub fn finish(self) -> (i64, u64) {
        (self.b.finish(), self.pos_mag.finish())
    }
}

/// Scalar oracle for the second consumer.
pub fn pos_mag_oracle(a: &[([u8; 64], u64)], b: &[([u8; 64], u64)]) -> u64 {
    let mut t = 0u64;
    for ((ma, pa), (_, pb)) in a.iter().zip(b) {
        let positive = !(pa ^ pb);
        for (i, &m) in ma.iter().enumerate() {
            if positive >> i & 1 == 1 {
                t += m as u64;
            }
        }
    }
    t
}
