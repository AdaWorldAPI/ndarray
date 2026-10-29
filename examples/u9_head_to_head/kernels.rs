//! The two U9 x U9 multiply-accumulate candidates, written only with
//! `ndarray::simd` facade words. Shared by `examples/u9_head_to_head` (timing)
//! and `tests/u9_head_to_head.rs` (correctness).
//!
//! Both candidates read the same physical block per operand: 64 `u8`
//! magnitudes and one `u64` plane in which bit `i` set means lane `i` is
//! positive. A lane with plane bit 0 and magnitude 0 is the DR512
//! never-touched sentinel; both candidates refuse a block that holds one
//! (`check`), so only the 511 numeric values -255..=255 enter the arithmetic
//! (touched zero is plane 1, magnitude 0).
//!
//! The product of lane `i` is negative when exactly one operand is negative:
//! `negative = plane_a ^ plane_b` (both planes mark the positive lanes).

#![allow(dead_code)]

use ndarray::simd::{F32x16, I32x16, I8x64, U16x32, U8x64};

/// One operand block: 64 magnitudes and the positive plane.
#[derive(Clone, Copy)]
pub struct Block {
    pub mag: U8x64,
    pub positive: u64,
}

/// A block held a never-touched lane (plane bit 0, magnitude 0); the mask
/// names the offending lanes.
#[derive(Debug, PartialEq, Eq)]
pub struct NeverTouched(pub u64);

/// Never-touched lanes of one block.
#[inline(always)]
pub fn never_touched(b: Block) -> u64 {
    b.mag.cmpeq_mask(U8x64::splat(0)) & !b.positive
}

/// Refuses a pair of blocks if either holds a never-touched lane.
#[inline(always)]
pub fn check(a: Block, b: Block) -> Result<(), NeverTouched> {
    let bad = never_touched(a) | never_touched(b);
    if bad == 0 {
        Ok(())
    } else {
        Err(NeverTouched(bad))
    }
}

/// The 64 bytes of `v` as four `F32x16` quarters, lanes 0..16, 16..32,
/// 32..48, 48..64. Every byte is exact in `f32`.
#[inline(always)]
fn f32_quarters(v: U8x64) -> [F32x16; 4] {
    let (lo, hi) = (U16x32::from_u8x64_lo(v), U16x32::from_u8x64_hi(v));
    [lo.to_f32x16_lo(), lo.to_f32x16_hi(), hi.to_f32x16_lo(), hi.to_f32x16_hi()]
}

// ── Candidate A: U8 x U8 x sign x mul_add ──────────────────────────────────

/// A's sign routing: `|a|` where the product is positive, and where it is
/// negative, with `negative = plane_a ^ plane_b`; the other lanes are 0.
#[inline(always)]
pub fn route_a(a: Block, b: Block) -> (U8x64, U8x64) {
    let negative = a.positive ^ b.positive;
    let zero = U8x64::splat(0);
    (U8x64::mask_blend(negative, a.mag, zero), U8x64::mask_blend(!negative, a.mag, zero))
}

/// Blocks an A accumulator may absorb before `flush`: each lane gains at most
/// 255 * 255 = 65_025 per block, so 256 blocks stay below 2^24, the range in
/// which every integer, and therefore every partial sum, is exact in `f32`.
pub const A_FLUSH_BLOCKS: usize = 256;

/// Candidate A state: the unsigned products of positive and of negative
/// lanes, summed separately per lane in `f32` (exact below 2^24).
pub struct AccA {
    pos: [F32x16; 4],
    neg: [F32x16; 4],
    blocks: usize,
    total: i64,
}

impl AccA {
    pub fn new() -> Self {
        let z = F32x16::splat(0.0);
        Self {
            pos: [z; 4],
            neg: [z; 4],
            blocks: 0,
            total: 0,
        }
    }

    /// Moves the exact per-lane sums into the `i64` total and clears them.
    #[inline(always)]
    pub fn flush(&mut self) {
        let mut lanes = I32x16::splat(0);
        for k in 0..4 {
            // Integers below 2^24: the difference and the conversion are exact.
            lanes += (self.pos[k] - self.neg[k]).cast_i32();
        }
        self.total += i64::from(lanes.reduce_sum());
        let z = F32x16::splat(0.0);
        self.pos = [z; 4];
        self.neg = [z; 4];
        self.blocks = 0;
    }

    /// One block, magnitudes kept unsigned: `|a| * |b|` is routed to the
    /// positive or the negative sum by `negative = plane_a ^ plane_b`, with one
    /// `mul_add` per quarter. Products are taken in `f32` after widening.
    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (a_pos, a_neg) = route_a(a, b);
        self.mac(a_pos, a_neg, b.mag);
    }

    /// The arithmetic alone, on magnitudes already routed by sign.
    #[inline(always)]
    pub fn mac(&mut self, a_pos: U8x64, a_neg: U8x64, b_mag: U8x64) {
        let (fp, fneg, fb) = (f32_quarters(a_pos), f32_quarters(a_neg), f32_quarters(b_mag));
        for k in 0..4 {
            self.pos[k] = fp[k].mul_add(fb[k], self.pos[k]);
            self.neg[k] = fneg[k].mul_add(fb[k], self.neg[k]);
        }
        self.blocks += 1;
        if self.blocks == A_FLUSH_BLOCKS {
            self.flush();
        }
    }

    /// Same as [`Self::step`], but the products are formed in 16 bits with
    /// `U16x32::mullo` (255 * 255 fits `u16`) and only then widened to `f32`
    /// for the signed add.
    #[inline(always)]
    pub fn step_mullo(&mut self, a: Block, b: Block) {
        let (a_pos, a_neg) = route_a(a, b);
        self.mac_mullo(a_pos, a_neg, b.mag);
    }

    /// The arithmetic of [`Self::step_mullo`] alone, on routed magnitudes.
    #[inline(always)]
    pub fn mac_mullo(&mut self, a_pos: U8x64, a_neg: U8x64, b_mag: U8x64) {
        let (b_lo, b_hi) = (U16x32::from_u8x64_lo(b_mag), U16x32::from_u8x64_hi(b_mag));
        let pl = U16x32::from_u8x64_lo(a_pos).mullo(b_lo);
        let ph = U16x32::from_u8x64_hi(a_pos).mullo(b_hi);
        let nl = U16x32::from_u8x64_lo(a_neg).mullo(b_lo);
        let nh = U16x32::from_u8x64_hi(a_neg).mullo(b_hi);
        let p = [pl.to_f32x16_lo(), pl.to_f32x16_hi(), ph.to_f32x16_lo(), ph.to_f32x16_hi()];
        let n = [nl.to_f32x16_lo(), nl.to_f32x16_hi(), nh.to_f32x16_lo(), nh.to_f32x16_hi()];
        for k in 0..4 {
            self.pos[k] += p[k];
            self.neg[k] += n[k];
        }
        self.blocks += 1;
        if self.blocks == A_FLUSH_BLOCKS {
            self.flush();
        }
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

impl Default for AccA {
    fn default() -> Self {
        Self::new()
    }
}

// ── Candidate B: U8 x I8 byte-dot ──────────────────────────────────────────

/// Blocks a B accumulator may absorb before `flush`. Per block an `i32` lane
/// of `lo` gains at most 4 * 255 * 127 = 129_540 and a lane of `hi`
/// 4 * 255 = 1_020; `reduce_sum` adds 16 lanes, so 1024 blocks keep
/// 16 * 1024 * 129_540 = 2_122_383_360 below `i32::MAX`.
pub const B_FLUSH_BLOCKS: usize = 1024;

/// `±v` as signed bytes for `v <= 127`, negated where `negative` has its bit
/// set, built only from facade words.
///
/// The facade has no `U8x64 -> I8x64` reinterpretation, so the bytes travel
/// u8 -> u16 -> f32 (sign applied as a factor 1 or -1) -> i32 -> i16 -> i8
/// through the saturating narrowings, which never saturate here (|v| <= 127).
#[inline(always)]
pub fn signed_bytes(v: U8x64, factor: &[F32x16; 4]) -> I8x64 {
    let f = f32_quarters(v);
    let q: [I32x16; 4] = core::array::from_fn(|k| (f[k] * factor[k]).cast_i32());
    q[0].narrow_saturating_i16(q[1])
        .narrow_saturating_i8(q[2].narrow_saturating_i16(q[3]))
}

/// The per-lane sign factor `1 - 2 * negative` as four `F32x16` quarters.
#[inline(always)]
pub fn sign_factor(negative: u64) -> [F32x16; 4] {
    let twos = U8x64::mask_blend(negative, U8x64::splat(0), U8x64::splat(2));
    let s = f32_quarters(twos);
    core::array::from_fn(|k| s[k].mul_add(F32x16::splat(-1.0), F32x16::splat(1.0)))
}

/// Candidate B state: `|a| * signed(b & 127)` and `|a| * signed(b >> 7)` as
/// two byte-dot accumulators; the result is `lo + 128 * hi`.
pub struct AccB {
    lo: I32x16,
    hi: I32x16,
    blocks: usize,
    total: i64,
}

impl AccB {
    pub fn new() -> Self {
        Self {
            lo: I32x16::splat(0),
            hi: I32x16::splat(0),
            blocks: 0,
            total: 0,
        }
    }

    #[inline(always)]
    pub fn flush(&mut self) {
        self.total += i64::from(self.lo.reduce_sum()) + 128 * i64::from(self.hi.reduce_sum());
        self.lo = I32x16::splat(0);
        self.hi = I32x16::splat(0);
        self.blocks = 0;
    }

    /// The arithmetic alone: two byte-dots on operands already in signed form.
    #[inline(always)]
    pub fn mac(&mut self, a_mag: U8x64, b_lo: I8x64, b_hi: I8x64) {
        self.lo = self.lo.dot4_u8_i8_add(a_mag, b_lo);
        self.hi = self.hi.dot4_u8_i8_add(a_mag, b_hi);
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    /// One block from the shared representation: split `|b|` into its low
    /// seven bits and its top bit, sign both by `negative = plane_a ^ plane_b`,
    /// then [`Self::mac`].
    #[inline(always)]
    pub fn step(&mut self, a: Block, b: Block) {
        let (lo, hi) = prepare_b(a, b);
        self.mac(a.mag, lo, hi);
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

impl Default for AccB {
    fn default() -> Self {
        Self::new()
    }
}

/// B's operand preparation for one block pair: `(±(|b| & 127), ±(|b| >> 7))`.
#[inline(always)]
pub fn prepare_b(a: Block, b: Block) -> (I8x64, I8x64) {
    let factor = sign_factor(a.positive ^ b.positive);
    // `shr_epi16(7)` moves bit 7 of each byte to bit 0 of the same byte (and
    // bit 15 of a pair to bit 8); masking with 1 keeps exactly that bit.
    let low7 = b.mag & U8x64::splat(0x7F);
    let top = b.mag.shr_epi16(7) & U8x64::splat(1);
    (signed_bytes(low7, &factor), signed_bytes(top, &factor))
}

// ── Scan forms: the query operand prepared once, the index operand per block ─
//
// In a search the query block is fixed while index blocks stream past, so each
// candidate may prepare the query side once and amortize it. Both scan forms
// get exactly that and nothing more; the index side is read from the shared
// U8 + plane representation every block.

/// A's query: `|q|` as four `f32` quarters, plus its plane.
pub struct QueryA {
    f: [F32x16; 4],
    positive: u64,
}

impl QueryA {
    pub fn new(q: Block) -> Self {
        Self {
            f: f32_quarters(q.mag),
            positive: q.positive,
        }
    }
}

impl AccA {
    /// One index block against a prepared query: route `|x|` by
    /// `negative = plane_x ^ plane_q`, widen it, and `mul_add` with `|q|`.
    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryA) {
        let negative = x.positive ^ q.positive;
        let zero = U8x64::splat(0);
        let x_pos = f32_quarters(U8x64::mask_blend(negative, x.mag, zero));
        let x_neg = f32_quarters(U8x64::mask_blend(!negative, x.mag, zero));
        for k in 0..4 {
            self.pos[k] = x_pos[k].mul_add(q.f[k], self.pos[k]);
            self.neg[k] = x_neg[k].mul_add(q.f[k], self.neg[k]);
        }
        self.blocks += 1;
        if self.blocks == A_FLUSH_BLOCKS {
            self.flush();
        }
    }
}

/// B's query: `±(|q| & 127)` and `±(|q| >> 7)` as signed bytes, signed by the
/// query's own plane only.
pub struct QueryB {
    lo: I8x64,
    hi: I8x64,
}

impl QueryB {
    pub fn new(q: Block) -> Self {
        let factor = sign_factor(!q.positive);
        let low7 = q.mag & U8x64::splat(0x7F);
        let top = q.mag.shr_epi16(7) & U8x64::splat(1);
        Self {
            lo: signed_bytes(low7, &factor),
            hi: signed_bytes(top, &factor),
        }
    }
}

/// B's scan state: the index sign is routed with `dot4_u8_i8_add_masked`
/// (#387), the query sign is already in the bytes. Four byte-dots per block;
/// result `(pos_lo - neg_lo) + 128 * (pos_hi - neg_hi)`. Same per-lane bound
/// as [`AccB`], so the same flush interval.
pub struct AccBScan {
    acc: [I32x16; 4],
    blocks: usize,
    total: i64,
}

impl AccBScan {
    pub fn new() -> Self {
        Self {
            acc: [I32x16::splat(0); 4],
            blocks: 0,
            total: 0,
        }
    }

    #[inline(always)]
    pub fn flush(&mut self) {
        let r = self.acc.map(|v| i64::from(v.reduce_sum()));
        self.total += (r[0] - r[1]) + 128 * (r[2] - r[3]);
        self.acc = [I32x16::splat(0); 4];
        self.blocks = 0;
    }

    #[inline(always)]
    pub fn scan_step(&mut self, x: Block, q: &QueryB) {
        let (p, n) = (x.positive, !x.positive);
        self.acc[0] = self.acc[0].dot4_u8_i8_add_masked(x.mag, p, q.lo);
        self.acc[1] = self.acc[1].dot4_u8_i8_add_masked(x.mag, n, q.lo);
        self.acc[2] = self.acc[2].dot4_u8_i8_add_masked(x.mag, p, q.hi);
        self.acc[3] = self.acc[3].dot4_u8_i8_add_masked(x.mag, n, q.hi);
        self.blocks += 1;
        if self.blocks == B_FLUSH_BLOCKS {
            self.flush();
        }
    }

    pub fn finish(mut self) -> i64 {
        self.flush();
        self.total
    }
}

impl Default for AccBScan {
    fn default() -> Self {
        Self::new()
    }
}

// ── Baseline and oracle ────────────────────────────────────────────────────

/// Plain u8 x i8 byte-dot, the substrate both candidates are measured
/// against: one `dot4_u8_i8_add` per block.
#[inline(always)]
pub fn baseline_step(acc: I32x16, a: U8x64, w: I8x64) -> I32x16 {
    acc.dot4_u8_i8_add(a, w)
}

/// Signed value of lane `i` of a block, or `None` for never-touched.
pub fn value(mag: &[u8; 64], positive: u64, i: usize) -> Option<i64> {
    let m = i64::from(mag[i]);
    match (positive >> i & 1 == 1, m) {
        (false, 0) => None,
        (true, _) => Some(m),
        (false, _) => Some(-m),
    }
}

/// Independent `i64` oracle over whole blocks; refuses never-touched lanes.
pub fn oracle(a: &[([u8; 64], u64)], b: &[([u8; 64], u64)]) -> Result<i64, NeverTouched> {
    let mut sum = 0i64;
    for ((am, ap), (bm, bp)) in a.iter().zip(b) {
        let mut bad = 0u64;
        for i in 0..64 {
            match (value(am, *ap, i), value(bm, *bp, i)) {
                (Some(x), Some(y)) => sum = sum.checked_add(x.checked_mul(y).unwrap()).unwrap(),
                _ => bad |= 1 << i,
            }
        }
        if bad != 0 {
            return Err(NeverTouched(bad));
        }
    }
    Ok(sum)
}
