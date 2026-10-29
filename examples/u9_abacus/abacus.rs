//! D-U9-ABACUS: exact U9 x U9 multiply-accumulate as schoolbook arithmetic on
//! bit planes, written in plain Rust bit operations. Probe-private; nothing
//! here is a facade word or a storage format.
//!
//! A block of 64 magnitudes is read as eight bit planes: bit `k` of
//! `plane[i]` is bit `i` of lane `k`. The product of two lanes is
//! `sum over i, j of (a_i AND b_j) * 2^(i + j)`, so one AND of two planes
//! yields one partial-product bit for all 64 lanes at significance `i + j`.
//! The sign is the XOR of the two polarity planes and only selects whether a
//! partial product joins the positive or the negative tally; no magnitude is
//! negated.
//!
//! Two ways to fold the partial products:
//!
//! * [`AccPopcnt`] counts each partial-product plane at once
//!   (`popcount * 2^(i + j)`) into an `i64`;
//! * [`AccAbacus`] keeps them uncounted in carry-save columns
//!   ([`Abacus`]): three planes in column `w` compress to one in `w` and one
//!   carry in `w + 1` ([`carry_save3`]); counting happens once, at `finish`.
//!
//! Extracting the planes from the existing `U8x64` magnitudes uses facade
//! words only (`BitAnd`, `cmpeq_mask`) and is timed as part of the cost.

#![allow(dead_code)]

use ndarray::simd::U8x64;

/// Full adder on 64 independent bit columns. For every bit position `k`,
/// `a_k + b_k + c_k = sum_k + 2 * carry_k`: `sum` stays in the current
/// significance, `carry` belongs to the next one. Nothing is shifted across
/// bit positions; each bit is one lane.
#[inline]
pub fn carry_save3(a: u64, b: u64, c: u64) -> (u64, u64) {
    let t = a ^ b;
    let sum = t ^ c;
    let carry = (a & b) ^ (t & c);
    (sum, carry)
}

/// The eight bit planes of 64 magnitudes: bit `k` of `plane[i]` is bit `i` of
/// lane `k`. Facade words only: one AND and one byte compare per plane.
#[inline(always)]
pub fn planes(mag: U8x64) -> [u64; 8] {
    core::array::from_fn(|i| {
        let bit = U8x64::splat(1 << i);
        (mag & bit).cmpeq_mask(bit)
    })
}

/// Significance columns an [`Abacus`] holds. Column `w` can only hold a set
/// bit for a lane whose running total is at least `2^w`, so 64 columns cover
/// any per-lane total below `2^64`; [`Abacus::add`] panics rather than drop a
/// carry past the last column.
pub const COLUMNS: usize = 64;

/// Uncounted carry-save tally of 64 independent lanes: column `w` holds at
/// most two planes of significance `2^w`. The represented per-lane value is
/// `sum over w of 2^w * (bits of that lane in column w)`.
#[derive(Clone)]
pub struct Abacus {
    col: [[u64; 2]; COLUMNS],
    n: [u8; COLUMNS],
}

impl Abacus {
    pub fn new() -> Self {
        Self {
            col: [[0; 2]; COLUMNS],
            n: [0; COLUMNS],
        }
    }

    /// Adds plane `p` at significance `2^w`. A third plane in a column is
    /// compressed with the two already there: the sum stays, the carry moves
    /// up one column and is added there the same way.
    #[inline(always)]
    pub fn add(&mut self, mut w: usize, mut p: u64) {
        loop {
            let k = usize::from(self.n[w]);
            if k < 2 {
                self.col[w][k] = p;
                self.n[w] += 1;
                return;
            }
            let (s, c) = carry_save3(self.col[w][0], self.col[w][1], p);
            self.col[w][0] = s;
            self.n[w] = 1;
            p = c;
            w += 1;
            assert!(w < COLUMNS, "abacus carry past column {COLUMNS}");
        }
    }

    /// Sum over all 64 lanes, counted once: `popcount(plane) * 2^w`, exact in
    /// `i128` for every abacus (at most 2 * 64 planes, 64 bits each, below 2^64).
    pub fn value(&self) -> i128 {
        let mut v = 0i128;
        for w in 0..COLUMNS {
            for k in 0..usize::from(self.n[w]) {
                v += i128::from(self.col[w][k].count_ones()) << w;
            }
        }
        v
    }
}

impl Default for Abacus {
    fn default() -> Self {
        Self::new()
    }
}

/// C, deferred: positive and negative partial products tallied in two
/// abaci, counted only at [`Self::finish`].
pub struct AccAbacus {
    pos: Abacus,
    neg: Abacus,
}

impl AccAbacus {
    pub fn new() -> Self {
        Self {
            pos: Abacus::new(),
            neg: Abacus::new(),
        }
    }

    /// One block from prepared planes. `negative` is `plane_a ^ plane_b`.
    #[inline(always)]
    pub fn mac(&mut self, a: &[u64; 8], negative: u64, b: &[u64; 8]) {
        for i in 0..8 {
            let (ap, an) = (a[i] & !negative, a[i] & negative);
            for j in 0..8 {
                self.pos.add(i + j, ap & b[j]);
                self.neg.add(i + j, an & b[j]);
            }
        }
    }

    pub fn finish(&self) -> i64 {
        i64::try_from(self.pos.value() - self.neg.value()).expect("total outside i64")
    }
}

impl Default for AccAbacus {
    fn default() -> Self {
        Self::new()
    }
}

/// C, immediate: each partial-product plane counted at once into `i64`.
/// Per block the change is at most 64 * 65_025; the `i64` total is exact
/// while it stays in range (about 2.2 * 10^12 worst-case blocks).
pub struct AccPopcnt {
    total: i64,
}

impl AccPopcnt {
    pub fn new() -> Self {
        Self { total: 0 }
    }

    #[inline(always)]
    pub fn mac(&mut self, a: &[u64; 8], negative: u64, b: &[u64; 8]) {
        let mut block = 0i64;
        for i in 0..8 {
            let (ap, an) = (a[i] & !negative, a[i] & negative);
            for j in 0..8 {
                let d = i64::from((ap & b[j]).count_ones()) - i64::from((an & b[j]).count_ones());
                block += d << (i + j);
            }
        }
        self.total += block;
    }

    pub fn finish(&self) -> i64 {
        self.total
    }
}

impl Default for AccPopcnt {
    fn default() -> Self {
        Self::new()
    }
}

/// A query block prepared once for a scan: its planes and polarity.
pub struct QueryC {
    pub planes: [u64; 8],
    pub positive: u64,
}

impl QueryC {
    pub fn new(mag: U8x64, positive: u64) -> Self {
        Self {
            planes: planes(mag),
            positive,
        }
    }
}
