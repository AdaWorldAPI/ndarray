//! D-DOMINO-MOV-0: positional substitution and lazy binary carry on
//! bit-sliced columns. Probe-private.
//!
//! A column is one `u64`: bit `j` belongs to lane `j`, and every column of a
//! counter carries the same 64 lane identities at its own significance. A
//! carry mask moves to the next column as it is; no lane position changes.
//!
//! Invariants:
//!
//! * bit flip (half adder, per lane): adding carry `c` to bit `b` gives
//!   `b' = b ^ c` and carries `b & c` to the next significance;
//!   `b + c = b' + 2 * (b & c)` in every lane.
//! * positional substitution: if a column is logically zero (EMPTY), adding
//!   mask `m` gives exactly `m` and no carry, so the column may be assigned
//!   without reading or clearing it.
//! * lazy FULL: if a column is logically all ones (FULL), adding `m` gives
//!   `!m` and carries `m`; when `m` is all ones the column becomes EMPTY and
//!   the whole mask moves on, so only the state changes.
//!
//! Four counters over the same columns, all exact:
//!
//! * [`Eager`] (F): ripple `col ^= m; m = old & m` until the carry is empty;
//! * [`Subst`] (P): `Eager` plus an EMPTY bitmap, assignment into EMPTY columns;
//! * [`LazyFull`] (L): `Eager` plus a FULL bitmap, an all-ones carry passes a
//!   FULL column by writing zero, without reading it;
//! * [`SubstLazy`] (PL): both bitmaps; an all-ones carry passes FULL columns
//!   by changing state bits only, the data words are not touched.

#![allow(dead_code)]

/// Half adder on 64 lanes: `(old ^ carry, old & carry)`.
#[inline(always)]
pub fn flip(old: u64, carry: u64) -> (u64, u64) {
    (old ^ carry, old & carry)
}

/// A carry left the highest column: the lanes in the mask would need one more
/// column. The counter then holds its value modulo `2^N` in those lanes.
#[derive(Debug, PartialEq, Eq)]
pub struct Overflow(pub u64);

/// The interface all four counters share.
pub trait Counter<const N: usize> {
    fn new() -> Self;
    /// Adds 1 at significance `2^k` to every lane set in `m`.
    fn add_at(&mut self, k: usize, m: u64) -> Result<(), Overflow>;
    /// The logical columns (EMPTY / FULL resolved).
    fn columns(&self) -> [u64; N];
    /// Writes the logical columns back and clears any state metadata.
    fn normalize(&mut self);

    /// Value of lane `j`, exact up to 128 columns.
    fn lane(&self, j: usize) -> u128 {
        let c = self.columns();
        (0..N).map(|k| u128::from(c[k] >> j & 1) << k).sum()
    }
    /// Sum over all 64 lanes.
    fn total(&self) -> u128 {
        let c = self.columns();
        (0..N).map(|k| u128::from(c[k].count_ones()) << k).sum()
    }
}

/// F: plain domino ripple.
#[derive(Clone)]
pub struct Eager<const N: usize> {
    col: [u64; N],
}

impl<const N: usize> Counter<N> for Eager<N> {
    fn new() -> Self {
        Self { col: [0; N] }
    }
    #[inline(always)]
    fn add_at(&mut self, mut k: usize, mut m: u64) -> Result<(), Overflow> {
        while m != 0 {
            if k == N {
                return Err(Overflow(m));
            }
            let (next, carry) = flip(self.col[k], m);
            self.col[k] = next;
            m = carry;
            k += 1;
        }
        Ok(())
    }
    fn columns(&self) -> [u64; N] {
        self.col
    }
    fn normalize(&mut self) {}
}

/// P: an EMPTY bitmap (bit `k` set: column `k` is logically zero whatever the
/// word holds). Adding into an EMPTY column assigns the mask.
#[derive(Clone)]
pub struct Subst<const N: usize> {
    col: [u64; N],
    empty: u128,
}

impl<const N: usize> Counter<N> for Subst<N> {
    fn new() -> Self {
        // The words are never read while EMPTY, so they need no clearing;
        // they are zero here only because Rust requires initialized memory.
        Self {
            col: [0; N],
            empty: u128::MAX,
        }
    }
    #[inline(always)]
    fn add_at(&mut self, mut k: usize, mut m: u64) -> Result<(), Overflow> {
        while m != 0 {
            if k == N {
                return Err(Overflow(m));
            }
            if self.empty >> k & 1 == 1 {
                self.col[k] = m;
                self.empty &= !(1u128 << k);
                return Ok(());
            }
            let (next, carry) = flip(self.col[k], m);
            self.col[k] = next;
            m = carry;
            k += 1;
        }
        Ok(())
    }
    fn columns(&self) -> [u64; N] {
        core::array::from_fn(|k| if self.empty >> k & 1 == 1 { 0 } else { self.col[k] })
    }
    fn normalize(&mut self) {
        self.col = self.columns();
        self.empty = 0;
    }
}

impl<const N: usize> Subst<N> {
    /// A zero counter over words that still hold `stale` (a reused buffer):
    /// every column is EMPTY, so the words are never read before written.
    pub fn stale(stale: u64) -> Self {
        Self {
            col: [stale; N],
            empty: u128::MAX,
        }
    }
}

/// L: a FULL bitmap (bit `k` set: column `k` is logically all ones). An
/// all-ones carry into a FULL column writes zero without reading the word; any
/// other carry materializes `!m`. A column becomes FULL when an update leaves
/// it all ones.
#[derive(Clone)]
pub struct LazyFull<const N: usize> {
    col: [u64; N],
    full: u128,
}

impl<const N: usize> Counter<N> for LazyFull<N> {
    fn new() -> Self {
        Self { col: [0; N], full: 0 }
    }
    #[inline(always)]
    fn add_at(&mut self, mut k: usize, mut m: u64) -> Result<(), Overflow> {
        while m != 0 {
            if k == N {
                return Err(Overflow(m));
            }
            let bit = 1u128 << k;
            if self.full & bit != 0 {
                // Logical value is all ones: the result is !m, the carry is m.
                self.col[k] = !m;
                self.full &= !bit;
            } else {
                let (next, carry) = flip(self.col[k], m);
                self.col[k] = next;
                if next == u64::MAX {
                    self.full |= bit;
                }
                m = carry;
            }
            k += 1;
        }
        Ok(())
    }
    fn columns(&self) -> [u64; N] {
        core::array::from_fn(|k| if self.full >> k & 1 == 1 { u64::MAX } else { self.col[k] })
    }
    fn normalize(&mut self) {
        self.col = self.columns();
        self.full = 0;
    }
}

/// PL: EMPTY and FULL bitmaps (never both set for one column). An all-ones
/// carry passes a FULL column as a state change FULL -> EMPTY; an EMPTY column
/// takes the mask by assignment (and turns FULL if the mask is all ones).
#[derive(Clone)]
pub struct SubstLazy<const N: usize> {
    col: [u64; N],
    empty: u128,
    full: u128,
}

impl<const N: usize> Counter<N> for SubstLazy<N> {
    fn new() -> Self {
        Self {
            col: [0; N],
            empty: u128::MAX,
            full: 0,
        }
    }
    #[inline(always)]
    fn add_at(&mut self, mut k: usize, mut m: u64) -> Result<(), Overflow> {
        while m != 0 {
            if k == N {
                return Err(Overflow(m));
            }
            let bit = 1u128 << k;
            if self.empty & bit != 0 {
                self.empty &= !bit;
                if m == u64::MAX {
                    self.full |= bit;
                } else {
                    self.col[k] = m;
                }
                return Ok(());
            }
            if self.full & bit != 0 {
                self.full &= !bit;
                if m == u64::MAX {
                    // FULL + all ones = EMPTY, carry all ones: state only.
                    self.empty |= bit;
                } else {
                    self.col[k] = !m;
                }
            } else {
                let (next, carry) = flip(self.col[k], m);
                self.col[k] = next;
                if next == u64::MAX {
                    // Same FULL policy as L: an ordinary ripple that leaves the
                    // column all ones marks it, so a later all-ones carry passes
                    // it as a state change.
                    self.full |= bit;
                }
                m = carry;
            }
            k += 1;
        }
        Ok(())
    }
    fn columns(&self) -> [u64; N] {
        core::array::from_fn(|k| {
            if self.empty >> k & 1 == 1 {
                0
            } else if self.full >> k & 1 == 1 {
                u64::MAX
            } else {
                self.col[k]
            }
        })
    }
    fn normalize(&mut self) {
        self.col = self.columns();
        self.empty = 0;
        self.full = 0;
    }
}

impl<const N: usize> LazyFull<N> {
    /// Bit `k` set: column `k` is held as FULL (logically all ones).
    pub fn full_bits(&self) -> u128 {
        self.full
    }
}

impl<const N: usize> SubstLazy<N> {
    /// Bit `k` set: column `k` is held as FULL (logically all ones).
    pub fn full_bits(&self) -> u128 {
        self.full
    }
    /// Bit `k` set: column `k` is held as EMPTY (logically zero).
    pub fn empty_bits(&self) -> u128 {
        self.empty
    }

    /// As [`Subst::stale`]: logically zero, words holding `stale`.
    pub fn stale(stale: u64) -> Self {
        Self {
            col: [stale; N],
            empty: u128::MAX,
            full: 0,
        }
    }
}

/// Why [`Abacus::try_add_at`] refused. The counter is unchanged either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbacusError {
    /// `significance >= BITS`.
    SignificanceOutOfRange { significance: usize, bits: usize },
    /// A carry left the highest column in these lanes.
    Overflow { lanes: u64 },
}

/// C (D-BINARY-ABACUS-POLYFILL-0 candidate): F's eager ripple built from
/// `ndarray::simd::half_add_mask`, with transactional overflow refusal.
///
/// Each of the 64 lanes is an independent unsigned counter of `BITS` bits.
/// `BITS <= 122`, so a lane value (< 2^122) and the total over 64 lanes
/// (< 2^128) are both exact in `u128`.
///
/// The success path is F's. On overflow the touched columns are restored by
/// recomputing the carry chain forward from the input mask:
/// `old_k = new_k ^ m_k` and `m_{k+1} = old_k & m_k`. No undo log is kept.
#[derive(Clone)]
pub struct Abacus<const BITS: usize> {
    col: [u64; BITS],
}

impl<const BITS: usize> Abacus<BITS> {
    const BOUND: () = assert!(BITS >= 1 && BITS <= 122, "Abacus: 1 <= BITS <= 122");

    pub fn new() -> Self {
        let () = Self::BOUND;
        Self { col: [0; BITS] }
    }

    /// Adds `2^significance` to every lane set in `lanes`.
    #[inline(always)]
    pub fn try_add_at(&mut self, significance: usize, lanes: u64) -> Result<(), AbacusError> {
        if significance >= BITS {
            return Err(AbacusError::SignificanceOutOfRange {
                significance,
                bits: BITS,
            });
        }
        let (mut k, mut m) = (significance, lanes);
        while m != 0 {
            if k == BITS {
                self.undo(significance, lanes);
                return Err(AbacusError::Overflow { lanes: m });
            }
            let (next, carry) = ndarray::simd::half_add_mask(self.col[k], m);
            self.col[k] = next;
            m = carry;
            k += 1;
        }
        Ok(())
    }

    /// Restores every column from `start` up after an add of `lanes` whose
    /// carry escaped the top. Cold: runs only on refusal.
    #[cold]
    fn undo(&mut self, start: usize, lanes: u64) {
        let mut m = lanes;
        for k in start..BITS {
            let old = self.col[k] ^ m;
            self.col[k] = old;
            m &= old;
            if m == 0 {
                break;
            }
        }
    }

    /// Value of lane `lane` (< 64).
    pub fn lane_value(&self, lane: usize) -> u128 {
        (0..BITS)
            .map(|k| u128::from(self.col[k] >> lane & 1) << k)
            .sum()
    }

    /// Sum over all 64 lanes.
    pub fn total(&self) -> u128 {
        (0..BITS)
            .map(|k| u128::from(self.col[k].count_ones()) << k)
            .sum()
    }

    pub fn columns(&self) -> [u64; BITS] {
        self.col
    }
}

impl<const N: usize> Counter<N> for Abacus<N> {
    fn new() -> Self {
        Abacus::new()
    }
    #[inline(always)]
    fn add_at(&mut self, k: usize, m: u64) -> Result<(), Overflow> {
        self.try_add_at(k, m).map_err(|e| match e {
            AbacusError::Overflow { lanes } => Overflow(lanes),
            AbacusError::SignificanceOutOfRange { .. } => Overflow(m),
        })
    }
    fn columns(&self) -> [u64; N] {
        self.col
    }
    fn normalize(&mut self) {}
}

/// Native baseline: 64 per-lane integers.
#[derive(Clone)]
pub struct Native {
    pub lanes: [u64; 64],
}

impl Native {
    pub fn new() -> Self {
        Self { lanes: [0; 64] }
    }
    /// Adds `2^k` to every lane set in `m` (`k < 64`).
    #[inline(always)]
    pub fn add_at(&mut self, k: usize, m: u64) {
        for (j, l) in self.lanes.iter_mut().enumerate() {
            *l = l.wrapping_add((m >> j & 1) << k);
        }
    }
}

// ── Two significance planes in 128 bits ─────────────────────────────────────

/// R1: two adjacent significance planes as two words. `lo` is the 2^0 plane,
/// `hi` the 2^1 plane, of the same 64 lanes (64 independent 2-bit counters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair128 {
    pub lo: u64,
    pub hi: u64,
}

/// Adds 1 to every lane set in `m`; returns the carry out of the 2-bit lanes.
#[inline(always)]
pub fn pair_add(p: Pair128, m: u64) -> (Pair128, u64) {
    let (lo, c) = flip(p.lo, m);
    let (hi, out) = flip(p.hi, c);
    (Pair128 { lo, hi }, out)
}

/// R2: the same two planes packed into one `u128` (low half = 2^0 plane).
#[inline(always)]
pub fn packed_add(x: u128, m: u64) -> (u128, u64) {
    let (lo, hi) = (x as u64, (x >> 64) as u64);
    let (lo, c) = flip(lo, m);
    let (hi, out) = flip(hi, c);
    (u128::from(hi) << 64 | u128::from(lo), out)
}

/// R1 <-> R2.
pub fn pack(p: Pair128) -> u128 {
    u128::from(p.hi) << 64 | u128::from(p.lo)
}
pub fn unpack(x: u128) -> Pair128 {
    Pair128 {
        lo: x as u64,
        hi: (x >> 64) as u64,
    }
}

// ── The 64K boundary: low 16 bits wrap, the carry enters the bit-16 column ──

use crate::carry16::carry_after;
use ndarray::simd::U16x32;

/// Carry-out lanes of two 32-lane halves as one 64-lane mask (lane `i` of
/// the low half is bit `i`, lane `i` of the high half bit `32 + i`).
#[inline(always)]
pub fn carry_mask(c_lo: U16x32, c_hi: U16x32) -> u64 {
    let one = U16x32::splat(1);
    u64::from(c_lo.cmpeq_mask(one)) | u64::from(c_hi.cmpeq_mask(one)) << 32
}

/// Accumulates `u16` products per lane as `low16 + 2^16 * high`, with `high`
/// held by a bit-sliced counter `C` whose column 0 is the bit-16 plane.
/// `normalize_each` normalizes the counter after every block (eager).
pub fn sum_bitsliced<C: Counter<N>, const N: usize>(prods: &[[U16x32; 2]], normalize_each: bool) -> u128 {
    let mut low = [U16x32::splat(0); 2];
    let mut high = C::new();
    for p in prods {
        let s0 = low[0] + p[0];
        let s1 = low[1] + p[1];
        let m = carry_mask(carry_after(low[0], p[0], s0), carry_after(low[1], p[1], s1));
        low = [s0, s1];
        high.add_at(0, m).expect("bit-sliced high counter overflow");
        if normalize_each {
            high.normalize();
        }
    }
    let lows: u128 = low.iter().flat_map(|v| v.to_array()).map(u128::from).sum();
    lows + (high.total() << 16)
}

/// D (#389): the same low word, `high` a `u16` count per lane (`high += carry`),
/// flushed every 65,535 blocks.
pub fn sum_d(prods: &[[U16x32; 2]]) -> u128 {
    let mut col = [crate::carry16::Col16::zero(); 2];
    let mut total = 0u128;
    for (n, p) in prods.iter().enumerate() {
        col[0].add::<false>(p[0]);
        col[1].add::<false>(p[1]);
        if n % crate::carry16::D_FLUSH_BLOCKS == crate::carry16::D_FLUSH_BLOCKS - 1 {
            total += (col[0].total() + col[1].total()) as u128;
            col = [crate::carry16::Col16::zero(); 2];
        }
    }
    total + (col[0].total() + col[1].total()) as u128
}

/// Native: one `u64` accumulator per lane, plain Rust (the compiler picks the
/// instructions).
pub fn sum_native(prods: &[[u16; 64]]) -> u128 {
    let mut acc = [0u64; 64];
    for p in prods {
        for (a, &x) in acc.iter_mut().zip(p) {
            *a += u64::from(x);
        }
    }
    acc.iter().map(|&a| u128::from(a)).sum()
}
