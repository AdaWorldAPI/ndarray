//! 64-lane byte permutations as **composable maps**, not data moves.
//!
//! A permutation here is a coordinate map. Composing two of them is one
//! cross-lane byte permute on the *index* register (`VPERMB` where the CPU has
//! AVX-512 VBMI, the facade's fallback elsewhere); the payload is untouched.
//! Data moves exactly once, at a call whose name starts with `materialize`.
//!
//! The shape follows NNUE's lazy accumulator update: each step records a delta
//! (its index map), the deltas compose, and the expensive thing (touching the
//! payload) happens only when the result is actually read.
//!
//! Three layers:
//!
//! - [`Perm64`] — one bijection on 64 byte lanes. [`Perm64::then`] composes,
//!   [`Perm64::inverse`] inverts, [`Perm64::conjugate_mask`] carries a lane
//!   mask through the map without moving any data.
//! - [`PermTable12`] — twelve fixed base steps selected by a 12-bit code
//!   (bit *i* = "step *i* applies", steps applied in index order). Stored as
//!   two 64-entry half tables (8 KiB) instead of one 4096-entry flat table
//!   (256 KiB): one extra compose per lookup, and the tables fit in L1.
//! - [`PermChain`] — a lazy accumulator of steps. It holds one [`Perm64`] no
//!   matter how many steps were pushed, and it has no method that takes a
//!   payload except [`PermChain::materialize_into`]. Eager application is not
//!   expressible through this type.
//!
//! All lane work goes through [`crate::simd::U8x64`]; this module contains no
//! intrinsics.

use crate::simd::U8x64;

/// Number of byte lanes in one permutation.
pub const LANES: usize = 64;

/// Why a byte array was rejected as a permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermError {
    /// `indices[at]` is not a lane index (it is `>= 64`).
    OutOfRange {
        /// Position of the offending entry.
        at: usize,
        /// The value found there.
        value: u8,
    },
    /// Lane `value` is selected twice, so some other lane is never selected.
    Duplicate {
        /// Position of the second occurrence.
        at: usize,
        /// The lane selected twice.
        value: u8,
    },
}

impl core::fmt::Display for PermError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PermError::OutOfRange { at, value } => {
                write!(f, "index {value} at position {at} is not a lane (must be < 64)")
            }
            PermError::Duplicate { at, value } => {
                write!(f, "lane {value} selected twice (second time at position {at})")
            }
        }
    }
}

impl std::error::Error for PermError {}

/// A bijection on 64 byte lanes, stored as its gather index vector.
///
/// Applying it to a block `src` produces `out[i] = src[idx[i]]`. The invariant
/// that `idx` is a permutation of `0..64` is established at construction and
/// preserved by every method, so [`Perm64::inverse`] always exists.
///
/// # Example
///
/// ```
/// use ndarray::hpc::perm::Perm64;
///
/// let rot = Perm64::rotate(1); // out[i] = src[(i + 1) % 64]
/// let back = Perm64::rotate(63);
/// assert_eq!(rot.then(back), Perm64::IDENTITY);
/// assert_eq!(rot.inverse(), back);
/// ```
#[derive(Clone, Copy)]
pub struct Perm64 {
    idx: [u8; LANES],
}

impl PartialEq for Perm64 {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for Perm64 {}

impl core::fmt::Debug for Perm64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Perm64").field(&&self.idx[..]).finish()
    }
}

impl Default for Perm64 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

const fn identity_indices() -> [u8; LANES] {
    let mut idx = [0u8; LANES];
    let mut i = 0;
    while i < LANES {
        idx[i] = i as u8;
        i += 1;
    }
    idx
}

impl Perm64 {
    /// The permutation that moves nothing.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// assert_eq!(Perm64::IDENTITY.indices()[5], 5);
    /// ```
    pub const IDENTITY: Self = Self {
        idx: identity_indices(),
    };

    /// Build from a gather index vector, checking that it is a bijection.
    ///
    /// ```
    /// use ndarray::hpc::perm::{Perm64, PermError};
    ///
    /// let mut idx: [u8; 64] = core::array::from_fn(|i| i as u8);
    /// idx.swap(0, 1);
    /// assert!(Perm64::from_indices(idx).is_ok());
    ///
    /// idx[2] = 0; // lane 0 now selected twice
    /// assert!(matches!(Perm64::from_indices(idx), Err(PermError::Duplicate { .. })));
    /// ```
    pub fn from_indices(idx: [u8; LANES]) -> Result<Self, PermError> {
        let mut seen = 0u64;
        for (at, &value) in idx.iter().enumerate() {
            if value as usize >= LANES {
                return Err(PermError::OutOfRange { at, value });
            }
            let bit = 1u64 << value;
            if seen & bit != 0 {
                return Err(PermError::Duplicate { at, value });
            }
            seen |= bit;
        }
        Ok(Self { idx })
    }

    /// Rotation by `k` lanes: `out[i] = src[(i + k) % 64]`.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// assert_eq!(Perm64::rotate(3).indices()[63], 2);
    /// ```
    pub fn rotate(k: usize) -> Self {
        Self {
            idx: core::array::from_fn(|i| ((i + k) % LANES) as u8),
        }
    }

    /// The gather index vector (`out[i] = src[indices()[i]]`).
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// assert_eq!(Perm64::IDENTITY.indices().len(), 64);
    /// ```
    pub fn indices(&self) -> [u8; LANES] {
        self.idx
    }

    /// Compose: the map that applies `self` first and `next` second.
    ///
    /// Applying `self` gives `d1[i] = d[s[i]]`; applying `next` to that gives
    /// `d2[i] = d1[n[i]] = d[s[n[i]]]`. So the composed index vector is `s`
    /// gathered by `n` — one cross-lane byte permute on the index register.
    /// No payload is read or written.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// assert_eq!(Perm64::rotate(2).then(Perm64::rotate(5)), Perm64::rotate(7));
    /// ```
    #[inline]
    pub fn then(self, next: Perm64) -> Perm64 {
        let composed = U8x64::from_array(self.idx).permute_bytes(U8x64::from_array(next.idx));
        // A composition of two bijections is a bijection; no re-check needed.
        Perm64 {
            idx: composed.to_array(),
        }
    }

    /// The map that undoes `self`: `self.then(self.inverse()) == IDENTITY`.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// let p = Perm64::rotate(9);
    /// assert_eq!(p.then(p.inverse()), Perm64::IDENTITY);
    /// assert_eq!(p.inverse().then(p), Perm64::IDENTITY);
    /// ```
    pub fn inverse(&self) -> Perm64 {
        let mut inv = [0u8; LANES];
        for (i, &s) in self.idx.iter().enumerate() {
            inv[s as usize] = i as u8;
        }
        Perm64 { idx: inv }
    }

    /// Carry a lane mask through the map instead of moving the data.
    ///
    /// If `mask` marks lanes of the *source* block, the result marks the same
    /// bytes at their positions in the *permuted* block: bit `i` of the result
    /// is bit `idx[i]` of `mask`. This lets a mask meet the payload at the one
    /// materialization, rather than forcing the payload to move first.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// // Source lane 1 lands at position 0 under rotate(1).
    /// assert_eq!(Perm64::rotate(1).conjugate_mask(0b10), 0b1);
    /// ```
    #[inline]
    pub fn conjugate_mask(&self, mask: u64) -> u64 {
        let flags: [u8; LANES] = core::array::from_fn(|i| if mask >> i & 1 == 1 { 0x80 } else { 0 });
        U8x64::from_array(flags)
            .permute_bytes(U8x64::from_array(self.idx))
            .movemask()
    }

    /// The one data move: write `out[i] = src[idx[i]]` for one 64-byte block.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// let src: [u8; 64] = core::array::from_fn(|i| i as u8);
    /// let mut out = [0u8; 64];
    /// Perm64::rotate(1).materialize_into(&src, &mut out);
    /// assert_eq!(out[0], 1);
    /// assert_eq!(out[63], 0);
    /// ```
    #[inline]
    pub fn materialize_into(&self, src: &[u8; LANES], out: &mut [u8; LANES]) {
        *out = U8x64::from_array(*src)
            .permute_bytes(U8x64::from_array(self.idx))
            .to_array();
    }

    /// Apply the map independently to every 64-byte block of `src`.
    ///
    /// # Panics
    ///
    /// If the lengths differ or are not a multiple of 64.
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// let src: Vec<u8> = (0..128).map(|i| i as u8).collect();
    /// let mut out = vec![0u8; 128];
    /// Perm64::rotate(1).materialize_blocks_into(&src, &mut out);
    /// assert_eq!((out[0], out[64]), (1, 65));
    /// ```
    pub fn materialize_blocks_into(&self, src: &[u8], out: &mut [u8]) {
        assert_eq!(src.len(), out.len(), "source and output lengths differ");
        assert_eq!(src.len() % LANES, 0, "length must be a multiple of 64");
        let idx = U8x64::from_array(self.idx);
        for (s, o) in src.chunks_exact(LANES).zip(out.chunks_exact_mut(LANES)) {
            U8x64::from_slice(s).permute_bytes(idx).copy_to_slice(o);
        }
    }
}

/// Number of base steps a [`PermTable12`] selects between.
pub const STEPS: usize = 12;

/// Twelve fixed base permutations, selected by a 12-bit code.
///
/// Bit `i` of the code says whether step `i` applies; the selected steps are
/// applied in increasing `i`. Because the order is fixed, the code splits
/// exactly into a low half (steps 0–5) and a high half (steps 6–11), and
/// `for_code(c) == lo[c & 63].then(hi[c >> 6])` holds even when the steps
/// do not commute.
///
/// Storage is two 64-entry tables of [`Perm64`] (8 KiB) rather than all 4096
/// compositions (256 KiB): one extra compose per lookup in exchange for tables
/// that stay in L1.
///
/// # Example
///
/// ```
/// use ndarray::hpc::perm::{Perm64, PermTable12};
///
/// let steps: [Perm64; 12] = core::array::from_fn(|i| Perm64::rotate(1 << (i % 6)));
/// let table = PermTable12::new(steps);
/// assert_eq!(table.for_code(0), Perm64::IDENTITY);
/// assert_eq!(table.for_code(0b1), Perm64::rotate(1));
/// ```
#[derive(Clone)]
pub struct PermTable12 {
    base: [Perm64; STEPS],
    lo: Box<[Perm64; 64]>,
    hi: Box<[Perm64; 64]>,
}

impl PermTable12 {
    /// Precompose the two half tables from the twelve base steps.
    pub fn new(base: [Perm64; STEPS]) -> Self {
        let half = |first: usize| -> Box<[Perm64; 64]> {
            let mut t = Box::new([Perm64::IDENTITY; 64]);
            for m in 1..64usize {
                // Build each entry from the entry with its highest set bit
                // removed, appending that step last: preserves index order.
                let top = 63 - (m as u64).leading_zeros() as usize;
                t[m] = t[m & !(1 << top)].then(base[first + top]);
            }
            t
        };
        let lo = half(0);
        let hi = half(6);
        Self { base, lo, hi }
    }

    /// The twelve base steps this table was built from.
    pub fn base(&self) -> &[Perm64; STEPS] {
        &self.base
    }

    /// The composed permutation for a 12-bit code (higher bits are ignored).
    ///
    /// ```
    /// use ndarray::hpc::perm::{Perm64, PermTable12};
    /// let t = PermTable12::new([Perm64::rotate(1); 12]);
    /// assert_eq!(t.for_code(0xFFF), Perm64::rotate(12));
    /// ```
    #[inline]
    pub fn for_code(&self, code: u16) -> Perm64 {
        let c = code as usize & 0xFFF;
        self.lo[c & 63].then(self.hi[c >> 6])
    }
}

/// A lazy chain of permutation steps: composes maps, moves data once.
///
/// The chain holds exactly one [`Perm64`] regardless of how many steps were
/// pushed. No method takes a payload except [`PermChain::materialize_into`]
/// and [`PermChain::materialize_blocks_into`], so a step cannot be applied to
/// data eagerly through this type.
///
/// # Example
///
/// ```
/// use ndarray::hpc::perm::{Perm64, PermChain};
///
/// let mut chain = PermChain::new();
/// for _ in 0..12 {
///     chain.push(Perm64::rotate(1)); // index work only
/// }
/// assert_eq!(chain.pending_steps(), 12);
///
/// let src: [u8; 64] = core::array::from_fn(|i| i as u8);
/// let mut out = [0u8; 64];
/// chain.materialize_into(&src, &mut out); // the one data move
/// assert_eq!(out[0], 12);
/// assert_eq!(chain.pending_steps(), 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PermChain {
    pending: Perm64,
    steps: u32,
}

impl PermChain {
    /// An empty chain (the identity, zero pending steps).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step. Composes on the index register; touches no payload.
    #[inline]
    pub fn push(&mut self, step: Perm64) {
        self.pending = self.pending.then(step);
        self.steps += 1;
    }

    /// Append the step a [`PermTable12`] assigns to `code`.
    #[inline]
    pub fn push_code(&mut self, table: &PermTable12, code: u16) {
        self.push(table.for_code(code));
    }

    /// The composed map of every step pushed since the last materialization.
    pub fn pending(&self) -> Perm64 {
        self.pending
    }

    /// How many steps have been composed since the last materialization.
    pub fn pending_steps(&self) -> u32 {
        self.steps
    }

    /// Apply every pending step to one block in a single move, then reset.
    pub fn materialize_into(&mut self, src: &[u8; LANES], out: &mut [u8; LANES]) {
        self.pending.materialize_into(src, out);
        *self = Self::new();
    }

    /// Apply every pending step to each 64-byte block in a single move per
    /// block, then reset.
    ///
    /// # Panics
    ///
    /// As [`Perm64::materialize_blocks_into`].
    pub fn materialize_blocks_into(&mut self, src: &[u8], out: &mut [u8]) {
        self.pending.materialize_blocks_into(src, out);
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random permutation (Fisher–Yates over SplitMix64).
    fn shuffled(seed: u64) -> Perm64 {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut idx = identity_indices();
        for i in (1..LANES).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        Perm64::from_indices(idx).unwrap()
    }

    /// Scalar oracle: apply one map to one block.
    fn apply(p: &Perm64, src: &[u8; LANES]) -> [u8; LANES] {
        core::array::from_fn(|i| src[p.idx[i] as usize])
    }

    fn payload() -> [u8; LANES] {
        core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
    }

    #[test]
    fn then_matches_applying_twice() {
        for seed in 0..200 {
            let (a, b) = (shuffled(seed), shuffled(seed + 1000));
            let d = payload();
            assert_eq!(apply(&a.then(b), &d), apply(&b, &apply(&a, &d)), "seed {seed}");
        }
    }

    #[test]
    fn then_is_not_commutative_on_these_inputs() {
        // Guards against a `then` that silently swaps its operands: with
        // non-commuting inputs the two orders must differ.
        let (a, b) = (shuffled(1), shuffled(2));
        assert_ne!(a.then(b), b.then(a));
    }

    #[test]
    fn inverse_undoes_both_sides() {
        for seed in 0..100 {
            let p = shuffled(seed);
            assert_eq!(p.then(p.inverse()), Perm64::IDENTITY);
            assert_eq!(p.inverse().then(p), Perm64::IDENTITY);
        }
    }

    #[test]
    fn from_indices_rejects_each_malformed_shape_and_accepts_a_real_one() {
        let mut idx = identity_indices();
        assert!(Perm64::from_indices(idx).is_ok());
        idx[10] = 64;
        assert_eq!(Perm64::from_indices(idx), Err(PermError::OutOfRange { at: 10, value: 64 }));
        idx[10] = 3;
        assert_eq!(Perm64::from_indices(idx), Err(PermError::Duplicate { at: 10, value: 3 }));
    }

    #[test]
    fn conjugate_mask_tracks_the_bytes_it_marks() {
        for seed in 0..100 {
            let p = shuffled(seed);
            let mask = 0xA5A5_0F0F_3C3C_9999u64.rotate_left(seed as u32);
            let d = payload();
            let out = apply(&p, &d);
            let carried = p.conjugate_mask(mask);
            for i in 0..LANES {
                // The byte now at position i came from source lane idx[i].
                let src_lane = p.idx[i] as usize;
                assert_eq!(carried >> i & 1, mask >> src_lane & 1, "seed {seed} lane {i}");
                assert_eq!(out[i], d[src_lane]);
            }
        }
    }

    /// The flat reference: apply the selected base steps one by one, in order.
    fn flat(base: &[Perm64; STEPS], code: u16) -> Perm64 {
        let mut p = Perm64::IDENTITY;
        for (i, step) in base.iter().enumerate() {
            if code >> i & 1 == 1 {
                p = p.then(*step);
            }
        }
        p
    }

    #[test]
    fn split_table_equals_the_flat_composition_for_every_code() {
        let base: [Perm64; STEPS] = core::array::from_fn(|i| shuffled(77 + i as u64));
        let table = PermTable12::new(base);
        for code in 0..4096u16 {
            assert_eq!(table.for_code(code), flat(&base, code), "code {code:#05x}");
        }
    }

    #[test]
    fn split_table_respects_step_order_across_the_half_boundary() {
        // Steps 5 and 6 sit on opposite sides of the split. With
        // non-commuting steps, the table must apply 5 before 6.
        let base: [Perm64; STEPS] = core::array::from_fn(|i| shuffled(500 + i as u64));
        assert_ne!(base[5].then(base[6]), base[6].then(base[5]));
        let table = PermTable12::new(base);
        assert_eq!(table.for_code((1 << 5) | (1 << 6)), base[5].then(base[6]));
    }

    #[test]
    fn chain_moves_data_once_and_equals_sequential_application() {
        let steps: Vec<Perm64> = (0..12).map(|s| shuffled(300 + s)).collect();
        let d = payload();

        let mut sequential = d;
        for s in &steps {
            sequential = apply(s, &sequential);
        }

        let mut chain = PermChain::new();
        for s in &steps {
            chain.push(*s);
        }
        assert_eq!(chain.pending_steps(), 12);
        // The chain holds one map and a counter, whatever its length.
        assert_eq!(core::mem::size_of::<PermChain>(), core::mem::size_of::<Perm64>() + core::mem::size_of::<u32>());
        let mut out = [0u8; LANES];
        chain.materialize_into(&d, &mut out);
        assert_eq!(out, sequential);
        assert_eq!(chain, PermChain::new(), "materialization resets the chain");
    }

    #[test]
    fn chain_of_codes_matches_sequential_codes() {
        let base: [Perm64; STEPS] = core::array::from_fn(|i| shuffled(900 + i as u64));
        let table = PermTable12::new(base);
        let codes = [0x001u16, 0xFFF, 0x0A5, 0x840, 0x3C3, 0x000, 0x7E1];
        let d = payload();

        let mut sequential = d;
        for &c in &codes {
            sequential = apply(&flat(&base, c), &sequential);
        }
        let mut chain = PermChain::new();
        for &c in &codes {
            chain.push_code(&table, c);
        }
        let mut out = [0u8; LANES];
        chain.materialize_into(&d, &mut out);
        assert_eq!(out, sequential);
    }

    #[test]
    fn blocks_are_permuted_independently() {
        let p = shuffled(42);
        let src: Vec<u8> = (0..LANES * 3).map(|i| (i * 7 % 251) as u8).collect();
        let mut out = vec![0u8; src.len()];
        p.materialize_blocks_into(&src, &mut out);
        for b in 0..3 {
            let block: [u8; LANES] = src[b * LANES..(b + 1) * LANES].try_into().unwrap();
            assert_eq!(&out[b * LANES..(b + 1) * LANES], &apply(&p, &block)[..]);
        }
    }
}
