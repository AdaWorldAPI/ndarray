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

    /// The map `R` that, applied first and followed by `basis`, gives `self`:
    /// `self.relative_to(basis).then(basis) == self`.
    ///
    /// This is what lets two views in different coordinate systems meet
    /// without normalizing both. If `a` is seen through `P` and `b` through
    /// `Q`, a lane-wise operation `f` satisfies
    /// `f(P·a, Q·b) == P·f(a, R·b)` with `R = Q.relative_to(P)`: only `b`
    /// moves, and only by `R`, and the result stays in basis `P`.
    ///
    /// (`then` composes left to right, so in right-to-left notation this is
    /// `R = P⁻¹ ∘ Q` only if you read `∘` as "applied after"; the method
    /// name and the round-trip law above are the unambiguous statement.)
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// let (p, q) = (Perm64::rotate(3), Perm64::rotate(10));
    /// assert_eq!(q.relative_to(p).then(p), q);
    /// assert_eq!(p.relative_to(p), Perm64::IDENTITY);
    /// ```
    pub fn relative_to(&self, basis: Perm64) -> Perm64 {
        self.then(basis.inverse())
    }

    /// Apply the map to one block and return the result (a data move).
    ///
    /// ```
    /// use ndarray::hpc::perm::Perm64;
    /// let src: [u8; 64] = core::array::from_fn(|i| i as u8);
    /// assert_eq!(Perm64::rotate(2).materialize(&src)[0], 2);
    /// ```
    #[inline]
    pub fn materialize(&self, src: &[u8; LANES]) -> [u8; LANES] {
        let mut out = [0u8; LANES];
        self.materialize_into(src, &mut out);
        out
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

    /// As [`PermTable12::for_code`], but without an index that depends on
    /// `code`: every entry of both half tables is read and the wanted one is
    /// selected with a mask, so the memory access pattern is the same for
    /// every code.
    ///
    /// Best effort: the source has no code-dependent branch or index, but Rust
    /// gives no constant-time guarantee against the optimizer. It costs about
    /// 8 KiB of reads per lookup instead of 128 bytes.
    ///
    /// ```
    /// use ndarray::hpc::perm::{Perm64, PermTable12};
    /// let t = PermTable12::new([Perm64::rotate(1); 12]);
    /// assert_eq!(t.for_code_constant_time(0x0A5), t.for_code(0x0A5));
    /// ```
    pub fn for_code_constant_time(&self, code: u16) -> Perm64 {
        let c = code as usize & 0xFFF;
        select_constant_time(&self.lo, c & 63).then(select_constant_time(&self.hi, c >> 6))
    }
}

/// Read every entry and keep the one at `want`, with no `want`-dependent
/// branch or index.
fn select_constant_time(table: &[Perm64; 64], want: usize) -> Perm64 {
    let mut acc = [0u8; LANES];
    for (m, entry) in table.iter().enumerate() {
        // 0xFF when m == want, 0x00 otherwise, computed without a branch.
        let hit = core::hint::black_box(((m ^ want) == 0) as u8).wrapping_neg();
        for (a, &e) in acc.iter_mut().zip(entry.idx.iter()) {
            *a |= e & hit;
        }
    }
    Perm64 { idx: acc }
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

// ---------------------------------------------------------------------------
// Batches of codes: the live set, the request remap, the schedule.
// ---------------------------------------------------------------------------

/// How a [`PermBatch`] may execute. There is no default: the caller must say
/// whether the codes are public.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// The codes are public. Duplicate codes are composed once, and the work
    /// order follows the set of distinct codes. Timing and memory access
    /// depend on the codes.
    Deduplicate,
    /// The codes may be secret. No live set is built, nothing is
    /// deduplicated or reordered, and every table lookup reads all entries
    /// ([`PermTable12::for_code_constant_time`]). One compose per request, in
    /// request order.
    ConstantTime,
}

/// The set of distinct live codes: a 4096-bit occupancy mask (512 bytes) plus
/// the distinct codes in first-appearance order.
///
/// This is a SET. It does not remember how often a code was requested or in
/// what order; [`PermBatch`] carries that. Use the field on its own only for
/// questions about the code domain ("which combinations are live", "how
/// many").
///
/// ```
/// use ndarray::hpc::perm::PermField;
/// let f = PermField::from_codes(&[0xA7A, 0x311, 0xA7A, 0xA72, 0x311]);
/// assert_eq!(f.len(), 3);
/// assert_eq!(f.distinct(), &[0xA7A, 0x311, 0xA72]);
/// assert!(f.contains(0x311) && !f.contains(0x312));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermField {
    live: [u64; 64],
    distinct: Vec<u16>,
}

impl PermField {
    /// Collect the distinct 12-bit codes (higher bits are ignored).
    pub fn from_codes(codes: &[u16]) -> Self {
        let mut live = [0u64; 64];
        let mut distinct = Vec::new();
        for &code in codes {
            let c = code & 0xFFF;
            let (word, bit) = ((c >> 6) as usize, 1u64 << (c & 63));
            if live[word] & bit == 0 {
                live[word] |= bit;
                distinct.push(c);
            }
        }
        Self { live, distinct }
    }

    /// Whether `code` (masked to 12 bits) is live.
    pub fn contains(&self, code: u16) -> bool {
        let c = code & 0xFFF;
        self.live[(c >> 6) as usize] >> (c & 63) & 1 == 1
    }

    /// Number of distinct live codes (the popcount of the occupancy mask).
    pub fn len(&self) -> usize {
        self.live.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether no code is live.
    pub fn is_empty(&self) -> bool {
        self.distinct.is_empty()
    }

    /// The occupancy mask: word `h` bit `l` is code `(h << 6) | l`.
    pub fn live_words(&self) -> &[u64; 64] {
        &self.live
    }

    /// The distinct codes in first-appearance order.
    pub fn distinct(&self) -> &[u16] {
        &self.distinct
    }
}

/// A request stream of 12-bit codes, with its execution policy.
///
/// Under [`Schedule::Deduplicate`] it holds a [`PermField`] plus, for every
/// request, the index of its distinct code — so multiplicity and order
/// survive deduplication. Under [`Schedule::ConstantTime`] it holds only the
/// requests.
///
/// ```
/// use ndarray::hpc::perm::{PermBatch, Schedule};
/// let b = PermBatch::new(&[0xA7A, 0x311, 0xA7A, 0xA72, 0x311], Schedule::Deduplicate);
/// assert_eq!(b.len(), 5);
/// assert_eq!(b.request_to_distinct(), Some(&[0, 1, 0, 2, 1][..]));
/// ```
#[derive(Clone, Debug)]
pub struct PermBatch {
    requests: Vec<u16>,
    schedule: Schedule,
    field: Option<PermField>,
    request_to_distinct: Vec<u32>,
}

impl PermBatch {
    /// Record a request stream under a schedule.
    pub fn new(codes: &[u16], schedule: Schedule) -> Self {
        let requests: Vec<u16> = codes.iter().map(|c| c & 0xFFF).collect();
        match schedule {
            Schedule::ConstantTime => Self {
                requests,
                schedule,
                field: None,
                request_to_distinct: Vec::new(),
            },
            Schedule::Deduplicate => {
                let field = PermField::from_codes(&requests);
                let mut slot = vec![u32::MAX; 4096];
                for (i, &c) in field.distinct().iter().enumerate() {
                    slot[c as usize] = i as u32;
                }
                let request_to_distinct = requests.iter().map(|&c| slot[c as usize]).collect();
                Self {
                    requests,
                    schedule,
                    field: Some(field),
                    request_to_distinct,
                }
            }
        }
    }

    /// Number of requests (not distinct codes).
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether there are no requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// The execution policy this batch was built with.
    pub fn schedule(&self) -> Schedule {
        self.schedule
    }

    /// The live set; `None` under [`Schedule::ConstantTime`].
    pub fn field(&self) -> Option<&PermField> {
        self.field.as_ref()
    }

    /// For each request, the index of its distinct code; `None` under
    /// [`Schedule::ConstantTime`].
    pub fn request_to_distinct(&self) -> Option<&[u32]> {
        self.field.as_ref().map(|_| &self.request_to_distinct[..])
    }

    /// Compose the permutation for every request.
    ///
    /// Deduplicate: one compose per distinct code. ConstantTime: one
    /// constant-time lookup and compose per request, in request order.
    pub fn compose(&self, table: &PermTable12) -> ComposedBatch {
        match self.schedule {
            Schedule::Deduplicate => {
                let field = self
                    .field
                    .as_ref()
                    .expect("Deduplicate batches carry a field");
                let perms: Vec<Perm64> = field
                    .distinct()
                    .iter()
                    .map(|&c| table.for_code(c))
                    .collect();
                ComposedBatch {
                    compositions: perms.len(),
                    perms,
                    request_to_perm: self.request_to_distinct.clone(),
                }
            }
            Schedule::ConstantTime => {
                let perms: Vec<Perm64> = self
                    .requests
                    .iter()
                    .map(|&c| table.for_code_constant_time(c))
                    .collect();
                ComposedBatch {
                    compositions: perms.len(),
                    request_to_perm: (0..perms.len() as u32).collect(),
                    perms,
                }
            }
        }
    }

    /// Answer a permutation-invariant fold for every request at once.
    ///
    /// Takes no table: because `f` is [`PermInvariant`], `f(P·data) ==
    /// f(data)` for every request's `P`, so no permutation is composed or
    /// applied. The single returned value is every request's answer.
    ///
    /// ```
    /// use ndarray::hpc::perm::{CountNonzero, PermBatch, Schedule};
    /// let data: [u8; 64] = core::array::from_fn(|i| (i % 3) as u8);
    /// let b = PermBatch::new(&[1, 2, 3], Schedule::Deduplicate);
    /// assert_eq!(b.fold_invariant(&CountNonzero, &data), 42);
    /// ```
    pub fn fold_invariant<F: PermInvariant>(&self, f: &F, data: &[u8; LANES]) -> F::Out {
        f.fold(data)
    }
}

/// The permutations of a composed [`PermBatch`], one per request.
#[derive(Clone, Debug)]
pub struct ComposedBatch {
    perms: Vec<Perm64>,
    request_to_perm: Vec<u32>,
    compositions: usize,
}

impl ComposedBatch {
    /// Number of requests.
    pub fn len(&self) -> usize {
        self.request_to_perm.len()
    }

    /// Whether there are no requests.
    pub fn is_empty(&self) -> bool {
        self.request_to_perm.is_empty()
    }

    /// The permutation for request `i`.
    pub fn perm(&self, i: usize) -> Perm64 {
        self.perms[self.request_to_perm[i] as usize]
    }

    /// How many compositions building this batch performed.
    pub fn compositions(&self) -> usize {
        self.compositions
    }

    /// The owed data moves: `out[i]` is `src` permuted by request `i`'s map.
    ///
    /// # Panics
    ///
    /// If `out.len()` differs from the number of requests.
    pub fn materialize_into(&self, src: &[u8; LANES], out: &mut [[u8; LANES]]) {
        assert_eq!(out.len(), self.len(), "one output block per request");
        for (i, o) in out.iter_mut().enumerate() {
            self.perm(i).materialize_into(src, o);
        }
    }
}

// ---------------------------------------------------------------------------
// Operation properties: invariant, equivariant, or coordinate-sensitive.
// ---------------------------------------------------------------------------

/// A reduction over one 64-lane block.
pub trait LaneFold {
    /// The reduction's result type.
    type Out;
    /// Reduce one block.
    fn fold(&self, block: &[u8; LANES]) -> Self::Out;
}

/// Marker: `fold(P·x) == fold(x)` for every [`Perm64`] `P`.
///
/// An operation implements this to advertise that a permutation can be
/// skipped entirely before it. It holds for reductions over the plain
/// multiset of lane values (count, any, sum, min, max). It does NOT hold, and
/// must not be implemented, for reductions that read a lane position: masked
/// or indexed reductions, "first nonzero", prefix scans, anything weighted by
/// lane index. A mask that selects lanes is itself coordinate-sensitive —
/// carry it with [`Perm64::conjugate_mask`] instead.
pub trait PermInvariant: LaneFold {}

/// A lane-wise operation over `N` operand blocks.
pub trait LaneOp<const N: usize> {
    /// Combine the operands.
    fn apply(&self, operands: [&[u8; LANES]; N]) -> [u8; LANES];
}

/// Marker: `apply([P·a₀, …, P·aₙ]) == P·apply([a₀, …, aₙ])` for every
/// [`Perm64`] `P`.
///
/// Holds exactly when output lane `i` depends only on input lane `i` of each
/// operand, with the same function for every lane. Anything that reads a
/// neighbouring lane, or behaves differently per lane index, must not
/// implement it.
pub trait PermEquivariant<const N: usize>: LaneOp<N> {}

/// Number of nonzero lanes.
#[derive(Clone, Copy, Debug, Default)]
pub struct CountNonzero;
impl LaneFold for CountNonzero {
    type Out = u32;
    fn fold(&self, b: &[u8; LANES]) -> u32 {
        b.iter().filter(|&&v| v != 0).count() as u32
    }
}
impl PermInvariant for CountNonzero {}

/// Whether any lane is nonzero.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnyNonzero;
impl LaneFold for AnyNonzero {
    type Out = bool;
    fn fold(&self, b: &[u8; LANES]) -> bool {
        b.iter().any(|&v| v != 0)
    }
}
impl PermInvariant for AnyNonzero {}

/// Sum of lane values.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumLanes;
impl LaneFold for SumLanes {
    type Out = u32;
    fn fold(&self, b: &[u8; LANES]) -> u32 {
        b.iter().map(|&v| v as u32).sum()
    }
}
impl PermInvariant for SumLanes {}

/// Smallest lane value.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinLane;
impl LaneFold for MinLane {
    type Out = u8;
    fn fold(&self, b: &[u8; LANES]) -> u8 {
        b.iter().copied().min().unwrap_or(0)
    }
}
impl PermInvariant for MinLane {}

/// Largest lane value.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxLane;
impl LaneFold for MaxLane {
    type Out = u8;
    fn fold(&self, b: &[u8; LANES]) -> u8 {
        b.iter().copied().max().unwrap_or(0)
    }
}
impl PermInvariant for MaxLane {}

macro_rules! lanewise_binary {
    ($name:ident, $doc:literal, $op:tt) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl LaneOp<2> for $name {
            fn apply(&self, [a, b]: [&[u8; LANES]; 2]) -> [u8; LANES] {
                (U8x64::from_array(*a) $op U8x64::from_array(*b)).to_array()
            }
        }
        impl PermEquivariant<2> for $name {}
    };
}
lanewise_binary!(LaneAnd, "Lane-wise AND.", &);
lanewise_binary!(LaneOr, "Lane-wise OR.", |);
lanewise_binary!(LaneXor, "Lane-wise XOR.", ^);

/// Lane-wise three-input boolean function, bit by bit, with the ternlog
/// immediate convention: output bit = bit `(a << 2) | (b << 1) | c` of `imm`.
#[derive(Clone, Copy, Debug)]
pub struct LaneTernlog(pub u8);
impl LaneOp<3> for LaneTernlog {
    fn apply(&self, [a, b, c]: [&[u8; LANES]; 3]) -> [u8; LANES] {
        core::array::from_fn(|i| {
            let mut out = 0u8;
            for bit in 0..8 {
                let sel = ((a[i] >> bit & 1) << 2) | ((b[i] >> bit & 1) << 1) | (c[i] >> bit & 1);
                out |= (self.0 >> sel & 1) << bit;
            }
            out
        })
    }
}
impl PermEquivariant<3> for LaneTernlog {}

/// A result that is still in a coordinate system: the true answer is
/// `basis` applied to `data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InBasis {
    /// The map the result is still owed.
    pub basis: Perm64,
    /// The result in unpermuted coordinates.
    pub data: [u8; LANES],
    /// How many operands had to be moved to meet the basis.
    pub aligned_moves: usize,
}

impl InBasis {
    /// The terminal move, if the consumer really needs coordinates.
    pub fn materialize(&self) -> [u8; LANES] {
        self.basis.materialize(&self.data)
    }
}

/// Combine operands that are each seen through their own permutation,
/// without normalizing them.
///
/// `operands[i] = (Pᵢ, aᵢ)` means "the block `aᵢ` viewed through `Pᵢ`".
/// Operand 0's map is the basis; every other operand is moved once by
/// `Pᵢ.relative_to(P₀)` — and not at all when `Pᵢ == P₀`. The result stays in
/// basis `P₀`; call [`InBasis::materialize`] only if coordinates are needed.
///
/// ```
/// use ndarray::hpc::perm::{combine_in_basis, LaneXor, Perm64};
/// let a = [1u8; 64];
/// let b: [u8; 64] = core::array::from_fn(|i| i as u8);
/// let (p, q) = (Perm64::rotate(1), Perm64::rotate(5));
/// let r = combine_in_basis(&LaneXor, [(p, &a), (q, &b)]);
/// assert_eq!(r.aligned_moves, 1);
/// let expected: [u8; 64] = core::array::from_fn(|i| p.materialize(&a)[i] ^ q.materialize(&b)[i]);
/// assert_eq!(r.materialize(), expected);
/// ```
pub fn combine_in_basis<Op: PermEquivariant<N>, const N: usize>(
    op: &Op, operands: [(Perm64, &[u8; LANES]); N],
) -> InBasis {
    let basis = operands[0].0;
    let mut aligned_moves = 0;
    let moved: [[u8; LANES]; N] = core::array::from_fn(|i| {
        let (p, a) = operands[i];
        if p == basis {
            *a
        } else {
            aligned_moves += 1;
            p.relative_to(basis).materialize(a)
        }
    });
    let refs: [&[u8; LANES]; N] = core::array::from_fn(|i| &moved[i]);
    InBasis {
        basis,
        data: op.apply(refs),
        aligned_moves,
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
    fn table(seed: u64) -> PermTable12 {
        PermTable12::new(core::array::from_fn(|i| shuffled(seed + i as u64)))
    }

    #[test]
    fn dedup_keeps_multiplicity_and_order_through_the_remap() {
        let t = table(1200);
        let codes = [0xA7A, 0x311, 0xA7A, 0xA72, 0x311];
        let batch = PermBatch::new(&codes, Schedule::Deduplicate);
        assert_eq!(batch.field().unwrap().len(), 3);
        assert_eq!(batch.request_to_distinct().unwrap(), &[0, 1, 0, 2, 1]);

        let composed = batch.compose(&t);
        assert_eq!(composed.compositions(), 3, "one compose per distinct code");
        let d = payload();
        let mut out = vec![[0u8; LANES]; codes.len()];
        composed.materialize_into(&d, &mut out);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(out[i], apply(&flat(t.base(), c), &d), "request {i}");
        }
    }

    #[test]
    fn constant_time_schedule_builds_no_field_and_composes_every_request() {
        let t = table(1300);
        let codes = [0xA7A, 0x311, 0xA7A, 0xA72, 0x311];
        let ct = PermBatch::new(&codes, Schedule::ConstantTime);
        assert!(ct.field().is_none() && ct.request_to_distinct().is_none());
        let composed = ct.compose(&t);
        assert_eq!(composed.compositions(), codes.len(), "no deduplication");
        let dedup = PermBatch::new(&codes, Schedule::Deduplicate).compose(&t);
        for i in 0..codes.len() {
            assert_eq!(composed.perm(i), dedup.perm(i), "request {i}");
        }
    }

    #[test]
    fn constant_time_lookup_matches_the_direct_lookup_for_every_code() {
        let t = table(1400);
        for code in 0..4096u16 {
            assert_eq!(t.for_code_constant_time(code), t.for_code(code), "code {code:#05x}");
        }
    }

    #[test]
    fn relative_to_round_trips_and_is_identity_on_itself() {
        for seed in 0..100 {
            let (p, q) = (shuffled(seed), shuffled(seed + 5000));
            assert_eq!(p.relative_to(p), Perm64::IDENTITY);
            assert_eq!(q.relative_to(p).then(p), q);
            assert_eq!(p.inverse().then(p), Perm64::IDENTITY);
            assert_eq!(p.then(p.inverse()), Perm64::IDENTITY);
        }
    }

    /// Normalizing every operand is the oracle `combine_in_basis` must match.
    fn normalized<Op: LaneOp<N>, const N: usize>(op: &Op, operands: [(Perm64, &[u8; LANES]); N]) -> [u8; LANES] {
        let moved: [[u8; LANES]; N] = core::array::from_fn(|i| operands[i].0.materialize(operands[i].1));
        op.apply(core::array::from_fn(|i| &moved[i]))
    }

    fn block(seed: u8) -> [u8; LANES] {
        core::array::from_fn(|i| (i as u8).wrapping_mul(seed | 1).wrapping_add(seed))
    }

    #[test]
    fn relative_basis_law_holds_for_every_equivariant_op() {
        for seed in 0..60u64 {
            let (p, q, r) = (shuffled(seed), shuffled(seed + 700), shuffled(seed + 1400));
            let (a, b, c) = (block(seed as u8), block(seed as u8 ^ 0x5A), block(seed as u8 ^ 0xC3));

            for got in [
                combine_in_basis(&LaneAnd, [(p, &a), (q, &b)]),
                combine_in_basis(&LaneOr, [(p, &a), (q, &b)]),
                combine_in_basis(&LaneXor, [(p, &a), (q, &b)]),
            ] {
                assert_eq!(got.basis, p);
                assert_eq!(got.aligned_moves, 1);
            }
            assert_eq!(
                combine_in_basis(&LaneAnd, [(p, &a), (q, &b)]).materialize(),
                normalized(&LaneAnd, [(p, &a), (q, &b)])
            );
            assert_eq!(
                combine_in_basis(&LaneOr, [(p, &a), (q, &b)]).materialize(),
                normalized(&LaneOr, [(p, &a), (q, &b)])
            );
            assert_eq!(
                combine_in_basis(&LaneXor, [(p, &a), (q, &b)]).materialize(),
                normalized(&LaneXor, [(p, &a), (q, &b)])
            );

            for imm in [0x96u8, 0xE8, 0xCA, 0x80, 0x1E] {
                let op = LaneTernlog(imm);
                let got = combine_in_basis(&op, [(p, &a), (q, &b), (r, &c)]);
                assert_eq!(got.aligned_moves, 2);
                assert_eq!(got.materialize(), normalized(&op, [(p, &a), (q, &b), (r, &c)]), "seed {seed} imm {imm:#x}");
            }
        }
    }

    #[test]
    fn shared_basis_moves_nothing_before_the_terminal() {
        let p = shuffled(9);
        let (a, b) = (block(1), block(2));
        let got = combine_in_basis(&LaneXor, [(p, &a), (p, &b)]);
        assert_eq!(got.aligned_moves, 0);
        assert_eq!(got.materialize(), normalized(&LaneXor, [(p, &a), (p, &b)]));
    }

    /// Reads lane 1 of `b` for output lane 0: NOT lane-wise. The law must
    /// fail for it, or the law test above could not catch a wrong marker.
    struct NeighbourXor;
    impl LaneOp<2> for NeighbourXor {
        fn apply(&self, [a, b]: [&[u8; LANES]; 2]) -> [u8; LANES] {
            core::array::from_fn(|i| a[i] ^ b[(i + 1) % LANES])
        }
    }

    #[test]
    fn a_non_lanewise_op_breaks_the_basis_law() {
        let (p, q) = (shuffled(3), shuffled(4));
        let (a, b) = (block(7), block(8));
        let moved_b = q.relative_to(p).materialize(&b);
        let in_basis = p.materialize(&NeighbourXor.apply([&a, &moved_b]));
        assert_ne!(in_basis, normalized(&NeighbourXor, [(p, &a), (q, &b)]));
    }

    fn invariant_matches_every_code<F: PermInvariant>(f: &F, t: &PermTable12, d: &[u8; LANES])
    where
        F::Out: PartialEq + core::fmt::Debug,
    {
        let all: Vec<u16> = (0..4096).collect();
        let batch = PermBatch::new(&all, Schedule::Deduplicate);
        let answer = batch.fold_invariant(f, d); // no table argument: no compose possible
        for &c in &all {
            assert_eq!(f.fold(&t.for_code(c).materialize(d)), answer, "code {c:#05x}");
        }
    }

    #[test]
    fn invariant_folds_answer_all_4096_codes_without_composing() {
        let t = table(1500);
        let d: [u8; LANES] = core::array::from_fn(|i| if i % 5 == 0 { 0 } else { (i * 13) as u8 });
        invariant_matches_every_code(&CountNonzero, &t, &d);
        invariant_matches_every_code(&AnyNonzero, &t, &d);
        invariant_matches_every_code(&SumLanes, &t, &d);
        invariant_matches_every_code(&MinLane, &t, &d);
        invariant_matches_every_code(&MaxLane, &t, &d);
    }

    #[test]
    fn a_position_reading_fold_is_not_invariant() {
        // "The value in lane 0" reads a coordinate. Some code must change it,
        // or the invariance test above could not catch a wrong marker.
        let t = table(1600);
        let d = payload();
        let first = d[0];
        assert!((0..4096u16).any(|c| t.for_code(c).materialize(&d)[0] != first));
    }
}
