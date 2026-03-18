//! Vector Symbolic Architecture: 10,000-dimensional binary operations.
//!
//! VSA is working memory. It fills (bundle), crystallizes (unbundle),
//! empties (clean), repeats. Like breathing.
//!
//! - bind/unbind: XOR (self-inverse, O(n))
//! - bundle: majority vote via i16 accumulator
//! - clean: iterative similarity search against codebook
//! - permute: cyclic shift for sequence encoding

/// VSA dimensionality: 10,000 bits.
pub const VSA_DIMS: usize = 10_000;

/// VSA bytes: ceil(10000/8) = 1250.
pub const VSA_BYTES: usize = 1250;

/// VSA u64 words: ceil(10000/64) = 157 (with 8 padding bits in last word).
pub const VSA_WORDS: usize = 157;

/// Number of meaningful bits in the last word: 10000 - 156*64 = 16.
const TAIL_BITS: usize = VSA_DIMS - (VSA_WORDS - 1) * 64;

/// Mask for the meaningful bits in the last word.
const TAIL_MASK: u64 = (1u64 << TAIL_BITS) - 1;

/// A 10,000-dimensional binary VSA vector.
///
/// Stored as 157 u64 words (10048 bits total), with only the first 10,000
/// bits meaningful. The upper 48 bits of the last word are always zero.
#[derive(Clone, PartialEq, Eq)]
pub struct VsaVector {
    /// 157 u64 words = 10048 bits, only first 10000 are meaningful.
    pub words: [u64; VSA_WORDS],
}

/// Accumulator for majority-vote bundling. Each dimension tracked as i16.
///
/// Add vectors with `+1` for set bits and `-1` for unset bits. Extract
/// produces a vector where positive tallies become 1, non-positive become 0.
pub struct VsaAccumulator {
    /// Per-dimension tally: +1 for set bit, -1 for unset.
    pub values: Vec<i16>, // length = VSA_DIMS
}

impl VsaVector {
    /// Create a zero vector (all bits unset).
    ///
    /// This is the identity element for XOR bind.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::VsaVector;
    /// let z = VsaVector::zero();
    /// assert!(z.words.iter().all(|&w| w == 0));
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self {
            words: [0u64; VSA_WORDS],
        }
    }

    /// Create a deterministic pseudo-random vector from a seed.
    ///
    /// Uses xorshift64 to fill all words, then masks the tail.
    /// The resulting vector has approximately 50% bit density.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::VsaVector;
    /// let a = VsaVector::random(42);
    /// let b = VsaVector::random(42);
    /// assert_eq!(a, b);  // deterministic
    /// ```
    pub fn random(seed: u64) -> Self {
        let mut state = seed;
        // Ensure nonzero state for xorshift
        if state == 0 {
            state = 0xDEAD_BEEF_CAFE_BABEu64;
        }
        let mut words = [0u64; VSA_WORDS];
        for w in words.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }
        words[VSA_WORDS - 1] &= TAIL_MASK;
        Self { words }
    }

    /// Create a VSA vector from a byte slice.
    ///
    /// If `data` is shorter than [`VSA_BYTES`] (1250), uses blake3 in XOF
    /// mode to expand it. If longer, only the first 1250 bytes are used.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::VsaVector;
    /// let v = VsaVector::from_bytes(b"hello world");
    /// ```
    pub fn from_bytes(data: &[u8]) -> Self {
        let buf = if data.len() >= VSA_BYTES {
            data[..VSA_BYTES].to_vec()
        } else {
            // Use blake3 XOF to expand short data to VSA_BYTES
            let mut hasher = blake3::Hasher::new();
            hasher.update(data);
            let mut reader = hasher.finalize_xof();
            let mut buf = vec![0u8; VSA_BYTES];
            reader.fill(&mut buf);
            buf
        };

        let mut words = [0u64; VSA_WORDS];
        for (i, w) in words.iter_mut().enumerate() {
            let offset = i * 8;
            if offset + 8 <= buf.len() {
                *w = u64::from_le_bytes([
                    buf[offset],
                    buf[offset + 1],
                    buf[offset + 2],
                    buf[offset + 3],
                    buf[offset + 4],
                    buf[offset + 5],
                    buf[offset + 6],
                    buf[offset + 7],
                ]);
            } else {
                // Last partial word
                let mut bytes = [0u8; 8];
                let remaining = buf.len() - offset;
                bytes[..remaining].copy_from_slice(&buf[offset..]);
                *w = u64::from_le_bytes(bytes);
            }
        }
        words[VSA_WORDS - 1] &= TAIL_MASK;
        Self { words }
    }

    /// Create a VSA vector from text using blake3 hash expansion.
    ///
    /// The text is hashed with blake3, then expanded via XOF mode to fill
    /// all 1250 bytes. Deterministic: same text always produces same vector.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::VsaVector;
    /// let a = VsaVector::from_text("cat");
    /// let b = VsaVector::from_text("cat");
    /// assert_eq!(a, b);
    /// ```
    pub fn from_text(text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        let mut reader = hasher.finalize_xof();
        let mut buf = vec![0u8; VSA_BYTES];
        reader.fill(&mut buf);
        Self::from_bytes(&buf)
    }

    /// Zero-copy view of the vector as a byte slice.
    ///
    /// Returns all `VSA_WORDS * 8` bytes (1256 bytes). The last 6 bytes
    /// contain only padding zeros.
    ///
    /// # Safety
    ///
    /// `[u64; VSA_WORDS]` is contiguous. `u8` has no stricter alignment
    /// than `u64`. Pointer cast is always valid.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: [u64; N] is contiguous, u64→u8 cast is alignment-safe.
        unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const u8, VSA_WORDS * 8)
        }
    }

    /// Population count: number of set bits (within the meaningful 10,000).
    #[inline]
    pub fn popcount(&self) -> u32 {
        super::bitwise::popcount_raw(self.as_bytes()) as u32
    }
}

impl std::fmt::Debug for VsaVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VsaVector[popcount={}, {:016x} {:016x} {:016x} {:016x} ...]",
            self.popcount(),
            self.words[0],
            self.words[1],
            self.words[2],
            self.words[3],
        )
    }
}

// ── Free functions ──────────────────────────────────────────────────

/// Bind two VSA vectors via XOR.
///
/// Binding is the fundamental association operation. It is commutative,
/// associative, and self-inverse: `bind(a, bind(a, b)) == b`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_bind};
/// let a = VsaVector::random(1);
/// let b = VsaVector::random(2);
/// let bound = vsa_bind(&a, &b);
/// let recovered = vsa_bind(&a, &bound);
/// assert_eq!(recovered, b);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn vsa_bind(a: &VsaVector, b: &VsaVector) -> VsaVector {
    let mut words = [0u64; VSA_WORDS];
    for i in 0..VSA_WORDS {
        words[i] = a.words[i] ^ b.words[i];
    }
    // Tail is already clean if both inputs are clean
    VsaVector { words }
}

/// Unbind a value from a bundle using XOR (self-inverse of bind).
///
/// `unbind(bind(key, value), key) == value`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_bind, vsa_unbind};
/// let key = VsaVector::random(10);
/// let val = VsaVector::random(20);
/// let bound = vsa_bind(&key, &val);
/// let recovered = vsa_unbind(&bound, &key);
/// assert_eq!(recovered, val);
/// ```
#[inline]
pub fn vsa_unbind(bundle: &VsaVector, key: &VsaVector) -> VsaVector {
    vsa_bind(bundle, key) // XOR is self-inverse
}

/// Bundle multiple vectors via majority vote.
///
/// Each bit position is tallied: +1 for set, -1 for unset.
/// The result bit is 1 if the tally is positive, 0 otherwise.
/// With odd count, the result is lossless for majority representation.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_bundle, vsa_similarity};
/// let a = VsaVector::random(1);
/// let b = VsaVector::random(2);
/// let bundled = vsa_bundle(&[a.clone(), a.clone(), a.clone(), b.clone()]);
/// assert!(vsa_similarity(&bundled, &a) > vsa_similarity(&bundled, &b));
/// ```
pub fn vsa_bundle(items: &[VsaVector]) -> VsaVector {
    if items.is_empty() {
        return VsaVector::zero();
    }
    let mut acc = VsaAccumulator::new();
    for item in items {
        acc.add(item);
    }
    acc.extract()
}

/// Normalized Hamming similarity in [0.0, 1.0].
///
/// Returns `1.0 - hamming_distance / VSA_DIMS`. Identical vectors yield 1.0,
/// complementary vectors yield approximately 0.0.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_similarity};
/// let a = VsaVector::random(42);
/// assert!((vsa_similarity(&a, &a) - 1.0).abs() < f32::EPSILON);
/// ```
#[inline]
pub fn vsa_similarity(a: &VsaVector, b: &VsaVector) -> f32 {
    1.0 - vsa_hamming(a, b) as f32 / VSA_DIMS as f32
}

/// Raw Hamming distance between two VSA vectors.
///
/// Counts the number of bit positions (out of 10,000) that differ.
/// Delegates to SIMD-accelerated bitwise operations.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_hamming};
/// let a = VsaVector::random(1);
/// assert_eq!(vsa_hamming(&a, &a), 0);
/// ```
pub fn vsa_hamming(a: &VsaVector, b: &VsaVector) -> u32 {
    super::bitwise::hamming_distance_raw(a.as_bytes(), b.as_bytes()) as u32
}

/// Cyclic bit permutation (left shift by `shift` positions within 10,000 bits).
///
/// Bit at position `i` moves to position `(i + shift) % VSA_DIMS`.
/// Used for sequence encoding: `permute(item, position)`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_permute, VSA_DIMS};
/// let v = VsaVector::random(1);
/// let shifted = vsa_permute(&v, 100);
/// let restored = vsa_permute(&shifted, VSA_DIMS - 100);
/// assert_eq!(restored, v);
/// ```
pub fn vsa_permute(v: &VsaVector, shift: usize) -> VsaVector {
    let shift = shift % VSA_DIMS;
    if shift == 0 {
        return v.clone();
    }

    let mut result = VsaVector::zero();

    // For each bit position in the result, find the source bit.
    // result_bit[i] = source_bit[(i - shift + VSA_DIMS) % VSA_DIMS]
    // Equivalently: source_bit[j] -> result_bit[(j + shift) % VSA_DIMS]
    //
    // We iterate source bits and place them in the destination.
    for src_bit in 0..VSA_DIMS {
        let dst_bit = (src_bit + shift) % VSA_DIMS;
        let src_word = src_bit / 64;
        let src_offset = src_bit % 64;
        let dst_word = dst_bit / 64;
        let dst_offset = dst_bit % 64;

        if v.words[src_word] & (1u64 << src_offset) != 0 {
            result.words[dst_word] |= 1u64 << dst_offset;
        }
    }

    // Tail should already be clean since we only write to positions < VSA_DIMS
    result
}

/// Encode an ordered sequence of vectors.
///
/// Each item is permuted by its index position, then all are bundled
/// via majority vote. This preserves order information:
/// `sequence([a, b]) != sequence([b, a])`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_sequence};
/// let a = VsaVector::random(1);
/// let b = VsaVector::random(2);
/// let seq_ab = vsa_sequence(&[a.clone(), b.clone()]);
/// let seq_ba = vsa_sequence(&[b, a]);
/// assert_ne!(seq_ab, seq_ba);
/// ```
pub fn vsa_sequence(items: &[VsaVector]) -> VsaVector {
    let permuted: Vec<VsaVector> = items
        .iter()
        .enumerate()
        .map(|(i, item)| vsa_permute(item, i))
        .collect();
    vsa_bundle(&permuted)
}

/// Find the closest codebook entry to a noisy vector.
///
/// Performs linear scan, returning the entry with minimum Hamming distance.
/// Returns `None` if the codebook is empty.
///
/// # Example
///
/// ```
/// use ndarray::hpc::vsa::{VsaVector, vsa_clean};
/// let codebook = vec![VsaVector::random(1), VsaVector::random(2)];
/// let noisy = VsaVector::random(1); // same seed → identical
/// let found = vsa_clean(&noisy, &codebook);
/// assert_eq!(found, Some(&codebook[0]));
/// ```
pub fn vsa_clean<'a>(dirty: &VsaVector, codebook: &'a [VsaVector]) -> Option<&'a VsaVector> {
    if codebook.is_empty() {
        return None;
    }
    let mut best_dist = u32::MAX;
    let mut best_entry = &codebook[0];
    for entry in codebook {
        let dist = vsa_hamming(dirty, entry);
        if dist < best_dist {
            best_dist = dist;
            best_entry = entry;
        }
    }
    Some(best_entry)
}

// ── Accumulator ─────────────────────────────────────────────────────

impl VsaAccumulator {
    /// Create a new zero accumulator.
    ///
    /// All 10,000 dimension tallies start at 0.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::VsaAccumulator;
    /// let acc = VsaAccumulator::new();
    /// assert!(acc.values.iter().all(|&v| v == 0));
    /// ```
    pub fn new() -> Self {
        Self {
            values: vec![0i16; VSA_DIMS],
        }
    }

    /// Accumulate a vector: +1 for each set bit, -1 for each unset bit.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::{VsaVector, VsaAccumulator};
    /// let mut acc = VsaAccumulator::new();
    /// let v = VsaVector::random(42);
    /// acc.add(&v);
    /// ```
    pub fn add(&mut self, v: &VsaVector) {
        for dim in 0..VSA_DIMS {
            let word = dim / 64;
            let bit = dim % 64;
            if v.words[word] & (1u64 << bit) != 0 {
                self.values[dim] += 1;
            } else {
                self.values[dim] -= 1;
            }
        }
    }

    /// Reverse-accumulate a vector: -1 for each set bit, +1 for each unset bit.
    ///
    /// This is the inverse of [`add`](Self::add), useful for removing a
    /// vector's contribution from a running bundle.
    pub fn subtract(&mut self, v: &VsaVector) {
        for dim in 0..VSA_DIMS {
            let word = dim / 64;
            let bit = dim % 64;
            if v.words[word] & (1u64 << bit) != 0 {
                self.values[dim] -= 1;
            } else {
                self.values[dim] += 1;
            }
        }
    }

    /// Extract the current tally as a VSA vector.
    ///
    /// Positive tallies become 1, non-positive tallies become 0.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::{VsaVector, VsaAccumulator};
    /// let mut acc = VsaAccumulator::new();
    /// let v = VsaVector::random(42);
    /// acc.add(&v);
    /// acc.add(&v);
    /// acc.add(&v);
    /// assert_eq!(acc.extract(), v);
    /// ```
    pub fn extract(&self) -> VsaVector {
        let mut words = [0u64; VSA_WORDS];
        for dim in 0..VSA_DIMS {
            if self.values[dim] > 0 {
                let word = dim / 64;
                let bit = dim % 64;
                words[word] |= 1u64 << bit;
            }
        }
        // Tail is clean because we only set bits for dim < VSA_DIMS
        VsaVector { words }
    }

    /// Reset all tallies to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::vsa::{VsaVector, VsaAccumulator};
    /// let mut acc = VsaAccumulator::new();
    /// acc.add(&VsaVector::random(1));
    /// acc.reset();
    /// assert!(acc.values.iter().all(|&v| v == 0));
    /// ```
    pub fn reset(&mut self) {
        self.values.fill(0);
    }
}

impl Default for VsaAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_self_inverse() {
        let a = VsaVector::random(1);
        let b = VsaVector::random(2);
        let bound = vsa_bind(&a, &b);
        let recovered = vsa_bind(&a, &bound);
        assert_eq!(recovered, b);
    }

    #[test]
    fn test_bundle_majority() {
        let a = VsaVector::random(10);
        let b = VsaVector::random(20);
        let bundled = vsa_bundle(&[a.clone(), a.clone(), a.clone(), b.clone()]);
        let sim_a = vsa_similarity(&bundled, &a);
        let sim_b = vsa_similarity(&bundled, &b);
        assert!(
            sim_a > sim_b,
            "bundled should be closer to a (sim_a={}, sim_b={})",
            sim_a,
            sim_b
        );
    }

    #[test]
    fn test_unbundle_recovery() {
        let key1 = VsaVector::random(100);
        let key2 = VsaVector::random(200);
        let val1 = VsaVector::random(300);
        let val2 = VsaVector::random(400);

        // Bind each value with its key
        let pair1 = vsa_bind(&key1, &val1);
        let pair2 = vsa_bind(&key2, &val2);

        // Bundle the pairs
        let bundled = vsa_bundle(&[pair1, pair2]);

        // Unbind with key1 should be closer to val1 than val2
        let recovered = vsa_unbind(&bundled, &key1);
        let sim_val1 = vsa_similarity(&recovered, &val1);
        let sim_val2 = vsa_similarity(&recovered, &val2);
        assert!(
            sim_val1 > sim_val2,
            "recovered should be closer to val1 (sim_val1={}, sim_val2={})",
            sim_val1,
            sim_val2
        );
    }

    #[test]
    fn test_similarity_self_one() {
        let a = VsaVector::random(42);
        let sim = vsa_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < f32::EPSILON,
            "self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_complement_zero() {
        let a = VsaVector::random(42);
        // Create complement: flip all meaningful bits
        let mut complement = VsaVector::zero();
        for i in 0..VSA_WORDS {
            complement.words[i] = !a.words[i];
        }
        complement.words[VSA_WORDS - 1] &= TAIL_MASK;

        let sim = vsa_similarity(&a, &complement);
        assert!(
            sim.abs() < 0.01,
            "complement similarity should be ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_permute_roundtrip() {
        let v = VsaVector::random(7);
        let k = 137;
        let shifted = vsa_permute(&v, k);
        let restored = vsa_permute(&shifted, VSA_DIMS - k);
        assert_eq!(restored, v);
    }

    #[test]
    fn test_sequence_order_matters() {
        let a = VsaVector::random(1);
        let b = VsaVector::random(2);
        let seq_ab = vsa_sequence(&[a.clone(), b.clone()]);
        let seq_ba = vsa_sequence(&[b, a]);
        assert_ne!(seq_ab, seq_ba);
    }

    #[test]
    fn test_clean_finds_closest() {
        let entry = VsaVector::random(50);
        let other1 = VsaVector::random(51);
        let other2 = VsaVector::random(52);

        // Add noise to the entry: flip ~5% of bits
        let mut noisy = entry.clone();
        let mut state = 0xBEEFu64;
        let mut flipped = 0;
        for dim in 0..VSA_DIMS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state % 20 == 0 {
                // ~5% flip rate
                let word = dim / 64;
                let bit = dim % 64;
                noisy.words[word] ^= 1u64 << bit;
                flipped += 1;
            }
        }
        noisy.words[VSA_WORDS - 1] &= TAIL_MASK;
        assert!(flipped > 0, "should have flipped some bits");

        let codebook = vec![other1, other2, entry.clone()];
        let found = vsa_clean(&noisy, &codebook).unwrap();
        assert_eq!(
            found, &entry,
            "clean should recover the closest codebook entry"
        );
    }

    #[test]
    fn test_accumulator_add_extract() {
        let v = VsaVector::random(99);
        let mut acc = VsaAccumulator::new();
        acc.add(&v);
        acc.add(&v);
        acc.add(&v);
        let extracted = acc.extract();
        assert_eq!(extracted, v);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = VsaAccumulator::new();
        acc.add(&VsaVector::random(1));
        acc.add(&VsaVector::random(2));
        acc.reset();
        let extracted = acc.extract();
        assert_eq!(extracted, VsaVector::zero());
    }

    #[test]
    fn test_from_text_deterministic() {
        let a = VsaVector::from_text("hello world");
        let b = VsaVector::from_text("hello world");
        assert_eq!(a, b);

        let c = VsaVector::from_text("goodbye world");
        assert_ne!(a, c);
    }

    #[test]
    fn test_random_near_half_density() {
        let v = VsaVector::random(12345);
        let ones = v.popcount();
        let density = ones as f32 / VSA_DIMS as f32;
        assert!(
            (density - 0.5).abs() < 0.05,
            "density should be ~50%, got {:.2}% ({} of {} bits)",
            density * 100.0,
            ones,
            VSA_DIMS
        );
    }

    #[test]
    fn test_constants() {
        assert_eq!(TAIL_BITS, 16);
        assert_eq!(TAIL_MASK, 0xFFFF);
        assert_eq!((VSA_WORDS - 1) * 64 + TAIL_BITS, VSA_DIMS);
    }

    #[test]
    fn test_zero_is_identity() {
        let a = VsaVector::random(1);
        let zero = VsaVector::zero();
        let result = vsa_bind(&a, &zero);
        assert_eq!(result, a);
    }

    #[test]
    fn test_accumulator_subtract() {
        let a = VsaVector::random(1);
        let b = VsaVector::random(2);
        let mut acc = VsaAccumulator::new();
        acc.add(&a);
        acc.add(&b);
        acc.add(&a);
        // a has weight +2 total (add twice), b has weight 0 after subtract
        acc.subtract(&b);
        // Now a contributed +2 (via add) and b contributed 0 (add then subtract cancel).
        // Each dim: if a=1,b=1 → +2-0 = +2; a=1,b=0 → +2+0=+2; a=0,b=1 → -2+0=-2; a=0,b=0 → -2+0=-2
        // Actually: add(a) twice means a=1→+2, a=0→-2 for a's contribution.
        // add(b) then subtract(b) means b=1→+1-1=0, b=0→-1+1=0.
        // Total: a=1→+2, a=0→-2, so extract should be a.
        let extracted = acc.extract();
        assert_eq!(extracted, a);
    }
}
