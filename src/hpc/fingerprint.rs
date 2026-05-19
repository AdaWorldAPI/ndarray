//! Const-generic binary fingerprint for holographic storage.
//!
//! `Fingerprint<N>` is a fixed-size binary vector of N×64 bits, stored as
//! `[u64; N]`. All holographic operations (XOR bind, Hamming distance,
//! delta layers) operate on this type.
//!
//! Standard sizes:
//! - `Fingerprint<256>` = 2048 bytes = 16384 bits (CogRecord container)
//! - `Fingerprint<128>` = 1024 bytes = 8192 bits
//! - `Fingerprint<1024>` = 8192 bytes = 65536 bits (64K recognition)

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// A fixed-size binary fingerprint stored as N words of u64.
///
/// Total bits = N × 64. Total bytes = N × 8.
///
/// The XOR group structure makes this the natural type for holographic
/// delta layers: ground truth is `&self`, writers own their delta as `&mut`.
///
/// # Memory layout
///
/// `#[repr(C)]` pins the layout so that `Fingerprint<N>` is exactly one
/// contiguous `[u64; N]` with no padding and no field reordering. This is a
/// hard precondition for the zero-copy reinterprets in [`Fingerprint::as_bytes`]
/// and [`Fingerprint::as_u8x64`] (the latter requires that a pointer to
/// `self.words` is the same as a pointer to `*self`, which `#[repr(C)]` on a
/// single-field struct guarantees explicitly rather than relying on the
/// Rust-reference single-field-struct layout rule).
#[derive(Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Fingerprint<const N: usize> {
    pub words: [u64; N],
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> Fingerprint<N> {
    /// Total number of bits in this fingerprint.
    pub const BITS: usize = N * 64;

    /// Total number of bytes in this fingerprint.
    pub const BYTES: usize = N * 8;

    /// Zero fingerprint (identity element for XOR).
    #[inline]
    pub fn zero() -> Self {
        Self { words: [0u64; N] }
    }

    /// All-ones fingerprint.
    #[inline]
    pub fn ones() -> Self {
        Self { words: [u64::MAX; N] }
    }

    /// Create from a word array.
    #[inline]
    pub fn from_words(words: [u64; N]) -> Self {
        Self { words }
    }

    /// Create from a byte slice. Panics if `bytes.len() < N * 8`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= N * 8, "need at least {} bytes, got {}", N * 8, bytes.len());
        let mut words = [0u64; N];
        for i in 0..N {
            let offset = i * 8;
            words[i] = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
        }
        Self { words }
    }

    /// Convert to bytes (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(N * 8);
        for w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Get a specific bit (0-indexed).
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index < Self::BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set a specific bit.
    #[inline]
    pub fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < Self::BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Toggle a specific bit.
    #[inline]
    pub fn toggle_bit(&mut self, index: usize) {
        debug_assert!(index < Self::BITS);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    /// Create a random fingerprint from a seed (xorshift128+).
    pub fn random(seed: u64) -> Self {
        let mut s0 = seed;
        let mut s1 = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut words = [0u64; N];
        for word in &mut words {
            let mut s = s0;
            s0 = s1;
            s ^= s << 23;
            s ^= s >> 18;
            s ^= s1;
            s ^= s1 >> 5;
            s1 = s;
            *word = s0.wrapping_add(s1);
        }
        Self { words }
    }

    /// Hamming distance (number of differing bits).
    /// Delegates to ndarray's SIMD dispatch (AVX-512 → AVX2 → scalar).
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        super::bitwise::hamming_distance_raw(self.as_bytes(), other.as_bytes()) as u32
    }

    /// Alias for `hamming_distance` (ladybug-rs compat).
    #[inline]
    pub fn hamming(&self, other: &Self) -> u32 {
        self.hamming_distance(other)
    }

    /// XOR bind (ladybug-rs compat). Returns a new fingerprint.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] ^ other.words[i];
        }
        Self { words }
    }

    /// AND (bitwise intersection).
    #[inline]
    pub fn and(&self, other: &Self) -> Self {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] & other.words[i];
        }
        Self { words }
    }

    /// Bitwise NOT.
    #[inline]
    pub fn not(&self) -> Self {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = !self.words[i];
        }
        Self { words }
    }

    /// Density: fraction of set bits (popcount / total bits).
    #[inline]
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / Self::BITS as f32
    }

    /// Access raw words as slice.
    #[inline]
    pub fn as_raw(&self) -> &[u64; N] {
        &self.words
    }

    /// Multi-lane SIMD view: iterate fingerprint as batches of 8 u64 words.
    ///
    /// At N=256 (16K fingerprint), this yields 32 chunks of 8 words each.
    /// Each chunk is one AVX-512 VPOPCNTDQ iteration (512 bits at a time).
    /// Consumer uses `U64x8::from_slice(chunk)` for SIMD popcount.
    #[inline]
    pub fn chunks_u64x8(&self) -> impl Iterator<Item = &[u64]> {
        self.words.chunks(8)
    }

    /// Multi-lane SIMD view: iterate as batches of 64 bytes.
    ///
    /// At N=256 (16K fingerprint), yields 32 chunks of 64 bytes.
    /// Each chunk = one U8x64 load for byte-level SIMD ops.
    #[inline]
    pub fn chunks_u8x64(&self) -> impl Iterator<Item = &[u8]> {
        self.as_bytes().chunks(64)
    }

    /// Bundle (majority vote) across multiple fingerprints.
    ///
    /// Returns a new fingerprint where each bit is set if more than
    /// half of the input fingerprints have it set.
    pub fn bundle(items: &[&Self]) -> Self {
        let n = items.len();
        if n == 0 {
            return Self::zero();
        }
        let threshold = n / 2;
        let mut result = [0u64; N];
        for w in 0..N {
            for bit in 0..64 {
                let count: usize = items
                    .iter()
                    .filter(|fp| (fp.words[w] >> bit) & 1 == 1)
                    .count();
                if count > threshold {
                    result[w] |= 1u64 << bit;
                }
            }
        }
        Self { words: result }
    }

    /// Create a quasi-orthogonal fingerprint from a seed.
    /// Uses golden-ratio-multiplied seeds to ensure near-orthogonality.
    pub fn orthogonal(seed: u64) -> Self {
        Self::random(seed.wrapping_mul(0x9E3779B97F4A7C15))
    }

    /// Bitwise OR.
    #[inline]
    pub fn or(&self, other: &Self) -> Self {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] | other.words[i];
        }
        Self { words }
    }

    /// Create from content string (SHA-256-like hash expansion).
    pub fn from_content(data: &str) -> Self {
        let mut h = 0x736f6d6570736575u64;
        for (i, b) in data.bytes().enumerate() {
            h ^= (b as u64) << ((i % 8) * 8);
            h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe6546b64);
        }
        Self::random(h)
    }

    /// Permute: circular bit shift by `positions` (positive = left).
    pub fn permute(&self, positions: i32) -> Self {
        let total = Self::BITS as i32;
        let shift = ((positions % total) + total) % total;
        if shift == 0 {
            return self.clone();
        }
        let mut result = Self::zero();
        for i in 0..Self::BITS {
            if self.get_bit(i) {
                let new_pos = ((i as i32 + shift) % total) as usize;
                result.set_bit(new_pos, true);
            }
        }
        result
    }

    /// Hamming weight (number of set bits).
    #[inline]
    pub fn popcount(&self) -> u32 {
        super::bitwise::popcount_raw(self.as_bytes()) as u32
    }

    /// Returns true if all bits are zero (identity element).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Hamming similarity in [0.0, 1.0]. Returns None on zero-width fingerprint.
    #[inline]
    pub fn similarity(&self, other: &Self) -> Option<f32> {
        if Self::BITS == 0 {
            return None;
        }
        Some(1.0 - self.hamming_distance(other) as f32 / Self::BITS as f32)
    }

    /// Zero-copy view of the fingerprint as a byte slice.
    ///
    /// SAFETY: `[u64; N]` is guaranteed contiguous in memory.
    /// `u8` has no alignment requirements stricter than `u64`.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: [u64; N] is contiguous. u64 is 8-byte aligned; u8 requires
        // 1-byte alignment. Pointer cast is always valid. Length N * 8 is exact.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, N * 8) }
    }

    /// Zero-copy mutable view as byte slice.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: Same as as_bytes(). Mutable borrow from &mut self guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, N * 8) }
    }
}

// XOR group operations

#[allow(clippy::needless_range_loop)]
impl<const N: usize> BitXor for Fingerprint<N> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] ^ rhs.words[i];
        }
        Self { words }
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> BitXor for &Fingerprint<N> {
    type Output = Fingerprint<N>;

    #[inline]
    fn bitxor(self, rhs: Self) -> Fingerprint<N> {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] ^ rhs.words[i];
        }
        Fingerprint { words }
    }
}

impl<const N: usize> BitXorAssign for Fingerprint<N> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.words[i] ^= rhs.words[i];
        }
    }
}

impl<const N: usize> BitXorAssign<&Fingerprint<N>> for Fingerprint<N> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.words[i] ^= rhs.words[i];
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> BitAnd for &Fingerprint<N> {
    type Output = Fingerprint<N>;

    #[inline]
    fn bitand(self, rhs: Self) -> Fingerprint<N> {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] & rhs.words[i];
        }
        Fingerprint { words }
    }
}

impl<const N: usize> BitAndAssign<&Fingerprint<N>> for Fingerprint<N> {
    #[inline]
    fn bitand_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.words[i] &= rhs.words[i];
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> BitOr for &Fingerprint<N> {
    type Output = Fingerprint<N>;

    #[inline]
    fn bitor(self, rhs: Self) -> Fingerprint<N> {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = self.words[i] | rhs.words[i];
        }
        Fingerprint { words }
    }
}

impl<const N: usize> BitOrAssign<&Fingerprint<N>> for Fingerprint<N> {
    #[inline]
    fn bitor_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.words[i] |= rhs.words[i];
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> Not for &Fingerprint<N> {
    type Output = Fingerprint<N>;

    #[inline]
    fn not(self) -> Fingerprint<N> {
        let mut words = [0u64; N];
        for i in 0..N {
            words[i] = !self.words[i];
        }
        Fingerprint { words }
    }
}

impl<const N: usize> std::fmt::Debug for Fingerprint<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fingerprint<{}>[popcount={}, ", N, self.popcount())?;
        let show = N.min(4);
        for i in 0..show {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{:016x}", self.words[i])?;
        }
        if N > 4 {
            write!(f, " ...")?;
        }
        write!(f, "]")
    }
}

/// Standard 2048-byte fingerprint (CogRecord container size).
pub type Fingerprint2K = Fingerprint<256>;

/// Standard 1024-byte fingerprint.
pub type Fingerprint1K = Fingerprint<128>;

/// 64K-bit fingerprint (recognition projections).
pub type Fingerprint64K = Fingerprint<1024>;

// ─── Vector width config (LazyLock, switchable) ─────────────────

use std::sync::LazyLock;

/// Supported vector widths for the BindSpace substrate.
///
/// NOTE: 4096 is NOT a vector width — it's the 0xFFF schema/command address
/// space (4096 CAM operations, verb vocabulary). Vectors are 8K or 16K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum VectorWidth {
    /// 8,192 bits = 128 words = 1 KB. Deprecated, still referenced in some code.
    W8K = 128,
    /// 16,384 bits = 256 words = 2 KB. Production default.
    W16K = 256,
}

/// Runtime vector configuration. Frozen on first access.
///
/// Like `simd_caps()` — detect once, read everywhere.
/// Controls serialization format, network protocol, and storage layout.
/// Does NOT change the Rust type (use the matching Fingerprint\<N\> alias).
#[derive(Clone, Copy, Debug)]
pub struct VectorConfig {
    pub width: VectorWidth,
    pub words: usize,
    pub bits: usize,
    pub bytes: usize,
}

impl VectorConfig {
    const fn from_width(w: VectorWidth) -> Self {
        let words = w as usize;
        VectorConfig {
            width: w,
            words,
            bits: words * 64,
            bytes: words * 8,
        }
    }
}

static VECTOR_WIDTH: LazyLock<VectorConfig> = LazyLock::new(|| {
    let w = std::env::var("NDARRAY_VECTOR_WIDTH")
        .ok()
        .and_then(|s| match s.as_str() {
            "8192" | "8k" | "8K" => Some(VectorWidth::W8K),
            "16384" | "16k" | "16K" => Some(VectorWidth::W16K),
            _ => None,
        })
        .unwrap_or(VectorWidth::W16K);
    VectorConfig::from_width(w)
});

/// Get the frozen vector width configuration.
///
/// Defaults to 16K (production). Override with `NDARRAY_VECTOR_WIDTH=8192`
/// env var before first access. After first call, width is frozen.
///
/// ```
/// use ndarray::hpc::fingerprint::vector_config;
/// let cfg = vector_config();
/// assert_eq!(cfg.bits, 16_384);  // default
/// assert_eq!(cfg.words, 256);
/// assert_eq!(cfg.bytes, 2_048);
/// ```
pub fn vector_config() -> &'static VectorConfig {
    &VECTOR_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_identity() {
        let a = Fingerprint::<4> {
            words: [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0],
        };
        let zero = Fingerprint::<4>::zero();
        assert_eq!(&a ^ &zero, a);
    }

    #[test]
    fn test_xor_self_inverse() {
        let a = Fingerprint::<4> {
            words: [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0],
        };
        let result = &a ^ &a;
        assert!(result.is_zero());
    }

    #[test]
    fn test_xor_associative() {
        let a = Fingerprint::<4> { words: [1, 2, 3, 4] };
        let b = Fingerprint::<4> { words: [5, 6, 7, 8] };
        let c = Fingerprint::<4> { words: [9, 10, 11, 12] };
        let ab_c = &(&a ^ &b) ^ &c;
        let a_bc = &a ^ &(&b ^ &c);
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_hamming_distance() {
        let a = Fingerprint::<2> { words: [0xFF, 0x00] };
        let b = Fingerprint::<2> { words: [0x00, 0x00] };
        assert_eq!(a.hamming_distance(&b), 8);
    }

    #[test]
    fn test_hamming_self_zero() {
        let a = Fingerprint::<4> {
            words: [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE],
        };
        assert_eq!(a.hamming_distance(&a), 0);
    }

    #[test]
    fn test_popcount() {
        let a = Fingerprint::<1> { words: [0xFF] };
        assert_eq!(a.popcount(), 8);

        let b = Fingerprint::<2> { words: [0xFF, 0xFF] };
        assert_eq!(b.popcount(), 16);
    }

    #[test]
    fn test_from_to_bytes_roundtrip() {
        let original = Fingerprint::<4> {
            words: [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0],
        };
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), 32);
        let restored = Fingerprint::<4>::from_bytes(&bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_similarity() {
        let a = Fingerprint::<2>::zero();
        let b = Fingerprint::<2>::zero();
        assert!((a.similarity(&b).unwrap() - 1.0).abs() < f32::EPSILON);

        let c = Fingerprint::<2>::ones();
        assert!((a.similarity(&c).unwrap() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_2k_size() {
        assert_eq!(Fingerprint2K::BYTES, 2048);
        assert_eq!(Fingerprint2K::BITS, 16384);
    }

    #[test]
    fn test_xor_assign() {
        let a = Fingerprint::<2> { words: [0xFF, 0x00] };
        let b = Fingerprint::<2> { words: [0x0F, 0xF0] };
        let mut c = a.clone();
        c ^= &b;
        assert_eq!(c, &a ^ &b);
    }

    #[test]
    fn test_as_bytes_roundtrip() {
        let fp = Fingerprint::<4> {
            words: [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0],
        };
        let bytes = fp.as_bytes();
        assert_eq!(bytes.len(), 32);
        let restored = Fingerprint::<4>::from_bytes(bytes);
        assert_eq!(fp, restored);
    }

    #[test]
    fn test_as_bytes_zero_copy() {
        let fp = Fingerprint::<4>::zero();
        let bytes_ptr = fp.as_bytes().as_ptr();
        let words_ptr = fp.words.as_ptr() as *const u8;
        assert_eq!(bytes_ptr, words_ptr, "as_bytes must be zero-copy");
    }
}

// ============================================================================
// PR-X1 — N==8 register-width view (carved-out draft, body lands in uncomment sprint)
// ============================================================================

impl Fingerprint<8> {
    /// Typed `&[u8; 64]` view over the fingerprint's backing bytes (PR-X1).
    ///
    /// Available only on `Fingerprint<8>` — 8 × `u64` = 64 bytes = one AVX-512
    /// `U8x64` register width. Returns the exact `&[u8; 64]` reference without
    /// copying; feed it into `crate::simd::U8x64::from_array(*win)` or
    /// `U8x64::from_slice(win)` for register-level byte ops.
    ///
    /// For other `N` widths, use [`Fingerprint::chunks_u8x64`] which yields
    /// 64-byte chunks without the compile-time length guarantee.
    ///
    /// # Design reference
    ///
    /// `.claude/knowledge/pr-x1-design.md` § "2. `Fingerprint::as_u8x64`".
    /// Body: pointer reinterpret of `self.words.as_ptr() as *const [u8; 64]`,
    /// justified by the `#[repr(C)]` + 8-`u64` = 64-byte layout invariant.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::hpc::fingerprint::Fingerprint;
    /// let fp: Fingerprint<8> = Fingerprint::zero();
    /// let window: &[u8; 64] = fp.as_u8x64();
    /// assert_eq!(window.len(), 64);
    /// assert!(window.iter().all(|&b| b == 0));
    /// ```
    ///
    /// # Examples — little-endian word layout
    ///
    /// ```
    /// use ndarray::hpc::fingerprint::Fingerprint;
    /// let fp: Fingerprint<8> = Fingerprint::from_words([
    ///     0x0102030405060708u64, 0, 0, 0, 0, 0, 0, 0,
    /// ]);
    /// let view = fp.as_u8x64();
    /// // Words are stored little-endian; word[0] low byte = 0x08.
    /// assert_eq!(view[0], 0x08);
    /// assert_eq!(view[7], 0x01);
    /// assert_eq!(view[8], 0x00); // word[1] is zero
    /// ```
    #[inline]
    pub fn as_u8x64(&self) -> &[u8; 64] {
        // SAFETY:
        // 1. `Fingerprint<N>` is `#[repr(C)]` with a single field `words: [u64; N]`,
        //    so `self.words.as_ptr()` points at the first byte of `*self` and the
        //    struct has no trailing padding or field reordering.
        // 2. For `N == 8`, `size_of::<[u64; 8]>() == 64 == size_of::<[u8; 64]>()`,
        //    so the reinterpret covers exactly the same 64 bytes — no out-of-bounds
        //    read, no truncation.
        // 3. `[u64; 8]` has alignment 8; `[u8; 64]` has alignment 1. The source
        //    alignment is strictly greater than the destination alignment, so the
        //    `*const [u8; 64]` is validly aligned for its pointee type.
        // 4. `u8` has no invalid bit patterns; any byte pattern in the `u64` words
        //    is a valid `u8` byte, so dereferencing the cast pointer is sound.
        // 5. The returned reference borrows from `&self`, so its lifetime cannot
        //    outlive `self`, satisfying the borrow-checker lifetime rule and
        //    preventing dangling references.
        unsafe { &*(self.words.as_ptr() as *const [u8; 64]) }
    }
}

#[cfg(test)]
mod pr_x1_as_u8x64_tests {
    use super::*;

    /// `as_u8x64` on `zero()` returns 64 zero bytes.
    #[test]
    fn as_u8x64_zero_all_zero() {
        let fp: Fingerprint<8> = Fingerprint::zero();
        let view = fp.as_u8x64();
        assert_eq!(view.len(), 64);
        assert!(view.iter().all(|&b| b == 0));
    }

    /// `as_u8x64` on `ones()` returns 64 `0xFF` bytes.
    #[test]
    fn as_u8x64_ones_all_ff() {
        let fp: Fingerprint<8> = Fingerprint::ones();
        let view = fp.as_u8x64();
        assert_eq!(view.len(), 64);
        assert!(view.iter().all(|&b| b == 0xFF));
    }

    /// Words land little-endian: low byte of `word[0]` is at byte offset 0.
    #[test]
    fn as_u8x64_little_endian_round_trip() {
        let fp: Fingerprint<8> = Fingerprint::from_words([
            0x0102030405060708u64,
            0x1112131415161718u64,
            0,
            0,
            0,
            0,
            0,
            0,
        ]);
        let view = fp.as_u8x64();
        // word[0] = 0x0102030405060708 → bytes 0..8 = [08 07 06 05 04 03 02 01]
        assert_eq!(view[0], 0x08);
        assert_eq!(view[1], 0x07);
        assert_eq!(view[7], 0x01);
        // word[1] = 0x1112131415161718 → bytes 8..16 = [18 17 16 15 14 13 12 11]
        assert_eq!(view[8], 0x18);
        assert_eq!(view[15], 0x11);
        // Remaining words are zero.
        assert!(view[16..].iter().all(|&b| b == 0));
    }

    /// `as_u8x64` does not allocate: the returned pointer equals the
    /// underlying `words` pointer cast to `*const u8`.
    #[test]
    fn as_u8x64_zero_copy_pointer_equality() {
        let fp: Fingerprint<8> = Fingerprint::zero();
        let view_ptr = fp.as_u8x64().as_ptr();
        let words_ptr = fp.words.as_ptr() as *const u8;
        assert_eq!(view_ptr, words_ptr, "as_u8x64 must be zero-copy");
    }

    /// Size in bytes matches the AVX-512 U8x64 register width.
    #[test]
    fn fingerprint8_is_exactly_64_bytes() {
        assert_eq!(core::mem::size_of::<Fingerprint<8>>(), 64);
        assert_eq!(Fingerprint::<8>::BYTES, 64);
    }
}
