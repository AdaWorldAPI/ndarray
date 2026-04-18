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
#[derive(Clone, PartialEq, Eq)]
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
        Self {
            words: [u64::MAX; N],
        }
    }

    /// Create from a word array.
    #[inline]
    pub fn from_words(words: [u64; N]) -> Self {
        Self { words }
    }

    /// Create from a byte slice. Panics if `bytes.len() < N * 8`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= N * 8,
            "need at least {} bytes, got {}",
            N * 8,
            bytes.len()
        );
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

    /// Hamming distance (number of differing bits).
    /// Delegates to ndarray's SIMD dispatch (AVX-512 → AVX2 → scalar).
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        super::bitwise::hamming_distance_raw(self.as_bytes(), other.as_bytes()) as u32
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
        VectorConfig { width: w, words, bits: words * 64, bytes: words * 8 }
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
        let a = Fingerprint::<4> {
            words: [1, 2, 3, 4],
        };
        let b = Fingerprint::<4> {
            words: [5, 6, 7, 8],
        };
        let c = Fingerprint::<4> {
            words: [9, 10, 11, 12],
        };
        let ab_c = &(&a ^ &b) ^ &c;
        let a_bc = &a ^ &(&b ^ &c);
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_hamming_distance() {
        let a = Fingerprint::<2> {
            words: [0xFF, 0x00],
        };
        let b = Fingerprint::<2> {
            words: [0x00, 0x00],
        };
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

        let b = Fingerprint::<2> {
            words: [0xFF, 0xFF],
        };
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
        let a = Fingerprint::<2> {
            words: [0xFF, 0x00],
        };
        let b = Fingerprint::<2> {
            words: [0x0F, 0xF0],
        };
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
