//! U8x32 / U8x64 portable-simd wrappers — round-3-portable-simd agent #3.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdUint;
use core::simd::{u8x32 as core_u8x32, u8x64 as core_u8x64, Simd};

// ════════════════════════════════════════════════════════════════════
// U8x64 — 64-lane unsigned byte (rasterizer / palette / NBT width)
// ════════════════════════════════════════════════════════════════════

/// 64-lane `u8` SIMD vector backed by `core::simd::u8x64`.
///
/// API mirrors `simd_avx512::U8x64` so consumer code is identical under
/// both the intrinsics and the portable-simd backend. Miri can execute
/// every method below.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::U8x64;
/// let a = U8x64::splat(10);
/// let b = U8x64::splat(20);
/// assert_eq!(a.simd_min(b).to_array()[0], 10);
/// # }
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U8x64(pub core_u8x64);

impl U8x64 {
    /// Number of `u8` lanes.
    pub const LANES: usize = 64;

    // ── Constructors ────────────────────────────────────────────────

    /// Broadcast a single byte to all 64 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// assert_eq!(U8x64::splat(7).to_array()[0], 7);
    /// # }
    /// ```
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(core_u8x64::splat(v))
    }

    /// Unaligned load 64 bytes from a slice. Panics if `s.len() < 64`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let data = [1u8; 64];
    /// let v = U8x64::from_slice(&data);
    /// assert_eq!(v.to_array()[0], 1);
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 64, "U8x64::from_slice needs ≥64 bytes");
        Self(core_u8x64::from_slice(s))
    }

    /// Load 64 bytes from a fixed-size array.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let arr = [2u8; 64];
    /// assert_eq!(U8x64::from_array(arr).to_array(), arr);
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_array(arr: [u8; 64]) -> Self {
        Self(core_u8x64::from_array(arr))
    }

    /// Store all 64 bytes into a `[u8; 64]` array.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let v = U8x64::splat(3);
    /// assert_eq!(v.to_array()[63], 3);
    /// # }
    /// ```
    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        self.0.to_array()
    }

    /// Copy all 64 bytes into a mutable slice. Panics if `s.len() < 64`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let mut buf = [0u8; 64];
    /// U8x64::splat(9).copy_to_slice(&mut buf);
    /// assert_eq!(buf[0], 9);
    /// # }
    /// ```
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 64, "U8x64::copy_to_slice needs ≥64 bytes");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping sum of all 64 lanes → `u8` (wraps at 256).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// assert_eq!(U8x64::splat(1).reduce_sum(), 64u8.wrapping_mul(1));
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        self.0.reduce_sum()
    }

    /// Unsigned minimum across all 64 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let mut arr = [10u8; 64];
    /// arr[5] = 2;
    /// assert_eq!(U8x64::from_array(arr).reduce_min(), 2);
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        self.0.reduce_min()
    }

    /// Unsigned maximum across all 64 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let mut arr = [10u8; 64];
    /// arr[5] = 200;
    /// assert_eq!(U8x64::from_array(arr).reduce_max(), 200);
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        self.0.reduce_max()
    }

    /// Horizontal byte sum as `u64` — does NOT wrap at 256.
    ///
    /// Promotes each byte lane to `u16`, then reduces. Range: 0..=64×255=16_320.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// assert_eq!(U8x64::splat(1).sum_bytes_u64(), 64);
    /// assert_eq!(U8x64::splat(255).sum_bytes_u64(), 64 * 255);
    /// # }
    /// ```
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        // Promote u8 × 64 → u16 × 64 to avoid 8-bit wrapping, then reduce.
        let v16: Simd<u16, 64> = self.0.cast::<u16>();
        v16.reduce_sum() as u64
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    /// Per-lane unsigned min.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(100);
    /// let b = U8x64::splat(50);
    /// assert_eq!(a.simd_min(b).to_array()[0], 50);
    /// # }
    /// ```
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Per-lane unsigned max.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(100);
    /// let b = U8x64::splat(50);
    /// assert_eq!(a.simd_max(b).to_array()[0], 100);
    /// # }
    /// ```
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ────────────────────────────────────────

    /// Per-lane saturating unsigned add: `min(a + b, 255)`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(200);
    /// let b = U8x64::splat(100);
    /// assert_eq!(a.saturating_add(b).to_array()[0], 255);
    /// # }
    /// ```
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Per-lane saturating unsigned sub: `max(a - b, 0)`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(10);
    /// let b = U8x64::splat(20);
    /// assert_eq!(a.saturating_sub(b).to_array()[0], 0);
    /// # }
    /// ```
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Per-lane unsigned rounded average: `(a + b + 1) >> 1`.
    ///
    /// `core::simd` has no native `avg_epu8`; computed via u16 promotion
    /// to avoid overflow. LLVM may lower to `vpavgb` on AVX-512 builds.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(10);
    /// let b = U8x64::splat(11);
    /// // (10+11+1)/2 = 11
    /// assert_eq!(a.pairwise_avg(b).to_array()[0], 11);
    /// # }
    /// ```
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        let a16: Simd<u16, 64> = self.0.cast::<u16>();
        let b16: Simd<u16, 64> = other.0.cast::<u16>();
        let avg = (a16 + b16 + Simd::splat(1u16)) >> Simd::splat(1u16);
        Self(avg.cast::<u8>())
    }

    // ── 16-bit-lane shifts (nibble pack/unpack) ─────────────────────

    /// Right shift each 16-bit lane by `imm` bits.
    ///
    /// Operates on 16-bit lanes (same semantics as `_mm512_srli_epi16`).
    /// Promotes to `u16 × 32`, shifts, truncates back to `u8 × 64`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// // Byte pattern 0x10 in high nibble, shift right 4 → 0x01 in low byte
    /// let v = U8x64::splat(0x10);
    /// assert_eq!(v.shr_epi16(4).to_array()[0], 0x01);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        // Reinterpret u8×64 as u16×32, apply shift, reinterpret back.
        let bytes = self.to_array();
        // SAFETY: [u8; 64] has same size as [u16; 32].
        let mut words: [u16; 32] = unsafe { core::mem::transmute(bytes) };
        for w in words.iter_mut() {
            *w >>= imm;
        }
        Self::from_array(unsafe { core::mem::transmute(words) })
    }

    /// Left shift each 16-bit lane by `imm` bits.
    ///
    /// Operates on 16-bit lanes (same semantics as `_mm512_slli_epi16`).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let v = U8x64::splat(0x01);
    /// // shl by 4 → high nibble: 0x10 in high byte of u16 lane (low byte of pair)
    /// assert_eq!(v.shl_epi16(4).to_array()[0], 0x10);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        let bytes = self.to_array();
        let mut words: [u16; 32] = unsafe { core::mem::transmute(bytes) };
        for w in words.iter_mut() {
            *w <<= imm;
        }
        Self::from_array(unsafe { core::mem::transmute(words) })
    }

    // ── Comparison → 64-bit bitmask ───────────────────────────────

    /// Per-lane equality → 64-bit bitmask (bit `i` set iff `self[i] == other[i]`).
    ///
    /// Matches `simd_avx512::U8x64::cmpeq_mask` shape.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(5);
    /// let b = U8x64::splat(5);
    /// assert_eq!(a.cmpeq_mask(b), u64::MAX);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        self.0.simd_eq(other.0).to_bitmask()
    }

    /// Per-lane unsigned greater-than → 64-bit bitmask.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(10);
    /// let b = U8x64::splat(5);
    /// assert_eq!(a.cmpgt_mask(b), u64::MAX);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        self.0.simd_gt(other.0).to_bitmask()
    }

    /// Extract MSB of each lane as a 64-bit mask.
    ///
    /// Bit `i` is set iff `self[i] >= 128`. Equivalent to `_mm512_movepi8_mask`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// assert_eq!(U8x64::splat(128).movemask(), u64::MAX);
    /// assert_eq!(U8x64::splat(127).movemask(), 0);
    /// # }
    /// ```
    #[inline(always)]
    pub fn movemask(self) -> u64 {
        // MSB set ⇔ value > 127 in unsigned comparison.
        self.0.simd_gt(core_u8x64::splat(0x7F)).to_bitmask()
    }

    // ── Nibble popcount LUT ────────────────────────────────────────

    /// Build the nibble-popcount lookup table (replicated across all 64 bytes).
    ///
    /// Entry `i` (for i in 0..16) = `popcount(i)`. Intended for use with the
    /// shuffle-based Mula SIMD popcount algorithm in `exotic_methods.rs`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let lut = U8x64::nibble_popcount_lut();
    /// // entry 0 → 0, entry 1 → 1, entry 3 → 2, entry 15 → 4
    /// assert_eq!(lut.to_array()[0], 0);
    /// assert_eq!(lut.to_array()[3], 2);
    /// assert_eq!(lut.to_array()[15], 4);
    /// # }
    /// ```
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        // LUT for nibbles 0..=15, replicated 4× to fill 64 bytes.
        Self::from_array([
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2,
            1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        ])
    }
}

impl Default for U8x64 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U8x32 — 32-lane unsigned byte (AVX2 width)
// ════════════════════════════════════════════════════════════════════

/// 32-lane `u8` SIMD vector backed by `core::simd::u8x32`.
///
/// API mirrors `simd_avx2::U8x32` so consumer code is identical under
/// both the intrinsics and the portable-simd backend. Miri can execute
/// every method below.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::U8x32;
/// let a = U8x32::splat(10);
/// let b = U8x32::splat(20);
/// assert_eq!(a.simd_min(b).to_array()[0], 10);
/// # }
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U8x32(pub core_u8x32);

impl U8x32 {
    /// Number of `u8` lanes.
    pub const LANES: usize = 32;

    // ── Constructors ────────────────────────────────────────────────

    /// Broadcast a single byte to all 32 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// assert_eq!(U8x32::splat(7).to_array()[0], 7);
    /// # }
    /// ```
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(core_u8x32::splat(v))
    }

    /// Unaligned load 32 bytes from a slice. Panics if `s.len() < 32`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let data = [1u8; 32];
    /// let v = U8x32::from_slice(&data);
    /// assert_eq!(v.to_array()[0], 1);
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 32, "U8x32::from_slice needs ≥32 bytes");
        Self(core_u8x32::from_slice(s))
    }

    /// Load 32 bytes from a fixed-size array.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let arr = [2u8; 32];
    /// assert_eq!(U8x32::from_array(arr).to_array(), arr);
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_array(arr: [u8; 32]) -> Self {
        Self(core_u8x32::from_array(arr))
    }

    /// Store all 32 bytes into a `[u8; 32]` array.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let v = U8x32::splat(3);
    /// assert_eq!(v.to_array()[31], 3);
    /// # }
    /// ```
    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
        self.0.to_array()
    }

    /// Copy all 32 bytes into a mutable slice. Panics if `s.len() < 32`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let mut buf = [0u8; 32];
    /// U8x32::splat(9).copy_to_slice(&mut buf);
    /// assert_eq!(buf[0], 9);
    /// # }
    /// ```
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 32, "U8x32::copy_to_slice needs ≥32 bytes");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping sum of all 32 lanes → `u8` (wraps at 256).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// assert_eq!(U8x32::splat(2).reduce_sum(), 64u8.wrapping_mul(1)); // 32*2=64
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        self.0.reduce_sum()
    }

    /// Unsigned minimum across all 32 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let mut arr = [10u8; 32];
    /// arr[5] = 2;
    /// assert_eq!(U8x32::from_array(arr).reduce_min(), 2);
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        self.0.reduce_min()
    }

    /// Unsigned maximum across all 32 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let mut arr = [10u8; 32];
    /// arr[5] = 200;
    /// assert_eq!(U8x32::from_array(arr).reduce_max(), 200);
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        self.0.reduce_max()
    }

    /// Horizontal byte sum as `u64` — does NOT wrap at 256.
    ///
    /// Promotes each byte lane to `u16`, then reduces. Range: 0..=32×255=8_160.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// assert_eq!(U8x32::splat(1).sum_bytes_u64(), 32);
    /// assert_eq!(U8x32::splat(255).sum_bytes_u64(), 32 * 255);
    /// # }
    /// ```
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        let v16: Simd<u16, 32> = self.0.cast::<u16>();
        v16.reduce_sum() as u64
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    /// Per-lane unsigned min.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(100);
    /// let b = U8x32::splat(50);
    /// assert_eq!(a.simd_min(b).to_array()[0], 50);
    /// # }
    /// ```
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Per-lane unsigned max.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(100);
    /// let b = U8x32::splat(50);
    /// assert_eq!(a.simd_max(b).to_array()[0], 100);
    /// # }
    /// ```
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ────────────────────────────────────────

    /// Per-lane saturating unsigned add: `min(a + b, 255)`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(200);
    /// let b = U8x32::splat(100);
    /// assert_eq!(a.saturating_add(b).to_array()[0], 255);
    /// # }
    /// ```
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Per-lane saturating unsigned sub: `max(a - b, 0)`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(10);
    /// let b = U8x32::splat(20);
    /// assert_eq!(a.saturating_sub(b).to_array()[0], 0);
    /// # }
    /// ```
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Per-lane unsigned rounded average: `(a + b + 1) >> 1`.
    ///
    /// `core::simd` has no native `avg_epu8`; computed via u16 promotion
    /// to avoid overflow. LLVM may lower to `vpavgb` on AVX2 builds.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(10);
    /// let b = U8x32::splat(11);
    /// // (10+11+1)/2 = 11
    /// assert_eq!(a.pairwise_avg(b).to_array()[0], 11);
    /// # }
    /// ```
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        let a16: Simd<u16, 32> = self.0.cast::<u16>();
        let b16: Simd<u16, 32> = other.0.cast::<u16>();
        let avg = (a16 + b16 + Simd::splat(1u16)) >> Simd::splat(1u16);
        Self(avg.cast::<u8>())
    }

    // ── 16-bit-lane shifts (nibble pack/unpack) ─────────────────────

    /// Right shift each 16-bit lane by `imm` bits.
    ///
    /// Operates on 16-bit lanes (same semantics as `_mm256_srli_epi16`).
    /// Reinterprets `u8 × 32` as `u16 × 16`, shifts, reinterprets back.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let v = U8x32::splat(0x10);
    /// assert_eq!(v.shr_epi16(4).to_array()[0], 0x01);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        let bytes = self.to_array();
        // SAFETY: [u8; 32] and [u16; 16] have identical size and alignment req ≤ 2.
        let mut words: [u16; 16] = unsafe { core::mem::transmute(bytes) };
        for w in words.iter_mut() {
            *w >>= imm;
        }
        Self::from_array(unsafe { core::mem::transmute(words) })
    }

    /// Left shift each 16-bit lane by `imm` bits.
    ///
    /// Operates on 16-bit lanes (same semantics as `_mm256_slli_epi16`).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let v = U8x32::splat(0x01);
    /// assert_eq!(v.shl_epi16(4).to_array()[0], 0x10);
    /// # }
    /// ```
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        let bytes = self.to_array();
        let mut words: [u16; 16] = unsafe { core::mem::transmute(bytes) };
        for w in words.iter_mut() {
            *w <<= imm;
        }
        Self::from_array(unsafe { core::mem::transmute(words) })
    }

    // ── Comparison → 32-bit bitmask ───────────────────────────────

    /// Per-lane equality → 32-bit bitmask (bit `i` set iff `self[i] == other[i]`).
    ///
    /// Matches `simd_avx2::U8x32::cmpeq_mask` shape. `core::simd::to_bitmask()`
    /// returns `u64` for all lane widths ≤ 64; the upper 32 bits are always zero.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(5);
    /// let b = U8x32::splat(5);
    /// assert_eq!(a.cmpeq_mask(b), u32::MAX);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane unsigned greater-than → 32-bit bitmask.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(10);
    /// let b = U8x32::splat(5);
    /// assert_eq!(a.cmpgt_mask(b), u32::MAX);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }

    /// Extract MSB of each lane as a 32-bit mask.
    ///
    /// Bit `i` is set iff `self[i] >= 128`. Equivalent to `_mm256_movemask_epi8`
    /// but limited to MSB of each byte.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// assert_eq!(U8x32::splat(128).movemask(), u32::MAX);
    /// assert_eq!(U8x32::splat(127).movemask(), 0);
    /// # }
    /// ```
    #[inline(always)]
    pub fn movemask(self) -> u32 {
        self.0.simd_gt(core_u8x32::splat(0x7F)).to_bitmask() as u32
    }

    // ── Nibble popcount LUT ────────────────────────────────────────

    /// Build the nibble-popcount lookup table (replicated across both 16-byte halves).
    ///
    /// Entry `i` (for i in 0..16) = `popcount(i)`. Intended for use with the
    /// shuffle-based Mula SIMD popcount algorithm in `exotic_methods.rs`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let lut = U8x32::nibble_popcount_lut();
    /// assert_eq!(lut.to_array()[0], 0);
    /// assert_eq!(lut.to_array()[3], 2);
    /// assert_eq!(lut.to_array()[15], 4);
    /// # }
    /// ```
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        // LUT replicated twice to fill 32 bytes (2 × 128-bit halves).
        Self::from_array([
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        ])
    }
}

impl Default for U8x32 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── U8x64 tests ─────────────────────────────────────────────────

    #[test]
    fn u8x64_splat_to_array() {
        let v = U8x64::splat(42);
        assert!(v.to_array().iter().all(|&b| b == 42));
    }

    #[test]
    fn u8x64_from_slice_roundtrip() {
        let src: [u8; 64] = core::array::from_fn(|i| i as u8);
        let mut dst = [0u8; 64];
        U8x64::from_slice(&src).copy_to_slice(&mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn u8x64_reduce_min_max() {
        let mut arr = [100u8; 64];
        arr[0] = 5;
        arr[63] = 200;
        let v = U8x64::from_array(arr);
        assert_eq!(v.reduce_min(), 5);
        assert_eq!(v.reduce_max(), 200);
    }

    #[test]
    fn u8x64_reduce_sum_wraps() {
        // 64 * 4 = 256 which wraps to 0
        assert_eq!(U8x64::splat(4).reduce_sum(), 0u8);
    }

    #[test]
    fn u8x64_sum_bytes_u64_no_wrap() {
        assert_eq!(U8x64::splat(255).sum_bytes_u64(), 64 * 255);
    }

    #[test]
    fn u8x64_simd_min_max() {
        let a = U8x64::splat(100);
        let b = U8x64::splat(50);
        assert!(a.simd_min(b).to_array().iter().all(|&x| x == 50));
        assert!(a.simd_max(b).to_array().iter().all(|&x| x == 100));
    }

    #[test]
    fn u8x64_saturating_add() {
        let a = U8x64::splat(200);
        let b = U8x64::splat(100);
        assert!(a.saturating_add(b).to_array().iter().all(|&x| x == 255));
    }

    #[test]
    fn u8x64_saturating_sub() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        assert!(a.saturating_sub(b).to_array().iter().all(|&x| x == 0));
    }

    #[test]
    fn u8x64_pairwise_avg() {
        // (10 + 11 + 1) / 2 = 11
        let a = U8x64::splat(10);
        let b = U8x64::splat(11);
        assert!(a.pairwise_avg(b).to_array().iter().all(|&x| x == 11));
        // (0 + 1 + 1) / 2 = 1
        let c = U8x64::splat(0);
        let d = U8x64::splat(1);
        assert!(c.pairwise_avg(d).to_array().iter().all(|&x| x == 1));
    }

    #[test]
    fn u8x64_cmpeq_mask_all_eq() {
        assert_eq!(U8x64::splat(5).cmpeq_mask(U8x64::splat(5)), u64::MAX);
    }

    #[test]
    fn u8x64_cmpeq_mask_none_eq() {
        assert_eq!(U8x64::splat(1).cmpeq_mask(U8x64::splat(2)), 0u64);
    }

    #[test]
    fn u8x64_cmpgt_mask() {
        assert_eq!(U8x64::splat(10).cmpgt_mask(U8x64::splat(5)), u64::MAX);
        assert_eq!(U8x64::splat(5).cmpgt_mask(U8x64::splat(10)), 0u64);
    }

    #[test]
    fn u8x64_movemask() {
        assert_eq!(U8x64::splat(128).movemask(), u64::MAX);
        assert_eq!(U8x64::splat(127).movemask(), 0u64);
    }

    #[test]
    fn u8x64_shr_epi16() {
        // 0x10 >> 4 = 0x01 in the low byte of each u16 lane.
        let v = U8x64::splat(0x10);
        assert_eq!(v.shr_epi16(4).to_array()[0], 0x01);
    }

    #[test]
    fn u8x64_shl_epi16() {
        let v = U8x64::splat(0x01);
        assert_eq!(v.shl_epi16(4).to_array()[0], 0x10);
    }

    #[test]
    fn u8x64_nibble_popcount_lut() {
        let lut = U8x64::nibble_popcount_lut();
        let arr = lut.to_array();
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[3], 2);
        assert_eq!(arr[7], 3);
        assert_eq!(arr[15], 4);
        // Verify second repetition at offset 16.
        assert_eq!(arr[16], 0);
        assert_eq!(arr[31], 4);
    }

    // ── U8x32 tests ─────────────────────────────────────────────────

    #[test]
    fn u8x32_splat_to_array() {
        let v = U8x32::splat(42);
        assert!(v.to_array().iter().all(|&b| b == 42));
    }

    #[test]
    fn u8x32_from_slice_roundtrip() {
        let src: [u8; 32] = core::array::from_fn(|i| i as u8);
        let mut dst = [0u8; 32];
        U8x32::from_slice(&src).copy_to_slice(&mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn u8x32_reduce_min_max() {
        let mut arr = [100u8; 32];
        arr[0] = 5;
        arr[31] = 200;
        let v = U8x32::from_array(arr);
        assert_eq!(v.reduce_min(), 5);
        assert_eq!(v.reduce_max(), 200);
    }

    #[test]
    fn u8x32_sum_bytes_u64_no_wrap() {
        assert_eq!(U8x32::splat(255).sum_bytes_u64(), 32 * 255);
    }

    #[test]
    fn u8x32_simd_min_max() {
        let a = U8x32::splat(100);
        let b = U8x32::splat(50);
        assert!(a.simd_min(b).to_array().iter().all(|&x| x == 50));
        assert!(a.simd_max(b).to_array().iter().all(|&x| x == 100));
    }

    #[test]
    fn u8x32_saturating_add() {
        let a = U8x32::splat(200);
        let b = U8x32::splat(100);
        assert!(a.saturating_add(b).to_array().iter().all(|&x| x == 255));
    }

    #[test]
    fn u8x32_saturating_sub() {
        let a = U8x32::splat(10);
        let b = U8x32::splat(20);
        assert!(a.saturating_sub(b).to_array().iter().all(|&x| x == 0));
    }

    #[test]
    fn u8x32_pairwise_avg() {
        let a = U8x32::splat(10);
        let b = U8x32::splat(11);
        assert!(a.pairwise_avg(b).to_array().iter().all(|&x| x == 11));
    }

    #[test]
    fn u8x32_cmpeq_mask_all_eq() {
        assert_eq!(U8x32::splat(5).cmpeq_mask(U8x32::splat(5)), u32::MAX);
    }

    #[test]
    fn u8x32_cmpgt_mask() {
        assert_eq!(U8x32::splat(10).cmpgt_mask(U8x32::splat(5)), u32::MAX);
        assert_eq!(U8x32::splat(5).cmpgt_mask(U8x32::splat(10)), 0u32);
    }

    #[test]
    fn u8x32_movemask() {
        assert_eq!(U8x32::splat(128).movemask(), u32::MAX);
        assert_eq!(U8x32::splat(127).movemask(), 0u32);
    }

    #[test]
    fn u8x32_shr_epi16() {
        let v = U8x32::splat(0x10);
        assert_eq!(v.shr_epi16(4).to_array()[0], 0x01);
    }

    #[test]
    fn u8x32_shl_epi16() {
        let v = U8x32::splat(0x01);
        assert_eq!(v.shl_epi16(4).to_array()[0], 0x10);
    }

    #[test]
    fn u8x32_nibble_popcount_lut() {
        let lut = U8x32::nibble_popcount_lut();
        let arr = lut.to_array();
        assert_eq!(arr[0], 0);
        assert_eq!(arr[3], 2);
        assert_eq!(arr[15], 4);
        assert_eq!(arr[16], 0);
        assert_eq!(arr[31], 4);
    }
}
