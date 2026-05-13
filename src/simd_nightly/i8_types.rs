//! I8x32 / I8x64 portable-simd wrappers — round-3-portable-simd agent #5.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdInt;
use core::simd::{i8x32, i8x64};

// ════════════════════════════════════════════════════════════════════
// I8x64 — 64-lane signed byte
// Mirrors `simd_avx512::I8x64` surface (AVX-512BW path).
// ════════════════════════════════════════════════════════════════════

/// 64-lane `i8` SIMD vector backed by `core::simd::i8x64`.
///
/// API mirrors `simd_avx512::I8x64` so consumer code is backend-agnostic.
/// Every method executes under miri, unlike the AVX-512 intrinsics path.
///
/// # Examples
///
/// ```rust,ignore
/// let a = I8x64::splat(3);
/// let b = I8x64::splat(5);
/// assert_eq!(a.reduce_max(), 3);
/// assert_eq!(a.saturating_add(b).to_array(), [8i8; 64]);
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I8x64(pub i8x64);

impl I8x64 {
    /// Number of `i8` lanes.
    pub const LANES: usize = 64;

    /// Broadcast `v` into every lane.
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(i8x64::splat(v))
    }

    /// Load from a slice of at least 64 elements (panics otherwise).
    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 64, "I8x64::from_slice needs ≥64 elements");
        Self(i8x64::from_slice(s))
    }

    /// Construct from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [i8; 64]) -> Self {
        Self(i8x64::from_array(arr))
    }

    /// Extract all lanes as a fixed-size array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 64] {
        self.0.to_array()
    }

    /// Store all lanes into `s[0..64]` (panics if `s.len() < 64`).
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 64, "I8x64::copy_to_slice needs ≥64 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all 64 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i8 {
        self.0.reduce_sum()
    }

    /// Minimum across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> i8 {
        self.0.reduce_min()
    }

    /// Maximum across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> i8 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    /// Per-lane minimum.
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Per-lane maximum.
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    /// Per-lane signed saturating add. Results clamp to `[i8::MIN, i8::MAX]`.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Per-lane signed saturating subtract. Results clamp to `[i8::MIN, i8::MAX]`.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality mask. Returns a 64-bit integer; bit i is set iff
    /// `self[i] == other[i]`. Matches the `__mmask64` shape of the
    /// AVX-512BW backend.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        self.0.simd_eq(other.0).to_bitmask()
    }

    /// Per-lane signed greater-than mask. Returns a 64-bit integer; bit i is
    /// set iff `self[i] > other[i]` (signed comparison). Matches the
    /// `_mm512_cmpgt_epi8_mask` shape of the AVX-512BW backend.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        self.0.simd_gt(other.0).to_bitmask()
    }
}

impl PartialEq for I8x64 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// I8x32 — 32-lane signed byte
// Mirrors `simd_avx512::I8x32` surface (AVX2/AVX-512BW path).
// ════════════════════════════════════════════════════════════════════

/// 32-lane `i8` SIMD vector backed by `core::simd::i8x32`.
///
/// API mirrors `simd_avx512::I8x32` so consumer code is backend-agnostic.
///
/// # Examples
///
/// ```rust,ignore
/// let a = I8x32::splat(-10);
/// let b = I8x32::splat(20);
/// assert_eq!(a.saturating_add(b).to_array(), [10i8; 32]);
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I8x32(pub i8x32);

impl I8x32 {
    /// Number of `i8` lanes.
    pub const LANES: usize = 32;

    /// Broadcast `v` into every lane.
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(i8x32::splat(v))
    }

    /// Load from a slice of at least 32 elements (panics otherwise).
    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 32, "I8x32::from_slice needs ≥32 elements");
        Self(i8x32::from_slice(s))
    }

    /// Construct from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [i8; 32]) -> Self {
        Self(i8x32::from_array(arr))
    }

    /// Extract all lanes as a fixed-size array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 32] {
        self.0.to_array()
    }

    /// Store all lanes into `s[0..32]` (panics if `s.len() < 32`).
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 32, "I8x32::copy_to_slice needs ≥32 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all 32 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i8 {
        self.0.reduce_sum()
    }

    /// Minimum across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> i8 {
        self.0.reduce_min()
    }

    /// Maximum across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> i8 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    /// Per-lane minimum.
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Per-lane maximum.
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    /// Per-lane signed saturating add. Results clamp to `[i8::MIN, i8::MAX]`.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Per-lane signed saturating subtract. Results clamp to `[i8::MIN, i8::MAX]`.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality mask. Returns a 32-bit integer; bit i is set iff
    /// `self[i] == other[i]`. Matches the 32-lane mask shape of the
    /// AVX2 backend's `movemask_epi8` output.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane signed greater-than mask. Returns a 32-bit integer; bit i is
    /// set iff `self[i] > other[i]` (signed comparison). Matches the
    /// `_mm256_movemask_epi8(_mm256_cmpgt_epi8(...))` shape of the AVX2 backend.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }
}

impl PartialEq for I8x32 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}
