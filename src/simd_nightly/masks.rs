//! Comparison-result mask wrappers — round-3-portable-simd agent #7.
#![cfg(feature = "nightly-simd")]

use core::simd::Mask;
use core::simd::prelude::Select;
use super::{F32x16, F32x8, F64x8, F64x4};

// ============================================================================
// F32Mask16 — 16-lane mask for F32x16 comparisons
// ============================================================================

/// 16-lane mask wrapping `core::simd::Mask<i32, 16>`.
///
/// Mirrors `simd_avx512::F32Mask16` so consumer code compiles unchanged
/// under `--features nightly-simd`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32Mask16(pub Mask<i32, 16>);

impl F32Mask16 {
    /// Convert to a 16-bit packed bitmask (matches the AVX-512 `__mmask16`
    /// shape). Bit i is set iff lane i of the mask is true.
    #[inline(always)]
    pub fn to_bitmask(self) -> u16 {
        self.0.to_bitmask() as u16
    }

    /// Reconstruct a mask from a 16-bit packed bitmask.
    #[inline(always)]
    pub fn from_bitmask(bits: u16) -> Self {
        Self(Mask::<i32, 16>::from_bitmask(bits as u64))
    }

    /// Per-lane select: returns `true_val[i]` where mask[i] is set,
    /// else `false_val[i]`.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        F32x16(self.0.select(true_val.0, false_val.0))
    }

    /// Returns `true` if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.0.all()
    }

    /// Returns `true` if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.0.any()
    }
}

// ============================================================================
// F32Mask8 — 8-lane mask for F32x8 comparisons
// ============================================================================

/// 8-lane mask wrapping `core::simd::Mask<i32, 8>`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32Mask8(pub Mask<i32, 8>);

impl F32Mask8 {
    /// Convert to an 8-bit packed bitmask. Bit i is set iff lane i is true.
    #[inline(always)]
    pub fn to_bitmask(self) -> u8 {
        self.0.to_bitmask() as u8
    }

    /// Reconstruct a mask from an 8-bit packed bitmask.
    #[inline(always)]
    pub fn from_bitmask(bits: u8) -> Self {
        Self(Mask::<i32, 8>::from_bitmask(bits as u64))
    }

    /// Per-lane select: returns `true_val[i]` where mask[i] is set,
    /// else `false_val[i]`.
    #[inline(always)]
    pub fn select(self, true_val: F32x8, false_val: F32x8) -> F32x8 {
        F32x8(self.0.select(true_val.0, false_val.0))
    }

    /// Returns `true` if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.0.all()
    }

    /// Returns `true` if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.0.any()
    }
}

// ============================================================================
// F64Mask8 — 8-lane mask for F64x8 comparisons
// ============================================================================

/// 8-lane mask wrapping `core::simd::Mask<i64, 8>`.
///
/// Mirrors `simd_avx512::F64Mask8` so consumer code compiles unchanged
/// under `--features nightly-simd`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64Mask8(pub Mask<i64, 8>);

impl F64Mask8 {
    /// Convert to an 8-bit packed bitmask. Bit i is set iff lane i is true.
    #[inline(always)]
    pub fn to_bitmask(self) -> u8 {
        self.0.to_bitmask() as u8
    }

    /// Reconstruct a mask from an 8-bit packed bitmask.
    #[inline(always)]
    pub fn from_bitmask(bits: u8) -> Self {
        Self(Mask::<i64, 8>::from_bitmask(bits as u64))
    }

    /// Per-lane select: returns `true_val[i]` where mask[i] is set,
    /// else `false_val[i]`.
    #[inline(always)]
    pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
        F64x8(self.0.select(true_val.0, false_val.0))
    }

    /// Returns `true` if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.0.all()
    }

    /// Returns `true` if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.0.any()
    }
}

// ============================================================================
// F64Mask4 — 4-lane mask for F64x4 comparisons
// ============================================================================

/// 4-lane mask wrapping `core::simd::Mask<i64, 4>`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64Mask4(pub Mask<i64, 4>);

impl F64Mask4 {
    /// Convert to a 4-bit packed bitmask. Bit i is set iff lane i is true.
    #[inline(always)]
    pub fn to_bitmask(self) -> u8 {
        self.0.to_bitmask() as u8
    }

    /// Reconstruct a mask from a 4-bit packed bitmask.
    #[inline(always)]
    pub fn from_bitmask(bits: u8) -> Self {
        Self(Mask::<i64, 4>::from_bitmask(bits as u64))
    }

    /// Per-lane select: returns `true_val[i]` where mask[i] is set,
    /// else `false_val[i]`.
    #[inline(always)]
    pub fn select(self, true_val: F64x4, false_val: F64x4) -> F64x4 {
        F64x4(self.0.select(true_val.0, false_val.0))
    }

    /// Returns `true` if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.0.all()
    }

    /// Returns `true` if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.0.any()
    }
}
