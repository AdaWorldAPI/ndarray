//! F64x8 / F64x4 portable-simd wrappers — round-3-portable-simd agent #2.
#![cfg(feature = "nightly-simd")]

use core::simd::{f64x8 as core_f64x8, f64x4 as core_f64x4};
use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdFloat;

// ════════════════════════════════════════════════════════════════════
// F64x8 — 8-lane double-precision float
// ════════════════════════════════════════════════════════════════════

/// 8-lane `f64` SIMD vector backed by `core::simd::f64x8`.
///
/// API mirrors `simd_avx512::F64x8` so consumer code is identical.
/// Miri can execute every method — unlike the intrinsics backend where
/// SIMD paths are opaque to miri.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64x8(pub core_f64x8);

impl F64x8 {
    /// Number of f64 lanes.
    pub const LANES: usize = 8;

    // ── Constructors ───────────────────────────────────────────────

    /// Broadcast a scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(core_f64x8::splat(v))
    }

    /// Load from the first 8 elements of `s`. Panics if `s.len() < 8`.
    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8, "F64x8::from_slice needs ≥8 elements");
        Self(core_f64x8::from_slice(s))
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self(core_f64x8::from_array(arr))
    }

    /// Convert to a fixed-size array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        self.0.to_array()
    }

    /// Store to the first 8 elements of `s`. Panics if `s.len() < 8`.
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8, "F64x8::copy_to_slice needs ≥8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ─────────────────────────────────────────────────

    /// Horizontal sum of all 8 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        self.0.reduce_sum()
    }

    /// Horizontal minimum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        self.0.reduce_min()
    }

    /// Horizontal maximum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max / clamp ────────────────────────────────

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

    /// Per-lane clamp to `[lo, hi]`.
    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    // ── FMA + math ─────────────────────────────────────────────────

    /// Fused multiply-add: `self * b + c`.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Per-lane square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Per-lane round-to-nearest-even.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Per-lane floor.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// Per-lane absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    // ── Bit reinterpretation ───────────────────────────────────────

    /// Reinterpret the bit pattern as a `U64x8`.
    #[inline(always)]
    pub fn to_bits(self) -> super::u_word_types::U64x8 {
        super::u_word_types::U64x8(self.0.to_bits())
    }

    // ── Comparisons → mask ─────────────────────────────────────────

    /// Per-lane `==`.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_eq(other.0))
    }

    /// Per-lane `!=`.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_ne(other.0))
    }

    /// Per-lane `<`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_lt(other.0))
    }

    /// Per-lane `<=`.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_le(other.0))
    }

    /// Per-lane `>`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_gt(other.0))
    }

    /// Per-lane `>=`.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> super::masks::F64Mask8 {
        super::masks::F64Mask8(self.0.simd_ge(other.0))
    }
}

// ════════════════════════════════════════════════════════════════════
// F64x4 — 4-lane double-precision float
// ════════════════════════════════════════════════════════════════════

/// 4-lane `f64` SIMD vector backed by `core::simd::f64x4`.
///
/// API mirrors `simd_avx512::F64x4` at half the F64x8 width.
/// Miri-executable via the `core::simd` portable backend.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64x4(pub core_f64x4);

impl F64x4 {
    /// Number of f64 lanes.
    pub const LANES: usize = 4;

    // ── Constructors ───────────────────────────────────────────────

    /// Broadcast a scalar to all 4 lanes.
    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(core_f64x4::splat(v))
    }

    /// Load from the first 4 elements of `s`. Panics if `s.len() < 4`.
    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 4, "F64x4::from_slice needs ≥4 elements");
        Self(core_f64x4::from_slice(s))
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self(core_f64x4::from_array(arr))
    }

    /// Convert to a fixed-size array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        self.0.to_array()
    }

    /// Store to the first 4 elements of `s`. Panics if `s.len() < 4`.
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 4, "F64x4::copy_to_slice needs ≥4 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ─────────────────────────────────────────────────

    /// Horizontal sum of all 4 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        self.0.reduce_sum()
    }

    /// Horizontal minimum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        self.0.reduce_min()
    }

    /// Horizontal maximum across all 4 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max / clamp ────────────────────────────────

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

    /// Per-lane clamp to `[lo, hi]`.
    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    // ── FMA + math ─────────────────────────────────────────────────

    /// Fused multiply-add: `self * b + c`.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Per-lane square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Per-lane round-to-nearest-even.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Per-lane floor.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// Per-lane absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    // ── Bit reinterpretation ───────────────────────────────────────

    /// Reinterpret the bit pattern as a `U64x4`.
    #[inline(always)]
    pub fn to_bits(self) -> super::u_word_types::U64x4 {
        super::u_word_types::U64x4(self.0.to_bits())
    }

    // ── Comparisons → mask ─────────────────────────────────────────

    /// Per-lane `==`.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_eq(other.0))
    }

    /// Per-lane `!=`.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_ne(other.0))
    }

    /// Per-lane `<`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_lt(other.0))
    }

    /// Per-lane `<=`.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_le(other.0))
    }

    /// Per-lane `>`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_gt(other.0))
    }

    /// Per-lane `>=`.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> super::masks::F64Mask4 {
        super::masks::F64Mask4(self.0.simd_ge(other.0))
    }
}
