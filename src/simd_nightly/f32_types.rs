//! F32x16 / F32x8 portable-simd wrappers — round-3-portable-simd agent #1.
#![cfg(feature = "nightly-simd")]

use core::simd::{f32x16 as core_f32x16, f32x8 as core_f32x8};
use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdFloat;
// `mul_add`, `sqrt`, `round`, `floor`, `abs` live in `StdFloat` (std-only nightly trait).
use std::simd::StdFloat;

use super::masks::{F32Mask16, F32Mask8};
use super::u_word_types::{U32x16, U32x8};

// ════════════════════════════════════════════════════════════════════
// F32x16 — 16-lane single-precision float
// ════════════════════════════════════════════════════════════════════

/// 16-lane `f32` SIMD vector backed by `core::simd::f32x16`.
///
/// API mirrors `simd_avx512::F32x16` so consumer code is identical.
/// Miri can execute every method below — unlike the intrinsics
/// backend, where SIMD paths are opaque to miri.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct F32x16(pub core_f32x16);

impl F32x16 {
    pub const LANES: usize = 16;

    // ── Constructors ──────────────────────────────────────────────

    /// Broadcast `v` to all 16 lanes.
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(core_f32x16::splat(v))
    }

    /// Load from the first 16 elements of `arr`.
    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self(core_f32x16::from_array(arr))
    }

    /// Load from the first 16 elements of `s`.
    ///
    /// # Panics
    /// Panics if `s.len() < 16`.
    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16, "F32x16::from_slice needs >= 16 elements");
        Self(core_f32x16::from_slice(s))
    }

    /// Copy all 16 lanes into a `[f32; 16]`.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        self.0.to_array()
    }

    /// Store all 16 lanes into the first 16 slots of `s`.
    ///
    /// # Panics
    /// Panics if `s.len() < 16`.
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16, "F32x16::copy_to_slice needs >= 16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Sum of all 16 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        self.0.reduce_sum()
    }

    /// Minimum lane value.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        self.0.reduce_min()
    }

    /// Maximum lane value.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max / clamp ───────────────────────────────

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

    /// Per-lane clamp: `lo <= self <= hi` for each lane.
    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    // ── FMA + math ────────────────────────────────────────────────

    /// Fused multiply-add: `self * b + c`.
    ///
    /// Maps to `_mm512_fmadd_ps` on AVX-512 builds via LLVM's portable-simd
    /// lowering; scalar `f32::mul_add` per lane otherwise.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Per-lane square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Per-lane round-to-nearest (ties to even).
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Per-lane floor (toward negative infinity).
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// Per-lane absolute value (clears sign bit).
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    // ── Bit reinterpretation ──────────────────────────────────────

    /// Reinterpret the 16 × 32-bit float lanes as `U32x16` (no conversion).
    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        U32x16(self.0.to_bits())
    }

    /// Reinterpret a `U32x16` as `F32x16` (no conversion).
    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        Self(core_f32x16::from_bits(bits.0))
    }

    // ── Comparisons → typed masks ─────────────────────────────────
    //
    // Return `super::masks::F32Mask16` (agent #7's type), which wraps
    // `core::simd::Mask<i32, 16>`.

    /// Per-lane equality: `self[i] == other[i]`.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_eq(other.0))
    }

    /// Per-lane inequality: `self[i] != other[i]`.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_ne(other.0))
    }

    /// Per-lane less-than: `self[i] < other[i]`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_lt(other.0))
    }

    /// Per-lane less-or-equal: `self[i] <= other[i]`.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_le(other.0))
    }

    /// Per-lane greater-than: `self[i] > other[i]`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_gt(other.0))
    }

    /// Per-lane greater-or-equal: `self[i] >= other[i]`.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_ge(other.0))
    }
}

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0.0)
    }
}

// ════════════════════════════════════════════════════════════════════
// F32x8 — 8-lane single-precision float
// ════════════════════════════════════════════════════════════════════

/// 8-lane `f32` SIMD vector backed by `core::simd::f32x8`.
///
/// API mirrors `simd_avx512::F32x16` / `F32x8` so consumer code is
/// identical.  Miri can execute every method below.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct F32x8(pub core_f32x8);

impl F32x8 {
    pub const LANES: usize = 8;

    // ── Constructors ──────────────────────────────────────────────

    /// Broadcast `v` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(core_f32x8::splat(v))
    }

    /// Load from the first 8 elements of `arr`.
    #[inline(always)]
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self(core_f32x8::from_array(arr))
    }

    /// Load from the first 8 elements of `s`.
    ///
    /// # Panics
    /// Panics if `s.len() < 8`.
    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 8, "F32x8::from_slice needs >= 8 elements");
        Self(core_f32x8::from_slice(s))
    }

    /// Copy all 8 lanes into a `[f32; 8]`.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        self.0.to_array()
    }

    /// Store all 8 lanes into the first 8 slots of `s`.
    ///
    /// # Panics
    /// Panics if `s.len() < 8`.
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 8, "F32x8::copy_to_slice needs >= 8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Sum of all 8 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        self.0.reduce_sum()
    }

    /// Minimum lane value.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        self.0.reduce_min()
    }

    /// Maximum lane value.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max / clamp ───────────────────────────────

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

    /// Per-lane clamp: `lo <= self <= hi` for each lane.
    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    // ── FMA + math ────────────────────────────────────────────────

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

    /// Per-lane round-to-nearest (ties to even).
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Per-lane floor (toward negative infinity).
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// Per-lane absolute value (clears sign bit).
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    // ── Bit reinterpretation ──────────────────────────────────────

    /// Reinterpret the 8 × 32-bit float lanes as `U32x8` (no conversion).
    #[inline(always)]
    pub fn to_bits(self) -> U32x8 {
        U32x8(self.0.to_bits())
    }

    /// Reinterpret a `U32x8` as `F32x8` (no conversion).
    #[inline(always)]
    pub fn from_bits(bits: U32x8) -> Self {
        Self(core_f32x8::from_bits(bits.0))
    }

    // ── Comparisons → typed masks ─────────────────────────────────
    //
    // Return `super::masks::F32Mask8` (agent #7's type), which wraps
    // `core::simd::Mask<i32, 8>`.

    /// Per-lane equality: `self[i] == other[i]`.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_eq(other.0))
    }

    /// Per-lane inequality: `self[i] != other[i]`.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_ne(other.0))
    }

    /// Per-lane less-than: `self[i] < other[i]`.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_lt(other.0))
    }

    /// Per-lane less-or-equal: `self[i] <= other[i]`.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_le(other.0))
    }

    /// Per-lane greater-than: `self[i] > other[i]`.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_gt(other.0))
    }

    /// Per-lane greater-or-equal: `self[i] >= other[i]`.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask8 {
        F32Mask8(self.0.simd_ge(other.0))
    }
}

impl Default for F32x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0.0)
    }
}
