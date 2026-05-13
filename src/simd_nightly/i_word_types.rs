//! I16x16 / I16x32 / I32x16 / I64x8 portable-simd wrappers — round-3-portable-simd agent #6.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdInt;
use core::simd::{i16x16, i16x32, i32x16, i64x8};

// ════════════════════════════════════════════════════════════════════
// I16x16 — 16-lane signed 16-bit integer
// ════════════════════════════════════════════════════════════════════

/// 16-lane `i16` SIMD vector backed by `core::simd::i16x16`.
///
/// API mirrors `simd_avx512::I16x16`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I16x16(pub i16x16);

impl I16x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(i16x16::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 16]) -> Self {
        Self(i16x16::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16, "I16x16::from_slice needs ≥16 elements");
        Self(i16x16::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 16, "I16x16::copy_to_slice needs ≥16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u16` matching the 16-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }
}

impl PartialEq for I16x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I16x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16x16({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I16x32 — 32-lane signed 16-bit integer
// ════════════════════════════════════════════════════════════════════

/// 32-lane `i16` SIMD vector backed by `core::simd::i16x32`.
///
/// API mirrors `simd_avx512::I16x32`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I16x32(pub i16x32);

impl I16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(i16x32::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self(i16x32::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 32, "I16x32::from_slice needs ≥32 elements");
        Self(i16x32::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 32] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 32, "I16x32::copy_to_slice needs ≥32 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u32` matching the 32-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }
}

impl PartialEq for I16x32 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I16x32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16x32({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I32x16 — 16-lane signed 32-bit integer
// ════════════════════════════════════════════════════════════════════

/// 16-lane `i32` SIMD vector backed by `core::simd::i32x16`.
///
/// API mirrors `simd_avx512::I32x16`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I32x16(pub i32x16);

impl I32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(i32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 16]) -> Self {
        Self(i32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 16, "I32x16::from_slice needs ≥16 elements");
        Self(i32x16::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 16, "I32x16::copy_to_slice needs ≥16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u16` matching the 16-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }
}

impl PartialEq for I32x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I32x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I32x16({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I64x8 — 8-lane signed 64-bit integer
// ════════════════════════════════════════════════════════════════════

/// 8-lane `i64` SIMD vector backed by `core::simd::i64x8`.
///
/// API mirrors `simd_avx512::I64x8`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I64x8(pub i64x8);

impl I64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(i64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 8]) -> Self {
        Self(i64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 8, "I64x8::from_slice needs ≥8 elements");
        Self(i64x8::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 8, "I64x8::copy_to_slice needs ≥8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u8` matching the 8-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl PartialEq for I64x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I64x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I64x8({:?})", &self.to_array()[..])
    }
}
