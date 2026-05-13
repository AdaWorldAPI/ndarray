//! U16x32 / U32x16 / U32x8 / U64x8 / U64x4 portable-simd wrappers — round-3-portable-simd agent #4.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdUint;
use core::simd::{u16x32, u32x16, u32x8, u64x4, u64x8};

// ════════════════════════════════════════════════════════════════════
// U64x8 — 8-lane u64
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u64` SIMD vector backed by `core::simd::u64x8`.
///
/// Also used as the return type of `F64x8::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U64x8(pub u64x8);

impl U64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 8, "U64x8::from_slice needs >=8 elements");
        Self(u64x8::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self(u64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 8, "U64x8::copy_to_slice needs >=8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U64x4 — 4-lane u64 (companion for F64x4::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 4-lane `u64` SIMD vector backed by `core::simd::u64x4`.
///
/// Return type of `F64x4::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U64x4(pub u64x4);

impl U64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x4::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 4, "U64x4::from_slice needs >=4 elements");
        Self(u64x4::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 4]) -> Self {
        Self(u64x4::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 4] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 4, "U64x4::copy_to_slice needs >=4 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit value (4 bits used; bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit value (4 bits used).
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U64x4 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x8 — 8-lane u32 (companion for F32x8::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u32` SIMD vector backed by `core::simd::u32x8`.
///
/// Return type of `F32x8::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U32x8(pub u32x8);

impl U32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 8, "U32x8::from_slice needs >=8 elements");
        Self(u32x8::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self(u32x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 8, "U32x8::copy_to_slice needs >=8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U32x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x16 — 16-lane u32
// ════════════════════════════════════════════════════════════════════

/// 16-lane `u32` SIMD vector backed by `core::simd::u32x16`.
///
/// Also used as the return type of `F32x16::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U32x16(pub u32x16);

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 16, "U32x16::from_slice needs >=16 elements");
        Self(u32x16::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(u32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 16, "U32x16::copy_to_slice needs >=16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns a 16-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane unsigned greater-than. Returns a 16-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }
}

impl Default for U32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U16x32 — 32-lane u16
// ════════════════════════════════════════════════════════════════════

/// 32-lane `u16` SIMD vector backed by `core::simd::u16x32`.
///
/// API mirrors `simd_avx512::U16x32`. Miri-executable.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U16x32(pub u16x32);

impl U16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(u16x32::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 32, "U16x32::from_slice needs >=32 elements");
        Self(u16x32::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u16; 32]) -> Self {
        Self(u16x32::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 32] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 32, "U16x32::copy_to_slice needs >=32 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all 32 lanes (result is u16, wraps on overflow).
    #[inline(always)]
    pub fn reduce_sum(self) -> u16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

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

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns a 32-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane unsigned greater-than. Returns a 32-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }
}

impl Default for U16x32 {
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

    #[test]
    fn u16x32_splat_reduce() {
        let v = U16x32::splat(3);
        assert_eq!(v.reduce_min(), 3);
        assert_eq!(v.reduce_max(), 3);
    }

    #[test]
    fn u16x32_cmpeq_mask_all_equal() {
        let a = U16x32::splat(10);
        let b = U16x32::splat(10);
        assert_eq!(a.cmpeq_mask(b), u32::MAX);
    }

    #[test]
    fn u16x32_saturating_add_clamps() {
        let a = U16x32::splat(60000u16);
        let b = U16x32::splat(10000u16);
        let c = a.saturating_add(b);
        assert!(c.to_array().iter().all(|&v| v == u16::MAX));
    }

    #[test]
    fn u32x16_splat_reduce() {
        let v = U32x16::splat(7);
        assert_eq!(v.reduce_sum(), 112u32); // 16 * 7
        assert_eq!(v.reduce_min(), 7);
        assert_eq!(v.reduce_max(), 7);
    }

    #[test]
    fn u32x16_cmpgt_mask() {
        let mut arr = [0u32; 16];
        for (i, x) in arr.iter_mut().enumerate() {
            *x = i as u32;
        }
        let v = U32x16::from_array(arr);
        let threshold = U32x16::splat(7);
        // Lanes 8..15 are > 7 -> bits 8..15 set -> 0xFF00
        assert_eq!(v.cmpgt_mask(threshold), 0xFF00u16);
    }

    #[test]
    fn u32x8_splat_reduce() {
        let v = U32x8::splat(5);
        assert_eq!(v.reduce_sum(), 40u32); // 8 * 5
        assert_eq!(v.reduce_min(), 5);
        assert_eq!(v.reduce_max(), 5);
    }

    #[test]
    fn u32x8_cmpeq_mask() {
        let a = U32x8::from_array([1, 2, 1, 2, 1, 2, 1, 2]);
        let b = U32x8::splat(1);
        // Lanes 0,2,4,6 equal -> bits 0,2,4,6 -> 0b01010101 = 0x55
        assert_eq!(a.cmpeq_mask(b), 0x55u8);
    }

    #[test]
    fn u64x8_splat_reduce() {
        let v = U64x8::splat(100);
        assert_eq!(v.reduce_sum(), 800u64); // 8 * 100
        assert_eq!(v.reduce_min(), 100);
        assert_eq!(v.reduce_max(), 100);
    }

    #[test]
    fn u64x8_cmpeq_mask_all() {
        let a = U64x8::splat(42);
        let b = U64x8::splat(42);
        assert_eq!(a.cmpeq_mask(b), 0xFFu8);
    }

    #[test]
    fn u64x4_splat_reduce() {
        let v = U64x4::splat(9);
        assert_eq!(v.reduce_sum(), 36u64); // 4 * 9
        assert_eq!(v.reduce_min(), 9);
        assert_eq!(v.reduce_max(), 9);
    }

    #[test]
    fn u64x4_cmpeq_mask_partial() {
        let a = U64x4::from_array([1, 2, 1, 2]);
        let b = U64x4::splat(1);
        // Lanes 0,2 equal -> bits 0,2 -> 0b0101 = 5
        assert_eq!(a.cmpeq_mask(b), 0x05u8);
    }

    #[test]
    fn u64x4_cmpgt_mask() {
        let a = U64x4::from_array([10, 1, 10, 1]);
        let b = U64x4::splat(5);
        // Lanes 0,2 > 5 -> bits 0,2 -> 0b0101 = 5
        assert_eq!(a.cmpgt_mask(b), 0x05u8);
    }
}
