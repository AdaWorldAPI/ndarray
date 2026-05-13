//! U16x32, U32x16, U64x8, U64x4 portable-simd wrappers — round-3-portable-simd agent #4.
//! (U64x4 added here at agent #2's request for F64x4::to_bits support.)
#![cfg(feature = "nightly-simd")]

use core::simd::{u16x32, u32x8, u32x16, u64x8, u64x4};

// ════════════════════════════════════════════════════════════════════
// U64x8 — 8-lane u64
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u64` SIMD vector backed by `core::simd::u64x8`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U64x8(pub u64x8);

impl U64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self(u64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        self.0.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// U64x4 — 4-lane u64 (companion for F64x4::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 4-lane `u64` SIMD vector backed by `core::simd::u64x4`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U64x4(pub u64x4);

impl U64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x4::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 4]) -> Self {
        Self(u64x4::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 4] {
        self.0.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x8 — 8-lane u32 (companion for F32x8::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u32` SIMD vector backed by `core::simd::u32x8`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U32x8(pub u32x8);

impl U32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x8::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self(u32x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        self.0.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x16 — 16-lane u32
// ════════════════════════════════════════════════════════════════════

/// 16-lane `u32` SIMD vector backed by `core::simd::u32x16`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U32x16(pub u32x16);

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(u32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        self.0.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// U16x32 — 32-lane u16
// ════════════════════════════════════════════════════════════════════

/// 32-lane `u16` SIMD vector backed by `core::simd::u16x32`.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U16x32(pub u16x32);

impl U16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(u16x32::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [u16; 32]) -> Self {
        Self(u16x32::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 32] {
        self.0.to_array()
    }
}
