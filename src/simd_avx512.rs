//! AVX-512 SIMD compatibility layer — stable Rust std::arch wrappers.
//!
//! Drop-in replacement for `std::simd` portable_simd types. Provides the same
//! API surface (methods, operators, type names) backed by `std::arch::x86_64`
//! intrinsics. All intrinsics used here are stable on Rust 1.89+.
//!
//! # Types
//!
//! | Compat type | portable_simd equiv | Backing type | Width |
//! |-------------|--------------------|--------------| ------|
//! | `F32x16`    | `f32x16`           | `__m512`     | 512b  |
//! | `F64x8`     | `f64x8`            | `__m512d`    | 512b  |
//! | `U8x64`     | `u8x64`            | `__m512i`    | 512b  |
//! | `I32x16`    | `i32x16`           | `__m512i`    | 512b  |
//! | `I64x8`     | `i64x8`            | `__m512i`    | 512b  |
//! | `U32x16`    | `u32x16`           | `__m512i`    | 512b  |
//! | `U64x8`     | `u64x8`            | `__m512i`    | 512b  |
//!
//! # Migration guide
//!
//! ```rust,ignore
//! // Before (nightly):
//! use std::simd::f32x16;
//! use std::simd::num::SimdFloat;
//!
//! // After (stable 1.93):
//! use crate::simd::f32x16;
//! // No trait imports needed — all methods are inherent.
//! ```

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

// ============================================================================
// Operator macros — reduce boilerplate for the 7 wrapper types
// ============================================================================

macro_rules! impl_bin_op {
    ($ty:ident, $trait:ident, $method:ident, $intr:path) => {
        impl $trait for $ty {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self(unsafe { $intr(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($ty:ident, $trait:ident, $method:ident, $intr:path) => {
        impl $trait for $ty {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.0 = unsafe { $intr(self.0, rhs.0) };
            }
        }
    };
}

// ============================================================================
// F32x16 — 16 × f32 in one AVX-512 register (__m512)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x16(pub __m512);

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm512_setzero_ps() })
    }
}

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm512_set1_ps(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_ps(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_ps(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let mut arr = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_ps(s.as_mut_ptr(), self.0) };
    }

    // --- Reductions ---

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        unsafe { _mm512_reduce_add_ps(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe { _mm512_reduce_min_ps(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe { _mm512_reduce_max_ps(self.0) }
    }

    // --- Element-wise min/max/clamp ---

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_ps(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_ps(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    // --- Math (StdFloat equivalents) ---

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_ps(self.0) })
    }

    /// Round to nearest integer (ties to even).
    #[inline(always)]
    pub fn round(self) -> Self {
        // IMM8: bits[1:0]=0 (nearest), bit[3]=1 (suppress exceptions) = 0x08
        Self(unsafe { _mm512_roundscale_ps::<0x08>(self.0) })
    }

    /// Floor (round toward negative infinity).
    #[inline(always)]
    pub fn floor(self) -> Self {
        // IMM8: bits[1:0]=1 (floor), bit[3]=1 (suppress exceptions) = 0x09
        Self(unsafe { _mm512_roundscale_ps::<0x09>(self.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm512_set1_epi32(0x7FFF_FFFFi32);
            Self(_mm512_castsi512_ps(_mm512_and_si512(
                _mm512_castps_si512(self.0),
                mask,
            )))
        }
    }

    // --- Bit reinterpretation ---

    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        U32x16(unsafe { _mm512_castps_si512(self.0) })
    }

    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        Self(unsafe { _mm512_castsi512_ps(bits.0) })
    }

    // --- Type casts ---

    /// Truncating cast f32→i32 (equivalent to `portable_simd .cast::<i32>()`).
    #[inline(always)]
    pub fn cast_i32(self) -> I32x16 {
        I32x16(unsafe { _mm512_cvttps_epi32(self.0) })
    }

    // --- Comparisons (return typed masks) ---

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_LE_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask16 {
        // GT(a, b) = LT(b, a)
        other.simd_lt(self)
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask16 {
        // GE(a, b) = LE(b, a)
        other.simd_le(self)
    }
}

impl_bin_op!(F32x16, Add, add, _mm512_add_ps);
impl_bin_op!(F32x16, Sub, sub, _mm512_sub_ps);
impl_bin_op!(F32x16, Mul, mul, _mm512_mul_ps);
impl_bin_op!(F32x16, Div, div, _mm512_div_ps);
impl_assign_op!(F32x16, AddAssign, add_assign, _mm512_add_ps);
impl_assign_op!(F32x16, SubAssign, sub_assign, _mm512_sub_ps);
impl_assign_op!(F32x16, MulAssign, mul_assign, _mm512_mul_ps);
impl_assign_op!(F32x16, DivAssign, div_assign, _mm512_div_ps);

impl Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm512_set1_epi32(i32::MIN); // 0x80000000
            Self(_mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_castps_si512(self.0),
                sign,
            )))
        }
    }
}

impl fmt::Debug for F32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32x16({:?})", self.to_array())
    }
}

impl PartialEq for F32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// F32Mask16 — 16-bit mask from f32 comparisons
// ============================================================================

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32Mask16(pub __mmask16);

impl F32Mask16 {
    /// Select: for each lane, if mask bit is 1 → true_val, else false_val.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        // _mm512_mask_blend_ps(k, a, b): if k[i] then b[i] else a[i]
        F32x16(unsafe { _mm512_mask_blend_ps(self.0, false_val.0, true_val.0) })
    }
}

// ============================================================================
// F64x8 — 8 × f64 in one AVX-512 register (__m512d)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F64x8(pub __m512d);

impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm512_setzero_pd() })
    }
}

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(unsafe { _mm512_set1_pd(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_pd(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_pd(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let mut arr = [0.0f64; 8];
        unsafe { _mm512_storeu_pd(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_pd(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe { _mm512_reduce_min_pd(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe { _mm512_reduce_max_pd(self.0) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_pd(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_pd(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_pd(self.0, b.0, c.0) })
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_pd(self.0) })
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x08>(self.0) })
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x09>(self.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFi64);
            Self(_mm512_castsi512_pd(_mm512_and_si512(
                _mm512_castpd_si512(self.0),
                mask,
            )))
        }
    }

    #[inline(always)]
    pub fn to_bits(self) -> U64x8 {
        U64x8(unsafe { _mm512_castpd_si512(self.0) })
    }

    #[inline(always)]
    pub fn from_bits(bits: U64x8) -> Self {
        Self(unsafe { _mm512_castsi512_pd(bits.0) })
    }

    // --- Comparisons ---

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_LT_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_LE_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F64Mask8 {
        other.simd_lt(self)
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F64Mask8 {
        other.simd_le(self)
    }
}

impl_bin_op!(F64x8, Add, add, _mm512_add_pd);
impl_bin_op!(F64x8, Sub, sub, _mm512_sub_pd);
impl_bin_op!(F64x8, Mul, mul, _mm512_mul_pd);
impl_bin_op!(F64x8, Div, div, _mm512_div_pd);
impl_assign_op!(F64x8, AddAssign, add_assign, _mm512_add_pd);
impl_assign_op!(F64x8, SubAssign, sub_assign, _mm512_sub_pd);
impl_assign_op!(F64x8, MulAssign, mul_assign, _mm512_mul_pd);
impl_assign_op!(F64x8, DivAssign, div_assign, _mm512_div_pd);

impl Neg for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm512_set1_epi64(i64::MIN); // 0x8000000000000000
            Self(_mm512_castsi512_pd(_mm512_xor_si512(
                _mm512_castpd_si512(self.0),
                sign,
            )))
        }
    }
}

impl fmt::Debug for F64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F64x8({:?})", self.to_array())
    }
}

impl PartialEq for F64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// F64Mask8 — 8-bit mask from f64 comparisons
// ============================================================================

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64Mask8(pub __mmask8);

impl F64Mask8 {
    #[inline(always)]
    pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
        F64x8(unsafe { _mm512_mask_blend_pd(self.0, false_val.0, true_val.0) })
    }
}

// ============================================================================
// U8x64 — 64 × u8 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U8x64(pub __m512i);

impl U8x64 {
    pub const LANES: usize = 64;

    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v as i8) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 64);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        let mut arr = [0u8; 64];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 64);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    /// Wrapping sum of all 64 bytes → u8 (matches portable_simd semantics).
    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        unsafe {
            // SAD against zero sums groups of 8 bytes → 8 × u64
            let sad = _mm512_sad_epu8(self.0, _mm512_setzero_si512());
            _mm512_reduce_add_epi64(sad) as u8
        }
    }

    /// Minimum of all 64 bytes.
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        // Tree reduction: 512→256→128→scalar
        let arr = self.to_array();
        let mut m = arr[0];
        for &val in arr.iter().skip(1) {
            if val < m {
                m = val;
            }
        }
        m
    }

    /// Maximum of all 64 bytes.
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        let arr = self.to_array();
        let mut m = arr[0];
        for &val in arr.iter().skip(1) {
            if val > m {
                m = val;
            }
        }
        m
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu8(self.0, other.0) })
    }

    // ── Byte-level operations for palette codec, nibble, byte scan ──────
    // Reference: Pumpkin/Minecraft-derived modules (palette_codec.rs,
    // nibble.rs, byte_scan.rs) use these for 4-bit packing and scanning.

    /// Byte-wise equality comparison. Returns 64-bit mask: bit i set if a[i] == b[i].
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        unsafe { _mm512_cmpeq_epi8_mask(self.0, other.0) }
    }

    /// Shift right each 16-bit lane by immediate bits (for nibble extraction).
    /// Note: operates on 16-bit lanes, not 8-bit — matches _mm512_srli_epi16.
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        // _mm512_srli_epi16 shifts each 16-bit lane right
        // Use match for const immediate (intrinsic requires const)
        Self(unsafe { match imm {
            1 => _mm512_srli_epi16(self.0, 1),
            2 => _mm512_srli_epi16(self.0, 2),
            3 => _mm512_srli_epi16(self.0, 3),
            4 => _mm512_srli_epi16(self.0, 4),
            5 => _mm512_srli_epi16(self.0, 5),
            6 => _mm512_srli_epi16(self.0, 6),
            7 => _mm512_srli_epi16(self.0, 7),
            8 => _mm512_srli_epi16(self.0, 8),
            _ => _mm512_setzero_si512(),
        }})
    }

    /// Saturating unsigned subtraction: max(a - b, 0) per byte.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(unsafe { _mm512_subs_epu8(self.0, other.0) })
    }

    /// Interleave low bytes: [a0,b0,a1,b1,...] from lower halves.
    #[inline(always)]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        Self(unsafe { _mm512_unpacklo_epi8(self.0, other.0) })
    }

    /// Interleave high bytes: [a8,b8,a9,b9,...] from upper halves.
    #[inline(always)]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        Self(unsafe { _mm512_unpackhi_epi8(self.0, other.0) })
    }

    /// Byte-wise shuffle: use `self` as a LUT, `idx` selects bytes within each 128-bit lane.
    /// Equivalent to `_mm512_shuffle_epi8(self.0, idx.0)`.
    #[inline(always)]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        Self(unsafe { _mm512_shuffle_epi8(self.0, idx.0) })
    }

    /// Sum all 64 bytes into a single `u64` without wrapping.
    ///
    /// Uses `_mm512_sad_epu8` (groups of 8 bytes → u64 lanes) then horizontal add.
    /// Range: 0..=64*255 = 16_320, always fits in u64.
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        unsafe {
            let sad = _mm512_sad_epu8(self.0, _mm512_setzero_si512());
            _mm512_reduce_add_epi64(sad) as u64
        }
    }

    /// Build a nibble-popcount lookup table (replicated across all 4 × 128-bit lanes).
    ///
    /// Entry `i` = popcount of `i` for i in 0..16. Used with `shuffle_bytes` for
    /// SIMD popcount via the Mula nibble-LUT algorithm.
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        // 0x04030302_03020201_03020201_02010100 replicated ×4
        Self(unsafe { _mm512_set4_epi32(
            0x04030302_u32 as i32,
            0x03020201_u32 as i32,
            0x03020201_u32 as i32,
            0x02010100_u32 as i32,
        )})
    }
}

// u8 add/sub use AVX-512BW instructions
impl_bin_op!(U8x64, Add, add, _mm512_add_epi8);
impl_bin_op!(U8x64, Sub, sub, _mm512_sub_epi8);
impl_assign_op!(U8x64, AddAssign, add_assign, _mm512_add_epi8);
impl_assign_op!(U8x64, SubAssign, sub_assign, _mm512_sub_epi8);

// u8 multiply — no single instruction; widen to u16, multiply, truncate back.
impl Mul for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Split into lower/upper 32-byte halves
            let a_lo = _mm512_castsi512_si256(self.0);
            let a_hi = _mm512_extracti64x4_epi64::<1>(self.0);
            let b_lo = _mm512_castsi512_si256(rhs.0);
            let b_hi = _mm512_extracti64x4_epi64::<1>(rhs.0);

            // Zero-extend u8→u16 (256→512 bits, 32 elements each)
            let a16_lo = _mm512_cvtepu8_epi16(a_lo);
            let a16_hi = _mm512_cvtepu8_epi16(a_hi);
            let b16_lo = _mm512_cvtepu8_epi16(b_lo);
            let b16_hi = _mm512_cvtepu8_epi16(b_hi);

            // Multiply as u16 (wrapping at 16-bit)
            let prod_lo = _mm512_mullo_epi16(a16_lo, b16_lo);
            let prod_hi = _mm512_mullo_epi16(a16_hi, b16_hi);

            // Truncate u16→u8 (keep low byte)
            let packed_lo = _mm512_cvtepi16_epi8(prod_lo);
            let packed_hi = _mm512_cvtepi16_epi8(prod_hi);

            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(packed_lo),
                packed_hi,
            ))
        }
    }
}

impl MulAssign for U8x64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// Bitwise ops for u8
impl_bin_op!(U8x64, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U8x64, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U8x64, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(U8x64, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(U8x64, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(U8x64, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi8(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl fmt::Debug for U8x64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U8x64({:?})", &self.to_array()[..])
    }
}

impl PartialEq for U8x64 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I32x16 — 16 × i32 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I32x16(pub __m512i);

impl I32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        let mut arr = [0i32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i32 {
        unsafe { _mm512_reduce_add_epi32(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        unsafe { _mm512_reduce_min_epi32(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        unsafe { _mm512_reduce_max_epi32(self.0) }
    }

    // ── Base17 i16[17] operations: load-widen, abs, narrow ──────────────
    // Used by bgz17_bridge.rs for L1 distance, weighted L1, sign agreement, xor_bind.

    /// Load 16 × i16 from slice, sign-extend to 16 × i32.
    /// This is the first step of every Base17 kernel: i16 → i32 to avoid overflow.
    #[inline(always)]
    pub fn from_i16_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_cvtepi16_epi32(_mm256_loadu_si256(s.as_ptr() as *const __m256i)) })
    }

    /// Absolute value per lane.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi32(self.0) })
    }

    /// Narrow 16 × i32 back to 16 × i16 (truncation, no saturation).
    #[inline(always)]
    pub fn to_i16_array(self) -> [i16; 16] {
        unsafe {
            let packed = _mm512_cvtepi32_epi16(self.0);
            let mut arr = [0i16; 16];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, packed);
            arr
        }
    }

    /// Compare >= 0: returns 16-bit mask. Bit i set where lane i >= 0.
    #[inline(always)]
    pub fn cmpge_zero_mask(self) -> u16 {
        unsafe { _mm512_cmpge_epi32_mask(self.0, _mm512_setzero_si512()) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi32(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi32(self.0, other.0) })
    }

    /// Cast i32→f32 (equivalent to `portable_simd .cast::<f32>()`).
    #[inline(always)]
    pub fn cast_f32(self) -> F32x16 {
        F32x16(unsafe { _mm512_cvtepi32_ps(self.0) })
    }
}

impl_bin_op!(I32x16, Add, add, _mm512_add_epi32);
impl_bin_op!(I32x16, Sub, sub, _mm512_sub_epi32);
impl_assign_op!(I32x16, AddAssign, add_assign, _mm512_add_epi32);
impl_assign_op!(I32x16, SubAssign, sub_assign, _mm512_sub_epi32);

// i32 multiply: _mm512_mullo_epi32 (AVX-512F)
impl_bin_op!(I32x16, Mul, mul, _mm512_mullo_epi32);
impl_assign_op!(I32x16, MulAssign, mul_assign, _mm512_mullo_epi32);

// i32 divide: no SIMD instruction — array fallback
impl Div for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i32; 16];
        for i in 0..16 {
            c[i] = a[i] / b[i];
        }
        Self::from_array(c)
    }
}

impl DivAssign for I32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Bitwise
impl_bin_op!(I32x16, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(I32x16, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(I32x16, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(I32x16, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(I32x16, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(I32x16, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl Neg for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { Self(_mm512_sub_epi32(_mm512_setzero_si512(), self.0)) }
    }
}

impl fmt::Debug for I32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I32x16({:?})", self.to_array())
    }
}

impl PartialEq for I32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I64x8 — 8 × i64 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I64x8(pub __m512i);

impl I64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
        let mut arr = [0i64; 8];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i64 {
        unsafe { _mm512_reduce_add_epi64(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        unsafe { _mm512_reduce_min_epi64(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        unsafe { _mm512_reduce_max_epi64(self.0) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi64(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi64(self.0, other.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi64(self.0) })
    }
}

impl_bin_op!(I64x8, Add, add, _mm512_add_epi64);
impl_bin_op!(I64x8, Sub, sub, _mm512_sub_epi64);
impl_assign_op!(I64x8, AddAssign, add_assign, _mm512_add_epi64);
impl_assign_op!(I64x8, SubAssign, sub_assign, _mm512_sub_epi64);

// i64 multiply: _mm512_mullo_epi64 (AVX-512DQ — available on all server CPUs)
impl Mul for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Fallback: array-based multiply (AVX-512DQ _mm512_mullo_epi64 may
        // not be available on all targets)
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i64; 8];
        for i in 0..8 {
            c[i] = a[i].wrapping_mul(b[i]);
        }
        Self::from_array(c)
    }
}

impl MulAssign for I64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// i64 divide: no SIMD instruction — array fallback
impl Div for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i64; 8];
        for i in 0..8 {
            c[i] = a[i] / b[i];
        }
        Self::from_array(c)
    }
}

impl DivAssign for I64x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Bitwise
impl_bin_op!(I64x8, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(I64x8, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(I64x8, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(I64x8, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(I64x8, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(I64x8, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi64(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl Neg for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { Self(_mm512_sub_epi64(_mm512_setzero_si512(), self.0)) }
    }
}

impl fmt::Debug for I64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I64x8({:?})", self.to_array())
    }
}

impl PartialEq for I64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// U32x16 — 16 × u32 in one AVX-512 register (__m512i)
// Used primarily for bit manipulation in transcendental functions (vml.rs).
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x16(pub __m512i);

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v as i32) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        let mut arr = [0u32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        unsafe { _mm512_reduce_add_epi32(self.0) as u32 }
    }
}

impl_bin_op!(U32x16, Add, add, _mm512_add_epi32);
impl_bin_op!(U32x16, Sub, sub, _mm512_sub_epi32);
impl_bin_op!(U32x16, Mul, mul, _mm512_mullo_epi32);
impl_assign_op!(U32x16, AddAssign, add_assign, _mm512_add_epi32);

// Bitwise
impl_bin_op!(U32x16, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U32x16, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U32x16, BitOr, bitor, _mm512_or_si512);

impl Not for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

// Shift operators for U32x16 (per-element variable shift)
impl Shr<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_srlv_epi32(self.0, rhs.0) })
    }
}

impl Shl<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sllv_epi32(self.0, rhs.0) })
    }
}

impl fmt::Debug for U32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U32x16({:?})", self.to_array())
    }
}

impl PartialEq for U32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// U64x8 — 8 × u64 in one AVX-512 register (__m512i)
// Used primarily for bit manipulation in transcendental functions and HDC.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U64x8(pub __m512i);

impl U64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v as i64) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        unsafe { _mm512_reduce_add_epi64(self.0) as u64 }
    }
}

impl_bin_op!(U64x8, Add, add, _mm512_add_epi64);
impl_bin_op!(U64x8, Sub, sub, _mm512_sub_epi64);
impl_assign_op!(U64x8, AddAssign, add_assign, _mm512_add_epi64);

// Bitwise
impl_bin_op!(U64x8, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U64x8, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U64x8, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(U64x8, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(U64x8, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(U64x8, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi64(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

// Shift operators for U64x8 (per-element variable shift)
impl Shr<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_srlv_epi64(self.0, rhs.0) })
    }
}

impl Shl<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sllv_epi64(self.0, rhs.0) })
    }
}

impl fmt::Debug for U64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U64x8({:?})", self.to_array())
    }
}

impl PartialEq for U64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// AVX2 wrapper types — 256-bit (F32x8, F64x4)
// ============================================================================
// Same pattern as AVX-512 wrappers above. Used by simd_avx2.rs when
// compiling with --features avx2 --no-default-features.
// All intrinsics are stable std::arch::x86_64 (avx/avx2).

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x8(pub __m256);

impl F32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm256_loadu_ps(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 8);
        unsafe { _mm256_storeu_ps(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        unsafe {
            // Extract upper 128 and add to lower 128
            let hi = _mm256_extractf128_ps(self.0, 1);
            let lo = _mm256_castps256_ps128(self.0);
            let sum128 = _mm_add_ps(lo, hi);
            // Horizontal reduce 4 floats
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
            let sum32 = _mm_add_ss(sum64, hi32);
            _mm_cvtss_f32(sum32)
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        // Clear sign bit: AND with 0x7FFFFFFF
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFi32));
            Self(_mm256_and_ps(self.0, mask))
        }
    }
}

impl Add for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }
}

impl AddAssign for F32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_ps(self.0, rhs.0) };
    }
}

impl Mul for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }
}

impl MulAssign for F32x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_mul_ps(self.0, rhs.0) };
    }
}

impl Sub for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }
}

impl SubAssign for F32x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_sub_ps(self.0, rhs.0) };
    }
}

impl Div for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }
}

impl DivAssign for F32x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_div_ps(self.0, rhs.0) };
    }
}

impl fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32x8({:?})", self.to_array())
    }
}

impl PartialEq for F32x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// --- F64x4 (AVX2: 4 × f64) ---

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F64x4(pub __m256d);

impl F64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(unsafe { _mm256_set1_pd(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 4);
        Self(unsafe { _mm256_loadu_pd(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(a: [f64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_pd(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 4);
        unsafe { _mm256_storeu_pd(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(self.0, 1);
            let lo = _mm256_castpd256_pd128(self.0);
            let sum128 = _mm_add_pd(lo, hi);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            let sum64 = _mm_add_sd(sum128, hi64);
            _mm_cvtsd_f64(sum64)
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFi64));
            Self(_mm256_and_pd(self.0, mask))
        }
    }
}

impl Add for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_pd(self.0, rhs.0) })
    }
}

impl AddAssign for F64x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_pd(self.0, rhs.0) };
    }
}

impl Mul for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mul_pd(self.0, rhs.0) })
    }
}

impl MulAssign for F64x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_mul_pd(self.0, rhs.0) };
    }
}

impl Sub for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_sub_pd(self.0, rhs.0) })
    }
}

impl SubAssign for F64x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_sub_pd(self.0, rhs.0) };
    }
}

impl Div for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_div_pd(self.0, rhs.0) })
    }
}

impl DivAssign for F64x4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_div_pd(self.0, rhs.0) };
    }
}

impl fmt::Debug for F64x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F64x4({:?})", self.to_array())
    }
}

impl PartialEq for F64x4 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// Type aliases — lowercase names matching portable_simd convention
// ============================================================================

#[allow(non_camel_case_types)]
pub type f32x16 = F32x16;
#[allow(non_camel_case_types)]
pub type f64x8 = F64x8;
#[allow(non_camel_case_types)]
pub type u8x64 = U8x64;
#[allow(non_camel_case_types)]
pub type i32x16 = I32x16;
#[allow(non_camel_case_types)]
pub type i64x8 = I64x8;
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;
#[allow(non_camel_case_types)]
pub type u64x8 = U64x8;

// AVX2 aliases (256-bit)
#[allow(non_camel_case_types)]
pub type f32x8 = F32x8;
#[allow(non_camel_case_types)]
pub type f64x4 = F64x4;

// ============================================================================
// BF16 conversion wrappers — AVX-512 BF16 hardware instructions
// ============================================================================
//
// Reference: https://doc.rust-lang.org/beta/src/core/stdarch/crates/core_arch/src/x86/avx512bf16.rs.html
//
// Hardware instructions (requires avx512bf16 + avx512vl):
//   _mm512_cvtpbh_ps:  16 BF16 → 16 f32   (__m256bh → __m512)
//   _mm256_cvtpbh_ps:   8 BF16 →  8 f32   (__m128bh → __m256)
//   _mm_cvtpbh_ps:      4 BF16 →  4 f32   (__m128bh → __m128)
//   _mm_cvtsbh_ss:      1 BF16 →  1 f32   (scalar)
//
//   _mm512_cvtneps_pbh: 16 f32 → 16 BF16  (__m512 → __m256bh)
//   _mm256_cvtneps_pbh:  8 f32 →  8 BF16  (__m256 → __m128bh)
//   _mm_cvtness_sbh:     1 f32 →  1 BF16  (scalar)
//
// These are NOT available on all AVX-512 CPUs — requires the BF16 extension.
// The scalar fallback (shift left 16) works everywhere.

/// BF16x16: 16 BF16 values packed in __m256bh. Converts to/from F32x16.
///
/// Primary use: bulk BF16→f32 hydration from GGUF source files.
/// One `vcvtneebf162ps` instruction converts 16 BF16 → 16 f32.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct BF16x16(pub __m256bh);

#[cfg(target_arch = "x86_64")]
impl BF16x16 {
    pub const LANES: usize = 16;

    /// Load 16 BF16 values from a u16 slice.
    ///
    /// SAFETY: Requires avx512bf16 at call site.
    /// Caller must ensure slice has >= 16 elements.
    #[inline]
    #[target_feature(enable = "avx512bf16")]
    pub unsafe fn from_u16_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 16);
        // __m256bh is 256 bits = 16 × u16. Load as __m256i then transmute.
        let raw = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
        Self(core::mem::transmute(raw))
    }

    /// Convert 16 BF16 → 16 f32 via hardware instruction.
    ///
    /// SAFETY: Requires avx512bf16 + avx512f at call site.
    /// Uses `vcvtneebf162ps` — one instruction, one cycle.
    #[inline]
    #[target_feature(enable = "avx512bf16,avx512f")]
    pub unsafe fn to_f32x16(self) -> F32x16 {
        F32x16(_mm512_cvtpbh_ps(self.0))
    }
}

/// BF16x8: 8 BF16 values packed in __m128bh. Converts to/from F32x8.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct BF16x8(pub __m128bh);

#[cfg(target_arch = "x86_64")]
impl BF16x8 {
    pub const LANES: usize = 8;

    /// Load 8 BF16 values from a u16 slice.
    #[inline]
    #[target_feature(enable = "avx512bf16")]
    pub unsafe fn from_u16_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8);
        let raw = _mm_loadu_si128(s.as_ptr() as *const __m128i);
        Self(core::mem::transmute(raw))
    }

    /// Convert 8 BF16 → 8 f32 via hardware instruction.
    #[inline]
    #[target_feature(enable = "avx512bf16,avx512vl")]
    pub unsafe fn to_f32x8(self) -> F32x8 {
        F32x8(_mm256_cvtpbh_ps(self.0))
    }
}

/// F32x16 → BF16x16 conversion (16 f32 → 16 BF16).
#[cfg(target_arch = "x86_64")]
impl F32x16 {
    /// Convert 16 f32 → 16 BF16 via hardware instruction.
    #[inline]
    #[target_feature(enable = "avx512bf16,avx512f")]
    pub unsafe fn to_bf16x16(self) -> BF16x16 {
        BF16x16(_mm512_cvtneps_pbh(self.0))
    }
}

/// F32x8 → BF16x8 conversion (8 f32 → 8 BF16).
#[cfg(target_arch = "x86_64")]
impl F32x8 {
    /// Convert 8 f32 → 8 BF16 via hardware instruction.
    #[inline]
    #[target_feature(enable = "avx512bf16,avx512vl")]
    pub unsafe fn to_bf16x8(self) -> BF16x8 {
        BF16x8(_mm256_cvtneps_pbh(self.0))
    }
}

// ── Scalar BF16 conversion (always available, no target_feature needed) ──

/// Scalar BF16 → f32: bit shift, one instruction, lossless.
/// Works on ALL platforms — this is the fallback when avx512bf16 is not available.
#[inline]
pub fn bf16_to_f32_scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Scalar f32 → BF16: truncate mantissa (lossy, 1 ULP).
#[inline]
pub fn f32_to_bf16_scalar(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Batch BF16 → f32 conversion: runtime feature detection + `as_chunks::<N>()`.
///
/// Uses stable Rust 1.94 `slice::as_chunks` for SIMD batch widths:
///   1. Runtime detect avx512bf16 + avx512vl
///   2. Process 16-wide chunks via `_mm512_cvtpbh_ps`
///   3. Process 8-wide remainder via `_mm256_cvtpbh_ps`
///   4. Finish scalar tail via bit shift
///
/// No LazyLock, no nightly. Just `as_chunks::<16>()` + `as_chunks::<8>()`.
pub fn bf16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output must be >= input length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bf16")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: feature detection confirmed avx512bf16 + avx512vl
            unsafe { convert_bf16_to_f32_avx512bf16(input, output); }
            return;
        }
    }

    // Scalar fallback (all platforms, all CPUs)
    for (src, dst) in input.iter().copied().zip(output.iter_mut()) {
        *dst = bf16_to_f32_scalar(src);
    }
}

/// Batch f32 → BF16 conversion: same pattern.
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) {
    assert!(output.len() >= input.len(), "output must be >= input length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bf16")
            && is_x86_feature_detected!("avx512vl")
        {
            unsafe { convert_f32_to_bf16_avx512bf16(input, output); }
            return;
        }
    }

    for (src, dst) in input.iter().copied().zip(output.iter_mut()) {
        *dst = f32_to_bf16_scalar(src);
    }
}

/// AVX-512 BF16 path: as_chunks::<16>() → as_chunks::<8>() → scalar tail.
///
/// Reference: https://doc.rust-lang.org/beta/src/core/stdarch/crates/core_arch/src/x86/avx512bf16.rs.html
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn convert_bf16_to_f32_avx512bf16(input: &[u16], output: &mut [f32]) {
    // 16-wide chunks
    let (chunks16, rem16) = input.as_chunks::<16>();
    let (out16, out_rem16) = output[..input.len()].as_chunks_mut::<16>();

    for (src, dst) in chunks16.iter().zip(out16.iter_mut()) {
        // SAFETY: [u16; 16] = 256 bits = __m256bh
        let v_bf16: __m256bh = core::mem::transmute(*src);
        let v_f32: __m512 = _mm512_cvtpbh_ps(v_bf16);
        *dst = core::mem::transmute(v_f32);
    }

    // 8-wide remainder chunks
    let (chunks8, rem8) = rem16.as_chunks::<8>();
    let (out8, out_rem8) = out_rem16.as_chunks_mut::<8>();

    for (src, dst) in chunks8.iter().zip(out8.iter_mut()) {
        let v_bf16: __m128bh = core::mem::transmute(*src);
        let v_f32: __m256 = _mm256_cvtpbh_ps(v_bf16);
        *dst = core::mem::transmute(v_f32);
    }

    // Scalar tail (0-7 remaining values)
    for (src, dst) in rem8.iter().copied().zip(out_rem8.iter_mut()) {
        *dst = f32::from_bits((src as u32) << 16);
    }
}

/// AVX-512 BF16 path for f32 → BF16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn convert_f32_to_bf16_avx512bf16(input: &[f32], output: &mut [u16]) {
    let (chunks16, rem16) = input.as_chunks::<16>();
    let (out16, out_rem16) = output[..input.len()].as_chunks_mut::<16>();

    for (src, dst) in chunks16.iter().zip(out16.iter_mut()) {
        let v_f32: __m512 = core::mem::transmute(*src);
        let v_bf16: __m256bh = _mm512_cvtneps_pbh(v_f32);
        *dst = core::mem::transmute(v_bf16);
    }

    // Scalar remainder (f32→BF16 has no 8-wide instruction worth using)
    for (src, dst) in rem16.iter().copied().zip(out_rem16.iter_mut()) {
        *dst = (src.to_bits() >> 16) as u16;
    }
}

#[cfg(test)]
mod bf16_tests {
    use super::*;

    #[test]
    fn scalar_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, 0.001, -0.001] {
            let bf16 = f32_to_bf16_scalar(v);
            let back = bf16_to_f32_scalar(bf16);
            let err = (v - back).abs() / v.abs().max(1e-6);
            assert!(err < 0.02, "roundtrip error for {}: {} → {} → {}, err={:.4}", v, v, bf16, back, err);
        }
    }

    #[test]
    fn batch_conversion_matches_scalar() {
        let input: Vec<u16> = (0..100).map(|i| f32_to_bf16_scalar(i as f32 * 0.1 - 5.0)).collect();
        let mut batch_output = vec![0.0f32; 100];
        bf16_to_f32_batch(&input, &mut batch_output);

        for (i, &bf16) in input.iter().enumerate() {
            let scalar = bf16_to_f32_scalar(bf16);
            assert_eq!(batch_output[i], scalar, "mismatch at index {}", i);
        }
    }

    #[test]
    fn batch_f32_to_bf16() {
        let input: Vec<f32> = (0..50).map(|i| i as f32 * 0.3 - 7.5).collect();
        let mut output = vec![0u16; 50];
        f32_to_bf16_batch(&input, &mut output);

        for (i, &v) in input.iter().enumerate() {
            let expected = f32_to_bf16_scalar(v);
            // Allow ±1 ULP: hardware uses round-to-nearest-even, scalar uses truncation
            let diff = (output[i] as i32 - expected as i32).unsigned_abs();
            assert!(diff <= 1, "mismatch at index {}: {} → {} vs {}, diff={}", i, v, output[i], expected, diff);
        }
    }
}
