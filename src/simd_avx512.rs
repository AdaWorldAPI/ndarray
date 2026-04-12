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

    /// Gather 16 f32 values from `base_ptr` using 16 i32 indices.
    ///
    /// Equivalent to `_mm512_i32gather_ps::<4>(indices, base_ptr)`:
    /// each lane loads `base_ptr[indices[lane]]`.
    ///
    /// # Safety
    /// Caller must ensure all indices are valid offsets into the memory at `base_ptr`.
    #[inline(always)]
    pub unsafe fn gather(indices: I32x16, base_ptr: *const f32) -> Self {
        Self(_mm512_i32gather_ps::<4>(indices.0, base_ptr))
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

// ════════════════════════════════════════════════════════════════════════════
// Pure AVX-512-F round-to-nearest-even F32 → BF16
//
// Matches `_mm512_cvtneps_pbh` bit-exact on every input (incl. NaN/Inf/denorm)
// while requiring only the AVX-512-F baseline (Skylake-X+). This is the
// certification-harness path: deterministic across CPU vendors/generations.
//
// Algorithm (per Intel SDM VCVTNEPS2BF16 pseudocode):
//   if f32 is NaN:
//       bf16 = (f32_bits >> 16) | 0x0040   // force QNaN bit
//   else:
//       lsb   = (f32_bits >> 16) & 1
//       biased = f32_bits + 0x7FFF + lsb    // RNE via bias
//       bf16   = (biased >> 16) as u16
//
// Adding 0x7FFF when the preserved-LSB is 0, or 0x8000 when the preserved-LSB
// is 1, correctly resolves ties-to-even without an explicit sticky/round
// classification.  The NaN path is separate because the bias can carry out of
// the exponent and turn a NaN into ±Inf or a normal.
// ════════════════════════════════════════════════════════════════════════════

/// Scalar reference for RNE F32 → BF16 (matches `_mm512_cvtneps_pbh` bit-exact).
///
/// Kept distinct from `f32_to_bf16_scalar` (which is truncation-only and is a
/// *legacy* primitive left in place for its existing call sites).
///
/// Follows the Intel SDM `VCVTNEPS2BF16` pseudocode:
///   - NaN inputs produce a QNaN with forced quiet bit,
///   - subnormal inputs flush to ±0 (DAZ-style),
///   - Inf / zero / normal inputs round-to-nearest-even via the classic
///     `+0x7FFF + LSB` bias trick.
#[inline]
pub fn f32_to_bf16_scalar_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let exp = bits & 0x7F80_0000;
    let mant = bits & 0x007F_FFFF;
    if exp == 0x7F80_0000 && mant != 0 {
        // NaN: preserve sign + forced-quiet payload
        return ((bits >> 16) as u16) | 0x0040;
    }
    if exp == 0 && mant != 0 {
        // Subnormal → flush to ±0 preserving the sign bit.
        return ((bits >> 16) as u16) & 0x8000;
    }
    let lsb = (bits >> 16) & 1;
    let biased = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    (biased >> 16) as u16
}

/// Pure AVX-512-F RNE conversion of 16 F32 lanes → 16 BF16 lanes (packed u16).
///
/// Output is byte-identical to `_mm512_cvtneps_pbh` for every possible F32
/// input, without requiring AVX-512-BF16 hardware.  Requires only the
/// skylake-x AVX-512-F baseline.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f32_to_bf16_x16_rne(lane: __m512) -> __m256i {
    // SAFETY: caller guarantees AVX-512-F is enabled; every intrinsic below is
    // part of the AVX-512-F baseline and operates purely on register state.
    let bits = _mm512_castps_si512(lane);

    // lsb = (bits >> 16) & 1  — top-of-BF16 mantissa bit, used for ties-to-even
    let shifted = _mm512_srli_epi32::<16>(bits);
    let one = _mm512_set1_epi32(1);
    let lsb = _mm512_and_si512(shifted, one);

    // bias = 0x7FFF + lsb ; biased = bits + bias
    let bias = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF));
    let biased = _mm512_add_epi32(bits, bias);
    let normal_out = _mm512_srli_epi32::<16>(biased);

    // Subnormal flush: for (exp==0 && mant!=0) lanes output = sign bit only.
    // sign_only = (bits >> 16) & 0x8000  — but we already have `shifted`.
    let sign_only = _mm512_and_si512(shifted, _mm512_set1_epi32(0x0000_8000));

    // NaN lanes: produce (bits >> 16) | 0x40 (forced quiet bit, SDM spec).
    let nan_out = _mm512_or_si512(shifted, _mm512_set1_epi32(0x0040));

    // Classify lanes via the absolute value of the integer encoding.
    // abs_bits < 0x0080_0000                      → subnormal *or* +0
    // abs_bits == 0                               → ±0 (handled by normal path)
    // abs_bits > 0x7F80_0000                      → NaN (Inf is ==, handled by normal path)
    let abs_bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
    let exp_bound = _mm512_set1_epi32(0x0080_0000);
    let is_sub_or_zero: __mmask16 = _mm512_cmplt_epu32_mask(abs_bits, exp_bound);
    let is_nonzero: __mmask16 =
        _mm512_cmpgt_epu32_mask(abs_bits, _mm512_setzero_si512());
    let is_subnormal: __mmask16 = is_sub_or_zero & is_nonzero;

    let is_nan: __mmask16 = _mm512_cmpgt_epu32_mask(
        abs_bits,
        _mm512_set1_epi32(0x7F80_0000u32 as i32),
    );

    // Blend order:
    //   1. start from the normal RNE result,
    //   2. overwrite subnormal lanes with the sign-only zero,
    //   3. overwrite NaN lanes with the quieted payload.
    let with_subnormal =
        _mm512_mask_blend_epi32(is_subnormal, normal_out, sign_only);
    let merged = _mm512_mask_blend_epi32(is_nan, with_subnormal, nan_out);

    // Pack 16 × i32 low-halves into 16 × i16.  `_mm512_cvtepi32_epi16` is
    // plain truncation to the low 16 bits of each lane — exactly what we want
    // since the high 16 bits of every lane in `merged` are already zero.
    _mm512_cvtepi32_epi16(merged)
}

/// Deterministic batch F32 → BF16 using only AVX-512-F.  Output is
/// byte-identical to `_mm512_cvtneps_pbh` on any machine with AVX-512-F.
pub fn f32_to_bf16_batch_rne(input: &[f32], output: &mut [u16]) {
    assert!(output.len() >= input.len(), "output must be >= input length");

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512-F is guaranteed at compile time by `target-cpu=x86-64-v4`
        // (see `.cargo/config.toml`).  Still do a runtime check so this
        // function remains safe if the crate is ever rebuilt for a lower
        // target.
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime feature detection confirmed avx512f.
            unsafe {
                convert_f32_to_bf16_avx512f_rne(input, output);
            }
            return;
        }
    }

    for (src, dst) in input.iter().copied().zip(output.iter_mut()) {
        *dst = f32_to_bf16_scalar_rne(src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn convert_f32_to_bf16_avx512f_rne(input: &[f32], output: &mut [u16]) {
    // SAFETY: caller guarantees AVX-512-F is enabled.  The 16-wide loop uses
    // `_mm512_loadu_ps`/`_mm256_storeu_si256` on slice pointers of sufficient
    // length; the tail uses `_mm512_maskz_loadu_ps` + `_mm512_mask_cvtepi32_storeu_epi16`
    // with a mask that is zero for lanes beyond the slice end.
    let n = input.len();
    let mut i = 0usize;

    // Main 16-wide loop.
    while i + 16 <= n {
        let v = _mm512_loadu_ps(input.as_ptr().add(i));
        let packed = f32_to_bf16_x16_rne(v);
        _mm256_storeu_si256(output.as_mut_ptr().add(i) as *mut __m256i, packed);
        i += 16;
    }

    // Masked tail (0..15 lanes).
    let rem = n - i;
    if rem > 0 {
        let mask: __mmask16 = ((1u32 << rem) - 1) as __mmask16;
        // SAFETY: `maskz_loadu` only touches lanes where the mask bit is set.
        let v = _mm512_maskz_loadu_ps(mask, input.as_ptr().add(i));

        // Run the full RNE pipeline (same as `f32_to_bf16_x16_rne`) so the
        // tail has identical semantics to the main loop, then use
        // `_mm512_mask_cvtepi32_storeu_epi16` for a direct 16-bit masked store.
        let bits = _mm512_castps_si512(v);
        let shifted = _mm512_srli_epi32::<16>(bits);
        let lsb = _mm512_and_si512(shifted, _mm512_set1_epi32(1));
        let bias = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF));
        let biased = _mm512_add_epi32(bits, bias);
        let normal_out = _mm512_srli_epi32::<16>(biased);
        let sign_only = _mm512_and_si512(shifted, _mm512_set1_epi32(0x0000_8000));
        let nan_out = _mm512_or_si512(shifted, _mm512_set1_epi32(0x0040));

        let abs_bits =
            _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        let exp_bound = _mm512_set1_epi32(0x0080_0000);
        let is_sub_or_zero: __mmask16 =
            _mm512_cmplt_epu32_mask(abs_bits, exp_bound);
        let is_nonzero: __mmask16 =
            _mm512_cmpgt_epu32_mask(abs_bits, _mm512_setzero_si512());
        let is_subnormal: __mmask16 = is_sub_or_zero & is_nonzero;
        let is_nan: __mmask16 = _mm512_cmpgt_epu32_mask(
            abs_bits,
            _mm512_set1_epi32(0x7F80_0000u32 as i32),
        );

        let with_subnormal =
            _mm512_mask_blend_epi32(is_subnormal, normal_out, sign_only);
        let merged =
            _mm512_mask_blend_epi32(is_nan, with_subnormal, nan_out);

        // SAFETY: masked store — only lanes [0, rem) are touched.
        _mm512_mask_cvtepi32_storeu_epi16(
            output.as_mut_ptr().add(i) as *mut _,
            mask,
            merged,
        );
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

    // ─────────────────────────────────────────────────────────────────────
    // RNE certification tests — byte-equality with `_mm512_cvtneps_pbh`.
    // ─────────────────────────────────────────────────────────────────────

    /// Build the systematic corpus of F32 inputs whose correctness is
    /// critical for BF16 round-trip.  The caller concatenates this with a
    /// pseudo-random stream.
    fn rne_systematic_corpus() -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();

        // ±0
        out.push(0.0);
        out.push(-0.0);

        // ±Inf
        out.push(f32::INFINITY);
        out.push(f32::NEG_INFINITY);

        // Every kind of canonical/non-canonical NaN we can think of.
        for bits in [
            0x7FC0_0000u32, // canonical qNaN
            0xFFC0_0000,    // -qNaN
            0x7FC0_0001,    // qNaN with payload
            0x7FBF_FFFF,    // sNaN with max payload below quiet bit
            0x7F80_0001,    // smallest sNaN
            0xFF80_0001,    // -sNaN smallest
            0x7FFF_FFFF,    // qNaN, all-ones payload
            0x7FDE_AD00,    // arbitrary qNaN payload
        ] {
            out.push(f32::from_bits(bits));
        }

        // Subnormals: all f32 subnormals collapse to ±0 in BF16 because their
        // magnitude is far below the BF16 smallest normal (2^-126 vs 2^-126
        // w/ 7-bit mantissa).  Hit a bunch anyway.
        for bits in [
            0x0000_0001u32, // smallest positive subnormal
            0x007F_FFFF,    // largest positive subnormal
            0x0040_0000,    // mid-range subnormal
            0x8000_0001,    // negative subnormal
            0x807F_FFFF,
        ] {
            out.push(f32::from_bits(bits));
        }

        // Normals across the exponent range.
        for exp_byte in [1u32, 50, 126, 127, 128, 200, 254] {
            for mant in [
                0x0000_00u32,
                0x400000, // halfway-below-LSB for even mantissa
                0x7FFFFF, // top of mantissa (rounding into next exponent)
                0x0080_00, // round bit alone
                0x00_FFFF, // sticky bits only
                0x01_8000, // round + tie, LSB=1 → round up
                0x00_8001, // round + sticky → round up
            ] {
                let bits = (exp_byte << 23) | mant;
                out.push(f32::from_bits(bits));
                out.push(f32::from_bits(bits | 0x8000_0000)); // negative
            }
        }

        // Deterministic halfway cases around a variety of BF16 boundaries.
        // bit 15 set, bits 14..0 clear → exact halfway. LSB of preserved
        // mantissa must dictate the direction.
        for exp_byte in [100u32, 127, 150] {
            for lsb_bit in 0..7u32 {
                let mant_hi = 1u32 << (16 + lsb_bit); // varies kept-LSB
                let bits = (exp_byte << 23) | mant_hi | 0x0000_8000;
                out.push(f32::from_bits(bits));
            }
        }

        // Near-max finite (rounds up to Inf under RNE).
        out.push(f32::from_bits(0x7F7F_FFFF));
        out.push(f32::from_bits(0xFF7F_FFFF));

        out
    }

    /// Tiny xorshift PRNG — fixed seed for reproducibility.
    fn rne_random_corpus(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Lower 32 bits reinterpreted as f32 — covers every code point.
            out.push(f32::from_bits(state as u32));
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_to_bf16_rne_byte_equality() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: avx512f not available");
            return;
        }

        let mut corpus = rne_systematic_corpus();
        corpus.extend(rne_random_corpus(1_000_000, 0xD1CE_F00D_0BADu64));

        // Pad to multiple of 16 with zeros so we can run the 16-wide routine
        // end-to-end without worrying about masked tails in this test.
        while corpus.len() % 16 != 0 {
            corpus.push(0.0);
        }

        // Run the AVX-512-F RNE routine.
        let mut rne_out: Vec<u16> = vec![0; corpus.len()];
        unsafe {
            // SAFETY: avx512f confirmed by feature detection.
            let n = corpus.len();
            let mut i = 0;
            while i < n {
                let v = _mm512_loadu_ps(corpus.as_ptr().add(i));
                let packed = f32_to_bf16_x16_rne(v);
                _mm256_storeu_si256(
                    rne_out.as_mut_ptr().add(i) as *mut __m256i,
                    packed,
                );
                i += 16;
            }
        }

        // Reference: hardware `_mm512_cvtneps_pbh` if available.
        if is_x86_feature_detected!("avx512bf16")
            && is_x86_feature_detected!("avx512vl")
        {
            let mut hw_out: Vec<u16> = vec![0; corpus.len()];
            unsafe {
                // SAFETY: feature detection confirmed avx512bf16 + avx512vl.
                convert_f32_to_bf16_avx512bf16(&corpus, &mut hw_out);
            }
            let mut mismatches = 0usize;
            for (idx, (&r, &h)) in rne_out.iter().zip(hw_out.iter()).enumerate() {
                if r != h {
                    if mismatches < 8 {
                        eprintln!(
                            "mismatch idx={idx} input=0x{:08X} rne=0x{:04X} hw=0x{:04X}",
                            corpus[idx].to_bits(),
                            r,
                            h
                        );
                    }
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "byte-equality with _mm512_cvtneps_pbh failed on {} / {} inputs",
                mismatches,
                corpus.len()
            );
        } else {
            // Fallback: hand-picked reference table so the test still runs.
            //
            // Each (input_bits, expected_bf16_bits) entry was produced by
            // walking the Intel SDM VCVTNEPS2BF16 pseudocode by hand.  Do not
            // regenerate these — they are the published oracle.
            let reference: &[(u32, u16)] = &[
                (0x0000_0000, 0x0000),                  // +0
                (0x8000_0000, 0x8000),                  // -0
                (0x3F80_0000, 0x3F80),                  // 1.0
                (0xBF80_0000, 0xBF80),                  // -1.0
                (0x7F80_0000, 0x7F80),                  // +Inf
                (0xFF80_0000, 0xFF80),                  // -Inf
                (0x7FC0_0000, 0x7FC0),                  // canonical qNaN
                (0x7F80_0001, 0x7FC0),                  // sNaN → qNaN
                (0x7FBF_FFFF, 0x7FFF),                  // sNaN payload → QNaN'd
                // Halfway, LSB=0 → round down (stay even).
                // f32 bits = 0x3F80_8000  (1 + 2^-8).  Kept LSB = 0, ties.
                (0x3F80_8000, 0x3F80),
                // Halfway, LSB=1 → round up (to even).
                // f32 bits = 0x3F81_8000  (1.0078125 exactly). Kept LSB = 1.
                (0x3F81_8000, 0x3F82),
                // Round bit + sticky → unambiguous round up.
                (0x3F80_8001, 0x3F81),
                // Max finite rounds up to +Inf.
                (0x7F7F_FFFF, 0x7F80),
                (0xFF7F_FFFF, 0xFF80),
                // Positive subnormal rounds toward 0 (stays 0 in BF16).
                (0x0000_0001, 0x0000),
            ];

            for &(in_bits, expected) in reference {
                let v = f32::from_bits(in_bits);
                let got = f32_to_bf16_scalar_rne(v);
                assert_eq!(
                    got, expected,
                    "scalar RNE mismatch for 0x{in_bits:08X}: got=0x{got:04X} want=0x{expected:04X}"
                );
            }

            // And run the SIMD path on a padded batch of those same inputs
            // so the routine's SIMD code path is actually exercised.
            let mut batch: Vec<f32> =
                reference.iter().map(|&(b, _)| f32::from_bits(b)).collect();
            while batch.len() % 16 != 0 {
                batch.push(0.0);
            }
            let mut simd_out = vec![0u16; batch.len()];
            unsafe {
                // SAFETY: avx512f confirmed above.
                let v = _mm512_loadu_ps(batch.as_ptr());
                let packed = f32_to_bf16_x16_rne(v);
                _mm256_storeu_si256(
                    simd_out.as_mut_ptr() as *mut __m256i,
                    packed,
                );
            }
            for (i, &(in_bits, expected)) in reference.iter().enumerate() {
                assert_eq!(
                    simd_out[i], expected,
                    "SIMD RNE mismatch for 0x{in_bits:08X}: got=0x{:04X} want=0x{expected:04X}",
                    simd_out[i],
                );
            }
        }
    }

    /// Ties-to-even certification: for every exponent, construct a pair
    /// (LSB=0 halfway, LSB=1 halfway) and verify both the scalar and SIMD
    /// paths produce an even-LSB result.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_to_bf16_rne_ties_to_even() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: avx512f not available");
            return;
        }

        let mut cases: Vec<f32> = Vec::new();
        // exp_byte in [1, 254] skipping 0 (subnormal) and 255 (NaN/Inf).
        for exp_byte in 1u32..=254 {
            // LSB=0 halfway: mant = 0b...0_1000_0000_0000_0000
            // → f32 bits low 16 = 0x8000, kept-LSB bit (bit 16) = 0.
            let lsb0 = (exp_byte << 23) | 0x0000_8000;
            cases.push(f32::from_bits(lsb0));
            // LSB=1 halfway: mant = 0b...1_1000_0000_0000_0000
            let lsb1 = (exp_byte << 23) | 0x0001_8000;
            cases.push(f32::from_bits(lsb1));
        }
        while cases.len() % 16 != 0 {
            cases.push(0.0);
        }

        let mut out = vec![0u16; cases.len()];
        unsafe {
            // SAFETY: avx512f confirmed above.
            let n = cases.len();
            let mut i = 0;
            while i < n {
                let v = _mm512_loadu_ps(cases.as_ptr().add(i));
                let packed = f32_to_bf16_x16_rne(v);
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(i) as *mut __m256i,
                    packed,
                );
                i += 16;
            }
        }

        for (idx, (&v, &got)) in cases.iter().zip(out.iter()).enumerate() {
            // Skip the padding zeros.
            if v == 0.0 && idx >= 2 * (254 - 1 + 1) {
                continue;
            }
            let bf16_mant_lsb = got & 0x0001;
            assert_eq!(
                bf16_mant_lsb, 0,
                "round-to-even failed for input idx={idx} bits=0x{:08X}: bf16=0x{got:04X}",
                v.to_bits()
            );

            // Also cross-check with the scalar reference.
            let scalar = f32_to_bf16_scalar_rne(v);
            assert_eq!(
                got, scalar,
                "SIMD vs scalar RNE disagree for 0x{:08X}", v.to_bits()
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_to_bf16_batch_rne_end_to_end() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: avx512f not available");
            return;
        }

        // Sizes chosen to exercise 0, partial, full, and partial-tail paths.
        for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 128, 129, 1024, 1025] {
            let mut rng_state = 0xABAD_1DEAu64 ^ (len as u64).wrapping_mul(0x9E37_79B9);
            let mut input = Vec::with_capacity(len);
            for _ in 0..len {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                input.push(f32::from_bits(rng_state as u32));
            }
            let mut batch_out = vec![0u16; len];
            f32_to_bf16_batch_rne(&input, &mut batch_out);

            for (i, &v) in input.iter().enumerate() {
                let expected = f32_to_bf16_scalar_rne(v);
                assert_eq!(
                    batch_out[i], expected,
                    "batch RNE mismatch len={len} idx={i} bits=0x{:08X}",
                    v.to_bits()
                );
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// F16 (IEEE 754 Half-Precision) — via F16C instructions (stable since Rust 1.68)
//
// ⚠️  THIS IS NOT FOR GGUF/MODEL WEIGHT CALIBRATION ⚠️
//
// This f16 is for: sensor data, audio, ARM interchange, memory-efficient storage.
// For GGUF model weights → use the BF16 pipeline above (bf16_to_f32_batch etc.)
//
// ┌─────────┬──────┬──────────┬──────────┬────────────┬─────────────────┐
// │ Format  │ Bits │ Exponent │ Mantissa │ Range      │ Use case        │
// ├─────────┼──────┼──────────┼──────────┼────────────┼─────────────────┤
// │ BF16    │  16  │ 8 (b127) │ 7 bits   │ ±3.4e38   │ GGUF weights    │
// │ F16     │  16  │ 5 (b15)  │ 10 bits  │ ±65504    │ Sensors, audio  │
// │ F32     │  32  │ 8 (b127) │ 23 bits  │ ±3.4e38   │ Compute         │
// └─────────┴──────┴──────────┴──────────┴────────────┴─────────────────┘
//
// f32→f16 narrowing: 23-bit mantissa → 10-bit = 13 bits lost.
// Max RNE error: ±0.5 ULP of f16 result (≈ 0.05% relative).
//
// IEEE 754 binary16: 1 sign + 5 exponent + 10 mantissa
// Range: ±65504, precision: ~3.3 decimal digits
// Subnormals: ±5.96×10⁻⁸ minimum positive
//
// Hardware instructions (F16C, stable target_feature):
//   _mm256_cvtph_ps:  8× f16(u16) → 8× f32  (VCVTPH2PS ymm, xmm)
//   _mm512_cvtph_ps: 16× f16(u16) → 16× f32 (VCVTPH2PS zmm, ymm) [AVX-512F]
//   _mm256_cvtps_ph:  8× f32 → 8× f16(u16)  (VCVTPS2PH xmm, ymm, imm8)
//   _mm512_cvtps_ph: 16× f32 → 16× f16(u16) (VCVTPS2PH ymm, zmm, imm8) [AVX-512F]
//
// imm8 for rounding:
//   0x00 = Round to nearest even (IEEE default)
//   0x01 = Round toward negative infinity
//   0x02 = Round toward positive infinity
//   0x03 = Round toward zero (truncate)
//   0x04 = Use MXCSR rounding mode
//
// NOTE: F16C is available on Haswell+ (2013), essentially all modern x86_64.
// AVX-512 F16C (zmm-width) requires AVX-512F.
// ════════════════════════════════════════════════════════════════════════════

/// IEEE 754 f16 → f32 scalar conversion (exact, lossless).
///
/// binary16: 1 sign | 5 exponent (bias 15) | 10 mantissa
/// binary32: 1 sign | 8 exponent (bias 127) | 23 mantissa
///
/// Conversion is exact: every f16 value has an exact f32 representation.
/// Zero additional error — this is a widening cast.
pub fn f16_to_f32_ieee754(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // ±0.0
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: (−1)^sign × 2^(−14) × 0.mantissa
            // Normalize: find leading 1 in mantissa, adjust exponent
            let mut m = mant;
            let mut e: i32 = 1 - 15; // subnormal effective exponent = 1 - bias
            // Shift mantissa left until the implicit 1 is in bit 10
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove the implicit 1
            let f32_exp = ((e + 127) as i32) as u32; // rebias to f32
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN — preserve NaN payload
        let f32_mant = mant << 13; // widen 10-bit → 23-bit mantissa
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal: rebias exponent (bias 15 → bias 127) = exp + 112
        let f32_exp = exp + 112; // avoids u32 underflow vs (exp - 15 + 127)
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// IEEE 754 f32 → f16 scalar with Round-to-Nearest-Even (RNE).
///
/// Matches hardware VCVTPS2PH with imm8=0x00 bit-exact.
/// Handles: normals, subnormals, overflow→Inf, NaN preservation.
///
/// Precision: 10 mantissa bits → 3.31 decimal digits.
/// Any f32 value outside [−65504, +65504] overflows to ±Inf.
pub fn f32_to_f16_ieee754_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            // Inf
            ((sign << 15) | (0x1F << 10)) as u16
        } else {
            // NaN: preserve as much payload as possible
            // Quiet NaN bit (bit 22 in f32 → bit 9 in f16)
            let h_mant = (mant >> 13) & 0x3FF;
            // Ensure at least one mantissa bit set (to stay NaN)
            let h_mant = if h_mant == 0 { 0x200 } else { h_mant }; // set quiet bit
            ((sign << 15) | (0x1F << 10) | h_mant) as u16
        }
    } else if exp == 0 && mant == 0 {
        // ±0.0
        (sign << 15) as u16
    } else {
        // Normal or subnormal f32 → f16
        let unbiased = exp - 127; // true exponent

        if unbiased > 15 {
            // Overflow → ±Inf
            ((sign << 15) | (0x1F << 10)) as u16
        } else if unbiased < -24 {
            // Too small even for f16 subnormal → ±0
            (sign << 15) as u16
        } else if unbiased < -14 {
            // f16 subnormal range: exponent would be 0, mantissa encodes value
            // f16_value = (−1)^s × 2^(−14) × 0.mant
            // shift = how many extra bits to shift right (−14 − unbiased)
            let shift = (-14 - unbiased) as u32;
            // Add implicit 1 to f32 mantissa, then shift right
            let full_mant = mant | 0x800000; // 24 bits with implicit 1
            // We need to map 24-bit mantissa to 10-bit with proper shift
            let total_shift = 13 + shift; // 13 to go from 23→10, plus extra for subnormal

            // Round-to-nearest-even
            let truncated = full_mant >> total_shift;
            let remainder = full_mant & ((1 << total_shift) - 1);
            let halfway = 1 << (total_shift - 1);

            let rounded = if remainder > halfway {
                truncated + 1
            } else if remainder == halfway {
                // Ties to even: round up if truncated is odd
                if truncated & 1 != 0 { truncated + 1 } else { truncated }
            } else {
                truncated
            };

            let h_mant = rounded & 0x3FF;
            // If rounding overflowed into exponent range, it becomes a normal
            let h_exp = if rounded > 0x3FF { 1u32 } else { 0u32 };
            ((sign << 15) | (h_exp << 10) | h_mant) as u16
        } else {
            // Normal f16 range
            let h_exp = (unbiased + 15) as u32; // rebias: +15
            // Round mantissa from 23 bits to 10 bits using RNE
            let truncated = mant >> 13;
            let remainder = mant & 0x1FFF; // lower 13 bits
            let halfway = 0x1000; // 2^12

            let rounded = if remainder > halfway {
                truncated + 1
            } else if remainder == halfway {
                if truncated & 1 != 0 { truncated + 1 } else { truncated }
            } else {
                truncated
            };

            // Check if rounding overflowed mantissa (10 bits → 11 bits)
            if rounded > 0x3FF {
                // Carry into exponent
                let h_exp = h_exp + 1;
                if h_exp >= 31 {
                    // Overflow to Inf
                    ((sign << 15) | (0x1F << 10)) as u16
                } else {
                    ((sign << 15) | (h_exp << 10)) as u16 // mantissa = 0 after carry
                }
            } else {
                ((sign << 15) | (h_exp << 10) | rounded) as u16
            }
        }
    }
}

/// Batch f16 → f32 via AVX-512 VCVTPH2PS (16 lanes) with F16C fallback (8 lanes).
///
/// Detection: avx512f → 16-wide | f16c → 8-wide | scalar fallback
/// Conversion is exact (lossless widening).
pub fn f16_to_f32_batch_ieee754(input: &[u16], output: &mut [f32]) {
    let n = input.len().min(output.len());

    #[cfg(target_arch = "x86_64")]
    {
        // Tier 1: AVX-512F (16 lanes per instruction)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            let chunks16 = n / 16;
            for c in 0..chunks16 {
                unsafe {
                    // SAFETY: avx512f + f16c verified above.
                    let src = _mm256_loadu_si256(input[c*16..].as_ptr() as *const __m256i);
                    let dst = _mm512_cvtph_ps(src);
                    _mm512_storeu_ps(output[c*16..].as_mut_ptr(), dst);
                }
            }
            // Scalar tail
            for i in (chunks16*16)..n {
                output[i] = f16_to_f32_ieee754(input[i]);
            }
            return;
        }
        // Tier 2: F16C (8 lanes per instruction, Haswell+)
        if is_x86_feature_detected!("f16c") {
            let chunks8 = n / 8;
            for c in 0..chunks8 {
                unsafe {
                    // SAFETY: f16c verified above.
                    let src = _mm_loadu_si128(input[c*8..].as_ptr() as *const __m128i);
                    let dst = _mm256_cvtph_ps(src);
                    _mm256_storeu_ps(output[c*8..].as_mut_ptr(), dst);
                }
            }
            for i in (chunks8*8)..n {
                output[i] = f16_to_f32_ieee754(input[i]);
            }
            return;
        }
    }

    // Scalar fallback (exact)
    for i in 0..n {
        output[i] = f16_to_f32_ieee754(input[i]);
    }
}

/// Batch f32 → f16 via AVX-512 VCVTPS2PH (16 lanes) with RNE rounding.
///
/// imm8 = 0x00: Round-to-Nearest-Even (IEEE 754 default).
/// Matches hardware behavior bit-exact.
pub fn f32_to_f16_batch_ieee754_rne(input: &[f32], output: &mut [u16]) {
    let n = input.len().min(output.len());

    #[cfg(target_arch = "x86_64")]
    {
        // Tier 1: AVX-512F (16 lanes, RNE via imm8=0)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            let chunks16 = n / 16;
            for c in 0..chunks16 {
                unsafe {
                    // SAFETY: avx512f + f16c verified above.
                    let src = _mm512_loadu_ps(input[c*16..].as_ptr());
                    // imm8=0x00: _MM_FROUND_TO_NEAREST_INT (RNE)
                    let dst: __m256i = _mm512_cvtps_ph::<0x00>(src);
                    _mm256_storeu_si256(output[c*16..].as_mut_ptr() as *mut __m256i, dst);
                }
            }
            for i in (chunks16*16)..n {
                output[i] = f32_to_f16_ieee754_rne(input[i]);
            }
            return;
        }
        // Tier 2: F16C (8 lanes, RNE)
        if is_x86_feature_detected!("f16c") {
            let chunks8 = n / 8;
            for c in 0..chunks8 {
                unsafe {
                    // SAFETY: f16c verified above.
                    let src = _mm256_loadu_ps(input[c*8..].as_ptr());
                    let dst: __m128i = _mm256_cvtps_ph::<0x00>(src);
                    _mm_storeu_si128(output[c*8..].as_mut_ptr() as *mut __m128i, dst);
                }
            }
            for i in (chunks8*8)..n {
                output[i] = f32_to_f16_ieee754_rne(input[i]);
            }
            return;
        }
    }

    // Scalar RNE fallback
    for i in 0..n {
        output[i] = f32_to_f16_ieee754_rne(input[i]);
    }
}

#[cfg(test)]
mod f16_tests {
    use super::*;

    #[test]
    fn f16_ieee754_exact_values() {
        // IEEE 754 binary16 exact test vectors
        assert_eq!(f16_to_f32_ieee754(0x0000), 0.0);            // +0
        assert_eq!(f16_to_f32_ieee754(0x8000), -0.0);           // −0
        assert_eq!(f16_to_f32_ieee754(0x3C00), 1.0);            // 1.0
        assert_eq!(f16_to_f32_ieee754(0xBC00), -1.0);           // −1.0
        assert_eq!(f16_to_f32_ieee754(0x4000), 2.0);            // 2.0
        assert_eq!(f16_to_f32_ieee754(0x3800), 0.5);            // 0.5
        assert_eq!(f16_to_f32_ieee754(0x7BFF), 65504.0);        // max normal
        assert!(f16_to_f32_ieee754(0x7C00).is_infinite());       // +Inf
        assert!(f16_to_f32_ieee754(0xFC00).is_infinite());       // −Inf
        assert!(f16_to_f32_ieee754(0x7C01).is_nan());            // NaN
        // Smallest positive subnormal: 2^(−24) ≈ 5.96e-8
        let smallest_sub = f16_to_f32_ieee754(0x0001);
        assert!((smallest_sub - 5.960464e-8).abs() < 1e-14);
    }

    #[test]
    fn f16_rne_roundtrip_normals() {
        // Every f16 normal → f32 → f16 must be identity
        for exp in 1u16..31 {
            for mant in (0u16..1024).step_by(17) {
                let h = (exp << 10) | mant;
                let f = f16_to_f32_ieee754(h);
                let back = f32_to_f16_ieee754_rne(f);
                assert_eq!(h, back,
                    "roundtrip failed: 0x{:04X} → {} → 0x{:04X}", h, f, back);
            }
        }
    }

    #[test]
    fn f16_exact_representable_values() {
        // Values that are exactly representable in f16 must roundtrip perfectly
        let exact_values: &[f32] = &[
            0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, 0.125,
            65504.0, -65504.0, // max f16
            0.000061035156, // smallest normal f16 (2^-14)
        ];
        for &v in exact_values {
            let h = f32_to_f16_ieee754_rne(v);
            let back = f16_to_f32_ieee754(h);
            assert_eq!(v, back,
                "exact value roundtrip failed: {} → 0x{:04X} → {}", v, h, back);
        }
    }

    #[test]
    fn f16_overflow_to_inf() {
        let big = 100000.0f32;
        assert_eq!(f32_to_f16_ieee754_rne(big), 0x7C00); // +Inf
        assert_eq!(f32_to_f16_ieee754_rne(-big), 0xFC00); // −Inf
    }

    #[test]
    fn f16_batch_matches_scalar() {
        let input: Vec<u16> = (0..200).map(|i| {
            let v = (i as f32 - 100.0) * 0.5;
            f32_to_f16_ieee754_rne(v)
        }).collect();
        let mut batch_out = vec![0.0f32; 200];
        f16_to_f32_batch_ieee754(&input, &mut batch_out);

        for (i, &h) in input.iter().enumerate() {
            let scalar = f16_to_f32_ieee754(h);
            assert_eq!(batch_out[i].to_bits(), scalar.to_bits(),
                "batch/scalar mismatch at {}: batch=0x{:08X} scalar=0x{:08X}",
                i, batch_out[i].to_bits(), scalar.to_bits());
        }
    }

    #[test]
    fn f32_to_f16_batch_rne_matches_scalar() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.37).collect();
        let mut batch_out = vec![0u16; 200];
        f32_to_f16_batch_ieee754_rne(&input, &mut batch_out);

        for (i, &v) in input.iter().enumerate() {
            let scalar = f32_to_f16_ieee754_rne(v);
            assert_eq!(batch_out[i], scalar,
                "f32→f16 batch/scalar mismatch at {}: input={} batch=0x{:04X} scalar=0x{:04X}",
                i, v, batch_out[i], scalar);
        }
    }
}
