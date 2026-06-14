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
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign,
    Neg, Not, Shl, Shr, Sub, SubAssign,
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
            Self(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(self.0), mask)))
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
            Self(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(self.0), sign)))
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
            Self(_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(self.0), mask)))
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
            Self(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(self.0), sign)))
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
        Self(unsafe {
            match imm {
                1 => _mm512_srli_epi16(self.0, 1),
                2 => _mm512_srli_epi16(self.0, 2),
                3 => _mm512_srli_epi16(self.0, 3),
                4 => _mm512_srli_epi16(self.0, 4),
                5 => _mm512_srli_epi16(self.0, 5),
                6 => _mm512_srli_epi16(self.0, 6),
                7 => _mm512_srli_epi16(self.0, 7),
                8 => _mm512_srli_epi16(self.0, 8),
                _ => _mm512_setzero_si512(),
            }
        })
    }

    /// Saturating unsigned subtraction: max(a - b, 0) per byte.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(unsafe { _mm512_subs_epu8(self.0, other.0) })
    }

    // ── Tier 1: seismon rasterizer primitives ─────────────────────────

    /// Pairwise unsigned byte average: (a[i] + b[i] + 1) >> 1 per byte.
    /// Core op for 4×4 mipmap downsample (vpavgb + horizontal pair = 2 ops).
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        // SAFETY: AVX-512BW instruction, operates on all 64 bytes.
        Self(unsafe { _mm512_avg_epu8(self.0, other.0) })
    }

    /// Byte-wise unsigned greater-than comparison. Returns 64-bit mask:
    /// bit i set if self[i] > other[i]. Symmetric to `cmpeq_mask`.
    /// Used for threshold density fields, depth/Z-test, hit-tests.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        // SAFETY: AVX-512BW instruction. Unsigned compare via _epu8.
        unsafe { _mm512_cmpgt_epu8_mask(self.0, other.0) }
    }

    /// Masked blend: for each bit in `mask`, select from `b` if set, else `a`.
    /// Sprite alpha blit: write atlas pixel where mask bit set, keep framebuffer otherwise.
    #[inline(always)]
    pub fn mask_blend(mask: u64, a: Self, b: Self) -> Self {
        // SAFETY: AVX-512BW instruction. mask selects between a and b per byte.
        Self(unsafe { _mm512_mask_blend_epi8(mask, a.0, b.0) })
    }

    /// Shift left each 16-bit lane by immediate bits (nibble write: place high nibble).
    /// Completes the nibble shift pair with `shr_epi16`.
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        Self(unsafe {
            match imm {
                1 => _mm512_slli_epi16(self.0, 1),
                2 => _mm512_slli_epi16(self.0, 2),
                3 => _mm512_slli_epi16(self.0, 3),
                4 => _mm512_slli_epi16(self.0, 4),
                5 => _mm512_slli_epi16(self.0, 5),
                6 => _mm512_slli_epi16(self.0, 6),
                7 => _mm512_slli_epi16(self.0, 7),
                8 => _mm512_slli_epi16(self.0, 8),
                _ => _mm512_setzero_si512(),
            }
        })
    }

    // ── Tier 2: sprite blit + palette LUT + cross-lane shuffle ────────

    /// Masked store: write only bytes where mask bit is set.
    /// Partial-tile writes at framebuffer edges without scalar fallback.
    ///
    /// # Safety
    /// `ptr` must point to at least 64 writable bytes (may be unaligned).
    #[inline(always)]
    pub unsafe fn mask_store(self, ptr: *mut u8, mask: u64) {
        // SAFETY: AVX-512BW masked store. Caller guarantees ptr validity.
        _mm512_mask_storeu_epi8(ptr as *mut i8, mask, self.0);
    }

    /// Saturating unsigned addition: min(a + b, 255) per byte.
    /// Additive blend without overflow wrap. Symmetric to `saturating_sub`.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        // SAFETY: AVX-512BW instruction.
        Self(unsafe { _mm512_adds_epu8(self.0, other.0) })
    }

    /// Cross-lane byte permute: rearrange all 64 bytes by index vector.
    /// `idx[i]` selects which byte of `self` appears at position `i & 63`.
    /// Unlike `shuffle_bytes` (within-lane), this crosses 128-bit lane boundaries.
    /// Needed for sprite atlas reorder and palette remap > 16 entries.
    ///
    /// Dispatch (one LazyLock check via `simd_caps()`):
    /// - VBMI present (Ice Lake+, Tiger Lake+, Sapphire Rapids+, Zen 4): hardware
    ///   `_mm512_permutexvar_epi8` — one instruction.
    /// - AVX-512F without VBMI (Skylake-X, Cascade Lake, Ice Lake-SP): scalar
    ///   permute via stack. Slower but does not SIGILL.
    #[inline]
    pub fn permute_bytes(self, idx: Self) -> Self {
        if crate::hpc::simd_caps::simd_caps().avx512vbmi {
            // SAFETY: avx512vbmi was verified by simd_caps() at startup
            // (one LazyLock detect for the whole process).
            unsafe { Self(permute_bytes_vbmi(self.0, idx.0)) }
        } else {
            // AVX-512F-only fallback: scalar permute via stack arrays.
            // Same shape as the AVX2-tier fallback in simd_avx2.rs:1435.
            let src = self.to_array();
            let idx_arr = idx.to_array();
            let mut out = [0u8; 64];
            for i in 0..64 {
                out[i] = src[(idx_arr[i] & 63) as usize];
            }
            Self::from_array(out)
        }
    }

    /// Extract sign bits of all 64 bytes as a 64-bit mask.
    /// Bit i is set if byte i has its MSB (bit 7) set.
    /// Useful for empty-tile skip ("any pixel non-zero in this 64-pixel row").
    #[inline(always)]
    pub fn movemask(self) -> u64 {
        // SAFETY: AVX-512BW. Compare each byte > 0x7F is equivalent to MSB set.
        // Using cmpgt with 0x7F splat: set bit if byte > 127 (i.e. MSB = 1).
        unsafe { _mm512_movepi8_mask(self.0) }
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
        Self(unsafe {
            _mm512_set4_epi32(
                0x04030302_u32 as i32, 0x03020201_u32 as i32, 0x03020201_u32 as i32, 0x02010100_u32 as i32,
            )
        })
    }
}

/// AVX-512VBMI cross-lane byte permute. Inner unsafe leaf — `#[target_feature]`
/// is required by Rust to call the VBMI intrinsic from a function not compiled
/// with VBMI globally. Caller (`U8x64::permute_bytes`) gates this behind
/// `simd_caps().avx512vbmi` so the SIGILL on Skylake-X / Cascade Lake / Ice
/// Lake-SP is impossible by construction.
///
/// SAFETY: caller must verify `simd_caps().avx512vbmi == true` before calling.
#[inline]
#[target_feature(enable = "avx512vbmi")]
unsafe fn permute_bytes_vbmi(v: __m512i, idx: __m512i) -> __m512i {
    _mm512_permutexvar_epi8(idx, v)
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

            Self(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(packed_lo), packed_hi))
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
// U16x32 — 32 × u16 in one AVX-512 register (__m512i)
// Weighted blends, 16-bit accumulation, palette LUT with wider indices.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U16x32(pub __m512i);

impl U16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        // SAFETY: AVX-512 set1 for 16-bit.
        Self(unsafe { _mm512_set1_epi16(v as i16) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 32);
        // SAFETY: 32 × u16 = 64 bytes = one __m512i. Unaligned load.
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u16; 32]) -> Self {
        // SAFETY: same layout guarantee.
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 32] {
        let mut arr = [0u16; 32];
        // SAFETY: store 64 bytes into 32 × u16.
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 32);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    /// Widen lower 32 bytes of a U8x64 to 32 × u16 (zero-extend).
    #[inline(always)]
    pub fn from_u8x64_lo(v: U8x64) -> Self {
        // SAFETY: _mm512_cvtepu8_epi16 takes __m256i (lower half of __m512i).
        Self(unsafe {
            let lo = _mm512_castsi512_si256(v.0);
            _mm512_cvtepu8_epi16(lo)
        })
    }

    /// Widen upper 32 bytes of a U8x64 to 32 × u16 (zero-extend).
    #[inline(always)]
    pub fn from_u8x64_hi(v: U8x64) -> Self {
        // SAFETY: extract high 256 bits, then widen.
        Self(unsafe {
            let hi = _mm512_extracti64x4_epi64(v.0, 1);
            _mm512_cvtepu8_epi16(hi)
        })
    }

    /// Narrow back to u8 with unsigned saturation (32 × u16 → lower 32 bytes of U8x64).
    #[inline(always)]
    pub fn pack_saturate_u8(self, other: Self) -> U8x64 {
        // SAFETY: _mm512_packus_epi16 packs two __m512i of 16-bit into one __m512i of 8-bit.
        U8x64(unsafe { _mm512_packus_epi16(self.0, other.0) })
    }

    /// Shift right each 16-bit lane by immediate.
    #[inline(always)]
    pub fn shr(self, imm: u32) -> Self {
        Self(unsafe {
            match imm {
                1 => _mm512_srli_epi16(self.0, 1),
                2 => _mm512_srli_epi16(self.0, 2),
                4 => _mm512_srli_epi16(self.0, 4),
                8 => _mm512_srli_epi16(self.0, 8),
                _ => _mm512_setzero_si512(),
            }
        })
    }

    /// Shift left each 16-bit lane by immediate.
    #[inline(always)]
    pub fn shl(self, imm: u32) -> Self {
        Self(unsafe {
            match imm {
                1 => _mm512_slli_epi16(self.0, 1),
                2 => _mm512_slli_epi16(self.0, 2),
                4 => _mm512_slli_epi16(self.0, 4),
                8 => _mm512_slli_epi16(self.0, 8),
                _ => _mm512_setzero_si512(),
            }
        })
    }

    /// Multiply and keep low 16 bits (wrapping).
    #[inline(always)]
    pub fn mullo(self, other: Self) -> Self {
        // SAFETY: AVX-512BW multiply low 16.
        Self(unsafe { _mm512_mullo_epi16(self.0, other.0) })
    }

    /// Horizontal sum of all 32 lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        let arr = self.to_array();
        arr.iter().map(|&v| v as u32).sum()
    }
}

impl Add for U16x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_add_epi16(self.0, rhs.0) })
    }
}
impl Sub for U16x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sub_epi16(self.0, rhs.0) })
    }
}
impl AddAssign for U16x32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm512_add_epi16(self.0, rhs.0) };
    }
}

impl fmt::Debug for U16x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U16x32({:?})", self.to_array())
    }
}

impl PartialEq for U16x32 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// F32x8 fused multiply-add (256-bit __m256). `self.mul_add(a, b) = self*a + b`
// in a single rounding step via `_mm256_fmadd_ps` (FMA3). The 8-wide companion
// to the existing `F32x16::mul_add`; consumed by the PQ4-ADC FastScan flush
// (turbovec's AVX2 kernel) where the per-query `fa = v_scale*partial + fa`
// reduction needs an 8-wide FMA.
impl F32x8 {
    /// Fused multiply-add: `self * a + b`, single rounding (`_mm256_fmadd_ps`).
    ///
    /// # Examples
    /// ```ignore
    /// let a = F32x8::splat(0.5);
    /// let b = F32x8::splat(2.0);
    /// let c = F32x8::splat(1.0);
    /// assert_eq!(a.mul_add(b, c).to_array(), [2.0; 8]); // 0.5*2.0 + 1.0
    /// ```
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // SAFETY: FMA3 intrinsic; reached only on FMA-capable targets via the
        // consumer's runtime dispatch / `#[target_feature(enable = "fma")]`.
        Self(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }

    /// Lane-wise `self > other` as an 8-bit mask: bit `i` set iff
    /// `self[i] > other[i]` (ordered, non-signaling). `_mm256_cmp_ps::<_CMP_GT_OQ>`
    /// + `_mm256_movemask_ps`. The FastScan heap threshold-prune uses it to skip
    /// an 8-lane score chunk that holds no candidate above the current heap-min
    /// in a single instruction — the SIMD early-out the scalar `>hmin` scan loses.
    ///
    /// # Examples
    /// ```ignore
    /// let a = F32x8::from_array([3.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    /// let b = F32x8::splat(1.0);
    /// // lanes 0,2,4,6 are > 1.0 ⇒ bits 0,2,4,6 set = 0b0101_0101 = 0x55.
    /// assert_eq!(a.cmp_gt_mask(b), 0x55);
    /// ```
    #[inline(always)]
    pub fn cmp_gt_mask(self, other: Self) -> u32 {
        // SAFETY: AVX `vcmpps` + `vmovmskps`; available wherever this 256-bit
        // float type is (x86-64-v2+).
        unsafe { _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0)) as u32 }
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
// I8x64 — 64 × i8 in one AVX-512 register (__m512i)
// AVX-512BW: byte-level add/sub/min/max, 64-bit cmpgt mask.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I8x64(pub __m512i);

impl I8x64 {
    pub const LANES: usize = 64;

    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 64);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 64] {
        let mut arr = [0i8; 64];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 64);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { _mm512_add_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { _mm512_sub_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi8(self.0, other.0) })
    }

    /// Compare-greater-than: returns 64-bit mask. Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u64 {
        unsafe { _mm512_cmpgt_epi8_mask(self.0, other.0) }
    }
}

impl_bin_op!(I8x64, Add, add, _mm512_add_epi8);
impl_bin_op!(I8x64, Sub, sub, _mm512_sub_epi8);
impl_assign_op!(I8x64, AddAssign, add_assign, _mm512_add_epi8);
impl_assign_op!(I8x64, SubAssign, sub_assign, _mm512_sub_epi8);

impl fmt::Debug for I8x64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I8x64({:?})", &self.to_array()[..])
    }
}
impl PartialEq for I8x64 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I8x32 — 32 × i8 in one AVX2 register (__m256i)
// Lives here so consumers get unified import paths across tiers.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I8x32(pub __m256i);

impl I8x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 32);
        Self(unsafe { _mm256_loadu_si256(s.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 32] {
        let mut arr = [0i8; 32];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 32);
        unsafe { _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { _mm256_add_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { _mm256_sub_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epi8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epi8(self.0, other.0) })
    }

    /// Compare-greater-than: returns 32-bit mask via packed-byte movemask.
    /// Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        unsafe { _mm256_movemask_epi8(_mm256_cmpgt_epi8(self.0, other.0)) as u32 }
    }
}

impl Add for I8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_epi8(self.0, rhs.0) })
    }
}
impl Sub for I8x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_sub_epi8(self.0, rhs.0) })
    }
}
impl AddAssign for I8x32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_epi8(self.0, rhs.0) };
    }
}
impl SubAssign for I8x32 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_sub_epi8(self.0, rhs.0) };
    }
}
impl fmt::Debug for I8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I8x32({:?})", &self.to_array()[..])
    }
}
impl PartialEq for I8x32 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I16x32 — 32 × i16 in one AVX-512 register (__m512i)
// AVX-512BW: 16-bit add/sub/min/max, 32-bit cmpgt mask.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I16x32(pub __m512i);

impl I16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(unsafe { _mm512_set1_epi16(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 32);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 32] {
        let mut arr = [0i16; 32];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 32);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { _mm512_add_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { _mm512_sub_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi16(self.0, other.0) })
    }

    /// Compare-greater-than: returns 32-bit mask. Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        unsafe { _mm512_cmpgt_epi16_mask(self.0, other.0) }
    }
}

impl_bin_op!(I16x32, Add, add, _mm512_add_epi16);
impl_bin_op!(I16x32, Sub, sub, _mm512_sub_epi16);
impl_assign_op!(I16x32, AddAssign, add_assign, _mm512_add_epi16);
impl_assign_op!(I16x32, SubAssign, sub_assign, _mm512_sub_epi16);

impl fmt::Debug for I16x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I16x32({:?})", &self.to_array()[..])
    }
}
impl PartialEq for I16x32 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I16x16 — 16 × i16 in one AVX2 register (__m256i)
// Lives here so consumers get unified import paths.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I16x16(pub __m256i);

impl I16x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm256_loadu_si256(s.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        let mut arr = [0i16; 16];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 16);
        unsafe { _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { _mm256_add_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { _mm256_sub_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epi16(self.0, other.0) })
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epi16(self.0, other.0) })
    }

    /// Compare-greater-than: returns 16-bit mask via packed-word movemask.
    /// Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        unsafe {
            // _mm256_cmpgt_epi16 produces 16-bit lanes of all-ones / all-zeros.
            // Pack to bytes (signed sat), then use movemask_epi8 — needs a
            // permute to undo the per-128-bit packing that packs_epi16 does.
            let cmp = _mm256_cmpgt_epi16(self.0, other.0);
            let packed = _mm256_packs_epi16(cmp, _mm256_setzero_si256());
            let perm = _mm256_permute4x64_epi64(packed, 0b0000_1000);
            let mask32 = _mm256_movemask_epi8(perm) as u32;
            (mask32 & 0xFFFF) as u16
        }
    }
}

impl Add for I16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_epi16(self.0, rhs.0) })
    }
}
impl Sub for I16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_sub_epi16(self.0, rhs.0) })
    }
}
impl AddAssign for I16x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_epi16(self.0, rhs.0) };
    }
}
impl SubAssign for I16x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_sub_epi16(self.0, rhs.0) };
    }
}
impl fmt::Debug for I16x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I16x16({:?})", &self.to_array()[..])
    }
}
impl PartialEq for I16x16 {
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

// I8/I16 SIMD aliases
#[allow(non_camel_case_types)]
pub type i8x64 = I8x64;
#[allow(non_camel_case_types)]
pub type i8x32 = I8x32;
#[allow(non_camel_case_types)]
pub type i16x32 = I16x32;
#[allow(non_camel_case_types)]
pub type i16x16 = I16x16;

// 256-bit int lanes — added 2026-05-20 missing-lanes sweep. These types
// don't have native `__m256i` wrappers in this module yet; re-exported
// from `simd_avx2.rs` (where they live as scalar-storage polyfills via
// the `avx2_int_type!` macro) so the v4 dispatch arm in `simd.rs` can
// surface them through `crate::simd::*` with the same names the v3 arm
// uses. Native AVX2 `__m256i` upgrades for these are TD-SIMD-3.
pub use crate::simd_avx2::{i32x8, i64x4, u16x16, u32x8, u64x4, I32x8, I64x4, U16x16, U32x8, U64x4};

// ============================================================================
// W1a SIMD primitives — AVX-512 backend
// ============================================================================
//
// Five new primitives per `.claude/knowledge/vertical-simd-consumer-contract.md`.
// These live in the AVX-512 backend file and are the "real-intrinsic" tier where
// applicable.  When AVX-512 doesn't provide a narrower type natively (I8x16,
// U16x8, U8x8), we define minimal scalar-storage wrappers so the cross-arch API
// is uniform.
//
// Types NEW to this backend (polyfill wrappers):
//   I8x16  — 16 × i8, scalar storage (no native __m128i wrapping is necessary
//             to match the API; AVX-512 native I8x64 is wider and the consumer
//             only needs the 16-lane primitive here).
//   U16x8  — 8 × u16, scalar storage polyfill.
//   U8x8   — 8 × u8, scalar storage polyfill.

// ─── I8x16 (scalar-storage polyfill for the AVX-512 backend) ─────────────────

/// 16-lane `i8` vector.  On the AVX-512 backend this is a scalar-storage
/// polyfill (no native 128-bit intrinsic wrapper is needed for the W1a API
/// surface); on NEON it is backed by `int8x16_t`.
///
/// Edge cases and lane layout are identical across backends; only performance
/// differs.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, PartialEq)]
#[repr(align(16))]
pub struct I8x16(pub [i8; 16]);

#[cfg(target_arch = "x86_64")]
impl I8x16 {
    pub const LANES: usize = 16;

    /// Broadcast a single `i8` value to all 16 lanes.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x16::splat(3);
    /// assert!(v.to_array().iter().all(|&x| x == 3));
    /// ```
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self([v; 16])
    }

    /// Load 16 lanes from a slice (at least 16 elements required).
    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 16);
        let mut a = [0i8; 16];
        a.copy_from_slice(&s[..16]);
        Self(a)
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [i8; 16]) -> Self {
        Self(arr)
    }

    /// Extract all 16 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        self.0
    }

    /// Copy lanes into a slice (must have at least 16 elements).
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 16);
        s[..16].copy_from_slice(&self.0);
    }

    // ── W1a-#1: from_i4_packed_u64 + lane_i8 ────────────────────────────────

    /// Unpack 16 signed i4 nibbles from a `u64` into 16 sign-extended `i8` lanes.
    ///
    /// Nibble layout: `lane[i] = sign_extend_i4((packed >> (4*i)) & 0xf)`.
    /// Values `0x0..=0x7` map to `0..=7`; values `0x8..=0xf` map to `-8..=-1`.
    ///
    /// On the x86_64 backend this is a scalar polyfill (no AVX-512 intrinsic path
    /// here since the spec shows multiple equivalent approaches and the NEON
    /// path is the primary SIMD path for narrow unpacking).
    ///
    /// # Example
    /// ```rust,ignore
    /// // All nibbles == 0x0 → all lanes == 0
    /// let z = I8x16::from_i4_packed_u64(0);
    /// assert_eq!(z.lane_i8::<0>(), 0);
    /// // Nibble 0xf → -1
    /// let neg = I8x16::from_i4_packed_u64(0xffff_ffff_ffff_ffff);
    /// assert_eq!(neg.lane_i8::<0>(), -1);
    /// // Nibble 0x8 → -8
    /// let min4 = I8x16::from_i4_packed_u64(0x8888_8888_8888_8888);
    /// assert_eq!(min4.lane_i8::<0>(), -8);
    /// ```
    #[inline(always)]
    pub fn from_i4_packed_u64(packed: u64) -> Self {
        let mut lanes = [0i8; 16];
        for i in 0..16 {
            let nibble = ((packed >> (4 * i)) & 0xf) as i8;
            // Sign-extend: if bit 3 is set the value is negative
            lanes[i] = if nibble > 7 { nibble - 16 } else { nibble };
        }
        Self(lanes)
    }

    /// Extract lane `N` as an `i8` (const-generic, checked at compile time).
    ///
    /// `N` must be in `0..16`; this is enforced by the array index.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x16::from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16i8]);
    /// assert_eq!(v.lane_i8::<0>(), 1);
    /// assert_eq!(v.lane_i8::<15>(), 16);
    /// ```
    #[inline(always)]
    pub fn lane_i8<const N: usize>(self) -> i8 {
        self.0[N]
    }

    // ── W1a-#2: saturating_abs ────────────────────────────────────────────────

    /// Lane-wise saturating absolute value.
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX` (127), unlike the hardware VPABSB
    /// which returns `i8::MIN` (−128) for the minimum value because +128 does
    /// not fit in `i8`.  On AVX-512 this is corrected with `_mm512_min_epu8`
    /// (VPMINUB) per the VPABSB correction in the consumer contract.  This
    /// x86_64 polyfill delegates to `i8::saturating_abs` on the scalar path.
    ///
    /// All lanes are independently saturated.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x16::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        // SAFETY: `_mm_abs_epi8` (SSSE3) and `_mm_min_epu8` (SSE2) are available
        // on every x86_64 build this file compiles for — the workspace pins
        // `x86-64-v3`, which includes SSSE3. The unaligned load/store match the
        // `[i8; 16]` storage. VPABSB returns 0x80 for `i8::MIN` (the bit pattern
        // of +128, which does not fit in i8); VPMINUB then clamps 0x80 (= 128
        // unsigned) down to 0x7f (= 127 = `i8::MAX`), producing the saturating
        // result bare VPABSB cannot — per the consumer contract's VPABSB
        // correction. All 16 lanes are saturated branchlessly.
        use core::arch::x86_64::*;
        unsafe {
            let v = _mm_loadu_si128(self.0.as_ptr() as *const __m128i);
            let clamped = _mm_min_epu8(_mm_abs_epi8(v), _mm_set1_epi8(0x7f_u8 as i8));
            let mut o = [0i8; 16];
            _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, clamped);
            Self(o)
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for I8x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I8x16({:?})", &self.0[..])
    }
}

// ─── U8x8 (scalar-storage polyfill for AVX-512 backend) ──────────────────────

/// 8-lane `u8` vector.  Scalar-storage polyfill used by `palette_lookup_u8x8`.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, PartialEq)]
#[repr(align(8))]
pub struct U8x8(pub [u8; 8]);

#[cfg(target_arch = "x86_64")]
impl U8x8 {
    pub const LANES: usize = 8;

    /// Broadcast a single `u8` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self([v; 8])
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u8; 8]) -> Self {
        Self(arr)
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 8] {
        self.0
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for U8x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "U8x8({:?})", &self.0[..])
    }
}

// ─── U16x8 (scalar-storage polyfill for AVX-512 backend) ─────────────────────

/// 8-lane `u16` vector.  On the NEON backend this is backed by `uint16x8_t`; on
/// x86_64 (both AVX-512 and AVX2) it is a scalar-storage polyfill with the same
/// API.
///
/// The W1a gather primitive is defined as a method on this type.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, PartialEq)]
#[repr(align(16))]
pub struct U16x8(pub [u16; 8]);

#[cfg(target_arch = "x86_64")]
impl U16x8 {
    pub const LANES: usize = 8;

    /// Broadcast a single `u16` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self([v; 8])
    }

    /// Load from a slice (at least 8 elements required).
    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8);
        let mut a = [0u16; 8];
        a.copy_from_slice(&s[..8]);
        Self(a)
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u16; 8]) -> Self {
        Self(arr)
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        self.0
    }

    // ── W1a-#3: gather_u16 ───────────────────────────────────────────────────

    /// Gather 8 `u16` values from `table` at the indices given by `self`.
    ///
    /// In debug builds, panics if any index is `>= table.len()`.
    /// In release builds, falls through to a scalar loop using `get()` so
    /// out-of-range indices return `0` safely instead of reading past the
    /// slice end.
    ///
    /// On x86_64 this is a scalar-loop polyfill (real AVX2 gather via
    /// `_mm256_i32gather_epi32` + downcast is tracked as a follow-up
    /// optimisation; the scalar path is the correctness anchor per the
    /// contract).
    ///
    /// # Example
    /// ```rust,ignore
    /// let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
    /// let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
    /// let result = U16x8::gather_u16(idx, &table);
    /// assert_eq!(result.to_array(), [10, 30, 50, 70, 20, 40, 60, 80]);
    /// ```
    #[inline(always)]
    pub fn gather_u16(indices: U16x8, table: &[u16]) -> Self {
        let idx = indices.to_array();
        // Bounds validation: debug panics, release falls back to safe get()
        #[cfg(debug_assertions)]
        for &i in &idx {
            assert!(
                (i as usize) < table.len(),
                "gather_u16: index {} out of bounds (table.len() = {})",
                i,
                table.len()
            );
        }
        let mut out = [0u16; 8];
        for k in 0..8 {
            // SAFETY: in debug we already panicked above; in release `get`
            // returns None for OOB and we fall back to 0.
            out[k] = table.get(idx[k] as usize).copied().unwrap_or(0);
        }
        Self(out)
    }

    /// Extract lane `k` as a `u16` (for use in gather loops).
    #[inline(always)]
    pub fn lane(self, k: usize) -> u16 {
        self.0[k]
    }
}

#[cfg(target_arch = "x86_64")]
impl core::fmt::Debug for U16x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "U16x8({:?})", &self.0[..])
    }
}

// ─── W1a-#3: palette_lookup_u8x8 (AVX-512 backend) ───────────────────────────

/// Look up 8 bytes from a `u8` LUT by `u16` indices.
///
/// Convenience wrapper over `U16x8::gather_u16`: widens each index to u16,
/// reads the byte at that position in `lut`, and returns an 8-lane `U8x8`.
///
/// Bounds: panics in debug if any index `>= lut.len()`; returns 0 safely in
/// release for out-of-range indices.
///
/// # Example
/// ```rust,ignore
/// let lut: Vec<u8> = (0..256).map(|x| x as u8).collect();
/// let idx = U16x8::from_array([0, 1, 127, 128, 254, 255, 10, 20]);
/// let result = palette_lookup_u8x8(idx, &lut);
/// assert_eq!(result.to_array(), [0, 1, 127, 128, 254, 255, 10, 20]);
/// ```
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn palette_lookup_u8x8(idx_v: U16x8, lut: &[u8]) -> U8x8 {
    let idx = idx_v.to_array();
    #[cfg(debug_assertions)]
    for &i in &idx {
        assert!((i as usize) < lut.len(), "palette_lookup_u8x8: index {} OOB (lut.len() = {})", i, lut.len());
    }
    let mut out = [0u8; 8];
    for k in 0..8 {
        out[k] = lut.get(idx[k] as usize).copied().unwrap_or(0);
    }
    U8x8(out)
}

// ─── W1a-#2: I8x32::saturating_abs (AVX-512 backend) ─────────────────────────
//
// The AVX-512 I8x32 type lives in this file (backed by `__m256i`).
// We add saturating_abs using the VPABSB correction from the spec:
//   1. _mm256_abs_epi8  (VPABSB on AVX2) gives raw abs; returns 0x80 for 0x80.
//   2. _mm256_min_epu8  (VPMINUB) clamps 0x80 → 0x7f.

impl I8x32 {
    /// Lane-wise saturating absolute value.
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX` (127).  Uses the VPABSB +
    /// VPMINUB correction because VPABSB alone returns `i8::MIN` for the
    /// minimum lane value (the bit-pattern of +128 does not fit in `i8`).
    ///
    /// All 32 lanes are independently saturated.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x32::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// let v2 = I8x32::from_array([-1i8; 32]);
    /// assert!(v2.saturating_abs().to_array().iter().all(|&x| x == 1));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        // SAFETY: _mm256_abs_epi8 (VPABSB) is an AVX2 intrinsic; we are in
        // the simd_avx512.rs file which is only compiled for x86_64.  The
        // `target_feature(enable = "avx2")` annotation on the calling code
        // path guarantees AVX2 availability.  The raw_abs result for 0x80
        // is 0x80 (bit-pattern +128); VPMINUB then clamps it to 0x7f.
        // UNVERIFIED: _mm256_abs_epi8 stability on Rust 1.94 stable — it is
        // in std::arch::x86_64 since Rust 1.0 for AVX2 so should compile.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let raw_abs = core::arch::x86_64::_mm256_abs_epi8(self.0);
            // VPMINUB: unsigned-byte minimum. 0x80 unsigned = 128 > 0x7f = 127
            // so min(0x80, 0x7f) = 0x7f.  All values < 0x80 pass through.
            let clamped =
                core::arch::x86_64::_mm256_min_epu8(raw_abs, core::arch::x86_64::_mm256_set1_epi8(0x7f_u8 as i8));
            I8x32(clamped)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Scalar fallback (unreachable in practice for AVX-512 builds)
            let mut o = [0i8; 32];
            let arr = self.to_array();
            for i in 0..32 {
                o[i] = arr[i].saturating_abs();
            }
            I8x32::from_array(o)
        }
    }
}

// ─── W1a-#5: U64x8::popcnt / xor_popcount (AVX-512 backend) ──────────────────

impl U64x8 {
    /// Lane-wise population count (number of set bits) for each of the 8
    /// `u64` lanes.  Each result lane holds a value in `0..=64`.
    ///
    /// On AVX-512 with `avx512vpopcntdq` the native `_mm512_popcnt_epi64`
    /// instruction is used.  Without that extension (or when compiling for
    /// the scalar polyfill path) a Mula-style byte-LUT via VPSHUFB is used,
    /// or the scalar `u64::count_ones` fused loop.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = U64x8::splat(u64::MAX);  // all bits set → 64 per lane
    /// let p = v.popcnt();
    /// assert!(p.to_array().iter().all(|&x| x == 64));
    /// let z = U64x8::splat(0);
    /// assert!(z.popcnt().to_array().iter().all(|&x| x == 0));
    /// ```
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        // UNVERIFIED: _mm512_popcnt_epi64 requires `avx512vpopcntdq`; the
        // cfg guard below selects it only when that feature is enabled at
        // compile time.  On Sapphire Rapids + Zen4 it should be available.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512vpopcntdq"))]
        {
            // SAFETY: avx512vpopcntdq is enabled at compile time (cfg guard).
            // _mm512_popcnt_epi64 is a stable intrinsic from std::arch::x86_64.
            // UNVERIFIED: exact Rust stable version this intrinsic landed in —
            // believed to be 1.72 but not confirmed against 1.94.
            unsafe {
                let result = core::arch::x86_64::_mm512_popcnt_epi64(self.0);
                U64x8(result)
            }
        }
        // Scalar fallback for AVX-512F builds without VPOPCNTDQ.
        // The Mula-algorithm via VPSHUFB + VPSADBW would be faster but
        // requires avx512bw which may not be present alongside avx512f.
        // Scalar u64::count_ones is ~4 cycles per lane on modern CPUs and
        // is the safe correctness anchor (TD follow-up: add avx512bw guard).
        // UNVERIFIED: whether avx512bw is guaranteed to co-exist with avx512f
        // on the production deployment targets; leaving as scalar until confirmed.
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512vpopcntdq")))]
        {
            let arr = self.to_array();
            let mut out = [0u64; 8];
            for i in 0..8 {
                out[i] = arr[i].count_ones() as u64;
            }
            U64x8::from_array(out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Scalar fallback (unreachable in practice for this backend file).
            let arr = self.to_array();
            let mut out = [0u64; 8];
            for i in 0..8 {
                out[i] = arr[i].count_ones() as u64;
            }
            U64x8::from_array(out)
        }
    }

    /// XOR two vectors lane-wise, popcount each lane, then sum across all 8
    /// lanes.  Optimised for Hamming-distance reductions.
    ///
    /// Equivalent to `(self ^ other).popcnt().reduce_sum()` but avoids a
    /// store/reload cycle when all 8 popcounts are needed only as a sum.
    ///
    /// # Example
    /// ```rust,ignore
    /// let a = U64x8::splat(u64::MAX);
    /// let b = U64x8::splat(0);
    /// // All bits different → 64 set bits per lane × 8 lanes = 512
    /// assert_eq!(a.xor_popcount(b), 512);
    /// let same = U64x8::splat(0xdead_beef_cafe_babe);
    /// assert_eq!(same.xor_popcount(same), 0);
    /// ```
    #[inline(always)]
    pub fn xor_popcount(self, other: Self) -> u64 {
        // XOR first, then popcount + horizontal sum.
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: BitXor on U64x8 uses _mm512_xor_si512; popcnt uses the
            // avx512 path above.  reduce_sum uses _mm512_reduce_add_epi64.
            let xored = self ^ other;
            xored.popcnt().reduce_sum()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let a = self.to_array();
            let b = other.to_array();
            let mut sum = 0u64;
            for i in 0..8 {
                sum += (a[i] ^ b[i]).count_ones() as u64;
            }
            sum
        }
    }
}

// ─── W1a-#5: U64x4::popcnt (AVX-512 backend via simd_avx2 polyfill) ──────────
//
// U64x4 lives in simd_avx2.rs as a scalar-storage polyfill (avx2_int_type!).
// The AVX-512 backend re-exports it from simd_avx2.rs (see the re-export at
// line ~2265: `pub use crate::simd_avx2::{…U64x4…}`).
// We add popcnt to U64x4 via an impl block in simd_avx2.rs (see that file).

// ─── W1a-#4: prefetch_read_t0/t1/t2 (x86_64) ────────────────────────────────

/// Hint that the cache line containing `ptr` will be read soon; load into L1
/// (T0) data cache.
///
/// `ptr` is allowed to be invalid (null, unmapped).  On x86_64 an invalid
/// address in `PREFETCHT0` is silently dropped by the hardware; no fault is
/// raised.  Do NOT `assert!` or dereference `ptr` in this function.
///
/// # Example
/// ```rust,ignore
/// let data = vec![0u8; 4096];
/// unsafe { prefetch_read_t0(data.as_ptr()); }
/// ```
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_read_t0(ptr: *const u8) {
    // SAFETY: _MM_HINT_T0 prefetch on x86_64 is a hint-only instruction;
    // it does NOT fault on invalid addresses per Intel SDM § PREFETCHT0.
    // The pointer is never dereferenced.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
}

/// Hint to load into L2 (T1) cache.  Same invalid-pointer semantics as
/// `prefetch_read_t0`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_read_t1(ptr: *const u8) {
    // SAFETY: same as prefetch_read_t0 — hint-only, no fault on invalid ptr.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
    }
}

/// Hint to load into L3 (T2) cache.  Same invalid-pointer semantics as
/// `prefetch_read_t0`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_read_t2(ptr: *const u8) {
    // SAFETY: same as prefetch_read_t0 — hint-only, no fault on invalid ptr.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T2 }>(ptr as *const i8);
    }
}

// ─── W1a-#1: batch_packed_i4_16 (x86_64 backend) ─────────────────────────────

/// Closure-parameterised batch over packed i4 data.
///
/// Iterates `min(packed.len(), aux.len())` times.  Each iteration unpacks
/// `packed[i]` into an `I8x16` (16 sign-extended nibbles) and passes it
/// together with `aux[i]` to the closure `f`, storing the result in `out[i]`.
///
/// Tail handling: if `out.len() < packed.len()` only `out.len()` iterations
/// run (no out-of-bounds write).
///
/// Bounds: panics if `packed.len() != aux.len()`.  An empty slice is valid.
///
/// # Example
/// ```rust,ignore
/// let packed = vec![0u64; 4];
/// let aux    = vec![0i8; 4];
/// let mut out = vec![0i8; 4];
/// batch_packed_i4_16(&packed, &aux, &mut out, |lanes, a| {
///     lanes.lane_i8::<0>().wrapping_add(a)
/// });
/// assert!(out.iter().all(|&v| v == 0));
/// ```
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn batch_packed_i4_16<E, F>(packed: &[u64], aux: &[i8], out: &mut [E], f: F)
where
    F: Fn(I8x16, i8) -> E + Sync + Send,
    E: Copy,
{
    assert_eq!(packed.len(), aux.len(), "batch_packed_i4_16: packed and aux must be same length");
    let n = packed.len().min(out.len());
    for i in 0..n {
        let lanes = I8x16::from_i4_packed_u64(packed[i]);
        out[i] = f(lanes, aux[i]);
    }
}

// ─── Aliases ──────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[allow(non_camel_case_types)]
pub type i8x16 = I8x16;
#[cfg(target_arch = "x86_64")]
#[allow(non_camel_case_types)]
pub type u16x8 = U16x8;
#[cfg(target_arch = "x86_64")]
#[allow(non_camel_case_types)]
pub type u8x8 = U8x8;

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
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vl") {
            // SAFETY: feature detection confirmed avx512bf16 + avx512vl
            unsafe {
                convert_bf16_to_f32_avx512bf16(input, output);
            }
            return;
        }
        // Middle tier: pure AVX-512F bit-shift (Skylake-X, Cascade Lake,
        // Ice Lake-SP — all AVX-512F CPUs without the bf16 extension).
        // BF16 → f32 is lossless: BF16 IS the upper 16 bits of f32, so
        // `(bf16_u16 as u32) << 16` reinterpreted as f32 IS the exact
        // value. Vectorized: one _mm512_cvtepu16_epi32 zero-extends 16
        // u16 → 16 u32, one _mm512_slli_epi32::<16> shifts each lane left
        // by 16, _mm512_castsi512_ps reinterprets the i32 bit pattern as
        // f32. Three AVX-512F instructions per 16-lane chunk vs 16
        // scalar shifts in the fallback below.
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detection confirmed avx512f.
            unsafe {
                convert_bf16_to_f32_avx512f(input, output);
            }
            return;
        }
    }

    // Scalar fallback (all platforms, all CPUs)
    for (src, dst) in input.iter().copied().zip(output.iter_mut()) {
        *dst = bf16_to_f32_scalar(src);
    }
}

/// Pure-AVX-512F BF16 → f32 conversion. Bit-exact against
/// `bf16_to_f32_scalar` on every input — BF16 is `f32_bits >> 16`, so
/// the inverse `(bf16 as u32) << 16` reconstructed as f32 is exact.
///
/// 16-lane main loop via `_mm512_cvtepu16_epi32` (zero-extend) +
/// `_mm512_slli_epi32::<16>` (shift up) + `_mm512_castsi512_ps`
/// (bit-cast). Scalar tail for the last `n % 16` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn convert_bf16_to_f32_avx512f(input: &[u16], output: &mut [f32]) {
    let n = input.len();
    let mut i = 0usize;

    // Main 16-wide loop.
    while i + 16 <= n {
        let raw256 = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);
        let extended = _mm512_cvtepu16_epi32(raw256);
        let shifted = _mm512_slli_epi32::<16>(extended);
        let as_f32 = _mm512_castsi512_ps(shifted);
        _mm512_storeu_ps(output.as_mut_ptr().add(i), as_f32);
        i += 16;
    }

    // Scalar tail (0..15 remaining lanes).
    while i < n {
        *output.get_unchecked_mut(i) = bf16_to_f32_scalar(*input.get_unchecked(i));
        i += 1;
    }
}

/// Batch f32 → BF16 conversion: same pattern.
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) {
    assert!(output.len() >= input.len(), "output must be >= input length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vl") {
            unsafe {
                convert_f32_to_bf16_avx512bf16(input, output);
            }
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
    let is_nonzero: __mmask16 = _mm512_cmpgt_epu32_mask(abs_bits, _mm512_setzero_si512());
    let is_subnormal: __mmask16 = is_sub_or_zero & is_nonzero;

    let is_nan: __mmask16 = _mm512_cmpgt_epu32_mask(abs_bits, _mm512_set1_epi32(0x7F80_0000u32 as i32));

    // Blend order:
    //   1. start from the normal RNE result,
    //   2. overwrite subnormal lanes with the sign-only zero,
    //   3. overwrite NaN lanes with the quieted payload.
    let with_subnormal = _mm512_mask_blend_epi32(is_subnormal, normal_out, sign_only);
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

        let abs_bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
        let exp_bound = _mm512_set1_epi32(0x0080_0000);
        let is_sub_or_zero: __mmask16 = _mm512_cmplt_epu32_mask(abs_bits, exp_bound);
        let is_nonzero: __mmask16 = _mm512_cmpgt_epu32_mask(abs_bits, _mm512_setzero_si512());
        let is_subnormal: __mmask16 = is_sub_or_zero & is_nonzero;
        let is_nan: __mmask16 = _mm512_cmpgt_epu32_mask(abs_bits, _mm512_set1_epi32(0x7F80_0000u32 as i32));

        let with_subnormal = _mm512_mask_blend_epi32(is_subnormal, normal_out, sign_only);
        let merged = _mm512_mask_blend_epi32(is_nan, with_subnormal, nan_out);

        // SAFETY: masked store — only lanes [0, rem) are touched.
        _mm512_mask_cvtepi32_storeu_epi16(output.as_mut_ptr().add(i) as *mut _, mask, merged);
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
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
        let input: Vec<u16> = (0..100)
            .map(|i| f32_to_bf16_scalar(i as f32 * 0.1 - 5.0))
            .collect();
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

    /// Direct test for the AVX-512F bit-shift BF16 → f32 arm, exercising
    /// the path the dispatcher would skip when avx512bf16 is available.
    /// Verifies bit-exact parity against the scalar reference across a
    /// pathological corpus (subnormal, NaN, Inf, sign ±0, every exponent
    /// boundary) and a 16-aligned-plus-tail length.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn batch_bf16_to_f32_avx512f_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f not detected on this host; skipping");
            return;
        }
        // Build a corpus: every bf16 value of interest. The dispatcher's
        // 16-wide loop is what matters most; pick a non-aligned total so
        // we also exercise the scalar tail.
        let mut input: Vec<u16> = Vec::new();
        // Sign × exponent × representative mantissa sweep
        for sign in [0u16, 0x8000] {
            for exp in 0..256u16 {
                for &mant in &[0u16, 1, 0x40, 0x7F] {
                    input.push(sign | (exp << 7) | mant);
                }
            }
        }
        // Add 5 bytes of tail to land on a non-16-aligned length.
        input.extend_from_slice(&[0x3F80, 0xBF80, 0x4000, 0xC000, 0x7F80]);

        let mut output = vec![0.0f32; input.len()];
        // SAFETY: avx512f confirmed above.
        unsafe { convert_bf16_to_f32_avx512f(&input, &mut output) };

        for (i, &bf16) in input.iter().enumerate() {
            let expected = bf16_to_f32_scalar(bf16);
            // BF16 → f32 is lossless: bits must be byte-equal (incl. NaN
            // payloads).
            assert_eq!(
                output[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {} (bf16=0x{:04x}): got {} (0x{:08x}) vs {} (0x{:08x})",
                i,
                bf16,
                output[i],
                output[i].to_bits(),
                expected,
                expected.to_bits()
            );
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
                0x0000_00u32, 0x400000,  // halfway-below-LSB for even mantissa
                0x7FFFFF,  // top of mantissa (rounding into next exponent)
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
                _mm256_storeu_si256(rne_out.as_mut_ptr().add(i) as *mut __m256i, packed);
                i += 16;
            }
        }

        // Reference: hardware `_mm512_cvtneps_pbh` if available.
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vl") {
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
                mismatches,
                0,
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
                (0x0000_0000, 0x0000), // +0
                (0x8000_0000, 0x8000), // -0
                (0x3F80_0000, 0x3F80), // 1.0
                (0xBF80_0000, 0xBF80), // -1.0
                (0x7F80_0000, 0x7F80), // +Inf
                (0xFF80_0000, 0xFF80), // -Inf
                (0x7FC0_0000, 0x7FC0), // canonical qNaN
                (0x7F80_0001, 0x7FC0), // sNaN → qNaN
                (0x7FBF_FFFF, 0x7FFF), // sNaN payload → QNaN'd
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
            let mut batch: Vec<f32> = reference.iter().map(|&(b, _)| f32::from_bits(b)).collect();
            while batch.len() % 16 != 0 {
                batch.push(0.0);
            }
            let mut simd_out = vec![0u16; batch.len()];
            unsafe {
                // SAFETY: avx512f confirmed above.
                let v = _mm512_loadu_ps(batch.as_ptr());
                let packed = f32_to_bf16_x16_rne(v);
                _mm256_storeu_si256(simd_out.as_mut_ptr() as *mut __m256i, packed);
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
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, packed);
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
                bf16_mant_lsb,
                0,
                "round-to-even failed for input idx={idx} bits=0x{:08X}: bf16=0x{got:04X}",
                v.to_bits()
            );

            // Also cross-check with the scalar reference.
            let scalar = f32_to_bf16_scalar_rne(v);
            assert_eq!(got, scalar, "SIMD vs scalar RNE disagree for 0x{:08X}", v.to_bits());
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
                assert_eq!(batch_out[i], expected, "batch RNE mismatch len={len} idx={i} bits=0x{:08X}", v.to_bits());
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
                if truncated & 1 != 0 {
                    truncated + 1
                } else {
                    truncated
                }
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
                if truncated & 1 != 0 {
                    truncated + 1
                } else {
                    truncated
                }
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
                    let src = _mm256_loadu_si256(input[c * 16..].as_ptr() as *const __m256i);
                    let dst = _mm512_cvtph_ps(src);
                    _mm512_storeu_ps(output[c * 16..].as_mut_ptr(), dst);
                }
            }
            // Scalar tail
            for i in (chunks16 * 16)..n {
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
                    let src = _mm_loadu_si128(input[c * 8..].as_ptr() as *const __m128i);
                    let dst = _mm256_cvtph_ps(src);
                    _mm256_storeu_ps(output[c * 8..].as_mut_ptr(), dst);
                }
            }
            for i in (chunks8 * 8)..n {
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
                    let src = _mm512_loadu_ps(input[c * 16..].as_ptr());
                    // imm8=0x00: _MM_FROUND_TO_NEAREST_INT (RNE)
                    let dst: __m256i = _mm512_cvtps_ph::<0x00>(src);
                    _mm256_storeu_si256(output[c * 16..].as_mut_ptr() as *mut __m256i, dst);
                }
            }
            for i in (chunks16 * 16)..n {
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
                    let src = _mm256_loadu_ps(input[c * 8..].as_ptr());
                    let dst: __m128i = _mm256_cvtps_ph::<0x00>(src);
                    _mm_storeu_si128(output[c * 8..].as_mut_ptr() as *mut __m128i, dst);
                }
            }
            for i in (chunks8 * 8)..n {
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

#[cfg(all(test, target_feature = "avx512f"))]
mod f16_tests {
    use super::*;

    #[test]
    fn f16_ieee754_exact_values() {
        // IEEE 754 binary16 exact test vectors
        assert_eq!(f16_to_f32_ieee754(0x0000), 0.0); // +0
        assert_eq!(f16_to_f32_ieee754(0x8000), -0.0); // −0
        assert_eq!(f16_to_f32_ieee754(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32_ieee754(0xBC00), -1.0); // −1.0
        assert_eq!(f16_to_f32_ieee754(0x4000), 2.0); // 2.0
        assert_eq!(f16_to_f32_ieee754(0x3800), 0.5); // 0.5
        assert_eq!(f16_to_f32_ieee754(0x7BFF), 65504.0); // max normal
        assert!(f16_to_f32_ieee754(0x7C00).is_infinite()); // +Inf
        assert!(f16_to_f32_ieee754(0xFC00).is_infinite()); // −Inf
        assert!(f16_to_f32_ieee754(0x7C01).is_nan()); // NaN
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
                assert_eq!(h, back, "roundtrip failed: 0x{:04X} → {} → 0x{:04X}", h, f, back);
            }
        }
    }

    #[test]
    fn f16_exact_representable_values() {
        // Values that are exactly representable in f16 must roundtrip perfectly
        let exact_values: &[f32] = &[
            0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, 0.125, 65504.0, -65504.0,       // max f16
            0.000061035156, // smallest normal f16 (2^-14)
        ];
        for &v in exact_values {
            let h = f32_to_f16_ieee754_rne(v);
            let back = f16_to_f32_ieee754(h);
            assert_eq!(v, back, "exact value roundtrip failed: {} → 0x{:04X} → {}", v, h, back);
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
        let input: Vec<u16> = (0..200)
            .map(|i| {
                let v = (i as f32 - 100.0) * 0.5;
                f32_to_f16_ieee754_rne(v)
            })
            .collect();
        let mut batch_out = vec![0.0f32; 200];
        f16_to_f32_batch_ieee754(&input, &mut batch_out);

        for (i, &h) in input.iter().enumerate() {
            let scalar = f16_to_f32_ieee754(h);
            assert_eq!(
                batch_out[i].to_bits(),
                scalar.to_bits(),
                "batch/scalar mismatch at {}: batch=0x{:08X} scalar=0x{:08X}",
                i,
                batch_out[i].to_bits(),
                scalar.to_bits()
            );
        }
    }

    #[test]
    fn f32_to_f16_batch_rne_matches_scalar() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.37).collect();
        let mut batch_out = vec![0u16; 200];
        f32_to_f16_batch_ieee754_rne(&input, &mut batch_out);

        for (i, &v) in input.iter().enumerate() {
            let scalar = f32_to_f16_ieee754_rne(v);
            assert_eq!(
                batch_out[i], scalar,
                "f32→f16 batch/scalar mismatch at {}: input={} batch=0x{:04X} scalar=0x{:04X}",
                i, v, batch_out[i], scalar
            );
        }
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
mod u8x64_rasterizer_tests {
    use super::U8x64;

    #[test]
    fn pairwise_avg_basic() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        let avg = a.pairwise_avg(b);
        let mut out = [0u8; 64];
        avg.copy_to_slice(&mut out);
        // (10 + 20 + 1) >> 1 = 15
        assert!(out.iter().all(|&v| v == 15));
    }

    #[test]
    fn pairwise_avg_rounding() {
        let a = U8x64::splat(1);
        let b = U8x64::splat(2);
        let avg = a.pairwise_avg(b);
        let mut out = [0u8; 64];
        avg.copy_to_slice(&mut out);
        // (1 + 2 + 1) >> 1 = 2  (rounds up)
        assert!(out.iter().all(|&v| v == 2));
    }

    #[test]
    fn cmpgt_mask_basic() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(5);
        assert_eq!(a.cmpgt_mask(b), u64::MAX); // all greater
        assert_eq!(b.cmpgt_mask(a), 0); // none greater
        assert_eq!(a.cmpgt_mask(a), 0); // equal = not greater
    }

    #[test]
    fn mask_blend_selects_correctly() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        // mask = 0: all from a
        let r0 = U8x64::mask_blend(0, a, b);
        let mut out = [0u8; 64];
        r0.copy_to_slice(&mut out);
        assert!(out.iter().all(|&v| v == 10));
        // mask = all 1s: all from b
        let r1 = U8x64::mask_blend(u64::MAX, a, b);
        r1.copy_to_slice(&mut out);
        assert!(out.iter().all(|&v| v == 20));
        // mask = bit 0 only: first byte from b, rest from a
        let r2 = U8x64::mask_blend(1, a, b);
        r2.copy_to_slice(&mut out);
        assert_eq!(out[0], 20);
        assert_eq!(out[1], 10);
    }

    #[test]
    fn shl_epi16_shift_4() {
        let mut data = [0u8; 64];
        data[0] = 0x0F;
        data[1] = 0x00; // u16 = 0x000F
        let v = U8x64::from_slice(&data);
        let shifted = v.shl_epi16(4);
        let mut out = [0u8; 64];
        shifted.copy_to_slice(&mut out);
        let result = u16::from_le_bytes([out[0], out[1]]);
        assert_eq!(result, 0x00F0);
    }

    #[test]
    fn saturating_add_clamps_at_255() {
        let a = U8x64::splat(200);
        let b = U8x64::splat(100);
        let sum = a.saturating_add(b);
        let mut out = [0u8; 64];
        sum.copy_to_slice(&mut out);
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn saturating_add_no_overflow() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        let sum = a.saturating_add(b);
        let mut out = [0u8; 64];
        sum.copy_to_slice(&mut out);
        assert!(out.iter().all(|&v| v == 30));
    }

    #[test]
    fn permute_bytes_identity() {
        let mut data = [0u8; 64];
        for i in 0..64 {
            data[i] = i as u8;
        }
        let v = U8x64::from_slice(&data);
        // Identity permutation
        let mut idx = [0u8; 64];
        for i in 0..64 {
            idx[i] = i as u8;
        }
        let perm = v.permute_bytes(U8x64::from_slice(&idx));
        let mut out = [0u8; 64];
        perm.copy_to_slice(&mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn permute_bytes_reverse() {
        let mut data = [0u8; 64];
        for i in 0..64 {
            data[i] = i as u8;
        }
        let v = U8x64::from_slice(&data);
        // Reverse permutation
        let mut idx = [0u8; 64];
        for i in 0..64 {
            idx[i] = (63 - i) as u8;
        }
        let perm = v.permute_bytes(U8x64::from_slice(&idx));
        let mut out = [0u8; 64];
        perm.copy_to_slice(&mut out);
        for i in 0..64 {
            assert_eq!(out[i], (63 - i) as u8);
        }
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tier3_tests {
    use super::{U16x32, U8x64};

    #[test]
    fn movemask_all_zero() {
        let v = U8x64::splat(0);
        assert_eq!(v.movemask(), 0);
    }

    #[test]
    fn movemask_all_high() {
        let v = U8x64::splat(0xFF);
        assert_eq!(v.movemask(), u64::MAX);
    }

    #[test]
    fn movemask_selective() {
        let mut data = [0u8; 64];
        data[0] = 0x80; // MSB set → bit 0
        data[3] = 0xFF; // MSB set → bit 3
        data[63] = 0x80; // MSB set → bit 63
        let v = U8x64::from_slice(&data);
        let mask = v.movemask();
        assert!(mask & 1 != 0);
        assert!(mask & (1 << 3) != 0);
        assert!(mask & (1 << 63) != 0);
        assert!(mask & (1 << 1) == 0);
    }

    #[test]
    fn u16x32_splat_and_roundtrip() {
        let v = U16x32::splat(1234);
        let arr = v.to_array();
        assert!(arr.iter().all(|&x| x == 1234));
    }

    #[test]
    fn u16x32_add() {
        let a = U16x32::splat(100);
        let b = U16x32::splat(200);
        let c = a + b;
        assert!(c.to_array().iter().all(|&x| x == 300));
    }

    #[test]
    fn u16x32_from_u8x64_lo() {
        let mut data = [0u8; 64];
        for i in 0..32 {
            data[i] = (i + 1) as u8;
        }
        let v = U8x64::from_slice(&data);
        let wide = U16x32::from_u8x64_lo(v);
        let arr = wide.to_array();
        for i in 0..32 {
            assert_eq!(arr[i], (i + 1) as u16);
        }
    }

    #[test]
    fn u16x32_from_u8x64_hi() {
        let mut data = [0u8; 64];
        for i in 32..64 {
            data[i] = i as u8;
        }
        let v = U8x64::from_slice(&data);
        let wide = U16x32::from_u8x64_hi(v);
        let arr = wide.to_array();
        for i in 0..32 {
            assert_eq!(arr[i], (32 + i) as u16);
        }
    }

    #[test]
    fn u16x32_pack_saturate_u8_contains_both() {
        let a = U16x32::splat(42);
        let b = U16x32::splat(200);
        let packed = a.pack_saturate_u8(b);
        let mut out = [0u8; 64];
        packed.copy_to_slice(&mut out);
        let count_42 = out.iter().filter(|&&v| v == 42).count();
        let count_200 = out.iter().filter(|&&v| v == 200).count();
        assert_eq!(count_42, 32, "should have 32 bytes of 42");
        assert_eq!(count_200, 32, "should have 32 bytes of 200");
    }

    #[test]
    fn u16x32_pack_saturate_clamps() {
        let v = U16x32::splat(1000); // > 255
        let packed = v.pack_saturate_u8(U16x32::zero());
        let mut out = [0u8; 64];
        packed.copy_to_slice(&mut out);
        let count_255 = out.iter().filter(|&&v| v == 255).count();
        let count_0 = out.iter().filter(|&&v| v == 0).count();
        assert_eq!(count_255, 32, "1000 clamps to 255");
        assert_eq!(count_0, 32, "zero stays 0");
    }

    #[test]
    fn u16x32_mullo() {
        let a = U16x32::splat(100);
        let b = U16x32::splat(3);
        let c = a.mullo(b);
        assert!(c.to_array().iter().all(|&x| x == 300));
    }

    #[test]
    fn u16x32_shr_shl_roundtrip() {
        let v = U16x32::splat(0x00F0);
        let shifted_right = v.shr(4);
        assert!(shifted_right.to_array().iter().all(|&x| x == 0x000F));
        let shifted_back = shifted_right.shl(4);
        assert!(shifted_back.to_array().iter().all(|&x| x == 0x00F0));
    }

    #[test]
    fn u16x32_reduce_sum() {
        let v = U16x32::splat(10);
        assert_eq!(v.reduce_sum(), 320); // 32 × 10
    }
}

// ────────────────────────────────────────────────────────────────────────
// I8/I16 SIMD tests — verify add/sub/min/max/cmp_gt against scalar
//
// On hosts without target_feature avx512f at compile time, the types in
// crate::simd come from `simd_avx2.rs` (scalar arrays for I8x64/I16x32) and
// `simd_avx512.rs` (AVX2 intrinsics for I8x32/I16x16). These tests exercise
// whichever path the linker selected.
// ────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod int_simd_tests {
    use crate::simd::{I16x16, I16x32, I8x32, I8x64};

    #[test]
    fn i8x64_add_pair_to_constant() {
        // [1..=64] + [64..=1] = [65; 64]
        let mut a = [0i8; 64];
        let mut b = [0i8; 64];
        for i in 0..64 {
            a[i] = (i + 1) as i8;
            b[i] = (64 - i) as i8;
        }
        let va = I8x64::from_slice(&a);
        let vb = I8x64::from_slice(&b);
        let vc = va.add(vb);
        let mut out = [0i8; 64];
        vc.copy_to_slice(&mut out);
        for i in 0..64 {
            assert_eq!(out[i], 65, "i8x64 add lane {} = {}", i, out[i]);
        }
    }

    #[test]
    fn i8x64_sub_min_max_boundary() {
        // Boundary values: -128 (i8::MIN) and 127 (i8::MAX).
        let a = I8x64::splat(127);
        let b = I8x64::splat(-128);
        let mx = a.max(b);
        assert!(mx.to_array().iter().all(|&v| v == 127));
        let mn = a.min(b);
        assert!(mn.to_array().iter().all(|&v| v == -128));
        let zero = a.sub(I8x64::splat(127));
        assert!(zero.to_array().iter().all(|&v| v == 0));
    }

    #[test]
    fn i8x64_cmp_gt_bitmask() {
        let mut a = [0i8; 64];
        for i in 0..64 {
            a[i] = (i as i32 - 32) as i8;
        }
        let va = I8x64::from_slice(&a);
        let vb = I8x64::splat(0);
        let mask = va.cmp_gt(vb);
        let mut expected: u64 = 0;
        for i in 0..64 {
            if a[i] > 0 {
                expected |= 1u64 << i;
            }
        }
        assert_eq!(mask, expected, "i8x64 cmp_gt mask");
    }

    #[test]
    fn i8x32_add_round_trip() {
        let mut a = [0i8; 32];
        let mut b = [0i8; 32];
        for i in 0..32 {
            a[i] = (i + 1) as i8;
            b[i] = (32 - i) as i8;
        }
        let va = I8x32::from_slice(&a);
        let vb = I8x32::from_slice(&b);
        let vc = va.add(vb);
        let out = vc.to_array();
        for i in 0..32 {
            assert_eq!(out[i], 33, "i8x32 add lane {} = {}", i, out[i]);
        }
    }

    #[test]
    fn i8x32_cmp_gt_bitmask() {
        let mut a = [0i8; 32];
        for i in 0..32 {
            a[i] = (i as i32 - 16) as i8;
        }
        let va = I8x32::from_slice(&a);
        let vb = I8x32::splat(0);
        let mask = va.cmp_gt(vb);
        let mut expected: u32 = 0;
        for i in 0..32 {
            if a[i] > 0 {
                expected |= 1u32 << i;
            }
        }
        assert_eq!(mask, expected, "i8x32 cmp_gt mask");
    }

    #[test]
    fn i16x32_add_and_boundary() {
        let a = I16x32::splat(i16::MAX);
        let b = I16x32::splat(1);
        let c = a.add(b);
        // i16::MAX + 1 wraps to i16::MIN under wrapping add.
        assert!(c.to_array().iter().all(|&v| v == i16::MIN));

        let zero = I16x32::splat(0);
        let bigneg = I16x32::splat(i16::MIN);
        let mx = a.max(bigneg);
        assert!(mx.to_array().iter().all(|&v| v == i16::MAX));
        let mn = a.min(zero);
        assert!(mn.to_array().iter().all(|&v| v == 0));
    }

    #[test]
    fn i16x32_cmp_gt_bitmask() {
        let mut a = [0i16; 32];
        for i in 0..32 {
            a[i] = (i as i16) - 16;
        }
        let va = I16x32::from_slice(&a);
        let vb = I16x32::splat(0);
        let mask = va.cmp_gt(vb);
        let mut expected: u32 = 0;
        for i in 0..32 {
            if a[i] > 0 {
                expected |= 1u32 << i;
            }
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn i16x16_add_round_trip_and_min() {
        let a = I16x16::from_array([-100, -50, 0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]);
        let b = I16x16::splat(10);
        let c = a.add(b);
        let exp: [i16; 16] = [-90, -40, 10, 60, 110, 210, 310, 410, 510, 610, 710, 810, 910, 1010, 1110, 1210];
        assert_eq!(c.to_array(), exp);

        let mn = a.min(I16x16::splat(0));
        let exp_min: [i16; 16] = [-100, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(mn.to_array(), exp_min);
    }

    #[test]
    fn i16x16_cmp_gt_bitmask() {
        let mut a = [0i16; 16];
        for i in 0..16 {
            a[i] = (i as i16) - 8;
        }
        let va = I16x16::from_slice(&a);
        let vb = I16x16::splat(0);
        let mask = va.cmp_gt(vb);
        let mut expected: u16 = 0;
        for i in 0..16 {
            if a[i] > 0 {
                expected |= 1u16 << i;
            }
        }
        assert_eq!(mask, expected, "i16x16 cmp_gt mask");
    }

    #[test]
    fn lane_constants_match_widths() {
        assert_eq!(I8x64::LANES, 64);
        assert_eq!(I8x32::LANES, 32);
        assert_eq!(I16x32::LANES, 32);
        assert_eq!(I16x16::LANES, 16);
    }

    // ── W1a primitive tests (binding per the consumer contract) ──────────────

    /// Binding: `saturating_abs(i8::MIN) == i8::MAX` for every lane (the VPABSB
    /// correction — bare VPABSB would return i8::MIN).
    #[test]
    fn w1a_saturating_abs_i8x16_min_saturates_to_max() {
        let r = I8x16::splat(i8::MIN).saturating_abs().to_array();
        assert!(r.iter().all(|&x| x == i8::MAX), "got {r:?}");
    }

    #[test]
    fn w1a_saturating_abs_i8x16_matches_scalar_reference() {
        let corpus: [i8; 16] = [i8::MIN, -128, -127, -1, 0, 1, 7, 8, 64, 126, i8::MAX, -64, -2, 2, 100, -100];
        let got = I8x16::from_array(corpus).saturating_abs().to_array();
        let mut want = [0i8; 16];
        for i in 0..16 {
            want[i] = corpus[i].saturating_abs();
        }
        assert_eq!(got, want);
    }

    #[test]
    fn w1a_saturating_abs_i8x32_min_saturates_to_max() {
        let r = I8x32::splat(i8::MIN).saturating_abs().to_array();
        assert!(r.iter().all(|&x| x == i8::MAX), "got {r:?}");
    }

    #[test]
    fn w1a_from_i4_packed_u64_sign_extends() {
        // 0x0 → 0, 0xf → -1, 0x8 → -8, 0x7 → 7
        assert_eq!(I8x16::from_i4_packed_u64(0).lane_i8::<0>(), 0);
        assert_eq!(I8x16::from_i4_packed_u64(u64::MAX).lane_i8::<0>(), -1);
        assert_eq!(I8x16::from_i4_packed_u64(0x8888_8888_8888_8888).lane_i8::<3>(), -8);
        assert_eq!(I8x16::from_i4_packed_u64(0x7777_7777_7777_7777).lane_i8::<5>(), 7);
        // Mixed: low nibble 0x3 → 3, next nibble 0xC → -4.
        let mixed = I8x16::from_i4_packed_u64(0xC3);
        assert_eq!(mixed.lane_i8::<0>(), 3);
        assert_eq!(mixed.lane_i8::<1>(), -4);
    }

    #[test]
    fn w1a_u64x8_popcnt_and_xor_popcount() {
        let ones = U64x8::splat(u64::MAX);
        assert!(ones.popcnt().to_array().iter().all(|&x| x == 64));
        assert!(U64x8::splat(0).popcnt().to_array().iter().all(|&x| x == 0));
        // Hamming: all-bits-different → 64 × 8 = 512; same → 0.
        assert_eq!(U64x8::splat(u64::MAX).xor_popcount(U64x8::splat(0)), 512);
        let v = U64x8::splat(0xdead_beef_cafe_babe);
        assert_eq!(v.xor_popcount(v), 0);
    }

    #[test]
    fn w1a_gather_u16_in_bounds() {
        let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
        let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
        let got = U16x8::gather_u16(idx, &table).to_array();
        assert_eq!(got, [10, 30, 50, 70, 20, 40, 60, 80]);
    }
}
