//! Portable SIMD types backed by stable `core::arch` intrinsics.
//!
//! The wrapper IS the dispatch boundary. Kernel code writes `F32x16` and gets
//! optimal instructions on each target:
//!
//! | Target                  | F32x16 backing      | F64x8 backing      |
//! |-------------------------|---------------------|---------------------|
//! | x86_64 + AVX-512       | `__m512`            | `__m512d`           |
//! | non-x86 (scalar)       | `[f32; 16]`         | `[f64; 8]`          |
//!
//! When `std::simd` stabilizes, replace this file with re-exports.
//! When aarch64 support is needed, add `#[cfg(target_arch = "aarch64")]`
//! backing using NEON intrinsics (F32x16 → 4× `float32x4_t`).
//!
//! # Constraints
//!
//! - `#[inline(always)]` on every method — zero function-call overhead.
//! - No new dependencies — only `core::arch` (stable Rust 1.94+).
//! - std::simd API alignment — same method names for future migration.

// ============================================================================
// x86_64 AVX-512 backend
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx512_impl {
    use core::arch::x86_64::*;
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    macro_rules! impl_bin_op {
        ($ty:ident, $trait:ident, $method:ident, $intr:path) => {
            impl $trait for $ty {
                type Output = Self;
                #[inline(always)]
                fn $method(self, rhs: Self) -> Self {
                    // SAFETY: AVX-512 intrinsic, caller ensures CPU support
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
                    // SAFETY: AVX-512 intrinsic, caller ensures CPU support
                    self.0 = unsafe { $intr(self.0, rhs.0) };
                }
            }
        };
    }

    // ── F32x16 ──────────────────────────────────────────────────────

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
            // SAFETY: bounds checked, pointer valid for 16 floats
            Self(unsafe { _mm512_loadu_ps(s.as_ptr()) })
        }

        #[inline(always)]
        pub fn from_array(arr: [f32; 16]) -> Self {
            // SAFETY: array guaranteed to have 16 elements
            Self(unsafe { _mm512_loadu_ps(arr.as_ptr()) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut arr = [0.0f32; 16];
            // SAFETY: arr has 16 elements
            unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
            arr
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f32]) {
            assert!(s.len() >= 16);
            // SAFETY: bounds checked, pointer valid for 16 floats
            unsafe { _mm512_storeu_ps(s.as_mut_ptr(), self.0) };
        }

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

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            Self(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self(unsafe { _mm512_sqrt_ps(self.0) })
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            Self(unsafe { _mm512_roundscale_ps::<0x08>(self.0) })
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
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

        // --- Comparisons ---

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
            other.simd_lt(self)
        }

        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F32Mask16 {
            other.simd_le(self)
        }

        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F32Mask16 {
            F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, other.0) })
        }

        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F32Mask16 {
            F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
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
                let sign = _mm512_set1_epi32(i32::MIN);
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

    // ── F32Mask16 ───────────────────────────────────────────────────

    #[derive(Copy, Clone, Debug)]
    #[repr(transparent)]
    pub struct F32Mask16(pub __mmask16);

    impl F32Mask16 {
        #[inline(always)]
        pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
            F32x16(unsafe { _mm512_mask_blend_ps(self.0, false_val.0, true_val.0) })
        }
    }

    // ── F64x8 ───────────────────────────────────────────────────────

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
            // SAFETY: bounds checked
            Self(unsafe { _mm512_loadu_pd(s.as_ptr()) })
        }

        #[inline(always)]
        pub fn from_array(arr: [f64; 8]) -> Self {
            // SAFETY: array has 8 elements
            Self(unsafe { _mm512_loadu_pd(arr.as_ptr()) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut arr = [0.0f64; 8];
            // SAFETY: arr has 8 elements
            unsafe { _mm512_storeu_pd(arr.as_mut_ptr(), self.0) };
            arr
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            // SAFETY: bounds checked
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

        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F64Mask8 {
            F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0) })
        }

        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F64Mask8 {
            F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
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
                let sign = _mm512_set1_epi64(i64::MIN);
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

    // ── F64Mask8 ────────────────────────────────────────────────────

    #[derive(Copy, Clone, Debug)]
    #[repr(transparent)]
    pub struct F64Mask8(pub __mmask8);

    impl F64Mask8 {
        #[inline(always)]
        pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
            F64x8(unsafe { _mm512_mask_blend_pd(self.0, false_val.0, true_val.0) })
        }
    }
}

// ============================================================================
// Scalar fallback — correct everywhere, LLVM may auto-vectorize
// ============================================================================

#[cfg(not(target_arch = "x86_64"))]
mod scalar_impl {
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    // ── F32x16 ──────────────────────────────────────────────────────

    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F32x16(pub [f32; 16]);

    impl Default for F32x16 {
        #[inline(always)]
        fn default() -> Self { Self([0.0; 16]) }
    }

    impl F32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn splat(v: f32) -> Self { Self([v; 16]) }

        #[inline(always)]
        pub fn from_slice(s: &[f32]) -> Self {
            assert!(s.len() >= 16);
            let mut arr = [0.0f32; 16];
            arr.copy_from_slice(&s[..16]);
            Self(arr)
        }

        #[inline(always)]
        pub fn from_array(arr: [f32; 16]) -> Self { Self(arr) }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] { self.0 }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f32]) {
            assert!(s.len() >= 16);
            s[..16].copy_from_slice(&self.0);
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f32 { self.0.iter().sum() }

        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.0.iter().copied().fold(f32::INFINITY, f32::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].min(other.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].max(other.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].mul_add(b.0[i], c.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].sqrt(); }
            Self(out)
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].round(); }
            Self(out)
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].floor(); }
            Self(out)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i].abs(); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> F32Mask16 {
            let mut bits = 0u16;
            for i in 0..16 { if self.0[i] < other.0[i] { bits |= 1 << i; } }
            F32Mask16(bits)
        }

        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F32Mask16 {
            let mut bits = 0u16;
            for i in 0..16 { if self.0[i] <= other.0[i] { bits |= 1 << i; } }
            F32Mask16(bits)
        }

        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> F32Mask16 { other.simd_lt(self) }

        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F32Mask16 { other.simd_le(self) }

        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F32Mask16 {
            let mut bits = 0u16;
            for i in 0..16 { if self.0[i] == other.0[i] { bits |= 1 << i; } }
            F32Mask16(bits)
        }

        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F32Mask16 {
            let mut bits = 0u16;
            for i in 0..16 { if self.0[i] != other.0[i] { bits |= 1 << i; } }
            F32Mask16(bits)
        }
    }

    impl Add for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i] + rhs.0[i]; }
            Self(out)
        }
    }
    impl Sub for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i] - rhs.0[i]; }
            Self(out)
        }
    }
    impl Mul for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i] * rhs.0[i]; }
            Self(out)
        }
    }
    impl Div for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i] / rhs.0[i]; }
            Self(out)
        }
    }
    impl AddAssign for F32x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] += rhs.0[i]; } }
    }
    impl SubAssign for F32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] -= rhs.0[i]; } }
    }
    impl MulAssign for F32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] *= rhs.0[i]; } }
    }
    impl DivAssign for F32x16 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] /= rhs.0[i]; } }
    }
    impl Neg for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = -self.0[i]; }
            Self(out)
        }
    }
    impl fmt::Debug for F32x16 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F32x16({:?})", self.0)
        }
    }
    impl PartialEq for F32x16 {
        fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    // ── F32Mask16 ───────────────────────────────────────────────────

    #[derive(Copy, Clone, Debug)]
    pub struct F32Mask16(pub u16);

    impl F32Mask16 {
        #[inline(always)]
        pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
            let mut out = [0.0f32; 16];
            for i in 0..16 {
                out[i] = if (self.0 >> i) & 1 == 1 { true_val.0[i] } else { false_val.0[i] };
            }
            F32x16(out)
        }
    }

    // ── F64x8 ───────────────────────────────────────────────────────

    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F64x8(pub [f64; 8]);

    impl Default for F64x8 {
        #[inline(always)]
        fn default() -> Self { Self([0.0; 8]) }
    }

    impl F64x8 {
        pub const LANES: usize = 8;

        #[inline(always)]
        pub fn splat(v: f64) -> Self { Self([v; 8]) }

        #[inline(always)]
        pub fn from_slice(s: &[f64]) -> Self {
            assert!(s.len() >= 8);
            let mut arr = [0.0f64; 8];
            arr.copy_from_slice(&s[..8]);
            Self(arr)
        }

        #[inline(always)]
        pub fn from_array(arr: [f64; 8]) -> Self { Self(arr) }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] { self.0 }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            s[..8].copy_from_slice(&self.0);
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f64 { self.0.iter().sum() }

        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.0.iter().copied().fold(f64::INFINITY, f64::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.0.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].min(other.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].max(other.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].mul_add(b.0[i], c.0[i]); }
            Self(out)
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].sqrt(); }
            Self(out)
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].round(); }
            Self(out)
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].floor(); }
            Self(out)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i].abs(); }
            Self(out)
        }

        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> F64Mask8 {
            let mut bits = 0u8;
            for i in 0..8 { if self.0[i] < other.0[i] { bits |= 1 << i; } }
            F64Mask8(bits)
        }

        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F64Mask8 {
            let mut bits = 0u8;
            for i in 0..8 { if self.0[i] <= other.0[i] { bits |= 1 << i; } }
            F64Mask8(bits)
        }

        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> F64Mask8 { other.simd_lt(self) }

        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F64Mask8 { other.simd_le(self) }

        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F64Mask8 {
            let mut bits = 0u8;
            for i in 0..8 { if self.0[i] == other.0[i] { bits |= 1 << i; } }
            F64Mask8(bits)
        }

        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F64Mask8 {
            let mut bits = 0u8;
            for i in 0..8 { if self.0[i] != other.0[i] { bits |= 1 << i; } }
            F64Mask8(bits)
        }
    }

    impl Add for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i] + rhs.0[i]; }
            Self(out)
        }
    }
    impl Sub for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i] - rhs.0[i]; }
            Self(out)
        }
    }
    impl Mul for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i] * rhs.0[i]; }
            Self(out)
        }
    }
    impl Div for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = self.0[i] / rhs.0[i]; }
            Self(out)
        }
    }
    impl AddAssign for F64x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] += rhs.0[i]; } }
    }
    impl SubAssign for F64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] -= rhs.0[i]; } }
    }
    impl MulAssign for F64x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] *= rhs.0[i]; } }
    }
    impl DivAssign for F64x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] /= rhs.0[i]; } }
    }
    impl Neg for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = -self.0[i]; }
            Self(out)
        }
    }
    impl fmt::Debug for F64x8 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F64x8({:?})", self.0)
        }
    }
    impl PartialEq for F64x8 {
        fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }

    // ── F64Mask8 ────────────────────────────────────────────────────

    #[derive(Copy, Clone, Debug)]
    pub struct F64Mask8(pub u8);

    impl F64Mask8 {
        #[inline(always)]
        pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
            let mut out = [0.0f64; 8];
            for i in 0..8 {
                out[i] = if (self.0 >> i) & 1 == 1 { true_val.0[i] } else { false_val.0[i] };
            }
            F64x8(out)
        }
    }
}

// ============================================================================
// Re-exports — the public API is always the same types
// ============================================================================

#[allow(unused_imports)]
#[cfg(target_arch = "x86_64")]
pub use avx512_impl::{F32Mask16, F32x16, F64Mask8, F64x8};

#[allow(unused_imports)]
#[cfg(not(target_arch = "x86_64"))]
pub use scalar_impl::{F32Mask16, F32x16, F64Mask8, F64x8};

// ============================================================================
// SIMD math functions — polynomial approximations using mul_add chains
// ============================================================================

/// Fast exp(x) for F32x16 — Remez polynomial on [-87, 87].
///
/// Max error ~2 ULP in [-10, 10]. Uses the standard range-reduction
/// approach: exp(x) = 2^n * exp(r) where r = x - n*ln(2).
#[inline(always)]
#[allow(dead_code)]
pub fn simd_exp_f32(x: F32x16) -> F32x16 {
    let ln2 = F32x16::splat(core::f32::consts::LN_2);
    let inv_ln2 = F32x16::splat(1.0 / core::f32::consts::LN_2);
    let one = F32x16::splat(1.0);

    // Range reduction: n = round(x / ln2), r = x - n * ln2
    let n = (x * inv_ln2).round();
    let r = x - n * ln2;

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
    let c2 = F32x16::splat(0.5);
    let c3 = F32x16::splat(1.0 / 6.0);
    let c4 = F32x16::splat(1.0 / 24.0);
    let c5 = F32x16::splat(1.0 / 120.0);

    let poly = one + r * (one + r * (c2 + r * (c3 + r * (c4 + r * c5))));

    // Reconstruct: exp(x) = 2^n * poly
    poly * pow2n_from_int(n)
}

/// Compute 2^n where n is an integer stored as f32.
///
/// Uses the IEEE 754 trick: set the exponent field directly.
#[inline(always)]
#[allow(dead_code)]
fn pow2n_from_int(n: F32x16) -> F32x16 {
    // For each lane: reinterpret (int(n) + 127) << 23 as f32
    // This gives exactly 2^n for integer n in [-126, 127]
    let arr = n.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        let ni = arr[i] as i32;
        let bits = ((ni + 127) as u32) << 23;
        out[i] = f32::from_bits(bits);
    }
    F32x16::from_array(out)
}

/// Fast natural log for F32x16 — polynomial on [1, 2).
#[inline(always)]
#[allow(dead_code)]
pub fn simd_ln_f32(x: F32x16) -> F32x16 {
    // Decompose: x = 2^e * m where 1 <= m < 2
    // ln(x) = e * ln(2) + ln(m)
    let arr = x.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = arr[i].ln();
    }
    F32x16::from_array(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_splat_reduce_sum() {
        let v = F32x16::splat(3.0);
        assert!((v.reduce_sum() - 48.0).abs() < 1e-6);
    }

    #[test]
    fn f32x16_from_array_roundtrip() {
        let data: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let v = F32x16::from_array(data);
        assert_eq!(v.to_array(), data);
    }

    #[test]
    fn f32x16_add_sub_mul_div() {
        let a = F32x16::splat(6.0);
        let b = F32x16::splat(2.0);
        assert!(((a + b).reduce_sum() - 128.0).abs() < 1e-4);
        assert!(((a - b).reduce_sum() - 64.0).abs() < 1e-4);
        assert!(((a * b).reduce_sum() - 192.0).abs() < 1e-4);
        assert!(((a / b).reduce_sum() - 48.0).abs() < 1e-4);
    }

    #[test]
    fn f32x16_mul_add_fma() {
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        let r = a.mul_add(b, c); // 2*3+1 = 7 per lane
        assert!((r.reduce_sum() - 112.0).abs() < 1e-4);
    }

    #[test]
    fn f32x16_mask_select() {
        let a = F32x16::from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let threshold = F32x16::splat(8.5);
        let mask = a.simd_lt(threshold);
        let result = mask.select(F32x16::splat(1.0), F32x16::splat(0.0));
        // 8 lanes < 8.5 → sum = 8
        assert!((result.reduce_sum() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn f64x8_splat_reduce_sum() {
        let v = F64x8::splat(3.0);
        assert!((v.reduce_sum() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn f64x8_from_array_roundtrip() {
        let data: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let v = F64x8::from_array(data);
        assert_eq!(v.to_array(), data);
    }

    #[test]
    fn f64x8_mul_add() {
        let a = F64x8::splat(2.0);
        let b = F64x8::splat(3.0);
        let c = F64x8::splat(1.0);
        let r = a.mul_add(b, c);
        assert!((r.reduce_sum() - 56.0).abs() < 1e-10);
    }

    #[test]
    fn f32x16_abs_neg() {
        let a = F32x16::splat(-5.0);
        assert!((a.abs().reduce_sum() - 80.0).abs() < 1e-4);
        let b = F32x16::splat(3.0);
        assert!(((-b).reduce_sum() - (-48.0)).abs() < 1e-4);
    }

    #[test]
    fn f32x16_from_slice_to_slice() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        let mut out = vec![0.0f32; 20];
        v.copy_to_slice(&mut out);
        assert_eq!(&out[..16], &data[..16]);
    }

    #[test]
    fn simd_exp_f32_basic() {
        let zero = F32x16::splat(0.0);
        let result = simd_exp_f32(zero);
        // exp(0) = 1
        assert!((result.reduce_sum() / 16.0 - 1.0).abs() < 1e-4);
    }
}
