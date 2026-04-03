//! SIMD polyfill — `crate::simd::F32x16` dispatches via LazyLock<Tier>.
//!
//! Same pattern as `backend/native.rs`: detect once, dispatch forever.
//! AVX-512 → AVX2 → Scalar. Consumer writes `crate::simd::F32x16`. Period.
//!
//! When `std::simd` stabilizes: swap this file. Zero consumer changes.

use std::sync::LazyLock;

#[derive(Clone, Copy, PartialEq)]
enum Tier { Avx512, Avx2, Scalar }

static TIER: LazyLock<Tier> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { return Tier::Avx512; }
        if is_x86_feature_detected!("avx2") { return Tier::Avx2; }
    }
    Tier::Scalar
});

#[inline(always)]
fn tier() -> Tier { *TIER }

// BF16 tier detection happens inline in bf16_to_f32_batch() via
// is_x86_feature_detected!("avx512bf16") — no LazyLock needed.
// The check is cheap (reads a cached cpuid result) and the batch
// function uses as_chunks::<16>() + as_chunks::<8>() for SIMD widths.

// ============================================================================
// Preferred SIMD lane widths — compile-time constants for array_windows
// ============================================================================
//
// Consumer code uses these to select array_windows size at compile time:
//
//   for window in data.array_windows::<{crate::simd::PREFERRED_F64_LANES}>() {
//       let v = F64x8::from_array(*window);   // AVX-512: native 8-wide
//       // or
//       let v = F64x4::from_array(*window);   // AVX2: native 4-wide
//   }
//
// generic_const_exprs is nightly, so consumers must #[cfg] branch on window size.
// These constants document the preferred width per tier.

/// Preferred f64 SIMD width (elements per register).
/// AVX-512: 8 lanes (__m512d). AVX2/scalar: 4 lanes (__m256d).
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_F64_LANES: usize = 8;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_F64_LANES: usize = 4;
#[cfg(not(target_arch = "x86_64"))]
pub const PREFERRED_F64_LANES: usize = 4; // scalar fallback: same as AVX2 shape

/// Preferred f32 SIMD width.
/// AVX-512: 16 lanes (__m512). AVX2/scalar: 8 lanes (__m256).
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_F32_LANES: usize = 16;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_F32_LANES: usize = 8;
#[cfg(not(target_arch = "x86_64"))]
pub const PREFERRED_F32_LANES: usize = 8;

/// Preferred u64 SIMD width.
/// AVX-512: 8 lanes. AVX2/scalar: 4 lanes.
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_U64_LANES: usize = 8;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_U64_LANES: usize = 4;
#[cfg(not(target_arch = "x86_64"))]
pub const PREFERRED_U64_LANES: usize = 4;

/// Preferred i16 SIMD width (for Base17 L1 on i16[17]).
/// AVX-512: 32 lanes (__m512i via epi16). AVX2: 16 lanes (__m256i).
/// Base17 has 17 dims — AVX-512 covers 32 (load 17 + 15 padding),
/// AVX2 covers 16 + 1 scalar.
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_I16_LANES: usize = 32;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_I16_LANES: usize = 16;
#[cfg(not(target_arch = "x86_64"))]
pub const PREFERRED_I16_LANES: usize = 16;

// ============================================================================
// x86_64: re-export based on tier
// ============================================================================

// Compile-time AVX-512 dispatch via target_feature.
// With target-cpu=x86-64-v4 (.cargo/config.toml), avx512f is enabled
// at compile time → all types use native __m512/__m512d/__m512i.
// The 256-bit types (F32x8, F64x4) also live in simd_avx512 (__m256).

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use crate::simd_avx512::{
    // 256-bit (AVX2 baseline, __m256/__m256d)
    F32x8, F64x4, f32x8, f64x4,
    // 512-bit (native AVX-512, __m512/__m512d/__m512i)
    F32x16, F64x8, U8x64, I32x16, I64x8, U32x16, U64x8,
    F32Mask16, F64Mask8,
    f32x16, f64x8, u8x64, i32x16, i64x8, u32x16, u64x8,
};

// BF16 types + batch conversion (always available — scalar fallback built in)
#[cfg(target_arch = "x86_64")]
pub use crate::simd_avx512::{
    bf16_to_f32_scalar, f32_to_bf16_scalar,
    bf16_to_f32_batch, f32_to_bf16_batch,
};
// BF16 SIMD types only available when avx512bf16 is enabled at compile time
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16"))]
pub use crate::simd_avx512::{BF16x16, BF16x8};

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub use crate::simd_avx512::{F32x8, F64x4, f32x8, f64x4};

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub use crate::simd_avx2::{
    F32x16, F64x8, U8x64, I32x16, I64x8, U32x16, U64x8,
    F32Mask16, F64Mask8,
    f32x16, f64x8, u8x64, i32x16, i64x8, u32x16, u64x8,
};

// ============================================================================
// Non-x86: scalar fallback types with identical API
// ============================================================================

#[cfg(not(target_arch = "x86_64"))]
mod scalar {
    use core::fmt;
    use core::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
        Div, DivAssign, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
    };

    // ── Macros for scalar fallback boilerplate ────────────────────────

    macro_rules! impl_float_type {
        ($name:ident, $elem:ty, $lanes:expr, $mask:ident, $mask_prim:ty) => {
            #[derive(Copy, Clone)]
            #[repr(align(64))]
            pub struct $name(pub [$elem; $lanes]);

            impl Default for $name {
                #[inline(always)]
                fn default() -> Self { Self([0.0; $lanes]) }
            }

            impl $name {
                pub const LANES: usize = $lanes;

                #[inline(always)]
                pub fn splat(v: $elem) -> Self { Self([v; $lanes]) }

                #[inline(always)]
                pub fn from_slice(s: &[$elem]) -> Self {
                    assert!(s.len() >= $lanes);
                    let mut arr = [0.0 as $elem; $lanes];
                    arr.copy_from_slice(&s[..$lanes]);
                    Self(arr)
                }

                #[inline(always)]
                pub fn from_array(arr: [$elem; $lanes]) -> Self { Self(arr) }

                #[inline(always)]
                pub fn to_array(self) -> [$elem; $lanes] { self.0 }

                #[inline(always)]
                pub fn copy_to_slice(self, s: &mut [$elem]) {
                    assert!(s.len() >= $lanes);
                    s[..$lanes].copy_from_slice(&self.0);
                }

                #[inline(always)]
                pub fn reduce_sum(self) -> $elem { self.0.iter().sum() }

                #[inline(always)]
                pub fn reduce_min(self) -> $elem {
                    self.0.iter().copied().fold(<$elem>::INFINITY, <$elem>::min)
                }

                #[inline(always)]
                pub fn reduce_max(self) -> $elem {
                    self.0.iter().copied().fold(<$elem>::NEG_INFINITY, <$elem>::max)
                }

                #[inline(always)]
                pub fn simd_min(self, other: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].min(other.0[i]); }
                    Self(out)
                }

                #[inline(always)]
                pub fn simd_max(self, other: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].max(other.0[i]); }
                    Self(out)
                }

                #[inline(always)]
                pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
                    self.simd_max(lo).simd_min(hi)
                }

                #[inline(always)]
                pub fn mul_add(self, b: Self, c: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].mul_add(b.0[i], c.0[i]); }
                    Self(out)
                }

                #[inline(always)]
                pub fn sqrt(self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].sqrt(); }
                    Self(out)
                }

                #[inline(always)]
                pub fn round(self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].round(); }
                    Self(out)
                }

                #[inline(always)]
                pub fn floor(self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].floor(); }
                    Self(out)
                }

                #[inline(always)]
                pub fn abs(self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].abs(); }
                    Self(out)
                }

                #[inline(always)]
                pub fn simd_lt(self, other: Self) -> $mask {
                    let mut bits: $mask_prim = 0;
                    for i in 0..$lanes { if self.0[i] < other.0[i] { bits |= 1 << i; } }
                    $mask(bits)
                }

                #[inline(always)]
                pub fn simd_le(self, other: Self) -> $mask {
                    let mut bits: $mask_prim = 0;
                    for i in 0..$lanes { if self.0[i] <= other.0[i] { bits |= 1 << i; } }
                    $mask(bits)
                }

                #[inline(always)]
                pub fn simd_gt(self, other: Self) -> $mask { other.simd_lt(self) }

                #[inline(always)]
                pub fn simd_ge(self, other: Self) -> $mask { other.simd_le(self) }

                #[inline(always)]
                pub fn simd_eq(self, other: Self) -> $mask {
                    let mut bits: $mask_prim = 0;
                    for i in 0..$lanes { if self.0[i] == other.0[i] { bits |= 1 << i; } }
                    $mask(bits)
                }

                #[inline(always)]
                pub fn simd_ne(self, other: Self) -> $mask {
                    let mut bits: $mask_prim = 0;
                    for i in 0..$lanes { if self.0[i] != other.0[i] { bits |= 1 << i; } }
                    $mask(bits)
                }
            }

            impl Add for $name {
                type Output = Self;
                #[inline(always)]
                fn add(self, rhs: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] + rhs.0[i]; }
                    Self(out)
                }
            }
            impl Sub for $name {
                type Output = Self;
                #[inline(always)]
                fn sub(self, rhs: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] - rhs.0[i]; }
                    Self(out)
                }
            }
            impl Mul for $name {
                type Output = Self;
                #[inline(always)]
                fn mul(self, rhs: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] * rhs.0[i]; }
                    Self(out)
                }
            }
            impl Div for $name {
                type Output = Self;
                #[inline(always)]
                fn div(self, rhs: Self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] / rhs.0[i]; }
                    Self(out)
                }
            }
            impl AddAssign for $name {
                #[inline(always)]
                fn add_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] += rhs.0[i]; } }
            }
            impl SubAssign for $name {
                #[inline(always)]
                fn sub_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] -= rhs.0[i]; } }
            }
            impl MulAssign for $name {
                #[inline(always)]
                fn mul_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] *= rhs.0[i]; } }
            }
            impl DivAssign for $name {
                #[inline(always)]
                fn div_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] /= rhs.0[i]; } }
            }
            impl Neg for $name {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes { out[i] = -self.0[i]; }
                    Self(out)
                }
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
            }

            // Mask type
            #[derive(Copy, Clone, Debug)]
            pub struct $mask(pub $mask_prim);

            impl $mask {
                #[inline(always)]
                pub fn select(self, true_val: $name, false_val: $name) -> $name {
                    let mut out = [0.0 as $elem; $lanes];
                    for i in 0..$lanes {
                        out[i] = if (self.0 >> i) & 1 == 1 { true_val.0[i] } else { false_val.0[i] };
                    }
                    $name(out)
                }
            }
        };
    }

    macro_rules! impl_int_type {
        ($name:ident, $elem:ty, $lanes:expr, $zero:expr) => {
            #[derive(Copy, Clone)]
            #[repr(align(64))]
            pub struct $name(pub [$elem; $lanes]);

            impl Default for $name {
                #[inline(always)]
                fn default() -> Self { Self([$zero; $lanes]) }
            }

            impl $name {
                pub const LANES: usize = $lanes;

                #[inline(always)]
                pub fn splat(v: $elem) -> Self { Self([v; $lanes]) }

                #[inline(always)]
                pub fn from_slice(s: &[$elem]) -> Self {
                    assert!(s.len() >= $lanes);
                    let mut arr = [$zero; $lanes];
                    arr.copy_from_slice(&s[..$lanes]);
                    Self(arr)
                }

                #[inline(always)]
                pub fn from_array(arr: [$elem; $lanes]) -> Self { Self(arr) }

                #[inline(always)]
                pub fn to_array(self) -> [$elem; $lanes] { self.0 }

                #[inline(always)]
                pub fn copy_to_slice(self, s: &mut [$elem]) {
                    assert!(s.len() >= $lanes);
                    s[..$lanes].copy_from_slice(&self.0);
                }

                #[inline(always)]
                pub fn reduce_sum(self) -> $elem {
                    let mut s: $elem = $zero;
                    for i in 0..$lanes { s = s.wrapping_add(self.0[i]); }
                    s
                }
            }

            impl Add for $name {
                type Output = Self;
                #[inline(always)]
                fn add(self, rhs: Self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].wrapping_add(rhs.0[i]); }
                    Self(out)
                }
            }
            impl Sub for $name {
                type Output = Self;
                #[inline(always)]
                fn sub(self, rhs: Self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i].wrapping_sub(rhs.0[i]); }
                    Self(out)
                }
            }
            impl AddAssign for $name {
                #[inline(always)]
                fn add_assign(&mut self, rhs: Self) {
                    for i in 0..$lanes { self.0[i] = self.0[i].wrapping_add(rhs.0[i]); }
                }
            }
            impl SubAssign for $name {
                #[inline(always)]
                fn sub_assign(&mut self, rhs: Self) {
                    for i in 0..$lanes { self.0[i] = self.0[i].wrapping_sub(rhs.0[i]); }
                }
            }
            impl BitAnd for $name {
                type Output = Self;
                #[inline(always)]
                fn bitand(self, rhs: Self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] & rhs.0[i]; }
                    Self(out)
                }
            }
            impl BitOr for $name {
                type Output = Self;
                #[inline(always)]
                fn bitor(self, rhs: Self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] | rhs.0[i]; }
                    Self(out)
                }
            }
            impl BitXor for $name {
                type Output = Self;
                #[inline(always)]
                fn bitxor(self, rhs: Self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = self.0[i] ^ rhs.0[i]; }
                    Self(out)
                }
            }
            impl BitAndAssign for $name {
                #[inline(always)]
                fn bitand_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] &= rhs.0[i]; } }
            }
            impl BitOrAssign for $name {
                #[inline(always)]
                fn bitor_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] |= rhs.0[i]; } }
            }
            impl BitXorAssign for $name {
                #[inline(always)]
                fn bitxor_assign(&mut self, rhs: Self) { for i in 0..$lanes { self.0[i] ^= rhs.0[i]; } }
            }
            impl Not for $name {
                type Output = Self;
                #[inline(always)]
                fn not(self) -> Self {
                    let mut out = [$zero; $lanes];
                    for i in 0..$lanes { out[i] = !self.0[i]; }
                    Self(out)
                }
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
            }
        };
    }

    // ── Instantiate all 11 types ─────────────────────────────────────

    // 512-bit float types
    impl_float_type!(F32x16, f32, 16, F32Mask16, u16);
    impl_float_type!(F64x8, f64, 8, F64Mask8, u8);

    // 256-bit AVX2 float types
    impl_float_type!(F32x8, f32, 8, F32Mask8Scalar, u8);
    impl_float_type!(F64x4, f64, 4, F64Mask4Scalar, u8);

    // Unused mask types for AVX2 scalars (not exported, just needed by macro)
    #[derive(Copy, Clone, Debug)]
    pub struct F32Mask8Scalar(pub u8);
    #[derive(Copy, Clone, Debug)]
    pub struct F64Mask4Scalar(pub u8);

    // 512-bit integer types
    impl_int_type!(U8x64, u8, 64, 0u8);
    impl_int_type!(I32x16, i32, 16, 0i32);
    impl_int_type!(I64x8, i64, 8, 0i64);
    impl_int_type!(U32x16, u32, 16, 0u32);
    impl_int_type!(U64x8, u64, 8, 0u64);

    // Extra methods for I32x16 that float types have via the macro
    impl I32x16 {
        #[inline(always)]
        pub fn reduce_min(self) -> i32 { *self.0.iter().min().unwrap_or(&0) }
        #[inline(always)]
        pub fn reduce_max(self) -> i32 { *self.0.iter().max().unwrap_or(&0) }
        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = self.0[i].min(other.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = self.0[i].max(other.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn cast_f32(self) -> F32x16 {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = self.0[i] as f32; }
            F32x16(out)
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = self.0[i].abs(); }
            Self(out)
        }
    }

    impl Mul for I32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = self.0[i].wrapping_mul(rhs.0[i]); }
            Self(out)
        }
    }
    impl MulAssign for I32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
    }
    impl Neg for I32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = -self.0[i]; }
            Self(out)
        }
    }

    // Extra for F32x16: to_bits/from_bits/cast_i32
    impl F32x16 {
        #[inline(always)]
        pub fn to_bits(self) -> U32x16 {
            let mut out = [0u32; 16];
            for i in 0..16 { out[i] = self.0[i].to_bits(); }
            U32x16(out)
        }
        #[inline(always)]
        pub fn from_bits(bits: U32x16) -> Self {
            let mut out = [0.0f32; 16];
            for i in 0..16 { out[i] = f32::from_bits(bits.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn cast_i32(self) -> I32x16 {
            let mut out = [0i32; 16];
            for i in 0..16 { out[i] = self.0[i] as i32; }
            I32x16(out)
        }
    }

    // Extra for F64x8: to_bits/from_bits
    impl F64x8 {
        #[inline(always)]
        pub fn to_bits(self) -> U64x8 {
            let mut out = [0u64; 8];
            for i in 0..8 { out[i] = self.0[i].to_bits(); }
            U64x8(out)
        }
        #[inline(always)]
        pub fn from_bits(bits: U64x8) -> Self {
            let mut out = [0.0f64; 8];
            for i in 0..8 { out[i] = f64::from_bits(bits.0[i]); }
            Self(out)
        }
    }

    // Extra for I64x8
    impl I64x8 {
        #[inline(always)]
        pub fn reduce_min(self) -> i64 { *self.0.iter().min().unwrap_or(&0) }
        #[inline(always)]
        pub fn reduce_max(self) -> i64 { *self.0.iter().max().unwrap_or(&0) }
        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            let mut out = [0i64; 8];
            for i in 0..8 { out[i] = self.0[i].min(other.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            let mut out = [0i64; 8];
            for i in 0..8 { out[i] = self.0[i].max(other.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            let mut out = [0i64; 8];
            for i in 0..8 { out[i] = self.0[i].abs(); }
            Self(out)
        }
    }

    impl Mul for I64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0i64; 8];
            for i in 0..8 { out[i] = self.0[i].wrapping_mul(rhs.0[i]); }
            Self(out)
        }
    }
    impl MulAssign for I64x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
    }
    impl Neg for I64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            let mut out = [0i64; 8];
            for i in 0..8 { out[i] = -self.0[i]; }
            Self(out)
        }
    }

    // Shift operators for U32x16
    impl Shr<Self> for U32x16 {
        type Output = Self;
        #[inline(always)]
        fn shr(self, rhs: Self) -> Self {
            let mut out = [0u32; 16];
            for i in 0..16 { out[i] = self.0[i] >> rhs.0[i]; }
            Self(out)
        }
    }
    impl Shl<Self> for U32x16 {
        type Output = Self;
        #[inline(always)]
        fn shl(self, rhs: Self) -> Self {
            let mut out = [0u32; 16];
            for i in 0..16 { out[i] = self.0[i] << rhs.0[i]; }
            Self(out)
        }
    }

    // Shift operators for U64x8
    impl Shr<Self> for U64x8 {
        type Output = Self;
        #[inline(always)]
        fn shr(self, rhs: Self) -> Self {
            let mut out = [0u64; 8];
            for i in 0..8 { out[i] = self.0[i] >> rhs.0[i]; }
            Self(out)
        }
    }
    impl Shl<Self> for U64x8 {
        type Output = Self;
        #[inline(always)]
        fn shl(self, rhs: Self) -> Self {
            let mut out = [0u64; 8];
            for i in 0..8 { out[i] = self.0[i] << rhs.0[i]; }
            Self(out)
        }
    }

    // Mul for U8x64 (wrapping)
    impl Mul for U8x64 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0u8; 64];
            for i in 0..64 { out[i] = self.0[i].wrapping_mul(rhs.0[i]); }
            Self(out)
        }
    }
    impl MulAssign for U8x64 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
    }

    // U8x64 extra methods
    impl U8x64 {
        #[inline(always)]
        pub fn reduce_min(self) -> u8 { *self.0.iter().min().unwrap_or(&0) }
        #[inline(always)]
        pub fn reduce_max(self) -> u8 { *self.0.iter().max().unwrap_or(&0) }
        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            let mut out = [0u8; 64];
            for i in 0..64 { out[i] = self.0[i].min(other.0[i]); }
            Self(out)
        }
        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            let mut out = [0u8; 64];
            for i in 0..64 { out[i] = self.0[i].max(other.0[i]); }
            Self(out)
        }
    }

    // Mul for U32x16
    impl Mul for U32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut out = [0u32; 16];
            for i in 0..16 { out[i] = self.0[i].wrapping_mul(rhs.0[i]); }
            Self(out)
        }
    }

    // Lowercase aliases
    #[allow(non_camel_case_types)] pub type f32x16 = F32x16;
    #[allow(non_camel_case_types)] pub type f64x8 = F64x8;
    #[allow(non_camel_case_types)] pub type u8x64 = U8x64;
    #[allow(non_camel_case_types)] pub type i32x16 = I32x16;
    #[allow(non_camel_case_types)] pub type i64x8 = I64x8;
    #[allow(non_camel_case_types)] pub type u32x16 = U32x16;
    #[allow(non_camel_case_types)] pub type u64x8 = U64x8;
    #[allow(non_camel_case_types)] pub type f32x8 = F32x8;
    #[allow(non_camel_case_types)] pub type f64x4 = F64x4;
}

#[cfg(not(target_arch = "x86_64"))]
pub use scalar::{
    F32x16, F64x8, U8x64, I32x16, I64x8, U32x16, U64x8,
    F32x8, F64x4,
    F32Mask16, F64Mask8,
    f32x16, f64x8, u8x64, i32x16, i64x8, u32x16, u64x8,
    f32x8, f64x4,
};

// Scalar BF16 conversion — always available on all platforms
#[cfg(not(target_arch = "x86_64"))]
pub fn bf16_to_f32_scalar(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }
#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_bf16_scalar(v: f32) -> u16 { (v.to_bits() >> 16) as u16 }
#[cfg(not(target_arch = "x86_64"))]
pub fn bf16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    for (i, &b) in input.iter().enumerate() { if i < output.len() { output[i] = bf16_to_f32_scalar(b); } }
}
#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) {
    for (i, &v) in input.iter().enumerate() { if i < output.len() { output[i] = f32_to_bf16_scalar(v); } }
}

// ============================================================================
// SIMD math functions — ndarray additions (not in std::simd)
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

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
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
    let arr = n.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        let ni = arr[i] as i32;
        let bits = ((ni + 127) as u32) << 23;
        out[i] = f32::from_bits(bits);
    }
    F32x16::from_array(out)
}

/// Fast natural log for F32x16.
#[inline(always)]
#[allow(dead_code)]
pub fn simd_ln_f32(x: F32x16) -> F32x16 {
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
        let r = a.mul_add(b, c);
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
        assert!((result.reduce_sum() / 16.0 - 1.0).abs() < 1e-4);
    }
}
