//! Pure-Rust scalar fallback backend for `crate::simd::*`.
//!
//! Selected by `src/simd.rs` dispatch on non-x86_64 / non-aarch64
//! targets (wasm32, riscv, thumbv6m, etc.) when `feature =
//! "nightly-simd"` is OFF. Mirrors the API of `simd_avx512`,
//! `simd_avx2`, and `simd_neon::aarch64_simd` so consumer code reading
//! `use crate::simd::F32x16` compiles and runs uniformly across all
//! supported targets.
//!
//! Storage is plain `[$elem; $lanes]` arrays aligned to 64 bytes; the
//! arithmetic is loop-unrolled scalar Rust. No SIMD intrinsics — the
//! point is a correct fallback, not performance.
//!
//! The file was extracted from `simd.rs` in Phase 4 of the integration
//! plan in `.claude/knowledge/simd-dispatch-architecture.md` (split out
//! 1271 inline lines so the dispatcher reads as a re-export catalog
//! rather than 1.6k LoC of macro expansions).

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign,
    Neg, Not, Shl, Shr, Sub, SubAssign,
};

// ── Macros for scalar fallback boilerplate ────────────────────────

macro_rules! impl_float_type {
    ($name:ident, $elem:ty, $lanes:expr, $mask:ident, $mask_prim:ty) => {
        #[derive(Copy, Clone)]
        #[repr(align(64))]
        pub struct $name(pub [$elem; $lanes]);

        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self([0.0; $lanes])
            }
        }

        impl $name {
            pub const LANES: usize = $lanes;

            #[inline(always)]
            pub fn splat(v: $elem) -> Self {
                Self([v; $lanes])
            }

            #[inline(always)]
            pub fn from_slice(s: &[$elem]) -> Self {
                assert!(s.len() >= $lanes);
                let mut arr = [0.0 as $elem; $lanes];
                arr.copy_from_slice(&s[..$lanes]);
                Self(arr)
            }

            #[inline(always)]
            pub fn from_array(arr: [$elem; $lanes]) -> Self {
                Self(arr)
            }

            #[inline(always)]
            pub fn to_array(self) -> [$elem; $lanes] {
                self.0
            }

            #[inline(always)]
            pub fn copy_to_slice(self, s: &mut [$elem]) {
                assert!(s.len() >= $lanes);
                s[..$lanes].copy_from_slice(&self.0);
            }

            #[inline(always)]
            pub fn reduce_sum(self) -> $elem {
                self.0.iter().sum()
            }

            #[inline(always)]
            pub fn reduce_min(self) -> $elem {
                self.0.iter().copied().fold(<$elem>::INFINITY, <$elem>::min)
            }

            #[inline(always)]
            pub fn reduce_max(self) -> $elem {
                self.0
                    .iter()
                    .copied()
                    .fold(<$elem>::NEG_INFINITY, <$elem>::max)
            }

            #[inline(always)]
            pub fn simd_min(self, other: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].min(other.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_max(self, other: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].max(other.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
                self.simd_max(lo).simd_min(hi)
            }

            #[inline(always)]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].mul_add(b.0[i], c.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn sqrt(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].sqrt();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn round(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].round();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn floor(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].floor();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn abs(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].abs();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] < other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_le(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] <= other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> $mask {
                other.simd_lt(self)
            }

            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> $mask {
                other.simd_le(self)
            }

            #[inline(always)]
            pub fn simd_eq(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] == other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_ne(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] != other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] + rhs.0[i];
                }
                Self(out)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] - rhs.0[i];
                }
                Self(out)
            }
        }
        impl Mul for $name {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] * rhs.0[i];
                }
                Self(out)
            }
        }
        impl Div for $name {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] / rhs.0[i];
                }
                Self(out)
            }
        }
        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] += rhs.0[i];
                }
            }
        }
        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] -= rhs.0[i];
                }
            }
        }
        impl MulAssign for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] *= rhs.0[i];
                }
            }
        }
        impl DivAssign for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] /= rhs.0[i];
                }
            }
        }
        impl Neg for $name {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = -self.0[i];
                }
                Self(out)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        // Mask type
        #[derive(Copy, Clone, Debug)]
        pub struct $mask(pub $mask_prim);

        impl $mask {
            #[inline(always)]
            pub fn select(self, true_val: $name, false_val: $name) -> $name {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = if (self.0 >> i) & 1 == 1 {
                        true_val.0[i]
                    } else {
                        false_val.0[i]
                    };
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
            fn default() -> Self {
                Self([$zero; $lanes])
            }
        }

        impl $name {
            pub const LANES: usize = $lanes;

            #[inline(always)]
            pub fn splat(v: $elem) -> Self {
                Self([v; $lanes])
            }

            #[inline(always)]
            pub fn from_slice(s: &[$elem]) -> Self {
                assert!(s.len() >= $lanes);
                let mut arr = [$zero; $lanes];
                arr.copy_from_slice(&s[..$lanes]);
                Self(arr)
            }

            #[inline(always)]
            pub fn from_array(arr: [$elem; $lanes]) -> Self {
                Self(arr)
            }

            #[inline(always)]
            pub fn to_array(self) -> [$elem; $lanes] {
                self.0
            }

            #[inline(always)]
            pub fn copy_to_slice(self, s: &mut [$elem]) {
                assert!(s.len() >= $lanes);
                s[..$lanes].copy_from_slice(&self.0);
            }

            #[inline(always)]
            pub fn reduce_sum(self) -> $elem {
                let mut s: $elem = $zero;
                for i in 0..$lanes {
                    s = s.wrapping_add(self.0[i]);
                }
                s
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].wrapping_add(rhs.0[i]);
                }
                Self(out)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].wrapping_sub(rhs.0[i]);
                }
                Self(out)
            }
        }
        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] = self.0[i].wrapping_add(rhs.0[i]);
                }
            }
        }
        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] = self.0[i].wrapping_sub(rhs.0[i]);
                }
            }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] & rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] | rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] ^ rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitAndAssign for $name {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] &= rhs.0[i];
                }
            }
        }
        impl BitOrAssign for $name {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] |= rhs.0[i];
                }
            }
        }
        impl BitXorAssign for $name {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] ^= rhs.0[i];
                }
            }
        }
        impl Not for $name {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = !self.0[i];
                }
                Self(out)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
    };
}

// ── Instantiate all 11 types ─────────────────────────────────────

// 512-bit float types
impl_float_type!(F32x16, f32, 16, F32Mask16, u16);
impl_float_type!(F64x8, f64, 8, F64Mask8, u8);

// 256-bit AVX2 float types
// The macro `impl_float_type!` already emits `pub struct $mask(pub $mask_prim);`,
// so calling it with `F32Mask8Scalar` / `F64Mask4Scalar` defines those mask
// structs. The previous explicit re-declaration was a duplicate that
// tripped E0428 + 6× E0119 on i686-unknown-linux-gnu (where this scalar
// module compiles — `#[cfg(not(target_arch = "x86_64"))]`).
impl_float_type!(F32x8, f32, 8, F32Mask8Scalar, u8);
impl_float_type!(F64x4, f64, 4, F64Mask4Scalar, u8);

// 512-bit integer types
impl_int_type!(U8x64, u8, 64, 0u8);
impl_int_type!(I32x16, i32, 16, 0i32);
impl_int_type!(I64x8, i64, 8, 0i64);
impl_int_type!(U16x32, u16, 32, 0u16);
impl_int_type!(U32x16, u32, 16, 0u32);
impl_int_type!(U64x8, u64, 8, 0u64);

// I8/I16 SIMD types (scalar fallback)
impl_int_type!(I8x64, i8, 64, 0i8);
impl_int_type!(I8x32, i8, 32, 0i8);
impl_int_type!(I16x32, i16, 32, 0i16);
impl_int_type!(I16x16, i16, 16, 0i16);

// 256-bit int lanes — scalar polyfills filling the gap surfaced by the
// 2026-05-20 matrix audit. Mirror the additions in `src/simd_avx2.rs`
// (via the `avx2_int_type!` macro) so consumers on every backend reach
// the same type names through `crate::simd::*`.
impl_int_type!(U16x16, u16, 16, 0u16);
impl_int_type!(U32x8, u32, 8, 0u32);
impl_int_type!(U64x4, u64, 4, 0u64);
impl_int_type!(I32x8, i32, 8, 0i32);
impl_int_type!(I64x4, i64, 4, 0i64);

// I8x64 / I8x32 / I16x32 / I16x16 — AVX-512BW-style methods (scalar shape)
impl I8x64 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i8; 64])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] > other.0[i] {
                m |= 1u64 << i;
            }
        }
        m
    }
}
impl I8x32 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i8; 32])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        let mut m: u32 = 0;
        for i in 0..32 {
            if self.0[i] > other.0[i] {
                m |= 1u32 << i;
            }
        }
        m
    }
}
impl I16x32 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i16; 32])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        let mut m: u32 = 0;
        for i in 0..32 {
            if self.0[i] > other.0[i] {
                m |= 1u32 << i;
            }
        }
        m
    }
}
impl I16x16 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i16; 16])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        let mut m: u16 = 0;
        for i in 0..16 {
            if self.0[i] > other.0[i] {
                m |= 1u16 << i;
            }
        }
        m
    }
}

// Extra methods for U16x32 (widen/narrow, shift, multiply)
impl U16x32 {
    #[inline(always)]
    pub fn from_u8x64_lo(v: U8x64) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = v.0[i] as u16;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn from_u8x64_hi(v: U8x64) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = v.0[32 + i] as u16;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn pack_saturate_u8(self, other: Self) -> U8x64 {
        let mut out = [0u8; 64];
        for i in 0..32 {
            out[i] = self.0[i].min(255) as u8;
        }
        for i in 0..32 {
            out[32 + i] = other.0[i].min(255) as u8;
        }
        U8x64(out)
    }
    #[inline(always)]
    pub fn shr(self, imm: u32) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = if imm < 16 { self.0[i] >> imm } else { 0 };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn shl(self, imm: u32) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = if imm < 16 { self.0[i] << imm } else { 0 };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn mullo(self, other: Self) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = self.0[i].wrapping_mul(other.0[i]);
        }
        Self(out)
    }
}

// Extra methods for I32x16 that float types have via the macro
impl I32x16 {
    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cast_f32(self) -> F32x16 {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i] as f32;
        }
        F32x16(out)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].abs();
        }
        Self(out)
    }
    #[inline(always)]
    pub fn from_i16_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16);
        let mut o = [0i32; 16];
        for i in 0..16 {
            o[i] = s[i] as i32;
        }
        Self(o)
    }
    #[inline(always)]
    pub fn to_i16_array(self) -> [i16; 16] {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i] as i16;
        }
        o
    }
    #[inline(always)]
    pub fn cmpge_zero_mask(self) -> u16 {
        let mut mask = 0u16;
        for i in 0..16 {
            if self.0[i] >= 0 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

impl Mul for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for I32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Neg for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

// Extra for F32x16: to_bits/from_bits/cast_i32
impl F32x16 {
    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i].to_bits();
        }
        U32x16(out)
    }
    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = f32::from_bits(bits.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cast_i32(self) -> I32x16 {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i] as i32;
        }
        I32x16(out)
    }
}

// Extra for F64x8: to_bits/from_bits
impl F64x8 {
    #[inline(always)]
    pub fn to_bits(self) -> U64x8 {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i].to_bits();
        }
        U64x8(out)
    }
    #[inline(always)]
    pub fn from_bits(bits: U64x8) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 {
            out[i] = f64::from_bits(bits.0[i]);
        }
        Self(out)
    }
}

// Extra for I64x8
impl I64x8 {
    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].abs();
        }
        Self(out)
    }
}

impl Mul for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for I64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Neg for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

// Shift operators for U32x16
impl Shr<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i] >> rhs.0[i];
        }
        Self(out)
    }
}
impl Shl<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i] << rhs.0[i];
        }
        Self(out)
    }
}

// Shift operators for U64x8
impl Shr<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i] >> rhs.0[i];
        }
        Self(out)
    }
}
impl Shl<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i] << rhs.0[i];
        }
        Self(out)
    }
}

// Mul for U8x64 (wrapping)
impl Mul for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for U8x64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// U8x64 extra methods — byte-level operations for palette codec, nibble, byte scan
impl U8x64 {
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if self.0[i] == other.0[i] {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        let mut out = [0u8; 64];
        for i in (0..64).step_by(2) {
            let val = u16::from_le_bytes([self.0[i], self.0[i + 1]]);
            let shifted = val >> imm;
            let bytes = shifted.to_le_bytes();
            out[i] = bytes[0];
            out[i + 1] = bytes[1];
        }
        Self(out)
    }
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].saturating_sub(other.0[i]);
        }
        Self(out)
    }
    // ── Tier 1: seismon rasterizer primitives (scalar fallbacks) ──
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = ((self.0[i] as u16 + other.0[i] as u16 + 1) >> 1) as u8;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] > other.0[i] {
                m |= 1 << i;
            }
        }
        m
    }
    #[inline(always)]
    pub fn mask_blend(mask: u64, a: Self, b: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = if mask & (1 << i) != 0 { b.0[i] } else { a.0[i] };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        let mut out = [0u8; 64];
        for i in (0..64).step_by(2) {
            let v = u16::from_le_bytes([self.0[i], self.0[i + 1]]);
            let s = if imm < 16 { v << imm } else { 0 };
            let b = s.to_le_bytes();
            out[i] = b[0];
            out[i + 1] = b[1];
        }
        Self(out)
    }
    // ── Tier 2: sprite blit + palette remap (scalar fallbacks) ──
    #[inline(always)]
    pub unsafe fn mask_store(self, ptr: *mut u8, mask: u64) {
        for i in 0..64 {
            if mask & (1 << i) != 0 {
                *ptr.add(i) = self.0[i];
            }
        }
    }
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].saturating_add(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn permute_bytes(self, idx: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[(idx.0[i] & 63) as usize];
        }
        Self(out)
    }
    #[inline(always)]
    pub fn movemask(self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] & 0x80 != 0 {
                m |= 1 << i;
            }
        }
        m
    }
    #[inline(always)]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..8 {
                out[b + i * 2] = self.0[b + i];
                out[b + i * 2 + 1] = other.0[b + i];
            }
        }
        Self(out)
    }
    #[inline(always)]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..8 {
                out[b + i * 2] = self.0[b + 8 + i];
                out[b + i * 2 + 1] = other.0[b + 8 + i];
            }
        }
        Self(out)
    }
    /// Byte-wise shuffle: use `self` as a LUT, `idx` selects bytes within each 128-bit (16-byte) lane.
    #[inline(always)]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..16 {
                out[b + i] = self.0[b + (idx.0[b + i] & 0x0F) as usize];
            }
        }
        Self(out)
    }
    /// Sum all 64 bytes into a single `u64` without wrapping.
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        self.0.iter().map(|&b| b as u64).sum()
    }
    /// Build a nibble-popcount lookup table (replicated across 4 x 16-byte lanes).
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        let lane: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
        let mut arr = [0u8; 64];
        for l in 0..4 {
            arr[l * 16..(l + 1) * 16].copy_from_slice(&lane);
        }
        Self(arr)
    }
}

// Mul for U32x16
impl Mul for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}

// Lowercase aliases
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
#[allow(non_camel_case_types)]
pub type f32x8 = F32x8;
#[allow(non_camel_case_types)]
pub type f64x4 = F64x4;
#[allow(non_camel_case_types)]
pub type i8x64 = I8x64;
#[allow(non_camel_case_types)]
pub type i8x32 = I8x32;
#[allow(non_camel_case_types)]
pub type i16x32 = I16x32;
#[allow(non_camel_case_types)]
pub type i16x16 = I16x16;
// Lowercase aliases for the 256-bit polyfills added in the 2026-05-20
// missing-lanes sweep.
#[allow(non_camel_case_types)]
pub type u16x16 = U16x16;
#[allow(non_camel_case_types)]
pub type u32x8 = U32x8;
#[allow(non_camel_case_types)]
pub type u64x4 = U64x4;
#[allow(non_camel_case_types)]
pub type i32x8 = I32x8;
#[allow(non_camel_case_types)]
pub type i64x4 = I64x4;
