//! Operator impls for all simd_nightly types — round-3-portable-simd agent #10.
//!
//! Two main macros:
//! - `impl_fp_ops!` — Add/Sub/Mul/Div/Neg + assign variants for float types.
//! - `impl_int_ops!` — Add/Sub/BitAnd/BitOr/BitXor + assign variants for int types.
//!
//! A separate `impl_default!` macro handles `Default` for types whose own
//! module has NOT already derived/implemented it (e.g. F32x16/F32x8 already
//! implement `Default` in f32_types.rs; those are excluded here).
#![cfg(feature = "nightly-simd")]

// ════════════════════════════════════════════════════════════════════
// impl_fp_ops — float arithmetic + Neg + assign variants
// ════════════════════════════════════════════════════════════════════

/// Implement `Add`, `Sub`, `Mul`, `Div`, `Neg`, and their `*Assign`
/// counterparts for a newtype wrapper whose inner field is `.0`.
///
/// The wrapped type must itself implement the corresponding `core::ops`
/// traits (which all `core::simd` vector types do).
macro_rules! impl_fp_ops {
    ($name:ty) => {
        impl core::ops::Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl core::ops::Mul for $name {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl core::ops::Div for $name {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }

        impl core::ops::Neg for $name {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl core::ops::AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0 + rhs.0;
            }
        }

        impl core::ops::SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0 - rhs.0;
            }
        }

        impl core::ops::MulAssign for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 = self.0 * rhs.0;
            }
        }

        impl core::ops::DivAssign for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                self.0 = self.0 / rhs.0;
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// impl_int_ops — integer Add/Sub/BitAnd/BitOr/BitXor + assign variants
//                (wrapping semantics come from core::simd automatically)
// ════════════════════════════════════════════════════════════════════

/// Implement `Add`, `Sub`, `BitAnd`, `BitOr`, `BitXor`, and their
/// `*Assign` counterparts for a newtype wrapper whose inner field is `.0`.
///
/// `Neg` is NOT included here; for signed integer types use
/// `impl_int_neg!` separately.
macro_rules! impl_int_ops {
    ($name:ty) => {
        impl core::ops::Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl core::ops::BitAnd for $name {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl core::ops::BitOr for $name {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl core::ops::BitXor for $name {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl core::ops::AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0 + rhs.0;
            }
        }

        impl core::ops::SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0 - rhs.0;
            }
        }

        impl core::ops::BitAndAssign for $name {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 = self.0 & rhs.0;
            }
        }

        impl core::ops::BitOrAssign for $name {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 = self.0 | rhs.0;
            }
        }

        impl core::ops::BitXorAssign for $name {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 = self.0 ^ rhs.0;
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// impl_int_neg — Neg for signed integer types (separate from int_ops
//                so unsigned types don't accidentally get it)
// ════════════════════════════════════════════════════════════════════

macro_rules! impl_int_neg {
    ($name:ty) => {
        impl core::ops::Neg for $name {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// impl_default — Default (zero / false splat) for types that do NOT
//                already provide a Default impl in their own module.
//
// F32x16 and F32x8 already have Default in f32_types.rs — excluded.
// ════════════════════════════════════════════════════════════════════

macro_rules! impl_default {
    ($name:ty) => {
        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self(Default::default())
            }
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// Float type invocations
// ════════════════════════════════════════════════════════════════════

// F32x16 and F32x8 — ops only; Default already in f32_types.rs.
impl_fp_ops!(super::f32_types::F32x16);
impl_fp_ops!(super::f32_types::F32x8);

// F64x8 and F64x4 — ops + Default (not in f64_types.rs).
impl_fp_ops!(super::f64_types::F64x8);
impl_default!(super::f64_types::F64x8);
impl_fp_ops!(super::f64_types::F64x4);
impl_default!(super::f64_types::F64x4);

// ════════════════════════════════════════════════════════════════════
// Unsigned integer type invocations
// ════════════════════════════════════════════════════════════════════

// U8x32, U8x64 — ops only; Default already in u8_types.rs.
impl_int_ops!(super::u8_types::U8x32);
impl_int_ops!(super::u8_types::U8x64);

// All u_word types have Default in u_word_types.rs — ops only.
impl_int_ops!(super::u_word_types::U16x32);
impl_int_ops!(super::u_word_types::U32x16);
impl_int_ops!(super::u_word_types::U32x8);
impl_int_ops!(super::u_word_types::U64x8);
impl_int_ops!(super::u_word_types::U64x4);

// ════════════════════════════════════════════════════════════════════
// Signed integer type invocations (int_ops + Neg + Default)
// ════════════════════════════════════════════════════════════════════

impl_int_ops!(super::i8_types::I8x32);
impl_int_neg!(super::i8_types::I8x32);
impl_default!(super::i8_types::I8x32);
impl_int_ops!(super::i8_types::I8x64);
impl_int_neg!(super::i8_types::I8x64);
impl_default!(super::i8_types::I8x64);

impl_int_ops!(super::i_word_types::I16x16);
impl_int_neg!(super::i_word_types::I16x16);
impl_default!(super::i_word_types::I16x16);
impl_int_ops!(super::i_word_types::I16x32);
impl_int_neg!(super::i_word_types::I16x32);
impl_default!(super::i_word_types::I16x32);
impl_int_ops!(super::i_word_types::I32x16);
impl_int_neg!(super::i_word_types::I32x16);
impl_default!(super::i_word_types::I32x16);
impl_int_ops!(super::i_word_types::I64x8);
impl_int_neg!(super::i_word_types::I64x8);
impl_default!(super::i_word_types::I64x8);
