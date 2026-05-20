//! Portable-SIMD polyfill backend — full 30-type coverage (NIGHTLY ONLY).
//!
//! Wraps `core::simd::*` so miri can execute the polyfill paths.
//! Intrinsics backends (`simd_avx512.rs` / `simd_avx2.rs`) are opaque
//! to miri; `core::simd` is not. With `--features nightly-simd`, consumer
//! code using `ndarray::simd_nightly::*` becomes miri-checkable.
//!
//! Gated entirely behind `#[cfg(feature = "nightly-simd")]`. Requires
//! `cargo +nightly` because the file pulls in `#![feature(portable_simd)]`
//! at the crate root (see `lib.rs`).
//!
//! # Coverage
//!
//! 30 types across 12 sub-modules, populated by the round-3-portable-simd
//! fleet. The module aggregator (this file) re-exports the public surface
//! flat so consumers write `use ndarray::simd_nightly::F32x16` rather
//! than `use ndarray::simd_nightly::f32_types::F32x16`.

#![cfg(feature = "nightly-simd")]

pub mod bf16_types;
pub mod exotic_methods;
pub mod f16_types;
pub mod f32_types;
pub mod f64_types;
pub mod i8_types;
pub mod i_word_types;
pub mod masks;
pub mod ops;
pub mod u8_types;
pub mod u_word_types;

#[cfg(test)]
mod tests;

// Flat re-exports — consumer surface matches `crate::simd::*` shape.
pub use bf16_types::{BF16x16, BF16x8};
pub use f16_types::F16x16;
pub use f32_types::{F32x16, F32x8};
pub use f64_types::{F64x4, F64x8};
pub use i8_types::{I8x32, I8x64};
pub use i_word_types::{I16x16, I16x32, I32x16, I64x8};
pub use masks::{F32Mask16, F32Mask8, F64Mask4, F64Mask8};
pub use u8_types::{U8x32, U8x64};
pub use u_word_types::{U16x32, U32x16, U32x8, U64x4, U64x8};

// Lowercase aliases — match the std::simd convention used by
// `simd_avx2.rs`, `simd_avx512.rs`, and the scalar fallback in
// `simd_scalar.rs`. Consumer docs and downstream code import names like
// `crate::simd::f32x16`; without these aliases, enabling `nightly-simd`
// would silently break those imports (codex P1 on PR #173).
#[allow(non_camel_case_types)]
pub type f32x16 = F32x16;
#[allow(non_camel_case_types)]
pub type f32x8 = F32x8;
#[allow(non_camel_case_types)]
pub type f64x8 = F64x8;
#[allow(non_camel_case_types)]
pub type f64x4 = F64x4;
#[allow(non_camel_case_types)]
pub type u8x64 = U8x64;
#[allow(non_camel_case_types)]
pub type u8x32 = U8x32;
#[allow(non_camel_case_types)]
pub type u16x32 = U16x32;
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;
#[allow(non_camel_case_types)]
pub type u32x8 = U32x8;
#[allow(non_camel_case_types)]
pub type u64x8 = U64x8;
#[allow(non_camel_case_types)]
pub type u64x4 = U64x4;
#[allow(non_camel_case_types)]
pub type i8x64 = I8x64;
#[allow(non_camel_case_types)]
pub type i8x32 = I8x32;
#[allow(non_camel_case_types)]
pub type i16x32 = I16x32;
#[allow(non_camel_case_types)]
pub type i16x16 = I16x16;
#[allow(non_camel_case_types)]
pub type i32x16 = I32x16;
#[allow(non_camel_case_types)]
pub type i64x8 = I64x8;
