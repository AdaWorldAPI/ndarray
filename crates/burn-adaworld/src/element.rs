//! Element types supported by the AdaWorld backend.
//!
//! Maps burn's element traits to ndarray-compatible types.

use burn_backend::Element;
use burn_tensor::{DType, ElementConversion};
use num_traits::ToPrimitive;

/// Marker trait for elements usable with our ndarray backend.
pub trait AdaElement: Element + ndarray::LinalgScalar + ndarray::ScalarOperand + Default + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(val: f32) -> Self;
}

impl AdaElement for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 { self }
    #[inline(always)]
    fn from_f32(val: f32) -> Self { val }
}

impl AdaElement for f64 {
    #[inline(always)]
    fn to_f32(self) -> f32 { self as f32 }
    #[inline(always)]
    fn from_f32(val: f32) -> Self { val as f64 }
}

/// Integer element trait.
pub trait AdaIntElement: Element + ndarray::LinalgScalar + ndarray::ScalarOperand + Default + 'static {
    fn to_i64(self) -> i64;
    fn from_i64(val: i64) -> Self;
}

impl AdaIntElement for i32 {
    #[inline(always)]
    fn to_i64(self) -> i64 { self as i64 }
    #[inline(always)]
    fn from_i64(val: i64) -> Self { val as i32 }
}

impl AdaIntElement for i64 {
    #[inline(always)]
    fn to_i64(self) -> i64 { self }
    #[inline(always)]
    fn from_i64(val: i64) -> Self { val }
}
