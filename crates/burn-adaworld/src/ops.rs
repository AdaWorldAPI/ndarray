//! Tensor operations for the AdaWorld backend.
//!
//! Implements burn's FloatTensorOps, IntTensorOps, BoolTensorOps by delegating
//! to ndarray operations accelerated by crate::simd.

pub mod float_ops;
pub mod int_ops;
pub mod bool_ops;
