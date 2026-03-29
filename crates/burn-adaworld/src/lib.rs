//! burn-adaworld: Burn backend powered by adaworldapi/ndarray SIMD.
//!
//! Implements burn's `Backend` trait using:
//! - `crate::simd::F32x16` via `LazyLock<SimdDispatch>` (AVX-512 → AVX2 → scalar)
//! - Optional `AttentionTable` for O(1) compiled attention (bgz-tensor)
//! - `SimilarityTable` as BF16-precision cosine replacement (256 levels)
//!
//! # Usage
//!
//! ```ignore
//! use burn_adaworld::AdaWorld;
//! use burn_tensor::Tensor;
//!
//! let a = Tensor::<AdaWorld, 2>::ones([3, 4], &Default::default());
//! let b = Tensor::<AdaWorld, 2>::ones([4, 5], &Default::default());
//! let c = a.matmul(b); // Uses crate::simd BLAS, or AttentionTable if compiled
//! ```

pub mod backend;
pub mod element;
pub mod tensor;
pub mod ops;

pub use backend::AdaWorld;
