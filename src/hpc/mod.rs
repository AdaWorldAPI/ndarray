#![allow(
    clippy::assign_op_pattern,
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::type_complexity
)]
//! HPC extensions for ndarray — ported from rustynum.
//!
//! This module provides high-performance computing extensions:
//! - BLAS Level 1/2/3 operations as extension traits
//! - Statistics (median, var, std, percentile)
//! - Activation functions (sigmoid, softmax, log_softmax)
//! - HDC (Hyperdimensional Computing) operations
//! - CogRecord 4-channel cognitive units
//! - Graph operations with VerbCodebook
//! - BF16 and Int8 quantized GEMM
//! - LAPACK factorizations (LU, Cholesky, QR)
//! - FFT (forward, inverse, real-to-complex)
//! - VML (vectorized math library)

pub mod blas_level1;
pub mod blas_level2;
pub mod blas_level3;
pub mod statistics;
pub mod activations;
pub mod hdc;
pub mod bitwise;
pub mod projection;
pub mod cogrecord;
pub mod graph;
pub mod quantized;
pub mod lapack;
pub mod fft;
pub mod vml;
