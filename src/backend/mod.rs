//! Pluggable SIMD backends with runtime CPU detection.
//!
//! Uses the `dispatch!` macro pattern: one `LazyLock<Tier>` detection at first call,
//! then every function dispatches to the fastest available tier (AVX-512 → AVX2 → scalar).
//!
//! Feature gates for external BLAS:
//! - `native` (default): Pure Rust with SIMD acceleration
//! - `intel-mkl`: Intel MKL via FFI
//! - `openblas`: OpenBLAS via FFI

#![allow(clippy::too_many_arguments)]

mod native;
#[cfg(target_arch = "x86_64")]
pub(crate) mod kernels_avx512;

#[cfg(feature = "intel-mkl")]
mod mkl;
#[cfg(feature = "openblas")]
mod openblas;

// Ensure MKL and OpenBLAS are mutually exclusive
#[cfg(all(feature = "intel-mkl", feature = "openblas"))]
compile_error!("Features `intel-mkl` and `openblas` are mutually exclusive. Enable only one.");

// ─── Re-export dispatch functions ─────────────────────────────────
//
// These are the public API. No trait, no struct — just functions.
// `backend::dot_f32(x, y)` dispatches to the best tier automatically.
//
// Priority: intel-mkl > openblas > native (pure Rust SIMD).

#[cfg(feature = "intel-mkl")]
pub use mkl::{
    dot_f32, dot_f64,
    axpy_f32, axpy_f64,
    scal_f32, scal_f64,
    nrm2_f32, nrm2_f64,
    asum_f32, asum_f64,
    gemm_f32, gemm_f64,
    gemv_f32, gemv_f64,
    sgemm_nr, sgemm_mr, dgemm_nr, dgemm_mr,
};

#[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
pub use openblas::{
    dot_f32, dot_f64,
    axpy_f32, axpy_f64,
    scal_f32, scal_f64,
    nrm2_f32, nrm2_f64,
    asum_f32, asum_f64,
    gemm_f32, gemm_f64,
    gemv_f32, gemv_f64,
    sgemm_nr, sgemm_mr, dgemm_nr, dgemm_mr,
};

#[cfg(not(any(feature = "intel-mkl", feature = "openblas")))]
pub use native::{
    dot_f32, dot_f64,
    axpy_f32, axpy_f64,
    scal_f32, scal_f64,
    nrm2_f32, nrm2_f64,
    asum_f32, asum_f64,
    gemm_f32, gemm_f64,
    gemv_f32, gemv_f64,
    sgemm_nr, sgemm_mr, dgemm_nr, dgemm_mr,
};

// ─── BlasFloat: type-level dispatch for generic code ──────────────
//
// HPC extension traits use `A: BlasFloat` to write generic code
// that works on both f32 and f64 without manual dispatch.

/// Trait for float types usable in BLAS operations.
pub trait BlasFloat: num_traits::Float + Default + Send + Sync + 'static {
    /// Dot product using the active backend.
    fn backend_dot(x: &[Self], y: &[Self]) -> Self;
    /// AXPY using the active backend.
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]);
    /// Scale using the active backend.
    fn backend_scal(alpha: Self, x: &mut [Self]);
    /// L2 norm using the active backend.
    fn backend_nrm2(x: &[Self]) -> Self;
    /// L1 norm using the active backend.
    fn backend_asum(x: &[Self]) -> Self;
    /// GEMM using the active backend.
    fn backend_gemm(
        m: usize, n: usize, k: usize,
        alpha: Self, a: &[Self], lda: usize,
        b: &[Self], ldb: usize,
        beta: Self, c: &mut [Self], ldc: usize,
    );
    /// GEMV using the active backend.
    fn backend_gemv(
        m: usize, n: usize,
        alpha: Self, a: &[Self], lda: usize,
        x: &[Self], beta: Self, y: &mut [Self],
    );
}

impl BlasFloat for f32 {
    fn backend_dot(x: &[Self], y: &[Self]) -> Self {
        dot_f32(x, y)
    }
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        axpy_f32(alpha, x, y);
    }
    fn backend_scal(alpha: Self, x: &mut [Self]) {
        scal_f32(alpha, x);
    }
    fn backend_nrm2(x: &[Self]) -> Self {
        nrm2_f32(x)
    }
    fn backend_asum(x: &[Self]) -> Self {
        asum_f32(x)
    }
    fn backend_gemm(
        m: usize, n: usize, k: usize,
        alpha: Self, a: &[Self], lda: usize,
        b: &[Self], ldb: usize,
        beta: Self, c: &mut [Self], ldc: usize,
    ) {
        gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(
        m: usize, n: usize,
        alpha: Self, a: &[Self], lda: usize,
        x: &[Self], beta: Self, y: &mut [Self],
    ) {
        gemv_f32(m, n, alpha, a, lda, x, beta, y);
    }
}

impl BlasFloat for f64 {
    fn backend_dot(x: &[Self], y: &[Self]) -> Self {
        dot_f64(x, y)
    }
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        axpy_f64(alpha, x, y);
    }
    fn backend_scal(alpha: Self, x: &mut [Self]) {
        scal_f64(alpha, x);
    }
    fn backend_nrm2(x: &[Self]) -> Self {
        nrm2_f64(x)
    }
    fn backend_asum(x: &[Self]) -> Self {
        asum_f64(x)
    }
    fn backend_gemm(
        m: usize, n: usize, k: usize,
        alpha: Self, a: &[Self], lda: usize,
        b: &[Self], ldb: usize,
        beta: Self, c: &mut [Self], ldc: usize,
    ) {
        gemm_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(
        m: usize, n: usize,
        alpha: Self, a: &[Self], lda: usize,
        x: &[Self], beta: Self, y: &mut [Self],
    ) {
        gemv_f64(m, n, alpha, a, lda, x, beta, y);
    }
}
