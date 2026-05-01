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

pub mod native;
#[cfg(target_arch = "x86_64")]
pub(crate) mod kernels_avx512;

#[cfg(feature = "intel-mkl")]
pub mod mkl;
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
// Default (no feature): native — SIMD-dispatched, zero C deps.
//
// CBLAS-compat aliases (`cblas_sgemm`, `cblas_dgemm`) are always
// available below, routing through the native backend regardless of
// feature flags. Code written against MKL/OpenBLAS CBLAS can switch
// to these with zero logic changes.

#[cfg(feature = "intel-mkl")]
pub use mkl::{
    asum_f32, asum_f64, axpy_f32, axpy_f64, dgemm_mr, dgemm_nr, dot_f32, dot_f64, gemm_f32, gemm_f64, gemv_f32,
    gemv_f64, nrm2_f32, nrm2_f64, scal_f32, scal_f64, sgemm_mr, sgemm_nr,
};

#[cfg(all(feature = "openblas", not(feature = "intel-mkl")))]
pub use openblas::{
    asum_f32, asum_f64, axpy_f32, axpy_f64, dgemm_mr, dgemm_nr, dot_f32, dot_f64, gemm_f32, gemm_f64, gemv_f32,
    gemv_f64, nrm2_f32, nrm2_f64, scal_f32, scal_f64, sgemm_mr, sgemm_nr,
};

#[cfg(not(any(feature = "intel-mkl", feature = "openblas")))]
pub use native::{
    asum_f32, asum_f64, axpy_f32, axpy_f64, dgemm_mr, dgemm_nr, dot_f32, dot_f64, gemm_f32, gemm_f64, gemv_f32,
    gemv_f64, nrm2_f32, nrm2_f64, scal_f32, scal_f64, sgemm_mr, sgemm_nr,
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
        m: usize, n: usize, k: usize, alpha: Self, a: &[Self], lda: usize, b: &[Self], ldb: usize, beta: Self,
        c: &mut [Self], ldc: usize,
    );
    /// GEMV using the active backend.
    fn backend_gemv(m: usize, n: usize, alpha: Self, a: &[Self], lda: usize, x: &[Self], beta: Self, y: &mut [Self]);
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
        m: usize, n: usize, k: usize, alpha: Self, a: &[Self], lda: usize, b: &[Self], ldb: usize, beta: Self,
        c: &mut [Self], ldc: usize,
    ) {
        gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(m: usize, n: usize, alpha: Self, a: &[Self], lda: usize, x: &[Self], beta: Self, y: &mut [Self]) {
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
        m: usize, n: usize, k: usize, alpha: Self, a: &[Self], lda: usize, b: &[Self], ldb: usize, beta: Self,
        c: &mut [Self], ldc: usize,
    ) {
        gemm_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(m: usize, n: usize, alpha: Self, a: &[Self], lda: usize, x: &[Self], beta: Self, y: &mut [Self]) {
        gemv_f64(m, n, alpha, a, lda, x, beta, y);
    }
}

// ─── CBLAS-compatible aliases ─────────────────────────────────────
//
// Drop-in replacements for MKL/OpenBLAS CBLAS calls. All route through
// the native SIMD-dispatched kernels (AVX-512/AVX2/NEON/AMX/VNNI).
// No external C library needed.
//
// Usage:
//   use ndarray::backend::{cblas_sgemm, cblas_dgemm};
//   cblas_sgemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

/// `cblas_sgemm` equivalent — pure Rust SIMD-dispatched f32 GEMM.
#[inline]
pub fn cblas_sgemm(
    m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, b: &[f32], ldb: usize, beta: f32, c: &mut [f32],
    ldc: usize,
) {
    gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

/// `cblas_dgemm` equivalent — pure Rust SIMD-dispatched f64 GEMM.
#[inline]
pub fn cblas_dgemm(
    m: usize, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64],
    ldc: usize,
) {
    gemm_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// ─── Unified INT8 / BF16 GEMM dispatch ───────────────────────────
//
// Auto-dispatched: AMX > VNNI > scalar. Consumer writes one call,
// gets the best available hardware path.

/// INT8 GEMM: C = A × B where A is u8, B is i8, C is i32.
///
/// Dispatch: AMX TDPBUSD → VNNI VPDPBUSD → scalar.
/// Same signature across all paths.
#[inline]
#[allow(clippy::needless_return)]
pub fn gemm_i8(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // VNNI path (Ice Lake, Sapphire Rapids, Zen 4) — includes AMX fallback
    #[cfg(feature = "std")]
    {
        crate::hpc::vnni_gemm::int8_gemm_vnni(a, b, c, m, n, k);
        return;
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (a, b, c, m, n, k);
        panic!("INT8 GEMM requires std feature");
    }
}

/// BF16 GEMM: C (f32) = A (BF16) × B (BF16), with f32 accumulation.
///
/// Dispatch: AMX TDPBF16PS → scalar tiled bf16_gemm_f32.
/// Input: raw u16 slices representing BF16 values (same layout as
/// `ndarray::hpc::quantized::BF16`).
#[inline]
#[allow(clippy::needless_return)]
pub fn gemm_bf16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Reinterpret u16 slices as BF16 slices (repr(transparent))
    #[cfg(feature = "std")]
    {
        let a_bf16: &[crate::hpc::quantized::BF16] = unsafe {
            // SAFETY: BF16 is #[repr(transparent)] over u16
            core::slice::from_raw_parts(a.as_ptr() as *const crate::hpc::quantized::BF16, a.len())
        };
        let b_bf16: &[crate::hpc::quantized::BF16] =
            unsafe { core::slice::from_raw_parts(b.as_ptr() as *const crate::hpc::quantized::BF16, b.len()) };
        crate::hpc::quantized::bf16_gemm_f32(a_bf16, b_bf16, c, m, n, k, 1.0, 0.0);
        return;
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (a, b, c, m, n, k);
        panic!("BF16 GEMM requires std feature");
    }
}

/// CBLAS-compat alias for INT8 GEMM.
#[inline]
pub fn cblas_gemm_s8s8s32(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    gemm_i8(a, b, c, m, n, k)
}

/// CBLAS-compat alias for BF16 GEMM.
#[inline]
pub fn cblas_gemm_bf16bf16f32(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    gemm_bf16(a, b, c, m, n, k)
}

// ─── Elementwise ops (SIMD-dispatched) ───────────────────────────
//
// Slice-level add/sub/mul/div for f32, dispatched through the AVX-512
// kernel with AVX2/scalar fallback. Both vec×vec and vec×scalar forms.
//
// Usage:
//   use ndarray::backend::{add_f32_vec, mul_f32_scalar};
//   let c = add_f32_vec(&a, &b);        // c[i] = a[i] + b[i]
//   let d = mul_f32_scalar(&a, 2.0);    // d[i] = a[i] * 2.0

#[cfg(target_arch = "x86_64")]
pub use kernels_avx512::{
    add_f32_scalar, add_f32_vec, div_f32_scalar, div_f32_vec, iamax_f32, iamax_f64, mul_f32_scalar, mul_f32_vec,
    sub_f32_scalar, sub_f32_vec,
};

// ─── Slice-level ops by dtype (unified re-exports) ──────────────
//
// All the SIMD-dispatched slice ops in one place.
// Integer: simd_int_ops. Half: simd_half. Float: kernels_avx512 + reductions.

#[cfg(feature = "std")]
pub use crate::simd_int_ops::{add_i16, add_i8, dot_i16, dot_i8, max_i8, min_i8, sub_i8};

#[cfg(feature = "std")]
pub use crate::simd_half::{
    add_bf16_inplace, add_f16_inplace, cast_bf16_to_f32_batch, cast_f16_to_f32_batch, cast_f32_to_bf16_batch,
    cast_f32_to_f16_batch, mul_bf16_inplace, mul_f16_inplace,
};

#[cfg(feature = "std")]
pub use crate::hpc::reductions::{
    argmax_f32, argmin_f32, max_f32, mean_f32, mean_f64, min_f32, nrm2_f32 as nrm2_f32_simd, sum_f32, sum_f64,
};
