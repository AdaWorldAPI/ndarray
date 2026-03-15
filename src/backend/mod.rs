//! Pluggable linear algebra backends.
//!
//! The `LinalgBackend` trait provides a generic interface for BLAS-like operations.
//! Implementations are selected via feature gates:
//! - `native` (default): Pure Rust with SIMD acceleration
//! - `intel-mkl`: Intel MKL via FFI
//! - `openblas`: OpenBLAS via FFI

#![allow(clippy::too_many_arguments)]

mod native;

#[cfg(feature = "intel-mkl")]
mod mkl;
#[cfg(feature = "openblas")]
mod openblas;

pub use native::NativeBackend;

#[cfg(feature = "intel-mkl")]
pub use mkl::MklBackend;
#[cfg(feature = "openblas")]
pub use openblas::OpenBlasBackend;

// Ensure MKL and OpenBLAS are mutually exclusive
#[cfg(all(feature = "intel-mkl", feature = "openblas"))]
compile_error!("Features `intel-mkl` and `openblas` are mutually exclusive. Enable only one.");

/// Trait for pluggable linear algebra backends.
///
/// All methods operate on contiguous slices in row-major order.
/// Implementations must be generic over f32/f64 via the `BlasFloat` trait.
///
/// # Example
///
/// ```ignore
/// use ndarray::backend::{LinalgBackend, NativeBackend};
///
/// let backend = NativeBackend;
/// let x = vec![1.0f32, 2.0, 3.0];
/// let y = vec![4.0f32, 5.0, 6.0];
/// let dot = backend.dot(&x, &y);
/// assert_eq!(dot, 32.0);
/// ```
pub trait LinalgBackend {
    /// Dot product: result = Σ x[i] * y[i]
    fn dot_f32(&self, x: &[f32], y: &[f32]) -> f32;

    /// Dot product: result = Σ x[i] * y[i]
    fn dot_f64(&self, x: &[f64], y: &[f64]) -> f64;

    /// AXPY: y = alpha * x + y
    fn axpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]);

    /// AXPY: y = alpha * x + y
    fn axpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]);

    /// Scale: x = alpha * x
    fn scal_f32(&self, alpha: f32, x: &mut [f32]);

    /// Scale: x = alpha * x
    fn scal_f64(&self, alpha: f64, x: &mut [f64]);

    /// L2 norm: sqrt(Σ x[i]²)
    fn nrm2_f32(&self, x: &[f32]) -> f32;

    /// L2 norm: sqrt(Σ x[i]²)
    fn nrm2_f64(&self, x: &[f64]) -> f64;

    /// L1 norm (absolute sum): Σ |x[i]|
    fn asum_f32(&self, x: &[f32]) -> f32;

    /// L1 norm (absolute sum): Σ |x[i]|
    fn asum_f64(&self, x: &[f64]) -> f64;

    /// GEMM: C = alpha * A * B + beta * C
    ///
    /// - `m`: rows of A and C
    /// - `n`: columns of B and C
    /// - `k`: columns of A and rows of B
    /// - `a`: m×k matrix (row-major, stride = lda)
    /// - `b`: k×n matrix (row-major, stride = ldb)
    /// - `c`: m×n matrix (row-major, stride = ldc)
    fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    );

    /// GEMM: C = alpha * A * B + beta * C
    fn gemm_f64(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    );

    /// GEMV: y = alpha * A * x + beta * y
    fn gemv_f32(
        &self,
        m: usize,
        n: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    );

    /// GEMV: y = alpha * A * x + beta * y
    fn gemv_f64(
        &self,
        m: usize,
        n: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    );
}

/// Trait for float types usable in BLAS operations.
pub trait BlasFloat: num_traits::Float + Default + Send + Sync + 'static {
    /// Perform dot product using the active backend.
    fn backend_dot(x: &[Self], y: &[Self]) -> Self;
    /// Perform axpy using the active backend.
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]);
    /// Perform scal using the active backend.
    fn backend_scal(alpha: Self, x: &mut [Self]);
    /// L2 norm using the active backend.
    fn backend_nrm2(x: &[Self]) -> Self;
    /// L1 norm using the active backend.
    fn backend_asum(x: &[Self]) -> Self;
    /// GEMM using the active backend.
    fn backend_gemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    );
    /// GEMV using the active backend.
    fn backend_gemv(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    );
}

impl BlasFloat for f32 {
    fn backend_dot(x: &[Self], y: &[Self]) -> Self {
        NativeBackend.dot_f32(x, y)
    }
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        NativeBackend.axpy_f32(alpha, x, y);
    }
    fn backend_scal(alpha: Self, x: &mut [Self]) {
        NativeBackend.scal_f32(alpha, x);
    }
    fn backend_nrm2(x: &[Self]) -> Self {
        NativeBackend.nrm2_f32(x)
    }
    fn backend_asum(x: &[Self]) -> Self {
        NativeBackend.asum_f32(x)
    }
    fn backend_gemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        NativeBackend.gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) {
        NativeBackend.gemv_f32(m, n, alpha, a, lda, x, beta, y);
    }
}

impl BlasFloat for f64 {
    fn backend_dot(x: &[Self], y: &[Self]) -> Self {
        NativeBackend.dot_f64(x, y)
    }
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        NativeBackend.axpy_f64(alpha, x, y);
    }
    fn backend_scal(alpha: Self, x: &mut [Self]) {
        NativeBackend.scal_f64(alpha, x);
    }
    fn backend_nrm2(x: &[Self]) -> Self {
        NativeBackend.nrm2_f64(x)
    }
    fn backend_asum(x: &[Self]) -> Self {
        NativeBackend.asum_f64(x)
    }
    fn backend_gemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) {
        NativeBackend.gemm_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    fn backend_gemv(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) {
        NativeBackend.gemv_f64(m, n, alpha, a, lda, x, beta, y);
    }
}
