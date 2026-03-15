//! OpenBLAS FFI backend.
//!
//! Provides CBLAS Level 1/2/3 via OpenBLAS shared library.
//! Enable with `--features openblas`. Mutually exclusive with `intel-mkl`.
//!
//! Links: `-lopenblas` (system library or vendored via `openblas-src`).

use std::os::raw::{c_double, c_float, c_int};

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

#[link(name = "openblas")]
extern "C" {
    fn cblas_sdot(n: c_int, x: *const c_float, incx: c_int, y: *const c_float, incy: c_int) -> c_float;
    fn cblas_ddot(n: c_int, x: *const c_double, incx: c_int, y: *const c_double, incy: c_int) -> c_double;
    fn cblas_saxpy(n: c_int, alpha: c_float, x: *const c_float, incx: c_int, y: *mut c_float, incy: c_int);
    fn cblas_daxpy(n: c_int, alpha: c_double, x: *const c_double, incx: c_int, y: *mut c_double, incy: c_int);
    fn cblas_sscal(n: c_int, alpha: c_float, x: *mut c_float, incx: c_int);
    fn cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
    fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    fn cblas_sasum(n: c_int, x: *const c_float, incx: c_int) -> c_float;
    fn cblas_dasum(n: c_int, x: *const c_double, incx: c_int) -> c_double;
    fn cblas_sgemm(
        layout: c_int, transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: c_float, a: *const c_float, lda: c_int,
        b: *const c_float, ldb: c_int,
        beta: c_float, c: *mut c_float, ldc: c_int,
    );
    fn cblas_dgemm(
        layout: c_int, transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: c_double, a: *const c_double, lda: c_int,
        b: *const c_double, ldb: c_int,
        beta: c_double, c: *mut c_double, ldc: c_int,
    );
    fn cblas_sgemv(
        layout: c_int, trans: c_int,
        m: c_int, n: c_int,
        alpha: c_float, a: *const c_float, lda: c_int,
        x: *const c_float, incx: c_int,
        beta: c_float, y: *mut c_float, incy: c_int,
    );
    fn cblas_dgemv(
        layout: c_int, trans: c_int,
        m: c_int, n: c_int,
        alpha: c_double, a: *const c_double, lda: c_int,
        x: *const c_double, incx: c_int,
        beta: c_double, y: *mut c_double, incy: c_int,
    );
}

// ─── Safe wrappers matching native.rs signatures ─────────────────

pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len()) as c_int;
    // SAFETY: slices are valid for n elements, incx/incy = 1 (contiguous)
    unsafe { cblas_sdot(n, x.as_ptr(), 1, y.as_ptr(), 1) }
}

pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as c_int;
    unsafe { cblas_ddot(n, x.as_ptr(), 1, y.as_ptr(), 1) }
}

pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len()) as c_int;
    unsafe { cblas_saxpy(n, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
}

pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len()) as c_int;
    unsafe { cblas_daxpy(n, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
}

pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    unsafe { cblas_sscal(x.len() as c_int, alpha, x.as_mut_ptr(), 1) }
}

pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    unsafe { cblas_dscal(x.len() as c_int, alpha, x.as_mut_ptr(), 1) }
}

pub fn nrm2_f32(x: &[f32]) -> f32 {
    unsafe { cblas_snrm2(x.len() as c_int, x.as_ptr(), 1) }
}

pub fn nrm2_f64(x: &[f64]) -> f64 {
    unsafe { cblas_dnrm2(x.len() as c_int, x.as_ptr(), 1) }
}

pub fn asum_f32(x: &[f32]) -> f32 {
    unsafe { cblas_sasum(x.len() as c_int, x.as_ptr(), 1) }
}

pub fn asum_f64(x: &[f64]) -> f64 {
    unsafe { cblas_dasum(x.len() as c_int, x.as_ptr(), 1) }
}

pub fn gemm_f32(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32, c: &mut [f32], ldc: usize,
) {
    // SAFETY: caller guarantees a is m×k (stride lda), b is k×n (stride ldb),
    // c is m×n (stride ldc), all row-major.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as c_int, n as c_int, k as c_int,
            alpha, a.as_ptr(), lda as c_int,
            b.as_ptr(), ldb as c_int,
            beta, c.as_mut_ptr(), ldc as c_int,
        );
    }
}

pub fn gemm_f64(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    beta: f64, c: &mut [f64], ldc: usize,
) {
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as c_int, n as c_int, k as c_int,
            alpha, a.as_ptr(), lda as c_int,
            b.as_ptr(), ldb as c_int,
            beta, c.as_mut_ptr(), ldc as c_int,
        );
    }
}

pub fn gemv_f32(
    m: usize, n: usize,
    alpha: f32, a: &[f32], lda: usize,
    x: &[f32], beta: f32, y: &mut [f32],
) {
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
            m as c_int, n as c_int,
            alpha, a.as_ptr(), lda as c_int,
            x.as_ptr(), 1, beta, y.as_mut_ptr(), 1,
        );
    }
}

pub fn gemv_f64(
    m: usize, n: usize,
    alpha: f64, a: &[f64], lda: usize,
    x: &[f64], beta: f64, y: &mut [f64],
) {
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
            m as c_int, n as c_int,
            alpha, a.as_ptr(), lda as c_int,
            x.as_ptr(), 1, beta, y.as_mut_ptr(), 1,
        );
    }
}

// Tile size constants (not meaningful for FFI, but needed for API compat)
pub const fn sgemm_nr() -> usize { 16 }
pub const fn sgemm_mr() -> usize { 6 }
pub const fn dgemm_nr() -> usize { 8 }
pub const fn dgemm_mr() -> usize { 6 }
