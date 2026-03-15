//! Intel MKL FFI backend.
//!
//! Provides CBLAS Level 1/2/3, LAPACKE, VML, and DFTI via Intel MKL.
//! Enable with `--features intel-mkl`. Mutually exclusive with `openblas`.
//!
//! Links: `-lmkl_intel_lp64 -lmkl_sequential -lmkl_core` (or dynamic `-lmkl_rt`).

#![allow(non_snake_case)]

use std::os::raw::{c_double, c_float, c_int, c_long, c_void};

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

// ═══════════════════════════════════════════════════════════════
// CBLAS (shared API surface with OpenBLAS)
// ═══════════════════════════════════════════════════════════════

#[link(name = "mkl_rt")]
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

// ═══════════════════════════════════════════════════════════════
// LAPACKE (MKL-only — OpenBLAS doesn't ship LAPACKE by default)
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn LAPACKE_sgetrf(layout: c_int, m: c_int, n: c_int, a: *mut c_float, lda: c_int, ipiv: *mut c_int) -> c_int;
    pub fn LAPACKE_dgetrf(layout: c_int, m: c_int, n: c_int, a: *mut c_double, lda: c_int, ipiv: *mut c_int) -> c_int;
    pub fn LAPACKE_sgetrs(layout: c_int, trans: u8, n: c_int, nrhs: c_int, a: *const c_float, lda: c_int, ipiv: *const c_int, b: *mut c_float, ldb: c_int) -> c_int;
    pub fn LAPACKE_dgetrs(layout: c_int, trans: u8, n: c_int, nrhs: c_int, a: *const c_double, lda: c_int, ipiv: *const c_int, b: *mut c_double, ldb: c_int) -> c_int;
    pub fn LAPACKE_spotrf(layout: c_int, uplo: u8, n: c_int, a: *mut c_float, lda: c_int) -> c_int;
    pub fn LAPACKE_dpotrf(layout: c_int, uplo: u8, n: c_int, a: *mut c_double, lda: c_int) -> c_int;
    pub fn LAPACKE_spotrs(layout: c_int, uplo: u8, n: c_int, nrhs: c_int, a: *const c_float, lda: c_int, b: *mut c_float, ldb: c_int) -> c_int;
    pub fn LAPACKE_sgeqrf(layout: c_int, m: c_int, n: c_int, a: *mut c_float, lda: c_int, tau: *mut c_float) -> c_int;
    pub fn LAPACKE_dgeqrf(layout: c_int, m: c_int, n: c_int, a: *mut c_double, lda: c_int, tau: *mut c_double) -> c_int;
}

// ═══════════════════════════════════════════════════════════════
// VML (Vector Math Library — MKL-only)
// ═══════════════════════════════════════════════════════════════

extern "C" {
    pub fn vsExp(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdExp(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsLn(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdLn(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsSqrt(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdSqrt(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsAbs(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vdAbs(n: c_int, a: *const c_double, y: *mut c_double);
    pub fn vsAdd(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsMul(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsDiv(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
    pub fn vsSin(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vsCos(n: c_int, a: *const c_float, y: *mut c_float);
    pub fn vsPow(n: c_int, a: *const c_float, b: *const c_float, y: *mut c_float);
}

// ═══════════════════════════════════════════════════════════════
// DFTI (Discrete Fourier Transform Interface — MKL-only)
// ═══════════════════════════════════════════════════════════════

pub type DftiDescriptorHandle = *mut c_void;

pub const DFTI_SINGLE: c_int = 35;
pub const DFTI_DOUBLE: c_int = 36;
pub const DFTI_COMPLEX: c_int = 32;
pub const DFTI_REAL: c_int = 33;
pub const DFTI_PLACEMENT: c_int = 11;
pub const DFTI_INPLACE: c_int = 43;
pub const DFTI_NOT_INPLACE: c_int = 44;
pub const DFTI_BACKWARD_SCALE: c_int = 5;

extern "C" {
    pub fn DftiCreateDescriptor(handle: *mut DftiDescriptorHandle, precision: c_int, domain: c_int, dimension: c_int, length: c_long) -> c_long;
    pub fn DftiSetValue(handle: DftiDescriptorHandle, param: c_int, ...) -> c_long;
    pub fn DftiCommitDescriptor(handle: DftiDescriptorHandle) -> c_long;
    pub fn DftiComputeForward(handle: DftiDescriptorHandle, x_inout: *mut c_void, ...) -> c_long;
    pub fn DftiComputeBackward(handle: DftiDescriptorHandle, x_inout: *mut c_void, ...) -> c_long;
    pub fn DftiFreeDescriptor(handle: *mut DftiDescriptorHandle) -> c_long;
}

// ═══════════════════════════════════════════════════════════════
// Safe wrappers matching native.rs signatures
// ═══════════════════════════════════════════════════════════════

pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len()) as c_int;
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

pub const fn sgemm_nr() -> usize { 16 }
pub const fn sgemm_mr() -> usize { 6 }
pub const fn dgemm_nr() -> usize { 8 }
pub const fn dgemm_mr() -> usize { 6 }
