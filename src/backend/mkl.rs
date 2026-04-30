//! Intel MKL FFI backend.
//!
//! Provides CBLAS Level 1/2/3, LAPACKE, VML, and DFTI via Intel MKL.
//! Enable with `--features intel-mkl`. Mutually exclusive with `openblas`.
//!
//! Links: `-lmkl_intel_lp64 -lmkl_sequential -lmkl_core` (or dynamic `-lmkl_rt`).

#![allow(non_snake_case)]

use crate::{ArrayView2, ArrayViewMut2};
use std::os::raw::{c_double, c_float, c_int, c_long, c_void};

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

// `cblas_gemm_s8u8s32` / `cblas_gemm_s8s8s32` use CBLAS_OFFSET enums for the
// final argument (offset mode). `RowOffset = 171`, `ColOffset = 172`,
// `FixOffset = 173` — we always use `FixOffset` with a zero offset, which
// matches Burn / rustyblas behaviour.
const CBLAS_OFFSET_FIX: c_int = 173;

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

    // Mixed-precision GEMM: BF16 inputs, F32 accumulator.
    // MKL takes `*const u16` for BF16 operands (no native bf16 type in C ABI).
    // Reference: oneAPI MKL Developer Reference, "cblas_gemm_bf16bf16f32".
    fn cblas_gemm_bf16bf16f32(
        layout: c_int, transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: c_float, a: *const u16, lda: c_int,
        b: *const u16, ldb: c_int,
        beta: c_float, c: *mut c_float, ldc: c_int,
    );

    // Integer GEMM: i8 × i8 → i32.
    // The trailing offset arguments take CBLAS_OFFSET (= FixOffset) plus a
    // pointer to the offset value. Passing zero offsets matches a plain
    // matmul without zero-point correction.
    // Reference: oneAPI MKL Developer Reference, "cblas_gemm_s8s8s32".
    fn cblas_gemm_s8s8s32(
        layout: c_int, transa: c_int, transb: c_int, offsetc: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: c_float,
        a: *const i8, lda: c_int, oa: i8,
        b: *const i8, ldb: c_int, ob: i8,
        beta: c_float, c: *mut i32, ldc: c_int,
        co: *const i32,
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

// ═══════════════════════════════════════════════════════════════
// Public ndarray-shaped GEMM API (Burn integration surface)
// ═══════════════════════════════════════════════════════════════
//
// These wrappers accept `ArrayView2` / `ArrayViewMut2` and forward to the
// CBLAS FFI declared above. They handle row-major / column-major layout
// detection from ndarray strides and return a structured error if the input
// is non-contiguous along its leading dimension (which CBLAS cannot express).

/// Errors returned from the MKL ndarray-shaped GEMM wrappers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MklError {
    /// Inner dimensions of A and B don't match (`A.cols != B.rows`).
    ShapeMismatch {
        a_shape: (usize, usize),
        b_shape: (usize, usize),
    },
    /// Output `C` dimensions don't match `(A.rows, B.cols)`.
    OutputShapeMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// One of the arrays is not stride-compatible with CBLAS.
    ///
    /// CBLAS requires that one of the two strides is `1` (the contiguous
    /// dimension) and the other is `>= the contiguous extent`. Arbitrary
    /// striding (e.g. from a non-contiguous slice) is not supported — copy
    /// to a contiguous buffer first.
    NonContiguous { which: &'static str },
    /// The bf16 / int8 routines are not available in this MKL build, or are
    /// stubbed out (e.g. older MKL versions predate `cblas_gemm_*`).
    Unsupported(&'static str),
}

impl core::fmt::Display for MklError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MklError::ShapeMismatch { a_shape, b_shape } => write!(
                f,
                "MKL GEMM shape mismatch: A is {:?}, B is {:?}",
                a_shape, b_shape
            ),
            MklError::OutputShapeMismatch { expected, got } => write!(
                f,
                "MKL GEMM output shape mismatch: expected {:?}, got {:?}",
                expected, got
            ),
            MklError::NonContiguous { which } => {
                write!(f, "MKL GEMM operand `{}` is not stride-compatible with CBLAS", which)
            }
            MklError::Unsupported(msg) => write!(f, "MKL GEMM unsupported: {}", msg),
        }
    }
}

impl std::error::Error for MklError {}

/// CBLAS layout descriptor extracted from an ndarray view.
struct BlasLayout {
    layout: c_int,
    trans: c_int,
    ld: c_int,
}

/// Inspect strides and produce a CBLAS layout/transpose/leading-dimension
/// triple for a 2D ndarray view. Returns `None` if neither dimension has
/// stride 1 (i.e. the matrix is non-contiguous in both directions).
fn blas_layout<S: crate::RawData>(view: &crate::ArrayBase<S, crate::Ix2>) -> Option<BlasLayout> {
    let (rows, cols) = view.dim();
    let strides = view.strides();
    let rs = strides[0];
    let cs = strides[1];
    // Row-major: stride between rows is the leading dim, columns are stride 1.
    if cs == 1 && (rs >= cols as isize || rows <= 1) {
        return Some(BlasLayout { layout: CBLAS_ROW_MAJOR, trans: CBLAS_NO_TRANS, ld: rs.max(1) as c_int });
    }
    // Column-major: stride between cols is the leading dim, rows are stride 1.
    // We expose this to CBLAS as a *row-major transposed* matrix so we keep a
    // single `layout` argument across all three operands.
    if rs == 1 && (cs >= rows as isize || cols <= 1) {
        return Some(BlasLayout { layout: CBLAS_ROW_MAJOR, trans: 112 /* CblasTrans */, ld: cs.max(1) as c_int });
    }
    None
}

/// `C := alpha * A * B + beta * C` for `f32` matrices via MKL `cblas_sgemm`.
pub fn sgemm(
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    mut c: ArrayViewMut2<f32>,
    alpha: f32,
    beta: f32,
) -> Result<(), MklError> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(MklError::ShapeMismatch { a_shape: a.dim(), b_shape: b.dim() });
    }
    if c.dim() != (m, n) {
        return Err(MklError::OutputShapeMismatch { expected: (m, n), got: c.dim() });
    }
    let la = blas_layout(&a).ok_or(MklError::NonContiguous { which: "a" })?;
    let lb = blas_layout(&b).ok_or(MklError::NonContiguous { which: "b" })?;
    let lc = blas_layout(&c).ok_or(MklError::NonContiguous { which: "c" })?;
    if lc.trans != CBLAS_NO_TRANS {
        return Err(MklError::NonContiguous { which: "c" });
    }
    unsafe {
        cblas_sgemm(
            lc.layout, la.trans, lb.trans,
            m as c_int, n as c_int, k as c_int,
            alpha, a.as_ptr(), la.ld,
            b.as_ptr(), lb.ld,
            beta, c.as_mut_ptr(), lc.ld,
        );
    }
    Ok(())
}

/// `C := alpha * A * B + beta * C` for `f64` matrices via MKL `cblas_dgemm`.
pub fn dgemm(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    mut c: ArrayViewMut2<f64>,
    alpha: f64,
    beta: f64,
) -> Result<(), MklError> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(MklError::ShapeMismatch { a_shape: a.dim(), b_shape: b.dim() });
    }
    if c.dim() != (m, n) {
        return Err(MklError::OutputShapeMismatch { expected: (m, n), got: c.dim() });
    }
    let la = blas_layout(&a).ok_or(MklError::NonContiguous { which: "a" })?;
    let lb = blas_layout(&b).ok_or(MklError::NonContiguous { which: "b" })?;
    let lc = blas_layout(&c).ok_or(MklError::NonContiguous { which: "c" })?;
    if lc.trans != CBLAS_NO_TRANS {
        return Err(MklError::NonContiguous { which: "c" });
    }
    unsafe {
        cblas_dgemm(
            lc.layout, la.trans, lb.trans,
            m as c_int, n as c_int, k as c_int,
            alpha, a.as_ptr(), la.ld,
            b.as_ptr(), lb.ld,
            beta, c.as_mut_ptr(), lc.ld,
        );
    }
    Ok(())
}

/// `C := alpha * A * B + beta * C` with BF16 inputs and `f32` accumulator,
/// via MKL `cblas_gemm_bf16bf16f32`.
///
/// This requires Intel MKL >= 2020 (for the bf16 GEMM kernel). On older MKL
/// builds the symbol is missing and linking will fail at runtime — there is
/// no compile-time fallback.
pub fn sgemm_bf16(
    a: ArrayView2<crate::hpc::quantized::BF16>,
    b: ArrayView2<crate::hpc::quantized::BF16>,
    mut c: ArrayViewMut2<f32>,
    alpha: f32,
    beta: f32,
) -> Result<(), MklError> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(MklError::ShapeMismatch { a_shape: a.dim(), b_shape: b.dim() });
    }
    if c.dim() != (m, n) {
        return Err(MklError::OutputShapeMismatch { expected: (m, n), got: c.dim() });
    }
    let la = blas_layout(&a).ok_or(MklError::NonContiguous { which: "a" })?;
    let lb = blas_layout(&b).ok_or(MklError::NonContiguous { which: "b" })?;
    let lc = blas_layout(&c).ok_or(MklError::NonContiguous { which: "c" })?;
    if lc.trans != CBLAS_NO_TRANS {
        return Err(MklError::NonContiguous { which: "c" });
    }
    // BF16 is `#[repr(transparent)] (pub u16)`, so the pointer cast is sound.
    unsafe {
        cblas_gemm_bf16bf16f32(
            lc.layout, la.trans, lb.trans,
            m as c_int, n as c_int, k as c_int,
            alpha,
            a.as_ptr() as *const u16, la.ld,
            b.as_ptr() as *const u16, lb.ld,
            beta, c.as_mut_ptr(), lc.ld,
        );
    }
    Ok(())
}

/// `C := A * B` with `i8` inputs and `i32` accumulator, via MKL
/// `cblas_gemm_s8s8s32` with zero offsets (no zero-point correction).
///
/// Note: alpha/beta are fixed at `1.0` / `0.0` for the simple `Burn`-style
/// signature. If you need scaling, call the FFI directly. This requires
/// Intel MKL >= 2018 (when integer GEMM was introduced).
pub fn sgemm_int8(
    a: ArrayView2<i8>,
    b: ArrayView2<i8>,
    mut c: ArrayViewMut2<i32>,
) -> Result<(), MklError> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return Err(MklError::ShapeMismatch { a_shape: a.dim(), b_shape: b.dim() });
    }
    if c.dim() != (m, n) {
        return Err(MklError::OutputShapeMismatch { expected: (m, n), got: c.dim() });
    }
    let la = blas_layout(&a).ok_or(MklError::NonContiguous { which: "a" })?;
    let lb = blas_layout(&b).ok_or(MklError::NonContiguous { which: "b" })?;
    let lc = blas_layout(&c).ok_or(MklError::NonContiguous { which: "c" })?;
    if lc.trans != CBLAS_NO_TRANS {
        return Err(MklError::NonContiguous { which: "c" });
    }
    let co: i32 = 0;
    unsafe {
        cblas_gemm_s8s8s32(
            lc.layout, la.trans, lb.trans, CBLAS_OFFSET_FIX,
            m as c_int, n as c_int, k as c_int,
            1.0_f32,
            a.as_ptr(), la.ld, 0_i8,
            b.as_ptr(), lb.ld, 0_i8,
            0.0_f32, c.as_mut_ptr(), lc.ld,
            &co as *const i32,
        );
    }
    Ok(())
}
