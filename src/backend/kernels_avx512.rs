//! AVX-512 SIMD kernels using portable compat types.
//!
//! All functions have `#[target_feature(enable = "avx512f")]`.
//! The dispatch! macro's LazyLock tier check ensures these are only called
//! on AVX-512 CPUs.
//!
//! BLAS-1 and element-wise functions use `F32x16`/`F64x8` from `crate::simd`.
//! GEMM microkernels retain raw intrinsics for masked stores and broadcast patterns.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::simd::{F32x16, F64x8};

// ═══════════════════════════════════════════════════════════════════
// BLAS Level 1 — 12 functions (compat types)
// ═══════════════════════════════════════════════════════════════════

/// Dot product: sum(x[i] * y[i]) using 4x-unrolled FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let mut acc0 = F32x16::splat(0.0);
    let mut acc1 = F32x16::splat(0.0);
    let mut acc2 = F32x16::splat(0.0);
    let mut acc3 = F32x16::splat(0.0);
    let mut i = 0;

    while i + 64 <= n {
        acc0 = F32x16::from_slice(&x[i..]).mul_add(F32x16::from_slice(&y[i..]), acc0);
        acc1 = F32x16::from_slice(&x[i + 16..]).mul_add(F32x16::from_slice(&y[i + 16..]), acc1);
        acc2 = F32x16::from_slice(&x[i + 32..]).mul_add(F32x16::from_slice(&y[i + 32..]), acc2);
        acc3 = F32x16::from_slice(&x[i + 48..]).mul_add(F32x16::from_slice(&y[i + 48..]), acc3);
        i += 64;
    }
    while i + 16 <= n {
        acc0 = F32x16::from_slice(&x[i..]).mul_add(F32x16::from_slice(&y[i..]), acc0);
        i += 16;
    }

    let mut total = ((acc0 + acc1) + (acc2 + acc3)).reduce_sum();
    while i < n {
        total += x[i] * y[i];
        i += 1;
    }
    total
}

/// Dot product f64: 4x-unrolled FMA (8 doubles each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut acc0 = F64x8::splat(0.0);
    let mut acc1 = F64x8::splat(0.0);
    let mut acc2 = F64x8::splat(0.0);
    let mut acc3 = F64x8::splat(0.0);
    let mut i = 0;

    while i + 32 <= n {
        acc0 = F64x8::from_slice(&x[i..]).mul_add(F64x8::from_slice(&y[i..]), acc0);
        acc1 = F64x8::from_slice(&x[i + 8..]).mul_add(F64x8::from_slice(&y[i + 8..]), acc1);
        acc2 = F64x8::from_slice(&x[i + 16..]).mul_add(F64x8::from_slice(&y[i + 16..]), acc2);
        acc3 = F64x8::from_slice(&x[i + 24..]).mul_add(F64x8::from_slice(&y[i + 24..]), acc3);
        i += 32;
    }
    while i + 8 <= n {
        acc0 = F64x8::from_slice(&x[i..]).mul_add(F64x8::from_slice(&y[i..]), acc0);
        i += 8;
    }

    let mut total = ((acc0 + acc1) + (acc2 + acc3)).reduce_sum();
    while i < n {
        total += x[i] * y[i];
        i += 1;
    }
    total
}

/// AXPY: y = alpha * x + y (f32, 16-wide FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    let alpha_v = F32x16::splat(alpha);
    let mut i = 0;
    while i + 16 <= n {
        let xv = F32x16::from_slice(&x[i..]);
        let yv = F32x16::from_slice(&y[i..]);
        alpha_v.mul_add(xv, yv).copy_to_slice(&mut y[i..]);
        i += 16;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

/// AXPY: y = alpha * x + y (f64, 8-wide FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    let alpha_v = F64x8::splat(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let xv = F64x8::from_slice(&x[i..]);
        let yv = F64x8::from_slice(&y[i..]);
        alpha_v.mul_add(xv, yv).copy_to_slice(&mut y[i..]);
        i += 8;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

/// Scale: x = alpha * x (f32, 16-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let alpha_v = F32x16::splat(alpha);
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&x[i..]) * alpha_v).copy_to_slice(&mut x[i..]);
        i += 16;
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

/// Scale: x = alpha * x (f64, 8-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let alpha_v = F64x8::splat(alpha);
    let mut i = 0;
    while i + 8 <= n {
        (F64x8::from_slice(&x[i..]) * alpha_v).copy_to_slice(&mut x[i..]);
        i += 8;
    }
    while i < n {
        x[i] *= alpha;
        i += 1;
    }
}

/// L1 norm: sum(|x[i]|) (f32, 16-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn asum_f32(x: &[f32]) -> f32 {
    let mut acc = F32x16::splat(0.0);
    for chunk in x.chunks_exact(16) {
        acc += F32x16::from_slice(chunk).abs();
    }
    let mut sum = acc.reduce_sum();
    for &v in x.chunks_exact(16).remainder() {
        sum += v.abs();
    }
    sum
}

/// L1 norm: sum(|x[i]|) (f64, 8-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn asum_f64(x: &[f64]) -> f64 {
    let mut acc = F64x8::splat(0.0);
    for chunk in x.chunks_exact(8) {
        acc += F64x8::from_slice(chunk).abs();
    }
    let mut sum = acc.reduce_sum();
    for &v in x.chunks_exact(8).remainder() {
        sum += v.abs();
    }
    sum
}

/// L2 norm: sqrt(sum(x[i]^2)) (f32, 16-wide FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut acc = F32x16::splat(0.0);
    while i + 16 <= n {
        let xv = F32x16::from_slice(&x[i..]);
        acc = xv.mul_add(xv, acc);
        i += 16;
    }
    let mut sum = acc.reduce_sum();
    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }
    sum.sqrt()
}

/// L2 norm: sqrt(sum(x[i]^2)) (f64, 8-wide FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let mut i = 0;
    let mut acc = F64x8::splat(0.0);
    while i + 8 <= n {
        let xv = F64x8::from_slice(&x[i..]);
        acc = xv.mul_add(xv, acc);
        i += 8;
    }
    let mut sum = acc.reduce_sum();
    while i < n {
        sum += x[i] * x[i];
        i += 1;
    }
    sum.sqrt()
}

/// Index of max absolute value (f32). Scalar — no AVX-512 specialization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn iamax_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() { return (0, 0.0); }
    let mut max_idx = 0;
    let mut max_val = x[0].abs();
    for (i, &v) in x.iter().enumerate().skip(1) {
        let a = v.abs();
        if a > max_val { max_val = a; max_idx = i; }
    }
    (max_idx, x[max_idx])
}

/// Index of max absolute value (f64). Scalar — no AVX-512 specialization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn iamax_f64(x: &[f64]) -> (usize, f64) {
    if x.is_empty() { return (0, 0.0); }
    let mut max_idx = 0;
    let mut max_val = x[0].abs();
    for (i, &v) in x.iter().enumerate().skip(1) {
        let a = v.abs();
        if a > max_val { max_val = a; max_idx = i; }
    }
    (max_idx, x[max_idx])
}

// ═══════════════════════════════════════════════════════════════════
// Element-wise f32 — 8 functions (16-wide, compat types)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, EwOp::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, EwOp::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, EwOp::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, EwOp::Div) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, EwOp::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, EwOp::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, EwOp::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, EwOp::Div) }

// ═══════════════════════════════════════════════════════════════════
// Element-wise f64 — 8 functions (8-wide, compat types)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, EwOp::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, EwOp::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, EwOp::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, EwOp::Div) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, EwOp::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, EwOp::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, EwOp::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, EwOp::Div) }

// ─── Element-wise helpers (compat types) ─────────────────────────

#[cfg(target_arch = "x86_64")]
enum EwOp { Add, Sub, Mul, Div }

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn apply_f32(a: F32x16, b: F32x16, op: &EwOp) -> F32x16 {
    match op {
        EwOp::Add => a + b,
        EwOp::Sub => a - b,
        EwOp::Mul => a * b,
        EwOp::Div => a / b,
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn apply_f64(a: F64x8, b: F64x8, op: &EwOp) -> F64x8 {
    match op {
        EwOp::Add => a + b,
        EwOp::Sub => a - b,
        EwOp::Mul => a * b,
        EwOp::Div => a / b,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f32_s(a: &[f32], scalar: f32, op: EwOp) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0f32; n];
    let sv = F32x16::splat(scalar);
    let mut i = 0;
    while i + 16 <= n {
        apply_f32(F32x16::from_slice(&a[i..]), sv, &op).copy_to_slice(&mut result[i..]);
        i += 16;
    }
    while i < n {
        result[i] = match op {
            EwOp::Add => a[i] + scalar,
            EwOp::Sub => a[i] - scalar,
            EwOp::Mul => a[i] * scalar,
            EwOp::Div => a[i] / scalar,
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f32_v(a: &[f32], b: &[f32], op: EwOp) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut result = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        apply_f32(F32x16::from_slice(&a[i..]), F32x16::from_slice(&b[i..]), &op)
            .copy_to_slice(&mut result[i..]);
        i += 16;
    }
    while i < n {
        result[i] = match op {
            EwOp::Add => a[i] + b[i],
            EwOp::Sub => a[i] - b[i],
            EwOp::Mul => a[i] * b[i],
            EwOp::Div => a[i] / b[i],
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f64_s(a: &[f64], scalar: f64, op: EwOp) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0f64; n];
    let sv = F64x8::splat(scalar);
    let mut i = 0;
    while i + 8 <= n {
        apply_f64(F64x8::from_slice(&a[i..]), sv, &op).copy_to_slice(&mut result[i..]);
        i += 8;
    }
    while i < n {
        result[i] = match op {
            EwOp::Add => a[i] + scalar,
            EwOp::Sub => a[i] - scalar,
            EwOp::Mul => a[i] * scalar,
            EwOp::Div => a[i] / scalar,
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f64_v(a: &[f64], b: &[f64], op: EwOp) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut result = vec![0.0f64; n];
    let mut i = 0;
    while i + 8 <= n {
        apply_f64(F64x8::from_slice(&a[i..]), F64x8::from_slice(&b[i..]), &op)
            .copy_to_slice(&mut result[i..]);
        i += 8;
    }
    while i < n {
        result[i] = match op {
            EwOp::Add => a[i] + b[i],
            EwOp::Sub => a[i] - b[i],
            EwOp::Mul => a[i] * b[i],
            EwOp::Div => a[i] / b[i],
        };
        i += 1;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// GEMM — Goto BLAS packed microkernel (raw intrinsics retained)
//
// The GEMM microkernels use masked stores, broadcast-FMA patterns,
// and packed memory layouts that are inherently architecture-specific.
// They stay as raw intrinsics; the compat layer covers BLAS-1 + element-wise.
// ═══════════════════════════════════════════════════════════════════

const SGEMM_MR: usize = 6;
const SGEMM_NR: usize = 16;
const SGEMM_KC: usize = 256;
const SGEMM_MC: usize = 72;
const SGEMM_NC: usize = 256;

const DGEMM_MR: usize = 6;
const DGEMM_NR: usize = 8;
const DGEMM_KC: usize = 192;
const DGEMM_MC: usize = 72;
const DGEMM_NC: usize = 128;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn pack_a_f32(a: &[f32], lda: usize, mc: usize, kc: usize, i_start: usize, k_start: usize, buf: &mut [f32]) {
    let mut idx = 0;
    let mut ii = 0;
    while ii + SGEMM_MR <= mc {
        for p in 0..kc {
            for ir in 0..SGEMM_MR {
                buf[idx] = a[(i_start + ii + ir) * lda + (k_start + p)];
                idx += 1;
            }
        }
        ii += SGEMM_MR;
    }
    if ii < mc {
        let rem = mc - ii;
        for p in 0..kc {
            for ir in 0..SGEMM_MR {
                buf[idx] = if ir < rem { a[(i_start + ii + ir) * lda + (k_start + p)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn pack_b_f32(b: &[f32], ldb: usize, kc: usize, nc: usize, k_start: usize, j_start: usize, buf: &mut [f32]) {
    let mut idx = 0;
    let mut jj = 0;
    while jj + SGEMM_NR <= nc {
        for p in 0..kc {
            for jr in 0..SGEMM_NR {
                buf[idx] = b[(k_start + p) * ldb + (j_start + jj + jr)];
                idx += 1;
            }
        }
        jj += SGEMM_NR;
    }
    if jj < nc {
        let rem = nc - jj;
        for p in 0..kc {
            for jr in 0..SGEMM_NR {
                buf[idx] = if jr < rem { b[(k_start + p) * ldb + (j_start + jj + jr)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// AVX-512 microkernel: C[MR×NR] += A_packed[MR×kc] * B_packed[kc×NR]
///
/// Uses raw intrinsics for broadcast-FMA and masked store patterns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_ukernel_6x16(
    kc: usize,
    alpha: f32,
    a_packed: &[f32],
    b_packed: &[f32],
    c: &mut [f32],
    ldc: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();

    for p in 0..kc {
        let b_off = p * SGEMM_NR;
        let bv = _mm512_loadu_ps(b_packed[b_off..].as_ptr());

        let a_off = p * SGEMM_MR;
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off]), bv, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 1]), bv, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 2]), bv, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 3]), bv, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 4]), bv, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 5]), bv, c5);
    }

    let alpha_v = _mm512_set1_ps(alpha);
    c0 = _mm512_mul_ps(c0, alpha_v);
    c1 = _mm512_mul_ps(c1, alpha_v);
    c2 = _mm512_mul_ps(c2, alpha_v);
    c3 = _mm512_mul_ps(c3, alpha_v);
    c4 = _mm512_mul_ps(c4, alpha_v);
    c5 = _mm512_mul_ps(c5, alpha_v);

    let rows = [c0, c1, c2, c3, c4, c5];
    for ir in 0..mr_eff {
        let row_ptr = c[ir * ldc..].as_mut_ptr();
        if nr_eff == SGEMM_NR {
            let cv = _mm512_loadu_ps(row_ptr);
            _mm512_storeu_ps(row_ptr, _mm512_add_ps(cv, rows[ir]));
        } else {
            let mask: u16 = (1u32 << nr_eff) as u16 - 1;
            let cv = _mm512_maskz_loadu_ps(mask, row_ptr);
            _mm512_mask_storeu_ps(row_ptr, mask, _mm512_add_ps(cv, rows[ir]));
        }
    }
}

/// Goto BLAS style blocked SGEMM with packing and AVX-512 microkernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    let mut a_packed = vec![0.0f32; SGEMM_MC * SGEMM_KC];
    let mut b_packed = vec![0.0f32; SGEMM_KC * SGEMM_NC];

    let mut kk = 0;
    while kk < k {
        let kc = SGEMM_KC.min(k - kk);
        let mut jj = 0;
        while jj < n {
            let nc = SGEMM_NC.min(n - jj);
            pack_b_f32(b, ldb, kc, nc, kk, jj, &mut b_packed);

            let mut ii = 0;
            while ii < m {
                let mc = SGEMM_MC.min(m - ii);
                pack_a_f32(a, lda, mc, kc, ii, kk, &mut a_packed);

                let mut ir = 0;
                while ir < mc {
                    let mr_eff = SGEMM_MR.min(mc - ir);
                    let mut jr = 0;
                    while jr < nc {
                        let nr_eff = SGEMM_NR.min(nc - jr);
                        let a_off = (ir / SGEMM_MR) * (SGEMM_MR * kc);
                        let b_off = (jr / SGEMM_NR) * (SGEMM_NR * kc);

                        // SAFETY: tier() verified AVX-512F, buffers sized correctly
                        unsafe {
                            sgemm_ukernel_6x16(
                                kc, alpha,
                                &a_packed[a_off..],
                                &b_packed[b_off..],
                                &mut c[(ii + ir) * ldc + (jj + jr)..],
                                ldc, mr_eff, nr_eff,
                            );
                        }
                        jr += SGEMM_NR;
                    }
                    ir += SGEMM_MR;
                }
                ii += mc;
            }
            jj += nc;
        }
        kk += kc;
    }
}

// ─── DGEMM (f64) blocked ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn pack_a_f64(a: &[f64], lda: usize, mc: usize, kc: usize, i_start: usize, k_start: usize, buf: &mut [f64]) {
    let mut idx = 0;
    let mut ii = 0;
    while ii + DGEMM_MR <= mc {
        for p in 0..kc {
            for ir in 0..DGEMM_MR {
                buf[idx] = a[(i_start + ii + ir) * lda + (k_start + p)];
                idx += 1;
            }
        }
        ii += DGEMM_MR;
    }
    if ii < mc {
        let rem = mc - ii;
        for p in 0..kc {
            for ir in 0..DGEMM_MR {
                buf[idx] = if ir < rem { a[(i_start + ii + ir) * lda + (k_start + p)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn pack_b_f64(b: &[f64], ldb: usize, kc: usize, nc: usize, k_start: usize, j_start: usize, buf: &mut [f64]) {
    let mut idx = 0;
    let mut jj = 0;
    while jj + DGEMM_NR <= nc {
        for p in 0..kc {
            for jr in 0..DGEMM_NR {
                buf[idx] = b[(k_start + p) * ldb + (j_start + jj + jr)];
                idx += 1;
            }
        }
        jj += DGEMM_NR;
    }
    if jj < nc {
        let rem = nc - jj;
        for p in 0..kc {
            for jr in 0..DGEMM_NR {
                buf[idx] = if jr < rem { b[(k_start + p) * ldb + (j_start + jj + jr)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// AVX-512 microkernel: C[6×8] += A_packed[6×kc] * B_packed[kc×8] (f64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dgemm_ukernel_6x8(
    kc: usize,
    alpha: f64,
    a_packed: &[f64],
    b_packed: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    let mut c0 = _mm512_setzero_pd();
    let mut c1 = _mm512_setzero_pd();
    let mut c2 = _mm512_setzero_pd();
    let mut c3 = _mm512_setzero_pd();
    let mut c4 = _mm512_setzero_pd();
    let mut c5 = _mm512_setzero_pd();

    for p in 0..kc {
        let b_off = p * DGEMM_NR;
        let bv = _mm512_loadu_pd(b_packed[b_off..].as_ptr());

        let a_off = p * DGEMM_MR;
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off]), bv, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 1]), bv, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 2]), bv, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 3]), bv, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 4]), bv, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 5]), bv, c5);
    }

    let alpha_v = _mm512_set1_pd(alpha);
    c0 = _mm512_mul_pd(c0, alpha_v);
    c1 = _mm512_mul_pd(c1, alpha_v);
    c2 = _mm512_mul_pd(c2, alpha_v);
    c3 = _mm512_mul_pd(c3, alpha_v);
    c4 = _mm512_mul_pd(c4, alpha_v);
    c5 = _mm512_mul_pd(c5, alpha_v);

    let rows = [c0, c1, c2, c3, c4, c5];
    for ir in 0..mr_eff {
        let row_ptr = c[ir * ldc..].as_mut_ptr();
        if nr_eff == DGEMM_NR {
            let cv = _mm512_loadu_pd(row_ptr);
            _mm512_storeu_pd(row_ptr, _mm512_add_pd(cv, rows[ir]));
        } else {
            let mask: u8 = (1u16 << nr_eff) as u8 - 1;
            let cv = _mm512_maskz_loadu_pd(mask, row_ptr);
            _mm512_mask_storeu_pd(row_ptr, mask, _mm512_add_pd(cv, rows[ir]));
        }
    }
}

/// Goto BLAS style blocked DGEMM with packing and AVX-512 microkernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    c: &mut [f64], ldc: usize,
) {
    let mut a_packed = vec![0.0f64; DGEMM_MC * DGEMM_KC];
    let mut b_packed = vec![0.0f64; DGEMM_KC * DGEMM_NC];

    let mut kk = 0;
    while kk < k {
        let kc = DGEMM_KC.min(k - kk);
        let mut jj = 0;
        while jj < n {
            let nc = DGEMM_NC.min(n - jj);
            pack_b_f64(b, ldb, kc, nc, kk, jj, &mut b_packed);

            let mut ii = 0;
            while ii < m {
                let mc = DGEMM_MC.min(m - ii);
                pack_a_f64(a, lda, mc, kc, ii, kk, &mut a_packed);

                let mut ir = 0;
                while ir < mc {
                    let mr_eff = DGEMM_MR.min(mc - ir);
                    let mut jr = 0;
                    while jr < nc {
                        let nr_eff = DGEMM_NR.min(nc - jr);
                        let a_off = (ir / DGEMM_MR) * (DGEMM_MR * kc);
                        let b_off = (jr / DGEMM_NR) * (DGEMM_NR * kc);

                        unsafe {
                            dgemm_ukernel_6x8(
                                kc, alpha,
                                &a_packed[a_off..],
                                &b_packed[b_off..],
                                &mut c[(ii + ir) * ldc + (jj + jr)..],
                                ldc, mr_eff, nr_eff,
                            );
                        }
                        jr += DGEMM_NR;
                    }
                    ir += DGEMM_MR;
                }
                ii += mc;
            }
            jj += nc;
        }
        kk += kc;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Binary / HDC — 4 functions (raw intrinsics — VPOPCNTDQ specific)
// ═══════════════════════════════════════════════════════════════════

/// Hamming distance using VPOPCNTDQ — 64 bytes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let mut i = 0;
    let mut total_acc = _mm512_setzero_si512();

    while i + 64 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_si512(a[i..].as_ptr() as *const _);
            let bv = _mm512_loadu_si512(b[i..].as_ptr() as *const _);
            let xor = _mm512_xor_si512(av, bv);
            let popcnt = _mm512_popcnt_epi64(xor);
            total_acc = _mm512_add_epi64(total_acc, popcnt);
        }
        i += 64;
    }

    let mut total = _mm512_reduce_add_epi64(total_acc) as u64;
    while i < n {
        total += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    total
}

/// Population count using VPOPCNTDQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub fn popcount(a: &[u8]) -> u64 {
    let n = a.len();
    let mut i = 0;
    let mut total_acc = _mm512_setzero_si512();

    while i + 64 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_si512(a[i..].as_ptr() as *const _);
            let popcnt = _mm512_popcnt_epi64(av);
            total_acc = _mm512_add_epi64(total_acc, popcnt);
        }
        i += 64;
    }

    let mut total = _mm512_reduce_add_epi64(total_acc) as u64;
    while i < n {
        total += a[i].count_ones() as u64;
        i += 1;
    }
    total
}

/// Int8 dot product (scalar — no AVX-512 VNNI specialization yet).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    let n = a.len().min(b.len());
    let mut sum = 0i64;
    for i in 0..n {
        sum += (a[i] as i8 as i64) * (b[i] as i8 as i64);
    }
    sum
}

/// Batch Hamming distance using VPOPCNTDQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    (0..num_rows)
        .map(|i| {
            let start = i * row_bytes;
            hamming_distance(query, &database[start..start + row_bytes])
        })
        .collect()
}
