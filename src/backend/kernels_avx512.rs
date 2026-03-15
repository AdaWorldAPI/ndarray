//! AVX-512 SIMD kernels — raw intrinsics, Rust 1.94.
//!
//! All functions have `#[target_feature(enable = "avx512f")]`.
//! In Rust 1.94, arithmetic intrinsics (setzero, add, fmadd, reduce, etc.)
//! are safe inside `#[target_feature]` functions. Only load/store intrinsics
//! that take raw pointers still require `unsafe`.
//!
//! The dispatch! macro's LazyLock tier check ensures these are only called
//! on AVX-512 CPUs.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ═══════════════════════════════════════════════════════════════════
// BLAS Level 1 — 12 functions
// ═══════════════════════════════════════════════════════════════════

/// Dot product: sum(x[i] * y[i]) using 4x-unrolled FMA on zmm registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 64 <= n {
        // SAFETY: bounds checked by while condition, slicing ensures valid pointers
        unsafe {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x[i..].as_ptr()), _mm512_loadu_ps(y[i..].as_ptr()), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x[i + 16..].as_ptr()), _mm512_loadu_ps(y[i + 16..].as_ptr()), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x[i + 32..].as_ptr()), _mm512_loadu_ps(y[i + 32..].as_ptr()), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x[i + 48..].as_ptr()), _mm512_loadu_ps(y[i + 48..].as_ptr()), acc3);
        }
        i += 64;
    }
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x[i..].as_ptr()), _mm512_loadu_ps(y[i..].as_ptr()), acc0);
        }
        i += 16;
    }

    let sum_vec = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(sum_vec);
    while i < n {
        total += x[i] * y[i];
        i += 1;
    }
    total
}

/// Dot product f64: 4x-unrolled FMA on zmm registers (8 doubles each).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();
    let mut acc2 = _mm512_setzero_pd();
    let mut acc3 = _mm512_setzero_pd();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(x[i..].as_ptr()), _mm512_loadu_pd(y[i..].as_ptr()), acc0);
            acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(x[i + 8..].as_ptr()), _mm512_loadu_pd(y[i + 8..].as_ptr()), acc1);
            acc2 = _mm512_fmadd_pd(_mm512_loadu_pd(x[i + 16..].as_ptr()), _mm512_loadu_pd(y[i + 16..].as_ptr()), acc2);
            acc3 = _mm512_fmadd_pd(_mm512_loadu_pd(x[i + 24..].as_ptr()), _mm512_loadu_pd(y[i + 24..].as_ptr()), acc3);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(x[i..].as_ptr()), _mm512_loadu_pd(y[i..].as_ptr()), acc0);
        }
        i += 8;
    }

    let sum_vec = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
    let mut total = _mm512_reduce_add_pd(sum_vec);
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
    let alpha_v = _mm512_set1_ps(alpha);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_ps(x[i..].as_ptr());
            let yv = _mm512_loadu_ps(y[i..].as_ptr());
            _mm512_storeu_ps(y[i..].as_mut_ptr(), _mm512_fmadd_ps(alpha_v, xv, yv));
        }
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
    let alpha_v = _mm512_set1_pd(alpha);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_pd(x[i..].as_ptr());
            let yv = _mm512_loadu_pd(y[i..].as_ptr());
            _mm512_storeu_pd(y[i..].as_mut_ptr(), _mm512_fmadd_pd(alpha_v, xv, yv));
        }
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
    let alpha_v = _mm512_set1_ps(alpha);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_ps(x[i..].as_ptr());
            _mm512_storeu_ps(x[i..].as_mut_ptr(), _mm512_mul_ps(alpha_v, xv));
        }
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
    let alpha_v = _mm512_set1_pd(alpha);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_pd(x[i..].as_ptr());
            _mm512_storeu_pd(x[i..].as_mut_ptr(), _mm512_mul_pd(alpha_v, xv));
        }
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
    let n = x.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_ps();
    let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFi32);
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_ps(x[i..].as_ptr());
            let absv = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(xv), abs_mask));
            acc = _mm512_add_ps(acc, absv);
        }
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < n {
        sum += x[i].abs();
        i += 1;
    }
    sum
}

/// L1 norm: sum(|x[i]|) (f64, 8-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn asum_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_pd();
    let abs_mask = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFi64);
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_pd(x[i..].as_ptr());
            let absv = _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(xv), abs_mask));
            acc = _mm512_add_pd(acc, absv);
        }
        i += 8;
    }
    let mut sum = _mm512_reduce_add_pd(acc);
    while i < n {
        sum += x[i].abs();
        i += 1;
    }
    sum
}

/// L2 norm: sqrt(sum(x[i]^2)) (f32, 16-wide FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_ps();
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_ps(x[i..].as_ptr());
            acc = _mm512_fmadd_ps(xv, xv, acc);
        }
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
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
    let mut acc = _mm512_setzero_pd();
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let xv = _mm512_loadu_pd(x[i..].as_ptr());
            acc = _mm512_fmadd_pd(xv, xv, acc);
        }
        i += 8;
    }
    let mut sum = _mm512_reduce_add_pd(acc);
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
// Element-wise f32 — 8 functions (16-wide)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, Op::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, Op::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, Op::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { ew_f32_s(a, scalar, Op::Div) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, Op::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, Op::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, Op::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { ew_f32_v(a, b, Op::Div) }

// ═══════════════════════════════════════════════════════════════════
// Element-wise f64 — 8 functions (8-wide)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, Op::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, Op::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, Op::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { ew_f64_s(a, scalar, Op::Div) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn add_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, Op::Add) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sub_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, Op::Sub) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn mul_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, Op::Mul) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn div_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { ew_f64_v(a, b, Op::Div) }

// ─── Element-wise helpers ────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
enum Op { Add, Sub, Mul, Div }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f32_s(a: &[f32], scalar: f32, op: Op) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0f32; n];
    let sv = _mm512_set1_ps(scalar);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_ps(a[i..].as_ptr());
            let rv = match op {
                Op::Add => _mm512_add_ps(av, sv),
                Op::Sub => _mm512_sub_ps(av, sv),
                Op::Mul => _mm512_mul_ps(av, sv),
                Op::Div => _mm512_div_ps(av, sv),
            };
            _mm512_storeu_ps(result[i..].as_mut_ptr(), rv);
        }
        i += 16;
    }
    while i < n {
        result[i] = match op {
            Op::Add => a[i] + scalar,
            Op::Sub => a[i] - scalar,
            Op::Mul => a[i] * scalar,
            Op::Div => a[i] / scalar,
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f32_v(a: &[f32], b: &[f32], op: Op) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut result = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_ps(a[i..].as_ptr());
            let bv = _mm512_loadu_ps(b[i..].as_ptr());
            let rv = match op {
                Op::Add => _mm512_add_ps(av, bv),
                Op::Sub => _mm512_sub_ps(av, bv),
                Op::Mul => _mm512_mul_ps(av, bv),
                Op::Div => _mm512_div_ps(av, bv),
            };
            _mm512_storeu_ps(result[i..].as_mut_ptr(), rv);
        }
        i += 16;
    }
    while i < n {
        result[i] = match op {
            Op::Add => a[i] + b[i],
            Op::Sub => a[i] - b[i],
            Op::Mul => a[i] * b[i],
            Op::Div => a[i] / b[i],
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f64_s(a: &[f64], scalar: f64, op: Op) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0f64; n];
    let sv = _mm512_set1_pd(scalar);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_pd(a[i..].as_ptr());
            let rv = match op {
                Op::Add => _mm512_add_pd(av, sv),
                Op::Sub => _mm512_sub_pd(av, sv),
                Op::Mul => _mm512_mul_pd(av, sv),
                Op::Div => _mm512_div_pd(av, sv),
            };
            _mm512_storeu_pd(result[i..].as_mut_ptr(), rv);
        }
        i += 8;
    }
    while i < n {
        result[i] = match op {
            Op::Add => a[i] + scalar,
            Op::Sub => a[i] - scalar,
            Op::Mul => a[i] * scalar,
            Op::Div => a[i] / scalar,
        };
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn ew_f64_v(a: &[f64], b: &[f64], op: Op) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut result = vec![0.0f64; n];
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: bounds checked by while condition
        unsafe {
            let av = _mm512_loadu_pd(a[i..].as_ptr());
            let bv = _mm512_loadu_pd(b[i..].as_ptr());
            let rv = match op {
                Op::Add => _mm512_add_pd(av, bv),
                Op::Sub => _mm512_sub_pd(av, bv),
                Op::Mul => _mm512_mul_pd(av, bv),
                Op::Div => _mm512_div_pd(av, bv),
            };
            _mm512_storeu_pd(result[i..].as_mut_ptr(), rv);
        }
        i += 8;
    }
    while i < n {
        result[i] = match op {
            Op::Add => a[i] + b[i],
            Op::Sub => a[i] - b[i],
            Op::Mul => a[i] * b[i],
            Op::Div => a[i] / b[i],
        };
        i += 1;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// GEMM — Goto BLAS packed microkernel
// ═══════════════════════════════════════════════════════════════════

// Tile parameters for AVX-512:
// MR=6 rows × NR=16 cols → 6 zmm registers for C tile
// KC chosen to fit A_panel + B_panel + C_tile in L1 (32KB)
const SGEMM_MR: usize = 6;
const SGEMM_NR: usize = 16;
const SGEMM_KC: usize = 256; // 6*256*4 + 256*16*4 + 6*16*4 = 6K+16K+384 ≈ 22KB < 32KB L1
const SGEMM_MC: usize = 72;  // 12 micro-panels of MR=6
const SGEMM_NC: usize = 256; // 16 micro-panels of NR=16

const DGEMM_MR: usize = 6;
const DGEMM_NR: usize = 8;
const DGEMM_KC: usize = 192;
const DGEMM_MC: usize = 72;
const DGEMM_NC: usize = 128;

/// Pack a panel of A (mc×kc) into column-major MR-wide strips.
/// Layout: for each k, for each MR-block of rows, store MR contiguous values.
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
    // Remainder rows (< MR): zero-pad
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

/// Pack a panel of B (kc×nc) into row-major NR-wide strips.
/// Layout: for each k, for each NR-block of cols, store NR contiguous values.
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
    // Remainder cols (< NR): zero-pad
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
/// Uses 6 zmm accumulators (one per MR row), each holding NR=16 floats.
/// Inner loop: broadcast a[ir] from A_packed, FMA with NR-wide B_packed row.
/// This is the Goto BLAS GEBP inner kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_ukernel_6x16(
    kc: usize,
    alpha: f32,
    a_packed: &[f32], // MR * kc elements, MR-strided
    b_packed: &[f32], // kc * NR elements, NR-strided
    c: &mut [f32],    // MR rows of C (scattered by ldc)
    ldc: usize,
    mr_eff: usize,    // effective rows (may be < MR at edge)
    nr_eff: usize,    // effective cols (may be < NR at edge)
) {
    // 6 accumulators for C tile rows
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();

    // Main GEBP loop: for each k, load NR-wide B row, broadcast each A element
    for p in 0..kc {
        let b_off = p * SGEMM_NR;
        let bv = _mm512_loadu_ps(b_packed[b_off..].as_ptr());

        let a_off = p * SGEMM_MR;
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 0]), bv, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 1]), bv, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 2]), bv, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 3]), bv, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 4]), bv, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 5]), bv, c5);
    }

    // Scale by alpha
    let alpha_v = _mm512_set1_ps(alpha);
    c0 = _mm512_mul_ps(c0, alpha_v);
    c1 = _mm512_mul_ps(c1, alpha_v);
    c2 = _mm512_mul_ps(c2, alpha_v);
    c3 = _mm512_mul_ps(c3, alpha_v);
    c4 = _mm512_mul_ps(c4, alpha_v);
    c5 = _mm512_mul_ps(c5, alpha_v);

    // Store: add to C (beta already applied)
    let rows = [c0, c1, c2, c3, c4, c5];
    for ir in 0..mr_eff {
        let row_ptr = c[ir * ldc..].as_mut_ptr();
        if nr_eff == SGEMM_NR {
            let cv = _mm512_loadu_ps(row_ptr);
            _mm512_storeu_ps(row_ptr, _mm512_add_ps(cv, rows[ir]));
        } else {
            // Masked store for edge tiles
            let mask: u16 = (1u32 << nr_eff) as u16 - 1;
            let cv = _mm512_maskz_loadu_ps(mask.into(), row_ptr);
            _mm512_mask_storeu_ps(row_ptr, mask.into(), _mm512_add_ps(cv, rows[ir]));
        }
    }
}

/// Goto BLAS style blocked SGEMM with packing and AVX-512 microkernel.
///
/// C = alpha * A * B + beta * C  (beta already applied by caller)
///
/// 5-loop structure: KC → MC → NC → MR × NR microkernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    // Pack buffers — allocated once, reused across tiles
    let mut a_packed = vec![0.0f32; SGEMM_MC * SGEMM_KC];
    let mut b_packed = vec![0.0f32; SGEMM_KC * SGEMM_NC];

    // Loop 1: KC blocks
    let mut kk = 0;
    while kk < k {
        let kc = SGEMM_KC.min(k - kk);

        // Loop 2: NC blocks
        let mut jj = 0;
        while jj < n {
            let nc = SGEMM_NC.min(n - jj);

            // Pack B panel (kc × nc)
            pack_b_f32(b, ldb, kc, nc, kk, jj, &mut b_packed);

            // Loop 3: MC blocks
            let mut ii = 0;
            while ii < m {
                let mc = SGEMM_MC.min(m - ii);

                // Pack A panel (mc × kc)
                pack_a_f32(a, lda, mc, kc, ii, kk, &mut a_packed);

                // Loop 4+5: micro-tiles MR × NR
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
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 0]), bv, c0);
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
            let cv = _mm512_maskz_loadu_pd(mask.into(), row_ptr);
            _mm512_mask_storeu_pd(row_ptr, mask.into(), _mm512_add_pd(cv, rows[ir]));
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
// Binary / HDC — 4 functions
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

// ═══════════════════════════════════════════════════════════════════
// GEMM — Goto BLAS with 6×16 AVX-512 microkernel
// ═══════════════════════════════════════════════════════════════════

/// Tile constants for AVX-512 f32 GEMM.
const MR: usize = 6;
const NR: usize = 16;
const KC: usize = 256;
const MC: usize = MR * 16; // 96
const NC: usize = NR * 64; // 1024

/// Pack A panel: rows [row_start..row_start+rows], cols [col_start..col_start+cols]
/// into MR-strided contiguous layout for cache-friendly microkernel access.
///
/// Layout: for each MR-block, store cols-many groups of MR elements.
/// `packed[block * MR * cols + p * MR + ir] = A[row_start + block*MR + ir, col_start + p]`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn pack_a_f32(
    a: &[f32], lda: usize,
    row_start: usize, col_start: usize,
    rows: usize, cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    let mut i_block = 0;
    while i_block < rows {
        let mr = MR.min(rows - i_block);
        for p in 0..cols {
            for ir in 0..MR {
                if ir < mr {
                    let i = row_start + i_block + ir;
                    let j = col_start + p;
                    packed[idx] = a[i * lda + j];
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
        i_block += MR;
    }
}

/// Pack B panel: rows [row_start..row_start+rows], cols [col_start..col_start+cols]
/// into NR-strided contiguous layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn pack_b_f32(
    b: &[f32], ldb: usize,
    row_start: usize, col_start: usize,
    rows: usize, cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    let mut j_block = 0;
    while j_block < cols {
        let nr = NR.min(cols - j_block);
        for p in 0..rows {
            for jr in 0..NR {
                if jr < nr {
                    let i = row_start + p;
                    let j = col_start + j_block + jr;
                    packed[idx] = b[i * ldb + j];
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
        j_block += NR;
    }
}

/// 6×16 AVX-512 microkernel: compute a MR×NR tile of C from packed A and B.
///
/// Uses 6 zmm accumulator registers (one per row), each holding 16 f32.
/// K-loop is 2x unrolled to hide FMA latency.
/// C is updated: C[row+ir, col+jr] += alpha * acc[ir][jr]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sgemm_microkernel_6x16(
    alpha: f32,
    packed_a: &[f32],  // starts at this block's A data
    packed_b: &[f32],  // starts at this block's B data
    c: &mut [f32],
    ldc: usize,
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    kb: usize,
) {
    // 6 accumulator registers
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();

    let full_b = nr >= NR;

    // 2x-unrolled K loop
    let kb_even = kb & !1;
    let mut p = 0;
    while p < kb_even {
        // K-step 0
        let b_base0 = p * NR;
        // SAFETY: packed_b has at least kb * NR elements; p < kb_even <= kb
        let b_vec0 = unsafe {
            if full_b {
                _mm512_loadu_ps(packed_b[b_base0..].as_ptr())
            } else {
                let mut tmp = [0.0f32; NR];
                tmp[..nr].copy_from_slice(&packed_b[b_base0..b_base0 + nr]);
                _mm512_loadu_ps(tmp.as_ptr())
            }
        };

        // K-step 1
        let b_base1 = (p + 1) * NR;
        // SAFETY: same bounds reasoning
        let b_vec1 = unsafe {
            if full_b {
                _mm512_loadu_ps(packed_b[b_base1..].as_ptr())
            } else {
                let mut tmp = [0.0f32; NR];
                tmp[..nr].copy_from_slice(&packed_b[b_base1..b_base1 + nr]);
                _mm512_loadu_ps(tmp.as_ptr())
            }
        };

        let a_base0 = p * MR;
        let a_base1 = (p + 1) * MR;

        // Interleaved FMA for each row
        macro_rules! fma_row {
            ($acc:ident, $ir:expr) => {
                if $ir < mr {
                    let a0 = _mm512_set1_ps(packed_a[a_base0 + $ir]);
                    $acc = _mm512_fmadd_ps(a0, b_vec0, $acc);
                    let a1 = _mm512_set1_ps(packed_a[a_base1 + $ir]);
                    $acc = _mm512_fmadd_ps(a1, b_vec1, $acc);
                }
            };
        }
        fma_row!(acc0, 0);
        fma_row!(acc1, 1);
        fma_row!(acc2, 2);
        fma_row!(acc3, 3);
        fma_row!(acc4, 4);
        fma_row!(acc5, 5);

        p += 2;
    }

    // Handle odd K step
    if kb & 1 != 0 {
        let b_base = kb_even * NR;
        // SAFETY: kb_even < kb, so b_base + NR <= kb * NR
        let b_vec = unsafe {
            if full_b {
                _mm512_loadu_ps(packed_b[b_base..].as_ptr())
            } else {
                let mut tmp = [0.0f32; NR];
                tmp[..nr].copy_from_slice(&packed_b[b_base..b_base + nr]);
                _mm512_loadu_ps(tmp.as_ptr())
            }
        };
        let a_base = kb_even * MR;
        macro_rules! fma_tail {
            ($acc:ident, $ir:expr) => {
                if $ir < mr {
                    let a_val = _mm512_set1_ps(packed_a[a_base + $ir]);
                    $acc = _mm512_fmadd_ps(a_val, b_vec, $acc);
                }
            };
        }
        fma_tail!(acc0, 0);
        fma_tail!(acc1, 1);
        fma_tail!(acc2, 2);
        fma_tail!(acc3, 3);
        fma_tail!(acc4, 4);
        fma_tail!(acc5, 5);
    }

    // Store results: C[row+ir, col..col+nr] += alpha * acc[ir]
    let alpha_v = _mm512_set1_ps(alpha);
    let accs = [acc0, acc1, acc2, acc3, acc4, acc5];

    for ir in 0..mr.min(MR) {
        let result = _mm512_mul_ps(accs[ir], alpha_v);
        let base = (row + ir) * ldc + col;

        if nr >= NR {
            // Full-width: SIMD load + add + store
            // SAFETY: base + NR <= m * ldc + n, caller guarantees C dimensions
            unsafe {
                let existing = _mm512_loadu_ps(c[base..].as_ptr());
                let sum = _mm512_add_ps(existing, result);
                _mm512_storeu_ps(c[base..].as_mut_ptr(), sum);
            }
        } else {
            // Partial: extract to array and scalar store
            let mut buf = [0.0f32; NR];
            // SAFETY: writing to stack-local array
            unsafe { _mm512_storeu_ps(buf.as_mut_ptr(), result); }
            for jr in 0..nr {
                c[base + jr] += buf[jr];
            }
        }
    }
}

/// Macro-kernel: dispatch MR×NR microkernels over packed panels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sgemm_macrokernel(
    alpha: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    ldc: usize,
    ic: usize, jc: usize,
    mb: usize, nb: usize, kb: usize,
) {
    let mr_blocks = mb.div_ceil(MR);
    let nr_blocks = nb.div_ceil(NR);

    for jr in 0..nr_blocks {
        let nr = NR.min(nb - jr * NR);
        for ir in 0..mr_blocks {
            let mr = MR.min(mb - ir * MR);
            let a_offset = ir * MR * kb;
            let b_offset = jr * NR * kb;
            sgemm_microkernel_6x16(
                alpha,
                &packed_a[a_offset..],
                &packed_b[b_offset..],
                c, ldc,
                ic + ir * MR,
                jc + jr * NR,
                mr, nr, kb,
            );
        }
    }
}

/// Cache-blocked GEMM: C = alpha * A * B + beta * C (f32, row-major).
///
/// Uses Goto BLAS algorithm:
/// 1. Tile the K dimension into KC-wide panels
/// 2. Pack A and B into contiguous cache-friendly buffers
/// 3. Dispatch 6×16 AVX-512 microkernels
///
/// The beta scaling of C must be done by the caller before invoking this.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let mc_padded = mc.div_ceil(MR) * MR;
    let nc_padded = nc.div_ceil(NR) * NR;

    let mut packed_b = vec![0.0f32; kc * nc_padded];
    let mut packed_a = vec![0.0f32; mc_padded * kc];

    let mut jc = 0;
    while jc < n {
        let jb = nc.min(n - jc);
        let mut pc = 0;
        while pc < k {
            let pb = kc.min(k - pc);
            pack_b_f32(b, ldb, pc, jc, pb, jb, &mut packed_b);
            let mut ic = 0;
            while ic < m {
                let ib = mc.min(m - ic);
                pack_a_f32(a, lda, ic, pc, ib, pb, &mut packed_a);
                sgemm_macrokernel(
                    alpha, &packed_a, &packed_b,
                    c, ldc, ic, jc, ib, jb, pb,
                );
                ic += mc;
            }
            pc += kc;
        }
        jc += nc;
    }
}

// ═══════════════════════════════════════════════════════════════════
// DGEMM — Goto BLAS with 6×8 AVX-512 microkernel (f64)
// ═══════════════════════════════════════════════════════════════════

const DMR: usize = 6;
const DNR: usize = 8;
const DKC: usize = 256;
const DMC: usize = DMR * 16; // 96
const DNC: usize = DNR * 64; // 512

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn pack_a_f64(
    a: &[f64], lda: usize,
    row_start: usize, col_start: usize,
    rows: usize, cols: usize,
    packed: &mut [f64],
) {
    let mut idx = 0;
    let mut i_block = 0;
    while i_block < rows {
        let mr = DMR.min(rows - i_block);
        for p in 0..cols {
            for ir in 0..DMR {
                if ir < mr {
                    packed[idx] = a[(row_start + i_block + ir) * lda + col_start + p];
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
        i_block += DMR;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn pack_b_f64(
    b: &[f64], ldb: usize,
    row_start: usize, col_start: usize,
    rows: usize, cols: usize,
    packed: &mut [f64],
) {
    let mut idx = 0;
    let mut j_block = 0;
    while j_block < cols {
        let nr = DNR.min(cols - j_block);
        for p in 0..rows {
            for jr in 0..DNR {
                if jr < nr {
                    packed[idx] = b[(row_start + p) * ldb + col_start + j_block + jr];
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
        j_block += DNR;
    }
}

/// 6×8 AVX-512 microkernel for f64: 6 rows × 8 doubles per zmm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dgemm_microkernel_6x8(
    alpha: f64,
    packed_a: &[f64],
    packed_b: &[f64],
    c: &mut [f64],
    ldc: usize,
    row: usize, col: usize,
    mr: usize, nr: usize, kb: usize,
) {
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();
    let mut acc2 = _mm512_setzero_pd();
    let mut acc3 = _mm512_setzero_pd();
    let mut acc4 = _mm512_setzero_pd();
    let mut acc5 = _mm512_setzero_pd();
    let full_b = nr >= DNR;

    let kb_even = kb & !1;
    let mut p = 0;
    while p < kb_even {
        let b0 = unsafe {
            if full_b { _mm512_loadu_pd(packed_b[p * DNR..].as_ptr()) }
            else { let mut t = [0.0f64; DNR]; t[..nr].copy_from_slice(&packed_b[p*DNR..p*DNR+nr]); _mm512_loadu_pd(t.as_ptr()) }
        };
        let b1 = unsafe {
            if full_b { _mm512_loadu_pd(packed_b[(p+1) * DNR..].as_ptr()) }
            else { let mut t = [0.0f64; DNR]; t[..nr].copy_from_slice(&packed_b[(p+1)*DNR..(p+1)*DNR+nr]); _mm512_loadu_pd(t.as_ptr()) }
        };
        let a0 = p * DMR;
        let a1 = (p + 1) * DMR;
        macro_rules! fma { ($acc:ident, $ir:expr) => { if $ir < mr {
            $acc = _mm512_fmadd_pd(_mm512_set1_pd(packed_a[a0+$ir]), b0, $acc);
            $acc = _mm512_fmadd_pd(_mm512_set1_pd(packed_a[a1+$ir]), b1, $acc);
        }}}
        fma!(acc0,0); fma!(acc1,1); fma!(acc2,2); fma!(acc3,3); fma!(acc4,4); fma!(acc5,5);
        p += 2;
    }
    if kb & 1 != 0 {
        let bv = unsafe {
            if full_b { _mm512_loadu_pd(packed_b[kb_even*DNR..].as_ptr()) }
            else { let mut t = [0.0f64; DNR]; t[..nr].copy_from_slice(&packed_b[kb_even*DNR..kb_even*DNR+nr]); _mm512_loadu_pd(t.as_ptr()) }
        };
        let ab = kb_even * DMR;
        macro_rules! ft { ($acc:ident, $ir:expr) => { if $ir < mr {
            $acc = _mm512_fmadd_pd(_mm512_set1_pd(packed_a[ab+$ir]), bv, $acc);
        }}}
        ft!(acc0,0); ft!(acc1,1); ft!(acc2,2); ft!(acc3,3); ft!(acc4,4); ft!(acc5,5);
    }

    let alpha_v = _mm512_set1_pd(alpha);
    let accs = [acc0, acc1, acc2, acc3, acc4, acc5];
    for ir in 0..mr.min(DMR) {
        let result = _mm512_mul_pd(accs[ir], alpha_v);
        let base = (row + ir) * ldc + col;
        if nr >= DNR {
            unsafe {
                let existing = _mm512_loadu_pd(c[base..].as_ptr());
                _mm512_storeu_pd(c[base..].as_mut_ptr(), _mm512_add_pd(existing, result));
            }
        } else {
            let mut buf = [0.0f64; DNR];
            unsafe { _mm512_storeu_pd(buf.as_mut_ptr(), result); }
            for jr in 0..nr { c[base + jr] += buf[jr]; }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub fn dgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f64,
    a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    c: &mut [f64], ldc: usize,
) {
    let mc = DMC.min(m);
    let nc = DNC.min(n);
    let kc = DKC.min(k);
    let mc_padded = mc.div_ceil(DMR) * DMR;
    let nc_padded = nc.div_ceil(DNR) * DNR;
    let mut packed_b = vec![0.0f64; kc * nc_padded];
    let mut packed_a = vec![0.0f64; mc_padded * kc];

    let mut jc = 0;
    while jc < n {
        let jb = nc.min(n - jc);
        let mut pc = 0;
        while pc < k {
            let pb = kc.min(k - pc);
            pack_b_f64(b, ldb, pc, jc, pb, jb, &mut packed_b);
            let mut ic = 0;
            while ic < m {
                let ib = mc.min(m - ic);
                pack_a_f64(a, lda, ic, pc, ib, pb, &mut packed_a);
                // macrokernel
                let mr_blocks = ib.div_ceil(DMR);
                let nr_blocks = jb.div_ceil(DNR);
                for jr in 0..nr_blocks {
                    let nr = DNR.min(jb - jr * DNR);
                    for ir in 0..mr_blocks {
                        let mr = DMR.min(ib - ir * DMR);
                        dgemm_microkernel_6x8(
                            alpha,
                            &packed_a[ir * DMR * pb..],
                            &packed_b[jr * DNR * pb..],
                            c, ldc,
                            ic + ir * DMR, jc + jr * DNR,
                            mr, nr, pb,
                        );
                    }
                }
                ic += mc;
            }
            pc += kc;
        }
        jc += nc;
    }
}
