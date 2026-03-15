//! SIMD dispatch: detect once, dispatch forever.
//!
//! One `LazyLock` detects the CPU tier at first call.
//! Every function is one line: `dispatch!(name(args) -> ret);`
//! Adding a new function = adding one line. That's it.
//!
//! Tiers: AVX-512 → AVX2+FMA → Scalar

#![allow(clippy::too_many_arguments)]

use std::sync::LazyLock;

// ─── Tier detection: happens ONCE, at first access ─────────────────

#[derive(Clone, Copy, PartialEq)]
enum Tier { Avx512, Avx2, Scalar }

static TIER: LazyLock<Tier> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { return Tier::Avx512; }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Tier::Avx2;
        }
    }
    Tier::Scalar
});

#[inline(always)]
fn tier() -> Tier { *TIER }

// ─── Runtime GEMM tile constants ───────────────────────────────────

/// SGEMM register-block width, selected by tier:
/// AVX-512 = 16, AVX2 = 8, Scalar = 4.
pub fn sgemm_nr() -> usize {
    match tier() {
        Tier::Avx512 => 16,
        Tier::Avx2   => 8,
        Tier::Scalar => 4,
    }
}

/// DGEMM register-block width, selected by tier.
pub fn dgemm_nr() -> usize {
    match tier() {
        Tier::Avx512 => 8,
        Tier::Avx2   => 4,
        Tier::Scalar => 4,
    }
}

// ─── The macro: one line per function ──────────────────────────────

/// Dispatch macro: generates a `pub fn` that matches on tier().
///
/// For functions where AVX-512 isn't implemented yet, we fall back to
/// AVX2 (or scalar). The `{ avx512_path, avx2_path, scalar_path }` form
/// lets you specify custom paths per tier.
macro_rules! dispatch {
    // Same name in avx2 and scalar modules (no avx512 yet → falls to avx2)
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) -> $ret {
            match tier() {
                // No AVX-512 kernels yet — fall through to AVX2
                Tier::Avx512 | Tier::Avx2 => avx2::$name($($arg),*),
                Tier::Scalar => scalar::$name($($arg),*),
            }
        }
    };
    // Same name, no return type (returns ())
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? )
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) {
            match tier() {
                Tier::Avx512 | Tier::Avx2 => avx2::$name($($arg),*),
                Tier::Scalar => scalar::$name($($arg),*),
            }
        }
    };
    // Custom: different paths per tier
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
        { $a512:expr, $a2:expr, $sc:expr }
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) -> $ret {
            match tier() {
                Tier::Avx512 => $a512($($arg),*),
                Tier::Avx2   => $a2($($arg),*),
                Tier::Scalar => $sc($($arg),*),
            }
        }
    };
    // Custom, no return type
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? )
        { $a512:expr, $a2:expr, $sc:expr }
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) {
            match tier() {
                Tier::Avx512 => $a512($($arg),*),
                Tier::Avx2   => $a2($($arg),*),
                Tier::Scalar => $sc($($arg),*),
            }
        }
    };
}

// ─── BLAS-1 dispatch ──────────────────────────────────────────────

dispatch!(
    /// Dot product: result = sum(x[i] * y[i]) (f32).
    dot_f32(x: &[f32], y: &[f32]) -> f32);
dispatch!(
    /// Dot product: result = sum(x[i] * y[i]) (f64).
    dot_f64(x: &[f64], y: &[f64]) -> f64);
dispatch!(
    /// AXPY: y = alpha * x + y (f32).
    axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]));
dispatch!(
    /// AXPY: y = alpha * x + y (f64).
    axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]));

// scal/nrm2/asum: no AVX2 impl yet, scalar for all tiers
dispatch!(
    /// Scale: x = alpha * x (f32).
    scal_f32(alpha: f32, x: &mut [f32])
    { scalar::scal_f32, scalar::scal_f32, scalar::scal_f32 });
dispatch!(
    /// Scale: x = alpha * x (f64).
    scal_f64(alpha: f64, x: &mut [f64])
    { scalar::scal_f64, scalar::scal_f64, scalar::scal_f64 });
dispatch!(
    /// L2 norm: sqrt(sum(x[i]^2)) (f32).
    nrm2_f32(x: &[f32]) -> f32
    { scalar::nrm2_f32, scalar::nrm2_f32, scalar::nrm2_f32 });
dispatch!(
    /// L2 norm: sqrt(sum(x[i]^2)) (f64).
    nrm2_f64(x: &[f64]) -> f64
    { scalar::nrm2_f64, scalar::nrm2_f64, scalar::nrm2_f64 });
dispatch!(
    /// L1 norm: sum(|x[i]|) (f32).
    asum_f32(x: &[f32]) -> f32
    { scalar::asum_f32, scalar::asum_f32, scalar::asum_f32 });
dispatch!(
    /// L1 norm: sum(|x[i]|) (f64).
    asum_f64(x: &[f64]) -> f64
    { scalar::asum_f64, scalar::asum_f64, scalar::asum_f64 });

// ─── GEMM dispatch ───────────────────────────────────────────────

/// GEMM: C = alpha * A * B + beta * C (f32, tiled)
pub fn gemm_f32(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32, c: &mut [f32], ldc: usize,
) {
    scalar::gemm_f32_tiled(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// GEMM: C = alpha * A * B + beta * C (f64, tiled)
pub fn gemm_f64(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    beta: f64, c: &mut [f64], ldc: usize,
) {
    scalar::gemm_f64_tiled(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ─── GEMV dispatch ───────────────────────────────────────────────

/// GEMV: y = alpha * A * x + beta * y (f32)
pub fn gemv_f32(
    m: usize, n: usize,
    alpha: f32, a: &[f32], lda: usize,
    x: &[f32], beta: f32, y: &mut [f32],
) {
    scalar::gemv_f32(m, n, alpha, a, lda, x, beta, y);
}

/// GEMV: y = alpha * A * x + beta * y (f64)
pub fn gemv_f64(
    m: usize, n: usize,
    alpha: f64, a: &[f64], lda: usize,
    x: &[f64], beta: f64, y: &mut [f64],
) {
    scalar::gemv_f64(m, n, alpha, a, lda, x, beta, y);
}

// ═══════════════════════════════════════════════════════════════════
// Scalar fallback implementations
// ═══════════════════════════════════════════════════════════════════

mod scalar {
    #![allow(clippy::too_many_arguments)]

    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len().min(y.len());
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 4 <= n {
            sum += x[i] * y[i] + x[i + 1] * y[i + 1]
                + x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3];
            i += 4;
        }
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        sum
    }

    pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        let mut sum = 0.0f64;
        let mut i = 0;
        while i + 4 <= n {
            sum += x[i] * y[i] + x[i + 1] * y[i + 1]
                + x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3];
            i += 4;
        }
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        sum
    }

    pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
        let n = x.len().min(y.len());
        for i in 0..n {
            y[i] += alpha * x[i];
        }
    }

    pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
        let n = x.len().min(y.len());
        for i in 0..n {
            y[i] += alpha * x[i];
        }
    }

    pub fn scal_f32(alpha: f32, x: &mut [f32]) {
        for v in x.iter_mut() {
            *v *= alpha;
        }
    }

    pub fn scal_f64(alpha: f64, x: &mut [f64]) {
        for v in x.iter_mut() {
            *v *= alpha;
        }
    }

    pub fn nrm2_f32(x: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &v in x {
            sum += v * v;
        }
        sum.sqrt()
    }

    pub fn nrm2_f64(x: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        for &v in x {
            sum += v * v;
        }
        sum.sqrt()
    }

    pub fn asum_f32(x: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &v in x {
            sum += v.abs();
        }
        sum
    }

    pub fn asum_f64(x: &[f64]) -> f64 {
        let mut sum = 0.0f64;
        for &v in x {
            sum += v.abs();
        }
        sum
    }

    /// Tiled GEMM: C = alpha * A * B + beta * C
    pub fn gemm_f32_tiled(
        m: usize, n: usize, k: usize,
        alpha: f32, a: &[f32], lda: usize,
        b: &[f32], ldb: usize,
        beta: f32, c: &mut [f32], ldc: usize,
    ) {
        const TILE: usize = 64;

        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] *= beta;
                }
            }
        }

        let mut kk = 0;
        while kk < k {
            let kb = TILE.min(k - kk);
            let mut ii = 0;
            while ii < m {
                let ib = TILE.min(m - ii);
                let mut jj = 0;
                while jj < n {
                    let jb = TILE.min(n - jj);
                    for i in 0..ib {
                        for p in 0..kb {
                            let a_val = alpha * a[(ii + i) * lda + (kk + p)];
                            for j in 0..jb {
                                c[(ii + i) * ldc + (jj + j)] += a_val * b[(kk + p) * ldb + (jj + j)];
                            }
                        }
                    }
                    jj += jb;
                }
                ii += ib;
            }
            kk += kb;
        }
    }

    /// Tiled GEMM (f64)
    pub fn gemm_f64_tiled(
        m: usize, n: usize, k: usize,
        alpha: f64, a: &[f64], lda: usize,
        b: &[f64], ldb: usize,
        beta: f64, c: &mut [f64], ldc: usize,
    ) {
        const TILE: usize = 64;

        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] *= beta;
                }
            }
        }

        let mut kk = 0;
        while kk < k {
            let kb = TILE.min(k - kk);
            let mut ii = 0;
            while ii < m {
                let ib = TILE.min(m - ii);
                let mut jj = 0;
                while jj < n {
                    let jb = TILE.min(n - jj);
                    for i in 0..ib {
                        for p in 0..kb {
                            let a_val = alpha * a[(ii + i) * lda + (kk + p)];
                            for j in 0..jb {
                                c[(ii + i) * ldc + (jj + j)] += a_val * b[(kk + p) * ldb + (jj + j)];
                            }
                        }
                    }
                    jj += jb;
                }
                ii += ib;
            }
            kk += kb;
        }
    }

    pub fn gemv_f32(
        m: usize, n: usize,
        alpha: f32, a: &[f32], lda: usize,
        x: &[f32], beta: f32, y: &mut [f32],
    ) {
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += a[i * lda + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    }

    pub fn gemv_f64(
        m: usize, n: usize,
        alpha: f64, a: &[f64], lda: usize,
        x: &[f64], beta: f64, y: &mut [f64],
    ) {
        for i in 0..m {
            let mut sum = 0.0f64;
            for j in 0..n {
                sum += a[i * lda + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// AVX2+FMA implementations
// ═══════════════════════════════════════════════════════════════════

mod avx2 {
    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: tier() already verified AVX2+FMA support before calling
            unsafe { dot_f32_avx2(x, y) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::scalar::dot_f32(x, y)
        }
    }

    pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: tier() already verified AVX2+FMA support before calling
            unsafe { dot_f64_avx2(x, y) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::scalar::dot_f64(x, y)
        }
    }

    pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: tier() already verified AVX2+FMA support before calling
            unsafe { axpy_f32_avx2(alpha, x, y) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::scalar::axpy_f32(alpha, x, y)
        }
    }

    pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: tier() already verified AVX2+FMA support before calling
            unsafe { axpy_f64_avx2(alpha, x, y) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::scalar::axpy_f64(alpha, x, y)
        }
    }

    // ── AVX2 intrinsic implementations ─────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_f32_avx2(x: &[f32], y: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let n = x.len().min(y.len());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(x0, y0, acc0);
            let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(x1, y1, acc1);
            i += 16;
        }
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(xv, yv, acc0);
            i += 8;
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut total = _mm_cvtss_f32(result);
        while i < n {
            total += x[i] * y[i];
            i += 1;
        }
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_f64_avx2(x: &[f64], y: &[f64]) -> f64 {
        use core::arch::x86_64::*;
        let n = x.len().min(y.len());
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
            let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
            acc0 = _mm256_fmadd_pd(x0, y0, acc0);
            let x1 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
            let y1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
            acc1 = _mm256_fmadd_pd(x1, y1, acc1);
            i += 8;
        }
        while i + 4 <= n {
            let xv = _mm256_loadu_pd(x.as_ptr().add(i));
            let yv = _mm256_loadu_pd(y.as_ptr().add(i));
            acc0 = _mm256_fmadd_pd(xv, yv, acc0);
            i += 4;
        }
        acc0 = _mm256_add_pd(acc0, acc1);
        let hi = _mm256_extractf128_pd(acc0, 1);
        let lo = _mm256_castpd256_pd128(acc0);
        let sum128 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let result = _mm_add_sd(sum128, hi64);
        let mut total = _mm_cvtsd_f64(result);
        while i < n {
            total += x[i] * y[i];
            i += 1;
        }
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn axpy_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
        use core::arch::x86_64::*;
        let n = x.len().min(y.len());
        let alpha_v = _mm256_set1_ps(alpha);
        let mut i = 0;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            let result = _mm256_fmadd_ps(alpha_v, xv, yv);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }
        while i < n {
            y[i] += alpha * x[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn axpy_f64_avx2(alpha: f64, x: &[f64], y: &mut [f64]) {
        use core::arch::x86_64::*;
        let n = x.len().min(y.len());
        let alpha_v = _mm256_set1_pd(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let xv = _mm256_loadu_pd(x.as_ptr().add(i));
            let yv = _mm256_loadu_pd(y.as_ptr().add(i));
            let result = _mm256_fmadd_pd(alpha_v, xv, yv);
            _mm256_storeu_pd(y.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < n {
            y[i] += alpha * x[i];
            i += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests — use public dispatch functions directly
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = dot_f32(&x, &y);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_f64() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        let y = vec![5.0f64, 6.0, 7.0, 8.0];
        let result = dot_f64(&x, &y);
        assert!((result - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_f32() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![4.0f32, 5.0, 6.0];
        axpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        scal_f32(3.0, &mut x);
        assert_eq!(x, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_nrm2_f64() {
        let x = vec![3.0f64, 4.0];
        let result = nrm2_f64(&x);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_f32() {
        let x = vec![-1.0f32, 2.0, -3.0];
        let result = asum_f32(&x);
        assert!((result - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        gemm_f32(2, 2, 3, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![5.0f32, 6.0];
        let mut y = vec![0.0f32; 2];
        gemv_f32(2, 2, 1.0, &a, 2, &x, 0.0, &mut y);
        assert!((y[0] - 17.0).abs() < 1e-5);
        assert!((y[1] - 39.0).abs() < 1e-5);
    }

    #[test]
    fn test_sgemm_nr_runtime() {
        let nr = sgemm_nr();
        // Should be one of the valid tier values
        assert!(nr == 4 || nr == 8 || nr == 16);
    }
}
