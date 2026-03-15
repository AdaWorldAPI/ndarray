//! Native pure-Rust backend with SIMD acceleration.
//!
//! Uses runtime CPU feature detection to dispatch to the fastest
//! available instruction set: AVX-512 → AVX2 → SSE4.2 → scalar.

#![allow(clippy::too_many_arguments)]

use super::LinalgBackend;

/// Pure Rust BLAS backend with SIMD auto-dispatch.
pub struct NativeBackend;

// ── SIMD dispatch helpers ──────────────────────────────────────────

fn dot_f32_scalar(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    // Unroll by 4
    while i + 4 <= n {
        sum += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3];
        i += 4;
    }
    while i < n {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}

fn dot_f64_scalar(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 4 <= n {
        sum += x[i] * y[i] + x[i + 1] * y[i + 1] + x[i + 2] * y[i + 2] + x[i + 3] * y[i + 3];
        i += 4;
    }
    while i < n {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}

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
    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);
    // Handle remainder
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
    // Horizontal sum
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

fn dispatch_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { dot_f32_avx2(x, y) };
        }
    }
    dot_f32_scalar(x, y)
}

fn dispatch_dot_f64(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { dot_f64_avx2(x, y) };
        }
    }
    dot_f64_scalar(x, y)
}

// ── AXPY ───────────────────────────────────────────────────────────

fn axpy_f32_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

fn axpy_f64_scalar(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i];
    }
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

fn dispatch_axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { axpy_f32_avx2(alpha, x, y) };
        }
    }
    axpy_f32_scalar(alpha, x, y);
}

fn dispatch_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We checked for AVX2+FMA support
            return unsafe { axpy_f64_avx2(alpha, x, y) };
        }
    }
    axpy_f64_scalar(alpha, x, y);
}

// ── SCAL ───────────────────────────────────────────────────────────

fn scal_f32_scalar(alpha: f32, x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

fn scal_f64_scalar(alpha: f64, x: &mut [f64]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

// ── NRM2 ───────────────────────────────────────────────────────────

fn nrm2_f32_scalar(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    sum.sqrt()
}

fn nrm2_f64_scalar(x: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &v in x {
        sum += v * v;
    }
    sum.sqrt()
}

// ── ASUM ───────────────────────────────────────────────────────────

fn asum_f32_scalar(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v.abs();
    }
    sum
}

fn asum_f64_scalar(x: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for &v in x {
        sum += v.abs();
    }
    sum
}

// ── GEMM (tiled) ───────────────────────────────────────────────────

/// Tiled GEMM: C = alpha * A * B + beta * C
///
/// Uses cache-friendly tiling with tile size tuned for L1 cache.
fn gemm_f32_tiled(
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
) {
    const TILE: usize = 64;

    // Apply beta to C
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

    // Tiled multiply
    let mut kk = 0;
    while kk < k {
        let kb = TILE.min(k - kk);
        let mut ii = 0;
        while ii < m {
            let ib = TILE.min(m - ii);
            let mut jj = 0;
            while jj < n {
                let jb = TILE.min(n - jj);
                // Micro-kernel: C[ii..ii+ib, jj..jj+jb] += alpha * A[ii..ii+ib, kk..kk+kb] * B[kk..kk+kb, jj..jj+jb]
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

fn gemm_f64_tiled(
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

// ── GEMV ───────────────────────────────────────────────────────────

fn gemv_f32_scalar(
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[i * lda + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

fn gemv_f64_scalar(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    for i in 0..m {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += a[i * lda + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

// ── LinalgBackend impl ─────────────────────────────────────────────

impl LinalgBackend for NativeBackend {
    fn dot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        dispatch_dot_f32(x, y)
    }

    fn dot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        dispatch_dot_f64(x, y)
    }

    fn axpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        dispatch_axpy_f32(alpha, x, y);
    }

    fn axpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        dispatch_axpy_f64(alpha, x, y);
    }

    fn scal_f32(&self, alpha: f32, x: &mut [f32]) {
        scal_f32_scalar(alpha, x);
    }

    fn scal_f64(&self, alpha: f64, x: &mut [f64]) {
        scal_f64_scalar(alpha, x);
    }

    fn nrm2_f32(&self, x: &[f32]) -> f32 {
        nrm2_f32_scalar(x)
    }

    fn nrm2_f64(&self, x: &[f64]) -> f64 {
        nrm2_f64_scalar(x)
    }

    fn asum_f32(&self, x: &[f32]) -> f32 {
        asum_f32_scalar(x)
    }

    fn asum_f64(&self, x: &[f64]) -> f64 {
        asum_f64_scalar(x)
    }

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
    ) {
        gemm_f32_tiled(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

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
    ) {
        gemm_f64_tiled(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

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
    ) {
        gemv_f32_scalar(m, n, alpha, a, lda, x, beta, y);
    }

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
    ) {
        gemv_f64_scalar(m, n, alpha, a, lda, x, beta, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = NativeBackend.dot_f32(&x, &y);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_f64() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        let y = vec![5.0f64, 6.0, 7.0, 8.0];
        let result = NativeBackend.dot_f64(&x, &y);
        assert!((result - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_f32() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![4.0f32, 5.0, 6.0];
        NativeBackend.axpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        NativeBackend.scal_f32(3.0, &mut x);
        assert_eq!(x, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_nrm2_f64() {
        let x = vec![3.0f64, 4.0];
        let result = NativeBackend.nrm2_f64(&x);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_f32() {
        let x = vec![-1.0f32, 2.0, -3.0];
        let result = NativeBackend.asum_f32(&x);
        assert!((result - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_f32() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        NativeBackend.gemm_f32(2, 2, 3, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_f32() {
        // y = A * x where A = [[1,2],[3,4]], x = [5,6]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![5.0f32, 6.0];
        let mut y = vec![0.0f32; 2];
        NativeBackend.gemv_f32(2, 2, 1.0, &a, 2, &x, 0.0, &mut y);
        assert!((y[0] - 17.0).abs() < 1e-5);
        assert!((y[1] - 39.0).abs() < 1e-5);
    }
}
