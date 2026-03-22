//! AVX2 SIMD primitives (256-bit): f32x8, f64x4, u8x32.
//!
//! Same API surface as simd.rs (AVX-512) but with half-width vectors.
//! Selected at compile time via `--features avx2 --no-default-features`.
//!
//! Targets: Intel Meteor Lake (U9 185H), Alder Lake, AMD Zen 2+, etc.
//! These CPUs have AVX2 + AVX-VNNI (256-bit) but no AVX-512.

use crate::simd_avx512::{f32x8, f64x4};

// ============================================================================
// AVX2 lane counts (half of AVX-512)
// ============================================================================

pub const F32_LANES: usize = 8;
pub const F64_LANES: usize = 4;
pub const U8_LANES: usize = 32;

// ============================================================================
// GEMM microkernel tile sizes for AVX2
// ============================================================================

/// GEMM microkernel: 6 rows x 8 columns (f32x8).
pub const SGEMM_MR: usize = 6;
pub const SGEMM_NR: usize = 8;

/// DGEMM microkernel: 4 rows x 4 columns (f64x4).
pub const DGEMM_MR: usize = 4;
pub const DGEMM_NR: usize = 4;

// ============================================================================
// Cache blocking parameters (same cache hierarchy, smaller tiles)
// ============================================================================

pub const L1_BLOCK: usize = 8192;
pub const L2_BLOCK: usize = 65536;
pub const L3_BLOCK: usize = 2_097_152;

pub const SGEMM_KC: usize = 256;
pub const SGEMM_MC: usize = 128;
pub const SGEMM_NC: usize = 2048;

pub const DGEMM_KC: usize = 256;
pub const DGEMM_MC: usize = 96;
pub const DGEMM_NC: usize = 1024;

// ============================================================================
// SIMD dot product (AVX2: f32x8, 4x unrolled)
// ============================================================================

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F32_LANES;
        acc0 += f32x8::from_slice(&a[base..]) * f32x8::from_slice(&b[base..]);
        acc1 +=
            f32x8::from_slice(&a[base + F32_LANES..]) * f32x8::from_slice(&b[base + F32_LANES..]);
        acc2 += f32x8::from_slice(&a[base + 2 * F32_LANES..])
            * f32x8::from_slice(&b[base + 2 * F32_LANES..]);
        acc3 += f32x8::from_slice(&a[base + 3 * F32_LANES..])
            * f32x8::from_slice(&b[base + 3 * F32_LANES..]);
    }

    for i in (full_iters * 4)..chunks {
        let base = i * F32_LANES;
        acc0 += f32x8::from_slice(&a[base..]) * f32x8::from_slice(&b[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F64_LANES;

    let mut acc0 = f64x4::splat(0.0);
    let mut acc1 = f64x4::splat(0.0);
    let mut acc2 = f64x4::splat(0.0);
    let mut acc3 = f64x4::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F64_LANES;
        acc0 += f64x4::from_slice(&a[base..]) * f64x4::from_slice(&b[base..]);
        acc1 +=
            f64x4::from_slice(&a[base + F64_LANES..]) * f64x4::from_slice(&b[base + F64_LANES..]);
        acc2 += f64x4::from_slice(&a[base + 2 * F64_LANES..])
            * f64x4::from_slice(&b[base + 2 * F64_LANES..]);
        acc3 += f64x4::from_slice(&a[base + 3 * F64_LANES..])
            * f64x4::from_slice(&b[base + 3 * F64_LANES..]);
    }

    for i in (full_iters * 4)..chunks {
        let base = i * F64_LANES;
        acc0 += f64x4::from_slice(&a[base..]) * f64x4::from_slice(&b[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// SIMD axpy, scal, asum, nrm2 (AVX2)
// ============================================================================

#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        let mut yv = f32x8::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x4::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
        let mut yv = f64x4::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        (alpha_v * xv).copy_to_slice(&mut x[base..base + F32_LANES]);
    }
    for v in x[chunks * F32_LANES..].iter_mut() {
        *v *= alpha;
    }
}

#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x4::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
        (alpha_v * xv).copy_to_slice(&mut x[base..base + F64_LANES]);
    }
    for v in x[chunks * F64_LANES..].iter_mut() {
        *v *= alpha;
    }
}

#[inline]
pub fn asum_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        acc += f32x8::from_slice(&x[base..]).abs();
    }

    let mut sum = acc.reduce_sum();
    for &v in &x[chunks * F32_LANES..] {
        sum += v.abs();
    }
    sum
}

#[inline]
pub fn asum_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x4::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        acc += f64x4::from_slice(&x[base..]).abs();
    }

    let mut sum = acc.reduce_sum();
    for &v in &x[chunks * F64_LANES..] {
        sum += v.abs();
    }
    sum
}

#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        acc += xv * xv;
    }

    let mut sum = acc.reduce_sum();
    for &v in &x[chunks * F32_LANES..] {
        sum += v * v;
    }
    sum.sqrt()
}

#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x4::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
        acc += xv * xv;
    }

    let mut sum = acc.reduce_sum();
    for &v in &x[chunks * F64_LANES..] {
        sum += v * v;
    }
    sum.sqrt()
}

// ============================================================================
// Hamming distance (portable — no VPOPCNTDQ on AVX2 hardware)
// ============================================================================

/// Hamming distance between two byte arrays (number of differing bits).
///
/// Uses scalar POPCNT on u64 chunks (~4x faster than byte-by-byte).
/// AVX2 hardware lacks VPOPCNTDQ, so this is the fast path.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let u64_chunks = len / 8;
    let mut sum: u64 = 0;

    for i in 0..u64_chunks {
        let base = i * 8;
        let a_u64 = u64::from_le_bytes([
            a[base],
            a[base + 1],
            a[base + 2],
            a[base + 3],
            a[base + 4],
            a[base + 5],
            a[base + 6],
            a[base + 7],
        ]);
        let b_u64 = u64::from_le_bytes([
            b[base],
            b[base + 1],
            b[base + 2],
            b[base + 3],
            b[base + 4],
            b[base + 5],
            b[base + 6],
            b[base + 7],
        ]);
        sum += (a_u64 ^ b_u64).count_ones() as u64;
    }

    for i in (u64_chunks * 8)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }

    sum
}

/// Batch Hamming distance: compute distances from `query` to each row in `database`.
#[inline]
pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    assert_eq!(query.len(), row_bytes);
    assert_eq!(database.len(), num_rows * row_bytes);

    let mut distances = vec![0u64; num_rows];

    let full = num_rows / 4;
    for i in 0..full {
        let base = i * 4;
        distances[base] =
            hamming_distance(query, &database[base * row_bytes..(base + 1) * row_bytes]);
        distances[base + 1] = hamming_distance(
            query,
            &database[(base + 1) * row_bytes..(base + 2) * row_bytes],
        );
        distances[base + 2] = hamming_distance(
            query,
            &database[(base + 2) * row_bytes..(base + 3) * row_bytes],
        );
        distances[base + 3] = hamming_distance(
            query,
            &database[(base + 3) * row_bytes..(base + 4) * row_bytes],
        );
    }
    for i in (full * 4)..num_rows {
        distances[i] = hamming_distance(query, &database[i * row_bytes..(i + 1) * row_bytes]);
    }

    distances
}

/// Top-k nearest neighbors by Hamming distance.
pub fn hamming_top_k(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    let distances = hamming_batch(query, database, num_rows, row_bytes);
    let k = k.min(num_rows);

    let mut indices: Vec<usize> = (0..num_rows).collect();
    indices.select_nth_unstable_by_key(k.saturating_sub(1), |&i| distances[i]);
    indices.truncate(k);
    indices.sort_unstable_by_key(|&i| distances[i]);

    let top_distances: Vec<u64> = indices.iter().map(|&i| distances[i]).collect();
    (indices, top_distances)
}

/// AVX2 popcount using Harley-Seal vpshufb nibble lookup.
pub fn popcount(a: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            let len = a.len();
            let chunks = len / 32;
            let low_mask = _mm256_set1_epi8(0x0f);
            let lookup = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            );
            let mut total = _mm256_setzero_si256();
            let blocks = chunks / 8;
            for block in 0..blocks {
                let mut local = _mm256_setzero_si256();
                for i in 0..8 {
                    let idx = (block * 8 + i) * 32;
                    let v = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(v, low_mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
                    let cnt = _mm256_add_epi8(
                        _mm256_shuffle_epi8(lookup, lo),
                        _mm256_shuffle_epi8(lookup, hi),
                    );
                    local = _mm256_add_epi8(local, cnt);
                }
                total = _mm256_add_epi64(total, _mm256_sad_epu8(local, _mm256_setzero_si256()));
            }
            if blocks * 8 < chunks {
                let mut local = _mm256_setzero_si256();
                for i in blocks * 8..chunks {
                    let idx = i * 32;
                    let v = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(v, low_mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
                    let cnt = _mm256_add_epi8(
                        _mm256_shuffle_epi8(lookup, lo),
                        _mm256_shuffle_epi8(lookup, hi),
                    );
                    local = _mm256_add_epi8(local, cnt);
                }
                total = _mm256_add_epi64(total, _mm256_sad_epu8(local, _mm256_setzero_si256()));
            }
            let arr: [i64; 4] = std::mem::transmute(total);
            let mut sum: u64 = arr.iter().map(|&v| v as u64).sum();
            for &byte in &a[chunks * 32..] {
                sum += byte.count_ones() as u64;
            }
            sum
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter().map(|&b| b.count_ones() as u64).sum()
    }
}

/// AVX2 int8 dot product using VPMADDUBSW + VPMADDWD with XOR-0x80 bias correction.
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            let len = a.len();
            let chunks = len / 32;
            let bias = _mm256_set1_epi8(-128i8);
            let ones_u8 = _mm256_set1_epi8(1);
            let ones_i16 = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            let mut b_sum = _mm256_setzero_si256();
            for i in 0..chunks {
                let base = i * 32;
                let av = _mm256_loadu_si256(a[base..].as_ptr() as *const __m256i);
                let bv = _mm256_loadu_si256(b[base..].as_ptr() as *const __m256i);
                let av_u = _mm256_xor_si256(av, bias);
                let prod = _mm256_maddubs_epi16(av_u, bv);
                let widened = _mm256_madd_epi16(prod, ones_i16);
                acc = _mm256_add_epi32(acc, widened);
                let b_abs = _mm256_maddubs_epi16(ones_u8, bv);
                let b_wide = _mm256_madd_epi16(b_abs, ones_i16);
                b_sum = _mm256_add_epi32(b_sum, b_wide);
            }
            let mut acc_vals = [0i32; 8];
            _mm256_storeu_si256(acc_vals.as_mut_ptr() as *mut __m256i, acc);
            let total_biased: i64 = acc_vals.iter().map(|&v| v as i64).sum();
            let mut bsum_vals = [0i32; 8];
            _mm256_storeu_si256(bsum_vals.as_mut_ptr() as *mut __m256i, b_sum);
            let total_b: i64 = bsum_vals.iter().map(|&v| v as i64).sum();
            let mut result = total_biased - 128 * total_b;
            for i in (chunks * 32)..len {
                result += (a[i] as i8 as i64) * (b[i] as i8 as i64);
            }
            result
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x as i8 as i64) * (y as i8 as i64)).sum()
    }
}

// ============================================================================
// GEMM — AVX2 fallback (delegates to scalar for now)
// ============================================================================

/// AVX2 blocked SGEMM fallback — delegates to scalar implementation.
///
/// A dedicated AVX2 microkernel (MR=6, NR=8 with ymm registers) could be
/// added later. For now, the scalar path with LLVM auto-vectorization is
/// sufficient as the AVX2 fallback tier.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    // Scalar fallback: row-by-row dot products
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}

/// AVX2 blocked DGEMM fallback — delegates to scalar implementation.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    c: &mut [f64], ldc: usize,
) {
    // Scalar fallback: row-by-row dot products
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        let result = dot_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1.0,
            "dot_f32: {} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_dot_f64() {
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..100).map(|i| (i * 2) as f64).collect();
        let result = dot_f64(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_axpy_f32() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![10.0f32, 20.0, 30.0, 40.0];
        axpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        scal_f32(3.0, &mut x);
        assert_eq!(x, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_asum_f32() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        assert_eq!(asum_f32(&x), 10.0);
    }

    #[test]
    fn test_nrm2_f32() {
        let x = vec![3.0f32, 4.0];
        assert!((nrm2_f32(&x) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xFFu8; 2048];
        let b = vec![0xFFu8; 2048];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0x00u8; 64];
        let b = vec![0xFFu8; 64];
        assert_eq!(hamming_distance(&a, &b), 512);
    }

    #[test]
    fn test_hamming_distance_2kb() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i + 1) % 256) as u8).collect();
        let dist = hamming_distance(&a, &b);
        let expected: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as u64)
            .sum();
        assert_eq!(dist, expected);
    }

    #[test]
    fn test_hamming_batch() {
        let query = vec![0xAAu8; 16];
        let mut database = vec![0u8; 16 * 4];
        database[0..16].fill(0xAA);
        database[16..32].fill(0x55);
        database[32..40].fill(0xAA);
        database[40..48].fill(0x55);
        database[48..64].fill(0xAA);
        database[48] = 0x55;

        let distances = hamming_batch(&query, &database, 4, 16);
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 128);
        assert_eq!(distances[2], 64);
        assert_eq!(distances[3], 8);
    }
}
