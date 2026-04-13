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

// ============================================================================
// AVX2 512-bit types: composed from 2× 256-bit halves
//
// Same API as simd_avx512::F32x16 etc. but backed by [F32x8; 2].
// Consumer sees crate::simd::F32x16 — simd.rs picks avx512 or avx2 via LazyLock.
// ============================================================================

use core::fmt;
use core::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg,
                BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 16×f32 via 2× AVX2 F32x8 (__m256). Same API as simd_avx512::F32x16.
#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct F32x16(pub f32x8, pub f32x8);

impl F32x16 {
    pub const LANES: usize = 16;
    #[inline(always)] pub fn splat(v: f32) -> Self { Self(f32x8::splat(v), f32x8::splat(v)) }
    #[inline(always)] pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        Self(f32x8::from_slice(&s[..8]), f32x8::from_slice(&s[8..16]))
    }
    #[inline(always)] pub fn from_array(a: [f32; 16]) -> Self {
        Self(f32x8::from_array(a[..8].try_into().unwrap()), f32x8::from_array(a[8..].try_into().unwrap()))
    }
    #[inline(always)] pub fn to_array(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        out[..8].copy_from_slice(&self.0.to_array());
        out[8..].copy_from_slice(&self.1.to_array());
        out
    }
    #[inline(always)] pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        self.0.copy_to_slice(&mut s[..8]);
        self.1.copy_to_slice(&mut s[8..16]);
    }
    #[inline(always)] pub fn reduce_sum(self) -> f32 { self.0.reduce_sum() + self.1.reduce_sum() }
    #[inline(always)] pub fn reduce_min(self) -> f32 {
        let a = self.to_array();
        a.iter().copied().fold(f32::INFINITY, f32::min)
    }
    #[inline(always)] pub fn reduce_max(self) -> f32 {
        let a = self.to_array();
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
    #[inline(always)] pub fn abs(self) -> Self { Self(self.0.abs(), self.1.abs()) }
    #[inline(always)] pub fn sqrt(self) -> Self {
        let a = self.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].sqrt(); } Self::from_array(o)
    }
    #[inline(always)] pub fn round(self) -> Self {
        let a = self.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].round(); } Self::from_array(o)
    }
    #[inline(always)] pub fn floor(self) -> Self {
        let a = self.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].floor(); } Self::from_array(o)
    }
    #[inline(always)] pub fn mul_add(self, b: Self, c: Self) -> Self {
        let a = self.to_array(); let ba = b.to_array(); let ca = c.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].mul_add(ba[i], ca[i]); } Self::from_array(o)
    }
    #[inline(always)] pub fn simd_min(self, other: Self) -> Self {
        let a = self.to_array(); let b = other.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].min(b[i]); } Self::from_array(o)
    }
    #[inline(always)] pub fn simd_max(self, other: Self) -> Self {
        let a = self.to_array(); let b = other.to_array();
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = a[i].max(b[i]); } Self::from_array(o)
    }
    #[inline(always)] pub fn simd_clamp(self, lo: Self, hi: Self) -> Self { self.simd_max(lo).simd_min(hi) }
    #[inline(always)] pub fn simd_lt(self, other: Self) -> F32Mask16 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u16 = 0; for i in 0..16 { if a[i] < b[i] { bits |= 1 << i; } } F32Mask16(bits)
    }
    #[inline(always)] pub fn simd_le(self, other: Self) -> F32Mask16 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u16 = 0; for i in 0..16 { if a[i] <= b[i] { bits |= 1 << i; } } F32Mask16(bits)
    }
    #[inline(always)] pub fn simd_gt(self, other: Self) -> F32Mask16 { other.simd_lt(self) }
    #[inline(always)] pub fn simd_ge(self, other: Self) -> F32Mask16 { other.simd_le(self) }
    #[inline(always)] pub fn simd_eq(self, other: Self) -> F32Mask16 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u16 = 0; for i in 0..16 { if a[i] == b[i] { bits |= 1 << i; } } F32Mask16(bits)
    }
    #[inline(always)] pub fn simd_ne(self, other: Self) -> F32Mask16 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u16 = 0; for i in 0..16 { if a[i] != b[i] { bits |= 1 << i; } } F32Mask16(bits)
    }
    /// Gather 16 f32 values from `base_ptr` using 16 i32 indices.
    ///
    /// # Safety
    /// Caller must ensure all indices are valid offsets into the memory at `base_ptr`.
    #[inline(always)]
    pub unsafe fn gather(indices: I32x16, base_ptr: *const f32) -> Self {
        let idx = indices.0;
        let mut o = [0.0f32; 16];
        for i in 0..16 { o[i] = *base_ptr.add(idx[i] as usize); }
        Self::from_array(o)
    }
    #[inline(always)] pub fn to_bits(self) -> U32x16 {
        let a = self.to_array();
        let mut o = [0u32; 16]; for i in 0..16 { o[i] = a[i].to_bits(); } U32x16(o)
    }
    #[inline(always)] pub fn from_bits(bits: U32x16) -> Self {
        let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = f32::from_bits(bits.0[i]); } Self::from_array(o)
    }
    #[inline(always)] pub fn cast_i32(self) -> I32x16 {
        let a = self.to_array();
        let mut o = [0i32; 16]; for i in 0..16 { o[i] = a[i] as i32; } I32x16(o)
    }
}

impl Add for F32x16 { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0, self.1 + rhs.1) } }
impl Sub for F32x16 { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0, self.1 - rhs.1) } }
impl Mul for F32x16 { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self(self.0 * rhs.0, self.1 * rhs.1) } }
impl Div for F32x16 { type Output = Self; #[inline(always)] fn div(self, rhs: Self) -> Self { Self(self.0 / rhs.0, self.1 / rhs.1) } }
impl AddAssign for F32x16 { #[inline(always)] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; } }
impl SubAssign for F32x16 { #[inline(always)] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; } }
impl MulAssign for F32x16 { #[inline(always)] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; } }
impl DivAssign for F32x16 { #[inline(always)] fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; } }
impl Neg for F32x16 { type Output = Self; #[inline(always)] fn neg(self) -> Self { let a = self.to_array(); let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = -a[i]; } Self::from_array(o) } }
impl fmt::Debug for F32x16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "F32x16({:?})", self.to_array()) } }
impl PartialEq for F32x16 { fn eq(&self, other: &Self) -> bool { self.to_array() == other.to_array() } }
impl Default for F32x16 { fn default() -> Self { Self::splat(0.0) } }

#[derive(Copy, Clone, Debug)]
pub struct F32Mask16(pub u16);
impl F32Mask16 {
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        let t = true_val.to_array(); let f = false_val.to_array();
        let mut o = [0.0f32; 16];
        for i in 0..16 { o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] }; }
        F32x16::from_array(o)
    }
}

/// 8×f64 via 2× AVX2 F64x4 (__m256d). Same API as simd_avx512::F64x8.
#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct F64x8(pub f64x4, pub f64x4);

impl F64x8 {
    pub const LANES: usize = 8;
    #[inline(always)] pub fn splat(v: f64) -> Self { Self(f64x4::splat(v), f64x4::splat(v)) }
    #[inline(always)] pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        Self(f64x4::from_slice(&s[..4]), f64x4::from_slice(&s[4..8]))
    }
    #[inline(always)] pub fn from_array(a: [f64; 8]) -> Self {
        Self(f64x4::from_array(a[..4].try_into().unwrap()), f64x4::from_array(a[4..].try_into().unwrap()))
    }
    #[inline(always)] pub fn to_array(self) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        out[..4].copy_from_slice(&self.0.to_array());
        out[4..].copy_from_slice(&self.1.to_array());
        out
    }
    #[inline(always)] pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        self.0.copy_to_slice(&mut s[..4]);
        self.1.copy_to_slice(&mut s[4..8]);
    }
    #[inline(always)] pub fn reduce_sum(self) -> f64 { self.0.reduce_sum() + self.1.reduce_sum() }
    #[inline(always)] pub fn reduce_min(self) -> f64 { let a = self.to_array(); a.iter().copied().fold(f64::INFINITY, f64::min) }
    #[inline(always)] pub fn reduce_max(self) -> f64 { let a = self.to_array(); a.iter().copied().fold(f64::NEG_INFINITY, f64::max) }
    #[inline(always)] pub fn abs(self) -> Self { let a = self.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].abs(); } Self::from_array(o) }
    #[inline(always)] pub fn sqrt(self) -> Self { let a = self.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].sqrt(); } Self::from_array(o) }
    #[inline(always)] pub fn round(self) -> Self { let a = self.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].round(); } Self::from_array(o) }
    #[inline(always)] pub fn floor(self) -> Self { let a = self.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].floor(); } Self::from_array(o) }
    #[inline(always)] pub fn mul_add(self, b: Self, c: Self) -> Self {
        let a = self.to_array(); let ba = b.to_array(); let ca = c.to_array();
        let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].mul_add(ba[i], ca[i]); } Self::from_array(o)
    }
    #[inline(always)] pub fn simd_min(self, other: Self) -> Self { let a = self.to_array(); let b = other.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].min(b[i]); } Self::from_array(o) }
    #[inline(always)] pub fn simd_max(self, other: Self) -> Self { let a = self.to_array(); let b = other.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = a[i].max(b[i]); } Self::from_array(o) }
    #[inline(always)] pub fn simd_clamp(self, lo: Self, hi: Self) -> Self { self.simd_max(lo).simd_min(hi) }
    #[inline(always)] pub fn simd_ge(self, other: Self) -> F64Mask8 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u8 = 0; for i in 0..8 { if a[i] >= b[i] { bits |= 1 << i; } } F64Mask8(bits)
    }
    #[inline(always)] pub fn simd_le(self, other: Self) -> F64Mask8 {
        let a = self.to_array(); let b = other.to_array();
        let mut bits: u8 = 0; for i in 0..8 { if a[i] <= b[i] { bits |= 1 << i; } } F64Mask8(bits)
    }
    #[inline(always)] pub fn to_bits(self) -> U64x8 {
        let a = self.to_array(); let mut o = [0u64; 8]; for i in 0..8 { o[i] = a[i].to_bits(); } U64x8(o)
    }
    #[inline(always)] pub fn from_bits(bits: U64x8) -> Self {
        let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = f64::from_bits(bits.0[i]); } Self::from_array(o)
    }
}

impl Add for F64x8 { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0, self.1 + rhs.1) } }
impl Sub for F64x8 { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0, self.1 - rhs.1) } }
impl Mul for F64x8 { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self(self.0 * rhs.0, self.1 * rhs.1) } }
impl Div for F64x8 { type Output = Self; #[inline(always)] fn div(self, rhs: Self) -> Self { Self(self.0 / rhs.0, self.1 / rhs.1) } }
impl AddAssign for F64x8 { #[inline(always)] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; } }
impl SubAssign for F64x8 { #[inline(always)] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; } }
impl MulAssign for F64x8 { #[inline(always)] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; } }
impl DivAssign for F64x8 { #[inline(always)] fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; } }
impl Neg for F64x8 { type Output = Self; #[inline(always)] fn neg(self) -> Self { let a = self.to_array(); let mut o = [0.0f64; 8]; for i in 0..8 { o[i] = -a[i]; } Self::from_array(o) } }
impl fmt::Debug for F64x8 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "F64x8({:?})", self.to_array()) } }
impl PartialEq for F64x8 { fn eq(&self, other: &Self) -> bool { self.to_array() == other.to_array() } }
impl Default for F64x8 { fn default() -> Self { Self::splat(0.0) } }

#[derive(Copy, Clone, Debug)]
pub struct F64Mask8(pub u8);
impl F64Mask8 {
    #[inline(always)]
    pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
        let t = true_val.to_array(); let f = false_val.to_array();
        let mut o = [0.0f64; 8];
        for i in 0..8 { o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] }; }
        F64x8::from_array(o)
    }
}

// ── Integer types: array-backed, use scalar ops (no AVX2 integer 512-bit) ──

macro_rules! avx2_int_type {
    ($name:ident, $elem:ty, $lanes:expr, $zero:expr) => {
        #[derive(Copy, Clone)]
        #[repr(align(64))]
        pub struct $name(pub [$elem; $lanes]);

        impl Default for $name { #[inline(always)] fn default() -> Self { Self([$zero; $lanes]) } }
        impl $name {
            pub const LANES: usize = $lanes;
            #[inline(always)] pub fn splat(v: $elem) -> Self { Self([v; $lanes]) }
            #[inline(always)] pub fn from_slice(s: &[$elem]) -> Self { assert!(s.len() >= $lanes); let mut a = [$zero; $lanes]; a.copy_from_slice(&s[..$lanes]); Self(a) }
            #[inline(always)] pub fn from_array(a: [$elem; $lanes]) -> Self { Self(a) }
            #[inline(always)] pub fn to_array(self) -> [$elem; $lanes] { self.0 }
            #[inline(always)] pub fn copy_to_slice(self, s: &mut [$elem]) { assert!(s.len() >= $lanes); s[..$lanes].copy_from_slice(&self.0); }
            #[inline(always)] pub fn reduce_sum(self) -> $elem { let mut s: $elem = $zero; for i in 0..$lanes { s = s.wrapping_add(self.0[i]); } s }
        }
        impl Add for $name { type Output = Self; #[inline(always)] fn add(self, r: Self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = self.0[i].wrapping_add(r.0[i]); } Self(o) } }
        impl Sub for $name { type Output = Self; #[inline(always)] fn sub(self, r: Self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = self.0[i].wrapping_sub(r.0[i]); } Self(o) } }
        impl BitAnd for $name { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = self.0[i] & r.0[i]; } Self(o) } }
        impl BitOr for $name { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = self.0[i] | r.0[i]; } Self(o) } }
        impl BitXor for $name { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = self.0[i] ^ r.0[i]; } Self(o) } }
        impl BitAndAssign for $name { #[inline(always)] fn bitand_assign(&mut self, r: Self) { for i in 0..$lanes { self.0[i] &= r.0[i]; } } }
        impl BitOrAssign for $name { #[inline(always)] fn bitor_assign(&mut self, r: Self) { for i in 0..$lanes { self.0[i] |= r.0[i]; } } }
        impl BitXorAssign for $name { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { for i in 0..$lanes { self.0[i] ^= r.0[i]; } } }
        impl Not for $name { type Output = Self; #[inline(always)] fn not(self) -> Self { let mut o = [$zero; $lanes]; for i in 0..$lanes { o[i] = !self.0[i]; } Self(o) } }
        impl AddAssign for $name { #[inline(always)] fn add_assign(&mut self, r: Self) { for i in 0..$lanes { self.0[i] = self.0[i].wrapping_add(r.0[i]); } } }
        impl SubAssign for $name { #[inline(always)] fn sub_assign(&mut self, r: Self) { for i in 0..$lanes { self.0[i] = self.0[i].wrapping_sub(r.0[i]); } } }
        impl fmt::Debug for $name { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, concat!(stringify!($name), "({:?})"), &self.0[..]) } }
        impl PartialEq for $name { fn eq(&self, other: &Self) -> bool { self.0 == other.0 } }
    };
}

avx2_int_type!(U8x64, u8, 64, 0u8);

// ── U8x64 byte-level operations (scalar fallback for AVX2 tier) ──────────
// These match the AVX-512 U8x64 methods in simd_avx512.rs.
impl U8x64 {
    /// Byte-wise equality mask: bit i set if self[i] == other[i].
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 { if self.0[i] == other.0[i] { mask |= 1u64 << i; } }
        mask
    }

    /// Shift right each 16-bit lane by imm bits (operates on pairs of u8 as u16).
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        let mut out = [0u8; 64];
        for i in (0..64).step_by(2) {
            let val = u16::from_le_bytes([self.0[i], self.0[i + 1]]);
            let shifted = val >> imm;
            let bytes = shifted.to_le_bytes();
            out[i] = bytes[0];
            out[i + 1] = bytes[1];
        }
        Self(out)
    }

    /// Saturating unsigned subtraction: max(a - b, 0) per byte.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 { out[i] = self.0[i].saturating_sub(other.0[i]); }
        Self(out)
    }

    /// Interleave low bytes within each 128-bit lane.
    #[inline(always)]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        // Operates per 16-byte lane (4 lanes in 512-bit)
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2] = self.0[base + i];
                out[base + i * 2 + 1] = other.0[base + i];
            }
        }
        Self(out)
    }

    /// Interleave high bytes within each 128-bit lane.
    #[inline(always)]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2] = self.0[base + 8 + i];
                out[base + i * 2 + 1] = other.0[base + 8 + i];
            }
        }
        Self(out)
    }

    /// Reduce min/max (not in macro).
    #[inline(always)] pub fn reduce_min(self) -> u8 { *self.0.iter().min().unwrap() }
    #[inline(always)] pub fn reduce_max(self) -> u8 { *self.0.iter().max().unwrap() }
    #[inline(always)] pub fn simd_min(self, other: Self) -> Self { let mut o = [0u8; 64]; for i in 0..64 { o[i] = self.0[i].min(other.0[i]); } Self(o) }
    #[inline(always)] pub fn simd_max(self, other: Self) -> Self { let mut o = [0u8; 64]; for i in 0..64 { o[i] = self.0[i].max(other.0[i]); } Self(o) }

    /// Byte-wise shuffle: use `self` as a LUT, `idx` selects bytes within each 16-byte lane.
    #[inline(always)]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..16 {
                out[b + i] = self.0[b + (idx.0[b + i] & 0x0F) as usize];
            }
        }
        Self(out)
    }

    /// Sum all 64 bytes into a single `u64` without wrapping.
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        self.0.iter().map(|&b| b as u64).sum()
    }

    /// Build a nibble-popcount lookup table (replicated across 4 x 16-byte lanes).
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        let lane: [u8; 16] = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4];
        let mut arr = [0u8; 64];
        for l in 0..4 { arr[l*16..(l+1)*16].copy_from_slice(&lane); }
        Self(arr)
    }
}

avx2_int_type!(I32x16, i32, 16, 0i32);
avx2_int_type!(I64x8, i64, 8, 0i64);
avx2_int_type!(U32x16, u32, 16, 0u32);
avx2_int_type!(U64x8, u64, 8, 0u64);

impl I32x16 {
    #[inline(always)] pub fn reduce_min(self) -> i32 { *self.0.iter().min().unwrap() }
    #[inline(always)] pub fn reduce_max(self) -> i32 { *self.0.iter().max().unwrap() }
    #[inline(always)] pub fn simd_min(self, other: Self) -> Self { let mut o = [0i32; 16]; for i in 0..16 { o[i] = self.0[i].min(other.0[i]); } Self(o) }
    #[inline(always)] pub fn simd_max(self, other: Self) -> Self { let mut o = [0i32; 16]; for i in 0..16 { o[i] = self.0[i].max(other.0[i]); } Self(o) }
    #[inline(always)] pub fn cast_f32(self) -> F32x16 { let mut o = [0.0f32; 16]; for i in 0..16 { o[i] = self.0[i] as f32; } F32x16::from_array(o) }
    #[inline(always)] pub fn abs(self) -> Self { let mut o = [0i32; 16]; for i in 0..16 { o[i] = self.0[i].abs(); } Self(o) }

    /// Load 16 × i16, sign-extend to 16 × i32.
    #[inline(always)]
    pub fn from_i16_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16);
        let mut o = [0i32; 16];
        for i in 0..16 { o[i] = s[i] as i32; }
        Self(o)
    }

    /// Narrow 16 × i32 to 16 × i16 (truncation).
    #[inline(always)]
    pub fn to_i16_array(self) -> [i16; 16] {
        let mut o = [0i16; 16];
        for i in 0..16 { o[i] = self.0[i] as i16; }
        o
    }

    /// Mask: bit i set where lane i >= 0.
    #[inline(always)]
    pub fn cmpge_zero_mask(self) -> u16 {
        let mut mask = 0u16;
        for i in 0..16 { if self.0[i] >= 0 { mask |= 1 << i; } }
        mask
    }
}
impl Mul for I32x16 { type Output = Self; #[inline(always)] fn mul(self, r: Self) -> Self { let mut o = [0i32; 16]; for i in 0..16 { o[i] = self.0[i].wrapping_mul(r.0[i]); } Self(o) } }
impl MulAssign for I32x16 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Neg for I32x16 { type Output = Self; #[inline(always)] fn neg(self) -> Self { let mut o = [0i32; 16]; for i in 0..16 { o[i] = -self.0[i]; } Self(o) } }

impl I64x8 {
    #[inline(always)] pub fn reduce_min(self) -> i64 { *self.0.iter().min().unwrap() }
    #[inline(always)] pub fn reduce_max(self) -> i64 { *self.0.iter().max().unwrap() }
    #[inline(always)] pub fn simd_min(self, other: Self) -> Self { let mut o = [0i64; 8]; for i in 0..8 { o[i] = self.0[i].min(other.0[i]); } Self(o) }
    #[inline(always)] pub fn simd_max(self, other: Self) -> Self { let mut o = [0i64; 8]; for i in 0..8 { o[i] = self.0[i].max(other.0[i]); } Self(o) }
}

/// Lowercase aliases (std::simd convention).
#[allow(non_camel_case_types)]
pub type f32x16 = F32x16;
#[allow(non_camel_case_types)]
pub type f64x8 = F64x8;
#[allow(non_camel_case_types)]
pub type u8x64 = U8x64;
#[allow(non_camel_case_types)]
pub type i32x16 = I32x16;
#[allow(non_camel_case_types)]
pub type i64x8 = I64x8;
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;
#[allow(non_camel_case_types)]
pub type u64x8 = U64x8;

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

// ════════════════════════════════════════════════════════════════════════════
// F16 IEEE 754 Precision Toolkit — AVX2-accelerated (F16C: 8 lanes per cycle)
//
// ⚠️  NOT FOR GGUF CALIBRATION — see simd_avx512.rs BF16 pipeline for that.
// This is for: sensor data, audio samples, ARM↔x86 interchange, memory savings.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ IEEE 754 binary16: 1 sign + 5 exponent (bias 15) + 10 mantissa        │
// │ Range: ±65504    Precision: 3.31 decimal digits    Subnormal: ±5.96e-8 │
// │                                                                         │
// │ f16→f32: ALWAYS EXACT (lossless widening, zero error)                   │
// │ f32→f16: LOSSY (23-bit → 10-bit mantissa = 13 bits lost)               │
// │          Max RNE error: ±0.5 ULP of f16 result (≈0.05% relative)       │
// └─────────────────────────────────────────────────────────────────────────┘
//
// Hardware: F16C (VCVTPH2PS / VCVTPS2PH) available on Haswell+ (2013).
//           AVX2 path uses __m128i → __m256 (8 lanes per instruction).
//           AVX-512F path (16 lanes) lives in simd_avx512.rs.
//
// Tricks implemented:
//   1. Double-f16 (Error-Free Split) — ~20-bit effective precision in 2×u16
//   2. Kahan-compensated f16 accumulation — eliminates cumulative error
//   3. Exponent-aligned scaling — optimal mantissa utilization in known ranges
//
// All scalar paths use the IEEE 754 functions from simd_avx512.rs.
// AVX2 batch paths use F16C hardware (8 lanes) with scalar tail.
// ════════════════════════════════════════════════════════════════════════════

// Re-use the exact IEEE 754 scalar functions from simd_avx512
pub use crate::simd_avx512::{
    f16_to_f32_ieee754,
    f32_to_f16_ieee754_rne,
    f16_to_f32_batch_ieee754,
    f32_to_f16_batch_ieee754_rne,
};

// ── Trick 1: Double-f16 (Error-Free Split) ──────────────────────────────
//
// Problem: f32→f16 loses 13 mantissa bits (23→10).
// Solution: store value as TWO f16 values: hi (main) + lo (residual).
//
// Encode:
//   hi = rne(value)                     // best f16 approximation
//   residual = value - f16_to_f32(hi)   // exact error (computed in f32)
//   lo = rne(residual)                  // error captured as second f16
//
// Decode:
//   value ≈ f16_to_f32(hi) + f16_to_f32(lo)   // both conversions exact
//
// Effective precision: ~20 mantissa bits (10 + ~10 from residual).
// Storage: 4 bytes (same as f32) but split across two u16 values.
// Use case: codebook centroids where f16 is too imprecise but f32 wastes RAM.
//
// Error analysis:
//   hi captures the value with ≤0.5 ULP_f16 error
//   lo captures the residual with ≤0.5 ULP_f16(residual) error
//   Total error: ≤0.5 ULP_f16(residual) ≈ 2^{-21} × |value|
//   vs single f16: ≤0.5 ULP_f16 ≈ 2^{-11} × |value|
//   → ~1000× better precision for same 4 bytes

/// Encode f32 as Double-f16 pair (hi, lo) with ~20-bit effective precision.
///
/// Both `hi` and `lo` are standard IEEE 754 f16 values (stored as u16).
/// Decode: `f16_to_f32(hi) + f16_to_f32(lo)` (both additions are exact).
///
/// # Precision
/// - Single f16: 10 mantissa bits → 3.31 decimal digits
/// - Double-f16: ~20 mantissa bits → 6.02 decimal digits
/// - f32:         23 mantissa bits → 7.22 decimal digits
#[inline]
pub fn f16_double_encode(value: f32) -> (u16, u16) {
    let hi = f32_to_f16_ieee754_rne(value);
    let hi_f32 = f16_to_f32_ieee754(hi); // exact (lossless widening)
    let residual = value - hi_f32;        // exact (f32 subtraction)
    let lo = f32_to_f16_ieee754_rne(residual);
    (hi, lo)
}

/// Decode Double-f16 pair back to f32. Both f16→f32 conversions are exact.
#[inline]
pub fn f16_double_decode(hi: u16, lo: u16) -> f32 {
    f16_to_f32_ieee754(hi) + f16_to_f32_ieee754(lo)
}

/// Batch encode: f32 slice → Double-f16 (separate hi/lo arrays).
///
/// AVX2 acceleration via F16C for the f32→f16 conversions.
pub fn f16_double_encode_batch(input: &[f32], output_hi: &mut [u16], output_lo: &mut [u16]) {
    let n = input.len().min(output_hi.len()).min(output_lo.len());

    // Step 1: encode hi values (AVX2 F16C batch)
    f32_to_f16_batch_ieee754_rne(input, &mut output_hi[..n]);

    // Step 2: compute residuals and encode lo values
    let mut residuals = vec![0.0f32; n];
    f16_to_f32_batch_ieee754(&output_hi[..n], &mut residuals);
    for i in 0..n {
        residuals[i] = input[i] - residuals[i];
    }
    f32_to_f16_batch_ieee754_rne(&residuals, &mut output_lo[..n]);
}

/// Batch decode: Double-f16 → f32. Uses AVX2 F16C + f32x8 addition.
pub fn f16_double_decode_batch(hi: &[u16], lo: &[u16], output: &mut [f32]) {
    let n = hi.len().min(lo.len()).min(output.len());

    f16_to_f32_batch_ieee754(&hi[..n], &mut output[..n]);

    let mut lo_f32 = vec![0.0f32; n];
    f16_to_f32_batch_ieee754(&lo[..n], &mut lo_f32);

    // AVX2-accelerated f32 addition (8 lanes per cycle)
    let chunks = n / F32_LANES;
    for c in 0..chunks {
        let base = c * F32_LANES;
        let out_v = f32x8::from_slice(&output[base..]);
        let lo_v = f32x8::from_slice(&lo_f32[base..]);
        (out_v + lo_v).copy_to_slice(&mut output[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..n {
        output[i] += lo_f32[i];
    }
}

// ── Trick 2: Kahan-compensated f16 accumulation ─────────────────────────
//
// Problem: summing many f16 values in f32 accumulates rounding error.
//   Naive sum of 10K × 0.1: error ≈ 0.05
//   Kahan sum of 10K × 0.1: error ≈ 0.0 (bounded by 2ε, independent of N)
//
// Precision: O(ε) total error instead of O(N·ε).
// Cost: ~2 extra f32 additions per element (negligible vs f16→f32).

/// Kahan-compensated sum of f16 values. Returns f32 with near-zero cumulative error.
///
/// Each f16→f32 conversion is exact (lossless widening).
/// Kahan algorithm tracks rounding error of each f32 addition.
///
/// # Error bound
/// - Naive sum of N values: error ≤ N × ε (ε ≈ 1.19e-7)
/// - Kahan sum of N values: error ≤ 2ε (independent of N!)
pub fn f16_kahan_sum(input: &[u16]) -> f32 {
    let mut f32_buf = vec![0.0f32; input.len()];
    f16_to_f32_batch_ieee754(input, &mut f32_buf);

    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &v in &f32_buf {
        let y = v - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan-compensated dot product of two f16 vectors.
///
/// AVX2-accelerated: F16C for f16→f32, f32x8 multiply, Kahan accumulate.
pub fn f16_kahan_dot(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len().min(b.len());
    let mut a_f32 = vec![0.0f32; n];
    let mut b_f32 = vec![0.0f32; n];
    f16_to_f32_batch_ieee754(&a[..n], &mut a_f32);
    f16_to_f32_batch_ieee754(&b[..n], &mut b_f32);

    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;

    // AVX2: multiply 8-wide, reduce_sum, Kahan-accumulate partial sums
    let chunks = n / F32_LANES;
    for c in 0..chunks {
        let base = c * F32_LANES;
        let av = f32x8::from_slice(&a_f32[base..]);
        let bv = f32x8::from_slice(&b_f32[base..]);
        let prod_sum = (av * bv).reduce_sum();
        let y = prod_sum - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    for i in (chunks * F32_LANES)..n {
        let prod = a_f32[i] * b_f32[i];
        let y = prod - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

// ── Trick 3: Exponent-aligned scaling ───────────────────────────────────
//
// Problem: f16 has 10 mantissa bits. Narrow-range values waste exponent bits.
//   Values in [0.001, 0.005]: only 3-4 mantissa bits significant → ~8 levels
//   After scale to [0.5, 2.0]: all 10 mantissa bits → ~1024 levels
//
// Precision improvement: up to ~128× for narrow-range data.
// Use case: codebook centroids, sensor readings, normalized weights.

/// Pre-computed scaling context for exponent-aligned f16 encoding.
///
/// Analyzes the input range, computes scale that maps |max| → 1.0,
/// then uses that scale for all encode/decode operations.
#[derive(Debug, Clone, Copy)]
pub struct F16Scaler {
    /// Multiply by this before f32→f16 (shifts into sweet spot)
    pub scale: f32,
    /// Multiply by this after f16→f32 (restores original range)
    pub inv_scale: f32,
}

impl F16Scaler {
    /// Create from known value range [min_val, max_val].
    pub fn from_range(min_val: f32, max_val: f32) -> Self {
        assert!(min_val < max_val, "min must be less than max");
        let abs_max = min_val.abs().max(max_val.abs());
        if abs_max < f32::EPSILON {
            return Self { scale: 1.0, inv_scale: 1.0 };
        }
        let scale = 1.0 / abs_max;
        Self { scale, inv_scale: abs_max }
    }

    /// Create by scanning data for min/max.
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self { scale: 1.0, inv_scale: 1.0 };
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in data {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        Self::from_range(min, max)
    }

    #[inline]
    pub fn encode(&self, value: f32) -> u16 {
        f32_to_f16_ieee754_rne(value * self.scale)
    }

    #[inline]
    pub fn decode(&self, bits: u16) -> f32 {
        f16_to_f32_ieee754(bits) * self.inv_scale
    }

    /// Batch encode with AVX2: f32x8 scale multiply → F16C convert.
    pub fn encode_batch(&self, input: &[f32], output: &mut [u16]) {
        let n = input.len().min(output.len());
        let mut scaled = vec![0.0f32; n];
        let scale_v = f32x8::splat(self.scale);
        let chunks = n / F32_LANES;
        for c in 0..chunks {
            let base = c * F32_LANES;
            let v = f32x8::from_slice(&input[base..]);
            (v * scale_v).copy_to_slice(&mut scaled[base..base + F32_LANES]);
        }
        for i in (chunks * F32_LANES)..n {
            scaled[i] = input[i] * self.scale;
        }
        f32_to_f16_batch_ieee754_rne(&scaled, &mut output[..n]);
    }

    /// Batch decode with AVX2: F16C convert → f32x8 inv_scale multiply.
    pub fn decode_batch(&self, input: &[u16], output: &mut [f32]) {
        let n = input.len().min(output.len());
        f16_to_f32_batch_ieee754(&input[..n], &mut output[..n]);
        let inv_v = f32x8::splat(self.inv_scale);
        let chunks = n / F32_LANES;
        for c in 0..chunks {
            let base = c * F32_LANES;
            let v = f32x8::from_slice(&output[base..]);
            (v * inv_v).copy_to_slice(&mut output[base..base + F32_LANES]);
        }
        for i in (chunks * F32_LANES)..n {
            output[i] *= self.inv_scale;
        }
    }
}

#[cfg(test)]
mod f16_precision_tests {
    use super::*;

    #[test]
    fn double_f16_better_than_single() {
        let value = std::f32::consts::PI;
        let single = f32_to_f16_ieee754_rne(value);
        let single_err = (value - f16_to_f32_ieee754(single)).abs();

        let (hi, lo) = f16_double_encode(value);
        let double_err = (value - f16_double_decode(hi, lo)).abs();

        assert!(double_err < single_err,
            "double should be better: single={:.8} double={:.8}", single_err, double_err);
        assert!(double_err < single_err / 100.0,
            "double should be >100× better: ratio={:.0}", single_err / double_err);
    }

    #[test]
    fn double_f16_batch_roundtrip() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.037).collect();
        let mut hi = vec![0u16; 100];
        let mut lo = vec![0u16; 100];
        f16_double_encode_batch(&input, &mut hi, &mut lo);

        let mut decoded = vec![0.0f32; 100];
        f16_double_decode_batch(&hi, &lo, &mut decoded);

        for i in 0..100 {
            let err = (input[i] - decoded[i]).abs();
            let tol = input[i].abs() * 1e-4 + 1e-7;
            assert!(err < tol, "at {}: {} → {} err={}", i, input[i], decoded[i], err);
        }
    }

    #[test]
    fn kahan_sum_consistent() {
        let val_f16 = f32_to_f16_ieee754_rne(0.1);
        let input = vec![val_f16; 10_000];
        let kahan = f16_kahan_sum(&input);
        let expected = 10_000.0 * f16_to_f32_ieee754(val_f16);
        let err = (kahan - expected).abs();
        assert!(err < 0.01, "kahan error too large: {} (expected {})", err, expected);
    }

    #[test]
    fn kahan_dot_vs_f64_reference() {
        let a: Vec<u16> = (0..64).map(|i| f32_to_f16_ieee754_rne(i as f32 * 0.1)).collect();
        let b: Vec<u16> = (0..64).map(|i| f32_to_f16_ieee754_rne(1.0 - i as f32 * 0.01)).collect();
        let dot = f16_kahan_dot(&a, &b);
        let mut ref_sum = 0.0f64;
        for i in 0..64 {
            ref_sum += f16_to_f32_ieee754(a[i]) as f64 * f16_to_f32_ieee754(b[i]) as f64;
        }
        assert!((dot as f64 - ref_sum).abs() < 0.01,
            "got={} expected={}", dot, ref_sum);
    }

    #[test]
    fn scaler_improves_small_values() {
        let data: Vec<f32> = (0..100).map(|i| 0.001 + (i as f32) * 0.00004).collect();

        let no_scale: Vec<u16> = data.iter().map(|&v| f32_to_f16_ieee754_rne(v)).collect();
        let no_scale_err: f64 = data.iter().enumerate()
            .map(|(i, &v)| (v as f64 - f16_to_f32_ieee754(no_scale[i]) as f64).powi(2)).sum();

        let scaler = F16Scaler::from_data(&data);
        let mut scaled = vec![0u16; 100];
        scaler.encode_batch(&data, &mut scaled);
        let mut back = vec![0.0f32; 100];
        scaler.decode_batch(&scaled, &mut back);
        let scaled_err: f64 = data.iter().enumerate()
            .map(|(i, &v)| (v as f64 - back[i] as f64).powi(2)).sum();

        assert!(scaled_err < no_scale_err,
            "scaling should help: unscaled={:.2e} scaled={:.2e}", no_scale_err, scaled_err);
    }

    #[test]
    fn scaler_roundtrip_batch() {
        let data: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.004).collect();
        let scaler = F16Scaler::from_data(&data);
        let mut enc = vec![0u16; 50];
        scaler.encode_batch(&data, &mut enc);
        let mut dec = vec![0.0f32; 50];
        scaler.decode_batch(&enc, &mut dec);
        for i in 0..50 {
            let err = (data[i] - dec[i]).abs();
            assert!(err < data[i].abs() * 0.01 + 1e-6,
                "at {}: {} → {} err={}", i, data[i], dec[i], err);
        }
    }
}
