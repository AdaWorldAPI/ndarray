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

/// Lowercase aliases (std::simd convention)
pub type f32x16 = F32x16;
pub type f64x8 = F64x8;
pub type u8x64 = U8x64;
pub type i32x16 = I32x16;
pub type i64x8 = I64x8;
pub type u32x16 = U32x16;
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
