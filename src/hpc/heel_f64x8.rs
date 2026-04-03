//! F64x8 HEEL distance: 8 f64 distances across 8 HEEL planes in one SIMD pass.
//!
//! p64 has 8 HEEL planes (u64 each). For weighted f64 distance computation,
//! each plane produces one f64 distance value → 8 values = one F64x8 register.
//!
//! This module provides the SIMD kernel; p64-bridge calls it.
//! ndarray = hardware acceleration, consumers use the kernel.
//!
//! Dispatch: AVX-512 (native __m512d) → AVX2 (2×__m256d) → scalar.
//! LazyLock selects at startup.

use std::sync::LazyLock;

/// Kernel signature: 8 distances in, weighted sum out.
/// `distances`: 8 f64 values (one per HEEL plane).
/// `weights`: 8 f64 weights (per-expert importance).
/// Returns: weighted sum = Σ(distance[i] × weight[i]).
type HeelF64x8DotFn = unsafe fn(&[f64; 8], &[f64; 8]) -> f64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn heel_dot_avx512(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    use std::arch::x86_64::*;
    let va = _mm512_loadu_pd(a.as_ptr());
    let vb = _mm512_loadu_pd(b.as_ptr());
    let prod = _mm512_mul_pd(va, vb);
    _mm512_reduce_add_pd(prod)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn heel_dot_avx2(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    use std::arch::x86_64::*;
    // 2 passes of 4 lanes
    let va0 = _mm256_loadu_pd(a.as_ptr());
    let vb0 = _mm256_loadu_pd(b.as_ptr());
    let p0 = _mm256_mul_pd(va0, vb0);

    let va1 = _mm256_loadu_pd(a[4..].as_ptr());
    let vb1 = _mm256_loadu_pd(b[4..].as_ptr());
    let p1 = _mm256_mul_pd(va1, vb1);

    let sum = _mm256_add_pd(p0, p1);
    // Horizontal sum of 4 f64
    let hi = _mm256_extractf128_pd(sum, 1);
    let lo = _mm256_castpd256_pd128(sum);
    let pair = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(pair, pair);
    let result = _mm_add_sd(pair, hi64);
    _mm_cvtsd_f64(result)
}

fn heel_dot_scalar(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..8 {
        sum += a[i] * b[i];
    }
    sum
}

static HEEL_DOT_KERNEL: LazyLock<HeelF64x8DotFn> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return heel_dot_avx512 as HeelF64x8DotFn;
        }
        if is_x86_feature_detected!("avx2") {
            return heel_dot_avx2 as HeelF64x8DotFn;
        }
    }
    heel_dot_scalar as HeelF64x8DotFn
});

/// Compute weighted dot product of 8 HEEL plane distances.
///
/// `distances[i]` = distance for HEEL plane i.
/// `weights[i]` = importance weight for plane i.
/// Returns: Σ(distances[i] × weights[i]).
///
/// One SIMD pass on AVX-512 (single `vmulpd` + `vreducepd`).
/// Two passes on AVX2. Scalar fallback for non-x86.
#[inline]
pub fn heel_weighted_distance(distances: &[f64; 8], weights: &[f64; 8]) -> f64 {
    unsafe { HEEL_DOT_KERNEL(distances, weights) }
}

/// Compute L1-like distance across 8 HEEL planes.
///
/// For each plane i: distance[i] = popcount(a[i] XOR b[i]) as f64.
/// Then weighted sum via F64x8 dot product.
///
/// This converts binary Hamming distances to f64 for weighted combination,
/// where each plane's contribution is scaled by expert importance.
pub fn heel_plane_distances(a: &[u64; 8], b: &[u64; 8]) -> [f64; 8] {
    let mut dists = [0.0f64; 8];
    for i in 0..8 {
        dists[i] = (a[i] ^ b[i]).count_ones() as f64;
    }
    dists
}

/// Full pipeline: 8 HEEL planes → Hamming per plane → weighted F64x8 dot → scalar distance.
#[inline]
pub fn heel_weighted_hamming(
    a_planes: &[u64; 8],
    b_planes: &[u64; 8],
    weights: &[f64; 8],
) -> f64 {
    let dists = heel_plane_distances(a_planes, b_planes);
    heel_weighted_distance(&dists, weights)
}

/// Uniform weights (all planes equal).
pub const UNIFORM_WEIGHTS: [f64; 8] = [1.0; 8];

/// HEEL-weighted (7 constructive + 1 contradiction at reduced weight).
/// Contradiction plane (index 7) gets 0.5× weight.
pub const HEEL_7PLUS1_WEIGHTS: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = heel_weighted_distance(&a, &b);
        assert!((result - 36.0).abs() < 1e-10, "1+2+3+4+5+6+7+8 = 36, got {}", result);
    }

    #[test]
    fn dot_product_weighted() {
        let distances = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let weights = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5];
        let result = heel_weighted_distance(&distances, &weights);
        assert!((result - 60.0).abs() < 1e-10, "10×2 + 80×0.5 = 60, got {}", result);
    }

    #[test]
    fn plane_distances_self_zero() {
        let planes = [0x1234u64; 8];
        let dists = heel_plane_distances(&planes, &planes);
        for d in &dists {
            assert_eq!(*d, 0.0);
        }
    }

    #[test]
    fn plane_distances_opposite() {
        let a = [0u64; 8];
        let b = [u64::MAX; 8];
        let dists = heel_plane_distances(&a, &b);
        for d in &dists {
            assert_eq!(*d, 64.0);
        }
    }

    #[test]
    fn full_pipeline_uniform() {
        let a = [0xFFFF_0000_FFFF_0000u64; 8];
        let b = [0x0000_FFFF_0000_FFFFu64; 8];
        let d = heel_weighted_hamming(&a, &b, &UNIFORM_WEIGHTS);
        // Each plane: all bits differ = 64
        assert!((d - 64.0 * 8.0).abs() < 1e-10, "8 planes × 64 bits = 512, got {}", d);
    }

    #[test]
    fn seven_plus_one_weights() {
        let a = [0u64; 8];
        let b = [u64::MAX; 8];
        let d_uniform = heel_weighted_hamming(&a, &b, &UNIFORM_WEIGHTS);
        let d_7plus1 = heel_weighted_hamming(&a, &b, &HEEL_7PLUS1_WEIGHTS);
        // 7+1: plane 7 at 0.5× = 7×64 + 0.5×64 = 480 vs 512
        assert!((d_uniform - 512.0).abs() < 1e-10);
        assert!((d_7plus1 - 480.0).abs() < 1e-10, "7×64 + 0.5×64 = 480, got {}", d_7plus1);
    }
}
