//! F64x8 HEEL distance: 8 f64 distances across 8 HEEL planes in one SIMD pass.
//!
//! p64 has 8 HEEL planes (u64 each). For weighted f64 distance computation,
//! each plane produces one f64 distance value → 8 values = one F64x8 register.
//!
//! Uses `crate::simd::F64x8` polyfill — automatic dispatch:
//!   AVX-512: native __m512d (one register)
//!   AVX2:    2× __m256d (two registers, same API)
//!   Scalar:  [f64; 8] fallback
//! Consumer writes `crate::simd::F64x8`. The polyfill handles the rest.

use crate::simd::F64x8;

/// Compute weighted dot product of 8 HEEL plane distances.
///
/// `distances[i]` = distance for HEEL plane i.
/// `weights[i]` = importance weight for plane i.
/// Returns: Σ(distances[i] × weights[i]).
///
/// One F64x8 multiply + reduce_sum. On AVX-512: single vmulpd + vreducepd.
/// On AVX2: 2× vmulpd + 2× haddpd. Scalar: 8 multiplies + sum.
#[inline]
pub fn heel_weighted_distance(distances: &[f64; 8], weights: &[f64; 8]) -> f64 {
    let vd = F64x8::from_slice(distances);
    let vw = F64x8::from_slice(weights);
    (vd * vw).reduce_sum()
}

/// Compute L1-like distance across 8 HEEL planes.
///
/// For each plane i: distance[i] = popcount(a[i] XOR b[i]) as f64.
/// This is Hamming on binary HEEL planes — valid because HEEL planes
/// ARE uniform binary data (unlike bgz17 i16 which must use L1).
pub fn heel_plane_distances(a: &[u64; 8], b: &[u64; 8]) -> [f64; 8] {
    let mut dists = [0.0f64; 8];
    for i in 0..8 {
        dists[i] = (a[i] ^ b[i]).count_ones() as f64;
    }
    dists
}

/// Full pipeline: 8 HEEL planes → Hamming per plane → weighted F64x8 dot → scalar.
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

// ═══════════════════════════════════════════════════════════════════════════
// SIMD cosine similarity via F64x8 — for CLAM cosine clustering
// ═══════════════════════════════════════════════════════════════════════════

/// SIMD dot product on f64 slices via F64x8.
///
/// Processes 8 elements per iteration. Remainder handled scalar.
/// Used by cosine_simd as the inner kernel.
pub fn dot_f64_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = F64x8::splat(0.0);
    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        let vb = F64x8::from_slice(&b[i * 8..]);
        acc = va.mul_add(vb, acc); // acc = va * vb + acc (FMA)
    }
    let mut sum = acc.reduce_sum();

    // Scalar remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        sum += a[offset + i] * b[offset + i];
    }
    sum
}

/// SIMD sum of squares via F64x8.
pub fn sum_sq_f64_simd(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = F64x8::splat(0.0);
    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        acc = va.mul_add(va, acc); // acc = va * va + acc
    }
    let mut sum = acc.reduce_sum();

    let offset = chunks * 8;
    for i in 0..remainder {
        sum += a[offset + i] * a[offset + i];
    }
    sum
}

/// SIMD cosine similarity on f64 slices.
///
/// Computes dot(a,b) / (||a|| × ||b||) using F64x8 FMA.
/// Single pass: accumulates dot, norm_a, norm_b simultaneously.
pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot_acc = F64x8::splat(0.0);
    let mut na_acc = F64x8::splat(0.0);
    let mut nb_acc = F64x8::splat(0.0);

    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        let vb = F64x8::from_slice(&b[i * 8..]);
        dot_acc = va.mul_add(vb, dot_acc);  // dot += a*b
        na_acc = va.mul_add(va, na_acc);    // na += a*a
        nb_acc = vb.mul_add(vb, nb_acc);    // nb += b*b
    }

    let mut dot = dot_acc.reduce_sum();
    let mut na = na_acc.reduce_sum();
    let mut nb = nb_acc.reduce_sum();

    let offset = chunks * 8;
    for i in 0..remainder {
        dot += a[offset + i] * b[offset + i];
        na += a[offset + i] * a[offset + i];
        nb += b[offset + i] * b[offset + i];
    }

    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

/// SIMD cosine similarity on f32 slices (converts to f64 internally for precision).
///
/// For hot paths where input is f32 but you need f64 precision cosine.
/// Converts 8 f32 → 8 f64 per chunk via scalar widening, then F64x8 FMA.
pub fn cosine_f32_to_f64_simd(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot_acc = F64x8::splat(0.0);
    let mut na_acc = F64x8::splat(0.0);
    let mut nb_acc = F64x8::splat(0.0);

    let mut buf_a = [0.0f64; 8];
    let mut buf_b = [0.0f64; 8];

    for i in 0..chunks {
        let off = i * 8;
        for j in 0..8 {
            buf_a[j] = a[off + j] as f64;
            buf_b[j] = b[off + j] as f64;
        }
        let va = F64x8::from_slice(&buf_a);
        let vb = F64x8::from_slice(&buf_b);
        dot_acc = va.mul_add(vb, dot_acc);
        na_acc = va.mul_add(va, na_acc);
        nb_acc = vb.mul_add(vb, nb_acc);
    }

    let mut dot = dot_acc.reduce_sum();
    let mut na = na_acc.reduce_sum();
    let mut nb = nb_acc.reduce_sum();

    let offset = chunks * 8;
    for i in 0..remainder {
        let ai = a[offset + i] as f64;
        let bi = b[offset + i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }

    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heel_dot_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0; 8];
        let result = heel_weighted_distance(&a, &b);
        assert!((result - 36.0).abs() < 1e-10, "1+2+...+8 = 36, got {}", result);
    }

    #[test]
    fn heel_dot_weighted() {
        let distances = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let weights = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5];
        let result = heel_weighted_distance(&distances, &weights);
        assert!((result - 60.0).abs() < 1e-10, "10×2 + 80×0.5 = 60, got {}", result);
    }

    #[test]
    fn plane_distances_self_zero() {
        let planes = [0x1234u64; 8];
        let dists = heel_plane_distances(&planes, &planes);
        for d in &dists { assert_eq!(*d, 0.0); }
    }

    #[test]
    fn plane_distances_opposite() {
        let a = [0u64; 8];
        let b = [u64::MAX; 8];
        let dists = heel_plane_distances(&a, &b);
        for d in &dists { assert_eq!(*d, 64.0); }
    }

    #[test]
    fn full_pipeline_uniform() {
        let a = [0xFFFF_0000_FFFF_0000u64; 8];
        let b = [0x0000_FFFF_0000_FFFFu64; 8];
        let d = heel_weighted_hamming(&a, &b, &UNIFORM_WEIGHTS);
        assert!((d - 512.0).abs() < 1e-10, "8×64 = 512, got {}", d);
    }

    #[test]
    fn seven_plus_one_weights() {
        let a = [0u64; 8];
        let b = [u64::MAX; 8];
        let d = heel_weighted_hamming(&a, &b, &HEEL_7PLUS1_WEIGHTS);
        assert!((d - 480.0).abs() < 1e-10, "7×64 + 0.5×64 = 480, got {}", d);
    }

    // ── SIMD cosine tests ───────────────────────────────────────────

    #[test]
    fn cosine_identical() {
        let a: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
        let c = cosine_f64_simd(&a, &a);
        assert!((c - 1.0).abs() < 1e-10, "self-cosine should be 1.0: {}", c);
    }

    #[test]
    fn cosine_opposite() {
        let a: Vec<f64> = (0..256).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = a.iter().map(|v| -v).collect();
        let c = cosine_f64_simd(&a, &b);
        assert!((c - (-1.0)).abs() < 1e-10, "opposite should be -1.0: {}", c);
    }

    #[test]
    fn cosine_orthogonal() {
        let mut a = vec![0.0f64; 256];
        let mut b = vec![0.0f64; 256];
        a[0] = 1.0;
        b[1] = 1.0;
        let c = cosine_f64_simd(&a, &b);
        assert!(c.abs() < 1e-10, "orthogonal should be 0.0: {}", c);
    }

    #[test]
    fn cosine_matches_scalar() {
        let a: Vec<f64> = (0..333).map(|i| (i as f64 * 0.037).sin()).collect();
        let b: Vec<f64> = (0..333).map(|i| (i as f64 * 0.023).cos()).collect();

        let simd_cos = cosine_f64_simd(&a, &b);

        // Scalar reference
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum();
        let nb: f64 = b.iter().map(|x| x * x).sum();
        let scalar_cos = dot / (na * nb).sqrt();

        assert!((simd_cos - scalar_cos).abs() < 1e-10,
            "SIMD {:.12} vs scalar {:.12}", simd_cos, scalar_cos);
    }

    #[test]
    fn cosine_f32_matches_f64() {
        let a_f32: Vec<f32> = (0..500).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_f32: Vec<f32> = (0..500).map(|i| (i as f32 * 0.02).cos()).collect();

        let a_f64: Vec<f64> = a_f32.iter().map(|&v| v as f64).collect();
        let b_f64: Vec<f64> = b_f32.iter().map(|&v| v as f64).collect();

        let cos_f64 = cosine_f64_simd(&a_f64, &b_f64);
        let cos_f32 = cosine_f32_to_f64_simd(&a_f32, &b_f32);

        assert!((cos_f64 - cos_f32).abs() < 1e-6,
            "f32 {:.10} vs f64 {:.10}", cos_f32, cos_f64);
    }

    #[test]
    fn dot_f64_simd_basic() {
        let a = [1.0f64; 24];
        let b = [2.0f64; 24];
        let d = dot_f64_simd(&a, &b);
        assert!((d - 48.0).abs() < 1e-10, "24×2 = 48, got {}", d);
    }
}
