//! Batch distance computations for spatial queries.
//!
//! SIMD-accelerated squared-distance, radius filtering, and K-nearest-neighbor
//! searches over contiguous point slices. All operations work on borrowed slices
//! with no internal copies. Scalar fallback is provided for non-x86 targets.
//!
//! # Slice-shape geometric distance (PR-X10 A6)
//!
//! For arbitrary-length f64 slices (non-3D-point shape), use:
//!
//! - [`l1_f64_simd`]  — Manhattan: `Σ |a_i − b_i|`
//! - [`l2_f64_simd`]  — Euclidean: `√Σ (a_i − b_i)²`
//! - [`linf_f64_simd`] — Chebyshev: `max |a_i − b_i|`
//!
//! These use the `F64x8` polyfill (no `target_feature`, no `unsafe`),
//! matching the [`crate::hpc::heel_f64x8::cosine_f64_simd`] idiom: F64x8
//! chunks with FMA / SIMD-max accumulator + scalar remainder. They are
//! the salvaged kernels from the rolled-back PR #160 cross-repo arc
//! (lance-graph `heel_f64x8::{l1, l2, linf}_f64_simd`), re-landed here
//! per the linalg-core design's A6 worker scope and the
//! `crate::hpc::linalg/mod.rs` hard boundary ("No distance metrics —
//! those live in `crate::hpc::distance`").

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

#[inline]
fn sq_dist_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn sq_dist_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ---------------------------------------------------------------------------
// SIMD (x86_64 AVX2) internals
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_impl {
    use crate::simd::F32x8;

    /// Compute squared distances for 8 points at a time using F32x8 polyfill.
    /// `query` components are broadcast; `points` is read in SOA-style chunks.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn squared_distances_avx2(query: [f32; 3], points: &[[f32; 3]], out: &mut Vec<f32>) {
        let n = points.len();
        out.clear();
        out.reserve(n);

        let qx = F32x8::splat(query[0]);
        let qy = F32x8::splat(query[1]);
        let qz = F32x8::splat(query[2]);

        // Each point is 3 floats => stride 3
        let mut i = 0usize;
        // Process 8 points at a time
        while i + 8 <= n {
            // Gather x, y, z for 8 points (scalar gather — AVX2 gather is slow
            // on many microarchitectures for non-contiguous strides).
            let mut xs = [0f32; 8];
            let mut ys = [0f32; 8];
            let mut zs = [0f32; 8];
            for j in 0..8 {
                xs[j] = points[i + j][0];
                ys[j] = points[i + j][1];
                zs[j] = points[i + j][2];
            }

            let vx = F32x8::from_array(xs);
            let vy = F32x8::from_array(ys);
            let vz = F32x8::from_array(zs);

            let dx = qx - vx;
            let dy = qy - vy;
            let dz = qz - vz;

            // dx*dx + dy*dy + dz*dz
            let acc = dx * dx + dy * dy + dz * dz;

            out.extend_from_slice(&acc.to_array());

            i += 8;
        }

        // Scalar tail
        for p in &points[i..n] {
            let dx = query[0] - p[0];
            let dy = query[1] - p[1];
            let dz = query[2] - p[2];
            out.push(dx * dx + dy * dy + dz * dz);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API — f32
// ---------------------------------------------------------------------------

/// Squared distance from one point to N points (f32).
///
/// Returns a `Vec<f32>` of length `points.len()`.
pub fn squared_distances_f32(query: [f32; 3], points: &[[f32; 3]]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if super::simd_caps::simd_caps().avx2 {
            let mut out = Vec::new();
            // SAFETY: feature detected above.
            unsafe { simd_impl::squared_distances_avx2(query, points, &mut out) };
            return out;
        }
    }
    // Scalar fallback
    points.iter().map(|p| sq_dist_f32(query, *p)).collect()
}

/// Filter points by max squared distance. Returns indices of survivors.
pub fn filter_by_radius_sq(query: [f32; 3], points: &[[f32; 3]], radius_sq: f32) -> Vec<usize> {
    let dists = squared_distances_f32(query, points);
    dists
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d <= radius_sq { Some(i) } else { None })
        .collect()
}

/// Find K nearest points (f32). Returns `(indices, squared_distances)` sorted
/// ascending by distance.
pub fn knn_f32(query: [f32; 3], points: &[[f32; 3]], k: usize) -> (Vec<usize>, Vec<f32>) {
    let dists = squared_distances_f32(query, points);
    let mut indexed: Vec<(usize, f32)> = dists.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let take = k.min(indexed.len());
    let indices: Vec<usize> = indexed[..take].iter().map(|&(i, _)| i).collect();
    let sq_dists: Vec<f32> = indexed[..take].iter().map(|&(_, d)| d).collect();
    (indices, sq_dists)
}

// ---------------------------------------------------------------------------
// Public API — f64
// ---------------------------------------------------------------------------

/// Squared distance from one point to N points (f64).
///
/// Uses scalar path (AVX2 f64 lanes are only 4-wide so the gain is marginal
/// for AOS-3 data).
pub fn squared_distances_f64(query: [f64; 3], points: &[[f64; 3]]) -> Vec<f64> {
    points.iter().map(|p| sq_dist_f64(query, *p)).collect()
}

/// Filter f64 points by squared-distance radius. Returns survivor indices.
pub fn filter_by_radius_sq_f64(query: [f64; 3], points: &[[f64; 3]], radius_sq: f64) -> Vec<usize> {
    let dists = squared_distances_f64(query, points);
    dists
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d <= radius_sq { Some(i) } else { None })
        .collect()
}

/// Find K nearest points (f64). Returns `(indices, squared_distances)` sorted
/// ascending by distance.
pub fn knn_f64(query: [f64; 3], points: &[[f64; 3]], k: usize) -> (Vec<usize>, Vec<f64>) {
    let dists = squared_distances_f64(query, points);
    let mut indexed: Vec<(usize, f64)> = dists.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let take = k.min(indexed.len());
    let indices: Vec<usize> = indexed[..take].iter().map(|&(i, _)| i).collect();
    let sq_dists: Vec<f64> = indexed[..take].iter().map(|&(_, d)| d).collect();
    (indices, sq_dists)
}

// ---------------------------------------------------------------------------
// Slice-shape geometric distance — PR-X10 A6
// ---------------------------------------------------------------------------
//
// Polyfilled F64x8 chunked path with scalar remainder; no `target_feature`,
// no `unsafe` — the polyfill in `crate::simd::F64x8` owns runtime feature
// dispatch (AVX-512 native zmm / AVX2 2×ymm / scalar [f64; 8]).
//
// All three kernels read `min(a.len(), b.len())` elements. Empty inputs
// return 0.0.

use crate::simd::F64x8;

/// L1 (Manhattan) distance between two f64 slices: `Σ |a_i − b_i|`.
///
/// EXACT precision class — the per-lane `(a - b).abs()` introduces no
/// rounding beyond the standard subtract, and the reduce-sum order is
/// lane-tree within each F64x8 chunk + sequential across chunks (matches
/// the [`crate::hpc::heel_f64x8::cosine_f64_simd`] order so callers can
/// reason about determinism the same way).
///
/// Reads `min(a.len(), b.len())` elements. Returns 0.0 for empty inputs.
pub fn l1_f64_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = F64x8::splat(0.0);
    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        let vb = F64x8::from_slice(&b[i * 8..]);
        acc += (va - vb).abs();
    }
    let mut sum = acc.reduce_sum();
    let offset = chunks * 8;
    for i in 0..(n - offset) {
        sum += (a[offset + i] - b[offset + i]).abs();
    }
    sum
}

/// L2 (Euclidean) distance between two f64 slices: `√Σ (a_i − b_i)²`.
///
/// VERIFY precision class — the final `sqrt` is one ULP; the sum is
/// lane-tree within each F64x8 + sequential across chunks (same order
/// pattern as L1). Determinism across runs holds for fixed slice
/// length and fixed chunking. For full order-independence use a
/// pairwise-reduce variant (see `blas_level1::nrm2`).
///
/// Reads `min(a.len(), b.len())` elements. Returns 0.0 for empty inputs.
pub fn l2_f64_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = F64x8::splat(0.0);
    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        let vb = F64x8::from_slice(&b[i * 8..]);
        let d = va - vb;
        acc = d.mul_add(d, acc); // acc += d*d (single FMA per chunk)
    }
    let mut sum_sq = acc.reduce_sum();
    let offset = chunks * 8;
    for i in 0..(n - offset) {
        let d = a[offset + i] - b[offset + i];
        sum_sq += d * d;
    }
    sum_sq.sqrt()
}

/// L∞ (Chebyshev) distance between two f64 slices: `max |a_i − b_i|`.
///
/// EXACT precision class — `(a - b).abs()` and `max` introduce no
/// rounding; the result is determined by the inputs alone (order-
/// independent across chunks since `max` is associative and commutative
/// under IEEE-754 for non-NaN inputs).
///
/// Reads `min(a.len(), b.len())` elements. Returns 0.0 for empty inputs.
///
/// # NaN handling
///
/// IEEE-754 `_mm512_max_pd` returns the second operand when either input
/// is NaN; callers passing NaN-tainted slices may observe non-deterministic
/// max across runs (an upstream constraint, not a kernel bug). Audit
/// upstream for NaN before relying on this kernel on production data.
pub fn linf_f64_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut max_v = F64x8::splat(0.0);
    for i in 0..chunks {
        let va = F64x8::from_slice(&a[i * 8..]);
        let vb = F64x8::from_slice(&b[i * 8..]);
        max_v = max_v.simd_max((va - vb).abs());
    }
    let mut max_d = max_v.reduce_max();
    let offset = chunks * 8;
    for i in 0..(n - offset) {
        let d = (a[offset + i] - b[offset + i]).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    // -- scalar parity --

    #[test]
    fn test_squared_distances_f32_matches_scalar() {
        let query = [1.0f32, 2.0, 3.0];
        let points: Vec<[f32; 3]> = (0..33)
            .map(|i| {
                let v = i as f32;
                [v, v + 1.0, v + 2.0]
            })
            .collect();
        let result = squared_distances_f32(query, &points);
        assert_eq!(result.len(), points.len());
        for (i, &d) in result.iter().enumerate() {
            let expected = sq_dist_f32(query, points[i]);
            assert!(approx_eq_f32(d, expected), "mismatch at {i}: {d} vs {expected}");
        }
    }

    #[test]
    fn test_squared_distances_f64_matches_scalar() {
        let query = [1.0f64, 2.0, 3.0];
        let points: Vec<[f64; 3]> = (0..33)
            .map(|i| {
                let v = i as f64;
                [v, v + 1.0, v + 2.0]
            })
            .collect();
        let result = squared_distances_f64(query, &points);
        for (i, &d) in result.iter().enumerate() {
            let expected = sq_dist_f64(query, points[i]);
            assert!(approx_eq_f64(d, expected), "mismatch at {i}: {d} vs {expected}");
        }
    }

    // -- filter --

    #[test]
    fn test_filter_by_radius_sq() {
        let query = [0.0f32, 0.0, 0.0];
        let points = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let result = filter_by_radius_sq(query, &points, 1.0);
        // Point 0: dist=1.0, pass; Point 1: dist=4.0, fail; Point 2: dist=0.25, pass
        assert_eq!(result, vec![0, 2]);
    }

    #[test]
    fn test_filter_by_radius_sq_f64() {
        let query = [0.0f64, 0.0, 0.0];
        let points = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let result = filter_by_radius_sq_f64(query, &points, 1.0);
        assert_eq!(result, vec![0, 2]);
    }

    #[test]
    fn test_filter_vs_brute_force_f32() {
        let query = [5.0f32, 5.0, 5.0];
        let points: Vec<[f32; 3]> = (0..100)
            .map(|i| {
                let v = i as f32 * 0.3;
                [v, v, v]
            })
            .collect();
        let radius_sq = 10.0f32;
        let result = filter_by_radius_sq(query, &points, radius_sq);
        let brute: Vec<usize> = points
            .iter()
            .enumerate()
            .filter(|(_, p)| sq_dist_f32(query, **p) <= radius_sq)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(result, brute);
    }

    // -- knn --

    #[test]
    fn test_knn_f32() {
        let query = [0.0f32, 0.0, 0.0];
        let points = vec![[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let (idx, dist) = knn_f32(query, &points, 2);
        assert_eq!(idx, vec![3, 1]); // 0.25, 1.0
        assert!(approx_eq_f32(dist[0], 0.25));
        assert!(approx_eq_f32(dist[1], 1.0));
    }

    #[test]
    fn test_knn_f64() {
        let query = [0.0f64, 0.0, 0.0];
        let points = vec![[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let (idx, dist) = knn_f64(query, &points, 2);
        assert_eq!(idx, vec![3, 1]);
        assert!(approx_eq_f64(dist[0], 0.25));
        assert!(approx_eq_f64(dist[1], 1.0));
    }

    #[test]
    fn test_knn_k_larger_than_n() {
        let query = [0.0f32, 0.0, 0.0];
        let points = vec![[1.0, 0.0, 0.0]];
        let (idx, dist) = knn_f32(query, &points, 10);
        assert_eq!(idx.len(), 1);
        assert_eq!(dist.len(), 1);
    }

    // -- edge cases --

    #[test]
    fn test_empty_points() {
        let query = [0.0f32, 0.0, 0.0];
        let empty: &[[f32; 3]] = &[];
        assert!(squared_distances_f32(query, empty).is_empty());
        assert!(filter_by_radius_sq(query, empty, 1.0).is_empty());
        let (idx, dist) = knn_f32(query, empty, 5);
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn test_single_point() {
        let query = [0.0f32, 0.0, 0.0];
        let points = vec![[1.0, 1.0, 1.0]];
        let result = squared_distances_f32(query, &points);
        assert_eq!(result.len(), 1);
        assert!(approx_eq_f32(result[0], 3.0));
    }

    #[test]
    fn test_zero_distance() {
        let query = [5.0f32, 10.0, 15.0];
        let points = vec![query];
        let result = squared_distances_f32(query, &points);
        assert!(approx_eq_f32(result[0], 0.0));
    }

    // -- PR-X10 A6 slice-shape L1 / L2 / L∞ --

    fn approx_eq_f64_tol(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Deterministic SplitMix64 — matches the pillar harness so the
    /// corpus is reproducible across runs and across machines.
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_vec_f64(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let bits = splitmix(&mut s) >> 11;
                (bits as f64) / (1u64 << 53) as f64 * 2.0 - 1.0 // uniform in [-1, 1)
            })
            .collect()
    }

    // -- L1 boundary + parity --

    #[test]
    fn l1_f64_simd_self_zero() {
        let a = random_vec_f64(0xC1A0, 200);
        assert_eq!(l1_f64_simd(&a, &a), 0.0);
    }

    #[test]
    fn l1_f64_simd_empty_is_zero() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(l1_f64_simd(&a, &b), 0.0);
    }

    #[test]
    fn l1_f64_simd_uniform_diff() {
        let a = vec![3.0f64; 17];
        let b = vec![1.0f64; 17];
        // 17 * |3 - 1| = 34
        assert!(approx_eq_f64_tol(l1_f64_simd(&a, &b), 34.0, 1e-12));
    }

    #[test]
    fn l1_f64_simd_matches_scalar() {
        // 200 elements covers chunked path (25 chunks of 8) + remainder of 0;
        // 199 covers chunked + remainder of 7.
        for &n in &[1usize, 7, 8, 15, 16, 17, 64, 199, 200, 1024] {
            let a = random_vec_f64(0xA110_C1A0, n);
            let b = random_vec_f64(0xB220_C1A0, n);
            let simd = l1_f64_simd(&a, &b);
            let scalar: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
            assert!(approx_eq_f64_tol(simd, scalar, 1e-11), "n={} simd={:.15} scalar={:.15}", n, simd, scalar);
        }
    }

    // -- L2 boundary + parity --

    #[test]
    fn l2_f64_simd_self_zero() {
        let a = random_vec_f64(0xC2A0, 200);
        assert_eq!(l2_f64_simd(&a, &a), 0.0);
    }

    #[test]
    fn l2_f64_simd_empty_is_zero() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(l2_f64_simd(&a, &b), 0.0);
    }

    #[test]
    fn l2_f64_simd_pythagoras() {
        // (3, 0, …) vs (0, 4, …): √(9 + 16) = 5
        let a = vec![3.0f64, 0.0];
        let b = vec![0.0f64, 4.0];
        assert!(approx_eq_f64_tol(l2_f64_simd(&a, &b), 5.0, 1e-12));
    }

    #[test]
    fn l2_f64_simd_matches_scalar() {
        for &n in &[1usize, 7, 8, 15, 16, 17, 64, 199, 200, 1024] {
            let a = random_vec_f64(0xA110_C2A0, n);
            let b = random_vec_f64(0xB220_C2A0, n);
            let simd = l2_f64_simd(&a, &b);
            let sum_sq: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
            let scalar = sum_sq.sqrt();
            // Sqrt is 1 ULP; cross-chunk summation order differs by chunks
            // of 8 vs sequential — allow generous relative tolerance.
            let rel = (simd - scalar).abs() / scalar.max(1e-12);
            assert!(rel < 1e-10, "n={} simd={:.15} scalar={:.15} rel={:.2e}", n, simd, scalar, rel);
        }
    }

    // -- L∞ boundary + parity --

    #[test]
    fn linf_f64_simd_self_zero() {
        let a = random_vec_f64(0xC1FF, 200);
        assert_eq!(linf_f64_simd(&a, &a), 0.0);
    }

    #[test]
    fn linf_f64_simd_empty_is_zero() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(linf_f64_simd(&a, &b), 0.0);
    }

    #[test]
    fn linf_f64_simd_picks_max_in_chunk() {
        // Max difference must land inside a chunked path (index 5 < 8) and
        // also outside (index 13 > 8) to exercise both halves.
        let mut a = vec![0.0f64; 16];
        let mut b = vec![0.0f64; 16];
        a[5] = 0.5;
        a[13] = -0.7; // |Δ| = 0.7 — should win
        b[2] = 0.1;
        assert!(approx_eq_f64_tol(linf_f64_simd(&a, &b), 0.7, 1e-12));
    }

    #[test]
    fn linf_f64_simd_matches_scalar() {
        for &n in &[1usize, 7, 8, 15, 16, 17, 64, 199, 200, 1024] {
            let a = random_vec_f64(0xA110_C1FF, n);
            let b = random_vec_f64(0xB220_C1FF, n);
            let simd = linf_f64_simd(&a, &b);
            let scalar: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max);
            assert!(approx_eq_f64_tol(simd, scalar, 1e-15), "n={} simd={:.15} scalar={:.15}", n, simd, scalar);
        }
    }

    /// Mismatched-length slices: must use the shorter length, no panic.
    #[test]
    fn slice_distances_mismatched_length_uses_min() {
        let a = vec![1.0f64; 17];
        let b = vec![2.0f64; 10];
        // L1 over min=10: 10 * |1 - 2| = 10
        assert!(approx_eq_f64_tol(l1_f64_simd(&a, &b), 10.0, 1e-12));
        // L2 over min=10: √(10 * 1) = √10
        assert!(approx_eq_f64_tol(l2_f64_simd(&a, &b), 10f64.sqrt(), 1e-12));
        // L∞ = 1
        assert!(approx_eq_f64_tol(linf_f64_simd(&a, &b), 1.0, 1e-12));
    }
}
