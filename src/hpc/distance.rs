//! Batch distance computations for spatial queries.
//!
//! SIMD-accelerated squared-distance, radius filtering, and K-nearest-neighbor
//! searches over contiguous point slices. All operations work on borrowed slices
//! with no internal copies. Scalar fallback is provided for non-x86 targets.

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
    pub(crate) unsafe fn squared_distances_avx2(
        query: [f32; 3],
        points: &[[f32; 3]],
        out: &mut Vec<f32>,
    ) {
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
        for j in i..n {
            let dx = query[0] - points[j][0];
            let dy = query[1] - points[j][1];
            let dz = query[2] - points[j][2];
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
pub fn filter_by_radius_sq(
    query: [f32; 3],
    points: &[[f32; 3]],
    radius_sq: f32,
) -> Vec<usize> {
    let dists = squared_distances_f32(query, points);
    dists
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d <= radius_sq { Some(i) } else { None })
        .collect()
}

/// Find K nearest points (f32). Returns `(indices, squared_distances)` sorted
/// ascending by distance.
pub fn knn_f32(
    query: [f32; 3],
    points: &[[f32; 3]],
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
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
pub fn filter_by_radius_sq_f64(
    query: [f64; 3],
    points: &[[f64; 3]],
    radius_sq: f64,
) -> Vec<usize> {
    let dists = squared_distances_f64(query, points);
    dists
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d <= radius_sq { Some(i) } else { None })
        .collect()
}

/// Find K nearest points (f64). Returns `(indices, squared_distances)` sorted
/// ascending by distance.
pub fn knn_f64(
    query: [f64; 3],
    points: &[[f64; 3]],
    k: usize,
) -> (Vec<usize>, Vec<f64>) {
    let dists = squared_distances_f64(query, points);
    let mut indexed: Vec<(usize, f64)> = dists.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let take = k.min(indexed.len());
    let indices: Vec<usize> = indexed[..take].iter().map(|&(i, _)| i).collect();
    let sq_dists: Vec<f64> = indexed[..take].iter().map(|&(_, d)| d).collect();
    (indices, sq_dists)
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
            assert!(
                approx_eq_f32(d, expected),
                "mismatch at {i}: {d} vs {expected}"
            );
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
            assert!(
                approx_eq_f64(d, expected),
                "mismatch at {i}: {d} vs {expected}"
            );
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
        let points = vec![
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
        let (idx, dist) = knn_f32(query, &points, 2);
        assert_eq!(idx, vec![3, 1]); // 0.25, 1.0
        assert!(approx_eq_f32(dist[0], 0.25));
        assert!(approx_eq_f32(dist[1], 1.0));
    }

    #[test]
    fn test_knn_f64() {
        let query = [0.0f64, 0.0, 0.0];
        let points = vec![
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
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
}
