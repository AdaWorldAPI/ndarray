//! Axis-aligned bounding box batch operations.
//!
//! Provides SIMD-accelerated batch intersection, expansion, and distance
//! queries for entity collision detection.

/// Axis-aligned bounding box stored as 6 `f32` values.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::aabb::Aabb;
///
/// let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
/// let b = Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
/// assert!(a.intersects(&b));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Create a new AABB from min and max corners.
    #[inline]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Test if this AABB intersects another (inclusive on boundaries).
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Expand the AABB by `(dx, dy, dz)` in both directions per axis.
    #[inline]
    pub fn expand(&self, dx: f32, dy: f32, dz: f32) -> Self {
        Self {
            min: [self.min[0] - dx, self.min[1] - dy, self.min[2] - dz],
            max: [self.max[0] + dx, self.max[1] + dy, self.max[2] + dz],
        }
    }

    /// Test if a point is inside (or on the boundary of) this AABB.
    #[inline]
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }

    /// Volume of the AABB. Returns 0 if any dimension is degenerate.
    #[inline]
    pub fn volume(&self) -> f32 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        dx * dy * dz
    }

    /// Center point of the AABB.
    #[inline]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

/// Squared distance from a point to the nearest point on an AABB.
#[inline]
fn sq_dist_point_aabb(point: [f32; 3], aabb: &Aabb) -> f32 {
    let mut dist_sq = 0.0f32;
    for axis in 0..3 {
        let v = point[axis];
        if v < aabb.min[axis] {
            let d = aabb.min[axis] - v;
            dist_sq += d * d;
        } else if v > aabb.max[axis] {
            let d = v - aabb.max[axis];
            dist_sq += d * d;
        }
    }
    dist_sq
}

/// Test one AABB against N candidates. Returns a `Vec<bool>` indicating
/// which candidates intersect the query.
pub fn aabb_intersect_batch(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: sse4.1 detected, slice access within bounds.
            unsafe {
                return aabb_intersect_batch_sse41(query, candidates);
            }
        }
    }

    aabb_intersect_batch_scalar(query, candidates)
}

fn aabb_intersect_batch_scalar(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    candidates.iter().map(|c| query.intersects(c)).collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn aabb_intersect_batch_sse41(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    use core::arch::x86_64::*;

    // Load query min/max into SSE registers (only need xyz, ignore w).
    let q_min = _mm_set_ps(0.0, query.min[2], query.min[1], query.min[0]);
    let q_max = _mm_set_ps(f32::MAX, query.max[2], query.max[1], query.max[0]);

    let mut result = Vec::with_capacity(candidates.len());
    for c in candidates {
        let c_min = _mm_set_ps(0.0, c.min[2], c.min[1], c.min[0]);
        let c_max = _mm_set_ps(f32::MAX, c.max[2], c.max[1], c.max[0]);

        // q.min <= c.max  AND  q.max >= c.min  (per component)
        let le = _mm_cmple_ps(q_min, c_max);    // q_min[i] <= c_max[i]
        let ge = _mm_cmpge_ps(q_max, c_min);    // q_max[i] >= c_min[i]
        let both = _mm_and_ps(le, ge);
        // All 4 lanes must be true (lane 3 is always true due to sentinel values).
        let mask = _mm_movemask_ps(both);
        result.push(mask == 0xF);
    }
    result
}

/// Expand all AABBs in-place by `(dx, dy, dz)` in both directions per axis.
pub fn aabb_expand_batch(aabbs: &mut [Aabb], dx: f32, dy: f32, dz: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 detected, operating on mutable slice in-bounds.
            unsafe {
                aabb_expand_batch_sse2(aabbs, dx, dy, dz);
                return;
            }
        }
    }

    aabb_expand_batch_scalar(aabbs, dx, dy, dz);
}

fn aabb_expand_batch_scalar(aabbs: &mut [Aabb], dx: f32, dy: f32, dz: f32) {
    for a in aabbs.iter_mut() {
        a.min[0] -= dx;
        a.min[1] -= dy;
        a.min[2] -= dz;
        a.max[0] += dx;
        a.max[1] += dy;
        a.max[2] += dz;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn aabb_expand_batch_sse2(aabbs: &mut [Aabb], dx: f32, dy: f32, dz: f32) {
    use core::arch::x86_64::*;

    let delta_min = _mm_set_ps(0.0, dz, dy, dx);
    let delta_max = _mm_set_ps(0.0, dz, dy, dx);

    for a in aabbs.iter_mut() {
        let min_v = _mm_set_ps(0.0, a.min[2], a.min[1], a.min[0]);
        let max_v = _mm_set_ps(0.0, a.max[2], a.max[1], a.max[0]);

        let new_min = _mm_sub_ps(min_v, delta_min);
        let new_max = _mm_add_ps(max_v, delta_max);

        // Store back. We cannot use _mm_storeu_ps directly into [f32;3],
        // so extract components.
        let mut min_arr = [0.0f32; 4];
        let mut max_arr = [0.0f32; 4];
        _mm_storeu_ps(min_arr.as_mut_ptr(), new_min);
        _mm_storeu_ps(max_arr.as_mut_ptr(), new_max);

        a.min = [min_arr[0], min_arr[1], min_arr[2]];
        a.max = [max_arr[0], max_arr[1], max_arr[2]];
    }
}

/// Squared distance from a point to the nearest point on each AABB.
pub fn aabb_squared_distance_batch(point: [f32; 3], aabbs: &[Aabb]) -> Vec<f32> {
    aabbs.iter().map(|a| sq_dist_point_aabb(point, a)).collect()
}

/// Filter AABBs by maximum squared distance from a point. Returns indices
/// of AABBs whose nearest point is within `max_sq_dist` of `point`.
pub fn aabb_filter_by_distance(
    point: [f32; 3],
    aabbs: &[Aabb],
    max_sq_dist: f32,
) -> Vec<usize> {
    let distances = aabb_squared_distance_batch(point, aabbs);
    distances
        .iter()
        .enumerate()
        .filter(|(_, &d)| d <= max_sq_dist)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // ---------- Aabb unit tests ----------

    #[test]
    fn test_intersects_overlap() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = Aabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn test_intersects_touching() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(a.intersects(&b)); // boundary inclusive
    }

    #[test]
    fn test_no_intersect() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_no_intersect_single_axis() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([0.0, 0.0, 1.5], [1.0, 1.0, 2.5]); // only z separates
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_contains_point() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(a.contains_point([0.5, 0.5, 0.5]));
        assert!(a.contains_point([0.0, 0.0, 0.0])); // boundary
        assert!(a.contains_point([1.0, 1.0, 1.0])); // boundary
        assert!(!a.contains_point([1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_expand() {
        let a = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let expanded = a.expand(0.5, 1.0, 1.5);
        assert!(approx_eq(expanded.min[0], 0.5));
        assert!(approx_eq(expanded.min[1], 1.0));
        assert!(approx_eq(expanded.min[2], 1.5));
        assert!(approx_eq(expanded.max[0], 4.5));
        assert!(approx_eq(expanded.max[1], 6.0));
        assert!(approx_eq(expanded.max[2], 7.5));
    }

    #[test]
    fn test_volume() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        assert!(approx_eq(a.volume(), 24.0));
    }

    #[test]
    fn test_volume_degenerate() {
        let a = Aabb::new([0.0, 0.0, 0.0], [0.0, 3.0, 4.0]);
        assert!(approx_eq(a.volume(), 0.0));
    }

    #[test]
    fn test_center() {
        let a = Aabb::new([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]);
        let c = a.center();
        assert!(approx_eq(c[0], 3.0));
        assert!(approx_eq(c[1], 4.0));
        assert!(approx_eq(c[2], 5.0));
    }

    // ---------- Batch tests ----------

    #[test]
    fn test_intersect_batch() {
        let query = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let candidates = vec![
            Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),  // yes
            Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]),  // no
            Aabb::new([-1.0, -1.0, -1.0], [0.5, 0.5, 0.5]), // yes
            Aabb::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),  // yes (touching)
        ];
        let results = aabb_intersect_batch(&query, &candidates);
        assert_eq!(results, vec![true, false, true, true]);
    }

    #[test]
    fn test_intersect_batch_empty() {
        let query = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let results = aabb_intersect_batch(&query, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_intersect_batch_scalar_parity() {
        let query = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let candidates: Vec<Aabb> = (0..100)
            .map(|i| {
                let f = i as f32 * 0.1;
                Aabb::new([f - 0.5, f - 0.5, f - 0.5], [f + 0.5, f + 0.5, f + 0.5])
            })
            .collect();

        let batch = aabb_intersect_batch(&query, &candidates);
        let scalar: Vec<bool> = candidates.iter().map(|c| query.intersects(c)).collect();
        assert_eq!(batch, scalar);
    }

    #[test]
    fn test_expand_batch() {
        let mut aabbs = vec![
            Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Aabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
        ];
        aabb_expand_batch(&mut aabbs, 0.5, 1.0, 1.5);

        assert!(approx_eq(aabbs[0].min[0], -0.5));
        assert!(approx_eq(aabbs[0].max[2], 2.5));
        assert!(approx_eq(aabbs[1].min[1], 4.0));
        assert!(approx_eq(aabbs[1].max[0], 6.5));
    }

    #[test]
    fn test_expand_batch_scalar_parity() {
        let base: Vec<Aabb> = (0..50)
            .map(|i| {
                let f = i as f32;
                Aabb::new([f, f, f], [f + 1.0, f + 2.0, f + 3.0])
            })
            .collect();

        let mut batch = base.clone();
        aabb_expand_batch(&mut batch, 0.25, 0.5, 0.75);

        for (i, orig) in base.iter().enumerate() {
            let expected = orig.expand(0.25, 0.5, 0.75);
            for axis in 0..3 {
                assert!(
                    approx_eq(batch[i].min[axis], expected.min[axis]),
                    "min mismatch at [{},{}]",
                    i,
                    axis
                );
                assert!(
                    approx_eq(batch[i].max[axis], expected.max[axis]),
                    "max mismatch at [{},{}]",
                    i,
                    axis
                );
            }
        }
    }

    // ---------- Distance tests ----------

    #[test]
    fn test_squared_distance_inside() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let d = sq_dist_point_aabb([1.0, 1.0, 1.0], &a);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_squared_distance_outside() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Point is 1 unit away on x-axis
        let d = sq_dist_point_aabb([2.0, 0.5, 0.5], &a);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_squared_distance_corner() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Point at (2,2,2): distance to corner (1,1,1) = sqrt(3), sq=3
        let d = sq_dist_point_aabb([2.0, 2.0, 2.0], &a);
        assert!(approx_eq(d, 3.0));
    }

    #[test]
    fn test_squared_distance_batch() {
        let aabbs = vec![
            Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Aabb::new([10.0, 10.0, 10.0], [11.0, 11.0, 11.0]),
        ];
        let dists = aabb_squared_distance_batch([0.5, 0.5, 0.5], &aabbs);
        assert!(approx_eq(dists[0], 0.0));   // inside
        assert!(dists[1] > 200.0);           // far away
    }

    #[test]
    fn test_filter_by_distance() {
        let aabbs = vec![
            Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),     // 0: dist=0
            Aabb::new([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]),     // 1: dist=2, sq=4
            Aabb::new([10.0, 10.0, 10.0], [11.0, 11.0, 11.0]),// 2: far
        ];
        let indices = aabb_filter_by_distance([0.5, 0.5, 0.5], &aabbs, 5.0);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_filter_by_distance_none() {
        let aabbs = vec![
            Aabb::new([100.0, 100.0, 100.0], [101.0, 101.0, 101.0]),
        ];
        let indices = aabb_filter_by_distance([0.0, 0.0, 0.0], &aabbs, 1.0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_filter_by_distance_all() {
        let aabbs = vec![
            Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
        ];
        let indices = aabb_filter_by_distance([0.7, 0.7, 0.7], &aabbs, 100.0);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_self_intersection() {
        let a = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert!(a.intersects(&a));
    }

    #[test]
    fn test_zero_volume_aabb_intersects() {
        let a = Aabb::new([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]); // point
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }
}
