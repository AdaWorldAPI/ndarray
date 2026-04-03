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

/// Ray definition for projectile collision testing.
///
/// `inv_dir` must be precomputed as `1.0 / direction` for each axis.
/// If a direction component is zero, the corresponding `inv_dir` should be
/// `f32::INFINITY` or `f32::NEG_INFINITY`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::aabb::Ray;
///
/// let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]); // +X direction
/// assert_eq!(ray.inv_dir[0], 1.0);
/// assert!(ray.inv_dir[1].is_infinite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Ray {
    pub origin: [f32; 3],
    pub inv_dir: [f32; 3],
}

impl Ray {
    /// Create a ray from origin and direction (auto-computes `inv_dir`).
    #[inline]
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self {
            origin,
            inv_dir: [
                1.0 / direction[0],
                1.0 / direction[1],
                1.0 / direction[2],
            ],
        }
    }

    /// Create a ray from origin and precomputed inverse direction.
    #[inline]
    pub fn from_inv_dir(origin: [f32; 3], inv_dir: [f32; 3]) -> Self {
        Self { origin, inv_dir }
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
        let caps = super::simd_caps::simd_caps();
        if caps.avx512f && candidates.len() >= 16 {
            // SAFETY: avx512f detected via simd_caps singleton.
            unsafe {
                return aabb_intersect_batch_avx512(query, candidates);
            }
        }
        if caps.sse41 {
            // SAFETY: sse4.1 detected via simd_caps singleton.
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

/// AVX-512 batch AABB intersection: tests 16 candidates per axis comparison.
///
/// Broadcasts query min/max per axis, gathers candidate coords into F32x16,
/// compares all 16 at once using `simd_le` / `simd_ge`, ANDs the 6 comparison
/// masks.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn aabb_intersect_batch_avx512(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    use crate::simd::{F32x16, F32Mask16};

    let mut result = Vec::with_capacity(candidates.len());

    // Process 16 candidates at a time
    let chunks = candidates.len() / 16;
    for c in 0..chunks {
        let base = c * 16;
        // Gather min/max coords for 16 candidates into SoA arrays
        let mut c_min_x = [0.0f32; 16];
        let mut c_max_x = [0.0f32; 16];
        let mut c_min_y = [0.0f32; 16];
        let mut c_max_y = [0.0f32; 16];
        let mut c_min_z = [0.0f32; 16];
        let mut c_max_z = [0.0f32; 16];

        for i in 0..16 {
            let cand = &candidates[base + i];
            c_min_x[i] = cand.min[0];
            c_max_x[i] = cand.max[0];
            c_min_y[i] = cand.min[1];
            c_max_y[i] = cand.max[1];
            c_min_z[i] = cand.min[2];
            c_max_z[i] = cand.max[2];
        }

        let v_c_min_x = F32x16::from_array(c_min_x);
        let v_c_max_x = F32x16::from_array(c_max_x);
        let v_c_min_y = F32x16::from_array(c_min_y);
        let v_c_max_y = F32x16::from_array(c_max_y);
        let v_c_min_z = F32x16::from_array(c_min_z);
        let v_c_max_z = F32x16::from_array(c_max_z);

        // Broadcast query bounds
        let q_min_x = F32x16::splat(query.min[0]);
        let q_max_x = F32x16::splat(query.max[0]);
        let q_min_y = F32x16::splat(query.min[1]);
        let q_max_y = F32x16::splat(query.max[1]);
        let q_min_z = F32x16::splat(query.min[2]);
        let q_max_z = F32x16::splat(query.max[2]);

        // 6 intersection conditions: q.min[i] <= c.max[i] && q.max[i] >= c.min[i]
        let m1 = q_min_x.simd_le(v_c_max_x);
        let m2 = q_max_x.simd_ge(v_c_min_x);
        let m3 = q_min_y.simd_le(v_c_max_y);
        let m4 = q_max_y.simd_ge(v_c_min_y);
        let m5 = q_min_z.simd_le(v_c_max_z);
        let m6 = q_max_z.simd_ge(v_c_min_z);

        let all = m1.0 & m2.0 & m3.0 & m4.0 & m5.0 & m6.0;

        for i in 0..16 {
            result.push((all >> i) & 1 != 0);
        }
    }

    // Scalar tail
    for i in (chunks * 16)..candidates.len() {
        result.push(query.intersects(&candidates[i]));
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn aabb_intersect_batch_sse41(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    // Scalar per-candidate test — LLVM auto-vectorizes with target-cpu=x86-64-v4
    let mut result = Vec::with_capacity(candidates.len());
    for c in candidates {
        let hit = query.min[0] <= c.max[0]
            && query.max[0] >= c.min[0]
            && query.min[1] <= c.max[1]
            && query.max[1] >= c.min[1]
            && query.min[2] <= c.max[2]
            && query.max[2] >= c.min[2];
        result.push(hit);
    }
    result
}

/// Batch ray-AABB slab test for projectile collision.
///
/// Returns `(hit_mask, t_values)` where `hit_mask[i]` is `true` if the ray
/// intersects `aabbs[i]`, and `t_values[i]` is the entry `t` parameter
/// (or `f32::MAX` if no hit).
///
/// Uses the slab method: `t_enter = max(t_x_enter, t_y_enter, t_z_enter)`,
/// `t_exit = min(t_x_exit, t_y_exit, t_z_exit)`. Intersection when
/// `t_enter <= t_exit && t_exit >= 0`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::aabb::{Aabb, Ray, ray_aabb_slab_test_batch};
///
/// let ray = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
/// let aabbs = vec![
///     Aabb::new([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),  // hit at t=2
///     Aabb::new([0.0, 5.0, 0.0], [1.0, 6.0, 1.0]),  // miss
/// ];
/// let (hits, ts) = ray_aabb_slab_test_batch(&ray, &aabbs);
/// assert!(hits[0]);
/// assert!(!hits[1]);
/// ```
pub fn ray_aabb_slab_test_batch(ray: &Ray, aabbs: &[Aabb]) -> (Vec<bool>, Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::simd_caps::simd_caps().avx512f && aabbs.len() >= 16 {
            // SAFETY: avx512f detected via simd_caps singleton.
            unsafe {
                return ray_aabb_slab_test_avx512(ray, aabbs);
            }
        }
    }
    ray_aabb_slab_test_scalar(ray, aabbs)
}

fn ray_aabb_slab_test_scalar(ray: &Ray, aabbs: &[Aabb]) -> (Vec<bool>, Vec<f32>) {
    let mut hits = Vec::with_capacity(aabbs.len());
    let mut t_values = Vec::with_capacity(aabbs.len());

    for aabb in aabbs {
        let mut t_enter = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;

        for axis in 0..3 {
            let t1 = (aabb.min[axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t2 = (aabb.max[axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t_near = t1.min(t2);
            let t_far = t1.max(t2);
            t_enter = t_enter.max(t_near);
            t_exit = t_exit.min(t_far);
        }

        let hit = t_enter <= t_exit && t_exit >= 0.0;
        hits.push(hit);
        t_values.push(if hit { t_enter.max(0.0) } else { f32::MAX });
    }

    (hits, t_values)
}

/// AVX-512 batch ray-AABB slab test: processes 16 AABBs per iteration.
///
/// Broadcasts ray origin and inv_dir per axis, gathers candidate min/max
/// coords into SoA arrays, computes slab intervals with `simd_min` /
/// `simd_max`, and combines masks with `simd_le` / `simd_ge`.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ray_aabb_slab_test_avx512(ray: &Ray, aabbs: &[Aabb]) -> (Vec<bool>, Vec<f32>) {
    use crate::simd::F32x16;

    let mut hits = Vec::with_capacity(aabbs.len());
    let mut t_values = Vec::with_capacity(aabbs.len());

    // Broadcast ray origin and inv_dir per axis
    let orig_x = F32x16::splat(ray.origin[0]);
    let orig_y = F32x16::splat(ray.origin[1]);
    let orig_z = F32x16::splat(ray.origin[2]);
    let inv_x = F32x16::splat(ray.inv_dir[0]);
    let inv_y = F32x16::splat(ray.inv_dir[1]);
    let inv_z = F32x16::splat(ray.inv_dir[2]);
    let zero = F32x16::splat(0.0);

    // Process 16 AABBs at a time
    let chunks = aabbs.len() / 16;
    for c in 0..chunks {
        let base = c * 16;

        // Gather min/max coords for 16 AABBs into SoA arrays
        let mut a_min_x = [0.0f32; 16];
        let mut a_max_x = [0.0f32; 16];
        let mut a_min_y = [0.0f32; 16];
        let mut a_max_y = [0.0f32; 16];
        let mut a_min_z = [0.0f32; 16];
        let mut a_max_z = [0.0f32; 16];

        for i in 0..16 {
            let aabb = &aabbs[base + i];
            a_min_x[i] = aabb.min[0];
            a_max_x[i] = aabb.max[0];
            a_min_y[i] = aabb.min[1];
            a_max_y[i] = aabb.max[1];
            a_min_z[i] = aabb.min[2];
            a_max_z[i] = aabb.max[2];
        }

        let v_min_x = F32x16::from_array(a_min_x);
        let v_max_x = F32x16::from_array(a_max_x);
        let v_min_y = F32x16::from_array(a_min_y);
        let v_max_y = F32x16::from_array(a_max_y);
        let v_min_z = F32x16::from_array(a_min_z);
        let v_max_z = F32x16::from_array(a_max_z);

        // X axis: t1 = (min - origin) * inv_dir, t2 = (max - origin) * inv_dir
        let t1_x = (v_min_x - orig_x) * inv_x;
        let t2_x = (v_max_x - orig_x) * inv_x;
        let t_near_x = t1_x.simd_min(t2_x);
        let t_far_x = t1_x.simd_max(t2_x);

        // Y axis
        let t1_y = (v_min_y - orig_y) * inv_y;
        let t2_y = (v_max_y - orig_y) * inv_y;
        let t_near_y = t1_y.simd_min(t2_y);
        let t_far_y = t1_y.simd_max(t2_y);

        // Z axis
        let t1_z = (v_min_z - orig_z) * inv_z;
        let t2_z = (v_max_z - orig_z) * inv_z;
        let t_near_z = t1_z.simd_min(t2_z);
        let t_far_z = t1_z.simd_max(t2_z);

        // t_enter = max(t_near_x, t_near_y, t_near_z)
        let t_enter = t_near_x.simd_max(t_near_y).simd_max(t_near_z);
        // t_exit = min(t_far_x, t_far_y, t_far_z)
        let t_exit = t_far_x.simd_min(t_far_y).simd_min(t_far_z);

        // hit = t_enter <= t_exit AND t_exit >= 0
        let m_le = t_enter.simd_le(t_exit);
        let m_ge = t_exit.simd_ge(zero);
        let hit_mask = m_le.0 & m_ge.0;

        // Clamp t_enter to 0 for origins inside box
        let t_enter_clamped = t_enter.simd_max(zero);
        let t_arr = t_enter_clamped.to_array();

        for i in 0..16 {
            let hit = (hit_mask >> i) & 1 != 0;
            hits.push(hit);
            t_values.push(if hit { t_arr[i] } else { f32::MAX });
        }
    }

    // Scalar tail for remainder
    for i in (chunks * 16)..aabbs.len() {
        let aabb = &aabbs[i];
        let mut t_enter = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;

        for axis in 0..3 {
            let t1 = (aabb.min[axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t2 = (aabb.max[axis] - ray.origin[axis]) * ray.inv_dir[axis];
            let t_near = t1.min(t2);
            let t_far = t1.max(t2);
            t_enter = t_enter.max(t_near);
            t_exit = t_exit.min(t_far);
        }

        let hit = t_enter <= t_exit && t_exit >= 0.0;
        hits.push(hit);
        t_values.push(if hit { t_enter.max(0.0) } else { f32::MAX });
    }

    (hits, t_values)
}

/// Expand all AABBs in-place by `(dx, dy, dz)` in both directions per axis.
pub fn aabb_expand_batch(aabbs: &mut [Aabb], dx: f32, dy: f32, dz: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::simd_caps::simd_caps().sse2 {
            // SAFETY: sse2 detected via simd_caps singleton.
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
    // Scalar per-AABB expand — LLVM auto-vectorizes with target-cpu=x86-64-v4
    for a in aabbs.iter_mut() {
        a.min[0] -= dx;
        a.min[1] -= dy;
        a.min[2] -= dz;
        a.max[0] += dx;
        a.max[1] += dy;
        a.max[2] += dz;
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
            Aabb::new([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),     // 1: nearest pt (2,0.5,0.5), dist=1.5, sq=2.25
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

    // ---------- AVX-512 batch intersect parity ----------

    #[test]
    fn test_intersect_batch_avx512_parity() {
        let query = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Generate enough candidates to exercise AVX-512 (>= 16) + tail
        let candidates: Vec<Aabb> = (0..100)
            .map(|i| {
                let f = i as f32 * 0.1;
                Aabb::new([f - 0.5, f - 0.5, f - 0.5], [f + 0.5, f + 0.5, f + 0.5])
            })
            .collect();

        let batch = aabb_intersect_batch(&query, &candidates);
        let scalar: Vec<bool> = candidates.iter().map(|c| query.intersects(c)).collect();
        assert_eq!(batch, scalar, "AVX-512 batch intersect must match scalar");
    }

    // ---------- Ray-AABB slab test ----------

    #[test]
    fn test_ray_aabb_hit_along_x() {
        let ray = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let aabbs = vec![Aabb::new([2.0, 0.0, 0.0], [3.0, 1.0, 1.0])];
        let (hits, ts) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert!(hits[0]);
        assert!(approx_eq(ts[0], 2.0));
    }

    #[test]
    fn test_ray_aabb_miss() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let aabbs = vec![Aabb::new([0.0, 5.0, 0.0], [1.0, 6.0, 1.0])];
        let (hits, _) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert!(!hits[0]);
    }

    #[test]
    fn test_ray_aabb_origin_inside() {
        let ray = Ray::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let aabbs = vec![Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
        let (hits, ts) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert!(hits[0]);
        assert!(approx_eq(ts[0], 0.0)); // origin inside → t=0
    }

    #[test]
    fn test_ray_aabb_behind_ray() {
        let ray = Ray::new([5.0, 0.5, 0.5], [1.0, 0.0, 0.0]); // +X from x=5
        let aabbs = vec![Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])]; // behind
        let (hits, _) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert!(!hits[0]);
    }

    #[test]
    fn test_ray_aabb_diagonal() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let aabbs = vec![Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])];
        let (hits, ts) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert!(hits[0]);
        assert!(approx_eq(ts[0], 2.0));
    }

    #[test]
    fn test_ray_aabb_batch_mixed() {
        let ray = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let aabbs = vec![
            Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),   // hit at t=1
            Aabb::new([0.0, 5.0, 0.0], [1.0, 6.0, 1.0]),   // miss
            Aabb::new([5.0, 0.0, 0.0], [6.0, 1.0, 1.0]),   // hit at t=5
            Aabb::new([-3.0, 0.0, 0.0], [-2.0, 1.0, 1.0]), // behind → miss
        ];
        let (hits, ts) = ray_aabb_slab_test_batch(&ray, &aabbs);
        assert_eq!(hits, vec![true, false, true, false]);
        assert!(approx_eq(ts[0], 1.0));
        assert!(approx_eq(ts[2], 5.0));
    }

    #[test]
    fn test_ray_new() {
        let ray = Ray::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!(approx_eq(ray.inv_dir[0], 0.5));
        assert!(ray.inv_dir[1].is_infinite());
    }

    // ---------- AVX-512 ray-AABB parity ----------

    #[test]
    fn test_ray_aabb_avx512_parity() {
        // 100 AABBs to exercise AVX-512 + tail
        let ray = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let aabbs: Vec<Aabb> = (0..100)
            .map(|i| {
                let f = i as f32;
                Aabb::new([f, 0.0, 0.0], [f + 1.0, 1.0, 1.0])
            })
            .collect();
        let (hits_batch, ts_batch) = ray_aabb_slab_test_batch(&ray, &aabbs);
        let (hits_scalar, ts_scalar) = ray_aabb_slab_test_scalar(&ray, &aabbs);
        assert_eq!(hits_batch, hits_scalar, "ray AVX-512 hit parity");
        for i in 0..100 {
            assert!(
                approx_eq(ts_batch[i], ts_scalar[i]),
                "ray AVX-512 t parity at {i}: {} vs {}",
                ts_batch[i], ts_scalar[i]
            );
        }
    }
}
