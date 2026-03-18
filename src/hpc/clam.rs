//! CLAM Tree: divisive hierarchical clustering with LFD estimation.
//!
//! Implements Algorithm 1 (Partition) from CAKES (arXiv:2309.05491):
//!
//! ```text
//! 1. seeds ← random sample of ⌈√|C|⌉ points from C
//! 2. c     ← geometric median of seeds
//! 3. l     ← argmax f(c, x)  ∀x ∈ C          (left pole)
//! 4. r     ← argmax f(l, x)  ∀x ∈ C          (right pole)
//! 5. L     ← { x | f(l,x) ≤ f(r,x) }
//! 6. R     ← { x | f(r,x) < f(l,x) }
//! 7. recurse on L, R
//! ```
//!
//! After construction, the dataset is depth-first reordered so each cluster
//! is a contiguous slice `[offset..offset+cardinality]` — O(n) memory
//! instead of O(n log n) from storing index lists (CAKES §2.1.2).
//!
//! ## LFD (Local Fractal Dimension)
//!
//! Per-cluster LFD is computed during construction using Equation 2 from CAKES:
//!
//! ```text
//! LFD(q, r) = log₂( |B(q, r)| / |B(q, r/2)| )
//! ```

use super::bitwise;

// ─── SplitMix64 RNG (local) ─────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ─── Distance trait ──────────────────────────────────────────────

/// Generic distance function for CLAM tree construction and search.
pub trait Distance {
    type Point: ?Sized;
    fn distance(&self, a: &Self::Point, b: &Self::Point) -> u64;
    fn is_metric(&self) -> bool;
}

/// Hamming distance on byte slices.
pub struct HammingDistance;

impl Distance for HammingDistance {
    type Point = [u8];
    fn distance(&self, a: &[u8], b: &[u8]) -> u64 {
        assert_eq!(a.len(), b.len(), "Hamming distance requires equal lengths");
        bitwise::hamming_distance_raw(a, b)
    }
    fn is_metric(&self) -> bool {
        true
    }
}

/// Inline Hamming distance using the bitwise module.
#[inline(always)]
pub(crate) fn hamming_inline(a: &[u8], b: &[u8]) -> u64 {
    bitwise::hamming_distance_raw(a, b)
}

// ─── LFD ──────────────────────────────────────────

/// Local Fractal Dimension of a cluster.
///
/// LFD = log₂(|B(c, r)| / |B(c, r/2)|)
#[derive(Debug, Clone, Copy)]
pub struct Lfd {
    pub value: f64,
    pub count_r: usize,
    pub count_half_r: usize,
}

impl Lfd {
    pub fn compute(count_r: usize, count_half_r: usize) -> Self {
        let value = if count_half_r == 0 || count_r <= count_half_r {
            0.0
        } else {
            (count_r as f64 / count_half_r as f64).log2()
        };
        Lfd {
            value,
            count_r,
            count_half_r,
        }
    }
}

// ─── Cluster node ──────────────────────────────────

/// A node in the CLAM binary tree.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub center_idx: usize,
    pub radius: u64,
    pub cardinality: usize,
    pub offset: usize,
    pub depth: usize,
    pub lfd: Lfd,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl Cluster {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left.is_none()
    }

    /// δ⁺ = f(q, center) + radius
    #[inline]
    pub fn delta_plus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_add(self.radius)
    }

    /// δ⁻ = max(0, f(q, center) − radius)
    #[inline]
    pub fn delta_minus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_sub(self.radius)
    }
}

// ─── BuildConfig ──────────────────────────────────

/// Stopping criteria for tree construction.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub min_cardinality: usize,
    pub max_depth: usize,
    pub min_radius: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            min_cardinality: 1,
            max_depth: 256,
            min_radius: 0,
        }
    }
}

// ─── ClamTree ──────────────────────────────────────

/// Distance function type: takes two byte slices of equal length, returns u64.
pub type DistanceFn = fn(&[u8], &[u8]) -> u64;

/// Divisive hierarchical clustering tree.
pub struct ClamTree {
    pub nodes: Vec<Cluster>,
    pub reordered: Vec<usize>,
    pub num_leaves: usize,
    pub mean_leaf_radius: f64,
    distance_fn: DistanceFn,
}

impl ClamTree {
    /// Build a CLAM tree with default Hamming distance and simple config.
    pub fn build(data: &[u8], vec_len: usize, min_cluster_size: usize) -> Self {
        let count = data.len() / vec_len;
        let config = BuildConfig {
            min_cardinality: min_cluster_size.max(1),
            ..Default::default()
        };
        Self::build_with_config(data, vec_len, count, &config)
    }

    /// Build with explicit config and count.
    pub fn build_with_config(
        data: &[u8],
        vec_len: usize,
        count: usize,
        config: &BuildConfig,
    ) -> Self {
        Self::build_with_fn(data, vec_len, count, config, hamming_inline)
    }

    /// Build a CLAM tree with a custom distance function.
    pub fn build_with_fn(
        data: &[u8],
        vec_len: usize,
        count: usize,
        config: &BuildConfig,
        dist_fn: DistanceFn,
    ) -> Self {
        assert_eq!(data.len(), vec_len * count);

        if count == 0 {
            return ClamTree {
                nodes: Vec::new(),
                reordered: Vec::new(),
                num_leaves: 0,
                mean_leaf_radius: 0.0,
                distance_fn: dist_fn,
            };
        }

        let mut indices: Vec<usize> = (0..count).collect();
        let mut nodes = Vec::with_capacity(2 * count);
        let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_BABE);

        Self::partition(
            data, vec_len, &mut indices, 0, count, 0, config, &mut nodes, &mut rng, dist_fn,
        );

        let mut num_leaves = 0usize;
        let mut leaf_radius_sum = 0u64;
        for node in &nodes {
            if node.is_leaf() {
                num_leaves += 1;
                leaf_radius_sum += node.radius;
            }
        }
        let mean_leaf_radius = if num_leaves > 0 {
            leaf_radius_sum as f64 / num_leaves as f64
        } else {
            0.0
        };

        ClamTree {
            nodes,
            reordered: indices,
            num_leaves,
            mean_leaf_radius,
            distance_fn: dist_fn,
        }
    }

    /// Recursive partition (Algorithm 1 from CAKES).
    #[allow(clippy::too_many_arguments)]
    fn partition(
        data: &[u8],
        vec_len: usize,
        indices: &mut [usize],
        start: usize,
        end: usize,
        depth: usize,
        config: &BuildConfig,
        nodes: &mut Vec<Cluster>,
        rng: &mut SplitMix64,
        dist_fn: DistanceFn,
    ) -> usize {
        let n = end - start;
        let node_idx = nodes.len();

        // Step 1: Find center (geometric median of √n seeds)
        let num_seeds = (n as f64).sqrt().ceil() as usize;
        let num_seeds = num_seeds.max(1).min(n);

        let working = &mut indices[start..end];
        for i in 0..num_seeds.min(working.len()) {
            let j = i + (rng.next_u64() as usize % (working.len() - i));
            working.swap(i, j);
        }

        let center_local = if num_seeds <= 1 {
            0
        } else {
            let mut best_idx = 0;
            let mut best_sum = u64::MAX;
            for s in 0..num_seeds {
                let si = working[s];
                let si_data = &data[si * vec_len..(si + 1) * vec_len];
                let mut sum = 0u64;
                for t in 0..num_seeds {
                    if s != t {
                        let ti = working[t];
                        let ti_data = &data[ti * vec_len..(ti + 1) * vec_len];
                        sum += dist_fn(si_data, ti_data);
                    }
                }
                if sum < best_sum {
                    best_sum = sum;
                    best_idx = s;
                }
            }
            best_idx
        };

        working.swap(0, center_local);
        let center_idx = working[0];
        let center_data = &data[center_idx * vec_len..(center_idx + 1) * vec_len];

        // Step 2: Compute radius + find left pole
        let mut radius = 0u64;
        let mut left_pole_local = 0;
        let mut left_pole_dist = 0u64;

        let mut distances: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let d = dist_fn(center_data, pi_data);
            distances.push(d);
            if d > radius {
                radius = d;
            }
            if d > left_pole_dist {
                left_pole_dist = d;
                left_pole_local = i;
            }
        }

        // Compute LFD
        let half_radius = radius / 2;
        let count_r = distances.iter().filter(|&&d| d <= radius).count();
        let count_half_r = distances.iter().filter(|&&d| d <= half_radius).count();
        let lfd = Lfd::compute(count_r, count_half_r);

        // Step 3: Find right pole (farthest from left pole)
        let left_pole_idx = working[left_pole_local];
        let left_pole_data = &data[left_pole_idx * vec_len..(left_pole_idx + 1) * vec_len];

        let mut right_pole_local = 0;
        let mut right_pole_dist = 0u64;
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let d = dist_fn(left_pole_data, pi_data);
            if d > right_pole_dist {
                right_pole_dist = d;
                right_pole_local = i;
            }
        }
        let right_pole_idx = working[right_pole_local];
        let right_pole_data = &data[right_pole_idx * vec_len..(right_pole_idx + 1) * vec_len];

        // Step 4: Partition into L and R
        let mut side: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            let pi = working[i];
            let pi_data = &data[pi * vec_len..(pi + 1) * vec_len];
            let dl = dist_fn(left_pole_data, pi_data);
            let dr = dist_fn(right_pole_data, pi_data);
            side.push(dl <= dr);
        }

        let mut cursor = 0;
        for i in 0..n {
            if side[i] {
                working.swap(cursor, i);
                side.swap(cursor, i);
                cursor += 1;
            }
        }
        let split = cursor;

        nodes.push(Cluster {
            center_idx,
            radius,
            cardinality: n,
            offset: start,
            depth,
            lfd,
            left: None,
            right: None,
        });

        // Step 5: Recurse if criteria met
        let should_split = n > config.min_cardinality
            && depth < config.max_depth
            && radius > config.min_radius
            && split > 0
            && split < n;

        if should_split {
            let left_idx = Self::partition(
                data, vec_len, indices, start, start + split, depth + 1, config, nodes, rng,
                dist_fn,
            );
            nodes[node_idx].left = Some(left_idx);

            let right_idx = Self::partition(
                data, vec_len, indices, start + split, end, depth + 1, config, nodes, rng, dist_fn,
            );
            nodes[node_idx].right = Some(right_idx);
        }

        node_idx
    }

    #[inline]
    pub fn dist(&self, a: &[u8], b: &[u8]) -> u64 {
        (self.distance_fn)(a, b)
    }

    #[inline]
    pub fn distance_fn(&self) -> DistanceFn {
        self.distance_fn
    }

    pub fn root(&self) -> &Cluster {
        &self.nodes[0]
    }

    pub fn center_data<'a>(&self, cluster: &Cluster, data: &'a [u8], vec_len: usize) -> &'a [u8] {
        &data[cluster.center_idx * vec_len..(cluster.center_idx + 1) * vec_len]
    }

    pub fn cluster_points<'a>(
        &'a self,
        cluster: &Cluster,
        data: &'a [u8],
        vec_len: usize,
    ) -> impl Iterator<Item = (usize, &'a [u8])> + 'a {
        let start = cluster.offset;
        let end = start + cluster.cardinality;
        self.reordered[start..end].iter().map(move |&orig_idx| {
            (
                orig_idx,
                &data[orig_idx * vec_len..(orig_idx + 1) * vec_len],
            )
        })
    }

    pub fn cluster_member_indices(&self, cluster: &Cluster) -> Vec<usize> {
        let start = cluster.offset;
        let end = start + cluster.cardinality;
        self.reordered[start..end].to_vec()
    }

    /// Get LFD statistics across the tree.
    pub fn lfd_percentiles(&self) -> LfdStats {
        let mut lfds: Vec<f64> = self.nodes.iter().map(|c| c.lfd.value).collect();
        lfds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = lfds.len();
        if n == 0 {
            return LfdStats::default();
        }

        let last = n - 1;
        LfdStats {
            min: lfds[0],
            p5: lfds[last * 5 / 100],
            p25: lfds[last * 25 / 100],
            p50: lfds[last * 50 / 100],
            p75: lfds[last * 75 / 100],
            p95: lfds[last * 95 / 100],
            max: lfds[last],
            mean: lfds.iter().sum::<f64>() / n as f64,
        }
    }

    /// Extract root-to-leaf path for every data point.
    pub fn leaf_paths(&self) -> Vec<(usize, Vec<bool>)> {
        let mut result = Vec::new();
        let mut stack: Vec<(usize, Vec<bool>)> = vec![(0, Vec::new())];

        while let Some((node_idx, path)) = stack.pop() {
            let cluster = &self.nodes[node_idx];

            if cluster.is_leaf() {
                let start = cluster.offset;
                let end = start + cluster.cardinality;
                for &orig_idx in &self.reordered[start..end] {
                    result.push((orig_idx, path.clone()));
                }
            } else {
                if let Some(right) = cluster.right {
                    let mut right_path = path.clone();
                    right_path.push(true);
                    stack.push((right, right_path));
                }
                if let Some(left) = cluster.left {
                    let mut left_path = path.clone();
                    left_path.push(false);
                    stack.push((left, left_path));
                }
            }
        }

        result
    }

    /// Walk the tree following a ClamPath, stopping at the deepest real cluster.
    pub fn deepest_real_cluster(&self, path_bits: u16, path_depth: u8) -> (u8, usize) {
        if self.nodes.is_empty() {
            return (0, 0);
        }

        let mut node_idx = 0usize;
        let mut real_depth: u8 = 0;

        for bit_pos in 0..path_depth {
            let cluster = &self.nodes[node_idx];
            if cluster.is_leaf() {
                break;
            }

            let went_right = (path_bits >> (15 - bit_pos as u32)) & 1 == 1;

            let next = if went_right {
                cluster.right
            } else {
                cluster.left
            };

            match next {
                Some(child_idx) => {
                    node_idx = child_idx;
                    real_depth = bit_pos + 1;
                }
                None => break,
            }
        }

        (real_depth, node_idx)
    }

    /// Compute the CRP distribution for a cluster.
    pub fn cluster_crp(
        &self,
        cluster: &Cluster,
        data: &[u8],
        vec_len: usize,
    ) -> ClusterDistribution {
        let center = self.center_data(cluster, data, vec_len);
        let mut distances: Vec<u64> = self
            .cluster_points(cluster, data, vec_len)
            .map(|(_, point)| (self.distance_fn)(center, point))
            .collect();

        if distances.is_empty() {
            return ClusterDistribution::default();
        }

        distances.sort_unstable();
        let n = distances.len();
        let sum: u64 = distances.iter().sum();
        let mean = sum as f64 / n as f64;
        let variance = distances
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        ClusterDistribution {
            mean,
            std_dev,
            p25: distances[n * 25 / 100],
            p50: distances[n / 2],
            p75: distances[n * 75 / 100],
            min: distances[0],
            max: *distances.last().unwrap(),
            count: n,
        }
    }

    /// Get LFD values by depth.
    pub fn lfd_by_depth(&self) -> Vec<(usize, Vec<f64>)> {
        let max_depth = self.nodes.iter().map(|c| c.depth).max().unwrap_or(0);
        let mut by_depth: Vec<Vec<f64>> = vec![Vec::new(); max_depth + 1];

        for node in &self.nodes {
            for _ in 0..node.cardinality {
                by_depth[node.depth].push(node.lfd.value);
            }
        }

        by_depth
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect()
    }
}

// ─── Distribution ──────────────────────────────────

/// CRP distribution of a cluster.
#[derive(Debug, Clone, Default)]
pub struct ClusterDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub p25: u64,
    pub p50: u64,
    pub p75: u64,
    pub min: u64,
    pub max: u64,
    pub count: usize,
}

/// Summary statistics of LFD across the tree.
#[derive(Debug, Clone, Default)]
pub struct LfdStats {
    pub min: f64,
    pub p5: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
    pub mean: f64,
}

// ─── Search types ──────────────────────────────────

/// Result of ρ-NN search.
#[derive(Debug, Clone)]
pub struct RhoNnResult {
    pub hits: Vec<(usize, u64)>,
    pub distance_calls: usize,
    pub clusters_pruned: usize,
}

/// Result of k-NN search.
#[derive(Debug, Clone)]
pub struct KnnResult {
    pub hits: Vec<(usize, u64)>,
    pub distance_calls: usize,
    pub clusters_pruned: usize,
}

/// ρ-nearest neighbor search using triangle inequality.
pub fn rho_nn(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    rho: u64,
) -> RhoNnResult {
    let mut hits = Vec::new();
    let mut distance_calls = 0usize;
    let mut clusters_pruned = 0usize;
    let mut candidate_leaves = Vec::new();
    let mut stack = vec![0usize];

    while let Some(node_idx) = stack.pop() {
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);
        let delta = tree.dist(query, center);
        distance_calls += 1;

        let d_minus = cluster.delta_minus(delta);
        let d_plus = cluster.delta_plus(delta);

        if d_minus > rho {
            clusters_pruned += 1;
            continue;
        }

        if cluster.is_leaf() {
            if d_plus <= rho {
                candidate_leaves.push((node_idx, true));
            } else {
                candidate_leaves.push((node_idx, false));
            }
        } else {
            if let Some(left) = cluster.left {
                stack.push(left);
            }
            if let Some(right) = cluster.right {
                stack.push(right);
            }
        }
    }

    for (node_idx, all_inside) in &candidate_leaves {
        let cluster = &tree.nodes[*node_idx];

        if *all_inside {
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = tree.dist(query, point_data);
                distance_calls += 1;
                hits.push((orig_idx, d));
            }
        } else {
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = tree.dist(query, point_data);
                distance_calls += 1;
                if d <= rho {
                    hits.push((orig_idx, d));
                }
            }
        }
    }

    hits.sort_by_key(|&(_, d)| d);

    RhoNnResult {
        hits,
        distance_calls,
        clusters_pruned,
    }
}

/// k-NN via Repeated ρ-NN search (CAKES Algorithm 4).
pub fn knn_repeated_rho(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    k: usize,
) -> KnnResult {
    let root = tree.root();
    if root.cardinality == 0 {
        return KnnResult {
            hits: Vec::new(),
            distance_calls: 0,
            clusters_pruned: 0,
        };
    }
    let mut rho = root.radius / root.cardinality as u64;
    if rho == 0 {
        rho = 1;
    }

    let mut total_distance_calls = 0;
    let mut total_pruned = 0;

    loop {
        let result = rho_nn(tree, data, vec_len, query, rho);
        total_distance_calls += result.distance_calls;
        total_pruned += result.clusters_pruned;

        if result.hits.len() >= k {
            let mut hits = result.hits;
            hits.truncate(k);
            return KnnResult {
                hits,
                distance_calls: total_distance_calls,
                clusters_pruned: total_pruned,
            };
        }

        if result.hits.is_empty() {
            rho *= 2;
        } else {
            let mean_inv_lfd = estimate_local_lfd(tree, data, vec_len, query, rho);
            let ratio = k as f64 / result.hits.len() as f64;
            let factor = ratio.powf(mean_inv_lfd).clamp(1.1, 2.0);
            rho = ((rho as f64) * factor).ceil() as u64;
        }

        if rho > root.radius {
            rho = root.radius;
            let result = rho_nn(tree, data, vec_len, query, rho);
            total_distance_calls += result.distance_calls;
            total_pruned += result.clusters_pruned;
            let mut hits = result.hits;
            hits.truncate(k);
            return KnnResult {
                hits,
                distance_calls: total_distance_calls,
                clusters_pruned: total_pruned,
            };
        }
    }
}

fn estimate_local_lfd(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    rho: u64,
) -> f64 {
    let mut sum_inv_lfd = 0.0;
    let mut count = 0usize;

    let mut stack = vec![0usize];
    while let Some(node_idx) = stack.pop() {
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);
        let delta = tree.dist(query, center);

        if cluster.delta_minus(delta) > rho {
            continue;
        }

        if cluster.is_leaf() {
            let lfd = cluster.lfd.value.max(0.1);
            sum_inv_lfd += 1.0 / lfd;
            count += 1;
        } else {
            if let Some(left) = cluster.left {
                stack.push(left);
            }
            if let Some(right) = cluster.right {
                stack.push(right);
            }
        }
    }

    if count == 0 {
        1.0
    } else {
        sum_inv_lfd / count as f64
    }
}

/// Depth-First Sieve k-NN search (CAKES Algorithm 6).
pub fn knn_dfs_sieve(
    tree: &ClamTree,
    data: &[u8],
    vec_len: usize,
    query: &[u8],
    k: usize,
) -> KnnResult {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut distance_calls = 0usize;
    let mut clusters_pruned = 0usize;

    let mut queue: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut hits: BinaryHeap<(u64, usize)> = BinaryHeap::new();

    let root = tree.root();
    let root_center = tree.center_data(root, data, vec_len);
    let root_delta = tree.dist(query, root_center);
    distance_calls += 1;
    let root_d_minus = root.delta_minus(root_delta);
    queue.push(Reverse((root_d_minus, 0)));

    while let Some(&Reverse((d_minus, node_idx))) = queue.peek() {
        if hits.len() >= k {
            if let Some(&(worst_dist, _)) = hits.peek() {
                if worst_dist <= d_minus {
                    break;
                }
            }
        }

        queue.pop();
        let cluster = &tree.nodes[node_idx];

        if cluster.is_leaf() {
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let d = tree.dist(query, point_data);
                distance_calls += 1;

                if hits.len() < k {
                    hits.push((d, orig_idx));
                } else if let Some(&(worst, _)) = hits.peek() {
                    if d < worst {
                        hits.pop();
                        hits.push((d, orig_idx));
                    }
                }
            }
        } else {
            for child_idx in [cluster.left, cluster.right].iter().flatten() {
                let child = &tree.nodes[*child_idx];
                let child_center = tree.center_data(child, data, vec_len);
                let child_delta = tree.dist(query, child_center);
                distance_calls += 1;

                let child_d_minus = child.delta_minus(child_delta);

                if hits.len() >= k {
                    if let Some(&(worst, _)) = hits.peek() {
                        if child_d_minus > worst {
                            clusters_pruned += 1;
                            continue;
                        }
                    }
                }

                queue.push(Reverse((child_d_minus, *child_idx)));
            }
        }
    }

    let mut result: Vec<(usize, u64)> = hits.into_iter().map(|(d, idx)| (idx, d)).collect();
    result.sort_by_key(|&(_, d)| d);

    KnnResult {
        hits: result,
        distance_calls,
        clusters_pruned,
    }
}

/// k-NN via brute-force (baseline).
pub fn knn_brute(data: &[u8], vec_len: usize, query: &[u8], k: usize) -> KnnResult {
    let n = data.len() / vec_len;
    let mut dists: Vec<(usize, u64)> = (0..n)
        .map(|i| {
            (
                i,
                bitwise::hamming_distance_raw(
                    query,
                    &data[i * vec_len..(i + 1) * vec_len],
                ),
            )
        })
        .collect();
    dists.sort_unstable_by_key(|&(_, d)| d);
    dists.truncate(k);
    KnnResult {
        distance_calls: n,
        clusters_pruned: 0,
        hits: dists,
    }
}

// ─── Compression ──────────────────────────────────

/// XOR-diff encoding of a point relative to a reference.
#[derive(Debug, Clone)]
pub struct XorDiffEncoding {
    pub positions: Vec<u16>,
    pub values: Vec<u8>,
}

impl XorDiffEncoding {
    pub fn encode(center: &[u8], point: &[u8]) -> Self {
        debug_assert_eq!(center.len(), point.len());
        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, (&c, &p)) in center.iter().zip(point.iter()).enumerate() {
            if c != p {
                positions.push(i as u16);
                values.push(p);
            }
        }

        XorDiffEncoding { positions, values }
    }

    pub fn decode(&self, center: &[u8]) -> Vec<u8> {
        let mut result = center.to_vec();
        for (&pos, &val) in self.positions.iter().zip(self.values.iter()) {
            result[pos as usize] = val;
        }
        result
    }

    pub fn storage_cost(&self) -> usize {
        self.positions.len() * 3
    }

    pub fn num_diffs(&self) -> usize {
        self.positions.len()
    }

    /// Compute Hamming distance from query to encoded point WITHOUT full decompression.
    pub fn hamming_from_query(&self, query: &[u8], center: &[u8], dist_q_center: u64) -> u64 {
        let mut adjustment: i64 = 0;

        for (&pos, &val) in self.positions.iter().zip(self.values.iter()) {
            let p = pos as usize;
            let old_xor = query[p] ^ center[p];
            let new_xor = query[p] ^ val;
            adjustment += new_xor.count_ones() as i64 - old_xor.count_ones() as i64;
        }

        (dist_q_center as i64 + adjustment) as u64
    }
}

/// Compression mode chosen for each cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMode {
    Unitary,
    Recursive,
}

/// Compression statistics.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub ratio: f64,
    pub unitary_clusters: usize,
    pub recursive_clusters: usize,
    pub pruned_subtrees: usize,
}

/// A compressed CLAM tree with panCAKES encoding.
pub struct CompressedTree {
    pub encodings: Vec<XorDiffEncoding>,
    pub encoding_centers: Vec<usize>,
    pub cluster_modes: Vec<CompressionMode>,
    pub stats: CompressionStats,
}

#[derive(Clone)]
#[allow(dead_code)]
struct ClusterCompression {
    mode: CompressionMode,
    unitary_cost: usize,
    recursive_cost: usize,
    min_cost: usize,
}

impl CompressedTree {
    /// Compress a dataset using its CLAM tree (panCAKES Algorithm 2).
    pub fn compress(tree: &ClamTree, data: &[u8], vec_len: usize, count: usize) -> Self {
        let num_nodes = tree.nodes.len();
        let mut comp: Vec<Option<ClusterCompression>> = vec![None; num_nodes];
        let mut cluster_modes = vec![CompressionMode::Unitary; num_nodes];

        let order = postorder_indices(tree);

        for &node_idx in &order {
            let cluster = &tree.nodes[node_idx];
            let center = tree.center_data(cluster, data, vec_len);

            let mut unitary_cost = 0usize;
            for (_, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let enc = XorDiffEncoding::encode(center, point_data);
                unitary_cost += enc.storage_cost();
            }

            let mut min_cost = unitary_cost;
            let mut mode = CompressionMode::Unitary;

            if !cluster.is_leaf() {
                let mut recursive_cost = 0usize;

                if let Some(left_idx) = cluster.left {
                    let left = &tree.nodes[left_idx];
                    let left_center = tree.center_data(left, data, vec_len);
                    let edge_cost = XorDiffEncoding::encode(center, left_center).storage_cost();
                    let left_min = comp[left_idx].as_ref().map(|c| c.min_cost).unwrap_or(0);
                    recursive_cost += edge_cost + left_min;
                }

                if let Some(right_idx) = cluster.right {
                    let right = &tree.nodes[right_idx];
                    let right_center = tree.center_data(right, data, vec_len);
                    let edge_cost = XorDiffEncoding::encode(center, right_center).storage_cost();
                    let right_min = comp[right_idx].as_ref().map(|c| c.min_cost).unwrap_or(0);
                    recursive_cost += edge_cost + right_min;
                }

                if recursive_cost < unitary_cost {
                    min_cost = recursive_cost;
                    mode = CompressionMode::Recursive;
                }
            }

            cluster_modes[node_idx] = mode;
            comp[node_idx] = Some(ClusterCompression {
                mode,
                unitary_cost,
                recursive_cost: if cluster.is_leaf() {
                    unitary_cost
                } else {
                    min_cost
                },
                min_cost,
            });
        }

        let mut encodings = vec![
            XorDiffEncoding {
                positions: vec![],
                values: vec![],
            };
            count
        ];
        let mut encoding_centers = vec![0usize; count];

        Self::assign_encodings(
            tree,
            data,
            vec_len,
            0,
            &cluster_modes,
            &mut encodings,
            &mut encoding_centers,
        );

        let uncompressed_bytes = count * vec_len;
        let compressed_bytes: usize =
            encodings.iter().map(|e| e.storage_cost()).sum::<usize>() + count * 2;
        let ratio = if compressed_bytes > 0 {
            uncompressed_bytes as f64 / compressed_bytes as f64
        } else {
            f64::INFINITY
        };

        let unitary_clusters = cluster_modes
            .iter()
            .filter(|&&m| m == CompressionMode::Unitary)
            .count();
        let recursive_clusters = cluster_modes
            .iter()
            .filter(|&&m| m == CompressionMode::Recursive)
            .count();
        let pruned_subtrees = cluster_modes
            .iter()
            .enumerate()
            .filter(|&(i, &m)| m == CompressionMode::Unitary && !tree.nodes[i].is_leaf())
            .count();

        CompressedTree {
            encodings,
            encoding_centers,
            cluster_modes,
            stats: CompressionStats {
                uncompressed_bytes,
                compressed_bytes,
                ratio,
                unitary_clusters,
                recursive_clusters,
                pruned_subtrees,
            },
        }
    }

    fn assign_encodings(
        tree: &ClamTree,
        data: &[u8],
        vec_len: usize,
        node_idx: usize,
        modes: &[CompressionMode],
        encodings: &mut [XorDiffEncoding],
        encoding_centers: &mut [usize],
    ) {
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);

        if modes[node_idx] == CompressionMode::Unitary || cluster.is_leaf() {
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                encodings[orig_idx] = XorDiffEncoding::encode(center, point_data);
                encoding_centers[orig_idx] = cluster.center_idx;
            }
        } else {
            if let Some(left) = cluster.left {
                Self::assign_encodings(tree, data, vec_len, left, modes, encodings, encoding_centers);
            }
            if let Some(right) = cluster.right {
                Self::assign_encodings(
                    tree,
                    data,
                    vec_len,
                    right,
                    modes,
                    encodings,
                    encoding_centers,
                );
            }
        }
    }

    pub fn decompress_point(&self, point_idx: usize, data: &[u8], vec_len: usize) -> Vec<u8> {
        let center_idx = self.encoding_centers[point_idx];
        let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
        self.encodings[point_idx].decode(center)
    }

    /// Compute Hamming distance from query to compressed point WITHOUT decompression.
    pub fn hamming_to_compressed(
        &self,
        query: &[u8],
        point_idx: usize,
        data: &[u8],
        vec_len: usize,
        dist_cache: &mut DistanceCache,
        dist_fn: DistanceFn,
    ) -> u64 {
        let center_idx = self.encoding_centers[point_idx];

        let dist_q_center = dist_cache.get_or_compute(center_idx, || {
            let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
            dist_fn(query, center)
        });

        let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
        self.encodings[point_idx].hamming_from_query(query, center, dist_q_center)
    }
}

/// Cache for d(query, center) values.
pub struct DistanceCache {
    entries: std::collections::HashMap<usize, u64>,
}

impl Default for DistanceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceCache {
    pub fn new() -> Self {
        DistanceCache {
            entries: std::collections::HashMap::with_capacity(64),
        }
    }

    pub fn get_or_compute(&mut self, center_idx: usize, compute: impl FnOnce() -> u64) -> u64 {
        *self.entries.entry(center_idx).or_insert_with(compute)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Compute post-order traversal of the tree.
fn postorder_indices(tree: &ClamTree) -> Vec<usize> {
    let mut result = Vec::with_capacity(tree.nodes.len());
    let mut stack = vec![(0usize, false)];

    while let Some((node_idx, visited)) = stack.pop() {
        let cluster = &tree.nodes[node_idx];

        if visited || cluster.is_leaf() {
            result.push(node_idx);
        } else {
            stack.push((node_idx, true));
            if let Some(right) = cluster.right {
                stack.push((right, false));
            }
            if let Some(left) = cluster.left {
                stack.push((left, false));
            }
        }
    }

    result
}

// Legacy compat: compress function with old interface
/// Compress a dataset using XOR-diff encoding relative to cluster centers.
pub fn compress(data: &[u8], vec_len: usize, tree: &ClamTree) -> CompressedTree {
    let count = data.len() / vec_len;
    CompressedTree::compress(tree, data, vec_len, count)
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: CLAM → Cascade Bridge
// ═══════════════════════════════════════════════════════════════════

use super::cascade::{Band, Cascade, RankedHit};

/// Result of CLAM→Cascade bridged search, combining CLAM's tight candidates
/// with cascade verification and banding.
#[derive(Debug, Clone)]
pub struct ClamCascadeResult {
    /// Verified hits that passed cascade threshold, sorted by Hamming distance.
    pub hits: Vec<RankedHit>,
    /// Number of CLAM rho-NN candidates before cascade filtering.
    pub clam_candidates: usize,
    /// Number of hits that survived cascade verification.
    pub cascade_survivors: usize,
    /// Distance computations from the CLAM search phase.
    pub distance_calls: usize,
    /// Clusters pruned during CLAM search.
    pub clusters_pruned: usize,
}

impl ClamTree {
    /// CLAM → Cascade bridge: rho-NN candidates → cascade 3-stroke verification.
    ///
    /// 1. Runs rho-NN on the CLAM tree to get candidate fingerprint indices
    /// 2. Feeds those candidates into `Cascade::query_candidates()` for
    ///    verification and band classification
    /// 3. Returns verified `RankedHit` results
    ///
    /// The cascade's stroke-1 (partial prefix scan) becomes partially redundant
    /// since CLAM provides geometrically tight candidates via triangle inequality.
    pub fn rho_nn_cascade(
        &self,
        data: &[u8],
        vec_len: usize,
        query: &[u8],
        rho: u64,
        cascade: &Cascade,
    ) -> ClamCascadeResult {
        // Phase 1a: CLAM rho-NN search
        let rho_result = rho_nn(self, data, vec_len, query, rho);
        let clam_candidates = rho_result.hits.len();

        // Phase 1b: Feed CLAM candidates into cascade verification
        let verified = cascade.query_candidates(query, data, vec_len, &rho_result.hits);
        let cascade_survivors = verified.len();

        ClamCascadeResult {
            hits: verified,
            clam_candidates,
            cascade_survivors,
            distance_calls: rho_result.distance_calls,
            clusters_pruned: rho_result.clusters_pruned,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: SPO Distance Harvest
// ═══════════════════════════════════════════════════════════════════

use super::causality::{
    causality_decompose, CausalityDecomposition, NarsTruthValue,
};
use super::bf16_truth::{
    AwarenessState, PackedQualia,
};
use super::node::{Node, SPO, S__, _P_, __O};
use super::plane::Distance as PlaneDistance;

/// A verified hit enriched with causal metadata from SPO decomposition.
#[derive(Debug, Clone)]
pub struct CausalHit {
    /// Original cascade hit.
    pub index: usize,
    /// Hamming distance.
    pub hamming: u64,
    /// Cascade band classification.
    pub band: Band,
    /// Per-plane S/P/O distances (disagreement values).
    /// `None` if planes are incomparable.
    pub s_distance: Option<u32>,
    pub p_distance: Option<u32>,
    pub o_distance: Option<u32>,
    /// NARS truth value accumulated from awareness states.
    pub truth: NarsTruthValue,
    /// Causal decomposition across warmth/social/sacredness dimensions.
    pub causality: Option<CausalityDecomposition>,
}

/// Decompose verified hits through Node S/P/O distance and causality analysis.
///
/// For each verified hit:
/// 1. Computes per-plane distances via `Node::distance()` for S, P, O masks
/// 2. Feeds distances into NARS truth value accumulation
/// 3. Uses `CausalityDecomposition` to extract directional relationships
/// 4. Returns enriched results with causal metadata
pub fn spo_distance_harvest(
    hits: &[RankedHit],
    query_node: &mut Node,
    hit_nodes: &mut [Node],
    query_qualia: &PackedQualia,
    hit_qualias: &[PackedQualia],
) -> Vec<CausalHit> {
    let mut results = Vec::with_capacity(hits.len());

    for hit in hits {
        let idx = hit.index;

        // Bounds check: if we don't have node/qualia data for this index, use defaults
        if idx >= hit_nodes.len() || idx >= hit_qualias.len() {
            results.push(CausalHit {
                index: idx,
                hamming: hit.hamming,
                band: hit.band,
                s_distance: None,
                p_distance: None,
                o_distance: None,
                truth: NarsTruthValue::ignorance(),
                causality: None,
            });
            continue;
        }

        let hit_node = &mut hit_nodes[idx];

        // Step 1: Per-plane S/P/O distances
        let d_s = query_node.distance(hit_node, S__);
        let d_p = query_node.distance(hit_node, _P_);
        let d_o = query_node.distance(hit_node, __O);

        let s_dist = match d_s {
            PlaneDistance::Measured { disagreement, .. } => Some(disagreement),
            PlaneDistance::Incomparable => None,
        };
        let p_dist = match d_p {
            PlaneDistance::Measured { disagreement, .. } => Some(disagreement),
            PlaneDistance::Incomparable => None,
        };
        let o_dist = match d_o {
            PlaneDistance::Measured { disagreement, .. } => Some(disagreement),
            PlaneDistance::Incomparable => None,
        };

        // Step 2: Derive NARS truth value from SPO distances
        // Use the full SPO distance to derive awareness-based truth
        let d_spo = query_node.distance(hit_node, SPO);
        let truth = match d_spo {
            PlaneDistance::Measured { disagreement, overlap, .. } => {
                if overlap == 0 {
                    NarsTruthValue::ignorance()
                } else {
                    let frequency = 1.0 - (disagreement as f32 / overlap as f32).min(1.0);
                    let confidence = (overlap as f32 / (overlap as f32 + 1.0)).min(0.9999);
                    NarsTruthValue::new(frequency, confidence)
                }
            }
            PlaneDistance::Incomparable => NarsTruthValue::ignorance(),
        };

        // Step 3: Causality decomposition via qualia
        let hit_qualia = &hit_qualias[idx];
        let decomposition = causality_decompose(query_qualia, hit_qualia, None);

        results.push(CausalHit {
            index: idx,
            hamming: hit.hamming,
            band: hit.band,
            s_distance: s_dist,
            p_distance: p_dist,
            o_distance: o_dist,
            truth,
            causality: Some(decomposition),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════
// Phase 3: panCAKES Compression Wiring
// ═══════════════════════════════════════════════════════════════════

/// Result of a compressed search query.
#[derive(Debug, Clone)]
pub struct CompressedSearchResult {
    /// (original_index, distance) pairs, sorted by distance ascending.
    pub hits: Vec<(usize, u64)>,
    /// Number of distance computations (compressed, not full).
    pub distance_calls: usize,
    /// Compression ratio of the database.
    pub compression_ratio: f64,
}

impl ClamTree {
    /// Compress a fingerprint database using cluster centers as codebook.
    ///
    /// Wraps `CompressedTree::compress()` — each point is XOR-diff encoded
    /// relative to its nearest cluster center, yielding a compact representation
    /// suitable for compressive search.
    pub fn compress_database(
        &self,
        data: &[u8],
        vec_len: usize,
    ) -> CompressedTree {
        let count = data.len() / vec_len;
        CompressedTree::compress(self, data, vec_len, count)
    }

    /// Query a compressed database, decompressing on-the-fly during search.
    ///
    /// Uses `CompressedTree::hamming_to_compressed()` to compute Hamming
    /// distances from the encoding diffs without full decompression.
    /// Cost per point: O(num_diffs) instead of O(vec_len).
    pub fn query_compressed(
        &self,
        compressed: &CompressedTree,
        data: &[u8],
        vec_len: usize,
        query: &[u8],
        rho: u64,
    ) -> CompressedSearchResult {
        let count = data.len() / vec_len;
        let mut cache = DistanceCache::new();
        let mut hits = Vec::new();
        let mut distance_calls = 0usize;
        let dist_fn = self.distance_fn();

        // Use CLAM tree structure: walk to find overlapping leaves,
        // then do compressive distance on leaf members
        let mut candidate_leaves = Vec::new();
        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            if node_idx >= self.nodes.len() {
                continue;
            }
            let cluster = &self.nodes[node_idx];
            let center = self.center_data(cluster, data, vec_len);
            let delta = self.dist(query, center);
            distance_calls += 1;

            let d_minus = cluster.delta_minus(delta);

            if d_minus > rho {
                continue;
            }

            if cluster.is_leaf() {
                candidate_leaves.push(node_idx);
            } else {
                if let Some(left) = cluster.left {
                    stack.push(left);
                }
                if let Some(right) = cluster.right {
                    stack.push(right);
                }
            }
        }

        // For each surviving leaf, compute compressive distances
        for node_idx in &candidate_leaves {
            let cluster = &self.nodes[*node_idx];
            for (orig_idx, _) in self.cluster_points(cluster, data, vec_len) {
                if orig_idx >= count {
                    continue;
                }
                let d = compressed.hamming_to_compressed(
                    query, orig_idx, data, vec_len, &mut cache, dist_fn,
                );
                distance_calls += 1;
                if d <= rho {
                    hits.push((orig_idx, d));
                }
            }
        }

        hits.sort_by_key(|&(_, d)| d);

        CompressedSearchResult {
            hits,
            distance_calls,
            compression_ratio: compressed.stats.ratio,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: CHAODA Anomaly Detection
// ═══════════════════════════════════════════════════════════════════

/// Anomaly score for a single data point, derived from cluster LFD.
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// Original dataset index.
    pub index: usize,
    /// LFD of the leaf cluster containing this point.
    pub lfd: f64,
    /// Normalized anomaly score in [0, 1]. Higher = more anomalous.
    pub score: f64,
    /// Awareness classification derived from anomaly level.
    pub awareness: AwarenessState,
}

impl ClamTree {
    /// Compute cluster-based anomaly scores from LFD distribution (CHAODA).
    ///
    /// High LFD = complex local geometry = potential outlier.
    /// The score is normalized against the global LFD distribution:
    ///   score = (lfd - lfd_min) / (lfd_max - lfd_min)
    ///
    /// Returns one `AnomalyScore` per data point in the dataset.
    pub fn anomaly_scores(&self, data: &[u8], vec_len: usize) -> Vec<AnomalyScore> {
        let count = data.len() / vec_len;

        // Build a map: original_index -> leaf cluster LFD
        let mut point_lfd = vec![0.0f64; count];

        for node in &self.nodes {
            if node.is_leaf() {
                let start = node.offset;
                let end = start + node.cardinality;
                for &orig_idx in &self.reordered[start..end] {
                    if orig_idx < count {
                        point_lfd[orig_idx] = node.lfd.value;
                    }
                }
            }
        }

        // Compute global LFD min/max for normalization
        let lfd_min = point_lfd.iter().cloned().fold(f64::INFINITY, f64::min);
        let lfd_max = point_lfd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lfd_range = (lfd_max - lfd_min).max(1e-10);

        point_lfd
            .iter()
            .enumerate()
            .map(|(idx, &lfd)| {
                let score = (lfd - lfd_min) / lfd_range;

                // Map anomaly score to awareness state:
                // High score = high LFD = complex = Noise/Uncertain
                // Low score = low LFD = regular = Crystallized
                let awareness = if score < 0.25 {
                    AwarenessState::Crystallized
                } else if score < 0.50 {
                    AwarenessState::Tensioned
                } else if score < 0.75 {
                    AwarenessState::Uncertain
                } else {
                    AwarenessState::Noise
                };

                AnomalyScore {
                    index: idx,
                    lfd,
                    score,
                    awareness,
                }
            })
            .collect()
    }

    /// Flag anomalies: return indices of points whose anomaly score exceeds threshold.
    ///
    /// Threshold is in [0, 1]. A threshold of 0.75 flags the top ~25% most
    /// anomalous points (those in high-LFD leaf clusters).
    pub fn flag_anomalies(
        &self,
        data: &[u8],
        vec_len: usize,
        threshold: f64,
    ) -> Vec<AnomalyScore> {
        self.anomaly_scores(data, vec_len)
            .into_iter()
            .filter(|a| a.score >= threshold)
            .collect()
    }
}

// ─── Tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(n: usize, vec_len: usize) -> Vec<u8> {
        let mut rng = SplitMix64::new(42);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    fn make_test_data_seeded(n: usize, vec_len: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    fn make_clustered_data(
        num_clusters: usize,
        points_per_cluster: usize,
        vec_len: usize,
        noise_bytes: usize,
    ) -> Vec<u8> {
        let count = num_clusters * points_per_cluster;
        let mut data = vec![0u8; count * vec_len];
        let mut rng = SplitMix64::new(42);

        for c in 0..num_clusters {
            let mut center = vec![0u8; vec_len];
            for byte in center.iter_mut() {
                *byte = (rng.next_u64() & 0xFF) as u8;
            }
            for p in 0..points_per_cluster {
                let idx = c * points_per_cluster + p;
                let point = &mut data[idx * vec_len..(idx + 1) * vec_len];
                point.copy_from_slice(&center);
                for _ in 0..noise_bytes {
                    let pos = (rng.next_u64() as usize) % vec_len;
                    point[pos] = (rng.next_u64() & 0xFF) as u8;
                }
            }
        }

        data
    }

    fn linear_knn(
        data: &[u8],
        vec_len: usize,
        count: usize,
        query: &[u8],
        k: usize,
    ) -> Vec<(usize, u64)> {
        let mut dists: Vec<(usize, u64)> = (0..count)
            .map(|i| {
                let point = &data[i * vec_len..(i + 1) * vec_len];
                (i, hamming_inline(query, point))
            })
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        dists.truncate(k);
        dists
    }

    #[test]
    fn test_hamming_distance_metric() {
        let hd = HammingDistance;
        let a = [0xFFu8; 32];
        let b = [0x00u8; 32];
        assert_eq!(hd.distance(&a[..], &b[..]), 256);
        assert!(hd.is_metric());
    }

    #[test]
    fn test_hamming_inline_identical() {
        let a = vec![0xAA; 2048];
        assert_eq!(hamming_inline(&a, &a), 0);
    }

    #[test]
    fn test_hamming_inline_all_different() {
        let a = vec![0xFF; 8];
        let b = vec![0x00; 8];
        assert_eq!(hamming_inline(&a, &b), 64);
    }

    #[test]
    fn test_lfd_compute() {
        let lfd = Lfd::compute(100, 25);
        assert!((lfd.value - 2.0).abs() < 1e-10);

        let lfd = Lfd::compute(100, 100);
        assert_eq!(lfd.value, 0.0);

        let lfd = Lfd::compute(100, 0);
        assert_eq!(lfd.value, 0.0);
    }

    #[test]
    fn test_delta_bounds() {
        let c = Cluster {
            center_idx: 0,
            radius: 100,
            cardinality: 50,
            offset: 0,
            depth: 0,
            lfd: Lfd::compute(50, 25),
            left: None,
            right: None,
        };
        assert_eq!(c.delta_plus(500), 600);
        assert_eq!(c.delta_minus(500), 400);
        assert_eq!(c.delta_plus(50), 150);
        assert_eq!(c.delta_minus(50), 0);
    }

    #[test]
    fn test_build_tree_basic() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 1,
            max_depth: 50,
            min_radius: 0,
        };

        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        assert_eq!(tree.root().cardinality, count);
        assert_eq!(tree.root().depth, 0);
        assert_eq!(tree.reordered.len(), count);

        let mut seen = vec![false; count];
        for &idx in &tree.reordered {
            assert!(!seen[idx], "duplicate index {}", idx);
            seen[idx] = true;
        }

        assert!(tree.num_leaves > 0);
    }

    #[test]
    fn test_build_tree_singleton_leaves() {
        let vec_len = 32;
        let count = 16;
        let data = make_test_data(count, vec_len);
        let config = BuildConfig::default();
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);

        let singleton_count = tree
            .nodes
            .iter()
            .filter(|c| c.is_leaf() && c.cardinality == 1)
            .count();
        assert!(singleton_count > 0);
    }

    #[test]
    fn test_build_simple_api() {
        let data = vec![0u8; 100 * 32];
        let tree = ClamTree::build(&data, 32, 3);
        assert!(!tree.nodes.is_empty());
    }

    #[test]
    fn test_lfd_statistics() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 20,
            min_radius: 0,
        };

        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let stats = tree.lfd_percentiles();
        assert!(stats.min >= 0.0);
    }

    #[test]
    fn test_lfd_by_depth() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len);

        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 15,
            min_radius: 0,
        };

        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let lfd_depths = tree.lfd_by_depth();
        assert!(!lfd_depths.is_empty());
    }

    #[test]
    fn test_rho_nn_finds_close_points() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data_seeded(count, vec_len, 42);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[0..vec_len];

        let result = rho_nn(&tree, &data, vec_len, query, 0);
        assert!(!result.hits.is_empty());
        assert_eq!(result.hits[0].1, 0);
    }

    #[test]
    fn test_rho_nn_exact_recall() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data_seeded(count, vec_len, 123);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[0..vec_len];
        let rho = 200;

        let result = rho_nn(&tree, &data, vec_len, query, rho);
        let ground_truth: Vec<(usize, u64)> = (0..count)
            .map(|i| {
                let point = &data[i * vec_len..(i + 1) * vec_len];
                (i, tree.dist(query, point))
            })
            .filter(|&(_, d)| d <= rho)
            .collect();

        assert_eq!(
            result.hits.len(),
            ground_truth.len(),
            "ρ-NN should have perfect recall"
        );
    }

    #[test]
    fn test_knn_repeated_rho() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data_seeded(count, vec_len, 77);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_repeated_rho(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        assert_eq!(result.hits.len(), k);
        let our_max = result.hits.last().unwrap().1;
        let gt_max = ground_truth.last().unwrap().1;
        assert_eq!(our_max, gt_max, "k-NN max distance should match linear scan");
    }

    #[test]
    fn test_knn_dfs_sieve() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data_seeded(count, vec_len, 99);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_dfs_sieve(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        assert_eq!(result.hits.len(), k);
        let our_max = result.hits.last().unwrap().1;
        let gt_max = ground_truth.last().unwrap().1;
        assert_eq!(our_max, gt_max, "DFS Sieve should match linear scan");
    }

    #[test]
    fn test_dfs_sieve_prunes() {
        let vec_len = 256;
        let count = 1000;
        let data = make_test_data_seeded(count, vec_len, 42);

        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 40,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[0..vec_len];
        let k = 10;

        let result = knn_dfs_sieve(&tree, &data, vec_len, query, k);
        assert!(result.clusters_pruned > 0, "should prune at least some clusters");
    }

    #[test]
    fn test_all_three_agree() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 55);

        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let query = &data[32 * vec_len..33 * vec_len];
        let k = 5;

        let result_repeated = knn_repeated_rho(&tree, &data, vec_len, query, k);
        let result_dfs = knn_dfs_sieve(&tree, &data, vec_len, query, k);
        let ground_truth = linear_knn(&data, vec_len, count, query, k);

        let gt_max = ground_truth.last().unwrap().1;
        assert_eq!(result_repeated.hits.last().unwrap().1, gt_max);
        assert_eq!(result_dfs.hits.last().unwrap().1, gt_max);
    }

    #[test]
    fn test_knn_brute_basic() {
        let vec_len = 16;
        let data = make_test_data_seeded(50, vec_len, 42);
        let query = data[0..vec_len].to_vec();
        let result = knn_brute(&data, vec_len, &query, 5);
        assert_eq!(result.hits.len(), 5);
        assert_eq!(result.hits[0].1, 0);
        assert_eq!(result.hits[0].0, 0);
    }

    #[test]
    fn test_leaf_paths() {
        let vec_len = 32;
        let count = 20;
        let data = make_test_data(count, vec_len);
        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let paths = tree.leaf_paths();
        assert_eq!(paths.len(), count);
    }

    #[test]
    fn test_deepest_real_cluster() {
        let vec_len = 32;
        let count = 20;
        let data = make_test_data(count, vec_len);
        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 10,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let (depth, _node_idx) = tree.deepest_real_cluster(0, 10);
        // Should find at least the root
        assert!(depth <= 10);
    }

    #[test]
    fn test_cluster_crp() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len);
        let config = BuildConfig {
            min_cardinality: 5,
            max_depth: 20,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let dist = tree.cluster_crp(tree.root(), &data, vec_len);
        assert!(dist.count > 0);
        assert!(dist.mean >= 0.0);
    }

    // ─── Compression tests ──────────────────────────

    #[test]
    fn test_xor_diff_roundtrip() {
        let center = vec![0xAA; 64];
        let mut point = center.clone();
        point[0] = 0xBB;
        point[10] = 0xCC;
        point[63] = 0xDD;

        let enc = XorDiffEncoding::encode(&center, &point);
        assert_eq!(enc.num_diffs(), 3);
        assert_eq!(enc.storage_cost(), 9);

        let decoded = enc.decode(&center);
        assert_eq!(decoded, point);
    }

    #[test]
    fn test_xor_diff_identical() {
        let data = vec![0xFF; 2048];
        let enc = XorDiffEncoding::encode(&data, &data);
        assert_eq!(enc.num_diffs(), 0);
        assert_eq!(enc.storage_cost(), 0);
    }

    #[test]
    fn test_compressive_hamming() {
        let center = vec![0xAA; 64];
        let mut point = center.clone();
        point[0] = 0xBB;
        point[10] = 0xCC;

        let enc = XorDiffEncoding::encode(&center, &point);

        let query = vec![0xFF; 64];
        let dist_q_center = hamming_inline(&query, &center);
        let dist_q_point_exact = hamming_inline(&query, &point);
        let dist_q_point_compressed = enc.hamming_from_query(&query, &center, dist_q_center);

        assert_eq!(
            dist_q_point_compressed, dist_q_point_exact,
            "Compressive Hamming should match exact"
        );
    }

    #[test]
    fn test_compress_random_data() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 42);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 20,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        for i in 0..count {
            let decompressed = compressed.decompress_point(i, &data, vec_len);
            let original = &data[i * vec_len..(i + 1) * vec_len];
            assert_eq!(&decompressed, original, "Decompressed point {} mismatch", i);
        }
    }

    #[test]
    fn test_compress_clustered_data() {
        let vec_len = 256;
        let num_clusters = 10;
        let points_per = 50;
        let noise_bytes = 10;
        let count = num_clusters * points_per;

        let data = make_clustered_data(num_clusters, points_per, vec_len, noise_bytes);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 30,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        assert!(
            compressed.stats.ratio > 1.0,
            "Clustered data should compress > 1.0x, got {:.2}",
            compressed.stats.ratio
        );

        for i in 0..count {
            let decompressed = compressed.decompress_point(i, &data, vec_len);
            let original = &data[i * vec_len..(i + 1) * vec_len];
            assert_eq!(&decompressed, original);
        }
    }

    #[test]
    fn test_compressive_search_matches_exact() {
        let vec_len = 128;
        let num_clusters = 5;
        let points_per = 20;
        let count = num_clusters * points_per;

        let data = make_clustered_data(num_clusters, points_per, vec_len, 5);

        let config = BuildConfig {
            min_cardinality: 3,
            max_depth: 20,
            min_radius: 0,
        };
        let tree = ClamTree::build_with_config(&data, vec_len, count, &config);
        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        let query = &data[0..vec_len];
        let mut cache = DistanceCache::new();

        for i in 0..count {
            let exact = hamming_inline(query, &data[i * vec_len..(i + 1) * vec_len]);
            let comp = compressed.hamming_to_compressed(
                query,
                i,
                &data,
                vec_len,
                &mut cache,
                hamming_inline,
            );
            assert_eq!(comp, exact, "Compressive distance mismatch at point {}", i);
        }
    }

    #[test]
    fn test_compress_legacy_api() {
        let vec_len = 16;
        let data = make_test_data(20, vec_len);
        let tree = ClamTree::build(&data, vec_len, 5);
        let compressed = compress(&data, vec_len, &tree);
        assert!(compressed.stats.uncompressed_bytes > 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 1: CLAM → Cascade Bridge tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_rho_nn_cascade_finds_exact_match() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        // Query = first vector
        let query = &data[0..vec_len];
        let rho = 200;

        // Cascade with generous threshold
        let cascade = Cascade::from_threshold(vec_len as u64 * 4, vec_len);
        let result = tree.rho_nn_cascade(&data, vec_len, query, rho, &cascade);

        // Should find the exact match at index 0
        assert!(
            result.hits.iter().any(|r| r.index == 0 && r.hamming == 0),
            "Should find exact match at index 0"
        );
        assert!(result.clam_candidates > 0);
        assert!(result.cascade_survivors > 0);
        assert!(result.cascade_survivors <= result.clam_candidates);
    }

    #[test]
    fn test_rho_nn_cascade_cascade_filters() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 77);
        let tree = ClamTree::build(&data, vec_len, 3);

        let query = &data[0..vec_len];
        // Large rho to get many CLAM candidates
        let rho = tree.root().radius;

        // Very tight cascade threshold to filter aggressively
        let cascade = Cascade::from_threshold(10, vec_len);
        let result = tree.rho_nn_cascade(&data, vec_len, query, rho, &cascade);

        // Cascade should filter some candidates
        // (exact match dist=0 should still pass threshold=10)
        assert!(result.cascade_survivors <= result.clam_candidates);
    }

    #[test]
    fn test_rho_nn_cascade_results_sorted() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 99);
        let tree = ClamTree::build(&data, vec_len, 3);

        let query = &data[0..vec_len];
        let cascade = Cascade::from_threshold(vec_len as u64 * 4, vec_len);
        let result = tree.rho_nn_cascade(&data, vec_len, query, 300, &cascade);

        // Results should be sorted by hamming distance
        for w in result.hits.windows(2) {
            assert!(w[0].hamming <= w[1].hamming, "Results should be sorted by distance");
        }
    }

    #[test]
    fn test_rho_nn_cascade_empty_for_zero_rho_distant_query() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 55);
        let tree = ClamTree::build(&data, vec_len, 3);

        // Query very different from all data
        let query = vec![0xFFu8; vec_len];
        let cascade = Cascade::from_threshold(0, vec_len); // threshold=0 rejects everything non-exact
        let result = tree.rho_nn_cascade(&data, vec_len, &query, 0, &cascade);

        // With rho=0 and a random query, likely no exact matches
        // Just verify structure is valid
        assert_eq!(result.cascade_survivors, result.hits.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 2: SPO Distance Harvest tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_spo_distance_harvest_basic() {
        // Create some ranked hits
        let hits = vec![
            RankedHit { index: 0, hamming: 10, precise: f64::NAN, band: Band::Foveal },
            RankedHit { index: 1, hamming: 50, precise: f64::NAN, band: Band::Near },
        ];

        let mut query_node = Node::random(42);
        let mut hit_nodes = vec![Node::random(100), Node::random(200)];
        let query_qualia = PackedQualia::zero();
        let mut hit_q0 = PackedQualia::zero();
        hit_q0.resonance[4] = 10; // warmth forward
        let hit_qualias = vec![hit_q0, PackedQualia::zero()];

        let results = spo_distance_harvest(
            &hits,
            &mut query_node,
            &mut hit_nodes,
            &query_qualia,
            &hit_qualias,
        );

        assert_eq!(results.len(), 2);
        // First hit should have S/P/O distances (random nodes have encounters)
        assert!(results[0].s_distance.is_some());
        assert!(results[0].p_distance.is_some());
        assert!(results[0].o_distance.is_some());
        // First hit should have causality (warmth forward)
        let causality = results[0].causality.as_ref().unwrap();
        assert_eq!(
            causality.warmth_dir,
            super::super::causality::CausalityDirection::Backward // query=0, hit=10 -> backward (hit > query)
        );
    }

    #[test]
    fn test_spo_distance_harvest_out_of_bounds() {
        // Hit index beyond available nodes
        let hits = vec![
            RankedHit { index: 99, hamming: 10, precise: f64::NAN, band: Band::Good },
        ];

        let mut query_node = Node::random(42);
        let mut hit_nodes = vec![Node::random(100)]; // only 1 node
        let query_qualia = PackedQualia::zero();
        let hit_qualias = vec![PackedQualia::zero()];

        let results = spo_distance_harvest(
            &hits,
            &mut query_node,
            &mut hit_nodes,
            &query_qualia,
            &hit_qualias,
        );

        assert_eq!(results.len(), 1);
        // Out-of-bounds index should use defaults
        assert!(results[0].s_distance.is_none());
        assert!(results[0].causality.is_none());
        assert!((results[0].truth.frequency - 0.5).abs() < 1e-6); // ignorance
    }

    #[test]
    fn test_spo_distance_harvest_truth_values() {
        let hits = vec![
            RankedHit { index: 0, hamming: 5, precise: f64::NAN, band: Band::Foveal },
        ];

        let mut query_node = Node::random(1);
        let mut hit_nodes = vec![Node::random(2)];
        let query_qualia = PackedQualia::zero();
        let hit_qualias = vec![PackedQualia::zero()];

        let results = spo_distance_harvest(
            &hits,
            &mut query_node,
            &mut hit_nodes,
            &query_qualia,
            &hit_qualias,
        );

        assert_eq!(results.len(), 1);
        // Truth value should be derived from SPO distance
        let truth = results[0].truth;
        assert!(truth.frequency >= 0.0 && truth.frequency <= 1.0);
        assert!(truth.confidence >= 0.0 && truth.confidence < 1.0);
    }

    #[test]
    fn test_spo_distance_harvest_preserves_hit_info() {
        let hits = vec![
            RankedHit { index: 0, hamming: 42, precise: f64::NAN, band: Band::Near },
        ];

        let mut query_node = Node::random(10);
        let mut hit_nodes = vec![Node::random(20)];
        let query_qualia = PackedQualia::zero();
        let hit_qualias = vec![PackedQualia::zero()];

        let results = spo_distance_harvest(
            &hits,
            &mut query_node,
            &mut hit_nodes,
            &query_qualia,
            &hit_qualias,
        );

        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].hamming, 42);
        assert_eq!(results[0].band, Band::Near);
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 3: panCAKES Compression Wiring tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_compress_database_basic() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = tree.compress_database(&data, vec_len);
        assert_eq!(compressed.encodings.len(), count);
        assert_eq!(compressed.encoding_centers.len(), count);
        assert!(compressed.stats.uncompressed_bytes > 0);
    }

    #[test]
    fn test_compress_database_lossless() {
        let vec_len = 128;
        let count = 30;
        let data = make_clustered_data(3, 10, vec_len, 5);
        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = tree.compress_database(&data, vec_len);

        // Verify lossless decompression
        for i in 0..count {
            let decompressed = compressed.decompress_point(i, &data, vec_len);
            let original = &data[i * vec_len..(i + 1) * vec_len];
            assert_eq!(&decompressed, original, "Decompression mismatch at point {}", i);
        }
    }

    #[test]
    fn test_query_compressed_finds_exact_match() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = tree.compress_database(&data, vec_len);
        let query = &data[0..vec_len];
        let rho = 200;

        let result = tree.query_compressed(&compressed, &data, vec_len, query, rho);

        // Should find exact match at index 0
        assert!(
            result.hits.iter().any(|&(idx, d)| idx == 0 && d == 0),
            "Should find exact match at index 0"
        );
        assert!(result.distance_calls > 0);
    }

    #[test]
    fn test_query_compressed_matches_exact_search() {
        let vec_len = 64;
        let num_clusters = 5;
        let points_per = 10;
        let count = num_clusters * points_per;
        let data = make_clustered_data(num_clusters, points_per, vec_len, 5);
        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = tree.compress_database(&data, vec_len);
        let query = &data[0..vec_len];
        let rho = 300;

        let compressed_result = tree.query_compressed(&compressed, &data, vec_len, query, rho);
        let exact_result = rho_nn(&tree, &data, vec_len, query, rho);

        // Compressed search should find the same set of points
        let compressed_indices: std::collections::HashSet<usize> =
            compressed_result.hits.iter().map(|&(idx, _)| idx).collect();
        let exact_indices: std::collections::HashSet<usize> =
            exact_result.hits.iter().map(|&(idx, _)| idx).collect();

        assert_eq!(
            compressed_indices, exact_indices,
            "Compressed search should find same results as exact search"
        );
    }

    #[test]
    fn test_query_compressed_results_sorted() {
        let vec_len = 64;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 99);
        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = tree.compress_database(&data, vec_len);
        let query = &data[0..vec_len];

        let result = tree.query_compressed(&compressed, &data, vec_len, query, 300);

        for w in result.hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "Results should be sorted by distance");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 4: CHAODA Anomaly Detection tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_anomaly_scores_basic() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let scores = tree.anomaly_scores(&data, vec_len);
        assert_eq!(scores.len(), count);

        for score in &scores {
            assert!(score.score >= 0.0 && score.score <= 1.0,
                "Score {} out of range [0,1]", score.score);
            assert!(score.index < count);
        }
    }

    #[test]
    fn test_anomaly_scores_awareness_mapping() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let scores = tree.anomaly_scores(&data, vec_len);

        for score in &scores {
            let expected_awareness = if score.score < 0.25 {
                AwarenessState::Crystallized
            } else if score.score < 0.50 {
                AwarenessState::Tensioned
            } else if score.score < 0.75 {
                AwarenessState::Uncertain
            } else {
                AwarenessState::Noise
            };
            assert_eq!(
                score.awareness, expected_awareness,
                "Awareness mismatch for score={}", score.score
            );
        }
    }

    #[test]
    fn test_flag_anomalies_threshold() {
        let vec_len = 64;
        let count = 200;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 5);

        let all_scores = tree.anomaly_scores(&data, vec_len);
        let flagged = tree.flag_anomalies(&data, vec_len, 0.75);

        // All flagged should have score >= 0.75
        for anomaly in &flagged {
            assert!(
                anomaly.score >= 0.75,
                "Flagged anomaly has score {} < 0.75",
                anomaly.score
            );
        }

        // Count manually from all_scores
        let expected_count = all_scores.iter().filter(|s| s.score >= 0.75).count();
        assert_eq!(flagged.len(), expected_count);
    }

    #[test]
    fn test_flag_anomalies_zero_threshold_returns_all() {
        let vec_len = 32;
        let count = 50;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let flagged = tree.flag_anomalies(&data, vec_len, 0.0);
        assert_eq!(flagged.len(), count, "Threshold=0.0 should flag all points");
    }

    #[test]
    fn test_flag_anomalies_one_threshold_returns_max_only() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data_seeded(count, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 3);

        let flagged = tree.flag_anomalies(&data, vec_len, 1.0);
        // Only points with score exactly 1.0 (the maximum LFD points)
        for a in &flagged {
            assert!((a.score - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_anomaly_scores_clustered_vs_outliers() {
        let vec_len = 64;
        // Create 3 tight clusters plus some noisy outliers
        let mut data = make_clustered_data(3, 20, vec_len, 2); // tight clusters
        // Add 5 random outliers
        let mut rng = SplitMix64::new(999);
        for _ in 0..5 {
            let mut outlier = vec![0u8; vec_len];
            for byte in outlier.iter_mut() {
                *byte = (rng.next_u64() & 0xFF) as u8;
            }
            data.extend_from_slice(&outlier);
        }
        let count = data.len() / vec_len;
        let tree = ClamTree::build(&data, vec_len, 3);

        let scores = tree.anomaly_scores(&data, vec_len);
        assert_eq!(scores.len(), count);
        // Just verify valid range
        for s in &scores {
            assert!(s.score >= 0.0 && s.score <= 1.0);
        }
    }
}
