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
}
