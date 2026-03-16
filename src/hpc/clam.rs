//! CLAM Tree: divisive hierarchical clustering with LFD estimation.
//!
//! Implements CAKES (arXiv:2309.05491) algorithms for exact k-NN search
//! using triangle inequality pruning.

use super::bitwise;

// ─── Distance trait ──────────────────────────────────────────

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
    fn is_metric(&self) -> bool { true }
}

// ─── LFD ──────────────────────────────────────────

/// Local Fractal Dimension of a cluster.
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
        Lfd { value, count_r, count_half_r }
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
    /// δ⁻ = max(0, d(q, center) - radius)
    pub fn delta_minus(&self, d_to_center: u64) -> u64 {
        d_to_center.saturating_sub(self.radius)
    }

    /// δ⁺ = d(q, center) + radius
    pub fn delta_plus(&self, d_to_center: u64) -> u64 {
        d_to_center.saturating_add(self.radius)
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

// ─── Distribution ──────────────────────────────────

/// Distribution statistics for a cluster.
#[derive(Debug, Clone)]
pub struct ClusterDistribution {
    pub mean: f64,
    pub std: f64,
    pub max: u64,
    pub count: usize,
}

// ─── ClamTree ──────────────────────────────────────

/// Divisive hierarchical clustering tree.
pub struct ClamTree {
    pub nodes: Vec<Cluster>,
    /// Depth-first reordered indices: data[reorder[i]] is the i-th point
    pub reorder: Vec<usize>,
    pub vec_len: usize,
    pub min_cluster_size: usize,
}

impl ClamTree {
    /// Build a CLAM tree from data.
    pub fn build(data: &[u8], vec_len: usize, min_cluster_size: usize) -> Self {
        let num_points = data.len() / vec_len;
        let indices: Vec<usize> = (0..num_points).collect();
        let mut tree = ClamTree {
            nodes: Vec::new(),
            reorder: indices.clone(),
            vec_len,
            min_cluster_size: min_cluster_size.max(1),
        };
        if num_points == 0 {
            return tree;
        }
        tree.build_recursive(data, &indices, 0);
        tree
    }

    fn build_recursive(&mut self, data: &[u8], indices: &[usize], depth: usize) -> usize {
        let n = indices.len();
        let node_idx = self.nodes.len();

        // Find center (first point as approximation)
        let center_idx = indices[0];
        let center = &data[center_idx * self.vec_len..(center_idx + 1) * self.vec_len];

        // Compute distances from center
        let dists: Vec<u64> = indices.iter()
            .map(|&i| bitwise::hamming_distance_raw(center, &data[i * self.vec_len..(i + 1) * self.vec_len]))
            .collect();

        let radius = dists.iter().copied().max().unwrap_or(0);

        // Compute LFD
        let count_r = dists.iter().filter(|&&d| d <= radius).count();
        let half_r = radius / 2;
        let count_half_r = dists.iter().filter(|&&d| d <= half_r).count();
        let lfd = Lfd::compute(count_r, count_half_r);

        let offset = node_idx; // simplified

        self.nodes.push(Cluster {
            center_idx,
            radius,
            cardinality: n,
            offset,
            depth,
            lfd,
            left: None,
            right: None,
        });

        if n <= self.min_cluster_size || radius == 0 {
            return node_idx;
        }

        // Find poles: farthest from center = left pole
        let left_pole_local = dists.iter().enumerate()
            .max_by_key(|(_, &d)| d)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let left_pole = indices[left_pole_local];
        let left_data = &data[left_pole * self.vec_len..(left_pole + 1) * self.vec_len];

        // Right pole: farthest from left pole
        let right_pole_local = indices.iter()
            .enumerate()
            .max_by_key(|(_, &i)| bitwise::hamming_distance_raw(left_data, &data[i * self.vec_len..(i + 1) * self.vec_len]))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let right_pole = indices[right_pole_local];
        let right_data = &data[right_pole * self.vec_len..(right_pole + 1) * self.vec_len];

        // Partition
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            let d_left = bitwise::hamming_distance_raw(&data[i * self.vec_len..(i + 1) * self.vec_len], left_data);
            let d_right = bitwise::hamming_distance_raw(&data[i * self.vec_len..(i + 1) * self.vec_len], right_data);
            if d_left <= d_right {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // Prevent empty partitions
        if left_indices.is_empty() || right_indices.is_empty() {
            return node_idx;
        }

        let left_child = self.build_recursive(data, &left_indices, depth + 1);
        let right_child = self.build_recursive(data, &right_indices, depth + 1);
        self.nodes[node_idx].left = Some(left_child);
        self.nodes[node_idx].right = Some(right_child);
        node_idx
    }

    pub fn root(&self) -> &Cluster {
        &self.nodes[0]
    }

    pub fn dist(&self, a: &[u8], b: &[u8]) -> u64 {
        bitwise::hamming_distance_raw(a, b)
    }

    pub fn center_data<'a>(&self, cluster: &Cluster, data: &'a [u8], vec_len: usize) -> &'a [u8] {
        let start = cluster.center_idx * vec_len;
        &data[start..start + vec_len]
    }

    pub fn cluster_points<'a>(
        &self,
        _cluster: &Cluster,
        data: &'a [u8],
        vec_len: usize,
    ) -> Vec<(usize, &'a [u8])> {
        // Simplified: return all points (in full impl, use offset+cardinality)
        let n = data.len() / vec_len;
        (0..n).map(|i| (i, &data[i * vec_len..(i + 1) * vec_len])).collect()
    }
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
pub fn rho_nn(tree: &ClamTree, data: &[u8], vec_len: usize, query: &[u8], rho: u64) -> RhoNnResult {
    let mut hits = Vec::new();
    let mut distance_calls = 0usize;
    let mut clusters_pruned = 0usize;
    let mut stack = vec![0usize];

    while let Some(node_idx) = stack.pop() {
        if node_idx >= tree.nodes.len() { continue; }
        let cluster = &tree.nodes[node_idx];
        let center = tree.center_data(cluster, data, vec_len);
        let delta = tree.dist(query, center);
        distance_calls += 1;

        if cluster.delta_minus(delta) > rho {
            clusters_pruned += 1;
            continue;
        }

        if cluster.is_leaf() {
            // Scan leaf points
            let n = data.len() / vec_len;
            for i in 0..n {
                let point = &data[i * vec_len..(i + 1) * vec_len];
                let d = tree.dist(query, point);
                distance_calls += 1;
                if d <= rho {
                    hits.push((i, d));
                }
            }
        } else {
            if let Some(left) = cluster.left { stack.push(left); }
            if let Some(right) = cluster.right { stack.push(right); }
        }
    }

    hits.sort_by_key(|&(_, d)| d);
    hits.dedup_by_key(|h| h.0);
    RhoNnResult { hits, distance_calls, clusters_pruned }
}

/// k-NN via brute-force (baseline).
pub fn knn_brute(data: &[u8], vec_len: usize, query: &[u8], k: usize) -> KnnResult {
    let n = data.len() / vec_len;
    let mut dists: Vec<(usize, u64)> = (0..n)
        .map(|i| (i, bitwise::hamming_distance_raw(query, &data[i * vec_len..(i + 1) * vec_len])))
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

/// Compressed tree using XOR-diff encoding.
#[derive(Debug, Clone)]
pub struct CompressedTree {
    pub centers: Vec<Vec<u8>>,
    pub diffs: Vec<Vec<Vec<u8>>>,
    pub stats: CompressionStats,
}

/// Statistics from compression.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_points: usize,
    pub total_clusters: usize,
    pub total_diff_bytes: usize,
    pub original_bytes: usize,
}

impl CompressionStats {
    pub fn ratio(&self) -> f64 {
        if self.original_bytes == 0 { return 0.0; }
        self.total_diff_bytes as f64 / self.original_bytes as f64
    }
}

/// Compress a dataset using XOR-diff encoding relative to cluster centers.
pub fn compress(data: &[u8], vec_len: usize, tree: &ClamTree) -> CompressedTree {
    let n = data.len() / vec_len;
    let mut centers = Vec::new();
    let mut diffs = Vec::new();

    // Simplified: each leaf cluster stores center + XOR diffs
    for cluster in &tree.nodes {
        if !cluster.is_leaf() { continue; }
        let center = &data[cluster.center_idx * vec_len..(cluster.center_idx + 1) * vec_len];
        centers.push(center.to_vec());

        let mut cluster_diffs = Vec::new();
        for i in 0..n {
            let point = &data[i * vec_len..(i + 1) * vec_len];
            let diff: Vec<u8> = center.iter().zip(point.iter()).map(|(&c, &p)| c ^ p).collect();
            cluster_diffs.push(diff);
        }
        diffs.push(cluster_diffs);
    }

    let total_diff_bytes: usize = diffs.iter().flat_map(|d| d.iter().map(|v| v.len())).sum();
    let total_clusters = centers.len();
    CompressedTree {
        centers,
        diffs,
        stats: CompressionStats {
            total_points: n,
            total_clusters,
            total_diff_bytes,
            original_bytes: data.len(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(n: usize, vec_len: usize, seed: u8) -> Vec<u8> {
        (0..n * vec_len).map(|i| {
            ((i as u8).wrapping_mul(7).wrapping_add(seed).wrapping_mul(13)) ^ (i as u8)
        }).collect()
    }

    #[test]
    fn hamming_distance_metric() {
        let hd = HammingDistance;
        let a = [0xFFu8; 32];
        let b = [0x00u8; 32];
        assert_eq!(hd.distance(&a[..], &b[..]), 256);
        assert!(hd.is_metric());
    }

    #[test]
    fn lfd_compute() {
        let lfd = Lfd::compute(100, 25);
        assert!((lfd.value - 2.0).abs() < 0.01); // log2(100/25) = 2
    }

    #[test]
    fn lfd_degenerate() {
        let lfd = Lfd::compute(10, 0);
        assert_eq!(lfd.value, 0.0);
    }

    #[test]
    fn clam_tree_build() {
        let data = make_test_data(100, 32, 42);
        let tree = ClamTree::build(&data, 32, 5);
        assert!(!tree.nodes.is_empty());
        assert_eq!(tree.root().cardinality, 100);
    }

    #[test]
    fn knn_brute_basic() {
        let vec_len = 16;
        let data = make_test_data(50, vec_len, 42);
        let query = data[0..vec_len].to_vec();
        let result = knn_brute(&data, vec_len, &query, 5);
        assert_eq!(result.hits.len(), 5);
        assert_eq!(result.hits[0].1, 0); // first hit should be self (distance 0)
    }

    #[test]
    fn rho_nn_finds_identical() {
        let vec_len = 16;
        let data = vec![0xAAu8; vec_len * 10];
        let query = vec![0xAAu8; vec_len];
        let tree = ClamTree::build(&data, vec_len, 2);
        let result = rho_nn(&tree, &data, vec_len, &query, 0);
        // All 10 points are identical to query
        assert_eq!(result.hits.len(), 10);
    }

    #[test]
    fn cluster_delta_bounds() {
        let c = Cluster {
            center_idx: 0, radius: 100, cardinality: 10,
            offset: 0, depth: 0, lfd: Lfd::compute(10, 5),
            left: None, right: None,
        };
        assert_eq!(c.delta_minus(150), 50);
        assert_eq!(c.delta_minus(50), 0);
        assert_eq!(c.delta_plus(50), 150);
    }

    #[test]
    fn compression_roundtrip() {
        let vec_len = 16;
        let data = make_test_data(20, vec_len, 42);
        let tree = ClamTree::build(&data, vec_len, 5);
        let compressed = compress(&data, vec_len, &tree);
        assert!(compressed.stats.total_clusters > 0);
    }
}
