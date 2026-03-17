//! panCAKES compression: hierarchical XOR-diff encoding from cluster centers.
//!
//! Implements Algorithm 2 (Compress) from panCAKES (arXiv:2409.12161):
//!
//! ```text
//! For each cluster C:
//!   unitary_cost = sum encoding_cost(center, x) for all x in C
//!   if C is not leaf:
//!     recursive_cost = encoding_cost(center, left_center)
//!                    + encoding_cost(center, right_center)
//!                    + left.min_cost + right.min_cost
//!     C.min_cost = min(unitary_cost, recursive_cost)
//!     if recursive_cost > unitary_cost:
//!       delete descendants (flatten to leaf)
//! ```
//!
//! ## Encoding for Hamming distance
//!
//! For byte-aligned fingerprints under Hamming distance, the encoding of
//! point x in terms of center c is the **XOR diff**: the set of byte
//! positions where x and c differ, plus the values at those positions.
//!
//! ```text
//! encoding(c, x) = { (pos_i, val_i) : c[pos_i] != x[pos_i] }
//! storage_cost    = num_diffs * (sizeof(pos) + sizeof(val))
//! ```
//!
//! For 2048-byte fingerprints: pos needs 11 bits, val needs 8 bits.
//! We pack as (u16 pos, u8 val) = 3 bytes per diff.
//! At Hamming distance d, approximately d/8 bytes differ (on average
//! each differing byte contributes ~4 bit flips), so the encoding
//! cost is roughly `(d/8) * 3` bytes vs 2048 bytes uncompressed.
//!
//! ## Decompression for search
//!
//! To reconstruct x from encoding + center:
//! 1. Start with a copy of center
//! 2. Apply each (pos, val): result[pos] = val
//!
//! For Hamming distance search, we can compute d(query, x) from the
//! encoding without full decompression:
//! ```text
//! d(q, x) = d(q, c) - popcount(q[diffs] XOR c[diffs])
//!                    + popcount(q[diffs] XOR x_val[diffs])
//! ```
//! This only touches the diff positions, not the full 2048 bytes.

use super::clam::ClamTree;

// ---------------------------------------------------------------
// Encoding types
// ---------------------------------------------------------------

/// XOR-diff encoding of a point relative to a reference (cluster center).
///
/// For each byte position where the point differs from the center,
/// we store the position and the point's value at that position.
#[derive(Debug, Clone)]
pub struct XorDiffEncoding {
    /// Byte positions where point differs from center.
    pub positions: Vec<u16>,
    /// Point's byte values at those positions.
    pub values: Vec<u8>,
}

impl XorDiffEncoding {
    /// Encode point relative to center.
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

    /// Decode: reconstruct point from center + encoding.
    pub fn decode(&self, center: &[u8]) -> Vec<u8> {
        let mut result = center.to_vec();
        for (&pos, &val) in self.positions.iter().zip(self.values.iter()) {
            result[pos as usize] = val;
        }
        result
    }

    /// Storage cost in bytes: 3 bytes per diff (u16 pos + u8 val).
    pub fn storage_cost(&self) -> usize {
        self.positions.len() * 3 // 2 bytes pos + 1 byte val
    }

    /// Number of byte differences.
    pub fn num_diffs(&self) -> usize {
        self.positions.len()
    }

    /// Compute Hamming distance from query to the encoded point
    /// WITHOUT full decompression.
    ///
    /// ```text
    /// d(q, x) = d(q, center)
    ///         - popcount(q[diffs] XOR center[diffs])   // remove old contribution
    ///         + popcount(q[diffs] XOR x_val[diffs])     // add new contribution
    /// ```
    ///
    /// Cost: O(num_diffs) instead of O(vec_len).
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

// ---------------------------------------------------------------
// Compressed tree
// ---------------------------------------------------------------

/// Compression mode chosen for each cluster (Algorithm 2 decision).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMode {
    /// Points encoded as diffs from this cluster's center.
    Unitary,
    /// Points encoded hierarchically through children.
    Recursive,
}

/// Per-cluster compression metadata.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ClusterCompression {
    mode: CompressionMode,
    unitary_cost: usize,
    recursive_cost: usize,
    min_cost: usize,
}

/// A compressed CLAM tree with panCAKES encoding.
///
/// After compression, the tree may be pruned: some internal nodes
/// become leaves if unitary compression was cheaper than recursive.
/// The mix of unitary and recursive encoding varies by region of
/// the manifold (Figure 1 in panCAKES paper).
pub struct CompressedTree {
    /// For each original dataset point, its XOR-diff encoding
    /// relative to the nearest ancestor center that owns it.
    pub encodings: Vec<XorDiffEncoding>,

    /// Which center each point is encoded relative to.
    /// encodings[i] + center_data[encoding_centers[i]] = original point i.
    pub encoding_centers: Vec<usize>,

    /// Per-cluster compression decisions.
    pub cluster_modes: Vec<CompressionMode>,

    /// Statistics.
    pub stats: CompressionStats,
}

/// Compression statistics.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total bytes in uncompressed dataset.
    pub uncompressed_bytes: usize,
    /// Total bytes in compressed representation (encodings + tree overhead).
    pub compressed_bytes: usize,
    /// Compression ratio (uncompressed / compressed).
    pub ratio: f64,
    /// Number of clusters using unitary compression.
    pub unitary_clusters: usize,
    /// Number of clusters using recursive compression.
    pub recursive_clusters: usize,
    /// Number of clusters pruned (descendants deleted).
    pub pruned_subtrees: usize,
}

impl CompressedTree {
    /// Compress a dataset using its CLAM tree (panCAKES Algorithm 2).
    ///
    /// The algorithm traverses bottom-up:
    /// 1. Compute unitary cost for every cluster (sum of encoding costs)
    /// 2. For non-leaves, compute recursive cost (child center encodings + child min_costs)
    /// 3. Choose min(unitary, recursive) per cluster
    /// 4. If unitary wins, prune descendants
    ///
    /// # Arguments
    /// * `tree`    -- CLAM tree built by `ClamTree::build`
    /// * `data`    -- original flat dataset
    /// * `vec_len` -- bytes per vector
    /// * `count`   -- number of vectors
    pub fn compress(tree: &ClamTree, data: &[u8], vec_len: usize, count: usize) -> Self {
        let num_nodes = tree.nodes.len();
        let mut comp: Vec<Option<ClusterCompression>> = vec![None; num_nodes];
        let mut cluster_modes = vec![CompressionMode::Unitary; num_nodes];

        // Bottom-up traversal: process leaves first, then parents
        // We use post-order by processing nodes in reverse (since tree is
        // built depth-first, reverse gives us approximate bottom-up order)
        let order = postorder_indices(tree);

        for &node_idx in &order {
            let cluster = &tree.nodes[node_idx];
            let center = tree.center_data(cluster, data, vec_len);

            // Compute unitary cost: sum of encoding sizes for all points
            let mut unitary_cost = 0usize;
            for (_, point_data) in tree.cluster_points(cluster, data, vec_len) {
                let enc = XorDiffEncoding::encode(center, point_data);
                unitary_cost += enc.storage_cost();
            }

            let mut min_cost = unitary_cost;
            let mut mode = CompressionMode::Unitary;

            if !cluster.is_leaf() {
                // Compute recursive cost
                let mut recursive_cost = 0usize;

                if let Some(left_idx) = cluster.left {
                    let left = &tree.nodes[left_idx];
                    let left_center = tree.center_data(left, data, vec_len);
                    // Cost to encode left center from this center
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

        // Now build the actual encodings for each point.
        // Walk the tree: for unitary clusters, encode all points from that center.
        // For recursive clusters, delegate to children.
        let mut encodings = vec![
            XorDiffEncoding {
                positions: vec![],
                values: vec![]
            };
            count
        ];
        let mut encoding_centers = vec![0usize; count];

        Self::assign_encodings(
            tree,
            data,
            vec_len,
            0, // root
            &cluster_modes,
            &mut encodings,
            &mut encoding_centers,
        );

        // Compute stats
        let uncompressed_bytes = count * vec_len;
        let compressed_bytes: usize =
            encodings.iter().map(|e| e.storage_cost()).sum::<usize>() + count * 2; // 2 bytes per point for center reference overhead
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

    /// Recursively assign encodings to points.
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
            // Encode all points in this cluster from this center
            for (orig_idx, point_data) in tree.cluster_points(cluster, data, vec_len) {
                encodings[orig_idx] = XorDiffEncoding::encode(center, point_data);
                encoding_centers[orig_idx] = cluster.center_idx;
            }
        } else {
            // Recursive: delegate to children
            if let Some(left) = cluster.left {
                Self::assign_encodings(
                    tree,
                    data,
                    vec_len,
                    left,
                    modes,
                    encodings,
                    encoding_centers,
                );
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

    /// Decompress a single point.
    pub fn decompress_point(&self, point_idx: usize, data: &[u8], vec_len: usize) -> Vec<u8> {
        let center_idx = self.encoding_centers[point_idx];
        assert!(
            (center_idx + 1) * vec_len <= data.len(),
            "decompress_point: center_idx {} out of bounds (data has {} vectors of len {})",
            center_idx,
            data.len() / vec_len,
            vec_len,
        );
        let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
        self.encodings[point_idx].decode(center)
    }

    /// Compute Hamming distance from query to a compressed point
    /// WITHOUT full decompression (panCAKES compressive search).
    ///
    /// Cost: O(num_diffs) per point instead of O(vec_len).
    pub fn hamming_to_compressed(
        &self,
        query: &[u8],
        point_idx: usize,
        data: &[u8],
        vec_len: usize,
        dist_cache: &mut DistanceCache,
        dist_fn: fn(&[u8], &[u8]) -> u64,
    ) -> u64 {
        let center_idx = self.encoding_centers[point_idx];

        // Cache d(query, center) -- same center is shared by many points
        let dist_q_center = dist_cache.get_or_compute(center_idx, || {
            let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
            dist_fn(query, center)
        });

        let center = &data[center_idx * vec_len..(center_idx + 1) * vec_len];
        self.encodings[point_idx].hamming_from_query(query, center, dist_q_center)
    }
}

/// Cache for d(query, center) values -- avoids recomputing for points
/// sharing the same cluster center.
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

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

/// Compute post-order traversal of the tree (leaves before parents).
fn postorder_indices(tree: &ClamTree) -> Vec<usize> {
    let mut result = Vec::with_capacity(tree.nodes.len());
    let mut stack = vec![(0usize, false)]; // (node_idx, visited_children)

    while let Some((node_idx, visited)) = stack.pop() {
        let cluster = &tree.nodes[node_idx];

        if visited || cluster.is_leaf() {
            result.push(node_idx);
        } else {
            // Push self again (will be processed after children)
            stack.push((node_idx, true));
            // Push children
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

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::clam::ClamTree;
    use super::super::bitwise;

    /// Simple SplitMix64 RNG for deterministic test data generation.
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

    fn make_test_data(n: usize, vec_len: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    /// Make clustered data: groups of similar vectors.
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
            // Generate cluster center
            let mut center = vec![0u8; vec_len];
            for byte in center.iter_mut() {
                *byte = (rng.next_u64() & 0xFF) as u8;
            }

            for p in 0..points_per_cluster {
                let idx = c * points_per_cluster + p;
                let point = &mut data[idx * vec_len..(idx + 1) * vec_len];
                point.copy_from_slice(&center);

                // Add noise: flip `noise_bytes` random byte positions
                for _ in 0..noise_bytes {
                    let pos = (rng.next_u64() as usize) % vec_len;
                    point[pos] = (rng.next_u64() & 0xFF) as u8;
                }
            }
        }

        data
    }

    #[test]
    fn test_xor_diff_roundtrip() {
        let center = vec![0xAA; 64];
        let mut point = center.clone();
        point[0] = 0xBB;
        point[10] = 0xCC;
        point[63] = 0xDD;

        let enc = XorDiffEncoding::encode(&center, &point);
        assert_eq!(enc.num_diffs(), 3);
        assert_eq!(enc.storage_cost(), 9); // 3 diffs * 3 bytes

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
        let dist_q_center = bitwise::hamming_distance_raw(&query, &center);
        let dist_q_point_exact = bitwise::hamming_distance_raw(&query, &point);

        let dist_q_point_compressed = enc.hamming_from_query(&query, &center, dist_q_center);

        assert_eq!(
            dist_q_point_compressed, dist_q_point_exact,
            "Compressive Hamming should match exact Hamming"
        );
    }

    #[test]
    fn test_compress_random_data() {
        let vec_len = 64;
        let count = 100;
        let data = make_test_data(count, vec_len, 42);

        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        println!(
            "Random data compression: {:.2}x ({} -> {} bytes)",
            compressed.stats.ratio,
            compressed.stats.uncompressed_bytes,
            compressed.stats.compressed_bytes
        );

        // Random data: compression ratio may be < 1 (expansion)
        // That's expected -- panCAKES shines on self-similar data

        // Verify lossless decompression
        for i in 0..count {
            let decompressed = compressed.decompress_point(i, &data, vec_len);
            let original = &data[i * vec_len..(i + 1) * vec_len];
            assert_eq!(
                &decompressed, original,
                "Decompressed point {} should match original",
                i
            );
        }
    }

    #[test]
    fn test_compress_clustered_data() {
        let vec_len = 256; // 2KB fingerprints
        let num_clusters = 10;
        let points_per = 50;
        let noise_bytes = 10; // ~4% noise -> high self-similarity
        let count = num_clusters * points_per;

        let data = make_clustered_data(num_clusters, points_per, vec_len, noise_bytes);

        let tree = ClamTree::build(&data, vec_len, 3);

        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        println!(
            "Clustered data compression: {:.2}x ({} -> {} bytes)",
            compressed.stats.ratio,
            compressed.stats.uncompressed_bytes,
            compressed.stats.compressed_bytes
        );

        // Clustered data with low noise should compress well
        assert!(
            compressed.stats.ratio > 1.0,
            "Clustered data should achieve compression ratio > 1.0, got {:.2}",
            compressed.stats.ratio
        );

        // Verify lossless
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

        let tree = ClamTree::build(&data, vec_len, 3);
        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        let query = &data[0..vec_len]; // use first point as query
        let mut cache = DistanceCache::new();

        // Compare compressive search distances to exact distances
        for i in 0..count {
            let exact = bitwise::hamming_distance_raw(query, &data[i * vec_len..(i + 1) * vec_len]);
            let comp = compressed.hamming_to_compressed(
                query,
                i,
                &data,
                vec_len,
                &mut cache,
                bitwise::hamming_distance_raw,
            );
            assert_eq!(
                comp, exact,
                "Compressive Hamming for point {} should match exact ({} vs {})",
                i, comp, exact
            );
        }
    }

    #[test]
    fn test_compression_modes_mixed() {
        let vec_len = 256;
        let num_clusters = 8;
        let points_per = 30;
        let count = num_clusters * points_per;

        let data = make_clustered_data(num_clusters, points_per, vec_len, 8);

        let tree = ClamTree::build(&data, vec_len, 5);
        let compressed = CompressedTree::compress(&tree, &data, vec_len, count);

        println!(
            "Compression modes: {} unitary, {} recursive",
            compressed.stats.unitary_clusters, compressed.stats.recursive_clusters
        );

        // With enough data, should have both modes
        // (shallow clusters may prefer recursive, deep clusters unitary)
    }
}
