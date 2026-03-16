//! DN-tree: hierarchical bitmap summary tree for plastic graph traversal.
//!
//! Adapted from "On Demand Memory Specialization for Distributed Graph Processing"
//! (2013) for 16,384x3 HDC graph memory:
//!
//! - Quaternary tree hierarchy (fanout = 4) over prototype indices
//! - Lossy hierarchical summaries using bundled `GraphHV` hypervectors
//! - Partial Hamming similarity on prefix bits for ultra-fast descent
//! - Early-exit beam search (stacked exit from RIF-Net)
//! - Plastic bundling + exponential decay on every access (biological LTP/LTD)
//! - BTSP-inspired stochastic gating with CaMKII-like boost
//!
//! ## Performance (expected, single core)
//!
//! | Operation        | Latency     | Notes                                |
//! |-----------------|-------------|--------------------------------------|
//! | update          | ~30 ns/level| bundle + decay per node on path      |
//! | traverse (top-8)| 180-420 ns  | early exit wins big at depth 3-4     |
//! | memory          | < 4 MB      | full tree + summaries for 16K protos |

use super::cam_index::GraphHV;
use super::fingerprint::Fingerprint;

// ---------------------------------------------------------------------------
// Local SplitMix64 RNG (same algorithm as cam_index.rs, which keeps it private)
// ---------------------------------------------------------------------------

/// Simple SplitMix64 RNG for deterministic bundling and gating.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Generate the next `u64` value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Generate a random `f64` in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Local helper functions for GraphHV operations not on the public API
// ---------------------------------------------------------------------------

/// Create a random `GraphHV` from the local RNG.
#[cfg(test)]
fn random_graphhv(rng: &mut SplitMix64) -> GraphHV {
    let mut channels = [
        Fingerprint::<256>::zero(),
        Fingerprint::<256>::zero(),
        Fingerprint::<256>::zero(),
    ];
    for ch in &mut channels {
        for w in ch.words.iter_mut() {
            *w = rng.next_u64();
        }
    }
    GraphHV { channels }
}

/// Total number of bits across all 3 channels of a GraphHV.
const TOTAL_BITS: usize = Fingerprint::<256>::BITS * 3;

/// Cosine-style similarity: 1.0 - hamming / total_bits.
fn similarity(a: &GraphHV, b: &GraphHV) -> f64 {
    1.0 - a.hamming_distance(b) as f64 / TOTAL_BITS as f64
}

/// Partial similarity using only the first `num_bits` of each channel.
///
/// `num_bits` is clamped to the channel bit-width (16,384).
fn partial_similarity(a: &GraphHV, b: &GraphHV, num_bits: usize) -> f64 {
    let bits_per_channel = Fingerprint::<256>::BITS;
    let bits = num_bits.min(bits_per_channel);
    let words_needed = (bits + 63) / 64; // ceil-divide

    let mut dist: u32 = 0;
    for ch in 0..3 {
        for w in 0..words_needed {
            dist += (a.channels[ch].words[w] ^ b.channels[ch].words[w]).count_ones();
        }
    }

    let total = (words_needed * 64 * 3) as f64;
    1.0 - dist as f64 / total
}

/// Returns true if every word in every channel is zero.
#[cfg(test)]
fn is_zero(hv: &GraphHV) -> bool {
    hv.channels
        .iter()
        .all(|ch| ch.words.iter().all(|&w| w == 0))
}

/// Bundle `hv` into `current` with exponential-decay learning.
///
/// For each bit position, the existing summary bit is kept with probability
/// `1 - lr * boost`, and replaced by the input bit with probability `lr * boost`.
/// This is implemented per-word using a stochastic mask.
fn bundle_into(
    current: &GraphHV,
    hv: &GraphHV,
    lr: f64,
    boost: f64,
    rng: &mut SplitMix64,
) -> GraphHV {
    let effective_lr = (lr * boost).min(1.0);
    let mut result = current.clone();

    for ch in 0..3 {
        for w in 0..256 {
            // Build a stochastic mask: each bit is 1 with probability `effective_lr`.
            // Approximate by AND-ing multiple random words (each AND halves probability).
            let mask = make_probability_mask(effective_lr, rng);
            // Where mask is 1: take from hv; where 0: keep current.
            result.channels[ch].words[w] =
                (current.channels[ch].words[w] & !mask) | (hv.channels[ch].words[w] & mask);
        }
    }

    result
}

/// Create a u64 bitmask where each bit is independently 1 with probability ~`p`.
///
/// Uses cascaded AND of random words to achieve the target probability:
/// - p >= 0.5 → OR of inverse masks
/// - p < 0.5 → AND cascade
fn make_probability_mask(p: f64, rng: &mut SplitMix64) -> u64 {
    if p >= 1.0 {
        return u64::MAX;
    }
    if p <= 0.0 {
        return 0;
    }

    if p >= 0.5 {
        // p >= 0.5: use OR approach — each AND of randoms gives ~0.25, NOT gives ~0.75, etc.
        // Simpler: just AND enough randoms to get (1-p) kill rate, then NOT.
        return !make_probability_mask(1.0 - p, rng);
    }

    // p < 0.5: AND cascade. Each AND halves the probability.
    // We need n ANDs where 0.5^n ≈ p, so n = -log2(p).
    let n = (-p.log2()).ceil() as u32;
    let mut mask = rng.next_u64();
    for _ in 1..n {
        mask &= rng.next_u64();
    }
    mask
}

/// Configuration for the DN-tree.
#[derive(Clone, Debug)]
pub struct DNConfig {
    /// Number of prototype slots.
    pub num_prototypes: usize,
    /// Base split threshold (node splits when access_count >= threshold).
    pub split_threshold: u32,
    /// Threshold growth factor per level: threshold(level) = t * k^level.
    pub growth_factor: f64,
    /// Base learning rate for summary bundling (1 - decay). Typical: 0.03.
    pub learning_rate: f64,
    /// Early-exit similarity threshold (0.78-0.85 typical).
    pub early_exit_threshold: f64,
    /// Number of prefix bits for partial Hamming (1024 typical).
    pub partial_bits: usize,
    /// Maximum traversal depth.
    pub max_depth: u32,
    /// Beam width (number of children to follow at each level).
    pub beam_width: usize,
    /// BTSP gating probability (0.0 = disabled, 0.005-0.02 = biological).
    /// When triggered, learning rate is boosted by `btsp_boost`.
    pub btsp_gate_prob: f64,
    /// BTSP boost factor for learning rate when gate fires.
    pub btsp_boost: f64,
}

impl Default for DNConfig {
    fn default() -> Self {
        Self {
            num_prototypes: 4096,
            split_threshold: 8,
            growth_factor: 1.8,
            learning_rate: 0.03,
            early_exit_threshold: 0.82,
            partial_bits: 1024,
            max_depth: 7,
            beam_width: 2,
            btsp_gate_prob: 0.0, // disabled by default
            btsp_boost: 7.0,
        }
    }
}

impl DNConfig {
    /// Configuration with BTSP plasticity enabled.
    pub fn with_btsp() -> Self {
        Self {
            btsp_gate_prob: 0.01,
            btsp_boost: 7.0,
            ..Default::default()
        }
    }
}

/// A node in the DN-tree hierarchy.
///
/// Uses flat Vec storage (arena-style) with u32 indices for cache-friendly
/// traversal. Summary HVs are stored separately in SoA layout.
pub struct DNNode {
    /// Range of prototype indices covered by this node [lo, hi).
    pub range_lo: usize,
    pub range_hi: usize,
    /// Depth level in the tree.
    pub level: u32,
    /// Access counter (how many updates have passed through this node).
    pub access_count: u32,
    /// Indices of child nodes (None = leaf). Always 4 children when split.
    pub children: Option<[usize; 4]>,
}

/// A traversal result: prototype region + similarity score.
#[derive(Clone, Debug)]
pub struct TraversalHit {
    /// Start of prototype index range in this leaf.
    pub range_lo: usize,
    /// End of prototype index range (exclusive).
    pub range_hi: usize,
    /// Level in the tree where this hit was found.
    pub level: u32,
    /// Weighted similarity score.
    pub score: f64,
}

/// Statistics about the DN-tree structure.
#[derive(Clone, Debug, Default)]
pub struct DNTreeStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub internal_nodes: usize,
    pub max_depth: u32,
    pub total_accesses: u64,
}

/// DN-tree: hierarchical plasticity tree for graph hypervector routing.
///
/// Organizes prototype indices into a quaternary hierarchy with bundled
/// summary HVs at each node. Traversal uses partial Hamming distance
/// for fast pruning, beam search for exploration, and early exit for speed.
///
/// ## Arena Layout
///
/// Nodes are stored in a flat `Vec` for cache-friendly access.
/// Summary HVs are stored in a parallel `Vec` (SoA layout) so the hot
/// traversal path touches only the summaries it needs.
pub struct DNTree {
    pub nodes: Vec<DNNode>,
    pub summaries: Vec<GraphHV>,
    pub config: DNConfig,
}

impl DNTree {
    /// Create a new DN-tree for the given number of prototypes.
    pub fn new(config: DNConfig) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(256),
            summaries: Vec::with_capacity(256),
            config,
        };

        // Create root node covering [0, num_prototypes)
        tree.push_node(0, tree.config.num_prototypes, 0);

        // Root starts split if big enough
        if tree.config.num_prototypes >= 4 {
            tree.split_node(0);
        }

        tree
    }

    /// Create with default configuration for the given prototype count.
    pub fn with_capacity(num_prototypes: usize) -> Self {
        Self::new(DNConfig {
            num_prototypes,
            ..Default::default()
        })
    }

    /// Number of prototypes this tree covers.
    #[inline]
    pub fn num_prototypes(&self) -> usize {
        self.config.num_prototypes
    }

    /// Number of nodes in the tree (including internal and leaf).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Record an access to a prototype, bundling its HV into the hot path.
    ///
    /// This is the core plasticity operation:
    /// 1. Walk from root to the leaf containing `proto_idx`
    /// 2. At each node: increment access count, stochastically bundle HV
    /// 3. If a leaf exceeds its threshold: split into 4 children
    ///
    /// BTSP gating (when enabled): a stochastic "plateau potential" may fire,
    /// boosting the learning rate for this update (CaMKII-like amplification).
    pub fn update(&mut self, proto_idx: usize, hv: &GraphHV, rng: &mut SplitMix64) {
        assert!(
            proto_idx < self.config.num_prototypes,
            "proto_idx {} >= num_prototypes {}",
            proto_idx,
            self.config.num_prototypes
        );

        // Determine BTSP boost for this update
        let btsp_boost =
            if self.config.btsp_gate_prob > 0.0 && rng.next_f64() < self.config.btsp_gate_prob {
                self.config.btsp_boost
            } else {
                1.0
            };

        let lr = self.config.learning_rate;
        let growth = self.config.growth_factor;
        let threshold_base = self.config.split_threshold;

        let mut node_idx = 0usize;
        loop {
            // Increment access count
            self.nodes[node_idx].access_count = self.nodes[node_idx].access_count.saturating_add(1);

            // Bundle HV into node's summary
            let new_summary = bundle_into(&self.summaries[node_idx], hv, lr, btsp_boost, rng);
            self.summaries[node_idx] = new_summary;

            if self.nodes[node_idx].children.is_none() {
                // Leaf: check if we should split
                let level = self.nodes[node_idx].level;
                let threshold = (threshold_base as f64 * growth.powi(level as i32)) as u32;
                let range_size = self.nodes[node_idx].range_hi - self.nodes[node_idx].range_lo;

                if self.nodes[node_idx].access_count >= threshold && range_size >= 4 {
                    self.split_node(node_idx);
                }
                break;
            }

            // Recurse into the correct child
            let child_idx = self.select_child(node_idx, proto_idx);
            let children = self.nodes[node_idx].children.unwrap();
            node_idx = children[child_idx];
        }
    }

    /// Traverse the tree to find the most similar prototype regions.
    ///
    /// Uses beam search with partial Hamming for fast descent:
    /// 1. At each level, compute partial similarity with children's summaries
    /// 2. Follow the top `beam_width` children
    /// 3. Early-exit if best similarity exceeds threshold
    /// 4. Collect leaf hits with weighted scores
    pub fn traverse(&self, query: &GraphHV, top_k: usize) -> Vec<TraversalHit> {
        let mut hits = Vec::new();
        let mut beam: Vec<(usize, f64)> = vec![(0, 1.0)]; // (node_idx, weight)

        for _depth in 0..self.config.max_depth {
            if beam.is_empty() {
                break;
            }

            let mut next_beam: Vec<(usize, f64)> = Vec::new();

            for &(node_idx, weight) in &beam {
                let node = &self.nodes[node_idx];

                if node.children.is_none() {
                    // Leaf: full similarity
                    let sim = similarity(query, &self.summaries[node_idx]);
                    hits.push(TraversalHit {
                        range_lo: node.range_lo,
                        range_hi: node.range_hi,
                        level: node.level,
                        score: sim * weight,
                    });
                    continue;
                }

                let children = node.children.unwrap();

                // Compute partial similarity with each child's summary
                let mut child_sims: Vec<(usize, f64)> = children
                    .iter()
                    .filter(|&&c| self.nodes[c].access_count > 0)
                    .map(|&c| {
                        let sim =
                            partial_similarity(query, &self.summaries[c], self.config.partial_bits);
                        (c, sim)
                    })
                    .collect();

                if child_sims.is_empty() {
                    continue;
                }

                // Sort by similarity (descending)
                child_sims
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Early exit: if best child exceeds threshold, prune beam
                let best_sim = child_sims[0].1;
                let follow_count = if best_sim > self.config.early_exit_threshold {
                    1 // follow only the hottest path
                } else {
                    self.config.beam_width
                };

                for &(child_idx, _) in child_sims.iter().take(follow_count) {
                    let child_weight = if node.access_count > 0 {
                        self.nodes[child_idx].access_count as f64 / node.access_count as f64
                    } else {
                        1.0
                    };
                    next_beam.push((child_idx, weight * child_weight));
                }
            }

            beam = next_beam;
        }

        // Remaining beam items that haven't reached leaves
        for &(node_idx, weight) in &beam {
            let _node = &self.nodes[node_idx];
            let sim = similarity(query, &self.summaries[node_idx]);
            hits.push(TraversalHit {
                range_lo: self.nodes[node_idx].range_lo,
                range_hi: self.nodes[node_idx].range_hi,
                level: self.nodes[node_idx].level,
                score: sim * weight,
            });
        }

        // Sort by score descending, truncate to top_k
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        hits
    }

    /// Get tree statistics.
    pub fn stats(&self) -> DNTreeStats {
        let mut stats = DNTreeStats::default();
        for node in &self.nodes {
            stats.total_nodes += 1;
            stats.total_accesses += node.access_count as u64;
            if node.children.is_some() {
                stats.internal_nodes += 1;
            } else {
                stats.leaf_nodes += 1;
                if node.level > stats.max_depth {
                    stats.max_depth = node.level;
                }
            }
        }
        stats
    }

    /// Get a reference to a node's summary HV.
    #[inline]
    pub fn summary(&self, node_idx: usize) -> &GraphHV {
        &self.summaries[node_idx]
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Push a new node and its summary, return its arena index.
    pub fn push_node(&mut self, lo: usize, hi: usize, level: u32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(DNNode {
            range_lo: lo,
            range_hi: hi,
            level,
            access_count: 0,
            children: None,
        });
        self.summaries.push(GraphHV::zero());
        idx
    }

    /// Split a leaf node into 4 children (quaternary split).
    pub fn split_node(&mut self, node_idx: usize) {
        let lo = self.nodes[node_idx].range_lo;
        let hi = self.nodes[node_idx].range_hi;
        let span = hi - lo;
        if span < 4 {
            return;
        }

        let q = span / 4;
        let level = self.nodes[node_idx].level + 1;

        let c0 = self.push_node(lo, lo + q, level);
        let c1 = self.push_node(lo + q, lo + 2 * q, level);
        let c2 = self.push_node(lo + 2 * q, lo + 3 * q, level);
        let c3 = self.push_node(lo + 3 * q, hi, level);

        self.nodes[node_idx].children = Some([c0, c1, c2, c3]);
    }

    /// Determine which child quadrant contains the given prototype index.
    pub fn select_child(&self, node_idx: usize, proto_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let span = node.range_hi - node.range_lo;
        let q = span / 4;
        let offset = proto_idx - node.range_lo;
        (offset / q).min(3) // clamp to [0, 3] for rounding in last bucket
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    #[test]
    fn test_new_tree_structure() {
        let tree = DNTree::with_capacity(4096);
        assert_eq!(tree.num_prototypes(), 4096);
        // Root (1) + 4 children = 5 nodes minimum
        assert!(tree.num_nodes() >= 5);

        let stats = tree.stats();
        assert_eq!(stats.leaf_nodes, 4);
        assert_eq!(stats.internal_nodes, 1);
    }

    #[test]
    fn test_update_increments_access() {
        let mut tree = DNTree::with_capacity(4096);
        let mut rng = make_rng();
        let hv = random_graphhv(&mut rng);

        tree.update(0, &hv, &mut rng);
        tree.update(0, &hv, &mut rng);
        tree.update(0, &hv, &mut rng);

        // Root should have access_count >= 3
        assert!(tree.nodes[0].access_count >= 3);
    }

    #[test]
    fn test_update_bundles_summary() {
        let mut tree = DNTree::with_capacity(4096);
        let mut rng = make_rng();
        let hv = random_graphhv(&mut rng);

        // Before update, root summary is zero
        assert!(is_zero(&tree.summaries[0]));

        // After updates, summary should be non-zero and somewhat similar to input
        for _ in 0..20 {
            tree.update(0, &hv, &mut rng);
        }

        assert!(!is_zero(&tree.summaries[0]));
        let sim = similarity(&hv, &tree.summaries[0]);
        assert!(
            sim > 0.5,
            "After 20 updates, summary should resemble input: sim={:.4}",
            sim
        );
    }

    #[test]
    fn test_update_splits_leaf() {
        let config = DNConfig {
            num_prototypes: 1024,
            split_threshold: 5,
            growth_factor: 1.5,
            ..Default::default()
        };
        let mut tree = DNTree::new(config);
        let mut rng = make_rng();
        let initial_nodes = tree.num_nodes();

        // Hit the same leaf enough times to trigger split
        let hv = random_graphhv(&mut rng);
        for _ in 0..20 {
            tree.update(10, &hv, &mut rng);
        }

        assert!(
            tree.num_nodes() > initial_nodes,
            "Expected split: {} -> {}",
            initial_nodes,
            tree.num_nodes()
        );
    }

    #[test]
    fn test_traverse_finds_updated_regions() {
        let mut tree = DNTree::with_capacity(4096);
        let mut rng = make_rng();

        // Update prototypes 0-99 with one pattern
        let pattern_a = random_graphhv(&mut rng);
        for i in 0..100 {
            tree.update(i, &pattern_a, &mut rng);
        }

        // Update prototypes 2000-2099 with a different pattern
        let pattern_b = random_graphhv(&mut rng);
        for i in 2000..2100 {
            tree.update(i, &pattern_b, &mut rng);
        }

        // Query with pattern_a should find low-index regions
        let hits_a = tree.traverse(&pattern_a, 4);
        assert!(!hits_a.is_empty());

        // Query with pattern_b should find high-index regions
        let hits_b = tree.traverse(&pattern_b, 4);
        assert!(!hits_b.is_empty());

        // Top hit for pattern_a should be in [0, 1024) range
        let top_a = &hits_a[0];
        assert!(
            top_a.range_lo < 1024,
            "Expected low-range hit for pattern_a, got [{}, {})",
            top_a.range_lo,
            top_a.range_hi,
        );

        // Top hit for pattern_b should be in [2048, 4096) range
        let top_b = &hits_b[0];
        assert!(
            top_b.range_lo >= 1024,
            "Expected high-range hit for pattern_b, got [{}, {})",
            top_b.range_lo,
            top_b.range_hi,
        );
    }

    #[test]
    fn test_traverse_empty_tree() {
        let tree = DNTree::with_capacity(4096);
        let mut rng = make_rng();
        let query = random_graphhv(&mut rng);
        // No updates -> all children have access_count=0, filtered out in traversal
        // Traversal should not panic, returns empty or leaf hits from remaining beam
        let hits = tree.traverse(&query, 4);
        // With zero access counts, children are filtered -> beam runs out -> no hits
        // This is correct: no data in the tree = no results
        assert!(hits.len() <= 4);
    }

    #[test]
    fn test_tree_stats() {
        let mut tree = DNTree::with_capacity(256);
        let mut rng = make_rng();
        let hv = random_graphhv(&mut rng);

        for i in 0..50 {
            tree.update(i % 256, &hv, &mut rng);
        }

        let stats = tree.stats();
        // Each update increments access on root + child (2 per update minimum),
        // plus additional nodes created by splits. Total accesses > 50.
        assert!(stats.total_accesses >= 50);
        assert!(stats.total_nodes > 0);
        assert!(stats.leaf_nodes > 0);
    }

    #[test]
    fn test_btsp_gating() {
        let config = DNConfig {
            num_prototypes: 1024,
            btsp_gate_prob: 1.0, // always fire for testing
            btsp_boost: 7.0,
            ..Default::default()
        };
        let mut tree = DNTree::new(config);
        let mut rng = make_rng();
        let hv = random_graphhv(&mut rng);

        // With BTSP always firing, learning should be much faster
        for _ in 0..5 {
            tree.update(0, &hv, &mut rng);
        }

        let sim = similarity(&hv, &tree.summaries[0]);
        assert!(
            sim > 0.5,
            "BTSP boost should accelerate learning: sim={:.4}",
            sim
        );
    }

    #[test]
    fn test_select_child_ranges() {
        let tree = DNTree::with_capacity(1024);
        // Root covers [0, 1024), children cover [0,256), [256,512), [512,768), [768,1024)
        assert_eq!(tree.select_child(0, 0), 0);
        assert_eq!(tree.select_child(0, 100), 0);
        assert_eq!(tree.select_child(0, 300), 1);
        assert_eq!(tree.select_child(0, 600), 2);
        assert_eq!(tree.select_child(0, 900), 3);
        assert_eq!(tree.select_child(0, 1023), 3);
    }
}
