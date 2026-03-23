//! Parallel search combining HHTL progressive cascade with CLAM tree pruning.
//!
//! Dual-path search strategy:
//! - HHTL (Hierarchical Hash Table Lookup): palette-level progressive refinement
//! - CLAM: archetype tree pruning using precomputed distance matrices
//!
//! Results are merged and filtered through TruthGate for evidence quality.

use super::bgz17_bridge::PaletteEdge;
use super::palette_distance::SpoDistanceMatrices;
use super::layered_distance::{TruthGate, read_palette_edge, read_truth};

/// Search result with distance and truth metadata.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index of the matching node in the scope.
    pub node_idx: usize,
    /// SPO distance (sum of 3 plane distances from precomputed matrix).
    pub distance: u32,
    /// Truth frequency of the matching node.
    pub frequency: f32,
    /// Truth confidence of the matching node.
    pub confidence: f32,
}

impl SearchResult {
    /// Compute expectation from truth values.
    pub fn expectation(&self) -> f32 {
        TruthGate::expectation(self.frequency, self.confidence)
    }
}

/// Scope: search-ready palette data for a set of nodes.
///
/// Contains the containers (256 u64 words each), extracted palette edges,
/// and precomputed SPO distance matrices for O(1) distance lookups.
pub struct PaletteScope {
    /// Palette indices extracted from each container's W125.
    pub palette_indices: Vec<PaletteEdge>,
    /// Precomputed SPO distance matrices.
    pub distances: SpoDistanceMatrices,
    /// Raw containers (for reading truth values and other metadata).
    pub containers: Vec<[u64; 256]>,
}

impl PaletteScope {
    /// Build from containers: extract palette edges from W125 of each.
    pub fn from_containers(
        containers: Vec<[u64; 256]>,
        distances: SpoDistanceMatrices,
    ) -> Self {
        let palette_indices: Vec<PaletteEdge> = containers
            .iter()
            .map(read_palette_edge)
            .collect();
        PaletteScope {
            palette_indices,
            distances,
            containers,
        }
    }

    /// Number of nodes in this scope.
    pub fn len(&self) -> usize {
        self.containers.len()
    }

    /// Whether the scope is empty.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Compute palette distance between query and node at index.
    #[inline]
    fn distance_to(&self, query: &PaletteEdge, idx: usize) -> u32 {
        let c = &self.palette_indices[idx];
        self.distances.spo_distance(
            query.s_idx, query.p_idx, query.o_idx,
            c.s_idx, c.p_idx, c.o_idx,
        )
    }

    /// HHTL search: progressive refinement using palette distances.
    ///
    /// Scans all nodes, computing palette distance and keeping the top-k
    /// nearest results. This is the "brute force" path that benefits from
    /// the O(1) palette distance lookups (3 array loads per comparison).
    pub fn hhtl_search(&self, query: &PaletteEdge, k: usize) -> Vec<(usize, u32)> {
        if self.palette_indices.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use a max-heap approach: track the k nearest
        let mut results: Vec<(usize, u32)> = Vec::with_capacity(k + 1);
        let mut threshold = u32::MAX;

        for idx in 0..self.palette_indices.len() {
            let d = self.distance_to(query, idx);
            if d < threshold || results.len() < k {
                results.push((idx, d));
                results.sort_unstable_by_key(|&(_, dist)| dist);
                if results.len() > k {
                    results.truncate(k);
                }
                if results.len() == k {
                    threshold = results[k - 1].1;
                }
            }
        }

        results
    }

    /// CLAM-style search: archetype-based pruning.
    ///
    /// Partitions nodes into archetypes (clusters) and prunes distant
    /// clusters using triangle inequality on palette distances.
    /// Falls back to exhaustive scan for small scopes.
    pub fn clam_search(&self, query: &PaletteEdge, k: usize) -> Vec<(usize, u32)> {
        let n = self.palette_indices.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        // For small datasets, exhaustive scan is faster than tree overhead
        if n <= 64 {
            return self.hhtl_search(query, k);
        }

        // Build archetype clusters: pick sqrt(n) archetypes via farthest-first
        let n_archetypes = (n as f64).sqrt().ceil() as usize;
        let n_archetypes = n_archetypes.max(2).min(n);

        // Pick first archetype as node 0
        let mut archetype_indices: Vec<usize> = Vec::with_capacity(n_archetypes);
        archetype_indices.push(0);

        // Farthest-first selection
        let mut min_dists = vec![u32::MAX; n];
        for _ in 1..n_archetypes {
            let last = *archetype_indices.last().unwrap();
            let last_pe = &self.palette_indices[last];
            // Update min distances
            for i in 0..n {
                let d = self.distances.spo_distance(
                    last_pe.s_idx, last_pe.p_idx, last_pe.o_idx,
                    self.palette_indices[i].s_idx,
                    self.palette_indices[i].p_idx,
                    self.palette_indices[i].o_idx,
                );
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }
            // Pick the farthest node
            let mut best_idx = 0;
            let mut best_d = 0u32;
            for i in 0..n {
                if min_dists[i] > best_d {
                    best_d = min_dists[i];
                    best_idx = i;
                }
            }
            archetype_indices.push(best_idx);
        }

        // Assign each node to nearest archetype
        let mut assignments = vec![0usize; n];
        let mut archetype_radii = vec![0u32; n_archetypes];

        for i in 0..n {
            let pe_i = &self.palette_indices[i];
            let mut best_arch = 0;
            let mut best_d = u32::MAX;
            for (a, &arch_idx) in archetype_indices.iter().enumerate() {
                let arch_pe = &self.palette_indices[arch_idx];
                let d = self.distances.spo_distance(
                    pe_i.s_idx, pe_i.p_idx, pe_i.o_idx,
                    arch_pe.s_idx, arch_pe.p_idx, arch_pe.o_idx,
                );
                if d < best_d {
                    best_d = d;
                    best_arch = a;
                }
            }
            assignments[i] = best_arch;
            if best_d > archetype_radii[best_arch] {
                archetype_radii[best_arch] = best_d;
            }
        }

        // Compute query distance to each archetype
        let mut archetype_dists: Vec<(usize, u32)> = archetype_indices
            .iter()
            .enumerate()
            .map(|(a, &arch_idx)| {
                let d = self.distance_to(query, arch_idx);
                (a, d)
            })
            .collect();
        archetype_dists.sort_unstable_by_key(|&(_, d)| d);

        // Collect candidates from non-pruned clusters
        let mut results: Vec<(usize, u32)> = Vec::new();
        let mut current_threshold = u32::MAX;

        for &(arch_a, arch_d) in &archetype_dists {
            // Prune: if archetype distance - radius > current threshold, skip
            if results.len() >= k && arch_d > current_threshold + archetype_radii[arch_a] {
                continue;
            }

            // Scan all nodes in this cluster
            for i in 0..n {
                if assignments[i] != arch_a {
                    continue;
                }
                let d = self.distance_to(query, i);
                if d < current_threshold || results.len() < k {
                    results.push((i, d));
                    results.sort_unstable_by_key(|&(_, dist)| dist);
                    if results.len() > k {
                        results.truncate(k);
                    }
                    if results.len() == k {
                        current_threshold = results[k - 1].1;
                    }
                }
            }
        }

        results
    }
}

/// Parallel search: run HHTL + CLAM, merge, apply TruthGate.
///
/// Both search paths run independently and their results are merged
/// to produce the best top-k, filtered by truth-value evidence quality.
pub fn parallel_search(
    scope: &PaletteScope,
    query: &PaletteEdge,
    k: usize,
    gate: &TruthGate,
) -> Vec<SearchResult> {
    if scope.is_empty() || k == 0 {
        return Vec::new();
    }

    // Run both search paths
    let hhtl = scope.hhtl_search(query, k);
    let clam = scope.clam_search(query, k);

    // Merge results
    let merged = merge_and_rerank(hhtl, clam, k);

    // Apply TruthGate filter and build SearchResults
    let mut results = Vec::with_capacity(merged.len());
    for (idx, distance) in merged {
        let (frequency, confidence) = read_truth(&scope.containers[idx]);
        if gate.passes(frequency, confidence) {
            results.push(SearchResult {
                node_idx: idx,
                distance,
                frequency,
                confidence,
            });
        }
    }

    // Re-truncate after filtering (gate may have removed some)
    if results.len() > k {
        results.truncate(k);
    }

    results
}

/// Merge and re-rank two result sets, taking union of top-k.
///
/// Deduplicates by node index, keeping the minimum distance for each node.
fn merge_and_rerank(
    hhtl: Vec<(usize, u32)>,
    clam: Vec<(usize, u32)>,
    k: usize,
) -> Vec<(usize, u32)> {
    // Collect all results into a map (node_idx -> min_distance)
    let mut map = std::collections::HashMap::new();
    for (idx, d) in hhtl.into_iter().chain(clam.into_iter()) {
        let entry = map.entry(idx).or_insert(u32::MAX);
        if d < *entry {
            *entry = d;
        }
    }

    // Sort by distance
    let mut results: Vec<(usize, u32)> = map.into_iter().collect();
    results.sort_unstable_by_key(|&(_, d)| d);
    results.truncate(k);
    results
}

/// Compute local fractal dimension from palette distances.
///
/// LFD = log2(|B(center, radius)| / |B(center, radius/2)|)
/// where B(c, r) is the set of nodes within distance r of center.
pub fn lfd_from_palette(
    scope: &PaletteScope,
    center_idx: usize,
    radius: u32,
) -> f64 {
    if radius == 0 || scope.is_empty() {
        return 0.0;
    }

    let center = &scope.palette_indices[center_idx];
    let half_radius = radius / 2;

    let mut count_r = 0usize;
    let mut count_half_r = 0usize;

    for (i, pe) in scope.palette_indices.iter().enumerate() {
        if i == center_idx {
            count_r += 1;
            count_half_r += 1;
            continue;
        }
        let d = scope.distances.spo_distance(
            center.s_idx, center.p_idx, center.o_idx,
            pe.s_idx, pe.p_idx, pe.o_idx,
        );
        if d <= radius {
            count_r += 1;
        }
        if d <= half_radius {
            count_half_r += 1;
        }
    }

    if count_half_r == 0 || count_r <= count_half_r {
        0.0
    } else {
        (count_r as f64 / count_half_r as f64).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::bgz17_bridge::{Base17, PaletteEdge};
    use super::super::palette_distance::{Palette, SpoDistanceMatrices};
    use super::super::layered_distance::write_palette_edge;

    fn make_test_scope(n: usize) -> PaletteScope {
        let entries: Vec<Base17> = (0..32)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        let pal = Palette { entries };
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let mut containers: Vec<[u64; 256]> = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = [0u64; 256];
            let pe = PaletteEdge {
                s_idx: (i % 32) as u8,
                p_idx: ((i * 3) % 32) as u8,
                o_idx: ((i * 7) % 32) as u8,
            };
            write_palette_edge(&mut c, pe);
            // Write truth: freq=0.8, conf=0.9
            super::super::layered_distance::write_truth(&mut c, 0.8, 0.9);
            containers.push(c);
        }

        PaletteScope::from_containers(containers, dm)
    }

    #[test]
    fn test_hhtl_search_basic() {
        let scope = make_test_scope(100);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = scope.hhtl_search(&query, 5);
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_hhtl_search_empty() {
        let scope = make_test_scope(0);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = scope.hhtl_search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hhtl_search_k_larger_than_n() {
        let scope = make_test_scope(3);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = scope.hhtl_search(&query, 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_clam_search_basic() {
        let scope = make_test_scope(100);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = scope.clam_search(&query, 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_clam_search_small_fallback() {
        // Small scope should fallback to HHTL
        let scope = make_test_scope(10);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let hhtl = scope.hhtl_search(&query, 5);
        let clam = scope.clam_search(&query, 5);
        // Should return the same results for small scope
        assert_eq!(hhtl.len(), clam.len());
        for (h, c) in hhtl.iter().zip(clam.iter()) {
            assert_eq!(h.0, c.0);
            assert_eq!(h.1, c.1);
        }
    }

    #[test]
    fn test_parallel_search_basic() {
        let scope = make_test_scope(100);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = parallel_search(&scope, &query, 5, &TruthGate::OPEN);
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_parallel_search_with_truth_gate() {
        // Build scope with mixed truth values
        let entries: Vec<Base17> = (0..32)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        let pal = Palette { entries };
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let mut containers: Vec<[u64; 256]> = Vec::new();
        for i in 0..20 {
            let mut c = [0u64; 256];
            let pe = PaletteEdge {
                s_idx: (i % 32) as u8,
                p_idx: ((i * 3) % 32) as u8,
                o_idx: ((i * 7) % 32) as u8,
            };
            write_palette_edge(&mut c, pe);
            // Alternate high and low truth
            if i % 2 == 0 {
                super::super::layered_distance::write_truth(&mut c, 0.9, 0.9);
            } else {
                super::super::layered_distance::write_truth(&mut c, 0.3, 0.2);
            }
            containers.push(c);
        }
        let scope = PaletteScope::from_containers(containers, dm);

        // CERTAIN gate should filter out low-truth nodes
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let all = parallel_search(&scope, &query, 20, &TruthGate::OPEN);
        let certain = parallel_search(&scope, &query, 20, &TruthGate::CERTAIN);

        // Certain should have fewer results
        assert!(certain.len() <= all.len());

        // All results in certain should have high expectation
        for r in &certain {
            let exp = TruthGate::expectation(r.frequency, r.confidence);
            assert!(
                exp >= 0.9,
                "result expectation {} should be >= 0.9",
                exp
            );
        }
    }

    #[test]
    fn test_parallel_search_empty() {
        let scope = make_test_scope(0);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = parallel_search(&scope, &query, 5, &TruthGate::OPEN);
        assert!(results.is_empty());
    }

    #[test]
    fn test_merge_and_rerank_dedup() {
        let a = vec![(0, 10), (1, 20), (2, 30)];
        let b = vec![(0, 5), (3, 25), (2, 15)];
        let merged = merge_and_rerank(a, b, 4);
        // node 0 should have min distance 5
        assert_eq!(merged[0], (0, 5));
        // node 2 should have min distance 15
        let node2 = merged.iter().find(|&&(idx, _)| idx == 2).unwrap();
        assert_eq!(node2.1, 15);
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn test_merge_and_rerank_truncate() {
        let a = vec![(0, 10), (1, 20), (2, 30)];
        let b = vec![(3, 5), (4, 25)];
        let merged = merge_and_rerank(a, b, 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, 3); // distance 5
        assert_eq!(merged[1].0, 0); // distance 10
        assert_eq!(merged[2].0, 1); // distance 20
    }

    #[test]
    fn test_lfd_from_palette() {
        let scope = make_test_scope(100);
        // Use a large radius so most nodes are within
        let lfd = lfd_from_palette(&scope, 0, u32::MAX);
        // With all nodes within radius and most within half, LFD should be small
        assert!(lfd >= 0.0, "LFD should be non-negative, got {}", lfd);
    }

    #[test]
    fn test_lfd_zero_radius() {
        let scope = make_test_scope(100);
        let lfd = lfd_from_palette(&scope, 0, 0);
        assert_eq!(lfd, 0.0);
    }

    #[test]
    fn test_lfd_empty_scope() {
        let scope = make_test_scope(0);
        let lfd = lfd_from_palette(&scope, 0, 100);
        assert_eq!(lfd, 0.0);
    }

    #[test]
    fn test_search_result_expectation() {
        let sr = SearchResult {
            node_idx: 0,
            distance: 100,
            frequency: 0.8,
            confidence: 0.9,
        };
        let exp = sr.expectation();
        let expected = 0.9 * (0.8 - 0.5) + 0.5; // 0.77
        assert!((exp - expected).abs() < 1e-6);
    }

    #[test]
    fn test_scope_from_containers() {
        let scope = make_test_scope(50);
        assert_eq!(scope.len(), 50);
        assert!(!scope.is_empty());
        assert_eq!(scope.palette_indices.len(), 50);
        assert_eq!(scope.containers.len(), 50);
    }

    #[test]
    fn test_hhtl_finds_exact_match() {
        // Node 0 has palette edge (0, 0, 0), query is (0, 0, 0) => distance 0
        let scope = make_test_scope(100);
        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = scope.hhtl_search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0, "exact match should have distance 0");
    }
}
