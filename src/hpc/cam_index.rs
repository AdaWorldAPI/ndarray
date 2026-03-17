//! Content-Addressable Memory (CAM) index using multi-probe LSH.
//!
//! Maps 3-channel `GraphHV` vectors (3 × `Fingerprint<256>` = 49,152 bits) to
//! 64-bit hash signatures for O(log N) candidate retrieval, followed by exact
//! Hamming distance verification on the shortlist.
//!
//! ## Architecture
//!
//! ```text
//! GraphHV (49,152 bits)
//!     |
//!     +-- LSH Table 0: hash(hv, proj_0) -> sorted Vec<(hash, idx)>
//!     +-- LSH Table 1: hash(hv, proj_1) -> sorted Vec<(hash, idx)>
//!     +-- LSH Table 2: hash(hv, proj_2) -> sorted Vec<(hash, idx)>
//!     +-- LSH Table 3: hash(hv, proj_3) -> sorted Vec<(hash, idx)>
//!           |
//!           +-- Union of candidates -> Exact Hamming -> Top-K
//! ```
//!
//! ## Parameters
//!
//! | Parameter    | Sweet Spot | Reason                                |
//! |-------------|-----------|---------------------------------------|
//! | num_tables  | 4         | Balances recall vs query cost          |
//! | sample_size | 8         | Bits per hash projection (LSH quality) |
//! | window_size | 32        | Scan window around hash insertion point|

use super::fingerprint::Fingerprint;

/// Simple SplitMix64 RNG for deterministic projection generation.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// A 3-channel graph hypervector: Subject × Predicate × Object.
///
/// Each channel is a `Fingerprint<256>` (16,384 bits). The full vector
/// is 49,152 bits — enough to encode holographic triples with near-zero
/// collision probability.
#[derive(Clone, PartialEq, Eq)]
pub struct GraphHV {
    pub channels: [Fingerprint<256>; 3],
}

impl GraphHV {
    /// Create a zero-initialized GraphHV.
    pub fn zero() -> Self {
        Self {
            channels: [
                Fingerprint::<256>::zero(),
                Fingerprint::<256>::zero(),
                Fingerprint::<256>::zero(),
            ],
        }
    }

    /// Create a random GraphHV from a PRNG.
    pub fn random(rng: &mut SplitMix64) -> Self {
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
        Self { channels }
    }

    /// Total Hamming distance across all 3 channels.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.channels[0].hamming_distance(&other.channels[0])
            + self.channels[1].hamming_distance(&other.channels[1])
            + self.channels[2].hamming_distance(&other.channels[2])
    }
}

impl std::fmt::Debug for GraphHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphHV[S:{:?}, P:{:?}, O:{:?}]",
            self.channels[0], self.channels[1], self.channels[2]
        )
    }
}

/// Configuration for the CAM index.
#[derive(Clone, Debug)]
pub struct CamConfig {
    /// Number of independent LSH hash tables (more = better recall, higher memory).
    pub num_tables: usize,
    /// Number of input bits sampled per hash bit (XOR parity projection).
    pub sample_size: usize,
    /// Scan window size around hash insertion point during query.
    pub window_size: usize,
}

impl Default for CamConfig {
    fn default() -> Self {
        Self {
            num_tables: 4,
            sample_size: 8,
            window_size: 32,
        }
    }
}

/// A single LSH projector: maps a GraphHV to a 64-bit hash via XOR parity.
///
/// Each of 64 hash bits is the XOR-parity of `sample_size` randomly chosen
/// input bits from across all 3 channels. This provides locality-sensitive
/// hashing: similar inputs produce similar (low Hamming distance) hashes.
///
/// ## Precomputed Masks
///
/// Instead of storing scattered `(channel, word, bit)` tuples and doing random
/// lookups, we precompute 3 × `Fingerprint<256>` masks per hash bit. The hash
/// computation becomes contiguous AND + popcount per channel — no scattered loads.
struct LshProjector {
    // For each of 64 hash bits: 3 masks (one per channel).
    // masks[i][ch] has bits set at the positions sampled for hash bit i.
    masks: Vec<[Fingerprint<256>; 3]>,
}

impl LshProjector {
    fn new(rng: &mut SplitMix64, sample_size: usize) -> Self {
        let mut masks = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut ch_masks = [
                Fingerprint::<256>::zero(),
                Fingerprint::<256>::zero(),
                Fingerprint::<256>::zero(),
            ];
            for _ in 0..sample_size {
                let ch = (rng.next_u64() % 3) as usize;
                let word = (rng.next_u64() % 256) as usize;
                let bit = 1u64 << (rng.next_u64() % 64);
                ch_masks[ch].words[word] |= bit;
            }
            masks.push(ch_masks);
        }
        Self { masks }
    }

    #[inline]
    fn hash(&self, hv: &GraphHV) -> u64 {
        let mut code = 0u64;
        for (i, ch_masks) in self.masks.iter().enumerate() {
            // AND + popcount across all 3 channels — contiguous, no scattered loads.
            // Parity = (total matching bits) & 1.
            let mut parity = 0u32;
            for (ch_mask, channel) in ch_masks.iter().zip(hv.channels.iter()) {
                for (m, w) in ch_mask.words.iter().zip(channel.words.iter()) {
                    if *m != 0 {
                        parity += (m & w).count_ones();
                    }
                }
            }
            if parity & 1 != 0 {
                code |= 1u64 << i;
            }
        }
        code
    }
}

/// Query result from the CAM index.
#[derive(Clone, Debug)]
pub struct CamHit {
    /// Prototype index in the store.
    pub index: usize,
    /// Exact Hamming distance to the query.
    pub distance: u32,
}

/// Content-Addressable Memory index using multi-probe LSH.
///
/// Stores graph hypervectors and provides fast approximate nearest-neighbor
/// queries via locality-sensitive hashing with exact verification on shortlists.
///
/// Insert: O(L * log N) where L = num_tables
/// Query: O(L * W + C * D) where W = window_size, C = candidates, D = vector dim
pub struct CamIndex {
    prototypes: Vec<GraphHV>,
    /// Sorted (hash, proto_idx) per table for binary-search lookup.
    tables: Vec<Vec<(u64, usize)>>,
    projectors: Vec<LshProjector>,
    config: CamConfig,
}

impl CamIndex {
    /// Create a new empty CAM index.
    pub fn new(config: CamConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let projectors: Vec<_> = (0..config.num_tables)
            .map(|_| LshProjector::new(&mut rng, config.sample_size))
            .collect();
        let tables = vec![Vec::new(); config.num_tables];
        Self {
            prototypes: Vec::new(),
            tables,
            projectors,
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(seed: u64) -> Self {
        Self::new(CamConfig::default(), seed)
    }

    /// Number of stored prototypes.
    #[inline]
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Returns true if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }

    /// Insert a prototype. Returns its index.
    pub fn insert(&mut self, hv: GraphHV) -> usize {
        let idx = self.prototypes.len();
        for (t, proj) in self.projectors.iter().enumerate() {
            let hash = proj.hash(&hv);
            let table = &mut self.tables[t];
            let pos = table.partition_point(|&(h, _)| h < hash);
            table.insert(pos, (hash, idx));
        }
        self.prototypes.push(hv);
        idx
    }

    /// Query for the top-K most similar prototypes.
    ///
    /// Returns `CamHit` entries sorted by Hamming distance (ascending).
    /// Uses multi-probe LSH to collect candidates, then exact Hamming verification.
    pub fn query(&self, query: &GraphHV, top_k: usize) -> Vec<CamHit> {
        if self.prototypes.is_empty() {
            return Vec::new();
        }

        // Collect unique candidate indices from all hash tables
        let mut seen = vec![false; self.prototypes.len()];
        let mut candidates = Vec::new();

        for (t, proj) in self.projectors.iter().enumerate() {
            let query_hash = proj.hash(query);
            let table = &self.tables[t];

            // Binary search for insertion point, scan window around it
            let pos = table.partition_point(|&(h, _)| h < query_hash);
            let half_window = self.config.window_size / 2;
            let start = pos.saturating_sub(half_window);
            let end = (pos + half_window).min(table.len());

            for &(_, idx) in &table[start..end] {
                if !seen[idx] {
                    seen[idx] = true;
                    candidates.push(idx);
                }
            }
        }

        // Exact Hamming distance on candidate shortlist
        let mut results: Vec<CamHit> = candidates
            .iter()
            .map(|&idx| CamHit {
                index: idx,
                distance: query.hamming_distance(&self.prototypes[idx]),
            })
            .collect();

        results.sort_by_key(|h| h.distance);
        results.truncate(top_k);
        results
    }

    /// Get a reference to a stored prototype by index.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&GraphHV> {
        self.prototypes.get(idx)
    }

    /// Rebuild all hash tables from scratch.
    ///
    /// Call after bulk modifications to prototypes (if supported in the future).
    pub fn rebuild(&mut self) {
        for t in 0..self.config.num_tables {
            self.tables[t].clear();
            for (idx, hv) in self.prototypes.iter().enumerate() {
                let hash = self.projectors[t].hash(hv);
                self.tables[t].push((hash, idx));
            }
            self.tables[t].sort_by_key(|&(h, _)| h);
        }
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
    fn test_insert_and_len() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(123);
        assert!(cam.is_empty());

        for _ in 0..10 {
            cam.insert(GraphHV::random(&mut rng));
        }
        assert_eq!(cam.len(), 10);
    }

    #[test]
    fn test_query_exact_match() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(123);

        for _ in 0..50 {
            cam.insert(GraphHV::random(&mut rng));
        }

        let target = GraphHV::random(&mut rng);
        let target_idx = cam.insert(target.clone());

        let results = cam.query(&target, 5);
        assert!(!results.is_empty());
        assert!(
            results
                .iter()
                .any(|h| h.index == target_idx && h.distance == 0),
            "Exact match not found in top-5 results"
        );
    }

    #[test]
    fn test_query_top_k_ordered() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(456);

        for _ in 0..100 {
            cam.insert(GraphHV::random(&mut rng));
        }

        let query = GraphHV::random(&mut rng);
        let results = cam.query(&query, 10);

        for i in 1..results.len() {
            assert!(
                results[i].distance >= results[i - 1].distance,
                "Results not sorted: {} > {}",
                results[i - 1].distance,
                results[i].distance,
            );
        }
    }

    #[test]
    fn test_query_empty() {
        let cam = CamIndex::with_defaults(789);
        let mut rng = make_rng();
        let query = GraphHV::random(&mut rng);
        let results = cam.query(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_finds_similar() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(111);

        for _ in 0..200 {
            cam.insert(GraphHV::random(&mut rng));
        }

        // Create a prototype and a noisy copy (flip ~5% of bits)
        let original = GraphHV::random(&mut rng);
        let original_idx = cam.insert(original.clone());

        let mut noisy = original.clone();
        let mut flip_rng = SplitMix64::new(9999);
        for ch in 0..3 {
            for w in 0..256 {
                // Flip ~5% of bits: AND 4 randoms = ~6.25% kill rate
                let kill = flip_rng.next_u64()
                    & flip_rng.next_u64()
                    & flip_rng.next_u64()
                    & flip_rng.next_u64();
                noisy.channels[ch].words[w] ^= kill;
            }
        }

        let results = cam.query(&noisy, 10);
        let found = results.iter().any(|h| h.index == original_idx);
        assert!(
            found,
            "Similar prototype not found (may be LSH collision miss — non-deterministic)"
        );
    }

    #[test]
    fn test_rebuild_consistency() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(222);

        for _ in 0..50 {
            cam.insert(GraphHV::random(&mut rng));
        }

        let query = GraphHV::random(&mut rng);
        let before = cam.query(&query, 5);
        cam.rebuild();
        let after = cam.query(&query, 5);

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.index, a.index);
            assert_eq!(b.distance, a.distance);
        }
    }

    #[test]
    fn test_lsh_hash_deterministic() {
        let mut rng = make_rng();
        let cam = CamIndex::with_defaults(42);
        let hv = GraphHV::random(&mut rng);

        let h1 = cam.projectors[0].hash(&hv);
        let h2 = cam.projectors[0].hash(&hv);
        assert_eq!(h1, h2, "Same vector must produce same hash");

        let hv2 = GraphHV::random(&mut rng);
        let h3 = cam.projectors[0].hash(&hv2);
        assert_ne!(
            h1, h3,
            "Random vectors should produce different hashes (probabilistic)"
        );
    }

    #[test]
    fn test_get_prototype() {
        let mut rng = make_rng();
        let mut cam = CamIndex::with_defaults(333);
        let hv = GraphHV::random(&mut rng);
        let idx = cam.insert(hv.clone());

        let retrieved = cam.get(idx).unwrap();
        assert_eq!(*retrieved, hv);
        assert!(cam.get(idx + 1).is_none());
    }
}
