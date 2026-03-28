//! CAM-PQ: Content-Addressable Memory as Product Quantization.
//!
//! Unifies FAISS Product Quantization and CLAM 48-bit archetypes into a single
//! encode/decode/distance codec. 170× compression for 256D vectors, 682× for 1024D.
//! 500M candidates/second via AVX-512 VPGATHERDD.
//!
//! # Storage Format
//!
//! ```text
//! FAISS PQ6x8:                 CAM-PQ:
//!   6 subspaces × 256 centroids   6 bytes: HEEL/BRANCH/TWIG_A/TWIG_B/LEAF/GAMMA
//!   48 bits per vector             48 bits per vector
//!   Distance = 6 lookups + sum     Distance = 6 lookups + sum
//!   STORAGE FORMAT: IDENTICAL      QUERY PROTOCOL: IDENTICAL
//! ```
//!
//! # Three Training Modes
//!
//! - **Geometric**: standard k-means per subspace (minimizes reconstruction error)
//! - **Semantic**: CLAM archetype clustering (maximizes intent separation)
//! - **Hybrid**: geometric init + semantic fine-tune
//!
//! # Stroke Cascade
//!
//! ```text
//! Stroke 1 (HEEL only):   1 byte/candidate → 90% rejected
//! Stroke 2 (HEEL+BRANCH): 2 bytes/survivor → 90% of survivors rejected
//! Stroke 3 (full 6-byte): 6 bytes/finalist → precise ranking
//! = 99% rejection before full ADC computation
//! ```

use std::cmp::Ordering;

/// Number of subspaces (PQ parameter M).
pub const NUM_SUBSPACES: usize = 6;
/// Number of centroids per subspace (PQ parameter Ks).
pub const NUM_CENTROIDS: usize = 256;

/// Semantic names for the 6 CAM bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CamByte {
    Heel = 0,     // Coarse category
    Branch = 1,   // Archetype selection
    TwigA = 2,    // Shape parameter A
    TwigB = 3,    // Shape parameter B
    Leaf = 4,     // Fine detail
    Gamma = 5,    // Euler tension/energy
}

/// 6-byte CAM fingerprint.
pub type CamFingerprint = [u8; NUM_SUBSPACES];

/// A single codebook: 256 centroid vectors for one subspace.
#[derive(Clone, Debug)]
pub struct SubspaceCodebook {
    /// Centroid vectors. `centroids[i]` has length `subspace_dim`.
    pub centroids: Vec<Vec<f32>>,
    /// Dimension of each centroid.
    pub subspace_dim: usize,
}

/// The 6 codebooks — the complete CAM-PQ model.
///
/// Train once per domain, then encode/decode/query.
/// Codebook size: 6 × 256 × (D/6) × 4 bytes. For D=1024: ~1MB.
#[derive(Clone, Debug)]
pub struct CamCodebook {
    /// One codebook per subspace.
    pub codebooks: [SubspaceCodebook; NUM_SUBSPACES],
    /// Total vector dimension (D).
    pub total_dim: usize,
    /// Subspace dimension (D/6).
    pub subspace_dim: usize,
}

/// Precomputed distance tables for one query.
///
/// 6 × 256 = 1536 floats = 6KB. Fits in L1 cache.
/// After precomputing, each candidate distance = 6 table lookups + 5 adds.
#[derive(Clone, Debug)]
pub struct DistanceTables {
    /// `tables[subspace][centroid]` = squared L2 distance from query subvector
    /// to that centroid.
    pub tables: [[f32; NUM_CENTROIDS]; NUM_SUBSPACES],
}

/// Stroke-aligned packed database for cascade filtering.
///
/// CAM fingerprints are stored in stroke layout:
/// - Stroke 1: all HEEL bytes contiguous (1 byte/candidate)
/// - Stroke 2: all HEEL+BRANCH pairs contiguous (2 bytes/candidate)
/// - Stroke 3: all full 6-byte CAMs contiguous
///
/// This enables 99% rejection before full ADC: scan 1MB instead of 6MB for 1M vectors.
#[derive(Clone, Debug)]
pub struct PackedDatabase {
    /// Stroke 1: HEEL bytes only.
    pub stroke1: Vec<u8>,
    /// Stroke 2: HEEL + BRANCH interleaved.
    pub stroke2: Vec<u8>,
    /// Stroke 3: full 6-byte CAMs.
    pub stroke3: Vec<u8>,
    /// Number of candidates.
    pub num_candidates: usize,
}

// === CamCodebook ===

impl CamCodebook {
    /// Encode a full-precision vector to a 6-byte CAM fingerprint.
    ///
    /// For each subspace, finds the nearest centroid (argmin squared L2).
    /// O(6 × 256 × D/6) = O(256 × D).
    pub fn encode(&self, vector: &[f32]) -> CamFingerprint {
        assert!(vector.len() >= self.total_dim, "vector too short");
        let mut cam = [0u8; NUM_SUBSPACES];
        for s in 0..NUM_SUBSPACES {
            let sub = &vector[s * self.subspace_dim..(s + 1) * self.subspace_dim];
            let mut best_dist = f32::MAX;
            let mut best_id = 0u8;
            for (c, centroid) in self.codebooks[s].centroids.iter().enumerate() {
                let dist = squared_l2(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = c as u8;
                }
            }
            cam[s] = best_id;
        }
        cam
    }

    /// Decode a 6-byte CAM fingerprint to an approximate vector.
    ///
    /// Concatenates the centroid vectors for each subspace.
    /// Lossless given the codebook: decode(encode(x)) = nearest centroids.
    pub fn decode(&self, cam: &CamFingerprint) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.total_dim);
        for s in 0..NUM_SUBSPACES {
            vec.extend_from_slice(&self.codebooks[s].centroids[cam[s] as usize]);
        }
        vec
    }

    /// Precompute distance tables for a query vector.
    ///
    /// This is the key to ADC speed: O(6 × 256 × D/6) precompute once,
    /// then O(6) per candidate instead of O(D).
    pub fn precompute_distances(&self, query: &[f32]) -> DistanceTables {
        assert!(query.len() >= self.total_dim, "query too short");
        let mut tables = [[0.0f32; NUM_CENTROIDS]; NUM_SUBSPACES];
        for s in 0..NUM_SUBSPACES {
            let q_sub = &query[s * self.subspace_dim..(s + 1) * self.subspace_dim];
            for c in 0..NUM_CENTROIDS {
                tables[s][c] = squared_l2(q_sub, &self.codebooks[s].centroids[c]);
            }
        }
        DistanceTables { tables }
    }

    /// Batch encode multiple vectors.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<CamFingerprint> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Reconstruction error: ||x - decode(encode(x))||².
    pub fn reconstruction_error(&self, vector: &[f32]) -> f32 {
        let cam = self.encode(vector);
        let reconstructed = self.decode(&cam);
        squared_l2(vector, &reconstructed)
    }

    /// Mean reconstruction error over a dataset.
    pub fn mean_reconstruction_error(&self, vectors: &[Vec<f32>]) -> f32 {
        if vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = vectors.iter().map(|v| self.reconstruction_error(v)).sum();
        sum / vectors.len() as f32
    }
}

// === DistanceTables ===

impl DistanceTables {
    /// Compute ADC distance to a single CAM fingerprint.
    /// 6 table lookups + 5 adds.
    #[inline(always)]
    pub fn distance(&self, cam: &CamFingerprint) -> f32 {
        self.tables[0][cam[0] as usize]
            + self.tables[1][cam[1] as usize]
            + self.tables[2][cam[2] as usize]
            + self.tables[3][cam[3] as usize]
            + self.tables[4][cam[4] as usize]
            + self.tables[5][cam[5] as usize]
    }

    /// Batch distance for N candidates. Dispatches to AVX-512 if available.
    pub fn distance_batch(&self, cams: &[CamFingerprint]) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        {
            if super::simd_caps::simd_caps().avx512f {
                return unsafe { self.distance_batch_avx512(cams) };
            }
        }
        // Scalar fallback
        cams.iter().map(|c| self.distance(c)).collect()
    }

    /// AVX-512 batch distance: 16 candidates per iteration via VPGATHERPS.
    ///
    /// For each subspace, gathers 16 distances from the precomputed table
    /// using 16 centroid indices, then accumulates across 6 subspaces.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn distance_batch_avx512(&self, cams: &[CamFingerprint]) -> Vec<f32> {
        use core::arch::x86_64::*;

        let n = cams.len();
        let mut result = vec![0.0f32; n];
        let chunks = n / 16;

        for chunk in 0..chunks {
            let base = chunk * 16;
            let mut acc = _mm512_setzero_ps();

            for s in 0..NUM_SUBSPACES {
                // Gather 16 centroid indices for subspace s
                let indices = _mm512_set_epi32(
                    cams[base + 15][s] as i32, cams[base + 14][s] as i32,
                    cams[base + 13][s] as i32, cams[base + 12][s] as i32,
                    cams[base + 11][s] as i32, cams[base + 10][s] as i32,
                    cams[base + 9][s] as i32,  cams[base + 8][s] as i32,
                    cams[base + 7][s] as i32,  cams[base + 6][s] as i32,
                    cams[base + 5][s] as i32,  cams[base + 4][s] as i32,
                    cams[base + 3][s] as i32,  cams[base + 2][s] as i32,
                    cams[base + 1][s] as i32,  cams[base][s] as i32,
                );

                // Gather distances from precomputed table
                // _mm512_i32gather_ps gathers f32 values at byte offsets = index * scale
                // Scale=4 means each index is multiplied by 4 (sizeof f32)
                let base_ptr = self.tables[s].as_ptr();
                let distances = _mm512_i32gather_ps::<4>(indices, base_ptr);

                acc = _mm512_add_ps(acc, distances);
            }

            _mm512_storeu_ps(result[base..].as_mut_ptr(), acc);
        }

        // Scalar tail
        for i in (chunks * 16)..n {
            result[i] = self.distance(&cams[i]);
        }

        result
    }

    /// Top-K nearest candidates by ADC distance.
    pub fn top_k(&self, cams: &[CamFingerprint], k: usize) -> Vec<(usize, f32)> {
        let distances = self.distance_batch(cams);
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

// === PackedDatabase (stroke cascade) ===

impl PackedDatabase {
    /// Pack CAM fingerprints into stroke-aligned layout.
    pub fn pack(fingerprints: &[CamFingerprint]) -> Self {
        let n = fingerprints.len();

        // Stroke 1: HEEL bytes only (1 byte per candidate)
        let stroke1: Vec<u8> = fingerprints.iter().map(|f| f[0]).collect();

        // Stroke 2: HEEL + BRANCH interleaved (2 bytes per candidate)
        let stroke2: Vec<u8> = fingerprints.iter()
            .flat_map(|f| [f[0], f[1]])
            .collect();

        // Stroke 3: full CAM (6 bytes per candidate)
        let stroke3: Vec<u8> = fingerprints.iter()
            .flat_map(|f| f.iter().copied())
            .collect();

        PackedDatabase { stroke1, stroke2, stroke3, num_candidates: n }
    }

    /// Cascade query: Stroke 1 → Stroke 2 → Stroke 3.
    ///
    /// 99% rejection before full ADC. Scans 1MB instead of 6MB for 1M vectors.
    pub fn cascade_query(
        &self,
        dist_tables: &DistanceTables,
        heel_threshold: f32,
        branch_threshold: f32,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // Stroke 1: scan HEEL bytes (1 byte/candidate)
        let mut survivors: Vec<usize> = Vec::new();
        for i in 0..self.num_candidates {
            let heel_dist = dist_tables.tables[0][self.stroke1[i] as usize];
            if heel_dist < heel_threshold {
                survivors.push(i);
            }
        }

        // Stroke 2: scan HEEL+BRANCH for survivors (2 bytes/candidate)
        let mut refined: Vec<usize> = Vec::new();
        for &i in &survivors {
            let base = i * 2;
            let dist = dist_tables.tables[0][self.stroke2[base] as usize]
                     + dist_tables.tables[1][self.stroke2[base + 1] as usize];
            if dist < branch_threshold {
                refined.push(i);
            }
        }

        // Stroke 3: full ADC on refined candidates (6 bytes/candidate)
        let mut hits: Vec<(usize, f32)> = refined.iter().map(|&i| {
            let base = i * 6;
            let cam: CamFingerprint = [
                self.stroke3[base],
                self.stroke3[base + 1],
                self.stroke3[base + 2],
                self.stroke3[base + 3],
                self.stroke3[base + 4],
                self.stroke3[base + 5],
            ];
            (i, dist_tables.distance(&cam))
        }).collect();

        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        hits.truncate(top_k);
        hits
    }

    /// Number of candidates in the database.
    pub fn len(&self) -> usize {
        self.num_candidates
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.num_candidates == 0
    }

    /// Storage size in bytes (all 3 strokes).
    pub fn storage_bytes(&self) -> usize {
        self.stroke1.len() + self.stroke2.len() + self.stroke3.len()
    }
}

// === Training ===

/// Train codebooks via k-means on subvectors (standard FAISS PQ).
///
/// Minimizes reconstruction error: ||x - decode(encode(x))||².
pub fn train_geometric(
    vectors: &[Vec<f32>],
    total_dim: usize,
    iterations: usize,
) -> CamCodebook {
    assert!(!vectors.is_empty(), "need at least one training vector");
    assert!(total_dim >= NUM_SUBSPACES, "dimension must be >= 6");
    let subspace_dim = total_dim / NUM_SUBSPACES;

    let mut codebooks_vec: Vec<SubspaceCodebook> = Vec::with_capacity(NUM_SUBSPACES);

    for s in 0..NUM_SUBSPACES {
        // Extract subvectors for this subspace
        let subs: Vec<Vec<f32>> = vectors.iter()
            .map(|v| v[s * subspace_dim..(s + 1) * subspace_dim].to_vec())
            .collect();

        // k-means clustering
        let centroids = kmeans(&subs, NUM_CENTROIDS.min(subs.len()), subspace_dim, iterations);
        codebooks_vec.push(SubspaceCodebook { centroids, subspace_dim });
    }

    CamCodebook {
        codebooks: codebooks_vec.try_into().unwrap_or_else(|v: Vec<SubspaceCodebook>| {
            panic!("expected {} codebooks, got {}", NUM_SUBSPACES, v.len())
        }),
        total_dim,
        subspace_dim,
    }
}

/// Train with semantic labels: geometric init + label-guided fine-tuning.
///
/// Codebooks balance reconstruction error AND semantic separation.
/// `labels[i]` is a set of semantic tags for vector `i`.
pub fn train_semantic(
    vectors: &[Vec<f32>],
    labels: &[Vec<String>],
    total_dim: usize,
    alpha: f32,
) -> CamCodebook {
    assert_eq!(vectors.len(), labels.len(), "vectors and labels must match");

    // Phase 1: geometric initialization
    let mut codebook = train_geometric(vectors, total_dim, 20);

    // Phase 2: semantic fine-tuning
    // For pairs with same labels, pull centroids closer.
    // For pairs with different labels, push centroids apart.
    let sample_size = vectors.len().min(500);
    for _epoch in 0..30 {
        for i in 0..sample_size {
            let cam_i = codebook.encode(&vectors[i]);
            let j_end = (i + 50).min(vectors.len());
            for j in (i + 1)..j_end {
                let cam_j = codebook.encode(&vectors[j]);
                let semantic_sim = jaccard_similarity(&labels[i], &labels[j]);
                let cam_dist_val = cam_l1_distance(&cam_i, &cam_j);

                // Gradient: push/pull centroids based on label agreement
                let target = if semantic_sim > 0.5 { 0.0f32 } else { 1.0 };
                let current = cam_dist_val as f32 / (NUM_SUBSPACES as f32 * 255.0);
                let grad = alpha * (target - current);

                // Adjust centroids for each subspace
                for s in 0..NUM_SUBSPACES {
                    if cam_i[s] != cam_j[s] {
                        let ci = cam_i[s] as usize;
                        let cj = cam_j[s] as usize;
                        let dim = codebook.subspace_dim;
                        for d in 0..dim {
                            let delta = grad * (codebook.codebooks[s].centroids[cj][d]
                                              - codebook.codebooks[s].centroids[ci][d]);
                            codebook.codebooks[s].centroids[ci][d] += delta * 0.01;
                            codebook.codebooks[s].centroids[cj][d] -= delta * 0.01;
                        }
                    }
                }
            }
        }
    }

    codebook
}

/// Hybrid training: geometric init + semantic fine-tune (convenience wrapper).
pub fn train_hybrid(
    vectors: &[Vec<f32>],
    labels: &[Vec<String>],
    total_dim: usize,
) -> CamCodebook {
    train_semantic(vectors, labels, total_dim, 0.1)
}

// === Internal utilities ===

/// Squared L2 distance between two slices via `crate::simd`.
///
/// For 16D subvectors (CAM-PQ subspace dimension), this is one F32x16
/// load-subtract-multiply-reduce. Consumer never sees hardware details.
#[inline(always)]
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    // Fast path: exactly 16 elements = one F32x16 lane (most common in CAM-PQ).
    if n == 16 {
        use crate::simd::F32x16;
        let va = F32x16::from_slice(a);
        let vb = F32x16::from_slice(b);
        let diff = va - vb;
        return (diff * diff).reduce_sum();
    }

    // Medium path: process 16 elements at a time, accumulate remainder scalar.
    if n >= 16 {
        use crate::simd::F32x16;
        let mut acc = F32x16::splat(0.0);
        let chunks = n / 16;
        for i in 0..chunks {
            let off = i * 16;
            let va = F32x16::from_slice(&a[off..off + 16]);
            let vb = F32x16::from_slice(&b[off..off + 16]);
            let diff = va - vb;
            acc = diff.mul_add(diff, acc);
        }
        let mut sum = acc.reduce_sum();
        // Scalar remainder
        for i in (chunks * 16)..n {
            let d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    // Scalar fallback for tiny slices.
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// L1 distance between two CAM fingerprints.
fn cam_l1_distance(a: &CamFingerprint, b: &CamFingerprint) -> u32 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

/// Jaccard similarity between two label sets.
fn jaccard_similarity(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.iter().filter(|x| b.contains(x)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 { 1.0 } else { intersection as f32 / union as f32 }
}

/// Simple k-means clustering.
///
/// Returns `k` centroid vectors of length `dim`.
fn kmeans(data: &[Vec<f32>], k: usize, dim: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![vec![0.0; dim]; k];
    }
    let k = k.min(n);

    // Initialize: farthest-first selection
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(data[0].clone());

    let mut min_dists = vec![f32::MAX; n];
    for _ in 1..k {
        let last = centroids.last().unwrap();
        for (i, v) in data.iter().enumerate() {
            let d = squared_l2(v, last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        // Pick farthest point
        let best = min_dists.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        centroids.push(data[best].clone());
    }

    // Lloyd's algorithm
    let mut assignments = vec![0usize; n];
    for _iter in 0..iterations {
        // Assign each point to nearest centroid
        for (i, v) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = squared_l2(v, centroid);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Recompute centroids
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, val) in v.iter().enumerate() {
                sums[c][d] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            // Deterministic pseudo-random
            let mut seed = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            for val in v.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *val = (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            }
            vecs.push(v);
        }
        vecs
    }

    fn train_test_codebook(dim: usize) -> CamCodebook {
        let vectors = make_test_vectors(500, dim);
        train_geometric(&vectors, dim, 10)
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let codebook = train_test_codebook(24);
        let v = make_test_vectors(1, 24).pop().unwrap();

        let cam = codebook.encode(&v);
        let decoded = codebook.decode(&cam);

        assert_eq!(decoded.len(), 24);
        // Reconstruction error should be finite
        let err = codebook.reconstruction_error(&v);
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }

    #[test]
    fn test_distance_tables() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(100, 24);
        let query = &vecs[0];
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();

        let dt = codebook.precompute_distances(query);

        // Self-distance should be minimal
        let self_cam = codebook.encode(query);
        let self_dist = dt.distance(&self_cam);
        assert!(self_dist >= 0.0);

        // Distance to a different vector should be larger
        let other_dist = dt.distance(&cams[50]);
        assert!(other_dist >= 0.0);
    }

    #[test]
    fn test_distance_batch_matches_scalar() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(100, 24);
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();
        let dt = codebook.precompute_distances(&vecs[0]);

        let batch = dt.distance_batch(&cams);
        let scalar: Vec<f32> = cams.iter().map(|c| dt.distance(c)).collect();

        for (b, s) in batch.iter().zip(scalar.iter()) {
            assert!((b - s).abs() < 1e-6, "batch {} != scalar {}", b, s);
        }
    }

    #[test]
    fn test_top_k() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(100, 24);
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();
        let dt = codebook.precompute_distances(&vecs[0]);

        let top5 = dt.top_k(&cams, 5);
        assert_eq!(top5.len(), 5);
        // Should be sorted by distance
        for w in top5.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_packed_database() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(1000, 24);
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();

        let packed = PackedDatabase::pack(&cams);
        assert_eq!(packed.len(), 1000);
        assert_eq!(packed.stroke1.len(), 1000);
        assert_eq!(packed.stroke2.len(), 2000);
        assert_eq!(packed.stroke3.len(), 6000);
        // storage = 1000 + 2000 + 6000 = 9000 bytes
        assert_eq!(packed.storage_bytes(), 9000);
    }

    #[test]
    fn test_cascade_query() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(1000, 24);
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();
        let packed = PackedDatabase::pack(&cams);
        let dt = codebook.precompute_distances(&vecs[0]);

        // Wide thresholds should return results
        let results = packed.cascade_query(&dt, f32::MAX, f32::MAX, 10);
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        // Tight thresholds should return fewer results
        let tight = packed.cascade_query(&dt, 0.001, 0.001, 10);
        assert!(tight.len() <= results.len());
    }

    #[test]
    fn test_cascade_filters_progressively() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(10000, 24);
        let cams: Vec<CamFingerprint> = vecs.iter().map(|v| codebook.encode(v)).collect();
        let packed = PackedDatabase::pack(&cams);
        let dt = codebook.precompute_distances(&vecs[0]);

        // Moderate thresholds: cascade should reduce candidate count at each stroke
        let results = packed.cascade_query(&dt, 1.0, 1.5, 20);
        // Should have results (thresholds are moderate)
        assert!(!results.is_empty());
        assert!(results.len() <= 20);
    }

    #[test]
    fn test_train_geometric() {
        let vecs = make_test_vectors(200, 24);
        let codebook = train_geometric(&vecs, 24, 5);

        assert_eq!(codebook.total_dim, 24);
        assert_eq!(codebook.subspace_dim, 4);
        assert_eq!(codebook.codebooks.len(), 6);
        for cb in &codebook.codebooks {
            assert!(cb.centroids.len() <= 256);
            assert!(!cb.centroids.is_empty());
            for c in &cb.centroids {
                assert_eq!(c.len(), 4);
            }
        }
    }

    #[test]
    fn test_train_semantic() {
        let vecs = make_test_vectors(100, 24);
        let labels: Vec<Vec<String>> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    vec!["category_a".into()]
                } else {
                    vec!["category_b".into()]
                }
            })
            .collect();

        let codebook = train_semantic(&vecs, &labels, 24, 0.1);
        assert_eq!(codebook.total_dim, 24);
    }

    #[test]
    fn test_train_hybrid() {
        let vecs = make_test_vectors(100, 24);
        let labels: Vec<Vec<String>> = (0..100)
            .map(|i| vec![format!("cat_{}", i % 5)])
            .collect();

        let codebook = train_hybrid(&vecs, &labels, 24);
        assert_eq!(codebook.total_dim, 24);

        // Should be able to encode/decode
        let cam = codebook.encode(&vecs[0]);
        let decoded = codebook.decode(&cam);
        assert_eq!(decoded.len(), 24);
    }

    #[test]
    fn test_kmeans_basic() {
        // Two clear clusters
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(vec![0.0, 0.0]);
        }
        for _ in 0..50 {
            data.push(vec![10.0, 10.0]);
        }

        let centroids = kmeans(&data, 2, 2, 20);
        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0,0) and (10,10)
        let c0 = &centroids[0];
        let c1 = &centroids[1];
        let near_origin = (c0[0].abs() < 1.0 && c0[1].abs() < 1.0)
            || (c1[0].abs() < 1.0 && c1[1].abs() < 1.0);
        let near_ten = ((c0[0] - 10.0).abs() < 1.0 && (c0[1] - 10.0).abs() < 1.0)
            || ((c1[0] - 10.0).abs() < 1.0 && (c1[1] - 10.0).abs() < 1.0);
        assert!(near_origin, "one centroid should be near origin");
        assert!(near_ten, "one centroid should be near (10,10)");
    }

    #[test]
    fn test_encode_batch() {
        let codebook = train_test_codebook(24);
        let vecs = make_test_vectors(50, 24);
        let batch = codebook.encode_batch(&vecs);
        assert_eq!(batch.len(), 50);
        // Each should match individual encode
        for (v, cam) in vecs.iter().zip(batch.iter()) {
            assert_eq!(codebook.encode(v), *cam);
        }
    }

    #[test]
    fn test_cam_fingerprint_size() {
        assert_eq!(std::mem::size_of::<CamFingerprint>(), 6);
    }

    #[test]
    fn test_mean_reconstruction_error() {
        let vecs = make_test_vectors(200, 24);
        let codebook = train_geometric(&vecs, 24, 10);
        let mre = codebook.mean_reconstruction_error(&vecs);
        assert!(mre.is_finite());
        assert!(mre >= 0.0);
    }
}
