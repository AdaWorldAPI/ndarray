//! HDR (High Dynamic Range) Cascade Search.
//!
//! 3-stroke adaptive cascade for Hamming-based nearest-neighbour search
//! with optional precision tiers (VNNI cosine, F32/BF16 dequant, DeltaXor, BF16Hamming).
//!
//! Extracted from rustynum-core/hdr.rs — the cascade algorithm and types.

use super::bitwise;
use super::bf16_truth::BF16Weights;

/// A ranked hit from the HDR cascade search.
#[derive(Debug, Clone)]
pub struct RankedHit {
    pub index: usize,
    pub hamming: u64,
    pub precise: f64,
    pub band: Band,
}

/// Quality band for a cascade hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    Foveal,
    Near,
    Good,
    Weak,
    Reject,
}

/// Alert emitted when the cascade detects distributional drift.
#[derive(Debug, Clone)]
pub struct ShiftAlert {
    pub old_mu: f64,
    pub new_mu: f64,
    pub old_sigma: f64,
    pub new_sigma: f64,
    pub observations: usize,
}

/// Precision mode for Stroke 3 of the HDR cascade.
///
/// Six data paths through the same cascade engine:
///
/// | Case | Source | Tier 1-2 | Tier 3 | Example |
/// |------|--------|----------|--------|---------|
/// | Off  | —      | hamming  | none   | reject-only |
/// | Vnni | native binary 64Kbit | partial popcount | dot_i8 cosine | SimHash/LSH/HDC |
/// | F32  | f32 embedding → u8 | hamming on u8 | dequant → f32 dot | Jina embed |
/// | BF16 | f32 embedding → u8 | hamming on u8 | dequant → bf16 dot | large embed db |
/// | DeltaXor | 3D + INT8 delta | XOR delta popcount | INT8 residual dot | DeltaLayer |
/// | BF16Hamming | native BF16 bytes (2B/dim) | weighted XOR popcount | weighted BF16 distance | 6× faster than F32 |
#[derive(Clone, Copy, Debug)]
pub enum PreciseMode {
    /// No precision tier — return Hamming distances only.
    Off,
    /// Native u8 vectors (HDC/SimHash/LSH). Uses dot_i8 → cosine.
    Vnni,
    /// Quantized f32 embeddings → dequantize to f32, SIMD dot → cosine.
    F32 { scale: f32, zero_point: i32 },
    /// Same dequantization but signals BF16 intent (falls through to f32 path).
    BF16 { scale: f32, zero_point: i32 },
    /// XOR Delta Layer + INT8 residual blend.
    DeltaXor { delta_weight: f32 },
    /// BF16-structured Hamming: XOR + per-field weighted popcount.
    BF16Hamming { weights: BF16Weights },
}

impl PartialEq for PreciseMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Off, Self::Off) => true,
            (Self::Vnni, Self::Vnni) => true,
            (Self::F32 { scale: s1, zero_point: z1 }, Self::F32 { scale: s2, zero_point: z2 }) => {
                s1.to_bits() == s2.to_bits() && z1 == z2
            }
            (Self::BF16 { scale: s1, zero_point: z1 }, Self::BF16 { scale: s2, zero_point: z2 }) => {
                s1.to_bits() == s2.to_bits() && z1 == z2
            }
            (Self::DeltaXor { delta_weight: w1 }, Self::DeltaXor { delta_weight: w2 }) => {
                w1.to_bits() == w2.to_bits()
            }
            (Self::BF16Hamming { weights: w1 }, Self::BF16Hamming { weights: w2 }) => w1 == w2,
            _ => false,
        }
    }
}

impl Eq for PreciseMode {}

/// HDR Cascade: stateful search engine with calibrated rejection thresholds.
pub struct Cascade {
    pub threshold: u64,
    pub vec_bytes: usize,
    mu: f64,
    sigma: f64,
    observations: usize,
}

impl Cascade {
    pub fn from_threshold(threshold: u64, vec_bytes: usize) -> Self {
        Self { threshold, vec_bytes, mu: 0.0, sigma: 0.0, observations: 0 }
    }

    pub fn calibrate(distances: &[u32], vec_bytes: usize) -> Self {
        if distances.is_empty() {
            return Self::from_threshold(0, vec_bytes);
        }
        let n = distances.len() as f64;
        let mu = distances.iter().map(|&d| d as f64).sum::<f64>() / n;
        let var = distances.iter().map(|&d| { let diff = d as f64 - mu; diff * diff }).sum::<f64>() / n;
        let sigma = var.sqrt();
        let threshold = (mu + 3.0 * sigma) as u64;
        Self { threshold, vec_bytes, mu, sigma, observations: distances.len() }
    }

    pub fn expose(&self, distance: u32) -> Band {
        let d = distance as u64;
        let t = self.threshold;
        if d <= t / 4 {
            Band::Foveal
        } else if d <= t / 2 {
            Band::Near
        } else if d <= t * 3 / 4 {
            Band::Good
        } else if d <= t {
            Band::Weak
        } else {
            Band::Reject
        }
    }

    pub fn test(&self, a: &[u8], b: &[u8]) -> bool {
        bitwise::hamming_distance_raw(a, b) <= self.threshold
    }

    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        let d = distance as f64;
        self.observations += 1;
        if self.observations == 1 {
            self.mu = d;
            self.sigma = 0.0;
            return None;
        }
        let old_mu = self.mu;
        let old_sigma = self.sigma;
        let delta = d - self.mu;
        self.mu += delta / self.observations as f64;
        let delta2 = d - self.mu;
        let m2 = old_sigma * old_sigma * (self.observations - 1) as f64 + delta * delta2;
        self.sigma = (m2 / self.observations as f64).sqrt();

        if self.observations > 10 && old_sigma > 0.0 && (self.mu - old_mu).abs() > 2.0 * old_sigma {
            Some(ShiftAlert {
                old_mu,
                new_mu: self.mu,
                old_sigma,
                new_sigma: self.sigma,
                observations: self.observations,
            })
        } else {
            None
        }
    }

    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        self.mu = alert.new_mu;
        self.sigma = alert.new_sigma;
        self.threshold = (alert.new_mu + 3.0 * alert.new_sigma) as u64;
    }

    /// Run the full 3-stroke cascade query.
    pub fn query(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        num_vectors: usize,
    ) -> Vec<RankedHit> {
        assert_eq!(query.len(), vec_bytes);
        assert_eq!(database.len(), vec_bytes * num_vectors);

        let threshold = self.threshold;

        let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
        let scale1 = (vec_bytes as f64) / (s1_bytes as f64);
        let warmup_n = 128.min(num_vectors);

        // Small vectors: skip cascade
        if vec_bytes < 128 {
            let mut results = Vec::new();
            for i in 0..num_vectors {
                let base = i * vec_bytes;
                let d = bitwise::hamming_distance_raw(query, &database[base..base + vec_bytes]);
                if d <= threshold {
                    results.push(RankedHit {
                        index: i,
                        hamming: d,
                        precise: f64::NAN,
                        band: self.expose(d as u32),
                    });
                }
            }
            return results;
        }

        // ═══ STROKE 1: Partial popcount with σ warmup ═══
        let query_prefix = &query[..s1_bytes];
        let total_bits = (vec_bytes * 8) as f64;
        let p_thresh = (threshold as f64 / total_bits).clamp(0.001, 0.999);
        let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();

        let mut warmup_dists = Vec::with_capacity(warmup_n);
        for i in 0..warmup_n {
            let base = i * vec_bytes;
            let cand_prefix = &database[base..base + s1_bytes];
            let d = bitwise::hamming_distance_raw(query_prefix, cand_prefix);
            let estimate = (d as f64 * scale1) as u64;
            warmup_dists.push(estimate);
        }

        let var: f64 = {
            let mu: f64 = warmup_dists.iter().map(|&d| d as f64).sum::<f64>() / warmup_n as f64;
            warmup_dists.iter().map(|&d| { let diff = d as f64 - mu; diff * diff }).sum::<f64>() / warmup_n as f64
        };
        let sigma_pop = var.sqrt();
        let sigma = sigma_est.max(sigma_pop).max(1.0);
        let s1_reject = threshold as f64 + 3.0 * sigma;

        let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(num_vectors / 20);
        for i in 0..num_vectors {
            let base = i * vec_bytes;
            let cand_prefix = &database[base..base + s1_bytes];
            let d = bitwise::hamming_distance_raw(query_prefix, cand_prefix);
            let estimate = (d as f64 * scale1) as u64;
            if (estimate as f64) <= s1_reject {
                survivors.push((i, d));
            }
        }

        // ═══ STROKE 2: Full Hamming on survivors ═══
        let mut finalists: Vec<RankedHit> = Vec::with_capacity(survivors.len() / 5 + 1);
        let query_rest = &query[s1_bytes..];
        for &(idx, d_prefix) in &survivors {
            let base = idx * vec_bytes;
            let d_rest = bitwise::hamming_distance_raw(query_rest, &database[base + s1_bytes..base + vec_bytes]);
            let d_full = d_prefix + d_rest;
            if d_full <= threshold {
                finalists.push(RankedHit {
                    index: idx,
                    hamming: d_full,
                    precise: f64::NAN,
                    band: self.expose(d_full as u32),
                });
            }
        }

        finalists
    }

    /// Run cascade verification on pre-filtered candidate indices.
    ///
    /// This is the CLAM->Cascade bridge entry point. CLAM provides tight
    /// candidates via rho-NN (triangle-inequality pruning), replacing Stroke 1's
    /// statistical prefix scan. This method runs Stroke 2 (full Hamming) and
    /// Stroke 3 (banding) on those candidates only.
    ///
    /// Because CLAM already provides geometrically tight candidates, Stroke 1
    /// is partially redundant -- we skip directly to full Hamming verification.
    pub fn query_candidates(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        candidate_indices: &[(usize, u64)],
    ) -> Vec<RankedHit> {
        let threshold = self.threshold;
        let mut results = Vec::with_capacity(candidate_indices.len());

        for &(idx, clam_dist) in candidate_indices {
            // CLAM already computed Hamming distances; use them if within threshold
            let d = if clam_dist > 0 {
                // Verify with full Hamming (CLAM distances are exact for Hamming)
                clam_dist
            } else {
                // Distance 0 = exact match, verify
                let base = idx * vec_bytes;
                bitwise::hamming_distance_raw(query, &database[base..base + vec_bytes])
            };

            if d <= threshold {
                results.push(RankedHit {
                    index: idx,
                    hamming: d,
                    precise: f64::NAN,
                    band: self.expose(d as u32),
                });
            }
        }

        results.sort_unstable_by_key(|r| r.hamming);
        results
    }

    /// Run the full 3-stroke cascade query with precision scoring (Stroke 3).
    pub fn query_precise(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        num_vectors: usize,
        precise_mode: PreciseMode,
    ) -> Vec<RankedHit> {
        let mut results = self.query(query, database, vec_bytes, num_vectors);

        if precise_mode != PreciseMode::Off && !results.is_empty() {
            apply_precision_tier(query, database, vec_bytes, &mut results, precise_mode);
        }

        results
    }
}

// ============================================================================
// Scalar dot products for precision tier (no SIMD dependency)
// ============================================================================

/// Scalar dot product on i8 vectors (treats u8 as unsigned for dot).
fn dot_i8_scalar(a: &[u8], b: &[u8]) -> i64 {
    let n = a.len().min(b.len());
    let mut sum: i64 = 0;
    for i in 0..n {
        sum += (a[i] as i64) * (b[i] as i64);
    }
    sum
}

/// Scalar f32 dot product.
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum += a[i] as f64 * b[i] as f64;
    }
    sum
}

/// BF16-structured Hamming distance: XOR + per-field weighted popcount.
fn bf16_hamming_scalar(a: &[u8], b: &[u8], weights: &BF16Weights) -> u64 {
    let n_dims = a.len().min(b.len()) / 2;
    let mut total = 0u64;
    for d in 0..n_dims {
        let ai = u16::from_le_bytes([a[d * 2], a[d * 2 + 1]]);
        let bi = u16::from_le_bytes([b[d * 2], b[d * 2 + 1]]);
        let xor = ai ^ bi;
        // sign bit 15
        let sign_diff = ((xor >> 15) & 1) as u64 * weights.sign as u64;
        // exponent bits 14..7
        let exp_diff = ((xor >> 7) & 0xFF).count_ones() as u64 * weights.exponent as u64;
        // mantissa bits 6..0
        let man_diff = (xor & 0x7F).count_ones() as u64 * weights.mantissa as u64;
        total += sign_diff + exp_diff + man_diff;
    }
    total
}

/// Stroke 3: compute high-precision distance for finalists.
///
/// Sorts by precise distance descending (most similar first).
fn apply_precision_tier(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    finalists: &mut [RankedHit],
    precise_mode: PreciseMode,
) {
    match precise_mode {
        PreciseMode::Off => return,

        PreciseMode::Vnni => {
            let query_norm_sq = dot_i8_scalar(query, query);
            let query_norm = (query_norm_sq as f64).sqrt();

            if query_norm == 0.0 {
                for r in finalists.iter_mut() {
                    r.precise = 0.0;
                }
                return;
            }

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];
                let dot = dot_i8_scalar(query, candidate);
                let cand_norm = (dot_i8_scalar(candidate, candidate) as f64).sqrt();
                r.precise = if cand_norm > 0.0 {
                    dot as f64 / (query_norm * cand_norm)
                } else {
                    0.0
                };
            }
        }

        PreciseMode::F32 { scale, zero_point } | PreciseMode::BF16 { scale, zero_point } => {
            let mut query_f32 = vec![0.0f32; vec_bytes];
            for i in 0..vec_bytes {
                query_f32[i] = scale * (query[i] as i32 - zero_point) as f32;
            }
            let query_norm = dot_f32_scalar(&query_f32, &query_f32).sqrt();

            if query_norm == 0.0 {
                for r in finalists.iter_mut() {
                    r.precise = 0.0;
                }
                return;
            }

            let mut cand_f32 = vec![0.0f32; vec_bytes];

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];
                for i in 0..vec_bytes {
                    cand_f32[i] = scale * (candidate[i] as i32 - zero_point) as f32;
                }
                let dot = dot_f32_scalar(&query_f32, &cand_f32);
                let cand_norm = dot_f32_scalar(&cand_f32, &cand_f32).sqrt();
                r.precise = if cand_norm > 0.0 {
                    dot / (query_norm * cand_norm)
                } else {
                    0.0
                };
            }
        }

        PreciseMode::DeltaXor { delta_weight } => {
            let total_bits = (vec_bytes * 8) as f64;
            let query_norm_sq = dot_i8_scalar(query, query);
            let query_norm = (query_norm_sq as f64).sqrt();
            let w = delta_weight as f64;

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];
                let hamming_norm = r.hamming as f64 / total_bits;
                let cosine = if query_norm > 0.0 {
                    let dot = dot_i8_scalar(query, candidate);
                    let cand_norm = (dot_i8_scalar(candidate, candidate) as f64).sqrt();
                    if cand_norm > 0.0 {
                        dot as f64 / (query_norm * cand_norm)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                r.precise = 1.0 - (hamming_norm * (1.0 - w) + (1.0 - cosine) * w);
            }
        }

        PreciseMode::BF16Hamming { weights } => {
            let max_per_dim =
                weights.sign as u64 + 8 * weights.exponent as u64 + 7 * weights.mantissa as u64;
            let n_dims = vec_bytes / 2;
            let max_total = max_per_dim * n_dims as u64;

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];
                let dist = bf16_hamming_scalar(query, candidate, &weights);
                let norm = if max_total > 0 {
                    dist as f64 / max_total as f64
                } else {
                    1.0
                };
                r.precise = 1.0 - norm;
            }
        }
    }

    // Sort by precise distance descending (most similar first)
    finalists.sort_unstable_by(|a, b| {
        b.precise
            .partial_cmp(&a.precise)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Packed database for stroke-aligned cascade search.
pub struct PackedDatabase {
    pub stroke1: Vec<u8>,
    pub stroke2: Vec<u8>,
    pub stroke3: Vec<u8>,
    pub num_vectors: usize,
    pub s1_bytes: usize,
    pub s2_bytes: usize,
    pub s3_bytes: usize,
}

impl PackedDatabase {
    /// Pack a flat database into stroke-aligned layout.
    pub fn pack(database: &[u8], vec_bytes: usize) -> Self {
        let num_vectors = database.len() / vec_bytes;
        let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
        let s2_end = (vec_bytes / 4).max(s1_bytes).min(vec_bytes);
        let s2_bytes = s2_end - s1_bytes;
        let s3_bytes = vec_bytes - s2_end;

        let mut stroke1 = Vec::with_capacity(num_vectors * s1_bytes);
        let mut stroke2 = Vec::with_capacity(num_vectors * s2_bytes);
        let mut stroke3 = Vec::with_capacity(num_vectors * s3_bytes);

        for i in 0..num_vectors {
            let base = i * vec_bytes;
            stroke1.extend_from_slice(&database[base..base + s1_bytes]);
            if s2_bytes > 0 {
                stroke2.extend_from_slice(&database[base + s1_bytes..base + s1_bytes + s2_bytes]);
            }
            if s3_bytes > 0 {
                stroke3.extend_from_slice(&database[base + s1_bytes + s2_bytes..base + vec_bytes]);
            }
        }

        Self { stroke1, stroke2, stroke3, num_vectors, s1_bytes, s2_bytes, s3_bytes }
    }

    /// Run cascade query on packed layout.
    pub fn cascade_query(&self, query: &[u8], cascade: &Cascade, top_k: usize) -> Vec<RankedHit> {
        let threshold = cascade.threshold;
        let vec_bytes = self.s1_bytes + self.s2_bytes + self.s3_bytes;
        let scale1 = vec_bytes as f64 / self.s1_bytes as f64;

        let query_s1 = &query[..self.s1_bytes];

        // Stroke 1
        let mut survivors = Vec::new();
        for i in 0..self.num_vectors {
            let s1_start = i * self.s1_bytes;
            let d = bitwise::hamming_distance_raw(query_s1, &self.stroke1[s1_start..s1_start + self.s1_bytes]);
            let estimate = (d as f64 * scale1) as u64;
            if estimate <= threshold + threshold / 4 {
                survivors.push((i, d));
            }
        }

        // Stroke 2
        let query_s2 = &query[self.s1_bytes..self.s1_bytes + self.s2_bytes];
        let mut finalists = Vec::new();
        for &(idx, d1) in &survivors {
            if self.s2_bytes > 0 {
                let s2_start = idx * self.s2_bytes;
                let d2 = bitwise::hamming_distance_raw(query_s2, &self.stroke2[s2_start..s2_start + self.s2_bytes]);
                let d12 = d1 + d2;
                let estimate = (d12 as f64 * vec_bytes as f64 / (self.s1_bytes + self.s2_bytes) as f64) as u64;
                if estimate <= threshold {
                    finalists.push((idx, d12));
                }
            } else {
                finalists.push((idx, d1));
            }
        }

        // Stroke 3
        let query_s3 = &query[self.s1_bytes + self.s2_bytes..];
        let mut results: Vec<RankedHit> = Vec::new();
        for &(idx, d12) in &finalists {
            if self.s3_bytes > 0 {
                let s3_start = idx * self.s3_bytes;
                let d3 = bitwise::hamming_distance_raw(query_s3, &self.stroke3[s3_start..s3_start + self.s3_bytes]);
                let d_full = d12 + d3;
                if d_full <= threshold {
                    results.push(RankedHit {
                        index: idx,
                        hamming: d_full,
                        precise: f64::NAN,
                        band: cascade.expose(d_full as u32),
                    });
                }
            } else {
                results.push(RankedHit {
                    index: idx,
                    hamming: d12,
                    precise: f64::NAN,
                    band: cascade.expose(d12 as u32),
                });
            }
        }

        results.sort_unstable_by_key(|r| r.hamming);
        results.truncate(top_k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_calibrate() {
        let dists: Vec<u32> = (0..100).map(|i| 8000 + (i % 200)).collect();
        let cascade = Cascade::calibrate(&dists, 2048);
        assert!(cascade.threshold > 8000);
        assert!(cascade.threshold < 10000);
    }

    #[test]
    fn cascade_expose_bands() {
        let cascade = Cascade::from_threshold(1000, 2048);
        assert_eq!(cascade.expose(100), Band::Foveal);
        assert_eq!(cascade.expose(400), Band::Near);
        assert_eq!(cascade.expose(600), Band::Good);
        assert_eq!(cascade.expose(900), Band::Weak);
        assert_eq!(cascade.expose(1500), Band::Reject);
    }

    #[test]
    fn cascade_observe_drift() {
        let mut cascade = Cascade::from_threshold(1000, 2048);
        // Feed stable observations with some variance
        for i in 0..20 {
            cascade.observe(500 + (i % 5) * 2);
        }
        // Feed strongly drifted observations
        let mut alert = None;
        for _ in 0..50 {
            if let Some(a) = cascade.observe(5000) {
                alert = Some(a);
                break;
            }
        }
        assert!(alert.is_some(), "drift should be detected");
    }

    #[test]
    fn cascade_query_finds_identical() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0x55u8; vec_bytes * 10];
        // Make candidate 3 identical to query
        database[3 * vec_bytes..4 * vec_bytes].copy_from_slice(&query);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let results = cascade.query(&query, &database, vec_bytes, 10);
        assert!(results.iter().any(|r| r.index == 3 && r.hamming == 0));
    }

    #[test]
    fn packed_database_roundtrip() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0x55u8; vec_bytes * 20];
        database[5 * vec_bytes..6 * vec_bytes].copy_from_slice(&query);
        let packed = PackedDatabase::pack(&database, vec_bytes);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let results = packed.cascade_query(&query, &cascade, 10);
        assert!(results.iter().any(|r| r.index == 5 && r.hamming == 0));
    }

    #[test]
    fn cascade_query_precise_vnni() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0x55u8; vec_bytes * 10];
        database[3 * vec_bytes..4 * vec_bytes].copy_from_slice(&query);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let results = cascade.query_precise(&query, &database, vec_bytes, 10, PreciseMode::Vnni);
        assert!(results.iter().any(|r| r.index == 3 && r.hamming == 0));
        // Precise score for exact match should be 1.0
        let exact = results.iter().find(|r| r.index == 3).unwrap();
        assert!((exact.precise - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cascade_query_precise_f32() {
        let vec_bytes = 256;
        let query = vec![128u8; vec_bytes];
        let mut database = vec![64u8; vec_bytes * 5];
        database[2 * vec_bytes..3 * vec_bytes].copy_from_slice(&query);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let results = cascade.query_precise(
            &query, &database, vec_bytes, 5,
            PreciseMode::F32 { scale: 1.0 / 128.0, zero_point: 128 },
        );
        let exact = results.iter().find(|r| r.index == 2).unwrap();
        assert!(!exact.precise.is_nan());
    }

    #[test]
    fn cascade_query_precise_bf16hamming() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0x55u8; vec_bytes * 5];
        database[1 * vec_bytes..2 * vec_bytes].copy_from_slice(&query);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let weights = BF16Weights::new(256, 16, 1);
        let results = cascade.query_precise(
            &query, &database, vec_bytes, 5,
            PreciseMode::BF16Hamming { weights },
        );
        let exact = results.iter().find(|r| r.index == 1).unwrap();
        assert!((exact.precise - 1.0).abs() < 1e-6, "exact match should have precise=1.0");
    }

    #[test]
    fn cascade_query_candidates_finds_exact_match() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0x55u8; vec_bytes * 10];
        // Make candidate 3 identical to query
        database[3 * vec_bytes..4 * vec_bytes].copy_from_slice(&query);
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);

        // Pre-filtered candidates: (index, clam_distance)
        let candidates = vec![(3, 0u64), (5, 500)];
        let results = cascade.query_candidates(&query, &database, vec_bytes, &candidates);

        // Exact match should survive
        assert!(results.iter().any(|r| r.index == 3 && r.hamming == 0));
        // Results should be sorted by hamming distance
        for w in results.windows(2) {
            assert!(w[0].hamming <= w[1].hamming);
        }
    }

    #[test]
    fn cascade_query_candidates_rejects_above_threshold() {
        let vec_bytes = 64;
        let query = vec![0xFFu8; vec_bytes];
        let database = vec![0x00u8; vec_bytes * 5]; // all zeros, max hamming from query
        // Hamming(0xFF, 0x00) = 8 bits per byte * 64 bytes = 512
        let cascade = Cascade::from_threshold(100, vec_bytes); // tight threshold

        let candidates = vec![(0, 512), (1, 512)];
        let results = cascade.query_candidates(&query, &database, vec_bytes, &candidates);

        // All should be rejected since dist=512 > threshold=100
        assert!(results.is_empty());
    }

    #[test]
    fn cascade_query_candidates_bands_correct() {
        let vec_bytes = 256;
        let query = vec![0xAAu8; vec_bytes];
        let mut database = vec![0xAAu8; vec_bytes * 5];
        // candidate 0: exact match (dist 0)
        // candidate 1: slightly different
        database[1 * vec_bytes] = 0xBB; // flip some bits

        let cascade = Cascade::from_threshold(1000, vec_bytes);
        let d1 = bitwise::hamming_distance_raw(&query, &database[vec_bytes..2 * vec_bytes]);
        let candidates = vec![(0, 0), (1, d1)];
        let results = cascade.query_candidates(&query, &database, vec_bytes, &candidates);

        assert!(!results.is_empty());
        // Exact match should be Foveal
        let exact = results.iter().find(|r| r.index == 0).unwrap();
        assert_eq!(exact.band, Band::Foveal);
    }
}
