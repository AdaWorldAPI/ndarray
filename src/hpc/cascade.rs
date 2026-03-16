//! HDR (High Dynamic Range) Cascade Search.
//!
//! 3-stroke adaptive cascade for Hamming-based nearest-neighbour search.

use super::bitwise;

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
}
