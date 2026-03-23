//! PackedDatabase: stroke-aligned layout for streaming cascade search.
//!
//! Panel packing (siboehm GEMM analogy): reorder database so each stroke
//! is contiguous across all candidates. The prefetcher handles sequential access.
//!
//! Memory bandwidth reduction: up to 11.3x for cascade with 90% rejection per stroke.
//!
//! # Example
//!
//! ```
//! use ndarray::hpc::packed::{PackedDatabase, FINGERPRINT_BYTES};
//!
//! // Create a small database of 3 candidates
//! let mut db = vec![0u8; 3 * FINGERPRINT_BYTES];
//! db[0] = 0xFF; // candidate 0 starts with 0xFF
//! let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
//! assert_eq!(packed.num_candidates(), 3);
//! ```

use super::bitwise;

/// Software prefetch: bring a cache line into L1 for the given byte slice.
///
/// No-op on non-x86 targets. On x86_64, uses `_mm_prefetch(_MM_HINT_T0)`.
/// The prefetch distance (how many candidates ahead) should be tuned per
/// cache hierarchy — 4 candidates × 128B = 512B ≈ 8 cache lines is a
/// reasonable default for Stroke 1 sequential scan.
#[inline(always)]
#[allow(unused_variables)]
fn prefetch_t0(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a CPU hint that cannot cause UB for any pointer
    // value — the CPU silently ignores invalid or unmapped addresses. The `ptr`
    // comes from a bounds-checked slice index in the caller.
    unsafe {
        #[cfg(target_feature = "sse")]
        {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                ptr as *const i8,
            );
        }
    }
}

/// Prefetch into L2 cache (temporal hint for data accessed in a later stage).
#[inline(always)]
#[allow(unused_variables)]
fn prefetch_t1(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a CPU hint that cannot cause UB for any pointer
    // value — the CPU silently ignores invalid or unmapped addresses. The `ptr`
    // comes from a bounds-checked slice index in the caller.
    unsafe {
        #[cfg(target_feature = "sse")]
        {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(
                ptr as *const i8,
            );
        }
    }
}

/// Number of candidates to prefetch ahead in the Stroke 1 scan.
const PREFETCH_DISTANCE: usize = 4;

/// Stroke 1: first 128 bytes (1024 bits) — coarse rejection (~90% eliminated).
pub const STROKE1_BYTES: usize = 128;
/// Stroke 2: bytes 128..512 (3072 bits) — medium filter (~90% of survivors).
pub const STROKE2_BYTES: usize = 384;
/// Stroke 3: bytes 512..2048 (12288 bits) — precise distance (final ranking).
pub const STROKE3_BYTES: usize = 1536;
/// Full fingerprint: 2048 bytes (16384 bits).
pub const FINGERPRINT_BYTES: usize = 2048;

/// Stroke-aligned database layout for streaming cascade search.
///
/// Each stroke's bytes for all candidates are stored contiguously, enabling
/// perfect sequential prefetching. This is the "panel packing" optimization
/// from Goto BLAS applied to binary cascade search.
///
/// # Layout
///
/// ```text
/// stroke1: [cand[0]_0..128 | cand[1]_0..128 | cand[2]_0..128 | ...]
/// stroke2: [cand[0]_128..512 | cand[1]_128..512 | ...]
/// stroke3: [cand[0]_512..2048 | cand[1]_512..2048 | ...]
/// ```
pub struct PackedDatabase {
    /// Stroke 1 data: N × STROKE1_BYTES contiguous bytes.
    stroke1: Vec<u8>,
    /// Stroke 2 data: N × STROKE2_BYTES contiguous bytes.
    stroke2: Vec<u8>,
    /// Stroke 3 data: N × STROKE3_BYTES contiguous bytes.
    stroke3: Vec<u8>,
    /// Original candidate indices (for result mapping back to source).
    index: Vec<u32>,
    /// Number of candidates.
    num_candidates: usize,
}

/// A ranked search result from cascade query.
#[derive(Debug, Clone)]
pub struct RankedHit {
    /// Original candidate index in the source database.
    pub index: usize,
    /// Full Hamming distance (sum of all 3 strokes).
    pub distance: u64,
}

impl PackedDatabase {
    /// Pack a flat database into stroke-aligned layout.
    ///
    /// This is done ONCE at database construction. O(N × 2048) memory copy.
    /// Amortized over all subsequent queries.
    ///
    /// # Panics
    ///
    /// Panics if `row_bytes != FINGERPRINT_BYTES`.
    pub fn pack(database: &[u8], row_bytes: usize) -> Self {
        assert_eq!(row_bytes, FINGERPRINT_BYTES, "row_bytes must be {FINGERPRINT_BYTES}");
        let n = database.len() / row_bytes;

        let mut stroke1 = Vec::with_capacity(n * STROKE1_BYTES);
        let mut stroke2 = Vec::with_capacity(n * STROKE2_BYTES);
        let mut stroke3 = Vec::with_capacity(n * STROKE3_BYTES);
        let mut index = Vec::with_capacity(n);

        for i in 0..n {
            let base = i * row_bytes;
            stroke1.extend_from_slice(&database[base..base + STROKE1_BYTES]);
            stroke2.extend_from_slice(
                &database[base + STROKE1_BYTES..base + STROKE1_BYTES + STROKE2_BYTES],
            );
            stroke3.extend_from_slice(
                &database[base + STROKE1_BYTES + STROKE2_BYTES..base + row_bytes],
            );
            index.push(i as u32);
        }

        Self { stroke1, stroke2, stroke3, index, num_candidates: n }
    }

    /// Get stroke 1 slice for candidate i (128 bytes).
    #[inline(always)]
    pub fn get_stroke1(&self, i: usize) -> &[u8] {
        &self.stroke1[i * STROKE1_BYTES..(i + 1) * STROKE1_BYTES]
    }

    /// Get stroke 2 slice for candidate i (384 bytes).
    #[inline(always)]
    pub fn get_stroke2(&self, i: usize) -> &[u8] {
        &self.stroke2[i * STROKE2_BYTES..(i + 1) * STROKE2_BYTES]
    }

    /// Get stroke 3 slice for candidate i (1536 bytes).
    #[inline(always)]
    pub fn get_stroke3(&self, i: usize) -> &[u8] {
        &self.stroke3[i * STROKE3_BYTES..(i + 1) * STROKE3_BYTES]
    }

    /// The raw contiguous stroke 1 buffer (for batch SIMD operations).
    pub fn stroke1_data(&self) -> &[u8] {
        &self.stroke1
    }

    /// The raw contiguous stroke 2 buffer.
    pub fn stroke2_data(&self) -> &[u8] {
        &self.stroke2
    }

    /// The raw contiguous stroke 3 buffer.
    pub fn stroke3_data(&self) -> &[u8] {
        &self.stroke3
    }

    /// Original candidate ID for result mapping.
    #[inline(always)]
    pub fn original_id(&self, i: usize) -> u32 {
        self.index[i]
    }

    /// Number of candidates in the database.
    pub fn num_candidates(&self) -> usize {
        self.num_candidates
    }

    /// Three-stroke cascade query with early rejection.
    ///
    /// - Stroke 1: sequential scan of 128B per candidate, reject above threshold
    /// - Stroke 2: scan survivors only (384B), reject above cumulative threshold
    /// - Stroke 3: scan final survivors (1536B), compute exact distance
    ///
    /// Returns top-k results sorted by distance ascending.
    pub fn cascade_query(
        &self,
        query: &[u8],
        reject_threshold_s1: u64,
        reject_threshold_s12: u64,
        k: usize,
    ) -> Vec<RankedHit> {
        assert!(query.len() >= FINGERPRINT_BYTES, "query must be at least {FINGERPRINT_BYTES} bytes");

        let query_s1 = &query[..STROKE1_BYTES];
        let query_s2 = &query[STROKE1_BYTES..STROKE1_BYTES + STROKE2_BYTES];
        let query_s3 = &query[STROKE1_BYTES + STROKE2_BYTES..FINGERPRINT_BYTES];

        // STROKE 1: coarse rejection — sequential scan through packed stroke1
        // Prefetch PREFETCH_DISTANCE candidates ahead to hide memory latency.
        let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(self.num_candidates / 10);
        for i in 0..self.num_candidates {
            // Prefetch stroke1 data for upcoming candidate
            if i + PREFETCH_DISTANCE < self.num_candidates {
                prefetch_t0(self.stroke1[(i + PREFETCH_DISTANCE) * STROKE1_BYTES..].as_ptr());
            }
            let d1 = bitwise::hamming_distance_raw(query_s1, self.get_stroke1(i));
            if d1 <= reject_threshold_s1 {
                survivors.push((i, d1));
            }
        }

        // STROKE 2: medium filter — scan survivors through packed stroke2
        // Prefetch stroke2 data for next survivor (sparse access pattern).
        let mut survivors2: Vec<(usize, u64)> = Vec::with_capacity(survivors.len() / 10);
        for (si, &(idx, d1)) in survivors.iter().enumerate() {
            // Prefetch next survivor's stroke2 data into L2
            if si + 1 < survivors.len() {
                let next_idx = survivors[si + 1].0;
                prefetch_t1(self.stroke2[next_idx * STROKE2_BYTES..].as_ptr());
            }
            let d2 = bitwise::hamming_distance_raw(query_s2, self.get_stroke2(idx));
            let d_cumul = d1 + d2;
            if d_cumul <= reject_threshold_s12 {
                survivors2.push((idx, d_cumul));
            }
        }

        // STROKE 3: precise distance — final ranking
        // Prefetch next survivor's stroke3 data.
        let mut results: Vec<RankedHit> = Vec::with_capacity(survivors2.len());
        for (si, &(idx, d12)) in survivors2.iter().enumerate() {
            if si + 1 < survivors2.len() {
                let next_idx = survivors2[si + 1].0;
                prefetch_t0(self.stroke3[next_idx * STROKE3_BYTES..].as_ptr());
            }
            let d3 = bitwise::hamming_distance_raw(query_s3, self.get_stroke3(idx));
            results.push(RankedHit {
                index: self.original_id(idx) as usize,
                distance: d12 + d3,
            });
        }

        // Sort and take top-k
        results.sort_unstable_by_key(|h| h.distance);
        results.truncate(k);
        results
    }

    /// Brute-force scan (no cascade) — for benchmarking comparison.
    ///
    /// Computes full Hamming distance for every candidate using all 3 strokes.
    pub fn brute_force_query(&self, query: &[u8], k: usize) -> Vec<RankedHit> {
        assert!(query.len() >= FINGERPRINT_BYTES);

        let query_s1 = &query[..STROKE1_BYTES];
        let query_s2 = &query[STROKE1_BYTES..STROKE1_BYTES + STROKE2_BYTES];
        let query_s3 = &query[STROKE1_BYTES + STROKE2_BYTES..FINGERPRINT_BYTES];

        let mut results: Vec<RankedHit> = (0..self.num_candidates)
            .map(|i| {
                let d1 = bitwise::hamming_distance_raw(query_s1, self.get_stroke1(i));
                let d2 = bitwise::hamming_distance_raw(query_s2, self.get_stroke2(i));
                let d3 = bitwise::hamming_distance_raw(query_s3, self.get_stroke3(i));
                RankedHit { index: self.original_id(i) as usize, distance: d1 + d2 + d3 }
            })
            .collect();

        if k < results.len() {
            results.select_nth_unstable_by_key(k, |h| h.distance);
            results.truncate(k);
            results.sort_unstable_by_key(|h| h.distance);
        } else {
            results.sort_unstable_by_key(|h| h.distance);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_roundtrip() {
        let n = 10;
        let db: Vec<u8> = (0..n * FINGERPRINT_BYTES).map(|i| i as u8).collect();
        let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
        assert_eq!(packed.num_candidates(), n);

        // Verify each candidate's strokes reconstruct the original
        for i in 0..n {
            let base = i * FINGERPRINT_BYTES;
            assert_eq!(packed.get_stroke1(i), &db[base..base + STROKE1_BYTES]);
            assert_eq!(
                packed.get_stroke2(i),
                &db[base + STROKE1_BYTES..base + STROKE1_BYTES + STROKE2_BYTES]
            );
            assert_eq!(
                packed.get_stroke3(i),
                &db[base + STROKE1_BYTES + STROKE2_BYTES..base + FINGERPRINT_BYTES]
            );
        }
    }

    #[test]
    fn test_brute_force_query() {
        let n = 5;
        // Each candidate has a unique pattern so distances are distinct
        let mut db = vec![0u8; n * FINGERPRINT_BYTES];
        // Candidate 0: all zeros → distance 0
        // Candidate 1: first byte 0xFF → distance 8
        db[1 * FINGERPRINT_BYTES] = 0xFF;
        // Candidate 2: first 2 bytes 0xFF → distance 16
        db[2 * FINGERPRINT_BYTES] = 0xFF;
        db[2 * FINGERPRINT_BYTES + 1] = 0xFF;
        // Candidate 3: first 3 bytes 0xFF → distance 24
        db[3 * FINGERPRINT_BYTES] = 0xFF;
        db[3 * FINGERPRINT_BYTES + 1] = 0xFF;
        db[3 * FINGERPRINT_BYTES + 2] = 0xFF;
        // Candidate 4: first 4 bytes 0xFF → distance 32
        db[4 * FINGERPRINT_BYTES] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 1] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 2] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 3] = 0xFF;

        let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
        let query = vec![0u8; FINGERPRINT_BYTES];
        let results = packed.brute_force_query(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].distance, 8);
        assert_eq!(results[1].index, 1);
        assert_eq!(results[2].distance, 16);
        assert_eq!(results[2].index, 2);
    }

    #[test]
    fn test_cascade_query() {
        let n = 5;
        let mut db = vec![0u8; n * FINGERPRINT_BYTES];
        // Candidate 0: all zeros (exact match)
        // Candidate 1: first byte 0xFF in stroke1 (distance 8)
        db[1 * FINGERPRINT_BYTES] = 0xFF;
        // Candidate 2: 2 bytes 0xFF (distance 16, should be rejected by stroke1 threshold=10)
        db[2 * FINGERPRINT_BYTES] = 0xFF;
        db[2 * FINGERPRINT_BYTES + 1] = 0xFF;
        // Candidate 3: 3 bytes 0xFF (distance 24)
        db[3 * FINGERPRINT_BYTES] = 0xFF;
        db[3 * FINGERPRINT_BYTES + 1] = 0xFF;
        db[3 * FINGERPRINT_BYTES + 2] = 0xFF;
        // Candidate 4: 4 bytes 0xFF (distance 32)
        db[4 * FINGERPRINT_BYTES] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 1] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 2] = 0xFF;
        db[4 * FINGERPRINT_BYTES + 3] = 0xFF;

        let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
        let query = vec![0u8; FINGERPRINT_BYTES];

        // Thresholds: allow up to 10 bits in stroke1, 20 bits cumulative in s1+s2
        let results = packed.cascade_query(&query, 10, 20, 5);

        // Candidates 0 (dist=0) and 1 (dist=8) survive stroke1 threshold=10
        // Candidates 2,3,4 (dist>=16 in stroke1 alone) are rejected
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[1].index, 1);
        assert_eq!(results[1].distance, 8);
    }

    #[test]
    fn test_cascade_vs_brute_force_consistency() {
        let n = 50;
        let db: Vec<u8> = (0..n * FINGERPRINT_BYTES).map(|i| (i * 7 + 13) as u8).collect();
        let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
        let query: Vec<u8> = (0..FINGERPRINT_BYTES).map(|i| (i * 3) as u8).collect();

        // With very high thresholds, cascade should return same results as brute force
        let brute = packed.brute_force_query(&query, 5);
        let cascade = packed.cascade_query(&query, u64::MAX, u64::MAX, 5);

        assert_eq!(brute.len(), cascade.len());
        // Distances must match (indices may differ when distances are tied)
        let brute_dists: Vec<u64> = brute.iter().map(|r| r.distance).collect();
        let cascade_dists: Vec<u64> = cascade.iter().map(|r| r.distance).collect();
        assert_eq!(brute_dists, cascade_dists);
    }

    #[test]
    fn test_stroke_data_contiguous() {
        let n = 5;
        let db: Vec<u8> = (0..n * FINGERPRINT_BYTES).map(|i| i as u8).collect();
        let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);

        // stroke1_data should be exactly n * STROKE1_BYTES
        assert_eq!(packed.stroke1_data().len(), n * STROKE1_BYTES);
        assert_eq!(packed.stroke2_data().len(), n * STROKE2_BYTES);
        assert_eq!(packed.stroke3_data().len(), n * STROKE3_BYTES);

        // First candidate's stroke1 should be at the start
        assert_eq!(&packed.stroke1_data()[..STROKE1_BYTES], packed.get_stroke1(0));
    }
}
