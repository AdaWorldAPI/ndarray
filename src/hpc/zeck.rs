//! ZeckF64: 8-byte progressive edge encoding for SPO triples.
//!
//! Each edge between two nodes is encoded as a single `u64`:
//!
//! - **Byte 0 (scent):** 7 boolean SPO band classifications + sign bit.
//!   The bits form a boolean lattice: `SP_=close` implies `S__=close AND _P_=close`.
//!   19 of 128 patterns are legal, giving ~85% built-in error detection.
//!
//! - **Bytes 1–7 (resolution):** Distance quantiles within each band mask.
//!   Each byte encodes 256 levels of refinement (0 = identical, 255 = max different).
//!
//! Progressive reading: byte 0 alone gives ρ ≈ 0.94 rank correlation.
//!
//! Ported from `lance-graph/crates/lance-graph/src/graph/neighborhood/zeckf64.rs`.
//! Batch and top-k operations added for symmetry with `hpc::bitwise`.

use super::bitwise::hamming_distance_raw;

// ============================================================================
// Constants
// ============================================================================

/// Maximum bits per plane (16384-bit fingerprint).
const D_MAX: u32 = 16384;

/// "Close" threshold: less than half the bits differ.
const THRESHOLD: u32 = D_MAX / 2;

// ============================================================================
// ZeckF64 encoding
// ============================================================================

/// Quantize a single distance into [0, 255].
#[inline]
fn quantile_1(d: u32) -> u8 {
    ((d as u64 * 255) / D_MAX as u64).min(255) as u8
}

/// Quantize max of 2 distances into [0, 255].
#[inline]
fn quantile_2(d1: u32, d2: u32) -> u8 {
    quantile_1(d1.max(d2))
}

/// Quantize max of 3 distances into [0, 255].
#[inline]
fn quantile_3(d1: u32, d2: u32, d3: u32) -> u8 {
    quantile_1(d1.max(d2).max(d3))
}

/// Compute ZeckF64 encoding from pre-computed Hamming distances.
///
/// `ds`, `dp`, `d_o` are per-plane Hamming distances (Subject, Predicate, Object).
/// Returns an 8-byte u64 with progressive precision.
pub fn zeckf64_from_distances(ds: u32, dp: u32, d_o: u32) -> u64 {
    let s_close = (ds < THRESHOLD) as u8;
    let p_close = (dp < THRESHOLD) as u8;
    let o_close = (d_o < THRESHOLD) as u8;
    let sp_close = s_close & p_close;
    let so_close = s_close & o_close;
    let po_close = p_close & o_close;
    let spo_close = sp_close & so_close & po_close;

    let byte0 = s_close
        | (p_close << 1)
        | (o_close << 2)
        | (sp_close << 3)
        | (so_close << 4)
        | (po_close << 5)
        | (spo_close << 6);

    let byte1 = quantile_3(ds, dp, d_o);
    let byte2 = quantile_2(dp, d_o);
    let byte3 = quantile_2(ds, d_o);
    let byte4 = quantile_2(ds, dp);
    let byte5 = quantile_1(d_o);
    let byte6 = quantile_1(dp);
    let byte7 = quantile_1(ds);

    (byte0 as u64)
        | ((byte1 as u64) << 8)
        | ((byte2 as u64) << 16)
        | ((byte3 as u64) << 24)
        | ((byte4 as u64) << 32)
        | ((byte5 as u64) << 40)
        | ((byte6 as u64) << 48)
        | ((byte7 as u64) << 56)
}

/// Compute ZeckF64 encoding from raw fingerprint byte slices.
///
/// Each triple is `(subject, predicate, object)` as `&[u8]` (16384-bit / 2048 bytes).
pub fn zeckf64(
    a: (&[u8], &[u8], &[u8]),
    b: (&[u8], &[u8], &[u8]),
) -> u64 {
    let ds = hamming_distance_raw(a.0, b.0) as u32;
    let dp = hamming_distance_raw(a.1, b.1) as u32;
    let d_o = hamming_distance_raw(a.2, b.2) as u32;
    zeckf64_from_distances(ds, dp, d_o)
}

// ============================================================================
// Accessors
// ============================================================================

/// Extract the scent byte (byte 0) from a ZeckF64.
#[inline]
pub fn scent(edge: u64) -> u8 {
    edge as u8
}

/// Extract a resolution byte (1–7) from a ZeckF64.
#[inline]
pub fn resolution(edge: u64, byte_n: u8) -> u8 {
    debug_assert!((1..=7).contains(&byte_n), "byte_n must be 1..=7");
    (edge >> (byte_n * 8)) as u8
}

/// Set the sign (causality direction) bit.
#[inline]
pub fn set_sign(edge: u64, sign: bool) -> u64 {
    if sign { edge | (1u64 << 7) } else { edge & !(1u64 << 7) }
}

/// Read the sign bit.
#[inline]
pub fn get_sign(edge: u64) -> bool {
    (edge & (1u64 << 7)) != 0
}

// ============================================================================
// Distance functions
// ============================================================================

/// L1 (Manhattan) distance on two ZeckF64 values (all 8 bytes).
/// Maximum possible distance: 8 × 255 = 2040.
pub fn zeckf64_distance(a: u64, b: u64) -> u32 {
    let mut dist = 0u32;
    for i in 0..8 {
        let ba = ((a >> (i * 8)) & 0xFF) as i16;
        let bb = ((b >> (i * 8)) & 0xFF) as i16;
        dist += (ba - bb).unsigned_abs() as u32;
    }
    dist
}

/// Scent-only L1 distance: byte 0 only. Range: 0–255.
#[inline]
pub fn zeckf64_scent_distance(a: u64, b: u64) -> u32 {
    let ba = (a & 0xFF) as i16;
    let bb = (b & 0xFF) as i16;
    (ba - bb).unsigned_abs() as u32
}

/// Scent-only Hamming distance: popcount(byte0_a ^ byte0_b). Range: 0–8.
#[inline]
pub fn zeckf64_scent_hamming(a: u64, b: u64) -> u32 {
    ((a as u8) ^ (b as u8)).count_ones()
}

/// Progressive L1 distance on bytes 0..=n (inclusive).
/// `n = 0`: scent only (1 byte). `n = 7`: full ZeckF64 (8 bytes).
pub fn zeckf64_progressive_distance(a: u64, b: u64, n: u8) -> u32 {
    let n = n.min(7) as usize;
    let mut dist = 0u32;
    for i in 0..=n {
        let ba = ((a >> (i * 8)) & 0xFF) as i16;
        let bb = ((b >> (i * 8)) & 0xFF) as i16;
        dist += (ba - bb).unsigned_abs() as u32;
    }
    dist
}

/// Validate the boolean lattice constraints of a scent byte.
///
/// Returns `true` if the pattern is legal (19 of 128 are legal).
pub fn is_legal_scent(byte0: u8) -> bool {
    let s = byte0 & 1;
    let p = (byte0 >> 1) & 1;
    let o = (byte0 >> 2) & 1;
    let sp = (byte0 >> 3) & 1;
    let so = (byte0 >> 4) & 1;
    let po = (byte0 >> 5) & 1;
    let spo = (byte0 >> 6) & 1;

    // Lattice: pair bit implies both individual bits
    if sp == 1 && (s == 0 || p == 0) { return false; }
    if so == 1 && (s == 0 || o == 0) { return false; }
    if po == 1 && (p == 0 || o == 0) { return false; }
    // Triple implies all three pairs
    if spo == 1 && (sp == 0 || so == 0 || po == 0) { return false; }
    true
}

// ============================================================================
// Batch operations (parallel to bitwise::hamming_batch_raw / hamming_top_k_raw)
// ============================================================================

/// Batch ZeckF64 distance: compute distance from `query` to each edge in `edges`.
///
/// Returns a Vec of `edges.len()` u32 distances.
pub fn zeckf64_batch(query: u64, edges: &[u64]) -> Vec<u32> {
    edges.iter().map(|&e| zeckf64_distance(query, e)).collect()
}

/// Batch scent-only distance: compute scent distance from `query` to each edge.
pub fn zeckf64_scent_batch(query: u64, edges: &[u64]) -> Vec<u32> {
    edges.iter().map(|&e| zeckf64_scent_distance(query, e)).collect()
}

/// Top-k nearest edges by ZeckF64 distance.
///
/// Returns (indices, distances) of the k closest edges.
/// Uses `select_nth_unstable` for O(n) partial sort.
pub fn zeckf64_top_k(query: u64, edges: &[u64], k: usize) -> (Vec<usize>, Vec<u32>) {
    let distances = zeckf64_batch(query, edges);
    let k = k.min(edges.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();
    indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
    indexed.truncate(k);
    indexed.sort_unstable_by_key(|&(_, d)| d);
    let indices = indexed.iter().map(|&(i, _)| i).collect();
    let dists = indexed.iter().map(|&(_, d)| d).collect();
    (indices, dists)
}

/// Top-k nearest edges by scent-only distance (byte 0 only).
///
/// Faster than full ZeckF64 top-k — only compares 1 byte per edge.
/// Good for the HEEL stage of HHTL cascade search.
pub fn zeckf64_scent_top_k(query: u64, edges: &[u64], k: usize) -> (Vec<usize>, Vec<u32>) {
    let distances = zeckf64_scent_batch(query, edges);
    let k = k.min(edges.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();
    indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
    indexed.truncate(k);
    indexed.sort_unstable_by_key(|&(_, d)| d);
    let indices = indexed.iter().map(|&(i, _)| i).collect();
    let dists = indexed.iter().map(|&(_, d)| d).collect();
    (indices, dists)
}

/// Batch ZeckF64 encoding: compute ZeckF64 for a query triple against each
/// row triple in a flat database.
///
/// `query` = (subject, predicate, object) as &[u8] slices (2048 bytes each).
/// `database` = flat buffer of concatenated S+P+O planes (6144 bytes per row).
/// Returns a Vec of `num_rows` u64 ZeckF64 values.
pub fn zeckf64_encode_batch(
    query: (&[u8], &[u8], &[u8]),
    database: &[u8],
    num_rows: usize,
    plane_bytes: usize,
) -> Vec<u64> {
    let row_bytes = plane_bytes * 3;
    (0..num_rows)
        .map(|i| {
            let offset = i * row_bytes;
            let s = &database[offset..offset + plane_bytes];
            let p = &database[offset + plane_bytes..offset + 2 * plane_bytes];
            let o = &database[offset + 2 * plane_bytes..offset + 3 * plane_bytes];
            zeckf64(query, (s, p, o))
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckf64_identical_triples() {
        let s = vec![0xAAu8; 2048];
        let p = vec![0xBBu8; 2048];
        let o = vec![0xCCu8; 2048];
        let edge = zeckf64((&s, &p, &o), (&s, &p, &o));
        // Identical → all close bits set, all quantiles = 0
        assert_eq!(scent(edge) & 0x7F, 0x7F);
        for b in 1..=7 {
            assert_eq!(resolution(edge, b), 0);
        }
    }

    #[test]
    fn test_zeckf64_distance_self_zero() {
        let edge = zeckf64_from_distances(100, 200, 300);
        assert_eq!(zeckf64_distance(edge, edge), 0);
    }

    #[test]
    fn test_zeckf64_distance_symmetry() {
        let a = zeckf64_from_distances(1000, 5000, 8000);
        let b = zeckf64_from_distances(3000, 7000, 2000);
        assert_eq!(zeckf64_distance(a, b), zeckf64_distance(b, a));
    }

    #[test]
    fn test_scent_distance() {
        let a = zeckf64_from_distances(0, 0, 0); // all close
        let b = zeckf64_from_distances(D_MAX, D_MAX, D_MAX); // none close
        assert!(zeckf64_scent_distance(a, b) > 0);
    }

    #[test]
    fn test_progressive_distance_monotonic() {
        let a = zeckf64_from_distances(1000, 5000, 8000);
        let b = zeckf64_from_distances(3000, 7000, 2000);
        let mut prev = 0;
        for n in 0..=7 {
            let d = zeckf64_progressive_distance(a, b, n);
            assert!(d >= prev, "progressive distance should be monotonic");
            prev = d;
        }
    }

    #[test]
    fn test_is_legal_scent() {
        // All close: 0111_1111 = 0x7F
        assert!(is_legal_scent(0x7F));
        // None close: 0000_0000
        assert!(is_legal_scent(0x00));
        // S only: 0000_0001
        assert!(is_legal_scent(0x01));
        // SP without S: illegal — bit 3 set but bit 0 unset
        assert!(!is_legal_scent(0x08));
        // SP with S and P: 0000_1011 = S + P + SP
        assert!(is_legal_scent(0x0B));
    }

    #[test]
    fn test_sign_bit() {
        let edge = zeckf64_from_distances(100, 200, 300);
        assert!(!get_sign(edge));
        let signed = set_sign(edge, true);
        assert!(get_sign(signed));
        let unsigned = set_sign(signed, false);
        assert!(!get_sign(unsigned));
    }

    #[test]
    fn test_zeckf64_batch() {
        let query = zeckf64_from_distances(1000, 2000, 3000);
        let edges = vec![
            zeckf64_from_distances(1000, 2000, 3000), // identical
            zeckf64_from_distances(5000, 8000, 10000), // far
            zeckf64_from_distances(1100, 2100, 3100), // close
        ];
        let dists = zeckf64_batch(query, &edges);
        assert_eq!(dists[0], 0); // identical → 0
        assert!(dists[1] > dists[2]); // far > close
    }

    #[test]
    fn test_zeckf64_top_k() {
        let query = zeckf64_from_distances(1000, 2000, 3000);
        let edges = vec![
            zeckf64_from_distances(5000, 8000, 10000), // far
            zeckf64_from_distances(1000, 2000, 3000), // identical
            zeckf64_from_distances(1100, 2100, 3100), // close
            zeckf64_from_distances(8000, 8000, 8000), // very far
        ];
        let (indices, dists) = zeckf64_top_k(query, &edges, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 1); // identical is closest
        assert_eq!(dists[0], 0);
    }

    #[test]
    fn test_zeckf64_scent_top_k() {
        let query = zeckf64_from_distances(0, 0, 0); // all close
        let edges = vec![
            zeckf64_from_distances(D_MAX, D_MAX, D_MAX), // none close
            zeckf64_from_distances(0, 0, 0), // all close (match)
            zeckf64_from_distances(100, 100, 100), // all close (close)
        ];
        let (indices, _) = zeckf64_scent_top_k(query, &edges, 2);
        // Both edges 1 and 2 have all-close scent bytes
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn test_zeckf64_encode_batch() {
        let qs = vec![0xAAu8; 2048];
        let qp = vec![0xBBu8; 2048];
        let qo = vec![0xCCu8; 2048];

        // Database: 2 rows, each row = S(2048) + P(2048) + O(2048)
        let mut db = Vec::new();
        db.extend_from_slice(&qs); // row 0 S = query S (identical)
        db.extend_from_slice(&qp);
        db.extend_from_slice(&qo);
        db.extend_from_slice(&vec![0x00u8; 2048]); // row 1 S = zeros (different)
        db.extend_from_slice(&vec![0x00u8; 2048]);
        db.extend_from_slice(&vec![0x00u8; 2048]);

        let edges = zeckf64_encode_batch((&qs, &qp, &qo), &db, 2, 2048);
        assert_eq!(edges.len(), 2);
        // Row 0 identical → scent should be all close
        assert_eq!(scent(edges[0]) & 0x7F, 0x7F);
        // Row 1 different → scent should show some open bits
        assert_ne!(scent(edges[1]) & 0x7F, 0x7F);
    }
}
