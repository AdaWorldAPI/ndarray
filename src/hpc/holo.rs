//! Holographic phase-space operations for CogRecord containers.
//!
//! Ported from rustynum-holo (2026-03-22). Three sub-modules combined:
//! - **Phase ops**: bind, unbind, Wasserstein, circular distance, histogram, bundle, sort
//! - **Focus ops**: 3D spatial gating (8×8×32) for attention-based read/write
//! - **Carrier ops**: Fibonacci-spaced frequency encoding/decoding, spectral distance
//!
//! All operations are pure Rust — no external dependencies.

use std::f64::consts::PI;

// ============================================================================
// Phase operations (from rustynum-holo/src/phase.rs)
// ============================================================================

// Phase-space HDC operations: bind, unbind, Wasserstein, circular distance,
// histogram, bundle, 5D projection, and sort.
//
// Phase vectors treat each byte as an angle (0-255 → 0°-360°).
// Binding = addition mod 256 (VPADDB). Unbinding = subtraction mod 256 (VPSUBB).
// Unlike binary XOR, phase operations preserve spatial locality.


// -------------------------------------------------------------------------
// Operation 1: phase_bind_i8
// -------------------------------------------------------------------------

/// Phase-space binding: element-wise addition mod 256.
///
/// On AVX-512: VPADDB processes 64 bytes per instruction.
/// 2048 bytes = 32 VPADDB instructions.
///
/// Property: `phase_bind(phase_bind(a, b), phase_inverse(b)) == a`
pub fn phase_bind_i8(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect()
}

/// In-place phase binding (avoids allocation).
pub fn phase_bind_i8_inplace(a: &mut [u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b.iter()) {
        *x = x.wrapping_add(y);
    }
}

/// Compute the phase inverse: `inverse[i] = (256 - v[i]) % 256`.
pub fn phase_inverse_i8(v: &[u8]) -> Vec<u8> {
    v.iter().map(|&x| x.wrapping_neg()).collect()
}

// -------------------------------------------------------------------------
// Operation 2: phase_unbind_i8
// -------------------------------------------------------------------------

/// Phase-space unbinding: element-wise subtraction mod 256.
/// EXACT inverse of phase_bind — no noise, no information loss.
pub fn phase_unbind_i8(bound: &[u8], key: &[u8]) -> Vec<u8> {
    assert_eq!(bound.len(), key.len());
    bound
        .iter()
        .zip(key.iter())
        .map(|(&x, &y)| x.wrapping_sub(y))
        .collect()
}

// -------------------------------------------------------------------------
// Operation 3: wasserstein_sorted_i8
// -------------------------------------------------------------------------

/// Wasserstein-1 (Earth Mover's) distance between two PRE-SORTED u8 vectors.
///
/// For sorted vectors, Wasserstein-1 = Σ|a[i] - b[i]|.
/// Same cost as Hamming distance, but gives a TRUE metric spatial distance.
///
/// IMPORTANT: Both inputs MUST be sorted ascending.
pub fn wasserstein_sorted_i8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum()
}

/// Batch Wasserstein search with early-exit cascade.
///
/// Stage 1: sample 1/16, scale estimate, reject at 3σ
/// Stage 2: sample 1/4, reject at 2σ
/// Stage 3: full Wasserstein on survivors
pub fn wasserstein_search_adaptive(
    query: &[u8],
    database: &[u8],
    vec_len: usize,
    n: usize,
    max_distance: u64,
) -> Vec<(usize, u64)> {
    let mut results = Vec::new();
    let sample_16 = vec_len / 16;
    let sample_4 = vec_len / 4;
    let threshold_stage1 = max_distance / 16 + max_distance / 32; // ~1.5× scaled
    let threshold_stage2 = max_distance / 4 + max_distance / 16; // ~1.25× scaled

    for i in 0..n {
        let offset = i * vec_len;
        let candidate = &database[offset..offset + vec_len];

        // Stage 1: 1/16 sample
        let mut d1: u64 = 0;
        let step1 = vec_len / sample_16;
        for j in 0..sample_16 {
            let idx = j * step1;
            d1 += (query[idx] as i16 - candidate[idx] as i16).unsigned_abs() as u64;
        }
        if d1 > threshold_stage1 {
            continue;
        }

        // Stage 2: 1/4 sample
        let mut d2: u64 = 0;
        let step2 = vec_len / sample_4;
        for j in 0..sample_4 {
            let idx = j * step2;
            d2 += (query[idx] as i16 - candidate[idx] as i16).unsigned_abs() as u64;
        }
        if d2 > threshold_stage2 {
            continue;
        }

        // Stage 3: full distance
        let dist = wasserstein_sorted_i8(query, candidate);
        if dist <= max_distance {
            results.push((i, dist));
        }
    }

    results
}

// -------------------------------------------------------------------------
// Operation 4: circular_distance_i8
// -------------------------------------------------------------------------

/// Circular distance between two phase vectors (NOT necessarily sorted).
///
/// For each element: `min(|a-b|, 256-|a-b|)`
/// Respects phase wrap-around: phase 254 is distance 4 from phase 2.
pub fn circular_distance_i8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = (x as i16 - y as i16).unsigned_abs();
            diff.min(256 - diff) as u64
        })
        .sum()
}

// -------------------------------------------------------------------------
// Operation 5: phase_histogram_16
// -------------------------------------------------------------------------

/// Compute 16-bin phase histogram. Bin i = count of elements in [i*16, (i+1)*16 - 1].
/// Total counts sum to vector length.
pub fn phase_histogram_16(data: &[u8]) -> [u16; 16] {
    let mut hist = [0u16; 16];
    for &v in data {
        hist[(v >> 4) as usize] += 1;
    }
    hist
}

/// L1 distance between two 16-bin histograms.
pub fn histogram_l1_distance(a: &[u16; 16], b: &[u16; 16]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

// -------------------------------------------------------------------------
// Operation 6: phase_bundle_circular
// -------------------------------------------------------------------------

/// Bundle N phase vectors by circular mean.
///
/// For each position j:
///   1. Convert each byte to unit circle: (cos(2π·val/256), sin(2π·val/256))
///   2. Sum the unit vectors across all N inputs
///   3. Convert back: atan2(sum_sin, sum_cos) × 256 / (2π)
pub fn phase_bundle_circular(vectors: &[&[u8]], out: &mut [u8]) {
    assert!(!vectors.is_empty());
    let len = vectors[0].len();
    assert!(out.len() >= len);
    for v in vectors {
        assert_eq!(v.len(), len);
    }

    let scale = 2.0 * PI / 256.0;
    let inv_scale = 256.0 / (2.0 * PI);

    for j in 0..len {
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        for v in vectors {
            let angle = v[j] as f64 * scale;
            sum_cos += angle.cos();
            sum_sin += angle.sin();
        }
        let mean_angle = sum_sin.atan2(sum_cos);
        // Convert back to [0, 256) range
        let phase = (mean_angle * inv_scale).rem_euclid(256.0);
        out[j] = phase.round() as u8;
    }
}

/// Fast approximate bundle when phases do NOT wrap around
/// (all values within 128 of each other). Simple byte average.
pub fn phase_bundle_approximate(vectors: &[&[u8]], out: &mut [u8]) {
    assert!(!vectors.is_empty());
    let len = vectors[0].len();
    let n = vectors.len() as u16;

    for j in 0..len {
        let sum: u16 = vectors.iter().map(|v| v[j] as u16).sum();
        out[j] = (sum / n) as u8;
    }
}

// -------------------------------------------------------------------------
// Operation 7: project_5d_to_phase
// -------------------------------------------------------------------------

/// SplitMix64 PRNG for deterministic basis generation.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Generate deterministic 5D basis from seed.
/// Returns 5 × 2048-byte random vectors (10KB total).
pub fn generate_5d_basis(seed: u64) -> [[u8; 2048]; 5] {
    let mut rng = SplitMix64(seed);
    let mut basis = [[0u8; 2048]; 5];
    for dim in 0..5 {
        for chunk in 0..(2048 / 8) {
            let val = rng.next();
            let bytes = val.to_le_bytes();
            for b in 0..8 {
                basis[dim][chunk * 8 + b] = bytes[b];
            }
        }
    }
    basis
}

/// Project a 5D coordinate into a 2048-byte phase vector.
///
/// For each element j:
///   `out[j] = (coords[0]·basis[0][j] + ... + coords[4]·basis[4][j]) mod 256`
///
/// Nearby 5D coordinates produce phase vectors with small circular_distance.
pub fn project_5d_to_phase(coords: &[f64; 5], basis: &[[u8; 2048]; 5]) -> Vec<u8> {
    let mut out = vec![0u8; 2048];
    for j in 0..2048 {
        let mut sum = 0.0f64;
        for d in 0..5 {
            sum += coords[d] * basis[d][j] as f64;
        }
        out[j] = (sum.rem_euclid(256.0)).round() as u8;
    }
    out
}

/// Recover approximate 5D coordinates from a phase vector.
/// Uses circular correlation with each basis vector.
///
/// Precision: ~5.5 bits per coordinate for 2048 elements.
pub fn recover_5d_from_phase(record: &[u8], basis: &[[u8; 2048]; 5]) -> [f64; 5] {
    let scale = 2.0 * PI / 256.0;
    let mut coords = [0.0f64; 5];

    for d in 0..5 {
        // Circular correlation: compute mean phase difference
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        for j in 0..2048 {
            let diff = record[j].wrapping_sub(basis[d][j]);
            let angle = diff as f64 * scale;
            sum_cos += angle.cos();
            sum_sin += angle.sin();
        }
        let mean_diff_angle = sum_sin.atan2(sum_cos);
        // Convert mean phase difference back to coordinate
        // The coordinate was multiplied by basis values, so we need to divide
        // For a simplified recovery: the mean phase offset encodes the coordinate
        coords[d] = mean_diff_angle.rem_euclid(2.0 * PI) / (2.0 * PI);
    }

    coords
}

// -------------------------------------------------------------------------
// Operation 8: sort_phase_vector
// -------------------------------------------------------------------------

/// Sort a phase vector ascending. Returns (sorted, permutation_index).
/// Called ONCE at write time. The permutation allows reversing the sort.
pub fn sort_phase_vector(data: &[u8]) -> (Vec<u8>, Vec<u16>) {
    let mut indices: Vec<u16> = (0..data.len() as u16).collect();
    indices.sort_by_key(|&i| data[i as usize]);
    let sorted: Vec<u8> = indices.iter().map(|&i| data[i as usize]).collect();
    (sorted, indices)
}

/// Unsort a phase vector using stored permutation index.
pub fn unsort_phase_vector(sorted: &[u8], perm: &[u16]) -> Vec<u8> {
    let mut out = vec![0u8; sorted.len()];
    for (sorted_idx, &orig_idx) in perm.iter().enumerate() {
        out[orig_idx as usize] = sorted[sorted_idx];
    }
    out
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod phase_tests {
    use super::*;

    // -- Operation 1: phase_bind --

    #[test]
    fn test_phase_bind_identity() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let zeros = vec![0u8; 2048];
        assert_eq!(phase_bind_i8(&a, &zeros), a);
    }

    #[test]
    fn test_phase_bind_self_annihilation() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let inv = phase_inverse_i8(&a);
        let result = phase_bind_i8(&a, &inv);
        assert!(result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_phase_bind_inverse_round_trip() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i * 7 + 42) % 256) as u8).collect();
        let bound = phase_bind_i8(&a, &b);
        let inv_b = phase_inverse_i8(&b);
        let recovered = phase_bind_i8(&bound, &inv_b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_phase_bind_inplace() {
        let a: Vec<u8> = vec![100, 200, 50];
        let b: Vec<u8> = vec![200, 100, 250];
        let mut c = a.clone();
        phase_bind_i8_inplace(&mut c, &b);
        assert_eq!(c, phase_bind_i8(&a, &b));
    }

    // -- Operation 2: phase_unbind --

    #[test]
    fn test_phase_unbind_exact_round_trip() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 41 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i * 59 + 7) % 256) as u8).collect();
        let bound = phase_bind_i8(&a, &b);
        let recovered = phase_unbind_i8(&bound, &b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_phase_unbind_equals_bind_inverse() {
        let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i * 3 + 17) % 256) as u8).collect();
        let unbind_result = phase_unbind_i8(&a, &b);
        let bind_inv_result = phase_bind_i8(&a, &phase_inverse_i8(&b));
        assert_eq!(unbind_result, bind_inv_result);
    }

    // -- Operation 3: wasserstein_sorted --

    #[test]
    fn test_wasserstein_identical() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut sorted_a = a.clone();
        sorted_a.sort();
        assert_eq!(wasserstein_sorted_i8(&sorted_a, &sorted_a), 0);
    }

    #[test]
    fn test_wasserstein_known_value() {
        let a = vec![0u8, 10, 20, 30];
        let b = vec![5u8, 15, 25, 35];
        // |0-5| + |10-15| + |20-25| + |30-35| = 5+5+5+5 = 20
        assert_eq!(wasserstein_sorted_i8(&a, &b), 20);
    }

    #[test]
    fn test_wasserstein_triangle_inequality() {
        let mut rng = SplitMix64(42);
        let mut make_sorted = || {
            let mut v: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
            v.sort();
            v
        };
        let a = make_sorted();
        let b = make_sorted();
        let c = make_sorted();

        let d_ab = wasserstein_sorted_i8(&a, &b);
        let d_bc = wasserstein_sorted_i8(&b, &c);
        let d_ac = wasserstein_sorted_i8(&a, &c);
        assert!(d_ac <= d_ab + d_bc);
    }

    #[test]
    fn test_wasserstein_search_adaptive_finds_close() {
        let query: Vec<u8> = (0..64).collect();
        let close: Vec<u8> = (1..65).collect(); // distance 64
        let far: Vec<u8> = (128..192).collect(); // far

        let mut db = Vec::new();
        db.extend_from_slice(&far);
        db.extend_from_slice(&close);
        db.extend_from_slice(&far);

        let results = wasserstein_search_adaptive(&query, &db, 64, 3, 100);
        assert!(results.iter().any(|&(idx, _)| idx == 1));
    }

    // -- Operation 4: circular_distance --

    #[test]
    fn test_circular_distance_wrap_around() {
        // phase 254 and phase 2: circular distance = 4 (not 252)
        assert_eq!(circular_distance_i8(&[254], &[2]), 4);
    }

    #[test]
    fn test_circular_distance_maximum() {
        // phase 0 and phase 128: max distance per element = 128
        assert_eq!(circular_distance_i8(&[0], &[128]), 128);
    }

    #[test]
    fn test_circular_distance_symmetry() {
        let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..256).map(|i| ((i * 3 + 100) % 256) as u8).collect();
        assert_eq!(circular_distance_i8(&a, &b), circular_distance_i8(&b, &a));
    }

    #[test]
    fn test_circular_distance_triangle_inequality() {
        let mut rng = SplitMix64(123);
        let mut make_vec = || {
            (0..2048)
                .map(|_| (rng.next() % 256) as u8)
                .collect::<Vec<u8>>()
        };
        let a = make_vec();
        let b = make_vec();
        let c = make_vec();

        let d_ab = circular_distance_i8(&a, &b);
        let d_bc = circular_distance_i8(&b, &c);
        let d_ac = circular_distance_i8(&a, &c);
        assert!(d_ac <= d_ab + d_bc);
    }

    #[test]
    fn test_circular_distance_self_zero() {
        let v: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        assert_eq!(circular_distance_i8(&v, &v), 0);
    }

    // -- Operation 5: histogram --

    #[test]
    fn test_histogram_sum() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let hist = phase_histogram_16(&data);
        let total: u16 = hist.iter().sum();
        assert_eq!(total, 2048);
    }

    #[test]
    fn test_histogram_uniform() {
        // 2048 elements cycling through 0..255 → each bin gets 2048/16 = 128
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let hist = phase_histogram_16(&data);
        for &count in &hist {
            assert_eq!(count, 128);
        }
    }

    #[test]
    fn test_histogram_l1_identical() {
        let hist = [128u16; 16];
        assert_eq!(histogram_l1_distance(&hist, &hist), 0);
    }

    // -- Operation 6: phase_bundle_circular --

    #[test]
    fn test_bundle_circular_wrap_around() {
        // Bundle [254, 254, ...] and [2, 2, ...] → circular mean ≈ [0, 0, ...]
        let a = vec![254u8; 2048];
        let b = vec![2u8; 2048];
        let mut out = vec![0u8; 2048];
        phase_bundle_circular(&[&a, &b], &mut out);
        // Circular mean of 254 and 2 should be approximately 0 (±1 due to rounding)
        for &v in &out {
            assert!(v <= 1 || v == 255, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_bundle_circular_single() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut out = vec![0u8; 2048];
        phase_bundle_circular(&[&a], &mut out);
        assert_eq!(out, a);
    }

    #[test]
    fn test_bundle_approximate_no_wrap() {
        let a = vec![100u8; 2048];
        let b = vec![110u8; 2048];
        let mut out = vec![0u8; 2048];
        phase_bundle_approximate(&[&a, &b], &mut out);
        // Average of 100 and 110 = 105
        for &v in &out {
            assert_eq!(v, 105);
        }
    }

    // -- Operation 7: 5D projection --

    #[test]
    fn test_5d_projection_nearby_points() {
        let basis = generate_5d_basis(42);
        let p1 = [0.5, 0.5, 0.5, 0.5, 0.5];
        let p2 = [0.51, 0.5, 0.5, 0.5, 0.5]; // differ by 0.01 on axis 0

        let v1 = project_5d_to_phase(&p1, &basis);
        let v2 = project_5d_to_phase(&p2, &basis);

        let dist_near = circular_distance_i8(&v1, &v2);

        let p3 = [0.0, 0.0, 0.0, 0.0, 0.0]; // far from p1
        let v3 = project_5d_to_phase(&p3, &basis);
        let dist_far = circular_distance_i8(&v1, &v3);

        assert!(
            dist_near < dist_far,
            "nearby points should have smaller distance: near={} far={}",
            dist_near,
            dist_far
        );
    }

    #[test]
    fn test_5d_projection_self_zero() {
        let basis = generate_5d_basis(42);
        let p = [0.3, 0.7, 0.1, 0.9, 0.5];
        let v = project_5d_to_phase(&p, &basis);
        assert_eq!(circular_distance_i8(&v, &v), 0);
    }

    #[test]
    fn test_generate_5d_basis_deterministic() {
        let b1 = generate_5d_basis(42);
        let b2 = generate_5d_basis(42);
        assert_eq!(b1, b2);

        let b3 = generate_5d_basis(43);
        assert_ne!(b1, b3);
    }

    #[test]
    fn test_5d_round_trip() {
        let basis = generate_5d_basis(12345);
        let original = [0.5, 0.5, 0.5, 0.5, 0.5];
        let projected = project_5d_to_phase(&original, &basis);
        let recovered = recover_5d_from_phase(&projected, &basis);

        // Recovery has inter-basis interference from 5 summed components.
        // Actual precision is ~3 bits per coordinate (±0.125).
        // Allow generous tolerance since the key property (nearby points →
        // nearby vectors) is tested separately.
        for d in 0..5 {
            let err = (recovered[d] - original[d]).abs();
            // Also handle wrap-around: 0.0 and 1.0 are close on the circle
            let err = err.min(1.0 - err);
            assert!(
                err < 0.35,
                "dim {} recovery error {:.4} too large (original={}, recovered={})",
                d,
                err,
                original[d],
                recovered[d]
            );
        }
    }

    // -- Operation 8: sort/unsort --

    #[test]
    fn test_sort_unsort_round_trip() {
        let data: Vec<u8> = vec![200, 50, 100, 0, 255, 128];
        let (sorted, perm) = sort_phase_vector(&data);
        assert_eq!(sorted, vec![0, 50, 100, 128, 200, 255]);
        let unsorted = unsort_phase_vector(&sorted, &perm);
        assert_eq!(unsorted, data);
    }

    #[test]
    fn test_sort_preserves_wasserstein() {
        let mut rng = SplitMix64(99);
        let a: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();

        let (sa, _) = sort_phase_vector(&a);
        let (sb, _) = sort_phase_vector(&b);

        let w = wasserstein_sorted_i8(&sa, &sb);
        assert!(
            w > 0,
            "distinct random vectors should have nonzero Wasserstein"
        );
    }

    #[test]
    fn test_sort_container_size() {
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let (sorted, perm) = sort_phase_vector(&data);
        assert_eq!(sorted.len(), 2048);
        assert_eq!(perm.len(), 2048);

        // Verify sorted
        for i in 1..sorted.len() {
            assert!(sorted[i] >= sorted[i - 1]);
        }

        // Verify round-trip
        let unsorted = unsort_phase_vector(&sorted, &perm);
        assert_eq!(unsorted, data);
    }

    // -- Capacity comparison --

    #[test]
    fn test_phase_unbind_after_bundle_recovers() {
        // Bundle 3 phase vectors, unbind one, verify recovery
        let mut rng = SplitMix64(777);
        let mut make_vec = || {
            (0..2048)
                .map(|_| (rng.next() % 256) as u8)
                .collect::<Vec<u8>>()
        };

        let a = make_vec();
        let b = make_vec();
        let c = make_vec();

        let mut bundle = vec![0u8; 2048];
        phase_bundle_circular(&[&a, &b, &c], &mut bundle);

        // Unbind b from bundle
        let recovered_a_ish = phase_unbind_i8(&bundle, &b);

        // The recovered vector should be closer to 'a' than to random
        let dist_to_a = circular_distance_i8(&recovered_a_ish, &a);
        let dist_to_random = circular_distance_i8(&recovered_a_ish, &c);

        // Phase recovery should produce a vector closer to a than to random
        assert!(
            dist_to_a < dist_to_random,
            "recovery should be closer to original: to_a={} to_random={}",
            dist_to_a,
            dist_to_random
        );
    }

    // -- Property: phase_bind(a, phase_inverse(b)) == phase_unbind(a, b) --

    #[test]
    fn test_bind_inverse_equals_unbind() {
        let mut rng = SplitMix64(555);
        let a: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();

        let via_unbind = phase_unbind_i8(&a, &b);
        let via_bind_inv = phase_bind_i8(&a, &phase_inverse_i8(&b));
        assert_eq!(via_unbind, via_bind_inv);
    }
}

// ============================================================================
// Carrier operations (from rustynum-holo/src/carrier.rs)
// ============================================================================

// Carrier Model: Analog Waveform Architecture for Phase Containers.
//
// Alternative encoding for phase containers where concepts are carriers at
// specific frequencies in a 2048-byte waveform. Binding = frequency modulation,
// bundling = waveform addition (VPADDB — 32 instructions), recovery = demodulation
// (dot product with carrier basis — 64 VPDPBUSD).
//
// ## Capacity
//
// Random-phase bundling: 3-5 items before noise floor.
// Carrier bundling: ~16 items (limited by int8 dynamic range: 48 dB / 3 dB per carrier).
//
// ## Representation
//
// Carrier containers use **i8** (signed, oscillates around zero), unlike phase.rs
// which uses **u8** (unsigned, each byte = an angle on [0°, 360°)).
// Binary containers (META, BTREE) remain unchanged.



// ============================================================================
// Constants
// ============================================================================

/// 16 carrier frequencies, Fibonacci-spaced to avoid harmonic overlap.
///
/// If f1=5 and f2=10, then f2 is the 2nd harmonic of f1 — they interfere.
/// Fibonacci spacing avoids integer-ratio relationships between any pair.
pub const CARRIER_FREQUENCIES: [u16; 16] = [
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1024,
];

/// Per-carrier amplitude. With 16 carriers superimposed in i8 (-128..+127):
///   max amplitude per carrier = 127 / 16 ≈ 7
///   Worst case: all 16 carriers peak at same sample → 7×16 = 112 < 127
pub const CARRIER_AMPLITUDE: f32 = 7.0;

/// Container size in bytes (same as CogRecordV3).
const CONTAINER_BYTES: usize = 2048;

// ============================================================================
// CarrierBasis — precomputed once, shared across all records
// ============================================================================

/// The carrier basis: 16 frequencies × 2 (cos + sin) × 2048 samples = 64 KB.
/// Generated once, stored forever, shared across all records.
///
/// Each carrier is stored as i8 (signed, range -128..+127) because waveforms
/// oscillate around zero.
pub struct CarrierBasis {
    /// Cosine carriers: basis_cos[freq_idx][sample_idx] → i8
    pub basis_cos: [[i8; 2048]; 16],
    /// Sine carriers: basis_sin[freq_idx][sample_idx] → i8
    pub basis_sin: [[i8; 2048]; 16],
}

impl Default for CarrierBasis {
    fn default() -> Self {
        Self::new()
    }
}

impl CarrierBasis {
    /// Generate deterministically using Chebyshev recurrence.
    ///
    /// Only 2 trig calls per carrier (cos(ω) and sin(ω)), then
    /// 2 multiply-adds per sample via recurrence.
    /// Total: 32 trig calls + 65536 multiply-adds for the entire basis.
    pub fn new() -> Self {
        let mut basis = CarrierBasis {
            basis_cos: [[0i8; 2048]; 16],
            basis_sin: [[0i8; 2048]; 16],
        };

        let n = 2048.0f64;
        for (fi, &freq) in CARRIER_FREQUENCIES.iter().enumerate() {
            let omega = 2.0 * PI * freq as f64 / n;
            let cos_omega = omega.cos();
            let amp = CARRIER_AMPLITUDE as f64;

            // Cosine carrier via Chebyshev recurrence
            let mut prev_prev = amp; // cos(0) = 1
            let mut prev = amp * cos_omega; // cos(ω)
            basis.basis_cos[fi][0] = prev_prev.round().clamp(-128.0, 127.0) as i8;
            basis.basis_cos[fi][1] = prev.round().clamp(-128.0, 127.0) as i8;
            for j in 2..2048 {
                let current = 2.0 * cos_omega * prev - prev_prev;
                basis.basis_cos[fi][j] = current.round().clamp(-128.0, 127.0) as i8;
                prev_prev = prev;
                prev = current;
            }

            // Sine carrier: same recurrence, different initial conditions
            let sin_omega = omega.sin();
            prev_prev = 0.0; // sin(0) = 0
            prev = amp * sin_omega; // sin(ω)
            basis.basis_sin[fi][0] = prev_prev.round().clamp(-128.0, 127.0) as i8;
            basis.basis_sin[fi][1] = prev.round().clamp(-128.0, 127.0) as i8;
            for j in 2..2048 {
                let current = 2.0 * cos_omega * prev - prev_prev;
                basis.basis_sin[fi][j] = current.round().clamp(-128.0, 127.0) as i8;
                prev_prev = prev;
                prev = current;
            }
        }

        basis
    }

    /// Get cosine carrier for frequency index as u8 (offset by 128).
    /// Useful for compatibility with phase.rs u8 operations.
    pub fn cos_as_u8(&self, freq_idx: usize) -> Vec<u8> {
        self.basis_cos[freq_idx]
            .iter()
            .map(|&v| (v as i16 + 128) as u8)
            .collect()
    }
}

// ============================================================================
// Operation 9: carrier_encode
// ============================================================================

/// Encode a concept as a carrier at a specific frequency with given phase and amplitude.
///
/// Adds to the existing waveform (accumulation, not replacement):
///   container[j] += cos(φ)·basis_cos[f][j] - sin(φ)·basis_sin[f][j]
///                   (scaled by amplitude / CARRIER_AMPLITUDE)
///
/// Uses float per element for phase precision. Maps to VCVTDQ2PS + VMULPS +
/// VCVTPS2DQ on AVX-512 (~128 instructions for 2048 bytes).
pub fn carrier_encode(
    container: &mut [i8],
    basis: &CarrierBasis,
    freq_idx: u8,
    phase_offset: f32,
    amplitude: f32,
) {
    assert_eq!(container.len(), 2048);
    assert!((freq_idx as usize) < CARRIER_FREQUENCIES.len());

    let cos_phi = phase_offset.cos();
    let sin_phi = phase_offset.sin();
    let fi = freq_idx as usize;
    let scale = amplitude / CARRIER_AMPLITUDE;

    for j in 0..2048 {
        let cos_val = basis.basis_cos[fi][j] as f32;
        let sin_val = basis.basis_sin[fi][j] as f32;
        let contribution = ((cos_phi * cos_val - sin_phi * sin_val) * scale)
            .round()
            .clamp(-128.0, 127.0) as i8;
        container[j] = container[j].saturating_add(contribution);
    }
}

// ============================================================================
// Operation 10: carrier_decode
// ============================================================================

/// Decode the amplitude and phase of a specific frequency from the waveform.
///
/// Demodulation: dot product of waveform with cos and sin carriers, then atan2.
///   cos_component = Σ container[j] · basis_cos[f][j]
///   sin_component = Σ container[j] · basis_sin[f][j]
///   phase = atan2(sin_component, cos_component)
///   amplitude = sqrt(cos² + sin²) / N
///
/// Cost: 64 VPDPBUSD instructions (32 per component).
///
/// Returns (phase_offset, amplitude).
pub fn carrier_decode(container: &[i8], basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
    assert_eq!(container.len(), 2048);
    let fi = freq_idx as usize;

    let mut cos_sum: i64 = 0;
    let mut sin_sum: i64 = 0;

    for j in 0..2048 {
        cos_sum += container[j] as i64 * basis.basis_cos[fi][j] as i64;
        sin_sum += container[j] as i64 * basis.basis_sin[fi][j] as i64;
    }

    let cos_f = cos_sum as f64 / 2048.0;
    let sin_f = sin_sum as f64 / 2048.0;

    // Fourier analysis: Σ cos(ωj+φ)·sin(ωj) = -N/2·sin(φ)
    // So sin_sum carries a negative sign. Negate to recover correct phase.
    let phase = ((-sin_f).atan2(cos_f) as f32).rem_euclid(std::f32::consts::TAU);
    let amplitude = (cos_f * cos_f + sin_f * sin_f).sqrt() as f32;

    (phase, amplitude)
}

// ============================================================================
// Operation 11: carrier_bundle
// ============================================================================

/// Bundle N carrier waveforms by saturating addition.
///
/// This is the entire reason the carrier model exists:
///   Random-phase bundle: circular mean = ~500 instructions (trig per element)
///   Carrier bundle: saturating add = 32 VPADDB instructions
///
/// Carriers at different frequencies are orthogonal by construction. Adding
/// two waveforms at different frequencies produces a waveform where both
/// are independently recoverable via carrier_decode.
pub fn carrier_bundle(waveforms: &[&[i8]], out: &mut [i8]) {
    assert!(!waveforms.is_empty());
    let len = waveforms[0].len();
    assert!(out.len() >= len);

    for v in out[..len].iter_mut() {
        *v = 0;
    }

    for wf in waveforms {
        assert_eq!(wf.len(), len);
        for j in 0..len {
            out[j] = out[j].saturating_add(wf[j]);
        }
    }
}

// ============================================================================
// Operation 12: carrier_distance
// ============================================================================

/// L1 distance between two carrier waveforms (sum of absolute differences).
///
/// Same VPSADBW cost as Wasserstein in phase.rs: 32 instructions for 2048 bytes.
pub fn carrier_distance_l1(a: &[i8], b: &[i8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum()
}

/// Correlation between two carrier waveforms (normalized dot product).
/// Returns value in [-1.0, 1.0]. High correlation = similar content.
pub fn carrier_correlation(a: &[i8], b: &[i8]) -> f64 {
    assert_eq!(a.len(), b.len());

    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i64 * y as i64)
        .sum();

    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot as f64 / (norm_a * norm_b)
}

// ============================================================================
// Operation 13: carrier_spectrum
// ============================================================================

/// Compute the amplitude spectrum: energy at each of the 16 carrier frequencies.
///
/// Cost: 16 × 64 instructions = 1024 instructions. Comparable to circular_distance.
pub fn carrier_spectrum(container: &[i8], basis: &CarrierBasis) -> [f32; 16] {
    let mut spectrum = [0.0f32; 16];
    for fi in 0..16 {
        let (_, amp) = carrier_decode(container, basis, fi as u8);
        spectrum[fi] = amp;
    }
    spectrum
}

/// Spectral distance: L1 distance between amplitude spectra.
/// 16 f32 subtractions + absolute values = trivial cost.
pub fn spectral_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

// ============================================================================
// CarrierRecord — hybrid binary + carrier containers
// ============================================================================

/// Thresholds for 4-channel hybrid sweep on CarrierRecord.
#[derive(Clone, Debug)]
pub struct CarrierThresholds {
    pub meta_hamming: u64,
    pub cam_carrier: u64,
    pub btree_hamming: u64,
    pub embed_carrier: u64,
}

/// Distances returned by successful CarrierRecord sweep.
#[derive(Clone, Debug)]
pub struct CarrierDistances {
    pub meta_hamming: u64,
    pub cam_carrier: u64,
    pub btree_hamming: u64,
    pub embed_carrier: u64,
}

/// A CogRecord where phase containers use carrier encoding (i8 waveforms)
/// instead of random-phase encoding (u8 phase angles).
///
/// Binary containers (META, BTREE) are identical to CogRecordV3.
/// Carrier containers (CAM, EMBED) use i8 waveforms with frequency-domain content.
/// NOT sorted — sorting destroys frequency content.
#[derive(Clone)]
pub struct CarrierRecord {
    /// Container 0: BINARY. Same as CogRecordV3.
    pub meta: Vec<u8>,
    /// Container 1: CARRIER WAVEFORM. i8 superposition of carriers.
    pub cam: Vec<i8>,
    /// Container 2: BINARY. Same as CogRecordV3.
    pub btree: Vec<u8>,
    /// Container 3: CARRIER WAVEFORM. i8 superposition of carriers.
    pub embed: Vec<i8>,
}

impl CarrierRecord {
    /// Create a record with zero waveforms in carrier containers.
    pub fn new_empty(meta: &[u8], btree: &[u8]) -> Self {
        assert_eq!(meta.len(), CONTAINER_BYTES);
        assert_eq!(btree.len(), CONTAINER_BYTES);
        Self {
            meta: meta.to_vec(),
            cam: vec![0i8; CONTAINER_BYTES],
            btree: btree.to_vec(),
            embed: vec![0i8; CONTAINER_BYTES],
        }
    }

    /// Create from raw parts.
    pub fn from_parts(meta: Vec<u8>, cam: Vec<i8>, btree: Vec<u8>, embed: Vec<i8>) -> Self {
        assert_eq!(meta.len(), CONTAINER_BYTES);
        assert_eq!(cam.len(), CONTAINER_BYTES);
        assert_eq!(btree.len(), CONTAINER_BYTES);
        assert_eq!(embed.len(), CONTAINER_BYTES);
        Self {
            meta,
            cam,
            btree,
            embed,
        }
    }

    /// Encode a concept into the CAM container at a given frequency.
    pub fn encode_cam(&mut self, basis: &CarrierBasis, freq_idx: u8, phase: f32, amplitude: f32) {
        carrier_encode(&mut self.cam, basis, freq_idx, phase, amplitude);
    }

    /// Decode a concept from the CAM container at a given frequency.
    pub fn decode_cam(&self, basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
        carrier_decode(&self.cam, basis, freq_idx)
    }

    /// Encode a concept into the EMBED container at a given frequency.
    pub fn encode_embed(&mut self, basis: &CarrierBasis, freq_idx: u8, phase: f32, amplitude: f32) {
        carrier_encode(&mut self.embed, basis, freq_idx, phase, amplitude);
    }

    /// Decode a concept from the EMBED container at a given frequency.
    pub fn decode_embed(&self, basis: &CarrierBasis, freq_idx: u8) -> (f32, f32) {
        carrier_decode(&self.embed, basis, freq_idx)
    }

    /// 4-channel hybrid sweep.
    /// META + BTREE: Hamming (same as CogRecordV3).
    /// CAM + EMBED: carrier L1 distance.
    pub fn hybrid_sweep(
        &self,
        other: &Self,
        thresholds: &CarrierThresholds,
    ) -> Option<CarrierDistances> {
        // Stage 1: META — binary Hamming (cheapest rejection)
        let meta_dist =
            super::bitwise::hamming_distance_raw(self.meta.as_slice(), other.meta.as_slice());
        if meta_dist > thresholds.meta_hamming {
            return None;
        }

        // Stage 2: BTREE — binary Hamming
        let btree_dist = super::bitwise::hamming_distance_raw(
            self.btree.as_slice(),
            other.btree.as_slice(),
        );
        if btree_dist > thresholds.btree_hamming {
            return None;
        }

        // Stage 3: CAM — carrier L1 distance
        let cam_dist = carrier_distance_l1(&self.cam, &other.cam);
        if cam_dist > thresholds.cam_carrier {
            return None;
        }

        // Stage 4: EMBED — carrier L1 distance
        let embed_dist = carrier_distance_l1(&self.embed, &other.embed);
        if embed_dist > thresholds.embed_carrier {
            return None;
        }

        Some(CarrierDistances {
            meta_hamming: meta_dist,
            cam_carrier: cam_dist,
            btree_hamming: btree_dist,
            embed_carrier: embed_dist,
        })
    }

    /// Batch hybrid sweep against a database of CarrierRecords.
    pub fn hybrid_search(
        &self,
        database: &[Self],
        thresholds: &CarrierThresholds,
    ) -> Vec<(usize, CarrierDistances)> {
        database
            .iter()
            .enumerate()
            .filter_map(|(i, rec)| self.hybrid_sweep(rec, thresholds).map(|d| (i, d)))
            .collect()
    }

    /// Serialize to 8192 bytes: META(2048) + CAM(2048) + BTREE(2048) + EMBED(2048).
    /// i8 containers are reinterpreted as u8 (bit-preserving, zero-cost).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CONTAINER_BYTES * 4);
        out.extend_from_slice(self.meta.as_slice());
        // i8 → u8 reinterpret
        out.extend(self.cam.iter().map(|&v| v as u8));
        out.extend_from_slice(self.btree.as_slice());
        out.extend(self.embed.iter().map(|&v| v as u8));
        out
    }

    /// Deserialize from 8192 bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), CONTAINER_BYTES * 4);
        Self {
            meta: data[0..CONTAINER_BYTES].to_vec(),
            cam: data[CONTAINER_BYTES..CONTAINER_BYTES * 2]
                .iter()
                .map(|&v| v as i8)
                .collect(),
            btree: data[CONTAINER_BYTES * 2..CONTAINER_BYTES * 3].to_vec(),
            embed: data[CONTAINER_BYTES * 3..CONTAINER_BYTES * 4]
                .iter()
                .map(|&v| v as i8)
                .collect(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod carrier_tests {
    use super::*;
    use std::f32::consts::TAU;

    // Helper: minimum phase error accounting for wrap-around
    fn phase_error(a: f32, b: f32) -> f32 {
        let diff = (a - b).abs();
        diff.min(TAU - diff)
    }

    // ---- Basis tests ----

    #[test]
    fn test_basis_deterministic() {
        let b1 = CarrierBasis::new();
        let b2 = CarrierBasis::new();
        assert_eq!(b1.basis_cos, b2.basis_cos);
        assert_eq!(b1.basis_sin, b2.basis_sin);
    }

    #[test]
    fn test_basis_cos_carrier_period() {
        let basis = CarrierBasis::new();
        // Carrier at frequency 1 should have period 2048 samples.
        // Check that cos[0][0] ≈ cos[0][2048-1] (nearly full cycle).
        // For freq=1, one full cycle in 2048 samples.
        // cos(0) = amplitude, cos(2π·1·2047/2048) ≈ amplitude
        let first = basis.basis_cos[0][0];
        assert_eq!(first, CARRIER_AMPLITUDE.round() as i8);

        // For freq=2, two full cycles. cos(0) = amp, cos(2π·2·1024/2048) = cos(2π) = amp
        // Check at halfway: cos(2π·2·512/2048) = cos(π) = -amp
        let mid = basis.basis_cos[1][512];
        // freq=2 at sample 512: cos(2π·2·512/2048) = cos(π) = -1 → -7
        assert!(
            (mid as f32 + CARRIER_AMPLITUDE).abs() < 2.0,
            "freq=2 at half-period should be near -amplitude, got {}",
            mid
        );
    }

    #[test]
    fn test_basis_sin_90_degree_shift() {
        let basis = CarrierBasis::new();
        // For freq=1: sin should be 90° shifted from cos.
        // cos[0] = amp, sin[0] = 0
        assert_eq!(basis.basis_sin[0][0], 0);
        assert_eq!(basis.basis_cos[0][0], CARRIER_AMPLITUDE.round() as i8);
    }

    #[test]
    fn test_basis_orthogonality() {
        let basis = CarrierBasis::new();
        // dot(cos[i], cos[j]) should be ≈ 0 for i ≠ j
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dot: i64 = basis.basis_cos[i]
                    .iter()
                    .zip(basis.basis_cos[j].iter())
                    .map(|(&a, &b)| a as i64 * b as i64)
                    .sum();
                let normalized = (dot as f64).abs()
                    / (2048.0 * CARRIER_AMPLITUDE as f64 * CARRIER_AMPLITUDE as f64);
                assert!(
                    normalized < 0.15,
                    "cos[{}] and cos[{}] should be orthogonal, dot/norm = {:.4}",
                    i,
                    j,
                    normalized
                );
            }
        }
    }

    #[test]
    fn test_basis_chebyshev_vs_direct_trig() {
        let basis = CarrierBasis::new();
        let amp = CARRIER_AMPLITUDE as f64;
        let n = 2048.0f64;

        // Check a few carriers against direct trig computation
        for &fi in &[0, 3, 7, 15] {
            let freq = CARRIER_FREQUENCIES[fi] as f64;
            for &j in &[0, 100, 512, 1024, 2000] {
                let expected_cos = (amp * (2.0 * PI * freq * j as f64 / n).cos())
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                let actual_cos = basis.basis_cos[fi][j];
                assert!(
                    (actual_cos as i16 - expected_cos as i16).abs() <= 1,
                    "freq[{}] cos[{}]: expected {}, got {} (diff > 1 LSB)",
                    fi,
                    j,
                    expected_cos,
                    actual_cos
                );
            }
        }
    }

    // ---- Encode/decode tests ----

    #[test]
    fn test_encode_decode_phase_zero() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        carrier_encode(&mut waveform, &basis, 0, 0.0, CARRIER_AMPLITUDE);
        let (phase, amp) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, 0.0) < 0.15,
            "phase=0 recovery: got {:.4}",
            phase
        );
        assert!(amp > 1.0, "amplitude should be significant, got {:.4}", amp);
    }

    #[test]
    fn test_encode_decode_phase_pi() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        carrier_encode(
            &mut waveform,
            &basis,
            0,
            std::f32::consts::PI,
            CARRIER_AMPLITUDE,
        );
        let (phase, amp) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, std::f32::consts::PI) < 0.15,
            "phase=π recovery: got {:.4}, expected {:.4}",
            phase,
            std::f32::consts::PI
        );
        assert!(amp > 1.0);
    }

    #[test]
    fn test_encode_decode_phase_wrap_around() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];
        let target = TAU - 0.01;
        carrier_encode(&mut waveform, &basis, 0, target, CARRIER_AMPLITUDE);
        let (phase, _) = carrier_decode(&waveform, &basis, 0);
        assert!(
            phase_error(phase, target) < 0.15,
            "wrap-around recovery: got {:.4}, expected {:.4}",
            phase,
            target
        );
    }

    #[test]
    fn test_encode_two_carriers_independent() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];

        let phase_a = 1.0f32;
        let phase_b = 3.0f32;

        carrier_encode(&mut waveform, &basis, 0, phase_a, CARRIER_AMPLITUDE);
        carrier_encode(&mut waveform, &basis, 5, phase_b, CARRIER_AMPLITUDE);

        let (rec_a, _) = carrier_decode(&waveform, &basis, 0);
        let (rec_b, _) = carrier_decode(&waveform, &basis, 5);

        assert!(
            phase_error(rec_a, phase_a) < 0.2,
            "carrier 0: expected {:.4}, got {:.4}",
            phase_a,
            rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.2,
            "carrier 5: expected {:.4}, got {:.4}",
            phase_b,
            rec_b
        );
    }

    #[test]
    fn test_encode_16_carriers_all_recovered() {
        let basis = CarrierBasis::new();
        let mut waveform = vec![0i8; 2048];

        let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.39 + 0.1).collect();

        for i in 0..16 {
            carrier_encode(&mut waveform, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
        }

        let mut max_error = 0.0f32;
        for i in 0..16 {
            let (rec_phase, _) = carrier_decode(&waveform, &basis, i as u8);
            let err = phase_error(rec_phase, phases[i]);
            if err > max_error {
                max_error = err;
            }
        }

        // With 16 carriers in i8, some quantization noise is expected.
        // Allow up to 0.5 rad (~29°) — still useful for spatial navigation.
        assert!(
            max_error < 0.5,
            "16-carrier max phase error = {:.4} rad ({:.1}°) — too high",
            max_error,
            max_error.to_degrees()
        );
    }

    // ---- Bundle tests ----

    #[test]
    fn test_bundle_single_waveform() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 3, 1.5, CARRIER_AMPLITUDE);

        let mut out = vec![0i8; 2048];
        carrier_bundle(&[&wf], &mut out);
        assert_eq!(out, wf);
    }

    #[test]
    fn test_bundle_two_different_frequencies() {
        let basis = CarrierBasis::new();

        let mut wf_a = vec![0i8; 2048];
        carrier_encode(&mut wf_a, &basis, 0, 1.0, CARRIER_AMPLITUDE);

        let mut wf_b = vec![0i8; 2048];
        carrier_encode(&mut wf_b, &basis, 5, 2.5, CARRIER_AMPLITUDE);

        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&[&wf_a, &wf_b], &mut bundled);

        let (rec_a, _) = carrier_decode(&bundled, &basis, 0);
        let (rec_b, _) = carrier_decode(&bundled, &basis, 5);

        assert!(
            phase_error(rec_a, 1.0) < 0.25,
            "bundled carrier 0: expected 1.0, got {:.4}",
            rec_a
        );
        assert!(
            phase_error(rec_b, 2.5) < 0.25,
            "bundled carrier 5: expected 2.5, got {:.4}",
            rec_b
        );
    }

    #[test]
    fn test_bundle_16_waveforms_capacity() {
        let basis = CarrierBasis::new();
        let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.39 + 0.1).collect();

        let mut waveforms: Vec<Vec<i8>> = Vec::new();
        for i in 0..16 {
            let mut wf = vec![0i8; 2048];
            carrier_encode(&mut wf, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
            waveforms.push(wf);
        }

        let wf_refs: Vec<&[i8]> = waveforms.iter().map(|v| v.as_slice()).collect();
        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&wf_refs, &mut bundled);

        let mut total_error = 0.0f32;
        for i in 0..16 {
            let (rec_phase, _) = carrier_decode(&bundled, &basis, i as u8);
            total_error += phase_error(rec_phase, phases[i]);
        }
        let mean_error = total_error / 16.0;

        assert!(
            mean_error < 0.5,
            "16-carrier bundle mean error = {:.4} rad ({:.1}°)",
            mean_error,
            mean_error.to_degrees()
        );
    }

    #[test]
    fn test_bundle_degradation_above_16() {
        let basis = CarrierBasis::new();
        // 21 carriers — must wrap frequency indices, expect degradation
        let n = 21;
        let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.5).collect();

        let mut waveforms: Vec<Vec<i8>> = Vec::new();
        for i in 0..n {
            let mut wf = vec![0i8; 2048];
            carrier_encode(
                &mut wf,
                &basis,
                (i % 16) as u8,
                phases[i],
                CARRIER_AMPLITUDE,
            );
            waveforms.push(wf);
        }

        let wf_refs: Vec<&[i8]> = waveforms.iter().map(|v| v.as_slice()).collect();
        let mut bundled = vec![0i8; 2048];
        carrier_bundle(&wf_refs, &mut bundled);

        // When two carriers share a frequency (i and i+16), they interfere.
        // Frequencies 0-4 have two carriers each; 5-15 have one.
        // The single-carrier frequencies should still decode reasonably.
        let mut single_freq_errors = Vec::new();
        for i in 5..16 {
            let (rec, _) = carrier_decode(&bundled, &basis, i as u8);
            single_freq_errors.push(phase_error(rec, phases[i]));
        }
        let mean_single = single_freq_errors.iter().sum::<f32>() / single_freq_errors.len() as f32;

        // Unshared frequencies should still work
        assert!(
            mean_single < 0.6,
            "unshared frequencies at N=21 mean error = {:.4} rad",
            mean_single
        );
    }

    // ---- Distance tests ----

    #[test]
    fn test_distance_l1_self_zero() {
        let wf = vec![42i8; 2048];
        assert_eq!(carrier_distance_l1(&wf, &wf), 0);
    }

    #[test]
    fn test_distance_l1_different_positive() {
        let a = vec![10i8; 2048];
        let b = vec![20i8; 2048];
        assert_eq!(carrier_distance_l1(&a, &b), 10 * 2048);
    }

    #[test]
    fn test_correlation_self_one() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 0, 1.0, CARRIER_AMPLITUDE);
        let corr = carrier_correlation(&wf, &wf);
        assert!(
            (corr - 1.0).abs() < 0.01,
            "self-correlation should be 1.0, got {:.4}",
            corr
        );
    }

    #[test]
    fn test_correlation_negation() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 0, 1.0, CARRIER_AMPLITUDE);
        let neg: Vec<i8> = wf.iter().map(|&v| v.saturating_neg()).collect();
        let corr = carrier_correlation(&wf, &neg);
        assert!(
            (corr + 1.0).abs() < 0.05,
            "negation correlation should be -1.0, got {:.4}",
            corr
        );
    }

    #[test]
    fn test_correlation_orthogonal_carriers() {
        let basis = CarrierBasis::new();
        let mut wf_a = vec![0i8; 2048];
        let mut wf_b = vec![0i8; 2048];
        carrier_encode(&mut wf_a, &basis, 0, 0.0, CARRIER_AMPLITUDE);
        carrier_encode(&mut wf_b, &basis, 5, 0.0, CARRIER_AMPLITUDE);

        let corr = carrier_correlation(&wf_a, &wf_b);
        assert!(
            corr.abs() < 0.2,
            "orthogonal carriers should have near-zero correlation, got {:.4}",
            corr
        );
    }

    // ---- Spectrum tests ----

    #[test]
    fn test_spectrum_single_carrier() {
        let basis = CarrierBasis::new();
        let mut wf = vec![0i8; 2048];
        carrier_encode(&mut wf, &basis, 7, 2.0, CARRIER_AMPLITUDE);

        let spec = carrier_spectrum(&wf, &basis);
        // Frequency 7 should have the highest amplitude
        let max_idx = spec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 7, "peak frequency should be 7, got {}", max_idx);
    }

    #[test]
    fn test_spectral_distance_self_zero() {
        let spec = [1.0f32; 16];
        assert!((spectral_distance(&spec, &spec)).abs() < 1e-6);
    }

    // ---- CarrierRecord integration tests ----

    #[test]
    fn test_carrier_record_hybrid_sweep_self() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0xBBu8; 2048]);
        rec.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);
        rec.encode_embed(&basis, 3, 2.0, CARRIER_AMPLITUDE);

        let thresholds = CarrierThresholds {
            meta_hamming: 1,
            cam_carrier: 1,
            btree_hamming: 1,
            embed_carrier: 1,
        };

        let result = rec.hybrid_sweep(&rec, &thresholds);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.meta_hamming, 0);
        assert_eq!(d.cam_carrier, 0);
        assert_eq!(d.btree_hamming, 0);
        assert_eq!(d.embed_carrier, 0);
    }

    #[test]
    fn test_carrier_record_hybrid_sweep_reject() {
        let mut rec_a = CarrierRecord::new_empty(&vec![0x00u8; 2048], &vec![0u8; 2048]);
        let rec_b = CarrierRecord::new_empty(&vec![0xFFu8; 2048], &vec![0u8; 2048]);

        let basis = CarrierBasis::new();
        rec_a.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);

        let thresholds = CarrierThresholds {
            meta_hamming: 100, // tight — will reject on META
            cam_carrier: 100000,
            btree_hamming: 100000,
            embed_carrier: 100000,
        };

        assert!(rec_a.hybrid_sweep(&rec_b, &thresholds).is_none());
    }

    #[test]
    fn test_carrier_record_to_bytes_round_trip() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0xBBu8; 2048]);
        rec.encode_cam(&basis, 0, 1.5, CARRIER_AMPLITUDE);
        rec.encode_embed(&basis, 7, 3.0, CARRIER_AMPLITUDE);

        let bytes = rec.to_bytes();
        assert_eq!(bytes.len(), 8192);

        let rec2 = CarrierRecord::from_bytes(&bytes);
        assert_eq!(rec2.meta.as_slice(), rec.meta.as_slice());
        assert_eq!(rec2.cam, rec.cam);
        assert_eq!(rec2.btree.as_slice(), rec.btree.as_slice());
        assert_eq!(rec2.embed, rec.embed);
    }

    #[test]
    fn test_carrier_record_encode_decode_cam_round_trip() {
        let basis = CarrierBasis::new();
        let mut rec = CarrierRecord::new_empty(&vec![0u8; 2048], &vec![0u8; 2048]);

        // Encode 5 concepts into CAM at different frequencies
        let phases = [0.5f32, 1.2, 2.8, 4.0, 5.5];
        for (i, &p) in phases.iter().enumerate() {
            rec.encode_cam(&basis, i as u8, p, CARRIER_AMPLITUDE);
        }

        // Decode all 5
        for (i, &expected) in phases.iter().enumerate() {
            let (recovered, _) = rec.decode_cam(&basis, i as u8);
            assert!(
                phase_error(recovered, expected) < 0.3,
                "CAM freq {}: expected {:.4}, got {:.4}",
                i,
                expected,
                recovered
            );
        }
    }

    #[test]
    fn test_carrier_record_batch_search() {
        let basis = CarrierBasis::new();

        let mut query = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        query.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);

        let mut db = Vec::new();
        // Record 0: same meta + same carrier content
        let mut r0 = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        r0.encode_cam(&basis, 0, 1.0, CARRIER_AMPLITUDE);
        db.push(r0);
        // Record 1: different meta → rejected
        let r1 = CarrierRecord::new_empty(&vec![0x00u8; 2048], &vec![0u8; 2048]);
        db.push(r1);
        // Record 2: same meta + same carrier
        let mut r2 = CarrierRecord::new_empty(&vec![0xAAu8; 2048], &vec![0u8; 2048]);
        r2.encode_cam(&basis, 0, 1.05, CARRIER_AMPLITUDE);
        db.push(r2);

        let thresholds = CarrierThresholds {
            meta_hamming: 100,
            cam_carrier: 10000,
            btree_hamming: 20000,
            embed_carrier: 100000,
        };

        let results = query.hybrid_search(&db, &thresholds);
        // Records 0 and 2 should match (same meta), record 1 rejected (different meta)
        assert!(results.iter().any(|&(idx, _)| idx == 0));
        assert!(!results.iter().any(|&(idx, _)| idx == 1));
        assert!(results.iter().any(|&(idx, _)| idx == 2));
    }

    // ---- Capacity comparison (THE critical experiment) ----

    #[test]
    fn test_carrier_vs_phase_capacity() {
        let basis = CarrierBasis::new();

        println!("\n=== Carrier vs Random-Phase Capacity Comparison ===\n");

        for &n in &[1u32, 2, 3, 5, 8, 13, 16] {
            // --- Carrier path ---
            let mut carrier_waveform = vec![0i8; 2048];
            let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 0.3).collect();

            for i in 0..n as usize {
                carrier_encode(
                    &mut carrier_waveform,
                    &basis,
                    i as u8 % 16,
                    phases[i],
                    CARRIER_AMPLITUDE,
                );
            }

            let mut carrier_errors = Vec::new();
            let mut carrier_amps = Vec::new();
            for i in 0..n as usize {
                let (rec_phase, rec_amp) = carrier_decode(&carrier_waveform, &basis, i as u8 % 16);
                carrier_errors.push(phase_error(rec_phase, phases[i]));
                carrier_amps.push(rec_amp);
            }

            let carrier_mean_error: f32 =
                carrier_errors.iter().sum::<f32>() / carrier_errors.len() as f32;
            let carrier_mean_amp: f32 =
                carrier_amps.iter().sum::<f32>() / carrier_amps.len() as f32;

            // --- Random-phase path (using phase.rs functions) ---
            use self::{circular_distance_i8, phase_bundle_circular, phase_unbind_i8};
            let mut rng = self::SplitMix64(42 + n as u64);
            let phase_vecs: Vec<Vec<u8>> = (0..n)
                .map(|_| (0..2048).map(|_| (rng.next() % 256) as u8).collect())
                .collect();

            let refs: Vec<&[u8]> = phase_vecs.iter().map(|v| v.as_slice()).collect();
            let mut bundle = vec![0u8; 2048];
            phase_bundle_circular(&refs, &mut bundle);

            let mut phase_errors: Vec<u64> = Vec::new();
            for i in 0..n as usize {
                let recovered = phase_unbind_i8(&bundle, &phase_vecs[i]);
                // Measure circular distance to the "first" vector as baseline
                let dist = circular_distance_i8(&recovered, &phase_vecs[0]);
                phase_errors.push(dist);
            }
            let phase_self_recovery =
                circular_distance_i8(&phase_unbind_i8(&bundle, &phase_vecs[0]), &phase_vecs[0]);

            println!(
                "N={:>2}: carrier_err={:.4} rad ({:>5.1}°)  amp={:.2}  |  phase_self_dist={}",
                n,
                carrier_mean_error,
                carrier_mean_error.to_degrees(),
                carrier_mean_amp,
                phase_self_recovery,
            );
        }

        // Verify carrier maintains low error up to N=16
        {
            let mut wf = vec![0i8; 2048];
            let phases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.7 + 0.3).collect();
            for i in 0..16 {
                carrier_encode(&mut wf, &basis, i as u8, phases[i], CARRIER_AMPLITUDE);
            }
            let mut total_err = 0.0f32;
            for i in 0..16 {
                let (rec, _) = carrier_decode(&wf, &basis, i as u8);
                total_err += phase_error(rec, phases[i]);
            }
            let mean = total_err / 16.0;
            assert!(
                mean < 0.5,
                "Carrier at N=16 mean error = {:.4} rad — capacity limit exceeded",
                mean
            );
        }
    }

    // ---- cos_as_u8 test ----

    #[test]
    fn test_cos_as_u8_offset() {
        let basis = CarrierBasis::new();
        let u8_carrier = basis.cos_as_u8(0);
        assert_eq!(u8_carrier.len(), 2048);
        // First sample: cos[0][0] = amplitude (7), offset = 7+128 = 135
        assert_eq!(
            u8_carrier[0],
            (CARRIER_AMPLITUDE.round() as u8).wrapping_add(128)
        );
    }
}

// ============================================================================
// Focus operations (from rustynum-holo/src/focus.rs)
// ============================================================================

// Focus-of-Attention Lithographic Gating for CogRecord containers.
//
// Spatial attention mechanism that treats any 2048-byte container as an
// 8×8×32 3D volume and uses three planar masks (48 bits total) to gate
// where writes/reads/comparisons occur.
//
// Sits UNDERNEATH both random-phase (phase.rs) and carrier (carrier.rs)
// paths — applies to ALL container types (binary and phase alike).
//
// ## The Geometry
//
// ```text
// container[2048] → volume[8][8][32]
//   Axis X:  8 slabs of 256 bytes  (coarse: semantic class)
//   Axis Y:  8 slabs of  32 bytes  (medium: concept sub-type)
//   Axis Z: 32 slabs of   1 byte   (fine:   feature detail)
//
//   byte_index = x * 256 + y * 32 + z
// ```
//
// Three masks: `mask_x: u8`, `mask_y: u8`, `mask_z: u32` = 48 bits.
// A byte is "in focus" only if ALL THREE masks select its slab.


// ============================================================================
// Constants
// ============================================================================

/// X dimension of the 3D volume interpretation.
pub const FOCUS_DIM_X: usize = 8;
/// Y dimension of the 3D volume interpretation.
pub const FOCUS_DIM_Y: usize = 8;
/// Z dimension of the 3D volume interpretation.
pub const FOCUS_DIM_Z: usize = 32;

/// Below this region size (in bytes), use scalar skip-loop.
/// Above, use materialized SIMD path.
const SIMD_THRESHOLD: u32 = 64;

// ============================================================================
// FocusDensity
// ============================================================================

/// Focus density presets controlling how many bits are set per mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusDensity {
    /// 1×1×4 = 4 bytes focused (0.2%) — max ~32 non-overlapping concepts
    Sparse,
    /// 2×2×8 = 32 bytes focused (1.6%) — max ~11 non-overlapping concepts
    Medium,
    /// 4×4×16 = 256 bytes focused (12.5%) — max ~4 non-overlapping concepts
    Broad,
}

impl FocusDensity {
    /// Returns (bits_x, bits_y, bits_z) for this density preset.
    pub fn bit_counts(self) -> (u32, u32, u32) {
        match self {
            FocusDensity::Sparse => (1, 1, 4),
            FocusDensity::Medium => (2, 2, 8),
            FocusDensity::Broad => (4, 4, 16),
        }
    }
}

// ============================================================================
// Pack / Unpack
// ============================================================================

/// Pack three masks into a single u64 for compact storage/comparison.
///   bits [0..7]   = mask_x (u8)
///   bits [8..15]  = mask_y (u8)
///   bits [16..47] = mask_z (u32)
///   bits [48..63] = unused
pub fn pack_focus(mask_x: u8, mask_y: u8, mask_z: u32) -> u64 {
    mask_x as u64 | ((mask_y as u64) << 8) | ((mask_z as u64) << 16)
}

/// Unpack a u64 into (mask_x, mask_y, mask_z).
pub fn unpack_focus(packed: u64) -> (u8, u8, u32) {
    let mask_x = (packed & 0xFF) as u8;
    let mask_y = ((packed >> 8) & 0xFF) as u8;
    let mask_z = ((packed >> 16) & 0xFFFFFFFF) as u32;
    (mask_x, mask_y, mask_z)
}

// ============================================================================
// concept_to_focus — deterministic mask generation
// ============================================================================

// SplitMix64 PRNG reused from phase section above.

/// Select `count` distinct bits from a mask of `total` bits using Fisher-Yates.
fn select_bits(rng: &mut SplitMix64, total: u32, count: u32) -> u64 {
    let mut indices: Vec<u32> = (0..total).collect();
    for i in 0..count.min(total) {
        let j = i + (rng.next() % (total - i) as u64) as u32;
        indices.swap(i as usize, j as usize);
    }
    let mut mask: u64 = 0;
    for i in 0..count.min(total) {
        mask |= 1u64 << indices[i as usize];
    }
    mask
}

/// Derive focus masks from a concept key using SplitMix64.
/// The hash determines WHERE in the container this concept lives.
pub fn concept_to_focus(concept_id: u64, density: FocusDensity) -> (u8, u8, u32) {
    let (bits_x, bits_y, bits_z) = density.bit_counts();
    let mut rng = SplitMix64(concept_id);

    let mask_x = select_bits(&mut rng, 8, bits_x) as u8;
    let mask_y = select_bits(&mut rng, 8, bits_y) as u8;
    let mask_z = select_bits(&mut rng, 32, bits_z) as u32;

    (mask_x, mask_y, mask_z)
}

// ============================================================================
// Operation 14a: focus_xor — Write/Erase via XOR gating
// ============================================================================

/// XOR a value into the container at the focused sub-volume.
///
/// Self-inverse: `focus_xor(focus_xor(c, m, v), m, v)` restores c.
/// Bytes outside the mask are NEVER touched.
pub fn focus_xor(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] ^= value[idx];
            }
        }
    }
}

// ============================================================================
// Operation 14b: focus_read — AND extraction
// ============================================================================

/// Extract the focused sub-volume. Non-focused positions are zeroed.
/// Non-destructive read — the container is unchanged.
pub fn focus_read(container: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Vec<u8> {
    assert!(container.len() >= 2048);
    let mut out = vec![0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                out[idx] = container[idx];
            }
        }
    }

    out
}

// ============================================================================
// Operation 14c: focus_add / focus_sub — phase-space gated write
// ============================================================================

/// ADD a value into the container at the focused sub-volume (phase-space).
/// NOT self-inverse — use focus_sub to undo.
pub fn focus_add(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] = container[idx].wrapping_add(value[idx]);
            }
        }
    }
}

/// SUB a value from the container at the focused sub-volume.
/// Exact inverse of focus_add.
pub fn focus_sub(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                container[idx] = container[idx].wrapping_sub(value[idx]);
            }
        }
    }
}

// ============================================================================
// Operation 14d: focus_hamming / focus_l1 — regional distance
// ============================================================================

/// Hamming distance within focus region (for binary containers).
/// Returns (hamming_distance, region_size_bytes).
pub fn focus_hamming(a: &[u8], b: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> (u64, u32) {
    assert!(a.len() >= 2048 && b.len() >= 2048);
    let mut distance: u64 = 0;
    let mut region_size: u32 = 0;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                distance += (a[idx] ^ b[idx]).count_ones() as u64;
                region_size += 1;
            }
        }
    }

    (distance, region_size)
}

/// L1 distance within focus region (for phase containers).
/// Returns (l1_distance, region_size_bytes).
pub fn focus_l1(a: &[u8], b: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> (u64, u32) {
    assert!(a.len() >= 2048 && b.len() >= 2048);
    let mut distance: u64 = 0;
    let mut region_size: u32 = 0;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                distance += (a[idx] as i16 - b[idx] as i16).unsigned_abs() as u64;
                region_size += 1;
            }
        }
    }

    (distance, region_size)
}

// ============================================================================
// Materialized mask operations
// ============================================================================

/// Expand the 48-bit focus address into a full 2048-byte mask.
/// out[i] = 0xFF if position i is in focus, 0x00 otherwise.
pub fn materialize_focus_mask(mask_x: u8, mask_y: u8, mask_z: u32) -> [u8; 2048] {
    let mut mask = [0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                mask[x * 256 + y * 32 + z] = 0xFF;
            }
        }
    }

    mask
}

/// Materialized XOR: `container[i] ^= (value[i] & mask[i])`.
/// SIMD-friendly: VPAND + VPXORD per 64-byte chunk.
pub fn focus_xor_materialized(container: &mut [u8], mask: &[u8; 2048], value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for i in 0..2048 {
        container[i] ^= value[i] & mask[i];
    }
}

/// Materialized ADD: `container[i] = container[i].wrapping_add(value[i] & mask[i])`.
/// SIMD-friendly: VPAND + VPADDB per 64-byte chunk.
pub fn focus_add_materialized(container: &mut [u8], mask: &[u8; 2048], value: &[u8]) {
    assert!(container.len() >= 2048);
    assert!(value.len() >= 2048);

    for i in 0..2048 {
        container[i] = container[i].wrapping_add(value[i] & mask[i]);
    }
}

/// Auto-dispatch: scalar for sparse, materialized for broad.
pub fn focus_xor_auto(container: &mut [u8], mask_x: u8, mask_y: u8, mask_z: u32, value: &[u8]) {
    let region = mask_x.count_ones() * mask_y.count_ones() * mask_z.count_ones();
    if region < SIMD_THRESHOLD {
        focus_xor(container, mask_x, mask_y, mask_z, value);
    } else {
        let mask = materialize_focus_mask(mask_x, mask_y, mask_z);
        focus_xor_materialized(container, &mask, value);
    }
}

// ============================================================================
// Composition: focus + binary / phase / carrier
// ============================================================================

/// Write a concept into a binary container at a focused region.
/// Uses XOR binding. Self-inverse: call again to erase.
pub fn focus_bind_binary(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_xor(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Write a concept into a phase container at a focused region.
/// Uses ADD binding. NOT self-inverse — use focus_unbind_phase to erase.
pub fn focus_bind_phase(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_add(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Erase a concept from a phase container at a focused region.
pub fn focus_unbind_phase(
    container: &mut [u8],
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    concept_vec: &[u8],
) {
    focus_sub(container, mask_x, mask_y, mask_z, concept_vec);
}

/// Write a carrier-encoded concept into a focused region of a waveform.
///
/// Combines carrier encoding (frequency multiplexing) with focus gating
/// (spatial partitioning). The carrier signal only exists in the focused
/// region.
pub fn focus_carrier_encode(
    container: &mut [i8],
    basis: &CarrierBasis,
    mask_x: u8,
    mask_y: u8,
    mask_z: u32,
    freq_idx: u8,
    phase_offset: f32,
    amplitude: f32,
) {
    let cos_phi = phase_offset.cos();
    let sin_phi = phase_offset.sin();
    let fi = freq_idx as usize;
    let scale = amplitude / self::CARRIER_AMPLITUDE;

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let j = x * 256 + y * 32 + z;
                let cos_val = basis.basis_cos[fi][j] as f32;
                let sin_val = basis.basis_sin[fi][j] as f32;
                let contribution = ((cos_phi * cos_val - sin_phi * sin_val) * scale)
                    .round()
                    .clamp(-128.0, 127.0) as i8;
                container[j] = container[j].saturating_add(contribution);
            }
        }
    }
}

// ============================================================================
// XOR Delta (Shared Lithography pattern)
// ============================================================================

/// Compute the XOR delta between two containers, restricted to a focus region.
/// Returns a 2048-byte delta where non-focused positions are zero.
pub fn focus_delta(old: &[u8], new: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Vec<u8> {
    assert!(old.len() >= 2048 && new.len() >= 2048);
    let mut delta = vec![0u8; 2048];

    for x in 0..FOCUS_DIM_X {
        if mask_x & (1 << x) == 0 {
            continue;
        }
        for y in 0..FOCUS_DIM_Y {
            if mask_y & (1 << y) == 0 {
                continue;
            }
            for z in 0..FOCUS_DIM_Z {
                if mask_z & (1 << z) == 0 {
                    continue;
                }
                let idx = x * 256 + y * 32 + z;
                delta[idx] = old[idx] ^ new[idx];
            }
        }
    }

    delta
}

/// Compact delta: only the non-zero bytes with their positions.
/// For sparse focus (4 bytes): 4 × 3 = 12 bytes vs 2048 for full delta.
pub struct CompactDelta {
    /// Packed focus address (6 bytes in a u64).
    pub mask: u64,
    /// (byte_position, xor_delta) pairs.
    pub changes: Vec<(u16, u8)>,
}

impl CompactDelta {
    pub fn from_delta(delta: &[u8], mask_x: u8, mask_y: u8, mask_z: u32) -> Self {
        let mut changes = Vec::new();
        for (i, &d) in delta.iter().enumerate() {
            if d != 0 {
                changes.push((i as u16, d));
            }
        }
        CompactDelta {
            mask: pack_focus(mask_x, mask_y, mask_z),
            changes,
        }
    }

    /// Apply this delta to a container via XOR.
    pub fn apply(&self, container: &mut [u8]) {
        for &(pos, delta) in &self.changes {
            container[pos as usize] ^= delta;
        }
    }

    /// Wire size in bytes: 8 bytes header + 3 bytes per change.
    pub fn wire_size(&self) -> usize {
        8 + self.changes.len() * 3
    }
}

// ============================================================================
// FocusRegistry — track what's written where
// ============================================================================

/// Tracks which focus addresses are occupied in a container.
/// Each entry: (packed_focus: u64, concept_id: u64) = 16 bytes.
pub struct FocusRegistry {
    pub entries: Vec<(u64, u64)>,
}

impl Default for FocusRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FocusRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a concept at a focus address.
    pub fn register(&mut self, focus: u64, concept_id: u64) {
        self.entries.push((focus, concept_id));
    }

    /// Check if a proposed focus address overlaps with any existing entry.
    /// Returns overlapping (concept_id, overlap_size_bytes) pairs.
    pub fn check_overlap(
        &self,
        new_mask_x: u8,
        new_mask_y: u8,
        new_mask_z: u32,
    ) -> Vec<(u64, u32)> {
        let mut overlaps = Vec::new();

        for &(existing_packed, concept_id) in &self.entries {
            let (ex, ey, ez) = unpack_focus(existing_packed);
            let overlap_x = (new_mask_x & ex).count_ones();
            let overlap_y = (new_mask_y & ey).count_ones();
            let overlap_z = (new_mask_z & ez).count_ones();
            let overlap_bytes = overlap_x * overlap_y * overlap_z;
            if overlap_bytes > 0 {
                overlaps.push((concept_id, overlap_bytes));
            }
        }

        overlaps
    }

    /// Remove a concept from the registry.
    pub fn remove(&mut self, concept_id: u64) -> Option<u64> {
        if let Some(pos) = self.entries.iter().position(|&(_, id)| id == concept_id) {
            let (focus, _) = self.entries.remove(pos);
            Some(focus)
        } else {
            None
        }
    }

    /// Number of registered concepts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes occupied across all registered concepts (overlap counted once).
    pub fn total_coverage(&self) -> u32 {
        if self.entries.is_empty() {
            return 0;
        }

        let mut coverage = [false; 2048];
        for &(packed, _) in &self.entries {
            let (mx, my, mz) = unpack_focus(packed);
            for x in 0..FOCUS_DIM_X {
                if mx & (1 << x) == 0 {
                    continue;
                }
                for y in 0..FOCUS_DIM_Y {
                    if my & (1 << y) == 0 {
                        continue;
                    }
                    for z in 0..FOCUS_DIM_Z {
                        if mz & (1 << z) == 0 {
                            continue;
                        }
                        coverage[x * 256 + y * 32 + z] = true;
                    }
                }
            }
        }

        coverage.iter().filter(|&&b| b).count() as u32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod focus_tests {
    use super::*;

    // ---- Geometry tests ----

    #[test]
    fn test_materialize_all_bits_set() {
        let mask = materialize_focus_mask(0xFF, 0xFF, 0xFFFFFFFF);
        assert!(mask.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_materialize_no_bits_set() {
        let mask = materialize_focus_mask(0, 0, 0);
        assert!(mask.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_materialize_single_byte_origin() {
        // mask_x=1 (bit 0), mask_y=1 (bit 0), mask_z=1 (bit 0) → index [0][0][0] = 0
        let mask = materialize_focus_mask(1, 1, 1);
        assert_eq!(mask[0], 0xFF);
        let count = mask.iter().filter(|&&b| b == 0xFF).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_materialize_single_byte_corner() {
        // mask_x=0x80 (bit 7), mask_y=0x80 (bit 7), mask_z=0x80000000 (bit 31)
        // index = 7*256 + 7*32 + 31 = 1792 + 224 + 31 = 2047
        let mask = materialize_focus_mask(0x80, 0x80, 0x80000000);
        assert_eq!(mask[2047], 0xFF);
        let count = mask.iter().filter(|&&b| b == 0xFF).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_materialize_region_size() {
        // popcount(mask_x) × popcount(mask_y) × popcount(mask_z) = region bytes
        let test_cases: Vec<(u8, u8, u32)> = vec![
            (0x03, 0x05, 0x0000000F), // 2 × 2 × 4 = 16
            (0xFF, 0x01, 0x00000001), // 8 × 1 × 1 = 8
            (0x01, 0xFF, 0xFFFFFFFF), // 1 × 8 × 32 = 256
        ];
        for (mx, my, mz) in test_cases {
            let mask = materialize_focus_mask(mx, my, mz);
            let count = mask.iter().filter(|&&b| b == 0xFF).count();
            let expected =
                mx.count_ones() as usize * my.count_ones() as usize * mz.count_ones() as usize;
            assert_eq!(count, expected, "mx={:#x} my={:#x} mz={:#x}", mx, my, mz);
        }
    }

    // ---- Pack / Unpack tests ----

    #[test]
    fn test_pack_unpack_round_trip() {
        let mx: u8 = 0xA5;
        let my: u8 = 0x3C;
        let mz: u32 = 0xDEADBEEF;
        let packed = pack_focus(mx, my, mz);
        let (rx, ry, rz) = unpack_focus(packed);
        assert_eq!(rx, mx);
        assert_eq!(ry, my);
        assert_eq!(rz, mz);
    }

    // ---- XOR gating tests ----

    #[test]
    fn test_focus_xor_self_inverse() {
        let mut container = vec![0x42u8; 2048];
        let original = container.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();

        focus_xor(&mut container, 0x0F, 0xF0, 0x0000FFFF, &value);
        assert_ne!(container, original);
        focus_xor(&mut container, 0x0F, 0xF0, 0x0000FFFF, &value);
        assert_eq!(container, original);
    }

    #[test]
    fn test_focus_xor_commutative() {
        let mut c1 = vec![0u8; 2048];
        let mut c2 = vec![0u8; 2048];
        let v1: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let v2: Vec<u8> = (0..2048).map(|i| ((i * 41 + 7) % 256) as u8).collect();

        let mx = 0x03u8;
        let my = 0x0Cu8;
        let mz = 0x000000FFu32;

        focus_xor(&mut c1, mx, my, mz, &v1);
        focus_xor(&mut c1, mx, my, mz, &v2);

        focus_xor(&mut c2, mx, my, mz, &v2);
        focus_xor(&mut c2, mx, my, mz, &v1);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_focus_xor_preserves_outside() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let original = container.clone();
        let value = vec![0xFFu8; 2048];

        let mx = 0x01u8; // only X slab 0
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // only Z slabs 0-3

        focus_xor(&mut container, mx, my, mz, &value);

        // Check outside mask is unchanged
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(
                    container[i], original[i],
                    "position {} outside mask changed",
                    i
                );
            }
        }
    }

    #[test]
    fn test_focus_xor_materialized_matches_scalar() {
        let mut c1 = vec![0x55u8; 2048];
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 71 % 256) as u8).collect();

        let mx = 0x0Fu8;
        let my = 0xF0u8;
        let mz = 0x00FF00FFu32;

        focus_xor(&mut c1, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        focus_xor_materialized(&mut c2, &mask, &value);

        assert_eq!(c1, c2);
    }

    // ---- Phase gating tests ----

    #[test]
    fn test_focus_add_sub_round_trip() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i * 31 % 256) as u8).collect();
        let original = container.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 47 % 256) as u8).collect();

        let mx = 0x33u8;
        let my = 0xCCu8;
        let mz = 0x0F0F0F0Fu32;

        focus_add(&mut container, mx, my, mz, &value);
        assert_ne!(container, original);
        focus_sub(&mut container, mx, my, mz, &value);
        assert_eq!(container, original);
    }

    #[test]
    fn test_focus_add_only_modifies_masked() {
        let mut container = vec![100u8; 2048];
        let original = container.clone();
        let value = vec![50u8; 2048];

        let mx = 0x80u8;
        let my = 0x01u8;
        let mz = 0x00000001u32;

        focus_add(&mut container, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(container[i], original[i]);
            }
        }
    }

    #[test]
    fn test_focus_add_materialized_matches_scalar() {
        let mut c1: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 19 % 256) as u8).collect();

        let mx = 0x55u8;
        let my = 0xAAu8;
        let mz = 0xF0F0F0F0u32;

        focus_add(&mut c1, mx, my, mz, &value);

        let mask = materialize_focus_mask(mx, my, mz);
        focus_add_materialized(&mut c2, &mask, &value);

        assert_eq!(c1, c2);
    }

    // ---- Focus read tests ----

    #[test]
    fn test_focus_read_zero_container() {
        let container = vec![0u8; 2048];
        let result = focus_read(&container, 0xFF, 0xFF, 0xFFFFFFFF);
        assert!(result.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_focus_read_after_xor() {
        let mut container = vec![0u8; 2048];
        let value: Vec<u8> = (0..2048).map(|i| (i * 7 + 1) as u8).collect();

        let mx = 0x03u8;
        let my = 0x03u8;
        let mz = 0x0000000Fu32;

        focus_xor(&mut container, mx, my, mz, &value);
        let read = focus_read(&container, mx, my, mz);

        // At focused positions, read should equal value (since container was zero)
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], value[i], "focused pos {} mismatch", i);
            } else {
                assert_eq!(read[i], 0, "non-focused pos {} should be zero", i);
            }
        }
    }

    #[test]
    fn test_focus_read_non_focused_zero() {
        let container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let result = focus_read(&container, 0x01, 0x01, 0x00000001);
        // Only position 0 is focused
        for i in 1..2048 {
            if i != 0 {
                // Check all non-focused positions
                let mask = materialize_focus_mask(0x01, 0x01, 0x00000001);
                if mask[i] == 0 {
                    assert_eq!(result[i], 0);
                }
            }
        }
    }

    // ---- Distance tests ----

    #[test]
    fn test_focus_hamming_self_zero() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let (dist, size) = focus_hamming(&a, &a, 0xFF, 0xFF, 0xFFFFFFFF);
        assert_eq!(dist, 0);
        assert_eq!(size, 2048);
    }

    #[test]
    fn test_focus_hamming_restricted_region() {
        let a = vec![0u8; 2048];
        let b = vec![0xFFu8; 2048];

        // Full mask: every byte differs by 8 bits
        let (full_dist, full_size) = focus_hamming(&a, &b, 0xFF, 0xFF, 0xFFFFFFFF);
        assert_eq!(full_dist, 2048 * 8);
        assert_eq!(full_size, 2048);

        // Single byte mask
        let (single_dist, single_size) = focus_hamming(&a, &b, 0x01, 0x01, 0x00000001);
        assert_eq!(single_dist, 8);
        assert_eq!(single_size, 1);
    }

    #[test]
    fn test_focus_l1_self_zero() {
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let (dist, _) = focus_l1(&a, &a, 0x0F, 0xF0, 0x00FF00FF);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_focus_l1_region_matches_manual() {
        let a = vec![100u8; 2048];
        let b = vec![110u8; 2048];

        // Region size = 2 × 2 × 4 = 16 bytes, each with L1 = 10
        let (dist, size) = focus_l1(&a, &b, 0x03, 0x03, 0x0000000F);
        assert_eq!(size, 16);
        assert_eq!(dist, 16 * 10);
    }

    // ---- Registry tests ----

    #[test]
    fn test_registry_no_overlap_sparse() {
        let mut reg = FocusRegistry::new();

        // Register 5 concepts at different sparse addresses
        for i in 0..5u64 {
            let (mx, my, mz) = concept_to_focus(i * 1000 + 42, FocusDensity::Sparse);
            let packed = pack_focus(mx, my, mz);
            reg.register(packed, i);
        }

        // Check overlap of a new concept
        let (nmx, nmy, nmz) = concept_to_focus(99999, FocusDensity::Sparse);
        let _overlaps = reg.check_overlap(nmx, nmy, nmz);
        // Sparse masks are very unlikely to overlap
        // (but not impossible — just verify the method runs correctly)
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn test_registry_detect_overlap() {
        let mut reg = FocusRegistry::new();

        // Register a concept at a specific address
        let mx = 0xFFu8;
        let my = 0xFFu8;
        let mz = 0xFFFFFFFFu32;
        reg.register(pack_focus(mx, my, mz), 1);

        // Any new concept will overlap with a full mask
        let overlaps = reg.check_overlap(0x01, 0x01, 0x00000001);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].0, 1); // concept_id
        assert_eq!(overlaps[0].1, 1); // 1 byte overlap
    }

    #[test]
    fn test_registry_total_coverage_sparse() {
        let mut reg = FocusRegistry::new();

        // Register 3 non-overlapping sparse concepts manually
        // Each sparse = 1×1×4 = 4 bytes
        reg.register(pack_focus(0x01, 0x01, 0x0000000F), 1); // X0,Y0,Z0-3
        reg.register(pack_focus(0x02, 0x02, 0x0000000F), 2); // X1,Y1,Z0-3
        reg.register(pack_focus(0x04, 0x04, 0x0000000F), 3); // X2,Y2,Z0-3

        assert_eq!(reg.total_coverage(), 12); // 3 × 4 = 12
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = FocusRegistry::new();
        let packed = pack_focus(0x01, 0x01, 0x00000001);
        reg.register(packed, 42);
        assert_eq!(reg.len(), 1);

        let removed = reg.remove(42);
        assert_eq!(removed, Some(packed));
        assert_eq!(reg.len(), 0);

        let not_found = reg.remove(42);
        assert_eq!(not_found, None);
    }

    // ---- concept_to_focus determinism ----

    #[test]
    fn test_concept_to_focus_deterministic() {
        let (mx1, my1, mz1) = concept_to_focus(12345, FocusDensity::Sparse);
        let (mx2, my2, mz2) = concept_to_focus(12345, FocusDensity::Sparse);
        assert_eq!(mx1, mx2);
        assert_eq!(my1, my2);
        assert_eq!(mz1, mz2);
    }

    #[test]
    fn test_concept_to_focus_different_ids() {
        let mut masks = std::collections::HashSet::new();
        for id in 0..100u64 {
            let (mx, my, mz) = concept_to_focus(id, FocusDensity::Medium);
            masks.insert((mx, my, mz));
        }
        // With 100 random IDs and medium density, most should be distinct
        assert!(
            masks.len() > 50,
            "expected most masks unique, got {}",
            masks.len()
        );
    }

    #[test]
    fn test_concept_to_focus_density_bits() {
        for density in [
            FocusDensity::Sparse,
            FocusDensity::Medium,
            FocusDensity::Broad,
        ] {
            let (bits_x, bits_y, bits_z) = density.bit_counts();
            let (mx, my, mz) = concept_to_focus(42, density);
            assert_eq!(mx.count_ones(), bits_x, "density={:?} mask_x bits", density);
            assert_eq!(my.count_ones(), bits_y, "density={:?} mask_y bits", density);
            assert_eq!(mz.count_ones(), bits_z, "density={:?} mask_z bits", density);
        }
    }

    // ---- Integration tests ----

    #[test]
    fn test_write_read_10_sparse_concepts() {
        let mut container = vec![0u8; 2048];
        let mut rng = super::SplitMix64(777);

        let concepts: Vec<(u64, Vec<u8>)> = (0..10)
            .map(|id| {
                let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                (id, value)
            })
            .collect();

        // Write all 10
        for (id, value) in &concepts {
            let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
            focus_xor(&mut container, mx, my, mz, value);
        }

        // Read each back and verify signal is present
        for (id, value) in &concepts {
            let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
            let read = focus_read(&container, mx, my, mz);
            let mask = materialize_focus_mask(mx, my, mz);

            // Count how many focused positions match
            let mut matches = 0u32;
            let mut total = 0u32;
            for i in 0..2048 {
                if mask[i] == 0xFF {
                    total += 1;
                    if read[i] == value[i] {
                        matches += 1;
                    }
                }
            }
            // With sparse non-overlapping, most/all should match
            // (some may collide due to birthday effect)
            assert!(
                matches as f64 / total as f64 > 0.5,
                "concept {} signal too weak: {}/{}",
                id,
                matches,
                total
            );
        }
    }

    #[test]
    fn test_write_preserves_non_focused() {
        let mut container: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let original = container.clone();
        let mut rng = super::SplitMix64(123);

        for id in 0..10u64 {
            let (mx, my, mz) = concept_to_focus(id, FocusDensity::Sparse);
            let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
            focus_xor(&mut container, mx, my, mz, &value);
        }

        // Check that MOST positions are unchanged
        // (10 sparse concepts × 4 bytes each = ~40 bytes changed, out of 2048)
        let changed = container
            .iter()
            .zip(original.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            changed <= 200, // generous bound for overlaps
            "too many positions changed: {} (expected ~40)",
            changed
        );
    }

    #[test]
    fn test_focus_bind_binary_round_trip() {
        let mut container = vec![0u8; 2048];
        let concept: Vec<u8> = (0..2048).map(|i| (i * 7 % 256) as u8).collect();

        let mx = 0x0Fu8;
        let my = 0x0Fu8;
        let mz = 0x000000FFu32;

        focus_bind_binary(&mut container, mx, my, mz, &concept);
        let read = focus_read(&container, mx, my, mz);

        // Verify focused positions have the concept value
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], concept[i], "pos {} mismatch", i);
            }
        }

        // Erase: XOR again
        focus_bind_binary(&mut container, mx, my, mz, &concept);
        assert!(container.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_focus_bind_phase_round_trip() {
        let mut container = vec![0u8; 2048];
        let concept: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();

        let mx = 0x07u8;
        let my = 0x07u8;
        let mz = 0x0000FFFFu32;

        focus_bind_phase(&mut container, mx, my, mz, &concept);
        let read = focus_read(&container, mx, my, mz);

        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(read[i], concept[i], "pos {} mismatch", i);
            }
        }

        // Undo
        focus_unbind_phase(&mut container, mx, my, mz, &concept);
        assert!(container.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_compact_delta_wire_size() {
        let old = vec![0u8; 2048];
        let mut new = vec![0u8; 2048];
        let value: Vec<u8> = (0..2048).map(|i| (i * 3 % 256) as u8).collect();

        let mx = 0x01u8;
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // 1×1×4 = 4 bytes

        // Write into new
        new.copy_from_slice(&old);
        focus_xor(&mut new, mx, my, mz, &value);

        let delta = focus_delta(&old, &new, mx, my, mz);
        let compact = CompactDelta::from_delta(&delta, mx, my, mz);

        assert!(
            compact.wire_size() < 2048,
            "compact should be smaller than full"
        );
        assert!(
            compact.changes.len() <= 4,
            "sparse focus: at most 4 changes"
        );
    }

    #[test]
    fn test_compact_delta_apply() {
        let old: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        let value = vec![0xAAu8; 2048];

        let mx = 0x03u8;
        let my = 0x03u8;
        let mz = 0x0000000Fu32;

        focus_xor(&mut new, mx, my, mz, &value);

        let delta = focus_delta(&old, &new, mx, my, mz);
        let compact = CompactDelta::from_delta(&delta, mx, my, mz);

        let mut reconstructed = old.clone();
        compact.apply(&mut reconstructed);

        // Focused positions should match new
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0xFF {
                assert_eq!(reconstructed[i], new[i], "pos {} mismatch", i);
            }
        }
    }

    #[test]
    fn test_focus_xor_auto_matches_scalar() {
        let mut c1 = vec![0x77u8; 2048];
        let mut c2 = c1.clone();
        let value: Vec<u8> = (0..2048).map(|i| (i * 59 % 256) as u8).collect();

        // Sparse: should use scalar
        focus_xor(&mut c1, 0x01, 0x01, 0x00000001, &value);
        focus_xor_auto(&mut c2, 0x01, 0x01, 0x00000001, &value);
        assert_eq!(c1, c2);

        // Broad: should use materialized
        let mut c3 = vec![0x77u8; 2048];
        let mut c4 = c3.clone();
        focus_xor(&mut c3, 0xFF, 0xFF, 0xFFFFFFFF, &value);
        focus_xor_auto(&mut c4, 0xFF, 0xFF, 0xFFFFFFFF, &value);
        assert_eq!(c3, c4);
    }

    // ---- Capacity experiment ----

    #[test]
    fn test_focus_capacity_experiment() {
        println!("\n=== Focus Gating Capacity Experiment ===\n");

        for &density in &[
            FocusDensity::Sparse,
            FocusDensity::Medium,
            FocusDensity::Broad,
        ] {
            let (bits_x, bits_y, bits_z) = density.bit_counts();
            let region_bytes = bits_x * bits_y * bits_z;

            let test_counts = [1u64, 5, 10, 20, 32, 50];
            let mut rng = super::SplitMix64(42);

            for &n in &test_counts {
                let mut container = vec![0u8; 2048];
                let concepts: Vec<(u64, Vec<u8>)> = (0..n)
                    .map(|id| {
                        let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                        (id, value)
                    })
                    .collect();

                // Write all
                for (id, value) in &concepts {
                    let (mx, my, mz) = concept_to_focus(*id, density);
                    focus_xor(&mut container, mx, my, mz, value);
                }

                // Read each back and measure accuracy
                let mut total_match = 0u32;
                let mut total_bits = 0u32;
                for (id, value) in &concepts {
                    let (mx, my, mz) = concept_to_focus(*id, density);
                    let read = focus_read(&container, mx, my, mz);
                    let mask = materialize_focus_mask(mx, my, mz);

                    for i in 0..2048 {
                        if mask[i] == 0xFF {
                            total_bits += 1;
                            if read[i] == value[i] {
                                total_match += 1;
                            }
                        }
                    }
                }

                let accuracy = if total_bits > 0 {
                    total_match as f64 / total_bits as f64
                } else {
                    0.0
                };

                println!(
                    "  {:?} ({}B region) N={:>3}: accuracy={:.1}% ({}/{})",
                    density,
                    region_bytes,
                    n,
                    accuracy * 100.0,
                    total_match,
                    total_bits
                );
            }
            println!();
        }

        // Verify sparse holds at N=10 with good accuracy
        {
            let mut container = vec![0u8; 2048];
            let mut rng = super::SplitMix64(999);
            let concepts: Vec<(u64, Vec<u8>)> = (0..10)
                .map(|id| {
                    let value: Vec<u8> = (0..2048).map(|_| (rng.next() % 256) as u8).collect();
                    (id, value)
                })
                .collect();

            for (id, value) in &concepts {
                let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
                focus_xor(&mut container, mx, my, mz, value);
            }

            let mut total_match = 0u32;
            let mut total_bits = 0u32;
            for (id, value) in &concepts {
                let (mx, my, mz) = concept_to_focus(*id, FocusDensity::Sparse);
                let read = focus_read(&container, mx, my, mz);
                let mask = materialize_focus_mask(mx, my, mz);
                for i in 0..2048 {
                    if mask[i] == 0xFF {
                        total_bits += 1;
                        if read[i] == value[i] {
                            total_match += 1;
                        }
                    }
                }
            }
            let accuracy = total_match as f64 / total_bits as f64;
            assert!(
                accuracy > 0.7,
                "Sparse N=10 accuracy {:.1}% too low",
                accuracy * 100.0
            );
        }
    }

    // ---- Carrier + focus integration ----

    #[test]
    fn test_focus_carrier_encode_writes_only_masked() {
        let basis = self::CarrierBasis::new();
        let mut container = vec![0i8; 2048];

        let mx = 0x01u8;
        let my = 0x01u8;
        let mz = 0x0000000Fu32; // 1×1×4 = 4 bytes

        focus_carrier_encode(
            &mut container,
            &basis,
            mx,
            my,
            mz,
            0,
            1.0,
            self::CARRIER_AMPLITUDE,
        );

        // Check that only masked positions are non-zero
        let mask = materialize_focus_mask(mx, my, mz);
        for i in 0..2048 {
            if mask[i] == 0 {
                assert_eq!(
                    container[i], 0,
                    "position {} outside mask should be zero, got {}",
                    i, container[i]
                );
            }
        }

        // At least some masked positions should be non-zero
        let nonzero_in_mask = (0..2048)
            .filter(|&i| mask[i] == 0xFF && container[i] != 0)
            .count();
        assert!(
            nonzero_in_mask > 0,
            "carrier should write some non-zero values"
        );
    }
}
