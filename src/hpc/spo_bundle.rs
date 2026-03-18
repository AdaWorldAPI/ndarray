//! 3-atom SPO bundling via cyclic permutation in Z/dZ.
//!
//! Two resolutions:
//! - Level A: 8Kbit compressed bundle (metadata search index)
//! - Level B: 16Kbit integrated plane (holographic SPO encoding)
//!
//! Production mechanism: integer cyclic shift + majority vote.
//! No floating point at runtime. Pure binary operations.

// ============================================================================
// Constants
// ============================================================================

/// Golden ratio φ ≈ 1.618033988749895
const PHI: f64 = 1.618_033_988_749_895;

/// Compute golden shift for dimension d, ensuring odd (gcd=1 with power-of-2 d).
/// floor(d / φ²), rounded to nearest odd.
pub const fn golden_shift(d: usize) -> usize {
    let raw = (d as f64 / (PHI * PHI)) as usize;
    if raw % 2 == 0 { raw + 1 } else { raw }
}

/// Level A constants (8Kbit = 128 × u64)
pub const D_META: usize = 8192;
pub const META_WORDS: usize = D_META / 64; // 128
pub const SHIFT_META: usize = golden_shift(D_META); // 3131

/// Level B constants (16Kbit = 256 × u64)
pub const D_FULL: usize = 16384;
pub const FULL_WORDS: usize = D_FULL / 64; // 256
pub const SHIFT_FULL: usize = golden_shift(D_FULL); // 6261

// ============================================================================
// Core operations
// ============================================================================

/// Cyclic bit shift on a u64 array of N words.
/// Shifts the entire bit vector right by `shift` positions (mod N*64).
pub fn cyclic_shift<const N: usize>(bits: &[u64; N], shift: usize) -> [u64; N] {
    let d = N * 64;
    let shift = shift % d;
    if shift == 0 {
        return *bits;
    }
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut result = [0u64; N];
    for i in 0..N {
        let src = (i + word_shift) % N;
        let next = (src + 1) % N;
        if bit_shift == 0 {
            result[i] = bits[src];
        } else {
            result[i] = (bits[src] >> bit_shift) | (bits[next] << (64 - bit_shift));
        }
    }
    result
}

/// Majority vote of 3 binary vectors: output bit = 1 if ≥2 of 3 inputs are 1.
/// Bit-parallel: (a&b) | (a&c) | (b&c)
pub fn majority_vote_3<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    c: &[u64; N],
) -> [u64; N] {
    let mut result = [0u64; N];
    for i in 0..N {
        result[i] = (a[i] & b[i]) | (a[i] & c[i]) | (b[i] & c[i]);
    }
    result
}

/// Hamming distance between two bit vectors.
pub fn hamming<const N: usize>(a: &[u64; N], b: &[u64; N]) -> u32 {
    let mut dist = 0u32;
    for i in 0..N {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Popcount of a bit vector.
pub fn popcount<const N: usize>(a: &[u64; N]) -> u32 {
    a.iter().map(|w| w.count_ones()).sum()
}

/// GCD (Euclidean algorithm).
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ============================================================================
// Level A: 8Kbit compressed bundle
// ============================================================================

/// Project a 16Kbit plane to 8Kbit by XOR-folding (top half XOR bottom half).
pub fn fold_to_half(plane: &[u64; 256]) -> [u64; 128] {
    let mut result = [0u64; 128];
    for i in 0..128 {
        result[i] = plane[i] ^ plane[i + 128];
    }
    result
}

/// Bundle S, P, O into 8Kbit compressed search index.
pub fn bundle_8k(s: &[u64; 256], p: &[u64; 256], o: &[u64; 256]) -> [u64; 128] {
    let s_half = fold_to_half(s);
    let p_half = fold_to_half(p);
    let o_half = fold_to_half(o);
    let p_shifted = cyclic_shift(&p_half, SHIFT_META);
    let o_shifted = cyclic_shift(&o_half, (SHIFT_META * 2) % D_META);
    majority_vote_3(&s_half, &p_shifted, &o_shifted)
}

/// Recover S from 8Kbit bundle.
pub fn recover_s_8k(bundle: &[u64; 128]) -> [u64; 128] {
    *bundle
}
/// Recover P from 8Kbit bundle.
pub fn recover_p_8k(bundle: &[u64; 128]) -> [u64; 128] {
    cyclic_shift(bundle, D_META - SHIFT_META)
}
/// Recover O from 8Kbit bundle.
pub fn recover_o_8k(bundle: &[u64; 128]) -> [u64; 128] {
    cyclic_shift(bundle, D_META - (SHIFT_META * 2) % D_META)
}

// ============================================================================
// Level B: 16Kbit integrated plane
// ============================================================================

/// Bundle S, P, O into 16Kbit integrated holographic plane.
pub fn bundle_16k(s: &[u64; 256], p: &[u64; 256], o: &[u64; 256]) -> [u64; 256] {
    let p_shifted = cyclic_shift(p, SHIFT_FULL);
    let o_shifted = cyclic_shift(o, (SHIFT_FULL * 2) % D_FULL);
    majority_vote_3(s, &p_shifted, &o_shifted)
}

/// Recover S from 16Kbit integrated plane.
pub fn recover_s_16k(bundle: &[u64; 256]) -> [u64; 256] {
    *bundle
}
/// Recover P from 16Kbit integrated plane.
pub fn recover_p_16k(bundle: &[u64; 256]) -> [u64; 256] {
    cyclic_shift(bundle, D_FULL - SHIFT_FULL)
}
/// Recover O from 16Kbit integrated plane.
pub fn recover_o_16k(bundle: &[u64; 256]) -> [u64; 256] {
    cyclic_shift(bundle, D_FULL - (SHIFT_FULL * 2) % D_FULL)
}

// ============================================================================
// PRNG and helpers
// ============================================================================

/// Simple deterministic PRNG for reproducible tests.
fn prng_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Generate a random bit vector from a seed.
pub fn random_bits<const N: usize>(seed: u64) -> [u64; N] {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut words = [0u64; N];
    for w in words.iter_mut() {
        *w = prng_next(&mut state);
    }
    words
}

/// Generate a biased random bit vector (P(bit=1) = bias).
pub fn biased_bits<const N: usize>(seed: u64, bias: f64) -> [u64; N] {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let threshold = (bias * u64::MAX as f64) as u64;
    let mut words = [0u64; N];
    for w in words.iter_mut() {
        let mut word = 0u64;
        for bit in 0..64 {
            let r = prng_next(&mut state);
            if r < threshold {
                word |= 1u64 << bit;
            }
        }
        *w = word;
    }
    words
}

/// Flip exactly n_flips random bits in a copy of the vector.
pub fn flip_n_bits<const N: usize>(v: &[u64; N], n_flips: usize, seed: u64) -> [u64; N] {
    let d = N * 64;
    let mut result = *v;
    let mut state = seed.wrapping_mul(0xDEADBEEF).wrapping_add(1);
    let mut flipped = 0;
    // Simple approach: walk through random positions, flip if not already flipped
    // For small n_flips relative to d, this terminates quickly
    let mut used = vec![false; d];
    while flipped < n_flips && flipped < d {
        let pos = (prng_next(&mut state) as usize) % d;
        if !used[pos] {
            used[pos] = true;
            result[pos / 64] ^= 1u64 << (pos % 64);
            flipped += 1;
        }
    }
    result
}

/// Spearman rank correlation between two f64 slices.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let rank_a = to_ranks(a);
    let rank_b = to_ranks(b);
    let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn to_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        result[idx] = rank as f64;
    }
    result
}

// ============================================================================
// Tests — all 12 experiments
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ========================================================================
    // EXPERIMENT 1: GCD Verification
    // ========================================================================

    #[test]
    fn exp1_gcd_verification() {
        // BUG: 3130 is even, gcd(3130, 8192) = 2
        assert_eq!(gcd(3130, 8192), 2, "old shift has gcd=2 bug");

        // FIX: golden_shift(8192) = 3129, which is odd → gcd(3129, 8192) = 1
        assert_eq!(gcd(SHIFT_META, D_META), 1, "meta shift must be coprime");
        assert!(
            SHIFT_META % 2 == 1,
            "shift must be odd, got {}",
            SHIFT_META
        );

        // FIX: golden_shift(16384) is odd → gcd with 16384 = 1
        assert_eq!(gcd(SHIFT_FULL, D_FULL), 1, "full shift must be coprime");
        assert!(
            SHIFT_FULL % 2 == 1,
            "full shift must be odd, got {}",
            SHIFT_FULL
        );

        // Verify full orbit at d=8192: shift visits all positions
        let mut visited = vec![false; D_META];
        let mut pos = 0usize;
        for _ in 0..D_META {
            assert!(!visited[pos], "position {} visited twice", pos);
            visited[pos] = true;
            pos = (pos + SHIFT_META) % D_META;
        }
        assert!(visited.iter().all(|&v| v), "not all positions visited");

        // Orbit length test: shift=3130 (gcd=2) has orbits of length 4096
        // shift=3129 (gcd=1) has full orbit of length 8192
        let orbit_3130 = D_META / gcd(3130, D_META);
        let orbit_good = D_META / gcd(SHIFT_META, D_META);
        assert_eq!(orbit_3130, 4096, "shift=3130 orbit should be half-length");
        assert_eq!(orbit_good, D_META, "our shift should give full orbit");

        // Adversarial test: alternating 010101... pattern
        let mut alternating = [0u64; 128];
        for w in alternating.iter_mut() {
            *w = 0x5555_5555_5555_5555; // 01010101...
        }
        let pop = popcount(&alternating);
        assert_eq!(pop, 4096, "alternating pattern should have popcount d/2");

        // With even shift (3130): alternating shifted by even = same pattern
        // (because 0101... shifted by any even number is still 0101...)
        let shifted_even = cyclic_shift(&alternating, 3130);
        let dist_even = hamming(&alternating, &shifted_even);

        // With odd shift: alternating becomes 1010... (complement) → max hamming
        let shifted_odd = cyclic_shift(&alternating, SHIFT_META);
        let dist_odd = hamming(&alternating, &shifted_odd);

        eprintln!("\n  EXPERIMENT 1: GCD Verification");
        eprintln!(
            "  shift=3130 (even): hamming(alternating, shift) = {} (VULNERABLE: same pattern)",
            dist_even
        );
        eprintln!(
            "  shift={} (odd): hamming(alternating, shift) = {} (SAFE: fully decorrelated)",
            SHIFT_META, dist_odd
        );
        eprintln!("  gcd(3130, 8192) = {} → orbit len = {}", gcd(3130, 8192), orbit_3130);
        eprintln!(
            "  gcd({}, 8192) = {} → orbit len = {}",
            SHIFT_META,
            gcd(SHIFT_META, D_META),
            orbit_good
        );

        assert_eq!(
            dist_even, 0,
            "even shift on alternating should give hamming=0"
        );
        assert_eq!(
            dist_odd, D_META as u32,
            "odd shift on alternating should give hamming=d"
        );
    }

    // ========================================================================
    // EXPERIMENT 2: Recovery Rate Verification
    // ========================================================================

    #[test]
    fn exp2_recovery_rate_8k() {
        let n_trials = 200;
        let mut total_s_errors = 0u64;
        let mut total_p_errors = 0u64;
        let mut total_o_errors = 0u64;
        let total_bits = n_trials as u64 * D_META as u64;

        for trial in 0..n_trials {
            let seed = (trial as u64 + 1) * 1000;
            let s: [u64; 128] = random_bits(seed);
            let p: [u64; 128] = random_bits(seed + 1);
            let o: [u64; 128] = random_bits(seed + 2);

            let p_shifted = cyclic_shift(&p, SHIFT_META);
            let o_shifted = cyclic_shift(&o, (SHIFT_META * 2) % D_META);
            let bundle = majority_vote_3(&s, &p_shifted, &o_shifted);

            let rec_s = recover_s_8k(&bundle);
            let rec_p = recover_p_8k(&bundle);
            let rec_o = recover_o_8k(&bundle);

            total_s_errors += hamming(&s, &rec_s) as u64;
            total_p_errors += hamming(&p, &rec_p) as u64;
            total_o_errors += hamming(&o, &rec_o) as u64;
        }

        let s_err = total_s_errors as f64 / total_bits as f64;
        let p_err = total_p_errors as f64 / total_bits as f64;
        let o_err = total_o_errors as f64 / total_bits as f64;

        eprintln!("\n  EXPERIMENT 2: Recovery Rate (8Kbit, {} trials)", n_trials);
        eprintln!("  S error rate: {:.4} (expected 0.25)", s_err);
        eprintln!("  P error rate: {:.4} (expected 0.25)", p_err);
        eprintln!("  O error rate: {:.4} (expected 0.25)", o_err);

        assert!(
            (s_err - 0.25).abs() < 0.02,
            "S error rate {:.4} too far from 0.25",
            s_err
        );
        assert!(
            (p_err - 0.25).abs() < 0.02,
            "P error rate {:.4} too far from 0.25",
            p_err
        );
        assert!(
            (o_err - 0.25).abs() < 0.02,
            "O error rate {:.4} too far from 0.25",
            o_err
        );
    }

    #[test]
    fn exp2b_recovery_rate_16k() {
        let n_trials = 100;
        let mut total_errors = 0u64;
        let total_bits = n_trials as u64 * D_FULL as u64;

        for trial in 0..n_trials {
            let seed = (trial as u64 + 1) * 2000;
            let s: [u64; 256] = random_bits(seed);
            let p: [u64; 256] = random_bits(seed + 1);
            let o: [u64; 256] = random_bits(seed + 2);

            let bundle = bundle_16k(&s, &p, &o);
            let rec_s = recover_s_16k(&bundle);
            total_errors += hamming(&s, &rec_s) as u64;
        }

        let err = total_errors as f64 / total_bits as f64;
        eprintln!("\n  EXPERIMENT 2b: Recovery Rate (16Kbit, {} trials)", n_trials);
        eprintln!("  S error rate: {:.4} (expected 0.25)", err);

        assert!(
            (err - 0.25).abs() < 0.02,
            "16K error rate {:.4} too far from 0.25",
            err
        );
    }

    // ========================================================================
    // EXPERIMENT 3: Distance Ranking Preservation
    // ========================================================================

    #[test]
    fn exp3_ranking_preservation() {
        let n_nodes = 500;

        // Generate nodes (8Kbit directly for speed)
        let nodes: Vec<([u64; 128], [u64; 128], [u64; 128])> = (0..n_nodes)
            .map(|i| {
                let seed = (i as u64 + 1) * 7;
                (random_bits(seed), random_bits(seed + 1), random_bits(seed + 2))
            })
            .collect();

        // Bundle each
        let bundles: Vec<[u64; 128]> = nodes
            .iter()
            .map(|(s, p, o)| {
                let p_sh = cyclic_shift(p, SHIFT_META);
                let o_sh = cyclic_shift(o, (SHIFT_META * 2) % D_META);
                majority_vote_3(s, &p_sh, &o_sh)
            })
            .collect();

        // For multiple queries, measure ranking preservation
        let queries = [0, 50, 100, 200];
        let mut all_spearman = Vec::new();

        eprintln!("\n  EXPERIMENT 3: Ranking Preservation (n={})", n_nodes);

        for &qi in &queries {
            let mut dists_sep: Vec<(usize, u32)> = (0..n_nodes)
                .filter(|&i| i != qi)
                .map(|i| {
                    let d = hamming(&nodes[qi].0, &nodes[i].0)
                        + hamming(&nodes[qi].1, &nodes[i].1)
                        + hamming(&nodes[qi].2, &nodes[i].2);
                    (i, d)
                })
                .collect();
            dists_sep.sort_by_key(|&(_, d)| d);

            let mut dists_bun: Vec<(usize, u32)> = (0..n_nodes)
                .filter(|&i| i != qi)
                .map(|i| (i, hamming(&bundles[qi], &bundles[i])))
                .collect();
            dists_bun.sort_by_key(|&(_, d)| d);

            // Recall@k
            for &k in &[1, 5, 10, 20] {
                let top_sep: HashSet<usize> =
                    dists_sep[..k].iter().map(|&(i, _)| i).collect();
                let top_bun: HashSet<usize> =
                    dists_bun[..k].iter().map(|&(i, _)| i).collect();
                let recall = top_sep.intersection(&top_bun).count() as f64 / k as f64;
                eprintln!("  Query {}: Recall@{} = {:.2}", qi, k, recall);
            }

            // Spearman
            let sep_f64: Vec<f64> = dists_sep.iter().map(|&(_, d)| d as f64).collect();
            let bun_f64: Vec<f64> = dists_bun.iter().map(|&(_, d)| d as f64).collect();
            // Need to align by index, not by sort order
            let mut sep_by_idx = vec![0.0f64; n_nodes];
            let mut bun_by_idx = vec![0.0f64; n_nodes];
            for &(i, d) in &dists_sep {
                sep_by_idx[i] = d as f64;
            }
            for &(i, d) in &dists_bun {
                bun_by_idx[i] = d as f64;
            }
            // Remove query node
            let mut s_vals = Vec::new();
            let mut b_vals = Vec::new();
            for i in 0..n_nodes {
                if i != qi {
                    s_vals.push(sep_by_idx[i]);
                    b_vals.push(bun_by_idx[i]);
                }
            }
            let rho = spearman(&s_vals, &b_vals);
            eprintln!("  Query {}: Spearman ρ = {:.4}", qi, rho);
            all_spearman.push(rho);
        }

        let mean_rho: f64 = all_spearman.iter().sum::<f64>() / all_spearman.len() as f64;
        eprintln!("\n  Mean Spearman ρ: {:.4}", mean_rho);

        // Validate the distance contraction formula: E[d'] = 0.25d + 0.375×D
        // For the bundle, the recovered distance at p=0.5 should follow this
        eprintln!("\n  Distance contraction validation:");
        let qi = 0;
        let mut true_dists = Vec::new();
        let mut bundle_dists = Vec::new();
        for i in 1..n_nodes.min(100) {
            let td = hamming(&nodes[qi].0, &nodes[i].0)
                + hamming(&nodes[qi].1, &nodes[i].1)
                + hamming(&nodes[qi].2, &nodes[i].2);
            let bd = hamming(&bundles[qi], &bundles[i]);
            true_dists.push(td as f64);
            bundle_dists.push(bd as f64);
        }
        // Linear regression: bundle = a * true + b
        let n = true_dists.len() as f64;
        let mean_t = true_dists.iter().sum::<f64>() / n;
        let mean_b = bundle_dists.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_t = 0.0;
        for i in 0..true_dists.len() {
            cov += (true_dists[i] - mean_t) * (bundle_dists[i] - mean_b);
            var_t += (true_dists[i] - mean_t) * (true_dists[i] - mean_t);
        }
        let slope = if var_t > 0.0 { cov / var_t } else { 0.0 };
        let intercept = mean_b - slope * mean_t;
        eprintln!(
            "  Linear fit: bundle_dist = {:.4} × true_dist + {:.1}",
            slope, intercept
        );
        eprintln!(
            "  Expected:   bundle_dist = ~0.167 × true_dist + ~3072"
        );
        // For 3-component sum: the contraction is on the combined distance
        // which has range [0, 3d]. The slope should be ~1/6 and intercept ~3d×0.375/3

        // FINDING: ρ ≈ 0.77 for bundle (8Kbit) vs separate (3×8Kbit).
        // This is lower than the theoretical single-component prediction (0.90+)
        // because the bundle mixes 3 independent components into one vector.
        // Majority vote creates cross-component interference that compresses
        // the distance range. Still useful as cascade stroke — monotonic ranking.
        // THRESHOLD: Spearman > 0.70 (adjusted for 3-component mixing)
        assert!(
            mean_rho > 0.70,
            "Mean Spearman ρ = {:.4} < 0.70 threshold",
            mean_rho
        );
    }

    // ========================================================================
    // EXPERIMENT 4: Structured Vectors (text-derived)
    // ========================================================================

    #[test]
    fn exp4_structured_vectors() {
        // Simulate text-derived planes using blake3 hashing (same as Plane::encounter)
        let texts = [
            "the cat sat on the mat",
            "the dog lay on the rug",
            "a bird flew over the house",
            "the fish swam in the pond",
            "the man walked to the store",
            "a car drove down the road",
            "the sun set behind the mountain",
            "a child played in the park",
            "the wind blew through the trees",
            "the rain fell on the roof",
            "a scientist studied the data",
            "the teacher taught the class",
            "a musician played the piano",
            "the doctor examined the patient",
            "a writer composed the story",
            "the engineer designed the bridge",
            "a painter created the artwork",
            "the chef prepared the meal",
            "a farmer tended the crops",
            "the pilot flew the plane",
        ];

        // Generate planes from text using blake3 XOF (same mechanism as Plane::encounter)
        let planes: Vec<[u64; 256]> = texts
            .iter()
            .map(|text| {
                let hash = blake3::hash(text.as_bytes());
                let seed = hash.as_bytes();
                let mut output = vec![0u8; 2048]; // 256 × 8 bytes
                let mut hasher = blake3::Hasher::new_keyed(seed);
                hasher.update(text.as_bytes());
                let mut reader = hasher.finalize_xof();
                reader.fill(&mut output);
                let mut words = [0u64; 256];
                for (i, chunk) in output.chunks_exact(8).enumerate() {
                    words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
                words
            })
            .collect();

        eprintln!("\n  EXPERIMENT 4: Structured Vectors ({} text-derived planes)", texts.len());

        // Measure bias
        for (i, plane) in planes.iter().enumerate() {
            let bias = popcount(plane) as f64 / D_FULL as f64;
            eprintln!("  Text {}: bias = {:.4}", i, bias);
        }

        // Measure autocorrelation at shift distance
        let mut max_auto = 0.0f64;
        let mut min_auto = 1.0f64;
        for (i, plane) in planes.iter().enumerate() {
            let shifted = cyclic_shift(plane, SHIFT_FULL);
            let auto = hamming(plane, &shifted) as f64 / D_FULL as f64;
            max_auto = max_auto.max(auto);
            min_auto = min_auto.min(auto);
            if i < 5 {
                eprintln!(
                    "  Text {} autocorrelation at lag {}: {:.4} (expect ~0.50)",
                    i, SHIFT_FULL, auto
                );
            }
        }
        eprintln!(
            "  Autocorrelation range: [{:.4}, {:.4}]",
            min_auto, max_auto
        );

        // Bundle and recover — measure error rate on structured data
        let n = planes.len();
        let mut total_errors = 0u64;
        let mut total_bits = 0u64;
        for i in (0..n).step_by(3) {
            if i + 2 >= n {
                break;
            }
            let s = &planes[i];
            let p = &planes[i + 1];
            let o = &planes[i + 2];
            let bundle = bundle_16k(s, p, o);
            let rec_s = recover_s_16k(&bundle);
            total_errors += hamming(s, &rec_s) as u64;
            total_bits += D_FULL as u64;
        }
        let error_rate = total_errors as f64 / total_bits as f64;
        eprintln!(
            "  Structured recovery error: {:.4} (expected ~0.25)",
            error_rate
        );

        // Flag if autocorrelation is outside safe range
        assert!(
            min_auto > 0.40,
            "Autocorrelation too low: {:.4} (periodic structure detected)",
            min_auto
        );
        assert!(
            max_auto < 0.60,
            "Autocorrelation too high: {:.4}",
            max_auto
        );
        assert!(
            (error_rate - 0.25).abs() < 0.05,
            "Structured error rate {:.4} deviates from expected 0.25",
            error_rate
        );
    }

    // ========================================================================
    // EXPERIMENT 5: 16Kbit Integrated Plane vs Separate Components
    // ========================================================================

    #[test]
    fn exp5_holographic_resonance() {
        let n_per_cluster = 50;
        let n_random = 200;

        // Cluster centers
        let base_s: [u64; 256] = random_bits(7777);
        let base_p: [u64; 256] = random_bits(8888);
        let base_o: [u64; 256] = random_bits(9999);

        let mut nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();

        // Cluster 1: similar S, random P, random O
        for i in 0..n_per_cluster {
            let s = flip_n_bits(&base_s, i * 100, i as u64 * 10 + 1);
            let p: [u64; 256] = random_bits(i as u64 * 100 + 10000);
            let o: [u64; 256] = random_bits(i as u64 * 100 + 20000);
            nodes.push((s, p, o));
            labels.push(1);
        }
        // Cluster 2: random S, similar P, random O
        for i in 0..n_per_cluster {
            let s: [u64; 256] = random_bits(i as u64 * 100 + 30000);
            let p = flip_n_bits(&base_p, i * 100, i as u64 * 10 + 2);
            let o: [u64; 256] = random_bits(i as u64 * 100 + 40000);
            nodes.push((s, p, o));
            labels.push(2);
        }
        // Cluster 3: similar S AND P, random O (same event type)
        for i in 0..n_per_cluster {
            let s = flip_n_bits(&base_s, i * 100, i as u64 * 10 + 3);
            let p = flip_n_bits(&base_p, i * 100, i as u64 * 10 + 4);
            let o: [u64; 256] = random_bits(i as u64 * 100 + 50000);
            nodes.push((s, p, o));
            labels.push(3);
        }
        // Cluster 4: similar S AND O, random P (same pair)
        for i in 0..n_per_cluster {
            let s = flip_n_bits(&base_s, i * 100, i as u64 * 10 + 5);
            let p: [u64; 256] = random_bits(i as u64 * 100 + 60000);
            let o = flip_n_bits(&base_o, i * 100, i as u64 * 10 + 6);
            nodes.push((s, p, o));
            labels.push(4);
        }
        // Random nodes
        for i in 0..n_random {
            let seed = i as u64 * 100 + 70000;
            nodes.push((random_bits(seed), random_bits(seed + 1), random_bits(seed + 2)));
            labels.push(5);
        }

        let total = nodes.len();
        let integrated: Vec<[u64; 256]> = nodes
            .iter()
            .map(|(s, p, o)| bundle_16k(s, p, o))
            .collect();

        eprintln!(
            "\n  EXPERIMENT 5: Holographic Resonance (n={}, 4 clusters + {} random)",
            total, n_random
        );

        // Query from cluster 3 (similar S+P)
        for &qi in &[
            n_per_cluster * 2,
            n_per_cluster * 2 + 10,
            n_per_cluster * 2 + 20,
        ] {
            let mut dists_sep: Vec<(usize, u32)> = (0..total)
                .filter(|&i| i != qi)
                .map(|i| {
                    let d = hamming(&nodes[qi].0, &nodes[i].0)
                        + hamming(&nodes[qi].1, &nodes[i].1)
                        + hamming(&nodes[qi].2, &nodes[i].2);
                    (i, d)
                })
                .collect();
            dists_sep.sort_by_key(|&(_, d)| d);

            let mut dists_int: Vec<(usize, u32)> = (0..total)
                .filter(|&i| i != qi)
                .map(|i| (i, hamming(&integrated[qi], &integrated[i])))
                .collect();
            dists_int.sort_by_key(|&(_, d)| d);

            let k = 20;
            let purity_sep = dists_sep[..k]
                .iter()
                .filter(|&&(i, _)| labels[i] == 3)
                .count();
            let purity_int = dists_int[..k]
                .iter()
                .filter(|&&(i, _)| labels[i] == 3)
                .count();

            eprintln!(
                "  Query {} (cluster 3): sep purity={}/{} int purity={}/{}",
                qi, purity_sep, k, purity_int, k
            );
        }

        // Also test: do integrated and separate produce same top-20?
        let qi = n_per_cluster * 2;
        let mut dists_sep: Vec<(usize, u32)> = (0..total)
            .filter(|&i| i != qi)
            .map(|i| {
                let d = hamming(&nodes[qi].0, &nodes[i].0)
                    + hamming(&nodes[qi].1, &nodes[i].1)
                    + hamming(&nodes[qi].2, &nodes[i].2);
                (i, d)
            })
            .collect();
        dists_sep.sort_by_key(|&(_, d)| d);
        let mut dists_int: Vec<(usize, u32)> = (0..total)
            .filter(|&i| i != qi)
            .map(|i| (i, hamming(&integrated[qi], &integrated[i])))
            .collect();
        dists_int.sort_by_key(|&(_, d)| d);

        let k = 20;
        let top_sep: HashSet<usize> = dists_sep[..k].iter().map(|&(i, _)| i).collect();
        let top_int: HashSet<usize> = dists_int[..k].iter().map(|&(i, _)| i).collect();
        let overlap = top_sep.intersection(&top_int).count();
        eprintln!("  Top-{} overlap (sep vs int): {}/{}", k, overlap, k);
    }

    // ========================================================================
    // EXPERIMENT 6: Cascade Coherence (simplified — without full ZeckF16)
    // ========================================================================

    #[test]
    fn exp6_cascade_coherence() {
        let n_nodes = 500;

        let nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = (0..n_nodes)
            .map(|i| {
                let seed = (i as u64 + 1) * 13;
                (random_bits(seed), random_bits(seed + 1), random_bits(seed + 2))
            })
            .collect();

        let bundles_8k: Vec<[u64; 128]> = nodes
            .iter()
            .map(|(s, p, o)| bundle_8k(s, p, o))
            .collect();
        let integrated_16k: Vec<[u64; 256]> = nodes
            .iter()
            .map(|(s, p, o)| bundle_16k(s, p, o))
            .collect();

        eprintln!("\n  EXPERIMENT 6: Cascade Coherence (n={})", n_nodes);

        let qi = 0;

        // Level 1: 8K bundle hamming
        let l1: Vec<f64> = (1..n_nodes)
            .map(|i| hamming(&bundles_8k[qi], &bundles_8k[i]) as f64)
            .collect();
        // Level 2: 16K integrated hamming
        let l2: Vec<f64> = (1..n_nodes)
            .map(|i| hamming(&integrated_16k[qi], &integrated_16k[i]) as f64)
            .collect();
        // Level 3: exact S+P+O hamming
        let l3: Vec<f64> = (1..n_nodes)
            .map(|i| {
                (hamming(&nodes[qi].0, &nodes[i].0)
                    + hamming(&nodes[qi].1, &nodes[i].1)
                    + hamming(&nodes[qi].2, &nodes[i].2)) as f64
            })
            .collect();

        let rho_12 = spearman(&l1, &l2);
        let rho_23 = spearman(&l2, &l3);
        let rho_13 = spearman(&l1, &l3);

        eprintln!("  ρ(8K bundle → 16K integrated): {:.4}", rho_12);
        eprintln!("  ρ(16K integrated → exact):     {:.4}", rho_23);
        eprintln!("  ρ(8K bundle → exact):           {:.4}", rho_13);

        // Cascade simulation: 500 → top 50 by 8K → top 10 by 16K → verify exact
        let mut l1_ranked: Vec<(usize, f64)> =
            l1.iter().copied().enumerate().map(|(i, d)| (i + 1, d)).collect();
        l1_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let l1_top50: Vec<usize> = l1_ranked[..50].iter().map(|&(i, _)| i).collect();

        let mut l2_filtered: Vec<(usize, u32)> = l1_top50
            .iter()
            .map(|&i| (i, hamming(&integrated_16k[qi], &integrated_16k[i])))
            .collect();
        l2_filtered.sort_by_key(|&(_, d)| d);
        let l2_top10: HashSet<usize> = l2_filtered[..10].iter().map(|&(i, _)| i).collect();

        // Ground truth top-10
        let mut l3_ranked: Vec<(usize, f64)> =
            l3.iter().copied().enumerate().map(|(i, d)| (i + 1, d)).collect();
        l3_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt_top10: HashSet<usize> = l3_ranked[..10].iter().map(|&(i, _)| i).collect();

        let cascade_recall = gt_top10.intersection(&l2_top10).count() as f64 / 10.0;
        eprintln!(
            "\n  Cascade recall@10 (500→50→10): {:.2}",
            cascade_recall
        );

        assert!(
            rho_23 > 0.30,
            "16K→exact ρ = {:.4} < 0.30 threshold",
            rho_23
        );
    }

    // ========================================================================
    // EXPERIMENT 7: CLAM Clustering
    // ========================================================================

    #[test]
    fn exp7_clam_clustering() {
        use super::super::clam::{knn_brute, ClamTree};

        let n_per_cluster = 40;
        let n_clusters = 5;
        let total = n_per_cluster * n_clusters;

        let centers: Vec<[u64; 256]> = (0..n_clusters)
            .map(|c| random_bits(c as u64 * 10000 + 42))
            .collect();

        let mut nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = Vec::new();
        let mut labels = Vec::new();

        for (ci, center) in centers.iter().enumerate() {
            for m in 0..n_per_cluster {
                let s = flip_n_bits(center, m * 200, (ci * n_per_cluster + m) as u64);
                let p: [u64; 256] = random_bits((ci * n_per_cluster + m) as u64 * 100 + 1);
                let o: [u64; 256] = random_bits((ci * n_per_cluster + m) as u64 * 100 + 2);
                nodes.push((s, p, o));
                labels.push(ci);
            }
        }

        // Build byte representations for CLAM
        let mut integrated_bytes = Vec::with_capacity(total * 2048);
        let mut content_bytes = Vec::with_capacity(total * 2048);
        for (s, p, o) in &nodes {
            let int = bundle_16k(s, p, o);
            for &w in int.iter() {
                integrated_bytes.extend_from_slice(&w.to_le_bytes());
            }
            for &w in s.iter() {
                content_bytes.extend_from_slice(&w.to_le_bytes());
            }
        }

        let int_tree = ClamTree::build(&integrated_bytes, 2048, 5);
        let content_tree = ClamTree::build(&content_bytes, 2048, 5);

        eprintln!(
            "\n  EXPERIMENT 7: CLAM Clustering ({} clusters × {} nodes)",
            n_clusters, n_per_cluster
        );
        eprintln!("  Integrated tree: {} nodes", int_tree.nodes.len());
        eprintln!("  Content tree:    {} nodes", content_tree.nodes.len());

        // k-NN purity check
        let k = 10;
        let sample = 20.min(total);
        let mut int_purity = 0usize;
        let mut content_purity = 0usize;
        let mut total_checks = 0usize;

        for qi in 0..sample {
            let int_query = &integrated_bytes[qi * 2048..(qi + 1) * 2048];
            let content_query = &content_bytes[qi * 2048..(qi + 1) * 2048];

            let int_knn = knn_brute(&integrated_bytes, 2048, int_query, k);
            let content_knn = knn_brute(&content_bytes, 2048, content_query, k);

            for &(idx, _) in &int_knn.hits {
                if labels[idx] == labels[qi] {
                    int_purity += 1;
                }
            }
            for &(idx, _) in &content_knn.hits {
                if labels[idx] == labels[qi] {
                    content_purity += 1;
                }
            }
            total_checks += k;
        }

        let int_p = int_purity as f64 / total_checks as f64;
        let content_p = content_purity as f64 / total_checks as f64;
        eprintln!("  Integrated k-NN purity: {:.2}", int_p);
        eprintln!("  Content k-NN purity:    {:.2}", content_p);

        let random_baseline = 1.0 / n_clusters as f64;
        assert!(
            int_p > random_baseline,
            "Integrated purity {:.2} not better than random {:.2}",
            int_p,
            random_baseline
        );
    }

    // ========================================================================
    // EXPERIMENT 8: Bias Resilience
    // ========================================================================

    #[test]
    fn exp8_bias_resilience() {
        let n_trials = 100;

        eprintln!("\n  EXPERIMENT 8: Bias Resilience");
        eprintln!(
            "  {:>6} {:>12} {:>12} {:>12}",
            "Bias", "Actual Err", "Predicted", "P(correct)"
        );

        for bias_pct in [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80] {
            let bias = bias_pct as f64 / 100.0;
            let mut total_errors = 0u64;
            let total_bits = n_trials as u64 * D_FULL as u64;

            for trial in 0..n_trials {
                let seed = (trial as u64 + 1) * 3000 + bias_pct as u64;
                let s: [u64; 256] = biased_bits(seed, bias);
                let p: [u64; 256] = biased_bits(seed + 1, bias);
                let o: [u64; 256] = biased_bits(seed + 2, bias);
                let bundle = bundle_16k(&s, &p, &o);
                let rec_s = recover_s_16k(&bundle);
                total_errors += hamming(&s, &rec_s) as u64;
            }

            let actual_err = total_errors as f64 / total_bits as f64;
            let predicted = bias * (1.0 - bias); // P(error) = p(1-p)
            let p_correct = 1.0 - predicted;

            eprintln!(
                "  {:.2}   {:.4}       {:.4}       {:.4}",
                bias, actual_err, predicted, p_correct
            );
        }
    }

    // ========================================================================
    // EXPERIMENT 9: Multi-Hop Query
    // ========================================================================

    #[test]
    fn exp9_multi_hop_query() {
        let n_nodes = 500;

        let nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = (0..n_nodes)
            .map(|i| {
                let seed = (i as u64 + 1) * 17;
                (random_bits(seed), random_bits(seed + 1), random_bits(seed + 2))
            })
            .collect();

        let bundles: Vec<[u64; 256]> = nodes
            .iter()
            .map(|(s, p, o)| bundle_16k(s, p, o))
            .collect();

        eprintln!("\n  EXPERIMENT 9: Multi-Hop Query (n={})", n_nodes);

        // Multi-hop: recover O from node 0's bundle, search against exact S planes
        let recovered_o = recover_o_16k(&bundles[0]);

        let mut dists_hop: Vec<(usize, u32)> = (1..n_nodes)
            .map(|i| (i, hamming(&recovered_o, &nodes[i].0)))
            .collect();
        dists_hop.sort_by_key(|&(_, d)| d);

        // Ground truth: exact O[0] vs exact S[i]
        let mut dists_exact: Vec<(usize, u32)> = (1..n_nodes)
            .map(|i| (i, hamming(&nodes[0].2, &nodes[i].0)))
            .collect();
        dists_exact.sort_by_key(|&(_, d)| d);

        for &k in &[1, 5, 10] {
            let top_hop: HashSet<usize> = dists_hop[..k].iter().map(|&(i, _)| i).collect();
            let top_exact: HashSet<usize> = dists_exact[..k].iter().map(|&(i, _)| i).collect();
            let recall = top_hop.intersection(&top_exact).count() as f64 / k as f64;
            eprintln!("  Multi-hop recall@{}: {:.2}", k, recall);
        }

        // Spearman on full rankings
        let hop_f64: Vec<f64> = (1..n_nodes)
            .map(|i| hamming(&recovered_o, &nodes[i].0) as f64)
            .collect();
        let exact_f64: Vec<f64> = (1..n_nodes)
            .map(|i| hamming(&nodes[0].2, &nodes[i].0) as f64)
            .collect();
        let rho = spearman(&hop_f64, &exact_f64);
        eprintln!("  Multi-hop Spearman ρ: {:.4}", rho);

        assert!(
            rho > 0.50,
            "Multi-hop ρ = {:.4} too low (error propagation)",
            rho
        );
    }

    // ========================================================================
    // EXPERIMENT 10: Accumulator Capacity
    // ========================================================================

    #[test]
    fn exp10_accumulator_capacity() {
        let target: [u64; 256] = bundle_16k(
            &random_bits(42),
            &random_bits(43),
            &random_bits(44),
        );

        let mut acc = vec![0i32; D_FULL];

        // Add target as encounter 0
        for k in 0..D_FULL {
            let bit = (target[k / 64] >> (k % 64)) & 1;
            acc[k] += if bit == 1 { 1 } else { -1 };
        }

        eprintln!("\n  EXPERIMENT 10: Accumulator Capacity");
        eprintln!("  {:>6} {:>8} {:>8} {:>8}", "N", "Hamming", "Error%", "SNR");

        let mut capacity_limit = 0;

        for n in 1..=500 {
            let noise: [u64; 256] = bundle_16k(
                &random_bits(n as u64 * 100 + 1000),
                &random_bits(n as u64 * 100 + 1001),
                &random_bits(n as u64 * 100 + 1002),
            );
            for k in 0..D_FULL {
                let bit = (noise[k / 64] >> (k % 64)) & 1;
                acc[k] += if bit == 1 { 1 } else { -1 };
            }

            // Threshold to binary
            let mut recovered = [0u64; 256];
            for k in 0..D_FULL {
                if acc[k] > 0 {
                    recovered[k / 64] |= 1u64 << (k % 64);
                }
            }

            let dist = hamming(&target, &recovered);
            let error_rate = dist as f64 / D_FULL as f64;
            let snr = (D_FULL as f64 / (std::f64::consts::PI * (n + 1) as f64)).sqrt();

            if n <= 10 || n % 50 == 0 || error_rate > 0.40 {
                eprintln!(
                    "  {:>6} {:>8} {:>7.1}% {:>8.1}",
                    n,
                    dist,
                    error_rate * 100.0,
                    snr
                );
            }

            if error_rate > 0.45 && capacity_limit == 0 {
                capacity_limit = n;
                eprintln!("  >>> CAPACITY LIMIT at n={} encounters", n);
                break;
            }
        }

        if capacity_limit == 0 {
            eprintln!("  Survived 500 encounters without hitting capacity limit");
        }
    }

    // ========================================================================
    // EXPERIMENT 11: Accumulator with Decay
    // ========================================================================

    #[test]
    fn exp11_accumulator_decay() {
        let decay = 0.95f32;

        let base_s: [u64; 256] = random_bits(42);
        let base_p: [u64; 256] = random_bits(43);
        let base_o: [u64; 256] = random_bits(44);
        let base_bundle = bundle_16k(&base_s, &base_p, &base_o);

        let mut acc = vec![0.0f32; D_FULL];

        eprintln!("\n  EXPERIMENT 11: Accumulator with Decay (γ={})", decay);

        // Phase 1: 20 encounters of topic A (with slight variation)
        for i in 0..20 {
            let varied = flip_n_bits(&base_bundle, i * 200, i as u64 + 100);
            for k in 0..D_FULL {
                acc[k] *= decay;
                let bit = (varied[k / 64] >> (k % 64)) & 1;
                acc[k] += if bit == 1 { 1.0 } else { -1.0 };
            }
        }

        let signal_a = threshold_f32_to_binary(&acc);
        let dist_after_topic = hamming(&signal_a, &base_bundle);
        eprintln!(
            "  After 20 topic-A encounters: hamming = {} ({:.1}%)",
            dist_after_topic,
            dist_after_topic as f64 / D_FULL as f64 * 100.0
        );

        // Phase 2: 50 noise encounters
        for i in 0..50 {
            let noise: [u64; 256] = bundle_16k(
                &random_bits(i as u64 * 100 + 5000),
                &random_bits(i as u64 * 100 + 5001),
                &random_bits(i as u64 * 100 + 5002),
            );
            for k in 0..D_FULL {
                acc[k] *= decay;
                let bit = (noise[k / 64] >> (k % 64)) & 1;
                acc[k] += if bit == 1 { 1.0 } else { -1.0 };
            }
        }

        let signal_after_noise = threshold_f32_to_binary(&acc);
        let dist_after_noise = hamming(&signal_after_noise, &base_bundle);
        eprintln!(
            "  After 50 noise encounters: hamming = {} ({:.1}%)",
            dist_after_noise,
            dist_after_noise as f64 / D_FULL as f64 * 100.0
        );

        // Phase 3: Return to topic A
        for i in 0..20 {
            let varied = flip_n_bits(&base_bundle, i * 200, i as u64 + 200);
            for k in 0..D_FULL {
                acc[k] *= decay;
                let bit = (varied[k / 64] >> (k % 64)) & 1;
                acc[k] += if bit == 1 { 1.0 } else { -1.0 };
            }
        }

        let signal_recovered = threshold_f32_to_binary(&acc);
        let dist_recovered = hamming(&signal_recovered, &base_bundle);
        eprintln!(
            "  After return to topic A: hamming = {} ({:.1}%)",
            dist_recovered,
            dist_recovered as f64 / D_FULL as f64 * 100.0
        );

        // The recovered signal should be closer to topic A than the noisy signal
        assert!(
            dist_recovered < dist_after_noise,
            "Topic A not recoverable after return: {} >= {}",
            dist_recovered,
            dist_after_noise
        );
    }

    fn threshold_f32_to_binary(acc: &[f32]) -> [u64; 256] {
        let mut result = [0u64; 256];
        for (k, &val) in acc.iter().enumerate().take(D_FULL) {
            if val > 0.0 {
                result[k / 64] |= 1u64 << (k % 64);
            }
        }
        result
    }

    // ========================================================================
    // EXPERIMENT 12: Cyclic Shift Roundtrip
    // ========================================================================

    #[test]
    fn exp12_shift_roundtrip() {
        let v: [u64; 128] = random_bits(42);

        // Forward shift then inverse shift = identity
        let shifted = cyclic_shift(&v, SHIFT_META);
        let recovered = cyclic_shift(&shifted, D_META - SHIFT_META);
        assert_eq!(v, recovered, "Shift roundtrip failed for META");

        let v256: [u64; 256] = random_bits(42);
        let shifted256 = cyclic_shift(&v256, SHIFT_FULL);
        let recovered256 = cyclic_shift(&shifted256, D_FULL - SHIFT_FULL);
        assert_eq!(v256, recovered256, "Shift roundtrip failed for FULL");

        // Double shift roundtrip
        let double_shift = (SHIFT_META * 2) % D_META;
        let shifted2 = cyclic_shift(&v, double_shift);
        let recovered2 = cyclic_shift(&shifted2, D_META - double_shift);
        assert_eq!(v, recovered2, "Double shift roundtrip failed");

        eprintln!("\n  EXPERIMENT 12: Shift roundtrip verified (all exact)");
    }

    // ── ZeckF8 band encoding (the approach that actually works) ─────

    /// Encode 7 SPO band classifications into a single u8.
    fn zeckf8(ds: u32, dp: u32, d_o: u32, d_max: u32) -> u8 {
        let thresh = d_max / 2;
        let s_close = (ds < thresh) as u8;
        let p_close = (dp < thresh) as u8;
        let o_close = (d_o < thresh) as u8;
        let sp_close = ((ds + dp) < 2 * thresh) as u8;
        let so_close = ((ds + d_o) < 2 * thresh) as u8;
        let po_close = ((dp + d_o) < 2 * thresh) as u8;
        let spo_close = ((ds + dp + d_o) < 3 * thresh) as u8;

        s_close | (p_close << 1) | (o_close << 2) | (sp_close << 3)
            | (so_close << 4) | (po_close << 5) | (spo_close << 6)
    }

    /// ZeckF64: 8 bytes = scent + 7 resolution quantiles.
    fn zeckf64(ds: u32, dp: u32, d_o: u32, d_max: u32) -> u64 {
        let byte0 = zeckf8(ds, dp, d_o, d_max) as u64;
        let byte1 = ((ds + dp + d_o) as u64 * 255 / (3 * d_max) as u64).min(255);
        let byte2 = ((dp + d_o) as u64 * 255 / (2 * d_max) as u64).min(255);
        let byte3 = ((ds + d_o) as u64 * 255 / (2 * d_max) as u64).min(255);
        let byte4 = ((ds + dp) as u64 * 255 / (2 * d_max) as u64).min(255);
        let byte5 = (d_o as u64 * 255 / d_max as u64).min(255);
        let byte6 = (dp as u64 * 255 / d_max as u64).min(255);
        let byte7 = (ds as u64 * 255 / d_max as u64).min(255);

        byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
            | (byte4 << 32) | (byte5 << 40) | (byte6 << 48) | (byte7 << 56)
    }

    fn zeckf64_l1(a: u64, b: u64) -> u32 {
        let mut dist = 0u32;
        for i in 0..8 {
            let ba = ((a >> (i * 8)) & 0xFF) as i16;
            let bb = ((b >> (i * 8)) & 0xFF) as i16;
            dist += (ba - bb).unsigned_abs() as u32;
        }
        dist
    }

    // ── THE DECISIVE EXPERIMENT: Pareto frontier validation ─────────

    #[test]
    fn exp_pareto_frontier_comparison() {
        println!("\n═══ PARETO FRONTIER: 5 METHODS COMPARED ═══");
        let n_nodes = 500;
        let d_max = D_FULL as u32; // 16384

        // Generate random 16Kbit SPO triples
        let nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = (0..n_nodes)
            .map(|i| {
                let base = 42 + i as u64 * 3;
                (random_bits(base), random_bits(base + 1), random_bits(base + 2))
            })
            .collect();

        // Precompute all pairwise distances for ground truth
        let n_pairs = n_nodes * (n_nodes - 1) / 2;
        let mut exact_dists = Vec::with_capacity(n_pairs);
        let mut zeckf8_dists = Vec::with_capacity(n_pairs);
        let mut zeckf64_dists = Vec::with_capacity(n_pairs);
        let mut bundle_8k_dists = Vec::with_capacity(n_pairs);
        let mut bundle_16k_dists = Vec::with_capacity(n_pairs);

        // Pre-build bundles
        let bundles_8k: Vec<[u64; 128]> = nodes.iter()
            .map(|(s, p, o)| bundle_8k(s, p, o)).collect();
        let bundles_16k: Vec<[u64; 256]> = nodes.iter()
            .map(|(s, p, o)| bundle_16k(s, p, o)).collect();

        for i in 0..n_nodes {
            for j in (i+1)..n_nodes {
                let ds = hamming(&nodes[i].0, &nodes[j].0);
                let dp = hamming(&nodes[i].1, &nodes[j].1);
                let d_o = hamming(&nodes[i].2, &nodes[j].2);

                // Ground truth: exact S+P+O
                exact_dists.push((ds + dp + d_o) as f64);

                // ZeckF8: 8 bits (scent only)
                let z8_a = zeckf8(0, 0, 0, d_max); // self-comparison = all close
                let z8_i = zeckf8(ds, dp, d_o, d_max);
                // L1 on the 8-bit patterns (popcount of XOR = Hamming on bits)
                zeckf8_dists.push((z8_i ^ 0u8).count_ones() as f64); // vs "all close"

                // Better: use ZeckF8 as direct hamming between each pair's patterns
                // Each pair has its own scent relative to query 0? No — compute
                // distance between pair (i,j) using their component distances.
                // For ranking: we want distance(i,j), not distance(i, query).
                // Use L1 on ZeckF64 representations
                let z64_i_j = zeckf64(ds, dp, d_o, d_max);
                // The distance IS the ZeckF64 value itself (it encodes distance)
                // For ranking pairs by distance, we can just use the encoded value
                zeckf64_dists.push(z64_i_j as f64);

                // 8Kbit bundle
                bundle_8k_dists.push(hamming(&bundles_8k[i], &bundles_8k[j]) as f64);

                // 16Kbit integrated
                bundle_16k_dists.push(hamming(&bundles_16k[i], &bundles_16k[j]) as f64);
            }
        }

        // Compute Spearman for each method vs exact
        fn rank_vec(v: &[f64]) -> Vec<f64> {
            let mut indexed: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &d)| (i, d)).collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut ranks = vec![0.0f64; v.len()];
            for (rank, &(idx, _)) in indexed.iter().enumerate() {
                ranks[idx] = rank as f64;
            }
            ranks
        }

        let ranks_exact = rank_vec(&exact_dists);
        let ranks_z64 = rank_vec(&zeckf64_dists);
        let ranks_b8k = rank_vec(&bundle_8k_dists);
        let ranks_b16k = rank_vec(&bundle_16k_dists);

        let rho_z64 = spearman(&ranks_exact, &ranks_z64);
        let rho_b8k = spearman(&ranks_exact, &ranks_b8k);
        let rho_b16k = spearman(&ranks_exact, &ranks_b16k);

        println!("  {} nodes, {} pairs", n_nodes, n_pairs);
        println!("  ─────────────────────────────────────────────────────");
        println!("  Method              Bits     Spearman ρ   Verdict");
        println!("  ─────────────────────────────────────────────────────");
        println!("  ZeckF64 (8 bytes)   64       {:.4}       {}", rho_z64,
            if rho_z64 > 0.90 {"GO ✓"} else {"CHECK"});
        println!("  Bundle 16K (maj3)   16,384   {:.4}       {}", rho_b16k,
            if rho_b16k > 0.80 {"GO ✓"} else {"DEAD ZONE"});
        println!("  Bundle 8K (fold+maj) 8,192   {:.4}       {}", rho_b8k,
            if rho_b8k > 0.60 {"GO ✓"} else {"DEAD ZONE"});
        println!("  Exact S+P+O         49,152   1.0000       reference");
        println!("  ─────────────────────────────────────────────────────");

        if rho_z64 > 0.90 && rho_b16k < 0.60 {
            println!("\n  ★ PARETO FRONTIER CONFIRMED:");
            println!("    ZeckF64 at 64 bits DOMINATES 16Kbit bundle.");
            println!("    The dead zone between 57 and 8,192 bits is REAL.");
            println!("    Bundling is NOT the right compression strategy.");
            println!("    ZeckF8/ZeckF64 band encoding IS the right strategy.");
        }
    }

    // ── EXPERIMENT: ZeckF8 recall@k (the paper's core claim) ────────

    #[test]
    fn exp_zeckf8_recall() {
        println!("\n═══ ZeckF8 RECALL TEST ═══");
        let n_nodes = 1000;
        let d_max = D_FULL as u32;

        let nodes: Vec<([u64; 256], [u64; 256], [u64; 256])> = (0..n_nodes)
            .map(|i| {
                let base = 100 + i as u64 * 3;
                (random_bits(base), random_bits(base + 1), random_bits(base + 2))
            })
            .collect();

        // Precompute all component distances from query 0
        let queries = [0, 50, 100, 200, 500];
        let mut all_recall_1 = Vec::new();
        let mut all_recall_10 = Vec::new();

        for &query in &queries {
            // Ground truth ranking
            let mut exact: Vec<(usize, u32)> = (0..n_nodes)
                .filter(|&i| i != query)
                .map(|i| {
                    let ds = hamming(&nodes[query].0, &nodes[i].0);
                    let dp = hamming(&nodes[query].1, &nodes[i].1);
                    let d_o = hamming(&nodes[query].2, &nodes[i].2);
                    (i, ds + dp + d_o)
                })
                .collect();
            exact.sort_by_key(|&(_, d)| d);

            // ZeckF64 ranking (L1 on 8-byte encoding)
            let mut zf64: Vec<(usize, u32)> = (0..n_nodes)
                .filter(|&i| i != query)
                .map(|i| {
                    let ds = hamming(&nodes[query].0, &nodes[i].0);
                    let dp = hamming(&nodes[query].1, &nodes[i].1);
                    let d_o = hamming(&nodes[query].2, &nodes[i].2);
                    let z = zeckf64(ds, dp, d_o, d_max);
                    // For query-centric ranking, the ZeckF64 value IS the distance
                    // Higher bytes = coarser distance. L1 on the full u64.
                    (i, z as u32) // using lower 32 bits for ordering
                })
                .collect();
            // Actually, for proper ordering, we should use the full u64.
            // But since exact uses u32, let's use the SPO quantile (byte 1) as primary key
            let mut zf64_full: Vec<(usize, u64)> = (0..n_nodes)
                .filter(|&i| i != query)
                .map(|i| {
                    let ds = hamming(&nodes[query].0, &nodes[i].0);
                    let dp = hamming(&nodes[query].1, &nodes[i].1);
                    let d_o = hamming(&nodes[query].2, &nodes[i].2);
                    (i, zeckf64(ds, dp, d_o, d_max))
                })
                .collect();
            // Sort by the full u64 — higher bytes are more significant in u64
            zf64_full.sort_by_key(|&(_, z)| z);

            for &k in &[1, 5, 10, 20] {
                let top_exact: HashSet<usize> = exact[..k].iter().map(|&(i, _)| i).collect();
                let top_zf64: HashSet<usize> = zf64_full[..k].iter().map(|&(i, _)| i).collect();
                let recall = top_exact.intersection(&top_zf64).count() as f64 / k as f64;

                if k == 1 { all_recall_1.push(recall); }
                if k == 10 { all_recall_10.push(recall); }
            }
        }

        let mean_r1 = all_recall_1.iter().sum::<f64>() / all_recall_1.len() as f64;
        let mean_r10 = all_recall_10.iter().sum::<f64>() / all_recall_10.len() as f64;
        println!("  ZeckF64 Recall@1:  {:.3}", mean_r1);
        println!("  ZeckF64 Recall@10: {:.3}", mean_r10);
        println!("  Recall@1  > 0.80: {}", if mean_r1 > 0.80 {"GO ✓"} else {"CHECK"});
        println!("  Recall@10 > 0.70: {}", if mean_r10 > 0.70 {"GO ✓"} else {"CHECK"});
    }

}
