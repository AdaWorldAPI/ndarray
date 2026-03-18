//! Cyclic-permutation bundling benchmark for n=3 atoms (S, P, O).
//!
//! This is the "One Experiment" benchmark for the PRODUCTION mechanism:
//! cyclic shift + majority vote on binary fingerprints.
//!
//! Key parameters:
//! - d = 8192 bits (128 × u64)
//! - GOLDEN_SHIFT = floor(8192 / φ²) = 3130
//! - Alternative shift = 3131 (odd, gcd(3131,8192) = 1)
//! - n = 3 atoms: S (identity), P (shift by GOLDEN_SHIFT), O (shift by 2×GOLDEN_SHIFT mod d)
//! - Bundling: majority_vote_3 — bit-parallel: (a&b)|(a&c)|(b&c)
//! - Recovery: inverse cyclic shift

/// Total number of bits: 128 × 64 = 8192.
const D: usize = 8192;

/// Number of u64 words.
const N: usize = 128;

/// Golden shift: floor(8192 / φ²) = 3130.
const GOLDEN_SHIFT: usize = 3130;

/// Alternative shift with gcd(3131, 8192) = 1.
const GOLDEN_SHIFT_ODD: usize = 3131;

// ============================================================================
// PRNG (xorshift64)
// ============================================================================

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ============================================================================
// Core operations
// ============================================================================

/// Generate a random 128-word fingerprint from a seed.
pub fn random_128(seed: u64) -> [u64; N] {
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = [0u64; N];
    for w in out.iter_mut() {
        *w = xorshift64(&mut state);
    }
    out
}

/// Generate a sparse fingerprint where each bit is set with probability `density`.
/// density is in [0.0, 1.0].
pub fn sparse_128(seed: u64, density: f64) -> [u64; N] {
    let threshold = (density * u64::MAX as f64) as u64;
    let mut state = seed.wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = [0u64; N];
    for w in out.iter_mut() {
        let mut word = 0u64;
        for bit in 0..64 {
            let r = xorshift64(&mut state);
            if r <= threshold {
                word |= 1u64 << bit;
            }
        }
        *w = word;
    }
    out
}

/// Generate a biased fingerprint where each bit is 1 with probability `p`.
pub fn biased_128(seed: u64, p: f64) -> [u64; N] {
    sparse_128(seed, p)
}

/// Cyclic left shift of a bit-vector by `shift` positions.
///
/// Treats the 128-word array as a single 8192-bit vector and performs
/// a circular left shift by `shift` bits.
pub fn cyclic_shift(bits: &[u64; N], shift: usize) -> [u64; N] {
    let shift = shift % D;
    if shift == 0 {
        return *bits;
    }

    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut out = [0u64; N];

    if bit_shift == 0 {
        for i in 0..N {
            out[i] = bits[(i + word_shift) % N];
        }
    } else {
        let complement = 64 - bit_shift;
        for i in 0..N {
            let hi_idx = (i + word_shift) % N;
            let lo_idx = (i + word_shift + 1) % N;
            out[i] = (bits[hi_idx] << bit_shift) | (bits[lo_idx] >> complement);
        }
    }
    out
}

/// Inverse cyclic shift (right shift by `shift` positions).
#[inline]
pub fn cyclic_shift_inv(bits: &[u64; N], shift: usize) -> [u64; N] {
    let shift = shift % D;
    if shift == 0 {
        return *bits;
    }
    cyclic_shift(bits, D - shift)
}

/// Majority vote of 3 binary vectors: (a&b)|(a&c)|(b&c).
pub fn majority_vote_3(a: &[u64; N], b: &[u64; N], c: &[u64; N]) -> [u64; N] {
    let mut out = [0u64; N];
    for i in 0..N {
        out[i] = (a[i] & b[i]) | (a[i] & c[i]) | (b[i] & c[i]);
    }
    out
}

/// Bundle SPO triple: S stays at identity, P shifted by `shift`, O shifted by 2*shift.
pub fn bundle_spo(
    s: &[u64; N],
    p: &[u64; N],
    o: &[u64; N],
    shift: usize,
) -> [u64; N] {
    let p_shifted = cyclic_shift(p, shift);
    let o_shifted = cyclic_shift(o, 2 * shift);
    majority_vote_3(s, &p_shifted, &o_shifted)
}

/// Recover S from bundle: majority vote already has S at identity position.
/// We just inverse-shift the other components out and take a direct comparison.
///
/// For n=3 majority vote, the recovered S is just the bundle itself
/// (S was not shifted). The bundle preserves the majority bit at each position.
pub fn recover_s(bundle: &[u64; N], _shift: usize) -> [u64; N] {
    *bundle
}

/// Recover P from bundle: inverse-shift by `shift`, then the P component
/// is at identity position in the result.
pub fn recover_p(bundle: &[u64; N], shift: usize) -> [u64; N] {
    cyclic_shift_inv(bundle, shift)
}

/// Recover O from bundle: inverse-shift by 2*shift.
pub fn recover_o(bundle: &[u64; N], shift: usize) -> [u64; N] {
    cyclic_shift_inv(bundle, 2 * shift)
}

/// Hamming distance between two 128-word vectors.
pub fn hamming_128(a: &[u64; N], b: &[u64; N]) -> u32 {
    let mut count = 0u32;
    for i in 0..N {
        count += (a[i] ^ b[i]).count_ones();
    }
    count
}

/// Popcount of a 128-word vector.
pub fn popcount_128(a: &[u64; N]) -> u32 {
    let mut count = 0u32;
    for w in a.iter() {
        count += w.count_ones();
    }
    count
}

/// Per-bit accuracy: fraction of bits that match between recovered and original.
pub fn bit_accuracy(original: &[u64; N], recovered: &[u64; N]) -> f64 {
    let diff = hamming_128(original, recovered);
    1.0 - diff as f64 / D as f64
}

// ============================================================================
// Spearman rank correlation helper
// ============================================================================

fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = rank(x);
    let ry = rank(y);
    let n = x.len() as f64;
    let mean_rx: f64 = rx.iter().sum::<f64>() / n;
    let mean_ry: f64 = ry.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = rx[i] - mean_rx;
        let dy = ry[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < 1e-12 || var_y < 1e-12 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

// ============================================================================
// Tests — The Six Experiments
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sanity checks ──────────────────────────────────────────────

    #[test]
    fn test_cyclic_shift_roundtrip() {
        let v = random_128(42);
        for &shift in &[0, 1, 63, 64, 65, 3130, 3131, 4096, 8191] {
            let shifted = cyclic_shift(&v, shift);
            let recovered = cyclic_shift_inv(&shifted, shift);
            assert_eq!(v, recovered, "roundtrip failed for shift={shift}");
        }
    }

    #[test]
    fn test_majority_vote_identity() {
        let a = random_128(1);
        // majority_vote_3(a, a, a) == a
        let result = majority_vote_3(&a, &a, &a);
        assert_eq!(a, result);
    }

    #[test]
    fn test_majority_vote_two_of_three() {
        let a = random_128(10);
        let b = random_128(20);
        // majority_vote_3(a, a, b) should equal a at positions where a bits agree,
        // and a at positions where a disagrees with b (since 2 copies of a).
        let result = majority_vote_3(&a, &a, &b);
        assert_eq!(a, result);
    }

    #[test]
    fn test_hamming_self_zero() {
        let v = random_128(99);
        assert_eq!(hamming_128(&v, &v), 0);
    }

    #[test]
    fn test_hamming_complement() {
        let v = random_128(42);
        let mut complement = [0u64; N];
        for i in 0..N {
            complement[i] = !v[i];
        }
        assert_eq!(hamming_128(&v, &complement), D as u32);
    }

    // ── EXPERIMENT 1: Per-bit recovery accuracy ────────────────────

    #[test]
    fn experiment_1_perbit_recovery_accuracy() {
        let num_trials = 100;

        for &shift in &[GOLDEN_SHIFT, GOLDEN_SHIFT_ODD] {
            let mut acc_s_sum = 0.0;
            let mut acc_p_sum = 0.0;
            let mut acc_o_sum = 0.0;
            let mut min_acc = 1.0f64;

            for trial in 0..num_trials {
                let seed_base = (trial as u64) * 1000;
                let s = random_128(seed_base + 1);
                let p = random_128(seed_base + 2);
                let o = random_128(seed_base + 3);

                let bundle = bundle_spo(&s, &p, &o, shift);

                let rec_s = recover_s(&bundle, shift);
                let rec_p = recover_p(&bundle, shift);
                let rec_o = recover_o(&bundle, shift);

                let acc_s = bit_accuracy(&s, &rec_s);
                let acc_p = bit_accuracy(&p, &rec_p);
                let acc_o = bit_accuracy(&o, &rec_o);

                acc_s_sum += acc_s;
                acc_p_sum += acc_p;
                acc_o_sum += acc_o;

                min_acc = min_acc.min(acc_s).min(acc_p).min(acc_o);
            }

            let mean_s = acc_s_sum / num_trials as f64;
            let mean_p = acc_p_sum / num_trials as f64;
            let mean_o = acc_o_sum / num_trials as f64;
            let mean_all = (mean_s + mean_p + mean_o) / 3.0;

            eprintln!(
                "[Experiment 1] shift={shift}: mean_S={mean_s:.4}, mean_P={mean_p:.4}, \
                 mean_O={mean_o:.4}, mean_all={mean_all:.4}, min={min_acc:.4}"
            );

            // Expect ~75% accuracy for n=3 majority vote
            assert!(
                mean_all > 0.70,
                "shift={shift}: mean accuracy {mean_all:.4} too low (expect ~0.75)"
            );
            assert!(
                mean_all < 0.80,
                "shift={shift}: mean accuracy {mean_all:.4} unexpectedly high"
            );
        }
    }

    // ── EXPERIMENT 2: gcd(3130,8192)=2 vulnerability ──────────────

    #[test]
    fn experiment_2_gcd_vulnerability() {
        // Test 1: Pure alternating pattern (period 2)
        let mut alternating = [0u64; N];
        for w in alternating.iter_mut() {
            *w = 0xAAAA_AAAA_AAAA_AAAA; // 1010...
        }

        // With shift=3130 (gcd=2), a period-2 pattern may have specific
        // hamming characteristics.
        let shifted_3130 = cyclic_shift(&alternating, GOLDEN_SHIFT);
        let shifted_3131 = cyclic_shift(&alternating, GOLDEN_SHIFT_ODD);

        let ham_3130 = hamming_128(&alternating, &shifted_3130);
        let ham_3131 = hamming_128(&alternating, &shifted_3131);

        eprintln!(
            "[Experiment 2] Alternating 0xAAA...: ham(orig, shift3130)={ham_3130}, \
             ham(orig, shift3131)={ham_3131}"
        );

        // For a perfect alternating pattern with even shift, shift preserves
        // the pattern or flips all bits. With odd shift, it should flip ~half.
        // 3130 is even: shift by 3130 on alternating is shift by 3130 mod 2 = 0 mod period,
        // so hamming should be 0 or D.
        // 3131 is odd: shift by 3131 on alternating is shift by 1 mod period,
        // so it flips all bits, hamming = D.

        // Test 2: Near-periodic vectors (period 2 with 5% noise)
        let mut noisy_sum_3130 = 0u64;
        let mut noisy_sum_3131 = 0u64;
        let num_noisy = 100;
        for trial in 0..num_noisy {
            let noisy = sparse_128(trial as u64 + 7777, 0.05);
            let mut vec2 = alternating;
            for i in 0..N {
                vec2[i] ^= noisy[i]; // flip ~5% of bits
            }

            let bundle_3130 = bundle_spo(&vec2, &random_128(trial as u64 + 100), &random_128(trial as u64 + 200), GOLDEN_SHIFT);
            let bundle_3131 = bundle_spo(&vec2, &random_128(trial as u64 + 100), &random_128(trial as u64 + 200), GOLDEN_SHIFT_ODD);

            let rec_3130 = recover_s(&bundle_3130, GOLDEN_SHIFT);
            let rec_3131 = recover_s(&bundle_3131, GOLDEN_SHIFT_ODD);

            noisy_sum_3130 += hamming_128(&vec2, &rec_3130) as u64;
            noisy_sum_3131 += hamming_128(&vec2, &rec_3131) as u64;
        }

        let avg_err_3130 = noisy_sum_3130 as f64 / (num_noisy as f64 * D as f64);
        let avg_err_3131 = noisy_sum_3131 as f64 / (num_noisy as f64 * D as f64);
        let acc_3130 = 1.0 - avg_err_3130;
        let acc_3131 = 1.0 - avg_err_3131;

        eprintln!(
            "[Experiment 2] Noisy period-2: acc_3130={acc_3130:.4}, acc_3131={acc_3131:.4}"
        );

        // Test 3: Period-4 vectors
        let mut period4 = [0u64; N];
        for w in period4.iter_mut() {
            *w = 0x9999_9999_9999_9999; // 1001 repeating = period 4
        }
        let ham4_3130 = hamming_128(&period4, &cyclic_shift(&period4, GOLDEN_SHIFT));
        let ham4_3131 = hamming_128(&period4, &cyclic_shift(&period4, GOLDEN_SHIFT_ODD));
        eprintln!(
            "[Experiment 2] Period-4: ham(orig, shift3130)={ham4_3130}, \
             ham(orig, shift3131)={ham4_3131}"
        );

        // Test 4: Period-8 vectors
        let mut period8 = [0u64; N];
        for w in period8.iter_mut() {
            *w = 0x8181_8181_8181_8181; // 10000001 repeating = period 8
        }
        let ham8_3130 = hamming_128(&period8, &cyclic_shift(&period8, GOLDEN_SHIFT));
        let ham8_3131 = hamming_128(&period8, &cyclic_shift(&period8, GOLDEN_SHIFT_ODD));
        eprintln!(
            "[Experiment 2] Period-8: ham(orig, shift3130)={ham8_3130}, \
             ham(orig, shift3131)={ham8_3131}"
        );

        // Both shifts should give reasonable accuracy on noisy data
        // The key question: does 3130 fail on periodic inputs?
        // With random P and O, even periodic S should bundle reasonably.
        assert!(
            acc_3130 > 0.60,
            "shift=3130 accuracy {acc_3130:.4} on noisy period-2 is catastrophically low"
        );
        assert!(
            acc_3131 > 0.60,
            "shift=3131 accuracy {acc_3131:.4} on noisy period-2 is catastrophically low"
        );
    }

    // ── EXPERIMENT 3: Ranking preservation ─────────────────────────

    #[test]
    fn experiment_3_ranking_preservation() {
        let num_pairs = 500;
        let shift = GOLDEN_SHIFT_ODD; // use the safe shift

        let mut true_dists = Vec::with_capacity(num_pairs);
        let mut recovered_dists = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            let seed_a = (i as u64) * 10000 + 42;
            let seed_b = (i as u64) * 10000 + 99;

            let s_a = random_128(seed_a);
            let s_b = random_128(seed_b);
            let p_a = random_128(seed_a + 1);
            let o_a = random_128(seed_a + 2);
            let p_b = random_128(seed_b + 1);
            let o_b = random_128(seed_b + 2);

            let true_dist = hamming_128(&s_a, &s_b) as f64;

            let bundle_a = bundle_spo(&s_a, &p_a, &o_a, shift);
            let bundle_b = bundle_spo(&s_b, &p_b, &o_b, shift);
            let rec_s_a = recover_s(&bundle_a, shift);
            let rec_s_b = recover_s(&bundle_b, shift);
            let recovered_dist = hamming_128(&rec_s_a, &rec_s_b) as f64;

            true_dists.push(true_dist);
            recovered_dists.push(recovered_dist);
        }

        let rho = spearman_correlation(&true_dists, &recovered_dists);
        eprintln!("[Experiment 3] Spearman rho = {rho:.4}");

        // Validate affine transform: E[recovered] ≈ 0.25 * true + 0.375 * d
        // For random vectors, true_dist ≈ d/2 = 4096
        // Expected recovered ≈ 0.25 * 4096 + 0.375 * 8192 = 1024 + 3072 = 4096
        let mean_true: f64 = true_dists.iter().sum::<f64>() / num_pairs as f64;
        let mean_rec: f64 = recovered_dists.iter().sum::<f64>() / num_pairs as f64;
        let expected_rec = 0.25 * mean_true + 0.375 * D as f64;

        eprintln!(
            "[Experiment 3] mean_true={mean_true:.1}, mean_rec={mean_rec:.1}, \
             expected_rec={expected_rec:.1}"
        );

        // Rank correlation should be positive and significant.
        // With n=3 majority vote and independent P,O per node, the noise from
        // P,O reduces rank correlation. rho ~ 0.20-0.30 is expected for
        // random independent triples. The ranking is partially preserved.
        assert!(
            rho > 0.15,
            "Spearman rho={rho:.4} too low — ranking not preserved at all"
        );
    }

    // ── EXPERIMENT 4: Search quality (Recall@10) ──────────────────

    #[test]
    fn experiment_4_search_quality() {
        let num_nodes = 200;
        let num_similar = 10;
        let shift = GOLDEN_SHIFT_ODD;
        let k = 10;

        // Generate a reference S-plane
        let s_ref = random_128(12345);

        // Create 200 nodes: first 10 share ~80% of bits with s_ref
        let mut s_planes = Vec::with_capacity(num_nodes);
        let mut bundles = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let s = if i < num_similar {
                // Similar: flip ~20% of bits from reference
                let noise = sparse_128(i as u64 + 5000, 0.20);
                let mut sim = s_ref;
                for j in 0..N {
                    sim[j] ^= noise[j];
                }
                sim
            } else {
                random_128(i as u64 + 9000)
            };

            let p = random_128(i as u64 + 20000);
            let o = random_128(i as u64 + 30000);
            let bundle = bundle_spo(&s, &p, &o, shift);

            s_planes.push(s);
            bundles.push(bundle);
        }

        // Query: s_ref itself
        let p_q = random_128(77777);
        let o_q = random_128(88888);
        let bundle_q = bundle_spo(&s_ref, &p_q, &o_q, shift);

        // Top-k by original S hamming
        let mut true_dists: Vec<(usize, u32)> = s_planes
            .iter()
            .enumerate()
            .map(|(i, s)| (i, hamming_128(&s_ref, s)))
            .collect();
        true_dists.sort_by_key(|&(_, d)| d);
        let true_topk: Vec<usize> = true_dists.iter().take(k).map(|&(i, _)| i).collect();

        // Top-k by bundle hamming
        let mut bundle_dists: Vec<(usize, u32)> = bundles
            .iter()
            .enumerate()
            .map(|(i, b)| (i, hamming_128(&bundle_q, b)))
            .collect();
        bundle_dists.sort_by_key(|&(_, d)| d);
        let bundle_topk: Vec<usize> = bundle_dists.iter().take(k).map(|&(i, _)| i).collect();

        // Recall@10
        let recall: f64 = true_topk
            .iter()
            .filter(|idx| bundle_topk.contains(idx))
            .count() as f64
            / k as f64;

        eprintln!("[Experiment 4] Recall@{k} = {recall:.2}");
        eprintln!("  True top-{k}: {true_topk:?}");
        eprintln!("  Bundle top-{k}: {bundle_topk:?}");

        // We expect decent recall since similar S-planes should produce
        // closer bundles
        assert!(
            recall >= 0.3,
            "Recall@{k}={recall:.2} is too low for practical use"
        );
    }

    // ── EXPERIMENT 5: CLAM clustering on cyclic bundles ────────────

    #[test]
    fn experiment_5_clam_clustering() {
        let num_clusters = 5;
        let nodes_per_cluster = 40;
        let total_nodes = num_clusters * nodes_per_cluster;
        let shift = GOLDEN_SHIFT_ODD;

        // Generate cluster centers
        let centers: Vec<[u64; N]> = (0..num_clusters)
            .map(|c| random_128(c as u64 * 1_000_000 + 42))
            .collect();

        // Generate nodes: each node shares ~80% bits with its cluster center
        let mut s_planes = Vec::with_capacity(total_nodes);
        let mut bundles = Vec::with_capacity(total_nodes);
        let mut labels = Vec::with_capacity(total_nodes);

        for c in 0..num_clusters {
            for j in 0..nodes_per_cluster {
                let noise = sparse_128((c * 1000 + j) as u64 + 55555, 0.20);
                let mut s = centers[c];
                for k in 0..N {
                    s[k] ^= noise[k];
                }

                let p = random_128((c * 1000 + j) as u64 + 66666);
                let o = random_128((c * 1000 + j) as u64 + 77777);
                let bundle = bundle_spo(&s, &p, &o, shift);

                s_planes.push(s);
                bundles.push(bundle);
                labels.push(c);
            }
        }

        // k-NN purity test: for each node, find k=5 nearest neighbors
        // using bundles vs using original S-planes.
        let kk = 5;
        let mut bundle_purity_sum = 0.0;
        let mut original_purity_sum = 0.0;
        let mut knn_recall_sum = 0.0;

        for i in 0..total_nodes {
            // k-NN by bundle distance
            let mut bundle_dists: Vec<(usize, u32)> = (0..total_nodes)
                .filter(|&j| j != i)
                .map(|j| (j, hamming_128(&bundles[i], &bundles[j])))
                .collect();
            bundle_dists.sort_by_key(|&(_, d)| d);
            let bundle_knn: Vec<usize> =
                bundle_dists.iter().take(kk).map(|&(j, _)| j).collect();

            // k-NN by original S distance
            let mut orig_dists: Vec<(usize, u32)> = (0..total_nodes)
                .filter(|&j| j != i)
                .map(|j| (j, hamming_128(&s_planes[i], &s_planes[j])))
                .collect();
            orig_dists.sort_by_key(|&(_, d)| d);
            let orig_knn: Vec<usize> =
                orig_dists.iter().take(kk).map(|&(j, _)| j).collect();

            // Purity: fraction of k-NN sharing same label
            let bundle_purity = bundle_knn
                .iter()
                .filter(|&&j| labels[j] == labels[i])
                .count() as f64
                / kk as f64;
            let orig_purity = orig_knn
                .iter()
                .filter(|&&j| labels[j] == labels[i])
                .count() as f64
                / kk as f64;

            // k-NN recall: how many of the true k-NN are in bundle k-NN
            let knn_recall = orig_knn
                .iter()
                .filter(|idx| bundle_knn.contains(idx))
                .count() as f64
                / kk as f64;

            bundle_purity_sum += bundle_purity;
            original_purity_sum += orig_purity;
            knn_recall_sum += knn_recall;
        }

        let mean_bundle_purity = bundle_purity_sum / total_nodes as f64;
        let mean_orig_purity = original_purity_sum / total_nodes as f64;
        let mean_knn_recall = knn_recall_sum / total_nodes as f64;

        eprintln!(
            "[Experiment 5] Cluster purity: bundle={mean_bundle_purity:.3}, \
             original={mean_orig_purity:.3}"
        );
        eprintln!("[Experiment 5] k-NN recall = {mean_knn_recall:.3}");

        // Original should have high purity (well-separated clusters)
        assert!(
            mean_orig_purity > 0.80,
            "Original purity {mean_orig_purity:.3} unexpectedly low — clusters not well-separated"
        );

        // Bundle purity should be meaningfully above random (1/5 = 0.2)
        assert!(
            mean_bundle_purity > 0.30,
            "Bundle purity {mean_bundle_purity:.3} near random — bundling destroys cluster structure"
        );
    }

    // ── EXPERIMENT 6: Bias sensitivity ─────────────────────────────

    #[test]
    fn experiment_6_bias_sensitivity() {
        let num_trials = 100;
        let shift = GOLDEN_SHIFT_ODD;

        eprintln!("[Experiment 6] Bias sensitivity table:");
        eprintln!("  p     | predicted | actual_S  | actual_P  | actual_O");
        eprintln!("  ------+-----------+-----------+-----------+---------");

        // Vary bias from 0.30 to 0.70 in steps of 0.05
        let biases: Vec<f64> = (0..9).map(|i| 0.30 + i as f64 * 0.05).collect();

        for &p in &biases {
            let predicted = 1.0 - p * (1.0 - p);

            let mut acc_s_sum = 0.0;
            let mut acc_p_sum = 0.0;
            let mut acc_o_sum = 0.0;

            for trial in 0..num_trials {
                let seed_base = (trial as u64) * 3000 + (p * 1000.0) as u64;
                let s = biased_128(seed_base + 1, p);
                let pv = biased_128(seed_base + 2, p);
                let o = biased_128(seed_base + 3, p);

                let bundle = bundle_spo(&s, &pv, &o, shift);

                let rec_s = recover_s(&bundle, shift);
                let rec_p = recover_p(&bundle, shift);
                let rec_o = recover_o(&bundle, shift);

                acc_s_sum += bit_accuracy(&s, &rec_s);
                acc_p_sum += bit_accuracy(&pv, &rec_p);
                acc_o_sum += bit_accuracy(&o, &rec_o);
            }

            let actual_s = acc_s_sum / num_trials as f64;
            let actual_p = acc_p_sum / num_trials as f64;
            let actual_o = acc_o_sum / num_trials as f64;

            eprintln!(
                "  {p:.2}  | {predicted:.4}    | {actual_s:.4}    | {actual_p:.4}    | {actual_o:.4}"
            );

            // Verify predicted vs actual are reasonably close
            // The formula P(correct) = 1 - p(1-p) is for two independent random
            // components interfering with the target. Allow some tolerance.
            let actual_avg = (actual_s + actual_p + actual_o) / 3.0;
            let error = (actual_avg - predicted).abs();
            assert!(
                error < 0.10,
                "p={p:.2}: predicted={predicted:.4}, actual={actual_avg:.4}, error={error:.4} > 0.10"
            );
        }
    }
}
