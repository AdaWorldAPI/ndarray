#![allow(missing_docs)]
//! Pillar-13 — HHTL cascade contraction certification.
//!
//! Substrate-tier pillar: certifies that the weighted Hamming-bundle
//! operator consumed by the dn-tree cascade (`crate::hpc::dn_tree::DNTree::update`)
//! and the cognitive HHTL spread-activation cascade is a *contraction*
//! in normalized Hamming distance, with a measurable per-level ratio
//! that matches the Bernoulli-mixture prediction.
//!
//! # The contraction claim
//!
//! Let `T(x, y) = bundle(x, y, lr)` be the per-step bundle operator where
//! `lr ∈ (0, 1]` and each bit position is independently replaced with
//! probability `lr`. For a fixed target `y*` and any initial state `x₀`,
//! the iterated cascade `x_{t+1} = T(x_t, y*)` satisfies in expectation:
//!
//! ```text
//!   E[ d_H(x_t, y*) / N ]  =  (1 − lr)^t · d_H(x_0, y*) / N
//! ```
//!
//! where `d_H` is unnormalized Hamming distance and `N` is the bit width.
//! This is the Bernoulli-mixture geometry: each bit that differs flips
//! with probability `lr` per step, independently. The expected
//! contraction ratio per step is therefore exactly `(1 − lr)`.
//!
//! # Why the substrate needs this
//!
//! The cognitive shader's (4×4)^4 cascade and dn-tree's quaternary
//! traversal both assume bounded depth via contraction: spread
//! activation must *terminate* rather than blow up, and any change
//! that introduces a non-contracting bundle path (e.g., a BTSP boost
//! factor `lr × boost ≥ 1` driving `(1 − effective_lr) ≤ 0`) will fail
//! this pillar visibly. The substrate currently asserts contraction
//! by construction; Pillar-13 makes that assertion *testable*.
//!
//! # Probe design (synthetic only)
//!
//! 1. Fix bit width `BIT_WIDTH = 1024` and depth `MAX_DEPTH = 4`
//!    (the HHTL cascade depth from the cognitive shader spec).
//! 2. Generate a single target vector `y*` via `SplitMix64`.
//! 3. For each of `N_CASCADES = 256` cascades, generate a random initial
//!    state `x_0` with expected Hamming distance to `y*` ≈ `BIT_WIDTH/2`.
//! 4. Run `MAX_DEPTH` cascade steps of `bundle(x, y*, lr)` with the
//!    canonical mid-range learning rate `LR = 0.5`.
//! 5. At every (cascade, level) pair, record the empirical Hamming
//!    distance `d_t / N` and compare to the predicted `(1 − lr)^t · d_0 / N`.
//! 6. Mark the pair as *contracting* iff
//!    `d_t / d_(t−1) ≤ (1 − lr) + RATIO_TOLERANCE` AND `d_t ≤ d_(t−1)`.
//!
//! # Cross-verification
//!
//! The bundle operator is implemented independently inside this probe — it
//! is *not* a re-export of `crate::hpc::dn_tree::bundle_into`. The intent
//! is to test the *math* against any future stack implementation: if
//! `dn_tree::bundle_into` drifts from the Bernoulli-mixture formula, an
//! integration test elsewhere will detect the drift; this pillar pins the
//! mathematics independently.
//!
//! # Pass criteria
//!
//! The probe tests two complementary properties:
//!
//! 1. *Almost-sure contraction.* For every `(cascade, level)` pair the
//!    Hamming distance to the target must satisfy `d_t ≤ d_{t-1}`. This
//!    holds with probability 1 under `Binomial(d_{t-1}, 1 − lr)`
//!    dynamics; any violation flags a bundle-operator bug.
//!    Reported as `psd_rate`; pass threshold `PASS_FRACTION = 1.0`.
//! 2. *Predicted per-level mean ratio.* For each level `t` the mean
//!    `d_t / d_{t-1}` over cascades must equal `(1 − lr)` within
//!    `MEAN_RATIO_TOLERANCE`. The mean is taken per-level — not
//!    per-cascade — so the test concentrates by CLT at
//!    `σ / √N_CASCADES ≈ 0.004` rather than by per-realisation
//!    Binomial variance at `σ / √d_{t-1}`. Reported as
//!    `lognorm_concentration = log(1 + max_t |mean_ratio_t − (1 − lr)|)`;
//!    pass threshold `LOGNORM_BUDGET = 0.05`.
//! 3. The final-level mean Hamming distance must be strictly less than
//!    the initial mean (contraction confirmed over the full depth).
//!
//! # References
//!
//! * Banach, S. (1922). *Sur les opérations dans les ensembles abstraits
//!   et leur application aux équations intégrales.* Fundam. Math. 3.
//! * Kanerva, P. (2009). *Hyperdimensional computing: An introduction to
//!   computing in distributed representation with high-dimensional
//!   random vectors.* Cognitive Computation 1(2). (HDC bundling primer.)
//!
//! # SEED
//!
//! `PILLAR_13_SEED = 0x_C13C_A5CADE_C0DE`

use crate::hpc::pillar::prove_runner::{PillarReport, SplitMix64};

// ── Constants ────────────────────────────────────────────────────────────────

/// Deterministic seed for all Pillar-13 RNG streams.
pub const PILLAR_13_SEED: u64 = 0x_C13C_A5CA_DEC0;

/// Bit width of the synthetic state vectors.
const BIT_WIDTH: usize = 1024;

/// Number of u64 words to represent `BIT_WIDTH` bits.
const WORDS: usize = BIT_WIDTH / 64;

/// Number of independent cascades.
const N_CASCADES: usize = 256;

/// HHTL cascade depth.
const MAX_DEPTH: usize = 4;

/// Canonical bundle learning rate. lr = 0.5 is the mid-range value where
/// the predicted contraction ratio (1 − lr) = 0.5 per step is far from
/// both lr → 0 (no contraction) and lr → 1 (instant collapse).
const LR: f32 = 0.5;

/// Per-level mean-ratio tolerance: the empirical mean of `d_t / d_{t-1}`
/// over cascades must lie within `|mean − (1 − LR)| ≤ MEAN_RATIO_TOLERANCE`
/// at every level. With `N_CASCADES = 256` and `σ(d_t/d_{t-1}) ≤ 0.07`
/// (worst case at the deepest level), CLT gives the mean a standard
/// error of roughly `0.07 / sqrt(256) ≈ 0.004`. The tolerance is set
/// to `0.02` — five standard errors out, generous enough that a
/// passing implementation has overwhelming probability of staying
/// inside, tight enough that a doubled or halved effective LR fails
/// visibly.
const MEAN_RATIO_TOLERANCE: f64 = 0.02;

/// Minimum fraction of `(cascade, level)` pairs that must contract
/// almost-surely (`d_t ≤ d_{t-1}`). Mathematically this is `1.0` because
/// `d_t | d_{t-1} = Binomial(d_{t-1}, 1 - LR) ≤ d_{t-1}`; the threshold
/// is set to `1.0` so any violation indicates a bundle-operator bug.
const PASS_FRACTION: f64 = 1.0;

/// Budget on `log(1 + max per-level |mean_ratio − (1 − LR)|)`. Picked so
/// `MEAN_RATIO_TOLERANCE` translates to `ln(1.02) ≈ 0.0198 < 0.05`.
const LOGNORM_BUDGET: f64 = 0.05;

// ── Bit-vector primitives ────────────────────────────────────────────────────

/// Generate a random bit vector of `WORDS` u64 words via `SplitMix64`.
fn random_bits(rng: &mut SplitMix64) -> [u64; WORDS] {
    let mut bits = [0u64; WORDS];
    for w in bits.iter_mut() {
        *w = rng.next_u64();
    }
    bits
}

/// Unnormalized Hamming distance via `count_ones` on the XOR.
fn hamming(a: &[u64; WORDS], b: &[u64; WORDS]) -> u32 {
    let mut d = 0u32;
    for i in 0..WORDS {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

/// Build a u64 mask where each bit is independently 1 with probability
/// approximately `p ∈ [0, 1]`. Uses the AND-cascade construction from
/// `dn_tree::make_probability_mask` style — independent re-derivation here.
///
/// For `p = 0.5`: a single random word (each bit IID Bernoulli(0.5)).
/// For `p < 0.5`: AND of `ceil(-log2(p))` random words.
/// For `p > 0.5`: complement of a `(1-p)` mask.
fn probability_mask(p: f32, rng: &mut SplitMix64) -> u64 {
    if p >= 1.0 {
        return u64::MAX;
    }
    if p <= 0.0 {
        return 0;
    }
    if p > 0.5 {
        return !probability_mask(1.0 - p, rng);
    }
    // p ≤ 0.5: AND cascade. n = ceil(-log2(p)) random words ANDed.
    let n = (-(p.ln() / 2.0_f32.ln())).ceil() as u32;
    let mut mask = rng.next_u64();
    for _ in 1..n {
        mask &= rng.next_u64();
    }
    mask
}

/// One cascade step: `bundle(x, y, lr)` returns a new vector where each
/// bit position independently takes y's value with probability `lr` and
/// keeps x's value with probability `(1 − lr)`.
fn bundle_step(x: &[u64; WORDS], y: &[u64; WORDS], lr: f32, rng: &mut SplitMix64) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    for i in 0..WORDS {
        let mask = probability_mask(lr, rng);
        // Where mask is 1: take y[i]; where 0: keep x[i].
        out[i] = (x[i] & !mask) | (y[i] & mask);
    }
    out
}

// ── Probe pipeline ──────────────────────────────────────────────────────────

/// Run all `N_CASCADES` cascades to depth `MAX_DEPTH` and collect the
/// per-level normalized Hamming distance to the target. Returns a
/// `[MAX_DEPTH + 1] × N_CASCADES` matrix flattened row-major (level-major).
///
/// `distances[level * N_CASCADES + cascade]` is `d_H(x_level, y*) / BIT_WIDTH`
/// at that level. Level 0 is the initial state.
fn run_cascades(rng: &mut SplitMix64) -> Vec<f64> {
    let y_star = random_bits(rng);

    // Cap of (MAX_DEPTH + 1) * N_CASCADES = 5 · 256 = 1280 entries.
    let mut distances = vec![0.0_f64; (MAX_DEPTH + 1) * N_CASCADES];

    for cascade in 0..N_CASCADES {
        let mut x = random_bits(rng);

        // Level 0.
        distances[cascade] = hamming(&x, &y_star) as f64 / BIT_WIDTH as f64;

        // Levels 1..=MAX_DEPTH.
        for level in 1..=MAX_DEPTH {
            x = bundle_step(&x, &y_star, LR, rng);
            distances[level * N_CASCADES + cascade] = hamming(&x, &y_star) as f64 / BIT_WIDTH as f64;
        }
    }

    distances
}

/// Predicted per-level normalized Hamming distance under the Bernoulli
/// mixture: `(1 − lr)^level · d_0`. Used only for cross-verification of
/// the empirical means.
fn predicted_distance(initial: f64, level: usize) -> f64 {
    initial * (1.0 - LR as f64).powi(level as i32)
}

// ── Public probe entry point ────────────────────────────────────────────────

/// Run the Pillar-13 HHTL cascade-contraction certification probe.
///
/// Generates `N_CASCADES` synthetic cascades against a fixed target,
/// runs `MAX_DEPTH = 4` bundle steps per cascade at learning rate
/// `LR = 0.5`, and tests two properties:
///
/// 1. *Almost-sure contraction.* For every `(cascade, level)` pair,
///    `d_t ≤ d_{t-1}`. This holds with probability 1 under
///    `Binomial(d_{t-1}, 1 - LR)` dynamics; any violation flags a
///    bundle-operator bug. Reported as `psd_rate`.
/// 2. *Predicted per-level mean ratio.* For each level `t`, the mean
///    `d_t / d_{t-1}` over cascades equals `(1 − LR)` within
///    `MEAN_RATIO_TOLERANCE`. Tested per-level (CLT-tight at
///    `σ/√N_CASCADES`) rather than per-cascade (Binomial-loose at
///    `σ/√d_{t-1}`). Reported as `lognorm_concentration`.
///
/// # Returns
///
/// A [`PillarReport`] with semantics:
/// - `n_paths` = `N_CASCADES`
/// - `n_hops` = `MAX_DEPTH`
/// - `psd_rate` = fraction of `(cascade, level≥1)` pairs satisfying
///   `d_t ≤ d_{t-1}` (almost-sure contraction).
/// - `lognorm_concentration` = `log(1 + max_t |mean_ratio_t − (1 − LR)|)`
///   over `t ∈ {1, …, MAX_DEPTH}`.
/// - `passed` = `psd_rate ≥ PASS_FRACTION ∧ lognorm_concentration ≤ LOGNORM_BUDGET`
///   AND final-level mean strictly less than initial mean.
///
/// # Example
///
/// ```ignore
/// use ndarray::hpc::pillar::hhtl_contraction::prove_pillar_13;
/// let report = prove_pillar_13();
/// report.print();
/// assert!(report.passed);
/// ```
pub fn prove_pillar_13() -> PillarReport {
    let mut rng = SplitMix64::new(PILLAR_13_SEED);
    let distances = run_cascades(&mut rng);

    let predicted_ratio = 1.0 - LR as f64;

    // Almost-sure contraction: count (cascade, level) pairs with d_t ≤ d_{t-1}.
    let mut contracting_pairs = 0u32;
    let mut total_pairs = 0u32;
    for cascade in 0..N_CASCADES {
        for level in 1..=MAX_DEPTH {
            let prev = distances[(level - 1) * N_CASCADES + cascade];
            let curr = distances[level * N_CASCADES + cascade];
            total_pairs += 1;
            if curr <= prev {
                contracting_pairs += 1;
            }
        }
    }
    let psd_rate = (contracting_pairs as f64) / (total_pairs.max(1) as f64);

    // Per-level mean ratio: for each level, average d_t / d_{t-1} over
    // cascades and compare to the predicted (1 − LR). Skip cascades
    // whose previous-level distance is zero (would yield a 0/0 ratio);
    // at lr = 0.5 with N = 1024 bits this never triggers in practice.
    let mut max_level_deviation = 0.0_f64;
    for level in 1..=MAX_DEPTH {
        let mut ratio_sum = 0.0_f64;
        let mut ratio_count = 0u32;
        for cascade in 0..N_CASCADES {
            let prev = distances[(level - 1) * N_CASCADES + cascade];
            let curr = distances[level * N_CASCADES + cascade];
            if prev > 0.0 {
                ratio_sum += curr / prev;
                ratio_count += 1;
            }
        }
        if ratio_count > 0 {
            let mean_ratio = ratio_sum / ratio_count as f64;
            let deviation = (mean_ratio - predicted_ratio).abs();
            if deviation > max_level_deviation {
                max_level_deviation = deviation;
            }
        }
    }
    let lognorm_concentration = (1.0 + max_level_deviation).ln();

    // Sanity: mean final distance < mean initial distance.
    let mut mean_initial = 0.0_f64;
    let mut mean_final = 0.0_f64;
    for cascade in 0..N_CASCADES {
        mean_initial += distances[cascade];
        mean_final += distances[MAX_DEPTH * N_CASCADES + cascade];
    }
    mean_initial /= N_CASCADES as f64;
    mean_final /= N_CASCADES as f64;
    let depth_contracts = mean_final < mean_initial;

    // The per-level deviation must also lie within MEAN_RATIO_TOLERANCE
    // — a strict-form test on the predicted ratio, independent of the
    // lognorm budget (which is its log-1-plus relaxation).
    let mean_within_tolerance = max_level_deviation <= MEAN_RATIO_TOLERANCE;

    let passed = psd_rate >= PASS_FRACTION
        && lognorm_concentration <= LOGNORM_BUDGET
        && mean_within_tolerance
        && depth_contracts;

    PillarReport {
        pillar_id: 13,
        seed: PILLAR_13_SEED,
        n_paths: N_CASCADES as u32,
        n_hops: MAX_DEPTH as u32,
        psd_rate,
        lognorm_concentration,
        passed,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_13_seed_matches_spec() {
        assert_eq!(PILLAR_13_SEED, 0x_C13C_A5CA_DEC0);
    }

    #[test]
    fn hamming_self_is_zero() {
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        let b = random_bits(&mut rng);
        assert_eq!(hamming(&b, &b), 0);
    }

    #[test]
    fn hamming_complementary_is_full() {
        let a = [0u64; WORDS];
        let mut b = [0u64; WORDS];
        for w in b.iter_mut() {
            *w = u64::MAX;
        }
        assert_eq!(hamming(&a, &b), BIT_WIDTH as u32);
    }

    #[test]
    fn probability_mask_zero_is_all_zero() {
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        assert_eq!(probability_mask(0.0, &mut rng), 0);
    }

    #[test]
    fn probability_mask_one_is_all_one() {
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        assert_eq!(probability_mask(1.0, &mut rng), u64::MAX);
    }

    #[test]
    fn probability_mask_half_is_balanced() {
        // For p = 0.5, mean popcount over 64 bits across 1000 trials should
        // be near 32 with stderr ≈ 4 / sqrt(1000) ≈ 0.13.
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        let n = 1000u32;
        let mut total = 0u32;
        for _ in 0..n {
            total += probability_mask(0.5, &mut rng).count_ones();
        }
        let mean = total as f64 / n as f64;
        assert!((mean - 32.0).abs() < 0.8, "p=0.5 mask: mean popcount {mean:.4} not near 32");
    }

    #[test]
    fn bundle_lr_zero_keeps_x() {
        // lr = 0 → mask is all zero → result = x.
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        let x = random_bits(&mut rng);
        let y = random_bits(&mut rng);
        let out = bundle_step(&x, &y, 0.0, &mut rng);
        assert_eq!(out, x);
    }

    #[test]
    fn bundle_lr_one_yields_y() {
        // lr = 1 → mask is all one → result = y.
        let mut rng = SplitMix64::new(PILLAR_13_SEED);
        let x = random_bits(&mut rng);
        let y = random_bits(&mut rng);
        let out = bundle_step(&x, &y, 1.0, &mut rng);
        assert_eq!(out, y);
    }

    #[test]
    fn predicted_distance_decays_geometrically() {
        // (1 − 0.5)^1 = 0.5, ^2 = 0.25, ^3 = 0.125, ^4 = 0.0625.
        let d0 = 0.5_f64;
        assert!((predicted_distance(d0, 1) - 0.25).abs() < 1e-12);
        assert!((predicted_distance(d0, 2) - 0.125).abs() < 1e-12);
        assert!((predicted_distance(d0, 4) - 0.03125).abs() < 1e-12);
    }

    #[test]
    fn prove_pillar_13_passes() {
        let report = prove_pillar_13();
        report.print();
        assert!(
            report.passed,
            "Pillar-13 FAILED: psd_rate={:.6} lognorm={:.6}",
            report.psd_rate, report.lognorm_concentration
        );
    }

    #[test]
    fn prove_pillar_13_seed_anchored() {
        let r1 = prove_pillar_13();
        let r2 = prove_pillar_13();
        assert_eq!(r1.passed, r2.passed);
        assert!((r1.psd_rate - r2.psd_rate).abs() < 1e-12);
        assert!((r1.lognorm_concentration - r2.lognorm_concentration).abs() < 1e-12);
    }

    /// Drift-detection: the pillar's `bundle_step` independently re-derives
    /// the bit-mixing bundle operator. The production code path at
    /// `crate::hpc::dn_tree::bundle_into` (PR #189, exposed `pub(crate)`)
    /// is the substrate the pillar is defending. This test runs both on
    /// seed-aligned SplitMix64 RNGs and asserts the first 16 u64 words of
    /// production's `GraphHV.channels[0]` agree bit-exactly with the
    /// pillar's `[u64; WORDS]` output.
    ///
    /// # Why this is a bit-exact (not ε-tolerant) check
    ///
    /// Per the substrate's bit-exactness contract (W1a + the data-flow
    /// rules), bundling is a *gated XOR* (Bernoulli-mixture per bit) —
    /// the mask draws come from `SplitMix64` which is bit-deterministic.
    /// Both pillar's `probability_mask` and production's
    /// `make_probability_mask` consume the same number of `next_u64()`
    /// draws at lr=0.25 (n=ceil(-log2(0.25))=2 per word), so the masks
    /// for the first `WORDS` words align exactly across the two
    /// functions. The remaining 240 words of channel 0 (and channels 1/2)
    /// consume extra RNG draws on the production side; those don't affect
    /// the first WORDS=16 words because each word is independent.
    ///
    /// # Why not lr=0.5
    ///
    /// Production's `make_probability_mask(0.5)` has a latent
    /// infinite-recursion bug: `p >= 0.5` recurses with `1.0 - 0.5 = 0.5`
    /// forever. Pillar's `probability_mask` uses `p > 0.5` (strict) and
    /// falls through to the AND-cascade at p=0.5. Real production usage
    /// (DNConfig default lr=0.03, boost up to ~30 → effective_lr~0.9)
    /// never hits 0.5 exactly, so the bug is dormant. This drift-check
    /// uses lr=0.25 where both implementations agree; the lr=0.5 case
    /// is recorded as a follow-up.
    #[test]
    fn pillar_13_matches_production_bundle_into() {
        use crate::hpc::cam_index::GraphHV;
        use crate::hpc::dn_tree::{bundle_into, SplitMix64 as DnSplitMix64};

        const N_TRIALS: u32 = 16;
        // Was 0.25 to avoid the latent p=0.5 infinite-recursion bug in
        // production's make_probability_mask; that bug is fixed in the
        // same commit/PR that updates this constant. lr=0.5 now matches
        // Pillar 13's canonical mid-range learning rate and exercises
        // the previously-broken branch.
        const TEST_LR: f64 = 0.5;

        // Both SplitMix64 implementations use identical algorithm (same
        // multiplier constants 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
        // 0x94D049BB133111EB and same shift sequence), so identical seeds
        // → identical sequences. Both functions consume the same number
        // of next_u64() draws per word at p=0.25 (n=ceil(-log2(0.25))=2),
        // so the mask sequences align bit-exactly across the first WORDS
        // positions of each call.
        //
        // The RNGs MUST be re-seeded per trial because production's
        // bundle_into consumes 48× more RNG draws per call (3 channels ×
        // 256 words × 2 draws = 1536) than pillar's bundle_step (16 words
        // × 2 draws = 32). Without re-seeding, post-trial-0 RNG states
        // diverge.

        for trial in 0..N_TRIALS {
            // Per-trial seed for both bundling RNGs (must be the same so
            // masks align). Inputs come from a separate stream so the
            // bundling RNG state isn't disturbed by input generation.
            let trial_seed = PILLAR_13_SEED.wrapping_add(trial as u64);
            let mut rng_pillar = SplitMix64::new(trial_seed);
            let mut rng_prod = DnSplitMix64::new(trial_seed);

            let mut rng_inputs = SplitMix64::new(trial_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let x = random_bits(&mut rng_inputs);
            let y = random_bits(&mut rng_inputs);

            // Pillar side: WORDS=16 u64 mixing
            let out_pillar = bundle_step(&x, &y, TEST_LR as f32, &mut rng_pillar);

            // Production side: pack x/y into channel 0 of a GraphHV,
            // zero the rest. Pillar's `bundle(x, y, lr)` is "keep x where
            // mask=0, take y where mask=1"; production's `bundle_into`
            // contract is the same with `current` ↔ x and `hv` ↔ y
            // (per src/hpc/dn_tree.rs line 125). boost=1.0 means
            // effective_lr = lr * 1.0 = TEST_LR (matching pillar).
            let mut hv_x = GraphHV::zero();
            let mut hv_y = GraphHV::zero();
            hv_x.channels[0].words[..WORDS].copy_from_slice(&x);
            hv_y.channels[0].words[..WORDS].copy_from_slice(&y);
            let hv_out = bundle_into(&hv_x, &hv_y, TEST_LR, 1.0, &mut rng_prod);

            // Compare first WORDS=16 u64 words bit-exactly
            for w in 0..WORDS {
                assert_eq!(
                    out_pillar[w], hv_out.channels[0].words[w],
                    "Pillar/bundle_into drift at trial {trial} word {w}: \
                     pillar=0x{:016x} prod=0x{:016x}",
                    out_pillar[w], hv_out.channels[0].words[w]
                );
            }
        }
    }
}
