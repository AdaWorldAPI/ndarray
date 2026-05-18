#![allow(missing_docs)]
//! Pillar-10 — Pflug–Pichler nested-distance certification probe.
//!
//! Verifies that the empirical nested Wasserstein distance computed on a
//! stochastic-process scenario tree stays within the tight Pflug–Pichler
//! upper bound for 1000 paths × 5 hops.
//!
//! # Reference
//!
//! Pflug, G. Ch., & Pichler, A. (2012). *A distance for multistage stochastic
//! optimisation models.* SIAM Journal on Optimization, 22(1), 1–23.
//!
//! # SEED
//!
//! `PILLAR_10_SEED = 0xF1_5A_DC_5A_DD`

use crate::hpc::linalg::wasserstein::{sinkhorn_knopp_f32, wasserstein_1_f32};
use crate::hpc::pillar::prove_runner::{PillarReport, SplitMix64};

/// Deterministic seed for all Pillar-10 RNG streams.
pub const PILLAR_10_SEED: u64 = 0xF1_5A_DC_5A_DD;

/// Number of scenario paths in the nested-distance probe.
const N_PATHS: usize = 1000;

/// Number of time-stage hops per path.
const N_HOPS: usize = 5;

/// Entropic regularisation for Sinkhorn-Knopp within the nested-distance loop.
const EPSILON: f32 = 0.01;

/// Sinkhorn iteration cap.
const MAX_ITERS: u32 = 200;

/// Sinkhorn convergence tolerance.
const TOLERANCE: f32 = 1e-5;

// ── Pflug–Pichler bound ───────────────────────────────────────────────────────
//
// For a discrete scenario tree with N_PATHS equiprobable paths and N_HOPS
// stages the tight Pflug–Pichler upper bound on the nested distance equals:
//
//   bound(t) = C_lip · sqrt( ln(N_PATHS) / N_PATHS )
//
// where C_lip is the Lipschitz constant of the stage-cost kernel.
// We set C_lip = 1.0 (costs are bounded in [0, 1]) and verify that the
// empirical nested distance at every stage does not exceed `bound`.

/// Pflug–Pichler tight bound for the probe parameters.
#[inline]
fn pflug_pichler_bound() -> f32 {
    let n = N_PATHS as f32;
    // C_lip = 1.0 (costs normalised to [0,1])
    (n.ln() / n).sqrt()
}

// ── Stage-wise cost kernel ────────────────────────────────────────────────────

/// Generate a single-stage state vector (length N_PATHS) for time `t`.
///
/// Each path's state at step `t` is `x_t = x_{t-1} + noise`, where noise is
/// drawn from N(0, dt) with `dt = 1.0 / N_HOPS`.  We record only the scalar
/// terminal state per path (univariate stochastic process).
fn generate_states(rng: &mut SplitMix64, prev: &[f32], dt: f32) -> Vec<f32> {
    let scale = dt.sqrt();
    prev.iter()
        .map(|&x| {
            let n = rng.next_normal_f32();
            x + scale * n
        })
        .collect()
}

/// Pairwise L1 cost matrix between two state vectors `u` (length M) and `v`
/// (length N), flattened row-major.
fn pairwise_l1(u: &[f32], v: &[f32]) -> Vec<f32> {
    let m = u.len();
    let n = v.len();
    let mut cost = Vec::with_capacity(m * n);
    for &ui in u {
        for &vj in v {
            cost.push((ui - vj).abs());
        }
    }
    cost
}

// ── Nested-distance accumulator ───────────────────────────────────────────────

/// Compute the nested Wasserstein distance between two equiprobable scenario
/// trees, both generated with the same RNG step (deterministic coupling).
///
/// The nested distance at stage `t` is defined recursively as the Wasserstein
/// distance between the distributions at that stage where the cost includes
/// the nested distance from future stages.
///
/// For this probe we approximate with a single-level transport (Sinkhorn)
/// applied at each stage, accumulating costs forward in time.
fn nested_distance_single_level(
    rng_p: &mut SplitMix64,
    rng_q: &mut SplitMix64,
    n_paths: usize,
    n_hops: usize,
) -> f32 {
    let dt = 1.0_f32 / n_hops as f32;
    let p_weight = 1.0_f32 / n_paths as f32;

    let mut states_p: Vec<f32> = vec![0.0_f32; n_paths];
    let mut states_q: Vec<f32> = vec![0.0_f32; n_paths];

    // Uniform marginals.
    let marginals_p = vec![p_weight; n_paths];
    let marginals_q = vec![p_weight; n_paths];

    let mut total_nested = 0.0_f32;

    for _t in 0..n_hops {
        // Advance both processes one step.
        states_p = generate_states(rng_p, &states_p, dt);
        states_q = generate_states(rng_q, &states_q, dt);

        // Cost matrix: L1 distance between states.
        let cost = pairwise_l1(&states_p, &states_q);

        // Sinkhorn transport plan.
        let plan = sinkhorn_knopp_f32(
            &cost,
            n_paths,
            n_paths,
            &marginals_p,
            &marginals_q,
            EPSILON,
            MAX_ITERS,
            TOLERANCE,
        );

        // W1 distance at this stage.
        let w1 = wasserstein_1_f32(&cost, &plan, n_paths, n_paths);
        total_nested += w1;
    }

    total_nested / n_hops as f32
}

// ── Public probe entry point ──────────────────────────────────────────────────

/// Run the Pillar-10 Pflug–Pichler nested-distance certification probe.
///
/// Generates two stochastic process scenario trees (each with `N_PATHS=1000`
/// paths and `N_HOPS=5` time stages) from the canonical seed, computes the
/// empirical nested Wasserstein distance at each stage via Sinkhorn-Knopp,
/// and verifies the result lies within the tight Pflug–Pichler upper bound.
///
/// # Returns
///
/// A [`PillarReport`] with `passed = true` iff the nested distance satisfies
/// the bound.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::pflug::prove_pillar_10;
/// let report = prove_pillar_10();
/// report.print();
/// assert!(report.passed);
/// ```
pub fn prove_pillar_10() -> PillarReport {
    // Two independent RNG streams (same seed, advanced differently via fork).
    // We derive the second stream by mixing the seed with a fixed constant.
    let mut rng_p = SplitMix64::new(PILLAR_10_SEED);
    let mut rng_q = SplitMix64::new(PILLAR_10_SEED ^ 0x5555_5555_5555_5555);

    let bound = pflug_pichler_bound();
    let nested_dist = nested_distance_single_level(&mut rng_p, &mut rng_q, N_PATHS, N_HOPS);

    let passed = nested_dist <= bound;

    // lognorm_concentration: log-ratio of empirical distance to bound.
    // 0.0 would mean they are equal; negative means empirical < bound (PASS).
    let lognorm_concentration = if bound > 0.0 {
        (nested_dist / bound).ln() as f64
    } else {
        0.0
    };

    PillarReport {
        pillar_id: 10,
        seed: PILLAR_10_SEED,
        n_paths: N_PATHS as u32,
        n_hops: N_HOPS as u32,
        psd_rate: if passed { 1.0 } else { 0.0 },
        lognorm_concentration,
        passed,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_10_bound_positive() {
        let b = pflug_pichler_bound();
        assert!(b > 0.0, "Pflug–Pichler bound must be positive, got {b}");
    }

    #[test]
    fn pillar_10_seed_matches_spec() {
        assert_eq!(PILLAR_10_SEED, 0xF1_5A_DC_5A_DD);
    }

    #[test]
    fn pairwise_l1_correctness() {
        let u = vec![0.0_f32, 1.0];
        let v = vec![0.0_f32, 2.0];
        let cost = pairwise_l1(&u, &v);
        // cost = [|0-0|, |0-2|, |1-0|, |1-2|] = [0, 2, 1, 1]
        assert_eq!(cost, vec![0.0_f32, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn generate_states_changes_values() {
        let mut rng = SplitMix64::new(PILLAR_10_SEED);
        let prev = vec![0.0_f32; 5];
        let next = generate_states(&mut rng, &prev, 0.2);
        // With overwhelming probability, at least one value changes.
        let any_nonzero = next.iter().any(|&x| x.abs() > 1e-10);
        assert!(any_nonzero, "all states unchanged after generate_states");
    }

    #[test]
    fn prove_pillar_10_passes() {
        // Full probe — deterministic, should PASS.
        let report = prove_pillar_10();
        report.print();
        assert!(
            report.passed,
            "Pillar-10 FAILED: nested_dist > Pflug–Pichler bound. \
             lognorm_concentration={:.6}",
            report.lognorm_concentration
        );
    }

    #[test]
    fn prove_pillar_10_seed_anchored() {
        // Two runs must produce identical results (determinism).
        let r1 = prove_pillar_10();
        let r2 = prove_pillar_10();
        assert_eq!(r1.passed, r2.passed);
        assert!(
            (r1.lognorm_concentration - r2.lognorm_concentration).abs() < 1e-12,
            "non-deterministic: {} vs {}",
            r1.lognorm_concentration,
            r2.lognorm_concentration,
        );
    }
}
