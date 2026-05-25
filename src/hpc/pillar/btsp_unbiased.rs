#![allow(missing_docs)]
//! Pillar-16 — BTSP-gated bundling unbiasedness certification.
//! **DEFERRED** pending stable BTSP plasticity API in
//! `ndarray::hpc::dn_tree`.
//!
//! Substrate-tier pillar: when activated, this pillar will certify that
//! the BTSP-gated stochastic bundling operator used by
//! `dn_tree::DNTree::update` (with `btsp_gate_prob > 0` and
//! `btsp_boost > 1`) is an *unbiased* estimator of the underlying
//! signal — i.e. that the gate does not introduce a systematic drift
//! in the long-run mean of the bundled summary.
//!
//! # The unbiasedness claim
//!
//! Let `B_t = bundle(B_{t−1}, hv_t, lr_eff(t))` where
//!
//! ```text
//!   lr_eff(t)  =  lr · boost     with probability p_btsp
//!   lr_eff(t)  =  lr             with probability 1 − p_btsp
//! ```
//!
//! For unbiasedness we require the BTSP-gated dynamics to converge to
//! the *same* long-run mean as the gate-disabled dynamics
//! (`lr_eff(t) ≡ lr` for all `t`). Equivalently, for any fixed input
//! distribution `μ` of hypervectors:
//!
//! ```text
//!   lim_{T→∞}  E[B_T | BTSP active]   =   lim_{T→∞}  E[B_T | BTSP disabled]
//! ```
//!
//! This holds if and only if the gate's expected step
//! `E[lr_eff] = lr · ((1 − p_btsp) + p_btsp · boost)` produces an
//! effective bundling weight that the bundling operator's fixed-point
//! equation tolerates without drift — formally, that `lr_eff` enters
//! linearly through Doob's optional stopping theorem on the
//! martingale-decomposable component of the bundle update.
//!
//! # Why the substrate needs this
//!
//! The BTSP gate is the substrate's mechanism for *importance-weighted*
//! plasticity — high-salience events get a 7× learning-rate boost
//! (mimicking CaMKII amplification in biological synapses). If the gate
//! is *biased*, the long-run summary drifts toward an artifact of the
//! gate schedule rather than the true input distribution. The drift is
//! invisible in lab conditions (Jan's 100% recall numbers at θ ∈ [1.45,
//! 1.60]) because lab inputs are themselves drawn from the target
//! distribution; the bias surfaces only when the input distribution
//! shifts under operational load.
//!
//! Pillar-16 pins the unbiasedness claim to a measurable Monte-Carlo
//! estimate of the long-run mean, with the gate-disabled run as the
//! reference.
//!
//! # Probe design (deferred until BTSP API stabilises)
//!
//! 1. Fix bit width `BIT_WIDTH = 1024`, base learning rate `lr = 0.05`,
//!    gate probability `p_btsp = 0.01`, boost `boost = 7.0`.
//! 2. Fix a target distribution `μ` over input hypervectors — a small
//!    discrete set of `K = 8` "prototype" vectors, drawn uniformly per
//!    step.
//! 3. Run `T = 10_000` bundle steps with BTSP disabled (`p_btsp = 0`);
//!    record `B_T` as the reference mean `M_ref`.
//! 4. Run `T` bundle steps with BTSP enabled; record `B_T` as `M_btsp`.
//! 5. Repeat both runs `N_REPLICAS = 64` times with different seeds.
//! 6. Verify that
//!    `‖ mean(M_btsp) − mean(M_ref) ‖_∞ < BIAS_TOLERANCE`
//!    and that the per-bit empirical bias falls within a
//!    `BIAS_CHI2_BUDGET` chi-squared confidence band.
//!
//! # Cross-verification
//!
//! When active, the probe will compare ndarray's `dn_tree::bundle_into`
//! output against an independent re-derivation of the Bernoulli-mixture
//! step (same approach as Pillar-13). The math is verifiable against a
//! numpy / scipy reference implementation in `scripts/pillar_16_oracle.py`
//! (to be written alongside activation).
//!
//! # Activation gate
//!
//! Active when:
//! - `ndarray::hpc::dn_tree::bundle_into` (or equivalent) has a stable
//!   public signature accepting `(current, hv, lr, boost, rng)`.
//! - `DNConfig::with_btsp` and the BTSP fire/boost semantics are
//!   contract-frozen (no further breaking changes).
//!
//! # Pass criteria (when active)
//!
//! - `‖ E[B_btsp] − E[B_ref] ‖_∞ < 0.05` (5% bit-bias tolerance).
//! - Chi-squared test on per-bit bias: `χ² ≤ critical at α = 0.01,
//!   df = BIT_WIDTH − 1`.
//! - `psd_rate` ≥ 0.999 (fraction of bit positions within tolerance).
//! - `lognorm_concentration` = `log(1 + max bit-bias)`.
//!
//! # References
//!
//! * Bittner, K. C., Milstein, A. D., Grienberger, C., Romani, S., &
//!   Magee, J. C. (2017). *Behavioral time scale synaptic plasticity
//!   underlies CA1 place fields.* Science 357(6355). (Original BTSP
//!   observation in hippocampal CA1.)
//! * Doob, J. L. (1953). *Stochastic Processes.* Wiley. (Optional
//!   stopping for the martingale decomposition that pins the
//!   unbiasedness criterion.)
//!
//! # SEED
//!
//! `PILLAR_16_SEED = 0x_C16_B751P_BC057` (BTSP boost, reserved).

use crate::hpc::pillar::prove_runner::PillarReport;

/// Deterministic seed for all Pillar-16 RNG streams.
pub const PILLAR_16_SEED: u64 = 0x_C16B_751B_BC05;

/// Run the Pillar-16 deferred placeholder.
///
/// **Currently DEFERRED.** Replace the body when the BTSP API is
/// stabilised; see the "Probe design" section in the module docs for
/// the activation plan.
pub fn prove_pillar_16() -> PillarReport {
    PillarReport {
        pillar_id: 16,
        seed: PILLAR_16_SEED,
        n_paths: 0,
        n_hops: 0,
        psd_rate: 0.0,
        lognorm_concentration: 0.0,
        passed: true,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_16_seed_matches_spec() {
        assert_eq!(PILLAR_16_SEED, 0x_C16B_751B_BC05);
    }

    #[test]
    fn pillar_16_deferred_shape() {
        let r = prove_pillar_16();
        assert_eq!(r.pillar_id, 16);
        assert_eq!(r.seed, PILLAR_16_SEED);
        assert_eq!(r.n_paths, 0);
        assert_eq!(r.n_hops, 0);
        assert_eq!(r.psd_rate, 0.0);
        assert_eq!(r.lognorm_concentration, 0.0);
        assert!(r.passed);
    }

    #[test]
    fn pillar_16_seed_anchored() {
        let r1 = prove_pillar_16();
        let r2 = prove_pillar_16();
        assert_eq!(r1.passed, r2.passed);
        assert_eq!(r1.psd_rate, r2.psd_rate);
        assert_eq!(r1.lognorm_concentration, r2.lognorm_concentration);
    }
}
