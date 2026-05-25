#![allow(missing_docs)]
//! Pillar-15 — Mexican-hat / Difference-of-Gaussians unimodality
//! certification. **DEFERRED** pending Mexican-hat resonance kernel
//! landing in `ndarray::hpc`.
//!
//! Substrate-tier pillar: when activated, this pillar will certify that
//! the Difference-of-Gaussians (DoG) center-surround resonance kernel
//! consumed by the dragonfly-tier cognitive shader has exactly one local
//! maximum at the origin and exactly one annular minimum at the surround
//! radius, across the full parameter band the production stack uses.
//!
//! # The unimodality claim
//!
//! For two centered isotropic Gaussians with standard deviations
//! `σ_c < σ_s`, the Difference-of-Gaussians is
//!
//! ```text
//!   DoG(r ; σ_c, σ_s)  =  G(r ; σ_c)  −  G(r ; σ_s)
//! ```
//!
//! Differentiating with respect to `r` and setting to zero gives the
//! single positive root `r_min(σ_c, σ_s)`. For DoG to behave as a
//! center-surround kernel — peak at origin, negative annulus, decay to
//! zero — we need:
//!
//! 1. `DoG(0) > 0` (positive at center)
//! 2. `DoG'(r) < 0` for `0 < r < r_min` (monotone decrease from peak)
//! 3. `DoG'(r) > 0` for `r > r_min` (monotone increase from annulus to zero)
//! 4. Exactly one root in `(0, ∞)` for the equation `DoG(r) = 0`
//!
//! These hold for any `σ_c < σ_s`, but the *measurable* sharpness of the
//! center-surround structure depends on the ratio `κ = σ_s / σ_c`. The
//! production stack uses `κ ∈ [1.5, 3.0]`; values outside this band
//! produce either too-broad surrounds (resonance bleeds into neighbors,
//! cascade misfires) or too-sharp surrounds (numerical instability at
//! the inflection point).
//!
//! # Why the substrate needs this
//!
//! The dragonfly Mexican-hat resonance is the center-surround operator
//! that implements lateral inhibition in the cognitive shader. Bad
//! parameter choices (κ outside the band, or accidental introduction
//! of a kernel that's a sum-of-Gaussians rather than a difference)
//! produce *bimodal* or *saddle* responses that look like resonance
//! but bias activation toward the wrong neighborhoods. The failure
//! mode is silent: the cascade still produces top-K results, just
//! systematically biased.
//!
//! # Probe design (deferred until kernel lands)
//!
//! When the Mexican-hat / DoG kernel lands in `ndarray::hpc::dragonfly`
//! (or wherever it ultimately lives), Pillar-15 will:
//!
//! 1. Sweep `κ = σ_s / σ_c ∈ {1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0}`.
//! 2. For each κ, evaluate the DoG kernel on a 1D radial grid of
//!    `N_RADIAL = 4096` samples covering `r ∈ [0, 6 σ_s]`.
//! 3. Numerically locate all critical points (sign changes in `DoG'`)
//!    via finite differences with `h = r_max / N_RADIAL`.
//! 4. Verify:
//!    - Exactly one positive critical point (the annular minimum).
//!    - `DoG(0) > 0` and is the global maximum on the grid.
//!    - `DoG(r) → 0` as `r → r_max` within `1e-6 · DoG(0)`.
//!    - For `κ ∈ [1.5, 3.0]`, the second-derivative ratio at the
//!      inflection point is within the production-band budget.
//!
//! # Activation gate
//!
//! Active when:
//! - `ndarray::hpc::dragonfly::mexican_hat` (or canonical location) is
//!   exposed as a public function `dog_eval(r: f32, sigma_c: f32, sigma_s: f32) -> f32`.
//! - This module's `prove_pillar_15` body is updated to call it; the
//!   current `DEFERRED` placeholder must be replaced with the real probe.
//!
//! # Pass criteria (when active)
//!
//! - All seven critical-point checks pass for every κ in the sweep.
//! - `psd_rate ≥ 1.0` (every κ-sample contributes a binary pass/fail;
//!   any failure tanks the rate).
//! - `lognorm_concentration` = `log(1 + max kappa-band deviation)`.
//!
//! # References
//!
//! * Marr, D., & Hildreth, E. (1980). *Theory of edge detection.*
//!   Proc. R. Soc. Lond. B 207(1167). (The original DoG / Marr-Hildreth
//!   operator, source of the Mexican-hat name.)
//! * Lindeberg, T. (1994). *Scale-Space Theory in Computer Vision.*
//!   Kluwer. (Scale-space justification for the σ_s / σ_c ratio band.)
//!
//! # SEED
//!
//! `PILLAR_15_SEED = 0x_C15_DAD51_DEFE` (Mexican-hat unimodality, reserved).

use crate::hpc::pillar::prove_runner::PillarReport;

/// Deterministic seed for all Pillar-15 RNG streams.
pub const PILLAR_15_SEED: u64 = 0x_C15D_AD51_DEFE;

/// Run the Pillar-15 deferred placeholder.
///
/// **Currently DEFERRED.** Returns a `PillarReport` with all-zero
/// numeric fields and `passed = true`, signalling "no probe ran".
/// Replace the body when the Mexican-hat kernel lands; see the
/// "Probe design" section in the module docs for the activation plan.
///
/// # Returns
///
/// A deferred-shape `PillarReport`:
/// - `n_paths = 0`, `n_hops = 0` (no probe executed)
/// - `psd_rate = 0.0`, `lognorm_concentration = 0.0`
/// - `passed = true` (the deferral itself is not a failure)
pub fn prove_pillar_15() -> PillarReport {
    PillarReport {
        pillar_id: 15,
        seed: PILLAR_15_SEED,
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
    fn pillar_15_seed_matches_spec() {
        assert_eq!(PILLAR_15_SEED, 0x_C15D_AD51_DEFE);
    }

    #[test]
    fn pillar_15_deferred_shape() {
        // DEFERRED probe must return a recognisable deferred-shape report:
        // zeros everywhere except pillar_id and seed, passed = true.
        let r = prove_pillar_15();
        assert_eq!(r.pillar_id, 15);
        assert_eq!(r.seed, PILLAR_15_SEED);
        assert_eq!(r.n_paths, 0);
        assert_eq!(r.n_hops, 0);
        assert_eq!(r.psd_rate, 0.0);
        assert_eq!(r.lognorm_concentration, 0.0);
        assert!(r.passed, "deferred pillar should not be marked failed");
    }

    #[test]
    fn pillar_15_seed_anchored() {
        let r1 = prove_pillar_15();
        let r2 = prove_pillar_15();
        assert_eq!(r1.passed, r2.passed);
        assert_eq!(r1.psd_rate, r2.psd_rate);
        assert_eq!(r1.lognorm_concentration, r2.lognorm_concentration);
    }
}
