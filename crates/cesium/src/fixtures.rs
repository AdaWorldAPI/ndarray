//! `fixtures` (group D) — golden scenes + parity gates.
//!
//! Defines the canonical test scenes used to validate `splat3d` parity against
//! reference renders, and the SSIM / PSNR thresholds that constitute a "pass".
//!
//! # Philosophy
//!
//! The oracle **measures** parity; this module **gates** it.  A `ParityGate`
//! bundles a scene descriptor with its acceptable metric bounds.  Whether to
//! treat a failed gate as a CI hard-failure is the caller's decision.
//!
//! # First reference scene
//!
//! The first reference is an Inria `.ply` scene, loaded via
//! `splat3d::ply::read_ply` (confirmed: `src/hpc/splat3d/ply.rs` line 110).
//! The `staehlli/3D-settlement-development` scene is the named fixture source
//! from `lib.rs` grounding references.
//!
//! # Threshold rationale
//!
//! | Gate                | SSIM  | PSNR (dB) | Rationale                                      |
//! |---------------------|-------|-----------|------------------------------------------------|
//! | `SYNTHETIC_TIGHT`   | 0.99  | 40.0      | Synthetic scene, controlled camera, no sort risk |
//! | `INRIA_PLY_NOMINAL` | 0.95  | 30.0      | Real scene; sort-order risk may cost ~3–5 dB   |
//! | `SMOKE_TEST`        | 0.80  | 20.0      | Minimal bar: confirms pipeline runs at all     |
//!
//! # Sort-order risk (see `oracle` module doc)
//!
//! All thresholds above are set conservatively to account for the front-to-back
//! vs back-to-front accumulation delta documented in `oracle.rs`.  The
//! `INRIA_PLY_NOMINAL` gate in particular leaves 5–10 dB of headroom that a
//! sort-order-aware comparison could reclaim.
//!
//! # Implementation status
//!
//! **Commented scaffold only.**  The `Fixture` and `ParityGate` structs are
//! live `std`-only.  All references to `splat3d` symbols are commented out
//! pending the ndarray dep re-enable + Opus + CodeRabbit review.
//!
//! # Source grounding
//!
//! Symbols cited (confirmed from local source):
//! - `read_ply` — `src/hpc/splat3d/ply.rs:110`
//! - Framebuffer layout — `raster.rs` doc: interleaved RGB, len = `3·W·H`
//! - `GaussianBatch::len` — `gaussian.rs:89` (active gaussian count)

// ─────────────────────────────────────────────────────────────────────────────
// Parity threshold constants (live)
// ─────────────────────────────────────────────────────────────────────────────

/// SSIM / PSNR pair for a parity gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Minimum acceptable SSIM in `[0, 1]`.
    pub min_ssim: f32,
    /// Minimum acceptable PSNR in dB.
    pub min_psnr_db: f32,
}

impl Thresholds {
    /// Synthetic scene, controlled camera, minimal sort-order risk.
    pub const SYNTHETIC_TIGHT: Self = Self {
        min_ssim: 0.99,
        min_psnr_db: 40.0,
    };

    /// Real Inria PLY scene; sort-order risk may cost ~3–5 dB of PSNR.
    /// Threshold is set conservatively to account for front-to-back vs
    /// back-to-front accumulation divergence (see `oracle` module doc).
    pub const INRIA_PLY_NOMINAL: Self = Self {
        min_ssim: 0.95,
        min_psnr_db: 30.0,
    };

    /// Smoke test: confirms the pipeline runs and produces a plausible image.
    pub const SMOKE_TEST: Self = Self {
        min_ssim: 0.80,
        min_psnr_db: 20.0,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture descriptor (live — std-only)
// ─────────────────────────────────────────────────────────────────────────────

/// Descriptor for a golden test scene.
///
/// A `Fixture` names a scene and records its provenance.  It does NOT hold the
/// actual scene data (too large for in-memory constants); the oracle pipeline
/// loads data from disk at test time.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Short identifier used in test output (e.g. `"settlement_dev"`).
    pub id: &'static str,
    /// Human-readable description of the scene and its provenance.
    pub description: &'static str,
    /// Expected number of Gaussians in the loaded batch (informational).
    /// `None` means unchecked.
    pub expected_gaussian_count: Option<usize>,
    /// Image width used for the reference render, in pixels.
    pub width: u32,
    /// Image height used for the reference render, in pixels.
    pub height: u32,
    /// Background colour `[R, G, B]` in `[0, 1]` used for the reference render.
    pub background: [f32; 3],
}

/// Named fixtures.  Path resolution is left to the test harness so the crate
/// has no compile-time dependency on the filesystem layout.
pub mod scenes {
    use super::Fixture;

    /// Inria 3DGS garden scene (small variant, ~300K Gaussians).
    /// Source: Kerbl et al. 2023 supplemental material.
    // UNVERIFIED: exact gaussian count — 300K is an estimate from the paper.
    pub const INRIA_GARDEN_SMALL: Fixture = Fixture {
        id: "inria_garden_small",
        description: "Inria 3DGS garden scene (small variant), Kerbl 2023. \
                      Loaded via splat3d::ply::read_ply. \
                      // UNVERIFIED: ~300K gaussians.",
        expected_gaussian_count: None, // UNVERIFIED
        width: 1920,
        height: 1080,
        background: [0.0, 0.0, 0.0],
    };

    /// staehlli/3D-settlement-development scene (named grounding ref in lib.rs).
    /// This is the primary parity fixture for the cesium crate.
    // UNVERIFIED: gaussian count and exact image dimensions for this scene.
    pub const SETTLEMENT_DEV: Fixture = Fixture {
        id: "settlement_dev",
        description: "staehlli/3D-settlement-development 3DGS scene. \
                      Grounding reference named in cesium/src/lib.rs. \
                      Loaded via splat3d::ply::read_ply. \
                      // UNVERIFIED: gaussian count and image dims.",
        expected_gaussian_count: None, // UNVERIFIED
        width: 1920,
        height: 1080,
        background: [0.0, 0.0, 0.0],
    };

    /// Synthetic unit scene: 4 Gaussians at known positions, controlled camera.
    /// Used for `SYNTHETIC_TIGHT` gate — no sort-order ambiguity.
    pub const SYNTHETIC_UNIT: Fixture = Fixture {
        id: "synthetic_unit",
        description: "4 Gaussians at axis-aligned positions, identity camera. \
                      Fully deterministic: no sort-order risk, SSIM ≥ 0.99 expected.",
        expected_gaussian_count: Some(4),
        width: 64,
        height: 64,
        background: [0.0, 0.0, 0.0],
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// ParityGate (live — std-only)
// ─────────────────────────────────────────────────────────────────────────────

/// A parity gate bundles a [`Fixture`] with its [`Thresholds`].
///
/// Use [`ParityGate::check`] to evaluate a [`crate::oracle::ParityMetrics`]
/// against the gate.  The gate reports pass/fail but does not panic — the
/// caller decides whether a failed gate is a CI hard failure.
#[derive(Debug, Clone)]
pub struct ParityGate {
    /// The scene this gate applies to.
    pub fixture: Fixture,
    /// SSIM / PSNR thresholds.
    pub thresholds: Thresholds,
}

/// Result of evaluating a parity gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GateResult {
    /// Whether the gate passed.
    pub passed: bool,
    /// Observed SSIM.
    pub ssim: f32,
    /// Observed PSNR in dB.
    pub psnr: f32,
    /// SSIM margin (observed − minimum; negative = failing).
    pub ssim_margin: f32,
    /// PSNR margin in dB (observed − minimum; negative = failing).
    pub psnr_margin_db: f32,
}

impl ParityGate {
    /// Evaluate the gate against observed metrics.
    pub fn check(&self, metrics: &crate::oracle::ParityMetrics) -> GateResult {
        let ssim_margin = metrics.ssim - self.thresholds.min_ssim;
        let psnr_margin_db = if metrics.psnr.is_infinite() {
            f32::INFINITY
        } else {
            metrics.psnr - self.thresholds.min_psnr_db
        };
        let passed = metrics.passes(self.thresholds.min_ssim, self.thresholds.min_psnr_db);
        GateResult {
            passed,
            ssim: metrics.ssim,
            psnr: metrics.psnr,
            ssim_margin,
            psnr_margin_db,
        }
    }
}

/// Pre-built gates for the named scenes.
pub mod gates {
    use super::{scenes, ParityGate, Thresholds};

    /// Smoke test gate for the settlement_dev scene.
    pub fn settlement_dev_smoke() -> ParityGate {
        ParityGate {
            fixture: scenes::SETTLEMENT_DEV.clone(),
            thresholds: Thresholds::SMOKE_TEST,
        }
    }

    /// Nominal gate for the settlement_dev scene (accounts for sort-order risk).
    pub fn settlement_dev_nominal() -> ParityGate {
        ParityGate {
            fixture: scenes::SETTLEMENT_DEV.clone(),
            thresholds: Thresholds::INRIA_PLY_NOMINAL,
        }
    }

    /// Tight gate for the synthetic unit scene.
    pub fn synthetic_unit_tight() -> ParityGate {
        ParityGate {
            fixture: scenes::SYNTHETIC_UNIT.clone(),
            thresholds: Thresholds::SYNTHETIC_TIGHT,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Commented scaffold — live oracle integration blocked on ndarray dep
// ─────────────────────────────────────────────────────────────────────────────

// DEFERRED: When ndarray dep is re-enabled, wire the full fixture runner:
//
// use crate::oracle::{load_ply_batch, render_batch, ParityMetrics, OracleError};
// use ndarray_hpc::hpc::splat3d::project::Camera;
//
// /// Run the `settlement_dev` smoke test fixture end-to-end.
// ///
// /// Loads `ply_path`, renders at `SETTLEMENT_DEV` resolution, diffs against
// /// `reference_fb`, and evaluates against `settlement_dev_smoke()`.
// ///
// /// # Sort-order risk
// ///
// /// The `SMOKE_TEST` threshold (SSIM ≥ 0.80, PSNR ≥ 20 dB) is intentionally
// /// loose to tolerate front-to-back vs back-to-front divergence.  See `oracle`
// /// module doc for the named risk.
// pub fn run_settlement_dev_smoke(
//     ply_path: &std::path::Path,
//     reference_fb: &[f32],
// ) -> Result<GateResult, OracleError> {
//     let fixture = &scenes::SETTLEMENT_DEV;
//     // UNVERIFIED: Camera::identity_at_origin signature — confirm in project.rs
//     let camera = Camera::identity_at_origin(fixture.width, fixture.height);
//     let metrics = crate::oracle::run_ply_oracle(
//         ply_path,
//         reference_fb,
//         &camera,
//         fixture.background,
//     )?;
//     Ok(gates::settlement_dev_smoke().check(&metrics))
// }

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (live — std-only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::ParityMetrics;

    fn make_metrics(ssim: f32, psnr: f32) -> ParityMetrics {
        ParityMetrics {
            ssim,
            psnr,
            mse: 0.0,
            sample_count: 48,
        }
    }

    #[test]
    fn thresholds_constants_are_ordered() {
        // SMOKE < INRIA < SYNTHETIC
        assert!(Thresholds::SMOKE_TEST.min_ssim < Thresholds::INRIA_PLY_NOMINAL.min_ssim);
        assert!(Thresholds::INRIA_PLY_NOMINAL.min_ssim < Thresholds::SYNTHETIC_TIGHT.min_ssim);
        assert!(Thresholds::SMOKE_TEST.min_psnr_db < Thresholds::INRIA_PLY_NOMINAL.min_psnr_db);
        assert!(Thresholds::INRIA_PLY_NOMINAL.min_psnr_db < Thresholds::SYNTHETIC_TIGHT.min_psnr_db);
    }

    #[test]
    fn gate_check_passes_above_threshold() {
        let gate = gates::synthetic_unit_tight();
        let m = make_metrics(0.995, 45.0);
        let r = gate.check(&m);
        assert!(r.passed, "metrics above threshold must pass");
        assert!(r.ssim_margin > 0.0, "ssim_margin must be positive");
        assert!(r.psnr_margin_db > 0.0, "psnr_margin_db must be positive");
    }

    #[test]
    fn gate_check_fails_below_ssim() {
        let gate = gates::synthetic_unit_tight();
        let m = make_metrics(0.97, 45.0); // SSIM below 0.99
        let r = gate.check(&m);
        assert!(!r.passed, "SSIM below threshold must fail");
        assert!(r.ssim_margin < 0.0, "ssim_margin must be negative");
    }

    #[test]
    fn gate_check_fails_below_psnr() {
        let gate = gates::synthetic_unit_tight();
        let m = make_metrics(0.995, 35.0); // PSNR below 40 dB
        let r = gate.check(&m);
        assert!(!r.passed, "PSNR below threshold must fail");
        assert!(r.psnr_margin_db < 0.0, "psnr_margin_db must be negative");
    }

    #[test]
    fn gate_check_infinite_psnr_passes() {
        let gate = gates::settlement_dev_nominal();
        let m = make_metrics(0.97, f32::INFINITY);
        let r = gate.check(&m);
        assert!(r.passed, "infinite PSNR (identical buffers) must pass any gate");
        assert!(r.psnr_margin_db.is_infinite(), "psnr_margin_db must be infinite");
    }

    #[test]
    fn smoke_gate_is_looser_than_nominal() {
        let smoke = gates::settlement_dev_smoke();
        let nominal = gates::settlement_dev_nominal();
        assert!(smoke.thresholds.min_ssim <= nominal.thresholds.min_ssim);
        assert!(smoke.thresholds.min_psnr_db <= nominal.thresholds.min_psnr_db);
    }

    #[test]
    fn fixture_ids_are_unique() {
        let fixtures = [scenes::INRIA_GARDEN_SMALL.id, scenes::SETTLEMENT_DEV.id, scenes::SYNTHETIC_UNIT.id];
        let mut seen = std::collections::HashSet::new();
        for id in &fixtures {
            assert!(seen.insert(*id), "duplicate fixture id: {id}");
        }
    }

    #[test]
    fn synthetic_unit_fixture_has_known_gaussian_count() {
        assert_eq!(scenes::SYNTHETIC_UNIT.expected_gaussian_count, Some(4));
    }
}
