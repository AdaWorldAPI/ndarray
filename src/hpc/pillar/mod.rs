//! Pillar probe certification module — `ndarray::hpc::pillar`.
//!
//! Houses the shared harness and per-pillar mathematical certification probes
//! that migrate from `lance-graph/crates/jc/` per PR-X11, plus the
//! substrate-tier pillars (12–17) that test ndarray's own cognitive-shader
//! substrate against established mathematics.
//!
//! # Tier structure
//!
//! - **Pillars 6–11 (cognitive-architecture tier)** migrate from
//!   `lance-graph/crates/jc/`. They test invariants about the cognitive
//!   architecture built *on top of* the substrate (EWA-sandwich for
//!   splat covariance push-forward, Pflug-Pichler nested distance for
//!   CAM-PQ tree quantization, Hambly-Lyons uniqueness for sigker
//!   signature classification, etc.).
//! - **Pillars 12–17 (substrate tier)** are native to ndarray. They test
//!   the substrate-level mathematics that the cognitive shader relies on
//!   regardless of any specific architectural wiring above it (partition
//!   of unity for Gaussian splats, contraction for HHTL cascade,
//!   partial-order axioms for the OGIT type gate, and so on).
//!
//! Pillars in both tiers share the same `PillarReport` shape and the same
//! `SplitMix64` deterministic RNG harness. Certification is about
//! **determinism + inspectability**, not which tier the math belongs to.
//!
//! # Structure
//!
//! ```text
//! src/hpc/pillar/
//! ├── mod.rs                ← this file; public surface + re-exports
//! ├── prove_runner.rs       ← B8: shared splitmix64 RNG, PillarReport, helpers
//! │
//! │   ─── Cognitive-architecture tier (migrated from lance-graph/crates/jc) ───
//! ├── ewa_sandwich_2d.rs    ← B1: Pillar-6  2D EWA sandwich + prove()
//! ├── ewa_sandwich_3d.rs    ← B2: Pillar-7  3D EWA sandwich + prove()
//! ├── koestenberger.rs      ← B3: Pillar-7.5 Koestenberger PSD path
//! ├── temporal_sandwich.rs  ← B4: Pillar-8  temporal drift sandwich
//! ├── cov_high_d.rs         ← B5: Pillar-9  Cov16384 CLT probe
//! ├── pflug.rs              ← B6: Pillar-10 Pflug-Pichler nested distance
//! ├── signature.rs          ← B7: Pillar-11 f32 depth-3 kernel-stability battery
//! ├── lattice_signature.rs  ← Pillar-11 lattice lane: bit-exact i128 signature + Annals Thm 5/6 certificate
//! │
//! │   ─── Substrate tier (native to ndarray) ───
//! ├── splat_invariants.rs   ← Pillar-12  Splat-construction rotation invariance
//! ├── hhtl_contraction.rs   ← Pillar-13  HHTL cascade contraction
//! ├── ogit_lattice.rs       ← Pillar-14  OGIT type-gate lattice closure
//! ├── mexican_hat.rs        ← Pillar-15  Mexican-hat unimodality (DEFERRED)
//! ├── btsp_unbiased.rs      ← Pillar-16  BTSP gating unbiasedness (DEFERRED)
//! └── tree_balance.rs       ← Pillar-17  Quaternary tree balance     (DEFERRED)
//! ```
//!
//! # Invariant 12
//!
//! Certification is about **determinism + inspectability**, not repo separation.
//! Every `prove()` probe is SEED-anchored, commits `RESULTS.md` lines per run,
//! and can be cross-verified against numpy/scipy/R.
//!
//! # Feature gate
//!
//! This module is compiled under `#[cfg(feature = "pillar")]`.
//! Enable with `--features std,linalg,pillar`.

/// Shared probe harness: splitmix64 RNG, [`PillarReport`], contractive-SPD helpers,
/// and PSD-rate assertion. Consumed by all B1–B7 pillar workers and the
/// substrate-tier pillars 12–17.
pub mod prove_runner;

// ── Cognitive-architecture tier — Pillars 6 through 11 ──────────────────────

/// Pillar-6: 2D EWA sandwich certification probe (B1).
pub mod ewa_sandwich_2d;

/// Pillar-7: 3D EWA sandwich certification probe (B2).
pub mod ewa_sandwich_3d;

/// Pillar-7.5: Koestenberger PSD path certification probe (B3).
pub mod koestenberger;

/// Pillar-8: Temporal drift sandwich certification probe (B4).
pub mod temporal_sandwich;

/// Pillar-9: Cov16384 Düker–Zoubouloglou CLT probe (B5).
pub mod cov_high_d;

/// Pillar-10: Pflug–Pichler nested Wasserstein distance (B6).
pub mod pflug;

/// Pillar-11: Hambly–Lyons iterated-integrals signature transform (B7).
pub mod signature;

/// Pillar-11 (lattice lane): the BIT-EXACT integer signature of a lattice
/// walk (`k!·S_k` as `i128`) and the finite-depth Hambly–Lyons Theorem 5/6
/// certificate — reduced words separated at `⌈4.7917·L⌉` (Annals Thm 5/6), tree-like words
/// exactly the identity, the depth-2 figure-of-8 class separated at level 3,
/// the `d = 1` collapse pinned. Integer depth policy, no floating point.
pub mod lattice_signature;

// ── Substrate tier — Pillars 12 through 17 (native to ndarray) ──────────────

/// Pillar-12: Anisotropic-splat construction-invariance certification.
///
/// Verifies that the splat covariance construction
/// `Σ = R(q) · diag(s²) · R(q)ᵀ` consumed by `crate::hpc::splat3d` is
/// symmetric positive-definite and that its trace, determinant, and
/// Frobenius norm match their closed-form rotation-invariant values
/// for every positive scale and every unit quaternion in the production
/// sample space.
pub mod splat_invariants;

/// Pillar-13: HHTL cascade contraction certification.
///
/// Verifies that the weighted Hamming-bundle operator consumed by the
/// dn-tree and HHTL cascade is a contraction in normalized Hamming
/// distance, with a measured per-level ratio matching the Bernoulli-
/// mixture prediction `(1 − lr)`.
pub mod hhtl_contraction;

/// Pillar-14: OGIT type-gate lattice-closure certification.
///
/// Verifies that synthetic `rdfs:subClassOf`-style type-compatibility
/// relations satisfy the three partial-order axioms (reflexivity,
/// antisymmetry, transitivity) after closure, and reports the lattice
/// width as an informational metric.
pub mod ogit_lattice;

/// Pillar-15: Mexican-hat / Difference-of-Gaussians unimodality.
/// **DEFERRED** pending Mexican-hat resonance kernel.
pub mod mexican_hat;

/// Pillar-16: BTSP-gated bundling unbiasedness.
/// **DEFERRED** pending stable BTSP plasticity API.
pub mod btsp_unbiased;

/// Pillar-17: Quaternary DN-tree balance under Zipf access.
/// **DEFERRED** pending Zipf workload simulator.
pub mod tree_balance;

// ── Re-exports — the core harness types at the `pillar::` surface ───────────

// Re-export the core harness types at the `pillar::` surface so pillar workers
// can write `use crate::hpc::pillar::{SplitMix64, PillarReport, ...};`
pub use prove_runner::{assert_psd_rate, random_contractive_spd2, random_contractive_spd3, PillarReport, SplitMix64};

// ── Re-exports — substrate-tier prove() entry points ────────────────────────

pub use btsp_unbiased::prove_pillar_16;
pub use hhtl_contraction::prove_pillar_13;
pub use lattice_signature::prove_pillar_11_lattice;
pub use mexican_hat::prove_pillar_15;
pub use ogit_lattice::prove_pillar_14;
pub use splat_invariants::prove_pillar_12;
pub use tree_balance::prove_pillar_17;

// ── Convenience runner — all substrate-tier pillars in sequence ─────────────

/// Run all substrate-tier pillars (12–17) in order, returning the reports.
///
/// Active probes (12, 13, 14) execute their full workload; deferred probes
/// (15, 16, 17) return their `DEFERRED`-shape report. Cognitive-architecture
/// pillars (6–11) are not run by this function — each has its own
/// `prove_pillar_N()` entry point and is run individually.
///
/// # Example
///
/// ```ignore
/// use ndarray::hpc::pillar::run_substrate_tier;
/// let reports = run_substrate_tier();
/// for r in &reports {
///     r.print();
/// }
/// assert!(reports.iter().all(|r| r.passed));
/// ```
pub fn run_substrate_tier() -> Vec<PillarReport> {
    vec![
        prove_pillar_12(),
        prove_pillar_13(),
        prove_pillar_14(),
        prove_pillar_15(),
        prove_pillar_16(),
        prove_pillar_17(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_substrate_tier_produces_six_reports() {
        let reports = run_substrate_tier();
        assert_eq!(reports.len(), 6);
        let ids: Vec<u8> = reports.iter().map(|r| r.pillar_id).collect();
        assert_eq!(ids, vec![12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn run_substrate_tier_all_pass() {
        // Active pillars must pass; deferred pillars are marked passed by
        // convention (the deferral itself is not a failure).
        let reports = run_substrate_tier();
        for r in &reports {
            assert!(r.passed, "Pillar-{} did not pass", r.pillar_id);
        }
    }

    #[test]
    fn substrate_tier_is_deterministic() {
        // Two consecutive runs must produce identical reports.
        let r1 = run_substrate_tier();
        let r2 = run_substrate_tier();
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.pillar_id, b.pillar_id);
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.n_paths, b.n_paths);
            assert_eq!(a.n_hops, b.n_hops);
            assert!((a.psd_rate - b.psd_rate).abs() < 1e-12);
            assert!((a.lognorm_concentration - b.lognorm_concentration).abs() < 1e-12);
            assert_eq!(a.passed, b.passed);
        }
    }
}
