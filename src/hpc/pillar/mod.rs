//! Pillar probe certification module — `ndarray::hpc::pillar`.
//!
//! Houses the shared harness and per-pillar mathematical certification probes
//! that migrate from `lance-graph/crates/jc/` per PR-X11.
//!
//! # Structure
//!
//! ```text
//! src/hpc/pillar/
//! ├── mod.rs              ← this file; public surface + re-exports
//! ├── prove_runner.rs     ← B8: shared splitmix64 RNG, PillarReport, helpers
//! ├── ewa_sandwich_2d.rs  ← B1: Pillar-6 2D EWA sandwich + prove()
//! ├── ewa_sandwich_3d.rs  ← B2: Pillar-7 3D EWA sandwich + prove()
//! ├── koestenberger.rs    ← B3: Pillar-7.5 Koestenberger PSD path
//! ├── temporal_sandwich.rs← B4: Pillar-8 temporal drift sandwich + prove()
//! ├── cov_high_d.rs       ← B5: Pillar-9 Cov16384 CLT probe
//! ├── pflug.rs            ← B6: Pillar-10 Pflug-Pichler nested distance
//! └── signature.rs        ← B7: Pillar-11 Hambly-Lyons signature transform
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
/// and PSD-rate assertion. Consumed by all B1–B7 pillar workers.
pub mod prove_runner;

// ── Pillar-6 through Pillar-11 (B1–B7) — stubs; workers land these in parallel ──
// Each module will export:
//   - The pillar's typed wrapper struct
//   - `pub fn prove() -> PillarReport`
//   - The math kernel as `pub` functions consuming `linalg::Spd{2,3,N}`

/// Pillar-6: 2D EWA sandwich certification probe (B1).
// pub mod ewa_sandwich_2d;

/// Pillar-7: 3D EWA sandwich certification probe (B2).
// pub mod ewa_sandwich_3d;

/// Pillar-7.5: Koestenberger PSD path certification probe (B3).
// pub mod koestenberger;

/// Pillar-8: Temporal drift sandwich certification probe (B4).
pub mod temporal_sandwich;

/// Pillar-9: Cov16384 Düker–Zoubouloglou CLT probe (B5).
// pub mod cov_high_d;

/// Pillar-10: Pflug–Pichler nested Wasserstein distance (B6).
// pub mod pflug;

/// Pillar-11: Hambly–Lyons iterated-integrals signature transform (B7).
// pub mod signature;

// Re-export the core harness types at the `pillar::` surface so B1–B7 can write
// `use crate::hpc::pillar::{SplitMix64, PillarReport, ...};`
pub use prove_runner::{assert_psd_rate, random_contractive_spd2, random_contractive_spd3, PillarReport, SplitMix64};
