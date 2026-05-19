//! Symmetric 3×3 SPD math v2 — temporal sandwich (Pillar 8).
//!
//! Placeholder module re-exporting v1 symbols.  The v2 spd3 module will
//! integrate the Pillar-8 temporal sandwich: `Spd3::sandwich(step_cov, cov)`
//! and `Spd3::sqrt` from PR-X11 (jc consolidation) for the L5/L6
//! moment-match and temporal-sandwich paths (done-criteria #4 from master
//! prompt).
//!
//! Per A1 brief § "File moves": `splat3d/spd3.rs` → `splat3d_v2/spd3.rs`.
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - Pillar-8 temporal sandwich: PR-X11 (jc consolidation) deliverable
//! - L5 moment-match `compose_cascade`: done-criteria #4
//! - L6 temporal sandwich `Spd3::sandwich/sqrt`: done-criteria #4
//! - PR-X11: `Spd3::sandwich`, `Spd3::sqrt`, `Spd3::from_rows`

//// TODO(A1 or later): integrate PR-X11 Spd3::sandwich/sqrt once PR-X11 lands
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§Carry-overs #PR-X11
//// TODO: add compose_cascade (L5 Gaussian-mixture moment-match) — done-criteria #4
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§Done criteria #4

pub use crate::hpc::splat3d::{sandwich, sandwich_x16, Spd3};
