//! Degree-3 spherical-harmonics evaluator v2 — inquiry-direction extension.
//!
//! Placeholder module re-exporting v1 symbols.  The v2 SH module (A3
//! deliverable, G1 gap) extends degree-3 SH to inquiry-direction evaluation
//! for the 4×4 anisotropic kernel: `vocab × thinking_style` directional
//! state instead of `view_direction × RGB color`.
//!
//! A3's parity gate: `splat3d_v2::sh_eval_deg3_inquiry` must agree with
//! `splat3d::sh::sh_eval_deg3` to bit-exact precision on the 16-basis
//! overlap (done-criteria #3/G1 from master prompt).
//!
//! Per A1 brief § "File moves": `splat3d/sh.rs` → `splat3d_v2/sh.rs`
//! with the note "(A3 expands)".
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//!   § "File moves" (A3 expands)
//! - G1 SH projection deg-3: A3 worker
//! - PR-X10 A8: `linalg::sh` deg 4-7 — A3 consumes this for higher-order SH

//// TODO(A3): add sh_eval_deg3_inquiry for vocab × thinking_style direction
//// — parity must be bit-exact vs splat3d::sh::sh_eval_deg3 on 16-basis overlap
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§G1 Hilbert-3D + Done criteria #3

pub use crate::hpc::splat3d::{sh_eval_deg3, sh_eval_deg3_x16, SH_BASIS_PER_CHANNEL};
