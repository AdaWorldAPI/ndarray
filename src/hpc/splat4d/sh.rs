//! `splat4d::sh` — G1 SH deg-3 inquiry-direction evaluation (A3 worker, PR-X4).
//!
//! Evaluates degree-3 spherical harmonics (16 coefficients per channel) in the
//! **inquiry direction** (vocab × thinking_style) rather than the view direction
//! used by the graphics splat in `crate::hpc::splat3d::sh`.
//!
//! The parity test for done criterion G1 must verify bit-exact agreement with
//! `crate::hpc::splat3d::sh::sh_eval_deg3` on the same coefficient slice and
//! direction.
//!
//! # References
//!
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G1 — SH deg-3"
//! - `crate::hpc::splat3d::sh::sh_eval_deg3` — parity reference

//// TODO(A3): implement inquiry-direction SH deg-3 evaluation.
//// Mirror `crate::hpc::splat3d::sh::sh_eval_deg3` with the same coefficient
//// layout (48 floats, channel-major). Parity test: bit-exact on 10K random
//// (direction, coefficients) pairs against the splat3d reference.
//// Ref: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G1"
