//! `splat4d::revise` — G3 NARS truth-revision kernel + G4 `fast_exp_x16` audit
//! (A5 worker, PR-X4).
//!
//! # G3 — NARS truth-revision kernel
//!
//! SIMD-staged inner loop implementing:
//!
//! ```text
//! revise(T1, T2).freq = (f1·c1 + f2·c2) / (c1 + c2)
//! revise(T1, T2).conf = (c1 + c2) / (c1 + c2 + k)    where k=1 (NARS convention)
//! ```
//!
//! Reference: `crate::hpc::nars::revise_belief`. Correctness gate: ULP ≤ 4
//! vs the scalar reference on 10K randomly-seeded revisions.
//!
//! # G4 — `fast_exp_x16` precision audit
//!
//! The existing `fast_exp_x16` ships 3% relative error — graphics-safe for
//! alpha-compositing but suspect for NARS confidence floor near boundaries.
//!
//! Audit deliverable: sweep 10K randomly-seeded NARS revisions comparing
//! `fast_exp_x16`-based conf path against an f64-reference. PASS bar:
//! `worst_conf_diff < 1e-3`. FAIL bar: open a `precise_exp_x16` follow-up
//! PR against PR-X10 A6 with the audit numbers.
//!
//! # Bundle consumed: B-Compose
//!
//! `revise_truth_f32x16` (activated by `splat4d-nars-compose` feature flag).
//! Under the default alpha path, `B-Compose` binds `hreduce_sum_f32x16`.
//!
//! # References
//!
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G3 NARS truth-revision"
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G4 fast_exp_x16 audit"
//! - `crate::hpc::nars::revise_belief` — scalar reference for G3 parity

//// TODO(A5): implement G3 SIMD NARS revision kernel + G4 audit.
//// G3 parity: within ULP ≤ 4 of `nars::revise_belief` scalar on 10K pairs.
//// G4 audit: worst_conf_diff < 1e-3 on 10K pairs vs f64 reference; if FAIL,
//// open precise_exp_x16 follow-up PR with audit numbers.
//// Ref: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` §§ G3, G4
