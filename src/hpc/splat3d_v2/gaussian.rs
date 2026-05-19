//! `GaussianBatch` v2 — SoA storage for 3D Gaussian Splatting cascade.
//!
//! Placeholder module re-exporting v1 symbols.  The v2 refactor (A3/A4
//! workers) will extend `GaussianBatch` with `SplatCell<D>` for the
//! INT4×32 packed dot (G2) and inquiry-direction SH (G1).
//!
//! Per A1 brief § "File moves": `splat3d/gaussian.rs` → `splat3d_v2/gaussian.rs`.
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - G2 INT4×32 packed dot: A4 worker (B-Pack-Dot bundle)
//! - `SplatCell<D>` forward-compat note: per master prompt § "Carry-overs"
//!   (PR-X7 typed cell-DSL consumes `SplatCell<D>`; no breaking changes after W5)

//// TODO(A3): add SplatCell<D> with INT4 thinking-style and qualia fields
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§G2 INT4×N packed dot
//// TODO(A4): wire B-Pack-Dot (pack_int4x32 ∘ dot_i4x32_to_i32 ∘ dequant_f32)
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§SIMD bundles table

pub use crate::hpc::splat3d::{
    Gaussian3D, GaussianBatch,
    SH_COEFFS_PER_CHANNEL, SH_COEFFS_PER_GAUSSIAN, SH_DEGREE,
};
