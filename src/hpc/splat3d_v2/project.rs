//! EWA projection kernel v2 — world-space 3D gaussians → screen-space 2D conics.
//!
//! Placeholder module re-exporting v1 symbols.  The v2 projection kernel
//! will integrate `CascadeAddr` (A2 worker) for multi-tier addressing and
//! will consume the B-Gather-FMA bundle (`gather_idx_f32x16 ∘ fmadd_f32x16`)
//! for the 16-neighbour evidence-aggregation step (A2/A3 deliverable).
//!
//! Per A1 brief § "File moves": `splat3d/project.rs` → `splat3d_v2/project.rs`.
//!
//! Cross-ref:
//! - A1 brief: `.claude/knowledge/pr-x4-planning/01-a1-tileinstance-v2-brief.md`
//! - B-Gather-FMA: A2/A3 bundle (gather_idx_f32x16 ∘ fmadd_f32x16)
//! - `CascadeAddr`: A2 worker, gated on PR-X10 A12b L4 Hilbert fix

//// TODO(A2): extend project_batch_v2 to emit CascadeAddr per gaussian
//// — gated on linalg::hilbert::hilbert3d_encode L4 fix (PR-X10 A12b)
//// TODO(A3): wire B-Gather-FMA for 16-neighbour evidence aggregation
//// — see hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md:§SIMD bundles table

pub use crate::hpc::splat3d::{project_batch, Camera, ProjectedBatch};
