//! `crate::hpc::linalg::*` — the canonical middle layer between BLAS L1/L2/L3
//! and per-domain math (splat3d, cognitive cascade, jc pillars).
//!
//! # Stack position
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────┐
//!   │  per-domain math: splat3d · cognitive cascade · jc pillars  │
//!   │  (Spd3 eig/sandwich, SH, EWA-projection, polar, mat-exp…)   │
//!   ├─────────────────────────────────────────────────────────────┤
//!   │  crate::hpc::linalg  ←── YOU ARE HERE                       │
//!   │  MatN<N> carrier · Mat2/3/4 · Spd2/Spd3 SPD-cone            │
//!   ├─────────────────────────────────────────────────────────────┤
//!   │  crate::hpc::blas_level{1,2,3}  (dot, gemv, gemm, …)        │
//!   └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! `linalg` is the first stable, feature-gated surface that all
//! PR-X10 workers (A2-A12: Quat, inverse, eig\_sym, SVD, polar,
//! mat\_exp, SH, conv, batched, RoPE, attention, loss) build upon.
//!
//! # SPD math reference
//!
//! The closed-form 3×3 symmetric eigendecomposition used by [`Spd3`]
//! (and foreshadowed for the PR-X10 A4 eig\_sym worker) is:
//!
//! > Smith, J.O. (1961). "Eigenvalues of a symmetric 3×3 matrix."
//! > *Communications of the ACM* **4**(4):168.
//!
//! That algorithm is currently implemented in `crate::hpc::splat3d::spd3`
//! and will migrate here in PR-X10 A4 (eig\_sym). Until then, this
//! module's [`Spd3`] re-exports from `splat3d` so consumers see a stable
//! path today.
//!
//! # Out of scope (hard boundary)
//!
//! - **No SIMD primitives** — use `crate::simd::{F32x16, …}` directly.
//! - **No `#[target_feature]` annotations** — those live in `simd_avx512.rs`.
//! - **No distance metrics** — those live in `crate::hpc::distance`.

mod matrix;
pub use matrix::{Mat2, Mat3, Mat4, MatN, Spd2, Spd3};
