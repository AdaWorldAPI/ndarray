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
//!   │  · Quat algebra (PR-X10 A2)                                 │
//!   │  · eig_sym Smith+Ferrari+Jacobi+QR (PR-X10 A4)              │
//!   ├─────────────────────────────────────────────────────────────┤
//!   │  crate::hpc::blas_level{1,2,3}  (dot, gemv, gemm, …)        │
//!   └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! `linalg` is the first stable, feature-gated surface that all
//! PR-X10 workers (A2-A12: Quat, inverse, eig_sym, SVD, polar,
//! mat_exp, SH, conv, batched, RoPE, attention, loss) build upon.
//!
//! # SPD math reference
//!
//! The closed-form 3×3 symmetric eigendecomposition used by [`Spd3`]
//! and by [`eig_sym::eig_sym_3`] (PR-X10 A4) is:
//!
//! > Smith, J.O. (1961). "Eigenvalues of a symmetric 3×3 matrix."
//! > *Communications of the ACM* **4**(4):168.
//!
//! # Routing guide (per joint savant ruling #10)
//!
//! For symmetric eigendecomposition:
//! - **N ∈ {2, 3, 4}**: use closed-form fast paths — `eig_sym::{eig_sym_2, eig_sym_3, eig_sym_4}`
//! - **N ≥ 5**: use `eig_sym::eig_sym_n::<N>` (dispatches Jacobi for [5,64], QR for >64)
//! - `eig_sym_n::<3>` is the correctness reference; do NOT use on hot paths
//!
//! # Out of scope (hard boundary)
//!
//! - **No SIMD primitives** — use `crate::simd::{F32x16, …}` directly.
//! - **No `#[target_feature]` annotations** — those live in `simd_avx512.rs`.
//! - **No distance metrics** — those live in `crate::hpc::distance`.

mod matrix;
pub use matrix::{Mat2, Mat3, Mat4, MatN, Spd2, Spd3};

pub mod quat;
pub use quat::{quat_mul_x16, Quat};

pub mod eig_sym;

pub mod inverse;
pub use inverse::{invert_affine_4x4, invert_mat3, invert_mat4, invert_mat_n};
