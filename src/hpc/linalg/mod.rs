//! Linear algebra routines — eigendecomposition and matrix utilities.
//!
//! # Contents
//!
//! - [`eig_sym`]: Symmetric eigendecomposition — closed-form (2×2, 3×3, 4×4),
//!   Jacobi rotations (5–64), implicit-shift QR (> 64).

#![allow(clippy::all, unused_imports, dead_code)]

pub mod eig_sym;
