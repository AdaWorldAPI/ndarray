//! `crate::hpc::splat4d::*` — Gaussian splat L1–L4 cascade onto
//! `BlockedGrid` (PR-X4): the cognitive-spacetime evolution kernel.
//!
//! This module generalizes the bespoke fixed-16×16 tile binner in
//! [`crate::hpc::splat3d`] onto the const-generic
//! [`BlockedGrid`](crate::hpc::blocked_grid::BlockedGrid) tier hierarchy.
//! A [`Splat<D>`] carries an anisotropic Gaussian footprint plus a typed
//! cognitive [`SplatCell`]; [`TileBinning`] places each splat into the
//! L1–L4 tiles its 3σ footprint covers; [`TileBinning::compose_l1`] /
//! [`compose_cascade`](TileBinning::compose_cascade) blend the splats into
//! one grid or a four-level [`SplatPyramid`].
//!
//! # Why this is not "just a refactor"
//!
//! 3D Gaussian splatting and the cognitive-shader L1–L4 cascade share the
//! same arithmetic shape: project → bin → sort → composite, at four
//! multi-resolution tiers (the `(4×4)×(4×4)×(4×4)×(4×4)` pyramid). The
//! splat tile-binning pipeline, lifted onto `BlockedGrid`, *is* the
//! cognitive shader's spacetime evolution operator — L4 read as the time
//! axis, the cascade run every tick. See
//! `.claude/knowledge/pr-x4-design.md`.
//!
//! # Module layout (per PR-X4 worker decomposition)
//!
//! - [`splat`] — A3: `Splat<D>` / `SplatCell` / `SplatCovariance<D>`.
//! - [`tile`] — A1: `TileInstance` + `TileBinning<BR, BC>` + accessors.
//! - [`bin`] — A2: `from_projected_l1` / `from_projected_cascade`.
//! - [`compose`] — A4: `compose_l1` / `compose_cascade` + `SplatPyramid`.
//!
//! # Layering / no-float scope
//!
//! Pure layout + scheduling — zero per-arch intrinsics, zero
//! `#[target_feature]`, zero distance-aware API. SIMD lives inside the
//! consumer's `blend` closure (W1a). `f32` is used for splat *geometry*
//! (the graphics lane, sibling of `splat3d`); the cognitive payload
//! ([`SplatCell`]) is integer-only.
//!
//! # Out of scope (v1)
//!
//! - `splat3d` `From`/`Into` bridge + `raster::rasterize_frame` parity → X4.1.
//! - Higher-D inquiry space (`Splat<4>`/`Splat<N>`) → X4.1.
//! - NARS truth-revision blend kernel → W7 (drop-in `blend` replacement).
//! - Feature gate: `feature = "splat4d"`.

pub mod bin;
pub mod compose;
pub mod splat;
pub mod tile;

pub use compose::SplatPyramid;
pub use splat::{Splat, SplatCell, SplatCovariance, LT_LEN_MAX, MAX_SPLAT_DIM};
pub use tile::{TileBinning, TileInstance, N_TIERS, TIER_STRIDES};
