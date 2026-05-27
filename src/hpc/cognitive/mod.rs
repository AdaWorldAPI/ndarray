//! `crate::hpc::cognitive::*` — lazy basin-relative grid storage (PR-X9).
//!
//! Where [`crate::hpc::blocked_grid`] stores a grid densely, this module
//! stores each `CausalEdge64` cell as `(basin_idx, mode)` *relative to the
//! shared [`CamCodebook`]* hydrated from the embedded OGIT Cognitive TTL by
//! [`crate::hpc::ogit_bridge`]. Coherent cognitive state is dominated by
//! cells that equal their basin (`Skip`), so the basin-relative form is far
//! smaller than dense `u64` storage while staying **bit-exact**.
//!
//! This is the storage analogue of an x265 coding-tree-unit: the
//! skip/merge/delta/escape [`CellMode`](crate::hpc::codec::CellMode)
//! taxonomy applied to a semantic codebook substrate instead of a pixel
//! substrate.
//!
//! # Module layout (PR-X9 worker decomposition)
//!
//! - [`storage`] — A1: the `GridStorage<BR, BC>` trait (dense + lazy).
//! - [`sparse`] — A3: `SparseGrid` columnar store + 2-bit mode packing.
//! - [`lazy`] — A4/A5: `LazyBlockedGrid` + lossless XOR-relative
//!   `encode_from_dense` / `decode_to_dense`.
//!
//! The codebook itself ([`CamCodebook`], [`BasinAtom`], `CognitiveBridge`)
//! is shipped by [`crate::hpc::ogit_bridge`] (PR-X13) and consumed here —
//! never redefined.
//!
//! # Scope (v1)
//!
//! - **Lossless XOR-relative** encoding (no quantization loss); the codec's
//!   lossy arithmetic `rdo_select` is not used (different distortion model).
//! - **u64-concrete** `GridStorage` (cognitive cells are `CausalEdge64`).
//! - Out of scope: NARS truth-revision blend (W7), online codebook
//!   learning, Lance persistence, generic-`T` storage, per-tier covariance.
//!
//! # Layering
//!
//! Pure storage + encoding: no `#[target_feature]`, no per-arch intrinsics,
//! no distance API. Basin matching is codebook XOR-min (mirrors
//! `ogit_bridge::CognitiveBridge::nearest_basin`), not a distance trait.
//!
//! Gated by `feature = "cognitive"` (pulls in `codec` + `ogit_bridge`).

pub mod lazy;
pub mod sparse;
pub mod storage;

pub use lazy::LazyBlockedGrid;
pub use sparse::{ModeBits, SparseGrid};
pub use storage::GridStorage;

// Re-export the codebook types consumed here, so downstream callers get
// the whole lazy-storage surface from one path.
pub use crate::hpc::ogit_bridge::{BasinAtom, CamCodebook, CognitiveBridge};
