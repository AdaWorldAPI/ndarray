//! `crate::hpc::codec::*` — cognitive-cell codec (PR-X12).
//!
//! Encodes one or more `BlockedGrid<_, BR, BC>` blocks ("CTUs") into a
//! compact bytestream using the x265-shaped mode-class taxonomy
//! (skip / merge / delta / escape). Decoded streams reconstruct the
//! original cells modulo a configurable `epsilon_floor`.
//!
//! # Module layout (per PR-X12 worker decomposition)
//!
//! - [`ctu`] — A1: `Ctu` carrier + `CtuPartition` enum + quad-tree
//!   split / merge ops.
//! - [`mode`] — A2: bit-pack / unpack helpers for the on-wire 16-bit
//!   header + per-mode tail (Skip/Merge/Delta/Escape).
//! - [`predict`] — A3-intra: encoder-side mode-decision kernel that
//!   picks the cheapest `LeafCu` from a cell + NESW neighbours.
//! - `transform`, `quantize`, `rdo`, `ans`, `stream` — A4-A8, queued as
//!   follow-up sprints.
//!
//! # Feature gate
//!
//! The module is gated by `feature = "codec"` (declared in `Cargo.toml`)
//! so callers who don't ship the codec don't pay the LoC tax.
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x12-codec-x265-design.md` — master design doc.

pub mod ctu;
pub mod mode;
pub mod predict;

pub use ctu::{CellMode, MergeDir, MAX_QUAD_TREE_NODES, MAX_SPLIT_DEPTH};
pub use ctu::{Ctu, CtuArena, CtuPartition, LeafCu, MaxSplitDepthReached, MergeError, NodeIdx};
pub use mode::{
    pack_header, pack_leaf, pack_merge_dir, packed_byte_len, unpack_header, unpack_leaf, unpack_merge_dir, BASIN_NONE,
    MAX_BASIN_IDX,
};
pub use predict::{is_no_basin, predict_intra, IntraConfig, IntraContext};
