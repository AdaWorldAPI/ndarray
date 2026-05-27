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
//!   picks the cheapest `LeafCu` from a cell + NESW neighbours (hard
//!   decision tree).
//! - [`rdo`] — A6: λ-rate-distortion mode selection — the soft,
//!   cost-weighted generalization of [`predict`] (integer fixed-point λ).
//! - [`ans`] — A7: rANS entropy coder over the 4-symbol mode alphabet.
//! - `transform` (A4, deferred to v2 per design Q2 — 1-D DCT doesn't
//!   help an 8-bit scalar residual) and `stream` (A8, byte-stream
//!   framing over [`ans`]) remain as follow-ups.
//!
//! # Feature gate
//!
//! The module is gated by `feature = "codec"` (declared in `Cargo.toml`)
//! so callers who don't ship the codec don't pay the LoC tax.
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x12-codec-x265-design.md` — master design doc.

pub mod ans;
pub mod ctu;
pub mod mode;
pub mod predict;
pub mod rdo;

pub use ans::{decode_modes, encode_modes, RansFreqTable, RANS_BYTE_L, RANS_M, RANS_SCALE_BITS};
pub use ctu::{CellMode, MergeDir, MAX_QUAD_TREE_NODES, MAX_SPLIT_DEPTH};
pub use ctu::{Ctu, CtuArena, CtuPartition, LeafCu, MaxSplitDepthReached, MergeError, NodeIdx};
pub use mode::{
    pack_header, pack_leaf, pack_merge_dir, packed_byte_len, unpack_header, unpack_leaf, unpack_merge_dir,
    MAX_BASIN_IDX,
};
pub use predict::{predict_intra, IntraConfig, IntraContext};
pub use rdo::{rdo_select, RdoChoice, RdoConfig, RdoContext};
