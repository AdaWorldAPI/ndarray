//! `osm_pbf` — OpenStreetMap **PBF** (binary protobuf, `.osm.pbf`) ingestion.
//!
//! # No-XML Mandate
//!
//! This module is THE no-XML path. OSM XML (`.osm` text) is **forbidden in the
//! hotpath** and must never be introduced here. The OSM PBF format is the
//! canonical wire format (~10× smaller than XML, ~100× faster to parse).
//!
//! # Crate plan reference
//!
//! See `lance-graph/.claude/plans/cesium-osm-substrate-v1.md` for the full
//! cross-workspace integration plan. This file is **D-OSM-1** — the stub
//! mirror of `arcgis_pbf.rs` shape. **D-OSM-2** wires `osmpbf` v0.4 (b-r-u)
//! as a real dependency; this stub does NOT depend on `osmpbf` yet so the
//! `ndarray` workspace compile cost is unchanged until that gate fires.
//!
//! # PBF format — the wire shape
//!
//! Source: <https://wiki.openstreetmap.org/wiki/PBF_Format> + b-r-u/osmpbf
//! decoder reference. The binary layout (verbatim summary):
//!
//! ```text
//! // ── File structure ───────────────────────────────────────────────────────
//! // .osm.pbf = sequence of fileblocks
//! // fileblock = BlobHeader-length (4-byte BE u32)
//! //           + BlobHeader (protobuf — type + datasize)
//! //           + Blob       (protobuf — raw OR zlib_data; zlib is the common case)
//! //
//! // BlobHeader.type ∈ { "OSMHeader", "OSMData" }
//! //   - "OSMHeader" once at the top: bounding box, required_features
//! //   - "OSMData" N times: PrimitiveBlock containing nodes/ways/relations
//! //
//! // ── PrimitiveBlock ───────────────────────────────────────────────────────
//! // message PrimitiveBlock {
//! //     StringTable stringtable    = 1;   // string interning for keys/values/usernames
//! //     repeated PrimitiveGroup primitivegroup = 2;
//! //     int32 granularity          = 17;  // default 100 (nanodegrees scale)
//! //     int64 lat_offset           = 19;  // default 0
//! //     int64 lon_offset           = 20;  // default 0
//! //     int32 date_granularity     = 18;  // default 1000 (milliseconds)
//! // }
//! //
//! // message PrimitiveGroup {
//! //     repeated Node      nodes      = 1;  // rare — dense_nodes is canonical
//! //     DenseNodes         dense      = 2;  // delta-encoded packed node arrays
//! //     repeated Way       ways       = 3;
//! //     repeated Relation  relations  = 4;
//! //     repeated ChangeSet changesets = 5;  // not consumed here
//! // }
//! //
//! // ── DenseNodes (the canonical node carrier) ─────────────────────────────
//! // message DenseNodes {
//! //     repeated sint64 id        = 1; // delta encoded — running cumsum
//! //     repeated sint64 lat       = 8; // delta encoded — scaled to nanodegrees
//! //     repeated sint64 lon       = 9; // delta encoded — scaled to nanodegrees
//! //     repeated int32 keys_vals  = 10; // (key1, val1, key2, val2, …, 0, …)
//! //                                      // 0 sentinel separates nodes
//! //     DenseInfo denseinfo       = 5;  // version/timestamp/changeset/uid
//! // }
//! //
//! // ── Way ─────────────────────────────────────────────────────────────────
//! // message Way {
//! //     int64 id                  = 1;
//! //     repeated uint32 keys      = 2; // stringtable indices
//! //     repeated uint32 vals      = 3; // stringtable indices
//! //     Info info                 = 4;
//! //     repeated sint64 refs      = 8; // delta encoded node IDs
//! // }
//! //
//! // ── Relation ────────────────────────────────────────────────────────────
//! // message Relation {
//! //     int64 id                  = 1;
//! //     repeated uint32 keys      = 2;
//! //     repeated uint32 vals      = 3;
//! //     Info info                 = 4;
//! //     repeated int32 roles_sid  = 8; // stringtable indices (member roles)
//! //     repeated sint64 memids    = 9; // delta encoded member IDs
//! //     repeated MemberType types = 10;
//! // }
//! //
//! // enum MemberType { NODE = 0; WAY = 1; RELATION = 2; }
//! // ```
//!
//! # Coordinate decoding
//!
//! Real-world WGS84 coordinates are recovered as:
//!
//! ```text
//! lat_deg = 1e-9 * (lat_offset + (granularity * lat_quantised))
//! lon_deg = 1e-9 * (lon_offset + (granularity * lon_quantised))
//! ```
//!
//! With default `granularity = 100` and zero offsets this is `1e-7` degrees
//! per quantised unit (~1.1 cm at the equator) — well below sub-mm AR
//! overlay precision.
//!
//! # Q3 ruling — OSM-XYZ → Cesium-TMS Y-axis flip
//!
//! Per `lance-graph/.claude/plans/cesium-osm-substrate-v1.md §2 Q3`:
//! OSM slippy-map tile coordinates use **XYZ-order** (Y top-down, web
//! standard); Cesium's `implicit_tiling` uses **TMS-style** (Y bottom-up).
//! These are not the same key. The boundary helper [`xyz_to_tms_y`] performs
//! the one-line subtract at ingest, matching the runtime path to the
//! existing Cesium tile coordinate system. See [`SseFrustum`] in `sse.rs`
//! and `implicit_tiling.rs` for the runtime consumers of this convention.
//!
//! This is the **only** Y-axis conversion point in the OSM ingest pipeline —
//! per `I-LEGACY-API-FEATURE-GATED`, the wire-format-vs-runtime boundary is
//! converted at ingest; the runtime sees only one shape (TMS).
//!
//! # Stub-only file
//!
//! All carrier types are unit-struct placeholders. Real fields ship in
//! D-OSM-2 along with the `osmpbf` dependency. The Y-axis helper is the
//! only **live** function — it's small, self-contained, and pins the Q3
//! ruling at the boundary so D-OSM-2 cannot regress it accidentally.

// ─────────────────────────────────────────────────────────────────────────────
// Live: OSM-XYZ → Cesium-TMS Y-axis flip (Q3 boundary helper)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an OSM slippy-map tile Y coordinate (top-down, web XYZ convention)
/// to the Cesium TMS convention (bottom-up).
///
/// # Why this exists
///
/// OSM tile services (OpenStreetMap.org, Mapbox, Tileserver-GL, etc.) use the
/// web slippy-map convention: at zoom level `z`, the world is divided into
/// `2^z × 2^z` tiles, with `(x=0, y=0)` at the upper-left (north-west corner)
/// and `y` increasing southward.
///
/// Cesium's `implicit_tiling.rs` + `sse.rs` consume TMS (Tile Map Service)
/// convention: at zoom level `z`, `(x=0, y=0)` is at the lower-left (south-
/// west corner) and `y` increases northward.
///
/// For any quadkey-derived NiblePath (see `cesium-osm-substrate-v1.md §2 Q2`)
/// to prefix-match against the Cesium subtree path, the Y axis MUST be in TMS
/// convention. This helper is the conversion point.
///
/// # Formula
///
/// ```text
/// y_tms = (2^z - 1) - y_xyz
/// ```
///
/// # Examples
///
/// ```
/// use cesium::osm_pbf::xyz_to_tms_y;
///
/// // Zoom 0: only 1×1 tile; flip is no-op.
/// assert_eq!(xyz_to_tms_y(0, 0), 0);
///
/// // Zoom 1: 2×2 tiles; (x=0, y=0) XYZ → (x=0, y=1) TMS.
/// assert_eq!(xyz_to_tms_y(1, 0), 1);
/// assert_eq!(xyz_to_tms_y(1, 1), 0);
///
/// // Zoom 10: top-row OSM tile y=0 → TMS y=1023.
/// assert_eq!(xyz_to_tms_y(10, 0), 1023);
/// assert_eq!(xyz_to_tms_y(10, 1023), 0);
///
/// // Idempotent: applying twice returns the original.
/// for z in 0..=20 {
///     for y in [0u32, 1, (1 << z) / 2, (1 << z) - 1] {
///         if y >= 1 << z { continue; }
///         assert_eq!(xyz_to_tms_y(z, xyz_to_tms_y(z, y)), y);
///     }
/// }
/// ```
///
/// # Panics
///
/// Panics in debug builds if `y_xyz >= 2^zoom` (invalid tile coordinate for
/// the zoom level). Release builds wrap and return a meaningless result —
/// callers MUST validate the input before calling.
#[inline]
pub fn xyz_to_tms_y(zoom: u8, y_xyz: u32) -> u32 {
    debug_assert!(zoom <= 30, "zoom level must be ≤ 30 (u32 quad limit)");
    let max = (1u32 << zoom).saturating_sub(1);
    debug_assert!(y_xyz <= max, "y_xyz = {} exceeds zoom-level {} max of {}", y_xyz, zoom, max);
    max - y_xyz
}

// ─────────────────────────────────────────────────────────────────────────────
// Compilable stub types (no `osmpbf` dep yet — D-OSM-2 wires it)
// ─────────────────────────────────────────────────────────────────────────────

/// OSM **Node** decoded from a PBF `DenseNodes` slice.
///
/// Carrier shape (per `cesium-osm-substrate-v1.md §3 D-OSM-2`):
///
/// ```rust
/// // pub struct OsmNode {
/// //     /// Globally unique OSM node ID.
/// //     pub id: u64,
/// //     /// WGS84 latitude in degrees (decoded from quantised PBF).
/// //     pub lat: f64,
/// //     /// WGS84 longitude in degrees (decoded from quantised PBF).
/// //     pub lon: f64,
/// //     /// Cesium-TMS quadkey path (after `xyz_to_tms_y` boundary conversion).
/// //     /// Encoded as the NiblePath prefix `qk:<level>/<x>/<y_tms>`.
/// //     pub qk_tms_path: [u8; 24],
/// //     /// Tag list per Q1 (b) v1 fallback — Arrow `List<Struct<key, value>>`.
/// //     /// Migrates to Q1 (c) `Tag` interned identity in a follow-up sprint.
/// //     pub tags: Vec<(Box<str>, Box<str>)>,
/// // }
/// ```
pub struct OsmNode;

/// OSM **Way** — ordered sequence of node references + tags.
///
/// Carrier shape:
///
/// ```rust
/// // pub struct OsmWay {
/// //     pub id: u64,
/// //     /// Ordered list of `OsmNode.id` references defining the way's geometry.
/// //     pub ref_node_ids: Vec<u64>,
/// //     /// Cesium-TMS quadkey path computed from the way's centroid bbox.
/// //     pub qk_tms_path: [u8; 24],
/// //     /// Tag list per Q1 (b) v1 fallback.
/// //     pub tags: Vec<(Box<str>, Box<str>)>,
/// // }
/// ```
pub struct OsmWay;

/// OSM **Relation** — typed members (Node/Way/Relation) with roles + tags.
///
/// Carrier shape:
///
/// ```rust
/// // pub struct OsmRelation {
/// //     pub id: u64,
/// //     /// Member tuples: (member_id, role, member_kind).
/// //     pub members: Vec<OsmRelationMember>,
/// //     /// Tag list per Q1 (b) v1 fallback.
/// //     pub tags: Vec<(Box<str>, Box<str>)>,
/// // }
/// //
/// // pub struct OsmRelationMember {
/// //     pub ref_id: u64,
/// //     pub role: Box<str>,
/// //     pub kind: OsmRelationMemberKind,
/// // }
/// //
/// // #[repr(u8)]
/// // pub enum OsmRelationMemberKind { Node = 0, Way = 1, Relation = 2 }
/// ```
pub struct OsmRelation;

/// OSM **`PrimitiveBlock`** carrier — a single PBF fileblock decoded into
/// the three element collections plus the bounding box.
///
/// Carrier shape:
///
/// ```rust
/// // pub struct OsmPbfBlock {
/// //     pub nodes:     Vec<OsmNode>,
/// //     pub ways:      Vec<OsmWay>,
/// //     pub relations: Vec<OsmRelation>,
/// //     /// Bounding box in WGS84 degrees: [min_lat, min_lon, max_lat, max_lon].
/// //     pub bbox: [f64; 4],
/// //     /// PBF source granularity (default 100 nanodegrees).
/// //     pub granularity: i32,
/// //     /// PBF source lat_offset (default 0 nanodegrees).
/// //     pub lat_offset: i64,
/// //     /// PBF source lon_offset (default 0 nanodegrees).
/// //     pub lon_offset: i64,
/// // }
/// ```
pub struct OsmPbfBlock;

/// OSM **`OSMHeader`** carrier — the once-per-file metadata block.
///
/// Carrier shape:
///
/// ```rust
/// // pub struct OsmPbfHeader {
/// //     /// Optional file bounding box in WGS84 nanodegrees.
/// //     pub bbox: Option<[i64; 4]>,
/// //     /// `required_features` from the PBF header (e.g.
/// //     /// `["OsmSchema-V0.6", "DenseNodes"]`).
/// //     pub required_features: Vec<Box<str>>,
/// //     /// Generator program (e.g. `"osmconvert"`, `"Osmium"`).
/// //     pub generator: Option<Box<str>>,
/// // }
/// ```
pub struct OsmPbfHeader;

// ─────────────────────────────────────────────────────────────────────────────
// Stub decoder API — wired in D-OSM-2 (osmpbf consumer)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode a single PBF fileblock from raw bytes into an [`OsmPbfBlock`].
///
/// # Status
///
/// Stub — returns [`OsmPbfError::NotImplemented`] until D-OSM-2 wires the
/// `osmpbf` v0.4 dependency. The function signature is pinned now so the
/// rest of the pipeline (D-OSM-3 SPO lift, D-OSM-5 splat-fit-geo) can
/// reference the decoder shape without waiting for the implementation.
///
/// # Future implementation sketch
///
/// ```rust
/// // pub fn decode_pbf(bytes: &[u8]) -> Result<OsmPbfBlock, OsmPbfError> {
/// //     let reader = osmpbf::reader::BlobReader::new(bytes);
/// //     let mut nodes     = Vec::new();
/// //     let mut ways      = Vec::new();
/// //     let mut relations = Vec::new();
/// //
/// //     for blob in reader {
/// //         let blob = blob.map_err(OsmPbfError::Read)?;
/// //         if let Some(prim) = blob.to_primitiveblock().ok() {
/// //             for group in prim.groups() {
/// //                 // dense_nodes → OsmNode (delta-decode + apply Q3 Y-flip at
/// //                 //   any qk_tms_path computation site)
/// //                 // ways       → OsmWay (resolve stringtable for tags;
/// //                 //   tags as Q1 (b) Vec<(Box<str>, Box<str>)>)
/// //                 // relations  → OsmRelation (members + roles)
/// //             }
/// //         }
/// //     }
/// //
/// //     Ok(OsmPbfBlock { nodes, ways, relations, /* … */ })
/// // }
/// ```
pub fn decode_pbf(_bytes: &[u8]) -> Result<OsmPbfBlock, OsmPbfError> {
    Err(OsmPbfError::NotImplemented("decode_pbf: gated on D-OSM-2 (osmpbf v0.4 dep wiring)"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the OSM PBF decoder path.
#[derive(Debug)]
pub enum OsmPbfError {
    /// Function is a scaffold stub and has not been implemented yet.
    NotImplemented(&'static str),
    // Future variants (commented until D-OSM-2 wires the impl):
    //
    // /// PBF blob header read failure (truncated file, bad length).
    // Read(osmpbf::Error),
    // /// Required-feature in the OSM PBF header is not supported (e.g.
    // /// `HistoricalInformation`).
    // UnsupportedRequiredFeature(Box<str>),
    // /// Quantised coordinate decoding overflowed (corrupt or non-canonical PBF).
    // CoordinateOverflow,
    // /// Stringtable index out of range (corrupt PBF).
    // StringTableOutOfRange { index: u32, len: usize },
}

impl std::fmt::Display for OsmPbfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OsmPbfError::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl std::error::Error for OsmPbfError {}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers — commented stubs for D-OSM-2 implementer
// ═══════════════════════════════════════════════════════════════════════════

// fn decode_dense_nodes(dense: osmpbf::DenseNodeIter, granularity: i32,
//                       lat_offset: i64, lon_offset: i64,
//                       stringtable: &[String]) -> Vec<OsmNode> { ... }
//
// fn decode_way(way: osmpbf::Way, stringtable: &[String]) -> OsmWay { ... }
//
// fn decode_relation(rel: osmpbf::Relation, stringtable: &[String]) -> OsmRelation { ... }
//
// fn compute_qk_tms_path(lat: f64, lon: f64, zoom: u8) -> [u8; 24] {
//     // 1. WGS84 (lat, lon) → slippy-map tile (x, y_xyz) at `zoom` per the
//     //    web mercator formula.
//     // 2. Apply `xyz_to_tms_y(zoom, y_xyz)` — the Q3 boundary helper.
//     // 3. Encode quadkey path as `qk:<zoom>/<x>/<y_tms>` ASCII bytes,
//     //    padded to 24 bytes with NUL or 0x20.
// }
//
// NOTE(grounding): the `osmpbf` v0.4 (b-r-u) crate is the only production-
// grade Rust OSM PBF reader. See https://crates.io/crates/osmpbf and
// https://github.com/b-r-u/osmpbf — multi-year stable, lazy-decoded,
// parallelized. D-OSM-2 wires it as a default-on dep on the cesium crate
// behind a `osm` feature flag (to keep optional consumers slim).

// ─────────────────────────────────────────────────────────────────────────────
// Tests (stub — only the live Y-flip helper has assertions)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyz_to_tms_y_zoom_0_is_noop() {
        assert_eq!(xyz_to_tms_y(0, 0), 0);
    }

    #[test]
    fn xyz_to_tms_y_zoom_1_flips_pair() {
        assert_eq!(xyz_to_tms_y(1, 0), 1);
        assert_eq!(xyz_to_tms_y(1, 1), 0);
    }

    #[test]
    fn xyz_to_tms_y_zoom_10_flips_full_range() {
        assert_eq!(xyz_to_tms_y(10, 0), 1023);
        assert_eq!(xyz_to_tms_y(10, 1023), 0);
        assert_eq!(xyz_to_tms_y(10, 512), 511);
    }

    #[test]
    fn xyz_to_tms_y_idempotent_under_double_application() {
        for zoom in 0..=20u8 {
            let max = if zoom == 0 { 0 } else { (1u32 << zoom) - 1 };
            for y in [0, 1, max / 2, max] {
                if y > max {
                    continue;
                }
                assert_eq!(xyz_to_tms_y(zoom, xyz_to_tms_y(zoom, y)), y, "zoom={zoom} y={y} not idempotent");
            }
        }
    }

    #[test]
    fn decode_pbf_stub_returns_not_implemented() {
        let result = decode_pbf(&[]);
        assert!(matches!(result, Err(OsmPbfError::NotImplemented(_))));
    }
}
