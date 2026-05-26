//! `arcgis_pbf` — ArcGIS FeatureService **PBF** (binary protobuf, `f=pbf`) ingestion.
//!
//! # No-JSON Mandate
//! This module is THE no-JSON path.  JSON (`f=json`) is **forbidden in the hotpath** and
//! must never be introduced here.  The ArcGIS FeatureService binary response (`f=pbf`) uses
//! Protocol Buffer encoding defined by Esri's open `arcgis-pbf` repository
//! (github.com/Esri/arcgis-pbf) and decodes ≥ 4× faster than the equivalent JSON at wire
//! size ≈ 30-50% smaller.  The cold import boundary (URL construction, HTTP) may tolerate a
//! small JSON config blob; the frame-decode loop must not.
//!
//! # Protobuf Schema — `FeatureCollectionPBuffer`
//! Source: `github.com/Esri/arcgis-pbf/proto/FeatureCollection/FeatureCollection.proto`
//! (confirmed 2026-05-26 via raw fetch).  All field numbers below are verbatim from that file.
//!
//! ```text
//! // package esriPBuffer;
//!
//! // ── Enumerations ────────────────────────────────────────────────────────────
//! enum GeometryType {
//!     esriGeometryTypePoint       = 0;
//!     esriGeometryTypeMultipoint  = 1;
//!     esriGeometryTypePolyline    = 2;
//!     esriGeometryTypePolygon     = 3;
//!     esriGeometryTypeMultipatch  = 4;
//!     esriGeometryTypeEnvelope    = 5;
//!     esriGeometryTypeNone        = 127;
//! }
//!
//! enum QuantizeOriginPostion {   // sic: "Postion" in the real proto
//!     upperLeft = 0;
//!     lowerLeft = 1;
//! }
//!
//! // ── Spatial reference ───────────────────────────────────────────────────────
//! message SpatialReference {
//!     uint32 wkid          = 1;   // well-known ID (horizontal)
//!     uint32 lastestWkid   = 2;   // sic: "lastest" in the real proto (latest effective)
//!     uint32 vcsWkid       = 3;   // vertical coordinate system WKID
//!     uint32 latestVcsWkid = 4;
//!     string wkt           = 5;
//!     string wkt2          = 6;
//! }
//!
//! // ── Quantised-coordinate transform ──────────────────────────────────────────
//! // Applied to Geometry.coords — multiply + translate to recover real-world coords.
//! message Scale     { double xScale = 1; double yScale = 2; double mScale = 3; double zScale = 4; }
//! message Translate { double xTranslate = 1; double yTranslate = 2; double mTranslate = 3; double zTranslate = 4; }
//! message Transform {
//!     QuantizeOriginPostion quantizeOriginPostion = 1;
//!     Scale     scale     = 2;
//!     Translate translate = 3;
//! }
//!
//! // ── Field descriptors ───────────────────────────────────────────────────────
//! message Field {
//!     string    name         = 1;
//!     FieldType fieldType    = 2;
//!     string    alias        = 3;
//!     SQLType   sqlType      = 4;
//!     string    domain       = 5;
//!     string    defaultValue = 6;
//! }
//!
//! // ── Geometry ────────────────────────────────────────────────────────────────
//! // `coords` is a delta-encoded, zigzag-signed sequence of quantised integers.
//! // `lengths` gives the ring/part lengths for multi-part geometries.
//! message Geometry {
//!     GeometryType   geometryType = 1;
//!     repeated uint32 lengths     = 2; // packed
//!     repeated sint64 coords      = 3; // packed — delta-zigzag quantised
//!     repeated uint64 ids         = 4; // packed — OIDs (multipoint)
//! }
//!
//! // ── Attribute value (oneof) ──────────────────────────────────────────────────
//! message Value {
//!     oneof value_type {
//!         string string_value  = 1;
//!         float  float_value   = 2;
//!         double double_value  = 3;
//!         sint32 sint_value    = 4;
//!         uint32 uint_value    = 5;
//!         int64  int64_value   = 6;
//!         uint64 uint64_value  = 7;
//!         sint64 sint64_value  = 8;
//!         bool   bool_value    = 9;
//!         bool   null_value    = 10; // true = NULL attribute
//!     }
//!     uint32 index = 11; // index into FeatureResult.fields for this value's column
//! }
//!
//! // ── Feature ─────────────────────────────────────────────────────────────────
//! message Feature {
//!     repeated Value attributes = 1;
//!     oneof compressed_geometry {
//!         Geometry       geometry       = 2;
//!         esriShapeBuffer shapeBuffer   = 3;
//!         CurveGeometry  curveGeometry  = 7;
//!     }
//!     Geometry centroid            = 4;
//!     repeated Geometry aggregateGeometries = 5;
//!     Envelope envelope            = 6;
//! }
//!
//! // ── Top-level result wrapper ─────────────────────────────────────────────────
//! message FeatureResult {
//!     string objectIdFieldName  = 1;
//!     UniqueIdField uniqueIdField = 2;
//!     string globalIdFieldName  = 3;
//!     string geohashFieldName   = 4;
//!     GeometryProperties geometryProperties = 5;
//!     ServerGens serverGens     = 6;
//!     GeometryType geometryType = 7;
//!     SpatialReference spatialReference = 8;
//!     bool exceededTransferLimit = 9;
//!     bool hasZ                 = 10;
//!     bool hasM                 = 11;
//!     Transform transform       = 12;
//!     repeated Field  fields    = 13;
//!     repeated Value  values    = 14; // column-ordered attribute pool
//!     repeated Feature features = 15;
//!     repeated GeometryField geometryFields = 16;
//! }
//!
//! message QueryResult {
//!     oneof results {
//!         FeatureResult    featureResult    = 1;
//!         CountResult      countResult      = 2;
//!         ObjectIdsResult  idsResult        = 3;
//!         ExtentCountResult extentCountResult = 4;
//!     }
//! }
//!
//! message FeatureCollectionPBuffer {
//!     string      version     = 1; // e.g. "2.1"
//!     QueryResult queryResult = 2;
//! }
//! ```
//!
//! # Coordinate dequantisation
//! `Geometry.coords` is a **delta-encoded, zigzag-signed** sint64 packed field.
//! The cumulative sum of decoded zigzag values gives quantised integers Q.
//! Real-world coordinates are recovered by:
//! ```text
//! x_real = (Q_x * transform.scale.xScale) + transform.translate.xTranslate
//! y_real = (Q_y * transform.scale.yScale) + transform.translate.yTranslate
//! z_real = (Q_z * transform.scale.zScale) + transform.translate.zTranslate  // if hasZ
//! ```
//! Origin convention (`QuantizeOriginPostion`): upperLeft ⟹ Y increases downward;
//! lowerLeft ⟹ standard math convention.
//!
//! # CAM SoA target
//! After decode the intermediate structs (`PbfFeatureCollection`, `PbfFeature`, `PbfGeometry`)
//! must be consumed by `to_cam_soa` (agent C) to produce `GaussianBatch` + `cam_pq`.
//! These structs do NOT survive past the import boundary.
//!
//! # Request construction (cold boundary — JSON allowed here, not in decode loop)
//! ```text
//! GET /FeatureServer/0/query
//!   ?f=pbf
//!   &outFields=*
//!   &returnGeometry=true
//!   &returnZ=true
//!   &quantizationParameters={"tolerance":0.001,"extent":{...},"mode":"view"}
//! ```
//! `quantizationParameters` is JSON but lives in the URL query string, not in the
//! binary response payload.  It is constructed once at cold import, not per frame.
//!
//! # Grounding
//! - Proto field numbers: `github.com/Esri/arcgis-pbf` (fetched 2026-05-26, confirmed)
//! - R-ArcGIS/arcpbf — R reference implementation (dequant, delta decode)
//! - dfridkin/arcGIS-Rust-runtime — Rust approach patterns
//! - EsriDevEvents/locup — Location Platform Rust usage
//! - AdaWorldAPI/arcgisutils — UNVERIFIED (private / unreachable); scaffold only

// ══════════════════════════════════════════════════════════════════════════════
// Intermediate structs — compile-only (no live impl until Opus review)
// ══════════════════════════════════════════════════════════════════════════════

/// Decoded `SpatialReference` from the PBF `FeatureResult` header.
/// Passed directly to `esri_crs::EsriCrs` for reprojection before CAM SoA hand-off.
///
/// ```rust
/// // pub struct PbfSpatialReference {
/// //     /// Field 1 — original WKID (horizontal).  May be legacy (e.g. 102100).
/// //     pub wkid: u32,
/// //     /// Field 2 — latest WKID (sic "lastest" in proto); use when non-zero.
/// //     pub latest_wkid: u32,
/// //     /// Field 3 — vertical CRS WKID (0 = none / assumed ellipsoidal).
/// //     pub vcs_wkid: u32,
/// //     /// Field 4 — latest vertical WKID.
/// //     pub latest_vcs_wkid: u32,
/// //     /// Field 5/6 — WKT fallback when WKID is 0; expensive, avoid in hotpath.
/// //     pub wkt: Option<String>,
/// // }
/// ```
pub struct PbfSpatialReference;

/// Quantised-coordinate transform decoded from `FeatureResult.transform` (field 12).
/// Must be applied to every `Geometry.coords` delta-sum before the result is meaningful.
///
/// ```rust
/// // pub struct PbfTransform {
/// //     /// upperLeft (0) or lowerLeft (1) — affects Y-axis direction.
/// //     pub origin: u8,
/// //     /// Multipliers: xScale, yScale, zScale (mScale unused for splats).
/// //     pub scale: [f64; 4],     // [x, y, m, z]
/// //     /// Additive offsets after scaling.
/// //     pub translate: [f64; 4], // [x, y, m, z]
/// // }
/// //
/// // impl PbfTransform {
/// //     /// Apply to a zigzag-decoded, delta-accumulated quantised coordinate triple.
/// //     /// `q` is [Q_x, Q_y, Q_z] after cumulative sum.
/// //     #[inline]
/// //     pub fn dequantise(&self, q: [i64; 3]) -> [f64; 3] {
/// //         [
/// //             q[0] as f64 * self.scale[0] + self.translate[0],
/// //             q[1] as f64 * self.scale[1] + self.translate[1],
/// //             q[2] as f64 * self.scale[3] + self.translate[3], // z uses index 3
/// //         ]
/// //     }
/// // }
/// ```
pub struct PbfTransform;

/// A single decoded feature geometry — positions only; attributes are stripped
/// at the import boundary (we only need xyz for CAM SoA splat seeding).
///
/// `coords_raw` contains the raw zigzag-decoded delta values **before** dequantisation
/// so that `PbfTransform::dequantise` can be applied in one pass without per-coord
/// allocation.
///
/// ```rust
/// // pub struct PbfGeometry {
/// //     /// Geometry type — we only process Point / Multipoint / Multipatch for splats.
/// //     pub geometry_type: u32,       // raw proto enum value
/// //     /// Delta-accumulated zigzag sint64 coords, stride = 2 (xy) or 3 (xyz, hasZ).
/// //     pub coords_raw: Vec<i64>,
/// //     /// Part/ring lengths for multi-part geometries.
/// //     pub lengths: Vec<u32>,
/// //     /// True when Z is present (from `FeatureResult.hasZ` field 10).
/// //     pub has_z: bool,
/// // }
/// ```
pub struct PbfGeometry;

/// One decoded feature (geometry + object ID only — other attributes dropped).
///
/// ```rust
/// // pub struct PbfFeature {
/// //     pub object_id: u64,
/// //     pub geometry: Option<PbfGeometry>,
/// //     /// Centroid geometry (field 4), present for polygon/polyline features.
/// //     pub centroid: Option<PbfGeometry>,
/// // }
/// ```
pub struct PbfFeature;

/// Top-level decoded PBF response — the unit consumed by `to_cam_soa`.
/// This struct must NOT outlive the import boundary; it is transmuted into CAM SoA
/// by agent C's `to_cam_soa::pbf_to_gaussian_batch`.
///
/// ```rust
/// // pub struct PbfFeatureCollection {
/// //     /// Protocol version string from `FeatureCollectionPBuffer.version` (field 1).
/// //     pub version: String,
/// //     /// Spatial reference for reprojection via `esri_crs`.
/// //     pub spatial_ref: PbfSpatialReference,
/// //     /// Coordinate dequantisation parameters.
/// //     pub transform: PbfTransform,
/// //     /// Geometry type from `FeatureResult.geometryType` (field 7).
/// //     pub geometry_type: u32,
/// //     /// Decoded features — geometry only, attributes stripped.
/// //     pub features: Vec<PbfFeature>,
/// //     /// True if the server truncated the result set.
/// //     pub exceeded_transfer_limit: bool,
/// // }
/// ```
pub struct PbfFeatureCollection;

// ══════════════════════════════════════════════════════════════════════════════
// Decode pipeline — commented until protobuf dep is wired
// ══════════════════════════════════════════════════════════════════════════════

/// Decode a raw `f=pbf` HTTP response body into a [`PbfFeatureCollection`].
///
/// # No-JSON guarantee
/// This function operates entirely on binary Protocol Buffer bytes.  No JSON
/// parsing occurs here or in any function it calls.  JSON is only permitted in
/// the URL query string at request construction time (cold boundary).
///
/// # Wire format
/// The bytes are a length-delimited protobuf encoding of `FeatureCollectionPBuffer`
/// (package `esriPBuffer`).  Decode with `prost` or `quick-protobuf` — either is
/// fine; the generated code is a compile-time artefact, not a runtime allocation.
///
/// # Decode steps
/// 1. Parse outer `FeatureCollectionPBuffer` → extract `version` + `QueryResult`.
/// 2. Assert `QueryResult` is `featureResult` variant (field 1); others (count/ids/extent)
///    carry no geometry and should return `Err(PbfError::NoGeometry)`.
/// 3. Read `FeatureResult` fields:
///    - field 8 → `PbfSpatialReference`
///    - field 10 (`hasZ`) + field 11 (`hasM`) → stride hint
///    - field 12 → `PbfTransform` (scale + translate + origin)
///    - field 9 (`exceededTransferLimit`)
/// 4. For each `Feature` (field 15): decode `Geometry` (field 2) or `centroid` (field 4).
///    - Delta-accumulate `Geometry.coords` (packed sint64, zigzag).
///    - Apply `PbfTransform::dequantise` → `[f64; 3]` per vertex.
///    - Collect into `PbfGeometry.coords_raw` (pre-dequant) to defer allocation.
/// 5. Drop all `Value` attribute fields (field 14 / feature field 1) — not needed for splats.
///
/// ```rust
/// // pub fn decode_pbf(bytes: &[u8]) -> Result<PbfFeatureCollection, PbfError> {
/// //     use prost::Message;
/// //     // Step 1 — outer wrapper
/// //     let pbf = esri_pb::FeatureCollectionPBuffer::decode(bytes)
/// //         .map_err(PbfError::Decode)?;
/// //     // Step 2 — must be featureResult
/// //     let fr = pbf.query_result
/// //         .and_then(|qr| qr.results)
/// //         .and_then(|r| if let Results::FeatureResult(fr) = r { Some(fr) } else { None })
/// //         .ok_or(PbfError::NoGeometry)?;
/// //     // Step 3 — header fields
/// //     let spatial_ref = decode_spatial_ref(fr.spatial_reference);
/// //     let transform   = decode_transform(fr.transform);
/// //     let has_z       = fr.has_z;
/// //     let has_m       = fr.has_m;
/// //     // Step 4 — features
/// //     let features: Vec<PbfFeature> = fr.features.into_iter()
/// //         .map(|f| decode_feature(f, has_z, has_m))
/// //         .collect::<Result<_, _>>()?;
/// //     Ok(PbfFeatureCollection {
/// //         version: pbf.version,
/// //         spatial_ref,
/// //         transform,
/// //         geometry_type: fr.geometry_type as u32,
/// //         features,
/// //         exceeded_transfer_limit: fr.exceeded_transfer_limit,
/// //     })
/// // }
/// ```
pub fn decode_pbf(_bytes: &[u8]) -> Result<PbfFeatureCollection, PbfError> {
    Err(PbfError::NotImplemented("scaffold only — not yet implemented"))
}

/// Zigzag-decode + delta-accumulate a packed sint64 coordinate sequence from a
/// `Geometry.coords` field.
///
/// The protobuf packed sint64 encoding stores each value as its zigzag-mapped uint64:
/// `zigzag(n) = (n << 1) ^ (n >> 63)`.  After decoding each zigzag value the running
/// cumulative sum gives the actual quantised integer.
///
/// ```rust
/// // fn decode_delta_coords(raw_zigzag: &[u64]) -> Vec<i64> {
/// //     let mut acc = 0i64;
/// //     raw_zigzag.iter().map(|&z| {
/// //         let delta = ((z >> 1) as i64) ^ -((z & 1) as i64); // zigzag decode
/// //         acc = acc.wrapping_add(delta);
/// //         acc
/// //     }).collect()
/// // }
/// ```
///
/// # Stride
/// After delta accumulation the flat `Vec<i64>` is strided:
/// - `hasZ = false`: stride 2, layout `[x0,y0, x1,y1, …]`
/// - `hasZ = true`:  stride 3, layout `[x0,y0,z0, x1,y1,z1, …]`
///
/// Dequantise with [`PbfTransform`] before passing to `to_cam_soa`.
pub(crate) fn decode_delta_coords(_raw: &[u64], _has_z: bool) -> Vec<i64> {
    unimplemented!("scaffold only")
}

// ══════════════════════════════════════════════════════════════════════════════
// Error type
// ══════════════════════════════════════════════════════════════════════════════

/// Errors produced by the PBF decode pipeline.
///
/// ```rust
/// // #[derive(Debug)]
/// // pub enum PbfError {
/// //     /// Protobuf wire-format decode failure (wraps prost::DecodeError).
/// //     Decode(prost::DecodeError),
/// //     /// The QueryResult did not contain a FeatureResult with geometry.
/// //     NoGeometry,
/// //     /// coord count is not divisible by stride (corrupted payload).
/// //     CoordStrideMismatch { count: usize, stride: usize },
/// //     /// Transform scale fields are zero — division by zero guard.
/// //     DegenerateTransform,
/// // }
/// ```
#[derive(Debug)]
pub enum PbfError {
    /// Function is a scaffold stub and has not been implemented yet.
    NotImplemented(&'static str),
}

impl std::fmt::Display for PbfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PbfError::NotImplemented(msg) => write!(f, "not implemented: {}", msg),
        }
    }
}

impl std::error::Error for PbfError {}

// ══════════════════════════════════════════════════════════════════════════════
// Internal helpers — commented stubs
// ══════════════════════════════════════════════════════════════════════════════

// fn decode_spatial_ref(sr: Option<esri_pb::SpatialReference>) -> PbfSpatialReference { ... }
// fn decode_transform(t: Option<esri_pb::Transform>) -> PbfTransform { ... }
// fn decode_feature(f: esri_pb::Feature, has_z: bool, has_m: bool)
//     -> Result<PbfFeature, PbfError> { ... }
//
// NOTE(grounding): `esri_pb` refers to the prost-generated module from
// `crates/cesium/proto/FeatureCollection.proto` (copy of Esri/arcgis-pbf).
// Do NOT hand-write the proto structs — generate them with `prost-build` in build.rs.
// The build.rs compileProtos call is also commented until dep is wired:
//
// fn main() {
//     prost_build::compile_protos(
//         &["proto/FeatureCollection/FeatureCollection.proto"],
//         &["proto/"],
//     ).unwrap();
// }
