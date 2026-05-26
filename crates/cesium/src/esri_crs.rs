//! `esri_crs` — `ESRI_crs` extension: WKID (horizontal + vertical) → WGS84 / local-origin.
//!
//! # What is ESRI_crs?
//! `ESRI_crs` is an Esri-proprietary 3D Tiles extension that attaches a full
//! coordinate reference system definition to a tileset or glTF asset via Well-Known IDs
//! (WKIDs).  It is required for Gaussian splat layers produced by ArcGIS Pro / ArcGIS
//! Online — the tileset will not reproject correctly without it.  The extension appears
//! in `tileset.json` under `extensions.ESRI_crs` and/or in per-tile glTF under
//! `extensions.ESRI_crs` at the asset level.
//!
//! # JSON schema (UNVERIFIED — grounded against I3S spec + ArcGIS Pro docs; no
//! dedicated ESRI_crs glTF schema URL was publicly reachable as of 2026-05-26)
//! ```json
//! {
//!   "extensions": {
//!     "ESRI_crs": {
//!       "wkid":          103142,   // original horizontal WKID
//!       "latestWkid":    6565,     // current WKID for same CRS (may differ from wkid)
//!       "vcsWkid":       105703,   // vertical CRS WKID (0 or absent = ellipsoidal)
//!       "latestVcsWkid": 6360      // current vertical WKID
//!     }
//!   }
//! }
//! ```
//! Sources: Esri I3S-spec `spatialReference.cmn.md` (confirmed); ArcGIS Pro
//! "Work with Gaussian splat layers" page (confirmed ESRI_crs is required);
//! Web-map specification `spatialReference` object (confirmed wkid/latestWkid/vcsWkid).
//!
//! # WKID semantics
//! - `wkid` — original ID assigned when the CRS was created.  May be a deprecated alias
//!   (e.g. 102100 = old Web Mercator; latestWkid = 3857).  Always prefer `latestWkid`
//!   when non-zero.
//! - `latestWkid` — current EPSG/Esri code.  Use for projection library lookups.
//! - `vcsWkid` — vertical CRS (0 = not specified / use ellipsoidal height).
//! - `latestVcsWkid` — current vertical WKID.
//!
//! # Common WKIDs for geospatial splats
//! | WKID  | latestWkid | Meaning                          |
//! |-------|------------|----------------------------------|
//! | 4326  | 4326       | WGS84 geographic (lat/lon/ellht) |
//! | 102100| 3857       | Web Mercator (metres, XY only)   |
//! | 4978  | 4978       | WGS84 geocentric ECEF            |
//! | 4979  | 4979       | WGS84 geographic 3D              |
//!
//! # Reprojection pipeline
//! This module only carries the parsed CRS + coordinate-transform math stubs.
//! Full PROJ/GDAL integration is a live-impl decision deferred to Opus review.
//! The intermediate output `EsriCrsReprojected` feeds `to_cam_soa` (agent C).
//!
//! ## WGS84 target
//! All coordinates must ultimately reach WGS84 geographic (EPSG:4979) or local-ENU
//! (east-north-up) before CAM SoA hand-off.  Steps:
//! 1. Parse `ESRI_crs` → `EsriCrs`.
//! 2. If `wkid == 4326` or `latestWkid == 4326` (or 4979): identity pass.
//! 3. If `wkid == 102100 / 3857` (Web Mercator):
//!    apply inverse Mercator projection → WGS84 lat/lon.
//! 4. Otherwise: invoke PROJ pipeline `EPSG:{latestWkid} → EPSG:4979`.
//!    (UNVERIFIED: exact PROJ string; mark as PROJ-required at runtime)
//! 5. Apply vertical transform if `vcsWkid != 0`:
//!    `EPSG:{latestVcsWkid} → ellipsoidal height`.
//!    (UNVERIFIED: geoid separation lookup not implemented in scaffold)
//! 6. Optionally rebase to local ENU origin (provided externally by caller).
//!
//! ## Web-Mercator inverse (inline, no PROJ dependency)
//! For WKID 3857 / 102100 this can be done with closed-form math (no external dep):
//! ```text
//! λ (lon, rad) = x / R_EARTH
//! φ (lat, rad) = 2·atan(exp(y / R_EARTH)) - π/2
//! R_EARTH = 6_378_137.0  // WGS84 semi-major axis, metres
//! ```
//!
//! # Grounding
//! - I3S-spec `spatialReference.cmn.md` (github.com/Esri/i3s-spec, confirmed 2026-05-26)
//! - ArcGIS Web Map spec `spatialReference` object (confirmed wkid/latestWkid/vcsWkid)
//! - ArcGIS PBF proto `SpatialReference` message (field numbers confirmed 2026-05-26)
//! - "ESRI_crs required for Gaussian splat layers": ArcGIS Pro docs (confirmed intent;
//!   exact JSON schema not independently fetched — UNVERIFIED for glTF-specific form)

// ══════════════════════════════════════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════════════════════════════════════

/// WGS84 semi-major axis in metres — used in Web Mercator inverse projection.
///
/// ```rust
/// // pub const WGS84_A: f64 = 6_378_137.0;
/// ```
pub const WGS84_A: f64 = 6_378_137.0;

/// WKID for WGS84 geographic 2D (EPSG:4326).
pub const WKID_WGS84_GEO2D: u32 = 4326;

/// WKID for WGS84 geographic 3D / lat-lon-ellht (EPSG:4979).
pub const WKID_WGS84_GEO3D: u32 = 4979;

/// WKID for WGS84 geocentric ECEF (EPSG:4978).
pub const WKID_WGS84_ECEF: u32 = 4978;

/// Legacy Web Mercator WKID (Esri alias for EPSG:3857).
pub const WKID_WEB_MERCATOR_LEGACY: u32 = 102100;

/// Current Web Mercator WKID (EPSG:3857).
pub const WKID_WEB_MERCATOR: u32 = 3857;

// ══════════════════════════════════════════════════════════════════════════════
// Parsed ESRI_crs
// ══════════════════════════════════════════════════════════════════════════════

/// Parsed `ESRI_crs` extension object.
///
/// Consumed from the cold JSON import boundary (tileset.json or glTF asset extension).
/// JSON parsing is allowed here — this is the cold boundary, not the hotpath.
///
/// ```rust
/// // #[derive(Debug, Clone)]
/// // pub struct EsriCrs {
/// //     /// Original horizontal WKID (may be legacy alias).
/// //     pub wkid: u32,
/// //     /// Latest/current horizontal WKID; use for projection if non-zero.
/// //     pub latest_wkid: u32,
/// //     /// Vertical CRS WKID; 0 = ellipsoidal height (no conversion needed).
/// //     pub vcs_wkid: u32,
/// //     /// Latest vertical CRS WKID.
/// //     pub latest_vcs_wkid: u32,
/// // }
/// //
/// // impl EsriCrs {
/// //     /// Return the effective horizontal WKID (prefer latestWkid over wkid).
/// //     pub fn effective_h_wkid(&self) -> u32 {
/// //         if self.latest_wkid != 0 { self.latest_wkid } else { self.wkid }
/// //     }
/// //
/// //     /// Return the effective vertical WKID (0 = none).
/// //     pub fn effective_v_wkid(&self) -> u32 {
/// //         if self.latest_vcs_wkid != 0 { self.latest_vcs_wkid } else { self.vcs_wkid }
/// //     }
/// //
/// //     /// True when the horizontal CRS is already WGS84 (identity pass).
/// //     pub fn is_wgs84(&self) -> bool {
/// //         let h = self.effective_h_wkid();
/// //         matches!(h, WKID_WGS84_GEO2D | WKID_WGS84_GEO3D | WKID_WGS84_ECEF)
/// //     }
/// //
/// //     /// True when this is Web Mercator (fast closed-form inverse available).
/// //     pub fn is_web_mercator(&self) -> bool {
/// //         let h = self.effective_h_wkid();
/// //         matches!(h, WKID_WEB_MERCATOR | WKID_WEB_MERCATOR_LEGACY)
/// //     }
/// // }
/// ```
pub struct EsriCrs;

/// Construct an `EsriCrs` from the four WKID fields that appear in both the
/// ArcGIS PBF `SpatialReference` proto message and the `ESRI_crs` JSON extension.
///
/// This is the bridge between the PBF decode path and the CRS handling path —
/// `PbfSpatialReference` maps 1:1 onto this struct.
///
/// ```rust
/// // pub fn from_pbf_spatial_ref(
/// //     wkid: u32,
/// //     latest_wkid: u32,
/// //     vcs_wkid: u32,
/// //     latest_vcs_wkid: u32,
/// // ) -> EsriCrs {
/// //     EsriCrs { wkid, latest_wkid, vcs_wkid, latest_vcs_wkid }
/// // }
/// ```
pub fn from_pbf_spatial_ref(_wkid: u32, _latest_wkid: u32, _vcs_wkid: u32, _latest_vcs_wkid: u32) -> EsriCrs {
    unimplemented!("scaffold only")
}

/// Parse `ESRI_crs` from a raw JSON value at the cold import boundary.
/// JSON is permitted here — this is NOT in the hotpath.
///
/// Expected JSON shape (see module-level doc for schema):
/// ```json
/// { "wkid": 4326, "latestWkid": 4326, "vcsWkid": 0, "latestVcsWkid": 0 }
/// ```
///
/// ```rust
/// // pub fn from_json_value(v: &serde_json::Value) -> Result<EsriCrs, CrsError> {
/// //     // NOTE: serde_json import is ONLY in this cold-boundary function.
/// //     //       Never call this from the frame-decode loop.
/// //     Ok(EsriCrs {
/// //         wkid:            v["wkid"].as_u64().unwrap_or(0) as u32,
/// //         latest_wkid:     v["latestWkid"].as_u64().unwrap_or(0) as u32,
/// //         vcs_wkid:        v["vcsWkid"].as_u64().unwrap_or(0) as u32,
/// //         latest_vcs_wkid: v["latestVcsWkid"].as_u64().unwrap_or(0) as u32,
/// //     })
/// // }
/// ```
// UNVERIFIED: field-name casing in the glTF ESRI_crs extension JSON may differ from
// the PBF proto (proto uses "lastestWkid" — sic — vs JSON uses "latestWkid").
// Confirm against an actual tileset.json produced by ArcGIS Pro before wiring.
pub fn from_json_value(_v: &()) -> Result<EsriCrs, CrsError> {
    unimplemented!("scaffold only — JSON parse wired at cold boundary; do not call from hotpath")
}

// ══════════════════════════════════════════════════════════════════════════════
// Reprojection output
// ══════════════════════════════════════════════════════════════════════════════

/// Coordinates reprojected to WGS84 geographic 3D (EPSG:4979): lon, lat, ellht.
/// This is the intermediate consumed by `to_cam_soa` before local-ENU rebasing.
///
/// ```rust
/// // pub struct Wgs84Coord {
/// //     pub lon_deg: f64,   // longitude, degrees
/// //     pub lat_deg: f64,   // latitude, degrees
/// //     pub ellht_m: f64,   // ellipsoidal height, metres
/// // }
/// ```
pub struct Wgs84Coord;

/// Local-ENU rebased coordinate (east, north, up in metres relative to origin).
/// Produced by `rebase_to_local_enu` after WGS84 reprojection.
/// Consumed by `to_cam_soa` as the final position passed into `GaussianBatch`.
///
/// ```rust
/// // pub struct EnuCoord {
/// //     pub east_m:  f32,
/// //     pub north_m: f32,
/// //     pub up_m:    f32,
/// // }
/// ```
pub struct EnuCoord;

// ══════════════════════════════════════════════════════════════════════════════
// Reprojection functions
// ══════════════════════════════════════════════════════════════════════════════

/// Reproject a batch of XYZ coordinates from the CRS described by `crs` to WGS84 (EPSG:4979).
///
/// `coords` is a flat `[x0,y0,z0, x1,y1,z1, …]` slice in source-CRS units.
/// Returns a `Vec<Wgs84Coord>` of length `coords.len() / 3`.
///
/// Dispatch logic:
/// 1. `crs.is_wgs84()` → identity (copy to `Wgs84Coord` with lon=x, lat=y, h=z)
/// 2. `crs.is_web_mercator()` → closed-form inverse Mercator (no external dep)
/// 3. Otherwise → UNVERIFIED: requires PROJ runtime; return `Err(CrsError::ProjRequired)`
///
/// ```rust
/// // pub fn reproject_to_wgs84(
/// //     crs: &EsriCrs,
/// //     coords: &[f64],
/// // ) -> Result<Vec<Wgs84Coord>, CrsError> {
/// //     assert_eq!(coords.len() % 3, 0, "coords must be xyz triples");
/// //     let n = coords.len() / 3;
/// //     let mut out = Vec::with_capacity(n);
/// //     if crs.is_wgs84() {
/// //         for i in 0..n {
/// //             out.push(Wgs84Coord {
/// //                 lon_deg: coords[i*3],
/// //                 lat_deg: coords[i*3+1],
/// //                 ellht_m: coords[i*3+2],
/// //             });
/// //         }
/// //     } else if crs.is_web_mercator() {
/// //         for i in 0..n {
/// //             let (lon, lat) = inverse_mercator(coords[i*3], coords[i*3+1]);
/// //             out.push(Wgs84Coord { lon_deg: lon, lat_deg: lat, ellht_m: coords[i*3+2] });
/// //         }
/// //     } else {
/// //         return Err(CrsError::ProjRequired(crs.effective_h_wkid()));
/// //     }
/// //     Ok(out)
/// // }
/// ```
pub fn reproject_to_wgs84(_crs: &EsriCrs, _coords: &[f64]) -> Result<Vec<Wgs84Coord>, CrsError> {
    unimplemented!("scaffold only")
}

/// Closed-form inverse Web Mercator: (x_m, y_m) → (lon_deg, lat_deg).
/// Valid for EPSG:3857 / WKID 102100.  No external dependency.
///
/// ```rust
/// // #[inline]
/// // fn inverse_mercator(x: f64, y: f64) -> (f64, f64) {
/// //     let lon = x / WGS84_A;
/// //     let lat = 2.0 * (y / WGS84_A).exp().atan() - std::f64::consts::FRAC_PI_2;
/// //     (lon.to_degrees(), lat.to_degrees())
/// // }
/// ```
pub fn inverse_mercator(_x: f64, _y: f64) -> (f64, f64) {
    unimplemented!("scaffold only")
}

/// Rebase a batch of WGS84 coordinates to a local ENU frame centred on `origin`.
///
/// `origin` is the WGS84 reference point (lon_deg, lat_deg, ellht_m) that maps to (0,0,0).
/// Uses the standard WGS84 → ECEF → ENU rotation (3×3 matrix, closed-form).
///
/// ```rust
/// // pub fn rebase_to_local_enu(
/// //     coords: &[Wgs84Coord],
/// //     origin: &Wgs84Coord,
/// // ) -> Vec<EnuCoord> {
/// //     // 1. Convert origin to ECEF
/// //     // 2. Build ENU rotation matrix from origin lat/lon
/// //     // 3. For each coord: ECEF → subtract origin ECEF → rotate → ENU
/// //     // UNVERIFIED: geoid separation for vertical not applied here;
/// //     //             caller must handle vcsWkid != 0 separately.
/// //     todo!()
/// // }
/// //
/// // fn wgs84_to_ecef(lon_deg: f64, lat_deg: f64, h_m: f64) -> [f64; 3] {
/// //     let a = WGS84_A;
/// //     let f = 1.0 / 298.257_223_563; // WGS84 flattening
/// //     let e2 = 2.0*f - f*f;
/// //     let lat = lat_deg.to_radians();
/// //     let lon = lon_deg.to_radians();
/// //     let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
/// //     [(n + h_m)*lat.cos()*lon.cos(),
/// //      (n + h_m)*lat.cos()*lon.sin(),
/// //      (n*(1.0-e2) + h_m)*lat.sin()]
/// // }
/// ```
pub fn rebase_to_local_enu(_coords: &[Wgs84Coord], _origin: &Wgs84Coord) -> Vec<EnuCoord> {
    unimplemented!("scaffold only")
}

// ══════════════════════════════════════════════════════════════════════════════
// Error type
// ══════════════════════════════════════════════════════════════════════════════

/// Errors produced by the CRS reprojection pipeline.
///
/// ```rust
/// // #[derive(Debug)]
/// // pub enum CrsError {
/// //     /// JSON parsing failed at the cold import boundary.
/// //     JsonParse(String),
/// //     /// WKID requires PROJ runtime (not implemented in scaffold).
/// //     ProjRequired(u32),
/// //     /// WKID is unknown / unrecognised.
/// //     UnknownWkid(u32),
/// //     /// Input coordinate slice is not a multiple of 3.
/// //     BadCoordStride(usize),
/// //     /// Vertical CRS conversion requires geoid grid (not yet wired).
/// //     VerticalCrsRequired(u32),
/// // }
/// ```
pub struct CrsError;
