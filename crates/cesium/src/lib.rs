//! # cesium — optional Cesium / ArcGIS reference + parity oracle
//!
//! **NOT a production dependency.** This crate is the *runnable legacy pillar*
//! (see `.claude/knowledge/crate-registry.md`, modesty clause): it
//! reverse-engineers external geospatial-splat formats into ndarray CAM SoA and
//! diffs our `splat3d` render against a reference, to keep "parity / better"
//! honest by **measurement**. It is a workspace member but **NOT** in
//! `default-members`, so it is optional/dev by construction. It relocates to
//! `lance-graph/crates/cesium` once flawless.
//!
//! ## Hard rules (from crate-registry.md)
//! - **Reverse-engineer ONLY** — extract data, rebuild as CAM SoA; never depend
//!   on, round-trip, or emit the source format. Refactor > reverse-engineer.
//! - **CAM SoA target** — output is ndarray's splat SoA
//!   (`splat3d::gaussian::GaussianBatch`) + `cam_pq` codebook indices. No AoS,
//!   no source-native structs survive past the import boundary.
//! - **NO JSON IN THE HOTPATH, EVER** — JSON only at the cold import boundary.
//!   Prefer the **ArcGIS PBF (protobuf, binary)** path over `f=json`.
//!
//! ## Grounding references (reverse-engineer against these — do NOT fabricate;
//! mark anything you cannot confirm against a source as `// UNVERIFIED:`)
//! - `CesiumGS/cesium` — 3D Tiles / HLOD / KHR_gaussian_splatting behaviour
//! - Khronos `KHR_gaussian_splatting` + `KHR_gaussian_splatting_compression_spz`
//! - `R-ArcGIS/arcpbf`, `AdaWorldAPI/arcgisutils`, `dfridkin/arcGIS-Rust-runtime`,
//!   `EsriDevEvents/locup` — ArcGIS FeatureService **PBF** (binary, no-JSON) + `ESRI_crs`
//! - `staehlli/3D-settlement-development` — a real 3DGS scene for fixtures
//! - Inria `.ply` via ndarray `splat3d::ply` — first reference render for the oracle
//!
//! ## Module map (12 slots — `///` commented scaffold ONLY until reviewed live
//! by Opus + CodeRabbit/Codex)
//! | module | group | purpose |
//! |---|---|---|
//! | [`tileset`] | A | 3D Tiles `tileset.json` / `.3tz` parse (cold import) |
//! | [`implicit_tiling`] | A | subtree availability bitstreams + Morton/Z-order |
//! | [`khr_gs`] | A | KHR_gaussian_splatting glTF parse (POSITION/ROTATION/SCALE/COLOR_0/SH_DEGREE_n) |
//! | [`arcgis_pbf`] | B | ArcGIS FeatureService **PBF** ingestion (binary, the no-JSON path) |
//! | [`spz`] | B | `KHR_gaussian_splatting_compression_spz` decode (quantize+gzip) |
//! | [`esri_crs`] | B | `ESRI_crs` WKID (h+v) reproject → WGS84 / local-origin |
//! | [`to_cam_soa`] | C | reverse-engineer parsed splats → CAM SoA (`GaussianBatch` + `cam_pq`) |
//! | [`sse`] | C | screen-space-error LOD selection (`geometricError·viewportH/(dist·2tan(fovy/2))`) |
//! | [`hlod`] | C | HLOD tree + refine ADD·REPLACE |
//! | [`point_fallback`] | D | point-cloud fallback (KHR base extension = point primitives) |
//! | [`oracle`] | D | reference-render invocation + SSIM/PSNR diff vs `splat3d` |
//! | [`fixtures`] | D | golden scenes + parity gates |

pub mod tileset;
pub mod implicit_tiling;
pub mod khr_gs;
pub mod arcgis_pbf;
pub mod spz;
pub mod esri_crs;
pub mod to_cam_soa;
pub mod sse;
pub mod hlod;
pub mod point_fallback;
pub mod oracle;
pub mod fixtures;
