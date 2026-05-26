//! `khr_gs` (group A) — glTF `KHR_gaussian_splatting` parse
//! (attributes POSITION, ROTATION quat, SCALE vec3, OPACITY scalar,
//!  SH_DEGREE_n_COEF_i coefficients; COLOR_0 fallback; cold import → intermediate structs)
//!
//! # Grounding
//! - Khronos `KHR_gaussian_splatting` extension README
//!   (KhronosGroup/glTF, extensions/2.0/Khronos/KHR_gaussian_splatting)
//! - glTF 2.0 core spec (accessor componentType / type encoding, primitive modes)
//!
//! # Design contract
//! - **Cold import only.** glTF JSON is parsed once; no serde/JSON values
//!   survive into [`GsSplats`].  Accessor binary data is read once and stored
//!   as typed intermediate arrays.
//! - **Intermediate structs → `to_cam_soa`.** [`GsSplats`] maps 1-to-1 to
//!   `splat3d::gaussian::GaussianBatch` fields.  The actual conversion is owned
//!   by [`crate::to_cam_soa`]; this module only parses.
//! - **No live implementation.** All planned code is `//`-commented scaffold;
//!   reviewed by Opus + CodeRabbit before any impl is uncommented.
//!
//! # Extension location (verified)
//! The extension object lives at:
//! ```text
//! glTF → meshes[i] → primitives[j] → extensions → KHR_gaussian_splatting
//! ```
//! The primitive `mode` MUST be `0` (POINTS).
//! Splat attributes live in `primitive.attributes`, NOT inside the extension object.
//!
//! # Extension object fields (verified)
//! | Field          | Type   | Required | Default           |
//! |---|---|---|---|
//! | `kernel`       | string | yes      | —                 |
//! | `colorSpace`   | string | yes      | —                 |
//! | `projection`   | string | no       | `"perspective"`   |
//! | `sortingMethod`| string | no       | `"cameraDistance"`|
//!
//! `kernel` is currently always `"ellipse"`.
//! `colorSpace` is one of `"srgb_rec709_display"` or `"lin_rec709_display"`.
//!
//! # Attribute semantics (verified)
//! All attributes are at `primitive.attributes`; the extension-namespaced ones
//! use the prefix `KHR_gaussian_splatting:`.
//!
//! | Attribute semantic                          | Accessor type | Component types           | Required |
//! |---|---|---|---|
//! | `POSITION`                                  | VEC3          | FLOAT                     | yes      |
//! | `KHR_gaussian_splatting:ROTATION`           | VEC4          | FLOAT, BYTE_NORM, SHORT_NORM | yes   |
//! | `KHR_gaussian_splatting:SCALE`              | VEC3          | FLOAT, UBYTE, UBYTE_NORM, USHORT, USHORT_NORM | yes |
//! | `KHR_gaussian_splatting:OPACITY`            | SCALAR        | FLOAT, UBYTE_NORM, USHORT_NORM | yes  |
//! | `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` | VEC3         | FLOAT                     | yes (if SH used) |
//! | `KHR_gaussian_splatting:SH_DEGREE_1_COEF_[0-2]` | VEC3    | FLOAT                     | optional |
//! | `KHR_gaussian_splatting:SH_DEGREE_2_COEF_[0-4]` | VEC3    | FLOAT                     | optional |
//! | `KHR_gaussian_splatting:SH_DEGREE_3_COEF_[0-6]` | VEC3    | FLOAT                     | optional |
//! | `COLOR_0`                                   | VEC4          | per glTF core spec        | optional fallback |
//!
//! ROTATION quaternion component order: (x, y, z, w) — standard glTF VEC4 order.
//! Quantized ROTATION: signed byte/short normalized to [-1, 1].
//! Quantized SCALE: unsigned byte/short (normalized or raw — see below).
//! OPACITY: linear value in [0.0, 1.0].
//! COLOR_0: fallback diffuse ≈ clamp(SH_0_0 * 0.282095 + 0.5, 0, 1).
//!
//! SH coefficients must be provided in ascending degree order;
//! lower degrees must be fully populated before higher ones are used.
//!
//! # UNVERIFIED items
//! - Whether quantized SCALE uses UBYTE (un-normalized) or UBYTE_NORM — the
//!   search result listed both; confirm against the normative schema.
//! - Exact glTF componentType integer codes for BYTE_NORM, SHORT_NORM, etc.
//!   (glTF core: BYTE=5120, UBYTE=5121, SHORT=5122, USHORT=5123, FLOAT=5126).

// ─────────────────────────────────────────────────────────────────────────────
// glTF accessor constants — verified against glTF 2.0 core spec §3.6.2
// ─────────────────────────────────────────────────────────────────────────────
//
// ```rust
// /// glTF accessor component type codes.
// mod gltf_component_type {
//     pub const BYTE:           u32 = 5120;
//     pub const UNSIGNED_BYTE:  u32 = 5121;
//     pub const SHORT:          u32 = 5122;
//     pub const UNSIGNED_SHORT: u32 = 5123;
//     pub const UNSIGNED_INT:   u32 = 5125;
//     pub const FLOAT:          u32 = 5126;
// }
//
// /// glTF accessor type strings (from `accessor.type`).
// mod gltf_accessor_type {
//     pub const SCALAR: &str = "SCALAR";
//     pub const VEC2:   &str = "VEC2";
//     pub const VEC3:   &str = "VEC3";
//     pub const VEC4:   &str = "VEC4";
//     pub const MAT2:   &str = "MAT2";
//     pub const MAT3:   &str = "MAT3";
//     pub const MAT4:   &str = "MAT4";
// }
//
// /// glTF primitive mode — POINTS is the only valid mode for KHR_gaussian_splatting.
// const PRIMITIVE_MODE_POINTS: u32 = 0;
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Attribute name constants  (verified against KHR_gaussian_splatting README)
// ─────────────────────────────────────────────────────────────────────────────
//
// ```rust
// /// Attribute semantic names as they appear in `primitive.attributes`.
// mod attr {
//     pub const POSITION:     &str = "POSITION";
//     pub const COLOR_0:      &str = "COLOR_0";
//     pub const ROTATION:     &str = "KHR_gaussian_splatting:ROTATION";
//     pub const SCALE:        &str = "KHR_gaussian_splatting:SCALE";
//     pub const OPACITY:      &str = "KHR_gaussian_splatting:OPACITY";
//     // Degree-0 (DC) spherical harmonic — required when SH is used.
//     pub const SH_0_0:       &str = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0";
//     // Degree-1 (3 coefficients: indices 0-2).
//     pub const SH_1: [&str; 3] = [
//         "KHR_gaussian_splatting:SH_DEGREE_1_COEF_0",
//         "KHR_gaussian_splatting:SH_DEGREE_1_COEF_1",
//         "KHR_gaussian_splatting:SH_DEGREE_1_COEF_2",
//     ];
//     // Degree-2 (5 coefficients: indices 0-4).
//     pub const SH_2: [&str; 5] = [
//         "KHR_gaussian_splatting:SH_DEGREE_2_COEF_0",
//         "KHR_gaussian_splatting:SH_DEGREE_2_COEF_1",
//         "KHR_gaussian_splatting:SH_DEGREE_2_COEF_2",
//         "KHR_gaussian_splatting:SH_DEGREE_2_COEF_3",
//         "KHR_gaussian_splatting:SH_DEGREE_2_COEF_4",
//     ];
//     // Degree-3 (7 coefficients: indices 0-6).
//     pub const SH_3: [&str; 7] = [
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_0",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_1",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_2",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_3",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_4",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_5",
//         "KHR_gaussian_splatting:SH_DEGREE_3_COEF_6",
//     ];
// }
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Planned types  (all COMMENTED OUT)
// ─────────────────────────────────────────────────────────────────────────────

// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Parsed KHR_gaussian_splatting extension object                          │
// │                                                                         │
// │ The extension object at mesh.primitive.extensions.KHR_gaussian_splatting│
// │ contains metadata only — attributes are parsed separately.              │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ```rust
// /// Decoded `KHR_gaussian_splatting` extension descriptor (metadata only).
// pub struct KhrGsExtension {
//     /// The Gaussian kernel shape. Currently only `"ellipse"` is defined.
//     pub kernel: KernelType,
//     /// Color space of reconstructed splat color.
//     pub color_space: ColorSpace,
//     /// Projection type. Default: `"perspective"`.
//     pub projection: ProjectionType,
//     /// Sorting method for alpha compositing. Default: `"cameraDistance"`.
//     pub sorting_method: SortingMethod,
// }
//
// /// Kernel types defined by the extension (extensible in future revisions).
// pub enum KernelType {
//     /// 2D ellipse projecting an ellipsoid (the only currently defined value).
//     Ellipse,
//     /// UNVERIFIED: forward-compat catch-all for unknown kernel strings.
//     Unknown(String),
// }
//
// /// Color space of the reconstructed splat color output.
// pub enum ColorSpace {
//     /// `"srgb_rec709_display"` — BT.709 sRGB display-referred.
//     SrgbRec709Display,
//     /// `"lin_rec709_display"` — BT.709 linear display-referred.
//     LinRec709Display,
//     /// UNVERIFIED: forward-compat for additional color spaces.
//     Unknown(String),
// }
//
// /// Projection type for Gaussian rasterisation.
// pub enum ProjectionType {
//     Perspective,
//     /// UNVERIFIED: whether orthographic is defined in the extension.
//     Unknown(String),
// }
//
// /// Sorting method for depth-ordered alpha compositing.
// pub enum SortingMethod {
//     CameraDistance,
//     /// UNVERIFIED: additional sorting methods may be defined.
//     Unknown(String),
// }
// ```

// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Intermediate parsed splat batch                                          │
// │                                                                         │
// │ All attribute accessors decoded into flat f32 arrays (SoA orientation). │
// │ Layout mirrors splat3d::gaussian::GaussianBatch so to_cam_soa can       │
// │ copy/transform with minimal indirection.                                 │
// │                                                                         │
// │ SH layout: coefficients packed contiguously per splat:                  │
// │   [sh_deg0 (3 f32)] [sh_deg1_0..2 (9 f32)] [sh_deg2_0..4 (15 f32)]   │
// │   [sh_deg3_0..6 (21 f32)]   (only present up to the degree loaded)     │
// │ Total SH floats per splat: 3 + 9 + 15 + 21 = 48 (degree-3 full).      │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ```rust
// /// Intermediate representation of all KHR_gaussian_splatting splats from
// /// one glTF mesh primitive.  Owned; no references into original bytes.
// ///
// /// All decoded values are in f32.  Quantized source data is dequantised
// /// during decode (not deferred to to_cam_soa).
// ///
// /// Field lengths: every non-empty Vec has length `splat_count` (for
// /// per-splat scalars) or `splat_count * N` (for per-splat VecN).
// pub struct GsSplats {
//     /// Number of Gaussian splats in this primitive.
//     pub splat_count: usize,
//     /// Extension metadata (kernel, color space, etc.).
//     pub extension: KhrGsExtension,
//     /// Splat centre positions in local space.  Layout: [x0,y0,z0, x1,y1,z1, ...].
//     /// Source attribute: `POSITION` (VEC3 FLOAT, dequant n/a).
//     pub position: Vec<f32>,       // len = splat_count * 3
//     /// Unit quaternions (x,y,z,w).  Layout: [x0,y0,z0,w0, x1,...].
//     /// Source attribute: `KHR_gaussian_splatting:ROTATION` (VEC4).
//     /// Quantized byte/short inputs are dequantized to f32 here.
//     pub rotation_xyzw: Vec<f32>,  // len = splat_count * 4
//     /// Scale along each principal axis.  Layout: [sx0,sy0,sz0, sx1,...].
//     /// Source attribute: `KHR_gaussian_splatting:SCALE` (VEC3).
//     /// UNVERIFIED: whether raw UBYTE means literal scale or needs a decode table.
//     pub scale: Vec<f32>,          // len = splat_count * 3
//     /// Linear opacity in [0.0, 1.0].
//     /// Source attribute: `KHR_gaussian_splatting:OPACITY` (SCALAR).
//     pub opacity: Vec<f32>,        // len = splat_count
//     /// Spherical harmonics coefficients (all degrees, contiguous per splat).
//     /// Absent (empty) if the primitive carries no SH attributes.
//     /// Source attributes: SH_DEGREE_n_COEF_i (VEC3 FLOAT).
//     pub sh_coefficients: Vec<f32>, // len = splat_count * (3 * (degree+1)^2), or 0
//     /// Highest SH degree loaded (0-3); 0 = only DC term.
//     /// None = no SH attributes present (use color_0_rgba fallback).
//     pub sh_degree: Option<u8>,
//     /// Optional fallback diffuse RGBA (from `COLOR_0`).
//     /// Present only when COLOR_0 attribute exists in the primitive.
//     /// Layout: [r0,g0,b0,a0, r1,...] in [0.0, 1.0].
//     pub color_0_rgba: Option<Vec<f32>>,  // len = splat_count * 4, or None
// }
// ```

// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Quantization dequantise helpers                                         │
// │                                                                         │
// │ These are called during cold decode to produce uniform f32 arrays.      │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ```rust
// /// Dequantise a signed byte normalized value to f32 in [-1.0, 1.0].
// /// glTF core spec §3.6.2.3: max(value / 127.0, -1.0).
// #[inline]
// fn dequant_i8_norm(v: i8) -> f32 {
//     ((v as f32) / 127.0_f32).max(-1.0_f32)
// }
//
// /// Dequantise a signed short normalized value to f32 in [-1.0, 1.0].
// /// glTF core spec §3.6.2.3: max(value / 32767.0, -1.0).
// #[inline]
// fn dequant_i16_norm(v: i16) -> f32 {
//     ((v as f32) / 32767.0_f32).max(-1.0_f32)
// }
//
// /// Dequantise an unsigned byte normalized value to f32 in [0.0, 1.0].
// /// glTF core spec §3.6.2.3: value / 255.0.
// #[inline]
// fn dequant_u8_norm(v: u8) -> f32 {
//     (v as f32) / 255.0_f32
// }
//
// /// Dequantise an unsigned short normalized value to f32 in [0.0, 1.0].
// /// glTF core spec §3.6.2.3: value / 65535.0.
// #[inline]
// fn dequant_u16_norm(v: u16) -> f32 {
//     (v as f32) / 65535.0_f32
// }
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Planned functions  (all COMMENTED OUT)
// ─────────────────────────────────────────────────────────────────────────────

// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Top-level parse entry point                                              │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ```rust
// /// Parse all KHR_gaussian_splatting splat primitives from a glTF binary
// /// (GLB) or JSON+BIN pair.
// ///
// /// Returns one [`GsSplats`] per qualifying mesh primitive (mode=POINTS,
// /// extension present).  A file may contain zero, one, or many.
// ///
// /// This is the ONLY function that reads JSON and binary accessor data.
// /// After it returns, no JSON/serde values survive.
// ///
// /// # Arguments
// /// - `gltf_json`: raw bytes of the glTF JSON (the JSON chunk of a GLB, or
// ///   a standalone `.gltf` file).
// /// - `bin_chunk`: the BIN chunk from a GLB, or an external `.bin` buffer.
// ///   Pass `None` if the file has no binary data.
// ///
// /// # Errors
// /// Returns [`KhrGsError`] on JSON parse failure, missing required attributes,
// /// unsupported accessor types, or buffer bounds violations.
// pub fn parse_khr_gaussian_splatting(
//     gltf_json: &[u8],
//     bin_chunk: Option<&[u8]>,
// ) -> Result<Vec<GsSplats>, KhrGsError> {
//     // 1. Decode glTF JSON → minimal internal repr (no serde dep yet):
//     //    - meshes[*].primitives[*] where mode == 0 (POINTS)
//     //    - filter primitives that have extensions.KHR_gaussian_splatting
//     // 2. For each qualifying primitive, call parse_gs_primitive().
//     // 3. Return Vec<GsSplats>.
//     todo!()
// }
//
// /// Parse one qualifying mesh primitive into [`GsSplats`].
// fn parse_gs_primitive(
//     // primitive attributes map (semantic → accessor index)
//     // extension object fields
//     // accessors array
//     // bufferViews array
//     // buffers array
//     // bin_chunk
// ) -> Result<GsSplats, KhrGsError> {
//     // 1. Parse extension object → KhrGsExtension.
//     // 2. Verify mode == PRIMITIVE_MODE_POINTS.
//     // 3. Extract POSITION accessor → decode_vec3_f32() → position vec.
//     // 4. Extract ROTATION accessor → decode_rotation() (handles quantized types).
//     // 5. Extract SCALE accessor → decode_scale() (handles quantized types).
//     // 6. Extract OPACITY accessor → decode_opacity() (handles quantized types).
//     // 7. Determine SH degree: probe SH_DEGREE_3_* first, step down.
//     // 8. Extract SH accessors for each present degree → decode_sh_vec3_f32().
//     // 9. Optionally extract COLOR_0 accessor.
//     // 10. Validate all lengths == splat_count.
//     // 11. Return GsSplats.
//     todo!()
// }
// ```

// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Accessor decode helpers                                                  │
// │                                                                         │
// │ Each helper resolves accessor → bufferView → buffer slice and returns   │
// │ a decoded Vec<f32>.  All dequantisation happens here (cold path).       │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ```rust
// /// Decode a VEC3 FLOAT accessor into a flat Vec<f32>.
// /// Used for POSITION and SH coefficients.
// fn decode_vec3_f32(
//     // accessor index, accessors, buffer_views, buffers, bin_chunk
// ) -> Result<Vec<f32>, KhrGsError> {
//     // Validate componentType == FLOAT, type == "VEC3".
//     // Slice buffer bytes, cast to &[f32] (checking alignment).
//     // Copy to owned Vec<f32> (respect byteStride if nonzero).
//     todo!()
// }
//
// /// Decode ROTATION VEC4 accessor; dequantise BYTE_NORM or SHORT_NORM if needed.
// /// Output: flat Vec<f32> with [x,y,z,w] per splat, all normalised.
// fn decode_rotation(
//     // accessor index + context
// ) -> Result<Vec<f32>, KhrGsError> {
//     // Match componentType:
//     //   FLOAT (5126): direct copy (4 f32 per element).
//     //   BYTE  (5120) normalized: dequant_i8_norm for each of 4 components.
//     //   SHORT (5122) normalized: dequant_i16_norm for each of 4 components.
//     //   Other: KhrGsError::UnsupportedComponentType.
//     todo!()
// }
//
// /// Decode SCALE VEC3 accessor; dequantise if quantized.
// /// UNVERIFIED: exact dequant formula for UBYTE (non-normalized) variant.
// fn decode_scale(
//     // accessor index + context
// ) -> Result<Vec<f32>, KhrGsError> {
//     // Match componentType:
//     //   FLOAT (5126): direct copy.
//     //   UNSIGNED_BYTE (5121) normalized: dequant_u8_norm.
//     //   UNSIGNED_BYTE (5121) not normalized: UNVERIFIED — cast as f32 directly?
//     //   UNSIGNED_SHORT (5123) normalized: dequant_u16_norm.
//     //   UNSIGNED_SHORT (5123) not normalized: UNVERIFIED.
//     //   Other: KhrGsError::UnsupportedComponentType.
//     todo!()
// }
//
// /// Decode OPACITY SCALAR accessor; dequantise normalized types.
// /// Output: flat Vec<f32> in [0.0, 1.0].
// fn decode_opacity(
//     // accessor index + context
// ) -> Result<Vec<f32>, KhrGsError> {
//     // Match componentType:
//     //   FLOAT (5126): direct copy; clamp to [0,1] in debug builds.
//     //   UNSIGNED_BYTE (5121) normalized: dequant_u8_norm.
//     //   UNSIGNED_SHORT (5123) normalized: dequant_u16_norm.
//     //   Other: KhrGsError::UnsupportedComponentType.
//     todo!()
// }
//
// /// Detect the highest SH degree present by probing attribute names.
// /// Returns 0 if only SH_DEGREE_0_COEF_0 is present, 3 if all degrees present.
// /// Returns None if SH_DEGREE_0_COEF_0 is absent (no SH data at all).
// fn detect_sh_degree(
//     // attribute name set
// ) -> Option<u8> {
//     // Probe SH_3[6] present → return Some(3).
//     // Probe SH_2[4] present → return Some(2).
//     // Probe SH_1[2] present → return Some(1).
//     // Probe SH_0_0 present  → return Some(0).
//     // None.
//     todo!()
// }
//
// /// Decode all SH coefficient accessors (all degrees up to `max_degree`)
// /// into a single contiguous Vec<f32> per splat.
// ///
// /// Layout per splat: [DC(3), deg1_coef0(3),..,deg1_coef2(3),
// ///                    deg2_coef0(3),..,deg2_coef4(3),
// ///                    deg3_coef0(3),..,deg3_coef6(3)]
// ///
// /// All coefficients are FLOAT (VEC3); no dequantisation needed.
// fn decode_sh_coefficients(
//     // attribute map, max_degree, accessor context
// ) -> Result<Vec<f32>, KhrGsError> {
//     todo!()
// }
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Error type  (COMMENTED OUT)
// ─────────────────────────────────────────────────────────────────────────────
//
// ```rust
// pub enum KhrGsError {
//     /// glTF JSON is invalid UTF-8.
//     JsonUtf8,
//     /// glTF JSON syntax error.
//     JsonSyntax(String),
//     /// Required primitive attribute is missing.
//     MissingAttribute(&'static str),
//     /// Accessor componentType is not supported for this attribute.
//     UnsupportedComponentType { attribute: &'static str, component_type: u32 },
//     /// Accessor type string mismatch (e.g. expected VEC3, got SCALAR).
//     AccessorTypeMismatch { attribute: &'static str, expected: &'static str, got: String },
//     /// Primitive mode is not POINTS (0).
//     NotPointsPrimitive(u32),
//     /// Buffer or bufferView index out of range.
//     BufferIndexOutOfRange { index: u32, max: usize },
//     /// Byte slice is out of bounds for the given accessor offset/count/stride.
//     AccessorBoundsError,
//     /// SH coefficient accessor count mismatch (lower degrees absent).
//     ShDegreeGap(u8),
//     /// Splat count inconsistency across attribute accessors.
//     SplatCountMismatch { attribute: &'static str, expected: usize, got: usize },
//     /// Extension object missing required field.
//     MissingExtensionField(&'static str),
//     /// UNVERIFIED: extension field had an unrecognised string value (forward-compat warn).
//     UnrecognizedExtensionValue { field: &'static str, value: String },
// }
// ```
