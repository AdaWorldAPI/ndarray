//! `spz` — `KHR_gaussian_splatting_compression_spz` decode.
//!
//! Niantic SPZ is the binary compressed format ratified as
//! `KHR_gaussian_splatting_compression_spz` inside the Khronos glTF Gaussian Splatting
//! extension family (release-candidate as of 2026-02).  It achieves ≈90% size reduction
//! vs equivalent PLY through per-attribute quantisation + per-stream ZSTD compression
//! (v4; earlier versions used gzip).
//!
//! # Format overview
//! Source: `github.com/nianticlabs/spz` (open source, fetched 2026-05-26).
//! All struct fields and constants below are grounded against `load-spz.cc` / `spz.h`.
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  NgspFileHeader  (32 bytes, plaintext, LE)       │
//! ├─────────────────────────────────────────────────┤
//! │  optional extension blocks  (tocByteOffset − 32)│
//! ├─────────────────────────────────────────────────┤
//! │  TOC entries  (numStreams × 16 bytes)            │
//! │    each: [compressedSize: u64, uncompSize: u64] │
//! ├─────────────────────────────────────────────────┤
//! │  compressed stream 0: positions                 │
//! │  compressed stream 1: alphas                    │
//! │  compressed stream 2: colors (DC SH coeff)      │
//! │  compressed stream 3: scales                    │
//! │  compressed stream 4: rotations                 │
//! │  compressed stream 5: SH (degree 1–4 rest)      │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # NgspFileHeader layout (32 bytes, little-endian)
//! ```text
//! offset  size  field
//! 0       4     magic          — 0x5053474e  ("NGSP" ASCII LE)
//! 4       4     version        — currently 4
//! 8       4     numPoints      — Gaussian count
//! 12      1     shDegree       — 0–4
//! 13      1     fractionalBits — fixed-point precision (default 12)
//! 14      1     flags          — FlagAntialiased=0x1, FlagHasExtensions=0x2
//! 15      1     numStreams     — typically 6
//! 16      4     tocByteOffset  — byte offset of TOC from file start
//! 20      12    reserved       — must be zero
//! ```
//!
//! # Quantisation details (grounded from `load-spz.cc`)
//!
//! ## Positions — stream 0
//! - **9 bytes/point** (3 coords × 3 bytes each)
//! - 24-bit signed fixed-point, little-endian, sign-extended from bit 23
//! - `real = fixed32 / (1 << fractionalBits)` where `fractionalBits=12` → ~0.25 mm
//! - Coordinate flip may be applied post-dequant (implementation-defined axis convention)
//!
//! ## Alphas — stream 1
//! - **1 byte/point** (uint8)
//! - `opacity_logit = invSigmoid(u8 / 255.0)` where `invSigmoid(x) = ln(x / (1-x))`
//! - Maps linear opacity to logit space for GaussianBatch compatibility
//!
//! ## Colors (DC) — stream 2
//! - **3 bytes/point** (uint8 × 3, RGB)
//! - `sh0[c] = (u8 / 255.0 - 0.5) / colorScale` where `colorScale = 0.15`
//! - These are SH degree-0 coefficients, not sRGB display values
//!
//! ## Scales — stream 3
//! - **3 bytes/point** (uint8 × 3, log-encoded)
//! - `scale[j] = (u8 / 16.0) - 10.0`  (log-space; exponentiate for Gaussian σ)
//!
//! ## Rotations — stream 4
//! - **v3 and earlier (3 bytes/point):** xyz of unit quaternion as int8, derive w
//! - **v3+ / "smallest-three" (4 bytes/point):**
//!   2-bit index of the omitted (largest) component + three 10-bit signed magnitudes
//!   packed into 32 bits.  `MIN_SMALLEST_THREE_QUATERNIONS_VERSION = 3`.
//!
//!   Decode sketch:
//!   ```text
//!   packed u32 → largest_idx (bits 30-31) + 3 × 10-bit signed components (bits 0-29)
//!   reconstruct omitted component = sqrt(1 - dot(smallest_three, smallest_three))
//!   renormalise → unit quaternion [w, x, y, z]
//!   ```
//!   // UNVERIFIED: exact bit-field layout within the 32-bit word; ground against
//!   //             nianticlabs/spz `unpackSmallestThreeQuaternion` before implementing.
//!
//! ## Spherical Harmonics — stream 5
//! - Only present if `shDegree > 0`
//! - **Bit widths:** degree 1 → 5 bits/coeff (default), degrees 2–4 → 4 bits/coeff
//!   (configurable via `sh1Bits`/`shRestBits` in range 1–8)
//! - In the stream as-stored all SH coefficients are repacked to **8 bits**:
//!   `sh = (u8 - 128.0) / 128.0`
//! - Layout: for each SH coefficient index across all points, then across R/G/B channels
//!   (column-major per coefficient, not interleaved per point)
//! - `shDim` = number of SH coefficients per channel:
//!   degree 0 → 1, degree 1 → 4, degree 2 → 9, degree 3 → 16
//!   // UNVERIFIED: confirm shDim formula against spz source (degree 0 is in colors stream)
//!
//! # KHR_gaussian_splatting_compression_spz glTF binding
//! The SPZ blob appears as a glTF `bufferView` referenced from the mesh primitive's
//! extension object under `KHR_gaussian_splatting_compression_spz`.  The extension
//! replaces the per-attribute accessor layout defined by `KHR_gaussian_splatting`;
//! the splat renderer reconstructs all attributes from the SPZ streams.
//!
//! # CAM SoA target
//! `spz_to_gaussian_batch` (agent C, `to_cam_soa`) consumes `SpzGaussianCloud` and
//! produces `GaussianBatch` + `cam_pq` indices.  This module stops at the intermediate.
//!
//! # Grounding
//! - `github.com/nianticlabs/spz` (MIT, fetched 2026-05-26)
//! - Khronos `KHR_gaussian_splatting_compression_spz` PR #2490 (release-candidate 2026-02)
//! - Polyvia3D SPZ format overview (secondary confirmation)

// ══════════════════════════════════════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════════════════════════════════════

/// SPZ file magic: bytes `NGSP` in ASCII little-endian → `0x5053474e`.
///
/// ```rust
/// // pub const NGSP_MAGIC: u32 = 0x5053_474e;
/// ```
pub const NGSP_MAGIC: u32 = 0x5053_474e;

/// Minimum SPZ version that uses "smallest-three" quaternion encoding.
/// Versions < 3 use raw int8 xyz + derived w (3 bytes/point).
///
/// ```rust
/// // pub const MIN_SMALLEST_THREE_VERSION: u32 = 3;
/// ```
pub const MIN_SMALLEST_THREE_VERSION: u32 = 3;

/// Default fixed-point fractional bit count (yields ~0.25 mm resolution).
///
/// ```rust
/// // pub const DEFAULT_FRACTIONAL_BITS: u8 = 12;
/// ```
pub const DEFAULT_FRACTIONAL_BITS: u8 = 12;

/// DC colour scaling factor applied when dequantising SH degree-0 coefficients.
/// `sh0 = (u8/255 - 0.5) / COLOR_SCALE`.
///
/// ```rust
/// // pub const COLOR_SCALE: f32 = 0.15;
/// ```
pub const COLOR_SCALE: f32 = 0.15;

// ══════════════════════════════════════════════════════════════════════════════
// Header
// ══════════════════════════════════════════════════════════════════════════════

/// Parsed SPZ file header (32 bytes, little-endian plaintext).
/// Grounded against `NgspFileHeader` in `nianticlabs/spz`.
///
/// ```rust
/// // #[repr(C)]
/// // pub struct SpzHeader {
/// //     pub magic:           u32,   // must == NGSP_MAGIC
/// //     pub version:         u32,   // must be 1–4; 4 = ZSTD streams
/// //     pub num_points:      u32,   // Gaussian count
/// //     pub sh_degree:       u8,    // 0–4
/// //     pub fractional_bits: u8,    // fixed-point precision
/// //     pub flags:           u8,    // bit 0 = antialiased; bit 1 = has_extensions
/// //     pub num_streams:     u8,    // typically 6
/// //     pub toc_byte_offset: u32,   // TOC start offset from file start
/// //     pub reserved:        [u8; 12], // must be zero
/// // }
/// //
/// // impl SpzHeader {
/// //     pub const SIZE: usize = 32;
/// //
/// //     pub fn from_bytes(b: &[u8; 32]) -> Result<Self, SpzError> {
/// //         let magic = u32::from_le_bytes(b[0..4].try_into().unwrap());
/// //         if magic != NGSP_MAGIC { return Err(SpzError::BadMagic(magic)); }
/// //         let version = u32::from_le_bytes(b[4..8].try_into().unwrap());
/// //         Ok(Self {
/// //             magic,
/// //             version,
/// //             num_points:      u32::from_le_bytes(b[8..12].try_into().unwrap()),
/// //             sh_degree:       b[12],
/// //             fractional_bits: b[13],
/// //             flags:           b[14],
/// //             num_streams:     b[15],
/// //             toc_byte_offset: u32::from_le_bytes(b[16..20].try_into().unwrap()),
/// //             reserved:        b[20..32].try_into().unwrap(),
/// //         })
/// //     }
/// //
/// //     pub fn has_extensions(&self) -> bool { self.flags & 0x2 != 0 }
/// //     pub fn is_antialiased(&self) -> bool { self.flags & 0x1 != 0 }
/// // }
/// ```
pub struct SpzHeader;

/// One entry in the stream Table of Contents (16 bytes each).
/// Located at `header.toc_byte_offset`, with `header.num_streams` entries.
///
/// ```rust
/// // pub struct SpzTocEntry {
/// //     pub compressed_size:   u64,
/// //     pub uncompressed_size: u64,
/// // }
/// ```
pub struct SpzTocEntry;

// ══════════════════════════════════════════════════════════════════════════════
// Decoded intermediate
// ══════════════════════════════════════════════════════════════════════════════

/// Fully decoded SPZ Gaussian cloud — intermediate struct consumed by `to_cam_soa`.
/// All values are in their natural floating-point representation post-dequantisation.
/// This struct must NOT outlive the import boundary.
///
/// Layout is **SoA** (structure-of-arrays) to match CAM SoA target and enable SIMD
/// conversion passes in `to_cam_soa`.
///
/// ```rust
/// // pub struct SpzGaussianCloud {
/// //     pub num_points:  usize,
/// //     pub sh_degree:   u8,
/// //     pub antialiased: bool,
/// //
/// //     /// Positions [x,y,z] — f32, world-space after fixed-point decode.
/// //     /// Length: num_points * 3.
/// //     pub positions: Vec<f32>,
/// //
/// //     /// Opacity logits (invSigmoid of linear alpha) — f32.
/// //     /// Length: num_points.
/// //     pub opacity_logits: Vec<f32>,
/// //
/// //     /// DC SH coefficients / base colour, logit-scaled — f32.
/// //     /// Length: num_points * 3 (RGB interleaved: r0,g0,b0, r1,g1,b1, …).
/// //     pub sh_dc: Vec<f32>,
/// //
/// //     /// Log-space scale [sx, sy, sz] per Gaussian — f32.
/// //     /// Length: num_points * 3.
/// //     pub log_scales: Vec<f32>,
/// //
/// //     /// Unit quaternions [w, x, y, z] per Gaussian — f32.
/// //     /// Length: num_points * 4.
/// //     pub rotations: Vec<f32>,
/// //
/// //     /// Higher-degree SH coefficients (degrees 1–shDegree) — f32.
/// //     /// Length: num_points * (shDim-1) * 3  where shDim = (shDegree+1)^2.
/// //     /// UNVERIFIED: confirm exact shDim formula and channel-interleave order.
/// //     pub sh_rest: Vec<f32>,
/// // }
/// ```
pub struct SpzGaussianCloud;

// ══════════════════════════════════════════════════════════════════════════════
// Decode pipeline
// ══════════════════════════════════════════════════════════════════════════════

/// Top-level entry point: decode a raw SPZ blob into [`SpzGaussianCloud`].
///
/// # Steps
/// 1. Parse [`SpzHeader`] from bytes[0..32].
/// 2. Parse TOC from `bytes[header.toc_byte_offset..]`.
/// 3. Locate stream data start = `toc_byte_offset + num_streams * 16`.
/// 4. For each stream: slice compressed bytes, decompress with ZSTD (v4) or gzip (v1–3).
/// 5. Decode each attribute stream (see below).
/// 6. Return `SpzGaussianCloud`.
///
/// # Version compatibility
/// - v1–2: gzip compression, 3-byte/point rotation (int8 xyz)
/// - v3:   gzip + smallest-three quaternion (4 bytes/point)
/// - v4:   ZSTD per-stream + smallest-three quaternion
///
/// ```rust
/// // pub fn decode_spz(bytes: &[u8]) -> Result<SpzGaussianCloud, SpzError> {
/// //     let hdr_bytes: &[u8; 32] = bytes[..32].try_into().map_err(|_| SpzError::TooShort)?;
/// //     let hdr = SpzHeader::from_bytes(hdr_bytes)?;
/// //     let toc = parse_toc(bytes, &hdr)?;
/// //     let stream_data_start = hdr.toc_byte_offset as usize + hdr.num_streams as usize * 16;
/// //     let streams = decompress_streams(bytes, &hdr, &toc, stream_data_start)?;
/// //     Ok(SpzGaussianCloud {
/// //         num_points:    hdr.num_points as usize,
/// //         sh_degree:     hdr.sh_degree,
/// //         antialiased:   hdr.is_antialiased(),
/// //         positions:     decode_positions(&streams[0], &hdr)?,
/// //         opacity_logits:decode_alphas(&streams[1], hdr.num_points as usize)?,
/// //         sh_dc:         decode_colors(&streams[2], hdr.num_points as usize)?,
/// //         log_scales:    decode_scales(&streams[3], hdr.num_points as usize)?,
/// //         rotations:     decode_rotations(&streams[4], &hdr)?,
/// //         sh_rest:       decode_sh_rest(&streams[5], &hdr)?,
/// //     })
/// // }
/// ```
pub fn decode_spz(_bytes: &[u8]) -> Result<SpzGaussianCloud, SpzError> {
    unimplemented!("scaffold only — wire zstd dep + uncomment after Opus review")
}

/// Decode stream 0: 24-bit fixed-point positions → f32 world coordinates.
///
/// ```rust
/// // fn decode_positions(stream: &[u8], hdr: &SpzHeader) -> Result<Vec<f32>, SpzError> {
/// //     let n = hdr.num_points as usize;
/// //     assert_eq!(stream.len(), n * 9, "positions stream size mismatch");
/// //     let scale = 1.0_f32 / (1u32 << hdr.fractional_bits) as f32;
/// //     let mut out = Vec::with_capacity(n * 3);
/// //     for i in 0..n {
/// //         for j in 0..3 {
/// //             let base = i * 9 + j * 3;
/// //             let mut v = stream[base] as u32
/// //                 | ((stream[base+1] as u32) << 8)
/// //                 | ((stream[base+2] as u32) << 16);
/// //             if v & 0x80_0000 != 0 { v |= 0xFF00_0000; } // sign extend
/// //             out.push((v as i32) as f32 * scale);
/// //         }
/// //     }
/// //     Ok(out)
/// // }
/// ```
pub fn decode_positions(_stream: &[u8], _num_points: usize, _fractional_bits: u8) -> Vec<f32> {
    unimplemented!("scaffold only")
}

/// Decode stream 1: uint8 alpha → opacity logit (invSigmoid).
///
/// ```rust
/// // fn decode_alphas(stream: &[u8], n: usize) -> Result<Vec<f32>, SpzError> {
/// //     assert_eq!(stream.len(), n);
/// //     Ok(stream.iter().map(|&a| {
/// //         let p = a as f32 / 255.0;
/// //         let p = p.clamp(1e-6, 1.0 - 1e-6); // guard against log(0)
/// //         (p / (1.0 - p)).ln()                 // invSigmoid / logit
/// //     }).collect())
/// // }
/// ```
pub fn decode_alphas(_stream: &[u8]) -> Vec<f32> {
    unimplemented!("scaffold only")
}

/// Decode stream 2: uint8 RGB → DC SH coefficients.
/// `sh0 = (u8 / 255.0 - 0.5) / COLOR_SCALE`.
///
/// ```rust
/// // fn decode_colors(stream: &[u8], n: usize) -> Result<Vec<f32>, SpzError> {
/// //     assert_eq!(stream.len(), n * 3);
/// //     Ok(stream.iter().map(|&c| {
/// //         (c as f32 / 255.0 - 0.5) / COLOR_SCALE
/// //     }).collect())
/// // }
/// ```
pub fn decode_colors(_stream: &[u8]) -> Vec<f32> {
    unimplemented!("scaffold only")
}

/// Decode stream 3: uint8 log-scale → f32 log-scale values.
/// `log_scale = u8 / 16.0 - 10.0`.
///
/// ```rust
/// // fn decode_scales(stream: &[u8], n: usize) -> Result<Vec<f32>, SpzError> {
/// //     assert_eq!(stream.len(), n * 3);
/// //     Ok(stream.iter().map(|&s| s as f32 / 16.0 - 10.0).collect())
/// // }
/// ```
pub fn decode_scales(_stream: &[u8]) -> Vec<f32> {
    unimplemented!("scaffold only")
}

/// Decode stream 4: quaternion bytes → [w, x, y, z] f32 unit quaternions.
///
/// Two sub-cases depending on `header.version`:
/// - version < `MIN_SMALLEST_THREE_VERSION`: int8 xyz, `w = sqrt(1 - x²-y²-z²)`
/// - version >= `MIN_SMALLEST_THREE_VERSION`: smallest-three packed u32
///
/// Smallest-three decode (UNVERIFIED — confirm bit layout against spz source):
/// ```text
/// packed u32:
///   bits 31-30: index of omitted (largest magnitude) component
///   bits 29-20: component A (10-bit signed)
///   bits 19-10: component B (10-bit signed)
///   bits 9-0:   component C (10-bit signed)
/// 10-bit signed scale factor: UNVERIFIED — ground against unpackSmallestThreeQuaternion
/// omitted = sqrt(1 - A²-B²-C²), sign = positive by convention
/// reconstruct full quaternion, renormalise
/// ```
///
/// ```rust
/// // fn decode_rotations(stream: &[u8], hdr: &SpzHeader) -> Result<Vec<f32>, SpzError> {
/// //     let n = hdr.num_points as usize;
/// //     if hdr.version < MIN_SMALLEST_THREE_VERSION {
/// //         // legacy: 3 bytes/point, int8 xyz
/// //         assert_eq!(stream.len(), n * 3);
/// //         // ... decode xyz, derive w ...
/// //     } else {
/// //         // smallest-three: 4 bytes/point
/// //         assert_eq!(stream.len(), n * 4);
/// //         // ... unpack 2-bit index + 3×10-bit signed ...
/// //     }
/// // }
/// ```
pub fn decode_rotations(_stream: &[u8], _version: u32, _num_points: usize) -> Vec<f32> {
    unimplemented!("scaffold only")
}

/// Decode stream 5: higher-degree SH coefficients (degrees 1–shDegree).
/// `sh = (u8 - 128.0) / 128.0`.
///
/// Stream size: `num_points × (shDim - 1) × 3` bytes
/// where `shDim = (shDegree + 1)^2` (UNVERIFIED: degree-0 already in colors stream).
///
/// ```rust
/// // fn decode_sh_rest(stream: &[u8], hdr: &SpzHeader) -> Result<Vec<f32>, SpzError> {
/// //     let sh_dim = ((hdr.sh_degree as usize + 1).pow(2)).saturating_sub(1); // UNVERIFIED
/// //     let expected = hdr.num_points as usize * sh_dim * 3;
/// //     assert_eq!(stream.len(), expected, "SH rest stream size mismatch");
/// //     Ok(stream.iter().map(|&v| (v as f32 - 128.0) / 128.0).collect())
/// // }
/// ```
pub fn decode_sh_rest(_stream: &[u8], _sh_degree: u8, _num_points: usize) -> Vec<f32> {
    unimplemented!("scaffold only")
}

// ══════════════════════════════════════════════════════════════════════════════
// Error type
// ══════════════════════════════════════════════════════════════════════════════

/// Errors produced by the SPZ decode pipeline.
///
/// ```rust
/// // #[derive(Debug)]
/// // pub enum SpzError {
/// //     /// File is shorter than the 32-byte header.
/// //     TooShort,
/// //     /// Magic bytes do not match NGSP_MAGIC.
/// //     BadMagic(u32),
/// //     /// Unsupported version (>4).
/// //     UnsupportedVersion(u32),
/// //     /// ZSTD or gzip decompression failure.
/// //     DecompressFailed(String),
/// //     /// Stream length does not match expected size for attribute.
/// //     StreamSizeMismatch { stream: usize, expected: usize, actual: usize },
/// // }
/// ```
pub struct SpzError;

// ══════════════════════════════════════════════════════════════════════════════
// Internal helpers — commented stubs
// ══════════════════════════════════════════════════════════════════════════════

// fn parse_toc(bytes: &[u8], hdr: &SpzHeader) -> Result<Vec<SpzTocEntry>, SpzError> { ... }
// fn decompress_streams(bytes: &[u8], hdr: &SpzHeader, toc: &[SpzTocEntry], start: usize)
//     -> Result<Vec<Vec<u8>>, SpzError> {
//     // v4: zstd::bulk::decompress per stream
//     // v1-3: flate2::read::GzDecoder
//     ...
// }
//
// NOTE: do NOT use gzip for v4 — the version field is the authoritative switch.
// NOTE: streams may be shorter than num_streams if shDegree=0 (stream 5 is absent/empty).
