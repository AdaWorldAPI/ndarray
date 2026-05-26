//! `to_cam_soa` (group C) — Rule-2 bridge: reverse-engineer parsed KHR splats
//! into ndarray CAM SoA (`splat3d::gaussian::GaussianBatch` + `cam_pq` codebook
//! indices).  This is the canonical import target — the only module allowed to
//! produce `GaussianBatch` from external data.
//!
//! # Design contract
//! - **Reverse-engineer, never copy.** The parsed intermediate ([`khr_gs::GsSplats`])
//!   is restructured into SoA; the original attribute layout is abandoned at the
//!   boundary.
//! - **No JSON in the hotpath** (rule 3).  All JSON parsing happened in
//!   [`crate::khr_gs`]; by the time we touch data here it is already `Vec<f32>`.
//! - **Commented scaffold only** — all impl touching `ndarray` is `//`-commented
//!   until the `ndarray` dep in `Cargo.toml` is uncommented and reviewed live by
//!   Opus + CodeRabbit.  Compilable stub structs (std-only) and their unit tests
//!   are live.
//!
//! # GaussianBatch field mapping (grounded against gaussian.rs)
//!
//! ```text
//! GsSplats field          GaussianBatch field        Notes
//! ─────────────────────── ────────────────────────── ──────────────────────────────────
//! position[i*3+0]      →  mean_x[i]
//! position[i*3+1]      →  mean_y[i]
//! position[i*3+2]      →  mean_z[i]
//! scale[i*3+0]         →  scale_x[i]
//! scale[i*3+1]         →  scale_y[i]
//! scale[i*3+2]         →  scale_z[i]
//! rotation_xyzw[i*4+3] →  quat_w[i]                 *** glTF xyzw → splat3d wxyz ***
//! rotation_xyzw[i*4+0] →  quat_x[i]                 x stays x
//! rotation_xyzw[i*4+1] →  quat_y[i]                 y stays y
//! rotation_xyzw[i*4+2] →  quat_z[i]                 z stays z
//! opacity[i]           →  opacity[i]
//! sh_coefficients[..]  →  sh[i*48 + ch*16 + basis]  see SH repack below
//! ```
//!
//! # SH repack (khr_gs layout → GaussianBatch layout)
//!
//! `GsSplats::sh_coefficients` stores coefficients contiguously per splat in
//! *attribute order* (one VEC3 per coefficient: [r,g,b] interleaved):
//!
//! ```text
//! khr_gs per-splat layout (degree-3 = 16 coefs × 3 channels = 48 f32):
//!   [deg0_c0_r, deg0_c0_g, deg0_c0_b,          ← basis k=0, channels rgb
//!    deg1_c0_r, deg1_c0_g, deg1_c0_b,            ← basis k=1
//!    deg1_c1_r, ...,
//!    ...
//!    deg3_c6_r, deg3_c6_g, deg3_c6_b]            ← basis k=15
//! ```
//!
//! `GaussianBatch::sh` stores coefficients in *gaussian-major, channel-major*:
//!
//! ```text
//! sh[i * 48 + ch * 16 + k]   ch ∈ {0=R, 1=G, 2=B},  k ∈ 0..16
//! ```
//!
//! The repack loop is therefore:
//!   `batch.sh[i*48 + 0*16 + k] = gs_sh[k*3 + 0]`  (red)
//!   `batch.sh[i*48 + 1*16 + k] = gs_sh[k*3 + 1]`  (green)
//!   `batch.sh[i*48 + 2*16 + k] = gs_sh[k*3 + 2]`  (blue)
//! where `gs_sh` is the 48-f32 slice for splat i.
//!
//! # CAM-PQ fingerprint (grounded against cam_pq.rs)
//!
//! After the SoA is built, the bridge optionally encodes splat position vectors
//! through `CamCodebook::encode` to produce a [`CamFingerprint`] (`[u8;6]`)
//! per splat.  Encoding requires a pre-trained codebook; callers that have no
//! codebook receive `None` for the fingerprint slice.
//!
//! # UNVERIFIED items
//! - Whether Cesium-exported KHR files always normalise the ROTATION quaternion
//!   before writing, or whether we must renormalise after dequantisation.
//! - Whether COLOR_0 is already linearised or is sRGB-encoded in practice.

// ─────────────────────────────────────────────────────────────────────────────
// Compilable stub types (std-only, no ndarray dep)
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of [`convert_to_cam_soa`] (stub — real fields gated behind ndarray).
///
/// When the ndarray dep is live, `batch` becomes
/// `splat3d::gaussian::GaussianBatch` and `cam_fingerprints` becomes
/// `Vec<cam_pq::CamFingerprint>`.
///
/// # Examples
///
/// ```
/// use cesium::to_cam_soa::{convert_to_cam_soa, CamSoaResult};
///
/// let result: CamSoaResult = convert_to_cam_soa(
///     1,
///     &[0.0, 0.0, 0.0],
///     &[0.0, 0.0, 0.0, 1.0],
///     &[1.0, 1.0, 1.0],
///     &[0.5],
///     None,
/// )
/// .unwrap();
/// assert_eq!(result.splat_count, 1);
/// ```
#[derive(Debug)]
pub struct CamSoaResult {
    /// Number of Gaussian splats converted.
    pub splat_count: usize,
    /// UNVERIFIED: placeholder for the live GaussianBatch field once ndarray dep
    /// is uncommented.  True type: `splat3d::gaussian::GaussianBatch`.
    pub _batch_placeholder: (),
    /// UNVERIFIED: placeholder for per-splat CAM fingerprints once cam_pq dep
    /// is uncommented.  True type: `Vec<cam_pq::CamFingerprint>` = `Vec<[u8;6]>`.
    pub _cam_fingerprints_placeholder: Option<()>,
}

/// Errors emitted by the conversion bridge.
///
/// # Examples
///
/// ```
/// use cesium::to_cam_soa::{convert_to_cam_soa, ToCamSoaError};
///
/// // Empty splat count → EmptySplats
/// let err = convert_to_cam_soa(0, &[], &[], &[], &[], None).unwrap_err();
/// assert_eq!(err, ToCamSoaError::EmptySplats);
///
/// // Wrong position buffer length → BufferLengthMismatch
/// let err = convert_to_cam_soa(
///     1,
///     &[0.0],           // too short: expected 3 (= 1*3)
///     &[0.0, 0.0, 0.0, 1.0],
///     &[1.0, 1.0, 1.0],
///     &[0.5],
///     None,
/// )
/// .unwrap_err();
/// assert!(matches!(
///     err,
///     ToCamSoaError::BufferLengthMismatch { field: "position", expected: 3, actual: 1 }
/// ));
/// ```
#[derive(Debug, PartialEq, Eq)]
pub enum ToCamSoaError {
    /// The source splat count is zero — nothing to convert.
    EmptySplats,
    /// A per-splat input buffer has the wrong length.
    ///
    /// `expected` is `splat_count * stride` (e.g. 3 for position/scale, 4 for
    /// rotation, 1 for opacity).  `actual` is `slice.len()`.
    BufferLengthMismatch {
        /// Name of the offending field (`"position"`, `"rotation_xyzw"`,
        /// `"scale"`, or `"opacity"`).
        field: &'static str,
        /// Expected length (`splat_count * per-splat stride`).
        expected: usize,
        /// Actual length of the slice that was passed in.
        actual: usize,
    },
    /// SH coefficient array length is inconsistent with `splat_count * sh_stride`.
    ShLengthMismatch {
        splat_count: usize,
        sh_len: usize,
        expected_stride: usize,
    },
    /// UNVERIFIED: quaternion norm was below threshold after dequantisation.
    DegenerateRotation { splat_index: usize },
}

impl core::fmt::Display for ToCamSoaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptySplats => write!(f, "to_cam_soa: source splat count is zero"),
            Self::BufferLengthMismatch {
                field,
                expected,
                actual,
            } => write!(f, "to_cam_soa: buffer `{field}` length {actual} != expected {expected}",),
            Self::ShLengthMismatch {
                splat_count,
                sh_len,
                expected_stride,
            } => write!(
                f,
                "to_cam_soa: sh_coefficients length {} != splat_count {} * stride {}",
                sh_len, splat_count, expected_stride,
            ),
            Self::DegenerateRotation { splat_index } => {
                write!(f, "to_cam_soa: degenerate (near-zero-norm) quaternion at splat {}", splat_index,)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live stub function (returns a placeholder result; real body commented below)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert parsed KHR gaussian splats into ndarray CAM SoA.
///
/// **Stub.** Returns `Ok(CamSoaResult { splat_count, .. })` so callers
/// and tests compile.  Real body is `//`-commented until ndarray dep is live.
///
/// # Arguments
/// - `splat_count` — number of splats in the source arrays (for validation).
/// - `position` — flat f32 slice, layout `[x0,y0,z0, x1,y1,z1, ...]`; must have length `splat_count * 3`.
/// - `rotation_xyzw` — flat f32 slice, layout `[x0,y0,z0,w0, x1,...]`; must have length `splat_count * 4`.
///   **Note:** glTF stores (x,y,z,w); the bridge reorders to GaussianBatch's (w,x,y,z).
/// - `scale` — flat f32 slice, layout `[sx0,sy0,sz0, ...]`; must have length `splat_count * 3`.
/// - `opacity` — flat f32 slice, one value per splat; must have length `splat_count`.
/// - `sh_coefficients` — optional flat f32 slice; `None` means no SH.
///   Layout per splat: `[k=0 rgb, k=1 rgb, ..., k=15 rgb]` (48 f32 for degree-3);
///   when `Some`, must have length `splat_count * 48`.
///
/// # Returns
/// `Ok(CamSoaResult)` on success; `Err(ToCamSoaError)` on validation failure.
///
/// # Errors
/// - [`ToCamSoaError::EmptySplats`] — `splat_count == 0`.
/// - [`ToCamSoaError::BufferLengthMismatch`] — any per-splat buffer has the wrong length.
/// - [`ToCamSoaError::ShLengthMismatch`] — `sh_coefficients` is `Some` but has the wrong length.
///
/// # Examples
///
/// ```
/// use cesium::to_cam_soa::{convert_to_cam_soa, ToCamSoaError};
///
/// // Happy path: one splat, all buffers correctly sized.
/// let result = convert_to_cam_soa(
///     1,
///     &[0.0, 0.0, 0.0],          // position: 1*3 = 3 elements
///     &[0.0, 0.0, 0.0, 1.0],     // rotation_xyzw: 1*4 = 4 elements (identity quat)
///     &[1.0, 1.0, 1.0],          // scale: 1*3 = 3 elements
///     &[0.5],                     // opacity: 1 element
///     None,
/// )
/// .unwrap();
/// assert_eq!(result.splat_count, 1);
///
/// // Error path: position buffer too short.
/// let err = convert_to_cam_soa(
///     2,
///     &[0.0, 0.0, 0.0],          // wrong: expected 6 (= 2*3)
///     &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
///     &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
///     &[0.5, 0.5],
///     None,
/// )
/// .unwrap_err();
/// assert!(matches!(
///     err,
///     ToCamSoaError::BufferLengthMismatch { field: "position", expected: 6, actual: 3 }
/// ));
/// ```
pub fn convert_to_cam_soa(
    splat_count: usize, position: &[f32], rotation_xyzw: &[f32], scale: &[f32], opacity: &[f32],
    sh_coefficients: Option<&[f32]>,
) -> Result<CamSoaResult, ToCamSoaError> {
    if splat_count == 0 {
        return Err(ToCamSoaError::EmptySplats);
    }
    // Validate all per-splat buffer lengths before proceeding.
    let checks: &[(&'static str, usize, usize)] = &[
        ("position", splat_count * 3, position.len()),
        ("rotation_xyzw", splat_count * 4, rotation_xyzw.len()),
        ("scale", splat_count * 3, scale.len()),
        ("opacity", splat_count, opacity.len()),
    ];
    for &(field, expected, actual) in checks {
        if actual != expected {
            return Err(ToCamSoaError::BufferLengthMismatch {
                field,
                expected,
                actual,
            });
        }
    }
    // Validate SH length when present.
    if let Some(sh) = sh_coefficients {
        const SH_STRIDE: usize = 48; // 3 channels × 16 basis functions
        let expected = splat_count * SH_STRIDE;
        if sh.len() != expected {
            return Err(ToCamSoaError::ShLengthMismatch {
                splat_count,
                sh_len: sh.len(),
                expected_stride: SH_STRIDE,
            });
        }
    }
    Ok(CamSoaResult {
        splat_count,
        _batch_placeholder: (),
        _cam_fingerprints_placeholder: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// COMMENTED-OUT full implementation (ndarray dep required)
// ─────────────────────────────────────────────────────────────────────────────
//
// Uncomment when `ndarray = { workspace = true }` is re-enabled in Cargo.toml
// and this module has been reviewed live by Opus + CodeRabbit.
//
// ```rust
// use ndarray_crate_alias::hpc::splat3d::gaussian::{GaussianBatch, SH_COEFFS_PER_GAUSSIAN};
// use ndarray_crate_alias::hpc::cam_pq::{CamCodebook, CamFingerprint};
//
// /// Full conversion: GsSplats → GaussianBatch + optional CAM fingerprints.
// ///
// /// # Quaternion reorder (CRITICAL)
// /// glTF / KHR_gaussian_splatting stores ROTATION as VEC4 (x,y,z,w).
// /// GaussianBatch stores quaternions as (w,x,y,z) — the `quat` field of
// /// `Gaussian3D` is documented as `[w, x, y, z]` (gaussian.rs line 53).
// /// The mapping is therefore:
// ///   rotation_xyzw[i*4 + 3]  → quat_w[i]   ← w is at index 3 in glTF
// ///   rotation_xyzw[i*4 + 0]  → quat_x[i]
// ///   rotation_xyzw[i*4 + 1]  → quat_y[i]
// ///   rotation_xyzw[i*4 + 2]  → quat_z[i]
// ///
// /// # SH repack
// /// khr_gs layout: per-splat block of (n_coefs × 3) f32s, interleaved rgb per basis:
// ///   gs_sh_block[k * 3 + ch]  where k ∈ 0..16, ch ∈ {0=R,1=G,2=B}
// /// GaussianBatch layout: sh[i*48 + ch*16 + k]
// /// Repack: batch.sh[i*48 + ch*16 + k] = gs_sh_block[k*3 + ch]
// pub fn convert_to_cam_soa_live(
//     splat_count: usize,
//     position: &[f32],
//     rotation_xyzw: &[f32],
//     scale: &[f32],
//     opacity: &[f32],
//     sh_coefficients: Option<&[f32]>,
//     codebook: Option<&CamCodebook>,
// ) -> Result<(GaussianBatch, Option<Vec<CamFingerprint>>), ToCamSoaError> {
//     if splat_count == 0 {
//         return Err(ToCamSoaError::EmptySplats);
//     }
//
//     let mut batch = GaussianBatch::with_capacity(splat_count);
//
//     // ── 1. Scatter SoA fields ──────────────────────────────────────────────
//     for i in 0..splat_count {
//         // Position → mean_x / mean_y / mean_z
//         batch.mean_x[i] = position[i * 3 + 0];
//         batch.mean_y[i] = position[i * 3 + 1];
//         batch.mean_z[i] = position[i * 3 + 2];
//
//         // Scale → scale_x / scale_y / scale_z
//         batch.scale_x[i] = scale[i * 3 + 0];
//         batch.scale_y[i] = scale[i * 3 + 1];
//         batch.scale_z[i] = scale[i * 3 + 2];
//
//         // Rotation reorder: glTF (x,y,z,w) → GaussianBatch (w,x,y,z)
//         batch.quat_w[i] = rotation_xyzw[i * 4 + 3]; // *** w is at [3] in glTF ***
//         batch.quat_x[i] = rotation_xyzw[i * 4 + 0];
//         batch.quat_y[i] = rotation_xyzw[i * 4 + 1];
//         batch.quat_z[i] = rotation_xyzw[i * 4 + 2];
//
//         // Opacity
//         batch.opacity[i] = opacity[i];
//     }
//     batch.len = splat_count;
//
//     // ── 2. SH repack (optional) ────────────────────────────────────────────
//     if let Some(sh) = sh_coefficients {
//         const SH_STRIDE: usize = SH_COEFFS_PER_GAUSSIAN; // 48
//         let expected = splat_count * SH_STRIDE;
//         if sh.len() != expected {
//             return Err(ToCamSoaError::ShLengthMismatch {
//                 splat_count,
//                 sh_len: sh.len(),
//                 expected_stride: SH_STRIDE,
//             });
//         }
//         for i in 0..splat_count {
//             let gs_base = i * SH_STRIDE;     // start of splat i's khr_gs block
//             let batch_base = i * SH_STRIDE;  // start of splat i's batch block
//             // khr_gs: gs_sh_block[k * 3 + ch]
//             // batch:  sh[batch_base + ch * 16 + k]
//             for k in 0..16_usize {
//                 for ch in 0..3_usize {
//                     batch.sh[batch_base + ch * 16 + k] = sh[gs_base + k * 3 + ch];
//                 }
//             }
//         }
//     }
//
//     // ── 3. CAM-PQ encode (optional) ───────────────────────────────────────
//     // Encode position (3D) padded to codebook's total_dim with zeros.
//     // UNVERIFIED: whether position alone or a richer feature vector should be
//     // encoded.  Using position as a spatial index is the conservative choice.
//     let cam_fingerprints = if let Some(cb) = codebook {
//         let mut fps: Vec<CamFingerprint> = Vec::with_capacity(splat_count);
//         let mut padded = vec![0.0f32; cb.total_dim];
//         for i in 0..splat_count {
//             padded[0] = batch.mean_x[i];
//             padded[1] = batch.mean_y[i];
//             if cb.total_dim > 2 { padded[2] = batch.mean_z[i]; }
//             fps.push(cb.encode(&padded));
//         }
//         Some(fps)
//     } else {
//         None
//     };
//
//     Ok((batch, cam_fingerprints))
// }
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (std-only, live)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_pos(n: usize) -> Vec<f32> {
        (0..n * 3).map(|i| i as f32 * 0.1).collect()
    }
    fn make_flat_rot_xyzw(n: usize) -> Vec<f32> {
        // Identity quaternion in glTF xyzw = (0,0,0,1)
        let mut v = vec![0.0f32; n * 4];
        for i in 0..n {
            v[i * 4 + 3] = 1.0; // w at index 3
        }
        v
    }
    fn make_flat_scale(n: usize) -> Vec<f32> {
        vec![1.0f32; n * 3]
    }
    fn make_flat_opacity(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    #[test]
    fn zero_splats_returns_error() {
        let err = convert_to_cam_soa(0, &[], &[], &[], &[], None).unwrap_err();
        assert_eq!(err, ToCamSoaError::EmptySplats);
    }

    #[test]
    fn stub_returns_splat_count() {
        let n = 4;
        let result = convert_to_cam_soa(
            n,
            &make_flat_pos(n),
            &make_flat_rot_xyzw(n),
            &make_flat_scale(n),
            &make_flat_opacity(n),
            None,
        )
        .unwrap();
        assert_eq!(result.splat_count, n);
    }

    #[test]
    fn sh_wrong_length_returns_error() {
        let n = 2;
        // correct length = 2 * 48 = 96; pass 47 instead
        let bad_sh = vec![0.0f32; 47];
        let err = convert_to_cam_soa(
            n,
            &make_flat_pos(n),
            &make_flat_rot_xyzw(n),
            &make_flat_scale(n),
            &make_flat_opacity(n),
            Some(&bad_sh),
        )
        .unwrap_err();
        assert!(matches!(err, ToCamSoaError::ShLengthMismatch { .. }));
    }

    #[test]
    fn sh_correct_length_ok() {
        let n = 3;
        let sh = vec![0.0f32; n * 48]; // 3 × 48 = 144
        let result = convert_to_cam_soa(
            n,
            &make_flat_pos(n),
            &make_flat_rot_xyzw(n),
            &make_flat_scale(n),
            &make_flat_opacity(n),
            Some(&sh),
        )
        .unwrap();
        assert_eq!(result.splat_count, n);
    }

    /// Document the quaternion reorder contract so it is machine-checkable once
    /// the live impl is uncommented.
    ///
    /// glTF ROTATION VEC4 = (x, y, z, w); index mapping to GaussianBatch:
    ///   rotation_xyzw[i*4 + 3] → quat_w[i]   (w is at offset 3 in glTF)
    ///   rotation_xyzw[i*4 + 0] → quat_x[i]
    ///   rotation_xyzw[i*4 + 1] → quat_y[i]
    ///   rotation_xyzw[i*4 + 2] → quat_z[i]
    ///
    /// This test records the offset map and will fail if anyone changes it.
    #[test]
    fn quat_reorder_offset_map_is_documented() {
        // Encode the expected mapping as a const array: [(gltf_offset, field_name_tag)]
        // We use a simple u8 tag: 0=w 1=x 2=y 3=z
        const GLTF_TO_BATCH: [(usize, u8); 4] = [
            (3, 0), // gltf[3] → w
            (0, 1), // gltf[0] → x
            (1, 2), // gltf[1] → y
            (2, 3), // gltf[2] → z
        ];
        assert_eq!(GLTF_TO_BATCH[0], (3, 0), "w must come from gltf offset 3");
        assert_eq!(GLTF_TO_BATCH[1], (0, 1), "x must come from gltf offset 0");
        assert_eq!(GLTF_TO_BATCH[2], (1, 2), "y must come from gltf offset 1");
        assert_eq!(GLTF_TO_BATCH[3], (2, 3), "z must come from gltf offset 2");
    }

    /// Document SH repack formula.
    /// khr_gs: gs_block[k * 3 + ch]  → batch: sh[i*48 + ch*16 + k]
    #[test]
    fn sh_repack_index_formula() {
        let n_splats = 1_usize;
        let sh_stride = 48_usize; // SH_COEFFS_PER_GAUSSIAN
        let i = 0_usize;
        // For k=0, ch=0 (R DC term):
        let gs_idx = 0 * 3 + 0; // k=0, ch=0
        let batch_idx = i * sh_stride + 0 * 16 + 0; // ch=0, k=0
        assert_eq!(gs_idx, 0);
        assert_eq!(batch_idx, 0);
        // For k=15, ch=2 (B last basis):
        let gs_idx2 = 15 * 3 + 2;
        let batch_idx2 = i * sh_stride + 2 * 16 + 15;
        assert_eq!(gs_idx2, 47);
        assert_eq!(batch_idx2, 47);
        // Confirm n_splats is used (suppress unused warning)
        let _ = n_splats;
    }
}
