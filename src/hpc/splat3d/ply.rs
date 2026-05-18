//! Minimal PLY reader for the 3DGS canonical scene format.
//!
//! The Inria 3D Gaussian Splatting format ships scenes as binary PLY
//! files with a documented vertex layout (see Kerbl 2023 §3.2):
//!
//! ```text
//! property float x
//! property float y
//! property float z
//! property float nx, ny, nz       (unused — normals from training)
//! property float f_dc_0, f_dc_1, f_dc_2     (SH degree 0 RGB, 3 floats)
//! property float f_rest_0 ... f_rest_44     (SH degrees 1-3, 45 floats)
//! property float opacity                     (logit-space; needs sigmoid)
//! property float scale_0, scale_1, scale_2  (log-space; needs exp)
//! property float rot_0, rot_1, rot_2, rot_3 (quaternion w,x,y,z; needs normalize)
//! ```
//!
//! Total per-vertex: 62 f32 = 248 bytes. For 500K-1M-gaussian scenes
//! this is ~125-250 MB on disk.
//!
//! # What this reader does
//!
//! - Parses the ASCII header up to `end_header\n`.
//! - Validates that the vertex layout matches the Inria spec
//!   (`x, y, z, nx, ny, nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*`).
//! - Reads the binary little-endian body into a [`GaussianBatch`].
//! - Applies the activation transforms inline: `sigmoid(opacity)`,
//!   `exp(scale_*)`, `normalize(quat)`.
//! - Reorders the SH coefficients into the gaussian-major,
//!   channel-major layout that [`crate::hpc::splat3d::sh::sh_eval_deg3`]
//!   expects: `sh[g * 48 + ch * 16 + basis_k]`.
//!
//! The Inria PLY stores SH as: 3 DC coeffs first (RGB), then 45 rest
//! coeffs interleaved AS `f_rest_0 = R_basis1, f_rest_1 = R_basis2, …,
//! f_rest_14 = R_basis15, f_rest_15 = G_basis1, …`. So the on-disk
//! layout is channel-major (all R coeffs, then all G, then all B);
//! our internal layout matches that — `sh[ch * 16 + 0..16]` for
//! channel ch — so the reorder is just slot-by-slot copy.
//!
//! # What this reader does NOT do
//!
//! - ASCII PLY bodies (the Inria scenes are always binary; ASCII
//!   variants are rejected with `PlyError::AsciiUnsupported`).
//! - Big-endian byte order.
//! - PLY files with EXTRA properties (camera intrinsics, custom
//!   tags). The spec must match exactly; deviations return
//!   `PlyError::UnexpectedProperty`.
//! - Streaming / memory-mapped reads. The full file is buffered.
//!   For 1 GB scenes use a memory-mapped variant in a follow-up.

use std::io::{BufRead, BufReader, Read};

use crate::hpc::splat3d::gaussian::{
    GaussianBatch, SH_COEFFS_PER_CHANNEL, SH_COEFFS_PER_GAUSSIAN,
};

/// Errors the PLY reader can return.
#[derive(Debug)]
pub enum PlyError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// File doesn't start with the `ply\n` magic.
    NotPly,
    /// Format line says something other than `binary_little_endian 1.0`.
    AsciiUnsupported,
    /// Unknown / big-endian format.
    UnsupportedFormat(String),
    /// Vertex element missing or wrong count.
    BadElement(String),
    /// A property in the header doesn't match the expected Inria layout.
    UnexpectedProperty(String),
    /// Body is shorter than the header claimed.
    Truncated,
}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        PlyError::Io(e)
    }
}

/// Expected property names in order. Total = 3 + 3 + 3 + 45 + 1 + 3 + 4 = 62.
fn expected_properties() -> Vec<&'static str> {
    let mut v = vec!["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"];
    for k in 0..45 {
        v.push(Box::leak(format!("f_rest_{k}").into_boxed_str()));
    }
    v.push("opacity");
    v.push("scale_0");
    v.push("scale_1");
    v.push("scale_2");
    v.push("rot_0");
    v.push("rot_1");
    v.push("rot_2");
    v.push("rot_3");
    v
}

/// Per-vertex float count = 62.
pub const PROPERTIES_PER_VERTEX: usize = 62;

/// Read a PLY file (Inria 3DGS canonical layout) into a `GaussianBatch`.
///
/// The reader applies the canonical activation transforms inline:
/// - `opacity = sigmoid(opacity_logit)`
/// - `scale = exp(scale_log)` per axis
/// - `quat = normalize(rot_0..3)`
///
/// SH coefficients are stored verbatim in the gaussian-major,
/// channel-major layout. Caller is responsible for whatever further
/// rotation / color-space conversion the downstream renderer needs.
pub fn read_ply<R: Read>(reader: R) -> Result<GaussianBatch, PlyError> {
    let mut buf = BufReader::new(reader);
    let mut line = String::new();

    // First line: "ply"
    line.clear();
    buf.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(PlyError::NotPly);
    }

    // Header parse until "end_header".
    let mut format_seen = false;
    let mut n_vertices: usize = 0;
    let mut properties: Vec<String> = Vec::new();

    loop {
        line.clear();
        let n = buf.read_line(&mut line)?;
        if n == 0 {
            return Err(PlyError::BadElement(
                "header ended without end_header".to_string(),
            ));
        }
        let trimmed = line.trim();
        if trimmed == "end_header" {
            break;
        }
        if let Some(fmt) = trimmed.strip_prefix("format ") {
            if fmt.starts_with("ascii") {
                return Err(PlyError::AsciiUnsupported);
            }
            if !fmt.starts_with("binary_little_endian") {
                return Err(PlyError::UnsupportedFormat(fmt.to_string()));
            }
            format_seen = true;
        } else if let Some(elem) = trimmed.strip_prefix("element vertex ") {
            n_vertices = elem
                .parse()
                .map_err(|_| PlyError::BadElement(format!("vertex count: {elem}")))?;
        } else if let Some(prop) = trimmed.strip_prefix("property float ") {
            properties.push(prop.to_string());
        } else if trimmed.starts_with("element ") || trimmed.starts_with("property ") {
            return Err(PlyError::UnexpectedProperty(trimmed.to_string()));
        }
        // Comments and other lines are silently ignored.
    }

    if !format_seen {
        return Err(PlyError::UnsupportedFormat("no format line".to_string()));
    }
    if n_vertices == 0 {
        return Err(PlyError::BadElement("vertex count = 0".to_string()));
    }

    // Validate the property list matches the Inria spec exactly.
    let expected = expected_properties();
    if properties.len() != expected.len() {
        return Err(PlyError::UnexpectedProperty(format!(
            "expected {} properties, got {}",
            expected.len(),
            properties.len()
        )));
    }
    for (actual, exp) in properties.iter().zip(expected.iter()) {
        if actual != exp {
            return Err(PlyError::UnexpectedProperty(format!(
                "expected `{exp}`, got `{actual}`"
            )));
        }
    }

    // Read the binary body — n_vertices × 62 f32 little-endian.
    let mut bytes = vec![0u8; n_vertices * PROPERTIES_PER_VERTEX * 4];
    buf.read_exact(&mut bytes).map_err(|_| PlyError::Truncated)?;

    // Convert into a GaussianBatch with activations applied.
    let mut batch = GaussianBatch::with_capacity(n_vertices);
    let stride = PROPERTIES_PER_VERTEX * 4;
    for i in 0..n_vertices {
        let base = i * stride;
        let mut read_f32 = |offset: usize| -> f32 {
            let s = base + offset * 4;
            f32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]])
        };

        // x, y, z at offsets 0, 1, 2. nx, ny, nz at 3, 4, 5 (skipped).
        let mean_x = read_f32(0);
        let mean_y = read_f32(1);
        let mean_z = read_f32(2);
        // f_dc_0..2 at offsets 6, 7, 8 — these are channel-0 SH coeff 0
        // for R, G, B respectively.
        let dc_r = read_f32(6);
        let dc_g = read_f32(7);
        let dc_b = read_f32(8);
        // f_rest_0..44 at offsets 9..54. Inria layout is channel-major:
        //   f_rest_0..14   = R basis 1..15
        //   f_rest_15..29  = G basis 1..15
        //   f_rest_30..44  = B basis 1..15
        let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
        sh[0] = dc_r;
        sh[SH_COEFFS_PER_CHANNEL] = dc_g;
        sh[2 * SH_COEFFS_PER_CHANNEL] = dc_b;
        for k in 0..15 {
            sh[1 + k] = read_f32(9 + k);
            sh[SH_COEFFS_PER_CHANNEL + 1 + k] = read_f32(9 + 15 + k);
            sh[2 * SH_COEFFS_PER_CHANNEL + 1 + k] = read_f32(9 + 30 + k);
        }
        // opacity at offset 54 (logit).
        let opacity_logit = read_f32(54);
        let opacity = 1.0 / (1.0 + (-opacity_logit).exp());
        // scale_0..2 at offsets 55, 56, 57 (log-space).
        let scale = [
            read_f32(55).exp(),
            read_f32(56).exp(),
            read_f32(57).exp(),
        ];
        // rot_0..3 at offsets 58, 59, 60, 61 (w, x, y, z; normalize).
        let mut quat = [read_f32(58), read_f32(59), read_f32(60), read_f32(61)];
        let qn = (quat[0] * quat[0]
            + quat[1] * quat[1]
            + quat[2] * quat[2]
            + quat[3] * quat[3])
            .sqrt()
            .max(1e-12);
        for q in &mut quat {
            *q /= qn;
        }

        batch.push(crate::hpc::splat3d::gaussian::Gaussian3D {
            mean: [mean_x, mean_y, mean_z],
            scale,
            quat,
            opacity,
            sh,
        });
    }
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build_minimal_ply_bytes(n: usize) -> Vec<u8> {
        let mut header = String::new();
        header.push_str("ply\n");
        header.push_str("format binary_little_endian 1.0\n");
        header.push_str(&format!("element vertex {n}\n"));
        for p in &expected_properties() {
            header.push_str(&format!("property float {p}\n"));
        }
        header.push_str("end_header\n");

        let mut bytes = header.into_bytes();
        for i in 0..n {
            for j in 0..PROPERTIES_PER_VERTEX {
                // Distinct value per (vertex, property) so tests can verify
                // the right offsets get read.
                let v = (i * 100 + j) as f32 * 0.01;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn rejects_non_ply_magic() {
        let result = read_ply(Cursor::new(b"not a ply file"));
        match result {
            Err(PlyError::NotPly) => {}
            Ok(_) => panic!("expected NotPly, got Ok(batch)"),
            Err(e) => panic!("expected NotPly, got {e:?}"),
        }
    }

    #[test]
    fn rejects_ascii_format() {
        let bytes = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
        match read_ply(Cursor::new(bytes)) {
            Err(PlyError::AsciiUnsupported) => {}
            Ok(_) => panic!("expected AsciiUnsupported, got Ok(batch)"),
            Err(e) => panic!("expected AsciiUnsupported, got {e:?}"),
        }
    }

    #[test]
    fn reads_minimal_2_vertex_ply() {
        let bytes = build_minimal_ply_bytes(2);
        let batch = read_ply(Cursor::new(bytes)).expect("read_ply failed");
        assert_eq!(batch.len, 2);
        // Vertex 0: x=0.00, y=0.01, z=0.02
        assert!((batch.mean_x[0] - 0.0).abs() < 1e-6);
        assert!((batch.mean_y[0] - 0.01).abs() < 1e-6);
        assert!((batch.mean_z[0] - 0.02).abs() < 1e-6);
        // Vertex 1: x=1.00, y=1.01, z=1.02
        assert!((batch.mean_x[1] - 1.0).abs() < 1e-6);
        assert!((batch.mean_y[1] - 1.01).abs() < 1e-6);
        assert!((batch.mean_z[1] - 1.02).abs() < 1e-6);
        // Opacity activation: sigmoid(0.54) = 1/(1+exp(-0.54)) ≈ 0.632
        let opacity_logit = 0.54f32;
        let expected_opacity = 1.0 / (1.0 + (-opacity_logit).exp());
        assert!(
            (batch.opacity[0] - expected_opacity).abs() < 1e-5,
            "expected sigmoid({opacity_logit}) = {expected_opacity}, got {}",
            batch.opacity[0]
        );
        // Scale activation: exp(0.55) ≈ 1.733
        let expected_scale_0 = 0.55f32.exp();
        assert!(
            (batch.scale_x[0] - expected_scale_0).abs() < 1e-5,
            "expected exp(0.55) = {expected_scale_0}, got {}",
            batch.scale_x[0]
        );
        // Quat normalization: components are (0.58, 0.59, 0.60, 0.61)
        // norm = sqrt(0.58² + 0.59² + 0.60² + 0.61²) ≈ 1.190
        let qn = (0.58_f32.powi(2) + 0.59_f32.powi(2) + 0.60_f32.powi(2) + 0.61_f32.powi(2))
            .sqrt();
        assert!(
            (batch.quat_w[0] - 0.58 / qn).abs() < 1e-5,
            "quat_w[0] = {}, expected {}", batch.quat_w[0], 0.58 / qn
        );
    }

    #[test]
    fn rejects_unexpected_property() {
        let mut bytes = b"ply\nformat binary_little_endian 1.0\n\
                          element vertex 1\n\
                          property float x\n\
                          property float foo\n\
                          end_header\n"
            .to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        match read_ply(Cursor::new(bytes)) {
            Err(PlyError::UnexpectedProperty(_)) => {}
            Ok(_) => panic!("expected UnexpectedProperty, got Ok(batch)"),
            Err(e) => panic!("expected UnexpectedProperty, got {e:?}"),
        }
    }
}
