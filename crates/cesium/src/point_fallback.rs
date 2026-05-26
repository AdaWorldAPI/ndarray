//! `point_fallback` (group D) — point-cloud fallback renderer.
//!
//! When a [`GaussianBatch`] cannot be rendered as full 3DGS splats (e.g. the
//! receiver only supports the **KHR_gaussian_splatting base extension**, which
//! mandates point-primitive output for coarse-tier HHTL preview), this module
//! extracts the mean positions `(x, y, z)` from the SoA buffers and emits a
//! flat `[f32]` point cloud suitable for upload as a KHR point-primitive mesh.
//!
//! # KHR_gaussian_splatting base extension
//!
//! The Khronos base extension (as of 2024-Q3 draft) requires the runtime to
//! fall back to rendering splats as point primitives when the implementation
//! does not support the full covariance / SH pipeline.  The HHTL coarse-tier
//! preview uses exactly this path: positions only, no colour or opacity.
//!
//! # What this module does NOT do
//!
//! - Project positions onto screen (that's `splat3d::project`).
//! - Emit colour / opacity / SH — point primitives carry `POSITION` only in
//!   the base extension.  A colour channel could be synthesised from the DC
//!   SH term (`sh[0]` per channel) but is left `// DEFERRED:` until the
//!   extension spec stabilises.
//!
//! # Implementation status
//!
//! **Commented scaffold only.**  No live impl that touches ndarray exists here;
//! `ndarray` is commented out in `Cargo.toml`.  The stub structs and unit
//! tests below are `std`-only and compile clean.
//!
//! # Source grounding
//!
//! Symbols cited from the local codebase (read before writing, not fabricated):
//! - `GaussianBatch` — `src/hpc/splat3d/gaussian.rs` (fields `mean_x`,
//!   `mean_y`, `mean_z`, `len`, `capacity`).
//! - `read_ply` — `src/hpc/splat3d/ply.rs` (loader entry point that produces
//!   `GaussianBatch` from an Inria binary-PLY reader).

// ─────────────────────────────────────────────────────────────────────────────
// Stub types (live — std-only, no ndarray dep)
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the point-cloud fallback extraction.
///
/// Flat interleaved XYZ: `[x0, y0, z0, x1, y1, z1, …]`, length `3 * count`.
/// Suitable for upload as a KHR_gaussian_splatting point-primitive `POSITION`
/// accessor (component type `FLOAT`, count = `point_count`).
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud {
    /// Flat XYZ buffer: length `3 * point_count`.
    pub xyz: Vec<f32>,
    /// Number of points (= `GaussianBatch::len` of the source batch).
    pub point_count: usize,
}

impl PointCloud {
    /// Return the position of point `i` as `[x, y, z]`.
    ///
    /// Panics if `i >= point_count`.
    pub fn position(&self, i: usize) -> [f32; 3] {
        assert!(i < self.point_count, "PointCloud::position: index {i} >= point_count {}", self.point_count);
        let base = i * 3;
        [self.xyz[base], self.xyz[base + 1], self.xyz[base + 2]]
    }

    /// Byte length of the XYZ buffer (for KHR accessor `byteLength`).
    pub fn byte_length(&self) -> usize {
        self.xyz.len() * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Commented scaffold — live impl blocked on ndarray dep
// ─────────────────────────────────────────────────────────────────────────────

// DEFERRED: uncomment when `ndarray = { workspace = true }` is re-enabled in
// Cargo.toml and the module goes live post Opus + CodeRabbit review.
//
// use ndarray_hpc::hpc::splat3d::gaussian::GaussianBatch;
//
// /// Extract mean positions from a [`GaussianBatch`] into a [`PointCloud`].
// ///
// /// Reads `batch.mean_x`, `batch.mean_y`, `batch.mean_z` (SoA, len =
// /// `batch.len`) and interleaves them into a flat `[x, y, z, …]` buffer.
// ///
// /// # Complexity
// ///
// /// O(n) time, O(3n) extra allocation.  No SIMD — this is a cold-path
// /// extraction, not a hot-path accumulation.
// ///
// /// # KHR accessor layout
// ///
// /// The output xyz buffer maps directly to a glTF `POSITION` accessor:
// /// ```text
// /// accessors[k] = {
// ///   bufferView: <view into xyz>,
// ///   componentType: 5126,  // FLOAT
// ///   type: "VEC3",
// ///   count: point_count,
// /// }
// /// ```
// pub fn extract_point_cloud(batch: &GaussianBatch) -> PointCloud {
//     let n = batch.len;  // GaussianBatch::len — active gaussian count
//     let mut xyz = Vec::with_capacity(3 * n);
//     for i in 0..n {
//         // SoA fields: batch.mean_x[i], batch.mean_y[i], batch.mean_z[i]
//         // (confirmed from gaussian.rs — pub Vec<f32> per axis)
//         xyz.push(batch.mean_x[i]);
//         xyz.push(batch.mean_y[i]);
//         xyz.push(batch.mean_z[i]);
//     }
//     PointCloud { xyz, point_count: n }
// }
//
// // DEFERRED: synthesise colour from DC SH term once KHR base-ext colour
// // semantics stabilise.
// //
// // The DC SH coefficient for channel R is `batch.sh[i * 48 + 0]`, G at
// // `i * 48 + 16`, B at `i * 48 + 32` (SH_COEFFS_PER_CHANNEL = 16 from
// // gaussian.rs).  The Inria colour offset (+0.5) must be applied before
// // clamping to [0, 1]:
// //   color_r = (SH_C0 * sh_r + 0.5).clamp(0.0, 1.0)
// // where SH_C0 ≈ 0.282095 (from sh.rs::SH_C0).
// //
// // UNVERIFIED: KHR point-primitive extension colour accessor name — may be
// // COLOR_0 (glTF standard) or a custom attribute.

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (live — std-only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point_cloud(xyzs: &[[f32; 3]]) -> PointCloud {
        let mut xyz = Vec::with_capacity(xyzs.len() * 3);
        for p in xyzs {
            xyz.extend_from_slice(p);
        }
        PointCloud {
            xyz,
            point_count: xyzs.len(),
        }
    }

    #[test]
    fn point_cloud_position_round_trips() {
        let pts = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 0.0]];
        let pc = make_point_cloud(&pts);
        assert_eq!(pc.point_count, 3);
        for (i, expected) in pts.iter().enumerate() {
            let got = pc.position(i);
            assert_eq!(got, *expected, "position({i}) mismatch");
        }
    }

    #[test]
    fn point_cloud_byte_length() {
        let pc = make_point_cloud(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        // 2 points × 3 floats × 4 bytes = 24
        assert_eq!(pc.byte_length(), 24);
    }

    #[test]
    fn point_cloud_empty() {
        let pc = PointCloud {
            xyz: vec![],
            point_count: 0,
        };
        assert_eq!(pc.byte_length(), 0);
        assert_eq!(pc.point_count, 0);
    }

    #[test]
    #[should_panic]
    fn point_cloud_position_out_of_bounds_panics() {
        let pc = make_point_cloud(&[[1.0, 2.0, 3.0]]);
        let _ = pc.position(1); // must panic
    }
}
