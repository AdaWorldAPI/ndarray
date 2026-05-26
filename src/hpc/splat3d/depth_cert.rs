//! Render-depth certification — per-splat depth confidence interval + error budget.
//!
//! # Mathematical claim
//!
//! A 3D Gaussian projected by the EWA pipeline (`super::project`) has a
//! camera-space depth `z = μ_cam.z` and a camera-space depth variance
//! `σ_z² = Σ_cam[2][2]` (the (2,2) entry of `W · Σ_world · Wᵀ`). The render
//! sees a **depth support interval**
//!
//! ```text
//!   z_center ± k · σ_z
//! ```
//!
//! where `k` is the confidence multiplier selected by the render budget.
//! The certificate keeps the seven error terms of the render-depth plan
//! separate for auditability:
//!
//! ```text
//!   E_depth_total =
//!       E_camera_transform
//!     + E_local_frame_quantization
//!     + E_covariance_projection      (= k · σ_z, the only term computed here)
//!     + E_splat_support_overlap      (set by TWIG / P3, neighbour-dependent)
//!     + E_sort_bucket_width          (= ½ · sort_bucket_width)
//!     + E_lod_substitution           (set when a coarser LOD was substituted)
//!     + E_sampling_discrepancy
//! ```
//!
//! All terms except `E_covariance_projection` are caller-supplied budgets
//! (`DepthCertParams`) so the certificate is auditable rather than opaque:
//! the codec supplies the quantization term, the sorter supplies the bucket
//! width, the cascade supplies the LOD/overlap terms.
//!
//! # Cesium knowledge transfer
//!
//! The screen-space-error helper [`screen_space_error`] is the production lift
//! of `crates/cesium::sse::sse_for_tile` (OGC 3D Tiles 1.1 §7.4):
//!
//! ```text
//!   sse = geometricError · viewportHeight / (distance · 2·tan(fovy/2))
//! ```
//!
//! For our pinhole camera `fy = (height/2) / tan(fovy/2)`, so
//! `viewportHeight / (2·tan(fovy/2)) = fy` and the formula collapses to
//! `sse = geometricError · fy / distance`. For a single splat the
//! `projected_radius_px` from the EWA projection *is* its screen footprint,
//! so SSE and projected radius are the same screen-space quantity — the
//! certificate carries both lenses.
//!
//! This is the scalar reference (validation plan Tier 1). The SIMD batch
//! path (P2) consumes [`super::project::ProjectedBatch`] and must match this
//! reference within tolerance.

use super::project::ProjectedBatch;
use super::spd3::Spd3;

/// Cesium / OGC 3D Tiles screen-space error for a feature of world-space
/// extent `geometric_error` seen at camera-space `distance`, lifted from
/// `cesium::sse`. `focal_y_px` is the pinhole focal length in pixels
/// (`fy`), which equals Cesium's `viewportHeight / (2·tan(fovy/2))` when the
/// viewport height equals the image height.
///
/// `distance` is clamped to a small epsilon to avoid division by zero
/// (Cesium clamps when the camera is inside the bounding volume).
#[inline]
pub fn screen_space_error(geometric_error: f32, focal_y_px: f32, distance: f32) -> f32 {
    let d = distance.max(1e-6);
    geometric_error * focal_y_px / d
}

/// Camera-space depth variance `Σ_cam[2][2] = w2 · Σ_world · w2ᵀ`, where
/// `w2` is the third row of the view matrix's upper-left 3×3 (the camera
/// look-axis in world space) and `Σ_world` is built from scale + quaternion.
///
/// For an identity view this is `scale_z²`; for a 90° rotation about +Y the
/// look-axis maps to world −X so it becomes `scale_x²` — matching the axis
/// permutation asserted in `project::tests::project_non_identity_view_rotation`.
///
/// `quat` is `[w, x, y, z]`; `view` is the row-major 4×4 from `Camera::view`.
pub fn camera_depth_variance(scale: [f32; 3], quat: [f32; 4], view: &[[f32; 4]; 4]) -> f32 {
    let s = Spd3::from_scale_quat(scale, quat).to_rows();
    let w2 = [view[2][0], view[2][1], view[2][2]];
    let mut acc = 0.0f32;
    for i in 0..3 {
        let mut row = 0.0f32;
        for j in 0..3 {
            row += s[i][j] * w2[j];
        }
        acc += w2[i] * row;
    }
    acc.max(0.0)
}

/// Caller-supplied per-splat error budgets. Each field maps to one term of
/// `E_depth_total` (see module docs); defaults are zero so a full-precision,
/// neighbour-free splat is certified purely on its covariance interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthCertParams {
    /// Confidence multiplier `k` for the depth interval `z ± k·σ_z`
    /// (k=2 ≈ 95% for a Gaussian; k=3 ≈ 99.7%).
    pub k_sigma: f32,
    /// `E_camera_transform` — float error of the world→view transform (world units).
    pub camera_transform_error: f32,
    /// `E_local_frame_quantization` — depth error from a quantized local frame.
    pub quantization_depth_error: f32,
    /// `E_splat_support_overlap` — neighbour support overlap (set by TWIG / P3).
    pub splat_support_overlap_error: f32,
    /// Depth sort-bucket width; contributes `½·width` as `E_sort_bucket_width`.
    pub sort_bucket_width: f32,
    /// `E_lod_substitution` — depth error from substituting a coarser LOD.
    pub lod_substitution_error: f32,
    /// `E_sampling_discrepancy` — sub-pixel / sampling depth error (world units).
    pub sampling_discrepancy: f32,
    /// Pass threshold: `passed = total_depth_error.is_finite() && total ≤ this`.
    pub max_total_depth_error: f32,
}

impl Default for DepthCertParams {
    fn default() -> Self {
        Self {
            k_sigma: 2.0,
            camera_transform_error: 0.0,
            quantization_depth_error: 0.0,
            splat_support_overlap_error: 0.0,
            sort_bucket_width: 0.0,
            lod_substitution_error: 0.0,
            sampling_discrepancy: 0.0,
            max_total_depth_error: f32::INFINITY,
        }
    }
}

/// Per-splat render-depth certificate. Field set is fixed by
/// `3DGS-render-depth-certification-plan.md`; terms are kept separate for
/// auditability rather than collapsed into one opaque score.
///
/// `#[repr(C)]` so it can cross the FFI boundary into `lance-graph` for tile
/// aggregation (per the error-certification-pillars plan: primitive splat
/// certificates live here, tile aggregation lives in `lance-graph`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderDepthCertificate {
    /// Near edge of the depth support interval, clamped to ≥ 0.
    pub min_depth: f32,
    /// Far edge of the depth support interval.
    pub max_depth: f32,
    /// Camera-space depth variance `σ_z² = Σ_cam[2][2]`.
    pub depth_variance: f32,
    /// 3σ screen-space radius (pixels), passed through from EWA projection.
    pub projected_radius_px: f32,
    /// Confidence the occlusion ordering is unambiguous (1 = no neighbour
    /// support overlaps the interval). Refined by TWIG (P3).
    pub occlusion_confidence: f32,
    /// Fraction of a depth sort-bucket the interval spans (0 = sub-bucket,
    /// 1 = spans a whole bucket or more → exact LEAF sort recommended).
    pub ordering_uncertainty: f32,
    /// `E_local_frame_quantization` (echoed for audit).
    pub quantization_depth_error: f32,
    /// `E_covariance_projection = k · σ_z` — the interval half-width.
    pub covariance_depth_error: f32,
    /// Sum of all seven `E_depth_total` terms.
    pub total_depth_error: f32,
    /// `true` iff `total_depth_error` is finite and within budget.
    pub passed: bool,
}

/// Scalar reference: certify one projected splat from its camera-space depth
/// `depth` (= μ_cam.z), camera-space depth variance `cov_zz` (= Σ_cam[2][2],
/// e.g. from [`camera_depth_variance`]), and `projected_radius_px` from the
/// EWA projection.
///
/// This is the deterministic reference the SIMD batch path must match.
pub fn certify_depth_scalar(
    depth: f32, cov_zz: f32, projected_radius_px: f32, params: &DepthCertParams,
) -> RenderDepthCertificate {
    let depth_variance = cov_zz.max(0.0);
    let sigma_z = depth_variance.sqrt();

    // E_covariance_projection — the only term we compute from the splat itself.
    let covariance_depth_error = params.k_sigma * sigma_z;
    // E_sort_bucket_width — half a bucket of depth-ordering granularity.
    let sort_bucket_term = 0.5 * params.sort_bucket_width;

    let total_depth_error = params.camera_transform_error
        + params.quantization_depth_error
        + covariance_depth_error
        + params.splat_support_overlap_error
        + sort_bucket_term
        + params.lod_substitution_error
        + params.sampling_discrepancy;

    let min_depth = (depth - covariance_depth_error).max(0.0);
    let max_depth = depth + covariance_depth_error;

    // Ordering uncertainty: how much the full interval (2·k·σ_z) blurs the
    // depth sort order, expressed as a fraction of one bucket.
    let ordering_uncertainty = if params.sort_bucket_width > 0.0 {
        (2.0 * covariance_depth_error / params.sort_bucket_width).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Occlusion confidence: 1.0 when no neighbour support overlaps; falls as
    // the support-overlap budget grows relative to our own depth spread.
    let occlusion_confidence = if params.splat_support_overlap_error > 0.0 {
        let denom = covariance_depth_error.max(1e-6);
        (1.0 - params.splat_support_overlap_error / denom).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let passed = total_depth_error.is_finite() && total_depth_error <= params.max_total_depth_error;

    RenderDepthCertificate {
        min_depth,
        max_depth,
        depth_variance,
        projected_radius_px,
        occlusion_confidence,
        ordering_uncertainty,
        quantization_depth_error: params.quantization_depth_error,
        covariance_depth_error,
        total_depth_error,
        passed,
    }
}

/// Certify every valid splat of a projected batch into `out` (one certificate
/// per active slot; culled slots get a zeroed, `passed = false` certificate).
///
/// `depth_var[i]` is the camera-space depth variance `Σ_cam[2][2]` for splat
/// `i` (see [`camera_depth_variance`]); it parallels `batch.depth`. This
/// scalar driver is the reference the P2 SIMD reduction must match; it does
/// not panic on malformed splats — culled lanes are simply not certified.
pub fn certify_batch_scalar(
    batch: &ProjectedBatch, depth_var: &[f32], params: &DepthCertParams, out: &mut Vec<RenderDepthCertificate>,
) {
    out.clear();
    out.reserve(batch.len);
    for i in 0..batch.len {
        if batch.valid[i] == 0 {
            out.push(RenderDepthCertificate {
                min_depth: 0.0,
                max_depth: 0.0,
                depth_variance: 0.0,
                projected_radius_px: 0.0,
                occlusion_confidence: 0.0,
                ordering_uncertainty: 0.0,
                quantization_depth_error: 0.0,
                covariance_depth_error: 0.0,
                total_depth_error: 0.0,
                passed: false,
            });
            continue;
        }
        let cov_zz = depth_var.get(i).copied().unwrap_or(0.0);
        out.push(certify_depth_scalar(batch.depth[i], cov_zz, batch.radius[i], params));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    const IDENTITY_VIEW: [[f32; 4]; 4] =
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

    // ── camera_depth_variance ────────────────────────────────────────────────

    #[test]
    fn depth_variance_identity_view_is_scale_z_squared() {
        // Identity view, identity quat → Σ_world = diag(sx², sy², sz²),
        // and the look-axis is +Z → Σ_cam[2][2] = sz².
        let cov = camera_depth_variance([2.0, 3.0, 0.5], [1.0, 0.0, 0.0, 0.0], &IDENTITY_VIEW);
        assert!(approx(cov, 0.25, 1e-5), "expected scale_z²=0.25, got {cov}");
    }

    #[test]
    fn depth_variance_y_rotation_permutes_axis() {
        // R_y(90°) view: row 2 = [-1, 0, 0] → look-axis maps to world −X →
        // Σ_cam[2][2] = scale_x². scale=[2,1,0.5] → 4.0. Mirrors the
        // analytical project.rs rotation test.
        let view = [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let cov = camera_depth_variance([2.0, 1.0, 0.5], [1.0, 0.0, 0.0, 0.0], &view);
        assert!(approx(cov, 4.0, 1e-4), "expected scale_x²=4.0 after 90° Y, got {cov}");
    }

    // ── screen_space_error (Cesium lift) ──────────────────────────────────────

    #[test]
    fn sse_matches_cesium_formula() {
        // fy as denominator: sse = geometric_error * fy / distance.
        let sse = screen_space_error(20.0, 935.307, 500.0);
        // 20 * 935.307 / 500 = 37.412 — same number the cesium::sse doctest asserts.
        assert!(approx(sse, 37.412, 0.01), "sse={sse}");
    }

    #[test]
    fn sse_inverse_in_distance_linear_in_error() {
        let near = screen_space_error(10.0, 512.0, 100.0);
        let far = screen_space_error(10.0, 512.0, 200.0);
        assert!(approx(near, 2.0 * far, 1e-3), "near={near} far={far}");
        let small = screen_space_error(10.0, 512.0, 100.0);
        let big = screen_space_error(20.0, 512.0, 100.0);
        assert!(approx(big, 2.0 * small, 1e-3), "small={small} big={big}");
    }

    #[test]
    fn sse_zero_distance_is_finite() {
        assert!(screen_space_error(10.0, 512.0, 0.0).is_finite());
    }

    // ── certify_depth_scalar ──────────────────────────────────────────────────

    #[test]
    fn certificate_depth_interval_is_z_plus_minus_k_sigma() {
        // cov_zz = 0.25 → σ_z = 0.5; k=2 → half-width = 1.0; depth=10.
        let params = DepthCertParams { k_sigma: 2.0, ..Default::default() };
        let c = certify_depth_scalar(10.0, 0.25, 7.0, &params);
        assert!(approx(c.depth_variance, 0.25, 1e-6));
        assert!(approx(c.covariance_depth_error, 1.0, 1e-6), "cov err={}", c.covariance_depth_error);
        assert!(approx(c.min_depth, 9.0, 1e-6), "min={}", c.min_depth);
        assert!(approx(c.max_depth, 11.0, 1e-6), "max={}", c.max_depth);
        assert!(approx(c.projected_radius_px, 7.0, 1e-6));
        assert!(c.passed, "no budget → infinite threshold → must pass");
    }

    #[test]
    fn certificate_min_depth_clamped_non_negative() {
        // depth 0.5, half-width 1.0 → raw min = -0.5, clamped to 0.
        let params = DepthCertParams { k_sigma: 2.0, ..Default::default() };
        let c = certify_depth_scalar(0.5, 0.25, 1.0, &params);
        assert_eq!(c.min_depth, 0.0, "min_depth must clamp to 0");
    }

    #[test]
    fn certificate_total_is_sum_of_separate_terms() {
        // Distinct values per term so a wiring mistake is visible.
        let params = DepthCertParams {
            k_sigma: 2.0,                       // σ_z=0.5 → cov term = 1.0
            camera_transform_error: 0.1,
            quantization_depth_error: 0.2,
            splat_support_overlap_error: 0.4,
            sort_bucket_width: 0.6,             // → 0.3
            lod_substitution_error: 0.8,
            sampling_discrepancy: 1.6,
            max_total_depth_error: f32::INFINITY,
        };
        let c = certify_depth_scalar(10.0, 0.25, 5.0, &params);
        let expected = 0.1 + 0.2 + 1.0 + 0.4 + 0.3 + 0.8 + 1.6;
        assert!(approx(c.total_depth_error, expected, 1e-5), "total={} expected={expected}", c.total_depth_error);
        assert!(approx(c.quantization_depth_error, 0.2, 1e-6));
        assert!(approx(c.covariance_depth_error, 1.0, 1e-6));
    }

    #[test]
    fn certificate_pass_fail_tracks_threshold() {
        let mut params = DepthCertParams { k_sigma: 2.0, max_total_depth_error: 0.5, ..Default::default() };
        // cov term alone = 1.0 > 0.5 → fail.
        assert!(!certify_depth_scalar(10.0, 0.25, 5.0, &params).passed);
        // Loosen the threshold → pass.
        params.max_total_depth_error = 2.0;
        assert!(certify_depth_scalar(10.0, 0.25, 5.0, &params).passed);
    }

    #[test]
    fn certificate_non_finite_fails() {
        let params = DepthCertParams { camera_transform_error: f32::INFINITY, ..Default::default() };
        let c = certify_depth_scalar(10.0, 0.25, 5.0, &params);
        assert!(!c.passed, "non-finite total must not pass");
        assert!(!c.total_depth_error.is_finite());
    }

    #[test]
    fn ordering_uncertainty_scales_inverse_with_bucket() {
        // half-width = 1.0 → full interval = 2.0.
        let wide = DepthCertParams { k_sigma: 2.0, sort_bucket_width: 8.0, ..Default::default() };
        let narrow = DepthCertParams { k_sigma: 2.0, sort_bucket_width: 4.0, ..Default::default() };
        let cw = certify_depth_scalar(10.0, 0.25, 5.0, &wide);
        let cn = certify_depth_scalar(10.0, 0.25, 5.0, &narrow);
        assert!(approx(cw.ordering_uncertainty, 0.25, 1e-6), "wide={}", cw.ordering_uncertainty);
        assert!(approx(cn.ordering_uncertainty, 0.5, 1e-6), "narrow={}", cn.ordering_uncertainty);
    }

    #[test]
    fn ordering_uncertainty_saturates_at_one() {
        let p = DepthCertParams { k_sigma: 2.0, sort_bucket_width: 0.1, ..Default::default() };
        let c = certify_depth_scalar(10.0, 0.25, 5.0, &p);
        assert!(approx(c.ordering_uncertainty, 1.0, 1e-6), "should clamp to 1, got {}", c.ordering_uncertainty);
    }

    #[test]
    fn occlusion_confidence_high_without_overlap_drops_with_overlap() {
        let isolated = DepthCertParams { k_sigma: 2.0, ..Default::default() };
        let c0 = certify_depth_scalar(10.0, 0.25, 5.0, &isolated);
        assert!(approx(c0.occlusion_confidence, 1.0, 1e-6), "isolated splat should be fully confident");

        // overlap = half the depth-spread half-width (1.0) → confidence 0.5.
        let overlap = DepthCertParams { k_sigma: 2.0, splat_support_overlap_error: 0.5, ..Default::default() };
        let c1 = certify_depth_scalar(10.0, 0.25, 5.0, &overlap);
        assert!(approx(c1.occlusion_confidence, 0.5, 1e-5), "got {}", c1.occlusion_confidence);
        assert!((0.0..=1.0).contains(&c1.occlusion_confidence));
    }
}
