//! `sse` (group C) — screen-space-error (SSE) LOD selection.
//!
//! Implements the OGC 3D Tiles / Cesium SSE formula used to decide whether a
//! tile's geometric error is small enough at the current camera to be accepted
//! as a leaf (no further refinement needed):
//!
//! ```text
//! sse = geometricError * viewportHeight / (distance * 2 * tan(fovy / 2))
//! ```
//!
//! A tile (or HLOD cluster) is accepted when `sse ≤ maximumScreenSpaceError`
//! (Cesium default: 16.0 pixels).
//!
//! # Grounding
//! - OGC 3D Tiles 1.1 spec §7.4 "Screen-Space Error"
//! - CesiumGS/cesium `Cesium3DTileset.maximumScreenSpaceError` (default 16)
//! - CesiumGS/cesium `Cesium3DTile._screenSpaceError` computation
//!
//! # UNVERIFIED items
//! - Whether Cesium additionally applies a `sseDenominatorFactor` or a
//!   per-tile content-error multiplier in newer 3D Tiles 1.1 revisions.
//! - Exact behaviour when `distance ≈ 0` (camera inside tile bounding volume);
//!   CesiumGS clamps to some minimum distance — exact value unconfirmed.
//!
//! # Commented scaffold only
//! All impl is `//`-commented until reviewed live by Opus + CodeRabbit.
//! The compilable stub types and their unit tests are live (std-only).

// ─────────────────────────────────────────────────────────────────────────────
// Compilable stub types (std-only)
// ─────────────────────────────────────────────────────────────────────────────

/// Camera frustum parameters needed for SSE computation.
///
/// # Fields
/// - `viewport_height_px` — render target height in pixels (e.g. 1080.0).
/// - `fovy_rad` — vertical field-of-view in radians (e.g. π/3 ≈ 1.047 for 60°).
///
/// # Pre-computation
/// The hot-path helper [`SseFrustum::sse_denominator`] pre-computes
/// `2 * tan(fovy/2)` once per frame so the per-tile call is a
/// multiply-divide only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SseFrustum {
    /// Render target height in pixels.
    pub viewport_height_px: f32,
    /// Vertical field-of-view in radians.
    pub fovy_rad: f32,
}

impl SseFrustum {
    /// Pre-compute the constant denominator factor `viewportHeight / (2 * tan(fovy/2))`.
    ///
    /// Cache this value for the duration of a frame; call [`sse_for_tile`] with
    /// the result to avoid recomputing `tan` per tile.
    #[inline]
    pub fn sse_denominator(self) -> f32 {
        // 2 * tan(fovy / 2)  — the angular subtension of the viewport
        let two_tan_half_fov = 2.0 * (self.fovy_rad * 0.5).tan();
        // UNVERIFIED: guard against zero fovy (degenerate frustum)
        if two_tan_half_fov.abs() < f32::EPSILON {
            return f32::MAX;
        }
        self.viewport_height_px / two_tan_half_fov
    }
}

/// Compute the screen-space error for one tile.
///
/// # Arguments
/// - `geometric_error` — tile's `geometricError` from `tileset.json` (world units).
/// - `distance` — distance from camera to tile bounding volume (world units,
///   must be positive).  Values ≤ 0 are clamped to a small epsilon to avoid
///   division by zero.
/// - `sse_denominator` — pre-computed `viewportHeight / (2 * tan(fovy/2))`;
///   obtain once per frame from [`SseFrustum::sse_denominator`].
///
/// # Returns
/// Screen-space error in pixels.  Compare against
/// `maximumScreenSpaceError` (Cesium default 16.0) to decide refinement.
///
/// # Formula
/// ```text
/// sse = geometricError * (viewportHeight / (distance * 2 * tan(fovy/2)))
///     = geometricError * sse_denominator / distance
/// ```
#[inline]
pub fn sse_for_tile(geometric_error: f32, distance: f32, sse_denominator: f32) -> f32 {
    // Clamp distance to avoid division by zero; UNVERIFIED: Cesium's exact
    // minimum-distance clamp value.
    let d = distance.max(1e-6_f32);
    geometric_error * sse_denominator / d
}

/// Returns `true` when the tile meets the LOD acceptance criterion.
///
/// Equivalent to `sse_for_tile(...) <= maximum_sse`.
#[inline]
pub fn tile_meets_sse(geometric_error: f32, distance: f32, sse_denominator: f32, maximum_sse: f32) -> bool {
    sse_for_tile(geometric_error, distance, sse_denominator) <= maximum_sse
}

// ─────────────────────────────────────────────────────────────────────────────
// COMMENTED-OUT extended helpers (no external deps required, but kept commented
// until the full HLOD traversal is reviewed live)
// ─────────────────────────────────────────────────────────────────────────────
//
// ```rust
// /// Approximate distance from a camera position to a tile's bounding-sphere
// /// centre, minus the sphere radius (so the distance to the *surface*).
// ///
// /// Returns 0.0 if the camera is inside the sphere.
// ///
// /// UNVERIFIED: Cesium uses oriented bounding box (OBB) or bounding sphere
// /// depending on the tile's `boundingVolume` type.  Sphere variant shown here.
// #[inline]
// pub fn distance_to_bounding_sphere(
//     camera: [f32; 3],
//     sphere_center: [f32; 3],
//     sphere_radius: f32,
// ) -> f32 {
//     let dx = camera[0] - sphere_center[0];
//     let dy = camera[1] - sphere_center[1];
//     let dz = camera[2] - sphere_center[2];
//     let dist_to_center = (dx*dx + dy*dy + dz*dz).sqrt();
//     (dist_to_center - sphere_radius).max(0.0)
// }
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (std-only, live)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Reference computation using the full formula directly (no pre-computation).
    fn sse_reference(geometric_error: f32, viewport_height: f32, distance: f32, fovy_rad: f32) -> f32 {
        let d = distance.max(1e-6);
        geometric_error * viewport_height / (d * 2.0 * (fovy_rad * 0.5).tan())
    }

    #[test]
    fn sse_denominator_matches_reference() {
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        // reference: 1080 / (2 * tan(π/6)) = 1080 / (2 * 0.57735) ≈ 935.3
        let expected = 1080.0_f32 / (2.0 * (std::f32::consts::FRAC_PI_3 / 2.0).tan());
        assert!(approx(denom, expected, 0.1), "denom={denom} expected={expected}");
    }

    #[test]
    fn sse_for_tile_matches_reference() {
        let ge = 20.0_f32;
        let vh = 1080.0_f32;
        let fovy = std::f32::consts::FRAC_PI_3;
        let dist = 500.0_f32;

        let frustum = SseFrustum {
            viewport_height_px: vh,
            fovy_rad: fovy,
        };
        let denom = frustum.sse_denominator();
        let got = sse_for_tile(ge, dist, denom);
        let expected = sse_reference(ge, vh, dist, fovy);

        assert!(approx(got, expected, 1e-3), "got={got} expected={expected}");
    }

    #[test]
    fn tile_meets_sse_accepts_small_error() {
        // geometricError=1.0, distance=1000.0 → very small SSE → accepted
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        assert!(tile_meets_sse(1.0, 1000.0, denom, 16.0));
    }

    #[test]
    fn tile_meets_sse_rejects_large_error() {
        // geometricError=10000.0, distance=1.0 → enormous SSE → rejected
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        assert!(!tile_meets_sse(10000.0, 1.0, denom, 16.0));
    }

    #[test]
    fn zero_geometric_error_always_accepted() {
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        // geometricError=0 → SSE=0 → always meets any positive threshold
        assert!(tile_meets_sse(0.0, 1.0, denom, 16.0));
    }

    #[test]
    fn very_small_distance_does_not_panic() {
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        let sse = sse_for_tile(10.0, 0.0, denom); // distance=0 → clamped
        assert!(sse.is_finite(), "SSE must be finite even at zero distance");
    }

    #[test]
    fn sse_scales_linearly_with_geometric_error() {
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        let s1 = sse_for_tile(10.0, 100.0, denom);
        let s2 = sse_for_tile(20.0, 100.0, denom);
        assert!(approx(s2, 2.0 * s1, 1e-4), "s1={s1} s2={s2}");
    }

    #[test]
    fn sse_inversely_proportional_to_distance() {
        let frustum = SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        };
        let denom = frustum.sse_denominator();
        let s1 = sse_for_tile(10.0, 100.0, denom);
        let s2 = sse_for_tile(10.0, 200.0, denom);
        assert!(approx(s1, 2.0 * s2, 1e-4), "s1={s1} s2={s2}");
    }
}
