//! HHTL depth cascade — HEEL → HIP → TWIG → LEAF block preselection that fuses
//! Cesium screen-space error with the render-depth certificate.
//!
//! # The cascade (render-depth + HHTL-CPU-cascade plans)
//!
//! ```text
//!   HEEL  reject blocks behind near/far or outside the frustum
//!   HIP   screen-space error (Cesium SSE) + projected screen radius → priority
//!   TWIG  depth certificate → ordering / occlusion / depth-error refinement
//!   LEAF  exact-projection / render action
//! ```
//!
//! # Cesium reverse-engineered with our advantages
//!
//! Vanilla Cesium accepts a tile when `sse ≤ maximumScreenSpaceError`. We keep
//! that test (HIP) but gate acceptance on the **depth certificate** (TWIG): a
//! block is only kept coarse when its SSE is small *and* its depth interval is
//! within budget *and* its occlusion confidence clears the threshold. The
//! certificate adds depth/ordering/occlusion uncertainty that flat SSE can't
//! express, so refinement decisions are auditable rather than a single scalar.
//!
//! All math goes through `crate::simd::F32x16` (HEEL batch reject) or the
//! scalar reference; no SIMD-dispatch code lives here.

use super::depth_cert::{certify_depth_scalar, screen_space_error, DepthCertParams, RenderDepthCertificate};
use super::project::Camera;
use crate::simd::F32x16;

/// World-space bounding sphere of a splat block (the broad-phase proxy; the
/// sphere variant of Cesium's `boundingVolume`, matching `cesium::hlod`'s
/// `distance_to_camera`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BlockBounds {
    /// World-space sphere centre.
    pub center: [f32; 3],
    /// World-space sphere radius (also used as the block's geometric extent).
    pub radius: f32,
}

/// Deepest cascade tier a block reached before a terminal decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HhtlTier {
    Heel = 0,
    Hip = 1,
    Twig = 2,
    Leaf = 3,
}

/// Terminal action for a block (HHTL-CPU-cascade plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HhtlAction {
    /// Outside frustum / behind near / beyond far — drop it.
    Reject = 0,
    /// Coarse LOD is good enough (Cesium SSE accepted + certificate passed).
    KeepCoarse = 1,
    /// Depth/ordering/occlusion too uncertain — fetch finer data.
    Refine = 2,
    /// Project exactly via `splat3d` (SSE too high or footprint too large).
    ProjectExact = 3,
    /// Project and rasterize exactly.
    RenderExact = 4,
}

/// Per-block budget: Cesium SSE acceptance + the depth-certificate gates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthCascadeBudget {
    /// Cesium `maximumScreenSpaceError` (px). SSE ≤ this ⇒ coarse acceptable.
    pub max_error_px: f32,
    /// Minimum occlusion confidence to keep coarse (else Refine).
    pub min_confidence: f32,
    /// Projected radius (px) above which a block must be projected exactly.
    pub max_projected_radius_px: f32,
    /// Depth-certificate budgets (k_sigma, error terms, pass threshold).
    pub cert_params: DepthCertParams,
}

impl Default for DepthCascadeBudget {
    fn default() -> Self {
        Self {
            max_error_px: 16.0, // Cesium default maximumScreenSpaceError
            min_confidence: 0.5,
            max_projected_radius_px: 256.0,
            cert_params: DepthCertParams::default(),
        }
    }
}

/// Decision for one block, with the certificate that justified it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockDepthDecision {
    pub block_index: usize,
    pub tier_reached: HhtlTier,
    pub action: HhtlAction,
    /// Screen-space error (px) — Cesium priority; higher ⇒ more refinement.
    pub priority: f32,
    /// Estimated screen error (px) used for the LOD decision (= SSE here).
    pub estimated_error_px: f32,
    /// Projected screen radius (px).
    pub projected_radius_px: f32,
    /// The render-depth certificate produced at TWIG (zeroed if rejected at HEEL).
    pub certificate: RenderDepthCertificate,
}

/// Transform a world point by the upper 3 rows of the row-major view matrix.
#[inline]
fn to_camera(view: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let row = |r: usize| view[r][0] * p[0] + view[r][1] * p[1] + view[r][2] * p[2] + view[r][3];
    [row(0), row(1), row(2)]
}

/// HEEL broad-phase reject (scalar reference): `true` ⇒ drop the block.
///
/// Rejects blocks entirely behind the near plane, entirely beyond the far
/// plane, or whose projected screen AABB is fully off-screen (same cull logic
/// as `project::project_chunk_x16`, at block granularity).
pub fn heel_reject_scalar(camera: &Camera, block: &BlockBounds) -> bool {
    let c = to_camera(&camera.view, block.center);
    let (cam_x, cam_y, cam_z) = (c[0], c[1], c[2]);
    let r = block.radius;
    if cam_z + r < camera.near || cam_z - r > camera.far {
        return true;
    }
    // Project centre; world radius → screen px via fy / depth.
    let z = cam_z.max(1e-6);
    let projr = r * camera.fy / z;
    let sx = camera.fx * cam_x / z + camera.cx;
    let sy = camera.fy * cam_y / z + camera.cy;
    let w = camera.width as f32;
    let h = camera.height as f32;
    sx + projr < 0.0 || sx - projr >= w || sy + projr < 0.0 || sy - projr >= h
}

/// Run the full HEEL→HIP→TWIG→LEAF cascade for one block (scalar reference).
pub fn cascade_block(
    camera: &Camera, block: &BlockBounds, budget: &DepthCascadeBudget, index: usize,
) -> BlockDepthDecision {
    // ── HEEL ─────────────────────────────────────────────────────────────
    if heel_reject_scalar(camera, block) {
        return BlockDepthDecision {
            block_index: index,
            tier_reached: HhtlTier::Heel,
            action: HhtlAction::Reject,
            priority: 0.0,
            estimated_error_px: 0.0,
            projected_radius_px: 0.0,
            certificate: super::depth_cert::certify_depth_scalar(0.0, 0.0, 0.0, &budget.cert_params),
        };
    }

    // ── HIP: Cesium SSE + projected radius ───────────────────────────────
    let c = to_camera(&camera.view, block.center);
    let cam_z = c[2].max(1e-6);
    let projected_radius_px = block.radius * camera.fy / cam_z;
    // Distance to the sphere surface (Cesium hlod style), clamped to near.
    let distance = (cam_z - block.radius).max(camera.near);
    // Geometric error proxy: block diameter (its world-space extent).
    let sse = screen_space_error(2.0 * block.radius, camera.fy, distance);

    // ── TWIG: render-depth certificate ───────────────────────────────────
    // Sphere radius ≈ 3σ extent ⇒ depth variance ≈ (r/3)².
    let sigma_z = block.radius / 3.0;
    let cov_zz = sigma_z * sigma_z;
    let certificate = certify_depth_scalar(cam_z, cov_zz, projected_radius_px, &budget.cert_params);

    // ── LEAF: fused decision ─────────────────────────────────────────────
    let (tier_reached, action) = if !certificate.passed {
        (HhtlTier::Twig, HhtlAction::Refine)
    } else if certificate.occlusion_confidence < budget.min_confidence {
        (HhtlTier::Twig, HhtlAction::Refine)
    } else if projected_radius_px > budget.max_projected_radius_px {
        (HhtlTier::Leaf, HhtlAction::ProjectExact)
    } else if sse <= budget.max_error_px {
        (HhtlTier::Hip, HhtlAction::KeepCoarse)
    } else {
        (HhtlTier::Leaf, HhtlAction::ProjectExact)
    };

    BlockDepthDecision {
        block_index: index,
        tier_reached,
        action,
        priority: sse,
        estimated_error_px: sse,
        projected_radius_px,
        certificate,
    }
}

/// Cascade every block (scalar driver). Does not allocate per block beyond the
/// output push; no panic on degenerate blocks.
pub fn cascade_blocks(
    camera: &Camera, blocks: &[BlockBounds], budget: &DepthCascadeBudget, out: &mut Vec<BlockDepthDecision>,
) {
    out.clear();
    out.reserve(blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        out.push(cascade_block(camera, b, budget, i));
    }
}

const CHUNK: usize = 16;

/// SIMD (16-wide) HEEL broad-phase reject — `out[i] = 1` ⇒ block `i` dropped.
/// The vectorized twin of [`heel_reject_scalar`]: camera transform, projected
/// radius and screen centre run through `F32x16`; the OR of reject predicates
/// is finished per lane (project.rs SIMD-bulk / scalar-tail split).
///
/// `out.len()` must be ≥ `blocks.len()`.
pub fn heel_reject_mask(camera: &Camera, blocks: &[BlockBounds], out: &mut [u8]) {
    let n = blocks.len();
    assert!(out.len() >= n, "heel_reject_mask: out.len() {} < blocks.len() {n}", out.len());
    if n == 0 {
        return;
    }
    let v = &camera.view;
    let row =
        |r: usize| (F32x16::splat(v[r][0]), F32x16::splat(v[r][1]), F32x16::splat(v[r][2]), F32x16::splat(v[r][3]));
    let (v00, v01, v02, v03) = row(0);
    let (v10, v11, v12, v13) = row(1);
    let (v20, v21, v22, v23) = row(2);
    let fx = F32x16::splat(camera.fx);
    let fy = F32x16::splat(camera.fy);
    let cx = F32x16::splat(camera.cx);
    let cy = F32x16::splat(camera.cy);
    let eps = F32x16::splat(1e-6);

    let mut cxs = [0.0f32; 16];
    let mut cys = [0.0f32; 16];
    let mut czs = [0.0f32; 16];
    let mut rs = [0.0f32; 16];

    let mut start = 0;
    while start < n {
        let end = (start + CHUNK).min(n);
        cxs.iter_mut().for_each(|x| *x = 0.0);
        cys.iter_mut().for_each(|x| *x = 0.0);
        czs.iter_mut().for_each(|x| *x = 0.0);
        rs.iter_mut().for_each(|x| *x = 0.0);
        for (k, b) in blocks[start..end].iter().enumerate() {
            cxs[k] = b.center[0];
            cys[k] = b.center[1];
            czs[k] = b.center[2];
            rs[k] = b.radius;
        }
        let mx = F32x16::from_slice(&cxs);
        let my = F32x16::from_slice(&cys);
        let mz = F32x16::from_slice(&czs);
        let r = F32x16::from_slice(&rs);

        let cam_x = v00 * mx + v01 * my + v02 * mz + v03;
        let cam_y = v10 * mx + v11 * my + v12 * mz + v13;
        let cam_z = v20 * mx + v21 * my + v22 * mz + v23;
        let z = cam_z.simd_max(eps);
        let projr = r * fy / z;
        let sx = fx * cam_x / z + cx;
        let sy = fy * cam_y / z + cy;

        let mut camz_a = [0.0f32; 16];
        let mut projr_a = [0.0f32; 16];
        let mut sx_a = [0.0f32; 16];
        let mut sy_a = [0.0f32; 16];
        cam_z.copy_to_slice(&mut camz_a);
        projr.copy_to_slice(&mut projr_a);
        sx.copy_to_slice(&mut sx_a);
        sy.copy_to_slice(&mut sy_a);

        let w = camera.width as f32;
        let h = camera.height as f32;
        for k in 0..(end - start) {
            let cz = camz_a[k];
            let rk = rs[k];
            let pr = projr_a[k];
            let reject = cz + rk < camera.near
                || cz - rk > camera.far
                || sx_a[k] + pr < 0.0
                || sx_a[k] - pr >= w
                || sy_a[k] + pr < 0.0
                || sy_a[k] - pr >= h;
            out[start + k] = reject as u8;
        }
        start += CHUNK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::identity_at_origin(512, 512)
    }

    #[test]
    fn heel_rejects_behind_near_plane() {
        let c = cam();
        let b = BlockBounds {
            center: [0.0, 0.0, -5.0],
            radius: 1.0,
        };
        assert!(heel_reject_scalar(&c, &b));
        let d = cascade_block(&c, &b, &DepthCascadeBudget::default(), 0);
        assert_eq!(d.action, HhtlAction::Reject);
        assert_eq!(d.tier_reached, HhtlTier::Heel);
    }

    #[test]
    fn heel_rejects_off_screen() {
        let c = cam();
        let b = BlockBounds {
            center: [1000.0, 0.0, 1.0],
            radius: 0.01,
        };
        assert!(heel_reject_scalar(&c, &b));
        let d = cascade_block(&c, &b, &DepthCascadeBudget::default(), 7);
        assert_eq!(d.action, HhtlAction::Reject);
        assert_eq!(d.block_index, 7);
    }

    #[test]
    fn small_distant_block_kept_coarse() {
        let c = cam();
        // tiny block far away → low SSE, passing certificate → KeepCoarse.
        let b = BlockBounds {
            center: [0.0, 0.0, 200.0],
            radius: 0.05,
        };
        let budget = DepthCascadeBudget {
            max_error_px: 16.0,
            ..Default::default()
        };
        let d = cascade_block(&c, &b, &budget, 0);
        assert_eq!(d.action, HhtlAction::KeepCoarse, "sse={} cert.passed={}", d.priority, d.certificate.passed);
        assert_eq!(d.tier_reached, HhtlTier::Hip);
        assert!(d.priority <= budget.max_error_px);
    }

    #[test]
    fn large_near_block_projected_exact() {
        let c = cam();
        // big block up close → huge SSE → ProjectExact.
        let b = BlockBounds {
            center: [0.0, 0.0, 2.0],
            radius: 1.5,
        };
        let budget = DepthCascadeBudget {
            max_error_px: 16.0,
            max_projected_radius_px: 1e9,
            ..Default::default()
        };
        let d = cascade_block(&c, &b, &budget, 0);
        assert_eq!(d.action, HhtlAction::ProjectExact, "sse={}", d.priority);
        assert_eq!(d.tier_reached, HhtlTier::Leaf);
        assert!(d.priority > budget.max_error_px);
    }

    #[test]
    fn oversized_footprint_forces_project_exact() {
        let c = cam();
        let b = BlockBounds {
            center: [0.0, 0.0, 100.0],
            radius: 0.02,
        };
        // Force the footprint gate by setting a tiny max projected radius.
        let budget = DepthCascadeBudget {
            max_error_px: 1e9,
            max_projected_radius_px: 0.0,
            ..Default::default()
        };
        let d = cascade_block(&c, &b, &budget, 0);
        assert_eq!(d.action, HhtlAction::ProjectExact);
        assert_eq!(d.tier_reached, HhtlTier::Leaf);
    }

    #[test]
    fn failing_depth_certificate_forces_refine() {
        let c = cam();
        let b = BlockBounds {
            center: [0.0, 0.0, 5.0],
            radius: 1.0,
        };
        // Tiny depth-error budget → certificate fails → Refine at TWIG.
        let mut budget = DepthCascadeBudget::default();
        budget.cert_params.max_total_depth_error = 1e-6;
        let d = cascade_block(&c, &b, &budget, 3);
        assert_eq!(d.action, HhtlAction::Refine);
        assert_eq!(d.tier_reached, HhtlTier::Twig);
        assert!(!d.certificate.passed);
    }

    #[test]
    fn low_confidence_forces_refine() {
        let c = cam();
        let b = BlockBounds {
            center: [0.0, 0.0, 50.0],
            radius: 0.1,
        };
        // Large neighbour overlap → low occlusion confidence; demand high.
        let mut budget = DepthCascadeBudget {
            max_error_px: 1e9,
            min_confidence: 0.9,
            ..Default::default()
        };
        budget.cert_params.splat_support_overlap_error = 100.0;
        let d = cascade_block(&c, &b, &budget, 0);
        assert_eq!(d.action, HhtlAction::Refine);
        assert_eq!(d.tier_reached, HhtlTier::Twig);
        assert!(d.certificate.occlusion_confidence < budget.min_confidence);
    }

    #[test]
    fn heel_reject_mask_matches_scalar() {
        let c = cam();
        let mut st = 0xBEEF_0001u32;
        let mut rng = |s: &mut u32| {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f32) / (u32::MAX as f32)
        };
        let mut blocks = Vec::new();
        for i in 0..37 {
            // Mix on-screen, behind-camera, off-screen, far.
            let z = match i % 4 {
                0 => -2.0,
                1 => 2000.0,
                _ => 1.0 + rng(&mut st) * 10.0,
            };
            let x = if i % 5 == 0 { 5000.0 } else { rng(&mut st) * 4.0 - 2.0 };
            blocks.push(BlockBounds {
                center: [x, rng(&mut st) * 4.0 - 2.0, z],
                radius: 0.1 + rng(&mut st),
            });
        }
        let mut mask = vec![0u8; blocks.len()];
        heel_reject_mask(&c, &blocks, &mut mask);
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(mask[i], heel_reject_scalar(&c, b) as u8, "block {i} mask mismatch");
        }
    }

    #[test]
    fn cascade_blocks_driver_lengths() {
        let c = cam();
        let blocks: Vec<_> = (0..20)
            .map(|i| BlockBounds {
                center: [0.0, 0.0, 2.0 + i as f32],
                radius: 0.2,
            })
            .collect();
        let mut out = Vec::new();
        cascade_blocks(&c, &blocks, &DepthCascadeBudget::default(), &mut out);
        assert_eq!(out.len(), 20);
        for (i, d) in out.iter().enumerate() {
            assert_eq!(d.block_index, i);
        }
    }
}
