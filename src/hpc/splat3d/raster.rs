//! Per-tile rasterizer — depth-sorted alpha-blend with F32x16 pixel rows.
//!
//! Math reference: Zwicker 2001 §4, Kerbl 2023 §4 (EWA splatting).
//!
//! # Alpha-blend formula (front-to-back)
//!
//! For each gaussian at screen position `(gx, gy)` with 2D conic (a, b, c):
//! ```text
//! dx    = gx - px
//! dy    = gy - py
//! power = -0.5 · (a·dx² + 2·b·dx·dy + c·dy²)   [Mahalanobis²]
//! alpha = min(0.99, opacity · exp(power))
//! C    += T · alpha · color         (if power ≤ 0 and alpha ≥ 1/255)
//! T    *= (1 - alpha)
//! pixel = C + T · background
//! ```
//!
//! # Early-out math
//!
//! Any gaussian behind a point where `T < ε` contributes
//! `< ε · alpha · color < ε · 1 · 1 = ε` to the final pixel —
//! below the 8-bit quantization floor (1/256 ≈ 0.0039) when `ε = 1e-4`.
//!
//! # Framebuffer layout
//!
//! Interleaved RGB: `[R0, G0, B0, R1, G1, B1, …]`, length `3 · width · height`.
//! Pixel `(x, y)` occupies indices `(y * width + x) * 3 .. (y * width + x) * 3 + 3`.
//!
//! # SIMD strategy
//!
//! One F32x16 per tile row (16 pixels × 1 row). The inner gaussian loop
//! broadcasts per-gaussian scalars and evaluates all 16 pixels in parallel.

use crate::hpc::splat3d::project::ProjectedBatch;
use crate::hpc::splat3d::tile::{TileBinning, TILE_SIZE};
use crate::simd::{simd_exp_f32, F32Mask16, F32x16};

/// Saturation threshold for the front-to-back early-out.
///
/// At `T < T_SATURATION_EPS` any subsequent gaussian's contribution is below the
/// 8-bit quantization floor (`color · alpha · T < 1/256`).
pub const T_SATURATION_EPS: f32 = 1e-4;

// ════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ════════════════════════════════════════════════════════════════════════════

/// Combine two F32Mask16 with bitwise AND (both conditions must be true).
#[inline(always)]
fn mask_and(a: F32Mask16, b: F32Mask16) -> F32Mask16 {
    F32Mask16(a.0 & b.0)
}

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

/// Render one full 16×16 tile to the framebuffer.
///
/// Processes the tile at grid position `(tile_x, tile_y)`.  The pixel region
/// written is `[tile_x*16 .. (tile_x+1)*16] × [tile_y*16 .. (tile_y+1)*16]`
/// (clamped to `width`/`height` for edge tiles).
///
/// # Parameters
/// - `tile_x`, `tile_y`: tile grid coordinates.
/// - `binning`: precomputed tile binning from PR 4.
/// - `projected`: per-gaussian projection data from PR 3.
/// - `framebuffer`: interleaved RGB, length `3 · width · height` (mutable sink).
/// - `width`, `height`: image dimensions in pixels.
/// - `background`: clear color composited under the residual transmittance.
pub fn rasterize_tile(
    tile_x: u32,
    tile_y: u32,
    binning: &TileBinning,
    projected: &ProjectedBatch,
    framebuffer: &mut [f32],
    width: u32,
    height: u32,
    background: [f32; 3],
) {
    let tile_instances = binning.tile_instances(tile_x, tile_y);

    let tile_x_base = (tile_x * TILE_SIZE) as f32;
    let tile_y_base = (tile_y * TILE_SIZE) as f32;

    // Build the pixel-X vector once — same for every row.
    let px = F32x16::from_array([
        tile_x_base,
        tile_x_base + 1.0,
        tile_x_base + 2.0,
        tile_x_base + 3.0,
        tile_x_base + 4.0,
        tile_x_base + 5.0,
        tile_x_base + 6.0,
        tile_x_base + 7.0,
        tile_x_base + 8.0,
        tile_x_base + 9.0,
        tile_x_base + 10.0,
        tile_x_base + 11.0,
        tile_x_base + 12.0,
        tile_x_base + 13.0,
        tile_x_base + 14.0,
        tile_x_base + 15.0,
    ]);

    let zero = F32x16::splat(0.0);
    let one = F32x16::splat(1.0);
    let alpha_max = F32x16::splat(0.99);
    let alpha_floor = F32x16::splat(1.0 / 255.0);
    let zero_thresh = F32x16::splat(0.0);

    for row in 0..TILE_SIZE {
        let py = F32x16::splat(tile_y_base + row as f32);

        // Per-pixel accumulators for this row.
        let mut t = F32x16::splat(1.0);
        let mut cr = zero;
        let mut cg = zero;
        let mut cb = zero;

        // Walk gaussians depth-ascending (front-to-back).
        for inst in tile_instances {
            let gid = inst.gaussian_id as usize;

            // Broadcast per-gaussian scalars across all 16 pixel lanes.
            let gx = F32x16::splat(projected.screen_x[gid]);
            let gy = F32x16::splat(projected.screen_y[gid]);
            let ca = F32x16::splat(projected.conic_a[gid]);
            let cb_ = F32x16::splat(projected.conic_b[gid]);
            let cc = F32x16::splat(projected.conic_c[gid]);
            let op = F32x16::splat(projected.opacity[gid]);
            let rr = F32x16::splat(projected.color_r[gid]);
            let gg = F32x16::splat(projected.color_g[gid]);
            let bb = F32x16::splat(projected.color_b[gid]);

            // 2D Mahalanobis distance squared (negated for the exponent).
            let dx = gx - px;
            let dy = gy - py;
            let power = F32x16::splat(-0.5)
                * (ca * dx * dx
                    + F32x16::splat(2.0) * cb_ * dx * dy
                    + cc * dy * dy);

            // exp(power) is the gaussian density at each pixel.
            let alpha_pre = op * simd_exp_f32(power);
            let alpha = alpha_pre.simd_min(alpha_max);

            // Mask: inside 3σ ellipse (power ≤ 0) AND above quantization floor.
            let in_ellipse = power.simd_le(zero_thresh);
            let above_floor = alpha.simd_ge(alpha_floor);
            let m = mask_and(in_ellipse, above_floor);

            // Conditional accumulate: only lanes where m is set.
            let contrib = t * alpha;
            cr = m.select(cr + contrib * rr, cr);
            cg = m.select(cg + contrib * gg, cg);
            cb = m.select(cb + contrib * bb, cb);
            t = m.select(t * (one - alpha), t);

            // Front-to-back early-out: all 16 lanes saturated.
            if t.reduce_max() < T_SATURATION_EPS {
                break;
            }
        }

        // Composite background under residual transmittance.
        let bgr = F32x16::splat(background[0]);
        let bgg = F32x16::splat(background[1]);
        let bgb = F32x16::splat(background[2]);
        cr = cr + t * bgr;
        cg = cg + t * bgg;
        cb = cb + t * bgb;

        // Scatter the 16 pixel values into the interleaved framebuffer.
        let cr_arr = cr.to_array();
        let cg_arr = cg.to_array();
        let cb_arr = cb.to_array();

        let row_base = ((tile_y * TILE_SIZE + row) * width) as usize;
        for k in 0..16_usize {
            let pix_x = tile_x * TILE_SIZE + k as u32;
            if pix_x >= width {
                break; // Partial tile at right edge of image.
            }
            let py_abs = tile_y * TILE_SIZE + row;
            if py_abs >= height {
                break; // Partial tile at bottom edge of image.
            }
            let idx = (row_base + pix_x as usize) * 3;
            framebuffer[idx] = cr_arr[k];
            framebuffer[idx + 1] = cg_arr[k];
            framebuffer[idx + 2] = cb_arr[k];
        }
    }
}

/// Render the full framebuffer by walking every tile in `binning`.
///
/// Single-threaded; rayon parallelization is a follow-on (PR 6 frame
/// double-buffer driver).  Tiles are visited in row-major order; each call
/// to `rasterize_tile` writes a disjoint `TILE_SIZE × TILE_SIZE` pixel
/// region.
///
/// # Parameters
/// - `binning`: precomputed tile binning from PR 4.
/// - `projected`: per-gaussian projection data from PR 3.
/// - `framebuffer`: interleaved RGB sink, length `3 · width · height`.
/// - `width`, `height`: image dimensions in pixels.
/// - `background`: clear color composited under residual transmittance.
pub fn rasterize_frame(
    binning: &TileBinning,
    projected: &ProjectedBatch,
    framebuffer: &mut [f32],
    width: u32,
    height: u32,
    background: [f32; 3],
) {
    for ty in 0..binning.tile_rows {
        for tx in 0..binning.tile_cols {
            rasterize_tile(tx, ty, binning, projected, framebuffer, width, height, background);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::splat3d::project::{Camera, ProjectedBatch};
    use crate::hpc::splat3d::tile::TileBinning;

    // ── Test helper ──────────────────────────────────────────────────────────

    /// Build a test scene directly from a list of gaussian parameters,
    /// bypassing the projection step.
    ///
    /// Each tuple: `(screen_x, screen_y, conic_a, conic_b, conic_c,
    ///               radius, color_r, color_g, color_b, opacity, depth)`
    #[allow(clippy::type_complexity)]
    fn make_test_scene(
        width: u32,
        height: u32,
        gaussians: &[(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)],
    ) -> (ProjectedBatch, TileBinning, Camera) {
        let n = gaussians.len();
        let mut projected = ProjectedBatch::with_capacity(n.max(1));
        projected.len = n;

        for (i, &(sx, sy, ca, cb, cc, rad, cr, cg, cbv, op, dep)) in
            gaussians.iter().enumerate()
        {
            projected.screen_x[i] = sx;
            projected.screen_y[i] = sy;
            projected.conic_a[i] = ca;
            projected.conic_b[i] = cb;
            projected.conic_c[i] = cc;
            projected.radius[i] = rad;
            projected.color_r[i] = cr;
            projected.color_g[i] = cg;
            projected.color_b[i] = cbv;
            projected.opacity[i] = op;
            projected.depth[i] = dep;
            projected.valid[i] = 1;
        }

        let camera = Camera::identity_at_origin(width, height);
        let binning = TileBinning::from_projected(&projected, &camera);
        (projected, binning, camera)
    }

    /// Read a single pixel from the framebuffer.
    fn get_pixel(fb: &[f32], x: u32, y: u32, width: u32) -> [f32; 3] {
        let idx = (y * width + x) as usize * 3;
        [fb[idx], fb[idx + 1], fb[idx + 2]]
    }

    // ── Test 1: empty scene returns background ────────────────────────────────

    #[test]
    fn rasterize_empty_scene_returns_background() {
        let w = 32u32;
        let h = 32u32;
        let bg = [0.2_f32, 0.4, 0.6];
        let (projected, binning, _) = make_test_scene(w, h, &[]);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        for y in 0..h {
            for x in 0..w {
                let p = get_pixel(&fb, x, y, w);
                assert!((p[0] - bg[0]).abs() < 1e-6, "R mismatch at ({x},{y}): {}", p[0]);
                assert!((p[1] - bg[1]).abs() < 1e-6, "G mismatch at ({x},{y}): {}", p[1]);
                assert!((p[2] - bg[2]).abs() < 1e-6, "B mismatch at ({x},{y}): {}", p[2]);
            }
        }
    }

    // ── Test 2: single opaque white gaussian paints center pixel white ────────

    #[test]
    fn rasterize_single_opaque_white_gaussian_at_center_paints_white() {
        let w = 32u32;
        let h = 32u32;
        let bg = [0.0_f32, 0.0, 0.0];
        // Gaussian at (16,16) with tight conic — large eigenvalues means
        // it falls off fast. conic_a=conic_c=1, conic_b=0.
        // opacity=0.99 so alpha = min(0.99, 0.99*exp(0)) = 0.99 at center.
        let gaussians = [(16.0f32, 16.0, 1.0, 0.0, 1.0, 5.0, 1.0, 1.0, 1.0, 0.99, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        // Center pixel should be close to white (alpha = 0.99 at center).
        let center = get_pixel(&fb, 16, 16, w);
        assert!(center[0] > 0.9, "Center R should be near white, got {}", center[0]);
        assert!(center[1] > 0.9, "Center G should be near white, got {}", center[1]);
        assert!(center[2] > 0.9, "Center B should be near white, got {}", center[2]);

        // Far pixel (0,0) should be nearly background (black).
        let far = get_pixel(&fb, 0, 0, w);
        assert!(far[0] < 0.01, "Far R should be near 0, got {}", far[0]);
    }

    // ── Test 3: two overlapping gaussians alpha-blend correctly ───────────────

    #[test]
    fn rasterize_two_overlapping_alpha_blend_correctly() {
        let w = 32u32;
        let h = 32u32;
        let bg = [0.0_f32, 0.0, 0.0];
        // Both gaussians at exact center pixel (8,8) in tile (0,0).
        // Front (depth=1): red, alpha=0.3 (opacity tuned via conic so
        // exp(power)=1 at center → opacity = 0.3).
        // Back (depth=2): blue, alpha=0.3.
        // Expected: R = 0.3, G = 0, B = 0.3*(1-0.3) = 0.21
        // Large negative conic_b=0 and tight a/c so exp(0)=1 at center.
        let gaussians = [
            // front: red
            (8.0f32, 8.0, 100.0, 0.0, 100.0, 5.0, 1.0, 0.0, 0.0, 0.3, 1.0),
            // back: blue
            (8.0f32, 8.0, 100.0, 0.0, 100.0, 5.0, 0.0, 0.0, 1.0, 0.3, 2.0),
        ];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        let p = get_pixel(&fb, 8, 8, w);
        // At center: power=-0.5*(100*0+0+100*0)=0, alpha_pre=0.3*exp(0)=0.3
        // Front: C += 1.0*0.3*red=[0.3,0,0], T=0.7
        // Back:  C += 0.7*0.3*blue=[0,0,0.21], T=0.49
        // Final: C=[0.3,0,0.21], T=0.49, bg=black → pixel=[0.3,0,0.21]
        assert!((p[0] - 0.3).abs() < 0.01, "R expected ~0.3, got {}", p[0]);
        assert!((p[1]).abs() < 0.01, "G expected ~0, got {}", p[1]);
        assert!((p[2] - 0.21).abs() < 0.02, "B expected ~0.21, got {}", p[2]);
    }

    // ── Test 4: 50-stack early-out (pixel must be opaque black, no bg bleed) ──

    #[test]
    fn rasterize_early_out_skips_saturated_pixel() {
        let w = 32u32;
        let h = 32u32;
        let bg = [1.0_f32, 1.0, 1.0]; // white background
        // 50 fully opaque black gaussians at center (8,8), increasing depth.
        let mut gaussians = Vec::new();
        for i in 0..50usize {
            gaussians.push((
                8.0f32,
                8.0,
                100.0_f32, // tight conic
                0.0,
                100.0,
                5.0,
                0.0f32, // black color
                0.0,
                0.0,
                0.99f32, // high opacity
                (i + 1) as f32, // increasing depth
            ));
        }
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        // Center pixel must be black (opaque frontmost black gaussians,
        // no background bleed). The early-out ensures correctness here.
        let p = get_pixel(&fb, 8, 8, w);
        assert!(p[0] < 1e-3, "R should be ~0 (black), got {}", p[0]);
        assert!(p[1] < 1e-3, "G should be ~0 (black), got {}", p[1]);
        assert!(p[2] < 1e-3, "B should be ~0 (black), got {}", p[2]);
    }

    // ── Test 5: outside 3σ ellipse skips contribution ─────────────────────────

    #[test]
    fn rasterize_outside_3sigma_ellipse_skips_contribution() {
        let w = 256u32;
        let h = 256u32;
        let bg = [0.5_f32, 0.5, 0.5];
        // Gaussian at (32, 32) in tile (2,2), very tight conic (large values).
        // Pixel (200, 200) is in tile (12,12) — will NOT receive any contribution.
        let gaussians = [(32.0f32, 32.0, 1000.0, 0.0, 1000.0, 1.0, 1.0, 0.0, 0.0, 0.99, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        // Pixel (200, 200) must be exactly background.
        let p = get_pixel(&fb, 200, 200, w);
        assert!((p[0] - bg[0]).abs() < 1e-6, "R at (200,200) should be background");
        assert!((p[1] - bg[1]).abs() < 1e-6, "G at (200,200) should be background");
        assert!((p[2] - bg[2]).abs() < 1e-6, "B at (200,200) should be background");
    }

    // ── Test 6: per-tile write isolation ─────────────────────────────────────

    #[test]
    fn rasterize_tile_writes_only_its_pixels() {
        let w = 96u32;
        let h = 96u32;
        let bg = [0.0_f32, 0.0, 0.0];
        let sentinel = 0.5_f32;

        // Put a gaussian in tile (5,5) = pixel region [80..96] × [80..96].
        let gaussians = [(88.0f32, 88.0, 1.0, 0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.99, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);

        // Pre-fill entire framebuffer with sentinel.
        let mut fb = vec![sentinel; (3 * w * h) as usize];

        // Only render tile (5, 5).
        rasterize_tile(5, 5, &binning, &projected, &mut fb, w, h, bg);

        // Pixels inside [80..96) × [80..96) were written — should NOT be sentinel.
        // Pixels OUTSIDE that region must remain sentinel.
        for y in 0..h {
            for x in 0..w {
                let in_tile = x >= 80 && x < 96 && y >= 80 && y < 96;
                let p = get_pixel(&fb, x, y, w);
                if !in_tile {
                    assert!(
                        (p[0] - sentinel).abs() < 1e-6
                            && (p[1] - sentinel).abs() < 1e-6
                            && (p[2] - sentinel).abs() < 1e-6,
                        "Pixel ({x},{y}) outside tile was modified: {p:?}"
                    );
                }
            }
        }
    }

    // ── Test 7: rasterize_frame == per-tile sum ───────────────────────────────

    #[test]
    fn rasterize_frame_matches_per_tile_sum() {
        let w = 32u32;
        let h = 32u32;
        let bg = [0.1_f32, 0.2, 0.3];
        let gaussians = [
            (8.0f32, 8.0, 1.0, 0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.5, 1.0),
            (24.0f32, 8.0, 1.0, 0.0, 1.0, 4.0, 0.0, 1.0, 0.0, 0.5, 2.0),
        ];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);

        let mut fb_frame = vec![0.0f32; (3 * w * h) as usize];
        rasterize_frame(&binning, &projected, &mut fb_frame, w, h, bg);

        let mut fb_tiles = vec![0.0f32; (3 * w * h) as usize];
        for ty in 0..binning.tile_rows {
            for tx in 0..binning.tile_cols {
                rasterize_tile(tx, ty, &binning, &projected, &mut fb_tiles, w, h, bg);
            }
        }

        for i in 0..(3 * w * h) as usize {
            assert!(
                (fb_frame[i] - fb_tiles[i]).abs() < 1e-6,
                "Mismatch at index {i}: frame={} tiles={}",
                fb_frame[i],
                fb_tiles[i]
            );
        }
    }

    // ── Test 8: partial image at right edge ───────────────────────────────────

    #[test]
    fn rasterize_partial_image_at_edge() {
        // width=17: one full tile (0..16) + one partial tile column (16..17).
        let w = 17u32;
        let h = 16u32;
        let bg = [0.3_f32, 0.3, 0.3];
        let gaussians = [(16.0f32, 8.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.5, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        // Pixel (16, 8) exists and should have been written (background at minimum).
        let p16 = get_pixel(&fb, 16, 8, w);
        // It's within bounds — should be background or blended with gaussian.
        assert!(p16[0] >= 0.0 && p16[0] <= 1.0, "Pixel (16,8) R out of range: {}", p16[0]);

        // No out-of-bounds write occurred (the framebuffer is exactly sized
        // for w×h, so this test verifies the `pix_x >= width` guard by
        // not panicking with an index-out-of-bounds).
    }

    // ── Test 9: background visible when alpha is low ──────────────────────────

    #[test]
    fn rasterize_background_visible_when_alpha_low() {
        let w = 16u32;
        let h = 16u32;
        let bg = [1.0_f32, 0.0, 0.0]; // red background
        // Gaussian at (8,8) with low opacity=0.1, white color.
        // At center: alpha = min(0.99, 0.1 * exp(0)) = 0.1
        // C = 1.0 * 0.1 * [1,1,1] = [0.1, 0.1, 0.1]
        // T = 0.9
        // Final: [0.1, 0.1, 0.1] + 0.9 * [1, 0, 0] = [1.0, 0.1, 0.1]
        let gaussians = [(8.0f32, 8.0, 100.0, 0.0, 100.0, 2.0, 1.0, 1.0, 1.0, 0.1, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        let p = get_pixel(&fb, 8, 8, w);
        // Red channel: gaussian contributes 0.1, background 0.9*1.0=0.9 → ~1.0
        assert!((p[0] - 1.0).abs() < 0.05, "R expected ~1.0, got {}", p[0]);
        // Green channel: gaussian contributes 0.1, background 0.9*0.0=0 → ~0.1
        assert!((p[1] - 0.1).abs() < 0.05, "G expected ~0.1, got {}", p[1]);
        // Blue channel: gaussian contributes 0.1, background 0.9*0.0=0 → ~0.1
        assert!((p[2] - 0.1).abs() < 0.05, "B expected ~0.1, got {}", p[2]);
    }

    // ── Test 10: empty tile keeps background ──────────────────────────────────

    #[test]
    fn rasterize_zero_gaussians_in_tile_keeps_background() {
        let w = 112u32; // 7 tiles wide
        let h = 112u32; // 7 tiles tall
        let bg = [0.7_f32, 0.3, 0.1];
        // Gaussian only in tile (6,6) = pixels [96..112)×[96..112).
        let gaussians = [(104.0f32, 104.0, 1.0, 0.0, 1.0, 4.0, 1.0, 1.0, 1.0, 0.99, 1.0)];
        let (projected, binning, _) = make_test_scene(w, h, &gaussians);
        let mut fb = vec![0.0f32; (3 * w * h) as usize];

        rasterize_frame(&binning, &projected, &mut fb, w, h, bg);

        // Tile (5,5) = pixels [80..96)×[80..96) has no gaussians → pure bg.
        for y in 80..96u32 {
            for x in 80..96u32 {
                let p = get_pixel(&fb, x, y, w);
                assert!(
                    (p[0] - bg[0]).abs() < 1e-6,
                    "Tile(5,5) pixel ({x},{y}) R should be bg, got {}",
                    p[0]
                );
                assert!(
                    (p[1] - bg[1]).abs() < 1e-6,
                    "Tile(5,5) pixel ({x},{y}) G should be bg, got {}",
                    p[1]
                );
                assert!(
                    (p[2] - bg[2]).abs() < 1e-6,
                    "Tile(5,5) pixel ({x},{y}) B should be bg, got {}",
                    p[2]
                );
            }
        }
    }
}
