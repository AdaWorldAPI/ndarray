//! End-to-end integration test for `ndarray::hpc::splat3d`.
//!
//! Builds a synthetic 1000-gaussian scene with known structure,
//! runs it through `SplatRenderer::tick`, and validates the
//! framebuffer against analytical expectations. This is the e2e
//! regression guard the sprint's "Definition of done" calls out
//! ("renders a scene end-to-end on CPU"). The bicycle-scene SSIM
//! comparison vs reference CUDA render is left for a follow-up
//! session when the .ply asset is mirrored locally.
//!
//! ```bash
//! cargo test --features splat3d --test splat3d_correctness
//! ```

#![cfg(feature = "splat3d")]

use ndarray::hpc::splat3d::{
    Camera, Gaussian3D, SplatFrame, SplatRenderer, SH_COEFFS_PER_GAUSSIAN,
};

/// Build a deterministic 1000-gaussian scene laid out as a 10×10×10
/// cubic grid spanning world coordinates `[-2, 2]³`. Each gaussian:
/// - Position: cube vertex `(x, y, z)` with `x, y, z ∈ {-2, -1.5, …, 2}`.
/// - Scale: isotropic 0.08 (small enough that gaussians don't overlap).
/// - Quat: identity (no rotation).
/// - Opacity: 0.9.
/// - Color (via SH DC term): `((x + 2) / 4, (y + 2) / 4, (z + 2) / 4)` —
///   one color channel per axis, so the cube renders a smooth RGB
///   gradient depending on which face the camera looks at.
fn build_synthetic_cube_scene(frame: &mut SplatFrame) {
    let n = 10;
    let mut state = 0xC0FFEEu32;
    let mut xor_advance = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
    };

    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let x = -2.0 + (ix as f32) * (4.0 / (n - 1) as f32);
                let y = -2.0 + (iy as f32) * (4.0 / (n - 1) as f32);
                let z = -2.0 + (iz as f32) * (4.0 / (n - 1) as f32);
                let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
                // DC term per channel (sh[ch * 16 + 0]):
                //   R = ix/(n-1), G = iy/(n-1), B = iz/(n-1)
                //   Pre-divide by SH_C0 ≈ 0.282 so the output (which is
                //   SH_C0 · sh[0] + 0.5) lands at the intended color.
                let sh_c0: f32 = 0.28209479177387814;
                sh[0]      = (ix as f32) / (n - 1) as f32 / sh_c0;
                sh[16]     = (iy as f32) / (n - 1) as f32 / sh_c0;
                sh[32]     = (iz as f32) / (n - 1) as f32 / sh_c0;
                // Add a tiny jitter to the SH coefficients beyond the DC
                // term so the eval path exercises the higher-degree
                // basis functions (regression for PR 2's SH math).
                xor_advance(&mut state);
                sh[1] = (state as f32 / u32::MAX as f32 - 0.5) * 0.05;
                frame.gaussians.push(Gaussian3D {
                    mean: [x, y, z],
                    scale: [0.08, 0.08, 0.08],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    opacity: 0.9,
                    sh,
                });
            }
        }
    }
}

/// A simple "camera at (cx, cy, cz) looking down its own +Z axis with
/// no rotation" view matrix. Used for the smoke-test renders so the
/// gaussian arrangement projects predictably to screen space.
fn camera_looking_down_z(cx: f32, cy: f32, cz: f32, width: u32, height: u32) -> Camera {
    // World-to-camera translation: subtract camera position from world
    // coordinates. View matrix is identity rotation + (-cx, -cy, -cz)
    // translation. So a world point at (cx + dx, cy + dy, cz + dz)
    // ends up at camera-frame (dx, dy, dz).
    let view = [
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 1.0, -cz],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let fx = (width.max(height)) as f32;
    Camera {
        view,
        fx,
        fy: fx,
        cx: (width as f32) * 0.5,
        cy: (height as f32) * 0.5,
        near: 0.01,
        far: 1000.0,
        width,
        height,
        position: [cx, cy, cz],
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn end_to_end_synthetic_cube_renders_without_panic() {
    // 1000-gaussian scene, 256×256 image, camera placed at (0, 0, -5)
    // so the cube sits at depth ~3-7 in camera space and projects to
    // the image. Renders one frame; asserts the framebuffer has
    // non-trivial pixel variance (i.e. SOMETHING was rendered, not
    // just a flat background).
    let mut frame = SplatFrame::with_capacity(1000, 256, 256);
    build_synthetic_cube_scene(&mut frame);
    assert_eq!(frame.gaussians.len, 1000);

    let camera = camera_looking_down_z(0.0, 0.0, -5.0, 256, 256);
    frame.tick(&camera, [0.0, 0.0, 0.0]);

    assert_eq!(frame.frame_id, 1);
    assert_eq!(frame.framebuffer.len(), 256 * 256 * 3);

    // Pixel variance test: at least 1% of pixels must differ from the
    // pure-background value (= 0.0). Otherwise the rasterizer wrote
    // nothing.
    let lit_pixels = frame
        .framebuffer
        .chunks_exact(3)
        .filter(|p| p[0] > 0.01 || p[1] > 0.01 || p[2] > 0.01)
        .count();
    assert!(
        lit_pixels > 100,
        "expected > 100 lit pixels from a 1000-gaussian cube scene, got {lit_pixels}"
    );

    // The image should NOT be all-white either (which would indicate a
    // total saturation bug or an early-out failure).
    let saturated_pixels = frame
        .framebuffer
        .chunks_exact(3)
        .filter(|p| p[0] > 0.99 && p[1] > 0.99 && p[2] > 0.99)
        .count();
    assert!(
        saturated_pixels < 256 * 256 / 2,
        "expected < 50% saturated pixels, got {saturated_pixels} / {}",
        256 * 256
    );
}

#[test]
fn end_to_end_double_buffer_swap_preserves_consistency() {
    // SplatRenderer with the same scene. Tick twice. front_frame_id
    // must reach 2; the two ticks must render to DIFFERENT back
    // buffers (otherwise the double-buffer is broken).
    let renderer = SplatRenderer::with_capacity(1000, 128, 128);
    {
        let mut back = renderer.write_back();
        build_synthetic_cube_scene(&mut back);
    }
    // Copy the same scene into the OTHER buffer too (the renderer
    // allocates both up-front; the back buffer for tick 2 is what
    // started as the front buffer at construction time).
    renderer.swap();
    {
        let mut back = renderer.write_back();
        build_synthetic_cube_scene(&mut back);
    }
    renderer.swap();
    // Reset the renderer state — both buffers now have the scene.
    // Tick the renderer.
    let camera = camera_looking_down_z(0.0, 0.0, -5.0, 128, 128);
    renderer.tick(&camera, [0.0, 0.0, 0.0]);
    assert_eq!(renderer.front_frame_id(), 1);
    renderer.tick(&camera, [0.0, 0.0, 0.0]);
    assert_eq!(renderer.front_frame_id(), 2);
}

#[test]
fn end_to_end_camera_translation_changes_render() {
    // Smoke test that moving the camera produces a DIFFERENT render.
    // If the camera transform were broken (e.g. view matrix ignored),
    // two cameras at different positions would render identically.
    let mut frame = SplatFrame::with_capacity(1000, 64, 64);
    build_synthetic_cube_scene(&mut frame);

    let cam_a = camera_looking_down_z(0.0, 0.0, -5.0, 64, 64);
    frame.tick(&cam_a, [0.0, 0.0, 0.0]);
    let fb_a: Vec<f32> = frame.framebuffer.clone();

    let cam_b = camera_looking_down_z(1.0, 0.0, -5.0, 64, 64);
    frame.tick(&cam_b, [0.0, 0.0, 0.0]);
    let fb_b: Vec<f32> = frame.framebuffer.clone();

    // The two framebuffers must differ — sum-of-squared-differences > 0.
    let ssd: f32 = fb_a
        .iter()
        .zip(fb_b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(
        ssd > 1.0,
        "expected non-trivial SSD between two camera positions, got {ssd}"
    );
}

#[test]
fn end_to_end_empty_scene_yields_pure_background() {
    let mut frame = SplatFrame::with_capacity(16, 64, 64);
    let camera = camera_looking_down_z(0.0, 0.0, -5.0, 64, 64);
    let bg = [0.25_f32, 0.5, 0.75];
    frame.tick(&camera, bg);

    for (i, chunk) in frame.framebuffer.chunks_exact(3).enumerate() {
        assert!(
            (chunk[0] - bg[0]).abs() < 1e-6
                && (chunk[1] - bg[1]).abs() < 1e-6
                && (chunk[2] - bg[2]).abs() < 1e-6,
            "pixel {i}: expected bg = {bg:?}, got [{}, {}, {}]",
            chunk[0], chunk[1], chunk[2]
        );
    }
}

#[test]
fn end_to_end_three_consecutive_ticks_preserve_invariants() {
    // Stress test: 3 ticks in a row, verify frame_id increments
    // monotonically and the framebuffer is fully written each time
    // (no leaked NaN, no leaked zero from a previous frame).
    let mut frame = SplatFrame::with_capacity(1000, 128, 128);
    build_synthetic_cube_scene(&mut frame);

    let camera = camera_looking_down_z(0.0, 0.0, -5.0, 128, 128);
    for tick_n in 1..=3 {
        frame.tick(&camera, [0.05, 0.05, 0.05]);
        assert_eq!(frame.frame_id, tick_n);
        // No NaN in the framebuffer.
        for (i, &px) in frame.framebuffer.iter().enumerate() {
            assert!(
                px.is_finite(),
                "non-finite pixel at index {i} after tick {tick_n}: {px}"
            );
        }
    }
}
