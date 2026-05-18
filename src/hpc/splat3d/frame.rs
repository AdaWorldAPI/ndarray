//! [`SplatFrame`] — one tick's full state. [`SplatRenderer`] — the
//! double-buffered driver that owns two frames and runs `tick()` on
//! the back while readers consume the front.
//!
//! Sibling of [`crate::hpc::renderer::RenderFrame`] / [`crate::hpc::renderer::Renderer`].
//! Same double-buffer shape: two `RwLock<SplatFrame>`s, `AtomicUsize front_idx`,
//! atomic `swap()`. The instance pattern (vs module-level globals) lets
//! medvol and lance-graph-render each own their own `SplatRenderer`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::hpc::splat3d::gaussian::GaussianBatch;
use crate::hpc::splat3d::project::{project_batch, Camera, ProjectedBatch};
use crate::hpc::splat3d::raster::rasterize_frame;
use crate::hpc::splat3d::tile::TileBinning;

// ════════════════════════════════════════════════════════════════════════════
// SplatFrame — one frame's full state
// ════════════════════════════════════════════════════════════════════════════

/// One rendered frame's full state — input scene + intermediate
/// projection + tile binning + output framebuffer.
///
/// `tick(&mut self, ...)` runs the full PR 1–5 pipeline:
///   `project_batch → TileBinning::from_projected → rasterize_frame → frame_id += 1`
///
/// The `gaussians` field is owned by the frame for simplicity; a future
/// lance-graph sprint may refactor to `Arc<GaussianBatch>` for sharing.
pub struct SplatFrame {
    /// Input scene data.
    pub gaussians: GaussianBatch,
    /// Per-frame EWA projection output.
    pub projected: ProjectedBatch,
    /// Per-frame tile binning.
    ///
    /// Starts as an empty default (`tile_cols = 0`, `tile_rows = 0`,
    /// `instances` empty, `tile_offsets = vec![0]`). `tick()` overwrites
    /// it via `TileBinning::from_projected`.
    pub binning: TileBinning,
    /// Output RGB pixel buffer, interleaved: `[R0, G0, B0, R1, G1, B1, …]`.
    /// Length = `3 * width * height`.
    pub framebuffer: Vec<f32>,
    /// Image width in pixels (immutable after construction).
    pub width: u32,
    /// Image height in pixels (immutable after construction).
    pub height: u32,
    /// Monotonically incrementing render count; starts at 0, incremented
    /// at the END of each successful `tick()`.
    pub frame_id: u64,
}

impl SplatFrame {
    /// Allocate empty frame with `n_gaussians` capacity (rounded up
    /// internally by `GaussianBatch::with_capacity`) and a `width × height`
    /// framebuffer. All output buffers are zero-initialized.
    pub fn with_capacity(n_gaussians: usize, width: u32, height: u32) -> Self {
        let fb_len = 3 * (width as usize) * (height as usize);
        Self {
            gaussians: GaussianBatch::with_capacity(n_gaussians),
            projected: ProjectedBatch::with_capacity(n_gaussians),
            // Empty-default TileBinning: valid but holds no instances.
            binning: TileBinning {
                tile_cols: 0,
                tile_rows: 0,
                instances: Vec::new(),
                tile_offsets: vec![0],
            },
            framebuffer: vec![0.0_f32; fb_len],
            width,
            height,
            frame_id: 0,
        }
    }

    /// Run the full forward pipeline: project → bin → rasterize.
    /// Increments `frame_id`. Reads `self.gaussians` as input; writes
    /// every other field.
    pub fn tick(&mut self, camera: &Camera, background: [f32; 3]) {
        // 1. EWA projection: world gaussians → screen-space conic + depth + color
        project_batch(&self.gaussians, camera, &mut self.projected);

        // 2. Tile binning: AABB intersection + radix-sort by (tile_id, depth)
        self.binning = TileBinning::from_projected(&self.projected, camera);

        // 3. Rasterize: depth-sorted alpha-blend into framebuffer
        rasterize_frame(&self.binning, &self.projected, &mut self.framebuffer, self.width, self.height, background);

        // 4. Advance frame counter
        self.frame_id += 1;
    }

    /// Total bytes resident in this frame's owned storage (debug / health).
    pub fn byte_footprint(&self) -> usize {
        // GaussianBatch: 11 f32 vecs × capacity + SH vec
        let g = &self.gaussians;
        let gaussian_bytes = (g.mean_x.len()
            + g.mean_y.len()
            + g.mean_z.len()
            + g.scale_x.len()
            + g.scale_y.len()
            + g.scale_z.len()
            + g.quat_w.len()
            + g.quat_x.len()
            + g.quat_y.len()
            + g.quat_z.len()
            + g.opacity.len())
            * 4
            + g.sh.len() * 4;

        // ProjectedBatch: 10 f32 vecs × capacity + 1 u8 vec
        let p = &self.projected;
        let projected_bytes = (p.screen_x.len()
            + p.screen_y.len()
            + p.depth.len()
            + p.conic_a.len()
            + p.conic_b.len()
            + p.conic_c.len()
            + p.radius.len()
            + p.color_r.len()
            + p.color_g.len()
            + p.color_b.len()
            + p.opacity.len())
            * 4
            + p.valid.len();

        // TileBinning
        let binning_bytes = self.binning.instances.len() * 16 + self.binning.tile_offsets.len() * 4;

        // Framebuffer
        let fb_bytes = self.framebuffer.len() * 4;

        gaussian_bytes + projected_bytes + binning_bytes + fb_bytes
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SplatRenderer — double-buffered driver
// ════════════════════════════════════════════════════════════════════════════

/// Double-buffered `SplatFrame` driver. Two pre-allocated `SplatFrame`s
/// live in `frames[0]` / `frames[1]`. `front_idx` (0 or 1) names the
/// frame readers see; the back frame is `1 - front_idx`. `swap()`
/// flips the index atomically — no allocation.
///
/// Readers acquire a read lock on the FRONT frame; the render cycle
/// acquires a write lock on the BACK frame. They never contend.
pub struct SplatRenderer {
    /// Two pre-allocated frames (front + back).
    pub frames: [RwLock<SplatFrame>; 2],
    /// Index of the frame currently visible to readers (0 or 1).
    front_idx: AtomicUsize,
    /// Global monotonic tick counter (incremented once per `tick()` call).
    /// Used to set each back frame's `frame_id` to the GLOBAL render count,
    /// not the per-slot render count, so `front_frame_id()` reflects the
    /// number of times `SplatRenderer::tick()` has been called.
    tick_count: AtomicU64,
}

impl SplatRenderer {
    /// Allocate a renderer with two `SplatFrame`s of the given capacity.
    pub fn with_capacity(n_gaussians: usize, width: u32, height: u32) -> Self {
        Self {
            frames: [
                RwLock::new(SplatFrame::with_capacity(n_gaussians, width, height)),
                RwLock::new(SplatFrame::with_capacity(n_gaussians, width, height)),
            ],
            front_idx: AtomicUsize::new(0),
            tick_count: AtomicU64::new(0),
        }
    }

    /// Index of the currently-front frame (0 or 1).
    #[inline]
    pub fn front_index(&self) -> usize {
        self.front_idx.load(Ordering::Acquire)
    }

    /// Index of the currently-back frame (`1 - front_idx`).
    #[inline]
    pub fn back_index(&self) -> usize {
        1 - self.front_index()
    }

    /// Read-lock the front frame (for REST / SSE consumers).
    pub fn read_front(&self) -> RwLockReadGuard<'_, SplatFrame> {
        self.frames[self.front_index()]
            .read()
            .expect("SplatRenderer: front lock poisoned")
    }

    /// Write-lock the back frame (for the render cycle to mutate).
    pub fn write_back(&self) -> RwLockWriteGuard<'_, SplatFrame> {
        self.frames[self.back_index()]
            .write()
            .expect("SplatRenderer: back lock poisoned")
    }

    /// Atomically swap front and back. Readers acquired BEFORE the swap
    /// keep observing the old front; subsequent readers see the new front.
    pub fn swap(&self) {
        // XOR-flip via fetch_xor — single atomic write, matches Renderer::swap.
        self.front_idx.fetch_xor(1, Ordering::AcqRel);
    }

    /// Render to the back frame, then atomically promote it to front.
    ///
    /// The gaussians in the BACK frame are used as input (set them up before
    /// calling tick, or copy from the front frame first if needed). Subsequent
    /// calls to `read_front()` will observe the newly-rendered frame.
    ///
    /// `frame_id` on the rendered frame is set to the GLOBAL tick count
    /// (1-based), not the per-slot render count, so `front_frame_id()` always
    /// reflects how many times `SplatRenderer::tick()` has been called.
    pub fn tick(&self, camera: &Camera, background: [f32; 3]) {
        let next_id = self.tick_count.fetch_add(1, Ordering::AcqRel) + 1;
        {
            let mut back = self.write_back();
            // Delegate to SplatFrame::tick (which uses its own per-slot counter),
            // then overwrite frame_id with the global monotonic count.
            back.tick(camera, background);
            back.frame_id = next_id;
        }
        self.swap();
    }

    /// `frame_id` of the currently-visible front frame.
    pub fn front_frame_id(&self) -> u64 {
        self.read_front().frame_id
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::splat3d::gaussian::Gaussian3D;

    // ── Test 1 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_frame_with_capacity_allocates_correctly() {
        let frame = SplatFrame::with_capacity(100, 64, 48);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.framebuffer.len(), 3 * 64 * 48);
        assert!(frame.gaussians.capacity >= 100, "capacity {} < 100", frame.gaussians.capacity);
        assert_eq!(frame.frame_id, 0);
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_frame_tick_runs_pipeline_and_increments_id() {
        let mut frame = SplatFrame::with_capacity(0, 32, 32);
        let camera = Camera::identity_at_origin(32, 32);
        frame.tick(&camera, [0.0, 0.0, 0.0]);
        assert_eq!(frame.frame_id, 1);
        // With zero gaussians, framebuffer must be all-black (background = black)
        assert!(
            frame.framebuffer.iter().all(|&v| v == 0.0),
            "framebuffer should be all black with zero gaussians and black background"
        );
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_frame_tick_renders_visible_gaussian() {
        let mut frame = SplatFrame::with_capacity(1, 64, 64);
        let camera = Camera::identity_at_origin(64, 64);

        // One bright opaque gaussian at (0, 0, 1) — directly in front of the camera
        let mut g = Gaussian3D::unit();
        g.mean = [0.0, 0.0, 1.0];
        g.opacity = 1.0;
        // Set the DC (l=0) SH coefficient for each channel to produce a bright color.
        // DC index is 0 per channel; layout: sh[ch*16 + basis_idx].
        // SH DC contribution: color = 0.5 + 0.282_095 * sh_dc
        // To get color > background (0.0), we need a positive DC.
        // Use a large positive value so the clamped output is clearly > 0.
        g.sh[0] = 3.0; // R channel DC
        g.sh[16] = 3.0; // G channel DC
        g.sh[32] = 3.0; // B channel DC
        g.scale = [0.5, 0.5, 0.5]; // Visible screen-space radius

        frame.gaussians.push(g);
        frame.tick(&camera, [0.0, 0.0, 0.0]);

        // Screen center pixel index: (cy * width + cx) * 3
        let cx = 32usize;
        let cy = 32usize;
        let idx = (cy * 64 + cx) * 3;
        let r = frame.framebuffer[idx];
        assert!(r > 0.0, "center pixel R={r} should be > 0 after rendering a bright gaussian");
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_frame_tick_monotonic_id() {
        let mut frame = SplatFrame::with_capacity(0, 16, 16);
        let camera = Camera::identity_at_origin(16, 16);
        for expected in 1u64..=5 {
            frame.tick(&camera, [0.0, 0.0, 0.0]);
            assert_eq!(frame.frame_id, expected);
        }
    }

    // ── Test 5 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_renderer_front_back_indices_are_complementary() {
        let r = SplatRenderer::with_capacity(0, 16, 16);
        assert_eq!(r.front_index(), 0);
        assert_eq!(r.back_index(), 1);
        r.swap();
        assert_eq!(r.front_index(), 1);
        assert_eq!(r.back_index(), 0);
    }

    // ── Test 6 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_renderer_swap_is_xor_flip() {
        let r = SplatRenderer::with_capacity(0, 16, 16);
        r.swap();
        r.swap();
        assert_eq!(r.front_index(), 0);
    }

    // ── Test 7 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_renderer_tick_advances_front_frame_id() {
        let r = SplatRenderer::with_capacity(0, 16, 16);
        let camera = Camera::identity_at_origin(16, 16);
        assert_eq!(r.front_frame_id(), 0);
        r.tick(&camera, [0.0, 0.0, 0.0]);
        assert_eq!(r.front_frame_id(), 1);
        r.tick(&camera, [0.0, 0.0, 0.0]);
        assert_eq!(r.front_frame_id(), 2);
    }

    // ── Test 8 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_renderer_concurrent_read_does_not_block_write() {
        use std::sync::Arc;
        let renderer = Arc::new(SplatRenderer::with_capacity(0, 16, 16));
        let renderer2 = Arc::clone(&renderer);

        // Spawn a thread that holds a read lock on the FRONT frame
        let handle = std::thread::spawn(move || {
            let _guard = renderer2.read_front();
            // Hold it briefly; drop at end of scope
        });

        // On the main thread, obtain a write lock on the BACK frame.
        // This must not block, since front and back are different locks.
        {
            let _back = renderer.write_back();
            // Back write succeeds even while front read is (or was) held
        }

        handle.join().expect("thread panicked");
    }

    // ── Test 9 ───────────────────────────────────────────────────────────────

    #[test]
    fn splat_frame_byte_footprint_nonzero() {
        let frame = SplatFrame::with_capacity(64, 32, 32);
        assert!(frame.byte_footprint() > 0, "byte_footprint should be > 0 for a non-empty frame");
    }

    // ── Test 10 ──────────────────────────────────────────────────────────────

    #[test]
    fn splat_renderer_two_ticks_render_to_different_buffers() {
        let r = SplatRenderer::with_capacity(0, 16, 16);
        let camera = Camera::identity_at_origin(16, 16);

        // After tick 1: the back (index 1) was written and swapped to front (now at 0).
        // Wait — let's track which physical slot is back each tick.
        // Before tick 1: front=0, back=1. Tick writes to slot 1, then swaps → front=1.
        // Before tick 2: front=1, back=0. Tick writes to slot 0, then swaps → front=0.
        // So after each tick, we capture the CURRENT back slot's framebuffer pointer.

        // Before tick 1, back is slot 1.
        let ptr_before_tick1: *const f32 = {
            let back = r.write_back();
            back.framebuffer.as_ptr()
        };

        r.tick(&camera, [0.0, 0.0, 0.0]);

        // After tick 1 (front swapped to 1), back is now slot 0.
        let ptr_before_tick2: *const f32 = {
            let back = r.write_back();
            back.framebuffer.as_ptr()
        };

        r.tick(&camera, [0.0, 0.0, 0.0]);

        assert_ne!(ptr_before_tick1, ptr_before_tick2, "two ticks must render to different physical frame buffers");
    }
}
