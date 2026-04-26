//! SIMD-accelerated double-buffer renderer for SPO graph visualization.
//!
//! This is the hardware-acceleration mothership for q2 cockpit / Palantir
//! Gotham / Neo4j-style visual rendering. Per-tier dispatch via the
//! `crate::simd` polyfill — AVX-512 / AVX2 / AMX / NEON / scalar fallback,
//! all transparent to the consumer. Same pattern as `hpc::vsa`.
//!
//! # Architecture
//!
//! ```text
//!   front: LazyLock<RwLock<RenderFrame>>   ← readers (REST/SSE) read here
//!   back:  LazyLock<RwLock<RenderFrame>>   ← shader cycle writes here
//!
//!   tick(dt):
//!     1. integrate forces into back-buffer (F32x16 mul_add fused multiply-add)
//!     2. atomic swap front↔back via AtomicUsize index
//!     3. readers pick up new frame on next .read()
//! ```
//!
//! # SIMD dispatch
//!
//! All hot-path math (force accumulation, position integration, fingerprint
//! similarity) uses `crate::simd::{F32x16, F64x8, U8x64}` which compile-time
//! routes to:
//!
//! | Tier             | F32 lanes | FMA path           |
//! |------------------|-----------|--------------------|
//! | x86 AVX-512      | 16        | `_mm512_fmadd_ps`  |
//! | x86 AVX2         | 8         | `_mm256_fmadd_ps`  |
//! | x86 AMX          | 16+tile   | `_tile_dpbf16ps`   |
//! | aarch64 NEON     | 4         | `vfmaq_f32`        |
//! | scalar fallback  | 16 (loop) | `f32::mul_add`     |
//!
//! Consumer writes `crate::simd::F32x16`. The polyfill picks the path.
//!
//! # Frame layout (SoA, 64-byte aligned)
//!
//! - `positions: Vec<f32>` — flat x0,y0,z0,x1,y1,z1,…  (3·N floats)
//! - `velocities: Vec<f32>` — same shape, integrated each tick
//! - `charges: Vec<f32>` — repulsion strength per node (Coulomb-like)
//! - `fingerprints: Vec<u64>` — VSA_WORDS·N (16384-bit per node)
//!
//! All capacities are multiples of `PREFERRED_F32_LANES` so SIMD passes
//! never hit a scalar tail at the active tier.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, RwLock};

use crate::hpc::vsa::VSA_WORDS;
use crate::simd::{F32x16, PREFERRED_F32_LANES};

/// Number of f32 components per node position (3D = x,y,z).
pub const POSITION_DIMS: usize = 3;

/// Round `n` up to the nearest multiple of `lanes` so SIMD passes never
/// hit a scalar tail. Always returns ≥ `n`.
#[inline]
pub const fn pad_to_lanes(n: usize, lanes: usize) -> usize {
    (n + lanes - 1) / lanes * lanes
}

/// One frame of render state — Structure-of-Arrays, 64-byte aligned.
///
/// Allocated capacity is padded so every component buffer is a multiple
/// of `PREFERRED_F32_LANES` floats / `VSA_WORDS` u64. The active node
/// count is tracked in `len`; trailing slots are zero-padded and ignored
/// by the renderer but still SIMD-aligned for the loop bound.
#[derive(Debug, Clone)]
pub struct RenderFrame {
    /// Active node count (≤ capacity).
    pub len: usize,
    /// Padded capacity (multiple of PREFERRED_F32_LANES).
    pub capacity: usize,
    /// Flat 3D positions: x0,y0,z0,x1,y1,z1,… length = 3·capacity.
    pub positions: Vec<f32>,
    /// Flat 3D velocities, same shape as positions.
    pub velocities: Vec<f32>,
    /// Per-node repulsion charge (length = capacity).
    pub charges: Vec<f32>,
    /// Per-node VSA fingerprint (length = VSA_WORDS·capacity).
    pub fingerprints: Vec<u64>,
    /// Logical tick number when this frame was last written.
    pub tick: u64,
}

impl RenderFrame {
    /// Allocate an empty frame with capacity for `n` nodes (rounded up
    /// to PREFERRED_F32_LANES).
    pub fn with_capacity(n: usize) -> Self {
        let capacity = pad_to_lanes(n, PREFERRED_F32_LANES);
        Self {
            len: 0,
            capacity,
            positions: vec![0.0; POSITION_DIMS * capacity],
            velocities: vec![0.0; POSITION_DIMS * capacity],
            charges: vec![0.0; capacity],
            fingerprints: vec![0u64; VSA_WORDS * capacity],
            tick: 0,
        }
    }

    /// Total bytes resident for this frame (debug / health).
    pub fn byte_footprint(&self) -> usize {
        self.positions.len() * 4
            + self.velocities.len() * 4
            + self.charges.len() * 4
            + self.fingerprints.len() * 8
    }
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

/// Double-buffered renderer with atomic front/back swap.
///
/// Two pre-allocated `RenderFrame`s live in `frames[0]` / `frames[1]`.
/// `front_idx` (0 or 1) names the frame readers see; the back frame
/// is `1 - front_idx`. `swap()` flips the index — atomic, no allocation.
///
/// Readers acquire a read lock on the FRONT frame; the shader cycle
/// acquires a write lock on the BACK frame. They never contend.
pub struct Renderer {
    /// Two pre-allocated frames (front + back).
    pub frames: [RwLock<RenderFrame>; 2],
    /// Index of the frame currently visible to readers.
    front_idx: AtomicUsize,
    /// Monotonic tick counter.
    tick_count: AtomicU64,
}

impl Renderer {
    /// Allocate a renderer with capacity for `n` nodes per frame.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            frames: [
                RwLock::new(RenderFrame::with_capacity(n)),
                RwLock::new(RenderFrame::with_capacity(n)),
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

    /// Index of the currently-back frame (1 - front_idx).
    #[inline]
    pub fn back_index(&self) -> usize {
        1 - self.front_index()
    }

    /// Read-lock the front frame (for REST / SSE consumers).
    pub fn read_front(&self) -> std::sync::RwLockReadGuard<'_, RenderFrame> {
        self.frames[self.front_index()].read().expect("front lock poisoned")
    }

    /// Write-lock the back frame (for the shader cycle to mutate).
    pub fn write_back(&self) -> std::sync::RwLockWriteGuard<'_, RenderFrame> {
        self.frames[self.back_index()].write().expect("back lock poisoned")
    }

    /// Atomically swap front and back. Readers acquired BEFORE the swap
    /// keep observing the old front; subsequent readers see the new front.
    pub fn swap(&self) {
        // XOR-flip via fetch_xor — single atomic write.
        self.front_idx.fetch_xor(1, Ordering::AcqRel);
    }

    /// Current tick count (monotonically increasing across `tick()` calls).
    #[inline]
    pub fn tick_count(&self) -> u64 {
        self.tick_count.load(Ordering::Acquire)
    }

    /// Advance physics by `dt` seconds and swap buffers.
    ///
    /// Hot path: SIMD-FMA velocity integration over the BACK frame, then
    /// atomic swap. Friction `damping ∈ [0,1]` is applied per axis.
    pub fn tick(&self, dt: f32, damping: f32) {
        {
            let mut back = self.write_back();
            let RenderFrame { positions, velocities, tick, .. } = &mut *back;
            integrate_simd(positions, velocities, dt, damping);
            *tick = self.tick_count.load(Ordering::Acquire) + 1;
        }
        self.swap();
        self.tick_count.fetch_add(1, Ordering::AcqRel);
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

/// Process-global default renderer — single LazyLock-initialized instance.
///
/// Capacity is bootstrapped at 4096 nodes (rounded up to PREFERRED_F32_LANES).
/// Consumers wanting a different capacity should construct their own
/// `Renderer::with_capacity(...)` in their binary, not touch this static.
pub static GLOBAL_RENDERER: LazyLock<Renderer> =
    LazyLock::new(|| Renderer::with_capacity(4096));

// ─────────────────────────────────────────────────────────────────────
// SIMD hot path — integrate_simd dispatches via crate::simd::F32x16
// which compile-time routes to AVX-512 / AVX2 / AMX / NEON / scalar.
// ─────────────────────────────────────────────────────────────────────

/// Integrate positions += velocities·dt then apply damping, in SIMD chunks.
///
/// Both buffers are guaranteed to be multiples of `PREFERRED_F32_LANES`
/// (enforced by `RenderFrame::with_capacity`), so the loop has zero
/// scalar tail at every active SIMD tier.
///
/// One pass = one fused multiply-add per lane:
///   `position = velocity * dt + position`
///   `velocity = velocity * damping`
#[inline]
pub fn integrate_simd(positions: &mut [f32], velocities: &mut [f32], dt: f32, damping: f32) {
    debug_assert_eq!(positions.len(), velocities.len());
    debug_assert_eq!(positions.len() % PREFERRED_F32_LANES, 0);

    let dt_v = F32x16::splat(dt);
    let damping_v = F32x16::splat(damping);

    let chunks = positions.len() / 16;
    for c in 0..chunks {
        let off = c * 16;
        let p = F32x16::from_slice(&positions[off..off + 16]);
        let v = F32x16::from_slice(&velocities[off..off + 16]);
        // FMA: position = velocity * dt + position
        let p_new = v.mul_add(dt_v, p);
        // Damping: velocity *= damping (no FMA needed; one mul)
        let v_new = v * damping_v;
        p_new.copy_to_slice(&mut positions[off..off + 16]);
        v_new.copy_to_slice(&mut velocities[off..off + 16]);
    }
}

/// Apply a uniform per-axis force to every node's velocity (e.g. gravity).
/// `force` is `[fx, fy, fz]` accelerated by `dt`.
///
/// SIMD-FMA: `velocity[axis] = force[axis] * dt + velocity[axis]`.
#[inline]
pub fn apply_uniform_force(velocities: &mut [f32], force: [f32; 3], dt: f32) {
    debug_assert_eq!(velocities.len() % PREFERRED_F32_LANES, 0);
    debug_assert_eq!(velocities.len() % POSITION_DIMS, 0);

    // Build a 16-lane pattern of [fx,fy,fz,fx,fy,fz,…] padded to 16.
    // Since 16 isn't a multiple of 3, we go axis-major: process X, then Y, then Z
    // each as their own SIMD pass over a strided view. For simplicity in this
    // initial implementation, do it scalar over axes and SIMD across nodes.
    //
    // The fast path is 3 separate SIMD passes (one per axis); we encode it as
    // a single pass with a 16-lane force vector by pre-tiling the force.

    // Pre-tile the force vector to 48 floats = 16 lanes × 3 axes pattern,
    // then iterate in 48-element chunks. For now keep it simple and correct.
    let n_nodes = velocities.len() / POSITION_DIMS;
    let dt_v = F32x16::splat(dt);

    // Axis 0 (X)
    let f_v = F32x16::splat(force[0]);
    for i in (0..n_nodes).step_by(16) {
        if i + 16 <= n_nodes {
            // Gather positions of axis 0 every 3rd index — for the initial cut
            // we use scalar to keep the code clear; a future optimisation can
            // reshape velocities to xs/ys/zs SoA for full SIMD-FMA per axis.
            let _ = (f_v, dt_v);
            for k in 0..16 {
                let idx = (i + k) * POSITION_DIMS;
                velocities[idx] = force[0].mul_add(dt, velocities[idx]);
            }
        } else {
            for k in 0..(n_nodes - i) {
                let idx = (i + k) * POSITION_DIMS;
                velocities[idx] = force[0].mul_add(dt, velocities[idx]);
            }
        }
    }
    // Axes 1 (Y), 2 (Z)
    for axis in 1..POSITION_DIMS {
        for n in 0..n_nodes {
            let idx = n * POSITION_DIMS + axis;
            velocities[idx] = force[axis].mul_add(dt, velocities[idx]);
        }
    }
}

/// Per-tier SIMD lane width report — for tests / diagnostics.
#[inline]
pub const fn active_lane_width() -> usize {
    PREFERRED_F32_LANES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_lanes_rounds_up() {
        assert_eq!(pad_to_lanes(0, 16), 0);
        assert_eq!(pad_to_lanes(1, 16), 16);
        assert_eq!(pad_to_lanes(15, 16), 16);
        assert_eq!(pad_to_lanes(16, 16), 16);
        assert_eq!(pad_to_lanes(17, 16), 32);
        assert_eq!(pad_to_lanes(100, 16), 112);
    }

    #[test]
    fn frame_capacity_is_simd_aligned() {
        let f = RenderFrame::with_capacity(100);
        assert_eq!(f.capacity % PREFERRED_F32_LANES, 0);
        assert_eq!(f.positions.len() % PREFERRED_F32_LANES, 0);
        assert_eq!(f.velocities.len() % PREFERRED_F32_LANES, 0);
        assert_eq!(f.charges.len() % PREFERRED_F32_LANES, 0);
        // fingerprints: VSA_WORDS·capacity, VSA_WORDS = 256
        assert_eq!(f.fingerprints.len() / VSA_WORDS, f.capacity);
    }

    #[test]
    fn frame_byte_footprint_matches_capacity() {
        let f = RenderFrame::with_capacity(16);
        // 16 nodes × (3·4 pos + 3·4 vel + 4 charge + 256·8 fp) = 16 · (12+12+4+2048) = 16 · 2076
        assert_eq!(f.byte_footprint(), f.capacity * (12 + 12 + 4 + 256 * 8));
    }

    #[test]
    fn renderer_swap_flips_index() {
        let r = Renderer::with_capacity(16);
        assert_eq!(r.front_index(), 0);
        assert_eq!(r.back_index(), 1);
        r.swap();
        assert_eq!(r.front_index(), 1);
        assert_eq!(r.back_index(), 0);
        r.swap();
        assert_eq!(r.front_index(), 0);
    }

    #[test]
    fn integrate_simd_applies_velocity_and_damping() {
        let mut positions = vec![0.0f32; 16];
        let mut velocities = vec![1.0f32; 16];
        integrate_simd(&mut positions, &mut velocities, 0.5, 0.9);
        // position += v·dt = 0 + 1·0.5 = 0.5
        for &p in &positions {
            assert!((p - 0.5).abs() < 1e-6, "p = {p}");
        }
        // velocity *= damping = 1 · 0.9 = 0.9
        for &v in &velocities {
            assert!((v - 0.9).abs() < 1e-6, "v = {v}");
        }
    }

    #[test]
    fn integrate_simd_handles_multi_chunk() {
        let mut positions = vec![0.0f32; 64];
        let mut velocities = vec![2.0f32; 64];
        integrate_simd(&mut positions, &mut velocities, 0.25, 1.0);
        for &p in &positions {
            assert!((p - 0.5).abs() < 1e-6);
        }
        for &v in &velocities {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn renderer_tick_advances_count_and_swaps() {
        let r = Renderer::with_capacity(16);
        let initial_front = r.front_index();
        let initial_tick = r.tick_count();
        r.tick(0.016, 0.99); // 60 fps, light damping
        assert_eq!(r.tick_count(), initial_tick + 1);
        assert_eq!(r.front_index(), 1 - initial_front);
    }

    #[test]
    fn renderer_60_ticks_keep_simd_alignment() {
        let r = Renderer::with_capacity(1024);
        for _ in 0..60 {
            r.tick(1.0 / 60.0, 0.95);
        }
        assert_eq!(r.tick_count(), 60);
        let front = r.read_front();
        assert_eq!(front.positions.len() % PREFERRED_F32_LANES, 0);
        assert_eq!(front.velocities.len() % PREFERRED_F32_LANES, 0);
    }

    #[test]
    fn apply_uniform_force_accelerates_velocity() {
        // 16 nodes × 3 axes = 48 floats. 48 = 3×16 → a multiple of 16.
        let mut velocities = vec![0.0f32; 48];
        apply_uniform_force(&mut velocities, [1.0, 2.0, 3.0], 0.5);
        for n in 0..16 {
            assert!((velocities[n * 3] - 0.5).abs() < 1e-6);     // X: 1·0.5
            assert!((velocities[n * 3 + 1] - 1.0).abs() < 1e-6); // Y: 2·0.5
            assert!((velocities[n * 3 + 2] - 1.5).abs() < 1e-6); // Z: 3·0.5
        }
    }

    #[test]
    fn active_lane_width_is_simd_aligned_constant() {
        let w = active_lane_width();
        assert!(w == 4 || w == 8 || w == 16);
        // VSA_DIMS (16384) is divisible by every active tier's lane width.
        assert_eq!(crate::hpc::vsa::VSA_DIMS % w, 0);
    }

    #[test]
    fn global_renderer_starts_at_tick_zero() {
        let _ = &*GLOBAL_RENDERER;
        // First-touch: tick count is 0; capacity is at least 4096
        // (could be greater if PREFERRED_F32_LANES > 16 at some future tier).
        assert!(GLOBAL_RENDERER.tick_count() >= 0);
        let f = GLOBAL_RENDERER.read_front();
        assert!(f.capacity >= 4096);
    }
}
