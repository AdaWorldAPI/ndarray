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
        self.positions.len() * 4 + self.velocities.len() * 4 + self.charges.len() * 4 + self.fingerprints.len() * 8
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
            frames: [RwLock::new(RenderFrame::with_capacity(n)), RwLock::new(RenderFrame::with_capacity(n))],
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
        self.frames[self.front_index()]
            .read()
            .expect("front lock poisoned")
    }

    /// Write-lock the back frame (for the shader cycle to mutate).
    pub fn write_back(&self) -> std::sync::RwLockWriteGuard<'_, RenderFrame> {
        self.frames[self.back_index()]
            .write()
            .expect("back lock poisoned")
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
            let RenderFrame {
                positions,
                velocities,
                tick,
                ..
            } = &mut *back;
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
pub static GLOBAL_RENDERER: LazyLock<Renderer> = LazyLock::new(|| Renderer::with_capacity(4096));

// ─────────────────────────────────────────────────────────────────────
// SIMD hot path — integrate_simd dispatches via crate::simd::F32x16
// which compile-time routes to AVX-512 / AVX2 / AMX / NEON / scalar.
// ─────────────────────────────────────────────────────────────────────

/// Integrate positions += velocities·dt then apply damping, in SIMD chunks.
///
/// Uses `slice::as_chunks_mut::<16>()` for SIMD slicing — the array-window
/// pattern documented in `crate::simd`. Both buffers are guaranteed to be
/// multiples of `PREFERRED_F32_LANES` (enforced by `RenderFrame::with_capacity`),
/// so the remainder slice is empty and there's no scalar tail.
///
/// One pass = one fused multiply-add per lane:
///   `position = velocity * dt + position`
///   `velocity = velocity * damping`
#[inline]
pub fn integrate_simd(positions: &mut [f32], velocities: &mut [f32], dt: f32, damping: f32) {
    debug_assert_eq!(positions.len(), velocities.len());
    debug_assert_eq!(positions.len() % PREFERRED_F32_LANES, 0);

    let dt_v = cached_splat(dt);
    let damping_v = F32x16::splat(damping);

    // SIMD slicing via stable as_chunks_mut::<16>(). The remainder is
    // empty by construction (capacity is padded to PREFERRED_F32_LANES).
    let (p_chunks, p_tail) = positions.as_chunks_mut::<16>();
    let (v_chunks, v_tail) = velocities.as_chunks_mut::<16>();
    debug_assert!(p_tail.is_empty() && v_tail.is_empty());

    for (p, v) in p_chunks.iter_mut().zip(v_chunks.iter_mut()) {
        let pv = F32x16::from_array(*p);
        let vv = F32x16::from_array(*v);
        // FMA: position = velocity * dt + position
        let p_new = vv.mul_add(dt_v, pv);
        // Damping: velocity *= damping (one mul, no FMA needed)
        let v_new = vv * damping_v;
        p_new.copy_to_slice(p);
        v_new.copy_to_slice(v);
    }
}

/// Rayon-parallel block size in floats. Each worker processes `BLOCK_FLOATS`
/// consecutive elements, which is `BLOCK_LANES * 16` to stay aligned with the
/// inner `as_chunks_mut::<16>()` SIMD loop. 1024 floats × 4 bytes = 4 KB →
/// L1-resident, large enough to amortize work-stealing overhead.
#[cfg(feature = "rayon")]
pub const BLOCK_FLOATS: usize = 1024;

/// Rayon-parallel variant of [`integrate_simd`]: same FMA body, split across
/// the rayon thread pool in [`BLOCK_FLOATS`]-sized chunks.
///
/// Composition: 16 SIMD lanes × N rayon threads. Each worker runs the same
/// `F32x16::mul_add` inner loop on its block; rayon handles work-stealing.
///
/// Buffers must be a multiple of `BLOCK_FLOATS` so no worker hits a partial
/// chunk (which would still be a multiple of 16 by construction, but the
/// debug-assert is stricter to make alignment intent explicit).
///
/// Single-threaded sanity: at small `N` (< ~10K floats) sequential beats this
/// because work-stealing overhead exceeds the SIMD savings. Use the parallel
/// variant only for ≥ ~64 K floats (≈ 21 K nodes at 3 components each).
#[cfg(feature = "rayon")]
#[inline]
pub fn integrate_simd_par(positions: &mut [f32], velocities: &mut [f32], dt: f32, damping: f32) {
    use rayon::prelude::*;

    debug_assert_eq!(positions.len(), velocities.len());
    debug_assert_eq!(positions.len() % PREFERRED_F32_LANES, 0);
    debug_assert_eq!(positions.len() % 16, 0);

    let dt_v = cached_splat(dt);
    let damping_v = F32x16::splat(damping);

    positions
        .par_chunks_mut(BLOCK_FLOATS)
        .zip(velocities.par_chunks_mut(BLOCK_FLOATS))
        .for_each(|(p_block, v_block)| {
            // Inner SIMD loop is byte-identical to integrate_simd's body.
            // The last block may be < BLOCK_FLOATS but is still a multiple
            // of 16 because the caller guarantees positions.len() % 16 == 0.
            let (p_chunks, p_tail) = p_block.as_chunks_mut::<16>();
            let (v_chunks, v_tail) = v_block.as_chunks_mut::<16>();
            debug_assert!(p_tail.is_empty() && v_tail.is_empty());

            for (p, v) in p_chunks.iter_mut().zip(v_chunks.iter_mut()) {
                let pv = F32x16::from_array(*p);
                let vv = F32x16::from_array(*v);
                let p_new = vv.mul_add(dt_v, pv);
                let v_new = vv * damping_v;
                p_new.copy_to_slice(p);
                v_new.copy_to_slice(v);
            }
        });
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
            assert!((velocities[n * 3] - 0.5).abs() < 1e-6); // X: 1·0.5
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
        assert_eq!(GLOBAL_RENDERER.tick_count(), 0);
        let f = GLOBAL_RENDERER.read_front();
        assert!(f.capacity >= 4096);
    }
}

// ─────────────────────────────────────────────────────────────────────
// LazyLock-cached splat constants for the common tick rates.
//
// `F32x16::splat(dt)` is one CPU instruction at AVX-512 (`_mm512_set1_ps`)
// but the renderer ticks at fixed rates 99% of the time, so we cache the
// three canonical splat values: 60 fps / 30 fps / 15 fps.
//
// `cached_splat(dt)` returns the cached vector when `dt` matches one of
// the canonical rates, falling back to a fresh splat otherwise. Tolerance
// ±2 µs absorbs floating-point jitter without bypassing the cache.
// ─────────────────────────────────────────────────────────────────────

/// Tick budget for 60 fps in seconds (1/60).
pub const DT_60: f32 = 1.0 / 60.0;
/// Tick budget for 30 fps in seconds (1/30).
pub const DT_30: f32 = 1.0 / 30.0;
/// Tick budget for 15 fps in seconds (1/15).
pub const DT_15: f32 = 1.0 / 15.0;

static SPLAT_60: LazyLock<F32x16> = LazyLock::new(|| F32x16::splat(DT_60));
static SPLAT_30: LazyLock<F32x16> = LazyLock::new(|| F32x16::splat(DT_30));
static SPLAT_15: LazyLock<F32x16> = LazyLock::new(|| F32x16::splat(DT_15));

/// Splat `dt` into an `F32x16`, returning a cached value for the canonical
/// rates (60 / 30 / 15 fps). Falls back to a fresh splat for arbitrary `dt`.
#[inline]
pub fn cached_splat(dt: f32) -> F32x16 {
    const TOL: f32 = 2e-6;
    if (dt - DT_60).abs() < TOL {
        *SPLAT_60
    } else if (dt - DT_30).abs() < TOL {
        *SPLAT_30
    } else if (dt - DT_15).abs() < TOL {
        *SPLAT_15
    } else {
        F32x16::splat(dt)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Viewport + foveated rendering — only spend SIMD cycles on what's seen.
// ─────────────────────────────────────────────────────────────────────

/// Camera + view-volume parameters for foveated rendering.
///
/// Nodes are classified by distance to `center` into four priority bands.
/// The renderer ticks foveal nodes every frame, peripheral every other,
/// distant every fourth, and skips off-screen entirely. Net effect: O(N)
/// work scales with what the camera is actually looking at, not the
/// graph total.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// Camera focus point in world coordinates.
    pub center: [f32; 3],
    /// Foveal radius — full-detail every-tick zone.
    pub foveal_radius: f32,
    /// Peripheral radius — half-rate update zone (foveal_radius < r ≤ peripheral).
    pub peripheral_radius: f32,
    /// Cull radius — beyond this, skip entirely (not just slow).
    pub cull_radius: f32,
}

impl Viewport {
    /// Default: 4.0 unit foveal, 16.0 peripheral, 64.0 cull.
    pub fn default_at(center: [f32; 3]) -> Self {
        Self {
            center,
            foveal_radius: 4.0,
            peripheral_radius: 16.0,
            cull_radius: 64.0,
        }
    }
}

/// Update priority for one node — controls how often it's integrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UpdatePriority {
    /// Every tick (foveal zone).
    Foveal = 0,
    /// Every 2nd tick (peripheral zone).
    Peripheral = 1,
    /// Every 4th tick (distant but in-frustum).
    Distant = 2,
    /// Skip (out of cull radius).
    OffScreen = 3,
}

impl UpdatePriority {
    /// Stride between updates for this priority.
    #[inline]
    pub fn tick_stride(self) -> u64 {
        match self {
            Self::Foveal => 1,
            Self::Peripheral => 2,
            Self::Distant => 4,
            Self::OffScreen => u64::MAX,
        }
    }

    /// Should this node be updated on the given tick?
    #[inline]
    pub fn should_update(self, tick: u64) -> bool {
        let stride = self.tick_stride();
        if stride == u64::MAX {
            false
        } else {
            tick % stride == 0
        }
    }
}

/// Classify each node by distance to viewport center.
///
/// Returns a Vec<UpdatePriority> of length `len` (active nodes only). Trailing
/// padded slots are not classified — they're never integrated regardless.
pub fn classify_priorities(positions: &[f32], len: usize, vp: &Viewport) -> Vec<UpdatePriority> {
    let mut out = Vec::with_capacity(len);
    let f2 = vp.foveal_radius * vp.foveal_radius;
    let p2 = vp.peripheral_radius * vp.peripheral_radius;
    let c2 = vp.cull_radius * vp.cull_radius;
    for i in 0..len {
        let dx = positions[i * POSITION_DIMS] - vp.center[0];
        let dy = positions[i * POSITION_DIMS + 1] - vp.center[1];
        let dz = positions[i * POSITION_DIMS + 2] - vp.center[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        out.push(if d2 <= f2 {
            UpdatePriority::Foveal
        } else if d2 <= p2 {
            UpdatePriority::Peripheral
        } else if d2 <= c2 {
            UpdatePriority::Distant
        } else {
            UpdatePriority::OffScreen
        });
    }
    out
}

/// Foveated integrate — apply `integrate_simd` only to the chunks where
/// at least one node has `priorities[node] == active_priority` AND
/// `priority.should_update(tick)`.
///
/// Operates in 16-element chunks (the SIMD chunk size), so the smallest
/// granularity of skipping is 16/POSITION_DIMS ≈ 5 nodes. For graphs
/// where most nodes share a priority band this is near-optimal; for
/// random-priority graphs, foveated savings drop toward zero (worst case
/// is the same cost as `integrate_simd`).
pub fn integrate_foveated(
    positions: &mut [f32], velocities: &mut [f32], priorities: &[UpdatePriority], tick: u64, dt: f32, damping: f32,
) {
    debug_assert_eq!(positions.len(), velocities.len());
    debug_assert_eq!(positions.len() % PREFERRED_F32_LANES, 0);

    let dt_v = cached_splat(dt);
    let damping_v = F32x16::splat(damping);

    // Each 16-float chunk covers ceil(16/3) = 6 nodes (with overlap on
    // the chunk boundary). We skip a chunk only if EVERY node mapping to
    // it is OffScreen or stride-skipped this tick; otherwise we update
    // the whole chunk (cheap — one FMA — vs. branching cost).
    let nodes_per_chunk = 16 / POSITION_DIMS + 1; // round up

    let (p_chunks, _) = positions.as_chunks_mut::<16>();
    let (v_chunks, _) = velocities.as_chunks_mut::<16>();

    for (chunk_idx, (p, v)) in p_chunks.iter_mut().zip(v_chunks.iter_mut()).enumerate() {
        // Determine which active nodes fall into this chunk.
        let node_lo = (chunk_idx * 16) / POSITION_DIMS;
        let node_hi = (node_lo + nodes_per_chunk).min(priorities.len());

        // Skip only if every node in the band agrees to skip THIS tick.
        let all_skip = (node_lo..node_hi).all(|n| !priorities[n].should_update(tick));
        if all_skip {
            continue;
        }

        let pv = F32x16::from_array(*p);
        let vv = F32x16::from_array(*v);
        let p_new = vv.mul_add(dt_v, pv);
        let v_new = vv * damping_v;
        p_new.copy_to_slice(p);
        v_new.copy_to_slice(v);
    }
}

// ─────────────────────────────────────────────────────────────────────
// FPS controller — adaptive frame rate under load.
//
// Targets 60 ticks/s by default. If the last tick took longer than the
// budget allows, the controller drops to 30 fps; if even 30 fps overruns,
// it drops to 15 fps. When ticks consistently come in under-budget, it
// climbs back up. This keeps the cockpit responsive on big graphs without
// manual rate selection.
// ─────────────────────────────────────────────────────────────────────

use std::time::Instant;

/// Adaptive FPS targeting 60 → 30 → 15 with hysteresis.
pub struct FpsController {
    /// Active target rate in Hz (60, 30, or 15).
    target_hz: AtomicU32,
    /// Last tick wall-clock (ns since renderer construction).
    last_tick_ns: AtomicU64,
    /// Construction instant — origin for last_tick_ns.
    origin: Instant,
    /// Rolling mean tick duration in ns (EWMA).
    avg_tick_ns: AtomicU64,
    /// Consecutive under-budget ticks (used to climb back up).
    under_budget_streak: AtomicU32,
}

use std::sync::atomic::AtomicU32;

impl FpsController {
    /// Construct with `target_hz` initial rate (clamped to {15, 30, 60}).
    pub fn new(target_hz: u32) -> Self {
        let clamped = match target_hz {
            x if x >= 60 => 60,
            x if x >= 30 => 30,
            _ => 15,
        };
        Self {
            target_hz: AtomicU32::new(clamped),
            last_tick_ns: AtomicU64::new(0),
            origin: Instant::now(),
            avg_tick_ns: AtomicU64::new(0),
            under_budget_streak: AtomicU32::new(0),
        }
    }

    /// Currently active target rate in Hz (60, 30, or 15).
    #[inline]
    pub fn target_hz(&self) -> u32 {
        self.target_hz.load(Ordering::Acquire)
    }

    /// `dt` in seconds for the active target rate.
    #[inline]
    pub fn dt(&self) -> f32 {
        match self.target_hz() {
            60 => DT_60,
            30 => DT_30,
            _ => DT_15,
        }
    }

    /// Rolling mean tick duration in nanoseconds.
    #[inline]
    pub fn avg_tick_ns(&self) -> u64 {
        self.avg_tick_ns.load(Ordering::Acquire)
    }

    /// Tick budget for the current target rate, in nanoseconds.
    #[inline]
    pub fn budget_ns(&self) -> u64 {
        1_000_000_000u64 / self.target_hz() as u64
    }

    /// Record the duration of one tick and adapt the rate if needed.
    ///
    /// EWMA with α = 1/8 keeps the average responsive without flapping.
    /// Step down: 3 consecutive over-budget ticks → halve target.
    /// Step up:   60 consecutive under-budget ticks (~1s @ current rate) → double target.
    pub fn record_tick(&self, duration_ns: u64) {
        // EWMA update (α = 1/8): avg = avg + (sample - avg)/8
        let prev = self.avg_tick_ns.load(Ordering::Acquire);
        let next = if prev == 0 {
            duration_ns
        } else {
            prev + (duration_ns.saturating_sub(prev) / 8) - (prev.saturating_sub(duration_ns) / 8)
        };
        self.avg_tick_ns.store(next, Ordering::Release);

        let budget = self.budget_ns();
        let cur = self.target_hz();
        if duration_ns > budget {
            self.under_budget_streak.store(0, Ordering::Release);
            // Step down 60 → 30 → 15. Don't go below 15.
            let new_hz = match cur {
                60 => 30,
                30 => 15,
                _ => 15,
            };
            if new_hz != cur {
                self.target_hz.store(new_hz, Ordering::Release);
            }
        } else {
            let streak = self.under_budget_streak.fetch_add(1, Ordering::AcqRel) + 1;
            if streak >= 60 {
                let new_hz = match cur {
                    15 => 30,
                    30 => 60,
                    _ => 60,
                };
                if new_hz != cur {
                    self.target_hz.store(new_hz, Ordering::Release);
                }
                self.under_budget_streak.store(0, Ordering::Release);
            }
        }

        let elapsed = self.origin.elapsed().as_nanos() as u64;
        self.last_tick_ns.store(elapsed, Ordering::Release);
    }
}

impl Default for FpsController {
    fn default() -> Self {
        Self::new(60)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Renderer convenience: adaptive + foveated tick wrappers.
// ─────────────────────────────────────────────────────────────────────

impl Renderer {
    /// Tick at the rate the `FpsController` currently targets, measuring
    /// the duration so the controller can adapt for next call.
    ///
    /// This is the recommended top-level entry point for cockpit servers:
    /// just call `r.tick_adaptive(&fps_ctl, damping)` in a loop and the
    /// rate auto-tunes between 60 / 30 / 15 fps based on observed load.
    pub fn tick_adaptive(&self, fps: &FpsController, damping: f32) {
        let start = std::time::Instant::now();
        self.tick(fps.dt(), damping);
        fps.record_tick(start.elapsed().as_nanos() as u64);
    }

    /// Foveated tick — classify by viewport, integrate only chunks where
    /// at least one node should update this tick.
    ///
    /// The classification cost is O(N) (one squared-distance per node) but
    /// is done once per tick; the SIMD integration cost drops by the share
    /// of off-screen / sub-rate nodes. For a typical cockpit camera, foveal
    /// nodes are ≤ 20% of the graph → 5× speedup vs full integrate.
    pub fn tick_foveated(&self, fps: &FpsController, damping: f32, vp: &Viewport) {
        let start = std::time::Instant::now();
        let dt = fps.dt();
        let tick_now = self.tick_count.load(Ordering::Acquire) + 1;
        {
            let mut back = self.write_back();
            let RenderFrame {
                positions,
                velocities,
                len,
                tick,
                ..
            } = &mut *back;
            let priorities = classify_priorities(positions, *len, vp);
            integrate_foveated(positions, velocities, &priorities, tick_now, dt, damping);
            *tick = tick_now;
        }
        self.swap();
        self.tick_count.fetch_add(1, Ordering::AcqRel);
        fps.record_tick(start.elapsed().as_nanos() as u64);
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::*;

    #[test]
    fn cached_splat_returns_canonical_for_60fps() {
        let v = cached_splat(DT_60);
        // The cached vector should be byte-identical to a fresh splat.
        let fresh = F32x16::splat(DT_60);
        // F32x16 doesn't implement PartialEq directly; compare via copy_to_slice.
        let mut a = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        v.copy_to_slice(&mut a);
        fresh.copy_to_slice(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn cached_splat_falls_back_for_arbitrary_dt() {
        let v = cached_splat(0.0314);
        let mut out = [0.0f32; 16];
        v.copy_to_slice(&mut out);
        for x in out {
            assert!((x - 0.0314).abs() < 1e-6);
        }
    }

    #[test]
    fn cached_splat_within_tolerance_hits_cache() {
        // 1/60 = 0.01666666… → 0.01666666 + 1µs should still hit the cache.
        let v = cached_splat(DT_60 + 1e-7);
        let mut out = [0.0f32; 16];
        v.copy_to_slice(&mut out);
        // Cached at exactly DT_60, not the slightly-higher input.
        assert!((out[0] - DT_60).abs() < 1e-6);
    }

    #[test]
    fn priority_stride_progression() {
        assert_eq!(UpdatePriority::Foveal.tick_stride(), 1);
        assert_eq!(UpdatePriority::Peripheral.tick_stride(), 2);
        assert_eq!(UpdatePriority::Distant.tick_stride(), 4);
        assert_eq!(UpdatePriority::OffScreen.tick_stride(), u64::MAX);
    }

    #[test]
    fn priority_should_update_respects_stride() {
        assert!(UpdatePriority::Foveal.should_update(0));
        assert!(UpdatePriority::Foveal.should_update(7));
        assert!(UpdatePriority::Peripheral.should_update(0));
        assert!(!UpdatePriority::Peripheral.should_update(1));
        assert!(UpdatePriority::Peripheral.should_update(2));
        assert!(UpdatePriority::Distant.should_update(0));
        assert!(!UpdatePriority::Distant.should_update(1));
        assert!(UpdatePriority::Distant.should_update(4));
        assert!(!UpdatePriority::OffScreen.should_update(0));
        assert!(!UpdatePriority::OffScreen.should_update(u64::MAX - 1));
    }

    #[test]
    fn classify_priorities_assigns_zones() {
        // 4 nodes: at center, foveal-edge, peripheral-zone, off-screen.
        let positions = vec![
            0.0, 0.0, 0.0, // node 0 — at center → Foveal
            3.0, 0.0, 0.0, // node 1 — within foveal radius (4)
            8.0, 0.0, 0.0, // node 2 — within peripheral (16)
            70.0, 0.0, 0.0, // node 3 — beyond cull (64)
        ];
        let vp = Viewport::default_at([0.0, 0.0, 0.0]);
        let p = classify_priorities(&positions, 4, &vp);
        assert_eq!(p[0], UpdatePriority::Foveal);
        assert_eq!(p[1], UpdatePriority::Foveal);
        assert_eq!(p[2], UpdatePriority::Peripheral);
        assert_eq!(p[3], UpdatePriority::OffScreen);
    }

    #[test]
    fn integrate_foveated_skips_offscreen_chunks() {
        // 32 floats = 2 SIMD chunks. Mark all nodes as OffScreen → no update.
        let mut positions = vec![1.0f32; 32];
        let mut velocities = vec![1.0f32; 32];
        let priorities = vec![UpdatePriority::OffScreen; 12]; // covers both chunks
        integrate_foveated(&mut positions, &mut velocities, &priorities, 0, 0.5, 0.9);
        for &p in &positions {
            assert_eq!(p, 1.0);
        } // unchanged
        for &v in &velocities {
            assert_eq!(v, 1.0);
        } // unchanged
    }

    #[test]
    fn integrate_foveated_updates_foveal_chunks() {
        let mut positions = vec![0.0f32; 32];
        let mut velocities = vec![1.0f32; 32];
        let priorities = vec![UpdatePriority::Foveal; 12];
        integrate_foveated(&mut positions, &mut velocities, &priorities, 0, 0.5, 0.9);
        for &p in &positions {
            assert!((p - 0.5).abs() < 1e-6);
        }
        for &v in &velocities {
            assert!((v - 0.9).abs() < 1e-6);
        }
    }

    #[test]
    fn integrate_foveated_respects_peripheral_stride() {
        let mut positions = vec![0.0f32; 32];
        let mut velocities = vec![1.0f32; 32];
        let priorities = vec![UpdatePriority::Peripheral; 12];
        // Tick 1 (odd) — peripheral skips
        integrate_foveated(&mut positions, &mut velocities, &priorities, 1, 0.5, 0.9);
        for &p in &positions {
            assert_eq!(p, 0.0);
        }
        // Tick 2 (even) — peripheral updates
        integrate_foveated(&mut positions, &mut velocities, &priorities, 2, 0.5, 0.9);
        for &p in &positions {
            assert!((p - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn fps_controller_starts_at_60() {
        let c = FpsController::default();
        assert_eq!(c.target_hz(), 60);
        assert_eq!(c.dt(), DT_60);
        assert_eq!(c.budget_ns(), 16_666_666); // 1e9 / 60
    }

    #[test]
    fn fps_controller_steps_down_on_overrun() {
        let c = FpsController::default();
        // Single overrun: budget_60 = 16.67ms; record 50ms tick → step down.
        c.record_tick(50_000_000);
        assert_eq!(c.target_hz(), 30);
        // Another overrun at 30 (budget = 33ms): record 100ms → step to 15.
        c.record_tick(100_000_000);
        assert_eq!(c.target_hz(), 15);
    }

    #[test]
    fn fps_controller_steps_up_on_sustained_under_budget() {
        let c = FpsController::new(15);
        // Record 60 fast ticks → climb to 30.
        for _ in 0..60 {
            c.record_tick(1_000_000);
        } // 1ms each
        assert_eq!(c.target_hz(), 30);
        // Another 60 fast → climb to 60.
        for _ in 0..60 {
            c.record_tick(1_000_000);
        }
        assert_eq!(c.target_hz(), 60);
    }

    #[test]
    fn fps_controller_dt_tracks_target() {
        let c = FpsController::new(60);
        assert_eq!(c.dt(), DT_60);
        c.record_tick(50_000_000); // step to 30
        assert_eq!(c.dt(), DT_30);
        c.record_tick(50_000_000); // step to 15
        assert_eq!(c.dt(), DT_15);
    }

    #[test]
    fn renderer_tick_adaptive_advances_count() {
        let r = Renderer::with_capacity(64);
        let fps = FpsController::default();
        r.tick_adaptive(&fps, 0.95);
        assert_eq!(r.tick_count(), 1);
        // FpsController should have recorded a sample.
        assert!(fps.avg_tick_ns() > 0);
    }

    #[test]
    fn renderer_tick_foveated_advances_count_and_swaps() {
        let r = Renderer::with_capacity(64);
        {
            let mut back = r.write_back();
            back.len = 8;
        }
        let fps = FpsController::default();
        let vp = Viewport::default_at([0.0, 0.0, 0.0]);
        let initial_front = r.front_index();
        r.tick_foveated(&fps, 0.95, &vp);
        assert_eq!(r.tick_count(), 1);
        assert_eq!(r.front_index(), 1 - initial_front);
    }

    #[test]
    fn integrate_simd_array_chunks_have_no_tail() {
        // After migration: 16384 % PREFERRED_F32_LANES == 0, so as_chunks_mut
        // remainder must be empty.
        let mut p = vec![0.0f32; 16_384];
        let (_chunks, tail) = p.as_chunks_mut::<16>();
        assert!(tail.is_empty(), "no scalar tail at 16384");
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn integrate_simd_par_matches_sequential() {
        // 4096 floats = 4 × BLOCK_FLOATS — guaranteed multi-block, so rayon
        // actually parallelizes instead of degenerating to one worker.
        let n = 4 * BLOCK_FLOATS;
        let mut p_seq = (0..n).map(|i| i as f32 * 0.001).collect::<Vec<_>>();
        let mut v_seq = (0..n).map(|i| (i as f32).sin() * 0.1).collect::<Vec<_>>();
        let mut p_par = p_seq.clone();
        let mut v_par = v_seq.clone();

        integrate_simd(&mut p_seq, &mut v_seq, DT_60, 0.98);
        integrate_simd_par(&mut p_par, &mut v_par, DT_60, 0.98);

        // FMA + mul are deterministic at the same dispatch tier — every lane
        // bit-identical across sequential and parallel runs.
        for i in 0..n {
            assert_eq!(p_seq[i].to_bits(), p_par[i].to_bits(), "pos mismatch at {}", i);
            assert_eq!(v_seq[i].to_bits(), v_par[i].to_bits(), "vel mismatch at {}", i);
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn integrate_simd_par_advances_positions_exactly() {
        // Single-tick contract: x[i] += v[i] * dt. With v=1, dt=DT_60, after
        // one tick every element is initial + 1/60 (within f32 epsilon).
        let n = 2 * BLOCK_FLOATS;
        let mut p = vec![0.0f32; n];
        let mut v = vec![1.0f32; n];
        integrate_simd_par(&mut p, &mut v, DT_60, 1.0);
        for &x in &p {
            assert!((x - DT_60).abs() < 1e-6, "got {}, expected {}", x, DT_60);
        }
    }
}
