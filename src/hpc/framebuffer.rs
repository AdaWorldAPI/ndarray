//! Palette-indexed framebuffer — ndarray IS the graphics card.
//!
//! Composes a screen as a `[u8; W*H]` palette-indexed bitmap. Wire format
//! is palette_codec-compressed (4-bit nibble at 16 colors → 8× smaller
//! than RGB888). q2 receives a ready-made bitmap and blits with
//! `canvas.putImageData(...)`.
//!
//! # Tier-adaptive palette
//!
//! The detected SIMD tier determines the palette depth AND foveal detail
//! budget. Lower-capability hardware gets fewer colors and simpler sprites
//! that compress better and process faster:
//!
//! | Tier        | Palette | Bits/px | Sprite | Wire KB (1024²) |
//! |-------------|---------|---------|--------|-----------------|
//! | AVX-512/AMX | 16      | 4       | 8×8    | 512             |
//! | AVX2        | 8       | 3       | 6×6    | 384             |
//! | NEON/scalar | 4       | 2       | 4×4    | 256             |
//!
//! # Views
//!
//! - **MRI view** — full-screen density heatmap, all nodes visible,
//!   palette maps to intensity (white=hot, black=cold). Overview radar.
//! - **Neo4j view** — nodes as dot sprites, edges as Bresenham lines,
//!   labels as glyph sprites. Interactive-style graph display.
//! - **Cloud view** — distant/peripheral nodes as a nibble-packed
//!   density field at mipmap L1/L2. Foveal region sharp, periphery fog.

use crate::hpc::palette_codec::{bits_for_palette_size, pack_indices};
use crate::simd::PREFERRED_F32_LANES;

// ─────────────────────────────────────────────────────────────────────
// Tier-adaptive palette selection
// ─────────────────────────────────────────────────────────────────────

/// Palette depth based on detected SIMD tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteTier {
    /// AVX-512 / AMX: 16 colors, 4 bits/pixel, 8×8 sprites.
    Full16,
    /// AVX2: 8 colors, 3 bits/pixel, 6×6 sprites.
    Mid8,
    /// NEON / scalar: 4 colors, 2 bits/pixel, 4×4 sprites.
    Low4,
}

impl PaletteTier {
    /// Auto-detect from the active SIMD lane width.
    pub fn detect() -> Self {
        match PREFERRED_F32_LANES {
            16 => Self::Full16,   // AVX-512 / AMX
            8  => Self::Mid8,     // AVX2
            _  => Self::Low4,     // NEON (4), scalar (≤4)
        }
    }

    /// Number of palette entries for this tier.
    #[inline]
    pub fn palette_size(self) -> usize {
        match self {
            Self::Full16 => 16,
            Self::Mid8 => 8,
            Self::Low4 => 4,
        }
    }

    /// Bits per pixel for this tier.
    #[inline]
    pub fn bits_per_pixel(self) -> usize {
        bits_for_palette_size(self.palette_size())
    }

    /// Sprite edge length (square) for node dots.
    #[inline]
    pub fn sprite_size(self) -> usize {
        match self {
            Self::Full16 => 8,
            Self::Mid8 => 6,
            Self::Low4 => 4,
        }
    }

    /// Wire size in bytes for a `width × height` framebuffer at this tier.
    #[inline]
    pub fn wire_bytes(self, width: usize, height: usize) -> usize {
        let total_px = width * height;
        let bpp = self.bits_per_pixel();
        (total_px * bpp + 7) / 8
    }
}

// ─────────────────────────────────────────────────────────────────────
// Framebuffer + SpriteAtlas
// ─────────────────────────────────────────────────────────────────────

/// View mode — determines how the framebuffer is composed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// Density heatmap — every node plots at its position, intensity =
    /// confidence. Palette maps linearly: 0 = background, max = hottest.
    Mri,
    /// Nodes as dot sprites, edges as Bresenham lines. Neo4j-style.
    Neo4j,
    /// Foveal sharp, peripheral density fog. Hybrid.
    Cloud,
}

/// Palette-indexed framebuffer. Each pixel is a u8 index into a palette
/// whose size is determined by the SIMD tier.
pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub tier: PaletteTier,
    /// Row-major palette indices, length = width × height.
    pub pixels: Vec<u8>,
    /// Dirty rectangle: (x0, y0, x1, y1). Only the region inside needs
    /// re-encoding on the wire. Reset to (0,0,0,0) after each `pack()`.
    pub dirty: (usize, usize, usize, usize),
}

impl Framebuffer {
    /// Allocate a cleared framebuffer at the given resolution and auto-detected tier.
    pub fn new(width: usize, height: usize) -> Self {
        let tier = PaletteTier::detect();
        Self {
            width,
            height,
            tier,
            pixels: vec![0u8; width * height],
            dirty: (0, 0, width, height),
        }
    }

    /// Allocate with an explicit tier (for testing or override).
    pub fn with_tier(width: usize, height: usize, tier: PaletteTier) -> Self {
        Self {
            width,
            height,
            tier,
            pixels: vec![0u8; width * height],
            dirty: (0, 0, width, height),
        }
    }

    /// Clear the entire framebuffer to palette index 0 (background).
    #[inline]
    pub fn clear(&mut self) {
        self.pixels.fill(0);
        self.dirty = (0, 0, self.width, self.height);
    }

    /// Set a single pixel (with bounds check). Expands dirty rect.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: u8) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
            self.expand_dirty(x, y, x + 1, y + 1);
        }
    }

    /// Plot a filled dot (square sprite) centered at (cx, cy).
    pub fn plot_dot(&mut self, cx: usize, cy: usize, color: u8) {
        let r = self.tier.sprite_size() / 2;
        let x0 = cx.saturating_sub(r);
        let y0 = cy.saturating_sub(r);
        let x1 = (cx + r).min(self.width);
        let y1 = (cy + r).min(self.height);
        for y in y0..y1 {
            let row = y * self.width;
            for x in x0..x1 {
                self.pixels[row + x] = color;
            }
        }
        self.expand_dirty(x0, y0, x1, y1);
    }

    /// Draw a Bresenham line from (x0,y0) to (x1,y1) with palette index.
    pub fn draw_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: u8) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i32 = if x0 < x1 { 1 } else { -1 };
        let sy: i32 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x0 >= 0 && y0 >= 0 && (x0 as usize) < self.width && (y0 as usize) < self.height {
                self.pixels[y0 as usize * self.width + x0 as usize] = color;
            }
            if x0 == x1 && y0 == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
        let (lx, rx) = (x0.min(x1).max(0) as usize, (x0.max(x1) as usize + 1).min(self.width));
        let (ly, ry) = (y0.min(y1).max(0) as usize, (y0.max(y1) as usize + 1).min(self.height));
        self.expand_dirty(lx, ly, rx, ry);
    }

    /// MRI density blit — for each node, increment the pixel at its projected
    /// position. Clamped to palette max so saturated regions show as hottest.
    pub fn blit_mri_density(&mut self, screen_xs: &[usize], screen_ys: &[usize]) {
        let max_idx = (self.tier.palette_size() - 1) as u8;
        for (&sx, &sy) in screen_xs.iter().zip(screen_ys.iter()) {
            if sx < self.width && sy < self.height {
                let idx = sy * self.width + sx;
                self.pixels[idx] = self.pixels[idx].saturating_add(1).min(max_idx);
            }
        }
        self.dirty = (0, 0, self.width, self.height);
    }

    /// Pack the framebuffer into palette_codec wire format.
    ///
    /// Returns `(packed_u64s, bits_per_pixel)`. The consumer unpacks with
    /// `palette_codec::unpack_indices(&packed, bpp, w*h)`.
    pub fn pack(&mut self) -> (Vec<u64>, usize) {
        let bpp = self.tier.bits_per_pixel();
        let packed = pack_indices(&self.pixels, bpp);
        self.dirty = (0, 0, 0, 0);
        (packed, bpp)
    }

    /// Byte count of the last `pack()` output (for bandwidth estimation).
    pub fn packed_byte_estimate(&self) -> usize {
        self.tier.wire_bytes(self.width, self.height)
    }

    fn expand_dirty(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        self.dirty.0 = self.dirty.0.min(x0);
        self.dirty.1 = self.dirty.1.min(y0);
        self.dirty.2 = self.dirty.2.max(x1);
        self.dirty.3 = self.dirty.3.max(y1);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Mipmap — bitwise 4× downsampling for LOD pyramid.
// ─────────────────────────────────────────────────────────────────────

/// Downsample a framebuffer 2× in each axis (4× total pixels).
///
/// Each 2×2 block maps to one pixel. Strategy: max (brightest wins),
/// matching the MRI heatmap "any signal in this region" semantic.
pub fn downsample_2x(src: &[u8], src_w: usize, src_h: usize) -> (Vec<u8>, usize, usize) {
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;
    let mut dst = vec![0u8; dst_w * dst_h];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx * 2;
            let sy = dy * 2;
            let a = src[sy * src_w + sx];
            let b = src[sy * src_w + sx + 1];
            let c = src[(sy + 1) * src_w + sx];
            let d = src[(sy + 1) * src_w + sx + 1];
            dst[dy * dst_w + dx] = a.max(b).max(c).max(d);
        }
    }
    (dst, dst_w, dst_h)
}

/// Full mipmap pyramid from L0 (original) down to the level where
/// both dimensions are < `min_dim`.
pub fn build_mipmap_pyramid(fb: &Framebuffer, min_dim: usize) -> Vec<(Vec<u8>, usize, usize)> {
    let mut levels = Vec::new();
    let mut cur = fb.pixels.clone();
    let mut w = fb.width;
    let mut h = fb.height;
    levels.push((cur.clone(), w, h));
    while w > min_dim && h > min_dim {
        let (down, dw, dh) = downsample_2x(&cur, w, h);
        levels.push((down.clone(), dw, dh));
        cur = down;
        w = dw;
        h = dh;
    }
    levels
}

// ─────────────────────────────────────────────────────────────────────
// Compose: RenderFrame → Framebuffer (the "graphics card" pipeline).
// ─────────────────────────────────────────────────────────────────────

/// Project a 3D position to 2D screen coordinates (orthographic).
///
/// Simple orthographic: x → screen_x, y → screen_y (z ignored).
/// Scale and offset are applied. This is the dumbest projection that
/// works; replace with perspective when q2 has a camera matrix.
#[inline]
pub fn project_ortho(
    pos_x: f32, pos_y: f32,
    scale: f32, offset_x: f32, offset_y: f32,
    screen_w: usize, screen_h: usize,
) -> (usize, usize) {
    let sx = ((pos_x * scale + offset_x) as usize).min(screen_w.saturating_sub(1));
    let sy = ((pos_y * scale + offset_y) as usize).min(screen_h.saturating_sub(1));
    (sx, sy)
}

use crate::hpc::renderer::RenderFrame;

/// Compose a Neo4j-style view: dots at nodes, lines for edges.
///
/// `edges` is a list of (source_idx, target_idx) pairs into the frame's
/// node arrays. `color_fn` maps node index → palette color.
pub fn compose_neo4j(
    fb: &mut Framebuffer,
    frame: &RenderFrame,
    edges: &[(usize, usize)],
    scale: f32,
    offset: (f32, f32),
    node_color: u8,
    edge_color: u8,
) {
    fb.clear();
    let w = fb.width;
    let h = fb.height;

    // Edges first (so nodes overdraw on top).
    for &(src, tgt) in edges {
        if src >= frame.len || tgt >= frame.len { continue; }
        let (sx0, sy0) = project_ortho(
            frame.positions[src * 3], frame.positions[src * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        let (sx1, sy1) = project_ortho(
            frame.positions[tgt * 3], frame.positions[tgt * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        fb.draw_line(sx0 as i32, sy0 as i32, sx1 as i32, sy1 as i32, edge_color);
    }

    // Nodes as dot sprites.
    for i in 0..frame.len {
        let (sx, sy) = project_ortho(
            frame.positions[i * 3], frame.positions[i * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        fb.plot_dot(sx, sy, node_color);
    }
}

/// Compose an MRI density heatmap view.
pub fn compose_mri(
    fb: &mut Framebuffer,
    frame: &RenderFrame,
    scale: f32,
    offset: (f32, f32),
) {
    fb.clear();
    let w = fb.width;
    let h = fb.height;

    let mut xs = Vec::with_capacity(frame.len);
    let mut ys = Vec::with_capacity(frame.len);
    for i in 0..frame.len {
        let (sx, sy) = project_ortho(
            frame.positions[i * 3], frame.positions[i * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        xs.push(sx);
        ys.push(sy);
    }
    fb.blit_mri_density(&xs, &ys);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::palette_codec::unpack_indices;

    #[test]
    fn tier_detect_matches_lane_width() {
        let tier = PaletteTier::detect();
        match PREFERRED_F32_LANES {
            16 => assert_eq!(tier, PaletteTier::Full16),
            8  => assert_eq!(tier, PaletteTier::Mid8),
            _  => assert_eq!(tier, PaletteTier::Low4),
        }
    }

    #[test]
    fn tier_palette_sizes() {
        assert_eq!(PaletteTier::Full16.palette_size(), 16);
        assert_eq!(PaletteTier::Mid8.palette_size(), 8);
        assert_eq!(PaletteTier::Low4.palette_size(), 4);
    }

    #[test]
    fn tier_bits_per_pixel() {
        assert_eq!(PaletteTier::Full16.bits_per_pixel(), 4);
        assert_eq!(PaletteTier::Mid8.bits_per_pixel(), 3);
        assert_eq!(PaletteTier::Low4.bits_per_pixel(), 2);
    }

    #[test]
    fn tier_sprite_sizes() {
        assert_eq!(PaletteTier::Full16.sprite_size(), 8);
        assert_eq!(PaletteTier::Mid8.sprite_size(), 6);
        assert_eq!(PaletteTier::Low4.sprite_size(), 4);
    }

    #[test]
    fn framebuffer_clear_sets_all_zero() {
        let mut fb = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        fb.pixels[100] = 5;
        fb.clear();
        assert!(fb.pixels.iter().all(|&p| p == 0));
    }

    #[test]
    fn plot_dot_size_matches_tier() {
        for tier in [PaletteTier::Full16, PaletteTier::Mid8, PaletteTier::Low4] {
            let mut fb = Framebuffer::with_tier(64, 64, tier);
            fb.plot_dot(32, 32, 1);
            let lit: usize = fb.pixels.iter().filter(|&&p| p > 0).count();
            let expected = tier.sprite_size() * tier.sprite_size();
            assert_eq!(lit, expected, "tier {:?}", tier);
        }
    }

    #[test]
    fn bresenham_horizontal_line() {
        let mut fb = Framebuffer::with_tier(32, 32, PaletteTier::Full16);
        fb.draw_line(2, 5, 10, 5, 3);
        for x in 2..=10 {
            assert_eq!(fb.pixels[5 * 32 + x], 3);
        }
    }

    #[test]
    fn bresenham_diagonal_line() {
        let mut fb = Framebuffer::with_tier(32, 32, PaletteTier::Full16);
        fb.draw_line(0, 0, 7, 7, 2);
        for i in 0..=7 {
            assert_eq!(fb.pixels[i * 32 + i], 2);
        }
    }

    #[test]
    fn mri_density_accumulates() {
        let mut fb = Framebuffer::with_tier(16, 16, PaletteTier::Full16);
        let xs = vec![5, 5, 5]; // same pixel hit 3 times
        let ys = vec![5, 5, 5];
        fb.blit_mri_density(&xs, &ys);
        assert_eq!(fb.pixels[5 * 16 + 5], 3);
    }

    #[test]
    fn mri_density_clamps_to_palette_max() {
        let mut fb = Framebuffer::with_tier(16, 16, PaletteTier::Low4);
        // Low4 = palette_size 4, max index = 3.
        let xs = vec![2; 10];
        let ys = vec![2; 10];
        fb.blit_mri_density(&xs, &ys);
        assert_eq!(fb.pixels[2 * 16 + 2], 3); // clamped
    }

    #[test]
    fn pack_roundtrips_through_palette_codec() {
        let mut fb = Framebuffer::with_tier(16, 16, PaletteTier::Full16);
        fb.plot_dot(8, 8, 7);
        let original = fb.pixels.clone();
        let (packed, bpp) = fb.pack();
        let recovered = unpack_indices(&packed, bpp, 16 * 16);
        assert_eq!(original, recovered);
    }

    #[test]
    fn downsample_2x_shrinks_dimensions() {
        let src = vec![1u8; 64 * 64];
        let (dst, w, h) = downsample_2x(&src, 64, 64);
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(dst.len(), 32 * 32);
        assert!(dst.iter().all(|&p| p == 1));
    }

    #[test]
    fn mipmap_pyramid_has_correct_levels() {
        let fb = Framebuffer::with_tier(256, 256, PaletteTier::Full16);
        let pyramid = build_mipmap_pyramid(&fb, 8);
        // 256 → 128 → 64 → 32 → 16 → 8 = 6 levels (including L0).
        assert!(pyramid.len() >= 5);
        assert_eq!(pyramid[0].1, 256);
        assert_eq!(pyramid[1].1, 128);
    }

    #[test]
    fn compose_neo4j_plots_nodes_and_edges() {
        let mut fb = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        let mut frame = RenderFrame::with_capacity(16);
        // Two nodes
        frame.len = 2;
        frame.positions[0] = 10.0; frame.positions[1] = 10.0; frame.positions[2] = 0.0;
        frame.positions[3] = 50.0; frame.positions[4] = 50.0; frame.positions[5] = 0.0;
        let edges = vec![(0, 1)];
        compose_neo4j(&mut fb, &frame, &edges, 1.0, (0.0, 0.0), 5, 2);
        // Node 0 should have a dot around (10, 10).
        assert_eq!(fb.pixels[10 * 64 + 10], 5);
        // Edge should have at least one pixel of color 2 on the diagonal.
        let edge_count = fb.pixels.iter().filter(|&&p| p == 2).count();
        assert!(edge_count > 0, "edge should have drawn pixels");
    }

    #[test]
    fn compose_mri_plots_density() {
        let mut fb = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        let mut frame = RenderFrame::with_capacity(16);
        frame.len = 3;
        // Three nodes at same spot → density = 3.
        for i in 0..3 {
            frame.positions[i * 3] = 20.0;
            frame.positions[i * 3 + 1] = 20.0;
        }
        compose_mri(&mut fb, &frame, 1.0, (0.0, 0.0));
        assert_eq!(fb.pixels[20 * 64 + 20], 3);
    }

    #[test]
    fn wire_bytes_decrease_with_lower_tier() {
        let full = PaletteTier::Full16.wire_bytes(1024, 768);
        let mid = PaletteTier::Mid8.wire_bytes(1024, 768);
        let low = PaletteTier::Low4.wire_bytes(1024, 768);
        assert!(full > mid, "16-color > 8-color wire");
        assert!(mid > low, "8-color > 4-color wire");
    }
}

// ─────────────────────────────────────────────────────────────────────
// Wobble spring — organic node displacement that masks layout jitter.
//
// When a node moves (velocity exceeds threshold), wobble energy is
// injected. It decays exponentially each tick. At render time, wobble
// is added to the projected position — the node oscillates around its
// physics-true location. The effect: the graph feels alive, and small
// layout inaccuracies are hidden behind spring motion.
// ─────────────────────────────────────────────────────────────────────

/// Per-node wobble state: displacement + decay.
#[derive(Debug, Clone)]
pub struct WobbleState {
    /// Per-node wobble displacement (x, y interleaved; length = 2·capacity).
    pub displace: Vec<f32>,
    /// Decay factor per tick [0, 1). 0.92 = ~12 frames to half-life.
    pub decay: f32,
    /// Velocity threshold: inject wobble when speed exceeds this.
    pub inject_threshold: f32,
    /// Injection amplitude: max wobble pixels on injection.
    pub amplitude: f32,
}

impl WobbleState {
    pub fn new(capacity: usize) -> Self {
        Self {
            displace: vec![0.0; capacity * 2],
            decay: 0.92,
            inject_threshold: 0.5,
            amplitude: 3.0,
        }
    }

    /// Inject wobble for nodes whose velocity exceeds the threshold,
    /// then decay all displacements. Call once per tick.
    pub fn tick(&mut self, velocities: &[f32], len: usize) {
        // Inject: if |v| > threshold, add random-ish displacement
        // (use velocity direction × amplitude for deterministic wobble).
        for i in 0..len {
            let vx = velocities[i * 3];
            let vy = velocities[i * 3 + 1];
            let speed = (vx * vx + vy * vy).sqrt();
            if speed > self.inject_threshold {
                // Perpendicular to velocity direction → organic wobble
                let norm = speed.recip();
                self.displace[i * 2]     += -vy * norm * self.amplitude;
                self.displace[i * 2 + 1] +=  vx * norm * self.amplitude;
            }
        }
        // Decay all
        for d in self.displace.iter_mut() {
            *d *= self.decay;
        }
    }

    /// Get wobble-adjusted screen position for node `i`.
    #[inline]
    pub fn adjust(&self, sx: usize, sy: usize, node_idx: usize) -> (usize, usize) {
        let dx = self.displace.get(node_idx * 2).copied().unwrap_or(0.0);
        let dy = self.displace.get(node_idx * 2 + 1).copied().unwrap_or(0.0);
        (
            (sx as f32 + dx).max(0.0) as usize,
            (sy as f32 + dy).max(0.0) as usize,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// Neuron firing — nodes pulse when the cognitive shader resolves them.
//
// fire_intensity[i] ∈ [0, 255]. The shader sets it to 255 on Commit,
// 200 on Epiphany, 128 on FailureTicket. Each tick it decays by
// `decay_rate`. The framebuffer maps fire_intensity to a brighter
// palette index (additive blend).
// ─────────────────────────────────────────────────────────────────────

/// Per-node fire intensity for visual neuron-pulse feedback.
#[derive(Debug, Clone)]
pub struct FireState {
    /// Intensity per node [0, 255]. 0 = dark, 255 = just fired.
    pub intensity: Vec<u8>,
    /// Subtracted per tick. Higher = faster fade.
    pub decay_rate: u8,
}

impl FireState {
    pub fn new(capacity: usize) -> Self {
        Self {
            intensity: vec![0u8; capacity],
            decay_rate: 16,
        }
    }

    /// Fire a node at the given intensity (255 = max).
    #[inline]
    pub fn fire(&mut self, node_idx: usize, intensity: u8) {
        if let Some(v) = self.intensity.get_mut(node_idx) {
            *v = (*v).max(intensity);
        }
    }

    /// Decay all intensities by `decay_rate`. Call once per tick.
    pub fn tick(&mut self) {
        for v in self.intensity.iter_mut() {
            *v = v.saturating_sub(self.decay_rate);
        }
    }

    /// Map fire intensity to a palette color boost.
    /// Returns 0 (no boost) or 1–3 extra palette indices to add.
    #[inline]
    pub fn color_boost(&self, node_idx: usize, palette_max: u8) -> u8 {
        let raw = self.intensity.get(node_idx).copied().unwrap_or(0);
        // Scale [0,255] → [0, palette_max/2] extra indices
        let boost = (raw as u16 * (palette_max as u16 / 2)) / 255;
        boost as u8
    }
}

// ─────────────────────────────────────────────────────────────────────
// Glyph atlas — 5×7 bitmap font for node labels.
//
// 95 printable ASCII characters (0x20–0x7E) stored as 5-byte columns
// (7 rows each). Total atlas: 95 × 5 = 475 bytes — fits in L1.
// ─────────────────────────────────────────────────────────────────────

/// 5×7 bitmap glyph for one character. Column-major: glyph[col] has 7 bits (rows).
pub type Glyph = [u8; 5];

/// Minimal 5×7 ASCII glyph set. Covers A-Z, 0-9, space, common punctuation.
/// Missing chars render as a filled block.
pub static GLYPH_ATLAS: [Glyph; 128] = {
    let mut atlas = [[0x7Fu8; 5]; 128]; // default = filled block
    // Space
    atlas[b' ' as usize] = [0, 0, 0, 0, 0];
    // Digits 0-9
    atlas[b'0' as usize] = [0x3E, 0x51, 0x49, 0x45, 0x3E];
    atlas[b'1' as usize] = [0x00, 0x42, 0x7F, 0x40, 0x00];
    atlas[b'2' as usize] = [0x62, 0x51, 0x49, 0x49, 0x46];
    atlas[b'3' as usize] = [0x22, 0x41, 0x49, 0x49, 0x36];
    atlas[b'4' as usize] = [0x18, 0x14, 0x12, 0x7F, 0x10];
    atlas[b'5' as usize] = [0x27, 0x45, 0x45, 0x45, 0x39];
    atlas[b'6' as usize] = [0x3C, 0x4A, 0x49, 0x49, 0x30];
    atlas[b'7' as usize] = [0x01, 0x71, 0x09, 0x05, 0x03];
    atlas[b'8' as usize] = [0x36, 0x49, 0x49, 0x49, 0x36];
    atlas[b'9' as usize] = [0x06, 0x49, 0x49, 0x29, 0x1E];
    // Letters A-Z
    atlas[b'A' as usize] = [0x7E, 0x09, 0x09, 0x09, 0x7E];
    atlas[b'B' as usize] = [0x7F, 0x49, 0x49, 0x49, 0x36];
    atlas[b'C' as usize] = [0x3E, 0x41, 0x41, 0x41, 0x22];
    atlas[b'D' as usize] = [0x7F, 0x41, 0x41, 0x41, 0x3E];
    atlas[b'E' as usize] = [0x7F, 0x49, 0x49, 0x49, 0x41];
    atlas[b'F' as usize] = [0x7F, 0x09, 0x09, 0x09, 0x01];
    atlas[b'G' as usize] = [0x3E, 0x41, 0x49, 0x49, 0x7A];
    atlas[b'H' as usize] = [0x7F, 0x08, 0x08, 0x08, 0x7F];
    atlas[b'I' as usize] = [0x00, 0x41, 0x7F, 0x41, 0x00];
    atlas[b'J' as usize] = [0x20, 0x40, 0x41, 0x3F, 0x01];
    atlas[b'K' as usize] = [0x7F, 0x08, 0x14, 0x22, 0x41];
    atlas[b'L' as usize] = [0x7F, 0x40, 0x40, 0x40, 0x40];
    atlas[b'M' as usize] = [0x7F, 0x02, 0x0C, 0x02, 0x7F];
    atlas[b'N' as usize] = [0x7F, 0x04, 0x08, 0x10, 0x7F];
    atlas[b'O' as usize] = [0x3E, 0x41, 0x41, 0x41, 0x3E];
    atlas[b'P' as usize] = [0x7F, 0x09, 0x09, 0x09, 0x06];
    atlas[b'Q' as usize] = [0x3E, 0x41, 0x51, 0x21, 0x5E];
    atlas[b'R' as usize] = [0x7F, 0x09, 0x19, 0x29, 0x46];
    atlas[b'S' as usize] = [0x26, 0x49, 0x49, 0x49, 0x32];
    atlas[b'T' as usize] = [0x01, 0x01, 0x7F, 0x01, 0x01];
    atlas[b'U' as usize] = [0x3F, 0x40, 0x40, 0x40, 0x3F];
    atlas[b'V' as usize] = [0x1F, 0x20, 0x40, 0x20, 0x1F];
    atlas[b'W' as usize] = [0x3F, 0x40, 0x38, 0x40, 0x3F];
    atlas[b'X' as usize] = [0x63, 0x14, 0x08, 0x14, 0x63];
    atlas[b'Y' as usize] = [0x03, 0x04, 0x78, 0x04, 0x03];
    atlas[b'Z' as usize] = [0x61, 0x51, 0x49, 0x45, 0x43];
    // Punctuation
    atlas[b'.' as usize] = [0x00, 0x60, 0x60, 0x00, 0x00];
    atlas[b'-' as usize] = [0x08, 0x08, 0x08, 0x08, 0x08];
    atlas[b'_' as usize] = [0x40, 0x40, 0x40, 0x40, 0x40];
    atlas[b':' as usize] = [0x00, 0x36, 0x36, 0x00, 0x00];
    atlas
};

impl Framebuffer {
    /// Blit a text label at (x, y) using the 5×7 glyph atlas.
    pub fn draw_label(&mut self, x: usize, y: usize, text: &str, color: u8) {
        let mut cx = x;
        for ch in text.bytes() {
            let idx = (ch as usize).min(127);
            let glyph = &GLYPH_ATLAS[idx];
            for col in 0..5 {
                let bits = glyph[col];
                for row in 0..7 {
                    if bits & (1 << row) != 0 {
                        let px = cx + col;
                        let py = y + row;
                        if px < self.width && py < self.height {
                            self.pixels[py * self.width + px] = color;
                        }
                    }
                }
            }
            cx += 6; // 5 pixels + 1 gap
        }
        let text_w = text.len() * 6;
        self.expand_dirty(x, y, (x + text_w).min(self.width), (y + 7).min(self.height));
    }
}

// ─────────────────────────────────────────────────────────────────────
// Flyby ring buffer — Amiga demo scene trick.
//
// Pre-render N frames of a mathematically computed satellite orbit
// around the graph. Store as a ring of palette_codec-packed framebuffers.
// During zoom/pan transitions or when the compute budget is spent,
// play from the ring. The loop is seamless (Lissajous orbit completes
// one full cycle over N frames). Higher N = smoother apparent frame rate
// at the cost of memory.
//
// 300 frames × 512 KB each (16-color 1024²) = 150 MB.
// 300 frames × 128 KB each (16-color 512²)  =  38 MB — fits L3.
// ─────────────────────────────────────────────────────────────────────

/// Pre-rendered flyby frame (palette_codec-packed + camera state).
#[derive(Clone)]
pub struct FlybyFrame {
    /// Packed pixel indices (via palette_codec).
    pub packed: Vec<u64>,
    /// Bits per pixel used for packing.
    pub bpp: usize,
    /// Camera position at this keyframe.
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_zoom: f32,
}

/// Ring buffer of pre-rendered flyby keyframes.
pub struct FlybyCache {
    pub frames: Vec<FlybyFrame>,
    /// Current playback position in [0, frames.len()).
    pub cursor: usize,
    /// Width/height of pre-rendered frames.
    pub width: usize,
    pub height: usize,
}

impl FlybyCache {
    /// Pre-render `n_frames` of a Lissajous satellite orbit.
    ///
    /// The orbit traces a figure-8 around the graph center, completing
    /// one full loop over `n_frames`. Scale determines the orbital radius
    /// in world units; zoom_range controls the min/max camera zoom.
    pub fn prerender(
        fb_template: &Framebuffer,
        frame: &RenderFrame,
        edges: &[(usize, usize)],
        n_frames: usize,
        orbit_radius: f32,
        zoom_range: (f32, f32),
        node_color: u8,
        edge_color: u8,
    ) -> Self {
        let mut frames = Vec::with_capacity(n_frames);
        let w = fb_template.width;
        let h = fb_template.height;
        let tier = fb_template.tier;

        for i in 0..n_frames {
            let t = (i as f32 / n_frames as f32) * std::f32::consts::TAU;
            // Lissajous: x = A·sin(t), y = A·sin(2t) → figure-8 orbit
            let cam_x = orbit_radius * t.sin() + (w as f32 / 2.0);
            let cam_y = orbit_radius * (2.0 * t).sin() + (h as f32 / 2.0);
            // Zoom oscillates between min and max over the orbit
            let zoom_t = (t.cos() + 1.0) * 0.5; // [0, 1]
            let cam_zoom = zoom_range.0 + (zoom_range.1 - zoom_range.0) * zoom_t;

            let mut fb = Framebuffer::with_tier(w, h, tier);
            compose_neo4j(
                &mut fb, frame, edges,
                cam_zoom, (-cam_x * cam_zoom + w as f32 / 2.0,
                           -cam_y * cam_zoom + h as f32 / 2.0),
                node_color, edge_color,
            );
            let (packed, bpp) = fb.pack();
            frames.push(FlybyFrame { packed, bpp, cam_x, cam_y, cam_zoom });
        }
        Self { frames, cursor: 0, width: w, height: h }
    }

    /// Advance the cursor and return the next keyframe (looping).
    pub fn next_frame(&mut self) -> &FlybyFrame {
        let frame = &self.frames[self.cursor];
        self.cursor = (self.cursor + 1) % self.frames.len();
        frame
    }

    /// Seek to the keyframe closest to the given camera position.
    /// Used when transitioning from interactive mode back to flyby.
    pub fn seek_nearest(&mut self, cam_x: f32, cam_y: f32) {
        let mut best_dist = f32::MAX;
        let mut best_idx = 0;
        for (i, f) in self.frames.iter().enumerate() {
            let dx = f.cam_x - cam_x;
            let dy = f.cam_y - cam_y;
            let d = dx * dx + dy * dy;
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        self.cursor = best_idx;
    }

    /// Total memory used by the cache.
    pub fn memory_bytes(&self) -> usize {
        self.frames.iter().map(|f| f.packed.len() * 8).sum()
    }

    /// Frame count.
    pub fn len(&self) -> usize { self.frames.len() }

    pub fn is_empty(&self) -> bool { self.frames.is_empty() }
}

// ─────────────────────────────────────────────────────────────────────
// Full composition: wobble + fire + labels + Neo4j view
// ─────────────────────────────────────────────────────────────────────

/// Full Neo4j-style compose with wobble, neuron fire, and labels.
pub fn compose_neo4j_full(
    fb: &mut Framebuffer,
    frame: &RenderFrame,
    edges: &[(usize, usize)],
    scale: f32,
    offset: (f32, f32),
    wobble: &WobbleState,
    fire: &FireState,
    labels: &[&str],
    node_base_color: u8,
    edge_color: u8,
    label_color: u8,
) {
    fb.clear();
    let w = fb.width;
    let h = fb.height;
    let pal_max = (fb.tier.palette_size() - 1) as u8;

    // 1. Edges (drawn first so nodes overdraw).
    for &(src, tgt) in edges {
        if src >= frame.len || tgt >= frame.len { continue; }
        let (sx0, sy0) = project_ortho(
            frame.positions[src * 3], frame.positions[src * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        let (sx1, sy1) = project_ortho(
            frame.positions[tgt * 3], frame.positions[tgt * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        let (wx0, wy0) = wobble.adjust(sx0, sy0, src);
        let (wx1, wy1) = wobble.adjust(sx1, sy1, tgt);
        fb.draw_line(wx0 as i32, wy0 as i32, wx1 as i32, wy1 as i32, edge_color);
    }

    // 2. Nodes as dot sprites with fire boost.
    for i in 0..frame.len {
        let (sx, sy) = project_ortho(
            frame.positions[i * 3], frame.positions[i * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        let (wx, wy) = wobble.adjust(sx, sy, i);
        let boost = fire.color_boost(i, pal_max);
        let color = (node_base_color + boost).min(pal_max);
        fb.plot_dot(wx, wy, color);
    }

    // 3. Labels (drawn last so text is on top).
    for (i, &label) in labels.iter().enumerate().take(frame.len) {
        if label.is_empty() { continue; }
        let (sx, sy) = project_ortho(
            frame.positions[i * 3], frame.positions[i * 3 + 1],
            scale, offset.0, offset.1, w, h,
        );
        let (wx, wy) = wobble.adjust(sx, sy, i);
        let label_y = wy + fb.tier.sprite_size() / 2 + 1;
        fb.draw_label(wx.saturating_sub(label.len() * 3), label_y, label, label_color);
    }
}

#[cfg(test)]
mod visual_tests {
    use super::*;
    use crate::hpc::renderer::RenderFrame;

    #[test]
    fn wobble_decays_toward_zero() {
        let mut w = WobbleState::new(4);
        w.displace[0] = 10.0;
        w.displace[1] = -5.0;
        let vels = vec![0.0f32; 12]; // no new injection
        for _ in 0..200 {
            w.tick(&vels, 4);
        }
        assert!(w.displace[0].abs() < 0.01, "got {}", w.displace[0]);
        assert!(w.displace[1].abs() < 0.01, "got {}", w.displace[1]);
    }

    #[test]
    fn wobble_injects_on_high_velocity() {
        let mut w = WobbleState::new(2);
        let vels = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        w.tick(&vels, 1);
        // Perpendicular to (10, 0) → displacement in Y
        assert!(w.displace[1].abs() > 0.1);
    }

    #[test]
    fn fire_decays_to_zero() {
        let mut f = FireState::new(4);
        f.fire(0, 255);
        assert_eq!(f.intensity[0], 255);
        for _ in 0..20 {
            f.tick();
        }
        assert_eq!(f.intensity[0], 0);
    }

    #[test]
    fn fire_color_boost_scales_with_intensity() {
        let mut f = FireState::new(4);
        f.fire(0, 255);
        let boost_full = f.color_boost(0, 15);
        assert!(boost_full > 0);
        f.intensity[0] = 0;
        let boost_zero = f.color_boost(0, 15);
        assert_eq!(boost_zero, 0);
    }

    #[test]
    fn draw_label_renders_pixels() {
        let mut fb = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        fb.draw_label(4, 4, "AB", 5);
        let lit: usize = fb.pixels.iter().filter(|&&p| p == 5).count();
        assert!(lit > 10, "A+B glyphs should light at least 10 pixels");
    }

    #[test]
    fn flyby_cache_loops_seamlessly() {
        let mut frame = RenderFrame::with_capacity(16);
        frame.len = 2;
        frame.positions[0] = 10.0; frame.positions[1] = 10.0;
        frame.positions[3] = 20.0; frame.positions[4] = 20.0;
        let edges = vec![(0, 1)];
        let fb_template = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        let mut cache = FlybyCache::prerender(
            &fb_template, &frame, &edges, 8, 10.0, (0.5, 2.0), 5, 2,
        );
        assert_eq!(cache.len(), 8);
        // Play through more than one loop — should not panic.
        for _ in 0..20 {
            let _ = cache.next_frame();
        }
        // Cursor wraps around.
        assert_eq!(cache.cursor, 20 % 8);
    }

    #[test]
    fn flyby_seek_nearest_finds_closest_frame() {
        let mut frame = RenderFrame::with_capacity(16);
        frame.len = 1;
        frame.positions[0] = 32.0; frame.positions[1] = 32.0;
        let fb_template = Framebuffer::with_tier(64, 64, PaletteTier::Full16);
        let mut cache = FlybyCache::prerender(
            &fb_template, &frame, &[], 16, 10.0, (1.0, 1.0), 5, 2,
        );
        cache.seek_nearest(32.0, 32.0);
        let f = &cache.frames[cache.cursor];
        let dx = f.cam_x - 32.0;
        let dy = f.cam_y - 32.0;
        assert!((dx * dx + dy * dy).sqrt() < 20.0);
    }

    #[test]
    fn compose_neo4j_full_with_wobble_fire_labels() {
        let mut fb = Framebuffer::with_tier(128, 128, PaletteTier::Full16);
        let mut frame = RenderFrame::with_capacity(16);
        frame.len = 2;
        frame.positions[0] = 30.0; frame.positions[1] = 30.0;
        frame.positions[3] = 90.0; frame.positions[4] = 90.0;
        let edges = vec![(0, 1)];
        let wobble = WobbleState::new(16);
        let mut fire = FireState::new(16);
        fire.fire(0, 255);
        let labels = vec!["NODE0", "NODE1"];
        compose_neo4j_full(
            &mut fb, &frame, &edges, 1.0, (0.0, 0.0),
            &wobble, &fire, &labels, 3, 1, 7,
        );
        // Node 0 should be brighter (fire boost) than base color 3.
        let node0_pixel = fb.pixels[30 * 128 + 30];
        assert!(node0_pixel >= 3, "node0 should have at least base color");
        // Label pixels should exist at color 7.
        let label_count = fb.pixels.iter().filter(|&&p| p == 7).count();
        assert!(label_count > 0, "labels should render");
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pyramid shader — heat diffusion through the cache-aligned pyramid.
//
// The inverse Stufenpyramide IS a GPU shader pipeline:
//   L1 (64²)   → 4 KB   → registers / L0 cache
//   L2 (256²)  → 64 KB  → L1 data cache
//   L3 (4K²)   → 2 MB   → L2 cache  (bit) / 16 MB (byte)
//   L4 (16K²)  → 32 MB  → L3 cache
//
// A perturbation enters at L1, diffuses at each level, then upscales
// 4× to the next. Each level physically runs in its matching CPU cache.
// The viewer sees cognition ripple through the hardware.
// ─────────────────────────────────────────────────────────────────────

/// 3×3 box-blur diffusion: each pixel = average of itself + 8 neighbors.
/// In-place via double buffer (src → dst, then swap pointers).
/// Palette-safe: result is clamped to [0, max_palette].
pub fn diffuse_step(
    src: &[u8], dst: &mut [u8],
    width: usize, height: usize,
    max_palette: u8,
) {
    for y in 0..height {
        for x in 0..width {
            let mut sum: u16 = 0;
            let mut count: u16 = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        sum += src[ny as usize * width + nx as usize] as u16;
                        count += 1;
                    }
                }
            }
            dst[y * width + x] = ((sum / count) as u8).min(max_palette);
        }
    }
}

/// Upscale 2× via nearest-neighbor (L_n → L_{n+1}).
pub fn upscale_2x(src: &[u8], src_w: usize, src_h: usize) -> (Vec<u8>, usize, usize) {
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut dst = vec![0u8; dst_w * dst_h];
    for sy in 0..src_h {
        for sx in 0..src_w {
            let v = src[sy * src_w + sx];
            let dy = sy * 2;
            let dx = sx * 2;
            dst[dy * dst_w + dx] = v;
            dst[dy * dst_w + dx + 1] = v;
            dst[(dy + 1) * dst_w + dx] = v;
            dst[(dy + 1) * dst_w + dx + 1] = v;
        }
    }
    (dst, dst_w, dst_h)
}

/// Four-level pyramid shader state.
///
/// Each level is a framebuffer at its native resolution. `tick()` runs
/// one diffusion step at each level, then upscales L1→L2→L3→L4.
/// Inject heat at L1 via `inject(x, y, intensity)`.
pub struct PyramidShader {
    /// L1: 64×64 (4 KB).
    pub l1: Vec<u8>,
    /// L2: 256×256 (64 KB).
    pub l2: Vec<u8>,
    /// L3: 1024×1024 (1 MB) — scaled down from 4K for practical display.
    pub l3: Vec<u8>,
    /// L4: 2048×2048 (4 MB) — the output surface.
    pub l4: Vec<u8>,
    /// Scratch buffer for double-buffer diffusion (same size as L4).
    scratch: Vec<u8>,
    /// Palette max (from tier).
    pub palette_max: u8,
    /// Tick counter.
    pub tick: u64,
}

impl PyramidShader {
    pub fn new(palette_max: u8) -> Self {
        Self {
            l1: vec![0u8; 64 * 64],
            l2: vec![0u8; 256 * 256],
            l3: vec![0u8; 1024 * 1024],
            l4: vec![0u8; 2048 * 2048],
            scratch: vec![0u8; 2048 * 2048],
            palette_max,
            tick: 0,
        }
    }

    /// Inject heat at L1 coordinates (0..64, 0..64).
    pub fn inject(&mut self, x: usize, y: usize, intensity: u8) {
        if x < 64 && y < 64 {
            self.l1[y * 64 + x] = self.l1[y * 64 + x].saturating_add(intensity).min(self.palette_max);
        }
    }

    /// One shader tick: diffuse each level, then cascade upward.
    ///
    /// This IS the cognitive shader made visible. Each level physically
    /// fits its CPU cache tier. The 4× widening at each step IS the
    /// cache hierarchy doubling pattern.
    pub fn tick(&mut self) {
        // 1. Diffuse at each level independently.
        //    L1: 64² = 4 KB → runs in registers / L0.
        let mut scratch_l1 = vec![0u8; 64 * 64];
        diffuse_step(&self.l1, &mut scratch_l1, 64, 64, self.palette_max);
        self.l1.copy_from_slice(&scratch_l1);

        //    L2: 256² = 64 KB → runs in L1 data cache.
        let mut scratch_l2 = vec![0u8; 256 * 256];
        diffuse_step(&self.l2, &mut scratch_l2, 256, 256, self.palette_max);
        self.l2.copy_from_slice(&scratch_l2);

        //    L3: 1024² = 1 MB → runs in L2 cache.
        let mut scratch_l3 = vec![0u8; 1024 * 1024];
        diffuse_step(&self.l3, &mut scratch_l3, 1024, 1024, self.palette_max);
        self.l3.copy_from_slice(&scratch_l3);

        // 2. Cascade: L1 upscales into L2, L2 into L3, L3 into L4.
        //    Additive blend (saturating) so existing diffusion + upscaled signal combine.
        let (up1, _, _) = upscale_2x(&self.l1, 64, 64);       // 128²
        let (up1b, _, _) = upscale_2x(&up1, 128, 128);         // 256²
        for (dst, src) in self.l2.iter_mut().zip(up1b.iter()) {
            *dst = dst.saturating_add(*src).min(self.palette_max);
        }

        let (up2, _, _) = upscale_2x(&self.l2, 256, 256);      // 512²
        let (up2b, _, _) = upscale_2x(&up2, 512, 512);          // 1024²
        for (dst, src) in self.l3.iter_mut().zip(up2b.iter()) {
            *dst = dst.saturating_add(*src).min(self.palette_max);
        }

        let (up3, _, _) = upscale_2x(&self.l3, 1024, 1024);    // 2048²
        for (dst, src) in self.l4.iter_mut().zip(up3.iter()) {
            *dst = dst.saturating_add(*src).min(self.palette_max);
        }

        // 3. Global decay on L4 (prevents saturation).
        for v in self.l4.iter_mut() {
            *v = v.saturating_sub(1);
        }

        self.tick += 1;
    }

    /// Compose a 2×2 panel view of all four levels into a framebuffer.
    ///
    /// Top-left = L1 (upscaled to panel size), top-right = L2,
    /// bottom-left = L3, bottom-right = L4. Each panel is `pw × ph`.
    pub fn compose_quad_view(&self, fb: &mut Framebuffer) {
        let pw = fb.width / 2;
        let ph = fb.height / 2;

        // L1 → top-left (upscale from 64² to pw×ph)
        blit_scaled(&self.l1, 64, 64, fb, 0, 0, pw, ph);
        // L2 → top-right (upscale from 256² to pw×ph)
        blit_scaled(&self.l2, 256, 256, fb, pw, 0, pw, ph);
        // L3 → bottom-left (downscale from 1024² to pw×ph)
        blit_scaled(&self.l3, 1024, 1024, fb, 0, ph, pw, ph);
        // L4 → bottom-right (downscale from 2048² to pw×ph)
        blit_scaled(&self.l4, 2048, 2048, fb, pw, ph, pw, ph);

        fb.dirty = (0, 0, fb.width, fb.height);
    }

    /// Memory footprint across all levels.
    pub fn memory_bytes(&self) -> usize {
        self.l1.len() + self.l2.len() + self.l3.len() + self.l4.len() + self.scratch.len()
    }
}

/// Nearest-neighbor scale-blit from src (src_w × src_h) into a region
/// of the framebuffer at (dst_x, dst_y) with size (dst_w × dst_h).
fn blit_scaled(
    src: &[u8], src_w: usize, src_h: usize,
    fb: &mut Framebuffer,
    dst_x: usize, dst_y: usize,
    dst_w: usize, dst_h: usize,
) {
    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            let px = dst_x + dx;
            let py = dst_y + dy;
            if px < fb.width && py < fb.height && sy < src_h && sx < src_w {
                fb.pixels[py * fb.width + px] = src[sy * src_w + sx];
            }
        }
    }
}

#[cfg(test)]
mod pyramid_tests {
    use super::*;

    #[test]
    fn pyramid_shader_inject_and_tick() {
        let mut ps = PyramidShader::new(15);
        ps.inject(32, 32, 15);
        assert_eq!(ps.l1[32 * 64 + 32], 15);
        ps.tick();
        // After one tick, heat should have diffused to neighbors at L1
        // and cascaded to L2/L3/L4.
        assert!(ps.l1[32 * 64 + 33] > 0, "L1 should diffuse right");
        assert!(ps.l2[128 * 256 + 128] > 0, "L2 should receive cascade");
    }

    #[test]
    fn pyramid_shader_decays_to_zero() {
        let mut ps = PyramidShader::new(15);
        ps.inject(32, 32, 15);
        for _ in 0..200 {
            ps.tick();
        }
        let l4_max = ps.l4.iter().copied().max().unwrap_or(0);
        assert_eq!(l4_max, 0, "L4 should decay to zero after enough ticks");
    }

    #[test]
    fn pyramid_shader_compose_quad_view() {
        let mut ps = PyramidShader::new(15);
        ps.inject(32, 32, 15);
        ps.tick();
        let mut fb = Framebuffer::with_tier(128, 128, PaletteTier::Full16);
        ps.compose_quad_view(&mut fb);
        // Top-left panel (L1 upscaled) should have nonzero pixels.
        let tl_sum: u32 = fb.pixels[..64 * 128].iter().map(|&v| v as u32).sum();
        assert!(tl_sum > 0, "L1 panel should show the injection");
    }

    #[test]
    fn pyramid_shader_memory_footprint() {
        let ps = PyramidShader::new(15);
        // L1=4K + L2=64K + L3=1M + L4=4M + scratch=4M ≈ 9.07 MB
        assert!(ps.memory_bytes() > 5_000_000);
        assert!(ps.memory_bytes() < 20_000_000);
    }

    #[test]
    fn upscale_2x_doubles_dimensions() {
        let src = vec![5u8; 8 * 8];
        let (dst, w, h) = upscale_2x(&src, 8, 8);
        assert_eq!(w, 16);
        assert_eq!(h, 16);
        assert!(dst.iter().all(|&v| v == 5));
    }

    #[test]
    fn diffuse_step_smooths_spike() {
        let mut src = vec![0u8; 16 * 16];
        src[8 * 16 + 8] = 15; // single hot pixel
        let mut dst = vec![0u8; 16 * 16];
        diffuse_step(&src, &mut dst, 16, 16, 15);
        // Center should have decreased (averaged with zero neighbors).
        assert!(dst[8 * 16 + 8] < 15);
        // At least one neighbor should be nonzero.
        let neighbor_sum: u16 = [
            dst[7 * 16 + 8], dst[9 * 16 + 8],
            dst[8 * 16 + 7], dst[8 * 16 + 9],
        ].iter().map(|&v| v as u16).sum();
        assert!(neighbor_sum > 0, "diffusion should spread to neighbors");
    }
}
