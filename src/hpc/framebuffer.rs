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
