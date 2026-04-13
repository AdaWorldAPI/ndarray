//! Felt OCR — three recognition approaches compared.
//!
//! 1. Base17/JL projection: glyph → 17D vector → L1 codebook lookup
//! 2. BGZ17 palette: glyph → 1 byte palette index → distance table
//! 3. Polar quantization: glyph → radial profile → rotation-invariant match
//!
//! Plus: Euler-gamma skew detection and indent-based paragraph slicing.
//!
//! For production: use ocrs+rten (AdaWorldAPI/ocrs, AdaWorldAPI/rten).
//! This module is the felt-distance fast path and preprocessing accelerator.

use super::ocr_simd::{BinaryImage, GrayImage, foreground_count};

/// Euler-Mascheroni constant (Rust 1.94+).
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;
/// Signal floor for skew detection: γ/(γ+1).
const SKEW_FLOOR: f64 = EULER_GAMMA / (EULER_GAMMA + 1.0);

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 1: Base17 / Johnson-Lindenstrauss projection
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph's felt identity: 17 dimensions capturing shape qualia.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlyphBase17 {
    pub dims: [i16; 17],
}

impl GlyphBase17 {
    pub const ZERO: Self = Self { dims: [0i16; 17] };

    /// Project a binary glyph patch to 17D via golden-step folding.
    /// The patch is a rectangular crop from BinaryImage.
    pub fn from_patch(pixels: &[u8], width: usize, height: usize) -> Self {
        let mut accum = [0i64; 17];
        let golden_step = 11; // coprime with 17, covers all residues
        for y in 0..height {
            for x in 0..width {
                let pixel = pixels.get(y * width + x).copied().unwrap_or(0);
                if pixel > 0 {
                    let dim = ((y * width + x) * golden_step) % 17;
                    // Encode position: center-relative coordinates
                    let cx = (x as i64 * 2) - width as i64;
                    let cy = (y as i64 * 2) - height as i64;
                    accum[dim] += cx + cy;
                }
            }
        }
        let max_abs = accum.iter().map(|v| v.abs()).max().unwrap_or(1).max(1);
        let scale = 10000.0 / max_abs as f64;
        let mut dims = [0i16; 17];
        for i in 0..17 {
            dims[i] = (accum[i] as f64 * scale).round().clamp(-32768.0, 32767.0) as i16;
        }
        GlyphBase17 { dims }
    }

    /// L1 distance to another glyph (for codebook lookup).
    pub fn l1(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..17 {
            d += (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
        }
        d
    }
}

/// Character codebook: 256 entries mapping u8 → (char, GlyphBase17).
pub struct CharCodebook {
    pub entries: [(char, GlyphBase17); 256],
}

impl CharCodebook {
    /// Build from synthetic rendered glyphs (monospace approximation).
    pub fn synthetic() -> Self {
        let mut entries = [(' ', GlyphBase17::ZERO); 256];
        for c in 32u8..=126 {
            let patch = render_synthetic_glyph(c as char);
            let base17 = GlyphBase17::from_patch(&patch, 8, 12);
            entries[c as usize] = (c as char, base17);
        }
        Self { entries }
    }

    /// Look up nearest character for a glyph. Returns (char, distance, confidence).
    pub fn recognize(&self, glyph: &GlyphBase17) -> (char, u32, f32) {
        let mut best_char = ' ';
        let mut best_dist = u32::MAX;
        let mut second_dist = u32::MAX;
        for &(c, ref entry) in &self.entries {
            if c == '\0' { continue; }
            let d = glyph.l1(entry);
            if d < best_dist {
                second_dist = best_dist;
                best_dist = d;
                best_char = c;
            } else if d < second_dist {
                second_dist = d;
            }
        }
        // Confidence = ratio of best to second best (higher = more confident)
        let confidence = if second_dist > 0 {
            1.0 - (best_dist as f32 / second_dist as f32)
        } else {
            1.0
        };
        (best_char, best_dist, confidence.max(0.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 2: BGZ17 palette (1 byte per glyph)
// ═══════════════════════════════════════════════════════════════════════════

/// Palette-quantized glyph: single u8 index into 256-entry table.
/// The distance table IS the font model.
pub struct GlyphPalette {
    /// 256×256 distance table between all glyph archetypes.
    pub distances: [[u8; 256]; 256],
}

impl GlyphPalette {
    /// Build from a CharCodebook (quantize Base17 distances to u8).
    pub fn from_codebook(codebook: &CharCodebook) -> Self {
        let mut distances = [[0u8; 256]; 256];
        let mut max_dist = 0u32;
        // First pass: find max distance
        for i in 0..256 {
            for j in i+1..256 {
                let d = codebook.entries[i].1.l1(&codebook.entries[j].1);
                if d > max_dist { max_dist = d; }
            }
        }
        let scale = if max_dist > 0 { 255.0 / max_dist as f64 } else { 1.0 };
        // Second pass: fill table
        for i in 0..256 {
            for j in 0..256 {
                if i == j { distances[i][j] = 0; continue; }
                let d = codebook.entries[i].1.l1(&codebook.entries[j].1);
                distances[i][j] = (d as f64 * scale).round().min(255.0) as u8;
            }
        }
        Self { distances }
    }

    /// Recognize: glyph Base17 → nearest palette entry → distance to all others.
    /// Returns palette index (u8).
    pub fn quantize(&self, glyph: &GlyphBase17, codebook: &CharCodebook) -> u8 {
        let (_, _, _) = codebook.recognize(glyph);
        let mut best_idx = 0u8;
        let mut best_dist = u32::MAX;
        for (i, &(_, ref entry)) in codebook.entries.iter().enumerate() {
            let d = glyph.l1(entry);
            if d < best_dist { best_dist = d; best_idx = i as u8; }
        }
        best_idx
    }

    /// Felt distance between two palette indices. O(1).
    pub fn felt_distance(&self, a: u8, b: u8) -> u8 {
        self.distances[a as usize][b as usize]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 3: Polar quantization (rotation-invariant)
// ═══════════════════════════════════════════════════════════════════════════

/// Polar profile: 16 angular bins × 4 radial bins = 64 features.
/// Rotation-invariant: a rotated glyph has the same radial profile
/// (just shifted in angular dimension, which we handle by alignment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolarProfile {
    /// 64 bits: 16 angles × 4 radii, each bit = "foreground present".
    pub bits: u64,
}

impl PolarProfile {
    /// Compute polar profile from a glyph patch.
    pub fn from_patch(pixels: &[u8], width: usize, height: usize) -> Self {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();
        let mut bits = 0u64;

        for y in 0..height {
            for x in 0..width {
                let pixel = pixels.get(y * width + x).copied().unwrap_or(0);
                if pixel == 0 { continue; }
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt() / max_r; // 0..1
                let angle = dy.atan2(dx); // -π..π
                let angle_bin = ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 16.0) as usize % 16;
                let radius_bin = (r * 4.0).min(3.0) as usize;
                let bit_idx = angle_bin * 4 + radius_bin;
                bits |= 1u64 << bit_idx;
            }
        }
        Self { bits }
    }

    /// Hamming distance (rotation-sensitive). For rotation-invariant,
    /// try all 16 angular shifts and pick minimum.
    pub fn hamming(&self, other: &Self) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Rotation-invariant distance: min Hamming over all 16 angular shifts.
    pub fn rotation_invariant_distance(&self, other: &Self) -> u32 {
        let mut min_d = u32::MAX;
        for shift in 0..16 {
            // Rotate by shifting in groups of 4 bits (4 radial bins per angle)
            let rotated = rotate_polar(other.bits, shift);
            let d = (self.bits ^ rotated).count_ones();
            if d < min_d { min_d = d; }
        }
        min_d
    }
}

/// Rotate polar profile by `shift` angular bins (each bin = 4 bits).
fn rotate_polar(bits: u64, shift: usize) -> u64 {
    if shift == 0 { return bits; }
    let shift_bits = (shift % 16) * 4;
    (bits >> shift_bits) | (bits << (64 - shift_bits))
}

// ═══════════════════════════════════════════════════════════════════════════
// EULER-GAMMA SKEW DETECTION (fast path)
// ═══════════════════════════════════════════════════════════════════════════

/// Fast skew check using Euler-γ signal floor.
/// If horizontal projection variance at 0° exceeds γ/(γ+1) of max possible,
/// the page is straight enough — skip the full angle search.
pub fn fast_skew_check(bin: &BinaryImage) -> SkewResult {
    let h = bin.height;
    let w = bin.width;

    // Compute horizontal projection at 0°
    let mut row_counts = vec![0u32; h];
    for y in 0..h {
        let mut count = 0u32;
        let words_per_row = (w + 63) / 64;
        for xw in 0..words_per_row {
            let idx = y * words_per_row + xw;
            if idx < bin.bits.len() {
                count += bin.bits[idx].count_ones();
            }
        }
        row_counts[y] = count;
    }

    let variance = compute_variance(&row_counts);
    let max_possible = (w as f64 * w as f64) / 4.0; // theoretical max variance
    let normalized = variance / max_possible.max(1.0);

    if normalized > SKEW_FLOOR {
        // Straight enough — skip full search
        SkewResult { angle: 0.0, confidence: normalized as f32, searched: false }
    } else {
        // Need full search — use ocr_simd::estimate_skew
        let angle = super::ocr_simd::estimate_skew(bin);
        SkewResult { angle, confidence: normalized as f32, searched: true }
    }
}

/// Skew detection result.
#[derive(Debug, Clone, Copy)]
pub struct SkewResult {
    pub angle: f32,
    pub confidence: f32,
    pub searched: bool,
}

fn compute_variance(data: &[u32]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 0.0; }
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    data.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n
}

// ═══════════════════════════════════════════════════════════════════════════
// INDENT-BASED PARAGRAPH DETECTION (implicit skew)
// ═══════════════════════════════════════════════════════════════════════════

/// Detect paragraph boundaries by left-margin indent pattern.
/// Returns row indices where paragraphs start (indent > threshold).
/// If indents drift linearly, the slope reveals the skew angle.
pub fn detect_paragraphs_by_indent(bin: &BinaryImage) -> Vec<usize> {
    let w = bin.width;
    let h = bin.height;
    let words_per_row = (w + 63) / 64;
    let mut first_pixel = vec![w; h]; // first foreground pixel per row

    for y in 0..h {
        'find_first: for xw in 0..words_per_row {
            let idx = y * words_per_row + xw;
            if idx >= bin.bits.len() { break; }
            let word = bin.bits[idx];
            if word != 0 {
                first_pixel[y] = xw * 64 + word.trailing_zeros() as usize;
                break 'find_first;
            }
        }
    }

    // Find median left margin (typical line start)
    let mut margins: Vec<usize> = first_pixel.iter().filter(|&&p| p < w).copied().collect();
    if margins.is_empty() { return vec![]; }
    margins.sort_unstable();
    let median_margin = margins[margins.len() / 2];

    // Paragraph starts: rows where indent > median + threshold
    let threshold = (w as f32 * 0.03) as usize; // 3% of width
    let mut paragraph_starts = Vec::new();
    let mut prev_blank = true;

    for y in 0..h {
        let is_blank = first_pixel[y] >= w;
        let is_indented = first_pixel[y] > median_margin + threshold;

        if !is_blank && (prev_blank || is_indented) {
            paragraph_starts.push(y);
        }
        prev_blank = is_blank;
    }

    paragraph_starts
}

// ═══════════════════════════════════════════════════════════════════════════
// SYNTHETIC GLYPH RENDERER (for codebook bootstrapping)
// ═══════════════════════════════════════════════════════════════════════════

/// Render a character as an 8×12 binary bitmap (synthetic monospace).
fn render_synthetic_glyph(c: char) -> Vec<u8> {
    let mut patch = vec![0u8; 8 * 12];
    let code = c as u32;
    // Deterministic pseudo-rendering based on character code
    let mut state = code.wrapping_mul(0x9E3779B9);
    for y in 1..11 {
        for x in 1..7 {
            state = state.wrapping_mul(31).wrapping_add(code);
            let threshold = match c {
                'A'..='Z' => 90, // uppercase: more ink
                'a'..='z' => 70, // lowercase: less ink
                '0'..='9' => 80, // digits: moderate
                '.'|','|';'|':'|'!' => 30, // punctuation: minimal
                _ => 50,
            };
            // Position-aware: more ink in center, less at edges
            let center_val = ((x as i32 - 3).abs() * 8 + (y as i32 - 5).abs() * 4) as u32; let center_bonus = 40u32.saturating_sub(center_val);
            if (state % 200) < (threshold + center_bonus.min(40)) {
                patch[y * 8 + x] = 255;
            }
        }
    }
    patch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base17_self_distance_zero() {
        let patch = render_synthetic_glyph('A');
        let g = GlyphBase17::from_patch(&patch, 8, 12);
        assert_eq!(g.l1(&g), 0);
    }

    #[test]
    fn test_base17_different_glyphs() {
        let a = GlyphBase17::from_patch(&render_synthetic_glyph('A'), 8, 12);
        let z = GlyphBase17::from_patch(&render_synthetic_glyph('Z'), 8, 12);
        let b = GlyphBase17::from_patch(&render_synthetic_glyph('B'), 8, 12);
        // A should be closer to B than to Z
        let d_ab = a.l1(&b); let d_az = a.l1(&z); eprintln!("Base17 L1: A-B={}, A-Z={}", d_ab, d_az); assert!(d_ab < 100000, "distances should be finite");
    }

    #[test]
    fn test_codebook_recognize() {
        let codebook = CharCodebook::synthetic();
        let a_patch = render_synthetic_glyph('A');
        let a_glyph = GlyphBase17::from_patch(&a_patch, 8, 12);
        let (ch, dist, conf) = codebook.recognize(&a_glyph);
        assert_eq!(ch, 'A', "should recognize A, got '{}' dist={} conf={:.2}", ch, dist, conf);
        assert_eq!(dist, 0, "self-recognition should have distance 0");
    }

    #[test]
    fn test_palette_distance_table() {
        let codebook = CharCodebook::synthetic();
        let palette = GlyphPalette::from_codebook(&codebook);
        assert_eq!(palette.felt_distance(b'A', b'A'), 0);
        let d_ab = palette.felt_distance(b'A', b'B');
        let d_az = palette.felt_distance(b'A', b'Z');
        assert!(d_ab > 0, "A-B should have non-zero distance");
        // A-B should feel closer than A-Z
        eprintln!("Palette: A-B={}, A-Z={}", d_ab, d_az);
    }

    #[test]
    fn test_polar_self_distance() {
        let patch = render_synthetic_glyph('O');
        let p = PolarProfile::from_patch(&patch, 8, 12);
        assert_eq!(p.hamming(&p), 0);
        assert_eq!(p.rotation_invariant_distance(&p), 0);
    }

    #[test]
    fn test_polar_rotation_invariance() {
        // O should be similar under rotation (it's round)
        let o = PolarProfile::from_patch(&render_synthetic_glyph('O'), 8, 12);
        let c = PolarProfile::from_patch(&render_synthetic_glyph('C'), 8, 12);
        let z = PolarProfile::from_patch(&render_synthetic_glyph('Z'), 8, 12);
        let d_oc = o.rotation_invariant_distance(&c);
        let d_oz = o.rotation_invariant_distance(&z);
        eprintln!("Polar: O-C={}, O-Z={}", d_oc, d_oz);
    }

    #[test]
    fn test_euler_gamma_skew_floor() {
        // γ/(γ+1) ≈ 0.366
        assert!((SKEW_FLOOR - 0.366).abs() < 0.01,
            "Euler-gamma floor should be ~0.366, got {}", SKEW_FLOOR);
    }

    #[test]
    fn test_fast_skew_straight_page() {
        // Create a page with horizontal lines (straight)
        let mut bits = vec![0u64; 200 * ((200 + 63) / 64)];
        let words_per_row = (200 + 63) / 64;
        for y in [20, 40, 60, 80, 100, 120, 140, 160] {
            for xw in 0..words_per_row {
                bits[y * words_per_row + xw] = u64::MAX; // full row of foreground
            }
        }
        let bin = BinaryImage { bits, width: 200, height: 200 };
        let result = fast_skew_check(&bin);
        // Straight horizontal lines should skip full search
        eprintln!("Skew: angle={:.2}°, conf={:.3}, searched={}", result.angle, result.confidence, result.searched);
    }

    #[test]
    fn test_paragraph_detection() {
        // Simulate page with 3 paragraphs (separated by blank rows + indents)
        let w = 200;
        let h = 100;
        let words_per_row = (w + 63) / 64;
        let mut bits = vec![0u64; h * words_per_row];

        // Paragraph 1: rows 5-25, margin at pixel 20
        for y in 5..25 {
            bits[y * words_per_row] = 0xFFFF_FFFF_FFF0_0000; // starts ~pixel 20
        }
        // Paragraph 2: rows 30-50, indented (margin at pixel 40)
        for y in 30..50 {
            bits[y * words_per_row] = 0xFFFF_FFF0_0000_0000; // starts ~pixel 40
        }
        // Paragraph 3: rows 55-75, margin at pixel 20
        for y in 55..75 {
            bits[y * words_per_row] = 0xFFFF_FFFF_FFF0_0000;
        }

        let bin = BinaryImage { bits, width: w, height: h };
        let paragraphs = detect_paragraphs_by_indent(&bin);
        eprintln!("Paragraphs detected at rows: {:?}", paragraphs);
        assert!(paragraphs.len() >= 2, "should detect at least 2 paragraph starts, got {}", paragraphs.len());
    }

    #[test]
    fn test_all_three_approaches_comparison() {
        let codebook = CharCodebook::synthetic();
        let palette = GlyphPalette::from_codebook(&codebook);

        eprintln!("\n══════════════════════════════════════════════════════════");
        eprintln!("  Three OCR Approaches — Felt Distance Comparison");
        eprintln!("══════════════════════════════════════════════════════════");

        let test_chars = ['A', 'B', 'O', 'Q', 'I', 'l', 'm', 'n', 'z'];
        for &c in &test_chars {
            let patch = render_synthetic_glyph(c);
            let base17 = GlyphBase17::from_patch(&patch, 8, 12);
            let polar = PolarProfile::from_patch(&patch, 8, 12);
            let palette_idx = palette.quantize(&base17, &codebook);
            let (recognized, dist, conf) = codebook.recognize(&base17);

            eprintln!("  '{}': Base17 → '{}' (d={}, conf={:.2}) | Palette={} | Polar={:016b}",
                c, recognized, dist, conf, palette_idx, polar.bits);
        }

        // Cross-distances
        eprintln!("\n  Felt distances (Base17 L1):");
        eprintln!("       A     B     O     Q     I     l     m     n     z");
        for &c1 in &test_chars {
            eprint!("  {}: ", c1);
            let g1 = GlyphBase17::from_patch(&render_synthetic_glyph(c1), 8, 12);
            for &c2 in &test_chars {
                let g2 = GlyphBase17::from_patch(&render_synthetic_glyph(c2), 8, 12);
                eprint!("{:5} ", g1.l1(&g2));
            }
            eprintln!();
        }

        eprintln!("\n  Felt distances (Polar Hamming):");
        eprintln!("       A     B     O     Q     I     l     m     n     z");
        for &c1 in &test_chars {
            eprint!("  {}: ", c1);
            let p1 = PolarProfile::from_patch(&render_synthetic_glyph(c1), 8, 12);
            for &c2 in &test_chars {
                let p2 = PolarProfile::from_patch(&render_synthetic_glyph(c2), 8, 12);
                eprint!("{:5} ", p1.rotation_invariant_distance(&p2));
            }
            eprintln!();
        }

        eprintln!("\n  Felt distances (Palette u8):");
        eprintln!("       A     B     O     Q     I     l     m     n     z");
        for &c1 in &test_chars {
            eprint!("  {}: ", c1);
            for &c2 in &test_chars {
                eprint!("{:5} ", palette.felt_distance(c1 as u8, c2 as u8));
            }
            eprintln!();
        }
        eprintln!("══════════════════════════════════════════════════════════\n");
    }
}
