//! CausalEdge64 integration for Jina token triples.
//!
//! Every SPO triple from token palette indices packs into one u64
//! with NARS truth, Pearl hierarchy, plasticity, and temporal index.

use super::codec::JinaPalette;

/// Pack an SPO triple + metadata into a CausalEdge64 (one u64).
///
/// Bit layout:
/// ```text
/// [S:8][P:8][O:8][freq:8][conf:8][pearl:3][dir:3][inf:3][plast:3][temporal:12]
/// ```
#[inline(always)]
pub fn pack_edge(
    s_palette: u8,
    p_palette: u8,
    o_palette: u8,
    frequency: f32,
    confidence: f32,
    pearl_mask: u8,
    temporal: u16,
) -> u64 {
    let f_u8 = (frequency.clamp(0.0, 1.0) * 255.0) as u8;
    let c_u8 = (confidence.clamp(0.0, 1.0) * 255.0) as u8;

    (s_palette as u64)
        | ((p_palette as u64) << 8)
        | ((o_palette as u64) << 16)
        | ((f_u8 as u64) << 24)
        | ((c_u8 as u64) << 32)
        | (((pearl_mask & 0x7) as u64) << 40)
        | (0b100u64 << 43) // default direction: S→O
        | (0b001u64 << 46) // default inference: observation
        | (0b100u64 << 49) // default plasticity: hot
        | (((temporal & 0xFFF) as u64) << 52)
}

/// Unpack S palette index from a CausalEdge64.
#[inline(always)]
pub fn edge_s(edge: u64) -> u8 {
    (edge & 0xFF) as u8
}

/// Unpack P palette index.
#[inline(always)]
pub fn edge_p(edge: u64) -> u8 {
    ((edge >> 8) & 0xFF) as u8
}

/// Unpack O palette index.
#[inline(always)]
pub fn edge_o(edge: u64) -> u8 {
    ((edge >> 16) & 0xFF) as u8
}

/// Unpack NARS frequency [0.0, 1.0].
#[inline(always)]
pub fn edge_freq(edge: u64) -> f32 {
    ((edge >> 24) & 0xFF) as f32 / 255.0
}

/// Unpack NARS confidence [0.0, 1.0].
#[inline(always)]
pub fn edge_conf(edge: u64) -> f32 {
    ((edge >> 32) & 0xFF) as f32 / 255.0
}

/// Unpack Pearl mask (3 bits: S=4, P=2, O=1).
#[inline(always)]
pub fn edge_pearl(edge: u64) -> u8 {
    ((edge >> 40) & 0x7) as u8
}

/// Unpack temporal index (12 bits: 0-4095).
#[inline(always)]
pub fn edge_temporal(edge: u64) -> u16 {
    ((edge >> 52) & 0xFFF) as u16
}

/// Unpack plasticity (3 bits).
#[inline(always)]
pub fn edge_plasticity(edge: u64) -> u8 {
    ((edge >> 49) & 0x7) as u8
}

/// Pearl-masked distance between two edges using palette distance table.
///
/// Only the planes selected by `mask` contribute to the distance.
/// mask=0b111 (SPO): full distance. mask=0b011 (_PO): interventional.
#[inline]
pub fn causal_distance(edge_a: u64, edge_b: u64, palette: &JinaPalette, mask: u8) -> u32 {
    let mut d = 0u32;
    if mask & 0b100 != 0 {
        d += palette.distance_table[edge_s(edge_a) as usize][edge_s(edge_b) as usize] as u32;
    }
    if mask & 0b010 != 0 {
        d += palette.distance_table[edge_p(edge_a) as usize][edge_p(edge_b) as usize] as u32;
    }
    if mask & 0b001 != 0 {
        d += palette.distance_table[edge_o(edge_a) as usize][edge_o(edge_b) as usize] as u32;
    }
    d
}

/// NARS revision: combine two truth values (old + new evidence).
#[inline]
pub fn nars_revision(old_freq: f32, old_conf: f32, new_freq: f32, new_conf: f32) -> (f32, f32) {
    let w1 = old_conf / (1.0 - old_conf + 1e-9);
    let w2 = new_conf / (1.0 - new_conf + 1e-9);
    let w = w1 + w2;
    let f = (w1 * old_freq + w2 * new_freq) / (w + 1e-9);
    let c = w / (w + 1.0);
    (f.clamp(0.0, 1.0), c.clamp(0.0, 0.999))
}

/// Update an edge with new evidence via NARS revision.
/// Returns the updated edge (new u64 with revised truth values).
#[inline]
pub fn revise_edge(edge: u64, evidence_freq: f32, evidence_conf: f32) -> u64 {
    let old_f = edge_freq(edge);
    let old_c = edge_conf(edge);
    let (new_f, new_c) = nars_revision(old_f, old_c, evidence_freq, evidence_conf);

    // Clear old truth bits, set new ones
    let mask = !(0xFF_u64 << 24 | 0xFF_u64 << 32);
    let cleared = edge & mask;
    let f_u8 = (new_f * 255.0) as u64;
    let c_u8 = (new_c * 255.0) as u64;
    cleared | (f_u8 << 24) | (c_u8 << 32)
}

/// NARS expectation: e = c × (f - 0.5) + 0.5.
#[inline(always)]
pub fn edge_expectation(edge: u64) -> f32 {
    let f = edge_freq(edge);
    let c = edge_conf(edge);
    c * (f - 0.5) + 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let edge = pack_edge(42, 187, 91, 0.75, 0.60, 0b111, 1234);
        assert_eq!(edge_s(edge), 42);
        assert_eq!(edge_p(edge), 187);
        assert_eq!(edge_o(edge), 91);
        assert!((edge_freq(edge) - 0.75).abs() < 0.01);
        assert!((edge_conf(edge) - 0.60).abs() < 0.01);
        assert_eq!(edge_pearl(edge), 0b111);
        assert_eq!(edge_temporal(edge), 1234);
    }

    #[test]
    fn test_temporal_sort() {
        let e1 = pack_edge(0, 0, 0, 0.5, 0.3, 0b111, 100);
        let e2 = pack_edge(0, 0, 0, 0.5, 0.3, 0b111, 200);
        let e3 = pack_edge(0, 0, 0, 0.5, 0.3, 0b111, 50);
        let mut edges = vec![e2, e3, e1];
        edges.sort(); // native u64 sort
        assert_eq!(edge_temporal(edges[0]), 50);
        assert_eq!(edge_temporal(edges[1]), 100);
        assert_eq!(edge_temporal(edges[2]), 200);
    }

    #[test]
    fn test_nars_revision_increases_confidence() {
        let edge = pack_edge(42, 187, 91, 0.5, 0.1, 0b111, 0);
        let revised = revise_edge(edge, 0.8, 0.3);
        assert!(edge_conf(revised) > edge_conf(edge));
    }

    #[test]
    fn test_nars_revision_10_observations() {
        let mut edge = pack_edge(42, 187, 91, 0.5, 0.1, 0b111, 0);
        for _ in 0..10 {
            edge = revise_edge(edge, 0.8, 0.3);
        }
        assert!(edge_conf(edge) > 0.7, "10 observations should give conf > 0.7");
        assert!(edge_freq(edge) > 0.6, "positive evidence should push freq up");
    }

    #[test]
    fn test_expectation() {
        let edge = pack_edge(0, 0, 0, 0.9, 0.8, 0b111, 0);
        let exp = edge_expectation(edge);
        // e = 0.8 * (0.9 - 0.5) + 0.5 = 0.8 * 0.4 + 0.5 = 0.82
        assert!((exp - 0.82).abs() < 0.02);
    }

    #[test]
    fn test_pearl_mask_distance() {
        use super::super::codec::{Base17Token, JinaPalette, BASE_DIM};

        // Build minimal palette for testing
        let tokens: Vec<Base17Token> = (0..256)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                dims[0] = (i as i16) * 100;
                dims[1] = (i as i16) * 50;
                Base17Token { dims }
            })
            .collect();
        let palette = JinaPalette::build(&tokens, 5);

        let e1 = pack_edge(10, 20, 30, 0.5, 0.3, 0b111, 0);
        let e2 = pack_edge(40, 50, 60, 0.5, 0.3, 0b111, 0);

        let d_spo = causal_distance(e1, e2, &palette, 0b111); // full
        let d_po = causal_distance(e1, e2, &palette, 0b011); // remove S
        let d_so = causal_distance(e1, e2, &palette, 0b101); // remove P
        let d_s = causal_distance(e1, e2, &palette, 0b100); // S only

        assert!(d_spo > d_po, "removing S should reduce distance");
        assert!(d_spo > d_so, "removing P should reduce distance");
        assert_eq!(d_spo, d_s + d_po - causal_distance(e1, e2, &palette, 0b000),
            "planes should be additive (within rounding)... actually just check ordering");
        assert!(d_s > 0, "different S should have positive distance");
    }
}
