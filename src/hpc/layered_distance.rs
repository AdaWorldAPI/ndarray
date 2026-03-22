//! Layered distance function for CLAM tree building and search.
//!
//! Provides O(1) distance lookups by reading palette indices from containers
//! (the lance-graph 256-word container format) and using precomputed
//! distance matrices.
//!
//! Also provides TruthGate for filtering results by minimum expectation
//! (frequency * confidence threshold).

use super::bgz17_bridge::PaletteEdge;
use super::palette_distance::SpoDistanceMatrices;

/// Container word layout constants (matching lance-graph container.rs).
/// W112..W128 hold Base17 data; W125 holds the palette edge.
pub const W_BASE17_START: usize = 112;
pub const W_PALETTE_WORD: usize = 125;

/// Word indices for truth values in container format.
/// W4 = frequency (f32 in lower 32 bits), W5 = confidence (f32 in lower 32 bits).
const W_FREQUENCY: usize = 4;
const W_CONFIDENCE: usize = 5;

/// Read palette edge from container W125.
///
/// The palette edge is packed into the low 24 bits of word 125:
/// bits [0..8) = s_idx, bits [8..16) = p_idx, bits [16..24) = o_idx.
pub fn read_palette_edge(container: &[u64; 256]) -> PaletteEdge {
    let w = container[W_PALETTE_WORD];
    PaletteEdge {
        s_idx: (w & 0xFF) as u8,
        p_idx: ((w >> 8) & 0xFF) as u8,
        o_idx: ((w >> 16) & 0xFF) as u8,
    }
}

/// Write palette edge into container W125.
pub fn write_palette_edge(container: &mut [u64; 256], pe: PaletteEdge) {
    let packed = pe.s_idx as u64
        | ((pe.p_idx as u64) << 8)
        | ((pe.o_idx as u64) << 16);
    // Preserve upper bits
    container[W_PALETTE_WORD] = (container[W_PALETTE_WORD] & !0xFF_FFFF) | packed;
}

/// Read truth value (frequency, confidence) from container W4-W5.
///
/// Both are stored as f32 reinterpreted into the lower 32 bits of the u64 word.
pub fn read_truth(container: &[u64; 256]) -> (f32, f32) {
    let freq = f32::from_bits(container[W_FREQUENCY] as u32);
    let conf = f32::from_bits(container[W_CONFIDENCE] as u32);
    (freq, conf)
}

/// Write truth value (frequency, confidence) into container W4-W5.
pub fn write_truth(container: &mut [u64; 256], frequency: f32, confidence: f32) {
    container[W_FREQUENCY] = (container[W_FREQUENCY] & !0xFFFF_FFFF)
        | frequency.to_bits() as u64;
    container[W_CONFIDENCE] = (container[W_CONFIDENCE] & !0xFFFF_FFFF)
        | confidence.to_bits() as u64;
}

/// Layered distance: O(1) palette lookup between two containers.
///
/// Reads palette edges from W125 of each container, then looks up the
/// precomputed SPO distance in the distance matrices.
pub fn palette_distance(
    dm: &SpoDistanceMatrices,
    a: &[u64; 256],
    b: &[u64; 256],
) -> u32 {
    let pe_a = read_palette_edge(a);
    let pe_b = read_palette_edge(b);
    dm.spo_distance(
        pe_a.s_idx, pe_a.p_idx, pe_a.o_idx,
        pe_b.s_idx, pe_b.p_idx, pe_b.o_idx,
    )
}

/// TruthGate: filter by minimum expectation.
///
/// Expectation is computed as: `confidence * (frequency - 0.5) + 0.5`
/// which maps (freq, conf) to [0, 1] where:
/// - expectation = 0.5 means no evidence either way
/// - expectation = 1.0 means certain positive
/// - expectation = 0.0 means certain negative
#[derive(Clone, Copy, Debug)]
pub struct TruthGate {
    pub min_expectation: f32,
}

impl TruthGate {
    /// No filtering: all results pass.
    pub const OPEN: Self = Self { min_expectation: 0.0 };
    /// Weak evidence threshold.
    pub const WEAK: Self = Self { min_expectation: 0.4 };
    /// Normal evidence threshold.
    pub const NORMAL: Self = Self { min_expectation: 0.6 };
    /// Strong evidence threshold.
    pub const STRONG: Self = Self { min_expectation: 0.75 };
    /// Near-certain evidence threshold.
    pub const CERTAIN: Self = Self { min_expectation: 0.9 };

    /// Check if a (frequency, confidence) pair passes the gate.
    #[inline]
    pub fn passes(&self, frequency: f32, confidence: f32) -> bool {
        let expectation = confidence * (frequency - 0.5) + 0.5;
        expectation >= self.min_expectation
    }

    /// Compute expectation from frequency and confidence.
    #[inline]
    pub fn expectation(frequency: f32, confidence: f32) -> f32 {
        confidence * (frequency - 0.5) + 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_container(s_idx: u8, p_idx: u8, o_idx: u8, freq: f32, conf: f32) -> [u64; 256] {
        let mut c = [0u64; 256];
        write_palette_edge(&mut c, PaletteEdge { s_idx, p_idx, o_idx });
        write_truth(&mut c, freq, conf);
        c
    }

    #[test]
    fn test_read_write_palette_edge_roundtrip() {
        let mut container = [0u64; 256];
        let pe = PaletteEdge { s_idx: 42, p_idx: 128, o_idx: 255 };
        write_palette_edge(&mut container, pe);
        let read = read_palette_edge(&container);
        assert_eq!(pe, read);
    }

    #[test]
    fn test_read_write_palette_edge_zero() {
        let mut container = [0u64; 256];
        let pe = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        write_palette_edge(&mut container, pe);
        let read = read_palette_edge(&container);
        assert_eq!(pe, read);
    }

    #[test]
    fn test_read_write_truth_roundtrip() {
        let mut container = [0u64; 256];
        write_truth(&mut container, 0.75, 0.9);
        let (f, c) = read_truth(&container);
        assert!((f - 0.75).abs() < 1e-6);
        assert!((c - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_read_write_truth_zero() {
        let mut container = [0u64; 256];
        write_truth(&mut container, 0.0, 0.0);
        let (f, c) = read_truth(&container);
        assert_eq!(f, 0.0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_palette_distance_self_zero() {
        use super::super::palette_distance::{Palette, SpoDistanceMatrices};
        use super::super::bgz17_bridge::Base17;

        let entries: Vec<Base17> = (0..16)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        let pal = Palette { entries };
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let c = make_container(5, 5, 5, 0.8, 0.9);
        assert_eq!(palette_distance(&dm, &c, &c), 0);
    }

    #[test]
    fn test_palette_distance_symmetric() {
        use super::super::palette_distance::{Palette, SpoDistanceMatrices};
        use super::super::bgz17_bridge::Base17;

        let entries: Vec<Base17> = (0..16)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        let pal = Palette { entries };
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let a = make_container(3, 7, 11, 0.8, 0.9);
        let b = make_container(5, 2, 14, 0.6, 0.7);
        assert_eq!(palette_distance(&dm, &a, &b), palette_distance(&dm, &b, &a));
    }

    #[test]
    fn test_truth_gate_open() {
        assert!(TruthGate::OPEN.passes(0.0, 0.0));
        assert!(TruthGate::OPEN.passes(1.0, 1.0));
        assert!(TruthGate::OPEN.passes(0.5, 0.5));
    }

    #[test]
    fn test_truth_gate_certain() {
        // freq=1.0, conf=1.0 => expectation = 1.0 * (1.0 - 0.5) + 0.5 = 1.0
        assert!(TruthGate::CERTAIN.passes(1.0, 1.0));
        // freq=0.5, conf=1.0 => expectation = 1.0 * (0.5 - 0.5) + 0.5 = 0.5
        assert!(!TruthGate::CERTAIN.passes(0.5, 1.0));
        // freq=0.9, conf=0.95 => expectation = 0.95 * 0.4 + 0.5 = 0.88
        assert!(!TruthGate::CERTAIN.passes(0.9, 0.95));
    }

    #[test]
    fn test_truth_gate_normal() {
        // freq=0.8, conf=0.8 => expectation = 0.8 * 0.3 + 0.5 = 0.74
        assert!(TruthGate::NORMAL.passes(0.8, 0.8));
        // freq=0.5, conf=0.5 => expectation = 0.5 * 0.0 + 0.5 = 0.5
        assert!(!TruthGate::NORMAL.passes(0.5, 0.5));
    }

    #[test]
    fn test_truth_gate_weak() {
        // freq=0.6, conf=0.5 => expectation = 0.5 * 0.1 + 0.5 = 0.55
        assert!(TruthGate::WEAK.passes(0.6, 0.5));
        // freq=0.3, conf=0.5 => expectation = 0.5 * -0.2 + 0.5 = 0.4
        assert!(TruthGate::WEAK.passes(0.3, 0.5));
    }

    #[test]
    fn test_expectation_formula() {
        // Perfect positive evidence
        assert!((TruthGate::expectation(1.0, 1.0) - 1.0).abs() < 1e-6);
        // No evidence
        assert!((TruthGate::expectation(0.5, 0.0) - 0.5).abs() < 1e-6);
        // Perfect negative evidence
        assert!((TruthGate::expectation(0.0, 1.0) - 0.0).abs() < 1e-6);
        // Zero confidence, any frequency => 0.5
        assert!((TruthGate::expectation(0.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((TruthGate::expectation(1.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_write_palette_edge_preserves_upper_bits() {
        let mut container = [0u64; 256];
        container[W_PALETTE_WORD] = 0xFFFF_FFFF_FF00_0000;
        let pe = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        write_palette_edge(&mut container, pe);
        // Upper bits should be preserved
        assert_eq!(container[W_PALETTE_WORD] & 0xFFFF_FFFF_FF00_0000, 0xFFFF_FFFF_FF00_0000);
        let read = read_palette_edge(&container);
        assert_eq!(read, pe);
    }
}
