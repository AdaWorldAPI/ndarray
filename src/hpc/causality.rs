//! Causality decomposition over qualia-space dimensions.
//!
//! Maps BF16-encoded qualia to causal structure using NARS-style truth values
//! and per-dimension awareness states.
//!
//! Ported from rustynum-core/causality.rs.

use super::bf16_truth::{AwarenessState, PackedQualia, SuperpositionState};

// ---------------------------------------------------------------------------
// qualia_dim — named dimension constants
// ---------------------------------------------------------------------------

/// Named qualia dimension indices (0..15).
pub mod qualia_dim {
    /// Luminance / brightness.
    pub const LUMINANCE: usize = 0;
    /// Red-green chromatic channel.
    pub const RED_GREEN: usize = 1;
    /// Blue-yellow chromatic channel.
    pub const BLUE_YELLOW: usize = 2;
    /// Pitch / auditory frequency.
    pub const PITCH: usize = 3;
    /// Warmth / temperature.
    pub const WARMTH: usize = 4;
    /// Pressure / tactile.
    pub const PRESSURE: usize = 5;
    /// Social / interpersonal valence.
    pub const SOCIAL: usize = 6;
    /// Temporal flow / duration.
    pub const TEMPORAL: usize = 7;
    /// Sacredness / numinosity.
    pub const SACREDNESS: usize = 8;
    /// Arousal / activation level.
    pub const AROUSAL: usize = 9;
    /// Valence / hedonic tone.
    pub const VALENCE: usize = 10;
    /// Agency / sense of control.
    pub const AGENCY: usize = 11;
    /// Spatial depth.
    pub const DEPTH: usize = 12;
    /// Texture / surface quality.
    pub const TEXTURE: usize = 13;
    /// Familiarity / recognition.
    pub const FAMILIARITY: usize = 14;
    /// Surprise / novelty.
    pub const SURPRISE: usize = 15;
}

/// The three causality-relevant dimensions: warmth, social, sacredness.
pub const CAUSALITY_DIMS: [usize; 3] = [
    qualia_dim::WARMTH,    // 4
    qualia_dim::SOCIAL,    // 6
    qualia_dim::SACREDNESS, // 8
];

// ---------------------------------------------------------------------------
// CausalityDirection
// ---------------------------------------------------------------------------

/// Direction of causal influence between two qualia points along a dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalityDirection {
    /// A causes B (A -> B).
    Forward,
    /// B causes A (B -> A).
    Backward,
    /// No directional relationship detected.
    None,
}

impl CausalityDirection {
    /// Derive direction from two qualia points along a specific dimension.
    ///
    /// Compares resonance values: if `a > b`, forward; if `b > a`, backward;
    /// otherwise none.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::causality::CausalityDirection;
    /// use ndarray::hpc::bf16_truth::PackedQualia;
    ///
    /// let mut a = PackedQualia::zero();
    /// let b = PackedQualia::zero();
    /// a.resonance[4] = 10;
    /// assert_eq!(CausalityDirection::from_qualia(&a, &b, 4), CausalityDirection::Forward);
    /// ```
    pub fn from_qualia(a: &PackedQualia, b: &PackedQualia, dim: usize) -> Self {
        assert!(dim < 16, "qualia dimension must be < 16, got {}", dim);
        let va = a.resonance[dim];
        let vb = b.resonance[dim];
        if va > vb {
            CausalityDirection::Forward
        } else if vb > va {
            CausalityDirection::Backward
        } else {
            CausalityDirection::None
        }
    }

    /// Reverse the direction.
    ///
    /// Forward <-> Backward, None stays None.
    pub fn flip(self) -> Self {
        match self {
            CausalityDirection::Forward => CausalityDirection::Backward,
            CausalityDirection::Backward => CausalityDirection::Forward,
            CausalityDirection::None => CausalityDirection::None,
        }
    }
}

// ---------------------------------------------------------------------------
// NarsTruthValue
// ---------------------------------------------------------------------------

/// NARS-style truth value with frequency and confidence.
///
/// Frequency (f): how often the proposition holds [0.0, 1.0].
/// Confidence (c): how much evidence supports the frequency [0.0, 1.0).
///
/// # Example
///
/// ```
/// use ndarray::hpc::causality::NarsTruthValue;
///
/// let tv = NarsTruthValue::new(0.9, 0.8);
/// assert!((tv.frequency - 0.9).abs() < 1e-6);
/// assert!((tv.confidence - 0.8).abs() < 1e-6);
/// assert!((tv.expectation() - 0.82).abs() < 1e-2);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct NarsTruthValue {
    /// Frequency: fraction of positive evidence.
    pub frequency: f32,
    /// Confidence: strength of evidence (0 = none, approaches 1 = infinite).
    pub confidence: f32,
}

impl NarsTruthValue {
    /// Create a truth value, clamping to valid ranges.
    ///
    /// Frequency is clamped to [0.0, 1.0].
    /// Confidence is clamped to [0.0, 1.0).
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 0.9999),
        }
    }

    /// Derive truth value from an awareness state.
    ///
    /// | State        | Frequency | Confidence |
    /// |-------------|-----------|------------|
    /// | Crystallized | 1.0       | 0.9        |
    /// | Tensioned    | 0.7       | 0.6        |
    /// | Uncertain    | 0.5       | 0.3        |
    /// | Noise        | 0.5       | 0.01       |
    pub fn from_awareness(state: AwarenessState) -> Self {
        match state {
            AwarenessState::Crystallized => Self::new(1.0, 0.9),
            AwarenessState::Tensioned => Self::new(0.7, 0.6),
            AwarenessState::Uncertain => Self::new(0.5, 0.3),
            AwarenessState::Noise => Self::new(0.5, 0.01),
        }
    }

    /// Total ignorance: frequency 0.5, confidence 0.
    pub fn ignorance() -> Self {
        Self { frequency: 0.5, confidence: 0.0 }
    }

    /// Expectation: `c * (f - 0.5) + 0.5`.
    ///
    /// Returns a value in [0.0, 1.0] that incorporates both frequency and
    /// confidence.
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }
}

impl Default for NarsTruthValue {
    fn default() -> Self {
        Self::ignorance()
    }
}

// ---------------------------------------------------------------------------
// CausalityDecomposition
// ---------------------------------------------------------------------------

/// Decomposition of causal structure across the three causality dimensions.
#[derive(Clone, Debug)]
pub struct CausalityDecomposition {
    /// Direction along warmth dimension.
    pub warmth_dir: CausalityDirection,
    /// Direction along social dimension.
    pub social_dir: CausalityDirection,
    /// Direction along sacredness dimension.
    pub sacredness_dir: CausalityDirection,
    /// NARS truth value for warmth.
    pub warmth_tv: NarsTruthValue,
    /// NARS truth value for social.
    pub social_tv: NarsTruthValue,
    /// NARS truth value for sacredness.
    pub sacredness_tv: NarsTruthValue,
    /// Overall causal strength: mean expectation across the three dimensions.
    pub overall_strength: f32,
}

/// Decompose the causal relationship between two qualia points.
///
/// Uses the three causality dimensions (warmth=4, social=6, sacredness=8)
/// to determine directional influence and truth values.
///
/// If a `SuperpositionState` is provided and has at least 9 dimensions, the
/// awareness state at each causality dimension is used to derive truth values.
/// Otherwise, defaults to ignorance.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::PackedQualia;
/// use ndarray::hpc::causality::causality_decompose;
///
/// let mut a = PackedQualia::zero();
/// let b = PackedQualia::zero();
/// a.resonance[4] = 10;  // warmth
/// a.resonance[6] = -5;  // social
/// a.resonance[8] = 3;   // sacredness
/// let dec = causality_decompose(&a, &b, None);
/// assert_eq!(dec.warmth_dir, ndarray::hpc::causality::CausalityDirection::Forward);
/// assert_eq!(dec.social_dir, ndarray::hpc::causality::CausalityDirection::Backward);
/// assert_eq!(dec.sacredness_dir, ndarray::hpc::causality::CausalityDirection::Forward);
/// ```
pub fn causality_decompose(
    a: &PackedQualia,
    b: &PackedQualia,
    superposition: Option<&SuperpositionState>,
) -> CausalityDecomposition {
    let warmth_dir = CausalityDirection::from_qualia(a, b, qualia_dim::WARMTH);
    let social_dir = CausalityDirection::from_qualia(a, b, qualia_dim::SOCIAL);
    let sacredness_dir = CausalityDirection::from_qualia(a, b, qualia_dim::SACREDNESS);

    let (warmth_tv, social_tv, sacredness_tv) = match superposition {
        Some(sp) if sp.n_dims >= 9 => (
            NarsTruthValue::from_awareness(sp.states[qualia_dim::WARMTH]),
            NarsTruthValue::from_awareness(sp.states[qualia_dim::SOCIAL]),
            NarsTruthValue::from_awareness(sp.states[qualia_dim::SACREDNESS]),
        ),
        _ => (
            NarsTruthValue::ignorance(),
            NarsTruthValue::ignorance(),
            NarsTruthValue::ignorance(),
        ),
    };

    let overall_strength = (warmth_tv.expectation()
        + social_tv.expectation()
        + sacredness_tv.expectation())
        / 3.0;

    CausalityDecomposition {
        warmth_dir,
        social_dir,
        sacredness_dir,
        warmth_tv,
        social_tv,
        sacredness_tv,
        overall_strength,
    }
}

/// Mask a BF16 value to keep only causality-relevant exponent bits.
///
/// Zeros out all bits except sign and the exponent bits at positions
/// corresponding to the three causality dimensions modulo 8.
///
/// Warmth=4 -> exponent bit 4, Social=6 -> exponent bit 6,
/// Sacredness=8 -> wraps to bit 0.
///
/// # Example
///
/// ```
/// use ndarray::hpc::causality::causality_mask_bf16;
///
/// let masked = causality_mask_bf16(0xFFFF);
/// // sign (bit 15) + exp bits 4,6,0 mapped into bits 14..7
/// // bit 15: sign = 1
/// // exp bit 0: position 7 -> set
/// // exp bit 4: position 11 -> set
/// // exp bit 6: position 13 -> set
/// // = 0x8000 | 0x2080 | 0x0800 = depends on exact mapping
/// assert_ne!(masked, 0xFFFF); // not all bits survive
/// ```
pub fn causality_mask_bf16(bf16: u16) -> u16 {
    // Keep sign bit (15)
    let sign = bf16 & 0x8000;

    // Exponent is bits 14..7. We keep only bits at offsets 0, 4, 6 within
    // the 8-bit exponent field.
    let exp_full = (bf16 >> 7) & 0xFF;
    let mask = (1u16 << 0) | (1u16 << 4) | (1u16 << 6); // bits 0, 4, 6
    let exp_masked = exp_full & mask;

    sign | (exp_masked << 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualia_dim_constants() {
        assert_eq!(qualia_dim::LUMINANCE, 0);
        assert_eq!(qualia_dim::WARMTH, 4);
        assert_eq!(qualia_dim::SOCIAL, 6);
        assert_eq!(qualia_dim::SACREDNESS, 8);
        assert_eq!(qualia_dim::SURPRISE, 15);
    }

    #[test]
    fn test_causality_dims() {
        assert_eq!(CAUSALITY_DIMS, [4, 6, 8]);
    }

    #[test]
    fn test_direction_from_qualia_forward() {
        let mut a = PackedQualia::zero();
        let b = PackedQualia::zero();
        a.resonance[4] = 10;
        assert_eq!(CausalityDirection::from_qualia(&a, &b, 4), CausalityDirection::Forward);
    }

    #[test]
    fn test_direction_from_qualia_backward() {
        let a = PackedQualia::zero();
        let mut b = PackedQualia::zero();
        b.resonance[6] = 5;
        assert_eq!(CausalityDirection::from_qualia(&a, &b, 6), CausalityDirection::Backward);
    }

    #[test]
    fn test_direction_from_qualia_none() {
        let a = PackedQualia::zero();
        let b = PackedQualia::zero();
        assert_eq!(CausalityDirection::from_qualia(&a, &b, 0), CausalityDirection::None);
    }

    #[test]
    fn test_direction_flip() {
        assert_eq!(CausalityDirection::Forward.flip(), CausalityDirection::Backward);
        assert_eq!(CausalityDirection::Backward.flip(), CausalityDirection::Forward);
        assert_eq!(CausalityDirection::None.flip(), CausalityDirection::None);
    }

    #[test]
    fn test_nars_truth_value_new() {
        let tv = NarsTruthValue::new(0.9, 0.8);
        assert!((tv.frequency - 0.9).abs() < 1e-6);
        assert!((tv.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_nars_truth_value_clamping() {
        let tv = NarsTruthValue::new(1.5, 1.0);
        assert!((tv.frequency - 1.0).abs() < 1e-6);
        assert!(tv.confidence < 1.0);
    }

    #[test]
    fn test_nars_truth_value_ignorance() {
        let tv = NarsTruthValue::ignorance();
        assert!((tv.frequency - 0.5).abs() < 1e-6);
        assert!((tv.confidence - 0.0).abs() < 1e-6);
        assert!((tv.expectation() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_nars_truth_value_expectation() {
        let tv = NarsTruthValue::new(1.0, 0.9);
        // c * (f - 0.5) + 0.5 = 0.9 * 0.5 + 0.5 = 0.95
        assert!((tv.expectation() - 0.95).abs() < 1e-2);
    }

    #[test]
    fn test_nars_from_awareness() {
        let tv = NarsTruthValue::from_awareness(AwarenessState::Crystallized);
        assert!((tv.frequency - 1.0).abs() < 1e-6);
        assert!((tv.confidence - 0.9).abs() < 1e-4);

        let tv = NarsTruthValue::from_awareness(AwarenessState::Noise);
        assert!((tv.frequency - 0.5).abs() < 1e-6);
        assert!(tv.confidence < 0.02);
    }

    #[test]
    fn test_causality_decompose_basic() {
        let mut a = PackedQualia::zero();
        let b = PackedQualia::zero();
        a.resonance[qualia_dim::WARMTH] = 10;
        a.resonance[qualia_dim::SOCIAL] = -5;
        a.resonance[qualia_dim::SACREDNESS] = 3;

        let dec = causality_decompose(&a, &b, None);
        assert_eq!(dec.warmth_dir, CausalityDirection::Forward);
        assert_eq!(dec.social_dir, CausalityDirection::Backward);
        assert_eq!(dec.sacredness_dir, CausalityDirection::Forward);
        // Without superposition, all truth values are ignorance
        assert!((dec.overall_strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_causality_decompose_with_superposition() {
        let a = PackedQualia::zero();
        let mut b = PackedQualia::zero();
        b.resonance[qualia_dim::WARMTH] = 1;

        // Build a superposition with at least 9 dims, all crystallized
        let sp = SuperpositionState {
            n_dims: 9,
            sign_consensus: vec![255; 9],
            exp_spread: vec![0; 9],
            mantissa_noise: vec![false; 9],
            states: vec![AwarenessState::Crystallized; 9],
            packed_states: vec![0; 3],
            crystallized_pct: 1.0,
            tensioned_pct: 0.0,
            uncertain_pct: 0.0,
            noise_pct: 0.0,
        };

        let dec = causality_decompose(&a, &b, Some(&sp));
        assert_eq!(dec.warmth_dir, CausalityDirection::Backward);
        // Crystallized => freq=1.0, conf=0.9, exp=0.95
        assert!(dec.overall_strength > 0.9);
    }

    #[test]
    fn test_causality_mask_bf16_zero() {
        assert_eq!(causality_mask_bf16(0x0000), 0x0000);
    }

    #[test]
    fn test_causality_mask_bf16_sign_preserved() {
        let masked = causality_mask_bf16(0x8000);
        assert_eq!(masked & 0x8000, 0x8000);
    }

    #[test]
    fn test_causality_mask_bf16_strips_mantissa() {
        // 0x007F = mantissa all 1s, rest 0 => masked should be 0
        let masked = causality_mask_bf16(0x007F);
        assert_eq!(masked, 0x0000);
    }

    #[test]
    fn test_causality_mask_bf16_keeps_selected_exp_bits() {
        // Set all exponent bits: bits 14..7 = 0xFF << 7 = 0x7F80
        let full_exp = 0x7F80u16;
        let masked = causality_mask_bf16(full_exp);
        // Should keep only exp bits 0, 4, 6 => binary 0b01010001 = 0x51
        // Shifted back: 0x51 << 7 = 0x2880
        assert_eq!(masked, 0x2880);
    }
}
