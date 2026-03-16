//! BF16-structured Hamming distance and truth encoding.
//!
//! XOR + per-field weighted popcount:
//!   sign (bit 15): weight 256
//!   exponent (bits 14-7): weight 16 per flipped bit
//!   mantissa (bits 6-0): weight 1 per flipped bit
//!
//! Ported from rustynum-core/bf16_hamming.rs — core types only (no SIMD dispatch).

/// BF16 field weights.
///
/// Controls how much each BF16 field contributes to the Hamming distance.
/// Default: sign=256, exponent=16, mantissa=1.
///
/// # Invariant
///
/// `sign + 8 * exponent + 7 * mantissa` must fit in a `u16` (<=65535).
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::BF16Weights;
///
/// let w = BF16Weights::default();
/// assert_eq!(w.sign, 256);
/// assert_eq!(w.exponent, 16);
/// assert_eq!(w.mantissa, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BF16Weights {
    pub sign: u16,
    pub exponent: u16,
    pub mantissa: u16,
}

impl BF16Weights {
    /// Create weights with overflow check.
    ///
    /// # Panics
    ///
    /// Panics if `sign + 8 * exponent + 7 * mantissa > 65535`.
    pub fn new(sign: u16, exponent: u16, mantissa: u16) -> Self {
        let max_per_elem = sign as u32 + 8 * exponent as u32 + 7 * mantissa as u32;
        assert!(
            max_per_elem <= 65535,
            "BF16Weights overflow: sign({}) + 8*exp({}) + 7*man({}) = {} > 65535",
            sign, exponent, mantissa, max_per_elem
        );
        Self { sign, exponent, mantissa }
    }
}

impl Default for BF16Weights {
    fn default() -> Self {
        Self { sign: 256, exponent: 16, mantissa: 1 }
    }
}

/// Awareness state per BF16 dimension.
///
/// Each dimension in a superposition snapshot is classified into one of
/// four states based on how its sign, exponent, and mantissa fields behave
/// across a batch of observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AwarenessState {
    /// All observations agree — stable, low-entropy dimension.
    Crystallized,
    /// Sign consensus is high but exponent spread is significant.
    Tensioned,
    /// Moderate disagreement — neither stable nor noise.
    Uncertain,
    /// High mantissa noise and exponent spread — effectively random.
    Noise,
}

impl AwarenessState {
    /// Pack into 2 bits: Crystallized=0, Tensioned=1, Uncertain=2, Noise=3.
    pub fn to_bits(self) -> u8 {
        match self {
            AwarenessState::Crystallized => 0,
            AwarenessState::Tensioned => 1,
            AwarenessState::Uncertain => 2,
            AwarenessState::Noise => 3,
        }
    }

    /// Unpack from 2 bits.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => AwarenessState::Crystallized,
            1 => AwarenessState::Tensioned,
            2 => AwarenessState::Uncertain,
            _ => AwarenessState::Noise,
        }
    }
}

/// Superposition state across all dimensions.
///
/// Summarizes how a batch of BF16-encoded observations relate to each other
/// across `n_dims` dimensions.
#[derive(Clone, Debug)]
pub struct SuperpositionState {
    /// Number of BF16 dimensions.
    pub n_dims: usize,
    /// Per-dimension sign consensus (0 = total disagreement, 255 = total agreement).
    pub sign_consensus: Vec<u8>,
    /// Per-dimension exponent spread (0 = identical, 255 = maximally spread).
    pub exp_spread: Vec<u8>,
    /// Per-dimension mantissa noise flag.
    pub mantissa_noise: Vec<bool>,
    /// Per-dimension awareness classification.
    pub states: Vec<AwarenessState>,
    /// Packed states: 4 states per byte (2 bits each), little-endian within byte.
    pub packed_states: Vec<u8>,
    /// Fraction of dimensions that are crystallized.
    pub crystallized_pct: f32,
    /// Fraction of dimensions that are tensioned.
    pub tensioned_pct: f32,
    /// Fraction of dimensions that are uncertain.
    pub uncertain_pct: f32,
    /// Fraction of dimensions that are noise.
    pub noise_pct: f32,
}

/// Awareness thresholds for classification.
///
/// Controls the boundaries between awareness states.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::AwarenessThresholds;
///
/// let t = AwarenessThresholds::default();
/// assert_eq!(t.exp_spread_limit, 2);
/// assert_eq!(t.noise_mantissa_bits, 5);
/// ```
#[derive(Clone, Debug)]
pub struct AwarenessThresholds {
    /// Maximum exponent spread for crystallized/tensioned (inclusive).
    pub exp_spread_limit: u8,
    /// Minimum mantissa popcount to be considered noise.
    pub noise_mantissa_bits: u8,
}

impl Default for AwarenessThresholds {
    fn default() -> Self {
        Self { exp_spread_limit: 2, noise_mantissa_bits: 5 }
    }
}

/// 16-dimensional packed qualia point.
///
/// Stores a 16-element `i8` resonance vector alongside a BF16-encoded scalar.
/// This is the fundamental unit for qualia-space computations.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::PackedQualia;
///
/// let pq = PackedQualia::new([1, -1, 2, -2, 3, -3, 4, -4,
///                             5, -5, 6, -6, 7, -7, 8, -8], 1.0);
/// let rt = pq.scalar_f32();
/// // BF16 truncation: 1.0 round-trips exactly.
/// assert!((rt - 1.0).abs() < 1e-6);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedQualia {
    /// 16-dimensional signed resonance vector.
    pub resonance: [i8; 16],
    /// BF16 scalar stored as 2 little-endian bytes.
    pub scalar: [u8; 2],
}

impl PackedQualia {
    /// Create from resonance array and f32 scalar (truncated to BF16).
    pub fn new(resonance: [i8; 16], scalar_f32: f32) -> Self {
        let bits = scalar_f32.to_bits();
        let bf16 = ((bits >> 16) & 0xFFFF) as u16;
        Self {
            resonance,
            scalar: bf16.to_le_bytes(),
        }
    }

    /// Zero-initialized qualia point.
    pub fn zero() -> Self {
        Self { resonance: [0i8; 16], scalar: [0u8; 2] }
    }

    /// Decode the BF16 scalar back to f32.
    pub fn scalar_f32(&self) -> f32 {
        let bf16 = u16::from_le_bytes(self.scalar);
        let bits = (bf16 as u32) << 16;
        f32::from_bits(bits)
    }
}

/// Compute BF16-structured Hamming distance (scalar fallback).
///
/// Interprets `a` and `b` as packed BF16 values (2 bytes each, little-endian)
/// and computes the weighted Hamming distance across all BF16 elements.
///
/// # Arguments
///
/// * `a` — first byte slice (length must be even and equal to `b.len()`)
/// * `b` — second byte slice
/// * `weights` — per-field weights
///
/// # Returns
///
/// Total weighted Hamming distance as `u64`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::{bf16_hamming_scalar, BF16Weights};
///
/// let a = [0x00u8, 0x00];
/// let b = [0x00u8, 0x00];
/// let w = BF16Weights::default();
/// assert_eq!(bf16_hamming_scalar(&a, &b, &w), 0);
/// ```
pub fn bf16_hamming_scalar(a: &[u8], b: &[u8], weights: &BF16Weights) -> u64 {
    assert_eq!(a.len(), b.len(), "bf16_hamming_scalar: length mismatch");
    assert!(a.len() % 2 == 0, "bf16_hamming_scalar: length must be even (BF16 = 2 bytes)");

    let n_elems = a.len() / 2;
    let mut total: u64 = 0;

    for i in 0..n_elems {
        let off = i * 2;
        let va = u16::from_le_bytes([a[off], a[off + 1]]);
        let vb = u16::from_le_bytes([b[off], b[off + 1]]);
        let xor = va ^ vb;

        // sign: bit 15
        let sign_flip = (xor >> 15) & 1;
        // exponent: bits 14..7 (8 bits)
        let exp_bits = (xor >> 7) & 0xFF;
        let exp_pop = exp_bits.count_ones() as u16;
        // mantissa: bits 6..0 (7 bits)
        let man_bits = xor & 0x7F;
        let man_pop = man_bits.count_ones() as u16;

        let dist = sign_flip * weights.sign
            + exp_pop * weights.exponent
            + man_pop * weights.mantissa;
        total += dist as u64;
    }

    total
}

/// Classify awareness state for each BF16 dimension by comparing two byte vectors.
///
/// For each BF16 element, extracts sign, exponent, and mantissa XOR fields
/// and classifies the dimension into one of four awareness states.
///
/// # Arguments
///
/// * `a` — first byte slice (BF16-packed, little-endian)
/// * `b` — second byte slice (same length as `a`)
/// * `n_dims` — number of BF16 dimensions to classify (must be <= a.len()/2)
/// * `thresholds` — classification thresholds
///
/// # Returns
///
/// A [`SuperpositionState`] summarizing all dimensions.
pub fn awareness_classify(
    a: &[u8],
    b: &[u8],
    n_dims: usize,
    thresholds: &AwarenessThresholds,
) -> SuperpositionState {
    assert_eq!(a.len(), b.len(), "awareness_classify: length mismatch");
    assert!(a.len() >= n_dims * 2, "awareness_classify: not enough bytes for n_dims");

    let mut sign_consensus = Vec::with_capacity(n_dims);
    let mut exp_spread = Vec::with_capacity(n_dims);
    let mut mantissa_noise = Vec::with_capacity(n_dims);
    let mut states = Vec::with_capacity(n_dims);

    let mut counts = [0u32; 4]; // Crystallized, Tensioned, Uncertain, Noise

    for i in 0..n_dims {
        let off = i * 2;
        let va = u16::from_le_bytes([a[off], a[off + 1]]);
        let vb = u16::from_le_bytes([b[off], b[off + 1]]);
        let xor = va ^ vb;

        // sign: bit 15 — consensus is 255 if same, 0 if different
        let sign_flip = (xor >> 15) & 1;
        let sc = if sign_flip == 0 { 255u8 } else { 0u8 };
        sign_consensus.push(sc);

        // exponent: bits 14..7 — spread is popcount of flipped exponent bits
        let exp_bits = (xor >> 7) & 0xFF;
        let es = exp_bits.count_ones() as u8;
        exp_spread.push(es);

        // mantissa: bits 6..0 — noise if popcount >= threshold
        let man_bits = xor & 0x7F;
        let man_pop = man_bits.count_ones() as u8;
        let is_noisy = man_pop >= thresholds.noise_mantissa_bits;
        mantissa_noise.push(is_noisy);

        // Classify
        let state = if sc == 255 && es <= thresholds.exp_spread_limit && !is_noisy {
            AwarenessState::Crystallized
        } else if sc == 255 && es > thresholds.exp_spread_limit {
            AwarenessState::Tensioned
        } else if is_noisy && es > thresholds.exp_spread_limit {
            AwarenessState::Noise
        } else {
            AwarenessState::Uncertain
        };

        counts[state.to_bits() as usize] += 1;
        states.push(state);
    }

    // Pack states: 4 per byte
    let packed_len = (n_dims + 3) / 4;
    let mut packed_states = vec![0u8; packed_len];
    for (i, st) in states.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        packed_states[byte_idx] |= st.to_bits() << bit_off;
    }

    let nd = n_dims as f32;
    SuperpositionState {
        n_dims,
        sign_consensus,
        exp_spread,
        mantissa_noise,
        states,
        packed_states,
        crystallized_pct: counts[0] as f32 / nd,
        tensioned_pct: counts[1] as f32 / nd,
        uncertain_pct: counts[2] as f32 / nd,
        noise_pct: counts[3] as f32 / nd,
    }
}

/// Pack 7 projection bands + finest distance + causality direction into a BF16 truth value.
///
/// Layout (16 bits):
/// - sign (bit 15) = causality direction (0=Forward, 1=Backward)
/// - exponent (bits 14-8) = 7 bits from 7 projection bands (Foveal/Near -> 1, else -> 0)
/// - mantissa (bits 7-1) = 7 bits of finest Hamming distance (normalized to 0..127)
/// - bit 0 = reserved (0)
///
/// The exponent encodes which projections are "close" (Foveal or Near),
/// giving a 7-bit fingerprint of the relationship shape across S/P/O masks.
/// The mantissa captures the finest-grained distance for ranking within a band.
///
/// `CausalityDirection::None` is treated as Forward (sign = 0).
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::bf16_from_projections;
/// use ndarray::hpc::cascade::Band;
/// use ndarray::hpc::causality::CausalityDirection;
///
/// let bands = [Band::Foveal; 7];
/// let packed = bf16_from_projections(&bands, 0, 1000, CausalityDirection::Forward);
/// assert_ne!(packed, 0);
/// ```
pub fn bf16_from_projections(
    bands: &[super::cascade::Band; 7],
    finest_distance: u32,
    finest_max: u32,
    direction: super::causality::CausalityDirection,
) -> u16 {
    use super::cascade::Band;
    use super::causality::CausalityDirection;

    let sign: u16 = match direction {
        CausalityDirection::Forward | CausalityDirection::None => 0,
        CausalityDirection::Backward => 1,
    };

    let mut exponent: u16 = 0;
    for (i, band) in bands.iter().enumerate() {
        match band {
            Band::Foveal | Band::Near => {
                exponent |= 1 << i;
            }
            _ => {}
        }
    }

    // Normalize finest distance to 7 bits (0..127)
    let mantissa: u16 = if finest_max > 0 {
        ((finest_distance as u64 * 127) / finest_max as u64).min(127) as u16
    } else {
        0
    };

    (sign << 15) | ((exponent & 0x7F) << 8) | ((mantissa & 0x7F) << 1)
}

/// Unpack a BF16 truth value assembled by [`bf16_from_projections`].
///
/// Returns `(direction, exponent_bits, mantissa_7bit)`.
///
/// Sign bit 0 maps to `Forward`, sign bit 1 maps to `Backward`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::{bf16_from_projections, bf16_unpack_projections};
/// use ndarray::hpc::cascade::Band;
/// use ndarray::hpc::causality::CausalityDirection;
///
/// let bands = [Band::Foveal; 7];
/// let packed = bf16_from_projections(&bands, 0, 1000, CausalityDirection::Forward);
/// let (dir, exp, man) = bf16_unpack_projections(packed);
/// assert_eq!(dir, CausalityDirection::Forward);
/// assert_eq!(exp, 0x7F);
/// assert_eq!(man, 0);
/// ```
pub fn bf16_unpack_projections(
    packed: u16,
) -> (super::causality::CausalityDirection, u8, u8) {
    use super::causality::CausalityDirection;

    let direction = if packed & 0x8000 != 0 {
        CausalityDirection::Backward
    } else {
        CausalityDirection::Forward
    };
    let exponent = ((packed >> 8) & 0x7F) as u8;
    let mantissa = ((packed >> 1) & 0x7F) as u8;
    (direction, exponent, mantissa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_default() {
        let w = BF16Weights::default();
        assert_eq!(w.sign, 256);
        assert_eq!(w.exponent, 16);
        assert_eq!(w.mantissa, 1);
    }

    #[test]
    fn test_weights_valid_new() {
        // Max safe: 256 + 8*16 + 7*1 = 391
        let w = BF16Weights::new(256, 16, 1);
        assert_eq!(w, BF16Weights::default());
    }

    #[test]
    #[should_panic(expected = "BF16Weights overflow")]
    fn test_weights_overflow() {
        // 60000 + 8*1000 + 7*1 = 68007 > 65535
        BF16Weights::new(60000, 1000, 1);
    }

    #[test]
    fn test_hamming_identical() {
        let data = [0x3F, 0x80, 0x40, 0x00]; // two BF16 values
        let w = BF16Weights::default();
        assert_eq!(bf16_hamming_scalar(&data, &data, &w), 0);
    }

    #[test]
    fn test_hamming_sign_flip() {
        // 0x0000 vs 0x8000 — only sign bit flipped
        let a = [0x00u8, 0x00];
        let b = [0x00u8, 0x80];
        let w = BF16Weights::default();
        assert_eq!(bf16_hamming_scalar(&a, &b, &w), 256);
    }

    #[test]
    fn test_hamming_exponent_single_bit() {
        // Flip one exponent bit: bit 7 (0x0080)
        let a = [0x00u8, 0x00];
        let b = [0x80u8, 0x00]; // bit 7 set in LE
        let w = BF16Weights::default();
        assert_eq!(bf16_hamming_scalar(&a, &b, &w), 16);
    }

    #[test]
    fn test_hamming_mantissa_single_bit() {
        // Flip one mantissa bit: bit 0 (0x0001)
        let a = [0x00u8, 0x00];
        let b = [0x01u8, 0x00];
        let w = BF16Weights::default();
        assert_eq!(bf16_hamming_scalar(&a, &b, &w), 1);
    }

    #[test]
    fn test_packed_qualia_zero() {
        let pq = PackedQualia::zero();
        assert_eq!(pq.resonance, [0i8; 16]);
        assert_eq!(pq.scalar, [0u8; 2]);
        assert_eq!(pq.scalar_f32(), 0.0);
    }

    #[test]
    fn test_packed_qualia_roundtrip() {
        let res = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
        let pq = PackedQualia::new(res, 1.0);
        assert_eq!(pq.resonance, res);
        assert!((pq.scalar_f32() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_packed_qualia_negative_scalar() {
        let pq = PackedQualia::new([0i8; 16], -3.5);
        let rt = pq.scalar_f32();
        assert!((rt - (-3.5)).abs() < 0.1, "got {}", rt);
    }

    #[test]
    fn test_awareness_state_roundtrip() {
        for state in &[
            AwarenessState::Crystallized,
            AwarenessState::Tensioned,
            AwarenessState::Uncertain,
            AwarenessState::Noise,
        ] {
            assert_eq!(AwarenessState::from_bits(state.to_bits()), *state);
        }
    }

    #[test]
    fn test_awareness_classify_identical() {
        let data = vec![0x3F, 0x80, 0x40, 0x00, 0x3E, 0x00, 0x41, 0x20];
        let t = AwarenessThresholds::default();
        let s = awareness_classify(&data, &data, 4, &t);
        assert_eq!(s.n_dims, 4);
        assert!((s.crystallized_pct - 1.0).abs() < 1e-6);
        assert!(s.states.iter().all(|st| *st == AwarenessState::Crystallized));
    }

    #[test]
    fn test_awareness_classify_all_noise() {
        // All bits flipped — sign differs, exponent fully flipped, mantissa fully flipped
        let a = vec![0x00u8, 0x00, 0x00, 0x00];
        let b = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let t = AwarenessThresholds::default();
        let s = awareness_classify(&a, &b, 2, &t);
        assert_eq!(s.n_dims, 2);
        // sign differs, exp_spread=8, mantissa noise=7 >= 5 => Noise
        assert!(s.states.iter().all(|st| *st == AwarenessState::Noise));
        assert!((s.noise_pct - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_awareness_classify_tensioned() {
        // Same sign, large exponent spread, low mantissa noise
        // a = 0x0000, b = 0x7F00 => XOR = 0x7F00
        // sign: 0 (same), exp bits 14..7 of 0x7F00 = 0xFE => popcount=7, man = 0
        let a = vec![0x00u8, 0x00];
        let b = vec![0x00u8, 0x7F]; // LE: 0x7F00
        let t = AwarenessThresholds::default();
        let s = awareness_classify(&a, &b, 1, &t);
        assert_eq!(s.states[0], AwarenessState::Tensioned);
    }

    #[test]
    fn test_bf16_from_projections_all_foveal_forward() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Foveal; 7];
        let packed = bf16_from_projections(&bands, 0, 1000, CausalityDirection::Forward);
        let (dir, exp, man) = bf16_unpack_projections(packed);
        assert_eq!(dir, CausalityDirection::Forward);
        assert_eq!(exp, 0x7F); // all 7 bits set
        assert_eq!(man, 0); // distance = 0
    }

    #[test]
    fn test_bf16_from_projections_all_reject_backward() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Reject; 7];
        let packed = bf16_from_projections(&bands, 500, 1000, CausalityDirection::Backward);
        let (dir, exp, man) = bf16_unpack_projections(packed);
        assert_eq!(dir, CausalityDirection::Backward);
        assert_eq!(exp, 0); // no close projections
        // mantissa: 500/1000 * 127 = 63
        assert_eq!(man, 63);
    }

    #[test]
    fn test_bf16_from_projections_mixed_bands() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [
            Band::Foveal, // bit 0
            Band::Near,   // bit 1
            Band::Good,   // bit 2 = 0
            Band::Weak,   // bit 3 = 0
            Band::Foveal, // bit 4
            Band::Reject, // bit 5 = 0
            Band::Near,   // bit 6
        ];
        let packed = bf16_from_projections(&bands, 100, 1000, CausalityDirection::Forward);
        let (dir, exp, man) = bf16_unpack_projections(packed);
        assert_eq!(dir, CausalityDirection::Forward);
        // bits 0,1,4,6 set = 0b1010011 = 0x53
        assert_eq!(exp, 0b1010011);
        // mantissa: 100/1000 * 127 = 12
        assert_eq!(man, 12);
    }

    #[test]
    fn test_bf16_from_projections_roundtrip_sign() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Good; 7];
        for dir in [CausalityDirection::Forward, CausalityDirection::Backward] {
            let packed = bf16_from_projections(&bands, 50, 100, dir);
            let (unpacked_dir, _, _) = bf16_unpack_projections(packed);
            assert_eq!(unpacked_dir, dir);
        }
    }

    #[test]
    fn test_bf16_from_projections_max_distance() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Foveal; 7];
        let packed = bf16_from_projections(&bands, 1000, 1000, CausalityDirection::Forward);
        let (_, _, man) = bf16_unpack_projections(packed);
        assert_eq!(man, 127); // saturates at 127
    }

    #[test]
    fn test_bf16_from_projections_zero_max() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Foveal; 7];
        let packed = bf16_from_projections(&bands, 500, 0, CausalityDirection::Forward);
        let (_, _, man) = bf16_unpack_projections(packed);
        assert_eq!(man, 0); // zero max -> mantissa 0
    }

    #[test]
    fn test_bf16_from_projections_none_direction() {
        use super::super::cascade::Band;
        use super::super::causality::CausalityDirection;

        let bands = [Band::Good; 7];
        let packed = bf16_from_projections(&bands, 50, 100, CausalityDirection::None);
        let (dir, _, _) = bf16_unpack_projections(packed);
        // None maps to Forward (sign=0)
        assert_eq!(dir, CausalityDirection::Forward);
    }

    #[test]
    fn test_packed_states_encoding() {
        let a = vec![0x00u8; 8];
        let b = vec![0x00u8; 8]; // identical => all crystallized
        let t = AwarenessThresholds::default();
        let s = awareness_classify(&a, &b, 4, &t);
        // 4 crystallized states packed: [0b00_00_00_00] = [0]
        assert_eq!(s.packed_states.len(), 1);
        assert_eq!(s.packed_states[0], 0);
    }
}
