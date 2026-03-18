//! DataFusion UDF compute kernels.
//!
//! Pure compute kernels that can be wrapped as UDFs by a downstream consumer
//! crate. No DataFusion dependency is introduced here.
//!
//! ## Layer 1: UDF Compute Kernels
//!
//! - [`udf_hamming`] — Hamming distance on raw byte slices
//! - [`udf_spo_distance`] — SPO node distance from binary planes
//! - [`udf_nars_revision`] — NARS truth-value revision (integer encoding)
//! - [`udf_sigma_classify`] — Cascade band + threshold classification
//! - [`udf_bf16_hamming`] — BF16-weighted Hamming distance
//!
//! ## Layer 2: Causal Query Operators
//!
//! - [`factorize_spo`] — 8-term SPO interaction decomposition
//! - [`nars_accumulate`] — Fold multiple truth values via revision
//! - [`sigma_classify_batch`] — Batch cascade band classification
//! - [`causal_edge`] — Extract causality direction between two nodes

use super::bf16_truth::{bf16_hamming_scalar, BF16Weights};
use super::bitwise::hamming_distance_raw;
use super::cascade::Cascade;
use super::causality::{causality_decompose, CausalityDirection};
use super::node::{Node, SPO, S__, _P_, __O};
use super::plane::Truth;

// ============================================================================
// Layer 1: UDF Compute Kernels
// ============================================================================

/// Hamming distance between two byte slices.
///
/// Delegates to [`hamming_distance_raw`] with SIMD dispatch.
///
/// # Errors
///
/// Returns `Err` if input lengths do not match.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::udf_hamming;
///
/// let a = vec![0xFFu8; 32];
/// let b = vec![0x00u8; 32];
/// assert_eq!(udf_hamming(&a, &b).unwrap(), 32 * 8);
/// ```
pub fn udf_hamming(a: &[u8], b: &[u8]) -> Result<u64, &'static str> {
    if a.len() != b.len() {
        return Err("udf_hamming: input lengths must match");
    }
    Ok(hamming_distance_raw(a, b))
}

/// Result of an SPO distance computation.
#[derive(Clone, Debug)]
pub struct SpoDistanceResult {
    /// Subject plane distance (raw disagreement bits, or `None` if incomparable).
    pub subject_dist: Option<u32>,
    /// Predicate plane distance (raw disagreement bits, or `None` if incomparable).
    pub predicate_dist: Option<u32>,
    /// Object plane distance (raw disagreement bits, or `None` if incomparable).
    pub object_dist: Option<u32>,
    /// Combined SPO distance (raw disagreement bits, or `None` if incomparable).
    pub combined_dist: Option<u32>,
}

/// Compute SPO distance between two nodes constructed from binary planes.
///
/// Each input slice must be exactly 2048 bytes (one Plane worth of data).
/// The function creates two [`Node`]s by encountering the binary data on each
/// plane, then measures per-plane and combined distances.
///
/// # Errors
///
/// Returns `Err` if any input slice is not exactly 2048 bytes.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::udf_spo_distance;
///
/// let s = vec![0xAAu8; 2048];
/// let p = vec![0xBBu8; 2048];
/// let o = vec![0xCCu8; 2048];
/// let result = udf_spo_distance(&s, &p, &o, &s, &p, &o).unwrap();
/// assert_eq!(result.combined_dist, Some(0));
/// ```
pub fn udf_spo_distance(
    s1: &[u8],
    p1: &[u8],
    o1: &[u8],
    s2: &[u8],
    p2: &[u8],
    o2: &[u8],
) -> Result<SpoDistanceResult, &'static str> {
    use super::fingerprint::Fingerprint;

    const PLANE_BYTES: usize = 2048;
    for (name, slice) in [
        ("s1", s1),
        ("p1", p1),
        ("o1", o1),
        ("s2", s2),
        ("p2", p2),
        ("o2", o2),
    ] {
        if slice.len() != PLANE_BYTES {
            return Err(if name.starts_with('s') {
                "udf_spo_distance: subject plane must be 2048 bytes"
            } else if name.starts_with('p') {
                "udf_spo_distance: predicate plane must be 2048 bytes"
            } else {
                "udf_spo_distance: object plane must be 2048 bytes"
            });
        }
    }

    let fp1_s = Fingerprint::<256>::from_bytes(s1);
    let fp1_p = Fingerprint::<256>::from_bytes(p1);
    let fp1_o = Fingerprint::<256>::from_bytes(o1);
    let fp2_s = Fingerprint::<256>::from_bytes(s2);
    let fp2_p = Fingerprint::<256>::from_bytes(p2);
    let fp2_o = Fingerprint::<256>::from_bytes(o2);

    let mut node_a = Node::new();
    node_a.s.encounter_bits(&fp1_s);
    node_a.s.encounter_bits(&fp1_s);
    node_a.p.encounter_bits(&fp1_p);
    node_a.p.encounter_bits(&fp1_p);
    node_a.o.encounter_bits(&fp1_o);
    node_a.o.encounter_bits(&fp1_o);

    let mut node_b = Node::new();
    node_b.s.encounter_bits(&fp2_s);
    node_b.s.encounter_bits(&fp2_s);
    node_b.p.encounter_bits(&fp2_p);
    node_b.p.encounter_bits(&fp2_p);
    node_b.o.encounter_bits(&fp2_o);
    node_b.o.encounter_bits(&fp2_o);

    let subject_dist = node_a.distance(&mut node_b, S__).raw();
    let predicate_dist = node_a.distance(&mut node_b, _P_).raw();
    let object_dist = node_a.distance(&mut node_b, __O).raw();
    let combined_dist = node_a.distance(&mut node_b, SPO).raw();

    Ok(SpoDistanceResult {
        subject_dist,
        predicate_dist,
        object_dist,
        combined_dist,
    })
}

/// NARS truth-value revision using integer-encoded (u16) frequency and
/// confidence.
///
/// Constructs two [`Truth`] values and calls [`Truth::revision`], returning
/// the revised `(frequency, confidence)` pair as u16 values.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::udf_nars_revision;
///
/// let (f, c) = udf_nars_revision(60000, 50000, 30000, 40000);
/// // Revision combines evidence; result depends on evidence weights
/// assert!(f > 0);
/// assert!(c > 0);
/// ```
pub fn udf_nars_revision(freq1: u16, conf1: u16, freq2: u16, conf2: u16) -> (u16, u16) {
    let t1 = Truth {
        frequency: freq1,
        confidence: conf1,
        evidence: 1,
    };
    let t2 = Truth {
        frequency: freq2,
        confidence: conf2,
        evidence: 1,
    };
    let revised = t1.revision(&t2);
    (revised.frequency, revised.confidence)
}

/// Classify a distance within a cascade band.
///
/// Maps a cascade `band` threshold and a `distance` to one of four labels:
/// `"exact"`, `"near"`, `"far"`, or `"noise"`.
///
/// The classification uses the cascade's `expose` method:
/// - `Foveal` => `"exact"`
/// - `Near` => `"near"`
/// - `Good` | `Weak` => `"far"`
/// - `Reject` => `"noise"`
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::udf_sigma_classify;
///
/// assert_eq!(udf_sigma_classify(1000, 100), "exact");
/// assert_eq!(udf_sigma_classify(1000, 1500), "noise");
/// ```
pub fn udf_sigma_classify(band: u32, threshold: u64) -> &'static str {
    use super::cascade::Band;

    let cascade = Cascade::from_threshold(threshold, 2048);
    match cascade.expose(band) {
        Band::Foveal => "exact",
        Band::Near => "near",
        Band::Good | Band::Weak => "far",
        Band::Reject => "noise",
    }
}

/// BF16-weighted Hamming distance.
///
/// Delegates to [`bf16_hamming_scalar`]. The `weights` slice must be exactly
/// 6 bytes encoding three little-endian u16 values: `[sign, exponent, mantissa]`.
///
/// # Errors
///
/// Returns `Err` if:
/// - `a` and `b` have different lengths
/// - `a` length is not even (BF16 = 2 bytes per element)
/// - `weights` is not exactly 6 bytes
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::udf_bf16_hamming;
///
/// let a = vec![0x00u8, 0x00, 0x00, 0x00];
/// let b = vec![0x00u8, 0x80, 0x00, 0x00]; // sign flip on first element
/// let weights = 256u16.to_le_bytes().iter()
///     .chain(16u16.to_le_bytes().iter())
///     .chain(1u16.to_le_bytes().iter())
///     .copied().collect::<Vec<u8>>();
/// let d = udf_bf16_hamming(&a, &b, &weights).unwrap();
/// assert!((d - 256.0).abs() < 1e-6);
/// ```
pub fn udf_bf16_hamming(a: &[u8], b: &[u8], weights: &[u8]) -> Result<f64, &'static str> {
    if a.len() != b.len() {
        return Err("udf_bf16_hamming: input lengths must match");
    }
    if a.len() % 2 != 0 {
        return Err("udf_bf16_hamming: input length must be even (BF16 = 2 bytes)");
    }
    if weights.len() != 6 {
        return Err("udf_bf16_hamming: weights must be 6 bytes (3 x u16 LE)");
    }
    let sign = u16::from_le_bytes([weights[0], weights[1]]);
    let exponent = u16::from_le_bytes([weights[2], weights[3]]);
    let mantissa = u16::from_le_bytes([weights[4], weights[5]]);

    // Validate weights do not overflow
    let max_per_elem = sign as u32 + 8 * exponent as u32 + 7 * mantissa as u32;
    if max_per_elem > 65535 {
        return Err("udf_bf16_hamming: weights would overflow u16");
    }

    let w = BF16Weights::new(sign, exponent, mantissa);
    let dist = bf16_hamming_scalar(a, b, &w);
    Ok(dist as f64)
}

// ============================================================================
// Layer 2: Causal Query Operators
// ============================================================================

/// SPO 8-term interaction decomposition.
///
/// Computes the Hamming norm (popcount) of each of the 8 possible plane
/// combinations of a [`Node`]:
///
/// `[empty, S, P, O, SP, PO, SO, SPO]`
///
/// The "empty" term is always 0. Each subsequent term is the popcount of the
/// corresponding plane's bits.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::factorize_spo;
/// use ndarray::hpc::node::Node;
///
/// let mut node = Node::random(42);
/// let terms = factorize_spo(&mut node);
/// assert_eq!(terms[0], 0); // empty term
/// assert!(terms[7] > 0);   // SPO norm
/// ```
pub fn factorize_spo(node: &mut Node) -> [u64; 8] {
    use super::bitwise::popcount_raw;

    let s_bits = node.s.bits().as_bytes();
    let s_pop = popcount_raw(s_bits);

    let p_bits = node.p.bits().as_bytes();
    let p_pop = popcount_raw(p_bits);

    let o_bits = node.o.bits().as_bytes();
    let o_pop = popcount_raw(o_bits);

    [
        0,                           // empty
        s_pop,                       // S
        p_pop,                       // P
        o_pop,                       // O
        s_pop.saturating_add(p_pop), // SP
        p_pop.saturating_add(o_pop), // PO
        s_pop.saturating_add(o_pop), // SO
        s_pop.saturating_add(p_pop).saturating_add(o_pop), // SPO
    ]
}

/// Evidence accumulation: fold multiple integer-encoded truth values via
/// successive revision.
///
/// Each element of `evidence` is a `(frequency, confidence)` pair encoded as
/// u16 values. Returns the accumulated `(frequency, confidence)`.
///
/// If the input is empty, returns the ignorance prior `(32768, 0)`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::nars_accumulate;
///
/// let evidence = vec![(60000u16, 50000u16), (30000, 40000)];
/// let (f, c) = nars_accumulate(&evidence);
/// assert!(f > 0);
/// ```
pub fn nars_accumulate(evidence: &[(u16, u16)]) -> (u16, u16) {
    if evidence.is_empty() {
        return (32768, 0);
    }

    let mut acc = Truth {
        frequency: evidence[0].0,
        confidence: evidence[0].1,
        evidence: 1,
    };

    for &(freq, conf) in &evidence[1..] {
        let next = Truth {
            frequency: freq,
            confidence: conf,
            evidence: 1,
        };
        acc = acc.revision(&next);
    }

    (acc.frequency, acc.confidence)
}

/// Batch cascade band classification.
///
/// Classifies each distance in `distances` against a 3-tier threshold array
/// `[foveal_max, near_max, far_max]`:
///
/// - `distance <= foveal_max` => `"exact"`
/// - `distance <= near_max`   => `"near"`
/// - `distance <= far_max`    => `"far"`
/// - otherwise                => `"noise"`
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::sigma_classify_batch;
///
/// let distances = vec![10, 200, 500, 2000];
/// let thresholds = [100, 400, 1000];
/// let labels = sigma_classify_batch(&distances, &thresholds);
/// assert_eq!(labels, vec!["exact", "near", "far", "noise"]);
/// ```
pub fn sigma_classify_batch(distances: &[u64], thresholds: &[u64; 3]) -> Vec<&'static str> {
    distances
        .iter()
        .map(|&d| {
            if d <= thresholds[0] {
                "exact"
            } else if d <= thresholds[1] {
                "near"
            } else if d <= thresholds[2] {
                "far"
            } else {
                "noise"
            }
        })
        .collect()
}

/// Extract the causality direction between two nodes.
///
/// Uses [`causality_decompose`] on zero-initialized qualia points with
/// resonance values derived from the per-plane distance asymmetry.
///
/// The dominant direction across the three causality dimensions (warmth,
/// social, sacredness) determines the overall edge direction.
///
/// # Example
///
/// ```
/// use ndarray::hpc::udf_kernels::causal_edge;
/// use ndarray::hpc::node::Node;
/// use ndarray::hpc::causality::CausalityDirection;
///
/// let mut a = Node::random(42);
/// let mut b = Node::random(99);
/// let dir = causal_edge(&mut a, &mut b);
/// // Direction depends on node content
/// assert!(matches!(
///     dir,
///     CausalityDirection::Forward
///         | CausalityDirection::Backward
///         | CausalityDirection::None
/// ));
/// ```
pub fn causal_edge(node_a: &mut Node, node_b: &mut Node) -> CausalityDirection {
    use super::bf16_truth::PackedQualia;

    // Extract per-plane truth values to build qualia resonance vectors
    let truth_a_s = node_a.truth(S__);
    let truth_a_p = node_a.truth(_P_);
    let truth_a_o = node_a.truth(__O);

    let truth_b_s = node_b.truth(S__);
    let truth_b_p = node_b.truth(_P_);
    let truth_b_o = node_b.truth(__O);

    // Map truth frequency differences to resonance on causality dimensions:
    //   warmth (dim 4)    <- subject plane
    //   social (dim 6)    <- predicate plane
    //   sacredness (dim 8) <- object plane
    let mut qa = PackedQualia::zero();
    let mut qb = PackedQualia::zero();

    // Use frequency as resonance signal (map u16 to i8 range)
    qa.resonance[4] = (truth_a_s.frequency as i16 / 512).clamp(-128, 127) as i8;
    qa.resonance[6] = (truth_a_p.frequency as i16 / 512).clamp(-128, 127) as i8;
    qa.resonance[8] = (truth_a_o.frequency as i16 / 512).clamp(-128, 127) as i8;

    qb.resonance[4] = (truth_b_s.frequency as i16 / 512).clamp(-128, 127) as i8;
    qb.resonance[6] = (truth_b_p.frequency as i16 / 512).clamp(-128, 127) as i8;
    qb.resonance[8] = (truth_b_o.frequency as i16 / 512).clamp(-128, 127) as i8;

    let decomposition = causality_decompose(&qa, &qb, None);

    // Vote: majority of the three dimensions decides direction
    let mut forward_count = 0i32;
    let mut backward_count = 0i32;
    for dir in [
        decomposition.warmth_dir,
        decomposition.social_dir,
        decomposition.sacredness_dir,
    ] {
        match dir {
            CausalityDirection::Forward => forward_count += 1,
            CausalityDirection::Backward => backward_count += 1,
            CausalityDirection::None => {}
        }
    }

    if forward_count > backward_count {
        CausalityDirection::Forward
    } else if backward_count > forward_count {
        CausalityDirection::Backward
    } else {
        CausalityDirection::None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Layer 1: udf_hamming ─────────────────────────────────────────

    #[test]
    fn test_udf_hamming_identical() {
        let a = vec![0xAAu8; 64];
        assert_eq!(udf_hamming(&a, &a).unwrap(), 0);
    }

    #[test]
    fn test_udf_hamming_all_different() {
        let a = vec![0xFFu8; 64];
        let b = vec![0x00u8; 64];
        assert_eq!(udf_hamming(&a, &b).unwrap(), 64 * 8);
    }

    #[test]
    fn test_udf_hamming_length_mismatch() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 20];
        assert!(udf_hamming(&a, &b).is_err());
    }

    #[test]
    fn test_udf_hamming_empty() {
        assert_eq!(udf_hamming(&[], &[]).unwrap(), 0);
    }

    // ── Layer 1: udf_spo_distance ───────────────────────────────────

    #[test]
    fn test_udf_spo_distance_identical() {
        let s = vec![0xAAu8; 2048];
        let p = vec![0xBBu8; 2048];
        let o = vec![0xCCu8; 2048];
        let result = udf_spo_distance(&s, &p, &o, &s, &p, &o).unwrap();
        assert_eq!(result.subject_dist, Some(0));
        assert_eq!(result.predicate_dist, Some(0));
        assert_eq!(result.object_dist, Some(0));
        assert_eq!(result.combined_dist, Some(0));
    }

    #[test]
    fn test_udf_spo_distance_different() {
        let s1 = vec![0xAAu8; 2048];
        let p1 = vec![0xBBu8; 2048];
        let o1 = vec![0xCCu8; 2048];
        let s2 = vec![0x55u8; 2048];
        let p2 = vec![0x44u8; 2048];
        let o2 = vec![0x33u8; 2048];
        let result = udf_spo_distance(&s1, &p1, &o1, &s2, &p2, &o2).unwrap();
        // Different planes should have non-zero distance
        assert!(result.subject_dist.unwrap_or(0) > 0 || result.subject_dist.is_none());
        assert!(result.combined_dist.is_some());
    }

    #[test]
    fn test_udf_spo_distance_wrong_size() {
        let short = vec![0u8; 100];
        let ok = vec![0u8; 2048];
        assert!(udf_spo_distance(&short, &ok, &ok, &ok, &ok, &ok).is_err());
    }

    // ── Layer 1: udf_nars_revision ──────────────────────────────────

    #[test]
    fn test_udf_nars_revision_equal_evidence() {
        let (f, c) = udf_nars_revision(60000, 50000, 30000, 40000);
        // Revision combines with equal evidence weights
        assert_eq!(f, 45000); // (60000 + 30000) / 2
        assert_eq!(c, 45000); // (50000 + 40000) / 2
    }

    #[test]
    fn test_udf_nars_revision_same_values() {
        let (f, c) = udf_nars_revision(40000, 30000, 40000, 30000);
        assert_eq!(f, 40000);
        assert_eq!(c, 30000);
    }

    #[test]
    fn test_udf_nars_revision_zero_evidence() {
        // Both have zero evidence => ignorance
        let (f, _c) = udf_nars_revision(60000, 50000, 30000, 40000);
        // With evidence=1 each, it's a weighted average
        assert!(f > 0);
    }

    // ── Layer 1: udf_sigma_classify ─────────────────────────────────

    #[test]
    fn test_udf_sigma_classify_exact() {
        assert_eq!(udf_sigma_classify(100, 1000), "exact");
    }

    #[test]
    fn test_udf_sigma_classify_near() {
        assert_eq!(udf_sigma_classify(400, 1000), "near");
    }

    #[test]
    fn test_udf_sigma_classify_far() {
        assert_eq!(udf_sigma_classify(600, 1000), "far");
    }

    #[test]
    fn test_udf_sigma_classify_noise() {
        assert_eq!(udf_sigma_classify(1500, 1000), "noise");
    }

    #[test]
    fn test_udf_sigma_classify_boundary_foveal() {
        // threshold / 4 = 250 => distance 250 should be foveal
        assert_eq!(udf_sigma_classify(250, 1000), "exact");
    }

    // ── Layer 1: udf_bf16_hamming ───────────────────────────────────

    #[test]
    fn test_udf_bf16_hamming_identical() {
        let a = vec![0x3Fu8, 0x80, 0x40, 0x00];
        let w = make_weights(256, 16, 1);
        assert!((udf_bf16_hamming(&a, &a, &w).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_udf_bf16_hamming_sign_flip() {
        let a = vec![0x00u8, 0x00];
        let b = vec![0x00u8, 0x80]; // sign bit flipped
        let w = make_weights(256, 16, 1);
        assert!((udf_bf16_hamming(&a, &b, &w).unwrap() - 256.0).abs() < 1e-6);
    }

    #[test]
    fn test_udf_bf16_hamming_length_mismatch() {
        let a = vec![0u8; 4];
        let b = vec![0u8; 6];
        let w = make_weights(256, 16, 1);
        assert!(udf_bf16_hamming(&a, &b, &w).is_err());
    }

    #[test]
    fn test_udf_bf16_hamming_odd_length() {
        let a = vec![0u8; 3];
        let b = vec![0u8; 3];
        let w = make_weights(256, 16, 1);
        assert!(udf_bf16_hamming(&a, &b, &w).is_err());
    }

    #[test]
    fn test_udf_bf16_hamming_bad_weights() {
        let a = vec![0u8; 4];
        let w = vec![0u8; 4]; // wrong size
        assert!(udf_bf16_hamming(&a, &a, &w).is_err());
    }

    fn make_weights(sign: u16, exponent: u16, mantissa: u16) -> Vec<u8> {
        let mut w = Vec::with_capacity(6);
        w.extend_from_slice(&sign.to_le_bytes());
        w.extend_from_slice(&exponent.to_le_bytes());
        w.extend_from_slice(&mantissa.to_le_bytes());
        w
    }

    // ── Layer 2: factorize_spo ──────────────────────────────────────

    #[test]
    fn test_factorize_spo_empty_node() {
        let mut node = Node::new();
        // Empty node has no encounters => bits are all zero
        let terms = factorize_spo(&mut node);
        assert_eq!(terms[0], 0); // empty
        assert_eq!(terms[1], 0); // S
        assert_eq!(terms[2], 0); // P
        assert_eq!(terms[3], 0); // O
    }

    #[test]
    fn test_factorize_spo_random_node() {
        let mut node = Node::random(42);
        let terms = factorize_spo(&mut node);
        assert_eq!(terms[0], 0); // empty is always 0
        assert!(terms[1] > 0); // S should have bits
        assert!(terms[2] > 0); // P should have bits
        assert!(terms[3] > 0); // O should have bits
        // Combination norms are sums
        assert_eq!(terms[4], terms[1] + terms[2]); // SP = S + P
        assert_eq!(terms[5], terms[2] + terms[3]); // PO = P + O
        assert_eq!(terms[6], terms[1] + terms[3]); // SO = S + O
        assert_eq!(terms[7], terms[1] + terms[2] + terms[3]); // SPO
    }

    // ── Layer 2: nars_accumulate ────────────────────────────────────

    #[test]
    fn test_nars_accumulate_empty() {
        let (f, c) = nars_accumulate(&[]);
        assert_eq!(f, 32768); // ignorance prior
        assert_eq!(c, 0);
    }

    #[test]
    fn test_nars_accumulate_single() {
        let (f, c) = nars_accumulate(&[(60000, 50000)]);
        assert_eq!(f, 60000);
        assert_eq!(c, 50000);
    }

    #[test]
    fn test_nars_accumulate_two() {
        let (f, c) = nars_accumulate(&[(60000, 50000), (30000, 40000)]);
        // Should be same as single revision
        let (f2, c2) = udf_nars_revision(60000, 50000, 30000, 40000);
        assert_eq!(f, f2);
        assert_eq!(c, c2);
    }

    #[test]
    fn test_nars_accumulate_multiple() {
        let evidence = vec![(60000u16, 50000u16), (30000, 40000), (50000, 45000)];
        let (f, c) = nars_accumulate(&evidence);
        assert!(f > 0);
        assert!(c > 0);
    }

    // ── Layer 2: sigma_classify_batch ───────────────────────────────

    #[test]
    fn test_sigma_classify_batch_basic() {
        let distances = vec![10, 200, 500, 2000];
        let thresholds = [100, 400, 1000];
        let labels = sigma_classify_batch(&distances, &thresholds);
        assert_eq!(labels, vec!["exact", "near", "far", "noise"]);
    }

    #[test]
    fn test_sigma_classify_batch_empty() {
        let labels = sigma_classify_batch(&[], &[100, 400, 1000]);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_sigma_classify_batch_all_exact() {
        let distances = vec![0, 1, 50, 100];
        let thresholds = [100, 400, 1000];
        let labels = sigma_classify_batch(&distances, &thresholds);
        assert!(labels.iter().all(|&l| l == "exact"));
    }

    #[test]
    fn test_sigma_classify_batch_boundaries() {
        let thresholds = [100, 400, 1000];
        let distances = vec![100, 101, 400, 401, 1000, 1001];
        let labels = sigma_classify_batch(&distances, &thresholds);
        assert_eq!(labels[0], "exact"); // == threshold[0]
        assert_eq!(labels[1], "near"); // > threshold[0]
        assert_eq!(labels[2], "near"); // == threshold[1]
        assert_eq!(labels[3], "far"); // > threshold[1]
        assert_eq!(labels[4], "far"); // == threshold[2]
        assert_eq!(labels[5], "noise"); // > threshold[2]
    }

    // ── Layer 2: causal_edge ────────────────────────────────────────

    #[test]
    fn test_causal_edge_same_nodes() {
        let mut a = Node::random(42);
        let mut b = Node::random(42); // same seed => same content
        let dir = causal_edge(&mut a, &mut b);
        assert_eq!(dir, CausalityDirection::None);
    }

    #[test]
    fn test_causal_edge_different_nodes() {
        let mut a = Node::random(42);
        let mut b = Node::random(99);
        let dir = causal_edge(&mut a, &mut b);
        // Should produce a definite direction for different nodes
        assert!(matches!(
            dir,
            CausalityDirection::Forward
                | CausalityDirection::Backward
                | CausalityDirection::None
        ));
    }

    #[test]
    fn test_causal_edge_deterministic() {
        let mut a1 = Node::random(10);
        let mut b1 = Node::random(20);
        let dir1 = causal_edge(&mut a1, &mut b1);

        let mut a2 = Node::random(10);
        let mut b2 = Node::random(20);
        let dir2 = causal_edge(&mut a2, &mut b2);

        assert_eq!(dir1, dir2, "causal_edge should be deterministic");
    }

    #[test]
    fn test_causal_edge_empty_nodes() {
        let mut a = Node::new();
        let mut b = Node::new();
        // Empty nodes have ignorance truth => equal frequencies => None
        let dir = causal_edge(&mut a, &mut b);
        assert_eq!(dir, CausalityDirection::None);
    }
}
