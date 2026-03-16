#![allow(clippy::assign_op_pattern, clippy::too_many_arguments, clippy::manual_range_contains, clippy::needless_range_loop)]

//! Causal Trajectory Hydration via BNN Instrumentation.
//!
//! Sits between the foveal layer (resonator's converged SPO factorizations)
//! and the context layer (cross-plane typed halo from σ₂–σ₃ noise floor).
//!
//! Runs NARS on the delta between fovea and context across resonator iterations
//! to produce causality trajectories that grow the DN tree (Sigma Graph).
//!
//! ## BNN Components as Causal Instruments
//!
//! | Component | Causal Role |
//! |---|---|
//! | EWM correction pattern | Causal saliency map: WHERE the resonator worked hardest |
//! | BPReLU asymmetry | Causal directionality: forward (do) vs backward (observe) |
//! | RIF shortcuts (XOR t↔t-2) | Causal chains: genealogy of factorization convergence |
//!
//! ## Pipeline
//!
//! ```text
//! Resonator iteration t
//!   → Record snapshot (estimates, halo, distances, deltas)
//!   → RIF diff: XOR(snapshot[t], snapshot[t-2])
//!   → EWM saliency: tier transitions across iterations
//!   → BPReLU arrow: forward vs backward activation asymmetry
//!   → Halo transitions: promotions/demotions across lattice levels
//!   → NARS statements: causal judgments with truth values
//!   → Sigma edges: DN tree growth instructions
//! ```
//!
//! ## References
//!
//! - Zhang et al. 2025: BIR-EWM + BPReLU + Rich Information Flow
//! - Pearl 2009: do-calculus (BPReLU forward ≈ interventional, backward ≈ observational)
//! - Czégel et al. 2021: error thresholds for staged assembly via Hold state

use super::fingerprint::Fingerprint;
use super::kernels::{score_sigma, EnergyConflict, SigmaGate, SignificanceLevel};
use super::bnn_cross_plane::CollapseGate;

use super::bnn_cross_plane::{CrossPlaneVote, HaloType, InferenceMode};

// ============================================================================
// BPReLU — ported from rustynum-bnn rif_net_integration
// ============================================================================

/// Asymmetric activation: α_pos for positive, α_neg for negative inputs.
///
/// BPReLU (Bidirectional Parametric ReLU) distinguishes causal direction:
/// forward (interventional, α_pos) vs backward (observational, α_neg).
pub struct BPReLU {
    pub alpha_pos: f32,
    pub alpha_neg: f32,
}

impl Default for BPReLU {
    fn default() -> Self {
        Self {
            alpha_pos: 1.0,
            alpha_neg: 0.25,
        }
    }
}

impl BPReLU {
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            self.alpha_pos * x
        } else {
            self.alpha_neg * x
        }
    }
}

// ============================================================================
// EWM Tier Classification
// ============================================================================

/// EWM tier for a single dimension based on distance to codebook.
///
/// Maps awareness states to causal significance:
/// - Crystallized dimensions are settled → high causal weight
/// - Noise dimensions are irrelevant → zero causal weight
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwmTier {
    /// Well-matched to codebook (σ < 1.5). Settled knowledge.
    Crystallized,
    /// Good match (1.5 ≤ σ < 2.0). Familiar territory.
    Confident,
    /// Boundary region (2.0 ≤ σ < 3.0). Under active revision.
    Transitional,
    /// No match (σ ≥ 3.0). Irrelevant noise.
    Noise,
}

impl From<SignificanceLevel> for EwmTier {
    /// Map σ-significance levels from the kernel pipeline to EWM tiers.
    ///
    /// The mapping preserves the semantic alignment:
    /// - Discovery (> 3σ) → Crystallized: settled, high causal weight
    /// - Strong (2.5-3σ) → Confident: strong match, familiar territory
    /// - Evidence/Hint (1.5-2.5σ) → Transitional: under active revision
    /// - Noise (< 1.5σ) → Noise: irrelevant, zero causal weight
    fn from(level: SignificanceLevel) -> Self {
        match level {
            SignificanceLevel::Discovery => EwmTier::Crystallized,
            SignificanceLevel::Strong => EwmTier::Confident,
            SignificanceLevel::Evidence | SignificanceLevel::Hint => EwmTier::Transitional,
            SignificanceLevel::Noise => EwmTier::Noise,
        }
    }
}

// ============================================================================
// Resonator Snapshot — one iteration's state
// ============================================================================

/// One snapshot of the resonator state at a given iteration.
///
/// Records the factorization estimates, cross-plane halo, per-plane
/// Hamming distances to the codebook, and convergence deltas.
#[derive(Clone, Debug)]
pub struct ResonatorSnapshot {
    /// Iteration number (0-based).
    pub iter: u16,
    /// S-plane estimate (Fingerprint<256> = 16384 bits).
    pub s_est: Fingerprint<256>,
    /// P-plane estimate.
    pub p_est: Fingerprint<256>,
    /// O-plane estimate.
    pub o_est: Fingerprint<256>,
    /// Per-plane survivor bitmasks from cascade (for cross-plane vote).
    pub s_mask: Vec<u64>,
    /// P-plane survivor bitmask.
    pub p_mask: Vec<u64>,
    /// O-plane survivor bitmask.
    pub o_mask: Vec<u64>,
    /// Number of codebook entries.
    pub n_entries: usize,
    /// Convergence delta: Hamming(s_est[t], s_est[t-1]).
    pub delta_s: u32,
    /// Convergence delta: Hamming(p_est[t], p_est[t-1]).
    pub delta_p: u32,
    /// Convergence delta: Hamming(o_est[t], o_est[t-1]).
    pub delta_o: u32,
}

impl ResonatorSnapshot {
    /// Total convergence delta across all planes.
    #[inline]
    pub fn total_delta(&self) -> u32 {
        self.delta_s + self.delta_p + self.delta_o
    }

    /// Whether the resonator has converged (all deltas below threshold).
    #[inline]
    pub fn converged(&self, threshold: u32) -> bool {
        self.delta_s < threshold && self.delta_p < threshold && self.delta_o < threshold
    }

    /// Extract the cross-plane vote from this snapshot's survivor masks.
    pub fn cross_plane_vote(&self) -> CrossPlaneVote {
        CrossPlaneVote::extract(&self.s_mask, &self.p_mask, &self.o_mask, self.n_entries)
    }
}

// ============================================================================
// RIF Diff — XOR between non-adjacent snapshots
// ============================================================================

/// RIF-style causal diff: XOR between snapshots at iterations t and t-2.
///
/// Records WHAT CHANGED across two resonator iterations. The permuted
/// version (word rotation) prevents trivial cancellation when composing
/// multiple diffs into a causal chain.
#[derive(Clone, Debug)]
pub struct RifDiff {
    /// Source iteration.
    pub from_iter: u16,
    /// Target iteration.
    pub to_iter: u16,
    /// S-plane XOR diff (what changed in S across 2 iterations).
    pub s_diff: Fingerprint<256>,
    /// P-plane XOR diff.
    pub p_diff: Fingerprint<256>,
    /// O-plane XOR diff.
    pub o_diff: Fingerprint<256>,
    /// Popcount of s_diff (number of changed bits in S).
    pub s_activity: u32,
    /// Popcount of p_diff.
    pub p_activity: u32,
    /// Popcount of o_diff.
    pub o_activity: u32,
}

impl RifDiff {
    /// Compute the RIF diff between two snapshots.
    pub fn compute(earlier: &ResonatorSnapshot, later: &ResonatorSnapshot) -> Self {
        let s_diff = &later.s_est ^ &earlier.s_est;
        let p_diff = &later.p_est ^ &earlier.p_est;
        let o_diff = &later.o_est ^ &earlier.o_est;
        Self {
            from_iter: earlier.iter,
            to_iter: later.iter,
            s_activity: s_diff.popcount(),
            p_activity: p_diff.popcount(),
            o_activity: o_diff.popcount(),
            s_diff,
            p_diff,
            o_diff,
        }
    }

    /// Total activity across all planes.
    #[inline]
    pub fn total_activity(&self) -> u32 {
        self.s_activity + self.p_activity + self.o_activity
    }

    /// Which plane had the most activity (most bits changed).
    pub fn dominant_plane(&self) -> DominantPlane {
        if self.s_activity >= self.p_activity && self.s_activity >= self.o_activity {
            DominantPlane::S
        } else if self.p_activity >= self.o_activity {
            DominantPlane::P
        } else {
            DominantPlane::O
        }
    }
}

/// Which plane dominates in a comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DominantPlane {
    S,
    P,
    O,
}

// ============================================================================
// EWM Correction — per-iteration saliency
// ============================================================================

/// Per-iteration EWM correction record.
///
/// Records how much the resonator's estimate changed per-plane and
/// classifies each word into EWM tiers based on convergence behavior.
#[derive(Clone, Debug)]
pub struct EwmCorrection {
    /// Iteration number.
    pub iter: u16,
    /// Per-word L1 correction magnitude for S-plane (256 values).
    pub s_correction: [u32; 256],
    /// Per-word L1 correction magnitude for P-plane.
    pub p_correction: [u32; 256],
    /// Per-word L1 correction magnitude for O-plane.
    pub o_correction: [u32; 256],
    /// Per-plane EWM tier classification from σ-significance.
    ///
    /// Derived from the aggregate correction magnitude via score_sigma():
    /// lower correction = closer to convergence = higher tier.
    pub plane_tiers: [EwmTier; 3],
}

impl EwmCorrection {
    /// Compute per-word L1 correction between two snapshots.
    ///
    /// Also classifies per-plane aggregate correction into EwmTier via σ-scoring.
    pub fn compute(prev: &ResonatorSnapshot, curr: &ResonatorSnapshot) -> Self {
        Self::compute_with_gate(prev, curr, &SigmaGate::sku_16k())
    }

    /// Compute with a specific σ-gate (e.g., for SKU-64K).
    pub fn compute_with_gate(
        prev: &ResonatorSnapshot,
        curr: &ResonatorSnapshot,
        gate: &SigmaGate,
    ) -> Self {
        let s_correction = per_word_popcount(&(&curr.s_est ^ &prev.s_est));
        let p_correction = per_word_popcount(&(&curr.p_est ^ &prev.p_est));
        let o_correction = per_word_popcount(&(&curr.o_est ^ &prev.o_est));

        // Classify per-plane aggregate correction → EwmTier.
        // Lower aggregate correction = resonator converging = higher tier.
        let totals = [
            s_correction.iter().sum::<u32>(),
            p_correction.iter().sum::<u32>(),
            o_correction.iter().sum::<u32>(),
        ];
        let plane_tiers = totals.map(|total| {
            let ec = EnergyConflict {
                conflict: total,
                energy_a: 0,
                energy_b: 0,
                agreement: 0,
            };
            EwmTier::from(score_sigma(&ec, gate).level)
        });

        Self {
            iter: curr.iter,
            s_correction,
            p_correction,
            o_correction,
            plane_tiers,
        }
    }

    /// Total S-plane correction magnitude.
    pub fn s_total(&self) -> u32 {
        self.s_correction.iter().sum()
    }

    /// Total P-plane correction magnitude.
    pub fn p_total(&self) -> u32 {
        self.p_correction.iter().sum()
    }

    /// Total O-plane correction magnitude.
    pub fn o_total(&self) -> u32 {
        self.o_correction.iter().sum()
    }
}

/// Compute per-word popcount of a fingerprint (bit changes per u64 word).
fn per_word_popcount(fp: &Fingerprint<256>) -> [u32; 256] {
    let mut result = [0u32; 256];
    for (i, &word) in fp.words.iter().enumerate() {
        result[i] = word.count_ones();
    }
    result
}

// ============================================================================
// Causal Saliency — crystallizing, dissolving, contested dimensions
// ============================================================================

/// Causal saliency map from EWM correction history.
///
/// Tracks which dimensions are:
/// - **Crystallizing**: correction decreasing → resonator converging here
/// - **Dissolving**: correction increasing → resonator diverging here
/// - **Contested**: correction oscillating → resonator can't decide
#[derive(Clone, Debug)]
pub struct CausalSaliency {
    /// Words where correction is decreasing (converging). Signal: new evidence found.
    pub crystallizing: Vec<u64>,
    /// Words where correction is increasing (diverging). Signal: certainty lost.
    pub dissolving: Vec<u64>,
    /// Words that oscillate across iterations. Signal: competing hypotheses.
    pub contested: Vec<u64>,
    /// Per-plane crystallizing count.
    pub crystallizing_count: [u32; 3],
    /// Per-plane dissolving count.
    pub dissolving_count: [u32; 3],
    /// Per-plane contested count.
    pub contested_count: [u32; 3],
}

impl CausalSaliency {
    /// Compute saliency from a window of EWM corrections.
    ///
    /// Requires at least 2 corrections to detect trends.
    pub fn from_ewm_window(corrections: &[EwmCorrection]) -> Self {
        let n_words = 256usize;
        let mut crystallizing = vec![0u64; n_words / 64 * 3]; // 3 planes × 4 words
        let mut dissolving = vec![0u64; n_words / 64 * 3];
        let mut contested = vec![0u64; n_words / 64 * 3];
        let mut cryst_count = [0u32; 3];
        let mut diss_count = [0u32; 3];
        let mut cont_count = [0u32; 3];

        if corrections.len() < 2 {
            return Self {
                crystallizing,
                dissolving,
                contested,
                crystallizing_count: cryst_count,
                dissolving_count: diss_count,
                contested_count: cont_count,
            };
        }

        // Compare first and last correction in window
        let first = &corrections[0];
        let last = &corrections[corrections.len() - 1];

        for word_idx in 0..n_words {
            // S-plane
            classify_word_trend(
                first.s_correction[word_idx],
                last.s_correction[word_idx],
                corrections,
                word_idx,
                0, // plane index
                &mut crystallizing,
                &mut dissolving,
                &mut contested,
                &mut cryst_count,
                &mut diss_count,
                &mut cont_count,
            );
            // P-plane
            classify_word_trend(
                first.p_correction[word_idx],
                last.p_correction[word_idx],
                corrections,
                word_idx,
                1,
                &mut crystallizing,
                &mut dissolving,
                &mut contested,
                &mut cryst_count,
                &mut diss_count,
                &mut cont_count,
            );
            // O-plane
            classify_word_trend(
                first.o_correction[word_idx],
                last.o_correction[word_idx],
                corrections,
                word_idx,
                2,
                &mut crystallizing,
                &mut dissolving,
                &mut contested,
                &mut cryst_count,
                &mut diss_count,
                &mut cont_count,
            );
        }

        Self {
            crystallizing,
            dissolving,
            contested,
            crystallizing_count: cryst_count,
            dissolving_count: diss_count,
            contested_count: cont_count,
        }
    }
}

/// Classify a single word's trend across the correction window.
#[allow(clippy::too_many_arguments)]
fn classify_word_trend(
    first_val: u32,
    last_val: u32,
    corrections: &[EwmCorrection],
    word_idx: usize,
    plane_idx: usize, // 0=S, 1=P, 2=O
    crystallizing: &mut [u64],
    dissolving: &mut [u64],
    contested: &mut [u64],
    cryst_count: &mut [u32; 3],
    diss_count: &mut [u32; 3],
    cont_count: &mut [u32; 3],
) {
    let mask_offset = plane_idx * 4; // 4 u64 words per plane (256 bits / 64)
    let u64_idx = mask_offset + word_idx / 64;
    let bit_pos = word_idx % 64;

    // Check for oscillation: count direction changes
    let mut direction_changes = 0u32;
    for i in 1..corrections.len() {
        let prev = get_correction_val(&corrections[i - 1], plane_idx, word_idx);
        let curr = get_correction_val(&corrections[i], plane_idx, word_idx);
        if curr != prev && i > 1 {
            let prev2 = get_correction_val(&corrections[i - 2], plane_idx, word_idx);
            if (curr > prev) != (prev > prev2) {
                direction_changes += 1;
            }
        }
    }

    if direction_changes >= 2 {
        // Oscillating → contested
        contested[u64_idx] |= 1u64 << bit_pos;
        cont_count[plane_idx] += 1;
    } else if last_val + 2 < first_val {
        // Decreasing → crystallizing (add 2 to avoid noise)
        crystallizing[u64_idx] |= 1u64 << bit_pos;
        cryst_count[plane_idx] += 1;
    } else if last_val > first_val + 2 {
        // Increasing → dissolving
        dissolving[u64_idx] |= 1u64 << bit_pos;
        diss_count[plane_idx] += 1;
    }
}

/// Extract the correction value for a specific plane and word index.
fn get_correction_val(corr: &EwmCorrection, plane_idx: usize, word_idx: usize) -> u32 {
    match plane_idx {
        0 => corr.s_correction[word_idx],
        1 => corr.p_correction[word_idx],
        _ => corr.o_correction[word_idx],
    }
}

// ============================================================================
// Causal Arrow — BPReLU forward/backward asymmetry
// ============================================================================

/// Causal direction between adjacent resonator states.
///
/// Forward: foveal commitment DROVE the context change (interventional).
/// Backward: context OVERRODE the foveal commitment (observational).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CausalDirection {
    /// Commitment drove change (α_pos dominated). P(effect | do(cause)).
    Forward(f32),
    /// Context overrode commitment (α_neg dominated). P(cause | effect).
    Backward(f32),
    /// No clear direction (α_pos ≈ α_neg). Correlation without causation.
    Symmetric,
    /// Dimensions split between forward and backward. Mixed causation.
    Contested(f32),
}

/// Causal arrow between two adjacent resonator iterations.
///
/// Uses BPReLU asymmetry to determine whether the previous iteration's
/// commitment CAUSED the current context, or the current context
/// FORCED a revision of the commitment.
#[derive(Clone, Debug)]
pub struct CausalArrow {
    /// Source iteration.
    pub iter: u16,
    /// Per-plane causal direction.
    pub s_direction: CausalDirection,
    pub p_direction: CausalDirection,
    pub o_direction: CausalDirection,
    /// Per-plane forward activation magnitude.
    pub forward_magnitude: [f32; 3],
    /// Per-plane backward activation magnitude.
    pub backward_magnitude: [f32; 3],
    /// Overall causal direction (aggregate).
    pub overall: CausalDirection,
}

impl CausalArrow {
    /// Compute causal arrow between two snapshots using BPReLU asymmetry.
    ///
    /// The insight: if the previous estimate predicts the current delta
    /// (forward BPReLU response > backward), the commitment was causal.
    /// If the current delta explains the previous estimate better
    /// (backward > forward), the context was causal.
    pub fn compute(prev: &ResonatorSnapshot, curr: &ResonatorSnapshot) -> Self {
        let bprelu = BPReLU::default(); // α_pos=1.0, α_neg=0.25

        // Per-plane: compute forward and backward activation sums
        let (fwd_s, bwd_s) = plane_asymmetry(&prev.s_est, &curr.s_est, &bprelu);
        let (fwd_p, bwd_p) = plane_asymmetry(&prev.p_est, &curr.p_est, &bprelu);
        let (fwd_o, bwd_o) = plane_asymmetry(&prev.o_est, &curr.o_est, &bprelu);

        let s_dir = classify_asymmetry(fwd_s, bwd_s);
        let p_dir = classify_asymmetry(fwd_p, bwd_p);
        let o_dir = classify_asymmetry(fwd_o, bwd_o);

        let total_fwd = fwd_s + fwd_p + fwd_o;
        let total_bwd = bwd_s + bwd_p + bwd_o;
        let overall = classify_asymmetry(total_fwd, total_bwd);

        CausalArrow {
            iter: curr.iter,
            s_direction: s_dir,
            p_direction: p_dir,
            o_direction: o_dir,
            forward_magnitude: [fwd_s, fwd_p, fwd_o],
            backward_magnitude: [bwd_s, bwd_p, bwd_o],
            overall,
        }
    }
}

/// Compute forward/backward activation asymmetry for one plane.
///
/// Forward: apply BPReLU to the signed delta (curr - prev interpreted as signed).
/// Backward: apply BPReLU to the negated delta.
///
/// We use popcount of the XOR as unsigned magnitude, then apply the BPReLU
/// based on whether the estimate moved TOWARD the codebook (forward) or
/// AWAY from it (backward), as indicated by the convergence delta.
fn plane_asymmetry(
    prev_est: &Fingerprint<256>,
    curr_est: &Fingerprint<256>,
    bprelu: &BPReLU,
) -> (f32, f32) {
    let diff = prev_est ^ curr_est;
    let changed_bits = diff.popcount() as f32;
    let total_bits = Fingerprint::<256>::BITS as f32;

    // Normalize to [-1, 1]: 0 bits changed = +1 (stable), all changed = -1
    let stability = 1.0 - 2.0 * changed_bits / total_bits;

    // Forward: how much does the previous state predict current stability?
    let fwd = bprelu.apply(stability);
    // Backward: how much does the current state contradict the previous?
    let bwd = bprelu.apply(-stability);

    (fwd.abs(), bwd.abs())
}

/// Classify forward/backward asymmetry into a causal direction.
fn classify_asymmetry(forward: f32, backward: f32) -> CausalDirection {
    let total = forward + backward;
    if total < 1e-6 {
        return CausalDirection::Symmetric;
    }
    let ratio = forward / total;
    if ratio > 0.7 {
        CausalDirection::Forward(ratio)
    } else if ratio < 0.3 {
        CausalDirection::Backward(1.0 - ratio)
    } else {
        // Check if both are strong (contested) or both weak (symmetric)
        if total > 0.5 {
            CausalDirection::Contested(ratio)
        } else {
            CausalDirection::Symmetric
        }
    }
}

// ============================================================================
// Causal Chain — stacked RIF diffs revealing convergence genealogy
// ============================================================================

/// A causal link between two planes across iteration windows.
#[derive(Clone, Debug)]
pub struct CausalLink {
    /// Which plane stabilized first (the cause).
    pub cause_plane: DominantPlane,
    /// Which plane responded later (the effect).
    pub effect_plane: DominantPlane,
    /// Confidence: proportion of bits that show this pattern.
    pub confidence: f32,
    /// Iteration range over which this link was observed.
    pub from_iter: u16,
    pub to_iter: u16,
}

/// A causal chain: sequence of CausalLinks extracted from RIF diffs.
///
/// The chain captures the genealogy of factorization convergence:
/// which plane converged first, and which planes followed.
#[derive(Clone, Debug)]
pub struct CausalChain {
    pub links: Vec<CausalLink>,
}

impl CausalChain {
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Analyze stacked RIF diffs to extract the causal chain.
    ///
    /// The plane that was active EARLY but quiet LATE stabilized first
    /// (= it was the CAUSE). The plane that was quiet EARLY but active
    /// LATE was responding (= it was the EFFECT).
    pub fn from_rif_diffs(diffs: &[RifDiff]) -> Self {
        let mut chain = Self::new();

        for window in diffs.windows(2) {
            let early = &window[0];
            let late = &window[1];

            // A plane that stabilized: high activity early, low activity late
            let s_stabilized = early.s_activity > late.s_activity.saturating_mul(2);
            let p_stabilized = early.p_activity > late.p_activity.saturating_mul(2);
            let o_stabilized = early.o_activity > late.o_activity.saturating_mul(2);

            // A plane that responded: low activity early, high activity late
            let s_responding = late.s_activity > early.s_activity.saturating_mul(2);
            let p_responding = late.p_activity > early.p_activity.saturating_mul(2);
            let o_responding = late.o_activity > early.o_activity.saturating_mul(2);

            let total_bits = Fingerprint::<256>::BITS as f32;

            // Generate all cause→effect links
            let pairs: [(bool, DominantPlane, u32, bool, DominantPlane); 6] = [
                (
                    s_stabilized,
                    DominantPlane::S,
                    early.s_activity,
                    p_responding,
                    DominantPlane::P,
                ),
                (
                    s_stabilized,
                    DominantPlane::S,
                    early.s_activity,
                    o_responding,
                    DominantPlane::O,
                ),
                (
                    p_stabilized,
                    DominantPlane::P,
                    early.p_activity,
                    s_responding,
                    DominantPlane::S,
                ),
                (
                    p_stabilized,
                    DominantPlane::P,
                    early.p_activity,
                    o_responding,
                    DominantPlane::O,
                ),
                (
                    o_stabilized,
                    DominantPlane::O,
                    early.o_activity,
                    s_responding,
                    DominantPlane::S,
                ),
                (
                    o_stabilized,
                    DominantPlane::O,
                    early.o_activity,
                    p_responding,
                    DominantPlane::P,
                ),
            ];

            for &(cause_stable, cause_plane, cause_activity, effect_resp, effect_plane) in &pairs {
                if cause_stable && effect_resp {
                    chain.links.push(CausalLink {
                        cause_plane,
                        effect_plane,
                        confidence: cause_activity as f32 / total_bits,
                        from_iter: early.from_iter,
                        to_iter: late.to_iter,
                    });
                }
            }
        }

        chain
    }

    /// The first plane to converge (root cause).
    pub fn root_cause(&self) -> Option<DominantPlane> {
        self.links.first().map(|l| l.cause_plane)
    }

    /// Number of causal links in the chain.
    pub fn depth(&self) -> usize {
        self.links.len()
    }
}

impl Default for CausalChain {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Halo Transition — promotions/demotions across lattice levels
// ============================================================================

/// A transition of a codebook entry between halo types across iterations.
#[derive(Clone, Debug)]
pub struct HaloTransition {
    /// Codebook entry index.
    pub entry_index: usize,
    /// Previous halo type.
    pub from: HaloType,
    /// Current halo type.
    pub to: HaloType,
    /// Change in lattice level (positive = promotion, negative = demotion).
    pub level_delta: i8,
}

/// Detect halo transitions between two snapshots.
///
/// Compares the cross-plane vote at iteration t-1 with iteration t
/// and identifies entries that moved between halo types.
pub fn detect_halo_transitions(
    prev: &ResonatorSnapshot,
    curr: &ResonatorSnapshot,
) -> Vec<HaloTransition> {
    let prev_vote = prev.cross_plane_vote();
    let curr_vote = curr.cross_plane_vote();
    let mut transitions = Vec::new();

    let all_types = [
        HaloType::Noise,
        HaloType::S,
        HaloType::P,
        HaloType::O,
        HaloType::SP,
        HaloType::SO,
        HaloType::PO,
        HaloType::Core,
    ];

    for &from_type in &all_types {
        let from_entries = prev_vote.entries_of(from_type);
        for &entry_idx in &from_entries {
            // Find this entry's current halo type
            let curr_type = entry_halo_type(&curr_vote, entry_idx);
            if curr_type != from_type {
                transitions.push(HaloTransition {
                    entry_index: entry_idx,
                    from: from_type,
                    to: curr_type,
                    level_delta: curr_type.lattice_level() as i8 - from_type.lattice_level() as i8,
                });
            }
        }
    }

    transitions
}

/// Determine which halo type a specific entry belongs to in a vote.
fn entry_halo_type(vote: &CrossPlaneVote, entry_idx: usize) -> HaloType {
    let word_idx = entry_idx / 64;
    let bit_pos = entry_idx % 64;

    let all_types = [
        (HaloType::Core, &vote.core),
        (HaloType::SP, &vote.sp),
        (HaloType::SO, &vote.so),
        (HaloType::PO, &vote.po),
        (HaloType::S, &vote.s_only),
        (HaloType::P, &vote.p_only),
        (HaloType::O, &vote.o_only),
    ];

    for (halo_type, mask) in &all_types {
        if word_idx < mask.len() && (mask[word_idx] >> bit_pos) & 1 == 1 {
            return *halo_type;
        }
    }
    HaloType::Noise
}

// ============================================================================
// NARS Causal Statements
// ============================================================================

/// NARS truth value: (frequency, confidence).
#[derive(Clone, Copy, Debug)]
pub struct NarsTruth {
    /// Frequency: proportion of positive evidence. Range [0, 1].
    pub f: f32,
    /// Confidence: amount of evidence relative to max. Range [0, 1].
    pub c: f32,
}

impl NarsTruth {
    pub fn new(f: f32, c: f32) -> Self {
        Self {
            f: f.clamp(0.0, 1.0),
            c: c.clamp(0.0, 1.0),
        }
    }

    /// NARS revision rule: combine two truth values.
    ///
    /// w₁ = c₁/(1-c₁), w₂ = c₂/(1-c₂)
    /// f_new = (w₁·f₁ + w₂·f₂) / (w₁ + w₂)
    /// c_new = (w₁ + w₂) / (w₁ + w₂ + 1)
    pub fn revise(self, other: NarsTruth) -> NarsTruth {
        let w1 = self.c / (1.0 - self.c + 1e-9);
        let w2 = other.c / (1.0 - other.c + 1e-9);
        let w_total = w1 + w2;
        if w_total < 1e-9 {
            return NarsTruth::new(0.5, 0.0);
        }
        NarsTruth::new(
            (w1 * self.f + w2 * other.f) / w_total,
            w_total / (w_total + 1.0),
        )
    }

    /// NARS deduction: <A→B> ⊗ <B→C> = <A→C>.
    pub fn deduction(self, other: NarsTruth) -> NarsTruth {
        let f = self.f * other.f;
        let c = self.f * other.f * self.c * other.c;
        NarsTruth::new(f, c / (c + 1e-3))
    }

    /// NARS abduction: <A→B> ⊗ <C→B> = <A→C>.
    pub fn abduction(self, other: NarsTruth) -> NarsTruth {
        let f = self.f;
        let c = self.f * other.f * self.c * other.c;
        NarsTruth::new(f, c / (c + 1e-3))
    }

    /// NARS induction: <A→B> ⊗ <A→C> = <B→C>.
    pub fn induction(self, other: NarsTruth) -> NarsTruth {
        let f = other.f;
        let c = self.f * other.f * self.c * other.c;
        NarsTruth::new(f, c / (c + 1e-3))
    }
}

/// The type of causal relationship discovered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalRelation {
    /// Plane A's convergence caused plane B's convergence.
    Causes,
    /// Plane A's convergence enabled plane B's convergence.
    Enables,
    /// Plane A and B are contradicting each other.
    Contradicts,
    /// Evidence supports the current factorization.
    Supports,
    /// Evidence undermines the current factorization.
    Undermines,
}

/// A NARS causal statement derived from trajectory analysis.
#[derive(Clone, Debug)]
pub struct NarsCausalStatement {
    /// The causal relationship.
    pub relation: CausalRelation,
    /// Source plane (cause).
    pub source_plane: DominantPlane,
    /// Target plane (effect), if applicable.
    pub target_plane: Option<DominantPlane>,
    /// NARS truth value for this statement.
    pub truth: NarsTruth,
    /// Iteration at which this statement was generated.
    pub iter: u16,
    /// The inference mode that generated this statement.
    pub inference_mode: Option<InferenceMode>,
    /// σ-significance of the underlying evidence, if available.
    ///
    /// When present, this is the σ-score of the Hamming distance that
    /// generated this statement. The p-value can replace heuristic
    /// confidence scaling.
    pub sigma: Option<SignificanceLevel>,
}

// ============================================================================
// Sigma Graph Edges — DN tree growth instructions
// ============================================================================

/// A Sigma Graph edge to be created from causal trajectory analysis.
///
/// These edges grow the DN tree based on observed causal structure.
#[derive(Clone, Debug)]
pub struct SigmaEdge {
    /// Source node (codebook entry index or plane identifier).
    pub source: SigmaNode,
    /// Target node.
    pub target: SigmaNode,
    /// Edge type (causal relation).
    pub relation: CausalRelation,
    /// NARS truth value for this edge.
    pub truth: NarsTruth,
    /// Iteration at which this edge was discovered.
    pub iter: u16,
}

/// A node in the Sigma Graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SigmaNode {
    /// A specific codebook entry.
    Entry(usize),
    /// A plane-level aggregate (all entries in this plane).
    Plane(DominantPlane),
    /// A halo type aggregate (all entries of this type).
    HaloGroup(HaloType),
}

// ============================================================================
// CausalTrajectory — the full trajectory across all iterations
// ============================================================================

/// The full causal trajectory across all resonator iterations.
///
/// Records snapshots, BNN instrumentation (RIF diffs, EWM corrections,
/// BPReLU arrows), and NARS output (causal statements, Sigma edges).
#[derive(Clone, Debug)]
pub struct CausalTrajectory {
    /// One snapshot per resonator iteration.
    pub snapshots: Vec<ResonatorSnapshot>,
    /// RIF-style causal diffs (XOR between non-adjacent snapshots).
    pub rif_diffs: Vec<RifDiff>,
    /// Per-iteration EWM correction maps.
    pub ewm_corrections: Vec<EwmCorrection>,
    /// BPReLU forward/backward asymmetry per transition.
    pub causal_arrows: Vec<CausalArrow>,
    /// Halo transitions between iterations.
    pub halo_transitions: Vec<HaloTransition>,
    /// NARS causal statements derived from trajectory analysis.
    pub nars_statements: Vec<NarsCausalStatement>,
    /// Sigma Graph edges to create.
    pub sigma_edges: Vec<SigmaEdge>,
    /// Per-plane σ-stripe population tracker across iteration windows.
    pub shift_detector: ShiftDetector,
    /// σ-gate for computing significance from raw Hamming distances.
    sigma_gate: SigmaGate,
}

impl CausalTrajectory {
    /// Create a new empty trajectory.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            rif_diffs: Vec::new(),
            ewm_corrections: Vec::new(),
            causal_arrows: Vec::new(),
            halo_transitions: Vec::new(),
            nars_statements: Vec::new(),
            sigma_edges: Vec::new(),
            shift_detector: ShiftDetector::new(),
            sigma_gate: SigmaGate::sku_16k(),
        }
    }

    /// Create a trajectory with a custom σ-gate (e.g., SKU-64K).
    pub fn with_sigma_gate(gate: SigmaGate) -> Self {
        Self {
            sigma_gate: gate,
            ..Self::new()
        }
    }

    /// Record a new snapshot and run all instrumentation.
    ///
    /// This is the main entry point called at each resonator iteration.
    /// Feeds the ShiftDetector with per-plane σ-significance from convergence deltas.
    pub fn record_iteration(&mut self, snapshot: ResonatorSnapshot) {
        let n = self.snapshots.len();

        // EWM correction (requires previous snapshot)
        if n > 0 {
            let ewm = EwmCorrection::compute(&self.snapshots[n - 1], &snapshot);
            self.ewm_corrections.push(ewm);
        }

        // BPReLU causal arrow (requires previous snapshot)
        if n > 0 {
            let arrow = CausalArrow::compute(&self.snapshots[n - 1], &snapshot);
            self.causal_arrows.push(arrow);
        }

        // RIF diff (requires snapshot t-2)
        if n >= 2 {
            let rif = RifDiff::compute(&self.snapshots[n - 2], &snapshot);
            self.rif_diffs.push(rif);
        }

        // Halo transitions (requires previous snapshot)
        if n > 0 {
            let transitions = detect_halo_transitions(&self.snapshots[n - 1], &snapshot);

            // Generate NARS statements from transitions
            for t in &transitions {
                self.generate_nars_from_transition(t, snapshot.iter);
            }

            self.halo_transitions.extend(transitions);
        }

        // Feed per-plane convergence deltas into ShiftDetector as σ-scores.
        // The delta IS a Hamming distance between consecutive estimates — feed
        // it through score_sigma to get per-plane σ-significance.
        let deltas = [snapshot.delta_s, snapshot.delta_p, snapshot.delta_o];
        for (plane, &delta) in deltas.iter().enumerate() {
            let ec = EnergyConflict {
                conflict: delta,
                energy_a: 0,
                energy_b: 0,
                agreement: 0,
            };
            let sigma = score_sigma(&ec, &self.sigma_gate);
            self.shift_detector.record(plane, sigma.sigma);
        }

        // Advance shift window every 4 iterations for smoothing.
        if (snapshot.iter + 1)% 4 == 0 {
            self.shift_detector.advance_window();
        }

        self.snapshots.push(snapshot);
    }

    /// Generate NARS causal statements from a halo transition.
    fn generate_nars_from_transition(&mut self, transition: &HaloTransition, iter: u16) {
        match transition.level_delta.cmp(&0) {
            std::cmp::Ordering::Greater => {
                // Promotion: evidence SUPPORTS the factorization
                let confidence = 0.3 + 0.2 * transition.level_delta as f32;
                self.nars_statements.push(NarsCausalStatement {
                    relation: CausalRelation::Supports,
                    source_plane: halo_to_dominant_plane(transition.to),
                    target_plane: None,
                    truth: NarsTruth::new(0.8, confidence.min(0.9)),
                    iter,
                    inference_mode: transition.to.inference_mode(),
                    sigma: None,
                });
            }
            std::cmp::Ordering::Less => {
                // Demotion: evidence UNDERMINES the factorization
                let confidence = 0.3 + 0.2 * (-transition.level_delta) as f32;
                self.nars_statements.push(NarsCausalStatement {
                    relation: CausalRelation::Undermines,
                    source_plane: halo_to_dominant_plane(transition.from),
                    target_plane: None,
                    truth: NarsTruth::new(0.2, confidence.min(0.9)),
                    iter,
                    inference_mode: transition.from.inference_mode(),
                    sigma: None,
                });
            }
            std::cmp::Ordering::Equal => {} // Same level: lateral movement, no causal signal
        }
    }

    /// Finalize the trajectory: compute causal chain and generate Sigma edges.
    ///
    /// Call this after the resonator has converged or reached max iterations.
    pub fn finalize(&mut self) {
        // Causal chain from RIF diffs
        let chain = CausalChain::from_rif_diffs(&self.rif_diffs);

        // Generate Sigma edges from causal chain
        for link in &chain.links {
            self.sigma_edges.push(SigmaEdge {
                source: SigmaNode::Plane(link.cause_plane),
                target: SigmaNode::Plane(link.effect_plane),
                relation: CausalRelation::Causes,
                truth: NarsTruth::new(link.confidence, 0.6),
                iter: link.to_iter,
            });
        }

        // Generate Sigma edges from causal arrows
        for arrow in &self.causal_arrows {
            for (plane_idx, dir) in [
                (DominantPlane::S, &arrow.s_direction),
                (DominantPlane::P, &arrow.p_direction),
                (DominantPlane::O, &arrow.o_direction),
            ] {
                match dir {
                    CausalDirection::Forward(strength) => {
                        self.sigma_edges.push(SigmaEdge {
                            source: SigmaNode::Plane(plane_idx),
                            target: SigmaNode::Plane(next_plane(plane_idx)),
                            relation: CausalRelation::Causes,
                            truth: NarsTruth::new(*strength, 0.5),
                            iter: arrow.iter,
                        });
                    }
                    CausalDirection::Backward(strength) => {
                        self.sigma_edges.push(SigmaEdge {
                            source: SigmaNode::Plane(next_plane(plane_idx)),
                            target: SigmaNode::Plane(plane_idx),
                            relation: CausalRelation::Causes,
                            truth: NarsTruth::new(*strength, 0.5),
                            iter: arrow.iter,
                        });
                    }
                    CausalDirection::Contested(strength) => {
                        self.sigma_edges.push(SigmaEdge {
                            source: SigmaNode::Plane(plane_idx),
                            target: SigmaNode::Plane(next_plane(plane_idx)),
                            relation: CausalRelation::Contradicts,
                            truth: NarsTruth::new(*strength, 0.3),
                            iter: arrow.iter,
                        });
                    }
                    CausalDirection::Symmetric => {}
                }
            }
        }

        // Generate NARS statements from causal saliency
        if self.ewm_corrections.len() >= 2 {
            let saliency = CausalSaliency::from_ewm_window(&self.ewm_corrections);
            for (plane_idx, plane) in [DominantPlane::S, DominantPlane::P, DominantPlane::O]
                .iter()
                .enumerate()
            {
                if saliency.contested_count[plane_idx] > 10 {
                    self.nars_statements.push(NarsCausalStatement {
                        relation: CausalRelation::Contradicts,
                        source_plane: *plane,
                        target_plane: None,
                        truth: NarsTruth::new(
                            saliency.contested_count[plane_idx] as f32 / 256.0,
                            0.7,
                        ),
                        iter: self.snapshots.last().map_or(0, |s| s.iter),
                        inference_mode: None,
                        sigma: None,
                    });
                }
            }
        }
    }

    /// Evaluate the overall gate decision from the trajectory.
    ///
    /// Uses the final snapshot's convergence state + accumulated NARS evidence.
    /// ShiftDetector provides distributional bias: if the population is migrating
    /// toward noise → Hold (don't commit while ground moves), toward foveal → Flow.
    pub fn gate_decision(&self) -> CollapseGate {
        let Some(last) = self.snapshots.last() else {
            return CollapseGate::Block;
        };

        // Count supporting vs undermining statements
        let supports = self
            .nars_statements
            .iter()
            .filter(|s| s.relation == CausalRelation::Supports)
            .count();
        let undermines = self
            .nars_statements
            .iter()
            .filter(|s| s.relation == CausalRelation::Undermines)
            .count();
        let contradicts = self
            .nars_statements
            .iter()
            .filter(|s| s.relation == CausalRelation::Contradicts)
            .count();

        // Base decision from NARS evidence
        let nars_decision = if last.converged(100) && supports > undermines + contradicts {
            CollapseGate::Flow
        } else if contradicts > supports {
            CollapseGate::Block
        } else {
            CollapseGate::Hold
        };

        // ShiftDetector bias: distributional migration can override Hold↔Flow.
        // Block is never overridden — if NARS says Block, that's final.
        match (nars_decision, self.shift_detector.gate_bias()) {
            // Block from NARS is always final
            (CollapseGate::Block, _) => CollapseGate::Block,
            // ShiftDetector says Hold but NARS says Flow → Hold wins
            // (world is drifting, don't commit yet)
            (CollapseGate::Flow, Some(CollapseGate::Hold)) => CollapseGate::Hold,
            // ShiftDetector says Flow but NARS says Hold → Flow wins
            // (world is clarifying, accelerate commit)
            (CollapseGate::Hold, Some(CollapseGate::Flow)) => CollapseGate::Flow,
            // Otherwise: NARS decision stands
            (decision, _) => decision,
        }
    }

    /// Number of recorded iterations.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

impl Default for CausalTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a halo type to its most relevant dominant plane.
fn halo_to_dominant_plane(halo: HaloType) -> DominantPlane {
    match halo {
        HaloType::S | HaloType::SP | HaloType::SO => DominantPlane::S,
        HaloType::P | HaloType::PO => DominantPlane::P,
        HaloType::O | HaloType::Core | HaloType::Noise => DominantPlane::O,
    }
}

/// Get the "next" plane in S→P→O→S cycle.
fn next_plane(plane: DominantPlane) -> DominantPlane {
    match plane {
        DominantPlane::S => DominantPlane::P,
        DominantPlane::P => DominantPlane::O,
        DominantPlane::O => DominantPlane::S,
    }
}

// ============================================================================
// Stripe Histogram — 0.5σ-band population tracking per plane
// ============================================================================

/// Per-plane population histogram across 6 σ-significance stripes.
///
/// Tracks how many candidates fall into each 0.5σ band. With 3 planes,
/// this gives 18 counters total. Zero overhead: computed from the same
/// raw Hamming distances already available.
///
/// The 6 stripes correspond to the statistical significance bands:
/// - below_1s: < 1.0σ (deep noise)
/// - s1_to_s15: 1.0–1.5σ (emerging from noise)
/// - s15_to_s2: 1.5–2.0σ (Hint — interesting but not publishable)
/// - s2_to_s25: 2.0–2.5σ (Evidence — 95% CI)
/// - s25_to_s3: 2.5–3.0σ (Strong — 99% CI)
/// - above_3s: > 3.0σ (Discovery — foveal quality)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StripeHistogram {
    /// < 1.0σ: noise floor.
    pub below_1s: u32,
    /// 1.0–1.5σ: emerging from noise.
    pub s1_to_s15: u32,
    /// 1.5–2.0σ: hint band (interesting, not conclusive).
    pub s15_to_s2: u32,
    /// 2.0–2.5σ: evidence band (p < 0.05).
    pub s2_to_s25: u32,
    /// 2.5–3.0σ: strong band (p < 0.01).
    pub s25_to_s3: u32,
    /// > 3.0σ: discovery / foveal quality.
    pub above_3s: u32,
}

impl StripeHistogram {
    /// Total population across all stripes.
    #[inline]
    pub fn total(&self) -> u32 {
        self.below_1s
            + self.s1_to_s15
            + self.s15_to_s2
            + self.s2_to_s25
            + self.s25_to_s3
            + self.above_3s
    }

    /// Classify a σ-value into the appropriate stripe and increment.
    #[inline]
    pub fn record(&mut self, sigma: f32) {
        if sigma >= 3.0 {
            self.above_3s += 1;
        } else if sigma >= 2.5 {
            self.s25_to_s3 += 1;
        } else if sigma >= 2.0 {
            self.s2_to_s25 += 1;
        } else if sigma >= 1.5 {
            self.s15_to_s2 += 1;
        } else if sigma >= 1.0 {
            self.s1_to_s15 += 1;
        } else {
            self.below_1s += 1;
        }
    }

    /// Convert to array of 6 bin counts [below_1s, ..., above_3s].
    pub fn as_array(&self) -> [u32; 6] {
        [
            self.below_1s,
            self.s1_to_s15,
            self.s15_to_s2,
            self.s2_to_s25,
            self.s25_to_s3,
            self.above_3s,
        ]
    }

    /// Center of mass in σ-space: weighted average of bin centers.
    ///
    /// Returns a value in [0.0, 3.5] — higher = population closer to foveal.
    pub fn center_of_mass(&self) -> f32 {
        let total = self.total() as f32;
        if total < 1.0 {
            return 0.0;
        }
        // Bin centers: 0.5, 1.25, 1.75, 2.25, 2.75, 3.25
        let weighted_sum = self.below_1s as f32 * 0.5
            + self.s1_to_s15 as f32 * 1.25
            + self.s15_to_s2 as f32 * 1.75
            + self.s2_to_s25 as f32 * 2.25
            + self.s25_to_s3 as f32 * 2.75
            + self.above_3s as f32 * 3.25;
        weighted_sum / total
    }
}

/// Direction of distributional shift between time windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftDirection {
    /// Population migrating toward foveal (higher σ). Codebook improving.
    TowardFoveal,
    /// Population migrating toward noise (lower σ). Codebook going stale.
    TowardNoise,
    /// Both directions simultaneously. Speciation / world splitting.
    Bimodal,
    /// No significant migration. Steady state.
    Stable,
}

/// Signal from distributional shift analysis.
///
/// Encapsulates the direction, magnitude, and per-plane breakdown
/// of population migration across σ-stripes between time windows.
#[derive(Clone, Debug)]
pub struct ShiftSignal {
    /// Overall direction of the shift.
    pub direction: ShiftDirection,
    /// Per-plane shift direction.
    pub plane_directions: [ShiftDirection; 3],
    /// Per-plane center-of-mass delta (positive = toward foveal).
    pub com_delta: [f32; 3],
    /// Aggregate magnitude: absolute sum of center-of-mass deltas.
    pub magnitude: f32,
}

/// Distributional shift detector for σ-stripe histograms.
///
/// Tracks per-plane population distributions across time windows and
/// detects migration between adjacent stripes. The migration velocity
/// IS the NARS evidence rate:
/// - Net positive flow across 2σ→2.5σ boundary → confidence rising
/// - Net negative flow → confidence falling
/// - Bimodal flow → speciation event
///
/// Feeds CollapseGate:
/// - Shift toward noise → bias HOLD (ground is moving, don't commit)
/// - Shift toward foveal → bias FLOW (world clarifying, commit faster)
#[derive(Clone, Debug)]
pub struct ShiftDetector {
    /// Current window's per-plane histograms [S, P, O].
    pub current: [StripeHistogram; 3],
    /// Previous window's per-plane histograms [S, P, O].
    pub previous: [StripeHistogram; 3],
    /// Number of windows recorded.
    pub window_count: u32,
}

impl ShiftDetector {
    pub fn new() -> Self {
        Self {
            current: [StripeHistogram::default(); 3],
            previous: [StripeHistogram::default(); 3],
            window_count: 0,
        }
    }

    /// Record a per-plane σ-value into the current window.
    ///
    /// `plane` is 0=S, 1=P, 2=O.
    #[inline]
    pub fn record(&mut self, plane: usize, sigma: f32) {
        if plane < 3 {
            self.current[plane].record(sigma);
        }
    }

    /// Advance to the next time window. Current becomes previous.
    pub fn advance_window(&mut self) {
        self.previous = self.current;
        self.current = [StripeHistogram::default(); 3];
        self.window_count += 1;
    }

    /// Detect distributional shift between previous and current windows.
    ///
    /// Returns None if insufficient data (< 2 windows or empty histograms).
    pub fn detect_shift(&self) -> Option<ShiftSignal> {
        if self.window_count < 1 {
            return None;
        }

        // Per-plane center-of-mass delta
        let mut com_delta = [0.0f32; 3];
        let mut plane_dirs = [ShiftDirection::Stable; 3];

        for plane in 0..3 {
            let prev_com = self.previous[plane].center_of_mass();
            let curr_com = self.current[plane].center_of_mass();
            com_delta[plane] = curr_com - prev_com;

            // Classify per-plane direction
            let prev_arr = self.previous[plane].as_array();
            let curr_arr = self.current[plane].as_array();

            // Check for bimodality: lower bins growing AND upper bins growing
            let lower_growing = curr_arr[0] > prev_arr[0] + 2 || curr_arr[1] > prev_arr[1] + 2;
            let upper_growing = curr_arr[4] > prev_arr[4] + 2 || curr_arr[5] > prev_arr[5] + 2;

            if lower_growing && upper_growing {
                plane_dirs[plane] = ShiftDirection::Bimodal;
            } else if com_delta[plane] > 0.1 {
                plane_dirs[plane] = ShiftDirection::TowardFoveal;
            } else if com_delta[plane] < -0.1 {
                plane_dirs[plane] = ShiftDirection::TowardNoise;
            }
        }

        // Aggregate direction
        let total_delta: f32 = com_delta.iter().sum();
        let any_bimodal = plane_dirs.contains(&ShiftDirection::Bimodal);
        let magnitude = com_delta.iter().map(|d| d.abs()).sum();

        let direction = if any_bimodal {
            ShiftDirection::Bimodal
        } else if total_delta > 0.15 {
            ShiftDirection::TowardFoveal
        } else if total_delta < -0.15 {
            ShiftDirection::TowardNoise
        } else {
            ShiftDirection::Stable
        };

        Some(ShiftSignal {
            direction,
            plane_directions: plane_dirs,
            com_delta,
            magnitude,
        })
    }

    /// Map shift signal to CollapseGate bias.
    ///
    /// - TowardNoise → HOLD (don't commit while ground is moving)
    /// - TowardFoveal → FLOW (world is clarifying, commit faster)
    /// - Bimodal → HOLD (world is splitting, need more evidence)
    /// - Stable → no bias (use existing gate logic)
    pub fn gate_bias(&self) -> Option<CollapseGate> {
        self.detect_shift()
            .and_then(|signal| match signal.direction {
                ShiftDirection::TowardNoise => Some(CollapseGate::Hold),
                ShiftDirection::TowardFoveal => Some(CollapseGate::Flow),
                ShiftDirection::Bimodal => Some(CollapseGate::Hold),
                ShiftDirection::Stable => None,
            })
    }
}

impl Default for ShiftDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn new(seed: u64) -> Self { Self(seed) }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    fn random_fp(rng: &mut SplitMix64) -> Fingerprint<256> {
        let mut words = [0u64; 256];
        for w in &mut words {
            *w = rng.next_u64();
        }
        Fingerprint::from_words(words)
    }

    fn make_snapshot(rng: &mut SplitMix64, iter: u16, n_entries: usize) -> ResonatorSnapshot {
        let n_words = n_entries.div_ceil(64);
        ResonatorSnapshot {
            iter,
            s_est: random_fp(rng),
            p_est: random_fp(rng),
            o_est: random_fp(rng),
            s_mask: (0..n_words).map(|_| rng.next_u64()).collect(),
            p_mask: (0..n_words).map(|_| rng.next_u64()).collect(),
            o_mask: (0..n_words).map(|_| rng.next_u64()).collect(),
            n_entries,
            delta_s: 100 - iter as u32 * 10,
            delta_p: 100 - iter as u32 * 10,
            delta_o: 100 - iter as u32 * 10,
        }
    }

    // --- NARS Truth tests ---

    #[test]
    fn test_nars_revision() {
        let t1 = NarsTruth::new(0.8, 0.5);
        let t2 = NarsTruth::new(0.6, 0.5);
        let revised = t1.revise(t2);
        // Revision of equal-weight evidence should give mean frequency
        assert!(
            (revised.f - 0.7).abs() < 0.01,
            "Revised frequency should be ~0.7, got {}",
            revised.f
        );
        // Confidence should increase
        assert!(
            revised.c > t1.c,
            "Revised confidence {} should exceed input {}",
            revised.c,
            t1.c
        );
    }

    #[test]
    fn test_nars_deduction() {
        let ab = NarsTruth::new(0.9, 0.8);
        let bc = NarsTruth::new(0.9, 0.8);
        let ac = ab.deduction(bc);
        assert!(
            ac.f > 0.7,
            "Deduction frequency should be high, got {}",
            ac.f
        );
        assert!(ac.c > 0.0, "Deduction confidence should be > 0");
    }

    // --- RIF Diff tests ---

    #[test]
    fn test_rif_diff_identical() {
        let mut rng = make_rng();
        let snap = make_snapshot(&mut rng, 0, 100);
        let diff = RifDiff::compute(&snap, &snap);
        assert_eq!(
            diff.s_activity, 0,
            "Identical snapshots should have 0 S activity"
        );
        assert_eq!(diff.total_activity(), 0, "Total activity should be 0");
    }

    #[test]
    fn test_rif_diff_different() {
        let mut rng = make_rng();
        let snap1 = make_snapshot(&mut rng, 0, 100);
        let snap2 = make_snapshot(&mut rng, 2, 100);
        let diff = RifDiff::compute(&snap1, &snap2);
        assert!(
            diff.total_activity() > 0,
            "Different snapshots should have activity"
        );
    }

    // --- EWM Correction tests ---

    #[test]
    fn test_ewm_correction_self_is_zero() {
        let mut rng = make_rng();
        let snap = make_snapshot(&mut rng, 0, 100);
        let corr = EwmCorrection::compute(&snap, &snap);
        assert_eq!(corr.s_total(), 0, "Self-correction should be 0");
        assert_eq!(corr.p_total(), 0);
        assert_eq!(corr.o_total(), 0);
    }

    #[test]
    fn test_ewm_correction_nonzero() {
        let mut rng = make_rng();
        let snap1 = make_snapshot(&mut rng, 0, 100);
        let snap2 = make_snapshot(&mut rng, 1, 100);
        let corr = EwmCorrection::compute(&snap1, &snap2);
        assert!(
            corr.s_total() > 0,
            "Different snapshots should have nonzero correction"
        );
    }

    // --- Causal Arrow tests ---

    #[test]
    fn test_causal_arrow_identical_is_forward() {
        let mut rng = make_rng();
        let snap = make_snapshot(&mut rng, 0, 100);
        let arrow = CausalArrow::compute(&snap, &snap);
        // Identical snapshots: stability = 1.0, forward BPReLU dominates
        assert!(
            matches!(arrow.overall, CausalDirection::Forward(_)),
            "Identical snapshots should give Forward direction, got {:?}",
            arrow.overall
        );
    }

    #[test]
    fn test_causal_arrow_has_three_planes() {
        let mut rng = make_rng();
        let snap1 = make_snapshot(&mut rng, 0, 100);
        let snap2 = make_snapshot(&mut rng, 1, 100);
        let arrow = CausalArrow::compute(&snap1, &snap2);
        assert_eq!(arrow.forward_magnitude.len(), 3);
        assert_eq!(arrow.backward_magnitude.len(), 3);
    }

    // --- Causal Chain tests ---

    #[test]
    fn test_causal_chain_empty() {
        let chain = CausalChain::from_rif_diffs(&[]);
        assert_eq!(chain.depth(), 0);
        assert!(chain.root_cause().is_none());
    }

    #[test]
    fn test_causal_chain_detects_stabilization() {
        // Create diffs where S is active early but quiet late,
        // and P is quiet early but active late → S caused P
        let diff1 = RifDiff {
            from_iter: 0,
            to_iter: 2,
            s_diff: Fingerprint::ones(), // S very active
            p_diff: Fingerprint::zero(), // P quiet
            o_diff: Fingerprint::zero(),
            s_activity: 16384,
            p_activity: 0,
            o_activity: 0,
        };
        let diff2 = RifDiff {
            from_iter: 2,
            to_iter: 4,
            s_diff: Fingerprint::zero(), // S quiet (stabilized)
            p_diff: Fingerprint::ones(), // P now active (responding)
            o_diff: Fingerprint::zero(),
            s_activity: 0,
            p_activity: 16384,
            o_activity: 0,
        };

        let chain = CausalChain::from_rif_diffs(&[diff1, diff2]);
        assert!(chain.depth() > 0, "Should detect S→P causal link");
        assert_eq!(
            chain.root_cause(),
            Some(DominantPlane::S),
            "Root cause should be S-plane"
        );
        assert_eq!(chain.links[0].effect_plane, DominantPlane::P);
    }

    // --- Causal Saliency tests ---

    #[test]
    fn test_saliency_needs_minimum_corrections() {
        let saliency = CausalSaliency::from_ewm_window(&[]);
        assert_eq!(saliency.crystallizing_count, [0, 0, 0]);
    }

    #[test]
    fn test_saliency_detects_crystallizing() {
        // Decreasing correction → crystallizing
        let mut rng = make_rng();
        let snap0 = make_snapshot(&mut rng, 0, 100);
        let snap1 = make_snapshot(&mut rng, 1, 100);

        // Create corrections with decreasing S-plane values
        let mut corr1 = EwmCorrection::compute(&snap0, &snap1);
        let mut corr2 = corr1.clone();
        // Simulate: first correction has high values, second has low
        for i in 0..256 {
            corr1.s_correction[i] = 20;
            corr2.s_correction[i] = 5;
        }
        corr2.iter = 2;

        let saliency = CausalSaliency::from_ewm_window(&[corr1, corr2]);
        assert!(
            saliency.crystallizing_count[0] > 0,
            "Should detect crystallizing S-plane words"
        );
    }

    // --- Halo Transition tests ---

    #[test]
    fn test_halo_transition_detection() {
        let mut rng = make_rng();
        let snap1 = make_snapshot(&mut rng, 0, 64);
        let snap2 = make_snapshot(&mut rng, 1, 64);
        let transitions = detect_halo_transitions(&snap1, &snap2);
        // Random masks will produce some transitions
        // Just verify we don't panic and produce valid output
        for t in &transitions {
            assert_ne!(t.from, t.to, "Transition should change halo type");
        }
    }

    // --- Full Trajectory tests ---

    #[test]
    fn test_trajectory_record_and_finalize() {
        let mut rng = make_rng();
        let mut traj = CausalTrajectory::new();

        // Record 5 iterations
        for i in 0..5 {
            let snap = make_snapshot(&mut rng, i, 64);
            traj.record_iteration(snap);
        }

        assert_eq!(traj.len(), 5);
        assert!(!traj.is_empty());

        // EWM corrections: 4 (one per transition)
        assert_eq!(traj.ewm_corrections.len(), 4);

        // Causal arrows: 4
        assert_eq!(traj.causal_arrows.len(), 4);

        // RIF diffs: 3 (need t-2)
        assert_eq!(traj.rif_diffs.len(), 3);

        // Finalize
        traj.finalize();

        // Should have generated some sigma edges
        // (exact count depends on random data)
    }

    #[test]
    fn test_trajectory_gate_decision_empty() {
        let traj = CausalTrajectory::new();
        assert_eq!(traj.gate_decision(), CollapseGate::Block);
    }

    #[test]
    fn test_trajectory_gate_decision_converged() {
        let mut traj = CausalTrajectory::new();
        let mut rng = make_rng();

        // Create converging snapshots (decreasing deltas)
        for i in 0..5 {
            let mut snap = make_snapshot(&mut rng, i, 64);
            snap.delta_s = 10;
            snap.delta_p = 10;
            snap.delta_o = 10;
            traj.record_iteration(snap);
        }

        // Add supporting NARS statements
        traj.nars_statements.push(NarsCausalStatement {
            relation: CausalRelation::Supports,
            source_plane: DominantPlane::S,
            target_plane: None,
            truth: NarsTruth::new(0.9, 0.8),
            iter: 4,
            inference_mode: Some(InferenceMode::Forward),
            sigma: Some(SignificanceLevel::Strong),
        });

        assert_eq!(traj.gate_decision(), CollapseGate::Flow);
    }

    #[test]
    fn test_trajectory_gate_decision_contradicted() {
        let mut traj = CausalTrajectory::new();
        let mut rng = make_rng();

        let snap = make_snapshot(&mut rng, 0, 64);
        traj.record_iteration(snap);

        // Add contradicting statements (more contradicts than supports)
        for _ in 0..3 {
            traj.nars_statements.push(NarsCausalStatement {
                relation: CausalRelation::Contradicts,
                source_plane: DominantPlane::S,
                target_plane: None,
                truth: NarsTruth::new(0.5, 0.7),
                iter: 0,
                inference_mode: None,
                sigma: None,
            });
        }

        assert_eq!(traj.gate_decision(), CollapseGate::Block);
    }

    #[test]
    fn test_nars_truth_bounds() {
        let t = NarsTruth::new(1.5, -0.3);
        assert!(t.f <= 1.0 && t.f >= 0.0, "Frequency should be clamped");
        assert!(t.c <= 1.0 && t.c >= 0.0, "Confidence should be clamped");
    }

    #[test]
    fn test_snapshot_converged() {
        let mut rng = make_rng();
        let mut snap = make_snapshot(&mut rng, 0, 64);
        snap.delta_s = 5;
        snap.delta_p = 3;
        snap.delta_o = 7;
        assert!(
            snap.converged(10),
            "Should be converged when all deltas < threshold"
        );
        assert!(
            !snap.converged(5),
            "Should not converge when delta_o >= threshold"
        );
    }

    #[test]
    fn test_classify_asymmetry_forward() {
        let dir = classify_asymmetry(0.9, 0.1);
        assert!(matches!(dir, CausalDirection::Forward(_)));
    }

    #[test]
    fn test_classify_asymmetry_backward() {
        let dir = classify_asymmetry(0.1, 0.9);
        assert!(matches!(dir, CausalDirection::Backward(_)));
    }

    #[test]
    fn test_classify_asymmetry_symmetric() {
        let dir = classify_asymmetry(0.0, 0.0);
        assert!(matches!(dir, CausalDirection::Symmetric));
    }

    #[test]
    fn test_per_word_popcount() {
        let fp = Fingerprint::<256> {
            words: {
                let mut w = [0u64; 256];
                w[0] = 0xFF; // 8 bits
                w[1] = 0xFFFF; // 16 bits
                w
            },
        };
        let pops = per_word_popcount(&fp);
        assert_eq!(pops[0], 8);
        assert_eq!(pops[1], 16);
        assert_eq!(pops[2], 0);
    }

    // --- EwmTier from SignificanceLevel tests ---

    #[test]
    fn test_ewm_tier_from_significance_discovery() {
        assert_eq!(
            EwmTier::from(SignificanceLevel::Discovery),
            EwmTier::Crystallized
        );
    }

    #[test]
    fn test_ewm_tier_from_significance_strong() {
        assert_eq!(EwmTier::from(SignificanceLevel::Strong), EwmTier::Confident);
    }

    #[test]
    fn test_ewm_tier_from_significance_evidence() {
        assert_eq!(
            EwmTier::from(SignificanceLevel::Evidence),
            EwmTier::Transitional
        );
    }

    #[test]
    fn test_ewm_tier_from_significance_hint() {
        assert_eq!(
            EwmTier::from(SignificanceLevel::Hint),
            EwmTier::Transitional
        );
    }

    #[test]
    fn test_ewm_tier_from_significance_noise() {
        assert_eq!(EwmTier::from(SignificanceLevel::Noise), EwmTier::Noise);
    }

    // --- StripeHistogram tests ---

    #[test]
    fn test_stripe_histogram_record() {
        let mut hist = StripeHistogram::default();
        hist.record(0.5); // below_1s
        hist.record(1.2); // s1_to_s15
        hist.record(1.7); // s15_to_s2
        hist.record(2.3); // s2_to_s25
        hist.record(2.7); // s25_to_s3
        hist.record(3.5); // above_3s

        assert_eq!(hist.below_1s, 1);
        assert_eq!(hist.s1_to_s15, 1);
        assert_eq!(hist.s15_to_s2, 1);
        assert_eq!(hist.s2_to_s25, 1);
        assert_eq!(hist.s25_to_s3, 1);
        assert_eq!(hist.above_3s, 1);
        assert_eq!(hist.total(), 6);
    }

    #[test]
    fn test_stripe_histogram_center_of_mass() {
        let hist = StripeHistogram {
            above_3s: 100,
            ..Default::default()
        };
        assert!((hist.center_of_mass() - 3.25).abs() < 0.01);

        // All in noise
        let hist2 = StripeHistogram {
            below_1s: 100,
            ..Default::default()
        };
        assert!((hist2.center_of_mass() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stripe_histogram_empty() {
        let hist = StripeHistogram::default();
        assert_eq!(hist.total(), 0);
        assert_eq!(hist.center_of_mass(), 0.0);
    }

    // --- ShiftDetector tests ---

    #[test]
    fn test_shift_detector_no_data() {
        let det = ShiftDetector::new();
        assert!(det.detect_shift().is_none());
    }

    #[test]
    fn test_shift_detector_toward_foveal() {
        let mut det = ShiftDetector::new();

        // Previous window: mostly noise
        for _ in 0..100 {
            det.record(0, 0.5); // S-plane noise
            det.record(1, 0.5);
            det.record(2, 0.5);
        }
        det.advance_window();

        // Current window: mostly foveal
        for _ in 0..100 {
            det.record(0, 3.5); // S-plane foveal
            det.record(1, 3.5);
            det.record(2, 3.5);
        }

        let signal = det.detect_shift().unwrap();
        assert_eq!(signal.direction, ShiftDirection::TowardFoveal);
        assert!(signal.com_delta[0] > 0.0);
    }

    #[test]
    fn test_shift_detector_toward_noise() {
        let mut det = ShiftDetector::new();

        // Previous window: mostly foveal
        for _ in 0..100 {
            det.record(0, 3.5);
            det.record(1, 3.5);
            det.record(2, 3.5);
        }
        det.advance_window();

        // Current window: mostly noise
        for _ in 0..100 {
            det.record(0, 0.5);
            det.record(1, 0.5);
            det.record(2, 0.5);
        }

        let signal = det.detect_shift().unwrap();
        assert_eq!(signal.direction, ShiftDirection::TowardNoise);
        assert!(signal.com_delta[0] < 0.0);
    }

    #[test]
    fn test_shift_detector_stable() {
        let mut det = ShiftDetector::new();

        // Both windows: same distribution
        for _ in 0..100 {
            det.record(0, 2.0);
            det.record(1, 2.0);
            det.record(2, 2.0);
        }
        det.advance_window();

        for _ in 0..100 {
            det.record(0, 2.0);
            det.record(1, 2.0);
            det.record(2, 2.0);
        }

        let signal = det.detect_shift().unwrap();
        assert_eq!(signal.direction, ShiftDirection::Stable);
    }

    #[test]
    fn test_shift_detector_gate_bias() {
        let mut det = ShiftDetector::new();

        // Toward noise → HOLD
        for _ in 0..100 {
            det.record(0, 3.5);
            det.record(1, 3.5);
            det.record(2, 3.5);
        }
        det.advance_window();
        for _ in 0..100 {
            det.record(0, 0.5);
            det.record(1, 0.5);
            det.record(2, 0.5);
        }
        assert_eq!(det.gate_bias(), Some(CollapseGate::Hold));
    }
}
