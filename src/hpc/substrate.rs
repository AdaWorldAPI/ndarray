//! Substrate routing types for multi-substrate cognitive processing.
//!
//! The cognitive architecture has four computational substrates, each answering
//! a different question about the same concept:
//!
//! - **Structural**: Binary Hamming — "Have I seen this exact pattern before?"
//! - **Soaking**: BCM organic registers — "Is this concept still forming or settled?"
//! - **Evidential**: NARS truth values — "How much should I trust this?"
//! - **Semantic**: f32 embeddings — "What else is this like?"
//!
//! This module provides the TYPES for substrate routing. The actual
//! ThinkingStyle → SubstrateRoute mapping lives upstream (crewai-rust)
//! because rustynum never imports style types. rustynum is Level 1 (Surface).
//!
//! # Architecture
//!
//! ```text
//! [crewai-rust: ThinkingStyle]
//!        ↓ maps to
//! [rustynum-core: SubstrateRoute]
//!        ↓ dispatches to
//! [rustynum-core: Substrate computation]
//!        ↓ emits
//! [rustynum-core: SubstrateSignals]
//!        ↓ feeds
//! [rustynum-core: CollapseGate decision]
//!        ↓ optionally recommends
//! [rustynum-core: Substrate transition]
//! ```
//!
//! # Zero IO
//!
//! All types and functions are pure compute / pure data. No allocation
//! beyond return values. No IO.

use super::organic::SynapseState;

// ---------------------------------------------------------------------------
// Substrate enum
// ---------------------------------------------------------------------------

/// Which computational substrate to route through.
///
/// Each substrate answers a fundamentally different question:
/// - Structural: exact match via binary Hamming distance
/// - Soaking: evidence accumulation via BCM plasticity
/// - Evidential: trust estimation via NARS (f,c,k) truth values
/// - Semantic: analogy finding via f32 cosine similarity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Substrate {
    /// 3×16384-bit binary: XOR bind, Hamming distance, CAM addressing.
    /// Fast, exact, no gradation. Best for known patterns.
    Structural = 0,
    /// 3×10000D organic: BCM θ soaking, evidence accumulation.
    /// Slow, adaptive, self-normalizing. Best for learning new concepts.
    Soaking = 1,
    /// NARS (f,c,k): evidential reasoning, revision, doubt.
    /// Explicit uncertainty tracking. Best for competing hypotheses.
    Evidential = 2,
    /// f32×1024 Jina: cosine similarity, semantic neighbors.
    /// Continuous, dense. Best for distant analogies and creative leaps.
    Semantic = 3,
}

impl Substrate {
    /// All four substrates in priority order.
    pub const ALL: [Substrate; 4] = [
        Substrate::Structural,
        Substrate::Soaking,
        Substrate::Evidential,
        Substrate::Semantic,
    ];

    /// Convert from raw u8.
    #[inline]
    pub fn from_raw(v: u8) -> Option<Self> {
        match v {
            0 => Some(Substrate::Structural),
            1 => Some(Substrate::Soaking),
            2 => Some(Substrate::Evidential),
            3 => Some(Substrate::Semantic),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SubstrateRoute
// ---------------------------------------------------------------------------

/// Routing priority for cognitive processing.
///
/// Determines which substrate does the heavy lifting (primary),
/// which is consulted for cross-check (secondary), and which
/// handles edge cases (tertiary).
///
/// Created by upstream code (crewai-rust ThinkingStyle mapping)
/// and consumed by the cortex dispatch loop.
#[derive(Clone, Debug)]
pub struct SubstrateRoute {
    /// Primary substrate: does the main computation.
    pub primary: Substrate,
    /// Secondary substrate: consulted for cross-check (if any).
    pub secondary: Option<Substrate>,
    /// Tertiary substrate: used for edge cases (if any).
    pub tertiary: Option<Substrate>,
    /// Weight for primary vs secondary blending.
    /// 0.0 = all secondary, 1.0 = all primary.
    pub primary_weight: f32,
    /// Whether to run substrates in parallel or sequentially.
    /// Parallel: all substrates run concurrently (wider but more expensive).
    /// Sequential: primary first, secondary only if needed (faster, narrower).
    pub parallel: bool,
}

impl SubstrateRoute {
    /// Create a single-substrate route (e.g., Focused/Intuitive styles).
    #[inline]
    pub fn single(substrate: Substrate) -> Self {
        Self {
            primary: substrate,
            secondary: None,
            tertiary: None,
            primary_weight: 1.0,
            parallel: false,
        }
    }

    /// Create a dual-substrate route.
    #[inline]
    pub fn dual(
        primary: Substrate,
        secondary: Substrate,
        primary_weight: f32,
        parallel: bool,
    ) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            tertiary: None,
            primary_weight,
            parallel,
        }
    }

    /// Create a triple-substrate route.
    #[inline]
    pub fn triple(
        primary: Substrate,
        secondary: Substrate,
        tertiary: Substrate,
        primary_weight: f32,
        parallel: bool,
    ) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            tertiary: Some(tertiary),
            primary_weight,
            parallel,
        }
    }

    /// All substrates this route touches (in priority order).
    pub fn substrates(&self) -> Vec<Substrate> {
        let mut out = vec![self.primary];
        if let Some(s) = self.secondary {
            out.push(s);
        }
        if let Some(s) = self.tertiary {
            out.push(s);
        }
        out
    }

    /// Number of active substrates in this route.
    #[inline]
    pub fn depth(&self) -> usize {
        1 + self.secondary.is_some() as usize + self.tertiary.is_some() as usize
    }

    /// Whether this route uses a specific substrate.
    #[inline]
    pub fn uses(&self, substrate: Substrate) -> bool {
        self.primary == substrate
            || self.secondary == Some(substrate)
            || self.tertiary == Some(substrate)
    }
}

// ---------------------------------------------------------------------------
// SubstrateSignals
// ---------------------------------------------------------------------------

/// Signals emitted by substrates during processing.
///
/// After each processing cycle, the active substrates report their
/// observations. These signals feed into the CollapseGate for
/// gate decisions and style transition recommendations.
///
/// All fields are optional (defaulting to 0/0.0) because not all
/// substrates are active in every cycle.
#[derive(Clone, Debug, Default)]
pub struct SubstrateSignals {
    // === STRUCTURAL signals ===
    /// Number of Hamming matches within σ-2 band.
    pub structural_hits: usize,
    /// Hamming distance to nearest match (0 = exact match).
    pub structural_nearest_distance: u32,

    // === SOAKING signals ===
    /// Average saturation ratio across active soaking registers.
    /// 0.0 = no saturation, 1.0 = all dimensions saturated.
    pub soaking_saturation: f32,
    /// Average BCM θ across active dimensions.
    pub theta_average: f32,
    /// Average maturity across active dimensions.
    pub maturity_average: f32,

    // === EVIDENTIAL signals ===
    /// Weighted confidence across active NARS statements.
    pub nars_confidence: f32,
    /// NARS frequency (truth expectation).
    pub nars_frequency: f32,
    /// Number of active contradictions (competing hypotheses).
    pub nars_contradictions: usize,
    /// Total evidence count (k parameter).
    pub nars_evidence_count: u32,

    // === SEMANTIC signals ===
    /// Cosine similarity to nearest known concept.
    pub semantic_nearest: f32,
    /// Number of concepts within cosine 0.3 (the "neighborhood").
    pub semantic_neighborhood_size: usize,
}

impl SubstrateSignals {
    /// Create empty signals (no substrate has reported yet).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any structural substrate reported results.
    #[inline]
    pub fn has_structural(&self) -> bool {
        self.structural_hits > 0 || self.structural_nearest_distance > 0
    }

    /// Whether soaking substrate reported results.
    #[inline]
    pub fn has_soaking(&self) -> bool {
        self.soaking_saturation > 0.0 || self.theta_average > 0.0
    }

    /// Whether evidential substrate reported results.
    #[inline]
    pub fn has_evidential(&self) -> bool {
        self.nars_confidence > 0.0 || self.nars_evidence_count > 0
    }

    /// Whether semantic substrate reported results.
    #[inline]
    pub fn has_semantic(&self) -> bool {
        self.semantic_nearest > 0.0 || self.semantic_neighborhood_size > 0
    }

    /// Count how many substrates reported results.
    pub fn active_count(&self) -> usize {
        self.has_structural() as usize
            + self.has_soaking() as usize
            + self.has_evidential() as usize
            + self.has_semantic() as usize
    }
}

// ---------------------------------------------------------------------------
// Cross-substrate coherence
// ---------------------------------------------------------------------------

/// Coherence classification: do the active substrates agree?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coherence {
    /// All active substrates agree (e.g., structural hit + high NARS confidence).
    Convergent,
    /// Active substrates partially agree (some agree, some neutral).
    Partial,
    /// Active substrates disagree (e.g., structural miss but semantic hit).
    Divergent,
    /// Only one substrate active — no coherence to measure.
    Singular,
}

/// Compute cross-substrate coherence from signals.
///
/// Checks whether the active substrates' signals point in the same direction:
/// - Structural hit + NARS confidence → convergent
/// - Structural miss + semantic hit → divergent (analogy vs. exact match)
/// - Soaking saturated + low NARS confidence → divergent (forming but untrusted)
pub fn coherence(signals: &SubstrateSignals) -> Coherence {
    let active = signals.active_count();
    if active <= 1 {
        return Coherence::Singular;
    }

    let mut agree = 0usize;
    let mut disagree = 0usize;

    // Structural vs. NARS: do exact matches align with confidence?
    if signals.has_structural() && signals.has_evidential() {
        if signals.structural_hits > 0 && signals.nars_confidence > 0.5 {
            agree += 1; // both say "known and trusted"
        } else if signals.structural_hits == 0 && signals.nars_confidence < 0.3 {
            agree += 1; // both say "unknown"
        } else {
            disagree += 1; // one says known, other says untrusted
        }
    }

    // Structural vs. Semantic: exact match vs. analogy
    if signals.has_structural() && signals.has_semantic() {
        if signals.structural_hits > 0 && signals.semantic_nearest > 0.5 {
            agree += 1; // exact match confirmed by semantic similarity
        } else if signals.structural_hits == 0 && signals.semantic_nearest < 0.2 {
            agree += 1; // both say truly novel
        } else if signals.structural_hits == 0 && signals.semantic_nearest > 0.5 {
            disagree += 1; // no exact match but semantically similar → analogy
        } else {
            // Ambiguous
        }
    }

    // Soaking vs. NARS: formation state vs. confidence
    if signals.has_soaking() && signals.has_evidential() {
        if signals.soaking_saturation > 0.85 && signals.nars_confidence > 0.7 {
            agree += 1; // saturated and trusted → ready to crystallize
        } else if signals.soaking_saturation < 0.3 && signals.nars_confidence < 0.3 {
            agree += 1; // both say "still forming, no confidence"
        } else if signals.soaking_saturation > 0.85 && signals.nars_confidence < 0.3 {
            disagree += 1; // saturated but not trusted — suspicious
        } else {
            // Ambiguous
        }
    }

    // Soaking vs. Semantic: formation state vs. neighborhood
    if signals.has_soaking() && signals.has_semantic() {
        if signals.soaking_saturation > 0.7 && signals.semantic_nearest > 0.5 {
            agree += 1; // forming and has neighbors → reinforcement
        } else if signals.soaking_saturation < 0.2 && signals.semantic_nearest < 0.2 {
            agree += 1; // new and isolated → genuinely novel
        } else {
            // Ambiguous
        }
    }

    if disagree == 0 && agree > 0 {
        Coherence::Convergent
    } else if agree > disagree {
        Coherence::Partial
    } else {
        Coherence::Divergent
    }
}

// ---------------------------------------------------------------------------
// Substrate transition recommendation
// ---------------------------------------------------------------------------

/// Recommended substrate transition based on cross-substrate signals.
///
/// This is a PURE FUNCTION: signals in → substrate suggestion out.
/// The actual ThinkingStyle transition lives upstream (crewai-rust),
/// but this provides the substrate-level recommendation.
///
/// Returns `None` if no transition is warranted.
pub fn recommend_transition(
    current_primary: Substrate,
    signals: &SubstrateSignals,
) -> Option<Substrate> {
    // === CRYSTALLIZATION: soaking saturated → move to structural ===
    if signals.soaking_saturation > 0.85
        && signals.theta_average > 100.0
        && (current_primary == Substrate::Soaking || current_primary == Substrate::Semantic)
    {
        return Some(Substrate::Structural); // ready to crystallize
    }

    // === DOUBT: NARS confidence dropping → move to evidential ===
    if signals.nars_confidence < 0.3
        && signals.nars_contradictions > 2
        && current_primary == Substrate::Structural
    {
        return Some(Substrate::Evidential); // need to re-examine
    }

    // === NOVELTY: no structural or semantic match → move to soaking ===
    if signals.structural_hits == 0
        && signals.semantic_nearest < 0.2
        && (current_primary == Substrate::Structural || current_primary == Substrate::Evidential)
    {
        return Some(Substrate::Soaking); // nothing known, start soaking
    }

    // === COMPLEXITY: large neighborhood + contradictions → evidential ===
    // (checked before association: contradictions outrank analogy)
    if signals.semantic_neighborhood_size > 10
        && signals.nars_contradictions > 1
        && current_primary != Substrate::Evidential
    {
        return Some(Substrate::Evidential); // complex situation, weigh carefully
    }

    // === ASSOCIATION: structural miss but semantic hit → move to semantic ===
    if signals.structural_hits == 0
        && signals.semantic_nearest > 0.5
        && current_primary == Substrate::Structural
    {
        return Some(Substrate::Semantic); // try analogy instead
    }

    // === CONVERGENCE: multiple structural hits + high confidence → lock in ===
    if signals.structural_hits > 3
        && signals.nars_confidence > 0.8
        && (current_primary == Substrate::Soaking || current_primary == Substrate::Semantic)
    {
        return Some(Substrate::Structural); // enough evidence, commit
    }

    None // no transition needed
}

// ---------------------------------------------------------------------------
// SubstrateSnapshot — provenance at crystallization time
// ---------------------------------------------------------------------------

/// Snapshot of substrate state at crystallization time.
///
/// When a concept crystallizes (gate → FLOW), this snapshot captures
/// WHAT the substrates observed. Carried as provenance on the stored atom.
///
/// This is the rustynum-side provenance. The full ThinkingAtom (which adds
/// ThinkingStyle, MerkleRoot, ClamPath) lives in crewai-rust.
#[derive(Clone, Debug)]
pub struct SubstrateSnapshot {
    /// Which substrate was PRIMARY during crystallization.
    pub birth_substrate: Substrate,

    /// The route that was active at crystallization.
    pub birth_route: SubstrateRoute,

    /// Signals at the moment of crystallization.
    pub birth_signals: SubstrateSignals,

    /// Cross-substrate coherence at crystallization.
    pub birth_coherence: Coherence,

    // === Per-substrate snapshots ===
    /// NARS truth at crystallization (if evidential substrate was involved).
    /// (frequency, confidence).
    pub nars_truth: Option<(f32, f32)>,

    /// BCM θ average at crystallization (if soaking was involved).
    /// Captures how "open" the system was when this concept formed.
    pub theta_at_birth: Option<f32>,

    /// Maturity at crystallization (if soaking was involved).
    pub maturity_at_birth: Option<u8>,

    /// Cosine similarity to nearest known concept at crystallization
    /// (if semantic substrate was involved).
    pub semantic_nearest_at_birth: Option<f32>,

    /// Saturation ratio at crystallization (if soaking was involved).
    pub saturation_at_birth: Option<f32>,
}

impl SubstrateSnapshot {
    /// Create a snapshot from current state.
    pub fn capture(primary: Substrate, route: &SubstrateRoute, signals: &SubstrateSignals) -> Self {
        let coh = coherence(signals);

        // Capture soaking state if soaking was active
        let (theta, maturity, saturation) =
            if route.uses(Substrate::Soaking) && signals.has_soaking() {
                (
                    Some(signals.theta_average),
                    Some(signals.maturity_average as u8),
                    Some(signals.soaking_saturation),
                )
            } else {
                (None, None, None)
            };

        // Capture NARS state if evidential was active
        let nars = if route.uses(Substrate::Evidential) && signals.has_evidential() {
            Some((signals.nars_frequency, signals.nars_confidence))
        } else {
            None
        };

        // Capture semantic state if semantic was active
        let semantic = if route.uses(Substrate::Semantic) && signals.has_semantic() {
            Some(signals.semantic_nearest)
        } else {
            None
        };

        Self {
            birth_substrate: primary,
            birth_route: route.clone(),
            birth_signals: signals.clone(),
            birth_coherence: coh,
            nars_truth: nars,
            theta_at_birth: theta,
            maturity_at_birth: maturity,
            semantic_nearest_at_birth: semantic,
            saturation_at_birth: saturation,
        }
    }

    /// Suggest which substrates should be activated when recalling this atom.
    ///
    /// An atom born under soaking/semantic will suggest those substrates
    /// for recall. This is how the system develops per-concept substrate
    /// preferences without explicit programming.
    pub fn recall_route(&self) -> SubstrateRoute {
        // Start from birth route but potentially adjust based on coherence
        match self.birth_coherence {
            Coherence::Convergent => {
                // Birth was clean convergence → same route for recall
                self.birth_route.clone()
            }
            Coherence::Divergent => {
                // Birth had substrate disagreement → add evidential for re-check
                if !self.birth_route.uses(Substrate::Evidential) {
                    SubstrateRoute {
                        primary: self.birth_route.primary,
                        secondary: Some(Substrate::Evidential),
                        tertiary: self.birth_route.secondary,
                        primary_weight: self.birth_route.primary_weight * 0.8,
                        parallel: true, // parallel to catch disagreement
                    }
                } else {
                    self.birth_route.clone()
                }
            }
            _ => self.birth_route.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Soaking-substrate diagnostics (pure compute helpers)
// ---------------------------------------------------------------------------

/// Compute aggregate soaking signals from a register of SynapseStates.
///
/// Pure function: takes a register, returns signals for the soaking substrate.
/// Used by upstream code to populate `SubstrateSignals.soaking_*` fields.
pub fn soaking_signals(register: &[SynapseState]) -> (f32, f32, f32) {
    if register.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let len = register.len() as f32;

    let saturation = super::organic::saturation_ratio(register, 100);
    let theta_avg = register.iter().map(|s| s.theta as f32).sum::<f32>() / len;
    let maturity_avg = register.iter().map(|s| s.maturity as f32).sum::<f32>() / len;

    (saturation, theta_avg, maturity_avg)
}

/// Populate the soaking fields of SubstrateSignals from a register.
pub fn fill_soaking_signals(signals: &mut SubstrateSignals, register: &[SynapseState]) {
    let (sat, theta, maturity) = soaking_signals(register);
    signals.soaking_saturation = sat;
    signals.theta_average = theta;
    signals.maturity_average = maturity;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_roundtrip() {
        for s in Substrate::ALL {
            let raw = s as u8;
            assert_eq!(Substrate::from_raw(raw), Some(s));
        }
        assert_eq!(Substrate::from_raw(4), None);
        assert_eq!(Substrate::from_raw(255), None);
    }

    #[test]
    fn test_substrate_route_single() {
        let route = SubstrateRoute::single(Substrate::Structural);
        assert_eq!(route.primary, Substrate::Structural);
        assert!(route.secondary.is_none());
        assert!(route.tertiary.is_none());
        assert_eq!(route.depth(), 1);
        assert!(route.uses(Substrate::Structural));
        assert!(!route.uses(Substrate::Soaking));
    }

    #[test]
    fn test_substrate_route_dual() {
        let route = SubstrateRoute::dual(Substrate::Semantic, Substrate::Soaking, 0.7, true);
        assert_eq!(route.depth(), 2);
        assert!(route.uses(Substrate::Semantic));
        assert!(route.uses(Substrate::Soaking));
        assert!(!route.uses(Substrate::Structural));
        let subs = route.substrates();
        assert_eq!(subs, vec![Substrate::Semantic, Substrate::Soaking]);
    }

    #[test]
    fn test_substrate_route_triple() {
        let route = SubstrateRoute::triple(
            Substrate::Structural,
            Substrate::Evidential,
            Substrate::Semantic,
            0.6,
            true,
        );
        assert_eq!(route.depth(), 3);
        assert!(route.uses(Substrate::Structural));
        assert!(route.uses(Substrate::Evidential));
        assert!(route.uses(Substrate::Semantic));
        assert!(!route.uses(Substrate::Soaking));
    }

    #[test]
    fn test_signals_default_empty() {
        let sig = SubstrateSignals::new();
        assert_eq!(sig.active_count(), 0);
        assert!(!sig.has_structural());
        assert!(!sig.has_soaking());
        assert!(!sig.has_evidential());
        assert!(!sig.has_semantic());
    }

    #[test]
    fn test_signals_structural_only() {
        let sig = SubstrateSignals {
            structural_hits: 3,
            structural_nearest_distance: 42,
            ..Default::default()
        };
        assert!(sig.has_structural());
        assert!(!sig.has_soaking());
        assert_eq!(sig.active_count(), 1);
    }

    #[test]
    fn test_signals_all_active() {
        let sig = SubstrateSignals {
            structural_hits: 2,
            soaking_saturation: 0.5,
            nars_confidence: 0.7,
            semantic_nearest: 0.4,
            ..Default::default()
        };
        assert_eq!(sig.active_count(), 4);
    }

    #[test]
    fn test_coherence_singular() {
        let sig = SubstrateSignals {
            structural_hits: 3,
            ..Default::default()
        };
        assert_eq!(coherence(&sig), Coherence::Singular);
    }

    #[test]
    fn test_coherence_convergent() {
        // Structural hit + high NARS confidence → convergent
        let sig = SubstrateSignals {
            structural_hits: 5,
            structural_nearest_distance: 10,
            nars_confidence: 0.8,
            nars_frequency: 0.9,
            nars_evidence_count: 10,
            ..Default::default()
        };
        assert_eq!(coherence(&sig), Coherence::Convergent);
    }

    #[test]
    fn test_coherence_divergent() {
        // No structural match but strong semantic → divergent (analogy)
        let sig = SubstrateSignals {
            structural_hits: 0,
            structural_nearest_distance: 500,
            semantic_nearest: 0.8,
            semantic_neighborhood_size: 5,
            ..Default::default()
        };
        assert_eq!(coherence(&sig), Coherence::Divergent);
    }

    #[test]
    fn test_recommend_transition_crystallize() {
        // Soaking saturated + high theta → should move to structural
        let sig = SubstrateSignals {
            soaking_saturation: 0.9,
            theta_average: 120.0,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Soaking, &sig);
        assert_eq!(rec, Some(Substrate::Structural));
    }

    #[test]
    fn test_recommend_transition_doubt() {
        // Low NARS confidence + contradictions → should move to evidential
        let sig = SubstrateSignals {
            nars_confidence: 0.2,
            nars_contradictions: 3,
            nars_evidence_count: 5,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Structural, &sig);
        assert_eq!(rec, Some(Substrate::Evidential));
    }

    #[test]
    fn test_recommend_transition_novelty() {
        // No structural or semantic match → should start soaking
        let sig = SubstrateSignals {
            structural_hits: 0,
            semantic_nearest: 0.1,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Structural, &sig);
        assert_eq!(rec, Some(Substrate::Soaking));
    }

    #[test]
    fn test_recommend_transition_association() {
        // Structural miss but semantic hit → try semantic
        let sig = SubstrateSignals {
            structural_hits: 0,
            structural_nearest_distance: 500,
            semantic_nearest: 0.7,
            semantic_neighborhood_size: 3,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Structural, &sig);
        assert_eq!(rec, Some(Substrate::Semantic));
    }

    #[test]
    fn test_recommend_transition_convergence() {
        // Multiple structural hits + high confidence → lock in
        let sig = SubstrateSignals {
            structural_hits: 5,
            nars_confidence: 0.9,
            nars_evidence_count: 20,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Soaking, &sig);
        assert_eq!(rec, Some(Substrate::Structural));
    }

    #[test]
    fn test_recommend_transition_complexity() {
        // Large neighborhood + contradictions → evidential
        let sig = SubstrateSignals {
            semantic_neighborhood_size: 15,
            semantic_nearest: 0.6,
            nars_contradictions: 3,
            nars_confidence: 0.5,
            nars_evidence_count: 5,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Structural, &sig);
        assert_eq!(rec, Some(Substrate::Evidential));
    }

    #[test]
    fn test_recommend_transition_none() {
        // Normal operation, nothing alarming → no transition
        let sig = SubstrateSignals {
            structural_hits: 2,
            nars_confidence: 0.6,
            nars_evidence_count: 5,
            ..Default::default()
        };
        let rec = recommend_transition(Substrate::Structural, &sig);
        assert_eq!(rec, None);
    }

    #[test]
    fn test_snapshot_capture_soaking() {
        let route = SubstrateRoute::dual(Substrate::Soaking, Substrate::Semantic, 0.6, true);
        let sig = SubstrateSignals {
            soaking_saturation: 0.8,
            theta_average: 80.0,
            maturity_average: 7.0,
            semantic_nearest: 0.5,
            semantic_neighborhood_size: 3,
            ..Default::default()
        };
        let snap = SubstrateSnapshot::capture(Substrate::Soaking, &route, &sig);
        assert_eq!(snap.birth_substrate, Substrate::Soaking);
        assert!(snap.theta_at_birth.is_some());
        assert!(snap.saturation_at_birth.is_some());
        assert!(snap.semantic_nearest_at_birth.is_some());
        assert!(snap.nars_truth.is_none()); // evidential not active
    }

    #[test]
    fn test_snapshot_capture_structural_evidential() {
        let route = SubstrateRoute::dual(Substrate::Structural, Substrate::Evidential, 0.85, false);
        let sig = SubstrateSignals {
            structural_hits: 3,
            structural_nearest_distance: 15,
            nars_confidence: 0.9,
            nars_frequency: 0.95,
            nars_evidence_count: 25,
            ..Default::default()
        };
        let snap = SubstrateSnapshot::capture(Substrate::Structural, &route, &sig);
        assert_eq!(snap.birth_substrate, Substrate::Structural);
        assert!(snap.nars_truth.is_some());
        let (f, c) = snap.nars_truth.unwrap();
        assert!((f - 0.95).abs() < f32::EPSILON);
        assert!((c - 0.9).abs() < f32::EPSILON);
        assert!(snap.theta_at_birth.is_none()); // soaking not active
    }

    #[test]
    fn test_recall_route_convergent() {
        let route = SubstrateRoute::single(Substrate::Structural);
        let sig = SubstrateSignals {
            structural_hits: 5,
            nars_confidence: 0.9,
            nars_evidence_count: 10,
            ..Default::default()
        };
        let snap = SubstrateSnapshot::capture(Substrate::Structural, &route, &sig);
        let recall = snap.recall_route();
        // Convergent birth → same route for recall
        assert_eq!(recall.primary, Substrate::Structural);
    }

    #[test]
    fn test_recall_route_divergent_adds_evidential() {
        let route = SubstrateRoute::dual(Substrate::Structural, Substrate::Semantic, 0.7, true);
        // Structural miss + semantic hit → divergent
        let sig = SubstrateSignals {
            structural_hits: 0,
            structural_nearest_distance: 500,
            semantic_nearest: 0.8,
            semantic_neighborhood_size: 5,
            ..Default::default()
        };
        let snap = SubstrateSnapshot::capture(Substrate::Structural, &route, &sig);
        assert_eq!(snap.birth_coherence, Coherence::Divergent);
        let recall = snap.recall_route();
        // Divergent → should add evidential for re-check
        assert!(recall.uses(Substrate::Evidential));
    }

    #[test]
    fn test_soaking_signals_from_register() {
        let register = vec![
            SynapseState {
                efficacy: 120,
                theta: 80,
                maturity: 10,
            },
            SynapseState {
                efficacy: -110,
                theta: 60,
                maturity: 8,
            },
            SynapseState {
                efficacy: 5,
                theta: 20,
                maturity: 1,
            },
            SynapseState {
                efficacy: 50,
                theta: 40,
                maturity: 5,
            },
        ];
        let (sat, theta_avg, mat_avg) = soaking_signals(&register);
        // 2 of 4 above threshold 100 → 0.5
        assert!((sat - 0.5).abs() < f32::EPSILON);
        // theta avg = (80+60+20+40)/4 = 50
        assert!((theta_avg - 50.0).abs() < f32::EPSILON);
        // maturity avg = (10+8+1+5)/4 = 6
        assert!((mat_avg - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fill_soaking_signals() {
        let register = vec![
            SynapseState {
                efficacy: 120,
                theta: 80,
                maturity: 10,
            },
            SynapseState {
                efficacy: -110,
                theta: 60,
                maturity: 8,
            },
        ];
        let mut sig = SubstrateSignals::new();
        fill_soaking_signals(&mut sig, &register);
        assert!(sig.has_soaking());
        assert!((sig.theta_average - 70.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soaking_signals_empty() {
        let (sat, theta, mat) = soaking_signals(&[]);
        assert_eq!(sat, 0.0);
        assert_eq!(theta, 0.0);
        assert_eq!(mat, 0.0);
    }
}
