//! Full NARS (Non-Axiomatic Reasoning System) engine.
//!
//! Seven inference rules + budget management + contradiction detection.
//! Separate from causality.rs which only has basic NarsTruthValue.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Evidential horizon parameter.
pub const HORIZON: f32 = 1.0;

// ---------------------------------------------------------------------------
// NarsTruth
// ---------------------------------------------------------------------------

/// NARS truth value with frequency and confidence.
#[derive(Clone, Copy, Debug)]
pub struct NarsTruth {
    /// Frequency: fraction of positive evidence [0.0, 1.0].
    pub frequency: f32,
    /// Confidence: strength of evidence [0.0, 1.0).
    pub confidence: f32,
}

impl NarsTruth {
    /// Create a truth value, clamping to valid ranges.
    ///
    /// Frequency is clamped to [0.0, 1.0].
    /// Confidence is clamped to [0.0, 1.0).
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsTruth;
    ///
    /// let tv = NarsTruth::new(0.9, 0.8);
    /// assert!((tv.frequency - 0.9).abs() < 1e-6);
    /// assert!((tv.confidence - 0.8).abs() < 1e-6);
    /// ```
    pub fn new(f: f32, c: f32) -> Self {
        Self {
            frequency: f.clamp(0.0, 1.0),
            confidence: c.clamp(0.0, 0.9999),
        }
    }

    /// Construct a truth value from positive and negative evidence counts.
    ///
    /// f = pos / (pos + neg), c = (pos + neg) / (pos + neg + HORIZON)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsTruth;
    ///
    /// let tv = NarsTruth::from_evidence(9.0, 1.0);
    /// assert!((tv.frequency - 0.9).abs() < 1e-2);
    /// ```
    pub fn from_evidence(positive: f32, negative: f32) -> Self {
        let total = positive + negative;
        if total <= 0.0 {
            return Self::ignorance();
        }
        let f = positive / total;
        let c = total / (total + HORIZON);
        Self::new(f, c)
    }

    /// Convert this truth value back to positive/negative evidence.
    ///
    /// w_pos = f * c * HORIZON / (1 - c)
    /// w_neg = (1 - f) * c * HORIZON / (1 - c)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsTruth;
    ///
    /// let tv = NarsTruth::from_evidence(9.0, 1.0);
    /// let ev = tv.to_evidence();
    /// assert!((ev.positive - 9.0).abs() < 0.5);
    /// assert!((ev.negative - 1.0).abs() < 0.5);
    /// ```
    pub fn to_evidence(&self) -> NarsEvidence {
        let denom = 1.0 - self.confidence;
        if denom <= 1e-9 {
            // Near-infinite evidence; return large values proportionally.
            return NarsEvidence {
                positive: self.frequency * 1e6,
                negative: (1.0 - self.frequency) * 1e6,
            };
        }
        let w = self.confidence * HORIZON / denom;
        NarsEvidence {
            positive: self.frequency * w,
            negative: (1.0 - self.frequency) * w,
        }
    }

    /// Expectation value: `c * (f - 0.5) + 0.5`.
    ///
    /// Returns a single scalar in [0.0, 1.0] that combines frequency and
    /// confidence into an actionable estimate.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsTruth;
    ///
    /// let tv = NarsTruth::new(1.0, 0.9);
    /// assert!((tv.expectation() - 0.95).abs() < 1e-2);
    /// ```
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    /// Total ignorance: frequency 0.5, confidence 0.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsTruth;
    ///
    /// let tv = NarsTruth::ignorance();
    /// assert!((tv.expectation() - 0.5).abs() < 1e-6);
    /// ```
    pub fn ignorance() -> Self {
        Self {
            frequency: 0.5,
            confidence: 0.0,
        }
    }

    /// Returns true if expectation > 0.5 (more positive than not).
    pub fn is_positive(&self) -> bool {
        self.expectation() > 0.5
    }

    /// Returns true if confidence > 0.5 (reasonably well-supported).
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.5
    }
}

impl Default for NarsTruth {
    fn default() -> Self {
        Self::ignorance()
    }
}

// ---------------------------------------------------------------------------
// NarsEvidence
// ---------------------------------------------------------------------------

/// Evidence tracking for a belief.
#[derive(Clone, Copy, Debug)]
pub struct NarsEvidence {
    /// Positive evidence count.
    pub positive: f32,
    /// Negative evidence count.
    pub negative: f32,
}

// ---------------------------------------------------------------------------
// NarsBudget
// ---------------------------------------------------------------------------

/// NARS budget value for attention allocation.
#[derive(Clone, Copy, Debug)]
pub struct NarsBudget {
    /// Priority: urgency of processing [0.0, 1.0].
    pub priority: f32,
    /// Durability: persistence over time [0.0, 1.0].
    pub durability: f32,
    /// Quality: usefulness of the item [0.0, 1.0].
    pub quality: f32,
}

impl NarsBudget {
    /// Create a budget value, clamping all components to [0.0, 1.0].
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsBudget;
    ///
    /// let b = NarsBudget::new(0.8, 0.5, 0.6);
    /// assert!((b.priority - 0.8).abs() < 1e-6);
    /// ```
    pub fn new(p: f32, d: f32, q: f32) -> Self {
        Self {
            priority: p.clamp(0.0, 1.0),
            durability: d.clamp(0.0, 1.0),
            quality: q.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// ContradictionResult
// ---------------------------------------------------------------------------

/// Result of contradiction detection.
#[derive(Clone, Debug, PartialEq)]
pub enum ContradictionResult {
    /// No contradiction found.
    None,
    /// Contradiction detected with severity and description.
    Detected {
        /// Severity of the contradiction [0.0, 1.0].
        severity: f32,
        /// Human-readable description.
        description: String,
    },
}

// ---------------------------------------------------------------------------
// NarsContext
// ---------------------------------------------------------------------------

/// Working memory context for inference chains.
#[derive(Clone, Debug)]
pub struct NarsContext {
    /// Active beliefs: (label, truth value).
    pub beliefs: Vec<(String, NarsTruth)>,
    /// Active goals: (label, truth value).
    pub goals: Vec<(String, NarsTruth)>,
    /// Total number of inference steps performed.
    pub inference_count: u64,
}

impl NarsContext {
    /// Create an empty context.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::NarsContext;
    ///
    /// let ctx = NarsContext::new();
    /// assert!(ctx.beliefs.is_empty());
    /// assert_eq!(ctx.inference_count, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            beliefs: Vec::new(),
            goals: Vec::new(),
            inference_count: 0,
        }
    }

    /// Add a belief to the context.
    pub fn add_belief(&mut self, label: &str, truth: NarsTruth) {
        self.beliefs.push((label.to_string(), truth));
    }

    /// Add a goal to the context.
    pub fn add_goal(&mut self, label: &str, truth: NarsTruth) {
        self.goals.push((label.to_string(), truth));
    }

    /// Return the belief with the highest expectation, or `None` if empty.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::nars::{NarsContext, NarsTruth};
    ///
    /// let mut ctx = NarsContext::new();
    /// ctx.add_belief("a", NarsTruth::new(0.9, 0.8));
    /// ctx.add_belief("b", NarsTruth::new(0.5, 0.3));
    /// let best = ctx.best_belief().unwrap();
    /// assert_eq!(best.0, "a");
    /// ```
    pub fn best_belief(&self) -> Option<&(String, NarsTruth)> {
        self.beliefs
            .iter()
            .max_by(|a, b| a.1.expectation().partial_cmp(&b.1.expectation()).unwrap())
    }

    /// Find a belief by label and revise it with new evidence.
    ///
    /// If the label is not found, the new evidence is added as a new belief.
    /// Increments `inference_count`.
    pub fn revise_belief(&mut self, label: &str, new_evidence: NarsTruth) {
        self.inference_count += 1;
        if let Some(entry) = self.beliefs.iter_mut().find(|(l, _)| l == label) {
            entry.1 = nars_revision(entry.1, new_evidence);
        } else {
            self.beliefs.push((label.to_string(), new_evidence));
        }
    }
}

impl Default for NarsContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Inference rules
// ---------------------------------------------------------------------------

/// Revision: combine two independent sources of evidence for the same
/// statement.
///
/// Converts both truth values to evidence, sums them, and recomputes.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_revision};
///
/// let a = NarsTruth::new(0.9, 0.5);
/// let b = NarsTruth::new(0.9, 0.5);
/// let r = nars_revision(a, b);
/// assert!((r.frequency - 0.9).abs() < 0.05);
/// assert!(r.confidence > a.confidence);
/// ```
pub fn nars_revision(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let ea = a.to_evidence();
    let eb = b.to_evidence();
    NarsTruth::from_evidence(ea.positive + eb.positive, ea.negative + eb.negative)
}

/// Deduction: A->B, B->C |- A->C.
///
/// f = f1 * f2, c = f1 * f2 * c1 * c2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_deduction};
///
/// let a = NarsTruth::new(0.9, 0.9);
/// let b = NarsTruth::new(0.9, 0.9);
/// let d = nars_deduction(a, b);
/// assert!(d.frequency > 0.7);
/// ```
pub fn nars_deduction(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f = a.frequency * b.frequency;
    let c = a.frequency * b.frequency * a.confidence * b.confidence;
    NarsTruth::new(f, c)
}

/// Abduction: A->B, C->B |- A->C.
///
/// f = f1, c = w / (w + HORIZON) where w = f2 * c1 * c2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_abduction};
///
/// let a = NarsTruth::new(0.9, 0.9);
/// let b = NarsTruth::new(0.9, 0.9);
/// let r = nars_abduction(a, b);
/// assert!(r.confidence < a.confidence);
/// ```
pub fn nars_abduction(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f = a.frequency;
    let w = b.frequency * a.confidence * b.confidence;
    let c = w / (w + HORIZON);
    NarsTruth::new(f, c)
}

/// Induction: A->B, A->C |- B->C.
///
/// f = f2, c = w / (w + HORIZON) where w = f1 * c1 * c2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_induction};
///
/// let a = NarsTruth::new(0.9, 0.9);
/// let b = NarsTruth::new(0.9, 0.9);
/// let r = nars_induction(a, b);
/// assert!(r.confidence < a.confidence);
/// ```
pub fn nars_induction(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f = b.frequency;
    let w = a.frequency * a.confidence * b.confidence;
    let c = w / (w + HORIZON);
    NarsTruth::new(f, c)
}

/// Comparison: A->B, C->B |- A<->C.
///
/// f = f1*f2 / (f1 + f2 - f1*f2), c = w / (w + HORIZON) where
/// w = f1 * f2 * c1 * c2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_comparison};
///
/// let a = NarsTruth::new(0.8, 0.8);
/// let b = NarsTruth::new(0.8, 0.8);
/// let r = nars_comparison(a, b);
/// assert!(r.frequency > 0.5);
/// ```
pub fn nars_comparison(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f1f2 = a.frequency * b.frequency;
    let denom = a.frequency + b.frequency - f1f2;
    let f = if denom > 1e-9 { f1f2 / denom } else { 0.0 };
    let w = f1f2 * a.confidence * b.confidence;
    let c = w / (w + HORIZON);
    NarsTruth::new(f, c)
}

/// Analogy: A->B, A<->C |- C->B.
///
/// f = f1 * f2, c = c1 * c2 * f2.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_analogy};
///
/// let a = NarsTruth::new(0.9, 0.9);
/// let b = NarsTruth::new(0.9, 0.9);
/// let r = nars_analogy(a, b);
/// assert!(r.frequency > 0.7);
/// ```
pub fn nars_analogy(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f = a.frequency * b.frequency;
    let c = a.confidence * b.confidence * b.frequency;
    NarsTruth::new(f, c)
}

/// Resemblance: A<->B, A<->C |- B<->C.
///
/// f = f1 * f2, c = c1 * c2 * (f1 + f2 - f1 * f2).
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_resemblance};
///
/// let a = NarsTruth::new(0.9, 0.9);
/// let b = NarsTruth::new(0.9, 0.9);
/// let r = nars_resemblance(a, b);
/// assert!(r.frequency > 0.7);
/// ```
pub fn nars_resemblance(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let f = a.frequency * b.frequency;
    let c = a.confidence * b.confidence * (a.frequency + b.frequency - a.frequency * b.frequency);
    NarsTruth::new(f, c)
}

/// #11 Contradiction between two beliefs: similar structure, opposing truth.
#[derive(Clone, Debug)]
pub struct Contradiction {
    pub structural_similarity: f32,
    pub truth_conflict: f32,
    pub resolution: NarsTruth,
}

/// #11 Detect contradictions: high structural similarity + opposing truth values.
/// Science: Wang (2006) revision, Priest (2002) paraconsistent logic, CHAODA.
pub fn detect_contradiction(
    truth_a: &NarsTruth,
    truth_b: &NarsTruth,
    structural_similarity: f32,
    threshold: f32,
) -> Option<Contradiction> {
    let truth_conflict = (truth_a.frequency - truth_b.frequency).abs();
    if structural_similarity > 0.7 && truth_conflict > threshold {
        Some(Contradiction {
            structural_similarity,
            truth_conflict,
            resolution: nars_revision(*truth_a, *truth_b),
        })
    } else {
        None
    }
}

/// #7 Adversarial Self-Critique result.
#[derive(Clone, Debug)]
pub struct Challenge {
    pub kind: ChallengeKind,
    pub alternative_truth: NarsTruth,
    pub survives: bool,
}

#[derive(Clone, Debug)]
pub enum ChallengeKind {
    /// What if the opposite is true?
    Negation,
    /// What breaks if this is false?
    Dependency,
}

/// #7 Adversarial Self-Critique — challenge a claim's truth value.
/// Science: Wang (2006) NARS negation, Mercier & Sperber (2011), Kahneman premortem.
pub fn adversarial_critique(truth: &NarsTruth) -> Vec<Challenge> {
    vec![
        // Negation: not<f,c> = <1-f, c*0.9>
        Challenge {
            kind: ChallengeKind::Negation,
            alternative_truth: NarsTruth::new(1.0 - truth.frequency, truth.confidence * 0.9),
            survives: truth.expectation() > NarsTruth::new(1.0 - truth.frequency, truth.confidence * 0.9).expectation(),
        },
        // Dependency: what if confidence drops?
        Challenge {
            kind: ChallengeKind::Dependency,
            alternative_truth: NarsTruth::new(truth.frequency, truth.confidence * 0.5),
            survives: truth.confidence > 0.5,
        },
    ]
}

// ---------------------------------------------------------------------------
// Budget operations
// ---------------------------------------------------------------------------

/// Merge two budget values: max priority, average durability, max quality.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsBudget, budget_merge};
///
/// let a = NarsBudget::new(0.8, 0.6, 0.7);
/// let b = NarsBudget::new(0.5, 0.4, 0.9);
/// let m = budget_merge(a, b);
/// assert!((m.priority - 0.8).abs() < 1e-6);
/// assert!((m.durability - 0.5).abs() < 1e-6);
/// assert!((m.quality - 0.9).abs() < 1e-6);
/// ```
pub fn budget_merge(a: NarsBudget, b: NarsBudget) -> NarsBudget {
    NarsBudget {
        priority: a.priority.max(b.priority),
        durability: (a.durability + b.durability) / 2.0,
        quality: a.quality.max(b.quality),
    }
}

/// Decay a budget by multiplying priority and durability by `factor`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsBudget, budget_decay};
///
/// let b = NarsBudget::new(0.8, 0.6, 0.7);
/// let d = budget_decay(b, 0.5);
/// assert!((d.priority - 0.4).abs() < 1e-6);
/// assert!((d.durability - 0.3).abs() < 1e-6);
/// assert!((d.quality - 0.7).abs() < 1e-6); // quality unchanged
/// ```
pub fn budget_decay(b: NarsBudget, factor: f32) -> NarsBudget {
    NarsBudget::new(b.priority * factor, b.durability * factor, b.quality)
}

// ---------------------------------------------------------------------------
// Contradiction detection
// ---------------------------------------------------------------------------

/// Detect contradiction between two truth values.
///
/// If both are confident (confidence > 0.5) and their frequencies differ by
/// more than 0.5, a contradiction is reported. Severity is the product of
/// the two confidences scaled by the frequency gap.
///
/// # Example
///
/// ```
/// use ndarray::hpc::nars::{NarsTruth, nars_contradiction_detect, ContradictionResult};
///
/// let a = NarsTruth::new(0.9, 0.8);
/// let b = NarsTruth::new(0.1, 0.8);
/// match nars_contradiction_detect(a, b) {
///     ContradictionResult::Detected { severity, .. } => assert!(severity > 0.4),
///     _ => panic!("expected contradiction"),
/// }
/// ```
pub fn nars_contradiction_detect(a: NarsTruth, b: NarsTruth) -> ContradictionResult {
    let freq_gap = (a.frequency - b.frequency).abs();
    if a.is_confident() && b.is_confident() && freq_gap > 0.5 {
        let severity = a.confidence * b.confidence * freq_gap;
        ContradictionResult::Detected {
            severity,
            description: format!(
                "Frequency gap {:.3} with confidences ({:.3}, {:.3})",
                freq_gap, a.confidence, b.confidence
            ),
        }
    } else {
        ContradictionResult::None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_evidence_roundtrip() {
        let tv = NarsTruth::from_evidence(9.0, 1.0);
        let ev = tv.to_evidence();
        assert!(
            (ev.positive - 9.0).abs() < 0.5,
            "positive: {} expected ~9.0",
            ev.positive
        );
        assert!(
            (ev.negative - 1.0).abs() < 0.5,
            "negative: {} expected ~1.0",
            ev.negative
        );
    }

    #[test]
    fn test_revision_equal_evidence() {
        let a = NarsTruth::new(0.9, 0.5);
        let b = NarsTruth::new(0.9, 0.5);
        let r = nars_revision(a, b);
        assert!(
            (r.frequency - 0.9).abs() < 0.05,
            "frequency: {} expected ~0.9",
            r.frequency
        );
        assert!(
            r.confidence > a.confidence,
            "revised confidence {} should exceed input {}",
            r.confidence,
            a.confidence
        );
    }

    #[test]
    fn test_deduction_strong() {
        let a = NarsTruth::new(0.9, 0.9);
        let b = NarsTruth::new(0.9, 0.9);
        let d = nars_deduction(a, b);
        assert!(d.frequency > 0.7, "f={}", d.frequency);
        assert!(d.confidence > 0.4, "c={}", d.confidence);
    }

    #[test]
    fn test_deduction_weak() {
        let a = NarsTruth::new(0.5, 0.3);
        let b = NarsTruth::new(0.5, 0.3);
        let d = nars_deduction(a, b);
        // Weak premises should yield weaker conclusion
        assert!(
            d.confidence < a.confidence,
            "weak deduction c={} should be < input c={}",
            d.confidence,
            a.confidence
        );
    }

    #[test]
    fn test_abduction_weaker() {
        let a = NarsTruth::new(0.9, 0.9);
        let b = NarsTruth::new(0.9, 0.9);
        let r = nars_abduction(a, b);
        assert!(
            r.confidence < a.confidence,
            "abduction c={} should be weaker than input c={}",
            r.confidence,
            a.confidence
        );
    }

    #[test]
    fn test_induction_weaker() {
        let a = NarsTruth::new(0.9, 0.9);
        let b = NarsTruth::new(0.9, 0.9);
        let r = nars_induction(a, b);
        assert!(
            r.confidence < a.confidence,
            "induction c={} should be weaker than input c={}",
            r.confidence,
            a.confidence
        );
    }

    #[test]
    fn test_comparison_symmetric() {
        let a = NarsTruth::new(0.8, 0.8);
        let b = NarsTruth::new(0.7, 0.7);
        let r1 = nars_comparison(a, b);
        let r2 = nars_comparison(b, a);
        // Comparison should be approximately symmetric in frequency
        assert!(
            (r1.frequency - r2.frequency).abs() < 1e-6,
            "comparison not symmetric: {} vs {}",
            r1.frequency,
            r2.frequency
        );
    }

    #[test]
    fn test_analogy_transfers() {
        let a = NarsTruth::new(0.9, 0.9);
        let b = NarsTruth::new(0.9, 0.9);
        let r = nars_analogy(a, b);
        assert!(r.frequency > 0.7, "f={}", r.frequency);
        assert!(r.confidence > 0.5, "c={}", r.confidence);
    }

    #[test]
    fn test_resemblance_bidirectional() {
        let a = NarsTruth::new(0.9, 0.9);
        let b = NarsTruth::new(0.8, 0.8);
        let r = nars_resemblance(a, b);
        assert!(r.frequency > 0.5, "f={}", r.frequency);
        assert!(r.confidence > 0.3, "c={}", r.confidence);
        // Resemblance frequency is symmetric
        let r2 = nars_resemblance(b, a);
        assert!(
            (r.frequency - r2.frequency).abs() < 1e-6,
            "resemblance not symmetric: {} vs {}",
            r.frequency,
            r2.frequency
        );
    }

    #[test]
    fn test_contradiction_detected() {
        let a = NarsTruth::new(0.9, 0.8);
        let b = NarsTruth::new(0.1, 0.8);
        let result = nars_contradiction_detect(a, b);
        match result {
            ContradictionResult::Detected { severity, .. } => {
                assert!(severity > 0.4, "severity={}", severity);
            }
            ContradictionResult::None => panic!("expected contradiction to be detected"),
        }
    }

    #[test]
    fn test_contradiction_none() {
        let a = NarsTruth::new(0.9, 0.8);
        let b = NarsTruth::new(0.85, 0.7);
        let result = nars_contradiction_detect(a, b);
        assert_eq!(result, ContradictionResult::None);
    }

    #[test]
    fn test_budget_merge() {
        let a = NarsBudget::new(0.8, 0.6, 0.7);
        let b = NarsBudget::new(0.5, 0.4, 0.9);
        let m = budget_merge(a, b);
        assert!((m.priority - 0.8).abs() < 1e-6, "p={}", m.priority);
        assert!((m.durability - 0.5).abs() < 1e-6, "d={}", m.durability);
        assert!((m.quality - 0.9).abs() < 1e-6, "q={}", m.quality);
    }

    #[test]
    fn test_budget_decay() {
        let b = NarsBudget::new(0.8, 0.6, 0.7);
        let d = budget_decay(b, 0.5);
        assert!((d.priority - 0.4).abs() < 1e-6, "p={}", d.priority);
        assert!((d.durability - 0.3).abs() < 1e-6, "d={}", d.durability);
        assert!((d.quality - 0.7).abs() < 1e-6, "q={}", d.quality);
    }

    #[test]
    fn test_context_revise() {
        let mut ctx = NarsContext::new();
        ctx.add_belief("bird_flies", NarsTruth::new(0.8, 0.5));
        ctx.revise_belief("bird_flies", NarsTruth::new(0.9, 0.6));
        assert_eq!(ctx.inference_count, 1);
        assert_eq!(ctx.beliefs.len(), 1);
        let tv = ctx.beliefs[0].1;
        // After revision, confidence should increase
        assert!(tv.confidence > 0.5, "c={}", tv.confidence);
    }

    #[test]
    fn test_adversarial_critique() {
        let strong = NarsTruth::new(0.9, 0.95);
        let challenges = adversarial_critique(&strong);
        assert_eq!(challenges.len(), 2);
        assert!(challenges[0].survives); // strong claim survives negation
        assert!(challenges[1].survives); // strong claim survives dependency

        let weak = NarsTruth::new(0.5, 0.3);
        let challenges = adversarial_critique(&weak);
        assert!(!challenges[1].survives); // weak confidence fails dependency
    }

    #[test]
    fn test_detect_contradiction() {
        let a = NarsTruth::new(0.9, 0.8);
        let b = NarsTruth::new(0.1, 0.8);
        // High similarity (0.9) + big truth gap (0.8) -> contradiction
        let c = detect_contradiction(&a, &b, 0.9, 0.5);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(c.truth_conflict > 0.5);

        // Low similarity -> no contradiction
        let c = detect_contradiction(&a, &b, 0.3, 0.5);
        assert!(c.is_none());
    }

    #[test]
    fn test_ignorance_expectation() {
        let tv = NarsTruth::ignorance();
        assert!(
            (tv.expectation() - 0.5).abs() < 1e-6,
            "expectation={}",
            tv.expectation()
        );
    }

    #[test]
    fn test_expectation_formula() {
        let tv = NarsTruth::new(1.0, 0.9);
        // c * (f - 0.5) + 0.5 = 0.9 * 0.5 + 0.5 = 0.95
        assert!(
            (tv.expectation() - 0.95).abs() < 1e-4,
            "expectation={}",
            tv.expectation()
        );

        let tv2 = NarsTruth::new(0.0, 0.8);
        // 0.8 * (0.0 - 0.5) + 0.5 = -0.4 + 0.5 = 0.1
        assert!(
            (tv2.expectation() - 0.1).abs() < 1e-4,
            "expectation={}",
            tv2.expectation()
        );
    }
}
