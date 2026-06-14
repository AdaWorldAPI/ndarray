//! Entropy ladder — the Staunen↔Wisdom coordinate that unifies NARS truth, the
//! 3×palette-256 SPO already inside a `CausalEdge64`, and the
//! syntax→semantics→pragmatics rungs.
//!
//! The organizing idea (operator framing, board canon): a fact's position on the
//! cognitive ladder IS an entropy level.
//!
//! * **Staunen** — high entropy — raw *stimulus*, not yet crystallized. Maps to
//!   **syntax** (surface form: many ways to say it).
//! * **Wisdom** — low entropy — *crystalline* knowledge / settled fact. Maps to
//!   **pragmatics** (one settled intent in context).
//! * **Semantics** sits between.
//!
//! Crucially this reads a `CausalEdge64`'s EXISTING fields — the three
//! palette-256 indices (`s_idx`/`p_idx`/`o_idx`), the Pearl 2³ causal mask, and
//! the packed NARS `(f, c)` — so there is NO re-quantization (the operator's
//! "the 3×palette256 SPO values inside CausalEdge64 are exact enough").
//!
//! The two axes (board's entropy×energy quadrant): **entropy** here, **energy**
//! supplied by the caller (e.g. `MailboxSoA.energy`). Together they classify the
//! [`Quadrant`] (Staunen / Confusion / Boredom / Wisdom).
//!
//! Validation: [`nars_entropy`] is a *reliability proxy* — it anti-correlates
//! with an edge's empirical prediction accuracy (see the module tests and the
//! `entropy_ladder_probe` example), grounded via [`crate::hpc::reliability`].
//!
//! ## Layering (PR #218 review — operator decision 2026-06-14)
//! This module carries cognition vocabulary (Staunen/Wisdom, SPO rungs) but has
//! NO dependency on the thinking/protocol crates: it is pure functions on
//! `(u8, f64)` + plain enums (`decompose_spo` takes raw palette indices, never a
//! `CausalEdge64`). It is VALIDATED here during the instrument phase, beside the
//! existing `hpc::nars` (NarsTruth) and `hpc::causal_diff` (CausalEdge64).
//! **Planned migration to the lance-graph thinking layer once the ladder
//! stabilizes** — tracked via the open coderabbit layering thread on PR #218.

/// Staunen↔Wisdom entropy of a NARS truth value `(f, c)`, in `[0, 1]`.
///
/// `entropy = 1 − c·|2f − 1|` = one minus the *decisiveness* `|2·E − 1|` of the
/// NARS expectation `E = c·(f − 0.5) + 0.5`.
///
/// * `0` (Wisdom): confident AND polarized — `c→1` and `f→0` or `f→1`.
/// * `1` (Staunen): unconfident (`c→0`) OR maximally ambiguous (`f→0.5`).
///
/// Inputs are clamped to `[0, 1]`.
///
/// # Examples
/// ```
/// use ndarray::hpc::entropy_ladder::nars_entropy;
/// assert!((nars_entropy(1.0, 1.0)).abs() < 1e-12);       // certain-true → Wisdom
/// assert!((nars_entropy(0.0, 1.0)).abs() < 1e-12);       // certain-false → Wisdom
/// assert!((nars_entropy(0.5, 1.0) - 1.0).abs() < 1e-12); // ambiguous → Staunen
/// assert!((nars_entropy(1.0, 0.0) - 1.0).abs() < 1e-12); // unconfident → Staunen
/// ```
#[inline]
pub fn nars_entropy(f: f64, c: f64) -> f64 {
    let f = f.clamp(0.0, 1.0);
    let c = c.clamp(0.0, 1.0);
    (1.0 - c * (2.0 * f - 1.0).abs()).clamp(0.0, 1.0)
}

/// The three linguistic rungs, read as an entropy ladder (high → low entropy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EntropyRung {
    /// High entropy — raw surface form / stimulus (Staunen end).
    Syntax = 0,
    /// Mid entropy — meaning.
    Semantics = 1,
    /// Low entropy — settled intent / fact (Wisdom end).
    Pragmatics = 2,
}

impl EntropyRung {
    /// Band an entropy value into a rung: `[0,⅓) → Pragmatics`, `[⅓,⅔) →
    /// Semantics`, `[⅔,1] → Syntax`.
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::entropy_ladder::EntropyRung;
    /// assert_eq!(EntropyRung::from_entropy(0.9), EntropyRung::Syntax);
    /// assert_eq!(EntropyRung::from_entropy(0.5), EntropyRung::Semantics);
    /// assert_eq!(EntropyRung::from_entropy(0.1), EntropyRung::Pragmatics);
    /// ```
    #[inline]
    pub fn from_entropy(h: f64) -> Self {
        if h >= 2.0 / 3.0 {
            EntropyRung::Syntax
        } else if h >= 1.0 / 3.0 {
            EntropyRung::Semantics
        } else {
            EntropyRung::Pragmatics
        }
    }
}

/// The entropy×energy quadrant (board canon, `LATEST_STATE` Staunen/Wisdom).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Quadrant {
    /// High entropy × low energy — stimulus that "needs entropy work".
    Staunen = 0,
    /// High entropy × high energy — in-progress climb (energy invested,
    /// entropy not yet collapsed).
    Confusion = 1,
    /// Low entropy × low energy — ordered but not energised.
    Boredom = 2,
    /// Low entropy × high energy — crystalline knowledge + plasticity.
    Wisdom = 3,
}

impl Quadrant {
    /// Classify by `(entropy, energy)`, both in `[0, 1]`, split at `0.5`.
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::entropy_ladder::Quadrant;
    /// assert_eq!(Quadrant::classify(0.9, 0.1), Quadrant::Staunen);
    /// assert_eq!(Quadrant::classify(0.1, 0.9), Quadrant::Wisdom);
    /// assert_eq!(Quadrant::classify(0.9, 0.9), Quadrant::Confusion);
    /// assert_eq!(Quadrant::classify(0.1, 0.1), Quadrant::Boredom);
    /// ```
    #[inline]
    pub fn classify(entropy: f64, energy: f64) -> Self {
        match (entropy >= 0.5, energy >= 0.5) {
            (true, false) => Quadrant::Staunen,
            (true, true) => Quadrant::Confusion,
            (false, false) => Quadrant::Boredom,
            (false, true) => Quadrant::Wisdom,
        }
    }
}

/// Pearl 2³ — the eight `(S, P, O)` active-subset masks, indexed by the
/// 3-bit causal mask (`bit0=S, bit1=P, bit2=O`). Index 0 = `∅`, index 7 = `SPO`.
pub const PEARL_SUBSETS: [(bool, bool, bool); 8] = [
    (false, false, false),
    (true, false, false),
    (false, true, false),
    (true, true, false),
    (false, false, true),
    (true, false, true),
    (false, true, true),
    (true, true, true),
];

/// One decomposed SPO ladder point, built from a `CausalEdge64`'s existing
/// fields with no re-quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpoLadderPoint {
    /// Staunen↔Wisdom entropy from the edge's NARS `(f, c)`.
    pub entropy: f64,
    /// The entropy rung (syntax/semantics/pragmatics).
    pub rung: EntropyRung,
    /// `(S, P, O)` active flags from the Pearl 2³ mask.
    pub active: (bool, bool, bool),
    /// HHTL-routable basin sub-key: `mask<<24 | s<<16 | p<<8 | o`, with inactive
    /// components zeroed. The active palette bytes ARE the basin coordinate that
    /// HHTL / helix / edge-residue route on downstream.
    pub basin_key: u32,
    /// 2-bit reliability class (0 = Wisdom … 3 = Staunen) for the 3 spare bits
    /// of `CausalEdge64` v2 (store via the consumer; this crate only computes it).
    pub class: u8,
}

/// Decompose an edge's existing fields into an [`SpoLadderPoint`].
///
/// `mask` is the 3-bit Pearl 2³ causal mask (`bit0=S, bit1=P, bit2=O`); only the
/// active palette indices contribute to `basin_key`. No re-quantization — the
/// palette indices are taken as-is.
///
/// # Examples
/// ```
/// use ndarray::hpc::entropy_ladder::{decompose_spo, EntropyRung};
/// // All three active (mask 0b111), crystalline truth (f=0.95, c=0.9).
/// let p = decompose_spo(10, 20, 30, 0b111, 0.95, 0.9);
/// assert_eq!(p.active, (true, true, true));
/// assert_eq!(p.basin_key, (0b111 << 24) | (10 << 16) | (20 << 8) | 30);
/// assert_eq!(p.rung, EntropyRung::Pragmatics); // low entropy → settled
/// ```
pub fn decompose_spo(s_idx: u8, p_idx: u8, o_idx: u8, mask: u8, f: f64, c: f64) -> SpoLadderPoint {
    let m = (mask & 0b111) as usize;
    let (sa, pa, oa) = PEARL_SUBSETS[m];
    let s = if sa { s_idx } else { 0 };
    let p = if pa { p_idx } else { 0 };
    let o = if oa { o_idx } else { 0 };
    let basin_key = ((m as u32) << 24) | ((s as u32) << 16) | ((p as u32) << 8) | (o as u32);
    let entropy = nars_entropy(f, c);
    SpoLadderPoint {
        entropy,
        rung: EntropyRung::from_entropy(entropy),
        active: (sa, pa, oa),
        basin_key,
        class: entropy_class(entropy),
    }
}

/// Quantize entropy into a 2-bit reliability class for the 3 spare bits of
/// `CausalEdge64` v2 (`[63:61]`): `0 = Wisdom` (most reliable) … `3 = Staunen`
/// (least). Reserve-don't-reclaim: the consumer writes this version-gated.
///
/// # Examples
/// ```
/// use ndarray::hpc::entropy_ladder::entropy_class;
/// assert_eq!(entropy_class(0.0), 0); // Wisdom
/// assert_eq!(entropy_class(1.0), 3); // Staunen
/// ```
#[inline]
pub fn entropy_class(h: f64) -> u8 {
    let h = h.clamp(0.0, 1.0);
    ((h * 4.0) as u8).min(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::reliability::spearman;

    fn splitmix(s: &mut u64) -> f64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn entropy_corners() {
        assert!(nars_entropy(1.0, 1.0) < 1e-12);
        assert!(nars_entropy(0.0, 1.0) < 1e-12);
        assert!((nars_entropy(0.5, 1.0) - 1.0).abs() < 1e-12);
        assert!((nars_entropy(1.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_monotone_in_confidence_and_polarization() {
        // More confidence on a polarized belief ⇒ lower entropy.
        assert!(nars_entropy(0.9, 0.9) < nars_entropy(0.9, 0.3));
        // More polarization at fixed confidence ⇒ lower entropy.
        assert!(nars_entropy(0.95, 0.8) < nars_entropy(0.6, 0.8));
    }

    #[test]
    fn rung_bands() {
        assert_eq!(EntropyRung::from_entropy(0.95), EntropyRung::Syntax);
        assert_eq!(EntropyRung::from_entropy(0.50), EntropyRung::Semantics);
        assert_eq!(EntropyRung::from_entropy(0.05), EntropyRung::Pragmatics);
    }

    #[test]
    fn quadrant_four_corners() {
        assert_eq!(Quadrant::classify(0.8, 0.2), Quadrant::Staunen);
        assert_eq!(Quadrant::classify(0.8, 0.8), Quadrant::Confusion);
        assert_eq!(Quadrant::classify(0.2, 0.2), Quadrant::Boredom);
        assert_eq!(Quadrant::classify(0.2, 0.8), Quadrant::Wisdom);
    }

    #[test]
    fn decompose_zeroes_inactive_components() {
        // Only P active (mask 0b010): s_idx and o_idx must not leak into the key.
        let p = decompose_spo(11, 22, 33, 0b010, 0.5, 0.5);
        assert_eq!(p.active, (false, true, false));
        assert_eq!(p.basin_key, (0b010 << 24) | (22 << 8));
    }

    #[test]
    fn entropy_class_spans_four() {
        assert_eq!(entropy_class(0.0), 0);
        assert_eq!(entropy_class(0.3), 1);
        assert_eq!(entropy_class(0.6), 2);
        assert_eq!(entropy_class(0.99), 3);
    }

    /// Validation: entropy is a reliability proxy. Build a population of edges
    /// whose belief `(f, c)` is estimated from `n_obs` Bernoulli(p) draws, then
    /// measure each edge's empirical prediction accuracy against fresh draws.
    /// `nars_entropy` must ANTI-correlate with accuracy (low entropy ⇒ the edge's
    /// belief reliably matches reality). Grounded via the reliability suite's
    /// Spearman ρ — this is the non-circular gate for the ladder coordinate.
    #[test]
    fn entropy_anticorrelates_with_prediction_accuracy() {
        let mut s = 0xC0FFEE_u64;
        let n = 3000usize;
        let mut entropies = Vec::with_capacity(n);
        let mut accuracies = Vec::with_capacity(n);
        for _ in 0..n {
            let p = splitmix(&mut s); // true Bernoulli rate (ground truth)
            let n_obs = 1 + (splitmix(&mut s) * 48.0) as u32; // 1..=48 observations
            let pos = (0..n_obs).filter(|_| splitmix(&mut s) < p).count() as f64;
            let f = pos / n_obs as f64;
            let c = n_obs as f64 / (n_obs as f64 + 1.0); // NARS horizon-1 confidence
            let e = c * (f - 0.5) + 0.5; // expectation → MAP prediction
            let predict_true = e > 0.5;
            // Empirical accuracy vs 64 fresh outcomes from the true rate p.
            let m = 64;
            let hits = (0..m)
                .filter(|_| (splitmix(&mut s) < p) == predict_true)
                .count() as f64;
            entropies.push(nars_entropy(f, c));
            accuracies.push(hits / m as f64);
        }
        let rho = spearman(&entropies, &accuracies);
        assert!(rho < -0.5, "entropy must anti-correlate with empirical accuracy (reliability proxy), got ρ={rho}");
    }
}
