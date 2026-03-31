//! #3 Structured Multi-Agent Debate — bundle + NARS revision on Base17 fingerprints.

use super::super::bgz17_bridge::Base17;
use super::super::nars::{NarsTruth, nars_revision};

/// One proposition in a debate: a fingerprint + truth value.
pub struct Proposition {
    pub fingerprint: Base17,
    pub truth: NarsTruth,
}

/// Result of a debate round.
pub struct DebateResult {
    pub consensus: Base17,
    pub truth: NarsTruth,
    pub rounds: u8,
    pub propositions: Vec<Proposition>,
}

/// Run structured debate: each "agent" is a Base17 perspective.
/// Perspectives are bundled (majority vote per dim), truth values revised.
/// Science: Wang (2006), Du et al. (2023), Kanerva (2009).
pub fn debate(
    input: &Base17,
    perspectives: &[Base17],
    rounds: u8,
) -> DebateResult {
    let mut propositions = Vec::new();

    for perspective in perspectives {
        // Each perspective "transforms" input by finding the nearest-like pattern
        // The L1 distance becomes evidence strength
        let dist = input.l1(perspective);
        let max_l1 = (17u32 * 65535) as f32;
        let resonance = 1.0 - (dist as f32 / max_l1);
        let truth = NarsTruth::from_evidence(
            resonance * 10.0,        // positive evidence proportional to resonance
            (1.0 - resonance) * 10.0, // negative evidence proportional to distance
        );
        propositions.push(Proposition { fingerprint: perspective.clone(), truth });
    }

    // Bundle: majority vote per dimension (mean of i16 values)
    let consensus = bundle_base17(&propositions.iter().map(|p| &p.fingerprint).collect::<Vec<_>>());

    // NARS revision across all truth values
    let mut consensus_truth = NarsTruth::new(0.5, 0.0);
    for prop in &propositions {
        consensus_truth = nars_revision(consensus_truth, prop.truth);
    }

    DebateResult { consensus, truth: consensus_truth, rounds, propositions }
}

/// Bundle Base17 fingerprints: mean per dimension (majority vote analog).
fn bundle_base17(fps: &[&Base17]) -> Base17 {
    if fps.is_empty() { return Base17 { dims: [0; 17] }; }
    let n = fps.len() as i32;
    let mut sums = [0i32; 17];
    for fp in fps {
        for d in 0..17 { sums[d] += fp.dims[d] as i32; }
    }
    let mut dims = [0i16; 17];
    for d in 0..17 { dims[d] = (sums[d] / n) as i16; }
    Base17 { dims }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debate_consensus() {
        let input = Base17 { dims: [100; 17] };
        let perspectives = vec![
            Base17 { dims: [90; 17] },
            Base17 { dims: [110; 17] },
            Base17 { dims: [100; 17] },
        ];
        let result = debate(&input, &perspectives, 1);
        assert_eq!(result.propositions.len(), 3);
        assert!(result.truth.confidence > 0.0);
        // Consensus should be near 100 (mean of 90, 110, 100)
        assert!((result.consensus.dims[0] - 100).abs() <= 7);
    }

    #[test]
    fn test_debate_truth_accumulates() {
        let input = Base17 { dims: [50; 17] };
        let perspectives: Vec<Base17> = (0..5).map(|i| {
            let mut dims = [50i16; 17];
            dims[0] += (i * 10) as i16;
            Base17 { dims }
        }).collect();
        let result = debate(&input, &perspectives, 1);
        // More perspectives → higher confidence
        assert!(result.truth.confidence > 0.5);
    }
}
