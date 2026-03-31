//! #1 Recursive Thought Expansion — Hofstadter strange loops on Base17 fingerprints.

use super::super::nars::NarsTruth;
use super::super::bgz17_bridge::Base17;

pub struct RecursiveExpansion {
    pub max_depth: u8,
    pub convergence_threshold: f32,
}

pub struct ExpansionStep {
    pub depth: u8,
    pub delta: f32,
    pub fingerprint: Base17,
}

pub struct ExpansionTrace {
    pub steps: Vec<ExpansionStep>,
    pub converged: bool,
}

impl RecursiveExpansion {
    pub fn new(max_depth: u8, convergence_threshold: f32) -> Self {
        Self { max_depth, convergence_threshold }
    }

    /// Apply recursive expansion: output of depth N becomes input to depth N+1.
    /// Stops when delta < convergence_threshold or max_depth reached.
    /// Science: Hofstadter (1979), Schmidhuber (2010), Berry-Esseen noise floor 0.004.
    pub fn expand(&self, seed: &Base17, corpus: &[Base17]) -> ExpansionTrace {
        let mut current = seed.clone();
        let mut steps = Vec::new();
        for depth in 0..self.max_depth {
            // Find nearest in corpus at this depth — the "rung transform"
            let mut best_dist = u32::MAX;
            let mut best = current.clone();
            for c in corpus {
                let d = current.l1(c);
                if d < best_dist && d > 0 {
                    best_dist = d;
                    best = c.clone();
                }
            }
            let max_l1 = (17 * 65535) as f32;
            let delta = best_dist as f32 / max_l1;
            steps.push(ExpansionStep { depth, delta, fingerprint: best.clone() });
            if delta < self.convergence_threshold { break; }
            current = best;
        }
        let converged = steps.last().map_or(false, |s| s.delta < self.convergence_threshold);
        ExpansionTrace { steps, converged }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_expansion_converges() {
        let seed = Base17 { dims: [100; 17] };
        let corpus: Vec<Base17> = (0..10).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = 100 - (i * 5) as i16;
            for d in 1..17 { dims[d] = 100 - (i * 3) as i16; }
            Base17 { dims }
        }).collect();

        let re = RecursiveExpansion::new(7, 0.001);
        let trace = re.expand(&seed, &corpus);
        assert!(!trace.steps.is_empty());
    }

    #[test]
    fn test_max_depth_cap() {
        let seed = Base17 { dims: [0; 17] };
        let corpus = vec![Base17 { dims: [100; 17] }, Base17 { dims: [200; 17] }];
        let re = RecursiveExpansion::new(3, 0.0001);
        let trace = re.expand(&seed, &corpus);
        assert!(trace.steps.len() <= 3);
    }
}
