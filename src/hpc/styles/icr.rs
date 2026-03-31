//! #31 Iterative Counterfactual Reasoning — systematic "what if" via Base17 binding.
//! Science: Pearl (2009), Lewis (1973), Squires & Uhler (2023).

use super::super::bgz17_bridge::Base17;
use super::super::nars::NarsTruth;

pub struct CounterfactualWorld {
    pub intervention_idx: usize,
    pub resulting: Base17,
    pub divergence: f32,
    pub truth: NarsTruth,
}

pub fn iterate_counterfactuals(
    base: &Base17,
    interventions: &[Base17],
    corpus: &[Base17],
) -> Vec<CounterfactualWorld> {
    let max_l1 = (17u32 * 65535) as f32;
    interventions.iter().enumerate().map(|(idx, intervention)| {
        let mut modified_dims = [0i16; 17];
        for d in 0..17 { modified_dims[d] = base.dims[d].wrapping_add(intervention.dims[d]); }
        let modified = Base17 { dims: modified_dims };
        let mut best_dist = u32::MAX;
        let mut best = modified.clone();
        for c in corpus {
            let d = modified.l1(c);
            if d < best_dist { best_dist = d; best = c.clone(); }
        }
        let divergence = base.l1(&best) as f32 / max_l1;
        let confidence = if best_dist < (max_l1 as u32) / 2 { 0.8 } else { 0.3 };
        CounterfactualWorld {
            intervention_idx: idx,
            resulting: best,
            divergence,
            truth: NarsTruth::new(1.0 - divergence, confidence),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_counterfactual() {
        let base = Base17 { dims: [100; 17] };
        let interventions = vec![Base17 { dims: [50; 17] }, Base17 { dims: [-200; 17] }];
        let corpus = vec![Base17 { dims: [150; 17] }, Base17 { dims: [-100; 17] }, Base17 { dims: [0; 17] }];
        let worlds = iterate_counterfactuals(&base, &interventions, &corpus);
        assert_eq!(worlds.len(), 2);
        assert!(worlds[0].divergence >= 0.0);
    }
}
