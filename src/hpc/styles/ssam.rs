//! #28 Self-Supervised Analogical Mapping — structural analogy via Base17 binding.
//! Science: Gentner (1983), Plate (2003), Turney (2006).

use super::super::bgz17_bridge::Base17;

pub struct AnalogyResult {
    pub source_idx: usize,
    pub predicted: Base17,
    pub strength: f32,
}

pub fn structural_analogy(relation: &Base17, domain: &[Base17], corpus: &[Base17]) -> Vec<AnalogyResult> {
    let max_l1 = (17u32 * 65535) as f32;
    domain.iter().enumerate().filter_map(|(idx, c)| {
        let mut predicted_dims = [0i16; 17];
        for d in 0..17 { predicted_dims[d] = c.dims[d].wrapping_add(relation.dims[d]); }
        let predicted = Base17 { dims: predicted_dims };
        let mut best_dist = u32::MAX;
        let mut best = predicted.clone();
        for target in corpus {
            let d = predicted.l1(target);
            if d < best_dist { best_dist = d; best = target.clone(); }
        }
        let strength = 1.0 - best_dist as f32 / max_l1;
        if strength > 0.6 { Some(AnalogyResult { source_idx: idx, predicted: best, strength }) } else { None }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_analogy() {
        let relation = Base17 { dims: [10; 17] }; // offset = +10
        let domain = vec![Base17 { dims: [100; 17] }];
        let corpus = vec![Base17 { dims: [110; 17] }, Base17 { dims: [500; 17] }];
        let results = structural_analogy(&relation, &domain, &corpus);
        assert!(!results.is_empty());
        assert!(results[0].strength > 0.9);
    }
}
