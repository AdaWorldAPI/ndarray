//! #30 Shadow Parallel Processing — precompute likely follow-up queries.
//! Science: Kahneman (2011) System 1/2, Friston (2010) free energy.

use super::super::bgz17_bridge::Base17;

pub struct ShadowResult {
    pub predictions: Vec<(usize, u32)>, // (corpus_idx, distance)
}

pub fn precompute_shadows(current: &Base17, corpus: &[Base17], depth: usize, top_k: usize) -> Vec<ShadowResult> {
    // Level 1: neighbors of current
    let mut neighbors: Vec<(usize, u32)> = corpus.iter().enumerate()
        .map(|(i, c)| (i, current.l1(c)))
        .collect();
    neighbors.sort_by_key(|&(_, d)| d);
    neighbors.truncate(top_k);

    let mut results = Vec::new();
    // Level 2: for each neighbor, find ITS neighbors
    if depth > 1 {
        for &(idx, _) in &neighbors {
            let mut sub: Vec<(usize, u32)> = corpus.iter().enumerate()
                .map(|(i, c)| (i, corpus[idx].l1(c)))
                .collect();
            sub.sort_by_key(|&(_, d)| d);
            sub.truncate(top_k);
            results.push(ShadowResult { predictions: sub });
        }
    }
    if results.is_empty() {
        results.push(ShadowResult { predictions: neighbors });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_shadow_precompute() {
        let current = Base17 { dims: [100; 17] };
        let corpus: Vec<Base17> = (0..20).map(|i| { let mut d = [0i16; 17]; d[0] = (i*50) as i16; Base17 { dims: d } }).collect();
        let shadows = precompute_shadows(&current, &corpus, 2, 5);
        assert!(!shadows.is_empty());
        assert!(shadows[0].predictions.len() <= 5);
    }
}
