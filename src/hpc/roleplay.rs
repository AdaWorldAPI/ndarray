//! #9 Iterative Roleplay Synthesis — perspective sweep on Base17 fingerprints.

use super::bgz17_bridge::Base17;

/// One perspective result: which role, what it produced, how novel.
pub struct PerspectiveResult {
    pub role_idx: usize,
    pub result: Base17,
    pub novelty: f32,
}

/// Perspective sweep: each role modulates the query via XOR-analog (dim-wise add),
/// then the nearest in corpus is found. Novelty = L1 from accumulated perspectives.
/// Science: Kanerva (2009) XOR binding, De Bono (1985), Galton (1907).
pub fn perspective_sweep(
    query: &Base17,
    roles: &[Base17],
    corpus: &[Base17],
) -> Vec<PerspectiveResult> {
    let max_l1 = (17u32 * 65535) as f32;
    let mut results = Vec::new();
    let mut seen = query.clone();

    for (idx, role) in roles.iter().enumerate() {
        // Role-modulate query: dim-wise addition (XOR-analog for i16)
        let mut modulated = Base17 { dims: [0; 17] };
        for d in 0..17 {
            modulated.dims[d] = query.dims[d].wrapping_add(role.dims[d]);
        }

        // Find nearest in corpus
        let mut best = corpus.first().cloned().unwrap_or(Base17 { dims: [0; 17] });
        let mut best_dist = u32::MAX;
        for c in corpus {
            let d = modulated.l1(c);
            if d < best_dist {
                best_dist = d;
                best = c.clone();
            }
        }

        // Novelty: how different from accumulated perspectives
        let novelty = best.l1(&seen) as f32 / max_l1;
        results.push(PerspectiveResult { role_idx: idx, result: best.clone(), novelty });

        // Accumulate: running mean
        for d in 0..17 {
            seen.dims[d] = ((seen.dims[d] as i32 + best.dims[d] as i32) / 2) as i16;
        }
    }

    results.sort_by(|a, b| b.novelty.partial_cmp(&a.novelty).unwrap());
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perspective_sweep() {
        let query = Base17 { dims: [100; 17] };
        let roles = vec![
            Base17 { dims: [10; 17] },
            Base17 { dims: [-50; 17] },
            Base17 { dims: [200; 17] },
        ];
        let corpus: Vec<Base17> = (0..20).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 50) as i16;
            Base17 { dims }
        }).collect();

        let results = perspective_sweep(&query, &roles, &corpus);
        assert_eq!(results.len(), 3);
        assert!(results[0].novelty >= results[1].novelty);
    }

    #[test]
    fn test_perspective_novelty() {
        let query = Base17 { dims: [0; 17] };
        let roles = vec![
            Base17 { dims: [0; 17] },
            Base17 { dims: [10000; 17] },
        ];
        let corpus = vec![
            Base17 { dims: [0; 17] },
            Base17 { dims: [10000; 17] },
        ];
        let results = perspective_sweep(&query, &roles, &corpus);
        assert!(results[0].novelty > results[1].novelty);
    }
}
