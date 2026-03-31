//! #13 Convergent & Divergent Thinking — oscillate between exploration and exploitation.
//! Science: Guilford (1967), Kanerva (2009), Sutton & Barto (2018).

use super::super::bgz17_bridge::Base17;

pub fn oscillate(query: &Base17, corpus: &[Base17], rounds: usize) -> (Base17, Vec<f32>) {
    let mut current = query.clone();
    let mut ratios = Vec::new();
    for round in 0..rounds {
        if round % 2 == 0 {
            // Diverge: bundle with farthest neighbors (mean of distant items)
            let mut farthest: Vec<(u32, usize)> = corpus.iter().enumerate()
                .map(|(i, c)| (current.l1(c), i)).collect();
            farthest.sort_by(|a, b| b.0.cmp(&a.0));
            let top5: Vec<&Base17> = farthest.iter().take(5).map(|(_, i)| &corpus[*i]).collect();
            current = bundle_base17(&top5, &current);
            ratios.push(1.0);
        } else {
            // Converge: snap to nearest
            let mut best_dist = u32::MAX;
            let mut best = current.clone();
            for c in corpus {
                let d = current.l1(c);
                if d < best_dist && d > 0 { best_dist = d; best = c.clone(); }
            }
            current = best;
            ratios.push(0.0);
        }
    }
    (current, ratios)
}

fn bundle_base17(items: &[&Base17], seed: &Base17) -> Base17 {
    let n = items.len() as i32 + 1;
    let mut dims = [0i32; 17];
    for d in 0..17 { dims[d] += seed.dims[d] as i32; }
    for item in items { for d in 0..17 { dims[d] += item.dims[d] as i32; } }
    let mut result = [0i16; 17];
    for d in 0..17 { result[d] = (dims[d] / n) as i16; }
    Base17 { dims: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_oscillate() {
        let query = Base17 { dims: [100; 17] };
        let corpus: Vec<Base17> = (0..20).map(|i| { let mut d = [0i16; 17]; d[0] = (i*50) as i16; Base17 { dims: d } }).collect();
        let (result, ratios) = oscillate(&query, &corpus, 4);
        assert_eq!(ratios.len(), 4);
        assert_eq!(ratios[0], 1.0); // diverge
        assert_eq!(ratios[1], 0.0); // converge
    }
}
