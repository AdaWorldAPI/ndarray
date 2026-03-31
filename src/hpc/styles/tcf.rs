//! #20 Thought Cascade Filtering — run multiple strategies, filter best.
//! Science: CAKES (Ishaq et al.), Wolpert & Macready (1997) No Free Lunch.

use super::super::bgz17_bridge::Base17;

pub fn cascade_filter(
    query: &Base17,
    corpus: &[Base17],
    quality_fn: &dyn Fn(&Base17) -> f32,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus.iter().enumerate()
        .map(|(i, c)| (i, quality_fn(c)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cascade_filter() {
        let query = Base17 { dims: [100; 17] };
        let corpus: Vec<Base17> = (0..10).map(|i| { let mut d = [0i16; 17]; d[0] = (i * 100) as i16; Base17 { dims: d } }).collect();
        let results = cascade_filter(&query, &corpus, &|c| -(query.l1(c) as f32), 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].1 >= results[1].1); // sorted by quality
    }
}
