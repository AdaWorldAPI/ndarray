//! #22 Emergent Task Decomposition — subtask structure emerges from corpus clusters.
//! Science: CLAM (Ishaq et al. 2019), Simon (1962), Bengio et al. (2013).

use super::super::bgz17_bridge::Base17;

pub struct Subtask { pub fingerprint: Base17, pub relevance: f32 }

pub fn emergent_decompose(task: &Base17, corpus: &[Base17], max_subtasks: usize) -> Vec<Subtask> {
    let max_l1 = (17u32 * 65535) as f32;
    // Find diverse set: furthest-point sampling from task
    let mut selected = Vec::new();
    let mut min_dists = vec![u32::MAX; corpus.len()];
    for _ in 0..max_subtasks.min(corpus.len()) {
        // Update distances
        let anchor = if selected.is_empty() { task } else { &corpus[*selected.last().unwrap()] };
        for (i, c) in corpus.iter().enumerate() {
            let d = anchor.l1(c);
            if d < min_dists[i] { min_dists[i] = d; }
        }
        // Pick farthest
        let best = min_dists.iter().enumerate()
            .filter(|(i, _)| !selected.contains(i))
            .max_by_key(|(_, d)| *d)
            .map(|(i, _)| i);
        if let Some(idx) = best { selected.push(idx); } else { break; }
    }
    selected.iter().map(|&i| {
        let relevance = 1.0 - task.l1(&corpus[i]) as f32 / max_l1;
        Subtask { fingerprint: corpus[i].clone(), relevance }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_emergent_decompose() {
        let task = Base17 { dims: [100; 17] };
        let corpus: Vec<Base17> = (0..20).map(|i| { let mut d = [0i16; 17]; d[0] = (i*100) as i16; Base17 { dims: d } }).collect();
        let subtasks = emergent_decompose(&task, &corpus, 5);
        assert_eq!(subtasks.len(), 5);
    }
}
