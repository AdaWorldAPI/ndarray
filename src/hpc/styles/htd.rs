//! #2 Hierarchical Thought Decomposition — CLAM-style bipolar split on Base17.

use super::super::bgz17_bridge::Base17;

pub struct DecompositionNode {
    pub centroid: Base17,
    pub radius: u32,
    pub count: usize,
    pub children: Vec<DecompositionNode>,
}

pub struct DecompositionTree {
    pub root: DecompositionNode,
    pub depth: usize,
}

/// Hierarchical decompose: CLAM-style bipolar split.
/// Find medoid, find farthest, partition into two clusters, recurse.
/// Science: Ishaq et al. (2019), Dasgupta & Long (2005), Simon (1962).
pub fn hierarchical_decompose(
    _query: &Base17,
    corpus: &[Base17],
    max_levels: usize,
) -> DecompositionTree {
    let root = decompose_recursive(corpus, max_levels, 0);
    let depth = tree_depth(&root);
    DecompositionTree { root, depth }
}

fn decompose_recursive(items: &[Base17], max_levels: usize, level: usize) -> DecompositionNode {
    if items.is_empty() {
        return DecompositionNode {
            centroid: Base17 { dims: [0; 17] },
            radius: 0, count: 0, children: Vec::new(),
        };
    }

    // Compute centroid (mean of all items)
    let centroid = compute_centroid(items);
    let radius = items.iter().map(|i| centroid.l1(i)).max().unwrap_or(0);

    if items.len() <= 2 || level >= max_levels {
        return DecompositionNode { centroid, radius, count: items.len(), children: Vec::new() };
    }

    // Bipolar split: find farthest from centroid, partition
    let farthest_idx = items.iter().enumerate()
        .max_by_key(|(_, i)| centroid.l1(i))
        .map(|(idx, _)| idx).unwrap_or(0);

    let pole = &items[farthest_idx];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for item in items {
        if centroid.l1(item) <= pole.l1(item) {
            left.push(item.clone());
        } else {
            right.push(item.clone());
        }
    }

    // Guard against degenerate splits
    if left.is_empty() || right.is_empty() {
        return DecompositionNode { centroid, radius, count: items.len(), children: Vec::new() };
    }

    let children = vec![
        decompose_recursive(&left, max_levels, level + 1),
        decompose_recursive(&right, max_levels, level + 1),
    ];

    DecompositionNode { centroid, radius, count: items.len(), children }
}

fn compute_centroid(items: &[Base17]) -> Base17 {
    let n = items.len() as i32;
    let mut dims = [0i32; 17];
    for item in items {
        for d in 0..17 { dims[d] += item.dims[d] as i32; }
    }
    let mut result = [0i16; 17];
    for d in 0..17 { result[d] = (dims[d] / n) as i16; }
    Base17 { dims: result }
}

fn tree_depth(node: &DecompositionNode) -> usize {
    if node.children.is_empty() { 0 }
    else { 1 + node.children.iter().map(tree_depth).max().unwrap_or(0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_basic() {
        let corpus: Vec<Base17> = (0..20).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100) as i16;
            Base17 { dims }
        }).collect();

        let query = Base17 { dims: [500; 17] };
        let tree = hierarchical_decompose(&query, &corpus, 4);
        assert!(tree.depth > 0);
        assert_eq!(tree.root.count, 20);
    }

    #[test]
    fn test_decompose_small() {
        let corpus = vec![Base17 { dims: [0; 17] }, Base17 { dims: [100; 17] }];
        let query = Base17 { dims: [50; 17] };
        let tree = hierarchical_decompose(&query, &corpus, 4);
        assert_eq!(tree.root.count, 2);
    }
}
