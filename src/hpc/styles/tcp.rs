//! #5 Thought Chain Pruning — Berry-Esseen noise floor on Base17 chains.

use super::super::bgz17_bridge::Base17;

pub struct ChainPruner {
    pub noise_floor: f32,
    pub max_branches: usize,
}

impl ChainPruner {
    /// Default: Berry-Esseen noise floor at d=17 (Base17 dimensions).
    pub fn new(max_branches: usize) -> Self {
        Self { noise_floor: 0.01, max_branches }
    }

    /// Prune chain: keep branches where L1 from accumulated bundle exceeds noise floor.
    /// Science: CAKES triangle inequality, Berry-Esseen, Rissanen (1978) MDL.
    pub fn prune(&self, chain: &[Base17]) -> Vec<usize> {
        let max_l1 = (17u32 * 65535) as f32;
        let mut kept = vec![0]; // Always keep root
        let mut bundle = chain[0].clone();

        for i in 1..chain.len() {
            let novelty = chain[i].l1(&bundle) as f32 / max_l1;
            if novelty > self.noise_floor {
                kept.push(i);
                // Update bundle: running mean
                for d in 0..17 {
                    bundle.dims[d] = ((bundle.dims[d] as i32 + chain[i].dims[d] as i32) / 2) as i16;
                }
            }
            if kept.len() >= self.max_branches { break; }
        }
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_keeps_novel() {
        let chain = vec![
            Base17 { dims: [0; 17] },
            Base17 { dims: [1000; 17] },  // very different -> keep
            Base17 { dims: [1; 17] },     // near duplicate of bundle -> prune
            Base17 { dims: [2000; 17] },  // very different -> keep
        ];
        let pruner = ChainPruner::new(10);
        let kept = pruner.prune(&chain);
        assert!(kept.contains(&0)); // root
        assert!(kept.contains(&1)); // novel
        assert!(kept.contains(&3)); // novel
    }

    #[test]
    fn test_prune_respects_max() {
        let chain: Vec<Base17> = (0..20).map(|i| {
            Base17 { dims: [(i * 1000) as i16; 17] }
        }).collect();
        let pruner = ChainPruner::new(3);
        let kept = pruner.prune(&chain);
        assert!(kept.len() <= 3);
    }
}
