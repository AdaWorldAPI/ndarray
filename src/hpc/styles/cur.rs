//! #26 Cascading Uncertainty Reduction — CRP percentiles drive resolution.
//! Science: Shannon (1948), Berry-Esseen, Rényi (1961).

use super::lsi::ClusterDistribution;

pub fn cascading_uncertainty(dist: &ClusterDistribution) -> Vec<(u8, f32)> {
    let max_l1 = (17u32 * 65535) as f32;
    vec![
        (1, 1.0 - dist.p25 / max_l1),  // INT1: coarsest
        (4, 1.0 - dist.p50 / max_l1),  // INT4
        (8, 1.0 - dist.p75 / max_l1),  // INT8
        (32, 1.0 - dist.p99 / max_l1), // INT32: finest
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_uncertainty_decreases() {
        let dist = ClusterDistribution { mu: 5000.0, sigma: 1000.0, p25: 4325.5, p50: 5000.0, p75: 5674.5, p95: 6644.9, p99: 7326.3 };
        let levels = cascading_uncertainty(&dist);
        assert!(levels[0].1 > levels[3].1); // coarsest has most uncertainty
    }
}
