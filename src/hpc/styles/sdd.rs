//! #32 Semantic Distortion Detection — measure meaning drift with Berry-Esseen floor.
//! Science: Shannon (1948), Berry-Esseen, Cohen (1988).

use super::super::bgz17_bridge::Base17;
use super::lsi::ClusterDistribution;

pub struct DistortionReport {
    pub information_loss: f32,
    pub structural_drift: f32,
    pub z_score: f32,
}

pub fn detect_distortion(original: &Base17, transformed: &Base17, dist: &ClusterDistribution) -> DistortionReport {
    let raw = original.l1(transformed) as f32;
    let noise_floor = dist.sigma * 0.01; // Berry-Esseen noise floor for Base17
    DistortionReport {
        information_loss: (raw - noise_floor).max(0.0) / (17.0 * 65535.0),
        structural_drift: if dist.mu > 0.0 { raw / dist.mu } else { 0.0 },
        z_score: if dist.sigma > 0.0 { (raw - dist.p50) / dist.sigma } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_no_distortion() {
        let a = Base17 { dims: [100; 17] };
        let dist = ClusterDistribution { mu: 5000.0, sigma: 1000.0, p25: 4325.0, p50: 5000.0, p75: 5675.0, p95: 6645.0, p99: 7326.0 };
        let report = detect_distortion(&a, &a, &dist);
        assert_eq!(report.information_loss, 0.0);
        assert!(report.z_score < 0.0); // below median
    }
    #[test]
    fn test_high_distortion() {
        let a = Base17 { dims: [0; 17] };
        let b = Base17 { dims: [30000; 17] };
        let dist = ClusterDistribution { mu: 5000.0, sigma: 1000.0, p25: 4325.0, p50: 5000.0, p75: 5675.0, p95: 6645.0, p99: 7326.0 };
        let report = detect_distortion(&a, &b, &dist);
        assert!(report.z_score > 2.0); // way above p99
        assert!(report.structural_drift > 1.0);
    }
}
