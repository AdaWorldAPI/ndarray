//! #15 Latent Space Introspection — CRP distribution from Base17 distances.
//! Science: Fisher (1925), Berry-Esseen (1941/42), Cohen (1988).

pub struct ClusterDistribution {
    pub mu: f32,
    pub sigma: f32,
    pub p25: f32, pub p50: f32, pub p75: f32, pub p95: f32, pub p99: f32,
}

impl ClusterDistribution {
    pub fn from_distances(distances: &[u32]) -> Self {
        if distances.is_empty() { return Self { mu: 0.0, sigma: 0.0, p25: 0.0, p50: 0.0, p75: 0.0, p95: 0.0, p99: 0.0 }; }
        let n = distances.len() as f32;
        let mu = distances.iter().sum::<u32>() as f32 / n;
        let sigma = (distances.iter().map(|d| (*d as f32 - mu).powi(2)).sum::<f32>() / n).sqrt();
        Self { mu, sigma, p25: mu - 0.6745 * sigma, p50: mu, p75: mu + 0.6745 * sigma, p95: mu + 1.6449 * sigma, p99: mu + 2.3263 * sigma }
    }

    pub fn mexican_hat(&self, distance: f32) -> f32 {
        if distance < self.p25 { 1.0 }
        else if distance < self.p75 { 0.5 }
        else if distance < self.p95 { 0.0 }
        else if distance < self.p99 { -0.5 }
        else { -1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_crp() {
        let dists: Vec<u32> = (0..100).map(|i| 1000 + i * 10).collect();
        let crp = ClusterDistribution::from_distances(&dists);
        assert!(crp.mu > 0.0);
        assert!(crp.p25 < crp.p50);
        assert!(crp.p50 < crp.p95);
        assert_eq!(crp.mexican_hat(crp.p25 - 1.0), 1.0);
        assert_eq!(crp.mexican_hat(crp.p99 + 1.0), -1.0);
    }
}
