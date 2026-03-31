//! #21 Self-Skepticism Reinforcement — skepticism grows with consecutive confidence.
//! Science: Descartes (1641), Wang (2006), Tetlock (2005).

use super::super::nars::NarsTruth;

pub struct SkepticismSchedule {
    consecutive_confident: u32,
    base_skepticism: f32,
}

impl SkepticismSchedule {
    pub fn new(base: f32) -> Self { Self { consecutive_confident: 0, base_skepticism: base } }

    pub fn update(&mut self, truth: &NarsTruth) -> f32 {
        if truth.confidence > 0.8 { self.consecutive_confident += 1; }
        else { self.consecutive_confident = 0; }
        self.base_skepticism + (self.consecutive_confident as f32 + 1.0).ln() * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_skepticism_grows() {
        let mut s = SkepticismSchedule::new(0.1);
        let high = NarsTruth::new(0.9, 0.95);
        let s1 = s.update(&high);
        let s2 = s.update(&high);
        let s3 = s.update(&high);
        assert!(s3 > s2);
        assert!(s2 > s1);
    }
    #[test]
    fn test_skepticism_resets() {
        let mut s = SkepticismSchedule::new(0.1);
        s.update(&NarsTruth::new(0.9, 0.95));
        s.update(&NarsTruth::new(0.9, 0.95));
        let after_reset = s.update(&NarsTruth::new(0.5, 0.3));
        let fresh = SkepticismSchedule::new(0.1).update(&NarsTruth::new(0.5, 0.3));
        assert!((after_reset - fresh).abs() < 0.01);
    }
}
