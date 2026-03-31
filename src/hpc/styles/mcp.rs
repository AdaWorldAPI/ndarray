//! #10 Meta-Cognition — monitor confidence reliability over time.

use super::super::nars::NarsTruth;

/// Meta-cognitive assessment result.
pub struct MetaAssessment {
    pub confidence: f32,
    pub meta_confidence: f32,
    pub should_admit_ignorance: bool,
}

/// Meta-cognition monitor: tracks confidence history, computes meta-confidence.
/// Science: Fleming & Dolan (2012), Brier (1950), Yeung & Summerfield (2012).
pub struct MetaCognition {
    history: Vec<f32>,
    max_history: usize,
    calibration_error: f32,
}

impl MetaCognition {
    pub fn new(max_history: usize) -> Self {
        Self { history: Vec::new(), max_history, calibration_error: 0.5 }
    }

    /// Assess meta-confidence: how reliable is our confidence?
    pub fn assess(&mut self, truth: &NarsTruth) -> MetaAssessment {
        let confidence = truth.confidence;
        self.history.push(confidence);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        let mean = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance = self.history.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>() / self.history.len() as f32;

        let meta_confidence = 1.0 - variance.sqrt();

        MetaAssessment {
            confidence,
            meta_confidence,
            should_admit_ignorance: confidence < 0.3 && self.calibration_error > 0.2,
        }
    }

    /// Update calibration error with actual outcome.
    pub fn update_calibration(&mut self, predicted_confidence: f32, was_correct: bool) {
        let outcome = if was_correct { 1.0 } else { 0.0 };
        let brier = (predicted_confidence - outcome).powi(2);
        // Exponential moving average
        self.calibration_error = 0.9 * self.calibration_error + 0.1 * brier;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacog_stable_confidence() {
        let mut mc = MetaCognition::new(100);
        for _ in 0..10 {
            mc.assess(&NarsTruth::new(0.8, 0.9));
        }
        let result = mc.assess(&NarsTruth::new(0.8, 0.9));
        assert!(result.meta_confidence > 0.9); // stable -> high meta-confidence
        assert!(!result.should_admit_ignorance);
    }

    #[test]
    fn test_metacog_unstable_confidence() {
        let mut mc = MetaCognition::new(100);
        for i in 0..10 {
            let c = if i % 2 == 0 { 0.9 } else { 0.1 };
            mc.assess(&NarsTruth::new(0.5, c));
        }
        let result = mc.assess(&NarsTruth::new(0.5, 0.5));
        assert!(result.meta_confidence < 0.7); // unstable -> low meta-confidence
    }

    #[test]
    fn test_calibration_update() {
        let mut mc = MetaCognition::new(100);
        // Overconfident predictions that are wrong
        for _ in 0..10 {
            mc.update_calibration(0.9, false);
        }
        assert!(mc.calibration_error > 0.3);
    }
}
