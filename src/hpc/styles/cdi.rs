//! #17 Cognitive Dissonance Induction — create productive tension.
//! Science: Festinger (1957), Berlyne (1960), Peng & Nisbett (1999).

use super::super::bgz17_bridge::Base17;
use super::super::nars::NarsTruth;

pub fn induce_dissonance(belief: &Base17, truth: &NarsTruth, corpus: &[Base17]) -> (Base17, NarsTruth) {
    // Find structurally similar but maximally different item
    let max_l1 = (17u32 * 65535) as f32;
    let mut best_tension = 0.0f32;
    let mut best = belief.clone();
    for c in corpus {
        let similarity = 1.0 - belief.l1(c) as f32 / max_l1;
        if similarity > 0.3 && similarity < 0.7 {
            let tension = similarity * (1.0 - similarity); // Maximum at 0.5
            if tension > best_tension { best_tension = tension; best = c.clone(); }
        }
    }
    // Dissonance = midpoint between belief and its tension partner
    let mut dims = [0i16; 17];
    for d in 0..17 { dims[d] = ((belief.dims[d] as i32 + best.dims[d] as i32) / 2) as i16; }
    let dissonance = Base17 { dims };
    let dissonant_truth = NarsTruth::new(0.5, truth.confidence * 0.5);
    (dissonance, dissonant_truth)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dissonance() {
        let belief = Base17 { dims: [100; 17] };
        let truth = NarsTruth::new(0.9, 0.8);
        let corpus = vec![Base17 { dims: [120; 17] }, Base17 { dims: [500; 17] }];
        let (dis, dt) = induce_dissonance(&belief, &truth, &corpus);
        assert_eq!(dt.frequency, 0.5); // maximum uncertainty
        assert!(dt.confidence < truth.confidence);
    }
}
