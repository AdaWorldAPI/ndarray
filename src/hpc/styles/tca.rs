//! #12 Temporal Context Augmentation — embed timestamps into Base17 fingerprints.

use super::super::bgz17_bridge::Base17;

/// Temporal context: event time, reference time, speech time (Reichenbach).
pub struct TemporalContext {
    pub event_time: u64,
    pub reference_time: u64,
    pub speech_time: u64,
}

/// Temporally augmented Base17: original fingerprint + temporal signal.
pub struct TemporalFingerprint {
    pub base: Base17,
    pub temporal: TemporalContext,
    pub recency: f32,
}

/// Augment a Base17 fingerprint with temporal context.
/// Recency decays with time distance from reference.
/// Science: Reichenbach (1947), Kamp & Reyle (1993 Ch.5), Vendler (1957).
pub fn temporalize(
    base: &Base17,
    event_time: u64,
    reference_time: u64,
) -> TemporalFingerprint {
    let speech_time = reference_time; // default: now = reference
    let time_delta = if event_time > reference_time {
        event_time - reference_time
    } else {
        reference_time - event_time
    };
    // Exponential decay: recency = exp(-delta / scale)
    let scale = 3600.0; // 1 hour in seconds
    let recency = (-1.0 * time_delta as f64 / scale).exp() as f32;

    TemporalFingerprint {
        base: base.clone(),
        temporal: TemporalContext { event_time, reference_time, speech_time },
        recency,
    }
}

/// Temporal similarity: combine Base17 L1 with recency weighting.
pub fn temporal_similarity(a: &TemporalFingerprint, b: &TemporalFingerprint) -> (u32, f32) {
    let spatial = a.base.l1(&b.base);
    let temporal_weight = (a.recency * b.recency).sqrt();
    (spatial, temporal_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporalize_recent() {
        let base = Base17 { dims: [100; 17] };
        let tf = temporalize(&base, 1000, 1000); // event = now
        assert!((tf.recency - 1.0).abs() < 0.01); // very recent
    }

    #[test]
    fn test_temporalize_old() {
        let base = Base17 { dims: [100; 17] };
        let tf = temporalize(&base, 0, 36000); // 10 hours ago
        assert!(tf.recency < 0.01); // very old
    }

    #[test]
    fn test_temporal_similarity() {
        let base = Base17 { dims: [100; 17] };
        let recent = temporalize(&base, 1000, 1000);
        let old = temporalize(&base, 0, 1000);
        let (spatial, weight) = temporal_similarity(&recent, &old);
        assert_eq!(spatial, 0); // same base -> 0 spatial distance
        assert!(weight < 1.0); // temporal discount
    }
}
