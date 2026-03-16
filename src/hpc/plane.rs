//! Plane: the i8 accumulator substrate for holographic cognition.
//!
//! One dimension of cognition. 16,384 bits of signal (L1 cache resident).
//! The i8 accumulator IS the ground truth. Everything else is derived:
//!   bits = sign(acc), alpha = |acc| > threshold, truth = alpha density.
//!
//! NaN is structurally impossible: i8 saturating arithmetic, no floats.
//! Width mismatch handled gracefully: compare on shorter prefix, alpha=0 on remainder.
//!
//! SIMD wired through: all bulk operations delegate to ndarray's bitwise dispatch.

use super::fingerprint::Fingerprint;
use std::cell::RefCell;

// ============================================================================
// Accumulator — 64-byte aligned for AVX-512
// ============================================================================

/// 16,384 i8 slots. 16 KB. Fits L1 cache. 64-byte aligned for AVX-512 loads.
#[repr(C, align(64))]
pub struct Acc16K {
    pub values: [i8; 16384],
}

impl Default for Acc16K {
    fn default() -> Self {
        Self {
            values: [0i8; 16384],
        }
    }
}

impl Clone for Acc16K {
    fn clone(&self) -> Self {
        let mut new = Self::default();
        new.values.copy_from_slice(&self.values);
        new
    }
}

// ============================================================================
// Plane — the core type
// ============================================================================

/// One dimension of cognition. 16,384 bits (256 × u64 words).
pub struct Plane {
    acc: Box<Acc16K>,
    bits: Fingerprint<256>,
    alpha: Fingerprint<256>,
    dirty: bool,
    encounters: u32,
}

impl Clone for Plane {
    fn clone(&self) -> Self {
        Self {
            acc: self.acc.clone(),
            bits: self.bits.clone(),
            alpha: self.alpha.clone(),
            dirty: self.dirty,
            encounters: self.encounters,
        }
    }
}

pub const PLANE_BITS: usize = 16384;
pub const PLANE_BYTES: usize = 2048;

impl Plane {
    pub const BITS: usize = PLANE_BITS;
    pub const BYTES: usize = PLANE_BYTES;

    pub fn new() -> Self {
        Self {
            acc: Box::new(Acc16K::default()),
            bits: Fingerprint::zero(),
            alpha: Fingerprint::zero(),
            dirty: false,
            encounters: 0,
        }
    }

    #[inline]
    pub fn encounters(&self) -> u32 {
        self.encounters
    }

    #[inline]
    pub fn bits(&mut self) -> &Fingerprint<256> {
        self.ensure_cache();
        &self.bits
    }

    #[inline]
    pub fn alpha(&mut self) -> &Fingerprint<256> {
        self.ensure_cache();
        &self.alpha
    }

    pub(crate) fn bits_bytes_ref(&self) -> &[u8] {
        self.bits.as_bytes()
    }

    pub(crate) fn alpha_bytes_ref(&self) -> &[u8] {
        self.alpha.as_bytes()
    }

    // ========================================================================
    // Encounter — evidence arrives
    // ========================================================================

    #[allow(clippy::needless_range_loop)]
    pub fn encounter_bits(&mut self, evidence: &Fingerprint<256>) {
        let bit_bytes = evidence.as_bytes();
        let acc = &mut self.acc.values;
        for k in 0..Self::BITS {
            let byte_idx = k / 8;
            let bit_idx = k % 8;
            let is_set = (bit_bytes[byte_idx] >> bit_idx) & 1 == 1;
            if is_set {
                acc[k] = acc[k].saturating_add(1);
            } else {
                acc[k] = acc[k].saturating_sub(1);
            }
        }
        self.encounters += 1;
        self.dirty = true;
    }

    pub fn encounter(&mut self, text: &str) {
        let fp = Self::text_to_fingerprint(text);
        self.encounter_bits(&fp);
    }

    fn text_to_fingerprint(text: &str) -> Fingerprint<256> {
        let hash = blake3::hash(text.as_bytes());
        let seed = hash.as_bytes();
        let mut output = vec![0u8; PLANE_BYTES];
        let mut hasher = blake3::Hasher::new_keyed(seed);
        hasher.update(text.as_bytes());
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut output);
        Fingerprint::from_bytes(&output)
    }

    // ========================================================================
    // Cache refresh
    // ========================================================================

    #[allow(clippy::needless_range_loop)]
    pub(crate) fn ensure_cache(&mut self) {
        if !self.dirty {
            return;
        }
        let threshold = self.alpha_threshold();
        let acc = &self.acc.values;
        for k in 0..Self::BITS {
            let word = k / 64;
            let bit = k % 64;
            if acc[k] > 0 {
                self.bits.words[word] |= 1u64 << bit;
            } else {
                self.bits.words[word] &= !(1u64 << bit);
            }
            if acc[k].unsigned_abs() > threshold {
                self.alpha.words[word] |= 1u64 << bit;
            } else {
                self.alpha.words[word] &= !(1u64 << bit);
            }
        }
        self.dirty = false;
    }

    fn alpha_threshold(&self) -> u8 {
        match self.encounters {
            0..=1 => 0,
            2..=5 => self.encounters as u8 / 2,
            6..=20 => self.encounters as u8 * 2 / 5,
            _ => {
                let isqrt = integer_sqrt(self.encounters);
                ((isqrt * 4) / 5).min(127) as u8
            }
        }
    }

    // ========================================================================
    // Distance — SIMD-accelerated, alpha-aware
    // ========================================================================

    pub fn distance(&mut self, other: &mut Plane) -> Distance {
        self.ensure_cache();
        other.ensure_cache();
        distance_slices(
            self.bits_bytes_ref(),
            self.alpha_bytes_ref(),
            other.bits_bytes_ref(),
            other.alpha_bytes_ref(),
        )
    }

    // ========================================================================
    // Truth — integer NARS truth
    // ========================================================================

    pub fn truth(&mut self) -> Truth {
        self.ensure_cache();
        let total_bits = Self::BITS as u32;
        let defined = super::bitwise::popcount_raw(self.alpha.as_bytes()) as u32;

        let mut buf = vec![0u8; Self::BYTES];
        let bits_bytes = self.bits.as_bytes();
        let alpha_bytes = self.alpha.as_bytes();
        for i in 0..Self::BYTES {
            buf[i] = bits_bytes[i] & alpha_bytes[i];
        }
        let positive = super::bitwise::popcount_raw(&buf) as u32;

        let frequency = if defined == 0 {
            32768u16
        } else {
            ((positive as u64 * 65535) / defined as u64) as u16
        };
        let confidence = ((defined as u64 * 65535) / total_bits as u64) as u16;

        Truth {
            frequency,
            confidence,
            evidence: self.encounters,
        }
    }

    #[inline]
    pub fn acc(&self) -> &[i8; 16384] {
        &self.acc.values
    }
}

impl Default for Plane {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Distance — the enum, not a float
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Measured {
        disagreement: u32,
        overlap: u32,
        penalty: u32,
    },
    Incomparable,
}

impl Distance {
    #[inline]
    pub fn normalized(&self) -> Option<f32> {
        match self {
            Distance::Measured {
                disagreement,
                overlap,
                penalty,
            } => {
                let denom = overlap + penalty;
                if denom == 0 {
                    return None;
                }
                Some((*disagreement + *penalty) as f32 / denom as f32)
            }
            Distance::Incomparable => None,
        }
    }

    #[inline]
    pub fn closer_than(&self, max_disagreement: u32) -> bool {
        match self {
            Distance::Measured { disagreement, .. } => *disagreement <= max_disagreement,
            Distance::Incomparable => false,
        }
    }

    #[inline]
    pub fn raw(&self) -> Option<u32> {
        match self {
            Distance::Measured { disagreement, .. } => Some(*disagreement),
            Distance::Incomparable => None,
        }
    }
}

// ============================================================================
// Truth — integer NARS truth value
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truth {
    pub frequency: u16,
    pub confidence: u16,
    pub evidence: u32,
}

impl Truth {
    #[inline]
    pub fn frequency_f32(&self) -> f32 {
        self.frequency as f32 / 65535.0
    }

    #[inline]
    pub fn confidence_f32(&self) -> f32 {
        self.confidence as f32 / 65535.0
    }

    #[inline]
    pub fn expectation(&self) -> u16 {
        let f = self.frequency as i32;
        let c = self.confidence as i32;
        let centered = f - 32768;
        let weighted = (c * centered) / 65535;
        (weighted + 32768).clamp(0, 65535) as u16
    }

    pub fn revision(&self, other: &Truth) -> Truth {
        let total_evidence = self.evidence.saturating_add(other.evidence);
        if total_evidence == 0 {
            return Truth {
                frequency: 32768,
                confidence: 0,
                evidence: 0,
            };
        }
        let f = ((self.frequency as u64 * self.evidence as u64)
            + (other.frequency as u64 * other.evidence as u64))
            / total_evidence as u64;
        let c = ((self.confidence as u64 * self.evidence as u64)
            + (other.confidence as u64 * other.evidence as u64))
            / total_evidence as u64;
        Truth {
            frequency: f.min(65535) as u16,
            confidence: c.min(65535) as u16,
            evidence: total_evidence,
        }
    }
}

// ============================================================================
// Free functions
// ============================================================================

#[repr(C, align(64))]
pub struct DistanceScratch {
    masked_xor: [u8; PLANE_BYTES],
    shared_alpha: [u8; PLANE_BYTES],
    not_alpha: [u8; PLANE_BYTES],
}

impl DistanceScratch {
    fn new() -> Self {
        Self {
            masked_xor: [0u8; PLANE_BYTES],
            shared_alpha: [0u8; PLANE_BYTES],
            not_alpha: [0u8; PLANE_BYTES],
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<DistanceScratch> = RefCell::new(DistanceScratch::new());
}

pub fn distance_slices(a_bits: &[u8], a_alpha: &[u8], b_bits: &[u8], b_alpha: &[u8]) -> Distance {
    let shared_len = a_bits.len().min(b_bits.len()).min(a_alpha.len()).min(b_alpha.len());
    if shared_len == 0 {
        return Distance::Incomparable;
    }
    let a = &a_bits[..shared_len];
    let b = &b_bits[..shared_len];
    let aa = &a_alpha[..shared_len];
    let ba = &b_alpha[..shared_len];

    let (disagreement, overlap, penalty) = if shared_len <= PLANE_BYTES {
        SCRATCH.with(|cell| {
            let scratch = &mut *cell.borrow_mut();
            for i in 0..shared_len {
                let xor = a[i] ^ b[i];
                scratch.shared_alpha[i] = aa[i] & ba[i];
                scratch.masked_xor[i] = xor & scratch.shared_alpha[i];
                scratch.not_alpha[i] = !aa[i];
            }
            (
                super::bitwise::popcount_raw(&scratch.masked_xor[..shared_len]) as u32,
                super::bitwise::popcount_raw(&scratch.shared_alpha[..shared_len]) as u32,
                super::bitwise::popcount_raw(&scratch.not_alpha[..shared_len]) as u32,
            )
        })
    } else {
        let mut shared_alpha_buf = vec![0u8; shared_len];
        let mut masked_xor_buf = vec![0u8; shared_len];
        let mut not_alpha_buf = vec![0u8; shared_len];
        for i in 0..shared_len {
            let xor = a[i] ^ b[i];
            shared_alpha_buf[i] = aa[i] & ba[i];
            masked_xor_buf[i] = xor & shared_alpha_buf[i];
            not_alpha_buf[i] = !aa[i];
        }
        (
            super::bitwise::popcount_raw(&masked_xor_buf) as u32,
            super::bitwise::popcount_raw(&shared_alpha_buf) as u32,
            super::bitwise::popcount_raw(&not_alpha_buf) as u32,
        )
    };

    let extra_bits = (a_bits.len().max(b_bits.len()) - shared_len) * 8;
    let total_penalty = penalty + extra_bits as u32;

    if overlap == 0 {
        return Distance::Incomparable;
    }

    Distance::Measured {
        disagreement,
        overlap,
        penalty: total_penalty,
    }
}

#[inline]
fn integer_sqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_new_is_empty() {
        let p = Plane::new();
        assert_eq!(p.encounters(), 0);
        assert!(p.bits.is_zero());
        assert!(p.alpha.is_zero());
    }

    #[test]
    fn plane_encounter_builds_signal() {
        let mut p = Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        p.encounter("hello");
        let t = p.truth();
        assert!(t.confidence > 0);
        assert_eq!(t.evidence, 3);
    }

    #[test]
    fn plane_nan_impossible() {
        let mut empty = Plane::new();
        let t = empty.truth();
        assert_eq!(t.frequency, 32768);
        assert_eq!(t.confidence, 0);
        let mut other = Plane::new();
        let d = empty.distance(&mut other);
        assert!(matches!(d, Distance::Incomparable));
    }

    #[test]
    fn plane_encounter_bits_direct() {
        let mut p = Plane::new();
        let all_ones = Fingerprint::<256>::ones();
        p.encounter_bits(&all_ones);
        p.encounter_bits(&all_ones);
        let t = p.truth();
        assert_eq!(t.evidence, 2);
        assert!(t.frequency > 32768);
    }

    #[test]
    fn distance_measured_between_similar_planes() {
        let mut a = Plane::new();
        let mut b = Plane::new();
        a.encounter("hello world");
        a.encounter("hello world");
        a.encounter("hello world");
        b.encounter("hello world");
        b.encounter("hello world");
        b.encounter("hello world");
        let d = a.distance(&mut b);
        match d {
            Distance::Measured {
                disagreement,
                overlap,
                ..
            } => {
                assert!(overlap > 0);
                assert_eq!(disagreement, 0);
            }
            Distance::Incomparable => panic!("expected Measured"),
        }
    }

    #[test]
    fn distance_closer_than() {
        let d = Distance::Measured {
            disagreement: 100,
            overlap: 8000,
            penalty: 200,
        };
        assert!(d.closer_than(100));
        assert!(d.closer_than(200));
        assert!(!d.closer_than(50));
        assert!(!Distance::Incomparable.closer_than(100));
    }

    #[test]
    fn distance_normalized() {
        let d = Distance::Measured {
            disagreement: 100,
            overlap: 900,
            penalty: 100,
        };
        let n = d.normalized().unwrap();
        assert!((n - 0.2).abs() < 0.001);
        assert!(Distance::Incomparable.normalized().is_none());
    }

    #[test]
    fn truth_revision_integer_only() {
        let t1 = Truth {
            frequency: 60000,
            confidence: 50000,
            evidence: 10,
        };
        let t2 = Truth {
            frequency: 30000,
            confidence: 40000,
            evidence: 5,
        };
        let revised = t1.revision(&t2);
        assert_eq!(revised.frequency, 50000);
        assert_eq!(revised.evidence, 15);
    }

    #[test]
    fn truth_expectation_no_confidence() {
        let t = Truth {
            frequency: 60000,
            confidence: 0,
            evidence: 0,
        };
        assert_eq!(t.expectation(), 32768);
    }

    #[test]
    fn truth_expectation_full_confidence() {
        let t = Truth {
            frequency: 65535,
            confidence: 65535,
            evidence: 100,
        };
        assert!(t.expectation() >= 65534);
    }

    #[test]
    fn integer_sqrt_correct() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(9), 3);
        assert_eq!(integer_sqrt(100), 10);
        assert_eq!(integer_sqrt(101), 10);
    }
}
