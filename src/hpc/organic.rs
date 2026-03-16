#![allow(clippy::assign_op_pattern, clippy::too_many_arguments, clippy::manual_range_contains, clippy::needless_range_loop)]

//! Organic Plasticity Model — BCM-inspired synapse dynamics.
//!
//! Replaces bare i8 saturating-add with biologically-grounded plasticity
//! rules from BCM (Bienenstock-Cooper-Munro) theory. Each dimension of the
//! soaking register becomes a `SynapseState` triple:
//!
//! - **efficacy** (i8): current synaptic strength, fast-changing (~seconds)
//! - **theta** (u8): BCM sliding modification threshold, medium-changing (~minutes)
//! - **maturity** (u8, clamped 0–15): structural stability counter, slow-changing (~hours)
//!
//! The BCM sliding threshold self-normalizes across concepts: high recent
//! activity raises theta (harder to potentiate), low activity lowers it
//! (easier to potentiate). This prevents first-mover dominance.
//!
//! # 5-State Quantization
//!
//! The biological synapse clusters around 5 effective states:
//! `{-2, -1, 0, +1, +2}` where 0 is Auslöschung (cancellation), not "no data".
//!
//! Packing: 3 values per byte (5³ = 125 < 128 = 2⁷).
//! 10000 dimensions at 5-state = ceil(10000/3) = 3334 bytes ≈ 3.3 KB per plane.
//!
//! # Zero IO
//!
//! All functions are pure compute. No allocation beyond return values.
//! No IO. Suitable for hot-path use in the bindspace surface.

use super::fingerprint::Fingerprint;

// ---------------------------------------------------------------------------
// 5-State quantization
// ---------------------------------------------------------------------------

/// The 5 effective synapse states, mapping to biological plasticity levels.
///
/// - `StrongNeg` (-2): Structurally depressed (consolidated LTD)
/// - `WeakNeg` (-1): Functionally depressed (recent LTD, theta above efficacy)
/// - `Silent` (0): Cancelled/silent (Auslöschung — destructive interference)
/// - `WeakPos` (+1): Functionally potentiated (recent LTP, efficacy above theta)
/// - `StrongPos` (+2): Structurally potentiated (consolidated LTP, high maturity)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FiveState {
    StrongNeg = 0,
    WeakNeg = 1,
    Silent = 2,
    WeakPos = 3,
    StrongPos = 4,
}

impl FiveState {
    /// Convert to signed integer value.
    #[inline]
    pub fn to_i8(self) -> i8 {
        match self {
            FiveState::StrongNeg => -2,
            FiveState::WeakNeg => -1,
            FiveState::Silent => 0,
            FiveState::WeakPos => 1,
            FiveState::StrongPos => 2,
        }
    }

    /// Convert from raw u8 (clamped to valid range).
    #[inline]
    pub fn from_raw(v: u8) -> Self {
        match v {
            0 => FiveState::StrongNeg,
            1 => FiveState::WeakNeg,
            2 => FiveState::Silent,
            3 => FiveState::WeakPos,
            4 => FiveState::StrongPos,
            _ => FiveState::Silent,
        }
    }
}

/// Pack 3 `FiveState` values into a single byte.
///
/// Encoding: `byte = a * 25 + b * 5 + c` where each value is in 0..5.
/// Maximum: 4*25 + 4*5 + 4 = 124 < 128.
#[inline]
pub fn pack_three(a: FiveState, b: FiveState, c: FiveState) -> u8 {
    (a as u8) * 25 + (b as u8) * 5 + (c as u8)
}

/// Unpack a single byte into 3 `FiveState` values.
#[inline]
pub fn unpack_three(byte: u8) -> (FiveState, FiveState, FiveState) {
    let a = byte / 25;
    let rem = byte % 25;
    let b = rem / 5;
    let c = rem % 5;
    (
        FiveState::from_raw(a),
        FiveState::from_raw(b),
        FiveState::from_raw(c),
    )
}

/// Pack a slice of `FiveState` values into bytes (3 values per byte).
///
/// Output length = ceil(states.len() / 3).
pub fn pack_five_states(states: &[FiveState]) -> Vec<u8> {
    let out_len = states.len().div_ceil(3);
    let mut out = Vec::with_capacity(out_len);
    let mut i = 0;
    while i + 2 < states.len() {
        out.push(pack_three(states[i], states[i + 1], states[i + 2]));
        i += 3;
    }
    // Handle remainder (1 or 2 trailing values)
    let remaining = states.len() - i;
    if remaining == 2 {
        out.push(pack_three(states[i], states[i + 1], FiveState::Silent));
    } else if remaining == 1 {
        out.push(pack_three(states[i], FiveState::Silent, FiveState::Silent));
    }
    out
}

/// Unpack bytes into `FiveState` values.
///
/// `count` is the total number of states to unpack (handles remainder).
pub fn unpack_five_states(packed: &[u8], count: usize) -> Vec<FiveState> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        if out.len() >= count {
            break;
        }
        let (a, b, c) = unpack_three(byte);
        out.push(a);
        if out.len() < count {
            out.push(b);
        }
        if out.len() < count {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SynapseState
// ---------------------------------------------------------------------------

/// Per-dimension synapse state: the minimum viable biology.
///
/// This replaces bare `i8` with a biological triple that implements
/// BCM (Bienenstock-Cooper-Munro) plasticity dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SynapseState {
    /// Efficacy: current synaptic strength (fast-changing, ~seconds).
    /// This is what was previously a bare i8 in the soaking register.
    pub efficacy: i8,

    /// Theta: BCM sliding modification threshold (medium-changing, ~minutes).
    /// When |efficacy| > theta → LTP (strengthen).
    /// When |efficacy| < theta → LTD (weaken).
    /// The threshold slides toward recent average |efficacy|.
    pub theta: u8,

    /// Maturity: structural stability counter (slow-changing, ~hours/days).
    /// Counts how many LTP/LTD cycles this dimension has survived.
    /// High maturity = resistant to change. Clamped to 0–15.
    pub maturity: u8,
}

impl SynapseState {
    /// Fresh synapse: no evidence, low threshold, zero maturity.
    #[inline]
    pub const fn new() -> Self {
        Self {
            efficacy: 0,
            theta: 5, // low initial threshold: easy to potentiate
            maturity: 0,
        }
    }

    /// Quantize this synapse to a 5-state value.
    ///
    /// Uses efficacy magnitude relative to theta to classify:
    /// - |eff| > theta AND maturity >= 8 → Strong (±2)
    /// - |eff| > theta/2 → Weak (±1)
    /// - otherwise → Silent (0)
    #[inline]
    pub fn quantize(&self) -> FiveState {
        let abs_eff = self.efficacy.unsigned_abs();
        let half_theta = self.theta / 2;

        if abs_eff <= half_theta {
            FiveState::Silent
        } else if self.efficacy > 0 {
            if abs_eff > self.theta && self.maturity >= 8 {
                FiveState::StrongPos
            } else {
                FiveState::WeakPos
            }
        } else if abs_eff > self.theta && self.maturity >= 8 {
            FiveState::StrongNeg
        } else {
            FiveState::WeakNeg
        }
    }

    /// Returns true if this synapse is at or near saturation.
    #[inline]
    pub fn is_saturated(&self) -> bool {
        self.efficacy.unsigned_abs() > 100 && self.maturity >= 12
    }
}

impl Default for SynapseState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BCM Plasticity: organic_deposit
// ---------------------------------------------------------------------------

/// BCM-style plasticity update for one dimension.
///
/// This replaces raw `saturating_add`. The update follows BCM theory:
/// - When |efficacy| > theta → LTP (potentiate in direction of evidence)
/// - When |efficacy| < theta but > theta/2 → LTD (depress toward zero)
/// - Near zero with low theta → full potentiation (new learning)
///
/// Maturity dampens the change rate: high maturity = small changes.
/// Theta slides toward |efficacy| with a slow time constant.
/// Maturity increments on each non-zero plasticity event (saturates at 15).
#[inline]
pub fn organic_deposit(state: &mut SynapseState, evidence: i8) {
    if evidence == 0 {
        return;
    }

    let eff = state.efficacy as i16;
    let theta = state.theta as i16;

    // BCM: compute phi(efficacy, theta)
    let phi = if eff.abs() > theta {
        // Above threshold: potentiate (strengthen in direction of evidence)
        evidence as i16
    } else if eff.abs() > theta / 2 {
        // Below threshold but above half: depress (weaken toward zero)
        -(eff.signum()) * (evidence.unsigned_abs() as i16 / 2).max(1)
    } else {
        // Near zero with low theta: full potentiation (new learning)
        evidence as i16
    };

    // Apply with maturity damping: high maturity = small changes
    let maturity_scale = 16i16 - state.maturity as i16; // 16 → 1 as maturity grows
    let delta = (phi * maturity_scale) / 16;

    state.efficacy = (eff + delta).clamp(-128, 127) as i8;

    // Slide theta: moves toward |efficacy| with slow time constant
    let target_theta = (eff.unsigned_abs().min(255)) as i16;
    let theta_delta = (target_theta - theta).clamp(-2, 2);
    state.theta = (theta + theta_delta).clamp(0, 255) as u8;

    // Maturity: increment on each plasticity event (saturating at 15)
    if delta != 0 && state.maturity < 15 {
        state.maturity += 1;
    }
}

/// Batch organic deposit: update a register of SynapseStates with evidence.
///
/// `states` and `evidence` must have the same length.
pub fn organic_deposit_batch(states: &mut [SynapseState], evidence: &[i8]) {
    assert_eq!(
        states.len(),
        evidence.len(),
        "states and evidence must have same length"
    );
    for (state, &ev) in states.iter_mut().zip(evidence.iter()) {
        organic_deposit(state, ev);
    }
}

// ---------------------------------------------------------------------------
// Homeostatic Scaling
// ---------------------------------------------------------------------------

/// Homeostatic scaling pass: adjusts theta across all active concepts
/// to prevent first-mover dominance.
///
/// Computes mean |efficacy| across all states. If mean is too high,
/// scales all theta upward (harder to potentiate). If too low,
/// scales theta downward (easier to potentiate).
///
/// `target_mean`: desired average |efficacy| (typically 30–60).
/// `scale_rate`: how aggressively to adjust (typically 1–3).
pub fn homeostatic_scale(states: &mut [SynapseState], target_mean: u8, scale_rate: u8) {
    if states.is_empty() {
        return;
    }

    // Compute mean absolute efficacy
    let sum: u64 = states
        .iter()
        .map(|s| s.efficacy.unsigned_abs() as u64)
        .sum();
    let mean = (sum / states.len() as u64) as u8;

    if mean == target_mean {
        return;
    }

    let rate = scale_rate.max(1) as i16;

    if mean > target_mean {
        // Activity too high: raise all theta (harder to potentiate)
        for state in states.iter_mut() {
            let new_theta = (state.theta as i16 + rate).min(255);
            state.theta = new_theta as u8;
        }
    } else {
        // Activity too low: lower all theta (easier to potentiate)
        for state in states.iter_mut() {
            let new_theta = (state.theta as i16 - rate).max(1);
            state.theta = new_theta as u8;
        }
    }
}

/// Compute the mean absolute efficacy of a register (diagnostic).
pub fn mean_efficacy(states: &[SynapseState]) -> f32 {
    if states.is_empty() {
        return 0.0;
    }
    let sum: u64 = states
        .iter()
        .map(|s| s.efficacy.unsigned_abs() as u64)
        .sum();
    sum as f32 / states.len() as f32
}

/// Count how many dimensions are at each 5-state level (diagnostic).
pub fn five_state_histogram(states: &[SynapseState]) -> [usize; 5] {
    let mut hist = [0usize; 5];
    for state in states {
        hist[state.quantize() as usize] += 1;
    }
    hist
}

/// Compute saturation ratio: fraction of dimensions at |efficacy| > threshold.
pub fn saturation_ratio(states: &[SynapseState], threshold: u8) -> f32 {
    if states.is_empty() {
        return 0.0;
    }
    let count = states
        .iter()
        .filter(|s| s.efficacy.unsigned_abs() > threshold)
        .count();
    count as f32 / states.len() as f32
}

// ---------------------------------------------------------------------------
// Crystallization: 5-state → binary
// ---------------------------------------------------------------------------

/// Crystallize a register of SynapseStates into a binary Fingerprint<256>.
///
/// Mapping:
/// - StrongPos, WeakPos → bit = 1
/// - Silent → bit = 0 (honest uncertainty — could also use random tiebreak)
/// - WeakNeg, StrongNeg → bit = 0
///
/// Only the first `min(states.len(), 16384)` dimensions map to bits.
/// Remaining bits (if states.len() < 16384) are zero.
pub fn crystallize<const N: usize>(states: &[SynapseState]) -> Fingerprint<N> {
    let mut fp = Fingerprint::<N>::zero();
    let max_bits = (N * 64).min(states.len());

    for (i, state) in states.iter().enumerate().take(max_bits) {
        if state.efficacy > 0 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            fp.words[word_idx] |= 1u64 << bit_idx;
        }
    }
    fp
}

/// Crystallize using 5-state quantization with explicit zero handling.
///
/// Same as `crystallize` but uses the quantized 5-state value,
/// giving a cleaner threshold (only structurally/functionally positive
/// bits become 1).
pub fn crystallize_quantized<const N: usize>(states: &[SynapseState]) -> Fingerprint<N> {
    let mut fp = Fingerprint::<N>::zero();
    let max_bits = (N * 64).min(states.len());

    for (i, state) in states.iter().enumerate().take(max_bits) {
        let q = state.quantize();
        if q == FiveState::WeakPos || q == FiveState::StrongPos {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            fp.words[word_idx] |= 1u64 << bit_idx;
        }
    }
    fp
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_five_state_values() {
        assert_eq!(FiveState::StrongNeg.to_i8(), -2);
        assert_eq!(FiveState::WeakNeg.to_i8(), -1);
        assert_eq!(FiveState::Silent.to_i8(), 0);
        assert_eq!(FiveState::WeakPos.to_i8(), 1);
        assert_eq!(FiveState::StrongPos.to_i8(), 2);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        for a in 0..5u8 {
            for b in 0..5u8 {
                for c in 0..5u8 {
                    let fa = FiveState::from_raw(a);
                    let fb = FiveState::from_raw(b);
                    let fc = FiveState::from_raw(c);
                    let packed = pack_three(fa, fb, fc);
                    assert!(packed < 125, "packed value must be < 125, got {packed}");
                    let (ua, ub, uc) = unpack_three(packed);
                    assert_eq!(ua, fa);
                    assert_eq!(ub, fb);
                    assert_eq!(uc, fc);
                }
            }
        }
    }

    #[test]
    fn test_pack_unpack_vec_roundtrip() {
        let states = vec![
            FiveState::StrongNeg,
            FiveState::WeakPos,
            FiveState::Silent,
            FiveState::StrongPos,
            FiveState::WeakNeg,
            FiveState::Silent,
            FiveState::WeakPos,
        ];
        let packed = pack_five_states(&states);
        assert_eq!(packed.len(), 3); // ceil(7/3) = 3
        let unpacked = unpack_five_states(&packed, states.len());
        assert_eq!(unpacked, states);
    }

    #[test]
    fn test_pack_unpack_exact_multiple() {
        let states = vec![FiveState::WeakPos, FiveState::Silent, FiveState::StrongNeg];
        let packed = pack_five_states(&states);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_five_states(&packed, 3);
        assert_eq!(unpacked, states);
    }

    #[test]
    fn test_pack_unpack_single() {
        let states = vec![FiveState::StrongPos];
        let packed = pack_five_states(&states);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_five_states(&packed, 1);
        assert_eq!(unpacked, states);
    }

    #[test]
    fn test_synapse_state_new() {
        let s = SynapseState::new();
        assert_eq!(s.efficacy, 0);
        assert_eq!(s.theta, 5);
        assert_eq!(s.maturity, 0);
        assert_eq!(s.quantize(), FiveState::Silent);
    }

    #[test]
    fn test_organic_deposit_basic_potentiation() {
        let mut s = SynapseState::new();
        // Deposit positive evidence into fresh synapse
        organic_deposit(&mut s, 10);
        assert!(s.efficacy > 0, "should potentiate positively");
        assert!(s.maturity > 0, "maturity should increment");
    }

    #[test]
    fn test_organic_deposit_negative_evidence() {
        let mut s = SynapseState::new();
        organic_deposit(&mut s, -10);
        assert!(s.efficacy < 0, "should potentiate negatively");
    }

    #[test]
    fn test_organic_deposit_zero_evidence_noop() {
        let s_before = SynapseState::new();
        let mut s = s_before;
        organic_deposit(&mut s, 0);
        assert_eq!(s, s_before);
    }

    #[test]
    fn test_bcm_theta_slides() {
        let mut s = SynapseState {
            efficacy: 80,
            theta: 20,
            maturity: 0,
        };
        let theta_before = s.theta;
        // Deposit — theta should slide toward |efficacy|
        organic_deposit(&mut s, 10);
        assert!(
            s.theta >= theta_before,
            "theta should slide upward toward high efficacy"
        );
    }

    #[test]
    fn test_maturity_dampens_change() {
        // High maturity synapse: should change less
        let mut high_mat = SynapseState {
            efficacy: 0,
            theta: 5,
            maturity: 14,
        };
        let mut low_mat = SynapseState {
            efficacy: 0,
            theta: 5,
            maturity: 0,
        };
        organic_deposit(&mut high_mat, 50);
        organic_deposit(&mut low_mat, 50);
        assert!(
            low_mat.efficacy.unsigned_abs() >= high_mat.efficacy.unsigned_abs(),
            "low maturity should change more: low={}, high={}",
            low_mat.efficacy,
            high_mat.efficacy
        );
    }

    #[test]
    fn test_organic_deposit_batch() {
        let mut states = vec![SynapseState::new(); 5];
        let evidence = vec![10, -5, 0, 20, -30];
        organic_deposit_batch(&mut states, &evidence);
        assert!(states[0].efficacy > 0);
        assert!(states[1].efficacy < 0);
        assert_eq!(states[2].efficacy, 0); // zero evidence = no change
        assert!(states[3].efficacy > 0);
        assert!(states[4].efficacy < 0);
    }

    #[test]
    fn test_homeostatic_scale_raises_theta() {
        // All synapses have high efficacy → theta should rise
        let mut states: Vec<SynapseState> = (0..10)
            .map(|_| SynapseState {
                efficacy: 100,
                theta: 20,
                maturity: 5,
            })
            .collect();
        let theta_before = states[0].theta;
        homeostatic_scale(&mut states, 30, 2);
        // Mean |efficacy| = 100 > target 30 → theta should increase
        assert!(states[0].theta > theta_before);
    }

    #[test]
    fn test_homeostatic_scale_lowers_theta() {
        // All synapses have low efficacy → theta should drop
        let mut states: Vec<SynapseState> = (0..10)
            .map(|_| SynapseState {
                efficacy: 5,
                theta: 100,
                maturity: 5,
            })
            .collect();
        let theta_before = states[0].theta;
        homeostatic_scale(&mut states, 50, 2);
        assert!(states[0].theta < theta_before);
    }

    #[test]
    fn test_saturation_ratio() {
        let states = vec![
            SynapseState {
                efficacy: 120,
                theta: 50,
                maturity: 10,
            },
            SynapseState {
                efficacy: -110,
                theta: 50,
                maturity: 10,
            },
            SynapseState {
                efficacy: 10,
                theta: 50,
                maturity: 10,
            },
            SynapseState {
                efficacy: -5,
                theta: 50,
                maturity: 10,
            },
        ];
        let ratio = saturation_ratio(&states, 100);
        assert!((ratio - 0.5).abs() < f32::EPSILON); // 2 of 4 above threshold
    }

    #[test]
    fn test_five_state_histogram() {
        let states = vec![
            SynapseState {
                efficacy: 80,
                theta: 20,
                maturity: 10,
            }, // StrongPos
            SynapseState {
                efficacy: -80,
                theta: 20,
                maturity: 10,
            }, // StrongNeg
            SynapseState {
                efficacy: 0,
                theta: 20,
                maturity: 0,
            }, // Silent
            SynapseState {
                efficacy: 15,
                theta: 20,
                maturity: 2,
            }, // WeakPos
            SynapseState {
                efficacy: -15,
                theta: 20,
                maturity: 2,
            }, // WeakNeg
        ];
        let hist = five_state_histogram(&states);
        assert_eq!(hist[FiveState::StrongNeg as usize], 1);
        assert_eq!(hist[FiveState::WeakNeg as usize], 1);
        assert_eq!(hist[FiveState::Silent as usize], 1);
        assert_eq!(hist[FiveState::WeakPos as usize], 1);
        assert_eq!(hist[FiveState::StrongPos as usize], 1);
    }

    #[test]
    fn test_crystallize_basic() {
        let states = vec![
            SynapseState {
                efficacy: 50,
                theta: 20,
                maturity: 5,
            },
            SynapseState {
                efficacy: -50,
                theta: 20,
                maturity: 5,
            },
            SynapseState {
                efficacy: 0,
                theta: 20,
                maturity: 0,
            },
            SynapseState {
                efficacy: 100,
                theta: 20,
                maturity: 5,
            },
        ];
        let fp: Fingerprint<1> = crystallize(&states);
        // Bit 0 = 1 (eff > 0), bit 1 = 0 (eff < 0), bit 2 = 0 (eff == 0), bit 3 = 1 (eff > 0)
        assert_eq!(fp.words[0] & 0xF, 0b1001);
    }

    #[test]
    fn test_crystallize_quantized() {
        let states = vec![
            SynapseState {
                efficacy: 80,
                theta: 20,
                maturity: 10,
            }, // StrongPos → 1
            SynapseState {
                efficacy: -80,
                theta: 20,
                maturity: 10,
            }, // StrongNeg → 0
            SynapseState {
                efficacy: 5,
                theta: 20,
                maturity: 0,
            }, // Silent → 0
            SynapseState {
                efficacy: 15,
                theta: 20,
                maturity: 2,
            }, // WeakPos → 1
        ];
        let fp: Fingerprint<1> = crystallize_quantized(&states);
        assert_eq!(fp.words[0] & 0xF, 0b1001);
    }

    #[test]
    fn test_convergence_self_normalization() {
        // Simulate depositing "love" 100 times then "kubernetes" 10 times.
        // BCM theta should self-normalize: love's theta rises, making
        // kubernetes easier to potentiate relative to its history.
        let mut love = SynapseState::new();
        let mut kube = SynapseState::new();

        for _ in 0..100 {
            organic_deposit(&mut love, 10);
        }
        for _ in 0..10 {
            organic_deposit(&mut kube, 10);
        }

        // Love should have higher theta (harder to potentiate further)
        assert!(
            love.theta > kube.theta,
            "love theta={} should be > kube theta={}",
            love.theta,
            kube.theta
        );
        // Love should have higher maturity
        assert!(love.maturity > kube.maturity);
    }

    #[test]
    fn test_5state_packing_10000d() {
        // Verify packing math for realistic dimension count
        let dim = 10000;
        let states: Vec<FiveState> = (0..dim)
            .map(|i| FiveState::from_raw((i % 5) as u8))
            .collect();
        let packed = pack_five_states(&states);
        // ceil(10000/3) = 3334
        assert_eq!(packed.len(), 3334);
        let unpacked = unpack_five_states(&packed, dim);
        assert_eq!(unpacked.len(), dim);
        assert_eq!(unpacked, states);
    }

    #[test]
    fn test_bcm_depression_below_threshold() {
        // When |efficacy| is between theta/2 and theta, evidence should
        // cause depression (move toward zero), not potentiation.
        // Need |eff| > theta/2 for the depression branch.
        let mut s = SynapseState {
            efficacy: 50,
            theta: 80, // theta/2 = 40, |50| > 40 and |50| < 80
            maturity: 0,
        };
        let eff_before = s.efficacy;
        organic_deposit(&mut s, 10); // positive evidence, but below threshold
                                     // Should depress (move toward zero) since |50| < 80 and |50| > 40
        assert!(
            s.efficacy < eff_before,
            "should depress: before={}, after={}",
            eff_before,
            s.efficacy
        );
    }
}
