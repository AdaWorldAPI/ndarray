//! `add_pm` probe: the timing harness and the scalar oracle for
//! [`ndarray::hpc::abakus::pm`], where the arithmetic now lives (one source
//! of truth). These wrappers return the saturated parts so the oracle can
//! check every lane, including the saturated byte of a carry lane.
//!
//! Subtraction is the same addition of the polarity-negated operand
//! (`neg_pm`, `sub_pm`). XOR governs polarity; magnitudes stay unsigned.
use ndarray::hpc::abakus::pm as lib;
// `neg_pm` and `touched` are used by tests/add_pm.rs, not by the timing example.
#[allow(unused_imports)]
pub use ndarray::hpc::abakus::pm::{neg_pm, touched, Pm};
use ndarray::simd::U8x64;

/// [`lib::add_pm`] as `(lanes, carry)`; carry lanes are saturated.
#[inline(always)]
pub fn add_pm(a: Pm, b: Pm) -> (Pm, u64) {
    lib::add_pm(a, b).into_saturated_parts()
}

/// [`lib::sub_pm`] as `(lanes, carry)`; carry lanes are saturated.
#[inline(always)]
pub fn sub_pm(a: Pm, b: Pm) -> (Pm, u64) {
    lib::sub_pm(a, b).into_saturated_parts()
}

// ------------------------------------------------------------------ scalar reference
// Decode to i32 and back: the path `add_pm` avoids, and the oracle it is checked by.

/// One lane's value: `None` = never touched.
pub fn decode_lane(m: u8, p: bool) -> Option<i32> {
    match (p, m) {
        (false, 0) => None,
        (false, m) => Some(-i32::from(m)),
        (true, m) => Some(i32::from(m)),
    }
}

/// A touched value in ±255 to its lane; zero is `+0`.
pub fn encode_lane(v: i32) -> (u8, bool) {
    assert!((-255..=255).contains(&v), "out of range: {v}");
    (v.unsigned_abs() as u8, v >= 0)
}

/// The reference sum of one lane: value (or never touched) and overflow.
pub fn add_ref(a: Option<i32>, b: Option<i32>) -> (Option<i32>, bool) {
    match (a, b) {
        (None, x) | (x, None) => (x, false),
        (Some(x), Some(y)) => (Some(x + y), (x + y).abs() > 255),
    }
}

/// The reference difference of one lane: `a - b`, never touched only when
/// both are, and overflow outside ±255.
pub fn sub_ref(a: Option<i32>, b: Option<i32>) -> (Option<i32>, bool) {
    add_ref(a, b.map(|y| -y))
}

/// `add_pm` by decode → i32 → encode, 64 lanes. Carry lanes are saturated.
pub fn add_decoded(a: Pm, b: Pm) -> (Pm, u64) {
    let (am, bm) = (a.mag.to_array(), b.mag.to_array());
    let mut mag = [0u8; 64];
    let (mut pos, mut carry) = (0u64, 0u64);
    for i in 0..64 {
        let x = decode_lane(am[i], a.pos >> i & 1 == 1);
        let y = decode_lane(bm[i], b.pos >> i & 1 == 1);
        let (s, over) = add_ref(x, y);
        if let Some(v) = s {
            let (m, p) = encode_lane(v.clamp(-255, 255));
            mag[i] = m;
            pos |= u64::from(p) << i;
        }
        carry |= u64::from(over) << i;
    }
    (
        Pm {
            mag: U8x64::from_array(mag),
            pos,
        },
        carry,
    )
}
