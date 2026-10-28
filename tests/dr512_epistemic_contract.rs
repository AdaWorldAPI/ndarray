#![cfg(feature = "std")]
//! Independent DR512 P×M truth-table and distance-oracle CI tripwires.
use ndarray::hpc::codec8::{Calibration, TiesAway, TiesEven};
use ndarray::hpc::codec9::{Codec9, EpiLane9};
use ndarray::simd::lut9;

struct Unit;
impl Calibration for Unit {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 0;
}

#[test]
fn all_512_codes_agree_in_codec9_and_lut9() {
    let identity: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mags: Vec<u8> = (0..512).map(|i| i as u8).collect();
    let presence: Vec<u64> = (0..8).map(|w| if w < 4 { 0 } else { u64::MAX }).collect();
    let mut out = vec![0u8; 512];
    let mut out_p = vec![u64::MAX; 8];
    lut9(&mags, &presence, &identity, &mut out, &mut out_p);
    assert_eq!(out, mags);
    assert_eq!(out_p, presence);

    let mut absent = 0;
    for code in 0..512u16 {
        let lane = EpiLane9::from_codeword(code).expect("9-bit code");
        let i = usize::from(code);
        assert_eq!(lane.codeword(), code);
        assert_eq!(lane.magnitude(), out[i]);
        assert_eq!(lane.presence(), (out_p[i / 64] >> (i % 64)) & 1 == 1);
        let expected = match code {
            0 => None,
            1..=255 => Some(-(code as i16)),
            _ => Some((code - 256) as i16),
        };
        assert_eq!(lane.value(), expected);
        absent += usize::from(lane.never_touched());
    }
    assert_eq!(absent, 1);
}

#[test]
fn where_zero_and_zero_needs_no_materialized_8k_mask() {
    let n = 65_536usize;
    let magnitudes: Vec<u8> = (0..n).map(|i| if i % 4 < 2 { 0 } else { 9 }).collect();
    let presence = vec![0xAAAA_AAAA_AAAA_AAAAu64; n / 64];
    let absent = (0..n)
        .filter(|&i| (presence[i / 64] >> (i % 64)) & 1 == 0 && magnitudes[i] == 0)
        .count();
    assert_eq!(absent, 16_384);
    // A negative nonzero lane must survive the same WHERE.
    assert_eq!(magnitudes[2], 9);
    assert_eq!((presence[0] >> 2) & 1, 0);
}

#[test]
fn touched_ceiling_to_valid_plus_zero_not_never_touched() {
    type Even = Codec9<TiesEven, Unit>;
    type Away = Codec9<TiesAway, Unit>;
    for x in [-0.0, -0.00001, -0.4, -0.49999, -0.5, 0.0, 0.4] {
        assert_eq!(Even::encode_touched(x).unwrap().codeword(), 0x100, "{x}");
    }
    assert_eq!(Away::encode_touched(-0.4).unwrap().codeword(), 0x100);
    assert_eq!(Away::encode_touched(-0.5).unwrap().codeword(), 1);
    assert_eq!(Even::encode_touched(-0.50001).unwrap().codeword(), 1);
    assert_eq!(EpiLane9::NEVER_TOUCHED.codeword(), 0);
}

/// The distance for equal numeric scale is abs(m1-m2) for equal polarity,
/// m1+m2 otherwise. Its maximum is 510: never narrow the result to u8.
#[test]
fn all_numeric_dr512_pairs_match_independent_i16_distance() {
    let mut checked = 0;
    let mut max_distance = 0;
    for ca in 1..512u16 {
        for cb in 1..512u16 {
            let a = EpiLane9::from_codeword(ca).unwrap();
            let b = EpiLane9::from_codeword(cb).unwrap();
            let derived = if a.presence() == b.presence() {
                u16::from(a.magnitude().abs_diff(b.magnitude()))
            } else {
                u16::from(a.magnitude()) + u16::from(b.magnitude())
            };
            let independent = (i32::from(a.value().unwrap()) - i32::from(b.value().unwrap())).unsigned_abs() as u16;
            assert_eq!(derived, independent, "codes {ca} {cb}");
            max_distance = max_distance.max(derived);
            checked += 1;
        }
    }
    assert_eq!(checked, 511 * 511);
    assert_eq!(max_distance, 510);
}

/// A hypothetical [-a,+b] calibrated range changes per-side scales,
/// but never changes DR512's P×M code interpretation. Mathematical oracle
/// only: NOT an admitted new Codec9 mode or a performance measurement.
#[test]
fn asymmetric_calibrated_distance_is_not_a_uniform_u8_lookup() {
    let (a_scale, b_scale) = (3.0f64 / 255.0, 7.0f64 / 255.0);
    for ca in 1..512u16 {
        let x = EpiLane9::from_codeword(ca).unwrap();
        let signed = |v: EpiLane9| -> f64 { f64::from(v.magnitude()) * if v.presence() { b_scale } else { -a_scale } };
        for cb in 1..512u16 {
            let y = EpiLane9::from_codeword(cb).unwrap();
            let folded = if x.presence() == y.presence() {
                f64::from(x.magnitude().abs_diff(y.magnitude())) * if x.presence() { b_scale } else { a_scale }
            } else {
                let p = if x.presence() { x } else { y };
                let n = if x.presence() { y } else { x };
                f64::from(p.magnitude()) * b_scale + f64::from(n.magnitude()) * a_scale
            };
            assert!((folded - (signed(x) - signed(y)).abs()).abs() < 1e-12);
        }
    }
}
