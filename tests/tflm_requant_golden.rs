//! Byte parity of the fixed-point requantization chain against TFLite-micro.
//!
//! `tests/data/tflm_requant_golden.txt` was produced by
//! `tests/data/tflm_requant_oracle.cc`, compiled against the real TFLM
//! `fixedpoint.h`, `common.cc` and `quantization_util.cc` (default build,
//! double rounding). Each line is `<tag> <inputs...> <expected>`.
#![cfg(feature = "std")]

use ndarray::simd::{
    multiply_by_quantized_multiplier_i32, multiply_by_quantized_multiplier_i64, rounding_divide_by_pot_i32, srdhm_i16,
    srdhm_i32, I32x16,
};

const GOLDEN: &str = include_str!("data/tflm_requant_golden.txt");

fn rows(tag: &str) -> Vec<Vec<i64>> {
    let v: Vec<Vec<i64>> = GOLDEN
        .lines()
        .filter(|l| l.split(' ').next() == Some(tag))
        .map(|l| {
            l.split(' ')
                .skip(1)
                .map(|t| t.parse::<i64>().unwrap())
                .collect()
        })
        .collect();
    assert!(!v.is_empty(), "no golden rows for {tag}");
    v
}

#[test]
fn scalar_chain_matches_tflm() {
    for r in rows("S") {
        assert_eq!(srdhm_i32(r[0] as i32, r[1] as i32) as i64, r[2], "srdhm_i32 {r:?}");
    }
    for r in rows("H") {
        assert_eq!(srdhm_i16(r[0] as i16, r[1] as i16) as i64, r[2], "srdhm_i16 {r:?}");
    }
    for r in rows("R") {
        assert_eq!(rounding_divide_by_pot_i32(r[0] as i32, r[1] as i32) as i64, r[2], "rdpot {r:?}");
    }
    for r in rows("M") {
        assert_eq!(
            multiply_by_quantized_multiplier_i32(r[0] as i32, r[1] as i32, r[2] as i32) as i64,
            r[3],
            "mbqm {r:?}"
        );
    }
    for r in rows("L") {
        assert_eq!(multiply_by_quantized_multiplier_i64(r[0], r[1] as i32, r[2] as i32) as i64, r[3], "mbqm64 {r:?}");
    }
}

#[test]
fn quantize_multiplier_matches_tflm() {
    for l in GOLDEN.lines().filter(|l| l.starts_with("Q ")) {
        let t: Vec<&str> = l.split(' ').collect();
        let d: f64 = t[1].parse().unwrap();
        let want = (t[2].parse::<i32>().unwrap(), t[3].parse::<i32>().unwrap());
        assert_eq!(ndarray::simd::quantize_multiplier(d), want, "quantize_multiplier({d})");
    }
}

/// The vector methods on whatever backend this build selected, 16 golden rows per call.
#[test]
fn vector_methods_match_tflm() {
    fn chunks(v: &[Vec<i64>], f: impl Fn(&[Vec<i64>]) -> ([i32; 16], [i32; 16])) {
        for c in v.chunks(16) {
            let (got, want) = f(c);
            assert_eq!(&got[..c.len()], &want[..c.len()], "rows {c:?}");
        }
    }
    let lane = |c: &[Vec<i64>], k: usize| -> [i32; 16] {
        let mut a = [0i32; 16];
        for (i, r) in c.iter().enumerate() {
            a[i] = r[k] as i32;
        }
        a
    };
    let s = rows("S");
    chunks(&s, |c| {
        let g = I32x16::from_array(lane(c, 0)).saturating_rounding_doubling_high_mul(I32x16::from_array(lane(c, 1)));
        (g.to_array(), lane(c, 2))
    });
    let r = rows("R");
    chunks(&r, |c| {
        let g = I32x16::from_array(lane(c, 0)).rounding_divide_by_pot(I32x16::from_array(lane(c, 1)));
        (g.to_array(), lane(c, 2))
    });
    let m = rows("M");
    chunks(&m, |c| {
        let g = I32x16::from_array(lane(c, 0))
            .multiply_by_quantized_multiplier(I32x16::from_array(lane(c, 1)), I32x16::from_array(lane(c, 2)));
        (g.to_array(), lane(c, 3))
    });
}
