//! Parity for plan `simd-fruits-integration-v1` F5: `is_nan`, `is_finite`,
//! `is_infinite`, `copysign` and `recip` on `F32x16`,
//! `F64x8`, `F32x8` and `F64x4`.
//!
//! Expectations are computed lane by lane with the `f32`/`f64` methods of
//! the same name, so this file grades whichever arm the build compiled.
//! Every result must be IEEE bit-exact and identical on every arm (the
//! polyfill contract): the masks bit for bit, `copysign` and `recip` by
//! `to_bits`.
#![cfg(feature = "std")]

use ndarray::simd::{F32x16, F32x8, F64x4, F64x8};

const POOL: [f64; 26] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -2.5,
    3.0,
    1e-3,
    -7.75,
    1e9,
    -1e-9,
    123.456,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -f64::NAN,
    2.0,
    -4.0,
    0.1,
    65504.0,
    -1e30,
    1e-30,
    1e300,
    -1e-300,
    f64::MIN_POSITIVE,
    7.0,
];

fn lanes_f64<const N: usize>(seed: usize) -> [f64; N] {
    core::array::from_fn(|i| POOL[(seed * 7 + i * 3) % POOL.len()])
}
fn lanes_f32<const N: usize>(seed: usize) -> [f32; N] {
    lanes_f64::<N>(seed).map(|x| x as f32)
}

fn mask_of(pred: impl Fn(usize) -> bool, lanes: usize) -> u32 {
    (0..lanes).filter(|&i| pred(i)).fold(0, |m, i| m | (1 << i))
}

#[test]
fn simd_float_classify_report_arm() {
    println!(
        "simd_float_classify: avx512f={} avx512vl={} avx2={} avx={} neon={} simd128={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512vl"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
}

macro_rules! classify_parity {
    ($vec:ident, $elem:ty, $lanes:expr, $gen:ident) => {
        for seed in 0..60 {
            let a: [$elem; $lanes] = $gen(seed);
            let s: [$elem; $lanes] = $gen(seed + 4);
            let (va, vs) = ($vec::from_array(a), $vec::from_array(s));
            let t = stringify!($vec);
            assert_eq!(u32::from(va.is_nan().to_bitmask()), mask_of(|i| a[i].is_nan(), $lanes), "{t}::is_nan {a:?}");
            assert_eq!(
                u32::from(va.is_finite().to_bitmask()),
                mask_of(|i| a[i].is_finite(), $lanes),
                "{t}::is_finite {a:?}"
            );
            assert_eq!(
                u32::from(va.is_infinite().to_bitmask()),
                mask_of(|i| a[i].is_infinite(), $lanes),
                "{t}::is_infinite {a:?}"
            );
            let cs = va.copysign(vs).to_array();
            let rc = va.recip().to_array();
            for i in 0..$lanes {
                assert_eq!(
                    cs[i].to_bits(),
                    a[i].copysign(s[i]).to_bits(),
                    "{t}::copysign lane {i}: {:?} sign {:?}",
                    a[i],
                    s[i]
                );
                let want = 1.0 / a[i];
                assert!(
                    (rc[i].is_nan() && want.is_nan()) || rc[i].to_bits() == want.to_bits(),
                    "{t}::recip lane {i}: 1/{:?} got {:?} want {:?}",
                    a[i],
                    rc[i],
                    want
                );
            }
        }
    };
}

#[test]
fn classification_copysign_recip_match_scalar() {
    classify_parity!(F32x16, f32, 16, lanes_f32);
    classify_parity!(F64x8, f64, 8, lanes_f64);
    classify_parity!(F32x8, f32, 8, lanes_f32);
    classify_parity!(F64x4, f64, 4, lanes_f64);
}
