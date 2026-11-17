//! Parity for plan `simd-fruits-integration-v1` F1 + F2 + F3.
//!
//! F1: the 256-bit `F32x8` / `F64x4` carry the full float method set
//! (compares, min/max/clamp, sqrt, round, floor, reductions, bit casts) on
//! every stable arm. F2: the AVX2 `F32Mask16` / `F64Mask8` are pairs of
//! 256-bit lane masks, and `F32x16` / `F64x8` compare, select, min/max,
//! sqrt, round and floor through those halves instead of per-lane loops.
//! F3: every mask has `from_bitmask`, `any`, `all`, `none` and `& | ^ !`.
//!
//! Every expectation is computed here, lane by lane, with plain `f32`/`f64`
//! operations, so the same file grades whichever arm the build compiled
//! (AVX-512, AVX2, AVX, x86-64-v2 scalar, NEON, wasm). Read the arm from the
//! printed header of `simd_lane_masks_report_arm`.
//!
//! NaN in min/max is deliberately not graded: x86 `vminps` returns the second
//! operand while the scalar arm's `f32::min` returns the non-NaN one; that
//! difference predates this plan and is documented on the methods.
#![cfg(feature = "std")]

use ndarray::simd::{F32Mask16, F32Mask8, F32x16, F32x8, F64Mask4, F64Mask8, F64x4, F64x8};

/// Deterministic test values: ordinary numbers, ±0, ties for rounding,
/// infinities and NaN, so ordered/unordered compares are both exercised.
const F32_POOL: [f32; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    1.5,
    2.5,
    -2.5,
    3.25,
    -7.75,
    1e-3,
    -1e9,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    2.0,
    4.0,
    -0.5,
    100.0,
    0.1,
    -3.5,
    8.0,
    1e30,
    -1e-30,
];

fn f32_lanes<const N: usize>(seed: usize) -> [f32; N] {
    core::array::from_fn(|i| F32_POOL[(seed * 7 + i * 5) % F32_POOL.len()])
}

fn f64_lanes<const N: usize>(seed: usize) -> [f64; N] {
    core::array::from_fn(|i| f64::from(F32_POOL[(seed * 11 + i * 3) % F32_POOL.len()]))
}

fn bits_of(pred: impl Fn(usize) -> bool, lanes: usize) -> u32 {
    (0..lanes).filter(|&i| pred(i)).fold(0, |m, i| m | (1 << i))
}

/// Bit-exact equality that treats every NaN as equal to every NaN.
fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}
fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

macro_rules! compare_parity {
    ($vec:ident, $elem:ty, $lanes:expr, $gen:ident, $seeds:expr) => {
        for seed in 0..$seeds {
            let a: [$elem; $lanes] = $gen(seed);
            let b: [$elem; $lanes] = $gen(seed + 3);
            let (va, vb) = ($vec::from_array(a), $vec::from_array(b));
            let cases: [(&str, u32, u32); 6] = [
                ("eq", va.simd_eq(vb).to_bitmask().into(), bits_of(|i| a[i] == b[i], $lanes)),
                ("ne", va.simd_ne(vb).to_bitmask().into(), bits_of(|i| a[i] != b[i], $lanes)),
                ("lt", va.simd_lt(vb).to_bitmask().into(), bits_of(|i| a[i] < b[i], $lanes)),
                ("le", va.simd_le(vb).to_bitmask().into(), bits_of(|i| a[i] <= b[i], $lanes)),
                ("gt", va.simd_gt(vb).to_bitmask().into(), bits_of(|i| a[i] > b[i], $lanes)),
                ("ge", va.simd_ge(vb).to_bitmask().into(), bits_of(|i| a[i] >= b[i], $lanes)),
            ];
            for (name, got, want) in cases {
                assert_eq!(got, want, "{}::simd_{} seed {}: a={:?} b={:?}", stringify!($vec), name, seed, a, b);
            }
        }
    };
}

#[test]
fn simd_lane_masks_report_arm() {
    // The arm this file graded. A green run says nothing about the arms
    // that were not compiled; run it again under each pinned config.
    println!(
        "simd_lane_masks: avx512f={} avx2={} avx={} neon={} simd128={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
}

#[test]
fn compares_match_scalar_on_every_width() {
    compare_parity!(F32x8, f32, 8, f32_lanes, 40);
    compare_parity!(F64x4, f64, 4, f64_lanes, 40);
    compare_parity!(F32x16, f32, 16, f32_lanes, 40);
    compare_parity!(F64x8, f64, 8, f64_lanes, 40);
}

macro_rules! unary_parity {
    ($vec:ident, $elem:ty, $lanes:expr, $gen:ident, $same:ident) => {
        for seed in 0..40 {
            let a: [$elem; $lanes] = $gen(seed);
            let v = $vec::from_array(a);
            let checks: [(&str, [$elem; $lanes], [$elem; $lanes]); 3] = [
                ("sqrt", v.sqrt().to_array(), a.map(<$elem>::sqrt)),
                ("round", v.round().to_array(), a.map(<$elem>::round_ties_even)),
                ("floor", v.floor().to_array(), a.map(<$elem>::floor)),
            ];
            for (name, got, want) in checks {
                for i in 0..$lanes {
                    assert!(
                        $same(got[i], want[i]),
                        "{}::{} lane {} of {:?}: got {:?} want {:?}",
                        stringify!($vec),
                        name,
                        i,
                        a,
                        got[i],
                        want[i]
                    );
                }
            }
        }
    };
}

#[test]
fn sqrt_round_floor_match_scalar() {
    unary_parity!(F32x8, f32, 8, f32_lanes, same_f32);
    unary_parity!(F64x4, f64, 4, f64_lanes, same_f64);
    unary_parity!(F32x16, f32, 16, f32_lanes, same_f32);
    unary_parity!(F64x8, f64, 8, f64_lanes, same_f64);
}

/// Pool without NaN: min/max/clamp/reductions are graded on ordered inputs.
fn no_nan_f32<const N: usize>(seed: usize) -> [f32; N] {
    f32_lanes::<N>(seed).map(|x| if x.is_nan() { 6.0 } else { x })
}
fn no_nan_f64<const N: usize>(seed: usize) -> [f64; N] {
    f64_lanes::<N>(seed).map(|x| if x.is_nan() { 6.0 } else { x })
}

macro_rules! minmax_parity {
    ($vec:ident, $elem:ty, $lanes:expr, $gen:ident) => {
        for seed in 0..40 {
            let a: [$elem; $lanes] = $gen(seed);
            let b: [$elem; $lanes] = $gen(seed + 5);
            let (va, vb) = ($vec::from_array(a), $vec::from_array(b));
            let lo = $vec::splat(-2.0);
            let hi = $vec::splat(3.0);
            let min: [$elem; $lanes] = core::array::from_fn(|i| a[i].min(b[i]));
            let max: [$elem; $lanes] = core::array::from_fn(|i| a[i].max(b[i]));
            let clamp: [$elem; $lanes] = a.map(|x| x.max(-2.0).min(3.0));
            // ±0 ties are implementation-defined in min/max: compare by value.
            assert_eq!(va.simd_min(vb).to_array(), min, "{} simd_min {:?} {:?}", stringify!($vec), a, b);
            assert_eq!(va.simd_max(vb).to_array(), max, "{} simd_max {:?} {:?}", stringify!($vec), a, b);
            assert_eq!(va.simd_clamp(lo, hi).to_array(), clamp, "{} simd_clamp {:?}", stringify!($vec), a);
            let rmin = a.iter().copied().fold(<$elem>::INFINITY, <$elem>::min);
            let rmax = a.iter().copied().fold(<$elem>::NEG_INFINITY, <$elem>::max);
            assert_eq!(va.reduce_min(), rmin, "{} reduce_min {:?}", stringify!($vec), a);
            assert_eq!(va.reduce_max(), rmax, "{} reduce_max {:?}", stringify!($vec), a);
        }
    };
}

#[test]
fn min_max_clamp_reduce_match_scalar() {
    minmax_parity!(F32x8, f32, 8, no_nan_f32);
    minmax_parity!(F64x4, f64, 4, no_nan_f64);
    minmax_parity!(F32x16, f32, 16, no_nan_f32);
    minmax_parity!(F64x8, f64, 8, no_nan_f64);
}

#[test]
fn bit_casts_round_trip_256() {
    let a: [f32; 8] = f32_lanes(1);
    let bits = F32x8::from_array(a).to_bits().to_array();
    assert_eq!(bits, a.map(f32::to_bits));
    let back = F32x8::from_bits(F32x8::from_array(a).to_bits()).to_array();
    assert!(back.iter().zip(&a).all(|(x, y)| same_f32(*x, *y)));

    let d: [f64; 4] = f64_lanes(2);
    assert_eq!(F64x4::from_array(d).to_bits().to_array(), d.map(f64::to_bits));
    let back = F64x4::from_bits(F64x4::from_array(d).to_bits()).to_array();
    assert!(back.iter().zip(&d).all(|(x, y)| same_f64(*x, *y)));
}

macro_rules! mask_api_parity {
    ($mask:ident, $vec:ident, $prim:ty, $lanes:expr) => {{
        let live: u32 = if $lanes == 32 { u32::MAX } else { (1u32 << $lanes) - 1 };
        // Every bit pattern for the narrow masks, a spread for the 16-lane one.
        let patterns: Vec<u32> = if $lanes <= 8 {
            (0..(1u32 << $lanes)).collect()
        } else {
            (0..=u32::from(u16::MAX))
                .step_by(97)
                .chain([0, 1, 0x8000, 0xFFFF])
                .collect()
        };
        for &p in &patterns {
            let m = $mask::from_bitmask(p as $prim);
            assert_eq!(u32::from(m.to_bitmask()), p & live, "{} from/to_bitmask {:#x}", stringify!($mask), p);
            assert_eq!(m.any(), p & live != 0, "{} any {:#x}", stringify!($mask), p);
            assert_eq!(m.all(), p & live == live, "{} all {:#x}", stringify!($mask), p);
            assert_eq!(m.none(), p & live == 0, "{} none {:#x}", stringify!($mask), p);
            assert_eq!(u32::from((!m).to_bitmask()), !p & live, "{} not {:#x}", stringify!($mask), p);
            let q = (p.wrapping_mul(0x9E37) ^ 0x5A5A) & live;
            let n = $mask::from_bitmask(q as $prim);
            assert_eq!(u32::from((m & n).to_bitmask()), p & q & live, "{} and", stringify!($mask));
            assert_eq!(u32::from((m | n).to_bitmask()), (p | q) & live, "{} or", stringify!($mask));
            assert_eq!(u32::from((m ^ n).to_bitmask()), (p ^ q) & live, "{} xor", stringify!($mask));
            let t: [_; $lanes] = core::array::from_fn(|i| (i + 1) as _);
            let f: [_; $lanes] = core::array::from_fn(|i| -((i + 1) as i32) as _);
            let sel = m
                .select($vec::from_array(t), $vec::from_array(f))
                .to_array();
            for i in 0..$lanes {
                let want = if (p >> i) & 1 == 1 { t[i] } else { f[i] };
                assert_eq!(sel[i], want, "{} select lane {} pattern {:#x}", stringify!($mask), i, p);
            }
        }
    }};
}

#[test]
fn mask_api_matches_bit_arithmetic() {
    mask_api_parity!(F32Mask8, F32x8, u8, 8);
    mask_api_parity!(F64Mask4, F64x4, u8, 4);
    mask_api_parity!(F64Mask8, F64x8, u8, 8);
    mask_api_parity!(F32Mask16, F32x16, u16, 16);
}

#[test]
fn compare_select_early_exit_loop() {
    // The shape of the Mandelbrot inner loop that motivated F1/F2: compare,
    // early-exit on `none()`, and count with `select`.
    let c_re = F64x8::from_array([-2.0, -1.0, -0.5, 0.0, 0.25, 0.3, 0.5, 1.0]);
    let (mut re, mut im) = (c_re, F64x8::splat(0.0));
    let mut count = F64x8::splat(0.0);
    for _ in 0..50 {
        let (rr, ii) = (re * re, im * im);
        let alive = (rr + ii).simd_le(F64x8::splat(4.0));
        if alive.none() {
            break;
        }
        count += alive.select(F64x8::splat(1.0), F64x8::splat(0.0));
        let ri = re * im;
        re = c_re + (rr - ii);
        im = ri + ri;
    }
    // Scalar reference with the same iteration rule.
    let want: [f64; 8] = c_re.to_array().map(|c| {
        let (mut r, mut i, mut n) = (c, 0.0f64, 0.0f64);
        for _ in 0..50 {
            if r * r + i * i > 4.0 {
                break;
            }
            n += 1.0;
            let ri = r * i;
            r = c + (r * r - i * i);
            i = ri + ri;
        }
        n
    });
    assert_eq!(count.to_array(), want);
}

/// NaN handling of min/max/reduce must stay exactly what `f32::min` /
/// `fold(INFINITY, f32::min)` give: that is the behaviour the AVX2 and
/// scalar 512-bit types had before plan F2 (it is a compatibility contract,
/// not a free choice), and the new 256-bit types follow it on every arm.
macro_rules! nan_minmax_contract {
    ($vec:ident, $elem:ty, $lanes:expr) => {{
        let pool: [$elem; 6] = [<$elem>::NAN, 1.0, -2.0, <$elem>::NAN, 3.0, -0.5];
        for seed in 0..12usize {
            let a: [$elem; $lanes] = core::array::from_fn(|i| pool[(seed + i) % 6]);
            let b: [$elem; $lanes] = core::array::from_fn(|i| pool[(seed * 5 + i * 7 + 1) % 6]);
            let (va, vb) = ($vec::from_array(a), $vec::from_array(b));
            let (mn, mx) = (va.simd_min(vb).to_array(), va.simd_max(vb).to_array());
            for i in 0..$lanes {
                let (wmin, wmax) = (a[i].min(b[i]), a[i].max(b[i]));
                assert!(
                    (mn[i].is_nan() && wmin.is_nan()) || mn[i] == wmin,
                    "{}::simd_min lane {i}: {:?} {:?} got {:?} want {:?}",
                    stringify!($vec),
                    a[i],
                    b[i],
                    mn[i],
                    wmin
                );
                assert!(
                    (mx[i].is_nan() && wmax.is_nan()) || mx[i] == wmax,
                    "{}::simd_max lane {i}: {:?} {:?} got {:?} want {:?}",
                    stringify!($vec),
                    a[i],
                    b[i],
                    mx[i],
                    wmax
                );
            }
            let rmin = a.iter().copied().fold(<$elem>::INFINITY, <$elem>::min);
            let rmax = a.iter().copied().fold(<$elem>::NEG_INFINITY, <$elem>::max);
            assert_eq!(va.reduce_min(), rmin, "{}::reduce_min {:?}", stringify!($vec), a);
            assert_eq!(va.reduce_max(), rmax, "{}::reduce_max {:?}", stringify!($vec), a);
        }
        // The `nightly-simd` backend has always delegated to `core::simd`'s
        // `reduce_min`/`reduce_max`, which return NaN for an all-NaN vector;
        // that pre-existing behaviour is kept unchanged, so only the stable
        // arms are held to the fold's ±INFINITY here.
        if !cfg!(feature = "nightly-simd") {
            let all_nan = $vec::splat(<$elem>::NAN);
            assert_eq!(all_nan.reduce_min(), <$elem>::INFINITY, "{} all-NaN reduce_min", stringify!($vec));
            assert_eq!(all_nan.reduce_max(), <$elem>::NEG_INFINITY, "{} all-NaN reduce_max", stringify!($vec));
        }
    }};
}

#[test]
fn nan_min_max_reduce_follow_f32_min_on_256_bit_types() {
    nan_minmax_contract!(F32x8, f32, 8);
    nan_minmax_contract!(F64x4, f64, 4);
}

/// The 512-bit types on the arms whose master implementation was the
/// `f32::min` loop: AVX2, AVX-without-AVX2 and the scalar arm (x86-64-v2 and
/// non-x86 targets other than aarch64/wasm, whose native types are unchanged).
#[cfg(not(any(target_feature = "avx512f", target_arch = "aarch64", target_arch = "wasm32")))]
#[test]
fn nan_min_max_reduce_unchanged_on_avx2_and_scalar_512_bit_types() {
    nan_minmax_contract!(F32x16, f32, 16);
    nan_minmax_contract!(F64x8, f64, 8);
}
