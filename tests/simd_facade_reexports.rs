//! The `ndarray::simd` re-exports of the VML words, the activations and the
//! batched f32 matmul resolve and name the same functions as their `hpc`
//! homes: each call through `simd::` gives bit-identical output to the call
//! through `hpc::`.
#![cfg(feature = "std")]

use ndarray::hpc;
use ndarray::simd;
use ndarray::{Array1, Array2, Axis};

fn input(n: usize) -> Array1<f32> {
    Array1::from_shape_fn(n, |i| ((i as f32) * 0.37 - 7.0).sin() * 3.0)
}

fn bits(a: &Array1<f32>) -> Vec<u32> {
    a.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn unary_vml_words_match_their_hpc_homes() {
    type U = fn(ndarray::ArrayView1<f32>, ndarray::ArrayViewMut1<f32>);
    let pairs: [(&str, U, U); 13] = [
        ("vsexp", simd::vsexp, hpc::vml::vsexp),
        ("vsln", simd::vsln, hpc::vml::vsln),
        ("vssqrt", simd::vssqrt, hpc::vml::vssqrt),
        ("vsabs", simd::vsabs, hpc::vml::vsabs),
        ("vssin", simd::vssin, hpc::vml::vssin),
        ("vscos", simd::vscos, hpc::vml::vscos),
        ("vstanh", simd::vstanh, hpc::vml::vstanh),
        ("vserf", simd::vserf, hpc::vml::vserf),
        ("vsfloor", simd::vsfloor, hpc::vml::vsfloor),
        ("vsceil", simd::vsceil, hpc::vml::vsceil),
        ("vsround", simd::vsround, hpc::vml::vsround),
        ("vsneg", simd::vsneg, hpc::vml::vsneg),
        ("vstrunc", simd::vstrunc, hpc::vml::vstrunc),
    ];
    let x = input(67).mapv(f32::abs) + 0.01;
    for (name, via_simd, via_hpc) in pairs {
        let (mut a, mut b) = (Array1::zeros(67), Array1::zeros(67));
        via_simd(x.view(), a.view_mut());
        via_hpc(x.view(), b.view_mut());
        assert_eq!(bits(&a), bits(&b), "{name}");
    }
}

#[test]
fn binary_vml_words_match_their_hpc_homes() {
    type B = fn(ndarray::ArrayView1<f32>, ndarray::ArrayView1<f32>, ndarray::ArrayViewMut1<f32>);
    let pairs: [(&str, B, B); 4] = [
        ("vsadd", simd::vsadd, hpc::vml::vsadd),
        ("vsmul", simd::vsmul, hpc::vml::vsmul),
        ("vsdiv", simd::vsdiv, hpc::vml::vsdiv),
        ("vspow", simd::vspow, hpc::vml::vspow),
    ];
    let a_in = input(41).mapv(f32::abs) + 0.5;
    let b_in = input(41).mapv(|v| v * 0.25 + 1.0);
    for (name, via_simd, via_hpc) in pairs {
        let (mut a, mut b) = (Array1::zeros(41), Array1::zeros(41));
        via_simd(a_in.view(), b_in.view(), a.view_mut());
        via_hpc(a_in.view(), b_in.view(), b.view_mut());
        assert_eq!(bits(&a), bits(&b), "{name}");
    }
}

#[test]
fn activations_match_their_hpc_homes() {
    let x = input(33);
    let (mut a, mut b) = (Array1::zeros(33), Array1::zeros(33));
    simd::sigmoid_f32(x.view(), a.view_mut());
    hpc::activations::sigmoid_f32(x.view(), b.view_mut());
    assert_eq!(bits(&a), bits(&b), "sigmoid_f32");
    simd::softmax_f32(x.view(), a.view_mut());
    hpc::activations::softmax_f32(x.view(), b.view_mut());
    assert_eq!(bits(&a), bits(&b), "softmax_f32");
    simd::log_softmax_f32(x.view(), a.view_mut());
    hpc::activations::log_softmax_f32(x.view(), b.view_mut());
    assert_eq!(bits(&a), bits(&b), "log_softmax_f32");

    let m = Array2::from_shape_fn((5, 7), |(i, j)| ((i * 7 + j) as f32 * 0.3).cos());
    for axis in [Axis(0), Axis(1)] {
        let (mut a, mut b) = (Array2::zeros((5, 7)), Array2::zeros((5, 7)));
        simd::softmax_axis_f32(m.view(), a.view_mut(), axis);
        hpc::activations::softmax_axis_f32(m.view(), b.view_mut(), axis);
        assert_eq!(a.mapv(f32::to_bits), b.mapv(f32::to_bits), "softmax_axis_f32 {axis:?}");
        simd::log_softmax_axis_f32(m.view(), a.view_mut(), axis);
        hpc::activations::log_softmax_axis_f32(m.view(), b.view_mut(), axis);
        assert_eq!(a.mapv(f32::to_bits), b.mapv(f32::to_bits), "log_softmax_axis_f32 {axis:?}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn batched_matmul_matches_its_hpc_home() {
    use ndarray::Array3;
    let lhs = Array3::from_shape_fn((3, 4, 5), |(b, i, k)| ((b * 20 + i * 5 + k) as f32 * 0.1).sin());
    let rhs = Array3::from_shape_fn((3, 5, 6), |(b, k, j)| ((b * 30 + k * 6 + j) as f32 * 0.2).cos());
    let (mut a, mut b) = (Array3::zeros((3, 4, 6)), Array3::zeros((3, 4, 6)));
    simd::matmul_f32_batched(lhs.view(), rhs.view(), a.view_mut()).unwrap();
    hpc::amx_matmul::matmul_f32_batched(lhs.view(), rhs.view(), b.view_mut()).unwrap();
    assert_eq!(a.mapv(f32::to_bits), b.mapv(f32::to_bits));
    // The error type is reachable through the same path.
    let mut bad = Array3::zeros((2, 4, 6));
    let err: simd::MatmulError = simd::matmul_f32_batched(lhs.view(), rhs.view(), bad.view_mut()).unwrap_err();
    assert!(matches!(err, simd::MatmulError::ShapeMismatch { .. }), "{err:?}");
}
