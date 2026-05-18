//! Activation functions: sigmoid, softmax, log_softmax.
//!
//! Generic trait impl via `mapv` + standalone SIMD-accelerated f32 ArrayView functions.

use crate::imp_prelude::*;
use crate::simd::{simd_exp_f32, F32x16};
use crate::{ArrayView, ArrayView1, ArrayViewMut, ArrayViewMut1, Dimension, Zip};
use num_traits::Float;

/// Neural network activation functions.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::activations::Activations;
///
/// let x = array![1.0f64, 2.0, 3.0];
/// let s = x.sigmoid();
/// assert!(s[0] > 0.7 && s[0] < 0.8);
/// ```
pub trait Activations<A> {
    /// Sigmoid: 1 / (1 + exp(-x))
    fn sigmoid(&self) -> Array<A, Ix1>;

    /// Softmax: exp(x_i) / Σ exp(x_j)
    ///
    /// Numerically stable: subtracts max before exp.
    fn softmax(&self) -> Array<A, Ix1>;

    /// Log-softmax: log(softmax(x))
    ///
    /// Numerically stable: x_i - max - log(Σ exp(x_j - max))
    fn log_softmax(&self) -> Array<A, Ix1>;
}

impl<A, S> Activations<A> for ArrayBase<S, Ix1>
where
    A: Float + 'static,
    S: Data<Elem = A>,
{
    fn sigmoid(&self) -> Array<A, Ix1> {
        self.mapv(|v| A::one() / (A::one() + (-v).exp()))
    }

    fn softmax(&self) -> Array<A, Ix1> {
        let max_val = self.iter().fold(A::neg_infinity(), |a, &b| a.max(b));
        let exps = self.mapv(|v| (v - max_val).exp());
        let sum: A = exps.iter().fold(A::zero(), |acc, &v| acc + v);
        exps.mapv(|v| v / sum)
    }

    fn log_softmax(&self) -> Array<A, Ix1> {
        let max_val = self.iter().fold(A::neg_infinity(), |a, &b| a.max(b));
        let shifted = self.mapv(|v| v - max_val);
        let log_sum_exp = shifted
            .mapv(|v| v.exp())
            .iter()
            .fold(A::zero(), |acc, &v| acc + v)
            .ln();
        shifted.mapv(|v| v - log_sum_exp)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Standalone SIMD-accelerated f32 ArrayView functions
//
// These accept ArrayView/ArrayViewMut so callers can pass any layout
// (contiguous, strided, owned, borrowed). The hot path flattens via
// `as_slice_memory_order` and dispatches to F32x16 polynomial kernels;
// the cold path (non-contiguous views) uses scalar fallback per the
// W2 ArrayView migration contract.
// ═══════════════════════════════════════════════════════════════════

/// SIMD sigmoid: `out[i] = 1 / (1 + exp(-x[i]))`.
///
/// Element-wise, generic over dimension. On the contiguous fast path
/// dispatches to the `F32x16` polynomial exp kernel (16 lanes at a
/// time); on the stride-only cold path falls back to a scalar `Zip`
/// traversal.
///
/// # Panics
/// Panics if `x.shape() != out.shape()`.
///
/// # Example
/// ```
/// use ndarray::arr1;
/// use ndarray::hpc::activations::sigmoid_f32;
/// let x = arr1(&[0.0_f32, 1.0, -1.0]);
/// let mut out = arr1(&[0.0_f32; 3]);
/// sigmoid_f32(x.view(), out.view_mut());
/// assert!((out[0] - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid_f32<D: Dimension>(x: ArrayView<f32, D>, mut out: ArrayViewMut<f32, D>) {
    assert_eq!(x.shape(), out.shape(), "sigmoid_f32: shape mismatch (x={:?} out={:?})", x.shape(), out.shape());
    if let (Some(xs), Some(os)) = (x.as_slice_memory_order(), out.as_slice_memory_order_mut()) {
        sigmoid_f32_slice(xs, os);
        return;
    }
    // Cold path: non-contiguous views (sliced/transposed) — stride-aware scalar.
    Zip::from(&mut out)
        .and(x)
        .for_each(|o, &v| *o = 1.0 / (1.0 + (-v).exp()));
}

/// Slice-flat hot path for `sigmoid_f32`. Kept private — public surface
/// is the `ArrayView`-taking wrapper above.
fn sigmoid_f32_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let one = F32x16::splat(1.0);
    let mut i = 0;
    while i + 16 <= n {
        let v = F32x16::from_slice(&x[i..]);
        let neg_v = -v;
        let exp_neg = simd_exp_f32(neg_v);
        let sigmoid = one / (one + exp_neg);
        sigmoid.copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = 1.0 / (1.0 + (-x[i]).exp());
        i += 1;
    }
}

/// SIMD softmax: `out = exp(x - max) / sum(exp(x - max))`.
///
/// Numerically stable (max-subtraction). Uses `F32x16` for both the
/// reduce_max and reduce_sum passes when the input/output are
/// contiguous; falls back to a scalar three-pass implementation when
/// either view is strided.
///
/// # 1-D only
/// This function takes `ArrayView1` / `ArrayViewMut1` by design.
/// Softmax has a sum-normalize step that needs a single axis. Callers
/// wanting N-D softmax should iterate `.lanes(axis)` /
/// `.lanes_mut(axis)` themselves and call this per lane until the
/// axis-aware variant `softmax_axis_f32` ships (out of scope for the
/// W2-2b wave; see `.claude/knowledge/w2-arrayview-migration.md` §
/// "Axis-aware reduction").
///
/// # Panics
/// Panics if `x.len() != out.len()`.
///
/// # Example
/// ```
/// use ndarray::arr1;
/// use ndarray::hpc::activations::softmax_f32;
/// let x = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
/// let mut out = arr1(&[0.0_f32; 4]);
/// softmax_f32(x.view(), out.view_mut());
/// let sum: f32 = out.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-3);
/// ```
pub fn softmax_f32(x: ArrayView1<f32>, mut out: ArrayViewMut1<f32>) {
    assert_eq!(x.len(), out.len(), "softmax_f32: length mismatch (x={} out={})", x.len(), out.len());
    if x.is_empty() {
        return;
    }
    if let (Some(xs), Some(os)) = (x.as_slice(), out.as_slice_mut()) {
        softmax_f32_slice(xs, os);
        return;
    }
    // Cold path: at least one view is non-contiguous (1-D strided slice).
    softmax_f32_scalar(x, out);
}

fn softmax_f32_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }

    // Pass 1: find max (SIMD reduce_max)
    let mut max_acc = F32x16::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= n {
        max_acc = max_acc.simd_max(F32x16::from_slice(&x[i..]));
        i += 16;
    }
    let mut max_val = max_acc.reduce_max();
    while i < n {
        max_val = max_val.max(x[i]);
        i += 1;
    }

    // Pass 2: compute exp(x - max) and accumulate sum
    let max_v = F32x16::splat(max_val);
    let mut sum_acc = F32x16::splat(0.0);
    i = 0;
    while i + 16 <= n {
        let shifted = F32x16::from_slice(&x[i..]) - max_v;
        let exp_v = simd_exp_f32(shifted);
        exp_v.copy_to_slice(&mut out[i..]);
        sum_acc = sum_acc + exp_v;
        i += 16;
    }
    let mut sum_val = sum_acc.reduce_sum();
    while i < n {
        let e = (x[i] - max_val).exp();
        out[i] = e;
        sum_val += e;
        i += 1;
    }

    // Pass 3: normalize by sum
    let inv_sum = F32x16::splat(1.0 / sum_val);
    i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&out[i..]) * inv_sum).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    let inv = 1.0 / sum_val;
    while i < n {
        out[i] *= inv;
        i += 1;
    }
}

fn softmax_f32_scalar(x: ArrayView1<f32>, mut out: ArrayViewMut1<f32>) {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        let e = (v - max_val).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// SIMD log-softmax: `out[i] = (x[i] - max) - ln(sum(exp(x - max)))`.
///
/// Numerically stable. Single pass for max, single pass for sum-exp,
/// final pass writes shifted-minus-log-sum.
///
/// # 1-D only
/// Takes `ArrayView1` / `ArrayViewMut1` for the same single-axis reason
/// as `softmax_f32`; N-D callers iterate `.lanes(axis)` themselves
/// until `log_softmax_axis_f32` ships.
///
/// # Panics
/// Panics if `x.len() != out.len()`.
///
/// # Example
/// ```
/// use ndarray::arr1;
/// use ndarray::hpc::activations::log_softmax_f32;
/// let x = arr1(&[1.0_f32, 2.0, 3.0]);
/// let mut out = arr1(&[0.0_f32; 3]);
/// log_softmax_f32(x.view(), out.view_mut());
/// // log_softmax is non-positive
/// assert!(out.iter().all(|&v| v <= 0.0));
/// ```
pub fn log_softmax_f32(x: ArrayView1<f32>, mut out: ArrayViewMut1<f32>) {
    assert_eq!(x.len(), out.len(), "log_softmax_f32: length mismatch (x={} out={})", x.len(), out.len());
    if x.is_empty() {
        return;
    }
    if let (Some(xs), Some(os)) = (x.as_slice(), out.as_slice_mut()) {
        log_softmax_f32_slice(xs, os);
        return;
    }
    log_softmax_f32_scalar(x, out);
}

fn log_softmax_f32_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 {
        return;
    }

    // Pass 1: find max
    let mut max_acc = F32x16::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= n {
        max_acc = max_acc.simd_max(F32x16::from_slice(&x[i..]));
        i += 16;
    }
    let mut max_val = max_acc.reduce_max();
    while i < n {
        max_val = max_val.max(x[i]);
        i += 1;
    }

    // Pass 2: compute sum(exp(x - max))
    let max_v = F32x16::splat(max_val);
    let mut sum_acc = F32x16::splat(0.0);
    i = 0;
    while i + 16 <= n {
        sum_acc = sum_acc + simd_exp_f32(F32x16::from_slice(&x[i..]) - max_v);
        i += 16;
    }
    let mut sum_val = sum_acc.reduce_sum();
    while i < n {
        sum_val += (x[i] - max_val).exp();
        i += 1;
    }

    // Pass 3: out[i] = (x[i] - max) - ln(sum)
    let log_sum = sum_val.ln();
    let log_sum_v = F32x16::splat(log_sum);
    i = 0;
    while i + 16 <= n {
        let shifted = F32x16::from_slice(&x[i..]) - max_v;
        (shifted - log_sum_v).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = (x[i] - max_val) - log_sum;
        i += 1;
    }
}

fn log_softmax_f32_scalar(x: ArrayView1<f32>, mut out: ArrayViewMut1<f32>) {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for &v in x.iter() {
        sum += (v - max_val).exp();
    }
    let log_sum = sum.ln();
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = (v - max_val) - log_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arr1, arr2, array, s, Array1, Array2};

    // ── Generic trait tests (kept untouched — already ArrayView-shaped) ──

    #[test]
    fn test_sigmoid() {
        let x = array![0.0f64];
        let s = x.sigmoid();
        assert!((s[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = array![1.0f64, 2.0, 3.0, 4.0];
        let s = x.softmax();
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_softmax_consistency() {
        let x = array![1.0f64, 2.0, 3.0];
        let ls = x.log_softmax();
        let s = x.softmax();
        for i in 0..3 {
            assert!((ls[i] - s[i].ln()).abs() < 1e-10);
        }
    }

    // ── SIMD f32 standalone ArrayView function tests ────────────────

    #[test]
    fn test_sigmoid_f32_zero() {
        let x = Array1::<f32>::zeros(32);
        let mut out = Array1::<f32>::zeros(32);
        sigmoid_f32(x.view(), out.view_mut());
        for &v in out.iter() {
            assert!((v - 0.5).abs() < 1e-3, "sigmoid(0) should be 0.5, got {}", v);
        }
    }

    #[test]
    fn test_sigmoid_f32_range() {
        let x: Array1<f32> = (-5..=5).map(|i| i as f32).collect();
        let mut out = Array1::<f32>::zeros(x.len());
        sigmoid_f32(x.view(), out.view_mut());
        for &v in out.iter() {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid must be in [0,1], got {}", v);
        }
        assert!(out[0] < 0.02, "sigmoid(-5) = {}", out[0]);
        assert!(out[10] > 0.98, "sigmoid(5) = {}", out[10]);
    }

    #[test]
    fn test_sigmoid_f32_tail() {
        // Non-multiple of 16 to exercise the scalar tail of the SIMD path
        let x = Array1::<f32>::zeros(5);
        let mut out = Array1::<f32>::zeros(5);
        sigmoid_f32(x.view(), out.view_mut());
        for &v in out.iter() {
            assert!((v - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_sigmoid_f32_strided() {
        // Cold path: non-contiguous input view forces Zip fallback
        let full = arr1(&[0.0_f32, 99.0, 0.0, 99.0, 0.0, 99.0]);
        let strided = full.slice(s![..;2]); // [0.0, 0.0, 0.0], stride=2
        assert!(strided.as_slice_memory_order().is_none());
        let mut out = Array1::<f32>::zeros(3);
        sigmoid_f32(strided, out.view_mut());
        for &v in out.iter() {
            assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) strided = {}", v);
        }
    }

    #[test]
    fn test_sigmoid_f32_2d() {
        // Generic-D verification: 2-D contiguous input works
        let x = arr2(&[[0.0_f32, 1.0], [-1.0, 2.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        sigmoid_f32(x.view(), out.view_mut());
        assert!((out[[0, 0]] - 0.5).abs() < 1e-6);
        assert!(out[[0, 1]] > 0.7 && out[[0, 1]] < 0.74);
        assert!(out[[1, 0]] > 0.26 && out[[1, 0]] < 0.3);
        assert!(out[[1, 1]] > 0.88 && out[[1, 1]] < 0.9);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_sigmoid_f32_shape_mismatch_panics() {
        let x = Array1::<f32>::zeros(4);
        let mut out = Array1::<f32>::zeros(5);
        sigmoid_f32(x.view(), out.view_mut());
    }

    #[test]
    fn test_softmax_f32_sums_to_one() {
        let x: Array1<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let mut out = Array1::<f32>::zeros(20);
        softmax_f32(x.view(), out.view_mut());
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax sum = {}", sum);
        for &v in out.iter() {
            assert!(v >= 0.0, "softmax must be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_softmax_f32_uniform() {
        let x = Array1::<f32>::zeros(16);
        let mut out = Array1::<f32>::zeros(16);
        softmax_f32(x.view(), out.view_mut());
        for &v in out.iter() {
            assert!((v - 1.0 / 16.0).abs() < 1e-3, "uniform softmax should be 1/16, got {}", v);
        }
    }

    #[test]
    fn test_softmax_f32_strided() {
        // Cold path: strided 1-D input must still sum to 1.0
        let full = arr1(&[1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0]);
        let strided = full.slice(s![..;2]); // [1.0, 2.0, 3.0, 4.0]
        assert!(strided.as_slice().is_none());
        let mut out = Array1::<f32>::zeros(4);
        softmax_f32(strided, out.view_mut());
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "strided softmax sum = {}", sum);
        // Monotonic in input → monotonic in output
        assert!(out[0] < out[1] && out[1] < out[2] && out[2] < out[3]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_softmax_f32_length_mismatch_panics() {
        let x = Array1::<f32>::zeros(4);
        let mut out = Array1::<f32>::zeros(5);
        softmax_f32(x.view(), out.view_mut());
    }

    #[test]
    fn test_log_softmax_f32_consistency() {
        let x: Array1<f32> = (0..10).map(|i| i as f32).collect();
        let mut softmax_out = Array1::<f32>::zeros(10);
        let mut logsm_out = Array1::<f32>::zeros(10);
        softmax_f32(x.view(), softmax_out.view_mut());
        log_softmax_f32(x.view(), logsm_out.view_mut());
        for i in 0..10 {
            let expected = softmax_out[i].ln();
            assert!(
                (logsm_out[i] - expected).abs() < 1e-2,
                "log_softmax[{}] = {}, expected {}",
                i,
                logsm_out[i],
                expected
            );
        }
    }

    #[test]
    fn test_log_softmax_f32_strided() {
        // Cold path: strided 1-D input matches softmax-then-ln
        let full = arr1(&[1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0]);
        let strided = full.slice(s![..;2]); // [1.0, 2.0, 3.0, 4.0]
        assert!(strided.as_slice().is_none());
        let mut logsm_out = Array1::<f32>::zeros(4);
        log_softmax_f32(strided, logsm_out.view_mut());
        let mut softmax_out = Array1::<f32>::zeros(4);
        softmax_f32(strided, softmax_out.view_mut());
        for i in 0..4 {
            let expected = softmax_out[i].ln();
            assert!(
                (logsm_out[i] - expected).abs() < 1e-4,
                "strided log_softmax[{}] = {}, expected {}",
                i,
                logsm_out[i],
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_log_softmax_f32_length_mismatch_panics() {
        let x = Array1::<f32>::zeros(4);
        let mut out = Array1::<f32>::zeros(5);
        log_softmax_f32(x.view(), out.view_mut());
    }
}
