//! Activation functions: sigmoid, softmax, log_softmax.
//!
//! Generic trait impl via `mapv` + standalone SIMD-accelerated f32 ArrayView functions.

use crate::imp_prelude::*;
use crate::simd::{simd_exp_f32, F32x16};
use crate::{ArrayView, ArrayView1, ArrayView2, ArrayViewMut, ArrayViewMut1, ArrayViewMut2, Axis, Dimension, Zip};
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
    // HOT PATH guard: input + output must share strides AND each be contiguous
    // in their own memory order. Without the strides-equality check a C-order
    // input + F-order output (same shape, both individually contiguous) would
    // both succeed at `as_slice_memory_order` but with mismatched logical
    // indexing — writing the wrong sigmoid value into each output coordinate.
    // Matches the dispatch_unary_contig guard in `hpc/vml.rs`.
    if x.strides() == out.strides() {
        if let (Some(xs), Some(os)) = (x.as_slice_memory_order(), out.as_slice_memory_order_mut()) {
            sigmoid_f32_slice(xs, os);
            return;
        }
    }
    // Cold path: non-contiguous views OR mismatched memory orders — stride-aware scalar.
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

// ═══════════════════════════════════════════════════════════════════
// Axis-aware 2-D variants
//
// These operate on a 2-D `ArrayView2<f32>` and apply the corresponding
// 1-D kernel along each lane parallel to the chosen `Axis`. They reuse
// the existing per-lane SIMD primitives (`softmax_f32` / `log_softmax_f32`)
// so the SIMD fast path fires for each contiguous lane.
//
// Iteration uses `lanes(axis)` / `lanes_mut(axis)` — NOT `axis_iter` —
// per the W2 migration contract (see `.claude/knowledge/
// w2-arrayview-migration.md` § "Axis-aware reduction"). `lanes(Axis(k))`
// yields 1-D lanes ALONG axis k; `axis_iter(Axis(k))` slices perpendicular
// to axis k and is the wrong primitive here.
// ═══════════════════════════════════════════════════════════════════

/// Axis-aware softmax over a 2-D `f32` matrix: each lane along `axis`
/// is normalized independently.
///
/// `out[i, j] = exp(x[i,j] - max_lane) / Σ exp(x[i,:] - max_lane)` for
/// `axis = Axis(1)` (normalize each row); for `axis = Axis(0)` each
/// column is normalized instead.
///
/// Numerically stable: max-shift is applied per lane before exponentiation.
/// The SIMD fast path (via `softmax_f32`) fires when the individual lane is
/// contiguous in memory (typically true for axis 1 on a C-order matrix and
/// axis 0 on an F-order matrix).
///
/// # Panics
/// - Panics if `axis.index() >= 2` (out-of-bounds axis for a 2-D array).
/// - Panics if `x.shape() != out.shape()`.
///
/// # Edge cases
/// - An empty matrix (either dimension is 0) returns immediately.
/// - A lane of length 0 is unreachable given the shape consistency check.
/// - A single-element lane produces `out = 1.0`.
///
/// # Example
/// ```
/// use ndarray::{arr2, Array2, Axis};
/// use ndarray::hpc::activations::softmax_axis_f32;
///
/// // 2×3 matrix, normalize each row (axis 1)
/// let x = arr2(&[[1.0_f32, 2.0, 3.0],
///                [0.0_f32, 0.0, 0.0]]);
/// let mut out = Array2::<f32>::zeros((2, 3));
/// softmax_axis_f32(x.view(), out.view_mut(), Axis(1));
///
/// // each row sums to 1.0
/// assert!((out.row(0).sum() - 1.0_f32).abs() < 1e-5);
/// assert!((out.row(1).sum() - 1.0_f32).abs() < 1e-5);
/// // uniform inputs → equal probabilities
/// assert!((out[[1, 0]] - 1.0 / 3.0).abs() < 1e-5);
/// ```
pub fn softmax_axis_f32(x: ArrayView2<f32>, mut out: ArrayViewMut2<f32>, axis: Axis) {
    assert!(axis.index() < 2, "softmax_axis_f32: axis {} is out of bounds for a 2-D array", axis.index());
    assert_eq!(x.shape(), out.shape(), "softmax_axis_f32: shape mismatch (x={:?} out={:?})", x.shape(), out.shape());
    // `lanes(axis)` yields 1-D views ALONG `axis`; `lanes_mut(axis)` yields
    // the corresponding mutable 1-D views of `out`. Zipping them visits every
    // lane exactly once.
    for (lane_in, lane_out) in x.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
        softmax_f32(lane_in, lane_out);
    }
}

/// Axis-aware log-softmax over a 2-D `f32` matrix: each lane along `axis`
/// is independently normalized in log-space.
///
/// `out[i, j] = (x[i,j] - max_lane) - ln(Σ exp(x[i,:] - max_lane))` for
/// `axis = Axis(1)` (per-row log-softmax); for `axis = Axis(0)` each
/// column is processed instead.
///
/// Numerically stable: max-shift is applied per lane before exponentiation.
/// All output values are ≤ 0, and `exp(out).sum_axis(axis)` ≈ 1.0 for each
/// lane (modulo floating-point rounding).
///
/// The SIMD fast path fires when the individual lane is contiguous in memory
/// (typically true for axis 1 on a C-order matrix).
///
/// # Panics
/// - Panics if `axis.index() >= 2` (out-of-bounds axis for a 2-D array).
/// - Panics if `x.shape() != out.shape()`.
///
/// # Edge cases
/// - An empty matrix (either dimension is 0) returns immediately.
/// - A single-element lane produces `out = 0.0` (log of 1.0).
///
/// # Example
/// ```
/// use ndarray::{arr2, Array2, Axis};
/// use ndarray::hpc::activations::log_softmax_axis_f32;
///
/// // 2×3 matrix, log-softmax along rows (axis 1)
/// let x = arr2(&[[1.0_f32, 2.0, 3.0],
///                [0.0_f32, 0.0, 0.0]]);
/// let mut out = Array2::<f32>::zeros((2, 3));
/// log_softmax_axis_f32(x.view(), out.view_mut(), Axis(1));
///
/// // all outputs must be ≤ 0
/// assert!(out.iter().all(|&v| v <= 0.0));
/// // uniform row: all log-softmax values equal ln(1/3)
/// let expected = (1.0_f32 / 3.0).ln();
/// assert!((out[[1, 0]] - expected).abs() < 1e-5);
/// assert!((out[[1, 1]] - expected).abs() < 1e-5);
/// assert!((out[[1, 2]] - expected).abs() < 1e-5);
/// ```
pub fn log_softmax_axis_f32(x: ArrayView2<f32>, mut out: ArrayViewMut2<f32>, axis: Axis) {
    assert!(axis.index() < 2, "log_softmax_axis_f32: axis {} is out of bounds for a 2-D array", axis.index());
    assert_eq!(
        x.shape(),
        out.shape(),
        "log_softmax_axis_f32: shape mismatch (x={:?} out={:?})",
        x.shape(),
        out.shape()
    );
    for (lane_in, lane_out) in x.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
        log_softmax_f32(lane_in, lane_out);
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
    fn test_sigmoid_f32_c_in_f_out_mismatched_strides() {
        // Regression for Codex PR #154 finding: same-shaped contig views with
        // different memory orders (C-order input + F-order output) both pass
        // `as_slice_memory_order` but with mismatched logical indexing. Without
        // the strides-equality guard, the flat SIMD primitive writes sigmoid
        // values into the wrong output coordinates. The fix re-routes such
        // cases to the stride-aware Zip cold path.
        use crate::{Array, Array2, ShapeBuilder};
        let x: Array2<f32> = arr2(&[[0.0_f32, 100.0], [-100.0, 0.0]]); // C-order
                                                                       // F-order output of the same shape, both individually contiguous,
                                                                       // but `x.strides() != out.strides()`.
        let mut out: Array2<f32> = Array::zeros((2, 2).f());
        assert!(x.as_slice_memory_order().is_some());
        assert!(out.as_slice_memory_order().is_some());
        assert_ne!(x.strides(), out.strides(), "test setup: strides must differ");

        sigmoid_f32(x.view(), out.view_mut());

        // Logical coordinates must carry the right sigmoid values regardless
        // of the underlying memory order.
        assert!((out[[0, 0]] - 0.5).abs() < 1e-6, "sigmoid(0) at [0,0] = {}", out[[0, 0]]);
        assert!((out[[0, 1]] - 1.0).abs() < 1e-4, "sigmoid(100) at [0,1] = {}", out[[0, 1]]);
        assert!((out[[1, 0]] - 0.0).abs() < 1e-4, "sigmoid(-100) at [1,0] = {}", out[[1, 0]]);
        assert!((out[[1, 1]] - 0.5).abs() < 1e-6, "sigmoid(0) at [1,1] = {}", out[[1, 1]]);
    }

    // Symmetric companion to `test_sigmoid_f32_c_in_f_out_mismatched_strides`:
    // F-order INPUT + C-order OUTPUT. The upstream test only covers the
    // C-in-F-out direction; if a future refactor accidentally guards only
    // the asymmetric case where x is C-order (e.g. `if x.is_standard_layout()
    // != out.is_standard_layout()`), the C→F test would still pass while
    // F→C silently regressed. Pinning both directions keeps the
    // strides-equality guard symmetric.
    #[test]
    fn test_sigmoid_f32_f_in_c_out_mismatched_strides() {
        use crate::{Array, Array2, ShapeBuilder};
        // Build an F-order input via `from_shape_vec` with the `.f()` shape
        // builder. Logical contents [[0, 100], [-100, 0]] same as the C-order
        // test — column-major buffer is [0, -100, 100, 0].
        let x: Array2<f32> =
            Array2::from_shape_vec((2, 2).f(), vec![0.0_f32, -100.0, 100.0, 0.0]).expect("F-order shape_vec");
        assert!(!x.is_standard_layout(), "x should be F-order");
        // Sanity-check the logical layout independent of memory order.
        assert!((x[[0, 0]] - 0.0).abs() < 1e-7);
        assert!((x[[0, 1]] - 100.0).abs() < 1e-7);
        assert!((x[[1, 0]] - (-100.0)).abs() < 1e-7);
        assert!((x[[1, 1]] - 0.0).abs() < 1e-7);

        // C-order output (the default for `Array::zeros((r, c))`).
        let mut out: Array2<f32> = Array::zeros((2, 2));
        assert!(out.is_standard_layout(), "out should be C-order");
        assert_ne!(x.strides(), out.strides(), "test setup: strides must differ");

        sigmoid_f32(x.view(), out.view_mut());

        // Logical coordinates must carry the right sigmoid values regardless
        // of memory order direction.
        assert!((out[[0, 0]] - 0.5).abs() < 1e-6, "sigmoid(0) at [0,0] = {}", out[[0, 0]]);
        assert!((out[[0, 1]] - 1.0).abs() < 1e-4, "sigmoid(100) at [0,1] = {}", out[[0, 1]]);
        assert!((out[[1, 0]] - 0.0).abs() < 1e-4, "sigmoid(-100) at [1,0] = {}", out[[1, 0]]);
        assert!((out[[1, 1]] - 0.5).abs() < 1e-6, "sigmoid(0) at [1,1] = {}", out[[1, 1]]);
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

    // ── softmax_axis_f32 ────────────────────────────────────────────

    #[test]
    fn test_softmax_axis1_rows_sum_to_one() {
        // Hand-computed: 2×3, normalize rows (axis 1)
        // Row 0: [1, 2, 3] → exp-shifted by 3 → [e^-2, e^-1, 1] → normalize
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0_f32, 4.0, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 3));
        softmax_axis_f32(x.view(), out.view_mut(), Axis(1));

        // Each row sums to 1.0
        assert!((out.row(0).sum() - 1.0).abs() < 1e-5, "row 0 sum = {}", out.row(0).sum());
        assert!((out.row(1).sum() - 1.0).abs() < 1e-5, "row 1 sum = {}", out.row(1).sum());

        // Uniform row → equal probabilities
        assert!((out[[1, 0]] - 1.0 / 3.0).abs() < 1e-5, "uniform: out[1,0] = {}", out[[1, 0]]);
        assert!((out[[1, 1]] - 1.0 / 3.0).abs() < 1e-5, "uniform: out[1,1] = {}", out[[1, 1]]);
        assert!((out[[1, 2]] - 1.0 / 3.0).abs() < 1e-5, "uniform: out[1,2] = {}", out[[1, 2]]);

        // Monotone row → monotone output
        assert!(out[[0, 0]] < out[[0, 1]] && out[[0, 1]] < out[[0, 2]]);
    }

    #[test]
    fn test_softmax_axis0_cols_sum_to_one() {
        // Normalize columns (axis 0): 2×3 matrix
        // Col 0: [1, 4], Col 1: [2, 4], Col 2: [3, 4]
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0_f32, 4.0, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 3));
        softmax_axis_f32(x.view(), out.view_mut(), Axis(0));

        // Each column sums to 1.0
        for j in 0..3 {
            let col_sum: f32 = out.column(j).sum();
            assert!((col_sum - 1.0).abs() < 1e-5, "col {} sum = {}", j, col_sum);
        }
        // All outputs are non-negative
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_softmax_axis1_correctness_hand_computed() {
        // 1×4: softmax_axis_f32 with axis 1 must match softmax_f32 on the single row
        let x = arr2(&[[0.0_f32, 1.0, 2.0, 3.0]]);
        let mut out_axis = Array2::<f32>::zeros((1, 4));
        softmax_axis_f32(x.view(), out_axis.view_mut(), Axis(1));

        let row = arr1(&[0.0_f32, 1.0, 2.0, 3.0]);
        let mut out_1d = Array1::<f32>::zeros(4);
        softmax_f32(row.view(), out_1d.view_mut());

        for j in 0..4 {
            assert!(
                (out_axis[[0, j]] - out_1d[j]).abs() < 1e-6,
                "j={}: axis={} vs 1d={}",
                j,
                out_axis[[0, j]],
                out_1d[j]
            );
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_softmax_axis_oob_panics() {
        let x = arr2(&[[1.0_f32, 2.0], [3.0_f32, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        softmax_axis_f32(x.view(), out.view_mut(), Axis(2));
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_softmax_axis_shape_mismatch_panics() {
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0_f32, 5.0, 6.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        softmax_axis_f32(x.view(), out.view_mut(), Axis(1));
    }

    // ── log_softmax_axis_f32 ────────────────────────────────────────

    #[test]
    fn test_log_softmax_axis1_all_nonpositive() {
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0_f32, 4.0, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 3));
        log_softmax_axis_f32(x.view(), out.view_mut(), Axis(1));

        assert!(out.iter().all(|&v| v <= 0.0), "log-softmax outputs must be ≤ 0");
    }

    #[test]
    fn test_log_softmax_axis1_uniform_row() {
        // Uniform row [0,0,0]: log-softmax = ln(1/3) for each element
        let x = arr2(&[[0.0_f32, 0.0, 0.0], [1.0_f32, 2.0, 3.0]]);
        let mut out = Array2::<f32>::zeros((2, 3));
        log_softmax_axis_f32(x.view(), out.view_mut(), Axis(1));

        let expected = (1.0_f32 / 3.0).ln();
        for j in 0..3 {
            assert!((out[[0, j]] - expected).abs() < 1e-5, "out[0,{}] = {} expected {}", j, out[[0, j]], expected);
        }
    }

    #[test]
    fn test_log_softmax_axis0_cols_nonpositive() {
        // Normalize columns (axis 0)
        let x = arr2(&[[1.0_f32, 0.0, 3.0], [2.0_f32, 0.0, 1.0]]);
        let mut out = Array2::<f32>::zeros((2, 3));
        log_softmax_axis_f32(x.view(), out.view_mut(), Axis(0));

        assert!(out.iter().all(|&v| v <= 0.0), "log-softmax outputs must be ≤ 0");
        // exp(out) along axis 0 must sum to ~1 per column
        for j in 0..3 {
            let exp_sum: f32 = out.column(j).mapv(f32::exp).sum();
            assert!((exp_sum - 1.0).abs() < 1e-5, "col {} exp-sum = {}", j, exp_sum);
        }
    }

    #[test]
    fn test_log_softmax_axis1_consistency_with_log_softmax_f32() {
        // log_softmax_axis_f32 on each row must match log_softmax_f32 applied directly
        let x = arr2(&[[2.0_f32, 1.0, 0.1, -1.0], [0.5_f32, 0.5, 0.5, 0.5]]);
        let mut out_axis = Array2::<f32>::zeros((2, 4));
        log_softmax_axis_f32(x.view(), out_axis.view_mut(), Axis(1));

        for i in 0..2 {
            let row = x.row(i).to_owned();
            let mut out_1d = Array1::<f32>::zeros(4);
            log_softmax_f32(row.view(), out_1d.view_mut());
            for j in 0..4 {
                assert!(
                    (out_axis[[i, j]] - out_1d[j]).abs() < 1e-5,
                    "row={} j={}: axis={} vs 1d={}",
                    i,
                    j,
                    out_axis[[i, j]],
                    out_1d[j]
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_log_softmax_axis_oob_panics() {
        let x = arr2(&[[1.0_f32, 2.0], [3.0_f32, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        log_softmax_axis_f32(x.view(), out.view_mut(), Axis(3));
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_log_softmax_axis_shape_mismatch_panics() {
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0_f32, 5.0_f32, 6.0_f32]]);
        let mut out = Array2::<f32>::zeros((3, 2));
        log_softmax_axis_f32(x.view(), out.view_mut(), Axis(1));
    }
}
