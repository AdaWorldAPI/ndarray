//! Activation functions: sigmoid, softmax, log_softmax.
//!
//! Generic trait impl via `mapv` + standalone SIMD-accelerated f32 slice functions.

use crate::simd::{simd_exp_f32, F32x16};
use crate::imp_prelude::*;
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
        let log_sum_exp = shifted.mapv(|v| v.exp()).iter().fold(A::zero(), |acc, &v| acc + v).ln();
        shifted.mapv(|v| v - log_sum_exp)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Standalone SIMD-accelerated f32 functions
//
// These operate on raw slices for maximum throughput. Use when you
// have contiguous f32 data and want to bypass the generic trait.
// ═══════════════════════════════════════════════════════════════════

/// SIMD sigmoid: out[i] = 1 / (1 + exp(-x[i]))
///
/// Processes 16 elements at a time via F32x16 polynomial exp.
pub fn sigmoid_f32(x: &[f32], out: &mut [f32]) {
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

/// SIMD softmax: out = exp(x - max) / sum(exp(x - max))
///
/// Numerically stable. Uses F32x16 for exp and reduce_sum.
pub fn softmax_f32(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 { return; }

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

/// SIMD log-softmax: out[i] = (x[i] - max) - ln(sum(exp(x - max)))
///
/// Numerically stable. Single pass for max, single pass for sum-exp.
pub fn log_softmax_f32(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    if n == 0 { return; }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

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

    // ── SIMD f32 standalone function tests ──────────────────────────

    #[test]
    fn test_sigmoid_f32_zero() {
        let x = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        sigmoid_f32(&x, &mut out);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-3, "sigmoid(0) should be 0.5, got {}", v);
        }
    }

    #[test]
    fn test_sigmoid_f32_range() {
        let x: Vec<f32> = (-5..=5).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; x.len()];
        sigmoid_f32(&x, &mut out);
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid must be in [0,1], got {}", v);
        }
        // sigmoid(-5) < 0.01, sigmoid(5) > 0.99
        assert!(out[0] < 0.02, "sigmoid(-5) = {}", out[0]);
        assert!(out[10] > 0.98, "sigmoid(5) = {}", out[10]);
    }

    #[test]
    fn test_softmax_f32_sums_to_one() {
        let x: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 20];
        softmax_f32(&x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "softmax sum = {}", sum);
        for &v in &out {
            assert!(v >= 0.0, "softmax must be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_softmax_f32_uniform() {
        let x = [0.0f32; 16];
        let mut out = [0.0f32; 16];
        softmax_f32(&x, &mut out);
        for &v in &out {
            assert!((v - 1.0 / 16.0).abs() < 1e-3, "uniform softmax should be 1/16, got {}", v);
        }
    }

    #[test]
    fn test_log_softmax_f32_consistency() {
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut softmax_out = vec![0.0f32; 10];
        let mut logsm_out = vec![0.0f32; 10];
        softmax_f32(&x, &mut softmax_out);
        log_softmax_f32(&x, &mut logsm_out);
        for i in 0..10 {
            let expected = softmax_out[i].ln();
            assert!((logsm_out[i] - expected).abs() < 1e-2,
                "log_softmax[{}] = {}, expected {}", i, logsm_out[i], expected);
        }
    }

    #[test]
    fn test_sigmoid_f32_tail() {
        // Non-multiple of 16 to test scalar tail
        let x = [0.0f32; 5];
        let mut out = [0.0f32; 5];
        sigmoid_f32(&x, &mut out);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-3);
        }
    }
}
