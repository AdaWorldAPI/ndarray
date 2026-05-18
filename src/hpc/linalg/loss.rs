//! Cross-entropy loss and fused softmax-backward for training loops.
//!
//! # Overview
//!
//! This module provides three primitives covering the forward/backward
//! cross-entropy path used in language-model training:
//!
//! | Function | Purpose |
//! |---|---|
//! | [`cross_entropy_with_logits_f32`] | Scalar loss for a single sample |
//! | [`cross_entropy_with_logits_batched_f32`] | Mean loss over a batch |
//! | [`softmax_xent_backward_f32`] | Fused softmax + ∂Loss/∂logits |
//!
//! # Numerical stability
//!
//! All three functions subtract the per-row maximum before computing
//! `exp`, preventing overflow for large logits (e.g. residual logits of
//! ~100 in LLMs).  The reduction over the vocabulary axis uses
//! **Kahan compensated summation** to limit floating-point rounding
//! error even for large vocabularies (128 k+).
//!
//! # Scope / hard boundaries
//!
//! - No SIMD intrinsics — any SIMD acceleration must go through
//!   `crate::hpc::vml` or `crate::simd::*`.
//! - No `unsafe` in this module.
//! - `f32` only; `f64` variants are out of scope for this wave.

// ============================================================================
// Helpers
// ============================================================================

/// Kahan-compensated sum over a slice.
///
/// Accumulates `iter` with O(1) error bound instead of O(n) for naive sum.
/// Used on every vocab-axis reduction.
#[inline]
fn kahan_sum(iter: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32; // running compensation
    for x in iter {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Compute log-sum-exp for a row slice using the max-subtraction trick.
///
/// Returns `(max_val, log_sum_exp)` where
/// `log_sum_exp = max_val + ln(kahan_sum(exp(row[i] - max_val)))`.
#[inline]
fn log_sum_exp_row(row: &[f32]) -> (f32, f32) {
    debug_assert!(!row.is_empty(), "log_sum_exp_row: empty row");
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = kahan_sum(row.iter().map(|&v| (v - max_val).exp()));
    (max_val, max_val + sum.ln())
}

// ============================================================================
// Public API
// ============================================================================

/// Cross-entropy loss with logits for a single sample.
///
/// Computes `-log(softmax(logits)[target])` using the numerically stable
/// log-sum-exp identity:
///
/// ```text
/// loss = log_sum_exp(logits) - logits[target]
///      = max + ln(Σ exp(logits[i] - max)) - logits[target]
/// ```
///
/// Kahan summation is used for the vocab-axis reduction.
///
/// # Arguments
///
/// - `logits` — unnormalised log-probabilities, shape `[vocab]`.
/// - `target` — ground-truth class index (must be `< logits.len()`).
///
/// # Panics
///
/// Panics if `logits` is empty or `target as usize >= logits.len()`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::loss::cross_entropy_with_logits_f32;
///
/// // Correct prediction: logits heavily favour class 0.
/// let loss = cross_entropy_with_logits_f32(&[10.0_f32, 0.0, 0.0], 0);
/// assert!(loss < 1e-4, "near-zero loss for confident correct pred, got {loss}");
///
/// // Wrong prediction: all probability mass on class 0 but target is 1.
/// let loss_wrong = cross_entropy_with_logits_f32(&[10.0_f32, 0.0, 0.0], 1);
/// assert!(loss_wrong > 9.9, "large loss for wrong pred, got {loss_wrong}");
/// ```
pub fn cross_entropy_with_logits_f32(logits: &[f32], target: u32) -> f32 {
    assert!(!logits.is_empty(), "cross_entropy_with_logits_f32: empty logits");
    let t = target as usize;
    assert!(
        t < logits.len(),
        "cross_entropy_with_logits_f32: target {target} out of range [0, {})",
        logits.len()
    );
    let (_, lse) = log_sum_exp_row(logits);
    lse - logits[t]
}

/// Batched cross-entropy loss with logits.
///
/// Computes the mean cross-entropy over a batch:
///
/// ```text
/// loss = (1/batch) * Σ_{b} cross_entropy(logits[b*vocab .. (b+1)*vocab], targets[b])
/// ```
///
/// Kahan summation is used for both the per-sample vocab reduction and
/// the outer batch mean.
///
/// # Arguments
///
/// - `logits` — flat row-major tensor, shape `[batch, vocab]`
///   (length must equal `batch * vocab`).
/// - `targets` — class indices, shape `[batch]`
///   (each entry must be `< vocab`).
/// - `batch` — number of samples.
/// - `vocab` — number of classes.
///
/// # Panics
///
/// Panics if:
/// - `logits.len() != batch * vocab`
/// - `targets.len() != batch`
/// - any `targets[b] as usize >= vocab`
/// - `batch == 0` or `vocab == 0`
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::loss::{
///     cross_entropy_with_logits_f32,
///     cross_entropy_with_logits_batched_f32,
/// };
///
/// let logits = [10.0_f32, 0.0, 0.0,   // sample 0: correct
///               10.0_f32, 0.0, 0.0];  // sample 1: correct
/// let targets = [0_u32, 0];
/// let batched = cross_entropy_with_logits_batched_f32(&logits, &targets, 2, 3);
/// let single  = cross_entropy_with_logits_f32(&logits[..3], 0);
/// assert!((batched - single).abs() < 1e-5);
/// ```
pub fn cross_entropy_with_logits_batched_f32(
    logits: &[f32],
    targets: &[u32],
    batch: usize,
    vocab: usize,
) -> f32 {
    assert!(batch > 0, "cross_entropy_with_logits_batched_f32: batch must be > 0");
    assert!(vocab > 0, "cross_entropy_with_logits_batched_f32: vocab must be > 0");
    assert_eq!(
        logits.len(),
        batch * vocab,
        "cross_entropy_with_logits_batched_f32: logits.len()={} != batch*vocab={}",
        logits.len(),
        batch * vocab,
    );
    assert_eq!(
        targets.len(),
        batch,
        "cross_entropy_with_logits_batched_f32: targets.len()={} != batch={}",
        targets.len(),
        batch,
    );

    let sum = kahan_sum((0..batch).map(|b| {
        let row = &logits[b * vocab..(b + 1) * vocab];
        let t = targets[b] as usize;
        assert!(
            t < vocab,
            "cross_entropy_with_logits_batched_f32: targets[{b}]={t} out of range [0, {vocab})"
        );
        let (_, lse) = log_sum_exp_row(row);
        lse - row[t]
    }));

    sum / batch as f32
}

/// Fused softmax + cross-entropy backward pass.
///
/// Computes the gradient of the mean cross-entropy loss with respect to
/// the logits in a single numerically-stable fused pass:
///
/// ```text
/// grad_out[b*vocab + i] = (softmax(logits[b])[i] - 1_{i == targets[b]}) / batch
/// ```
///
/// This is the canonical training-loop primitive: it eliminates a
/// separate softmax allocation and is the minimal representation of the
/// backward signal sent to the embedding / attention layers.
///
/// Kahan summation is used for the exp-sum reduction.
///
/// # Arguments
///
/// - `logits` — flat row-major tensor, shape `[batch, vocab]`.
/// - `targets` — class indices, shape `[batch]`.
/// - `grad_out` — output gradient buffer (same shape as `logits`,
///   `batch * vocab` elements).  The function **overwrites** every
///   element; callers need not zero the buffer beforehand.
/// - `batch` — number of samples.
/// - `vocab` — number of classes.
///
/// # Panics
///
/// Panics if:
/// - `logits.len() != batch * vocab`
/// - `targets.len() != batch`
/// - `grad_out.len() != batch * vocab`
/// - any `targets[b] as usize >= vocab`
/// - `batch == 0` or `vocab == 0`
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::loss::softmax_xent_backward_f32;
///
/// let logits  = [1.0_f32, 2.0, 3.0];
/// let targets = [2_u32];          // class 2 is correct
/// let mut grad = [0.0_f32; 3];
/// softmax_xent_backward_f32(&logits, &targets, &mut grad, 1, 3);
///
/// // Gradient at the target index must be ≤ 0.
/// assert!(grad[2] <= 0.0, "grad at target should be ≤ 0, got {}", grad[2]);
/// // Gradients at non-target indices must be ≥ 0.
/// assert!(grad[0] >= 0.0 && grad[1] >= 0.0);
/// // Gradients sum to zero (partition of unity).
/// let sum: f32 = grad.iter().sum();
/// assert!(sum.abs() < 1e-5, "grad sum should be ~0, got {sum}");
/// ```
pub fn softmax_xent_backward_f32(
    logits: &[f32],
    targets: &[u32],
    grad_out: &mut [f32],
    batch: usize,
    vocab: usize,
) {
    assert!(batch > 0, "softmax_xent_backward_f32: batch must be > 0");
    assert!(vocab > 0, "softmax_xent_backward_f32: vocab must be > 0");
    assert_eq!(
        logits.len(),
        batch * vocab,
        "softmax_xent_backward_f32: logits.len()={} != batch*vocab={}",
        logits.len(),
        batch * vocab,
    );
    assert_eq!(
        targets.len(),
        batch,
        "softmax_xent_backward_f32: targets.len()={} != batch={}",
        targets.len(),
        batch,
    );
    assert_eq!(
        grad_out.len(),
        batch * vocab,
        "softmax_xent_backward_f32: grad_out.len()={} != batch*vocab={}",
        grad_out.len(),
        batch * vocab,
    );

    let scale = 1.0_f32 / batch as f32;

    for b in 0..batch {
        let row_start = b * vocab;
        let row = &logits[row_start..row_start + vocab];
        let t = targets[b] as usize;
        assert!(
            t < vocab,
            "softmax_xent_backward_f32: targets[{b}]={t} out of range [0, {vocab})"
        );

        // Find max for numerical stability.
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // First pass: compute exp(row[i] - max) and write to grad_out.
        // We accumulate the sum using Kahan summation.
        let grad_row = &mut grad_out[row_start..row_start + vocab];
        let mut kahan_c = 0.0_f32;
        let mut exp_sum = 0.0_f32;
        for (g, &v) in grad_row.iter_mut().zip(row.iter()) {
            let e = (v - max_val).exp();
            *g = e;
            // Kahan step
            let y = e - kahan_c;
            let t_sum = exp_sum + y;
            kahan_c = (t_sum - exp_sum) - y;
            exp_sum = t_sum;
        }

        // Second pass: normalize to softmax, subtract one-hot, scale by 1/batch.
        let inv_sum = 1.0_f32 / exp_sum;
        for (i, g) in grad_row.iter_mut().enumerate() {
            let softmax_i = *g * inv_sum;
            let one_hot = if i == t { 1.0_f32 } else { 0.0_f32 };
            *g = (softmax_i - one_hot) * scale;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Gate 1: single-sample forward ───────────────────────────────────────

    /// One-hot target: confident correct prediction → near-zero loss.
    #[test]
    fn test_xent_correct_prediction_near_zero() {
        // softmax([10,0,0]) ≈ [1,0,0]; -log(1) ≈ 0
        let loss = cross_entropy_with_logits_f32(&[10.0_f32, 0.0, 0.0], 0);
        assert!(
            loss < 1e-4,
            "near-zero loss for confident correct pred, got {loss}"
        );
    }

    /// One-hot target pointing to the wrong class → large loss (~10).
    #[test]
    fn test_xent_wrong_prediction_large_loss() {
        // softmax([10,0,0]) ≈ [1,0,0]; -log(softmax[1]) ≈ 10
        let loss = cross_entropy_with_logits_f32(&[10.0_f32, 0.0, 0.0], 1);
        assert!(
            loss > 9.9,
            "large loss expected for wrong prediction, got {loss}"
        );
    }

    /// Uniform logits → loss equals ln(vocab).
    #[test]
    fn test_xent_uniform_logits() {
        let vocab = 8_usize;
        let logits = vec![0.0_f32; vocab];
        let loss = cross_entropy_with_logits_f32(&logits, 0);
        let expected = (vocab as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "uniform logits loss={loss}, expected ln({vocab})={expected}"
        );
    }

    /// Numerically large logits do not produce NaN or Inf.
    #[test]
    fn test_xent_large_logits_stable() {
        let logits = [1000.0_f32, 999.0, 998.0];
        let loss = cross_entropy_with_logits_f32(&logits, 0);
        assert!(loss.is_finite(), "loss should be finite for large logits, got {loss}");
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
    }

    // ── Gate 2: batched == mean of unbatched ─────────────────────────────────

    /// Batched on two identical samples matches single unbatched call.
    #[test]
    fn test_batched_matches_unbatched() {
        let row = [10.0_f32, 0.0, 0.0];
        let logits: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let targets = [0_u32, 0];
        let batched = cross_entropy_with_logits_batched_f32(&logits, &targets, 2, 3);
        let single = cross_entropy_with_logits_f32(&row, 0);
        assert!(
            (batched - single).abs() < 1e-5,
            "batched={batched} should equal single={single}"
        );
    }

    /// Batched mean is the arithmetic mean of individual losses.
    #[test]
    fn test_batched_is_mean() {
        let logits = [1.0_f32, 2.0, 3.0,   // sample 0
                      3.0_f32, 2.0, 1.0];  // sample 1
        let targets = [2_u32, 0];
        let batched = cross_entropy_with_logits_batched_f32(&logits, &targets, 2, 3);
        let l0 = cross_entropy_with_logits_f32(&logits[..3], 2);
        let l1 = cross_entropy_with_logits_f32(&logits[3..], 0);
        let mean = (l0 + l1) / 2.0;
        assert!(
            (batched - mean).abs() < 1e-5,
            "batched={batched} expected mean={mean}"
        );
    }

    // ── Gate 3: backward gradient sign ──────────────────────────────────────

    /// Gradient at target index is negative; at non-target is positive.
    #[test]
    fn test_backward_gradient_sign() {
        let logits = [1.0_f32, 2.0, 3.0];
        let targets = [2_u32]; // class 2 is correct
        let mut grad = [0.0_f32; 3];
        softmax_xent_backward_f32(&logits, &targets, &mut grad, 1, 3);

        assert!(
            grad[2] < 0.0,
            "grad at target index should be negative (softmax - 1), got {}",
            grad[2]
        );
        assert!(
            grad[0] > 0.0 && grad[1] > 0.0,
            "grad at non-target indices should be positive, got [{}, {}]",
            grad[0],
            grad[1]
        );
    }

    /// Gradients sum to zero (partition of unity × 1/batch).
    #[test]
    fn test_backward_grad_sum_zero() {
        let logits = [0.5_f32, 1.5, -0.5, 2.0];
        let targets = [1_u32];
        let mut grad = [0.0_f32; 4];
        softmax_xent_backward_f32(&logits, &targets, &mut grad, 1, 4);
        let sum: f32 = grad.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "grad should sum to 0 (Σ softmax = 1, Σ one_hot = 1), got {sum}"
        );
    }

    // ── Gate 4: backward approximates finite-difference gradient ────────────

    /// Finite-difference check: softmax_xent_backward agrees with numerical gradient.
    #[test]
    fn test_backward_finite_difference() {
        let logits = [0.3_f32, -0.2, 1.1, 0.5];
        let target = 2_u32;
        let eps = 1e-3_f32;
        let vocab = 4_usize;

        // Analytical gradient (single sample → batch=1)
        let mut grad = [0.0_f32; 4];
        softmax_xent_backward_f32(&logits, &[target], &mut grad, 1, vocab);

        // Finite differences
        for i in 0..vocab {
            let mut l_plus = logits.to_vec();
            let mut l_minus = logits.to_vec();
            l_plus[i] += eps;
            l_minus[i] -= eps;
            let f_plus = cross_entropy_with_logits_f32(&l_plus, target);
            let f_minus = cross_entropy_with_logits_f32(&l_minus, target);
            let fd = (f_plus - f_minus) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 5e-4,
                "finite-diff check failed at i={i}: analytical={}, fd={fd}",
                grad[i]
            );
        }
    }

    // ── Gate 5: batched backward consistency ────────────────────────────────

    /// Batched backward on identical samples equals scaled single-sample gradient.
    #[test]
    fn test_batched_backward_matches_unbatched() {
        let row = [1.0_f32, 2.0, 3.0];
        let logits: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let targets = [1_u32, 1];
        let mut grad_batch = [0.0_f32; 6];
        softmax_xent_backward_f32(&logits, &targets, &mut grad_batch, 2, 3);

        // Single-sample backward (batch=1)
        let mut grad_single = [0.0_f32; 3];
        softmax_xent_backward_f32(&row, &[1_u32], &mut grad_single, 1, 3);

        // Each row of the batched result should equal the single result
        // (both are scaled by 1/batch, so batch=2 gives half the magnitude)
        for i in 0..3 {
            let expected = grad_single[i] * 0.5; // batch=2 halves each
            assert!(
                (grad_batch[i] - expected).abs() < 1e-6,
                "batch grad[{i}]={} expected {expected}",
                grad_batch[i]
            );
            assert!(
                (grad_batch[3 + i] - expected).abs() < 1e-6,
                "batch grad[{}]={} expected {expected}",
                3 + i,
                grad_batch[3 + i]
            );
        }
    }

    /// Multi-sample backward: gradients from all rows sum correctly.
    #[test]
    fn test_batched_backward_different_targets() {
        let logits = [
            2.0_f32, 1.0, 0.0,  // sample 0, target 0
            0.0_f32, 1.0, 2.0,  // sample 1, target 2
        ];
        let targets = [0_u32, 2];
        let mut grad = [0.0_f32; 6];
        softmax_xent_backward_f32(&logits, &targets, &mut grad, 2, 3);

        // Target index for sample 0 (index 0) should be negative
        assert!(grad[0] < 0.0, "sample 0 target grad should be < 0, got {}", grad[0]);
        // Target index for sample 1 (index 5) should be negative
        assert!(grad[5] < 0.0, "sample 1 target grad should be < 0, got {}", grad[5]);

        // Each row should sum to zero (grad sum of one sample = (Σ_softmax - 1) / batch = 0/batch)
        let row0_sum: f32 = grad[0..3].iter().sum();
        let row1_sum: f32 = grad[3..6].iter().sum();
        assert!(row0_sum.abs() < 1e-6, "row0 grad sum={row0_sum}");
        assert!(row1_sum.abs() < 1e-6, "row1 grad sum={row1_sum}");
    }
}
