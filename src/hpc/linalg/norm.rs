//! Normalization layers — LayerNorm, RMSNorm, GroupNorm.
//!
//! All functions operate **in-place** on `&mut [f32]` slices (a single row /
//! embedding vector or a flat group).  The caller is responsible for chunking
//! multi-row tensors into per-row calls.
//!
//! # Formulas
//!
//! | Function | Formula |
//! |---|---|
//! | [`layer_norm_f32`] | `(x − μ) / √(σ² + ε) * γ + β` |
//! | [`rms_norm_f32`]   | `x / √(mean(x²) + ε) * γ` |
//! | [`group_norm_f32`] | LayerNorm applied to each of `G` equal-sized groups |
//!
//! # Example
//!
//! ```rust
//! use ndarray::hpc::linalg::norm::rms_norm_f32;
//!
//! let mut x     = vec![1.0f32, 2.0, 3.0, 4.0];
//! let     gamma = vec![1.0f32; 4];
//! rms_norm_f32(&mut x, &gamma, 1e-6);
//! // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386;  x / RMS ≈ [0.365, 0.730, 1.095, 1.460]
//! assert!((x[0] - 1.0 / (30.0f32 / 4.0).sqrt()).abs() < 1e-5);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────────

/// In-place LayerNorm: `x ← (x − μ) / √(σ² + ε) * γ + β`.
///
/// - `x`     — input/output vector; length `d`.
/// - `gamma` — per-element scale; length `d`.
/// - `beta`  — per-element bias; length `d`.
/// - `eps`   — small constant for numerical stability (e.g. `1e-5`).
///
/// # Panics
///
/// Panics if `gamma.len() != x.len()` or `beta.len() != x.len()`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::norm::layer_norm_f32;
///
/// let mut x     = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let     gamma = vec![1.0f32; 8];
/// let     beta  = vec![0.0f32; 8];
/// layer_norm_f32(&mut x, &gamma, &beta, 1e-5);
/// // Mean = 5, Var = 4; normalised x[0] = (2-5)/2 = -1.5
/// assert!((x[0] - (-1.5f32)).abs() < 1e-4);
/// ```
pub fn layer_norm_f32(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let d = x.len();
    assert_eq!(gamma.len(), d, "layer_norm_f32: gamma length mismatch");
    assert_eq!(beta.len(),  d, "layer_norm_f32: beta length mismatch");

    // Mean
    let mean: f32 = x.iter().sum::<f32>() / d as f32;

    // Variance (population)
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;

    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..d {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// In-place RMSNorm: `x_i ← x_i / √(mean(x²) + ε) * γ_i`.
///
/// This is the normalisation used in LLaMA / T5 family models.  No mean
/// subtraction is performed — only RMS scaling.
///
/// - `x`     — input/output vector; length `d`.
/// - `gamma` — per-element scale; length `d`.
/// - `eps`   — small constant for numerical stability (e.g. `1e-6`).
///
/// # Panics
///
/// Panics if `gamma.len() != x.len()`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::norm::rms_norm_f32;
///
/// let mut x     = vec![1.0f32; 4];
/// let     gamma = vec![1.0f32; 4];
/// rms_norm_f32(&mut x, &gamma, 1e-8);
/// // RMS([1,1,1,1]) = 1.0  →  x unchanged
/// assert!((x[0] - 1.0).abs() < 1e-6);
/// ```
pub fn rms_norm_f32(x: &mut [f32], gamma: &[f32], eps: f32) {
    let d = x.len();
    assert_eq!(gamma.len(), d, "rms_norm_f32: gamma length mismatch");

    // mean(x²)
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / d as f32;
    let inv_rms = 1.0 / (ms + eps).sqrt();

    for i in 0..d {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GroupNorm
// ─────────────────────────────────────────────────────────────────────────────

/// In-place GroupNorm: applies LayerNorm independently to each of `groups`
/// equal-sized partitions of `x`.
///
/// `x` must have length that is divisible by `groups`.  `gamma` and `beta`
/// are applied per-element (length `d`), not per-group.
///
/// # Panics
///
/// Panics if:
/// - `x.len() % groups != 0`
/// - `gamma.len() != x.len()` or `beta.len() != x.len()`
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::norm::group_norm_f32;
///
/// // 4 channels, 2 groups → each group has 2 elements.
/// let mut x     = vec![1.0f32, 3.0,  7.0, 9.0];
/// let     gamma = vec![1.0f32; 4];
/// let     beta  = vec![0.0f32; 4];
/// group_norm_f32(&mut x, &gamma, &beta, 2, 1e-5);
/// // group 0: mean=2, var=1 → normalised to [-1,+1]
/// assert!((x[0] - (-1.0f32)).abs() < 1e-4);
/// assert!((x[1] - (1.0f32)).abs() < 1e-4);
/// ```
pub fn group_norm_f32(x: &mut [f32], gamma: &[f32], beta: &[f32], groups: usize, eps: f32) {
    let d = x.len();
    assert!(
        d % groups == 0,
        "group_norm_f32: x.len() ({d}) not divisible by groups ({groups})"
    );
    assert_eq!(gamma.len(), d, "group_norm_f32: gamma length mismatch");
    assert_eq!(beta.len(),  d, "group_norm_f32: beta length mismatch");

    let g_size = d / groups;

    for g in 0..groups {
        let start = g * g_size;
        let end   = start + g_size;
        let chunk = &mut x[start..end];

        // Mean
        let mean: f32 = chunk.iter().sum::<f32>() / g_size as f32;

        // Variance
        let var: f32 = chunk.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / g_size as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..g_size {
            let gi = start + i;
            chunk[i] = (chunk[i] - mean) * inv_std * gamma[gi] + beta[gi];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_ones_returns_ones() {
        // RMS([1,1,1,1]) = 1  →  output = gamma * 1 / 1 = gamma
        let mut x     = vec![1.0f32; 4];
        let     gamma = vec![1.0f32; 4];
        rms_norm_f32(&mut x, &gamma, 1e-8);
        for &v in &x {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn rms_norm_formula_correctness() {
        // Verify: x_i / sqrt(mean(x^2) + eps) * gamma
        let mut x     = vec![1.0f32, 2.0, 3.0, 4.0];
        let     gamma = vec![2.0f32; 4];
        let eps = 0.0f32;
        rms_norm_f32(&mut x, &gamma, eps);
        // mean(x^2) of [1,2,3,4] = 30/4 = 7.5; rms = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        let expected = [1.0 / rms * 2.0, 2.0 / rms * 2.0, 3.0 / rms * 2.0, 4.0 / rms * 2.0];
        for (got, exp) in x.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut x     = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let     gamma = vec![1.0f32; 8];
        let     beta  = vec![0.0f32; 8];
        layer_norm_f32(&mut x, &gamma, &beta, 1e-10);

        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var:  f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;

        assert!(mean.abs() < 1e-5, "mean should be 0, got {mean}");
        assert!((var - 1.0).abs() < 1e-4, "var should be 1, got {var}");
    }

    #[test]
    fn group_norm_two_groups() {
        // 4 channels, 2 groups → LayerNorm applied to [1,3] and [7,9] separately.
        let mut x     = vec![1.0f32, 3.0, 7.0, 9.0];
        let     gamma = vec![1.0f32; 4];
        let     beta  = vec![0.0f32; 4];
        group_norm_f32(&mut x, &gamma, &beta, 2, 1e-10);

        // Group 0: mean=2, var=1 → [-1, 1]
        assert!((x[0] - (-1.0)).abs() < 1e-4);
        assert!((x[1] - 1.0).abs() < 1e-4);
        // Group 1: mean=8, var=1 → [-1, 1]
        assert!((x[2] - (-1.0)).abs() < 1e-4);
        assert!((x[3] - 1.0).abs() < 1e-4);
    }
}
