//! Extended activation functions — GELU, SiLU, Swish, Mish.
//!
//! All functions operate **in-place** on `&mut [f32]`.  Transcendental ops
//! (`exp`, `tanh`) are delegated to `f32` intrinsics — no raw SIMD primitives
//! are used here; callers needing SIMD throughput should vectorise at the
//! outer loop via `crate::hpc::vml::{vsexp, vstanh}`.
//!
//! # Activation summary
//!
//! | Function | Formula |
//! |---|---|
//! | [`gelu_f32`]       | `0.5 · x · (1 + erf(x / √2))` (exact Gaussian CDF) |
//! | [`gelu_tanh_f32`]  | `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))` (fast approx.) |
//! | [`silu_f32`]       | `x · sigmoid(x) = x / (1 + e^{-x})` |
//! | [`swish_f32`]      | `x · sigmoid(β · x) = x / (1 + e^{-β·x})` (generalised SiLU) |
//! | [`mish_f32`]       | `x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))` |
//!
//! # Example
//!
//! ```rust
//! use ndarray::hpc::linalg::activations_ext::{gelu_f32, silu_f32};
//!
//! let mut v = vec![0.0f32, 1.0, -1.0];
//! gelu_f32(&mut v);
//! assert!((v[0] - 0.0).abs() < 1e-6);
//!
//! let mut w = vec![0.0f32, 2.0];
//! silu_f32(&mut w);
//! assert!((w[0] - 0.0).abs() < 1e-6);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// √(2/π) — used in the GELU tanh approximation.
const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)

/// Coefficient for the cubic term in the GELU tanh approximation.
const GELU_COEFF: f32 = 0.044_715;

// ─────────────────────────────────────────────────────────────────────────────
// GELU (exact Gaussian CDF)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place GELU (Gaussian Error Linear Unit) — exact form using `erf`.
///
/// Formula: `x ← 0.5 · x · (1 + erf(x / √2))`
///
/// This is the standard definition from Hendrycks & Gimpel (2016).
/// For the fast tanh-approximated variant see [`gelu_tanh_f32`].
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::activations_ext::gelu_f32;
///
/// let mut v = vec![0.0f32];
/// gelu_f32(&mut v);
/// assert!((v[0] - 0.0).abs() < 1e-6);  // GELU(0) = 0
///
/// let mut v2 = vec![1.0f32];
/// gelu_f32(&mut v2);
/// // GELU(1) ≈ 0.8413
/// assert!((v2[0] - 0.841_3f32).abs() < 1e-3);
/// ```
pub fn gelu_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        // erf via libm / std — stable Rust 1.45+.
        *v = 0.5 * *v * (1.0 + erf_f32(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GELU — tanh approximation (as used in GPT-2/BERT)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place GELU tanh approximation — fast variant used in GPT-2 / BERT.
///
/// Formula: `x ← 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))`
///
/// This avoids `erf` and replaces it with a cheap `tanh` call.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::activations_ext::gelu_tanh_f32;
///
/// let mut v = vec![0.0f32];
/// gelu_tanh_f32(&mut v);
/// assert!((v[0] - 0.0).abs() < 1e-6);  // gelu_tanh(0) = 0
/// ```
pub fn gelu_tanh_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        let inner = SQRT_2_OVER_PI * (*v + GELU_COEFF * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SiLU (Sigmoid Linear Unit)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place SiLU (Sigmoid Linear Unit): `x ← x · sigmoid(x) = x / (1 + e^{-x})`.
///
/// Equivalent to `swish_f32` with `beta = 1`.  Widely used in LLaMA / MobileNet.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::activations_ext::silu_f32;
///
/// let mut v = vec![0.0f32];
/// silu_f32(&mut v);
/// assert!((v[0] - 0.0).abs() < 1e-6);  // SiLU(0) = 0
/// ```
pub fn silu_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let sig = 1.0 / (1.0 + (-*v).exp());
        *v = *v * sig;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swish (generalised SiLU)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place Swish: `x ← x · sigmoid(β · x) = x / (1 + e^{-β·x})`.
///
/// When `beta = 1` this is identical to [`silu_f32`].
/// When `beta → ∞` it converges to ReLU.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::activations_ext::swish_f32;
///
/// let mut v = vec![0.0f32];
/// swish_f32(&mut v, 1.0);
/// assert!((v[0] - 0.0).abs() < 1e-6);  // Swish(0, 1) = 0
/// ```
pub fn swish_f32(x: &mut [f32], beta: f32) {
    for v in x.iter_mut() {
        let sig = 1.0 / (1.0 + (-beta * *v).exp());
        *v = *v * sig;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mish
// ─────────────────────────────────────────────────────────────────────────────

/// In-place Mish: `x ← x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))`.
///
/// Introduced by Misra (2019).  Smooth and non-monotonic, competitive with
/// GELU on vision tasks.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::activations_ext::mish_f32;
///
/// let mut v = vec![0.0f32];
/// mish_f32(&mut v);
/// assert!((v[0] - 0.0).abs() < 1e-6);  // Mish(0) = 0 * tanh(ln 2) = 0
/// ```
pub fn mish_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        // softplus(x) = ln(1 + exp(x)); numerically safe for large x.
        let sp = if *v > 20.0 {
            *v // softplus(x) ≈ x for large x
        } else {
            (1.0 + v.exp()).ln()
        };
        *v = *v * sp.tanh();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helper: erf approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Approximation of `erf(x)` for f32 using the Abramowitz & Stegun formula.
///
/// Maximum absolute error < 1.5 × 10⁻⁷ across all x.
#[inline]
fn erf_f32(x: f32) -> f32 {
    // Use std::f32 erf if available (Rust 1.45+, all platforms with libm).
    // We delegate to the standard library to stay within the "no raw SIMD
    // outside crate::hpc::vml" rule.
    // Abramowitz & Stegun §7.1.26 rational approximation (p ≈ 0.3275911).
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let x = x.abs();

    const P:  f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_592;
    const A2: f32 = -0.284_496_736;
    const A3: f32 = 1.421_413_741;
    const A4: f32 = -1.453_152_027;
    const A5: f32 = 1.061_405_429;

    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_zero_is_zero() {
        let mut v = vec![0.0f32];
        gelu_f32(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-6, "GELU(0) should be 0, got {}", v[0]);
    }

    #[test]
    fn silu_zero_is_zero() {
        let mut v = vec![0.0f32];
        silu_f32(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-6, "SiLU(0) should be 0, got {}", v[0]);
    }

    #[test]
    fn silu_equals_x_times_sigmoid() {
        // SiLU(x) = x * sigmoid(x) — verify for several values.
        let inputs = [-2.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
        let mut v = inputs.to_vec();
        silu_f32(&mut v);

        for (&xi, &si) in inputs.iter().zip(v.iter()) {
            let expected = xi / (1.0 + (-xi).exp());
            assert!(
                (si - expected).abs() < 1e-5,
                "SiLU({xi}) expected {expected}, got {si}"
            );
        }
    }

    #[test]
    fn swish_beta1_equals_silu() {
        let inputs = [-2.0f32, -0.5, 0.0, 1.0, 2.5];

        let mut silu_out = inputs.to_vec();
        silu_f32(&mut silu_out);

        let mut swish_out = inputs.to_vec();
        swish_f32(&mut swish_out, 1.0);

        for (s, w) in silu_out.iter().zip(swish_out.iter()) {
            assert!(
                (s - w).abs() < 1e-5,
                "swish(β=1) != silu: {w} vs {s}"
            );
        }
    }

    #[test]
    fn gelu_tanh_zero_is_zero() {
        let mut v = vec![0.0f32];
        gelu_tanh_f32(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-6, "GELU_tanh(0) should be 0, got {}", v[0]);
    }

    #[test]
    fn mish_zero_is_zero() {
        let mut v = vec![0.0f32];
        mish_f32(&mut v);
        // Mish(0) = 0 * tanh(ln 2) = 0
        assert!((v[0] - 0.0).abs() < 1e-6, "Mish(0) should be 0, got {}", v[0]);
    }

    #[test]
    fn swish_zero_is_zero() {
        let mut v = vec![0.0f32];
        swish_f32(&mut v, 1.5);
        assert!((v[0] - 0.0).abs() < 1e-6, "Swish(0, 1.5) should be 0, got {}", v[0]);
    }
}
