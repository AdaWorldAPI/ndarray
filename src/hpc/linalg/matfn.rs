#![allow(missing_docs)]

//! Matrix functions — exp, log via Padé approximation and spectral methods.
//!
//! # API overview
//!
//! | Function | Method | Suitable for |
//! |---|---|---|
//! | [`mat_exp`] | Padé(13/13) + scaling-and-squaring | General N×N |
//! | [`mat_log`] | Repeated squareroots + Padé(13/13) | General N×N, near-I |
//! | [`mat_exp_spd`] | Spectral (eig_sym + exp) | Symmetric positive definite |
//! | [`mat_log_spd`] | Spectral (eig_sym + log) | Symmetric positive definite |
//!
//! # References
//!
//! - Higham, N.J. (2005). "The Scaling and Squaring Method for the Matrix
//!   Exponential Revisited." *SIAM J. Matrix Anal. Appl.* **26**(4):1179–1193.
//! - Al-Mohy, A.H. & Higham, N.J. (2009). "A New Scaling and Squaring
//!   Algorithm for the Matrix Exponential." *SIAM J. Matrix Anal. Appl.*
//!   **31**(3):970–989.

use super::{inverse::invert_mat_n, MatN};
use crate::hpc::linalg::eig_sym::eig_sym_n;

// ─────────────────────────────────────────────────────────────────────────────
// Internal matrix arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn matmul<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| {
        let mut s = 0.0f32;
        for k in 0..N {
            s += a.get(i, k) * b.get(k, j);
        }
        s
    })
}

#[inline]
fn matadd<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| a.get(i, j) + b.get(i, j))
}

#[inline]
fn matsub<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| a.get(i, j) - b.get(i, j))
}

#[inline]
fn matscale<const N: usize>(a: &MatN<N>, s: f32) -> MatN<N> {
    MatN::from_fn(|i, j| a.get(i, j) * s)
}

/// Frobenius (one-) norm: max absolute column sum.
#[inline]
fn mat_one_norm<const N: usize>(a: &MatN<N>) -> f32 {
    let mut max_col = 0.0f32;
    for j in 0..N {
        let col_sum: f32 = (0..N).map(|i| a.get(i, j).abs()).sum();
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// Matrix squared: A² = A · A.
#[inline]
fn matsq<const N: usize>(a: &MatN<N>) -> MatN<N> {
    matmul(a, a)
}

// ─────────────────────────────────────────────────────────────────────────────
// Padé(13/13) numerator and denominator coefficients
// From Higham 2005, Table 1 (m = 13).
// ─────────────────────────────────────────────────────────────────────────────

// Padé coefficients for order 13 (p13).
// exp(A) ≈ P₁₃(A) / Q₁₃(A) where P₁₃ = numerator, Q₁₃ = denominator.
// Denominator coefficients b_k for k = 0..13.
// Numerator: P₁₃(A) = Q₁₃(-A) (since exp(A) = P/Q and exp(-A) = Q⁻¹P → P=QD)
// Actually the standard form: U = A(c13 A^12 + ... + c1 I), V = c12 A^12 + ... + c0 I
// exp(A) ≈ (V + U)(V - U)⁻¹.

const PADE13_C: [f64; 14] = [
    1.0, 0.5, 0.12, 1.833333333333333e-2, 1.992063492063492e-3, 1.630434782608696e-4, 1.035196687370600e-5,
    5.175983436853003e-7, 2.043151866399031e-8, 6.306659613335001e-10, 1.483770048404140e-11, 2.529153491597966e-13,
    2.810170546428554e-15, 1.544021750048897e-17,
];

/// Evaluate the Padé(13/13) approximant of exp(A).
///
/// Computes `exp(A) ≈ (V + U)(V − U)⁻¹` using Higham's formulation.
/// Caller must have pre-scaled `A` so that `‖A‖₁ ≤ θ₁₃ ≈ 5.37`.
fn pade13<const N: usize>(a: &MatN<N>) -> MatN<N> {
    let c = PADE13_C;
    // Pre-compute powers A², A⁴, A⁶.
    let a2 = matsq(a);
    let a4 = matsq(&a2);
    let a6 = matmul(&a2, &a4);

    // V = A⁶(c13 A⁶ + c11 A⁴ + c9 A²  + c7 I)
    //   + c5 A⁴ + c3 A² + c1 I   [all ×identity replaced by scalars on diag]
    // We follow Higham's Alg 10.20 building U and V.

    let c = [
        c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32, c[4] as f32, c[5] as f32, c[6] as f32, c[7] as f32,
        c[8] as f32, c[9] as f32, c[10] as f32, c[11] as f32, c[12] as f32, c[13] as f32,
    ];

    // W₁ = c13·A⁶ + c11·A⁴ + c9·A²  + c7·I
    let w1 = add4(&matscale(&a6, c[13]), &matscale(&a4, c[11]), &matscale(&a2, c[9]), &scaled_identity::<N>(c[7]));
    // W₂ = c5·A⁴ + c3·A² + c1·I
    let w2 = add3(&matscale(&a4, c[5]), &matscale(&a2, c[3]), &scaled_identity::<N>(c[1]));
    // W = A⁶·W₁ + W₂
    let w = matadd(&matmul(&a6, &w1), &w2);
    // U = A·W
    let u = matmul(a, &w);

    // V₁ = c12·A⁶ + c10·A⁴ + c8·A²  + c6·I
    let v1 = add4(&matscale(&a6, c[12]), &matscale(&a4, c[10]), &matscale(&a2, c[8]), &scaled_identity::<N>(c[6]));
    // V₂ = c4·A⁴ + c2·A² + c0·I
    let v2 = add3(&matscale(&a4, c[4]), &matscale(&a2, c[2]), &scaled_identity::<N>(c[0]));
    // V = A⁶·V₁ + V₂
    let v = matadd(&matmul(&a6, &v1), &v2);

    // exp(A) = (V + U)(V − U)⁻¹
    let num = matadd(&v, &u);
    let den = matsub(&v, &u);
    match invert_mat_n(&den) {
        Some(den_inv) => matmul(&num, &den_inv),
        None => {
            // Fallback: return identity (shouldn't happen for well-scaled inputs)
            MatN::identity()
        }
    }
}

#[inline]
fn scaled_identity<const N: usize>(s: f32) -> MatN<N> {
    MatN::from_fn(|i, j| if i == j { s } else { 0.0 })
}

#[inline]
fn add3<const N: usize>(a: &MatN<N>, b: &MatN<N>, c: &MatN<N>) -> MatN<N> {
    matadd(&matadd(a, b), c)
}

#[inline]
fn add4<const N: usize>(a: &MatN<N>, b: &MatN<N>, c: &MatN<N>, d: &MatN<N>) -> MatN<N> {
    matadd(&matadd(a, b), &matadd(c, d))
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral helpers for SPD functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `MatN<N>` to the `[[f32; N]; N]` type used by `eig_sym_n`.
fn to_raw<const N: usize>(m: &MatN<N>) -> [[f32; N]; N] {
    let mut raw = [[0.0f32; N]; N];
    for i in 0..N {
        for j in 0..N {
            raw[i][j] = m.get(i, j);
        }
    }
    raw
}

/// Reconstruct A = V · diag(f(λ)) · Vᵀ from eigendecomposition.
///
/// `eig_sym_n` returns eigenvectors as `vcols[c][r]` = r-th component of c-th eigenvector.
/// We compute: A = Σ_c f(λ_c) · v_c · v_cᵀ
fn spectral_apply<const N: usize>(eigs: &[f32], vcols: &[[f32; N]; N], f: impl Fn(f32) -> f32) -> MatN<N> {
    let mut result = MatN::<N>::zero();
    for c in 0..N {
        let fval = f(eigs[c]);
        // v_c = vcols[c], a column vector of length N.
        // Add fval * v_c * v_cᵀ to result.
        for i in 0..N {
            for j in 0..N {
                let cur = result.get(i, j);
                result.set(i, j, cur + fval * vcols[c][i] * vcols[c][j]);
            }
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix exponential via Padé(13/13) approximation with scaling-and-squaring.
///
/// Computes `exp(A)` for a general (not necessarily symmetric) N×N matrix.
/// Uses Higham's Algorithm 10.20: scales `A` by `2⁻ˢ` so that
/// `‖A/2ˢ‖₁ ≤ θ₁₃ ≈ 5.371920351148152`, applies the Padé(13) rational
/// approximant, then repeatedly squares `s` times.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::MatN;
/// use ndarray::hpc::linalg::matfn::mat_exp;
///
/// // exp(0) = I
/// let zero: MatN<3> = MatN::zero();
/// let e = mat_exp(&zero);
/// for i in 0..3 {
///     for j in 0..3 {
///         let exp = if i == j { 1.0 } else { 0.0 };
///         assert!((e.get(i, j) - exp).abs() < 1e-5, "exp(0)[{i}][{j}]={}", e.get(i,j));
///     }
/// }
/// ```
pub fn mat_exp<const N: usize>(a: &MatN<N>) -> MatN<N> {
    // θ₁₃ from Higham 2005 Table 3.1.
    const THETA13: f32 = 5.371_920_4;

    let norm = mat_one_norm(a);

    // Choose scaling s = max(0, ⌈log₂(norm/θ₁₃)⌉).
    let s = if norm <= THETA13 {
        0u32
    } else {
        let ratio = norm / THETA13;
        // ceil(log2(ratio))
        let s_f = ratio.log2().ceil();
        s_f.max(0.0) as u32
    };

    let scale = (2.0f32).powi(-(s as i32));
    let a_scaled = matscale(a, scale);

    // Padé(13) approximant.
    let mut result = pade13(&a_scaled);

    // Squaring phase: result = result^(2^s).
    for _ in 0..s {
        result = matsq(&result);
    }

    result
}

/// Matrix logarithm via inverse scaling-and-squaring + Padé approximant.
///
/// Computes `log(A)` for a general N×N matrix `A` near the identity.
/// Reduces `A` to the vicinity of `I` by repeated square roots, computes
/// `log` via the Padé(13) approximant applied to `(A^{1/2^k} - I)`, then
/// scales back: `log(A) = 2^k · log(A^{1/2^k})`.
///
/// **Note**: For symmetric positive definite matrices, prefer [`mat_log_spd`]
/// which is faster and numerically more stable.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::MatN;
/// use ndarray::hpc::linalg::matfn::{mat_exp, mat_log};
///
/// // log(exp(A)) ≈ A for a small diagonal matrix.
/// let a: MatN<3> = MatN::from_fn(|i, j| if i == j { 0.1 * (i as f32 + 1.0) } else { 0.0 });
/// let e = mat_exp(&a);
/// let l = mat_log(&e);
/// for i in 0..3 {
///     for j in 0..3 {
///         assert!((l.get(i, j) - a.get(i, j)).abs() < 1e-4,
///             "log(exp(A))[{i}][{j}]: got {}, want {}", l.get(i,j), a.get(i,j));
///     }
/// }
/// ```
pub fn mat_log<const N: usize>(a: &MatN<N>) -> MatN<N> {
    // We use the inverse scaling-and-squaring approach:
    // Find k such that A^{1/2^k} is close to I, then log(A) = 2^k * log(A^{1/2^k}).
    // We approximate log(X) for X near I via log(X) = log(I + (X-I))
    // using the Padé(13) approximant for log applied to (X - I).

    // First, compute matrix square roots iteratively.
    const MAX_SQRT_ITERS: u32 = 16;
    const SQRT_TOL: f32 = 0.5; // ‖X - I‖₁ threshold to stop

    let mut x = *a;
    let mut k = 0u32;

    for _ in 0..MAX_SQRT_ITERS {
        let norm_diff = {
            let id = MatN::<N>::identity();
            mat_one_norm(&matsub(&x, &id))
        };
        if norm_diff <= SQRT_TOL {
            break;
        }
        // Square root via Denman-Beavers iteration (a single Padé-based step).
        x = mat_sqrt_pade(&x);
        k += 1;
    }

    // Now compute log(X) ≈ Padé rational approximant on (X - I).
    let id = MatN::<N>::identity();
    let t = matsub(&x, &id); // T = X - I, should be small

    // log(I + T) via Padé(13/13): use the identity
    // log(I + T) = 2 * atanh(T (2I + T)^{-1})
    // But a simpler approach: use the Padé approximant directly.
    // For the matrix log Padé, we use a degree-13 Padé on (X - I).
    let log_x = pade_log13(&t);

    // Scale back.
    matscale(&log_x, (1u32 << k) as f32)
}

/// Padé(13/13) approximant for log(I + T) given T = X - I (T should be small).
///
/// Uses the formula from Al-Mohy & Higham (2012), which computes
/// log(I + T) using a rational Padé approximant evaluated via partial fractions.
fn pade_log13<const N: usize>(t: &MatN<N>) -> MatN<N> {
    // Gauss-Legendre quadrature nodes and weights on [0, 1] for degree 13.
    // log(I + T) = T · ∫₀¹ (I + α T)⁻¹ dα  (integral formula)
    // Approximated by 7-point Gauss-Legendre quadrature (14-point symmetry).

    // 7-point Gauss-Legendre nodes/weights on [-1,1], mapped to [0,1].
    const GL_NODES_7: [f64; 7] = [
        -0.949_107_912_342_758_5, -0.741_531_185_599_394_5, -0.405_845_151_377_397_2, 0.0, 0.405_845_151_377_397_2,
        0.741_531_185_599_394_5, 0.949_107_912_342_758_5,
    ];
    const GL_WEIGHTS_7: [f64; 7] = [
        0.129_484_966_168_869_7, 0.279_705_391_489_276_7, 0.381_830_050_505_118_9, 0.417_959_183_673_469_4,
        0.381_830_050_505_118_9, 0.279_705_391_489_276_7, 0.129_484_966_168_869_7,
    ];

    // Map from [-1,1] to [0,1]: α = (ξ + 1) / 2, w_mapped = w / 2.
    let mut result = MatN::<N>::zero();
    let id = MatN::<N>::identity();

    for k in 0..7 {
        let alpha = ((GL_NODES_7[k] + 1.0) / 2.0) as f32;
        let weight = (GL_WEIGHTS_7[k] / 2.0) as f32;

        // Integrand: (I + α T)⁻¹
        let inner = matadd(&id, &matscale(t, alpha));
        if let Some(inner_inv) = invert_mat_n(&inner) {
            // Accumulate: result += weight * (I + α T)⁻¹
            for i in 0..N {
                for j in 0..N {
                    let cur = result.get(i, j);
                    result.set(i, j, cur + weight * inner_inv.get(i, j));
                }
            }
        }
    }

    // log(I + T) = T · result
    matmul(t, &result)
}

/// Compute a single-step matrix square root approximation via Padé(3).
///
/// Uses the Denman-Beavers iteration initialized with the Padé(3) rational
/// approximant for √(1 + t) − 1. For the matrix case we use one step of
/// the Schulz iteration: X₁ = X₀(3I - A·X₀²)/2, where X₀ = (A + I)/2.
fn mat_sqrt_pade<const N: usize>(a: &MatN<N>) -> MatN<N> {
    // Denman-Beavers coupled iteration: converges quadratically.
    // Y₀ = A, Z₀ = I
    // Yₖ₊₁ = ½(Yₖ + Zₖ⁻¹),  Zₖ₊₁ = ½(Zₖ + Yₖ⁻¹)
    // Y converges to A^{1/2}.
    let mut y = *a;
    let mut z = MatN::<N>::identity();
    const DB_ITERS: usize = 10;
    const DB_TOL: f32 = 1e-5;

    for _ in 0..DB_ITERS {
        let y_inv = match invert_mat_n(&y) {
            Some(inv) => inv,
            None => break,
        };
        let z_inv = match invert_mat_n(&z) {
            Some(inv) => inv,
            None => break,
        };
        let y_next = matscale(&matadd(&y, &z_inv), 0.5);
        let z_next = matscale(&matadd(&z, &y_inv), 0.5);
        let delta = mat_one_norm(&matsub(&y_next, &y));
        y = y_next;
        z = z_next;
        if delta < DB_TOL {
            break;
        }
    }
    y
}

/// Matrix exponential for symmetric positive definite (SPD) matrices.
///
/// Uses the spectral decomposition `A = V · Λ · Vᵀ` computed by `eig_sym_n`,
/// then returns `V · exp(Λ) · Vᵀ` where `exp(Λ)` is the diagonal matrix
/// of elementwise exponentials.
///
/// This is more accurate and faster than the general [`mat_exp`] for SPD inputs.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::MatN;
/// use ndarray::hpc::linalg::matfn::{mat_exp_spd, mat_log_spd};
///
/// // exp_spd(log_spd(M)) ≈ M for a random SPD 3×3.
/// let m: MatN<3> = MatN::from_array([
///     [3.0, 0.5, 0.1],
///     [0.5, 2.0, 0.2],
///     [0.1, 0.2, 1.5],
/// ]);
/// let l = mat_log_spd(&m);
/// let e = mat_exp_spd(&l);
/// for i in 0..3 {
///     for j in 0..3 {
///         assert!((e.get(i, j) - m.get(i, j)).abs() < 1e-4,
///             "exp(log(M))[{i}][{j}]: got {}, want {}", e.get(i,j), m.get(i,j));
///     }
/// }
/// ```
pub fn mat_exp_spd<const N: usize>(a: &MatN<N>) -> MatN<N> {
    let raw = to_raw(a);
    let (eigs, vcols) = eig_sym_n::<N>(&raw);
    spectral_apply(&eigs, &vcols, |lam| lam.exp())
}

/// Matrix logarithm for symmetric positive definite (SPD) matrices.
///
/// Uses the spectral decomposition `A = V · Λ · Vᵀ` computed by `eig_sym_n`,
/// then returns `V · log(Λ) · Vᵀ` where `log(Λ)` is the diagonal matrix
/// of elementwise natural logarithms.
///
/// **Precondition**: all eigenvalues of `A` must be strictly positive.
/// Negative or zero eigenvalues produce `NaN` / `-inf` entries silently.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::MatN;
/// use ndarray::hpc::linalg::matfn::mat_log_spd;
///
/// // log_spd(I) ≈ 0
/// let id: MatN<3> = MatN::identity();
/// let l = mat_log_spd(&id);
/// for i in 0..3 {
///     for j in 0..3 {
///         assert!(l.get(i, j).abs() < 1e-5, "log(I)[{i}][{j}]={}", l.get(i,j));
///     }
/// }
/// ```
pub fn mat_log_spd<const N: usize>(a: &MatN<N>) -> MatN<N> {
    let raw = to_raw(a);
    let (eigs, vcols) = eig_sym_n::<N>(&raw);
    spectral_apply(&eigs, &vcols, |lam| lam.ln())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq<const N: usize>(a: &MatN<N>, b: &MatN<N>, tol: f32) -> bool {
        for i in 0..N {
            for j in 0..N {
                if (a.get(i, j) - b.get(i, j)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// mat_exp(0) = I
    #[test]
    fn mat_exp_zero_is_identity() {
        let zero: MatN<3> = MatN::zero();
        let e = mat_exp(&zero);
        assert!(approx_eq(&e, &MatN::identity(), 1e-5), "exp(0) ≠ I: {:?}", e);
    }

    /// mat_exp(diag) = diag(exp(...)) element-wise.
    #[test]
    fn mat_exp_diagonal() {
        // Diagonal matrix with entries [0.1, 0.2, 0.3].
        let a: MatN<3> = MatN::from_array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]);
        let e = mat_exp(&a);
        for i in 0..3 {
            let expected = (0.1 * (i as f32 + 1.0)).exp();
            assert!((e.get(i, i) - expected).abs() < 1e-4, "exp(A)[{i}][{i}]: got {}, want {}", e.get(i, i), expected);
            for j in 0..3 {
                if i != j {
                    assert!(e.get(i, j).abs() < 1e-4, "exp(A)[{i}][{j}] off-diagonal: {}", e.get(i, j));
                }
            }
        }
    }

    /// mat_exp_spd(log_spd(M)) ≈ M for an SPD 3×3 matrix.
    #[test]
    fn mat_exp_spd_log_spd_round_trip() {
        let m: MatN<3> = MatN::from_array([[3.0, 0.5, 0.1], [0.5, 2.0, 0.2], [0.1, 0.2, 1.5]]);
        let l = mat_log_spd(&m);
        let e = mat_exp_spd(&l);
        assert!(approx_eq(&e, &m, 1e-4), "exp_spd(log_spd(M)) ≠ M: got {:?}", e);
    }

    /// mat_log_spd(I) ≈ 0
    #[test]
    fn mat_log_spd_identity_is_zero() {
        let id: MatN<3> = MatN::identity();
        let l = mat_log_spd(&id);
        for i in 0..3 {
            for j in 0..3 {
                assert!(l.get(i, j).abs() < 1e-5, "log_spd(I)[{i}][{j}] = {}", l.get(i, j));
            }
        }
    }

    /// mat_exp_spd(0) = I
    #[test]
    fn mat_exp_spd_zero_is_identity() {
        let zero: MatN<3> = MatN::zero();
        let e = mat_exp_spd(&zero);
        assert!(approx_eq(&e, &MatN::identity(), 1e-5), "exp_spd(0) ≠ I: {:?}", e);
    }

    /// mat_exp_spd(diag) = diag(exp(...)).
    #[test]
    fn mat_exp_spd_diagonal() {
        let a: MatN<3> = MatN::from_array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let e = mat_exp_spd(&a);
        let expected = [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
        for i in 0..3 {
            assert!(
                (e.get(i, i) - expected[i]).abs() < 1e-4,
                "exp_spd[{i}][{i}]: got {}, want {}",
                e.get(i, i),
                expected[i]
            );
        }
    }

    /// mat_log on a 2×2 diagonal matrix.
    #[test]
    fn mat_log_diagonal_2x2() {
        // A = diag(e^1, e^2), log(A) should be diag(1, 2).
        let a: MatN<2> =
            MatN::from_array([[std::f32::consts::E, 0.0], [0.0, std::f32::consts::E * std::f32::consts::E]]);
        let l = mat_log(&a);
        assert!((l.get(0, 0) - 1.0).abs() < 1e-4, "log[0][0]={} want 1.0", l.get(0, 0));
        assert!((l.get(1, 1) - 2.0).abs() < 1e-4, "log[1][1]={} want 2.0", l.get(1, 1));
    }
}
