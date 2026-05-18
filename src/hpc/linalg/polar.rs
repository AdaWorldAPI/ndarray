#![allow(missing_docs)]

//! Polar decomposition — A = U · P where U is orthogonal and P is SPD.
//!
//! # Algorithm
//!
//! Uses Newton iteration on the orthogonal factor:
//!
//! ```text
//!   U₀ = A
//!   Uₖ₊₁ = ½ (Uₖ + Uₖ⁻ᵀ)
//! ```
//!
//! which converges quadratically to the nearest orthogonal matrix when all
//! singular values of `A` are positive. Once `U` converges, `P = Uᵀ · A`.
//!
//! # References
//!
//! - Higham, N.J. (1986). "Computing the polar decomposition — with applications."
//!   *SIAM J. Sci. Stat. Comput.* **7**(4):1160–1174.

use super::{inverse::invert_mat_n, MatN};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a polar decomposition `A = U · P`.
///
/// - `u` — the orthogonal (rotation / improper rotation) factor.
/// - `p` — the symmetric positive semi-definite factor.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::{MatN, Mat3};
/// use ndarray::hpc::linalg::polar::polar;
///
/// // Polar decomposition of the identity is (I, I).
/// let id: Mat3 = MatN::identity();
/// let dec = polar(&id);
/// for i in 0..3 {
///     for j in 0..3 {
///         let exp = if i == j { 1.0 } else { 0.0 };
///         assert!((dec.u.get(i, j) - exp).abs() < 1e-5, "u[{i}][{j}]");
///         assert!((dec.p.get(i, j) - exp).abs() < 1e-5, "p[{i}][{j}]");
///     }
/// }
/// ```
pub struct Polar<const N: usize> {
    /// Orthogonal factor U (det = ±1).
    pub u: MatN<N>,
    /// Symmetric positive semi-definite factor P.
    pub p: MatN<N>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Transpose an N×N `MatN`.
#[inline]
fn transpose<const N: usize>(m: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| m.get(j, i))
}

/// Multiply two N×N `MatN` matrices.
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

/// Scale every element of `m` by `s`.
#[inline]
fn scale<const N: usize>(m: &MatN<N>, s: f32) -> MatN<N> {
    MatN::from_fn(|i, j| m.get(i, j) * s)
}

/// Element-wise sum of two N×N matrices.
#[inline]
fn add<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| a.get(i, j) + b.get(i, j))
}

/// Frobenius norm of (a - b).
#[inline]
fn frob_diff<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N {
        for j in 0..N {
            let d = a.get(i, j) - b.get(i, j);
            s += d * d;
        }
    }
    s.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the polar decomposition `A = U · P`.
///
/// Uses Newton's iteration to converge on the orthogonal factor `U`, then
/// recovers `P = Uᵀ · A`. Converges in at most 100 iterations (typically < 10).
///
/// For singular or nearly-singular `A`, convergence may be slow or imprecise;
/// the best available approximation is returned.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::{MatN, Mat3};
/// use ndarray::hpc::linalg::polar::polar;
///
/// // Polar of identity is (I, I).
/// let id: Mat3 = MatN::identity();
/// let dec = polar(&id);
/// for i in 0..3 {
///     for j in 0..3 {
///         let exp = if i == j { 1.0 } else { 0.0 };
///         assert!((dec.u.get(i, j) - exp).abs() < 1e-5);
///         assert!((dec.p.get(i, j) - exp).abs() < 1e-5);
///     }
/// }
/// ```
pub fn polar<const N: usize>(a: &MatN<N>) -> Polar<N> {
    // Newton iteration: Uₖ₊₁ = ½ (Uₖ + (Uₖ⁻¹)ᵀ)
    // Equivalent to: Uₖ₊₁ = ½ (Uₖ + Uₖ⁻ᵀ)
    let mut u = *a;
    const MAX_ITER: usize = 100;
    const TOL: f32 = 1e-6;

    for _ in 0..MAX_ITER {
        let u_inv = match invert_mat_n(&u) {
            Some(inv) => inv,
            None => break, // singular — return best approximation
        };
        let u_inv_t = transpose(&u_inv);
        let u_next = scale(&add(&u, &u_inv_t), 0.5);
        let delta = frob_diff(&u_next, &u);
        u = u_next;
        if delta < TOL {
            break;
        }
    }

    // P = Uᵀ · A  (symmetric positive semi-definite)
    let ut = transpose(&u);
    let p = matmul(&ut, a);

    // Symmetrize P to reduce floating-point asymmetry.
    let p_sym = MatN::from_fn(|i, j| (p.get(i, j) + p.get(j, i)) * 0.5);

    Polar { u, p: p_sym }
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

    /// polar(I) = (I, I)
    #[test]
    fn polar_identity_is_identity() {
        let id: MatN<3> = MatN::identity();
        let dec = polar(&id);
        assert!(approx_eq(&dec.u, &MatN::identity(), 1e-5), "U ≠ I: {:?}", dec.u);
        assert!(approx_eq(&dec.p, &MatN::identity(), 1e-5), "P ≠ I: {:?}", dec.p);
    }

    /// For an orthogonal matrix R, polar(R) = (R, I).
    #[test]
    fn polar_orthogonal_returns_r_and_identity() {
        // 90° rotation around Z axis.
        let r: MatN<3> = MatN::from_array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let dec = polar(&r);
        assert!(approx_eq(&dec.u, &r, 1e-5), "U ≠ R for orthogonal input: {:?}", dec.u);
        assert!(approx_eq(&dec.p, &MatN::identity(), 1e-4), "P ≠ I for orthogonal input: {:?}", dec.p);
    }

    /// Verify A = U · P reconstruction.
    #[test]
    fn polar_reconstruction() {
        // A simple SPD matrix scaled by 2.
        let a: MatN<3> = MatN::from_array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 1.5]]);
        let dec = polar(&a);
        let reconstructed = matmul(&dec.u, &dec.p);
        assert!(approx_eq(&reconstructed, &a, 1e-4), "U·P ≠ A: {:?}", reconstructed);
    }

    /// Polar of 2×2 identity.
    #[test]
    fn polar_identity_2x2() {
        let id: MatN<2> = MatN::identity();
        let dec = polar(&id);
        assert!(approx_eq(&dec.u, &MatN::identity(), 1e-5));
        assert!(approx_eq(&dec.p, &MatN::identity(), 1e-5));
    }
}
