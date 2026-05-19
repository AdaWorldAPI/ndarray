//! Matrix inverse — 3×3 / 4×4 closed-form + general LU + affine specialization.
//!
//! # Algorithm summary
//!
//! | Function | Method | Ops | Precision |
//! |---|---|---|---|
//! | [`invert_mat3`] | Adjugate / determinant | ~30 | EXACT |
//! | [`invert_mat4`] | Cofactor expansion | ~70 | EXACT |
//! | [`invert_mat_n`] | Partial-pivot LU + back-solve | O(N³) | VERIFY |
//! | [`invert_affine_4x4`] | (R\|t) → (Rᵀ \| −Rᵀ·t) | ~40 | EXACT |
//!
//! **EXACT** means the computation is free-function algebra — no iterative
//! convergence, no stability parameter. **VERIFY** means results should be
//! validated against a reference when N > 4.
//!
//! # Singularity handling
//!
//! The closed-form and LU paths return `None` when the absolute value of the
//! determinant (or pivot) is ≤ `1e-7`. `invert_affine_4x4` is not guarded —
//! callers are responsible for ensuring the rotation block is orthogonal.

use super::{Mat3, Mat4, MatN};

// ───────────────────────────────────────────────────────────────────────────
// Threshold below which a determinant / pivot is considered zero.
// ───────────────────────────────────────────────────────────────────────────

const SINGULARITY_EPS: f32 = 1e-7;

// ───────────────────────────────────────────────────────────────────────────
// invert_mat3
// ───────────────────────────────────────────────────────────────────────────

/// Invert a 3×3 f32 matrix using the closed-form adjugate / determinant method.
///
/// Returns `None` when `|det(a)| ≤ 1e-7` (singular or near-singular).
/// The computation uses ~30 multiply-add operations with no branches inside
/// the hot path — suitable for per-frame camera and splat-projection work.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Mat3;
/// # use ndarray::hpc::linalg::inverse::invert_mat3;
/// let inv = invert_mat3(&Mat3::identity());
/// assert_eq!(inv, Some(Mat3::identity()));
/// ```
pub fn invert_mat3(a: &Mat3) -> Option<Mat3> {
    // Extract elements by row for clarity.
    let a00 = a.get(0, 0);
    let a01 = a.get(0, 1);
    let a02 = a.get(0, 2);
    let a10 = a.get(1, 0);
    let a11 = a.get(1, 1);
    let a12 = a.get(1, 2);
    let a20 = a.get(2, 0);
    let a21 = a.get(2, 1);
    let a22 = a.get(2, 2);

    // Cofactors (= transposed adjugate entries).
    let c00 = a11 * a22 - a12 * a21;
    let c01 = -(a10 * a22 - a12 * a20);
    let c02 = a10 * a21 - a11 * a20;

    let det = a00 * c00 + a01 * c01 + a02 * c02;

    if det.abs() <= SINGULARITY_EPS {
        return None;
    }

    let inv_det = 1.0 / det;

    let c10 = -(a01 * a22 - a02 * a21);
    let c11 = a00 * a22 - a02 * a20;
    let c12 = -(a00 * a21 - a01 * a20);

    let c20 = a01 * a12 - a02 * a11;
    let c21 = -(a00 * a12 - a02 * a10);
    let c22 = a00 * a11 - a01 * a10;

    // The inverse is (1/det) * adjugate = (1/det) * cofactor-matrix^T.
    // cofactor^T means row i of the inverse = column i of the cofactor matrix.
    Some(Mat3::from_array([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ]))
}

// ───────────────────────────────────────────────────────────────────────────
// invert_mat4
// ───────────────────────────────────────────────────────────────────────────

/// Invert a 4×4 f32 matrix using the closed-form cofactor expansion.
///
/// Returns `None` when `|det(a)| ≤ 1e-7`. Uses ~70 multiply-add operations
/// with no iterative step — suitable for view/projection matrices in tight
/// render loops.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Mat4;
/// # use ndarray::hpc::linalg::inverse::invert_mat4;
/// let inv = invert_mat4(&Mat4::identity());
/// assert_eq!(inv, Some(Mat4::identity()));
/// ```
pub fn invert_mat4(a: &Mat4) -> Option<Mat4> {
    // Flatten to named scalars for the cofactor expansion.
    let (m00, m01, m02, m03) = (a.get(0, 0), a.get(0, 1), a.get(0, 2), a.get(0, 3));
    let (m10, m11, m12, m13) = (a.get(1, 0), a.get(1, 1), a.get(1, 2), a.get(1, 3));
    let (m20, m21, m22, m23) = (a.get(2, 0), a.get(2, 1), a.get(2, 2), a.get(2, 3));
    let (m30, m31, m32, m33) = (a.get(3, 0), a.get(3, 1), a.get(3, 2), a.get(3, 3));

    // Pre-compute 2×2 sub-determinants used multiple times.
    let s0 = m00 * m11 - m10 * m01;
    let s1 = m00 * m12 - m10 * m02;
    let s2 = m00 * m13 - m10 * m03;
    let s3 = m01 * m12 - m11 * m02;
    let s4 = m01 * m13 - m11 * m03;
    let s5 = m02 * m13 - m12 * m03;

    let c5 = m22 * m33 - m32 * m23;
    let c4 = m21 * m33 - m31 * m23;
    let c3 = m21 * m32 - m31 * m22;
    let c2 = m20 * m33 - m30 * m23;
    let c1 = m20 * m32 - m30 * m22;
    let c0 = m20 * m31 - m30 * m21;

    // det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    if det.abs() <= SINGULARITY_EPS {
        return None;
    }

    let inv_det = 1.0 / det;

    // Build the 4×4 inverse row by row using cofactors.
    let r00 = (m11 * c5 - m12 * c4 + m13 * c3) * inv_det;
    let r01 = (-m01 * c5 + m02 * c4 - m03 * c3) * inv_det;
    let r02 = (m31 * s5 - m32 * s4 + m33 * s3) * inv_det;
    let r03 = (-m21 * s5 + m22 * s4 - m23 * s3) * inv_det;

    let r10 = (-m10 * c5 + m12 * c2 - m13 * c1) * inv_det;
    let r11 = (m00 * c5 - m02 * c2 + m03 * c1) * inv_det;
    let r12 = (-m30 * s5 + m32 * s2 - m33 * s1) * inv_det;
    let r13 = (m20 * s5 - m22 * s2 + m23 * s1) * inv_det;

    let r20 = (m10 * c4 - m11 * c2 + m13 * c0) * inv_det;
    let r21 = (-m00 * c4 + m01 * c2 - m03 * c0) * inv_det;
    let r22 = (m30 * s4 - m31 * s2 + m33 * s0) * inv_det;
    let r23 = (-m20 * s4 + m21 * s2 - m23 * s0) * inv_det;

    let r30 = (-m10 * c3 + m11 * c1 - m12 * c0) * inv_det;
    let r31 = (m00 * c3 - m01 * c1 + m02 * c0) * inv_det;
    let r32 = (-m30 * s3 + m31 * s1 - m32 * s0) * inv_det;
    let r33 = (m20 * s3 - m21 * s1 + m22 * s0) * inv_det;

    Some(Mat4::from_array([
        [r00, r01, r02, r03],
        [r10, r11, r12, r13],
        [r20, r21, r22, r23],
        [r30, r31, r32, r33],
    ]))
}

// ───────────────────────────────────────────────────────────────────────────
// invert_mat_n — general N×N via partial-pivot LU
// ───────────────────────────────────────────────────────────────────────────

/// Invert a general N×N f32 matrix using partial-pivoting LU decomposition.
///
/// Returns `None` when any pivot's absolute value is ≤ `1e-7` (singular or
/// near-singular). Time complexity is O(N³); allocation-free.
///
/// For N ≤ 4, prefer [`invert_mat3`] / [`invert_mat4`] which use closed-form
/// cofactor methods and are significantly faster.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::{MatN};
/// # use ndarray::hpc::linalg::inverse::invert_mat_n;
/// let inv = invert_mat_n(&MatN::<4>::identity());
/// assert_eq!(inv, Some(MatN::<4>::identity()));
/// ```
pub fn invert_mat_n<const N: usize>(a: &MatN<N>) -> Option<MatN<N>> {
    // We perform LU decomposition with partial pivoting on a copy of `a`,
    // simultaneously building the augmented identity for back-substitution.

    // lu[i][j] — the in-place LU matrix (starts as a copy of a).
    let mut lu = [[0.0f32; N]; N];
    for i in 0..N {
        for j in 0..N {
            lu[i][j] = a.get(i, j);
        }
    }

    // inv[i][j] — starts as the N×N identity, ends as the inverse.
    let mut inv = [[0.0f32; N]; N];
    for i in 0..N {
        inv[i][i] = 1.0;
    }

    // Pivot row tracking.
    let mut piv = [0usize; N];
    for i in 0..N {
        piv[i] = i;
    }

    for col in 0..N {
        // Find the row with the maximum absolute value in this column (≥ col).
        let mut max_abs = lu[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..N {
            let v = lu[row][col].abs();
            if v > max_abs {
                max_abs = v;
                max_row = row;
            }
        }

        if max_abs <= SINGULARITY_EPS {
            return None;
        }

        // Swap rows in both lu and inv if needed.
        if max_row != col {
            lu.swap(col, max_row);
            inv.swap(col, max_row);
        }

        let pivot = lu[col][col];
        let inv_pivot = 1.0 / pivot;

        // Eliminate below the pivot.
        for row in (col + 1)..N {
            let factor = lu[row][col] * inv_pivot;
            lu[row][col] = 0.0; // Explicitly zero the lower triangle.
            for k in (col + 1)..N {
                lu[row][k] -= factor * lu[col][k];
            }
            // Apply same row operation to the augmented identity.
            for k in 0..N {
                inv[row][k] -= factor * inv[col][k];
            }
        }
    }

    // Back-substitution: solve U·X = inv_so_far column by column.
    for col in 0..N {
        for row in (0..N).rev() {
            // Divide by the diagonal (upper triangular pivot).
            inv[row][col] /= lu[row][row];
            let val = inv[row][col];
            // Eliminate upward.
            for k in 0..row {
                inv[k][col] -= lu[k][row] * val;
            }
        }
    }

    Some(MatN::from_fn(|i, j| inv[i][j]))
}

// ───────────────────────────────────────────────────────────────────────────
// invert_affine_4x4
// ───────────────────────────────────────────────────────────────────────────

/// Invert an affine 4×4 view matrix of the form `(R | t; 0 0 0 1)`.
///
/// Exploits the block structure: the inverse is `(Rᵀ | −Rᵀ·t; 0 0 0 1)`.
/// This is ~40 operations vs the ~70 of the general cofactor path.
///
/// **Precondition**: the upper-left 3×3 block must be orthogonal (|det| = 1).
/// No singularity check is performed — callers must ensure the matrix is a
/// valid rigid-body transform.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Mat4;
/// # use ndarray::hpc::linalg::inverse::invert_affine_4x4;
/// let inv = invert_affine_4x4(&Mat4::identity());
/// assert_eq!(inv, Mat4::identity());
/// ```
pub fn invert_affine_4x4(view: &Mat4) -> Mat4 {
    // Extract the 3×3 rotation block R.
    let r00 = view.get(0, 0);
    let r01 = view.get(0, 1);
    let r02 = view.get(0, 2);
    let r10 = view.get(1, 0);
    let r11 = view.get(1, 1);
    let r12 = view.get(1, 2);
    let r20 = view.get(2, 0);
    let r21 = view.get(2, 1);
    let r22 = view.get(2, 2);

    // Extract translation t (last column, rows 0..2).
    let tx = view.get(0, 3);
    let ty = view.get(1, 3);
    let tz = view.get(2, 3);

    // Rᵀ is just the transpose of R.
    // −Rᵀ·t:
    let ntx = -(r00 * tx + r10 * ty + r20 * tz);
    let nty = -(r01 * tx + r11 * ty + r21 * tz);
    let ntz = -(r02 * tx + r12 * ty + r22 * tz);

    Mat4::from_array([[r00, r10, r20, ntx], [r01, r11, r21, nty], [r02, r12, r22, ntz], [0.0, 0.0, 0.0, 1.0]])
}

// ───────────────────────────────────────────────────────────────────────────
// Internal helper: N×N matrix multiply (for tests only)
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
fn matmul<const N: usize>(a: &MatN<N>, b: &MatN<N>) -> MatN<N> {
    MatN::from_fn(|i, j| {
        let mut sum = 0.0f32;
        for k in 0..N {
            sum += a.get(i, k) * b.get(k, j);
        }
        sum
    })
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// True if all elements of `a` and `b` differ by less than `eps`.
    fn mat_approx_eq<const N: usize>(a: &MatN<N>, b: &MatN<N>, eps: f32) -> bool {
        for i in 0..N {
            for j in 0..N {
                if (a.get(i, j) - b.get(i, j)).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    // ── invert_mat3: identity ─────────────────────────────────────────────

    #[test]
    fn invert_mat3_identity_is_identity() {
        let inv = invert_mat3(&Mat3::identity());
        assert_eq!(inv, Some(Mat3::identity()));
    }

    // ── invert_mat3: singular returns None ────────────────────────────────

    #[test]
    fn invert_mat3_singular_returns_none() {
        // Rows 1 and 2 are identical → rank-deficient.
        let singular = Mat3::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]);
        assert_eq!(invert_mat3(&singular), None);
    }

    // ── invert_mat4: identity ─────────────────────────────────────────────

    #[test]
    fn invert_mat4_identity_is_identity() {
        let inv = invert_mat4(&Mat4::identity());
        assert_eq!(inv, Some(Mat4::identity()));
    }

    // ── invert_mat3: M * inv(M) ≈ I for 10 random non-singular matrices ──

    #[test]
    fn invert_mat3_round_trip_random() {
        // 10 fixed pseudo-random non-singular 3×3 matrices.
        // Generated deterministically — no rand dependency required.
        let matrices: [[[f32; 3]; 3]; 10] = [
            [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]],
            [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
            [[3.0, 0.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0]],
            [[4.0, 7.0, 2.0], [1.0, 3.0, 1.0], [2.0, 5.0, 3.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
            [[5.0, 3.0, 1.0], [3.0, 5.0, 3.0], [1.0, 3.0, 5.0]],
            [[1.0, 2.0, 0.0], [3.0, 1.0, 2.0], [0.0, 2.0, 1.0]],
            [[6.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 7.0]],
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ];

        let identity = Mat3::identity();

        for (idx, arr) in matrices.iter().enumerate() {
            let m = Mat3::from_array(*arr);
            let inv = invert_mat3(&m).unwrap_or_else(|| panic!("Matrix {idx} should be invertible"));
            let product = matmul(&m, &inv);
            assert!(mat_approx_eq(&product, &identity, 1e-5), "M * inv(M) ≠ I for matrix {idx}: product = {product:?}");
        }
    }

    // ── invert_affine_4x4: round-trip test ────────────────────────────────

    #[test]
    fn invert_affine_4x4_round_trip() {
        // A simple camera view matrix: 90° rotation about Y-axis + translation.
        let view = Mat4::from_array([
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 1.0, 0.0, -2.0],
            [-1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let inv = invert_affine_4x4(&view);
        let product = matmul(&view, &inv);
        let identity = Mat4::identity();

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (product.get(i, j) - identity.get(i, j)).abs() < 1e-5,
                    "view * inv(view) ≠ I at [{i}][{j}]: got {}",
                    product.get(i, j)
                );
            }
        }
    }

    #[test]
    fn invert_affine_4x4_identity() {
        let inv = invert_affine_4x4(&Mat4::identity());
        assert_eq!(inv, Mat4::identity());
    }

    // ── invert_mat_n: identity ────────────────────────────────────────────

    #[test]
    fn invert_mat_n_identity_4x4() {
        let inv = invert_mat_n(&MatN::<4>::identity());
        assert_eq!(inv, Some(MatN::<4>::identity()));
    }

    // ── invert_mat_n: 5×5 known matrix ───────────────────────────────────

    #[test]
    fn invert_mat_n_5x5_round_trip() {
        // A known well-conditioned 5×5 matrix (diagonally dominant).
        let m = MatN::<5>::from_array([
            [10.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 10.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 10.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 10.0],
        ]);

        let inv = invert_mat_n(&m).expect("5×5 diagonally dominant matrix must be invertible");
        let product = matmul::<5>(&m, &inv);
        let identity = MatN::<5>::identity();

        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (product.get(i, j) - identity.get(i, j)).abs() < 1e-5,
                    "M * inv(M) ≠ I at [{i}][{j}] for 5×5: got {}",
                    product.get(i, j)
                );
            }
        }
    }

    // ── invert_mat_n: singular returns None ───────────────────────────────

    #[test]
    fn invert_mat_n_singular_returns_none() {
        // All-zero matrix is trivially singular.
        let z = MatN::<3>::zero();
        assert_eq!(invert_mat_n(&z), None);
    }

    // ── Cross-check: invert_mat3 vs invert_mat_n ─────────────────────────

    #[test]
    fn invert_mat3_agrees_with_invert_mat_n() {
        let m = Mat3::from_array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);

        let closed = invert_mat3(&m).expect("should be invertible");
        let lu = invert_mat_n(&m).expect("should be invertible (LU)");

        assert!(mat_approx_eq(&closed, &lu, 1e-5), "closed-form and LU disagree: closed = {closed:?}, lu = {lu:?}");
    }

    // ── Cross-check: invert_mat4 vs invert_mat_n ─────────────────────────

    #[test]
    fn invert_mat4_agrees_with_invert_mat_n() {
        let m =
            Mat4::from_array([[4.0, 3.0, 2.0, 1.0], [3.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 3.0], [1.0, 2.0, 3.0, 4.0]]);

        let closed = invert_mat4(&m).expect("should be invertible");
        let lu = invert_mat_n(&m).expect("should be invertible (LU)");

        assert!(
            mat_approx_eq(&closed, &lu, 1e-4),
            "closed-form and LU disagree:\nclosed = {closed:?}\nlu     = {lu:?}"
        );
    }
}
