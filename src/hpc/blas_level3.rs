//! BLAS Level 3 operations: matrix-matrix operations.
//!
//! Provides gemm, syrk (symmetric rank-k update), trsm (triangular solve),
//! symm (symmetric matrix multiply).

use crate::imp_prelude::*;
use crate::backend::BlasFloat;
use super::blas_level2::Uplo;

/// Side specification for operations like symm and trsm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// A is on the left: C = alpha * A * B + beta * C
    Left,
    /// A is on the right: C = alpha * B * A + beta * C
    Right,
}

/// BLAS Level 3 operations on 2-D arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::blas_level3::BlasLevel3;
///
/// let a = array![[1.0f32, 2.0], [3.0, 4.0]];
/// let b = array![[5.0f32, 6.0], [7.0, 8.0]];
/// let c = a.blas_gemm(1.0, &b, 0.0);
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-4);
/// ```
pub trait BlasLevel3<A> {
    /// General matrix multiply: result = alpha * self * B + beta * C_init
    ///
    /// If C_init is None, assumes zero initialization.
    fn blas_gemm(
        &self,
        alpha: A,
        b: &Self,
        beta: A,
    ) -> Array<A, Ix2>;

    /// General matrix multiply with explicit C: C = alpha * self * B + beta * C
    fn blas_gemm_into(
        &self,
        alpha: A,
        b: &Self,
        beta: A,
        c: &mut Array<A, Ix2>,
    );

    /// Symmetric rank-k update: C = alpha * A * A^T + beta * C_init
    fn blas_syrk(
        &self,
        uplo: Uplo,
        alpha: A,
        beta: A,
        c_init: Option<&Self>,
    ) -> Array<A, Ix2>;

    /// Symmetric matrix multiply: C = alpha * A * B + beta * C_init
    ///
    /// A is the symmetric matrix (specified by `side`).
    fn blas_symm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        b: &Self,
        beta: A,
        c_init: Option<&Self>,
    ) -> Array<A, Ix2>;

    /// Triangular matrix-matrix multiply: B = alpha * op(A) * B (Left)
    /// or B = alpha * B * op(A) (Right).
    ///
    /// `a` is the triangular matrix. Only the triangle specified by `uplo` is read.
    fn blas_trmm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        a_tri: &Self,
    ) -> Array<A, Ix2>;

    /// Triangular solve (matrix): solve A * X = alpha * B for X
    fn blas_trsm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        b: &Self,
    ) -> Array<A, Ix2>;
}

impl<A, S> BlasLevel3<A> for ArrayBase<S, Ix2>
where
    A: BlasFloat + num_traits::Float + core::ops::AddAssign,
    S: Data<Elem = A>,
{
    fn blas_gemm(
        &self,
        alpha: A,
        b: &Self,
        beta: A,
    ) -> Array<A, Ix2> {
        let (m, k) = (self.nrows(), self.ncols());
        let (k2, n) = (b.nrows(), b.ncols());
        assert_eq!(k, k2, "Inner dimensions must match for GEMM");

        let mut c = Array::zeros((m, n));

        if let (Some(a_s), Some(b_s), Some(c_s)) =
            (self.as_slice(), b.as_slice(), c.as_slice_mut())
        {
            A::backend_gemm(m, n, k, alpha, a_s, k, b_s, n, beta, c_s, n);
        } else {
            // Fallback for non-contiguous
            for i in 0..m {
                for j in 0..n {
                    let mut sum = A::zero();
                    for p in 0..k {
                        sum = sum + self[[i, p]] * b[[p, j]];
                    }
                    c[[i, j]] = alpha * sum;
                }
            }
        }
        c
    }

    fn blas_gemm_into(
        &self,
        alpha: A,
        b: &Self,
        beta: A,
        c: &mut Array<A, Ix2>,
    ) {
        let (m, k) = (self.nrows(), self.ncols());
        let (k2, n) = (b.nrows(), b.ncols());
        assert_eq!(k, k2, "Inner dimensions must match for GEMM");
        assert_eq!(c.nrows(), m);
        assert_eq!(c.ncols(), n);

        if let (Some(a_s), Some(b_s), Some(c_s)) =
            (self.as_slice(), b.as_slice(), c.as_slice_mut())
        {
            A::backend_gemm(m, n, k, alpha, a_s, k, b_s, n, beta, c_s, n);
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = A::zero();
                    for p in 0..k {
                        sum = sum + self[[i, p]] * b[[p, j]];
                    }
                    c[[i, j]] = alpha * sum + beta * c[[i, j]];
                }
            }
        }
    }

    fn blas_syrk(
        &self,
        uplo: Uplo,
        alpha: A,
        beta: A,
        c_init: Option<&Self>,
    ) -> Array<A, Ix2> {
        let (m, k) = (self.nrows(), self.ncols());
        let mut c = match c_init {
            Some(ci) => ci.to_owned(),
            None => Array::zeros((m, m)),
        };

        for i in 0..m {
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, m),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                let mut sum = A::zero();
                for p in 0..k {
                    sum = sum + self[[i, p]] * self[[j, p]];
                }
                c[[i, j]] = alpha * sum + beta * c[[i, j]];
            }
        }
        c
    }

    fn blas_symm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        b: &Self,
        beta: A,
        c_init: Option<&Self>,
    ) -> Array<A, Ix2> {
        let (m, n) = (b.nrows(), b.ncols());
        let mut c = match c_init {
            Some(ci) => ci.to_owned(),
            None => Array::zeros((m, n)),
        };

        let sym = self;
        let sym_n = sym.nrows();
        assert_eq!(sym.ncols(), sym_n, "Symmetric matrix must be square");

        for i in 0..m {
            for j in 0..n {
                let mut sum = A::zero();
                match side {
                    Side::Left => {
                        assert_eq!(sym_n, m);
                        for p in 0..m {
                            let a_val = match uplo {
                                Uplo::Upper => {
                                    if p >= i { sym[[i, p]] } else { sym[[p, i]] }
                                }
                                Uplo::Lower => {
                                    if p <= i { sym[[i, p]] } else { sym[[p, i]] }
                                }
                            };
                            sum = sum + a_val * b[[p, j]];
                        }
                    }
                    Side::Right => {
                        assert_eq!(sym_n, n);
                        for p in 0..n {
                            let a_val = match uplo {
                                Uplo::Upper => {
                                    if p >= j { sym[[j, p]] } else { sym[[p, j]] }
                                }
                                Uplo::Lower => {
                                    if p <= j { sym[[j, p]] } else { sym[[p, j]] }
                                }
                            };
                            sum = sum + b[[i, p]] * a_val;
                        }
                    }
                }
                c[[i, j]] = alpha * sum + beta * c[[i, j]];
            }
        }
        c
    }

    fn blas_trmm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        a_tri: &Self,
    ) -> Array<A, Ix2> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut result = Array::zeros((m, n));
        let b = self;

        match side {
            Side::Left => {
                // result = alpha * A * B
                let k = a_tri.nrows();
                assert_eq!(a_tri.ncols(), k, "Triangular matrix must be square");
                assert_eq!(k, m, "A rows must equal B rows for Left side");
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = A::zero();
                        match uplo {
                            Uplo::Upper => {
                                // A[i, p] is nonzero for p >= i
                                for p in i..m {
                                    sum = sum + a_tri[[i, p]] * b[[p, j]];
                                }
                            }
                            Uplo::Lower => {
                                // A[i, p] is nonzero for p <= i
                                for p in 0..=i {
                                    sum = sum + a_tri[[i, p]] * b[[p, j]];
                                }
                            }
                        }
                        result[[i, j]] = alpha * sum;
                    }
                }
            }
            Side::Right => {
                // result = alpha * B * A
                let k = a_tri.nrows();
                assert_eq!(a_tri.ncols(), k, "Triangular matrix must be square");
                assert_eq!(k, n, "A rows must equal B columns for Right side");
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = A::zero();
                        match uplo {
                            Uplo::Upper => {
                                // A[p, j] is nonzero for p <= j
                                for p in 0..=j {
                                    sum = sum + b[[i, p]] * a_tri[[p, j]];
                                }
                            }
                            Uplo::Lower => {
                                // A[p, j] is nonzero for p >= j
                                for p in j..n {
                                    sum = sum + b[[i, p]] * a_tri[[p, j]];
                                }
                            }
                        }
                        result[[i, j]] = alpha * sum;
                    }
                }
            }
        }
        result
    }

    fn blas_trsm(
        &self,
        side: Side,
        uplo: Uplo,
        alpha: A,
        b: &Self,
    ) -> Array<A, Ix2> {
        let (m, n) = (b.nrows(), b.ncols());
        let mut x = b.mapv(|v| alpha * v);
        let a = self;

        match side {
            Side::Left => {
                assert_eq!(a.nrows(), m);
                assert_eq!(a.ncols(), m);
                match uplo {
                    Uplo::Lower => {
                        for j in 0..n {
                            for i in 0..m {
                                for k in 0..i {
                                    x[[i, j]] = x[[i, j]] - a[[i, k]] * x[[k, j]];
                                }
                                x[[i, j]] = x[[i, j]] / a[[i, i]];
                            }
                        }
                    }
                    Uplo::Upper => {
                        for j in 0..n {
                            for i in (0..m).rev() {
                                for k in (i + 1)..m {
                                    x[[i, j]] = x[[i, j]] - a[[i, k]] * x[[k, j]];
                                }
                                x[[i, j]] = x[[i, j]] / a[[i, i]];
                            }
                        }
                    }
                }
            }
            Side::Right => {
                assert_eq!(a.nrows(), n);
                assert_eq!(a.ncols(), n);
                match uplo {
                    Uplo::Lower => {
                        for i in 0..m {
                            for j in (0..n).rev() {
                                for k in (j + 1)..n {
                                    x[[i, j]] = x[[i, j]] - x[[i, k]] * a[[k, j]];
                                }
                                x[[i, j]] = x[[i, j]] / a[[j, j]];
                            }
                        }
                    }
                    Uplo::Upper => {
                        for i in 0..m {
                            for j in 0..n {
                                for k in 0..j {
                                    x[[i, j]] = x[[i, j]] - x[[i, k]] * a[[k, j]];
                                }
                                x[[i, j]] = x[[i, j]] / a[[j, j]];
                            }
                        }
                    }
                }
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_gemm() {
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = a.blas_gemm(1.0, &b, 0.0);
        assert!((c[[0, 0]] - 58.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 64.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 139.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_syrk() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let c = a.blas_syrk(Uplo::Upper, 1.0, 0.0, None);
        // C = A * A^T
        // C[0,0] = 1*1+2*2 = 5
        // C[0,1] = 1*3+2*4 = 11
        // C[1,1] = 3*3+4*4 = 25
        assert!((c[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 11.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_left_upper() {
        // U = [[2, 3], [0, 4]], B = [[1, 2], [3, 4]]
        // result = 1.0 * U * B
        // row 0: [2*1+3*3, 2*2+3*4] = [11, 16]
        // row 1: [0*1+4*3, 0*2+4*4] = [12, 16]
        let b = array![[1.0f64, 2.0], [3.0, 4.0]];
        let u = array![[2.0f64, 3.0], [0.0, 4.0]];
        let result = b.blas_trmm(Side::Left, Uplo::Upper, 1.0, &u);
        assert!((result[[0, 0]] - 11.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 16.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 12.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmm_right_lower() {
        // L = [[2, 0], [1, 3]], B = [[1, 2], [3, 4]]
        // result = 1.0 * B * L
        // row 0: [1*2+2*1, 1*0+2*3] = [4, 6]
        // row 1: [3*2+4*1, 3*0+4*3] = [10, 12]
        let b = array![[1.0f64, 2.0], [3.0, 4.0]];
        let l = array![[2.0f64, 0.0], [1.0, 3.0]];
        let result = b.blas_trmm(Side::Right, Uplo::Lower, 1.0, &l);
        assert!((result[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 6.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 10.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_lower_left() {
        // Solve L * X = B where L = [[2, 0], [1, 3]], B = [[4, 6], [7, 9]]
        let l = array![[2.0f64, 0.0], [1.0, 3.0]];
        let b = array![[4.0f64, 6.0], [7.0, 9.0]];
        let x = l.blas_trsm(Side::Left, Uplo::Lower, 1.0, &b);
        // x[0,0] = 4/2 = 2, x[1,0] = (7 - 1*2)/3 = 5/3
        assert!((x[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((x[[1, 0]] - 5.0 / 3.0).abs() < 1e-10);
    }
}
