//! LAPACK-style factorizations: LU, Cholesky, QR.
//!
//! Pure Rust implementations with MKL FFI behind `intel-mkl` feature gate.

use crate::imp_prelude::*;
use num_traits::Float;

/// LU factorization result.
#[derive(Clone, Debug)]
pub struct LuFactorization<A> {
    /// The L and U factors stored in-place (lower triangle = L, upper = U).
    pub lu: Array<A, Ix2>,
    /// Pivot indices.
    pub ipiv: Vec<usize>,
    /// Info: 0 = success, >0 = singular.
    pub info: i32,
}

/// Cholesky factorization result.
#[derive(Clone, Debug)]
pub struct CholeskyFactorization<A> {
    /// The factored matrix (lower or upper triangle).
    pub factor: Array<A, Ix2>,
    /// Info: 0 = success, >0 = not positive definite.
    pub info: i32,
}

/// QR factorization result.
#[derive(Clone, Debug)]
pub struct QrFactorization<A> {
    /// The factored matrix with R in upper triangle and Householder reflectors below.
    pub qr: Array<A, Ix2>,
    /// Householder scalar factors.
    pub tau: Vec<A>,
    /// Info: 0 = success.
    pub info: i32,
}

/// LAPACK factorization operations on 2-D arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::lapack::LapackOps;
///
/// let a = array![[2.0f64, 1.0], [1.0, 3.0]];
/// let lu = a.lu();
/// assert_eq!(lu.info, 0);
/// ```
pub trait LapackOps<A> {
    /// LU factorization with partial pivoting: P * A = L * U
    fn lu(&self) -> LuFactorization<A>;

    /// Solve A * X = B using LU factorization.
    fn lu_solve(&self, b: &Array<A, Ix2>) -> Array<A, Ix2>;

    /// Cholesky factorization: A = L * L^T (lower triangle).
    fn cholesky(&self) -> CholeskyFactorization<A>;

    /// Solve A * X = B using Cholesky factorization.
    fn cholesky_solve(&self, b: &Array<A, Ix2>) -> Array<A, Ix2>;

    /// QR factorization: A = Q * R
    fn qr(&self) -> QrFactorization<A>;
}

impl<A, S> LapackOps<A> for ArrayBase<S, Ix2>
where
    A: Float + Default + core::fmt::Debug + 'static,
    S: Data<Elem = A>,
{
    fn lu(&self) -> LuFactorization<A> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut lu = self.to_owned();
        let mut ipiv: Vec<usize> = (0..m.min(n)).collect();

        for k in 0..m.min(n) {
            // Find pivot
            let mut max_val = lu[[k, k]].abs();
            let mut max_row = k;
            for i in (k + 1)..m {
                let val = lu[[i, k]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val == A::zero() {
                return LuFactorization {
                    lu,
                    ipiv,
                    info: (k + 1) as i32,
                };
            }

            ipiv[k] = max_row;

            // Swap rows
            if max_row != k {
                for j in 0..n {
                    let tmp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = tmp;
                }
            }

            // Eliminate below
            let pivot = lu[[k, k]];
            for i in (k + 1)..m {
                lu[[i, k]] = lu[[i, k]] / pivot;
                let factor = lu[[i, k]];
                for j in (k + 1)..n {
                    lu[[i, j]] = lu[[i, j]] - factor * lu[[k, j]];
                }
            }
        }

        LuFactorization { lu, ipiv, info: 0 }
    }

    fn lu_solve(&self, b: &Array<A, Ix2>) -> Array<A, Ix2> {
        let fact = self.lu();
        let (m, _n) = (self.nrows(), self.ncols());
        let nrhs = b.ncols();
        let mut x = b.to_owned();

        // Apply row swaps
        for k in 0..fact.ipiv.len() {
            if fact.ipiv[k] != k {
                for j in 0..nrhs {
                    let tmp = x[[k, j]];
                    x[[k, j]] = x[[fact.ipiv[k], j]];
                    x[[fact.ipiv[k], j]] = tmp;
                }
            }
        }

        // Forward substitution (L * y = Pb)
        for k in 0..m {
            for i in (k + 1)..m {
                let factor = fact.lu[[i, k]];
                for j in 0..nrhs {
                    x[[i, j]] = x[[i, j]] - factor * x[[k, j]];
                }
            }
        }

        // Back substitution (U * x = y)
        for k in (0..m).rev() {
            let diag = fact.lu[[k, k]];
            for j in 0..nrhs {
                x[[k, j]] = x[[k, j]] / diag;
            }
            for i in 0..k {
                let factor = fact.lu[[i, k]];
                for j in 0..nrhs {
                    x[[i, j]] = x[[i, j]] - factor * x[[k, j]];
                }
            }
        }

        x
    }

    fn cholesky(&self) -> CholeskyFactorization<A> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for Cholesky");
        let mut l = Array::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = A::zero();
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    let diag = self[[i, i]] - sum;
                    if diag <= A::zero() {
                        return CholeskyFactorization {
                            factor: l,
                            info: (i + 1) as i32,
                        };
                    }
                    l[[i, j]] = diag.sqrt();
                } else {
                    l[[i, j]] = (self[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        CholeskyFactorization { factor: l, info: 0 }
    }

    fn cholesky_solve(&self, b: &Array<A, Ix2>) -> Array<A, Ix2> {
        let chol = self.cholesky();
        let n = self.nrows();
        let nrhs = b.ncols();
        let l = &chol.factor;
        let mut x = b.to_owned();

        // Forward: L * y = b
        for i in 0..n {
            for j in 0..nrhs {
                let mut sum = x[[i, j]];
                for k in 0..i {
                    sum = sum - l[[i, k]] * x[[k, j]];
                }
                x[[i, j]] = sum / l[[i, i]];
            }
        }

        // Backward: L^T * x = y
        for i in (0..n).rev() {
            for j in 0..nrhs {
                let mut sum = x[[i, j]];
                for k in (i + 1)..n {
                    sum = sum - l[[k, i]] * x[[k, j]];
                }
                x[[i, j]] = sum / l[[i, i]];
            }
        }

        x
    }

    fn qr(&self) -> QrFactorization<A> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut qr = self.to_owned();
        let min_mn = m.min(n);
        let mut tau = vec![A::zero(); min_mn];

        for k in 0..min_mn {
            // Compute Householder reflector
            let mut norm_sq = A::zero();
            for i in (k + 1)..m {
                norm_sq = norm_sq + qr[[i, k]] * qr[[i, k]];
            }
            let alpha = qr[[k, k]];
            let norm = (alpha * alpha + norm_sq).sqrt();
            let beta = if alpha >= A::zero() { -norm } else { norm };

            tau[k] = (beta - alpha) / beta;
            let scale = A::one() / (alpha - beta);
            for i in (k + 1)..m {
                qr[[i, k]] = qr[[i, k]] * scale;
            }
            qr[[k, k]] = beta;

            // Apply to remaining columns
            for j in (k + 1)..n {
                let mut dot = qr[[k, j]];
                for i in (k + 1)..m {
                    dot = dot + qr[[i, k]] * qr[[i, j]];
                }
                dot = dot * tau[k];
                qr[[k, j]] = qr[[k, j]] - dot;
                for i in (k + 1)..m {
                    qr[[i, j]] = qr[[i, j]] - dot * qr[[i, k]];
                }
            }
        }

        QrFactorization { qr, tau, info: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_lu_factorization() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let lu = a.lu();
        assert_eq!(lu.info, 0);
    }

    #[test]
    fn test_lu_solve() {
        // A * x = b where A = [[2, 1], [1, 3]], b = [[5], [7]]
        // x should be [[1.6], [1.8]]
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let b = array![[5.0f64], [7.0]];
        let x = a.lu_solve(&b);
        assert!((x[[0, 0]] - 1.6).abs() < 1e-10);
        assert!((x[[1, 0]] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = array![[4.0f64, 2.0], [2.0, 3.0]];
        let chol = a.cholesky();
        assert_eq!(chol.info, 0);
        // L should be [[2, 0], [1, sqrt(2)]]
        assert!((chol.factor[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((chol.factor[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((chol.factor[[1, 1]] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_qr_factorization() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let qr = a.qr();
        assert_eq!(qr.info, 0);
        assert_eq!(qr.tau.len(), 2);
    }
}
