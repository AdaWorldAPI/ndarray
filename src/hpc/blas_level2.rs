//! BLAS Level 2 operations: matrix-vector operations.
//!
//! Provides gemv, ger (rank-1 update), symv (symmetric matrix-vector),
//! trmv/trsv (triangular multiply/solve).

use crate::imp_prelude::*;
use crate::backend::BlasFloat;

/// Upper or lower triangle specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    /// Upper triangle.
    Upper,
    /// Lower triangle.
    Lower,
}

/// Diagonal specification for triangular operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    /// Non-unit diagonal (use actual diagonal values).
    NonUnit,
    /// Unit diagonal (assume diagonal is all ones).
    Unit,
}

/// BLAS Level 2 operations on 2-D arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::blas_level2::BlasLevel2;
///
/// let a = array![[1.0f32, 2.0], [3.0, 4.0]];
/// let x = array![1.0f32, 1.0];
/// let y = a.blas_gemv(1.0, &x, 0.0, &array![0.0f32, 0.0]);
/// assert!((y[0] - 3.0).abs() < 1e-5);
/// assert!((y[1] - 7.0).abs() < 1e-5);
/// ```
pub trait BlasLevel2<A> {
    /// General matrix-vector multiply: y = alpha * A * x + beta * y_init
    fn blas_gemv(
        &self,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;

    /// Rank-1 update: A = alpha * x * y^T + A (returns new array)
    fn blas_ger(
        &self,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        y: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2>;

    /// Symmetric matrix-vector multiply: y = alpha * A * x + beta * y_init
    ///
    /// Only reads the triangle specified by `uplo`.
    fn blas_symv(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;

    /// Triangular matrix-vector multiply: x = A * x
    fn blas_trmv(
        &self,
        uplo: Uplo,
        diag: Diag,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;

    /// Triangular solve: solve A * result = x for result
    fn blas_trsv(
        &self,
        uplo: Uplo,
        diag: Diag,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;

    /// Symmetric rank-1 update: A = alpha * x * x^T + A
    ///
    /// Only updates the triangle specified by `uplo`.
    fn blas_syr(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2>;

    /// Symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
    ///
    /// Only updates the triangle specified by `uplo`.
    fn blas_syr2(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        y: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2>;

    /// General banded matrix-vector multiply: y = alpha * A * x + beta * y_init
    ///
    /// `kl` is the number of sub-diagonals, `ku` is the number of super-diagonals.
    /// The matrix `A` is stored in band storage with `kl + ku + 1` rows and `n` columns.
    fn blas_gbmv(
        &self,
        m: usize,
        kl: usize,
        ku: usize,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;

    /// Symmetric banded matrix-vector multiply: y = alpha * A * x + beta * y_init
    ///
    /// `k` is the number of super-diagonals. The matrix is stored in band storage
    /// with `k + 1` rows and `n` columns. Only the triangle specified by `uplo` is read.
    fn blas_sbmv(
        &self,
        uplo: Uplo,
        k: usize,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1>;
}

impl<A, S> BlasLevel2<A> for ArrayBase<S, Ix2>
where
    A: BlasFloat + num_traits::Float + core::ops::AddAssign,
    S: Data<Elem = A>,
{
    fn blas_gemv(
        &self,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let (m, n) = (self.nrows(), self.ncols());
        assert_eq!(x.len(), n, "x length must equal number of columns");
        assert_eq!(y_init.len(), m, "y length must equal number of rows");

        // Try contiguous fast path
        if let (Some(a_slice), Some(x_slice)) = (self.as_slice(), x.as_slice()) {
            let mut y = y_init.to_owned();
            if let Some(y_slice) = y.as_slice_mut() {
                A::backend_gemv(m, n, alpha, a_slice, n, x_slice, beta, y_slice);
                return y;
            }
        }

        // Fallback
        let mut y = Array::zeros(m);
        for i in 0..m {
            let mut sum = A::zero();
            for j in 0..n {
                sum = sum + self[[i, j]] * x[j];
            }
            y[i] = alpha * sum + beta * y_init[i];
        }
        y
    }

    fn blas_ger(
        &self,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        y: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2> {
        let (m, n) = (self.nrows(), self.ncols());
        assert_eq!(x.len(), m, "x length must equal number of rows");
        assert_eq!(y.len(), n, "y length must equal number of columns");

        let mut result = self.to_owned();
        for i in 0..m {
            for j in 0..n {
                result[[i, j]] = result[[i, j]] + alpha * x[i] * y[j];
            }
        }
        result
    }

    fn blas_symv(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for symv");
        assert_eq!(x.len(), n);
        assert_eq!(y_init.len(), n);

        let mut y = Array::zeros(n);
        for i in 0..n {
            let mut sum = A::zero();
            for j in 0..n {
                let a_ij = match uplo {
                    Uplo::Upper => {
                        if j >= i { self[[i, j]] } else { self[[j, i]] }
                    }
                    Uplo::Lower => {
                        if j <= i { self[[i, j]] } else { self[[j, i]] }
                    }
                };
                sum = sum + a_ij * x[j];
            }
            y[i] = alpha * sum + beta * y_init[i];
        }
        y
    }

    fn blas_trmv(
        &self,
        uplo: Uplo,
        diag: Diag,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for trmv");
        assert_eq!(x.len(), n);

        let mut result = Array::zeros(n);
        match uplo {
            Uplo::Upper => {
                for i in 0..n {
                    let diag_val = match diag {
                        Diag::Unit => A::one(),
                        Diag::NonUnit => self[[i, i]],
                    };
                    result[i] = diag_val * x[i];
                    for j in (i + 1)..n {
                        result[i] = result[i] + self[[i, j]] * x[j];
                    }
                }
            }
            Uplo::Lower => {
                for i in 0..n {
                    for j in 0..i {
                        result[i] = result[i] + self[[i, j]] * x[j];
                    }
                    let diag_val = match diag {
                        Diag::Unit => A::one(),
                        Diag::NonUnit => self[[i, i]],
                    };
                    result[i] = result[i] + diag_val * x[i];
                }
            }
        }
        result
    }

    fn blas_trsv(
        &self,
        uplo: Uplo,
        diag: Diag,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for trsv");
        assert_eq!(x.len(), n);

        let mut result = x.to_owned();
        match uplo {
            Uplo::Lower => {
                for i in 0..n {
                    for j in 0..i {
                        result[i] = result[i] - self[[i, j]] * result[j];
                    }
                    if diag == Diag::NonUnit {
                        result[i] = result[i] / self[[i, i]];
                    }
                }
            }
            Uplo::Upper => {
                for i in (0..n).rev() {
                    for j in (i + 1)..n {
                        result[i] = result[i] - self[[i, j]] * result[j];
                    }
                    if diag == Diag::NonUnit {
                        result[i] = result[i] / self[[i, i]];
                    }
                }
            }
        }
        result
    }

    fn blas_syr(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for syr");
        assert_eq!(x.len(), n);

        let mut result = self.to_owned();
        for i in 0..n {
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                result[[i, j]] = result[[i, j]] + alpha * x[i] * x[j];
            }
        }
        result
    }

    fn blas_syr2(
        &self,
        uplo: Uplo,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        y: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix2> {
        let n = self.nrows();
        assert_eq!(self.ncols(), n, "Matrix must be square for syr2");
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        let mut result = self.to_owned();
        for i in 0..n {
            let (j_start, j_end) = match uplo {
                Uplo::Upper => (i, n),
                Uplo::Lower => (0, i + 1),
            };
            for j in j_start..j_end {
                result[[i, j]] = result[[i, j]] + alpha * (x[i] * y[j] + y[i] * x[j]);
            }
        }
        result
    }

    fn blas_gbmv(
        &self,
        m: usize,
        kl: usize,
        ku: usize,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let n = x.len();
        assert_eq!(y_init.len(), m);
        // self is the band storage matrix with shape (kl + ku + 1, n)
        assert_eq!(self.nrows(), kl + ku + 1, "Band matrix must have kl + ku + 1 rows");
        assert_eq!(self.ncols(), n, "Band matrix columns must equal n");

        let mut y = Array::zeros(m);
        for i in 0..m {
            let mut sum = A::zero();
            let j_start = if i > kl { i - kl } else { 0 };
            let j_end = core::cmp::min(i + ku + 1, n);
            for j in j_start..j_end {
                // In row-major band storage, element A(i,j) is stored at
                // band_row = ku + i - j, band_col = j
                let band_row = ku + i - j;
                sum = sum + self[[band_row, j]] * x[j];
            }
            y[i] = alpha * sum + beta * y_init[i];
        }
        y
    }

    fn blas_sbmv(
        &self,
        uplo: Uplo,
        k: usize,
        alpha: A,
        x: &ArrayBase<impl Data<Elem = A>, Ix1>,
        beta: A,
        y_init: &ArrayBase<impl Data<Elem = A>, Ix1>,
    ) -> Array<A, Ix1> {
        let n = x.len();
        assert_eq!(y_init.len(), n);
        assert_eq!(self.nrows(), k + 1, "Symmetric band matrix must have k + 1 rows");
        assert_eq!(self.ncols(), n, "Symmetric band matrix columns must equal n");

        let mut y = Array::zeros(n);
        for i in 0..n {
            let mut sum = A::zero();
            for j in 0..n {
                let diff = if i > j { i - j } else { j - i };
                if diff > k {
                    continue;
                }
                // Access band storage element
                let a_val = match uplo {
                    Uplo::Upper => {
                        if j >= i {
                            self[[k - (j - i), j]]
                        } else {
                            self[[k - (i - j), i]]
                        }
                    }
                    Uplo::Lower => {
                        if j <= i {
                            self[[j.abs_diff(i), j.max(i)]]
                        } else {
                            self[[i.abs_diff(j), i.max(j)]]
                        }
                    }
                };
                sum = sum + a_val * x[j];
            }
            y[i] = alpha * sum + beta * y_init[i];
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_gemv() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let x = array![5.0f64, 6.0];
        let y0 = array![0.0f64, 0.0];
        let y = a.blas_gemv(1.0, &x, 0.0, &y0);
        assert!((y[0] - 17.0).abs() < 1e-10);
        assert!((y[1] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_with_beta() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let x = array![1.0f32, 1.0];
        let y0 = array![10.0f32, 20.0];
        let y = a.blas_gemv(1.0, &x, 1.0, &y0);
        assert!((y[0] - 13.0).abs() < 1e-5); // 1+2+10
        assert!((y[1] - 27.0).abs() < 1e-5); // 3+4+20
    }

    #[test]
    fn test_ger() {
        let a = array![[0.0f64, 0.0], [0.0, 0.0]];
        let x = array![1.0f64, 2.0];
        let y = array![3.0f64, 4.0];
        let result = a.blas_ger(1.0, &x, &y);
        assert!((result[[0, 0]] - 3.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 6.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_symv() {
        // Symmetric matrix (upper stored)
        let a = array![[1.0f64, 2.0], [0.0, 3.0]];
        let x = array![1.0f64, 1.0];
        let y0 = array![0.0f64, 0.0];
        let y = a.blas_symv(Uplo::Upper, 1.0, &x, 0.0, &y0);
        // Full symmetric: [[1,2],[2,3]]
        assert!((y[0] - 3.0).abs() < 1e-10); // 1+2
        assert!((y[1] - 5.0).abs() < 1e-10); // 2+3
    }

    #[test]
    fn test_trsv_lower() {
        // L = [[2, 0], [1, 3]], solve L*x = [4, 7]
        let l = array![[2.0f64, 0.0], [1.0, 3.0]];
        let b = array![4.0f64, 7.0];
        let x = l.blas_trsv(Uplo::Lower, Diag::NonUnit, &b);
        // x[0] = 4/2 = 2, x[1] = (7 - 1*2)/3 = 5/3
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_upper() {
        let u = array![[2.0f64, 3.0], [0.0, 4.0]];
        let x = array![1.0f64, 2.0];
        let result = u.blas_trmv(Uplo::Upper, Diag::NonUnit, &x);
        assert!((result[0] - 8.0).abs() < 1e-10); // 2*1 + 3*2
        assert!((result[1] - 8.0).abs() < 1e-10); // 4*2
    }

    #[test]
    fn test_syr_upper() {
        let a = array![[1.0f64, 2.0], [0.0, 3.0]];
        let x = array![1.0f64, 2.0];
        let result = a.blas_syr(Uplo::Upper, 1.0, &x);
        // Upper: A[0,0] += 1*1=1 → 2, A[0,1] += 1*2=2 → 4, A[1,1] += 2*2=4 → 7
        assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 7.0).abs() < 1e-10);
        // Lower triangle unchanged
        assert!((result[[1, 0]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_syr2_upper() {
        let a = array![[0.0f64, 0.0], [0.0, 0.0]];
        let x = array![1.0f64, 2.0];
        let y = array![3.0f64, 4.0];
        let result = a.blas_syr2(Uplo::Upper, 1.0, &x, &y);
        // A[0,0] += x[0]*y[0] + y[0]*x[0] = 6
        // A[0,1] += x[0]*y[1] + y[0]*x[1] = 4+6 = 10
        // A[1,1] += x[1]*y[1] + y[1]*x[1] = 16
        assert!((result[[0, 0]] - 6.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 10.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_gbmv() {
        // Full matrix: [[1, 2, 0], [3, 4, 5], [0, 6, 7]]
        // kl=1, ku=1, band storage (3 rows x 3 cols):
        //   row 0 (super-diag): [*, 2, 5]
        //   row 1 (diagonal):   [1, 4, 7]
        //   row 2 (sub-diag):   [3, 6, *]
        let band = array![
            [0.0f64, 2.0, 5.0],
            [1.0, 4.0, 7.0],
            [3.0, 6.0, 0.0]
        ];
        let x = array![1.0f64, 2.0, 3.0];
        let y0 = array![0.0f64, 0.0, 0.0];
        let y = band.blas_gbmv(3, 1, 1, 1.0, &x, 0.0, &y0);
        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 3*1 + 4*2 + 5*3 = 26
        // y[2] = 6*2 + 7*3 = 33
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 26.0).abs() < 1e-10);
        assert!((y[2] - 33.0).abs() < 1e-10);
    }

    #[test]
    fn test_sbmv_upper() {
        // Symmetric tridiagonal: [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
        // k=1, upper band storage (2 rows x 3 cols):
        //   row 0 (super-diag): [*, 1, 1]
        //   row 1 (diagonal):   [2, 3, 4]
        let band = array![
            [0.0f64, 1.0, 1.0],
            [2.0, 3.0, 4.0]
        ];
        let x = array![1.0f64, 2.0, 3.0];
        let y0 = array![0.0f64, 0.0, 0.0];
        let y = band.blas_sbmv(Uplo::Upper, 1, 1.0, &x, 0.0, &y0);
        // y[0] = 2*1 + 1*2 = 4
        // y[1] = 1*1 + 3*2 + 1*3 = 10
        // y[2] = 1*2 + 4*3 = 14
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 10.0).abs() < 1e-10);
        assert!((y[2] - 14.0).abs() < 1e-10);
    }
}
