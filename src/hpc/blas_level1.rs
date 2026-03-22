//! BLAS Level 1 operations as extension traits on ndarray arrays.
//!
//! Provides vector-vector operations: dot, axpy, scal, nrm2, asum,
//! iamax, copy, swap, and element-wise scalar/vector arithmetic.

use crate::imp_prelude::*;
use crate::backend::BlasFloat;

/// BLAS Level 1 operations on 1-D arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::blas_level1::BlasLevel1;
///
/// let x = array![1.0f32, 2.0, 3.0];
/// let y = array![4.0f32, 5.0, 6.0];
/// assert!((x.blas_dot(&y) - 32.0).abs() < 1e-5);
/// ```
pub trait BlasLevel1<A> {
    /// Dot product: Σ x[i] * y[i]
    fn blas_dot(&self, other: &Self) -> A;

    /// AXPY: self = alpha * other + self (modifies self in-place)
    fn blas_axpy(&mut self, alpha: A, x: &Self);

    /// Scale: self = alpha * self
    fn blas_scal(&mut self, alpha: A);

    /// L2 norm: sqrt(Σ self[i]²)
    fn blas_nrm2(&self) -> A;

    /// L1 norm (absolute sum): Σ |self[i]|
    fn blas_asum(&self) -> A;

    /// Index of maximum absolute value element.
    fn blas_iamax(&self) -> usize;

    /// Copy elements from other into self.
    fn blas_copy_from(&mut self, other: &Self);

    /// Swap elements between self and other.
    fn blas_swap(&mut self, other: &mut Self);
}

impl<A, S> BlasLevel1<A> for ArrayBase<S, Ix1>
where
    A: BlasFloat + num_traits::Float,
    S: Data<Elem = A> + DataMut,
{
    fn blas_dot(&self, other: &Self) -> A {
        if let (Some(xs), Some(ys)) = (self.as_slice(), other.as_slice()) {
            A::backend_dot(xs, ys)
        } else {
            // Fallback for non-contiguous
            self.iter()
                .zip(other.iter())
                .fold(A::zero(), |acc, (&a, &b)| acc + a * b)
        }
    }

    fn blas_axpy(&mut self, alpha: A, x: &Self) {
        if let (Some(ys), Some(xs)) = (self.as_slice_mut(), x.as_slice()) {
            A::backend_axpy(alpha, xs, ys);
        } else {
            self.zip_mut_with(x, |y, &xv| *y = *y + alpha * xv);
        }
    }

    fn blas_scal(&mut self, alpha: A) {
        if let Some(xs) = self.as_slice_mut() {
            A::backend_scal(alpha, xs);
        } else {
            self.mapv_inplace(|v| v * alpha);
        }
    }

    fn blas_nrm2(&self) -> A {
        if let Some(xs) = self.as_slice() {
            A::backend_nrm2(xs)
        } else {
            self.iter().fold(A::zero(), |acc, &v| acc + v * v).sqrt()
        }
    }

    fn blas_asum(&self) -> A {
        if let Some(xs) = self.as_slice() {
            A::backend_asum(xs)
        } else {
            self.iter().fold(A::zero(), |acc, &v| acc + v.abs())
        }
    }

    fn blas_iamax(&self) -> usize {
        let mut max_idx = 0;
        let mut max_val = A::neg_infinity();
        for (i, &v) in self.iter().enumerate() {
            let abs_v = v.abs();
            if abs_v > max_val {
                max_val = abs_v;
                max_idx = i;
            }
        }
        max_idx
    }

    fn blas_copy_from(&mut self, other: &Self) {
        self.assign(other);
    }

    fn blas_swap(&mut self, other: &mut Self) {
        crate::Zip::from(self).and(other).for_each(|a, b| {
            core::mem::swap(a, b);
        });
    }
}

/// Givens rotation parameters.
///
/// Returned by [`blas_rotg`](GivensRotation::blas_rotg). Contains the cosine
/// and sine of the rotation that zeroes out the second component.
#[derive(Clone, Copy, Debug)]
pub struct GivensRotation<A> {
    /// The modified first element (r).
    pub r: A,
    /// Cosine of the rotation angle.
    pub c: A,
    /// Sine of the rotation angle.
    pub s: A,
}

/// Generate a Givens rotation.
///
/// Given scalars `a` and `b`, compute `r`, `c`, `s` such that:
///
/// ```text
/// [ c  s ] [ a ] = [ r ]
/// [-s  c ] [ b ]   [ 0 ]
/// ```
///
/// # Example
///
/// ```
/// use ndarray::hpc::blas_level1::blas_rotg;
///
/// let rot = blas_rotg(3.0f64, 4.0f64);
/// assert!((rot.r - 5.0).abs() < 1e-10);
/// assert!((rot.c - 0.6).abs() < 1e-10);
/// assert!((rot.s - 0.8).abs() < 1e-10);
/// ```
pub fn blas_rotg<A: num_traits::Float>(a: A, b: A) -> GivensRotation<A> {
    if a == A::zero() && b == A::zero() {
        return GivensRotation {
            r: A::zero(),
            c: A::one(),
            s: A::zero(),
        };
    }
    let scale = a.abs() + b.abs();
    let r = scale * ((a / scale).powi(2) + (b / scale).powi(2)).sqrt();
    // Sign of r follows the larger-magnitude input
    let r = if a.abs() > b.abs() {
        r.copysign(a)
    } else {
        r.copysign(b)
    };
    let c = a / r;
    let s = b / r;
    GivensRotation { r, c, s }
}

/// Element-wise scalar arithmetic operations.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::blas_level1::ScalarArith;
///
/// let x = array![1.0f32, 2.0, 3.0];
/// let y = x.add_scalar_elem(10.0);
/// assert_eq!(y, array![11.0, 12.0, 13.0]);
/// ```
pub trait ScalarArith<A> {
    /// Add a scalar to every element.
    fn add_scalar_elem(&self, scalar: A) -> Array<A, Ix1>;
    /// Subtract a scalar from every element.
    fn sub_scalar_elem(&self, scalar: A) -> Array<A, Ix1>;
    /// Multiply every element by a scalar.
    fn mul_scalar_elem(&self, scalar: A) -> Array<A, Ix1>;
    /// Divide every element by a scalar.
    fn div_scalar_elem(&self, scalar: A) -> Array<A, Ix1>;
}

impl<A, S> ScalarArith<A> for ArrayBase<S, Ix1>
where
    A: BlasFloat + num_traits::Float,
    S: Data<Elem = A>,
{
    fn add_scalar_elem(&self, scalar: A) -> Array<A, Ix1> {
        self.mapv(|v| v + scalar)
    }

    fn sub_scalar_elem(&self, scalar: A) -> Array<A, Ix1> {
        self.mapv(|v| v - scalar)
    }

    fn mul_scalar_elem(&self, scalar: A) -> Array<A, Ix1> {
        self.mapv(|v| v * scalar)
    }

    fn div_scalar_elem(&self, scalar: A) -> Array<A, Ix1> {
        self.mapv(|v| v / scalar)
    }
}

/// Element-wise vector arithmetic operations.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::blas_level1::VecArith;
///
/// let a = array![1.0f32, 2.0, 3.0];
/// let b = array![4.0f32, 5.0, 6.0];
/// let c = a.add_vec(&b);
/// assert_eq!(c, array![5.0, 7.0, 9.0]);
/// ```
pub trait VecArith<A> {
    /// Element-wise addition.
    fn add_vec(&self, other: &Self) -> Array<A, Ix1>;
    /// Element-wise subtraction.
    fn sub_vec(&self, other: &Self) -> Array<A, Ix1>;
    /// Element-wise multiplication.
    fn mul_vec(&self, other: &Self) -> Array<A, Ix1>;
    /// Element-wise division.
    fn div_vec(&self, other: &Self) -> Array<A, Ix1>;
}

impl<A, S> VecArith<A> for ArrayBase<S, Ix1>
where
    A: BlasFloat + num_traits::Float,
    S: Data<Elem = A>,
{
    fn add_vec(&self, other: &Self) -> Array<A, Ix1> {
        self + other
    }

    fn sub_vec(&self, other: &Self) -> Array<A, Ix1> {
        self - other
    }

    fn mul_vec(&self, other: &Self) -> Array<A, Ix1> {
        self * other
    }

    fn div_vec(&self, other: &Self) -> Array<A, Ix1> {
        self / other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_blas_dot() {
        let x = array![1.0f32, 2.0, 3.0, 4.0];
        let y = array![5.0f32, 6.0, 7.0, 8.0];
        assert!((x.blas_dot(&y) - 70.0).abs() < 1e-4);
    }

    #[test]
    fn test_blas_axpy() {
        let x = array![1.0f64, 2.0, 3.0];
        let mut y = array![4.0f64, 5.0, 6.0];
        y.blas_axpy(2.0, &x);
        assert_eq!(y, array![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_blas_scal() {
        let mut x = array![1.0f32, 2.0, 3.0];
        x.blas_scal(3.0);
        assert_eq!(x, array![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_blas_nrm2() {
        let x = array![3.0f64, 4.0];
        assert!((x.blas_nrm2() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_blas_asum() {
        let x = array![-1.0f32, 2.0, -3.0];
        assert!((x.blas_asum() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_blas_iamax() {
        let x = array![1.0f32, -5.0, 3.0, -2.0];
        assert_eq!(x.blas_iamax(), 1);
    }

    #[test]
    fn test_blas_swap() {
        let mut x = array![1.0f32, 2.0, 3.0];
        let mut y = array![4.0f32, 5.0, 6.0];
        x.blas_swap(&mut y);
        assert_eq!(x, array![4.0, 5.0, 6.0]);
        assert_eq!(y, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scalar_arith() {
        let x = array![1.0f32, 2.0, 3.0];
        assert_eq!(x.add_scalar_elem(10.0), array![11.0, 12.0, 13.0]);
        assert_eq!(x.mul_scalar_elem(2.0), array![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_blas_rotg() {
        let rot = super::blas_rotg(3.0f64, 4.0f64);
        assert!((rot.r - 5.0).abs() < 1e-10);
        assert!((rot.c - 0.6).abs() < 1e-10);
        assert!((rot.s - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_blas_rotg_zero() {
        let rot = super::blas_rotg(0.0f32, 0.0f32);
        assert_eq!(rot.r, 0.0);
        assert_eq!(rot.c, 1.0);
        assert_eq!(rot.s, 0.0);
    }

    #[test]
    fn test_vec_arith() {
        let a = array![1.0f64, 2.0, 3.0];
        let b = array![4.0f64, 5.0, 6.0];
        assert_eq!(a.add_vec(&b), array![5.0, 7.0, 9.0]);
        assert_eq!(a.mul_vec(&b), array![4.0, 10.0, 18.0]);
    }
}
