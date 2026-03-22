//! Statistical operations: median, var, std, percentile.
//!
//! Extends ndarray's existing mean/sum with additional statistics
//! ported from rustynum.

use crate::imp_prelude::*;
use num_traits::{Float, FromPrimitive, Zero};
use core::ops::{Add, Div, Sub, Mul};

/// Statistical operations on arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::statistics::Statistics;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// assert!((x.median() - 3.0).abs() < 1e-10);
/// assert!((x.variance() - 2.0).abs() < 1e-10);
/// ```
pub trait Statistics<A> {
    /// Median value of the array.
    fn median(&self) -> A;

    /// Population variance: E[(X - μ)²]
    fn variance(&self) -> A;

    /// Variance along a given axis.
    fn var_axis(&self, axis: Axis) -> Array<A, IxDyn>;

    /// Population standard deviation: sqrt(variance)
    fn std_dev(&self) -> A;

    /// Standard deviation along a given axis.
    fn std_axis(&self, axis: Axis) -> Array<A, IxDyn>;

    /// Percentile (0-100).
    ///
    /// Uses linear interpolation between nearest ranks.
    fn percentile(&self, p: A) -> A;

    /// Sort elements (returns a new 1-D sorted array).
    fn sorted(&self) -> Array<A, Ix1>;

    /// Argmin: index of minimum element.
    fn argmin(&self) -> usize;

    /// Argmax: index of maximum element.
    fn argmax(&self) -> usize;

    /// Top-k: returns (indices, values) of the k largest elements.
    fn top_k(&self, k: usize) -> (Vec<usize>, Vec<A>);

    /// Cumulative sum along the flat array.
    fn cumsum(&self) -> Array<A, Ix1>;

    /// Cosine similarity between two arrays.
    fn cosine_similarity(&self, other: &Self) -> A;

    /// Generalized norm: ||x||_p
    fn norm(&self, p: u32) -> A;
}

impl<A, S, D> Statistics<A> for ArrayBase<S, D>
where
    A: Float + FromPrimitive + Zero + Add<Output = A> + Sub<Output = A>
        + Mul<Output = A> + Div<Output = A> + PartialOrd + 'static,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn median(&self) -> A {
        let mut sorted: Vec<A> = self.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return A::zero();
        }
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / A::from_usize(2).unwrap()
        } else {
            sorted[n / 2]
        }
    }

    fn variance(&self) -> A {
        let n = self.len();
        if n == 0 {
            return A::zero();
        }
        let n_a = A::from_usize(n).unwrap();
        let mean = self.iter().fold(A::zero(), |acc, &v| acc + v) / n_a;
        self.iter().fold(A::zero(), |acc, &v| {
            let diff = v - mean;
            acc + diff * diff
        }) / n_a
    }

    fn var_axis(&self, axis: Axis) -> Array<A, IxDyn> {
        let shape = self.raw_dim();
        let ax = axis.index();
        let ax_len = shape[ax];

        // Guard: zero-length axis would cause division by zero
        if ax_len == 0 {
            let mut out_shape: Vec<usize> = Vec::new();
            for (i, &s) in shape.slice().iter().enumerate() {
                if i != ax {
                    out_shape.push(s);
                }
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            let out_dim = IxDyn(&out_shape);
            let n_out: usize = out_shape.iter().product();
            return Array::from_shape_vec(out_dim, vec![A::zero(); n_out]).unwrap();
        }

        let n_a = A::from_usize(ax_len).unwrap();

        // Compute mean along axis
        let mut out_shape: Vec<usize> = Vec::new();
        for (i, &s) in shape.slice().iter().enumerate() {
            if i != ax {
                out_shape.push(s);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_dim = IxDyn(&out_shape);
        let n_out: usize = out_shape.iter().product();
        let mut means = vec![A::zero(); n_out];
        let mut vars = vec![A::zero(); n_out];

        // Compute means
        for (idx, lane) in self.lanes(axis).into_iter().enumerate() {
            let mean = lane.iter().fold(A::zero(), |acc, &v| acc + v) / n_a;
            means[idx] = mean;
        }
        // Compute variances
        for (idx, lane) in self.lanes(axis).into_iter().enumerate() {
            let mean = means[idx];
            let var = lane.iter().fold(A::zero(), |acc, &v| {
                let diff = v - mean;
                acc + diff * diff
            }) / n_a;
            vars[idx] = var;
        }

        Array::from_shape_vec(out_dim, vars).unwrap()
    }

    fn std_dev(&self) -> A {
        self.variance().sqrt()
    }

    fn std_axis(&self, axis: Axis) -> Array<A, IxDyn> {
        self.var_axis(axis).mapv(|v| v.sqrt())
    }

    fn percentile(&self, p: A) -> A {
        let mut sorted: Vec<A> = self.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return A::zero();
        }
        if n == 1 {
            return sorted[0];
        }
        let hundred = A::from_f64(100.0).unwrap();
        let rank = p / hundred * A::from_usize(n - 1).unwrap();
        let lo = rank.floor().to_usize().unwrap().min(n - 1);
        let hi = rank.ceil().to_usize().unwrap().min(n - 1);
        if lo == hi {
            sorted[lo]
        } else {
            let frac = rank - A::from_usize(lo).unwrap();
            sorted[lo] * (A::one() - frac) + sorted[hi] * frac
        }
    }

    fn sorted(&self) -> Array<A, Ix1> {
        let mut v: Vec<A> = self.iter().cloned().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        Array::from_vec(v)
    }

    fn argmin(&self) -> usize {
        let mut min_idx = 0;
        let mut min_val = A::infinity();
        for (i, &v) in self.iter().enumerate() {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        min_idx
    }

    fn argmax(&self) -> usize {
        let mut max_idx = 0;
        let mut max_val = A::neg_infinity();
        for (i, &v) in self.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        max_idx
    }

    fn top_k(&self, k: usize) -> (Vec<usize>, Vec<A>) {
        let mut indexed: Vec<(usize, A)> = self.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        let k = k.min(indexed.len());
        let indices: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
        let values: Vec<A> = indexed[..k].iter().map(|&(_, v)| v).collect();
        (indices, values)
    }

    fn cumsum(&self) -> Array<A, Ix1> {
        let mut result = Vec::with_capacity(self.len());
        let mut acc = A::zero();
        for &v in self.iter() {
            acc = acc + v;
            result.push(acc);
        }
        Array::from_vec(result)
    }

    fn cosine_similarity(&self, other: &Self) -> A {
        let dot: A = self.iter().zip(other.iter()).fold(A::zero(), |acc, (&a, &b)| acc + a * b);
        let norm_a: A = self.iter().fold(A::zero(), |acc, &v| acc + v * v).sqrt();
        let norm_b: A = other.iter().fold(A::zero(), |acc, &v| acc + v * v).sqrt();
        if norm_a == A::zero() || norm_b == A::zero() {
            A::zero()
        } else {
            dot / (norm_a * norm_b)
        }
    }

    fn norm(&self, p: u32) -> A {
        match p {
            0 => {
                // L0 "norm": count of non-zero elements
                A::from_usize(self.iter().filter(|&&v| v != A::zero()).count()).unwrap()
            }
            1 => {
                self.iter().fold(A::zero(), |acc, &v| acc + v.abs())
            }
            2 => {
                self.iter().fold(A::zero(), |acc, &v| acc + v * v).sqrt()
            }
            _ => {
                let p_f = A::from_u32(p).unwrap();
                let inv_p = A::one() / p_f;
                self.iter()
                    .fold(A::zero(), |acc, &v| acc + v.abs().powf(p_f))
                    .powf(inv_p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_median_odd() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0];
        assert!((x.median() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let x = array![1.0f64, 2.0, 3.0, 4.0];
        assert!((x.median() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let x = array![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = x.variance();
        assert!((var - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let x = array![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((x.std_dev() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert!((x.percentile(50.0) - 3.0).abs() < 1e-10);
        assert!((x.percentile(0.0) - 1.0).abs() < 1e-10);
        assert!((x.percentile(100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_argmin_argmax() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(x.argmin(), 1);
        assert_eq!(x.argmax(), 4);
    }

    #[test]
    fn test_top_k() {
        let x = array![1.0f64, 5.0, 3.0, 4.0, 2.0];
        let (indices, values) = x.top_k(3);
        assert_eq!(indices, vec![1, 3, 2]);
        assert_eq!(values, vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_cumsum() {
        let x = array![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(x.cumsum(), array![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0f64, 0.0, 0.0];
        let b = array![0.0f64, 1.0, 0.0];
        assert!((a.cosine_similarity(&b)).abs() < 1e-10); // orthogonal = 0
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-10); // same = 1
    }

    #[test]
    fn test_norm() {
        let x = array![3.0f64, 4.0];
        assert!((x.norm(2) - 5.0).abs() < 1e-10);
        assert!((x.norm(1) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn var_axis_zero_length_axis_no_nan() {
        use crate::Array2;
        // 0 rows, 3 columns — axis 0 has length 0
        let a = Array2::<f64>::zeros((0, 3));
        let dyn_a = a.into_dyn();
        let result = dyn_a.var_axis(Axis(0));
        assert_eq!(result.len(), 3);
        for &v in result.iter() {
            assert!(!v.is_nan(), "var_axis produced NaN on zero-length axis");
            assert_eq!(v, 0.0);
        }
    }
}
