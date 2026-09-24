//! Statistical operations: median, var, std, percentile.
//!
//! Extends ndarray's existing mean/sum with additional statistics
//! ported from rustynum.

use crate::imp_prelude::*;
use core::ops::{Add, Div, Mul, Sub};
use num_traits::{Float, FromPrimitive, Zero};

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
    A: Float
        + FromPrimitive
        + Zero
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + PartialOrd
        + 'static,
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
        let dot: A = self
            .iter()
            .zip(other.iter())
            .fold(A::zero(), |acc, (&a, &b)| acc + a * b);
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
            1 => self.iter().fold(A::zero(), |acc, &v| acc + v.abs()),
            2 => self.iter().fold(A::zero(), |acc, &v| acc + v * v).sqrt(),
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

// ── Batch moments for shard-parallel Welford ──────────────────────────────

/// Exact first and second moments of a `u32` sample: count, `Σx` and `Σx²`.
///
/// These three integers are the sufficient statistics of a Welford rolling
/// floor. Unlike a running `(mean, M2)` pair they merge by plain addition, so
/// [`MomentsU32::merge`] is exact, associative and commutative: shards of a
/// sample can be reduced in parallel, in any order, and combined into the
/// same result as one sequential pass. Floats appear only in [`mean`] and
/// [`variance`], at the end.
///
/// [`mean`]: MomentsU32::mean
/// [`variance`]: MomentsU32::variance
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MomentsU32 {
    /// Number of values.
    pub n: u64,
    /// `Σx`.
    pub sum: u128,
    /// `Σx²`.
    pub sum_sq: u128,
}

impl MomentsU32 {
    /// Moments of the union of two samples — exact integer addition.
    #[inline]
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            n: self.n + other.n,
            sum: self.sum + other.sum,
            sum_sq: self.sum_sq + other.sum_sq,
        }
    }

    /// Sample mean; `0.0` for an empty sample.
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum as f64 / self.n as f64
        }
    }

    /// Population variance `E[(X - μ)²]`; `0.0` for an empty sample.
    ///
    /// Computed as `(n·Σx² − (Σx)²) / n²`. The numerator is formed exactly in
    /// `u128` whenever it fits (always, for Hamming-scale data: n ≤ 2³², x ≤
    /// 2¹⁷), so there is no cancellation between two large floats; only the
    /// final division rounds. Past that range it centres the sums on the
    /// integer part of the mean in `u128` first, so the `f64` step still
    /// works on small, variance-sized quantities.
    pub fn variance(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let n = u128::from(self.n);
        match (n.checked_mul(self.sum_sq), self.sum.checked_mul(self.sum)) {
            (Some(a), Some(b)) => (a - b) as f64 / (self.n as f64 * self.n as f64),
            _ => {
                // Centre on the integer part of the mean, q = ⌊Σx / n⌋, with
                // remainder r = Σx − n·q < n. Then, exactly in u128,
                // Σ(x − q)² = Σx² − q·Σx − q·r, which is < n·2⁶⁴ and never
                // negative at any step. The true M2 is that minus r²/n, and
                // both terms are O(n·(σ² + 1)), so the one float subtraction
                // cannot cancel the variance away.
                let q = self.sum / n;
                let r = self.sum % n;
                let centred = self.sum_sq - q * self.sum - q * r;
                let rf = r as f64;
                let nf = self.n as f64;
                ((centred as f64 - rf * rf / nf) / nf).max(0.0)
            }
        }
    }
}

/// Exact [`MomentsU32`] of `values`, eight lanes at a time through `U64x8`.
///
/// Each value is widened to `u64` and its square split into 32-bit halves, so
/// every lane add is below 2³², and the lanes are drained into `u128` totals
/// every 2²⁸ chunks, before an 8-lane reduction could overflow `u64`. The
/// widening multiply is plain lane-wise Rust that the compiler vectorizes
/// (`vpmuludq` on x86); the accumulation and the final reduction go through
/// the polyfill, so every backend runs the same code.
///
/// # Example
///
/// ```
/// use ndarray::hpc::statistics::moments_u32;
///
/// let m = moments_u32(&[1, 2, 3, 4]);
/// assert_eq!((m.n, m.sum, m.sum_sq), (4, 10, 30));
/// assert_eq!(m.mean(), 2.5);
/// assert_eq!(m.variance(), 1.25);
/// ```
pub fn moments_u32(values: &[u32]) -> MomentsU32 {
    use crate::simd::U64x8;

    /// Chunks per drain. Every lane add is below 2^32, so after 2^28 chunks
    /// a lane is below 2^60 and the 8-lane `reduce_sum` below 2^63 — the
    /// reduction, not the lane, is the binding limit.
    const DRAIN: usize = 1 << 28;

    let (chunks, tail) = values.as_chunks::<8>();
    let mut out = MomentsU32 {
        n: values.len() as u64,
        ..MomentsU32::default()
    };
    for block in chunks.chunks(DRAIN) {
        let (mut s, mut lo, mut hi) = (U64x8::splat(0), U64x8::splat(0), U64x8::splat(0));
        for c in block {
            let x: [u64; 8] = core::array::from_fn(|i| u64::from(c[i]));
            let sq: [u64; 8] = core::array::from_fn(|i| x[i] * x[i]);
            s += U64x8::from_array(x);
            lo += U64x8::from_array(core::array::from_fn(|i| sq[i] & 0xFFFF_FFFF));
            hi += U64x8::from_array(core::array::from_fn(|i| sq[i] >> 32));
        }
        out.sum += u128::from(s.reduce_sum());
        out.sum_sq += u128::from(lo.reduce_sum()) + (u128::from(hi.reduce_sum()) << 32);
    }
    for &v in tail {
        out.sum += u128::from(v);
        out.sum_sq += u128::from(v) * u128::from(v);
    }
    out
}

#[cfg(test)]
mod moments_tests {
    use super::*;

    /// Past the exact-`u128` range the variance must not cancel: 2³³ values
    /// split evenly between `u32::MAX` and `u32::MAX - 1` have variance
    /// exactly 0.25, and `n·Σx²` overflows `u128`, so this takes the
    /// fallback path.
    #[test]
    fn variance_fallback_does_not_cancel() {
        let half = 1u128 << 32;
        let hi = u128::from(u32::MAX);
        let lo = hi - 1;
        let m = MomentsU32 {
            n: 1u64 << 33,
            sum: half * (hi + lo),
            sum_sq: half * (hi * hi + lo * lo),
        };
        assert!(u128::from(m.n).checked_mul(m.sum_sq).is_none(), "fixture must take the fallback");
        assert!((m.variance() - 0.25).abs() < 1e-9, "variance {}", m.variance());
    }

    fn xorshift(n: usize, mut s: u64, mask: u32) -> Vec<u32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as u32) & mask
            })
            .collect()
    }

    fn reference(x: &[u32]) -> MomentsU32 {
        MomentsU32 {
            n: x.len() as u64,
            sum: x.iter().map(|&v| u128::from(v)).sum(),
            sum_sq: x.iter().map(|&v| u128::from(v) * u128::from(v)).sum(),
        }
    }

    /// Exact against a u128 reference at every length across the 8-lane
    /// chunk boundary, for small (Hamming-scale) and full-range values.
    #[test]
    fn moments_u32_is_exact() {
        for mask in [0x3FFF, u32::MAX] {
            let x = xorshift(1000, 0x9E37_79B9_7F4A_7C15, mask);
            for n in (0..=40).chain([63, 64, 65, 999, 1000]) {
                assert_eq!(moments_u32(&x[..n]), reference(&x[..n]), "n={n} mask={mask:#x}");
            }
        }
    }

    /// Worst case for the square accumulator: every square is (2^32-1)^2,
    /// which alone nearly fills a u64. A lane that summed whole squares
    /// would overflow on the second element.
    #[test]
    fn moments_u32_does_not_overflow_at_u32_max() {
        let x = vec![u32::MAX; 100_003];
        let m = moments_u32(&x);
        let v = u128::from(u32::MAX);
        assert_eq!(m.n, 100_003);
        assert_eq!(m.sum, 100_003 * v);
        assert_eq!(m.sum_sq, 100_003 * v * v);
    }

    /// Merging is exact integer addition: the moments of a concatenation
    /// equal the merge of the parts' moments at any split point and in
    /// either order — which is what makes shard-parallel statistics exact.
    #[test]
    fn merge_is_exact_and_order_independent() {
        let x = xorshift(777, 42, u32::MAX);
        let whole = moments_u32(&x);
        for split in [0, 1, 7, 8, 9, 400, 776, 777] {
            let (a, b) = x.split_at(split);
            assert_eq!(moments_u32(a).merge(moments_u32(b)), whole, "split={split}");
            assert_eq!(moments_u32(b).merge(moments_u32(a)), whole, "reversed split={split}");
        }
    }

    /// Mean and population variance against a two-pass f64 reference.
    #[test]
    fn mean_and_variance_match_two_pass() {
        for mask in [0x3FFF, u32::MAX] {
            let x = xorshift(5000, 7, mask);
            let n = x.len() as f64;
            let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let var = x
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / n;
            let m = moments_u32(&x);
            assert!((m.mean() - mean).abs() <= 1e-9 * mean.abs(), "mean mask={mask:#x}");
            assert!((m.variance() - var).abs() <= 1e-9 * var, "variance mask={mask:#x}");
        }
        assert_eq!(MomentsU32::default().mean(), 0.0);
        assert_eq!(MomentsU32::default().variance(), 0.0);
        assert_eq!(moments_u32(&[5, 5, 5]).variance(), 0.0, "constant input has zero variance");
    }
}
