//! Hyperdimensional Computing (HDC) operations.
//!
//! Provides bind (XOR), permute (rotate), bundle (majority),
//! and int8 dot product for hypervectors represented as `Array<u8, Ix1>`.

use crate::imp_prelude::*;

/// HDC operations on binary hypervectors (u8 arrays).
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::hdc::HdcOps;
///
/// let a = array![0xFFu8, 0x00, 0xAA];
/// let b = array![0x0Fu8, 0xF0, 0x55];
/// let bound = a.hdc_bind(&b);
/// assert_eq!(bound, array![0xF0, 0xF0, 0xFF]);
/// ```
pub trait HdcOps {
    /// Bind two hypervectors via XOR.
    fn hdc_bind(&self, other: &Self) -> Array<u8, Ix1>;

    /// Permute (circular rotate) a hypervector by k bytes.
    fn hdc_permute(&self, k: usize) -> Array<u8, Ix1>;

    /// Bundle multiple hypervectors via element-wise majority vote.
    ///
    /// For each bit position, the output bit is 1 if more than half
    /// of the input vectors have a 1 at that position.
    fn hdc_bundle(vectors: &[&ArrayBase<impl Data<Elem = u8>, Ix1>]) -> Array<u8, Ix1>;

    /// Bundle from raw byte slices.
    fn hdc_bundle_byte_slices(slices: &[&[u8]]) -> Vec<u8>;

    /// Int8 dot product (treating u8 as i8).
    fn hdc_dot_i8(&self, other: &Self) -> i64;
}

impl<S> HdcOps for ArrayBase<S, Ix1>
where S: Data<Elem = u8>
{
    fn hdc_bind(&self, other: &Self) -> Array<u8, Ix1> {
        let n = self.len().min(other.len());
        let mut result = Array::zeros(n);
        for i in 0..n {
            result[i] = self[i] ^ other[i];
        }
        result
    }

    fn hdc_permute(&self, k: usize) -> Array<u8, Ix1> {
        let n = self.len();
        if n == 0 {
            return Array::zeros(0);
        }
        let k = k % n;
        let mut result = Array::zeros(n);
        for i in 0..n {
            result[i] = self[(i + n - k) % n];
        }
        result
    }

    fn hdc_bundle(vectors: &[&ArrayBase<impl Data<Elem = u8>, Ix1>]) -> Array<u8, Ix1> {
        if vectors.is_empty() {
            return Array::zeros(0);
        }
        let n = vectors[0].len();
        let threshold = vectors.len() / 2;
        let mut result = Array::zeros(n);

        for byte_idx in 0..n {
            let mut out_byte = 0u8;
            for bit in 0..8 {
                let mut count = 0usize;
                for v in vectors {
                    if byte_idx < v.len() && (v[byte_idx] >> bit) & 1 == 1 {
                        count += 1;
                    }
                }
                if count > threshold {
                    out_byte |= 1 << bit;
                }
            }
            result[byte_idx] = out_byte;
        }
        result
    }

    fn hdc_bundle_byte_slices(slices: &[&[u8]]) -> Vec<u8> {
        if slices.is_empty() {
            return Vec::new();
        }
        let n = slices[0].len();
        let threshold = slices.len() / 2;
        let mut result = vec![0u8; n];

        for byte_idx in 0..n {
            let mut out_byte = 0u8;
            for bit in 0..8 {
                let mut count = 0usize;
                for s in slices {
                    if byte_idx < s.len() && (s[byte_idx] >> bit) & 1 == 1 {
                        count += 1;
                    }
                }
                if count > threshold {
                    out_byte |= 1 << bit;
                }
            }
            result[byte_idx] = out_byte;
        }
        result
    }

    fn hdc_dot_i8(&self, other: &Self) -> i64 {
        let n = self.len().min(other.len());
        let mut sum = 0i64;
        for i in 0..n {
            sum += (self[i] as i8 as i64) * (other[i] as i8 as i64);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_bind() {
        let a = array![0xFFu8, 0x00, 0xAA];
        let b = array![0x0Fu8, 0xF0, 0x55];
        let result = a.hdc_bind(&b);
        assert_eq!(result, array![0xF0, 0xF0, 0xFF]);
    }

    #[test]
    fn test_bind_self_inverse() {
        let a = array![0xABu8, 0xCD, 0xEF];
        let b = array![0x12u8, 0x34, 0x56];
        let bound = a.hdc_bind(&b);
        let recovered = bound.hdc_bind(&b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_permute() {
        let a = array![1u8, 2, 3, 4];
        let p = a.hdc_permute(1);
        assert_eq!(p, array![4, 1, 2, 3]);
    }

    #[test]
    fn test_bundle() {
        // 3 vectors: majority vote
        let a = array![0xFFu8, 0x00];
        let b = array![0xFFu8, 0xFF];
        let c = array![0x00u8, 0xFF];
        let a_ref: &Array<u8, Ix1> = &a;
        let b_ref: &Array<u8, Ix1> = &b;
        let c_ref: &Array<u8, Ix1> = &c;
        let result = Array::<u8, Ix1>::hdc_bundle(&[a_ref, b_ref, c_ref]);
        assert_eq!(result, array![0xFF, 0xFF]); // majority: FF for both
    }

    #[test]
    fn test_dot_i8() {
        // 1, -1 as i8 = 0x01, 0xFF
        let a = array![1u8, 0xFFu8]; // i8: 1, -1
        let b = array![2u8, 0xFEu8]; // i8: 2, -2
        let result = a.hdc_dot_i8(&b);
        assert_eq!(result, 1 * 2 + (-1i64) * (-2i64)); // 2 + 2 = 4
    }
}
