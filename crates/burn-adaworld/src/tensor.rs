//! Tensor primitive: wraps ndarray::ArcArray for burn's Backend trait.

use ndarray::{ArcArray, IxDyn};
use std::sync::Arc;

/// The tensor primitive for the AdaWorld backend.
///
/// Wraps ndarray's `ArcArray<E, IxDyn>` with reference-counted shared ownership.
/// Zero-copy when possible (ArcArray uses copy-on-write).
#[derive(Debug, Clone)]
pub struct AdaTensor<E: Clone + 'static> {
    /// The underlying ndarray with dynamic dimensionality.
    pub array: ArcArray<E, IxDyn>,
}

impl<E: Clone + Default + 'static> AdaTensor<E> {
    /// Create from an owned ndarray.
    pub fn new(array: ndarray::Array<E, IxDyn>) -> Self {
        Self {
            array: array.into_shared(),
        }
    }

    /// Create from a shared ndarray (zero-copy).
    pub fn from_shared(array: ArcArray<E, IxDyn>) -> Self {
        Self { array }
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.array.ndim()
    }

    /// Get a contiguous slice of the data (if layout is standard).
    pub fn as_slice(&self) -> Option<&[E]> {
        self.array.as_slice()
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self
    where
        E: num_traits::Zero,
    {
        Self::new(ndarray::Array::zeros(IxDyn(shape)))
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self
    where
        E: num_traits::One,
    {
        Self::new(ndarray::Array::ones(IxDyn(shape)))
    }

    /// Reshape (zero-copy if contiguous).
    pub fn reshape(self, shape: &[usize]) -> Self {
        let array = self.array.into_owned();
        Self::new(array.into_shape_with_order(IxDyn(shape)).expect("reshape: incompatible shape"))
    }
}
