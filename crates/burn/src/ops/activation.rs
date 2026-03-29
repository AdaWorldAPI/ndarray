use crate::{
    NdArray, NdArrayStorage, NdArrayTensor, SharedArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
    execute_with_numeric_dtype,
    ops::NdArrayMathOps,
};
use burn_backend::{ElementConversion, TensorMetadata, ops::ActivationOps, tensor::FloatTensor};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ActivationOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_numeric_dtype!(tensor, |array| NdArrayMathOps::clamp_min(array, 0.elem()))
    }

    /// Sigmoid via ndarray::hpc::activations::sigmoid_f32 (fused F32x16 SIMD).
    ///
    /// Default impl decomposes into 6 separate ops: neg, exp, add, log, neg, exp.
    /// Our version does `1 / (1 + exp(-x))` in one SIMD pass with F32x16.
    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        if let NdArrayTensor::F32(ref storage) = tensor {
            let view = storage.view();
            if view.is_standard_layout() {
                if let Some(input) = view.as_slice() {
                    let mut output = alloc::vec![0.0f32; input.len()];
                    ndarray::hpc::activations::sigmoid_f32(input, &mut output);
                    let shape: alloc::vec::Vec<usize> = view.shape().to_vec();
                    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), output)
                        .expect("sigmoid output shape mismatch");
                    return NdArrayTensor::F32(NdArrayStorage::Owned(array.into_shared()));
                }
            }
        }
        // Fallback: decomposed sigmoid via Backend ops (non-f32 or non-contiguous).
        use burn_backend::ops::FloatTensorOps;
        let tensor_neg = Self::float_neg(tensor);
        let tensor_exp = Self::float_exp(tensor_neg);
        let tensor_add = Self::float_add_scalar(tensor_exp, 1.0.into());
        Self::float_recip(tensor_add)
    }
}
