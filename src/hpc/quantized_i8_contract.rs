//! Independent, explicit-scale signed-i8 quantization contracts.
//!
//! These are *numeric* codecs, not Palette256 needle IDs. They do not read
//! codebooks or FamilyGamma and do not change the legacy quantized.rs API.
//!
//! The first follows the finite-f32, scalar-scale, signed-int8 portion of
//! ONNX QuantizeLinear: ties-to-even, optional signed zero point, [-128, 127].
//! The second uses the fully negatable [-127, 127] code set, with no marker
//! semantics assigned to -128. It is intentionally NOT ONNX QuantizeLinear.
//!
//! Both operate at an EXPLICIT caller-provided scale, so experiments can
//! compare range policy separately from scale calibration.

/// Errors for scalar-scale, finite-data signed-i8 quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I8QuantError {
    /// Scale must be finite and strictly positive.
    InvalidScale,
    /// The finite-input contract excludes NaN and infinity.
    NonFiniteInput { index: usize },
    /// Symmetric127 cannot decode the excluded signed code -128.
    OutOfRangeSymmetricCode { index: usize },
}

#[inline]
fn check_scale(scale: f32) -> Result<(), I8QuantError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(I8QuantError::InvalidScale)
    }
}

/// Quantize finite f32 inputs to signed i8 using ONNX's scalar-scale formula.
///
/// `q = clamp(round_ties_even(x / scale) + zero_point, -128, 127)`.
///
/// This covers the per-tensor signed-i8 numerical core of QuantizeLinear,
/// NOT per-axis/blocked broadcasting or other ONNX element types.
/// Invalid scales and non-finite inputs are rejected instead of choosing
/// undocumented NaN/Inf conversion semantics.
pub fn quantize_i8_onnx_per_tensor(
    values: &[f32],
    scale: f32,
    zero_point: i8,
) -> Result<Vec<i8>, I8QuantError> {
    check_scale(scale)?;
    let mut codes = Vec::with_capacity(values.len());
    for (index, &x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(I8QuantError::NonFiniteInput { index });
        }
        let rounded = (x / scale).round_ties_even();
        let q = (rounded + zero_point as f32).clamp(-128.0, 127.0);
        codes.push(q as i8);
    }
    Ok(codes)
}

/// Quantize finite f32 inputs to the sign-symmetric [-127, 127] code set.
///
/// `q = clamp(round_ties_even(x / scale), -127, 127)`.
/// There is no zero point and no reserved-code NaN or negative-zero policy.
/// In particular, this function never emits -128. The caller must not
/// relabel this as ONNX QuantizeLinear: ONNX permits -128.
pub fn quantize_i8_symmetric127_per_tensor(values: &[f32], scale: f32) -> Result<Vec<i8>, I8QuantError> {
    check_scale(scale)?;
    let mut codes = Vec::with_capacity(values.len());
    for (index, &x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(I8QuantError::NonFiniteInput { index });
        }
        codes.push((x / scale).round_ties_even().clamp(-127.0, 127.0) as i8);
    }
    Ok(codes)
}

/// Dequantize ONNX signed-i8 codes with explicit scalar scale and zero point.
///
/// `x = (q - zero_point) * scale`. Code -128 is a valid numeric value.
pub fn dequantize_i8_onnx_per_tensor(
    codes: &[i8],
    scale: f32,
    zero_point: i8,
) -> Result<Vec<f32>, I8QuantError> {
    check_scale(scale)?;
    Ok(codes.iter().map(|&q| (i32::from(q) - i32::from(zero_point)) as f32 * scale).collect())
}

/// Dequantize sign-symmetric codes; -128 is outside this code domain.
///
/// Excluding -128 does not imply a NaN, sentinel, or negative-zero meaning.
pub fn dequantize_i8_symmetric127_per_tensor(codes: &[i8], scale: f32) -> Result<Vec<f32>, I8QuantError> {
    check_scale(scale)?;
    let mut values = Vec::with_capacity(codes.len());
    for (index, &q) in codes.iter().enumerate() {
        if q == i8::MIN {
            return Err(I8QuantError::OutOfRangeSymmetricCode { index });
        }
        values.push(q as f32 * scale);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_256_i8_inputs_show_exactly_one_range_difference() {
        let inputs: Vec<f32> = (i8::MIN as i16..=i8::MAX as i16).map(|q| q as f32).collect();
        let onnx = quantize_i8_onnx_per_tensor(&inputs, 1.0, 0).unwrap();
        let sym = quantize_i8_symmetric127_per_tensor(&inputs, 1.0).unwrap();
        assert_eq!(onnx[0], -128);
        assert_eq!(sym[0], -127);
        assert!(sym.iter().all(|&q| q >= -127));
        assert_eq!(&onnx[1..], &sym[1..]);
        assert_eq!(dequantize_i8_onnx_per_tensor(&onnx, 1.0, 0).unwrap(), inputs);
        assert_eq!(
            dequantize_i8_symmetric127_per_tensor(&sym, 1.0).unwrap()[0],
            -127.0
        );
    }

    #[test]
    fn ties_to_even_is_not_rust_round_away_from_zero() {
        let values = [-3.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5];
        let expected = [-4, -2, -2, 0, 0, 0, 0, 2, 2, 4];
        assert_eq!(quantize_i8_onnx_per_tensor(&values, 1.0, 0).unwrap(), expected);
        assert_eq!(quantize_i8_symmetric127_per_tensor(&values, 1.0).unwrap(), expected);
    }

    #[test]
    fn negative_endpoint_and_shifted_zero_point_are_observable() {
        let inputs = [-1000.0, -128.0, -127.5, -127.0, 0.0, 127.0, 128.0, 1000.0];
        let o = quantize_i8_onnx_per_tensor(&inputs, 1.0, 0).unwrap();
        let s = quantize_i8_symmetric127_per_tensor(&inputs, 1.0).unwrap();
        assert_eq!(o, [-128, -128, -128, -127, 0, 127, 127, 127]);
        assert_eq!(s, [-127, -127, -127, -127, 0, 127, 127, 127]);
        assert_eq!(quantize_i8_onnx_per_tensor(&[-1000.0, -2.0, 0.0, 2.0, 1000.0], 2.0, 17).unwrap(), [-128, 16, 17, 18, 127]);
        assert_eq!(dequantize_i8_onnx_per_tensor(&[17, 18, 16], 2.0, 17).unwrap(), [0.0, 2.0, -2.0]);
    }

    #[test]
    fn symmetric_negation_is_exact_for_all_valid_codes() {
        let values: Vec<f32> = (-127i16..=127).map(|v| v as f32).collect();
        let codes = quantize_i8_symmetric127_per_tensor(&values, 1.0).unwrap();
        for &q in &codes {
            assert_eq!(quantize_i8_symmetric127_per_tensor(&[-(q as f32)], 1.0).unwrap(), [-q]);
        }
        assert_eq!(
            dequantize_i8_symmetric127_per_tensor(&[1, -128], 1.0),
            Err(I8QuantError::OutOfRangeSymmetricCode { index: 1 })
        );
    }

    #[test]
    fn identical_absmax_over_127_scale_gives_identical_codes() {
        let data = [-2.0, -1.5, -0.5, -0.0, 0.0, 0.25, 0.5, 1.5, 2.0];
        let scale = 2.0 / 127.0;
        assert_eq!(
            quantize_i8_onnx_per_tensor(&data, scale, 0),
            quantize_i8_symmetric127_per_tensor(&data, scale)
        );
    }

    #[test]
    fn invalid_and_non_finite_inputs_are_explicitly_rejected() {
        for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(quantize_i8_onnx_per_tensor(&[0.0], scale, 0), Err(I8QuantError::InvalidScale));
            assert_eq!(quantize_i8_symmetric127_per_tensor(&[0.0], scale), Err(I8QuantError::InvalidScale));
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                quantize_i8_onnx_per_tensor(&[1.0, bad], 1.0, 0),
                Err(I8QuantError::NonFiniteInput { index: 1 })
            );
            assert_eq!(
                quantize_i8_symmetric127_per_tensor(&[1.0, bad], 1.0),
                Err(I8QuantError::NonFiniteInput { index: 1 })
            );
        }
    }
}
