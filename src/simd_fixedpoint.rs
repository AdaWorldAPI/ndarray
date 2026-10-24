//! gemmlowp / TFLite-Micro fixed-point requantization primitives.
//!
//! Scalar reference semantics (bit-exact with the TFLM *double-rounding*
//! default, i.e. `TFLITE_SINGLE_ROUNDING` undefined) for the integer
//! requantization epilogue of quantized FC/Conv kernels, plus a batch
//! epilogue ([`requantize_i32_to_i8`]) that rides the `I32x16` typed-wrapper
//! methods (`saturating_rounding_doubling_high_mul`, `rounding_divide_by_pot`,
//! `multiply_by_quantized_multiplier`) of whichever backend `crate::simd`
//! selected.
//!
//! Ported from (tflite-micro `tensorflow/lite/micro/kernels/internal/`):
//!
//! | here | TFLM |
//! |---|---|
//! | [`srdhm_i32`], [`srdhm_i16`] | `fixedpoint.h:319-348` `SaturatingRoundingDoublingHighMul` |
//! | [`rounding_divide_by_pot_i32`] | `fixedpoint.h:352-366` `RoundingDivideByPOT` |
//! | [`multiply_by_quantized_multiplier_i32`] | `common.cc:85-92` (double-rounding branch) |
//! | [`multiply_by_quantized_multiplier_single_rounding_i32`] | `common.cc:30-45` |
//! | [`multiply_by_quantized_multiplier_i64`] | `common.cc:47-110` (identical in both modes) |
//! | [`quantize_multiplier`] | `quantization_util.cc:54-106` `QuantizeMultiplier` |
//! | [`requantize_i32_to_i8`] | `reference/integer_ops/fully_connected.h:60-80,165-180` epilogue |

/// `SaturatingRoundingDoublingHighMul` for `i32` (the NEON `VQRDMULH` semantics).
///
/// Computes `round(2·a·b / 2^32)` with the C++ truncating division after a
/// sign-dependent nudge; `i32::MIN · i32::MIN` saturates to `i32::MAX`.
///
/// ```
/// use ndarray::simd::srdhm_i32;
/// assert_eq!(srdhm_i32(i32::MIN, i32::MIN), i32::MAX);
/// assert_eq!(srdhm_i32(1 << 15, 1 << 15), 1); // +0.5 rounds up
/// assert_eq!(srdhm_i32(-(1 << 15), 1 << 15), 0); // -0.5 rounds toward +inf
/// ```
#[inline(always)]
pub fn srdhm_i32(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let p = a as i64 * b as i64;
    let nudge: i64 = if p >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    // `/` truncates toward zero (C++ semantics); it is NOT an arithmetic shift.
    ((p + nudge) / (1i64 << 31)) as i32
}

/// `SaturatingRoundingDoublingHighMul` for `i16` (scalar reference only).
///
/// ```
/// use ndarray::simd::srdhm_i16;
/// assert_eq!(srdhm_i16(i16::MIN, i16::MIN), i16::MAX);
/// assert_eq!(srdhm_i16(1 << 7, 1 << 7), 1);
/// ```
#[inline(always)]
pub fn srdhm_i16(a: i16, b: i16) -> i16 {
    if a == b && a == i16::MIN {
        return i16::MAX;
    }
    let p = a as i32 * b as i32;
    let nudge: i32 = if p >= 0 { 1 << 14 } else { 1 - (1 << 14) };
    ((p + nudge) / (1i32 << 15)) as i16
}

/// Correctly-rounded (ties away from zero) division by `2^e`, `e` in `0..=31`.
///
/// `e` outside `0..=31` is a contract violation (`debug_assert`); in release it
/// is clamped to 31 so the call is never UB.
///
/// ```
/// use ndarray::simd::rounding_divide_by_pot_i32;
/// assert_eq!(rounding_divide_by_pot_i32(1, 1), 1); // 0.5  -> 1
/// assert_eq!(rounding_divide_by_pot_i32(-1, 1), -1); // -0.5 -> -1
/// assert_eq!(rounding_divide_by_pot_i32(-3, 1), -2); // -1.5 -> -2
/// ```
#[inline(always)]
pub fn rounding_divide_by_pot_i32(x: i32, e: i32) -> i32 {
    debug_assert!((0..=31).contains(&e), "rounding_divide_by_pot exponent {e} outside 0..=31");
    let e = e.clamp(0, 31);
    let mask = ((1i64 << e) - 1) as i32;
    let rem = x & mask;
    let thr = (mask >> 1) + (x < 0) as i32;
    (x >> e) + (rem > thr) as i32
}

/// TFLM `MultiplyByQuantizedMultiplier(int32, int32, int)`, double-rounding.
///
/// Contract: `multiplier >= 0`, `-31 <= shift <= 30`, and `x << max(shift, 0)`
/// must not overflow. TFLM asserts the last point; this function **wraps**.
///
/// ```
/// use ndarray::simd::multiply_by_quantized_multiplier_i32;
/// // multiplier 0.5 in Q31 with shift 1 is a x1.0 scale.
/// assert_eq!(multiply_by_quantized_multiplier_i32(1000, 1 << 30, 1), 1000);
/// assert_eq!(multiply_by_quantized_multiplier_i32(1000, 1 << 30, -1), 250);
/// ```
#[inline(always)]
pub fn multiply_by_quantized_multiplier_i32(x: i32, multiplier: i32, shift: i32) -> i32 {
    debug_assert!(multiplier >= 0, "multiplier must be >= 0");
    debug_assert!((-31..=30).contains(&shift), "shift {shift} outside -31..=30");
    let left = shift.max(0) as u32;
    let right = (-shift).max(0);
    rounding_divide_by_pot_i32(srdhm_i32(x.wrapping_mul(1i32.wrapping_shl(left)), multiplier), right)
}

/// TFLM single-rounding `MultiplyByQuantizedMultiplier(int32, int32, int)`.
///
/// ```
/// use ndarray::simd::multiply_by_quantized_multiplier_single_rounding_i32 as f;
/// assert_eq!(f(1000, 1 << 30, 1), 1000);
/// assert_eq!(f(1, 1 << 30, 0), 1);
/// assert_eq!(f(-1, 1 << 30, 0), 0);
/// ```
#[inline(always)]
pub fn multiply_by_quantized_multiplier_single_rounding_i32(x: i32, multiplier: i32, shift: i32) -> i32 {
    debug_assert!(multiplier >= 0, "multiplier must be >= 0");
    debug_assert!((-31..=30).contains(&shift), "shift {shift} outside -31..=30");
    let total = (31 - shift).clamp(1, 62);
    let round = 1i64 << (total - 1);
    ((x as i64 * multiplier as i64 + round) >> total) as i32
}

/// TFLM `MultiplyByQuantizedMultiplier(int64, int32, int)` (same in both
/// rounding modes). Used by the int16-activation / int64-accumulator kernels.
///
/// Contract: `multiplier >= 0`, `-31 <= shift < 8`, `-(1<<47) <= x < (1<<47)`.
///
/// ```
/// use ndarray::simd::multiply_by_quantized_multiplier_i64;
/// assert_eq!(multiply_by_quantized_multiplier_i64(1 << 20, 1 << 30, 0), 1 << 19);
/// ```
#[inline(always)]
pub fn multiply_by_quantized_multiplier_i64(x: i64, multiplier: i32, shift: i32) -> i32 {
    debug_assert!(multiplier >= 0, "multiplier must be >= 0");
    debug_assert!((-31..8).contains(&shift), "shift {shift} outside -31..8");
    debug_assert!((-(1i64 << 47)..(1i64 << 47)).contains(&x), "x outside +-2^47");
    let reduced: i32 = if multiplier < 0x7FFF_0000 {
        (multiplier + (1 << 15)) >> 16
    } else {
        0x7FFF
    };
    let total = (15 - shift).clamp(1, 62);
    ((x.wrapping_mul(reduced as i64)
        .wrapping_add(1i64 << (total - 1)))
        >> total) as i32
}

/// `frexp` for finite non-zero `f64`: `(q, e)` with `m = q * 2^e`, `|q|` in `[0.5, 1)`.
#[cfg(feature = "std")]
fn frexp(m: f64) -> (f64, i32) {
    debug_assert!(m.is_finite() && m != 0.0);
    let mut bits = m.to_bits();
    let mut adjust = 0i32;
    if (bits >> 52) & 0x7FF == 0 {
        // Subnormal: pre-scale by 2^54 so the exponent field is populated.
        bits = (m * (1u64 << 54) as f64).to_bits();
        adjust = -54;
    }
    let exp_field = ((bits >> 52) & 0x7FF) as i32;
    let q = f64::from_bits((bits & !(0x7FFu64 << 52)) | (1022u64 << 52));
    (q, exp_field - 1022 + adjust)
}

#[cfg(feature = "std")]
fn quantize_multiplier_core(m: f64) -> (i64, i32) {
    assert!(m.is_finite(), "quantize_multiplier: non-finite multiplier");
    if m == 0.0 {
        return (0, 0);
    }
    let (q, mut shift) = frexp(m);
    // `round` is round-half-away-from-zero, the same as TfLiteRound / std::round.
    let mut q_fixed = (q * (1i64 << 31) as f64).round() as i64;
    assert!(q_fixed <= (1i64 << 31), "quantize_multiplier: q_fixed overflow");
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    assert!(q_fixed <= i32::MAX as i64);
    // A shift below -31 would shift every bit out; flush to zero as TFLM does.
    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }
    (q_fixed, shift)
}

/// TFLM `QuantizeMultiplier` (double-rounding build): split a real multiplier
/// into a Q31 `(multiplier, shift)` pair so `m ~= multiplier * 2^(shift - 31)`.
///
/// `0.0` maps to `(0, 0)`; magnitudes below about `2^-32` flush to `(0, 0)`.
///
/// ```
/// use ndarray::simd::quantize_multiplier;
/// assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
/// assert_eq!(quantize_multiplier(1.0), (1 << 30, 1));
/// assert_eq!(quantize_multiplier(0.0), (0, 0));
/// ```
#[cfg(feature = "std")]
pub fn quantize_multiplier(m: f64) -> (i32, i32) {
    let (q, shift) = quantize_multiplier_core(m);
    (q as i32, shift)
}

/// TFLM `QuantizeMultiplier` for the `TFLITE_SINGLE_ROUNDING` build: as
/// [`quantize_multiplier`], and additionally saturates `shift > 30` to
/// `(i32::MAX, 30)`.
///
/// ```
/// use ndarray::simd::quantize_multiplier_single_rounding;
/// assert_eq!(quantize_multiplier_single_rounding(2f64.powi(40)), (i32::MAX, 30));
/// ```
#[cfg(feature = "std")]
pub fn quantize_multiplier_single_rounding(m: f64) -> (i32, i32) {
    let (mut q, mut shift) = quantize_multiplier_core(m);
    if shift > 30 {
        shift = 30;
        q = (1i64 << 31) - 1;
    }
    (q as i32, shift)
}

/// Per-tensor / per-channel requantization epilogue `i32 -> i8`.
///
/// `channels = multiplier.len() = shift.len() >= 1`; element `i` uses channel
/// `i % channels` (per-tensor = one channel). For each element:
/// `v = multiply_by_quantized_multiplier(acc, m, s) + output_offset` (wrapping),
/// clamped to `[act_min, act_max]`, stored as `i8`.
///
/// Panics (never saturates silently) if the slice lengths are inconsistent,
/// `acc.len() % channels != 0`, or `act_min > act_max` / either is outside the
/// `i8` range.
///
/// ```
/// use ndarray::simd::requantize_i32_to_i8;
/// let acc = [100, -100, 0, 1_000_000];
/// let mut out = [0i8; 4];
/// requantize_i32_to_i8(&acc, &[1 << 30], &[0], 3, -128, 127, &mut out);
/// assert_eq!(out, [53, -47, 3, 127]);
/// ```
pub fn requantize_i32_to_i8(
    acc: &[i32], multiplier: &[i32], shift: &[i32], output_offset: i32, act_min: i32, act_max: i32, out: &mut [i8],
) {
    use crate::simd::I32x16;
    let channels = multiplier.len();
    assert!(channels >= 1 && shift.len() == channels, "multiplier/shift must have equal non-zero length");
    assert_eq!(acc.len(), out.len(), "acc/out length mismatch");
    assert!(acc.len().is_multiple_of(channels), "acc.len() must be a multiple of the channel count");
    assert!(
        act_min <= act_max && act_min >= i8::MIN as i32 && act_max <= i8::MAX as i32,
        "activation range must satisfy -128 <= act_min <= act_max <= 127"
    );

    // Channel pattern is identical for every 16-block when 16 % channels == 0.
    let invariant = 16 % channels == 0;
    let mut mv = [0i32; 16];
    let mut sv = [0i32; 16];
    for k in 0..16 {
        mv[k] = multiplier[k % channels];
        sv[k] = shift[k % channels];
    }
    let off = I32x16::splat(output_offset);
    let lo = I32x16::splat(act_min);
    let hi = I32x16::splat(act_max);

    let n16 = acc.len() / 16 * 16;
    let mut base = 0;
    while base < n16 {
        if !invariant {
            for k in 0..16 {
                mv[k] = multiplier[(base + k) % channels];
                sv[k] = shift[(base + k) % channels];
            }
        }
        let x = I32x16::from_array(core::array::from_fn(|k| acc[base + k]));
        let v = x.multiply_by_quantized_multiplier(I32x16::from_array(mv), I32x16::from_array(sv)) + off;
        let r = v.simd_max(lo).simd_min(hi).to_array();
        for k in 0..16 {
            out[base + k] = r[k] as i8;
        }
        base += 16;
    }
    for i in n16..acc.len() {
        let c = i % channels;
        let v = multiply_by_quantized_multiplier_i32(acc[i], multiplier[c], shift[c]).wrapping_add(output_offset);
        out[i] = v.max(act_min).min(act_max) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srdhm_min_min_saturates() {
        assert_eq!(srdhm_i32(i32::MIN, i32::MIN), i32::MAX);
        assert_eq!(srdhm_i16(i16::MIN, i16::MIN), i16::MAX);
        // MIN * MAX is not the saturating case.
        assert_eq!(srdhm_i32(i32::MIN, i32::MAX), -(i32::MAX));
        assert_eq!(srdhm_i32(i32::MAX, i32::MAX), 0x7FFF_FFFE);
    }

    #[test]
    fn srdhm_negative_ties_round_toward_positive_infinity() {
        // p = +-2^30 (+-0.5), +-3*2^30 (+-1.5)
        assert_eq!(srdhm_i32(1 << 15, 1 << 15), 1);
        assert_eq!(srdhm_i32(-(1 << 15), 1 << 15), 0);
        assert_eq!(srdhm_i32(3 << 15, 1 << 15), 2);
        assert_eq!(srdhm_i32(-(3 << 15), 1 << 15), -1);
        assert_eq!(srdhm_i16(1 << 7, 1 << 7), 1);
        assert_eq!(srdhm_i16(-(1 << 7), 1 << 7), 0);
    }

    #[test]
    fn rdpot_ties_both_signs_at_e1() {
        assert_eq!(rounding_divide_by_pot_i32(1, 1), 1);
        assert_eq!(rounding_divide_by_pot_i32(-1, 1), -1);
        assert_eq!(rounding_divide_by_pot_i32(3, 1), 2);
        assert_eq!(rounding_divide_by_pot_i32(-3, 1), -2);
        assert_eq!(rounding_divide_by_pot_i32(0, 1), 0);
    }

    #[test]
    fn rdpot_ties_both_signs_at_e31() {
        assert_eq!(rounding_divide_by_pot_i32(1 << 30, 31), 1);
        assert_eq!(rounding_divide_by_pot_i32(-(1 << 30), 31), -1);
        assert_eq!(rounding_divide_by_pot_i32(i32::MAX, 31), 1);
        assert_eq!(rounding_divide_by_pot_i32(i32::MIN, 31), -1);
        assert_eq!(rounding_divide_by_pot_i32((1 << 30) - 1, 31), 0);
        assert_eq!(rounding_divide_by_pot_i32(-(1 << 30) + 1, 31), 0);
    }

    #[test]
    fn rdpot_e0_is_identity() {
        for x in [i32::MIN, -7, 0, 7, i32::MAX] {
            assert_eq!(rounding_divide_by_pot_i32(x, 0), x);
        }
    }

    #[test]
    fn mbqm_basics_and_rounding_modes() {
        assert_eq!(multiply_by_quantized_multiplier_i32(1000, 1 << 30, 1), 1000);
        assert_eq!(multiply_by_quantized_multiplier_i32(1000, 1 << 30, -1), 250);
        assert_eq!(multiply_by_quantized_multiplier_i32(1, 1 << 30, 0), 1);
        assert_eq!(multiply_by_quantized_multiplier_i32(-1, 1 << 30, 0), 0);
        assert_eq!(multiply_by_quantized_multiplier_single_rounding_i32(1000, 1 << 30, 1), 1000);
        assert_eq!(multiply_by_quantized_multiplier_single_rounding_i32(1000, 1 << 30, -1), 250);
        assert_eq!(multiply_by_quantized_multiplier_single_rounding_i32(1, 1 << 30, 0), 1);
        assert_eq!(multiply_by_quantized_multiplier_single_rounding_i32(-1, 1 << 30, 0), 0);
        assert_eq!(multiply_by_quantized_multiplier_i64(1 << 20, 1 << 30, 0), 1 << 19);
    }

    #[cfg(feature = "std")]
    fn check_reconstruction(m: f64) {
        let (q, s) = quantize_multiplier(m);
        assert!(q >= 1 << 30, "q {q} not normalised for {m}");
        let back = q as f64 / (1i64 << 31) as f64 * 2f64.powi(s);
        assert!((back - m).abs() <= m.abs() * 2f64.powi(-30), "m={m} q={q} s={s} back={back}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn quantize_multiplier_known_values() {
        assert_eq!(quantize_multiplier(0.0), (0, 0));
        assert_eq!(quantize_multiplier(1.0), (1 << 30, 1));
        assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
        assert_eq!(quantize_multiplier(2f64.powi(-32)), (1 << 30, -31));
        // Below the representable range: flushed to zero.
        assert_eq!(quantize_multiplier(2f64.powi(-33)), (0, 0));
        // Subnormal input flushes the same way (and must not panic).
        assert_eq!(quantize_multiplier(f64::MIN_POSITIVE / 4.0), (0, 0));
        // 1e-10 has frexp exponent -33 < -31, so TFLM flushes it to zero
        // (the golden fixture records `Q 1e-10 0 0`).
        assert_eq!(quantize_multiplier(1e-10), (0, 0));
        // Exact value from the TFLM golden fixture.
        assert_eq!(quantize_multiplier(3.7), (1_986_422_374, 2));
        check_reconstruction(3.7);
        check_reconstruction(0.999_999_999_9);
        // Rounding up to 2^31 must renormalise: q_fixed = 2^31 -> 2^30, shift + 1.
        let just_below_one = 1.0 - 2f64.powi(-53);
        assert_eq!(quantize_multiplier(just_below_one), (1 << 30, 1));
    }

    #[cfg(feature = "std")]
    #[test]
    fn quantize_multiplier_single_rounding_saturates_high_shift() {
        assert_eq!(quantize_multiplier_single_rounding(1.0), (1 << 30, 1));
        assert_eq!(quantize_multiplier_single_rounding(2f64.powi(40)), (i32::MAX, 30));
        assert_eq!(quantize_multiplier_single_rounding(0.0), (0, 0));
        // The double-rounding variant keeps the large shift.
        assert_eq!(quantize_multiplier(2f64.powi(40)), (1 << 30, 41));
    }

    fn reference_epilogue(acc: &[i32], m: &[i32], s: &[i32], off: i32, lo: i32, hi: i32) -> Vec<i8> {
        acc.iter()
            .enumerate()
            .map(|(i, &a)| {
                let c = i % m.len();
                multiply_by_quantized_multiplier_i32(a, m[c], s[c])
                    .wrapping_add(off)
                    .max(lo)
                    .min(hi) as i8
            })
            .collect()
    }

    fn pseudo(n: usize, seed: u32) -> Vec<i32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                (st as i32) >> 12 // up to +-2^19 so left shifts cannot overflow
            })
            .collect()
    }

    #[test]
    fn requantize_per_tensor_with_tail() {
        for len in [0usize, 1, 15, 16, 17, 37, 64, 100] {
            let acc = pseudo(len, 7 + len as u32);
            let mut out = vec![0i8; len];
            requantize_i32_to_i8(&acc, &[1_518_500_250], &[-3], 5, -100, 100, &mut out);
            assert_eq!(out, reference_epilogue(&acc, &[1_518_500_250], &[-3], 5, -100, 100), "len {len}");
        }
        // Hand-checked values.
        let mut out = [0i8; 4];
        requantize_i32_to_i8(&[100, -100, 0, 1_000_000], &[1 << 30], &[0], 3, -128, 127, &mut out);
        assert_eq!(out, [53, -47, 3, 127]);
    }

    #[test]
    fn requantize_per_channel_with_tail() {
        // 3 channels (does not divide 16), 39 elements: two full blocks + 7-element tail.
        let m3 = [1 << 30, 1_518_500_250, 2_000_000_000];
        let s3 = [0, -2, -5];
        let acc = pseudo(39, 11);
        let mut out = vec![0i8; 39];
        requantize_i32_to_i8(&acc, &m3, &s3, -7, -128, 127, &mut out);
        assert_eq!(out, reference_epilogue(&acc, &m3, &s3, -7, -128, 127));
        // 4 channels (divides 16), 36 elements: two blocks + 4 tail.
        let m4 = [1 << 30, 1_100_000_000, 1_900_000_000, 1_300_000_000];
        let s4 = [1, -1, -4, 0];
        let acc = pseudo(36, 13);
        let mut out = vec![0i8; 36];
        requantize_i32_to_i8(&acc, &m4, &s4, 2, -50, 60, &mut out);
        assert_eq!(out, reference_epilogue(&acc, &m4, &s4, 2, -50, 60));
        // Clamping is exercised, not vacuous.
        assert!(out.contains(&-50) || out.contains(&60));
    }

    #[test]
    #[should_panic]
    fn requantize_rejects_bad_activation_range() {
        let mut out = [0i8; 1];
        requantize_i32_to_i8(&[1], &[1 << 30], &[0], 0, -200, 127, &mut out);
    }

    #[test]
    #[should_panic]
    fn requantize_rejects_ragged_channels() {
        let mut out = [0i8; 5];
        requantize_i32_to_i8(&[1; 5], &[1 << 30, 1 << 30], &[0, 0], 0, -128, 127, &mut out);
    }
}
