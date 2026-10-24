//! 8-bit code spaces as one codec, selected at compile time.
//!
//! [`Codec8`] is parameterised by four sealed marker types, and a calibration is a fifth
//! marker passed to each call:
//!
//! | parameter | markers | fixes |
//! |---|---|---|
//! | storage | [`I8`], [`U8`] | the element type, and how a code is stored |
//! | code space | [`Asym`], [`Sym`], [`UnitFull`] | the code range and the decode rule |
//! | rounding | [`TiesAway`], [`TiesEven`], [`Floor`] | how a scaled value becomes a code |
//! | NaN policy | [`NoNaN`], [`NaNAtMin`] | whether a code is reserved for NaN |
//! | calibration | any type implementing [`Calibration`] | scale and zero point |
//!
//! Every combination is a separate monomorphised function. Nothing is selected at run
//! time: there is no mode enum, no trait object and no function pointer, and the NaN
//! branch disappears for [`NoNaN`].
//!
//! Combinations that make no sense do not compile. [`NaNAtMin`] exists only for the
//! narrow space [`Sym`], because in [`Asym`] the code -128 is a value:
//!
//! ```compile_fail,E0277
//! use ndarray::hpc::codec8::{Asym, Codec8, NaNAtMin, TiesAway, I8};
//! let _: Codec8<I8, Asym, TiesAway, NaNAtMin>;
//! ```
//!
//! The same type with [`Sym`] compiles:
//!
//! ```
//! use ndarray::hpc::codec8::{Codec8, NaNAtMin, Sym, TiesAway, I8};
//! let _: Codec8<I8, Sym, TiesAway, NaNAtMin>;
//! ```
//!
//! [`UnitFull`] codes run from 0 to 255, so they cannot be stored as `i8`:
//!
//! ```compile_fail,E0277
//! use ndarray::hpc::codec8::{Codec8, Floor, NoNaN, UnitFull, I8};
//! let _: Codec8<I8, UnitFull, Floor, NoNaN>;
//! ```
//!
//! The plan and the council record are in `.claude/plans/quantized-codec8-v2.md`.

// `pub mod hpc` allows `clippy::all` for the whole module tree; this file opts back in.
#![warn(clippy::all)]

use core::fmt;
use core::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// Code spaces
// ---------------------------------------------------------------------------

/// The range of codes a [`Codec8`] uses.
pub trait CodeSpace: sealed::Sealed {
    /// Smallest code the encoder emits for a value.
    const MIN: i32;
    /// Largest code the encoder emits for a value.
    const MAX: i32;
}

/// A code space whose codes decode linearly through a [`Calibration`]:
/// `value = (code - ZERO_POINT) * SCALE`.
pub trait Linear: CodeSpace {}

/// Full signed range, -128..=127. Every code is a value, including -128.
///
/// This is the range of today's `quantize_f32_to_i8`, of TFLite activations and of ONNX
/// `QuantizeLinear` with an `int8` output.
#[derive(Debug, Clone, Copy)]
pub struct Asym;

/// Narrow signed range, -127..=127. The encoder never emits -128.
///
/// This is the range TFLite uses for weights. With [`NaNAtMin`] the unused code -128
/// means NaN; with [`NoNaN`] it is never produced and is rejected by
/// [`Codec8::decode_checked`].
#[derive(Debug, Clone, Copy)]
pub struct Sym;

/// Unit interval on 0..=255: code `c` stands for `c / 255`, so 0 and 255 are exactly 0.0
/// and 1.0.
///
/// This is the spoFC truth encoding (frequency and confidence). It has no calibration and
/// no signed counterpart, so it is stored only as `u8`.
#[derive(Debug, Clone, Copy)]
pub struct UnitFull;

impl sealed::Sealed for Asym {}
impl sealed::Sealed for Sym {}
impl sealed::Sealed for UnitFull {}

impl CodeSpace for Asym {
    const MIN: i32 = -128;
    const MAX: i32 = 127;
}
impl CodeSpace for Sym {
    const MIN: i32 = -127;
    const MAX: i32 = 127;
}
impl CodeSpace for UnitFull {
    const MIN: i32 = 0;
    const MAX: i32 = 255;
}
impl Linear for Asym {}
impl Linear for Sym {}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// How a code is stored.
pub trait Storage: sealed::Sealed {
    /// The stored element type.
    type Elem: Copy + fmt::Debug + PartialEq;
}

/// Codes stored as `i8`; the stored byte is the code itself.
#[derive(Debug, Clone, Copy)]
pub struct I8;

/// Codes stored as `u8`.
///
/// For [`Asym`] and [`Sym`] the stored byte is the signed code XOR 0x80, the same
/// relation TFLite Micro's int8↔uint8 requantize uses. For [`UnitFull`] it is the code
/// itself.
#[derive(Debug, Clone, Copy)]
pub struct U8;

impl sealed::Sealed for I8 {}
impl sealed::Sealed for U8 {}
impl Storage for I8 {
    type Elem = i8;
}
impl Storage for U8 {
    type Elem = u8;
}

/// A storage that can hold the codes of code space `S`.
pub trait Stores<S: CodeSpace>: Storage {
    /// Store a code that is already within `S::MIN..=S::MAX`, or the NaN code.
    fn store(code: i32) -> Self::Elem;
    /// Read the code back.
    fn load(elem: Self::Elem) -> i32;
}

impl Stores<Asym> for I8 {
    #[inline(always)]
    fn store(code: i32) -> i8 {
        code as i8
    }
    #[inline(always)]
    fn load(elem: i8) -> i32 {
        i32::from(elem)
    }
}
impl Stores<Sym> for I8 {
    #[inline(always)]
    fn store(code: i32) -> i8 {
        code as i8
    }
    #[inline(always)]
    fn load(elem: i8) -> i32 {
        i32::from(elem)
    }
}
impl Stores<Asym> for U8 {
    #[inline(always)]
    fn store(code: i32) -> u8 {
        (code + 128) as u8
    }
    #[inline(always)]
    fn load(elem: u8) -> i32 {
        i32::from(elem) - 128
    }
}
impl Stores<Sym> for U8 {
    #[inline(always)]
    fn store(code: i32) -> u8 {
        (code + 128) as u8
    }
    #[inline(always)]
    fn load(elem: u8) -> i32 {
        i32::from(elem) - 128
    }
}
impl Stores<UnitFull> for U8 {
    #[inline(always)]
    fn store(code: i32) -> u8 {
        code as u8
    }
    #[inline(always)]
    fn load(elem: u8) -> i32 {
        i32::from(elem)
    }
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// How a scaled value or an integer ratio becomes a code.
pub trait Rounding: sealed::Sealed {
    /// Round a finite or non-finite `f32` to an integral `f32`. NaN and ±inf pass through.
    fn round(x: f32) -> f32;
    /// Round `n / d` to an integer, for `d > 0`.
    fn div_round(n: u128, d: u128) -> u128;
}

/// Round half away from zero, as `f32::round` and TFLite (`std::round`).
#[derive(Debug, Clone, Copy)]
pub struct TiesAway;

/// Round half to even, as ONNX `QuantizeLinear`.
#[derive(Debug, Clone, Copy)]
pub struct TiesEven;

/// Round toward negative infinity, as today's spoFC integer encoders.
#[derive(Debug, Clone, Copy)]
pub struct Floor;

impl sealed::Sealed for TiesAway {}
impl sealed::Sealed for TiesEven {}
impl sealed::Sealed for Floor {}

impl Rounding for TiesAway {
    #[inline(always)]
    fn round(x: f32) -> f32 {
        x.round()
    }
    #[inline(always)]
    fn div_round(n: u128, d: u128) -> u128 {
        (2 * n + d) / (2 * d)
    }
}
impl Rounding for TiesEven {
    #[inline(always)]
    fn round(x: f32) -> f32 {
        x.round_ties_even()
    }
    #[inline(always)]
    fn div_round(n: u128, d: u128) -> u128 {
        let (q, r) = (n / d, n % d);
        match (2 * r).cmp(&d) {
            core::cmp::Ordering::Less => q,
            core::cmp::Ordering::Greater => q + 1,
            core::cmp::Ordering::Equal => q + (q & 1),
        }
    }
}
impl Rounding for Floor {
    #[inline(always)]
    fn round(x: f32) -> f32 {
        x.floor()
    }
    #[inline(always)]
    fn div_round(n: u128, d: u128) -> u128 {
        n / d
    }
}

// ---------------------------------------------------------------------------
// NaN policy
// ---------------------------------------------------------------------------

/// Whether code space `S` reserves a code for NaN.
pub trait NanPolicy<S: CodeSpace>: sealed::Sealed {
    /// The code that means NaN, if any.
    const NAN_CODE: Option<i32>;
}

/// No NaN code. A NaN input is rejected by the checked encoder.
#[derive(Debug, Clone, Copy)]
pub struct NoNaN;

/// The code -128 means NaN. Only valid with [`Sym`], whose values never use -128.
#[derive(Debug, Clone, Copy)]
pub struct NaNAtMin;

impl sealed::Sealed for NoNaN {}
impl sealed::Sealed for NaNAtMin {}

impl<S: CodeSpace> NanPolicy<S> for NoNaN {
    const NAN_CODE: Option<i32> = None;
}
impl NanPolicy<Sym> for NaNAtMin {
    const NAN_CODE: Option<i32> = Some(-128);
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// A compile-time calibration: `value = (code - ZERO_POINT) * SCALE`.
///
/// Production calibrations are marker types, usually generated from a calibration
/// manifest. Calibrations chosen at run time exist only behind the `lab-calibration`
/// feature.
///
/// ```
/// use ndarray::hpc::codec8::Calibration;
///
/// struct Unit127;
/// impl Calibration for Unit127 {
///     const SCALE: f32 = 1.0 / 127.0;
///     const ZERO_POINT: i32 = 0;
/// }
/// assert_eq!(Unit127::ZERO_POINT, 0);
/// ```
pub trait Calibration {
    /// Width of one code step. Must be finite and positive; a calibration that breaks
    /// this, or puts `ZERO_POINT` outside the code space, fails to compile when used.
    const SCALE: f32;
    /// The code that stands for 0.0.
    const ZERO_POINT: i32;
}

// ---------------------------------------------------------------------------
// The codec
// ---------------------------------------------------------------------------

/// Why a checked encode or decode refused its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec8Error {
    /// The source and destination slices have different lengths.
    LengthMismatch {
        /// Length of the source slice.
        src: usize,
        /// Length of the destination slice.
        dst: usize,
    },
    /// A NaN input under [`NoNaN`].
    NanInput {
        /// Position of the first NaN.
        index: usize,
    },
    /// A stored code outside the code space that is not the NaN code.
    InvalidCode {
        /// Position of the first invalid code.
        index: usize,
    },
}

impl fmt::Display for Codec8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Codec8Error::LengthMismatch { src, dst } => {
                write!(f, "source has {src} elements, destination {dst}")
            }
            Codec8Error::NanInput { index } => write!(f, "NaN input at index {index}"),
            Codec8Error::InvalidCode { index } => write!(f, "invalid code at index {index}"),
        }
    }
}

impl std::error::Error for Codec8Error {}

/// Carries the marker types without owning a value of any of them.
type Markers<T, S, R, N> = fn() -> (T, S, R, N);

/// An 8-bit codec fixed at compile time by storage `T`, code space `S`, rounding `R` and
/// NaN policy `N`.
///
/// It is a zero-sized type; all operations are associated functions.
///
/// ```
/// use ndarray::hpc::codec8::{Calibration, OnnxQ};
///
/// struct Half;
/// impl Calibration for Half {
///     const SCALE: f32 = 0.5;
///     const ZERO_POINT: i32 = 0;
/// }
/// let mut codes = [0i8; 4];
/// OnnxQ::encode_checked::<Half>(&[0.25, 0.75, -1.0, 100.0], &mut codes).unwrap();
/// // 0.5 and 1.5 round to the even neighbour; 200 saturates.
/// assert_eq!(codes, [0, 2, -2, 127]);
/// ```
pub struct Codec8<T, S, R, N>(PhantomData<Markers<T, S, R, N>>)
where
    T: Stores<S>,
    S: CodeSpace,
    R: Rounding,
    N: NanPolicy<S>;

impl<T, S, R, N> Codec8<T, S, R, N>
where
    T: Stores<S>,
    S: Linear,
    R: Rounding,
    N: NanPolicy<S>,
{
    /// Encode one value. Total: NaN gives the NaN code, or the zero-point code under
    /// [`NoNaN`]; ±inf and out-of-range values saturate to `S::MIN` / `S::MAX`.
    #[inline(always)]
    pub(crate) fn encode_one<C: Calibration>(v: f32) -> T::Elem {
        // A true division, not a multiplication by a precomputed reciprocal: the two
        // differ in f32 exactly at half-ties, where the rounding mode decides the code.
        const {
            assert!(C::SCALE > 0.0 && C::SCALE < f32::INFINITY, "SCALE must be finite and positive");
            assert!(C::ZERO_POINT >= S::MIN && C::ZERO_POINT <= S::MAX, "ZERO_POINT must lie in the code space");
        }
        let r = R::round(v / C::SCALE);
        // Clamp to the range shifted by the zero point, then add it as an integer. For
        // integral `r` this equals `clamp(r + zp)`, and with a zero point of 0 the add
        // vanishes; an f32 `+ 0.0` would not, because it changes -0.0.
        // `as i32` maps NaN to 0, as today's quantizer does.
        let lo = (S::MIN - C::ZERO_POINT) as f32;
        let hi = (S::MAX - C::ZERO_POINT) as f32;
        let q = r.clamp(lo, hi) as i32 + C::ZERO_POINT;
        let q = match N::NAN_CODE {
            Some(nan) if v.is_nan() => nan,
            _ => q,
        };
        T::store(q)
    }

    /// Decode one stored code. Total: the NaN code gives NaN; every other byte decodes
    /// linearly, including codes outside the code space.
    #[inline(always)]
    pub(crate) fn decode_one<C: Calibration>(e: T::Elem) -> f32 {
        let c = T::load(e);
        match N::NAN_CODE {
            Some(nan) if c == nan => f32::NAN,
            _ => (c - C::ZERO_POINT) as f32 * C::SCALE,
        }
    }

    /// Encode without validation. Defined for every input (see [`Self::encode_one`]).
    #[inline(always)]
    pub(crate) fn encode_unchecked<C: Calibration>(src: &[f32], dst: &mut [T::Elem]) {
        for (o, &v) in dst.iter_mut().zip(src) {
            *o = Self::encode_one::<C>(v);
        }
    }

    /// Decode without validation. Defined for every input (see [`Self::decode_one`]).
    #[inline(always)]
    pub(crate) fn decode_unchecked<C: Calibration>(src: &[T::Elem], dst: &mut [f32]) {
        for (o, &e) in dst.iter_mut().zip(src) {
            *o = Self::decode_one::<C>(e);
        }
    }

    /// Encode `src` into `dst` with calibration `C`.
    ///
    /// The input is validated once, then encoded in one pass. Values outside the range
    /// saturate; ±inf saturates. Under [`NoNaN`] a NaN input is an error and nothing is
    /// written; under [`NaNAtMin`] it becomes the code -128.
    ///
    /// ```
    /// use ndarray::hpc::codec8::{Calibration, Codec8, NaNAtMin, Sym, TiesEven, I8};
    ///
    /// struct One;
    /// impl Calibration for One {
    ///     const SCALE: f32 = 1.0;
    ///     const ZERO_POINT: i32 = 0;
    /// }
    /// type Lane = Codec8<I8, Sym, TiesEven, NaNAtMin>;
    /// let mut codes = [0i8; 3];
    /// Lane::encode_checked::<One>(&[f32::NAN, -500.0, 2.5], &mut codes).unwrap();
    /// assert_eq!(codes, [-128, -127, 2]);
    /// ```
    pub fn encode_checked<C: Calibration>(src: &[f32], dst: &mut [T::Elem]) -> Result<(), Codec8Error> {
        if src.len() != dst.len() {
            return Err(Codec8Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        if N::NAN_CODE.is_none() {
            if let Some(index) = src.iter().position(|v| v.is_nan()) {
                return Err(Codec8Error::NanInput { index });
            }
        }
        Self::encode_unchecked::<C>(src, dst);
        Ok(())
    }

    /// Decode `src` into `dst` with calibration `C`.
    ///
    /// A stored code outside the code space that is not the NaN code (for [`Sym`] with
    /// [`NoNaN`], the code -128) is an error and nothing is written.
    ///
    /// ```
    /// use ndarray::hpc::codec8::{Calibration, TfliteWeight, Codec8Error};
    ///
    /// struct One;
    /// impl Calibration for One {
    ///     const SCALE: f32 = 1.0;
    ///     const ZERO_POINT: i32 = 0;
    /// }
    /// let mut out = [0.0f32; 2];
    /// TfliteWeight::decode_checked::<One>(&[-127, 5], &mut out).unwrap();
    /// assert_eq!(out, [-127.0, 5.0]);
    /// assert_eq!(
    ///     TfliteWeight::decode_checked::<One>(&[3, -128], &mut out),
    ///     Err(Codec8Error::InvalidCode { index: 1 })
    /// );
    /// ```
    pub fn decode_checked<C: Calibration>(src: &[T::Elem], dst: &mut [f32]) -> Result<(), Codec8Error> {
        if src.len() != dst.len() {
            return Err(Codec8Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        let invalid = |&e: &T::Elem| {
            let c = T::load(e);
            (c < S::MIN || c > S::MAX) && N::NAN_CODE != Some(c)
        };
        if let Some(index) = src.iter().position(invalid) {
            return Err(Codec8Error::InvalidCode { index });
        }
        Self::decode_unchecked::<C>(src, dst);
        Ok(())
    }
}

impl<R: Rounding> Codec8<U8, UnitFull, R, NoNaN> {
    /// Encode values in [0, 1] as `round(v * 255)`, saturating outside the interval.
    /// A NaN input is an error and nothing is written.
    ///
    /// ```
    /// use ndarray::hpc::codec8::SpoFcTruth;
    ///
    /// let mut codes = [0u8; 3];
    /// SpoFcTruth::encode_checked(&[0.0, 1.0, 0.999], &mut codes).unwrap();
    /// assert_eq!(codes, [0, 255, 254]);
    /// ```
    pub fn encode_checked(src: &[f32], dst: &mut [u8]) -> Result<(), Codec8Error> {
        if src.len() != dst.len() {
            return Err(Codec8Error::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        if let Some(index) = src.iter().position(|v| v.is_nan()) {
            return Err(Codec8Error::NanInput { index });
        }
        for (o, &v) in dst.iter_mut().zip(src) {
            *o = R::round(v * 255.0).clamp(0.0, 255.0) as u8;
        }
        Ok(())
    }

    /// Decode codes as `c / 255`. Every byte is a valid code; 255 decodes to exactly 1.0.
    ///
    /// ```
    /// use ndarray::hpc::codec8::SpoFcTruth;
    ///
    /// let mut out = [0.0f32; 2];
    /// SpoFcTruth::decode(&[0, 255], &mut out);
    /// assert_eq!(out, [0.0, 1.0]);
    /// ```
    ///
    /// # Panics
    /// Panics if the slices have different lengths.
    pub fn decode(src: &[u8], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len(), "decode: length mismatch");
        for (o, &c) in dst.iter_mut().zip(src) {
            // A division, so that 255 / 255 is exactly 1.0.
            *o = f32::from(c) / 255.0;
        }
    }

    /// Encode the integer ratio `n / d` as a unit code, `round(n * 255 / d)` capped at
    /// 255, without going through floating point. Returns `None` when `d` is 0.
    ///
    /// With [`Floor`] this is the spoFC encoder: frequency
    /// `encode_ratio(cooccur, antecedent)` and confidence `encode_ratio(m, m + k)`.
    ///
    /// ```
    /// use ndarray::hpc::codec8::SpoFcTruth;
    ///
    /// assert_eq!(SpoFcTruth::encode_ratio(1, 2), Some(127));
    /// assert_eq!(SpoFcTruth::encode_ratio(5, 4), Some(255));
    /// assert_eq!(SpoFcTruth::encode_ratio(1, 0), None);
    /// ```
    pub fn encode_ratio(n: u64, d: u64) -> Option<u8> {
        if d == 0 {
            return None;
        }
        let q = R::div_round(u128::from(n) * 255, u128::from(d));
        Some(q.min(255) as u8)
    }
}

// ---------------------------------------------------------------------------
// Aliases for external formats
// ---------------------------------------------------------------------------

/// TFLite / TFLite Micro int8 activations: full range, ties away from zero.
pub type TfliteAct = Codec8<I8, Asym, TiesAway, NoNaN>;

/// TFLite int8 weights: narrow range (-127..=127), ties away from zero.
pub type TfliteWeight = Codec8<I8, Sym, TiesAway, NoNaN>;

/// ONNX `QuantizeLinear` / `DequantizeLinear` with `int8`: full range, ties to even.
pub type OnnxQ = Codec8<I8, Asym, TiesEven, NoNaN>;

/// spoFC truth bytes (NARS frequency and confidence): unit range, floor.
pub type SpoFcTruth = Codec8<U8, UnitFull, Floor, NoNaN>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::quantized::quantize_f32_to_i8;

    struct One;
    impl Calibration for One {
        const SCALE: f32 = 1.0;
        const ZERO_POINT: i32 = 0;
    }
    struct Shift3;
    impl Calibration for Shift3 {
        const SCALE: f32 = 1.0;
        const ZERO_POINT: i32 = -3;
    }

    type AsymEven = Codec8<I8, Asym, TiesEven, NoNaN>;
    type SymAway = Codec8<I8, Sym, TiesAway, NoNaN>;
    type LaneEven = Codec8<I8, Sym, TiesEven, NaNAtMin>;
    type LaneAway = Codec8<I8, Sym, TiesAway, NaNAtMin>;

    fn enc<T, S, R, N, C>(src: &[f32]) -> Vec<T::Elem>
    where
        T: Stores<S>,
        S: Linear,
        R: Rounding,
        N: NanPolicy<S>,
        C: Calibration,
    {
        src.iter()
            .map(|&v| Codec8::<T, S, R, N>::encode_one::<C>(v))
            .collect()
    }

    /// A deterministic corpus with ties, saturating values, signed zeros, NaN and ±inf.
    fn corpus() -> Vec<f32> {
        let mut v = vec![0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -127.5, -128.5];
        v.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e9, -1e9]);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..4000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.push(((s >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0);
        }
        v
    }

    // G2: TfliteAct with the exact fitted scale reproduces `quantize_f32_to_i8`.
    #[test]
    fn tflite_act_matches_quantize_f32_to_i8_bit_for_bit() {
        // Finite data with an abs max of 3.0 (the corpus generator stays inside ±3).
        let mut data: Vec<f32> = corpus()
            .into_iter()
            .filter(|v| v.is_finite() && v.abs() <= 3.0)
            .collect();
        data.push(3.0);
        let (codes, params) = quantize_f32_to_i8(&data);
        struct Fitted;
        impl Calibration for Fitted {
            const SCALE: f32 = f32::from_bits(0x3cc1_8306); // 3.0 / 127.0
            const ZERO_POINT: i32 = 0;
        }
        assert_ne!(params.scale, 1.0, "the corpus must not hit the 1.0 fallback");
        assert_eq!(params.scale.to_bits(), Fitted::SCALE.to_bits(), "fitted scale bits");
        assert_eq!(enc::<I8, Asym, TiesAway, NoNaN, Fitted>(&data), codes);
        // NaN and ±inf: today's quantizer gives 0 and ±saturation; so does the
        // unchecked path. Compare element-wise through the same scale.
        let special = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let today: Vec<i8> = special
            .iter()
            .map(|&v| (v / params.scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        assert_eq!(enc::<I8, Asym, TiesAway, NoNaN, Fitted>(&special), today);
        assert_eq!(today, [0, 127, -128, 0]);
    }

    // The division matters: a precomputed reciprocal changes codes at half-ties.
    #[test]
    fn division_and_reciprocal_disagree_at_a_tie() {
        struct S;
        impl Calibration for S {
            const SCALE: f32 = 0.0137;
            const ZERO_POINT: i32 = 0;
        }
        let v = -1.73305f32;
        assert_eq!(v / S::SCALE, -126.5);
        assert_ne!(v * (1.0 / S::SCALE), -126.5);
        assert_eq!(enc::<I8, Asym, TiesAway, NoNaN, S>(&[v]), [-127]);
    }

    // P-02: ties, both directions, both rounding modes; a non-tie twin.
    #[test]
    fn ties_round_away_or_to_even() {
        let xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, -126.5];
        assert_eq!(enc::<I8, Asym, TiesAway, NoNaN, One>(&xs), [-3, -2, -1, 1, 2, 3, 127, -127]);
        assert_eq!(enc::<I8, Asym, TiesEven, NoNaN, One>(&xs), [-2, -2, 0, 0, 2, 2, 126, -126]);
        let near = [-2.4999998, 2.5000002];
        assert_eq!(enc::<I8, Asym, TiesAway, NoNaN, One>(&near), [-2, 3]);
        assert_eq!(enc::<I8, Asym, TiesEven, NoNaN, One>(&near), [-2, 3]);
    }

    #[test]
    fn saturation_and_signed_zero() {
        let xs = [-1e9, -128.5, -127.6, 127.6, 1e9, -0.0, 0.0];
        assert_eq!(enc::<I8, Asym, TiesEven, NoNaN, One>(&xs), [-128, -128, -128, 127, 127, 0, 0]);
        assert_eq!(enc::<I8, Sym, TiesEven, NoNaN, One>(&xs), [-127, -127, -127, 127, 127, 0, 0]);
    }

    #[test]
    fn nonzero_zero_point_shifts_codes() {
        let xs = [-200.0, -2.5, 0.0, 2.5, 200.0];
        assert_eq!(enc::<I8, Asym, TiesEven, NoNaN, Shift3>(&xs), [-128, -5, -3, -1, 127]);
        let mut out = [0.0f32; 2];
        AsymEven::decode_checked::<Shift3>(&[-3, -5], &mut out).unwrap();
        assert_eq!(out, [0.0, -2.0]);
    }

    // P-04: Sym + NaNAtMin contract (±inf saturates: proposal, plan §8 item 5).
    #[test]
    fn sym_nan_at_min_contract() {
        let xs = [f32::NAN, 1e9, -1e9, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0];
        for codes in [enc::<I8, Sym, TiesEven, NaNAtMin, One>(&xs), enc::<I8, Sym, TiesAway, NaNAtMin, One>(&xs)] {
            assert_eq!(codes, [-128, 127, -127, 127, -127, 0, 0]);
        }
        let mut out = [0.0f32; 3];
        LaneEven::decode_checked::<One>(&[-128, -127, 127], &mut out).unwrap();
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[-127.0, 127.0]);
    }

    // Sym + NoNaN and Sym + NaNAtMin agree everywhere except NaN input and code -128.
    #[test]
    fn nan_policy_changes_only_nan_and_minus_128() {
        let finite: Vec<f32> = corpus().into_iter().filter(|v| !v.is_nan()).collect();
        assert_eq!(enc::<I8, Sym, TiesAway, NoNaN, One>(&finite), enc::<I8, Sym, TiesAway, NaNAtMin, One>(&finite));
        assert!(enc::<I8, Sym, TiesAway, NaNAtMin, One>(&finite)
            .iter()
            .all(|&c| c != -128));
        for c in -127i8..=127 {
            assert_eq!(SymAway::decode_one::<One>(c).to_bits(), LaneAway::decode_one::<One>(c).to_bits());
        }
        assert_eq!(SymAway::decode_one::<One>(-128), -128.0);
        assert!(LaneAway::decode_one::<One>(-128).is_nan());
    }

    // P-12: encode(decode(c)) == c for every code of every linear alias.
    #[test]
    fn encode_decode_is_identity_on_every_code() {
        fn check<T: Stores<S>, S: Linear, R: Rounding, N: NanPolicy<S>>(codes: &[T::Elem]) {
            for &e in codes {
                let back = Codec8::<T, S, R, N>::encode_one::<One>(Codec8::<T, S, R, N>::decode_one::<One>(e));
                assert_eq!(back, e);
            }
        }
        let i8_full: Vec<i8> = (-128..=127).collect();
        let i8_narrow: Vec<i8> = (-127..=127).collect();
        let u8_full: Vec<u8> = (0..=255).collect();
        let u8_narrow: Vec<u8> = (1..=255).collect();
        check::<I8, Asym, TiesAway, NoNaN>(&i8_full);
        check::<I8, Asym, TiesEven, NoNaN>(&i8_full);
        check::<I8, Sym, TiesAway, NoNaN>(&i8_narrow);
        check::<I8, Sym, TiesEven, NaNAtMin>(&i8_full);
        check::<U8, Asym, TiesEven, NoNaN>(&u8_full);
        check::<U8, Sym, TiesEven, NoNaN>(&u8_narrow);
    }

    // G4: u8 storage is the i8 code XOR 0x80, on all 256 codes, for every Asym/Sym alias.
    #[test]
    fn u8_storage_is_i8_xor_0x80() {
        let xs: Vec<f32> = (-130..=130).map(|i| i as f32).chain([f32::NAN]).collect();
        fn pair<S: Linear, R: Rounding, N: NanPolicy<S>>(xs: &[f32])
        where
            I8: Stores<S>,
            U8: Stores<S>,
        {
            let a = enc::<I8, S, R, N, One>(xs);
            let b = enc::<U8, S, R, N, One>(xs);
            for (&x, &y) in a.iter().zip(&b) {
                assert_eq!((x as u8) ^ 0x80, y);
            }
            for byte in 0u8..=255 {
                let signed = (byte ^ 0x80) as i8;
                assert_eq!(
                    Codec8::<I8, S, R, N>::decode_one::<One>(signed).to_bits(),
                    Codec8::<U8, S, R, N>::decode_one::<One>(byte).to_bits()
                );
            }
        }
        pair::<Asym, TiesAway, NoNaN>(&xs);
        pair::<Asym, TiesEven, NoNaN>(&xs);
        pair::<Sym, TiesAway, NoNaN>(&xs);
        pair::<Sym, TiesEven, NaNAtMin>(&xs);
    }

    #[test]
    fn checked_paths_reject_and_write_nothing() {
        let mut codes = [7i8; 3];
        assert_eq!(
            OnnxQ::encode_checked::<One>(&[1.0, f32::NAN, 2.0], &mut codes),
            Err(Codec8Error::NanInput { index: 1 })
        );
        assert_eq!(codes, [7, 7, 7]);
        assert_eq!(
            OnnxQ::encode_checked::<One>(&[1.0], &mut codes),
            Err(Codec8Error::LengthMismatch { src: 1, dst: 3 })
        );
        LaneEven::encode_checked::<One>(&[1.0, f32::NAN, 2.0], &mut codes).unwrap();
        assert_eq!(codes, [1, -128, 2]);
        let mut out = [9.0f32; 2];
        assert_eq!(
            Codec8::<U8, Sym, TiesEven, NoNaN>::decode_checked::<One>(&[128, 0], &mut out),
            Err(Codec8Error::InvalidCode { index: 1 })
        );
        assert_eq!(out, [9.0, 9.0]);
    }

    // D5: the unchecked path is total; a missed validation gives a pinned value.
    #[test]
    fn unchecked_path_is_defined_on_bad_input() {
        let mut codes = [0i8; 2];
        OnnxQ::encode_unchecked::<One>(&[f32::NAN, f32::INFINITY], &mut codes);
        assert_eq!(codes, [0, 127]);
        let mut out = [0.0f32; 1];
        SymAway::decode_unchecked::<One>(&[-128], &mut out);
        assert_eq!(out, [-128.0]);
    }

    #[test]
    fn unit_full_round_trip_and_endpoints() {
        let codes: Vec<u8> = (0..=255).collect();
        let mut vals = vec![0.0f32; 256];
        SpoFcTruth::decode(&codes, &mut vals);
        assert_eq!((vals[0], vals[255]), (0.0, 1.0));
        let mut back = vec![0u8; 256];
        Codec8::<U8, UnitFull, TiesEven, NoNaN>::encode_checked(&vals, &mut back).unwrap();
        assert_eq!(back, codes);
        assert_eq!(SpoFcTruth::encode_checked(&[f32::NAN], &mut [0u8]), Err(Codec8Error::NanInput { index: 0 }));
    }

    // G13: encode_ratio with Floor equals the spoFC encoder in lance-graph
    // `crates/lance-graph-arm-discovery/src/translator.rs` (`arm_to_truth_u8`), transcribed:
    //   frequency  = ((cooccur * 255) / antecedent).min(255)   (antecedent > 0)
    //   confidence = (m * 255) / (m + k)                        (k > 0)
    #[test]
    fn floor_ratio_matches_spofc_encoder() {
        for k in [1u64, 2, 5] {
            for m in 0u64..=4096 {
                assert_eq!(SpoFcTruth::encode_ratio(m, m + k), Some(((m * 255) / (m + k)) as u8));
            }
        }
        for a in 1u64..=300 {
            for c in 0u64..=a + 5 {
                assert_eq!(SpoFcTruth::encode_ratio(c, a), Some(((c * 255) / a).min(255) as u8));
            }
        }
    }

    // Rounded confidence reaches 255 (manufactured certainty); floor never does.
    #[test]
    fn rounded_confidence_can_reach_255_floor_cannot() {
        type Even = Codec8<U8, UnitFull, TiesEven, NoNaN>;
        type Away = Codec8<U8, UnitFull, TiesAway, NoNaN>;
        assert_eq!(Even::encode_ratio(510, 511), Some(255));
        assert_eq!(Away::encode_ratio(509, 510), Some(255)); // exact tie 254.5
        assert_eq!(Even::encode_ratio(509, 510), Some(254)); // tie to even
        for m in 0u64..=100_000 {
            assert!(SpoFcTruth::encode_ratio(m, m + 1).unwrap() < 255);
        }
    }
}
