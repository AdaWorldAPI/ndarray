//! Canonical DR512 numerical Codec9, reusing Codec8's compile-time calibration
//! and nearest-rounding types without borrowing Codec8's storage or NaN policy.
//!
//! One u8 magnitude plus ONE epistemic presence/polarity bit per lane:
//!
//! | P | Magnitude | Meaning |
//! |---|-----------|---------|
//! | 0 | 0         | -0 = NaN = epistemically NEVER TOUCHED |
//! | 0 | 1..=255   | negative -1..=-255 |
//! | 1 | 0         | valid +0 |
//! | 1 | 1..=255   | positive +1..=+255 |
//!
//! 512 codewords, 511 numeric values, 1 never-touched sentinel. A value
//! quantized from -0.4 to zero **must** emit +0, never the -0 sentinel.
//! A WHERE(P=0 AND magnitude=0) is sufficient; do not allocate a second mask.
//! Epistemic presence is NOT a source-write or alpha-claim authorization.
//!
//! Storage, polarity, Sym and NaN policy are intentionally NOT configurable.
//! An Asym mode, a separate sign/validity plane, or an optional NaN policy
//! would violate this contract. The only compile-time knobs are the existing
//! nearest rounding mode and scale calibration.
//!
//! ```
//! use ndarray::hpc::codec8::{Calibration, TiesEven};
//! use ndarray::hpc::codec9::{Codec9, EpiLane9};
//! struct Unit;
//! impl Calibration for Unit {
//!     const SCALE: f32 = 1.0;
//!     const ZERO_POINT: i32 = 0;
//! }
//! type Q = Codec9<TiesEven, Unit>;
//! assert_eq!(Q::encode_touched(-0.4).unwrap().codeword(), 0x100);
//! assert_eq!(EpiLane9::NEVER_TOUCHED.codeword(), 0);
//! ```

use core::marker::PhantomData;

use super::codec8::{Calibration, Rounding, TiesAway, TiesEven};

/// Restricts DR512 quantization to nearest rounding, never Floor.
/// Floor would turn -0.4 into -1 rather than the required +0.
pub trait NearestRounding9: Rounding {}

impl NearestRounding9 for TiesEven {}
impl NearestRounding9 for TiesAway {}

/// One of the canonical 512 DR512 codewords.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpiLane9 {
    magnitude: u8,
    positive: bool,
}

impl EpiLane9 {
    /// The unique -0 codeword, meaning epistemically never touched.
    pub const NEVER_TOUCHED: Self = Self {
        magnitude: 0,
        positive: false,
    };

    /// Returns the canonical codeword, with presence in bit 8.
    #[must_use]
    pub const fn codeword(self) -> u16 {
        (self.magnitude as u16) | ((self.positive as u16) << 8)
    }

    /// Reads one 9-bit codeword, refusing values beyond 511.
    #[must_use]
    pub const fn from_codeword(code: u16) -> Option<Self> {
        if code > 511 {
            None
        } else {
            Some(Self {
                magnitude: code as u8,
                positive: code & 0x100 != 0,
            })
        }
    }

    /// Evaluates WHERE(P=0 AND M=0), without creating another bitmap.
    #[must_use]
    pub const fn never_touched(self) -> bool {
        !self.positive && self.magnitude == 0
    }

    /// Returns a signed numeric value, or None only for untouched -0.
    #[must_use]
    pub const fn value(self) -> Option<i16> {
        if self.never_touched() {
            None
        } else if self.positive {
            Some(self.magnitude as i16)
        } else {
            Some(-(self.magnitude as i16))
        }
    }

    /// Returns the stored unsigned magnitude.
    #[must_use]
    pub const fn magnitude(self) -> u8 {
        self.magnitude
    }

    /// Returns the canonical epistemic presence/polarity bit.
    #[must_use]
    pub const fn presence(self) -> bool {
        self.positive
    }
}

/// A finite-value-only encoder never invents "never touched".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec9Error {
    /// An explicitly touched NaN or infinity is not a numeric value.
    NonFinite,
}

/// Compile-time-calibrated canonical DR512 numeric codec.
///
/// Reuses Codec8's `Rounding` and `Calibration` traits. Neither the
/// bit meaning nor storage/NaN policy can be selected or adapted.
/// The calibration must have positive finite SCALE and ZERO_POINT=0.
///
/// ```compile_fail
/// use ndarray::hpc::codec8::{Calibration, Floor};
/// use ndarray::hpc::codec9::Codec9;
/// struct Unit;
/// impl Calibration for Unit {
///     const SCALE: f32 = 1.0;
///     const ZERO_POINT: i32 = 0;
/// }
/// let _ = Codec9::<Floor, Unit>::encode_touched(-0.4);
/// ```
pub struct Codec9<R: NearestRounding9, C: Calibration>(PhantomData<(R, C)>);

impl<R: NearestRounding9, C: Calibration> Codec9<R, C> {
    #[inline(always)]
    fn assert_calibration() {
        const {
            assert!(C::SCALE > 0.0 && C::SCALE < f32::INFINITY, "Codec9 scale must be finite and positive");
            assert!(C::ZERO_POINT == 0, "Codec9 requires Sym with ZERO_POINT=0");
        }
    }

    /// Encodes a TOUCHED finite value. Any rounded zero is canonical +0,
    /// regardless of input sign; -0/NaN is never manufactured by rounding.
    /// Finite out-of-range values saturate at magnitude 255.
    pub fn encode_touched(value: f32) -> Result<EpiLane9, Codec9Error> {
        Self::assert_calibration();
        if !value.is_finite() {
            return Err(Codec9Error::NonFinite);
        }
        let quantized = R::round(value / C::SCALE);
        let magnitude = quantized.abs().clamp(0.0, 255.0) as u8;
        Ok(EpiLane9 {
            magnitude,
            positive: quantized >= 0.0 || magnitude == 0,
        })
    }

    /// Decodes -0 into a floating NaN and all other codes using calibration.
    /// This is an explicit numeric view, not a permission or NULL decision.
    #[must_use]
    pub fn decode(lane: EpiLane9) -> f32 {
        Self::assert_calibration();
        match lane.value() {
            None => f32::NAN,
            Some(v) => f32::from(v) * C::SCALE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Unit;
    impl Calibration for Unit {
        const SCALE: f32 = 1.0;
        const ZERO_POINT: i32 = 0;
    }
    type Even = Codec9<TiesEven, Unit>;
    type Away = Codec9<TiesAway, Unit>;

    #[test]
    fn all_512_codewords_are_unique_and_absence_is_exactly_zero_and_zero() {
        let mut absent = 0usize;
        for code in 0..512u16 {
            let lane = EpiLane9::from_codeword(code).unwrap();
            assert_eq!(lane.codeword(), code);
            if lane.never_touched() {
                absent += 1;
                assert_eq!(code, 0);
                assert!(Even::decode(lane).is_nan());
            } else {
                let expected = if code < 256 {
                    -(code as i16)
                } else {
                    (code - 256) as i16
                };
                assert_eq!(lane.value(), Some(expected));
                assert_eq!(Even::decode(lane), expected as f32);
            }
        }
        assert_eq!(absent, 1);
        assert_eq!(EpiLane9::from_codeword(512), None);
    }

    #[test]
    fn negative_ceiling_zero_never_fabricates_epistemic_absence() {
        for v in [-0.0, -0.00001, -0.4, -0.49999, -0.5, 0.0, 0.4] {
            let code = Even::encode_touched(v).unwrap();
            assert_eq!(code.codeword(), 0x100, "touched {v} must be valid +0");
            assert!(!code.never_touched());
        }
        assert_eq!(Away::encode_touched(-0.4).unwrap().codeword(), 0x100);
        assert_eq!(Away::encode_touched(-0.5).unwrap().codeword(), 0x001);
        assert_eq!(Even::encode_touched(-0.50001).unwrap().codeword(), 0x001);
        assert_eq!(Even::encode_touched(-1.0).unwrap().codeword(), 0x001);
        assert_eq!(Even::encode_touched(0.0).unwrap().codeword(), 0x100);
        assert_eq!(Even::encode_touched(255.0).unwrap().codeword(), 0x1ff);
        assert_eq!(Even::encode_touched(-255.0).unwrap().codeword(), 0x0ff);
        assert_eq!(Even::encode_touched(1000.0).unwrap().codeword(), 0x1ff);
        assert_eq!(Even::encode_touched(-1000.0).unwrap().codeword(), 0x0ff);
    }

    #[test]
    fn only_explicitly_unvisited_lanes_get_the_sentinel() {
        assert_eq!(EpiLane9::NEVER_TOUCHED.codeword(), 0);
        for value in -255..=255 {
            let encoded = Even::encode_touched(value as f32).unwrap();
            assert!(!encoded.never_touched());
            assert_eq!(encoded.value(), Some(value));
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(Even::encode_touched(invalid), Err(Codec9Error::NonFinite));
        }
    }

    #[test]
    fn decode_and_reencode_touched_values_preserve_the_entire_numeric_domain() {
        for code in 1..512u16 {
            let lane = EpiLane9::from_codeword(code).unwrap();
            assert_eq!(Even::encode_touched(Even::decode(lane)).unwrap(), lane);
        }
    }
}
