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
//! would violate this contract. The only knobs are the nearest rounding mode
//! (always a type parameter) and the scale.
//!
//! The scale comes in two forms that share one encoder, so equal scales give
//! byte-identical codewords:
//!
//! - [`Codec9`]: the scale is a compile-time [`Calibration`]. Use it where a
//!   table over codewords (`lut9`, a `LUT[a, b]`) must stay valid for every
//!   value ever encoded with it.
//! - [`Codec9Runtime`]: the scale is a value checked once at construction.
//!   Use it where the scale is learned or calibrated from data, for example a
//!   training loop that re-encodes features as its scale moves. It adds the
//!   batch forms [`Codec9Runtime::encode_touched_batch`] and
//!   [`Codec9Runtime::decode_batch`], which write and read the magnitude
//!   lane plus the one-bit presence plane (64 lanes per `u64`).
//!
//! A table built under one scale is not valid under another; a consumer that
//! changes the scale rebuilds its tables.
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
    /// A runtime scale that is zero, negative, NaN or infinite.
    InvalidScale,
    /// Batch slices whose lengths do not describe the same lanes.
    LengthMismatch,
}

/// The one encoder both scale forms call.
#[inline(always)]
fn encode_with_scale<R: NearestRounding9>(value: f32, scale: f32) -> Result<EpiLane9, Codec9Error> {
    if !value.is_finite() {
        return Err(Codec9Error::NonFinite);
    }
    let quantized = R::round(value / scale);
    let magnitude = quantized.abs().clamp(0.0, 255.0) as u8;
    Ok(EpiLane9 {
        magnitude,
        positive: quantized >= 0.0 || magnitude == 0,
    })
}

/// The one decoder both scale forms call.
#[inline(always)]
fn decode_with_scale(lane: EpiLane9, scale: f32) -> f32 {
    match lane.value() {
        None => f32::NAN,
        Some(v) => f32::from(v) * scale,
    }
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
        encode_with_scale::<R>(value, C::SCALE)
    }

    /// Decodes -0 into a floating NaN and all other codes using calibration.
    /// This is an explicit numeric view, not a permission or NULL decision.
    #[must_use]
    pub fn decode(lane: EpiLane9) -> f32 {
        Self::assert_calibration();
        decode_with_scale(lane, C::SCALE)
    }
}

/// DR512 Codec9 with a scale chosen at run time.
///
/// Same storage, polarity, Sym and NaN contract as [`Codec9`], and the same
/// encoder: for a scale equal to a [`Calibration`]'s `SCALE` the codewords are
/// identical. The scale must be positive and finite; there is no zero point.
///
/// ```
/// use ndarray::hpc::codec8::TiesEven;
/// use ndarray::hpc::codec9::Codec9Runtime;
/// let c = Codec9Runtime::<TiesEven>::new(0.5).unwrap();
/// assert_eq!(c.encode_touched(-0.2).unwrap().codeword(), 0x100); // touched zero is +0
/// let (mut m, mut p) = ([0u8; 2], [0u64; 1]);
/// c.encode_touched_batch(&[1.0, -1.0], &mut m, &mut p).unwrap();
/// assert_eq!((m, p[0]), ([2, 2], 0b01));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Codec9Runtime<R: NearestRounding9> {
    scale: f32,
    _rounding: PhantomData<R>,
}

impl<R: NearestRounding9> Codec9Runtime<R> {
    /// Checks the scale once; every later call relies on it.
    pub fn new(scale: f32) -> Result<Self, Codec9Error> {
        if scale > 0.0 && scale < f32::INFINITY {
            Ok(Self {
                scale,
                _rounding: PhantomData,
            })
        } else {
            Err(Codec9Error::InvalidScale)
        }
    }

    /// The scale this codec encodes with.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Encodes one touched finite value; see [`Codec9::encode_touched`].
    pub fn encode_touched(&self, value: f32) -> Result<EpiLane9, Codec9Error> {
        encode_with_scale::<R>(value, self.scale)
    }

    /// Decodes one lane; never-touched becomes NaN.
    #[must_use]
    pub fn decode(&self, lane: EpiLane9) -> f32 {
        decode_with_scale(lane, self.scale)
    }

    /// Encodes `values` as touched lanes: `magnitudes[i]` and bit `i % 64` of
    /// `presence[i / 64]`.
    ///
    /// `magnitudes` must have `values.len()` entries and `presence` exactly
    /// `values.len().div_ceil(64)` words. Presence bits past the last value
    /// are cleared: those lanes do not exist, so they read as never touched.
    /// Every value is checked before anything is written, so a non-finite
    /// value leaves both outputs unchanged.
    pub fn encode_touched_batch(
        &self, values: &[f32], magnitudes: &mut [u8], presence: &mut [u64],
    ) -> Result<(), Codec9Error> {
        if magnitudes.len() != values.len() || presence.len() != values.len().div_ceil(64) {
            return Err(Codec9Error::LengthMismatch);
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Codec9Error::NonFinite);
        }
        for (w, word) in presence.iter_mut().enumerate() {
            let lanes = (values.len() - 64 * w).min(64);
            let mut bits = 0u64;
            for k in 0..lanes {
                let i = 64 * w + k;
                let lane = encode_with_scale::<R>(values[i], self.scale)?;
                magnitudes[i] = lane.magnitude;
                bits |= u64::from(lane.positive) << k;
            }
            *word = bits;
        }
        Ok(())
    }

    /// Decodes a magnitude lane and its presence plane into `out`;
    /// never-touched lanes (P=0, M=0) become NaN.
    ///
    /// `out` must have `magnitudes.len()` entries and `presence` exactly
    /// `magnitudes.len().div_ceil(64)` words.
    pub fn decode_batch(&self, magnitudes: &[u8], presence: &[u64], out: &mut [f32]) -> Result<(), Codec9Error> {
        if out.len() != magnitudes.len() || presence.len() != magnitudes.len().div_ceil(64) {
            return Err(Codec9Error::LengthMismatch);
        }
        for (i, (&m, o)) in magnitudes.iter().zip(out.iter_mut()).enumerate() {
            let positive = presence[i / 64] >> (i % 64) & 1 == 1;
            *o = decode_with_scale(EpiLane9 { magnitude: m, positive }, self.scale);
        }
        Ok(())
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

    struct Tenth;
    impl Calibration for Tenth {
        const SCALE: f32 = 0.1;
        const ZERO_POINT: i32 = 0;
    }

    fn probe_values() -> Vec<f32> {
        let mut v: Vec<f32> = (-3000..=3000).map(|i| i as f32 * 0.0137).collect();
        v.extend([0.0, -0.0, -0.04, -0.05, -0.050001, 0.05, 1e-30, -1e-30, 25.5, -25.5, 1e9, -1e9]);
        v
    }

    #[test]
    fn runtime_scale_gives_the_same_codewords_as_the_compile_time_codec() {
        let even = Codec9Runtime::<TiesEven>::new(0.1).unwrap();
        let away = Codec9Runtime::<TiesAway>::new(0.1).unwrap();
        for v in probe_values() {
            assert_eq!(even.encode_touched(v), Codec9::<TiesEven, Tenth>::encode_touched(v), "{v}");
            assert_eq!(away.encode_touched(v), Codec9::<TiesAway, Tenth>::encode_touched(v), "{v}");
        }
        for code in 0..512u16 {
            let lane = EpiLane9::from_codeword(code).unwrap();
            let (rt, ct) = (even.decode(lane), Codec9::<TiesEven, Tenth>::decode(lane));
            assert!(rt.to_bits() == ct.to_bits() || (rt.is_nan() && ct.is_nan()), "{code}");
        }
    }

    #[test]
    fn runtime_scale_keeps_the_touched_zero_rule() {
        let c = Codec9Runtime::<TiesEven>::new(0.5).unwrap();
        for v in [-0.0, -0.1, -0.249, -0.25] {
            assert_eq!(c.encode_touched(v).unwrap().codeword(), 0x100, "{v}");
        }
        assert_eq!(c.encode_touched(-0.2501).unwrap().codeword(), 0x001);
        assert_eq!(c.encode_touched(f32::NAN), Err(Codec9Error::NonFinite));
    }

    #[test]
    fn runtime_scale_refuses_an_unusable_scale() {
        for s in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(Codec9Runtime::<TiesEven>::new(s).err(), Some(Codec9Error::InvalidScale), "{s}");
        }
        assert!(Codec9Runtime::<TiesEven>::new(f32::MIN_POSITIVE).is_ok());
    }

    #[test]
    fn a_different_scale_gives_different_codewords() {
        // Can-fire half: the scale is used, not ignored.
        let a = Codec9Runtime::<TiesEven>::new(0.1).unwrap();
        let b = Codec9Runtime::<TiesEven>::new(0.2).unwrap();
        assert_ne!(a.encode_touched(1.0), b.encode_touched(1.0));
    }

    #[test]
    fn batch_encode_matches_per_value_encode_and_writes_the_presence_plane() {
        let c = Codec9Runtime::<TiesEven>::new(0.0137).unwrap();
        let values = probe_values();
        let n = values.len();
        let mut mags = vec![0xAAu8; n];
        let mut pres = vec![u64::MAX; n.div_ceil(64)];
        c.encode_touched_batch(&values, &mut mags, &mut pres)
            .unwrap();
        let mut positives = 0;
        for (i, &v) in values.iter().enumerate() {
            let one = c.encode_touched(v).unwrap();
            let bit = pres[i / 64] >> (i % 64) & 1 == 1;
            assert_eq!((mags[i], bit), (one.magnitude(), one.presence()), "lane {i} value {v}");
            positives += bit as usize;
        }
        // Anti-vacuity: both polarities occur, and no touched lane is the sentinel.
        assert!(positives > 0 && positives < n);
        assert!((0..n).all(|i| mags[i] != 0 || pres[i / 64] >> (i % 64) & 1 == 1));
        // Bits past the last lane are cleared: those lanes were never touched.
        let tail = n % 64;
        if tail != 0 {
            assert_eq!(pres[n / 64] >> tail, 0);
        }
    }

    #[test]
    fn batch_round_trip_decodes_the_same_floats() {
        let c = Codec9Runtime::<TiesAway>::new(0.25).unwrap();
        let values: Vec<f32> = (-600..600).map(|i| i as f32 * 0.11).collect();
        let n = values.len();
        let (mut mags, mut pres) = (vec![0u8; n], vec![0u64; n.div_ceil(64)]);
        c.encode_touched_batch(&values, &mut mags, &mut pres)
            .unwrap();
        let mut out = vec![0f32; n];
        c.decode_batch(&mags, &pres, &mut out).unwrap();
        for i in 0..n {
            assert_eq!(out[i], c.decode(c.encode_touched(values[i]).unwrap()), "lane {i}");
        }
    }

    #[test]
    fn batch_decode_turns_never_touched_lanes_into_nan() {
        let c = Codec9Runtime::<TiesEven>::new(1.0).unwrap();
        let mags = [0u8, 0, 7, 7];
        let pres = [0b0110u64];
        let mut out = [0f32; 4];
        c.decode_batch(&mags, &pres, &mut out).unwrap();
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[0.0, 7.0, -7.0]);
    }

    #[test]
    fn batch_encode_refuses_non_finite_input_and_bad_lengths() {
        let c = Codec9Runtime::<TiesEven>::new(1.0).unwrap();
        let (mut m, mut p) = ([0u8; 3], [0u64; 1]);
        assert_eq!(c.encode_touched_batch(&[1.0, f32::INFINITY, 2.0], &mut m, &mut p), Err(Codec9Error::NonFinite));
        assert_eq!((m, p), ([0u8; 3], [0u64; 1]), "a refused batch must not write");
        assert_eq!(c.encode_touched_batch(&[1.0; 3], &mut [0u8; 2], &mut p), Err(Codec9Error::LengthMismatch));
        assert_eq!(c.encode_touched_batch(&[1.0; 65], &mut [0u8; 65], &mut p), Err(Codec9Error::LengthMismatch));
        let mut out = [0f32; 3];
        assert_eq!(c.decode_batch(&m, &[], &mut out), Err(Codec9Error::LengthMismatch));
    }

    #[test]
    fn decode_and_reencode_touched_values_preserve_the_entire_numeric_domain() {
        for code in 1..512u16 {
            let lane = EpiLane9::from_codeword(code).unwrap();
            assert_eq!(Even::encode_touched(Even::decode(lane)).unwrap(), lane);
        }
    }
}
