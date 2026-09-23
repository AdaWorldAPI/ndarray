//! The z-space entry points: every similarity or distance enters the
//! substrate as a z-score before anything thresholds, averages, bands or
//! ranks it by margin.
//!
//! Two doors, no third:
//!
//! - **Cosine-shaped** values (cosine, Pearson r, normalized dot products —
//!   anything bounded in `[-1, 1]`) pay the Fisher-Z entry tax:
//!   [`ZGamma`] (codes) / [`fisher_z`] (one scalar). No exceptions, including lab and
//!   calibration code. A raw cosine is not a z-score: its variance shrinks
//!   towards the rim, so equal steps do not mean equal evidence and a fixed
//!   `cos > t` threshold means a different confidence at every `t`.
//! - **Bitpacked** values go through HDR popcount stacking on the Hamming
//!   distance itself, never through a cosine reinterpretation such as
//!   `1 - 2d/N`. Their null distribution is known exactly — `Binomial(N, ½)`
//!   for two random N-bit vectors — so [`hamming_null_z`] needs no samples.
//!
//! The Fisher transform here is the helix convention
//! (`helix::fisher_z::Similarity::fisher_z`, mirrored by `jc::stats::fisher_2z`):
//! `z = ½·(ln(1+r) − ln(1−r))` in `ln` form, with `r` clamped to
//! `[−1+ε, 1−ε]`, `ε = `[`FISHER_CLAMP_EPS`] `= 1e-9`, so every finite input
//! gives a finite z and NaN propagates. Hyperbolic depth `2z` is
//! [`hyperbolic_depth`].
//!
//! **z is never materialized.** Populations are encoded through a
//! [`ZGamma`] envelope straight to `i8` codes and stay there; there is no
//! z buffer and no inverse back to cosine. [`fisher_z`] exists for single
//! scalars — a report statistic, a threshold — not for arrays.

use crate::simd::{simd_ln_f32, F32x16};

/// Rim clamp for the `f64` Fisher transform — identical to
/// `helix::fisher_z::Similarity::CLAMP_EPS`, so this and helix agree bit for
/// bit. At `r = 1 − ε` the transform is `≈ 10.708`.
pub const FISHER_CLAMP_EPS: f64 = 1e-9;

/// Largest `f32` below `1.0` (`1 − 2⁻²⁴`): the `f32` rim clamp.
///
/// `1 − 1e-9` is not representable in `f32` — it rounds to exactly `1.0`,
/// where `ln(1 − r)` is `−∞` — so the batch path clamps to the nearest
/// representable value instead. At that rim the transform is `≈ 8.664`.
pub const FISHER_CLAMP_F32: f32 = 1.0 - f32::EPSILON / 2.0;

/// Fisher-Z transform `z = atanh(r) = ½·(ln(1+r) − ln(1−r))`.
///
/// `r` is clamped to `[−1+ε, 1−ε]` ([`FISHER_CLAMP_EPS`]), so the result is
/// finite for every finite input, including exact `±1` and out-of-range
/// values; NaN propagates. Under the usual bivariate-normal model `z` is
/// approximately normal with variance `1/(n−3)`, which is what makes
/// confidence bands on it mean the same thing everywhere on the scale.
///
/// # Example
///
/// ```
/// use ndarray::hpc::zspace::fisher_z;
///
/// assert_eq!(fisher_z(0.0), 0.0);
/// assert!((fisher_z(0.5) - 0.5_f64.atanh()).abs() < 1e-15);
/// assert!(fisher_z(1.0).is_finite());
/// ```
#[inline]
pub fn fisher_z(r: f64) -> f64 {
    if r.is_nan() {
        return f64::NAN;
    }
    let s = r.clamp(-1.0 + FISHER_CLAMP_EPS, 1.0 - FISHER_CLAMP_EPS);
    0.5 * ((1.0 + s).ln() - (1.0 - s).ln())
}

/// Hyperbolic (Poincaré-disk) depth `2·atanh(r)` — exactly `2 ×` [`fisher_z`],
/// the "Fisher 2z" of helix and `jc`.
#[inline]
pub fn hyperbolic_depth(r: f64) -> f64 {
    2.0 * fisher_z(r)
}

/// The z-space envelope: the metadata that makes an `i8` code mean a z-score.
///
/// **Fisher-Z is never materialized.** There is no buffer of z values and no
/// way back to cosine. A cosine enters, is transformed in registers, is
/// normalized against this envelope, and leaves as an `i8` code. Everything
/// after that — thresholds, bands, ranking — works on codes. A threshold is
/// converted into code space once with [`ZGamma::threshold_code`]; it is never
/// converted back.
///
/// The code is `((z − z_min) / z_range)·254 − 127`, clamped to
/// `[−128, 127]` and truncated toward zero. `z_min`/`z_range` travel with the
/// codes as 8 bytes of little-endian `f32` ([`ZGamma::to_le_bytes`]). The byte
/// layout and the code formula match
/// `bgz_tensor::fisher_z::FamilyGamma`, so an envelope written by one can be
/// read by the other. The two clamp the cosine differently at the rim
/// (bgz-tensor uses `0.9999`, this uses [`FISHER_CLAMP_F32`]), so codes agree
/// bit for bit only for `|cos| ≤ 0.9999`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZGamma {
    /// z of the smallest cosine the envelope covers.
    pub z_min: f32,
    /// `z_max − z_min`; always `> 0`.
    pub z_range: f32,
}

impl ZGamma {
    /// Serialized size: two little-endian `f32`s.
    pub const BYTES: usize = 8;

    /// Fit the envelope to a population of cosines.
    ///
    /// Fisher-Z is monotone, so the z range is the transform of the cosine
    /// range: only the cosine minimum and maximum are scanned, and no z value
    /// is stored. NaN elements are ignored. An empty, all-NaN, or constant
    /// population gets `z_range = 1.0` so that codes stay finite.
    pub fn fit(cosines: &[f32]) -> Self {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &c in cosines {
            if !c.is_nan() {
                lo = lo.min(c);
                hi = hi.max(c);
            }
        }
        if lo > hi {
            return Self {
                z_min: 0.0,
                z_range: 1.0,
            };
        }
        let z_min = fisher_z_f32(lo);
        let range = fisher_z_f32(hi) - z_min;
        Self {
            z_min,
            z_range: if range > 0.0 { range } else { 1.0 },
        }
    }

    /// Encode one cosine as a z-space code.
    ///
    /// NaN encodes as `0`, the code for the envelope midpoint, because Rust's
    /// saturating `as i8` maps NaN to zero.
    #[inline]
    pub fn encode(&self, cosine: f32) -> i8 {
        let n = (fisher_z_f32(cosine) - self.z_min) / self.z_range;
        (n * 254.0 - 127.0).clamp(-128.0, 127.0) as i8
    }

    /// Encode a slice of cosines, sixteen lanes at a time.
    ///
    /// Clamp, Fisher transform and normalization are fused in `F32x16`
    /// registers; the only thing written is the `i8` code. The result is
    /// bit-identical to [`ZGamma::encode`] per element.
    ///
    /// # Panics
    ///
    /// Panics if `cosines.len() != codes.len()`.
    pub fn encode_batch(&self, cosines: &[f32], codes: &mut [i8]) {
        assert_eq!(cosines.len(), codes.len(), "ZGamma::encode_batch: length mismatch");
        let (cs, ts) = cosines.as_chunks::<16>();
        let (cd, td) = codes.as_chunks_mut::<16>();
        let one = F32x16::splat(1.0);
        let half = F32x16::splat(0.5);
        let z_min = F32x16::splat(self.z_min);
        let z_range = F32x16::splat(self.z_range);
        let scale = F32x16::splat(254.0);
        let shift = F32x16::splat(127.0);
        for (s, d) in cs.iter().zip(cd) {
            let x = F32x16::from_array(core::array::from_fn(|i| clamp_f32(s[i])));
            let z = (simd_ln_f32(one + x) - simd_ln_f32(one - x)) * half;
            let v = (((z - z_min) / z_range) * scale - shift).to_array();
            *d = core::array::from_fn(|i| v[i].clamp(-128.0, 127.0) as i8);
        }
        for (s, d) in ts.iter().zip(td) {
            *d = self.encode(*s);
        }
    }

    /// Convert a cosine threshold into code space, once.
    ///
    /// Because the code is monotone in the cosine, `encode(c) >= t` with
    /// `t = threshold_code(c₀)` holds for every `c ≥ c₀`. A cosine slightly
    /// below `c₀` can share its code, so the test is inclusive at the
    /// resolution of one code step.
    #[inline]
    pub fn threshold_code(&self, cosine: f32) -> i8 {
        self.encode(cosine)
    }

    /// The envelope as 8 little-endian bytes: `z_min` then `z_range`.
    pub fn to_le_bytes(&self) -> [u8; Self::BYTES] {
        let mut b = [0u8; Self::BYTES];
        b[..4].copy_from_slice(&self.z_min.to_le_bytes());
        b[4..].copy_from_slice(&self.z_range.to_le_bytes());
        b
    }

    /// Read an envelope written by [`ZGamma::to_le_bytes`].
    pub fn from_le_bytes(b: [u8; Self::BYTES]) -> Self {
        Self {
            z_min: f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            z_range: f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        }
    }
}

/// Scalar `f32` Fisher-Z with the `f32` rim clamp. Private on purpose: its
/// result only ever lives in a register on its way to a code.
#[inline]
fn fisher_z_f32(r: f32) -> f32 {
    let s = clamp_f32(r);
    ((1.0 + s).ln() - (1.0 - s).ln()) * 0.5
}

#[inline]
fn clamp_f32(r: f32) -> f32 {
    // `f32::clamp` returns NaN for a NaN input, so NaN propagates.
    r.clamp(-FISHER_CLAMP_F32, FISHER_CLAMP_F32)
}

/// z-score of a Hamming distance against the random-vector null.
///
/// Two independent uniformly random `n_bits`-bit vectors differ in
/// `Binomial(n_bits, ½)` positions: `μ = n/2`, `σ = √n/2`. The result is
/// `(d − μ)/σ`, so similar vectors score **negative** (`−3` is three sigma
/// closer than chance) with no calibration samples at all. Returns `0.0` for
/// `n_bits == 0`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::zspace::hamming_null_z;
///
/// // 16 384 bits: μ = 8192, σ = 64.
/// assert_eq!(hamming_null_z(8192, 16_384), 0.0);
/// assert_eq!(hamming_null_z(8192 - 3 * 64, 16_384), -3.0);
/// ```
#[inline]
pub fn hamming_null_z(d: u64, n_bits: u64) -> f64 {
    if n_bits == 0 {
        return 0.0;
    }
    let n = n_bits as f64;
    (d as f64 - n / 2.0) / (n.sqrt() / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_z_pins_known_values() {
        assert_eq!(fisher_z(0.0), 0.0);
        assert!((fisher_z(0.5) - 0.549_306_144_334_054_8).abs() < 1e-15);
        // Rim value with ε = 1e-9, computed independently in f64.
        assert!((fisher_z(1.0) - 10.708_206_522_644_144).abs() < 1e-9);
        assert_eq!(hyperbolic_depth(0.5), 2.0 * fisher_z(0.5));
    }

    #[test]
    fn fisher_z_is_odd_and_monotone() {
        let grid: Vec<f64> = (-99..=99).map(|i| f64::from(i) / 100.0).collect();
        for w in grid.windows(2) {
            assert!(fisher_z(w[0]) < fisher_z(w[1]), "not increasing at {}", w[0]);
        }
        for &r in &grid {
            assert_eq!(fisher_z(-r), -fisher_z(r), "not odd at {r}");
        }
    }

    #[test]
    fn fisher_z_is_finite_everywhere_and_propagates_nan() {
        for r in [1.0, -1.0, 2.0, -2.0, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(fisher_z(r).is_finite(), "r = {r}");
        }
        assert_eq!(fisher_z(1.0), fisher_z(5.0), "everything past the rim clamps to it");
        assert!(fisher_z(f64::NAN).is_nan());
    }

    fn cosines() -> Vec<f32> {
        let mut v: Vec<f32> = (0..200)
            .map(|i| ((i * 37) % 199) as f32 / 99.0 - 1.0)
            .collect();
        v[3] = 1.0;
        v[20] = -1.0;
        v[33] = 7.5;
        v[50] = f32::NAN;
        v
    }

    /// The fused batch path must equal the scalar encoder bit for bit, at
    /// every length across the 16-lane boundary, at the rim, and on NaN.
    #[test]
    fn encode_batch_is_bit_identical_to_encode() {
        let src = cosines();
        let g = ZGamma::fit(&src[..150]);
        for n in (0..=40).chain([199, 200]) {
            let mut dst = vec![0i8; n];
            g.encode_batch(&src[..n], &mut dst);
            for (i, (&c, &k)) in src[..n].iter().zip(&dst).enumerate() {
                assert_eq!(k, g.encode(c), "n={n} i={i} c={c}");
            }
        }
    }

    /// The fitted envelope spans the full code range: the population minimum
    /// lands at −127, the maximum at 127.
    #[test]
    fn fit_spans_the_code_range() {
        let pop = [-0.2f32, 0.1, 0.4, 0.9];
        let g = ZGamma::fit(&pop);
        assert_eq!(g.encode(-0.2), -127);
        assert_eq!(g.encode(0.9), 127);
        assert!(g.encode(0.1) > -127 && g.encode(0.4) < 127);
        // Outside the envelope saturates rather than wrapping.
        assert_eq!(g.encode(-0.9), -128);
        assert_eq!(g.encode(0.99), 127);
    }

    /// Codes are spaced in z, not in cosine: two cosine steps of the same
    /// size get more codes near the rim, where the evidence is stronger.
    #[test]
    fn codes_are_spaced_in_z_not_in_cosine() {
        let g = ZGamma::fit(&[0.0, 0.99]);
        let mid = i32::from(g.encode(0.10)) - i32::from(g.encode(0.00));
        let rim = i32::from(g.encode(0.99)) - i32::from(g.encode(0.89));
        assert!(rim > 2 * mid, "rim step {rim} vs centre step {mid}");
    }

    #[test]
    fn fit_is_finite_on_degenerate_populations() {
        for pop in [&[][..], &[f32::NAN][..], &[0.5, 0.5][..], &[1.0, 1.0][..]] {
            let g = ZGamma::fit(pop);
            assert!(g.z_min.is_finite() && g.z_range > 0.0, "{pop:?}");
            assert!(g.encode(0.3) >= -128);
        }
        assert_eq!(ZGamma::fit(&[f32::NAN]).encode(f32::NAN), 0);
    }

    #[test]
    fn threshold_code_is_inclusive_and_monotone() {
        let src = cosines();
        let g = ZGamma::fit(&src);
        let t = g.threshold_code(0.3);
        for &c in src.iter().filter(|c| !c.is_nan()) {
            if c >= 0.3 {
                assert!(g.encode(c) >= t, "c={c}");
            }
        }
    }

    #[test]
    fn envelope_round_trips_through_8_le_bytes() {
        let g = ZGamma {
            z_min: -0.75,
            z_range: 3.25,
        };
        let b = g.to_le_bytes();
        assert_eq!(b.len(), 8);
        assert_eq!(&b[..4], &(-0.75f32).to_le_bytes());
        assert_eq!(ZGamma::from_le_bytes(b), g);
    }

    /// Same formula as `bgz_tensor::fisher_z::FamilyGamma::encode`, written
    /// out independently: codes must agree away from the rim.
    #[test]
    fn matches_the_family_gamma_formula() {
        let g = ZGamma {
            z_min: -0.5,
            z_range: 2.0,
        };
        for i in -99..=99 {
            let c = i as f32 / 100.0;
            let z = c.clamp(-0.9999, 0.9999).atanh();
            let want = (((z - g.z_min) / g.z_range) * 254.0 - 127.0).clamp(-128.0, 127.0) as i8;
            let got = g.encode(c);
            assert!((i32::from(got) - i32::from(want)).abs() <= 1, "c={c} got={got} want={want}");
        }
    }

    #[test]
    fn hamming_null_z_matches_the_binomial_null() {
        assert_eq!(hamming_null_z(8192, 16_384), 0.0);
        assert_eq!(hamming_null_z(8192 - 64, 16_384), -1.0);
        assert_eq!(hamming_null_z(8192 + 192, 16_384), 3.0);
        assert_eq!(hamming_null_z(5, 0), 0.0);
    }
}
