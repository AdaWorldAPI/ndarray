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

use crate::simd::F32x16;

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
/// **Not bit-exact across targets.** This is the `f64` scalar for single
/// values (a report statistic, a threshold) and uses the platform `ln`; its
/// last ulp can differ between targets (measured: x86-64 glibc vs wasm32).
/// The substrate path — [`ZGamma`] codes — does not use it and is bit-exact.
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
/// The code is `((z − z_min) / z_range)·254 − 127`, **rounded** to the
/// nearest integer (ties to even) and saturated to the **symmetric** range
/// `[−127, 127]`. `z_min`/`z_range` travel with the codes as 8 bytes of
/// little-endian `f32` ([`ZGamma::to_le_bytes`]).
///
/// Two deliberate departures from `bgz_tensor::fisher_z::FamilyGamma`, whose
/// byte layout this shares but whose code does not:
///
/// - **Rounding, not truncation.** `as i8` truncates toward zero, which makes
///   the code-0 bucket twice as wide as every other and pulls every code half
///   a step toward the centre (measured: mean error `+0.49` below the centre,
///   `−0.49` above it). Rounding removes that median bias.
/// - **Symmetric range with a sentinel.** Two's-complement `i8` is asymmetric
///   (`−128..=127`); using `−128` for data would give one side an extra level.
///   Data codes stay in `−127..=127` and `−128` is reserved as
///   [`ZGamma::NAN_CODE`], so NaN can never be mistaken for a real value.
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

    /// The code for a NaN input. It is outside the data range `−127..=127`,
    /// so no finite cosine ever encodes to it.
    pub const NAN_CODE: i8 = i8::MIN;

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
    /// NaN encodes as [`ZGamma::NAN_CODE`]; values outside the envelope
    /// saturate to `±127`.
    #[inline]
    pub fn encode(&self, cosine: f32) -> i8 {
        let n = (fisher_z_f32(cosine) - self.z_min) / self.z_range;
        quantize(n * 254.0 - 127.0)
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
            let (p, m) = ((one + x).to_array(), (one - x).to_array());
            let lp = F32x16::from_array(core::array::from_fn(|i| ln_det(p[i])));
            let lm = F32x16::from_array(core::array::from_fn(|i| ln_det(m[i])));
            let z = (lp - lm) * half;
            let v = (((z - z_min) / z_range) * scale - shift).to_array();
            *d = core::array::from_fn(|i| quantize(v[i]));
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

/// Round to the nearest code (ties to even), saturate to the symmetric data
/// range, and map NaN to the sentinel. Shared by the scalar and batch paths so
/// they cannot drift.
#[inline]
fn quantize(v: f32) -> i8 {
    if v.is_nan() {
        return ZGamma::NAN_CODE;
    }
    v.round_ties_even().clamp(-127.0, 127.0) as i8
}

/// Scalar `f32` Fisher-Z with the `f32` rim clamp. Private on purpose: its
/// result only ever lives in a register on its way to a code.
///
/// Uses [`ln_det`], not `f32::ln`: the platform libm differs in the last one
/// or two ulp between targets (measured: 624 of 8 539 grid values differ
/// between x86-64 glibc and wasm32), and a one-ulp difference at a rounding
/// boundary flips a code.
#[inline]
fn fisher_z_f32(r: f32) -> f32 {
    let s = clamp_f32(r);
    (ln_det(1.0 + s) - ln_det(1.0 - s)) * 0.5
}

/// Deterministic natural log for positive normal `f32`, bit-identical on
/// every target.
///
/// Built only from IEEE add, sub, mul, div and bit operations, evaluated in a
/// fixed order (Rust never contracts into FMA), so every conforming target
/// produces the same bits. Algorithm: musl / FreeBSD `logf` — split off the
/// exponent `k`, reduce the mantissa to `[√½, √2)`, and evaluate a degree-8
/// polynomial in `s = f/(2+f)`. Error is below one ulp.
///
/// Domain: the Fisher path only ever passes `1 ± s` with `|s| ≤ 1 − 2⁻²⁴`,
/// i.e. `[2⁻²⁴, 2)` — positive and normal. NaN propagates. Zero, negative,
/// subnormal and infinite inputs are outside the domain and not handled.
#[inline]
fn ln_det(x: f32) -> f32 {
    const LN2_HI: f32 = f32::from_bits(0x3f31_7180); // 6.9313812256e-01
    const LN2_LO: f32 = f32::from_bits(0x3717_f7d1); // 9.0580006145e-06
    const LG1: f32 = f32::from_bits(0x3f2a_aaaa); // 0.66666662693
    const LG2: f32 = f32::from_bits(0x3ecc_ce13); // 0.40000972152
    const LG3: f32 = f32::from_bits(0x3e91_e9ee); // 0.28498786688
    const LG4: f32 = f32::from_bits(0x3e78_9e26); // 0.24279078841
    if x.is_nan() {
        return x;
    }
    let mut ix = x.to_bits();
    ix = ix.wrapping_add(0x3f80_0000 - 0x3f35_04f3);
    let k = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007f_ffff) + 0x3f35_04f3;
    let f = f32::from_bits(ix) - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let dk = k as f32;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
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
        // Outside the envelope saturates, symmetrically, rather than wrapping.
        assert_eq!(g.encode(-0.9), -127);
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
            assert!(g.encode(0.3) >= -127);
        }
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

    /// No median bias: over z uniform on the envelope, code 0 holds the same
    /// share as its neighbours and the signed error is centred on both sides.
    /// Truncation toward zero fails this: code 0 gets twice its share and
    /// every code is pulled half a step inward.
    #[test]
    fn rounding_has_no_median_bias() {
        let g = ZGamma {
            z_min: -1.0,
            z_range: 2.0,
        };
        let mut count = [0u32; 256];
        let (mut err_neg, mut n_neg, mut err_pos, mut n_pos) = (0.0f64, 0u32, 0.0f64, 0u32);
        for i in 0..=20_000 {
            let z = -1.0 + 2.0 * i as f32 / 20_000.0;
            let k = g.encode(z.tanh());
            count[(i32::from(k) + 128) as usize] += 1;
            let v = f64::from(((z - g.z_min) / g.z_range) * 254.0 - 127.0);
            let e = f64::from(k) - v;
            if v < -0.5 {
                err_neg += e;
                n_neg += 1;
            } else if v > 0.5 {
                err_pos += e;
                n_pos += 1;
            }
        }
        let (c0, c1) = (count[128], count[129]);
        assert!(c0 * 10 <= c1 * 12, "code 0 over-full: {c0} vs neighbour {c1}");
        assert!((err_neg / f64::from(n_neg)).abs() < 0.05, "bias below centre {}", err_neg / f64::from(n_neg));
        assert!((err_pos / f64::from(n_pos)).abs() < 0.05, "bias above centre {}", err_pos / f64::from(n_pos));
    }

    /// NaN has its own code, and no finite input — however far outside the
    /// envelope — can produce it.
    #[test]
    fn nan_sentinel_is_reserved() {
        let g = ZGamma::fit(&[-0.2, 0.3]);
        assert_eq!(g.encode(f32::NAN), ZGamma::NAN_CODE);
        let mut out = [0i8; 1];
        g.encode_batch(&[f32::NAN], &mut out);
        assert_eq!(out[0], ZGamma::NAN_CODE);
        for c in [-1.0f32, -0.99, -0.5, 0.0, 0.5, 1.0, 5.0, -5.0, f32::INFINITY, f32::NEG_INFINITY] {
            let k = g.encode(c);
            assert!((-127..=127).contains(&k), "c={c} -> {k}");
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

#[cfg(test)]
mod golden {
    use super::*;

    fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    fn grid() -> Vec<f32> {
        // Dense in the centre, and walking every f32 bit pattern near the rim,
        // where ln(1 − r) is most sensitive to the last ulp.
        let mut v: Vec<f32> = (-4096..=4096).map(|i| i as f32 / 4096.0).collect();
        let mut r = 0.999f32;
        while r < 1.0 {
            v.push(r);
            v.push(-r);
            r = f32::from_bits(r.to_bits() + 97);
        }
        v
    }

    /// `ln_det` stays within one ulp of the correctly rounded result over
    /// its whole domain `[2⁻²⁴, 2)`, sampled every 257th bit pattern.
    #[test]
    fn ln_det_is_within_one_ulp() {
        let (lo, hi) = (2f32.powi(-24).to_bits(), 2f32.to_bits());
        let mut worst = 0u32;
        let mut b = lo;
        while b < hi {
            let x = f32::from_bits(b);
            let exact = (f64::from(x).ln()) as f32;
            let got = ln_det(x);
            let ulp = got.to_bits().abs_diff(exact.to_bits());
            // The sign can differ only at x = 1, where both are ±0.
            if exact != 0.0 {
                worst = worst.max(ulp);
            }
            b += 257;
        }
        assert!(worst <= 1, "worst error {worst} ulp");
        assert!(ln_det(f32::NAN).is_nan());
        assert_eq!(ln_det(1.0), 0.0);
    }

    /// Pinned digests over a fixed grid (dense centre plus every 97th f32
    /// bit pattern near the rim). The same constants must hold on every
    /// target — x86-64 (all realizations), aarch64 and wasm32. Measured
    /// before `ln_det`: with libm, 624 of the 8 539 z values differed between
    /// x86-64 glibc and wasm32.
    pub const GOLDEN_CODES: u64 = 0xfb0b_294d_2a65_3dbb;
    pub const GOLDEN_Z32: u64 = 0x2e92_a08e_fd1f_cb69;

    #[test]
    fn digests_are_pinned() {
        let g = grid();
        assert_eq!(g.len(), 8539);
        let env = ZGamma::fit(&g);
        let mut codes = vec![0i8; g.len()];
        env.encode_batch(&g, &mut codes);
        assert_eq!(fnv(codes.iter().map(|&c| c as u8)), GOLDEN_CODES, "codes drifted");
        let z = fnv(g
            .iter()
            .flat_map(|&r| fisher_z_f32(r).to_bits().to_le_bytes()));
        assert_eq!(z, GOLDEN_Z32, "z bits drifted");
    }
}
