//! The z-space entry points: every similarity or distance enters the
//! substrate as a z-score before anything thresholds, averages, bands or
//! ranks it by margin.
//!
//! Two doors, no third:
//!
//! - **Cosine-shaped** values (cosine, Pearson r, normalized dot products —
//!   anything bounded in `[-1, 1]`) pay the Fisher-Z entry tax:
//!   [`fisher_z`] / [`fisher_z_f32_batch`]. No exceptions, including lab and
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

/// Inverse of [`fisher_z`]: `r = tanh(z)`.
#[inline]
pub fn fisher_z_inv(z: f64) -> f64 {
    z.tanh()
}

/// Hyperbolic (Poincaré-disk) depth `2·atanh(r)` — exactly `2 ×` [`fisher_z`],
/// the "Fisher 2z" of helix and `jc`.
#[inline]
pub fn hyperbolic_depth(r: f64) -> f64 {
    2.0 * fisher_z(r)
}

/// Fisher-Z over a slice of `f32` similarities, sixteen lanes at a time.
///
/// Each element is clamped to `[−`[`FISHER_CLAMP_F32`]`, `[`FISHER_CLAMP_F32`]`]`
/// and transformed with the same `ln` form as [`fisher_z`], through `F32x16`
/// and [`simd_ln_f32`]. The result is bit-identical to applying that formula
/// to each element in scalar `f32`; NaN propagates.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::zspace::fisher_z_f32_batch;
///
/// let r = [0.0f32, 0.5, -0.5, 1.0];
/// let mut z = [0.0f32; 4];
/// fisher_z_f32_batch(&r, &mut z);
/// assert_eq!(z[0], 0.0);
/// assert_eq!(z[1], -z[2]);
/// assert!(z[3].is_finite());
/// ```
pub fn fisher_z_f32_batch(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "fisher_z_f32_batch: length mismatch");
    let (cs, ts) = src.as_chunks::<16>();
    let (cd, td) = dst.as_chunks_mut::<16>();
    let one = F32x16::splat(1.0);
    let half = F32x16::splat(0.5);
    for (s, d) in cs.iter().zip(cd) {
        let x = F32x16::from_array(core::array::from_fn(|i| clamp_f32(s[i])));
        *d = ((simd_ln_f32(one + x) - simd_ln_f32(one - x)) * half).to_array();
    }
    for (s, d) in ts.iter().zip(td) {
        *d = fisher_z_f32(*s);
    }
}

/// Scalar `f32` Fisher-Z with the batch path's clamp — the per-element
/// definition [`fisher_z_f32_batch`] must reproduce.
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

    #[test]
    fn fisher_z_inverse_round_trips() {
        for i in -99..=99 {
            let r = f64::from(i) / 100.0;
            assert!((fisher_z_inv(fisher_z(r)) - r).abs() < 1e-12, "r = {r}");
        }
    }

    /// The batch path must equal the scalar f32 definition bit for bit, at
    /// every length across the 16-lane boundary and at the rim.
    #[test]
    fn batch_is_bit_identical_to_scalar_f32() {
        let mut src: Vec<f32> = (0..200)
            .map(|i| ((i * 37) % 199) as f32 / 99.0 - 1.0)
            .collect();
        src[3] = 1.0;
        src[20] = -1.0;
        src[33] = 7.5;
        for n in (0..=40).chain([199, 200]) {
            let mut dst = vec![0.0f32; n];
            fisher_z_f32_batch(&src[..n], &mut dst);
            for (i, (&r, &z)) in src[..n].iter().zip(&dst).enumerate() {
                assert_eq!(z.to_bits(), fisher_z_f32(r).to_bits(), "n={n} i={i} r={r}");
                assert!(z.is_finite(), "n={n} i={i} r={r}");
            }
        }
    }

    /// The f32 rim clamp is load-bearing: without it, `r = 1.0` in f32 is
    /// `ln(0) = −∞` and the result is infinite.
    #[test]
    fn f32_rim_is_finite_and_close_to_f64() {
        let mut z = [0.0f32; 2];
        fisher_z_f32_batch(&[1.0, -1.0], &mut z);
        assert!(z[0].is_finite() && z[1].is_finite());
        assert!((f64::from(z[0]) - fisher_z(f64::from(FISHER_CLAMP_F32))).abs() < 1e-3);
        let mut nan = [0.0f32; 1];
        fisher_z_f32_batch(&[f32::NAN], &mut nan);
        assert!(nan[0].is_nan());
    }

    #[test]
    fn hamming_null_z_matches_the_binomial_null() {
        assert_eq!(hamming_null_z(8192, 16_384), 0.0);
        assert_eq!(hamming_null_z(8192 - 64, 16_384), -1.0);
        assert_eq!(hamming_null_z(8192 + 192, 16_384), 3.0);
        assert_eq!(hamming_null_z(5, 0), 0.0);
    }
}
