//! Sign-magnitude 9-bit lanes with NULL carried by the presence plane.
//!
//! A follow-up to the Palette512 / SignedMagnitude9 probe (#380). There, the
//! codeword "sign set, magnitude 0" was reserved as the NaN sentinel. Here NaN
//! is not a codeword: a row is NaN (NULL) because its presence bit is clear,
//! whatever its payload holds. That is the NaN/presence rule `ndarray::simd`
//! ships for float lanes (`not_nan_*_to_mask`, `nan_where_clear_*`,
//! `presence_nan_mismatches_*`), applied to an integer lane, where the presence
//! plane is the only carrier.
//!
//! Per 64-lane block: 64 `u8` magnitudes, one `u64` sign plane, one `u64`
//! presence plane. Every magnitude/sign pair is a value, -255..=+255; the
//! writer stores +0 for zero and for missing rows, so each row has one encoding.
//!
//! What it measures:
//!
//! 1. The masked dot product: lanes split by `sign_a ^ sign_b` into a positive
//!    and a negative unsigned accumulator, NULL rows excluded by the presence
//!    AND, `n` the popcount of the same mask. Checked exhaustively against a
//!    decoded oracle. Scalar reference only: the SIMD cost of the u8 x u8
//!    accumulation is not measured here.
//! 2. tanh exactness against i8 symmetric (-127..=127), both as a 256-entry
//!    lookup table and as output quantization alone. tanh is odd, so the
//!    sign-magnitude table is indexed by magnitude and the sign plane passes
//!    through untouched.
//!
//! Run `cargo run --release --example sm9_presence_probe` for the table and
//! `cargo test --release --example sm9_presence_probe` for the checks.

use ndarray::simd::nan_where_clear_f64;

/// One 64-lane block. `presence` bit i clear means lane i is NULL (NaN).
#[derive(Clone, Debug)]
struct Sm9Block {
    magnitudes: [u8; 64],
    signs: u64,
    presence: u64,
}

impl Sm9Block {
    /// Store `values[i]` (`None` = NULL). A NULL or zero row is written as +0.
    fn from_values(values: &[Option<i16>; 64]) -> Self {
        let mut b = Self {
            magnitudes: [0; 64],
            signs: 0,
            presence: 0,
        };
        for (i, v) in values.iter().enumerate() {
            if let Some(v) = *v {
                assert!((-255..=255).contains(&v), "value {v} outside -255..=255");
                b.presence |= 1 << i;
                b.magnitudes[i] = v.unsigned_abs() as u8;
                if v < 0 {
                    b.signs |= 1 << i;
                }
            }
        }
        b
    }

    /// The value of lane `i`, `None` when the presence bit is clear.
    fn value(&self, i: usize) -> Option<i16> {
        if (self.presence >> i) & 1 == 0 {
            return None;
        }
        let m = i16::from(self.magnitudes[i]);
        Some(if (self.signs >> i) & 1 == 1 { -m } else { m })
    }

    /// Masked dot product over the rows both blocks hold (`filter` narrows it
    /// further). Returns `(dot, n)`. The sign is a mask, not arithmetic: lanes
    /// whose signs differ go to the negative accumulator.
    fn masked_dot(&self, rhs: &Self, filter: u64) -> (i64, u32) {
        let sel = self.presence & rhs.presence & filter;
        let negative = self.signs ^ rhs.signs;
        let (mut pos, mut neg) = (0u64, 0u64);
        let mut bits = sel;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            let p = u64::from(self.magnitudes[i]) * u64::from(rhs.magnitudes[i]);
            if (negative >> i) & 1 == 1 {
                neg += p;
            } else {
                pos += p;
            }
            bits &= bits - 1;
        }
        (pos as i64 - neg as i64, sel.count_ones())
    }

    /// Decode to an f64 lane at `scale`, NaN where the presence bit is clear.
    fn decode(&self, scale: f64) -> [f64; 64] {
        let mut out = [0.0f64; 64];
        for (i, o) in out.iter_mut().enumerate() {
            let m = f64::from(self.magnitudes[i]) * scale;
            *o = if (self.signs >> i) & 1 == 1 { -m } else { m };
        }
        nan_where_clear_f64(&[self.presence], &mut out);
        out
    }
}

// ── tanh exactness ──────────────────────────────────────────────────────────

/// i8 symmetric code for `x` at `scale`: round half to even, saturate at ±127.
fn q_i8(x: f64, scale: f64) -> i32 {
    ((x / scale).round_ties_even() as i32).clamp(-127, 127)
}

/// Sign-magnitude code for `x` at `scale`: (negative, magnitude 0..=255);
/// zero is always stored as +0.
fn q_sm9(x: f64, scale: f64) -> (bool, u8) {
    let m = ((x.abs() / scale).round_ties_even() as i32).clamp(0, 255) as u8;
    (x < 0.0 && m > 0, m)
}

/// Error statistics of an approximation `f` to tanh over a grid on [-r, r].
#[derive(Debug, Clone, Copy)]
struct Err {
    max: f64,
    rms: f64,
}

fn tanh_error(r: f64, points: usize, f: impl Fn(f64) -> f64) -> Err {
    let (mut max, mut sq) = (0.0f64, 0.0f64);
    for k in 0..points {
        let x = -r + 2.0 * r * (k as f64) / ((points - 1) as f64);
        let e = (f(x) - x.tanh()).abs();
        max = max.max(e);
        sq += e * e;
    }
    Err {
        max,
        rms: (sq / points as f64).sqrt(),
    }
}

/// The two lookup tables for inputs on [-r, r].
struct Tables {
    r: f64,
    /// i8 table: input code -127..=127 at offset +127, output code.
    lut_i8: [i8; 255],
    /// Sign-magnitude table: input magnitude -> output magnitude. tanh is odd,
    /// so the sign is not looked up.
    lut_sm9: [u8; 256],
}

impl Tables {
    fn new(r: f64) -> Self {
        let mut lut_i8 = [0i8; 255];
        for (k, o) in lut_i8.iter_mut().enumerate() {
            let x = (k as f64 - 127.0) * (r / 127.0);
            *o = q_i8(x.tanh(), 1.0 / 127.0) as i8;
        }
        let mut lut_sm9 = [0u8; 256];
        for (m, o) in lut_sm9.iter_mut().enumerate() {
            let x = m as f64 * (r / 255.0);
            *o = q_sm9(x.tanh(), 1.0 / 255.0).1;
        }
        Self { r, lut_i8, lut_sm9 }
    }

    fn tanh_i8(&self, x: f64) -> f64 {
        let c = q_i8(x, self.r / 127.0);
        f64::from(self.lut_i8[(c + 127) as usize]) / 127.0
    }

    fn tanh_sm9(&self, x: f64) -> f64 {
        let (neg, m) = q_sm9(x, self.r / 255.0);
        let y = f64::from(self.lut_sm9[m as usize]) / 255.0;
        if neg {
            -y
        } else {
            y
        }
    }
}

/// The largest |x| at which the quantized tanh output is still below the top
/// code, i.e. how far into saturation the format still tells values apart.
fn saturation_onset(top_code: f64) -> f64 {
    // tanh(x) rounds to the top code once tanh(x) >= (top - 0.5) / top.
    ((top_code - 0.5) / top_code).atanh()
}

fn main() {
    const POINTS: usize = 2_000_001;
    println!("tanh over [-r, r], {POINTS} points; errors are |approx - tanh(x)| in absolute units");
    println!();
    println!("output quantization only (exact input):");
    let out_i8 = tanh_error(4.0, POINTS, |x| f64::from(q_i8(x.tanh(), 1.0 / 127.0)) / 127.0);
    let out_sm9 = tanh_error(4.0, POINTS, |x| {
        let (neg, m) = q_sm9(x.tanh(), 1.0 / 255.0);
        let y = f64::from(m) / 255.0;
        if neg {
            -y
        } else {
            y
        }
    });
    println!(
        "  i8 sym  max {:.6}  rms {:.6}\n  sm9     max {:.6}  rms {:.6}   (max ratio {:.2}x, rms ratio {:.2}x)",
        out_i8.max,
        out_i8.rms,
        out_sm9.max,
        out_sm9.rms,
        out_i8.max / out_sm9.max,
        out_i8.rms / out_sm9.rms
    );
    println!();
    println!("256-entry lookup table (input and output quantized):");
    println!("     r   i8 max    i8 rms    sm9 max   sm9 rms   max x   rms x");
    for r in [2.0, 3.0, 4.0, 6.0, 8.0] {
        let t = Tables::new(r);
        let a = tanh_error(r, POINTS, |x| t.tanh_i8(x));
        let b = tanh_error(r, POINTS, |x| t.tanh_sm9(x));
        println!(
            "  {r:>4.1}  {:.6}  {:.6}  {:.6}  {:.6}  {:>5.2}x  {:>5.2}x",
            a.max,
            a.rms,
            b.max,
            b.rms,
            a.max / b.max,
            a.rms / b.rms
        );
    }
    println!();
    println!(
        "saturation onset (largest |x| still below the top output code): i8 {:.3}, sm9 {:.3}",
        saturation_onset(127.0),
        saturation_onset(255.0)
    );

    // The masked dot on one block: every third row NULL, signs alternating.
    let a: [Option<i16>; 64] = std::array::from_fn(|i| (i % 3 != 0).then_some(if i % 2 == 0 { 255 } else { -255 }));
    let b: [Option<i16>; 64] = std::array::from_fn(|_| Some(-255));
    let (ba, bb) = (Sm9Block::from_values(&a), Sm9Block::from_values(&b));
    let (dot, n) = ba.masked_dot(&bb, u64::MAX);
    let nulls = ba.decode(1.0 / 255.0).iter().filter(|x| x.is_nan()).count();
    let first = ba.value(1).map_or(0, i16::from);
    println!();
    println!("masked dot: dot {dot}, n {n}, NULL rows {nulls} (decode NaN), lane 1 = {first}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::simd::presence_nan_mismatches_f64;

    /// Every (value or NULL) x (value or NULL) pair: the masked product
    /// equals the decoded product, and NULL contributes nothing to dot or n.
    #[test]
    fn exhaustive_masked_product_matches_decoded_oracle() {
        let mut checked = 0u64;
        let all: Vec<Option<i16>> = std::iter::once(None)
            .chain((-255..=255).map(Some))
            .collect();
        for &a in &all {
            for &b in &all {
                let mut va = [None; 64];
                let mut vb = [None; 64];
                va[37] = a;
                vb[37] = b;
                let (ba, bb) = (Sm9Block::from_values(&va), Sm9Block::from_values(&vb));
                let want = match (a, b) {
                    (Some(x), Some(y)) => (i64::from(x) * i64::from(y), 1),
                    _ => (0, 0),
                };
                assert_eq!(ba.masked_dot(&bb, u64::MAX), want, "a={a:?} b={b:?}");
                checked += 1;
            }
        }
        // 511 values plus NULL on each side.
        assert_eq!(checked, 512 * 512);
    }

    /// Zero and NULL rows are stored as +0, so no row has two encodings, and
    /// the payload under a clear presence bit never reaches the result.
    #[test]
    fn null_and_zero_store_canonical_plus_zero_and_payload_is_ignored() {
        let mut v = [Some(7i16); 64];
        v[3] = None;
        v[4] = Some(0);
        let mut b = Sm9Block::from_values(&v);
        assert_eq!((b.magnitudes[3], (b.signs >> 3) & 1), (0, 0));
        assert_eq!((b.magnitudes[4], (b.signs >> 4) & 1), (0, 0));
        let ones = Sm9Block::from_values(&[Some(1i16); 64]);
        let before = b.masked_dot(&ones, u64::MAX);
        // Garbage under the clear bit changes nothing.
        b.magnitudes[3] = 255;
        b.signs |= 1 << 3;
        assert_eq!(b.masked_dot(&ones, u64::MAX), before);
        assert_eq!(before, (7 * 62, 63));
    }

    /// Random blocks with NULLs and a filter, against a wide decoded oracle.
    #[test]
    fn random_blocks_match_wide_oracle_and_n_is_the_selection() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut nulls_skipped = 0u32;
        for _ in 0..2000 {
            let mut va = [None; 64];
            let mut vb = [None; 64];
            for i in 0..64 {
                let r = next();
                va[i] = (r % 5 != 0).then(|| ((r >> 8) % 511) as i16 - 255);
                vb[i] = ((r >> 32) % 7 != 0).then(|| ((r >> 40) % 511) as i16 - 255);
            }
            let filter = next();
            let (a, b) = (Sm9Block::from_values(&va), Sm9Block::from_values(&vb));
            let mut want = (0i64, 0u32);
            for i in 0..64 {
                if (filter >> i) & 1 == 1 {
                    match (a.value(i), b.value(i)) {
                        (Some(x), Some(y)) => {
                            want.0 += i64::from(x) * i64::from(y);
                            want.1 += 1;
                        }
                        _ => nulls_skipped += 1,
                    }
                }
            }
            assert_eq!(a.masked_dot(&b, filter), want);
        }
        assert!(nulls_skipped > 10_000, "only {nulls_skipped} NULL rows skipped");
    }

    /// Decoding gives a float lane on which the NaN/presence bijection holds.
    #[test]
    fn decode_satisfies_the_float_bijection() {
        let mut v = [Some(-200i16); 64];
        for i in (0..64).step_by(3) {
            v[i] = None;
        }
        let b = Sm9Block::from_values(&v);
        let lane = b.decode(1.0 / 255.0);
        assert_eq!(presence_nan_mismatches_f64(&lane, &[b.presence]), 0);
        assert_eq!(lane.iter().filter(|x| x.is_nan()).count(), 22);
    }

    /// The sign-magnitude table is odd by construction; check it, and check
    /// that it is more exact than the i8 table at every tested range.
    #[test]
    fn sm9_tanh_table_is_odd_and_more_exact_than_i8() {
        for r in [2.0, 4.0, 8.0] {
            let t = Tables::new(r);
            for k in 0..=4000 {
                let x = r * k as f64 / 4000.0;
                assert_eq!(t.tanh_sm9(-x), -t.tanh_sm9(x), "r={r} x={x}");
            }
            let a = tanh_error(r, 200_001, |x| t.tanh_i8(x));
            let b = tanh_error(r, 200_001, |x| t.tanh_sm9(x));
            assert!(b.max < a.max && b.rms < a.rms, "r={r}: sm9 {b:?} vs i8 {a:?}");
        }
    }

    /// The inline i8 quantizer used as the baseline is the shipped codec.
    #[test]
    fn i8_baseline_matches_codec8_sym() {
        use ndarray::hpc::codec8::{Calibration, Codec8, NoNaN, Sym, TiesEven, I8};
        struct Out;
        impl Calibration for Out {
            const SCALE: f32 = 1.0 / 127.0;
            const ZERO_POINT: i32 = 0;
        }
        let xs: Vec<f32> = (0..=2000).map(|k| -1.0 + k as f32 / 1000.0).collect();
        let mut codes = vec![0i8; xs.len()];
        Codec8::<I8, Sym, TiesEven, NoNaN>::encode_checked::<Out>(&xs, &mut codes).unwrap();
        let mut differ = 0;
        for (&x, &c) in xs.iter().zip(&codes) {
            // The codec divides in f32; allow the inline f64 version to differ
            // only at an exact f32/f64 half-tie.
            if q_i8(f64::from(x), 1.0 / 127.0) != i32::from(c) {
                differ += 1;
            }
        }
        assert!(differ <= 2, "{differ} codes differ from Codec8<I8, Sym, TiesEven>");
    }
}
