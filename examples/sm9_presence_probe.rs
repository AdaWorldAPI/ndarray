//! DR512 P × magnitude laboratory. ONE bit plane: P=0,M=0 is -0/NaN/never touched.
//! P=0,M>0 negative; P=1,M=0 +0; P=1,M>0 positive.
//! No independent sign or NULL bit and no second bitmap for WHERE(P=0 && M=0).
//! Odoo/DT64 source authority and alpha-claim admission are separate.
//! Retains the original comparative i8 vs magnitude-LUT numerical experiments.
//!
//! Run cargo test --example sm9_presence_probe for exhaustive lab evidence.

/// One 64-lane DR512 block: u8 magnitude + ONE epistemic presence bit.
/// P=0,M=0 is never touched (-0/NaN), P=0,M>0 negative.
/// P=1,M=0 valid +0, P=1,M>0 positive.
/// Never conflate this numeric code with source-write authorization.
#[derive(Clone, Debug)]
struct Sm9Block {
    magnitudes: [u8; 64],
    presence: u64,
}

impl Sm9Block {
    fn from_values(values: &[Option<i16>; 64]) -> Self {
        let mut out = Self { magnitudes: [0; 64], presence: 0 };
        for (i, v) in values.iter().enumerate() {
            if let Some(v) = *v {
                assert!((-255..=255).contains(&v), "out of DR512 range: {v}");
                out.magnitudes[i] = v.unsigned_abs() as u8;
                // Touched valid zero is ALWAYS +0. Negative nonzero clears P.
                if v >= 0 {
                    out.presence |= 1u64 << i;
                }
            }
        }
        out
    }

    fn value(&self, i: usize) -> Option<i16> {
        let magnitude = i16::from(self.magnitudes[i]);
        let positive = (self.presence >> i) & 1 != 0;
        if !positive && magnitude == 0 { None }
        else { Some(if positive { magnitude } else { -magnitude }) }
    }

    fn never_touched(&self, i: usize) -> bool {
        (self.presence >> i) & 1 == 0 && self.magnitudes[i] == 0
    }

    /// Filter/WHERE is a selector, not proof of source authority.
    fn masked_dot(&self, rhs: &Self, filter: u64) -> (i64, u32) {
        let mut acc = 0i64;
        let mut count = 0u32;
        let mut selected = filter;
        while selected != 0 {
            let i = selected.trailing_zeros() as usize;
            if let (Some(a), Some(b)) = (self.value(i), rhs.value(i)) {
                acc += i64::from(a) * i64::from(b);
                count += 1;
            }
            selected &= selected - 1;
        }
        (acc, count)
    }

    fn decode(&self, scale: f64) -> [f64; 64] {
        std::array::from_fn(|i| match self.value(i) {
            Some(value) => f64::from(value) * scale,
            None => f64::NAN,
        })
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

    #[test]
    fn never_touched_negative_zero_positive_zero_and_negative_values() {
        let mut values = [Some(7i16); 64];
        values[3] = None;
        values[4] = Some(0);
        values[5] = Some(-255);
        let block = Sm9Block::from_values(&values);
        assert_eq!((block.presence >> 3) & 1, 0);
        assert_eq!((block.presence >> 4) & 1, 1);
        assert_eq!((block.presence >> 5) & 1, 0);
        assert!(block.never_touched(3));
        assert!(!block.never_touched(4));
        assert!(!block.never_touched(5));
        assert_eq!(block.value(3), None);
        assert_eq!(block.value(4), Some(0));
        assert_eq!(block.value(5), Some(-255));
        let ones = Sm9Block::from_values(&[Some(1i16); 64]);
        assert_eq!(block.masked_dot(&ones, u64::MAX), (7 * 61 - 255, 63));
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

    /// Decoding returns NaN precisely for P=0,M=0, not every P=0 row.
    #[test]
    fn decode_satisfies_the_float_bijection() {
        let mut v = [Some(-200i16); 64];
        for i in (0..64).step_by(3) {
            v[i] = None;
        }
        let b = Sm9Block::from_values(&v);
        let lane = b.decode(1.0 / 255.0);
        for (i, x) in lane.iter().enumerate() {
            assert_eq!(x.is_nan(), b.never_touched(i), "row {i}");
        }
        assert_eq!(lane.iter().filter(|x| x.is_nan()).count(), 22);
    }

    /// The 3σ gate: at a tolerance fixed before measuring, the fraction of
    /// inputs whose table error stays within it must reach 0.9973 (the
    /// two-sided 3σ coverage). On 200,000 independent uniform inputs over
    /// [-4, 4], sign-magnitude passes and i8 fails, and the paired mean
    /// difference of the errors is far above 3 standard errors.
    ///
    /// The tolerance 0.010 sits between the two formats' measured 99.73%
    /// error quantiles at r = 4 (about 0.0079 and 0.0157), so neither half of
    /// the gate is true by construction.
    #[test]
    fn three_sigma_coverage_gate_passes_for_sm9_and_fails_for_i8() {
        const TOL: f64 = 0.010;
        const N: usize = 200_000;
        const COVERAGE_3SIGMA: f64 = 0.9973;
        let t = Tables::new(4.0);
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        let (mut in8, mut in9) = (0usize, 0usize);
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for _ in 0..N {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // 53 random bits to a uniform x in [-4, 4).
            let x = -4.0 + 8.0 * ((s >> 11) as f64 / (1u64 << 53) as f64);
            let e8 = (t.tanh_i8(x) - x.tanh()).abs();
            let e9 = (t.tanh_sm9(x) - x.tanh()).abs();
            in8 += usize::from(e8 <= TOL);
            in9 += usize::from(e9 <= TOL);
            let d = e8 - e9;
            sum += d;
            sum_sq += d * d;
        }
        let (cov8, cov9) = (in8 as f64 / N as f64, in9 as f64 / N as f64);
        let mean = sum / N as f64;
        let var = (sum_sq - N as f64 * mean * mean) / (N - 1) as f64;
        let z = mean / (var / N as f64).sqrt();
        eprintln!("3σ gate: tol {TOL}, n {N}: sm9 {cov9:.4}, i8 {cov8:.4}, paired z {z:.1}");
        assert!(cov9 >= COVERAGE_3SIGMA, "sm9 coverage {cov9:.4} below 3σ at tol {TOL}");
        assert!(cov8 < COVERAGE_3SIGMA, "i8 coverage {cov8:.4} unexpectedly reaches 3σ at tol {TOL}");
        assert!(z > 3.0, "paired z {z:.1} not significant at 3σ");
        // Anti-vacuity: i8 is not merely a hair below the line.
        assert!(cov8 < 0.99, "i8 coverage {cov8:.4}: the gate barely separates the formats");
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
