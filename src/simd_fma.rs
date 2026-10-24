//! Correctly rounded scalar fused multiply-add for the backends that have no
//! FMA instruction at compile time.
//!
//! `f32::mul_add` / `f64::mul_add` on such a target is a libm call that picks
//! an FMA3, FMA4 or software path by CPUID at run time. The polyfill selects
//! its realization at compile time, so the per-lane `mul_add` of the scalar,
//! AVX-without-FMA and x86-64-v2 backends (and the wasm `F64x8` lanes outside
//! the vector path's range) use [`ExactFma`] instead:
//!
//! * with an FMA instruction (`target_feature = "fma"` on x86, always on
//!   aarch64) it is `mul_add`, which lowers to that instruction;
//! * otherwise `f32` is computed in f64, where the product is exact, with the
//!   sum rounded to odd before the final rounding (Boldo & Melquiond, IEEE TC
//!   2008), and `f64` in integer arithmetic.
//!
//! Both round once, to nearest-even, for every input including subnormals,
//! zeros, infinities and NaN (any NaN result is a NaN, payload unspecified).

// With an FMA instruction only the tests reach the emulation.
#![cfg_attr(
    any(
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
        target_arch = "aarch64"
    ),
    allow(dead_code)
)]

/// `a * b + c` rounded once.
pub(crate) trait ExactFma: Copy {
    fn exact_fma(self, b: Self, c: Self) -> Self;
}

#[cfg(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
    target_arch = "aarch64"
))]
impl ExactFma for f32 {
    #[inline(always)]
    fn exact_fma(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
    target_arch = "aarch64"
))]
impl ExactFma for f64 {
    #[inline(always)]
    fn exact_fma(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(not(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
    target_arch = "aarch64"
)))]
impl ExactFma for f32 {
    #[inline(always)]
    fn exact_fma(self, b: Self, c: Self) -> Self {
        fma_f32(self, b, c)
    }
}

#[cfg(not(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
    target_arch = "aarch64"
)))]
impl ExactFma for f64 {
    #[inline(always)]
    fn exact_fma(self, b: Self, c: Self) -> Self {
        fma_f64(self, b, c)
    }
}

/// f32 FMA through f64: the product of two f32 is exact in f64, and the sum
/// rounded to odd makes the single rounding to f32 correct.
#[inline]
pub(crate) fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64;
    let c = c as f64;
    let s = p + c;
    // Two-sum error of p + c (exact for finite operands).
    let cc = s - p;
    let e = (p - (s - cc)) + (c - cc);
    let mut bits = s.to_bits();
    if e != 0.0 && s.is_finite() && bits & 1 == 0 {
        // One ulp toward the exact sum: up in magnitude when e has the sign of s.
        if (e < 0.0) == (s < 0.0) {
            bits += 1;
        } else {
            bits -= 1;
        }
    }
    f64::from_bits(bits) as f32
}

/// `x = (-1)^neg * m * 2^e` for a finite nonzero f64.
#[inline]
fn split_f64(x: f64) -> (bool, u64, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if exp == 0 {
        (bits >> 63 != 0, frac, -1074)
    } else {
        (bits >> 63 != 0, frac | (1u64 << 52), exp - 1075)
    }
}

/// `m * 2^shift` as a u128 scaled to `2^base`, with every bit shifted out
/// OR-ed into bit 0 (sticky).
#[inline]
fn align(m: u128, exp: i32, base: i32) -> u128 {
    let d = exp - base;
    if d >= 0 {
        m << d
    } else {
        let k = (-d) as u32;
        if k >= 128 {
            (m != 0) as u128
        } else {
            (m >> k) | ((m & ((1u128 << k) - 1) != 0) as u128)
        }
    }
}

/// `2^e` for `-1074 <= e <= 1023`.
#[inline]
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// f64 FMA in integer arithmetic: the exact sum held in a 128-bit window
/// with a sticky bit, then one round to nearest-even.
#[inline]
pub(crate) fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
        // The product is an exact zero, an infinity or NaN; one rounding.
        return a * b + c;
    }
    if !c.is_finite() {
        // A finite product never turns an infinite or NaN c into anything else.
        return c + c;
    }
    if c == 0.0 {
        // a*b is RN(a*b); adding ±0 to it would lose the sign of an underflow.
        return a * b;
    }
    let (sa, ma, ea) = split_f64(a);
    let (sb, mb, eb) = split_f64(b);
    let (sc, mc, ec) = split_f64(c);
    let sp = sa != sb;
    let p = ma as u128 * mb as u128;
    let ep = ea + eb;
    let lp = 128 - p.leading_zeros() as i32;
    let lc = 64 - mc.leading_zeros() as i32;
    // The larger term ends at bit 126 of the window, so it is never shifted
    // right and a carry still fits; only a term far below it gets a sticky bit.
    let top = (ep + lp).max(ec + lc);
    let base = top - 126;
    let pa = align(p, ep, base);
    let ca = align(mc as u128, ec, base);
    let (neg, s) = if sp == sc {
        (sp, pa + ca)
    } else if pa >= ca {
        (sp, pa - ca)
    } else {
        (sc, ca - pa)
    };
    if s == 0 {
        return 0.0;
    }
    let len = 128 - s.leading_zeros() as i32;
    // Keep 53 bits, or fewer where the result is subnormal (granularity 2^-1074).
    let shift = (len - 53).max(-1074 - base);
    let (q, e2) = if shift <= 0 {
        (s << (-shift) as u32, base + shift)
    } else {
        let k = shift as u32;
        let q = if k > 128 {
            // s < 2^128 <= 2^(k-1): below half the smallest step.
            0
        } else {
            let (q, rem, half) = if k == 128 {
                (0u128, s, 1u128 << 127)
            } else {
                (s >> k, s & ((1u128 << k) - 1), 1u128 << (k - 1))
            };
            if rem > half || (rem == half && q & 1 == 1) {
                q + 1
            } else {
                q
            }
        };
        (q, base + shift)
    };
    // q <= 2^53 is exact in f64 and q * 2^e2 is representable or overflows.
    let mag = if e2 > 1023 {
        q as f64 * pow2(1023) * pow2((e2 - 1023).min(1023))
    } else {
        q as f64 * pow2(e2)
    };
    if neg {
        -mag
    } else {
        mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn same32(x: f32, y: f32) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }
    fn same64(x: f64, y: f64) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }

    /// Against the platform `mul_add` (hardware FMA on the CI hosts), over
    /// random bit patterns, near-cancelling c, ties and subnormal results.
    #[test]
    fn emulated_fma_matches_mul_add() {
        let mut r = Rng(0xF3A_0001);
        let mut differs_unfused = 0usize;
        for round in 0..200_000u32 {
            let (a, b, c) = match round % 4 {
                0 => {
                    (f32::from_bits(r.next() as u32), f32::from_bits(r.next() as u32), f32::from_bits(r.next() as u32))
                }
                1 => {
                    let a = f32::from_bits((r.next() as u32 & 0x807F_FFFF) | 0x3F00_0000);
                    let b = f32::from_bits((r.next() as u32 & 0x807F_FFFF) | 0x3F00_0000);
                    (a, b, -(a * b))
                }
                2 => {
                    let x = 1.0f32 + f32::EPSILON * 2048.0;
                    let s = 2f32.powi((r.next() % 100) as i32 - 50);
                    (
                        x * s,
                        x * s,
                        s * s * 2f32.powi(-40 - (r.next() % 16) as i32) * if r.next() & 1 == 0 { 1.0 } else { -1.0 },
                    )
                }
                _ => {
                    // Results in and around the subnormal range.
                    let a = f32::from_bits((r.next() as u32 & 0x807F_FFFF) | (((r.next() % 40) as u32 + 40) << 23));
                    let b = f32::from_bits((r.next() as u32 & 0x807F_FFFF) | (((r.next() % 40) as u32 + 20) << 23));
                    (a, b, f32::from_bits(r.next() as u32 & 0x80FF_FFFF))
                }
            };
            let want = a.mul_add(b, c);
            assert!(
                same32(fma_f32(a, b, c), want),
                "f32 fma({a:e}, {b:e}, {c:e}) = {:e}, want {want:e}",
                fma_f32(a, b, c)
            );
            if !same32(a * b + c, want) {
                differs_unfused += 1;
            }
        }
        assert!(differs_unfused > 1000, "fixtures must separate fused from unfused ({differs_unfused})");

        let mut differs_unfused = 0usize;
        let (ya, yb) = (1.0f64 + f64::EPSILON * 2f64.powi(26), 1.0f64 + f64::EPSILON * 2f64.powi(25));
        for round in 0..200_000u32 {
            let (a, b, c) = match round % 5 {
                0 => (f64::from_bits(r.next()), f64::from_bits(r.next()), f64::from_bits(r.next())),
                1 => {
                    let a = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);
                    let b = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);
                    (a, b, -(a * b))
                }
                2 => {
                    let k = (r.next() % 1200) as i32 - 600;
                    let sign = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
                    (ya * 2f64.powi(k), yb, sign * 2f64.powi(k - 60 - (r.next() % 80) as i32))
                }
                3 => {
                    // Products whose exponent lands near the subnormal range or overflow.
                    let ea = 1 + r.next() % 2046;
                    let target = if r.next() & 1 == 0 {
                        r.next() % 70
                    } else {
                        1990 + r.next() % 70
                    };
                    let eb = (target + 1023).saturating_sub(ea).clamp(1, 2046);
                    let a = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | (ea << 52));
                    let b = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | (eb << 52));
                    (a, b, f64::from_bits(r.next() & 0x801F_FFFF_FFFF_FFFF))
                }
                _ => {
                    let e = |r: &mut Rng| (r.next() % 2047) << 52;
                    let a = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | e(&mut r));
                    let b = f64::from_bits((r.next() & 0x800F_FFFF_FFFF_FFFF) | e(&mut r));
                    (a, b, -(a * b) * (1.0 + f64::EPSILON))
                }
            };
            let want = a.mul_add(b, c);
            assert!(
                same64(fma_f64(a, b, c), want),
                "f64 fma({a:e}, {b:e}, {c:e}) = {:e}, want {want:e}",
                fma_f64(a, b, c)
            );
            if !same64(a * b + c, want) {
                differs_unfused += 1;
            }
        }
        assert!(differs_unfused > 1000, "fixtures must separate fused from unfused ({differs_unfused})");
    }

    #[test]
    fn emulated_fma_special_values() {
        let v64 = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.5,
            3.0,
        ];
        for &a in &v64 {
            for &b in &v64 {
                for &c in &v64 {
                    assert!(same64(fma_f64(a, b, c), a.mul_add(b, c)), "f64 fma({a:e}, {b:e}, {c:e})");
                    let (x, y, z) = (a as f32, b as f32, c as f32);
                    assert!(same32(fma_f32(x, y, z), x.mul_add(y, z)), "f32 fma({x:e}, {y:e}, {z:e})");
                }
            }
        }
    }
}
