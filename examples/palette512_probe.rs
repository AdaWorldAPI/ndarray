//! Palette512 / SignedMagnitude9 isolated arithmetic probe.
//!
//! This is NOT a 512-centroid codebook. It uses one u8 magnitude per lane
//! and a 64-bit sign plane for 64 lanes: 72 bytes total. Of 512 codewords,
//! sign=1,magnitude=0 is the sole NaN sentinel; the other 511 are numbers.
//! No production codec, public ABI, calibration, or wire format is changed.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeError {
    Nan,
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SignedMagnitude9 {
    magnitude: u8,
    negative: bool,
}

impl SignedMagnitude9 {
    /// Bit 8 is the sign; bits 0..7 are the magnitude.
    fn from_code(code: u16) -> Self {
        assert!(code < 512, "not a 9-bit codeword");
        Self {
            magnitude: code as u8,
            negative: (code & 0x100) != 0,
        }
    }

    fn code(self) -> u16 {
        u16::from(self.magnitude) | ((self.negative as u16) << 8)
    }

    fn value(self) -> Result<i16, ProbeError> {
        if self.negative && self.magnitude == 0 {
            return Err(ProbeError::Nan);
        }
        let value = i16::from(self.magnitude);
        Ok(if self.negative { -value } else { value })
    }

    /// Handle sign in the accumulator without a signed-8-bit operand.
    fn mul_add(self, rhs: Self, acc: i32) -> Result<i32, ProbeError> {
        if (self.negative && self.magnitude == 0)
            || (rhs.negative && rhs.magnitude == 0)
        {
            return Err(ProbeError::Nan);
        }
        let product = i32::from(self.magnitude) * i32::from(rhs.magnitude);
        if self.negative ^ rhs.negative {
            acc.checked_sub(product).ok_or(ProbeError::Overflow)
        } else {
            acc.checked_add(product).ok_or(ProbeError::Overflow)
        }
    }
}

/// Bit i of signs belongs to lane i of magnitudes.
/// repr(C) pins the 72-byte in-memory footprint, NOT a persisted wire format.
#[repr(C)]
#[derive(Clone, Debug)]
struct MagnitudeSignBlock64 {
    magnitudes: [u8; 64],
    signs: u64,
}

impl MagnitudeSignBlock64 {
    fn from_codes(codes: &[u16; 64]) -> Self {
        let mut magnitudes = [0u8; 64];
        let mut signs = 0u64;
        for (i, &code) in codes.iter().enumerate() {
            let lane = SignedMagnitude9::from_code(code);
            magnitudes[i] = lane.magnitude;
            if lane.negative {
                signs |= 1u64 << i;
            }
        }
        Self { magnitudes, signs }
    }

    fn lane(&self, i: usize) -> SignedMagnitude9 {
        SignedMagnitude9 {
            magnitude: self.magnitudes[i],
            negative: ((self.signs >> i) & 1) != 0,
        }
    }

    /// Scalar semantic reference for future SIMD lowering, not a speed claim.
    fn fused_dot(&self, rhs: &Self) -> Result<i32, ProbeError> {
        let mut acc = 0i32;
        for i in 0..64 {
            acc = self.lane(i).mul_add(rhs.lane(i), acc)?;
        }
        Ok(acc)
    }
}

fn main() {
    let plus = SignedMagnitude9::from_code(0x0ff);
    let minus = SignedMagnitude9::from_code(0x1ff);
    let signed_product = plus.mul_add(minus, 0).expect("valid product");
    let lanes = [plus.code(); 64];
    let block = MagnitudeSignBlock64::from_codes(&lanes);
    let dot = block.fused_dot(&block).expect("valid 64-lane dot");
    println!(
        "Palette512 SM9: +{} * {} = {}, 64-lane dot = {}, block = {} B",
        plus.value().expect("valid value"),
        minus.value().expect("valid value"),
        signed_product,
        dot,
        std::mem::size_of::<MagnitudeSignBlock64>()
    );
    println!("Run cargo test --example palette512_probe for exhaustive evidence");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_512_codewords_are_unique_and_one_is_nan() {
        let mut valid = 0;
        let mut nan = 0;
        for code in 0..512u16 {
            let lane = SignedMagnitude9::from_code(code);
            assert_eq!(lane.code(), code);
            match lane.value() {
                Ok(v) => {
                    assert!((-255..=255).contains(&v));
                    let expected = if code < 256 { code as i16 } else { -((code & 255) as i16) };
                    assert_eq!(v, expected);
                    valid += 1;
                }
                Err(ProbeError::Nan) => {
                    assert_eq!(code, 0x100);
                    nan += 1;
                }
                Err(ProbeError::Overflow) => panic!("decoding cannot overflow"),
            }
        }
        assert_eq!((valid, nan), (511, 1));
        assert_eq!(SignedMagnitude9::from_code(0).value(), Ok(0));
        assert_eq!(SignedMagnitude9::from_code(0x100).value(), Err(ProbeError::Nan));
    }

    #[test]
    fn exhaustive_fused_mul_add_matches_independent_signed_oracle() {
        for a_code in 0..512u16 {
            for b_code in 0..512u16 {
                let a = SignedMagnitude9::from_code(a_code);
                let b = SignedMagnitude9::from_code(b_code);
                let expected = match (a.value(), b.value()) {
                    (Ok(x), Ok(y)) => Ok(i32::from(x) * i32::from(y) + 17),
                    _ => Err(ProbeError::Nan),
                };
                assert_eq!(a.mul_add(b, 17), expected, "a={a_code},b={b_code}");
            }
        }
    }

    #[test]
    fn sixty_four_lane_sign_plane_matches_wide_reference() {
        assert_eq!(std::mem::size_of::<MagnitudeSignBlock64>(), 72);
        let lhs = std::array::from_fn(|i| match i % 4 {
            0 => 0x0FF,               // +255
            1 => 0x1FF,               // -255
            2 => 0x000,               // +0
            _ => 0x100 | i as u16,   // negative magnitudes 3..63
        });
        let rhs = std::array::from_fn(|i| match i % 3 {
            0 => 0x1FF,
            1 => 0x0FF,
            _ => 0x000,
        });
        let a = MagnitudeSignBlock64::from_codes(&lhs);
        let b = MagnitudeSignBlock64::from_codes(&rhs);
        let wide_reference: i64 = (0..64)
            .map(|i| i64::from(a.lane(i).value().unwrap()) * i64::from(b.lane(i).value().unwrap()))
            .sum();
        assert_eq!(a.fused_dot(&b), Ok(wide_reference as i32));
        assert_eq!(wide_reference, i64::from(wide_reference as i32));
    }

    #[test]
    fn nan_propagates_even_when_other_operand_is_zero() {
        let zeros = [0u16; 64];
        let mut negzero = zeros;
        negzero[19] = 0x100;
        let z = MagnitudeSignBlock64::from_codes(&zeros);
        let n = MagnitudeSignBlock64::from_codes(&negzero);
        assert_eq!(n.fused_dot(&z), Err(ProbeError::Nan));
        assert_eq!(z.fused_dot(&n), Err(ProbeError::Nan));
    }

    #[test]
    fn overflow_is_not_misreported_as_nan() {
        let plus = SignedMagnitude9::from_code(255);
        let minus = SignedMagnitude9::from_code(511);
        assert_eq!(plus.mul_add(plus, i32::MAX), Err(ProbeError::Overflow));
        assert_eq!(minus.mul_add(plus, i32::MIN), Err(ProbeError::Overflow));
    }
}
