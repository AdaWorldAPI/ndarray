//! F16x16 portable-simd wrapper (scalar IEEE-754 binary16 emulation — no
//! core::simd half-precision). Round-3-portable-simd agent #9.
#![cfg(feature = "nightly-simd")]

/// 16-lane IEEE-754 binary16 (half-precision) vector backed by `[u16; 16]`.
///
/// `core::simd` has no native `f16` lane type, so this is a full scalar
/// emulation.  All conversions use the same IEEE-754-correct
/// round-to-nearest-even logic as `src/hpc/quantized.rs` (see
/// `F16::from_f32_rounded`, lines 193-264, and `F16::to_f32`, lines 267-301).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F16x16(pub [u16; 16]);

impl F16x16 {
    /// Number of lanes.
    pub const LANES: usize = 16;

    // ── IEEE-754 binary16 scalar helpers ────────────────────────────────────
    //
    // Copied from src/hpc/quantized.rs, `F16::from_f32_rounded` (lines 193-264)
    // and `F16::to_f32` (lines 267-301).

    /// Convert a single `f32` value to its IEEE-754 binary16 bit pattern
    /// (round-to-nearest-even).
    ///
    /// Logic mirrors `F16::from_f32_rounded` in `src/hpc/quantized.rs:193`.
    #[inline]
    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        if exp == 0xFF {
            // Inf / NaN
            if mant == 0 {
                return sign | 0x7C00;
            }
            let mant16 = (mant >> 13) as u16;
            let mant16 = if mant16 == 0 { 0x200 } else { mant16 | 0x200 };
            return sign | 0x7C00 | mant16;
        }

        let new_exp = exp - 127 + 15;

        if new_exp >= 0x1F {
            return sign | 0x7C00; // overflow → Inf
        }
        if new_exp >= 1 {
            // Normal range — round-to-nearest-even on the 13 dropped bits.
            let half_bits = mant & 0x0000_1FFF;
            let truncated = (mant >> 13) as u32;
            let half = 0x1000u32;
            let round_up = if half_bits > half {
                1u32
            } else if half_bits < half {
                0
            } else {
                truncated & 1 // tie → round to even
            };
            let mant16 = truncated + round_up;
            if mant16 == 0x400 {
                // mantissa overflow bumps exponent
                let new_exp = new_exp + 1;
                if new_exp >= 0x1F {
                    return sign | 0x7C00;
                }
                return sign | ((new_exp as u16) << 10);
            }
            return sign | ((new_exp as u16) << 10) | (mant16 as u16);
        }

        // Subnormal / underflow
        if new_exp < -10 {
            return sign; // underflow → ±0
        }
        let mant_full = mant | 0x0080_0000; // 24-bit with implicit 1
        let shift = (14 - new_exp) as u32;
        let truncated = mant_full >> shift;
        let dropped_mask = (1u32 << shift) - 1;
        let dropped = mant_full & dropped_mask;
        let half = 1u32 << (shift - 1);
        let round_up = if dropped > half {
            1u32
        } else if dropped < half {
            0
        } else {
            truncated & 1
        };
        let mant16 = (truncated + round_up) as u16;
        sign | mant16
    }

    /// Convert an IEEE-754 binary16 bit pattern to `f32` (lossless).
    ///
    /// Logic mirrors `F16::to_f32` in `src/hpc/quantized.rs:267`.
    #[inline]
    fn f16_bits_to_f32(h: u16) -> f32 {
        let h = h as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x03FF;

        let bits = if exp == 0 {
            if mant == 0 {
                sign // ±0
            } else {
                // Subnormal: normalize
                let mut m = mant;
                let mut e: i32 = 1;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                let m = m & 0x03FF;
                let new_exp = (e - 1 + 127 - 14) as u32;
                sign | (new_exp << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            if mant == 0 {
                sign | 0x7F80_0000 // ±Inf
            } else {
                sign | 0x7F80_0000 | (mant << 13) // NaN
            }
        } else {
            let new_exp = exp + (127 - 15);
            sign | (new_exp << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    // ── Constructors ────────────────────────────────────────────────────────

    /// Broadcast `v` (converted from `f32` to IEEE-754 binary16) across all 16 lanes.
    #[inline]
    pub fn splat(v: f32) -> Self {
        F16x16([Self::f32_to_f16_bits(v); 16])
    }

    /// Load 16 binary16 values from a `u16` slice (must have `len >= 16`).
    #[inline]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 16, "F16x16::from_slice: need >= 16 elements, got {}", s.len());
        let mut arr = [0u16; 16];
        arr.copy_from_slice(&s[..16]);
        F16x16(arr)
    }

    /// Construct directly from a raw `[u16; 16]` array of binary16 bit patterns.
    #[inline]
    pub fn from_array(arr: [u16; 16]) -> Self {
        F16x16(arr)
    }

    /// Return the raw `[u16; 16]` array of binary16 bit patterns.
    #[inline]
    pub fn to_array(self) -> [u16; 16] {
        self.0
    }

    /// Write all 16 binary16 bit patterns into a `u16` slice (must have `len >= 16`).
    #[inline]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 16, "F16x16::copy_to_slice: need >= 16 elements, got {}", s.len());
        s[..16].copy_from_slice(&self.0);
    }

    // ── Conversions ─────────────────────────────────────────────────────────

    /// Upcast all 16 binary16 lanes to `f32`.
    #[inline]
    pub fn to_f32_array(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = Self::f16_bits_to_f32(self.0[i]);
        }
        out
    }

    /// Convert 16 `f32` values to binary16 and pack into a new `F16x16`.
    #[inline]
    pub fn from_f32_array(arr: [f32; 16]) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            out[i] = Self::f32_to_f16_bits(arr[i]);
        }
        F16x16(out)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_roundtrip() {
        let v = F16x16::splat(1.0_f32);
        let arr = v.to_f32_array();
        for lane in arr {
            assert!((lane - 1.0).abs() < 1e-5, "splat(1.0) lane = {lane}");
        }
    }

    #[test]
    fn from_array_to_array_identity() {
        let raw = [0x3C00u16; 16]; // 1.0 in binary16
        let v = F16x16::from_array(raw);
        assert_eq!(v.to_array(), raw);
    }

    #[test]
    fn from_slice_copy_to_slice_roundtrip() {
        let src: [u16; 16] = core::array::from_fn(|i| i as u16 + 0x3C00);
        let v = F16x16::from_slice(&src);
        let mut dst = [0u16; 16];
        v.copy_to_slice(&mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn from_f32_array_to_f32_array_roundtrip() {
        let inputs: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5);
        let v = F16x16::from_f32_array(inputs);
        let out = v.to_f32_array();
        for (i, (&orig, &back)) in inputs.iter().zip(out.iter()).enumerate() {
            assert!(
                (orig - back).abs() < 0.001,
                "lane {i}: {orig} → f16 → {back}"
            );
        }
    }

    #[test]
    fn special_values_inf_nan() {
        let v = F16x16::splat(f32::INFINITY);
        for lane in v.to_f32_array() {
            assert!(lane.is_infinite() && lane > 0.0);
        }
        let v = F16x16::splat(f32::NAN);
        for lane in v.to_f32_array() {
            assert!(lane.is_nan());
        }
    }

    #[test]
    fn lanes_constant() {
        assert_eq!(F16x16::LANES, 16);
    }
}
