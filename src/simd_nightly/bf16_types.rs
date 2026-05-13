//! BF16x16 / BF16x8 portable-simd wrappers (scalar emulation — no core::simd
//! half-precision types). Round-3-portable-simd agent #8.
#![cfg(feature = "nightly-simd")]

/// 16-lane BF16 vector backed by `[u16; 16]` (scalar emulation).
///
/// `core::simd` has no native half-precision type, so bit patterns are stored
/// as `u16` and operations upcast through `f32` where needed.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct BF16x16(pub [u16; 16]);

/// 8-lane BF16 vector backed by `[u16; 8]` (scalar emulation).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct BF16x8(pub [u16; 8]);

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

/// Truncate an `f32` to BF16 bit pattern (drop low 16 mantissa bits).
#[inline(always)]
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Reconstruct an `f32` from a BF16 bit pattern (lossless).
#[inline(always)]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// BF16x16
// ---------------------------------------------------------------------------

impl BF16x16 {
    /// Number of BF16 lanes.
    pub const LANES: usize = 16;

    /// Broadcast `v` (converted f32 → BF16 via truncation) across all 16 lanes.
    #[inline]
    pub fn splat(v: f32) -> Self {
        Self([f32_to_bf16_bits(v); 16])
    }

    /// Load 16 BF16 bit-patterns from a `u16` slice (must have `len >= 16`).
    #[inline]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 16, "BF16x16::from_slice: need >= 16 elements, got {}", s.len());
        let mut arr = [0u16; 16];
        arr.copy_from_slice(&s[..16]);
        Self(arr)
    }

    /// Construct from an array of 16 BF16 bit-patterns.
    #[inline]
    pub fn from_array(arr: [u16; 16]) -> Self {
        Self(arr)
    }

    /// Return the underlying array of 16 BF16 bit-patterns.
    #[inline]
    pub fn to_array(self) -> [u16; 16] {
        self.0
    }

    /// Write the 16 BF16 bit-patterns into `dst` (must have `len >= 16`).
    #[inline]
    pub fn copy_to_slice(self, dst: &mut [u16]) {
        assert!(dst.len() >= 16, "BF16x16::copy_to_slice: need >= 16 elements, got {}", dst.len());
        dst[..16].copy_from_slice(&self.0);
    }

    /// Convert all 16 BF16 lanes to `f32` (lossless: zero-extend bits 15:0 → bits 31:16).
    #[inline]
    pub fn to_f32_lossy(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = bf16_bits_to_f32(self.0[i]);
        }
        out
    }

    /// Build from 16 `f32` values, truncating (not rounding) the low 16 mantissa bits.
    #[inline]
    pub fn from_f32_truncate(arr: [f32; 16]) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            out[i] = f32_to_bf16_bits(arr[i]);
        }
        Self(out)
    }
}

// ---------------------------------------------------------------------------
// BF16x8
// ---------------------------------------------------------------------------

impl BF16x8 {
    /// Number of BF16 lanes.
    pub const LANES: usize = 8;

    /// Broadcast `v` (converted f32 → BF16 via truncation) across all 8 lanes.
    #[inline]
    pub fn splat(v: f32) -> Self {
        Self([f32_to_bf16_bits(v); 8])
    }

    /// Load 8 BF16 bit-patterns from a `u16` slice (must have `len >= 8`).
    #[inline]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8, "BF16x8::from_slice: need >= 8 elements, got {}", s.len());
        let mut arr = [0u16; 8];
        arr.copy_from_slice(&s[..8]);
        Self(arr)
    }

    /// Construct from an array of 8 BF16 bit-patterns.
    #[inline]
    pub fn from_array(arr: [u16; 8]) -> Self {
        Self(arr)
    }

    /// Return the underlying array of 8 BF16 bit-patterns.
    #[inline]
    pub fn to_array(self) -> [u16; 8] {
        self.0
    }

    /// Write the 8 BF16 bit-patterns into `dst` (must have `len >= 8`).
    #[inline]
    pub fn copy_to_slice(self, dst: &mut [u16]) {
        assert!(dst.len() >= 8, "BF16x8::copy_to_slice: need >= 8 elements, got {}", dst.len());
        dst[..8].copy_from_slice(&self.0);
    }

    /// Convert all 8 BF16 lanes to `f32` (lossless).
    #[inline]
    pub fn to_f32_lossy(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = bf16_bits_to_f32(self.0[i]);
        }
        out
    }

    /// Build from 8 `f32` values, truncating the low 16 mantissa bits.
    #[inline]
    pub fn from_f32_truncate(arr: [f32; 8]) -> Self {
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = f32_to_bf16_bits(arr[i]);
        }
        Self(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BF16x16 ------------------------------------------------------------

    #[test]
    fn bf16x16_splat_roundtrip() {
        let v = BF16x16::splat(3.14f32);
        let f32s = v.to_f32_lossy();
        // splat then to_f32_lossy must all be identical
        assert!(f32s.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn bf16x16_from_f32_truncate_to_f32_lossy() {
        let input: [f32; 16] = core::array::from_fn(|i| i as f32 * 1.5);
        let vec = BF16x16::from_f32_truncate(input);
        let out = vec.to_f32_lossy();
        for i in 0..16 {
            // truncation then expand must match bit-shift arithmetic
            let expected = f32::from_bits((input[i].to_bits() >> 16) << 16);
            assert_eq!(out[i], expected, "lane {i} mismatch");
        }
    }

    #[test]
    fn bf16x16_from_slice_copy_to_slice_roundtrip() {
        let bits: [u16; 16] = core::array::from_fn(|i| (i as u16) * 0x100);
        let v = BF16x16::from_slice(&bits);
        let mut out = [0u16; 16];
        v.copy_to_slice(&mut out);
        assert_eq!(bits, out);
    }

    #[test]
    fn bf16x16_from_array_to_array_roundtrip() {
        let arr: [u16; 16] = core::array::from_fn(|i| i as u16 + 1);
        assert_eq!(BF16x16::from_array(arr).to_array(), arr);
    }

    #[test]
    fn bf16x16_lanes_const() {
        assert_eq!(BF16x16::LANES, 16);
    }

    // -- BF16x8 -------------------------------------------------------------

    #[test]
    fn bf16x8_splat_roundtrip() {
        let v = BF16x8::splat(2.71f32);
        let f32s = v.to_f32_lossy();
        assert!(f32s.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn bf16x8_from_f32_truncate_to_f32_lossy() {
        let input: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5);
        let vec = BF16x8::from_f32_truncate(input);
        let out = vec.to_f32_lossy();
        for i in 0..8 {
            let expected = f32::from_bits((input[i].to_bits() >> 16) << 16);
            assert_eq!(out[i], expected, "lane {i} mismatch");
        }
    }

    #[test]
    fn bf16x8_from_slice_copy_to_slice_roundtrip() {
        let bits: [u16; 8] = core::array::from_fn(|i| (i as u16) * 0x200);
        let v = BF16x8::from_slice(&bits);
        let mut out = [0u16; 8];
        v.copy_to_slice(&mut out);
        assert_eq!(bits, out);
    }

    #[test]
    fn bf16x8_from_array_to_array_roundtrip() {
        let arr: [u16; 8] = core::array::from_fn(|i| i as u16 + 10);
        assert_eq!(BF16x8::from_array(arr).to_array(), arr);
    }

    #[test]
    fn bf16x8_lanes_const() {
        assert_eq!(BF16x8::LANES, 8);
    }

    // -- Known BF16 bit patterns -------------------------------------------

    #[test]
    fn bf16_one_point_zero() {
        // 1.0f32 = 0x3F800000; BF16 = 0x3F80
        let v = BF16x16::splat(1.0f32);
        assert_eq!(v.0[0], 0x3F80u16);
        let f32s = v.to_f32_lossy();
        assert_eq!(f32s[0], 1.0f32);
    }

    #[test]
    fn bf16_negative_one() {
        // -1.0f32 = 0xBF800000; BF16 = 0xBF80
        let v = BF16x8::splat(-1.0f32);
        assert_eq!(v.0[0], 0xBF80u16);
        let f32s = v.to_f32_lossy();
        assert_eq!(f32s[0], -1.0f32);
    }
}
