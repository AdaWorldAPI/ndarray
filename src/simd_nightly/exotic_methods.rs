//! Scalar fallbacks for U8x32 / U8x64 methods `core::simd` doesn't
//! natively support (cross-lane permute, within-lane shuffle, bitmask
//! blend, lane interleave, nibble-popcount LUT).
//! Round-3-portable-simd agent #11.
#![cfg(feature = "nightly-simd")]

use super::u8_types::{U8x32, U8x64};

// ════════════════════════════════════════════════════════════════════
// U8x64 extension methods
// ════════════════════════════════════════════════════════════════════

impl U8x64 {
    /// Cross-lane byte permute: rearrange all 64 bytes by index vector.
    ///
    /// `idx[i]` selects which byte of `self` appears at position `i`.
    /// The low 6 bits of each index are used (`idx[i] & 63`).
    ///
    /// `core::simd::Swizzle::swizzle` requires a `const N: usize` index
    /// and cannot take a runtime `idx` vector. This scalar fallback via
    /// `to_array()` / `from_array()` matches the AVX-512F-without-VBMI
    /// path in `simd_avx512.rs::U8x64::permute_bytes` (lines ~695–710).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let v: U8x64 = U8x64::from_array(core::array::from_fn(|i| i as u8));
    /// // Identity permute
    /// let idx: U8x64 = U8x64::from_array(core::array::from_fn(|i| i as u8));
    /// assert_eq!(v.permute_bytes(idx).to_array(), v.to_array());
    /// # }
    /// ```
    #[inline]
    pub fn permute_bytes(self, idx: Self) -> Self {
        let src = self.to_array();
        let idx_arr = idx.to_array();
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = src[(idx_arr[i] & 63) as usize];
        }
        Self::from_array(out)
    }

    /// Within-128-bit-lane byte shuffle.
    ///
    /// `self` is the LUT; `idx[i]` selects within the same 16-byte lane.
    /// If the high bit (`0x80`) of `idx[i]` is set, the output lane is
    /// zeroed. Only the low 4 bits select the source index within the lane.
    /// Equivalent to `_mm512_shuffle_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let lut = U8x64::nibble_popcount_lut();
    /// // All zeros → nibble popcount 0
    /// let idx = U8x64::splat(0);
    /// assert_eq!(lut.shuffle_bytes(idx).to_array()[0], 0);
    /// // High bit set → zeroed
    /// let zero_idx = U8x64::splat(0x80);
    /// assert_eq!(lut.shuffle_bytes(zero_idx).to_array()[0], 0);
    /// # }
    /// ```
    #[inline]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        let src = self.to_array();
        let idx_arr = idx.to_array();
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..16 {
                let ix = idx_arr[base + i];
                // High bit set → zero; low 4 bits → index within lane.
                out[base + i] = if ix & 0x80 != 0 {
                    0
                } else {
                    src[base + (ix & 0x0F) as usize]
                };
            }
        }
        Self::from_array(out)
    }

    /// Bitmask-driven blend: select `b[i]` where bit `i` of `mask` is set,
    /// else `a[i]`.
    ///
    /// Mirrors `_mm512_mask_blend_epi8(mask, a, b)` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(10);
    /// let b = U8x64::splat(20);
    /// // Bit 0 set → lane 0 comes from b.
    /// let r = U8x64::mask_blend(1u64, a, b);
    /// assert_eq!(r.to_array()[0], 20);
    /// assert_eq!(r.to_array()[1], 10);
    /// # }
    /// ```
    #[inline]
    pub fn mask_blend(mask: u64, a: Self, b: Self) -> Self {
        let a_arr = a.to_array();
        let b_arr = b.to_array();
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = if mask & (1u64 << i) != 0 { b_arr[i] } else { a_arr[i] };
        }
        Self::from_array(out)
    }

    /// Interleave low bytes within each 128-bit lane.
    ///
    /// For each of the 4 × 16-byte lanes, takes the first 8 bytes of
    /// `self` and `other` and interleaves them:
    /// `[self[0], other[0], self[1], other[1], ..., self[7], other[7]]`.
    ///
    /// Mirrors `_mm512_unpacklo_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(0xAA);
    /// let b = U8x64::splat(0xBB);
    /// let r = a.unpack_lo_epi8(b);
    /// assert_eq!(r.to_array()[0], 0xAA);
    /// assert_eq!(r.to_array()[1], 0xBB);
    /// # }
    /// ```
    #[inline]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        let mut out = [0u8; 64];
        // 4 × 128-bit lanes, each 16 bytes wide.
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2]     = a[base + i];
                out[base + i * 2 + 1] = b[base + i];
            }
        }
        Self::from_array(out)
    }

    /// Interleave high bytes within each 128-bit lane.
    ///
    /// For each of the 4 × 16-byte lanes, takes the last 8 bytes of
    /// `self` and `other` and interleaves them:
    /// `[self[8], other[8], self[9], other[9], ..., self[15], other[15]]`.
    ///
    /// Mirrors `_mm512_unpackhi_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x64;
    /// let a = U8x64::splat(0xAA);
    /// let b = U8x64::splat(0xBB);
    /// let r = a.unpack_hi_epi8(b);
    /// assert_eq!(r.to_array()[0], 0xAA);
    /// assert_eq!(r.to_array()[1], 0xBB);
    /// # }
    /// ```
    #[inline]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2]     = a[base + 8 + i];
                out[base + i * 2 + 1] = b[base + 8 + i];
            }
        }
        Self::from_array(out)
    }
}

// ════════════════════════════════════════════════════════════════════
// U8x32 extension methods
// ════════════════════════════════════════════════════════════════════

impl U8x32 {
    /// Cross-lane byte permute: rearrange all 32 bytes by index vector.
    ///
    /// `idx[i]` selects which byte of `self` appears at position `i`.
    /// The low 5 bits of each index are used (`idx[i] & 31`).
    ///
    /// `core::simd::Swizzle::swizzle` requires a `const N: usize` index
    /// and cannot take a runtime `idx` vector. Scalar fallback.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let v = U8x32::from_array(core::array::from_fn(|i| i as u8));
    /// let idx = U8x32::from_array(core::array::from_fn(|i| (31 - i) as u8));
    /// let r = v.permute_bytes(idx);
    /// assert_eq!(r.to_array()[0], 31);
    /// assert_eq!(r.to_array()[31], 0);
    /// # }
    /// ```
    #[inline]
    pub fn permute_bytes(self, idx: Self) -> Self {
        let src = self.to_array();
        let idx_arr = idx.to_array();
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = src[(idx_arr[i] & 31) as usize];
        }
        Self::from_array(out)
    }

    /// Within-128-bit-lane byte shuffle.
    ///
    /// `self` is the LUT; `idx[i]` selects within the same 16-byte lane.
    /// If the high bit (`0x80`) of `idx[i]` is set, the output lane is
    /// zeroed. Only the low 4 bits select the source index within the lane.
    /// Equivalent to `_mm256_shuffle_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let lut = U8x32::nibble_popcount_lut();
    /// let idx = U8x32::splat(0);
    /// assert_eq!(lut.shuffle_bytes(idx).to_array()[0], 0);
    /// // High bit set → zeroed
    /// let zero_idx = U8x32::splat(0x80);
    /// assert_eq!(lut.shuffle_bytes(zero_idx).to_array()[0], 0);
    /// # }
    /// ```
    #[inline]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        let src = self.to_array();
        let idx_arr = idx.to_array();
        let mut out = [0u8; 32];
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..16 {
                let ix = idx_arr[base + i];
                // High bit set → zero; low 4 bits → index within lane.
                out[base + i] = if ix & 0x80 != 0 {
                    0
                } else {
                    src[base + (ix & 0x0F) as usize]
                };
            }
        }
        Self::from_array(out)
    }

    /// Bitmask-driven blend: select `b[i]` where bit `i` of `mask` is set,
    /// else `a[i]`.
    ///
    /// Uses a 32-bit bitmask (one bit per lane), matching the U8x32 width.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(10);
    /// let b = U8x32::splat(20);
    /// // Bit 0 set → lane 0 comes from b.
    /// let r = U8x32::mask_blend(1u32, a, b);
    /// assert_eq!(r.to_array()[0], 20);
    /// assert_eq!(r.to_array()[1], 10);
    /// # }
    /// ```
    #[inline]
    pub fn mask_blend(mask: u32, a: Self, b: Self) -> Self {
        let a_arr = a.to_array();
        let b_arr = b.to_array();
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = if mask & (1u32 << i) != 0 { b_arr[i] } else { a_arr[i] };
        }
        Self::from_array(out)
    }

    /// Interleave low bytes within each 128-bit lane.
    ///
    /// For each of the 2 × 16-byte lanes, takes the first 8 bytes of
    /// `self` and `other` and interleaves them:
    /// `[self[0], other[0], self[1], other[1], ..., self[7], other[7]]`.
    ///
    /// Mirrors `_mm256_unpacklo_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(0xAA);
    /// let b = U8x32::splat(0xBB);
    /// let r = a.unpack_lo_epi8(b);
    /// assert_eq!(r.to_array()[0], 0xAA);
    /// assert_eq!(r.to_array()[1], 0xBB);
    /// # }
    /// ```
    #[inline]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        let mut out = [0u8; 32];
        // 2 × 128-bit lanes, each 16 bytes wide.
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2]     = a[base + i];
                out[base + i * 2 + 1] = b[base + i];
            }
        }
        Self::from_array(out)
    }

    /// Interleave high bytes within each 128-bit lane.
    ///
    /// For each of the 2 × 16-byte lanes, takes the last 8 bytes of
    /// `self` and `other` and interleaves them:
    /// `[self[8], other[8], self[9], other[9], ..., self[15], other[15]]`.
    ///
    /// Mirrors `_mm256_unpackhi_epi8` semantics.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x32;
    /// let a = U8x32::splat(0xAA);
    /// let b = U8x32::splat(0xBB);
    /// let r = a.unpack_hi_epi8(b);
    /// assert_eq!(r.to_array()[0], 0xAA);
    /// assert_eq!(r.to_array()[1], 0xBB);
    /// # }
    /// ```
    #[inline]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        let mut out = [0u8; 32];
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..8 {
                out[base + i * 2]     = a[base + 8 + i];
                out[base + i * 2 + 1] = b[base + 8 + i];
            }
        }
        Self::from_array(out)
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── U8x64 tests ─────────────────────────────────────────────────

    #[test]
    fn u8x64_permute_bytes_identity() {
        let v = U8x64::from_array(core::array::from_fn(|i| i as u8));
        let idx = U8x64::from_array(core::array::from_fn(|i| i as u8));
        assert_eq!(v.permute_bytes(idx).to_array(), v.to_array());
    }

    #[test]
    fn u8x64_permute_bytes_reverse() {
        let v = U8x64::from_array(core::array::from_fn(|i| i as u8));
        let idx = U8x64::from_array(core::array::from_fn(|i| (63 - i) as u8));
        let r = v.permute_bytes(idx);
        for i in 0..64 {
            assert_eq!(r.to_array()[i], (63 - i) as u8, "lane {i}");
        }
    }

    #[test]
    fn u8x64_shuffle_bytes_lut() {
        let lut = U8x64::nibble_popcount_lut();
        // Index 0..16 → popcount of nibble.
        let idx_arr: [u8; 64] = core::array::from_fn(|i| (i % 16) as u8);
        let idx = U8x64::from_array(idx_arr);
        let r = lut.shuffle_bytes(idx);
        let expected = [0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..16 {
                assert_eq!(r.to_array()[base + i], expected[i], "lane {lane} pos {i}");
            }
        }
    }

    #[test]
    fn u8x64_shuffle_bytes_high_bit_zeroes() {
        let lut = U8x64::nibble_popcount_lut();
        let idx = U8x64::splat(0x80); // high bit set → zero
        assert!(lut.shuffle_bytes(idx).to_array().iter().all(|&b| b == 0));
    }

    #[test]
    fn u8x64_mask_blend_all_a() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        let r = U8x64::mask_blend(0u64, a, b);
        assert!(r.to_array().iter().all(|&x| x == 10));
    }

    #[test]
    fn u8x64_mask_blend_all_b() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        let r = U8x64::mask_blend(u64::MAX, a, b);
        assert!(r.to_array().iter().all(|&x| x == 20));
    }

    #[test]
    fn u8x64_mask_blend_lane0_from_b() {
        let a = U8x64::splat(10);
        let b = U8x64::splat(20);
        let r = U8x64::mask_blend(1u64, a, b);
        assert_eq!(r.to_array()[0], 20);
        assert_eq!(r.to_array()[1], 10);
    }

    #[test]
    fn u8x64_unpack_lo_interleave() {
        let a = U8x64::splat(0xAA);
        let b = U8x64::splat(0xBB);
        let r = a.unpack_lo_epi8(b);
        let arr = r.to_array();
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                assert_eq!(arr[base + i * 2],     0xAA, "lane {lane} pos {}", i * 2);
                assert_eq!(arr[base + i * 2 + 1], 0xBB, "lane {lane} pos {}", i * 2 + 1);
            }
        }
    }

    #[test]
    fn u8x64_unpack_hi_interleave() {
        let mut a_arr = [0u8; 64];
        let mut b_arr = [0u8; 64];
        for lane in 0..4 {
            for i in 0..16 {
                a_arr[lane * 16 + i] = i as u8;
                b_arr[lane * 16 + i] = (i + 100) as u8;
            }
        }
        let a = U8x64::from_array(a_arr);
        let b = U8x64::from_array(b_arr);
        let r = a.unpack_hi_epi8(b);
        let arr = r.to_array();
        for lane in 0..4 {
            let base = lane * 16;
            for i in 0..8 {
                assert_eq!(arr[base + i * 2],     (8 + i) as u8, "lane {lane} a[{i}]");
                assert_eq!(arr[base + i * 2 + 1], (108 + i) as u8, "lane {lane} b[{i}]");
            }
        }
    }

    // ── U8x32 tests ─────────────────────────────────────────────────

    #[test]
    fn u8x32_permute_bytes_reverse() {
        let v = U8x32::from_array(core::array::from_fn(|i| i as u8));
        let idx = U8x32::from_array(core::array::from_fn(|i| (31 - i) as u8));
        let r = v.permute_bytes(idx);
        for i in 0..32 {
            assert_eq!(r.to_array()[i], (31 - i) as u8, "lane {i}");
        }
    }

    #[test]
    fn u8x32_shuffle_bytes_lut() {
        let lut = U8x32::nibble_popcount_lut();
        let idx_arr: [u8; 32] = core::array::from_fn(|i| (i % 16) as u8);
        let idx = U8x32::from_array(idx_arr);
        let r = lut.shuffle_bytes(idx);
        let expected = [0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..16 {
                assert_eq!(r.to_array()[base + i], expected[i], "lane {lane} pos {i}");
            }
        }
    }

    #[test]
    fn u8x32_shuffle_bytes_high_bit_zeroes() {
        let lut = U8x32::nibble_popcount_lut();
        let idx = U8x32::splat(0x80);
        assert!(lut.shuffle_bytes(idx).to_array().iter().all(|&b| b == 0));
    }

    #[test]
    fn u8x32_mask_blend_all_a() {
        let a = U8x32::splat(10);
        let b = U8x32::splat(20);
        let r = U8x32::mask_blend(0u32, a, b);
        assert!(r.to_array().iter().all(|&x| x == 10));
    }

    #[test]
    fn u8x32_mask_blend_all_b() {
        let a = U8x32::splat(10);
        let b = U8x32::splat(20);
        let r = U8x32::mask_blend(u32::MAX, a, b);
        assert!(r.to_array().iter().all(|&x| x == 20));
    }

    #[test]
    fn u8x32_mask_blend_lane0_from_b() {
        let a = U8x32::splat(10);
        let b = U8x32::splat(20);
        let r = U8x32::mask_blend(1u32, a, b);
        assert_eq!(r.to_array()[0], 20);
        assert_eq!(r.to_array()[1], 10);
    }

    #[test]
    fn u8x32_unpack_lo_interleave() {
        let a = U8x32::splat(0xAA);
        let b = U8x32::splat(0xBB);
        let r = a.unpack_lo_epi8(b);
        let arr = r.to_array();
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..8 {
                assert_eq!(arr[base + i * 2],     0xAA);
                assert_eq!(arr[base + i * 2 + 1], 0xBB);
            }
        }
    }

    #[test]
    fn u8x32_unpack_hi_interleave() {
        let mut a_arr = [0u8; 32];
        let mut b_arr = [0u8; 32];
        for lane in 0..2 {
            for i in 0..16 {
                a_arr[lane * 16 + i] = i as u8;
                b_arr[lane * 16 + i] = (i + 100) as u8;
            }
        }
        let a = U8x32::from_array(a_arr);
        let b = U8x32::from_array(b_arr);
        let r = a.unpack_hi_epi8(b);
        let arr = r.to_array();
        for lane in 0..2 {
            let base = lane * 16;
            for i in 0..8 {
                assert_eq!(arr[base + i * 2],     (8 + i) as u8, "lane {lane} a[{i}]");
                assert_eq!(arr[base + i * 2 + 1], (108 + i) as u8, "lane {lane} b[{i}]");
            }
        }
    }
}
