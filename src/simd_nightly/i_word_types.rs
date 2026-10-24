//! I16x16 / I16x32 / I32x16 / I64x8 portable-simd wrappers — round-3-portable-simd agent #6.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdInt;
use core::simd::{i16x16, i16x32, i32x16, i32x8, i64x4, i64x8};

// ════════════════════════════════════════════════════════════════════
// I16x16 — 16-lane signed 16-bit integer
// ════════════════════════════════════════════════════════════════════

/// 16-lane `i16` SIMD vector backed by `core::simd::i16x16`.
///
/// API mirrors `simd_avx512::I16x16`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I16x16(pub i16x16);

impl I16x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(i16x16::splat(v))
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 16]) -> Self {
        Self(i16x16::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16, "I16x16::from_slice needs ≥16 elements");
        Self(i16x16::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 16, "I16x16::copy_to_slice needs ≥16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Lane-wise signed minimum — mirrors `simd_avx512::I16x16::min` (`_mm512_min_epi16`).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Lane-wise signed maximum — mirrors `simd_avx512::I16x16::max` (`_mm512_max_epi16`).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Lane-wise wrapping add — mirrors `simd_avx512::I16x16::add` (`_mm512_add_epi16`).
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Lane-wise wrapping subtract — mirrors `simd_avx512::I16x16::sub` (`_mm512_sub_epi16`).
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u16` matching the 16-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }

    /// Bitmask of `self > other`, one bit per lane — mirrors `simd_avx512::I16x16::cmp_gt`.
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        self.cmpgt_mask(other)
    }
}

impl PartialEq for I16x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I16x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16x16({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I16x32 — 32-lane signed 16-bit integer
// ════════════════════════════════════════════════════════════════════

/// 32-lane `i16` SIMD vector backed by `core::simd::i16x32`.
///
/// API mirrors `simd_avx512::I16x32`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I16x32(pub i16x32);

impl I16x32 {
    pub const LANES: usize = 32;

    /// Narrow two 32×i16 vectors to one 64×i8 vector with signed saturation: lanes 0..32
    /// come from `self`, lanes 32..64 from `hi`, each clamped to `-128..=127`.
    ///
    /// ```
    /// use ndarray::simd::I16x32;
    /// let mut a = [0i16; 32];
    /// a[..8].copy_from_slice(&[i16::MIN, -129, -128, -1, 1, 127, 128, i16::MAX]);
    /// let v = I16x32::from_array(a).narrow_saturating_i8(I16x32::splat(-300)).to_array();
    /// assert_eq!(v[..8], [-128, -128, -128, -1, 1, 127, 127, 127]);
    /// assert!(v[32..].iter().all(|&x| x == -128));
    /// ```
    #[inline(always)]
    pub fn narrow_saturating_i8(self, hi: Self) -> super::i8_types::I8x64 {
        let (min, max) = (i16x32::splat(-128), i16x32::splat(127));
        let mut out = [0i8; 64];
        out[..32].copy_from_slice(&self.0.simd_clamp(min, max).cast::<i8>().to_array());
        out[32..].copy_from_slice(&hi.0.simd_clamp(min, max).cast::<i8>().to_array());
        super::i8_types::I8x64::from_array(out)
    }

    /// Narrow two 32×i16 vectors to one 64×u8 vector with saturation: lanes 0..32 come
    /// from `self`, lanes 32..64 from `hi`, each clamped to `0..=255`.
    ///
    /// ```
    /// use ndarray::simd::I16x32;
    /// let mut a = [0i16; 32];
    /// a[..8].copy_from_slice(&[i16::MIN, -1, 0, 1, 254, 255, 256, i16::MAX]);
    /// let v = I16x32::from_array(a).narrow_saturating_u8(I16x32::splat(300)).to_array();
    /// assert_eq!(v[..8], [0, 0, 0, 1, 254, 255, 255, 255]);
    /// assert!(v[32..].iter().all(|&x| x == 255));
    /// ```
    #[inline(always)]
    pub fn narrow_saturating_u8(self, hi: Self) -> super::u8_types::U8x64 {
        let (min, max) = (i16x32::splat(0), i16x32::splat(255));
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&self.0.simd_clamp(min, max).cast::<u8>().to_array());
        out[32..].copy_from_slice(&hi.0.simd_clamp(min, max).cast::<u8>().to_array());
        super::u8_types::U8x64::from_array(out)
    }

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(i16x32::splat(v))
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self(i16x32::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 32, "I16x32::from_slice needs ≥32 elements");
        Self(i16x32::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 32] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 32, "I16x32::copy_to_slice needs ≥32 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Lane-wise signed minimum — mirrors `simd_avx512::I16x32::min` (`_mm512_min_epi16`).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Lane-wise signed maximum — mirrors `simd_avx512::I16x32::max` (`_mm512_max_epi16`).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Lane-wise wrapping add — mirrors `simd_avx512::I16x32::add` (`_mm512_add_epi16`).
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Lane-wise wrapping subtract — mirrors `simd_avx512::I16x32::sub` (`_mm512_sub_epi16`).
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    // ── Saturating arithmetic ─────────────────────────────────────

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u32` matching the 32-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }

    /// Bitmask of `self > other`, one bit per lane — mirrors `simd_avx512::I16x32::cmp_gt`.
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        self.cmpgt_mask(other)
    }
}

impl PartialEq for I16x32 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I16x32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16x32({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I32x16 — 16-lane signed 32-bit integer
// ════════════════════════════════════════════════════════════════════

/// 16-lane `i32` SIMD vector backed by `core::simd::i32x16`.
///
/// API mirrors `simd_avx512::I32x16`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I32x16(pub i32x16);

impl I32x16 {
    pub const LANES: usize = 16;

    /// Per-lane `SaturatingRoundingDoublingHighMul` (gemmlowp / NEON `VQRDMULH`):
    /// `round(2*a*b / 2^32)`, ties toward +inf, `MIN * MIN` saturating to `MAX`.
    ///
    /// ```
    /// use ndarray::simd::I32x16;
    /// let a = I32x16::from_array([i32::MIN, 1 << 15, -(1 << 15), 3 << 15, 0, 1, -1, 7, 7, 7, 7, 7, 7, 7, 7, 7]);
    /// let b = I32x16::from_array([i32::MIN, 1 << 15, 1 << 15, 1 << 15, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    /// let v = a.saturating_rounding_doubling_high_mul(b).to_array();
    /// assert_eq!(v[..7], [i32::MAX, 1, 0, 2, 0, 0, 0]);
    /// ```
    #[inline(always)]
    pub fn saturating_rounding_doubling_high_mul(self, b: Self) -> Self {
        let (x, y) = (self.to_array(), b.to_array());
        Self::from_array(core::array::from_fn(|i| crate::simd_fixedpoint::srdhm_i32(x[i], y[i])))
    }

    /// Per-lane rounding arithmetic right shift (ties away from zero) by a
    /// per-lane `exponent` in `0..=31`. Out-of-range exponents are a contract
    /// violation (`debug_assert` in the scalar reference; never UB).
    ///
    /// ```
    /// use ndarray::simd::I32x16;
    /// let x = I32x16::from_array([1, -1, 3, -3, 1 << 30, -(1 << 30), 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// let e = I32x16::from_array([1, 1, 1, 1, 31, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// assert_eq!(x.rounding_divide_by_pot(e).to_array()[..7], [1, -1, 2, -2, 1, -1, 5]);
    /// ```
    #[inline(always)]
    pub fn rounding_divide_by_pot(self, exponent: Self) -> Self {
        let (x, e) = (self.to_array(), exponent.to_array());
        Self::from_array(core::array::from_fn(|i| crate::simd_fixedpoint::rounding_divide_by_pot_i32(x[i], e[i])))
    }

    /// Per-lane TFLM `MultiplyByQuantizedMultiplier` (double rounding) with
    /// per-lane `multiplier` (Q31, `>= 0`) and `shift` (`-31..=30`). `x << shift`
    /// must not overflow (it wraps).
    ///
    /// ```
    /// use ndarray::simd::I32x16;
    /// let x = I32x16::splat(1000);
    /// let m = I32x16::splat(1 << 30);
    /// let s = I32x16::from_array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// assert_eq!(x.multiply_by_quantized_multiplier(m, s).to_array()[..3], [1000, 250, 500]);
    /// ```
    #[inline(always)]
    pub fn multiply_by_quantized_multiplier(self, multiplier: Self, shift: Self) -> Self {
        let (x, m, s) = (self.to_array(), multiplier.to_array(), shift.to_array());
        Self::from_array(core::array::from_fn(|i| {
            crate::simd_fixedpoint::multiply_by_quantized_multiplier_i32(x[i], m[i], s[i])
        }))
    }

    /// Narrow two 16×i32 vectors to one 32×i16 vector with signed saturation: lanes 0..16
    /// come from `self`, lanes 16..32 from `hi`, each clamped to `i16::MIN..=i16::MAX`.
    ///
    /// ```
    /// use ndarray::simd::I32x16;
    /// let lo = I32x16::from_array([i32::MIN, -32769, -32768, -1, 0, 1, 32767, 32768, i32::MAX, 5, -5, 7, -7, 9, -9, 0]);
    /// let hi = I32x16::splat(70_000);
    /// let v = lo.narrow_saturating_i16(hi).to_array();
    /// assert_eq!(v[..9], [-32768, -32768, -32768, -1, 0, 1, 32767, 32767, 32767]);
    /// assert!(v[16..].iter().all(|&x| x == 32767));
    /// ```
    #[inline(always)]
    pub fn narrow_saturating_i16(self, hi: Self) -> I16x32 {
        let (min, max) = (i32x16::splat(i16::MIN as i32), i32x16::splat(i16::MAX as i32));
        let (a, b) = (self.0.simd_clamp(min, max).cast::<i16>(), hi.0.simd_clamp(min, max).cast::<i16>());
        let mut out = [0i16; 32];
        out[..16].copy_from_slice(&a.to_array());
        out[16..].copy_from_slice(&b.to_array());
        I16x32(i16x32::from_array(out))
    }

    /// Cast i32→f32 (equivalent to `portable_simd .cast::<f32>()`).
    #[inline(always)]
    pub fn cast_f32(self) -> super::f32_types::F32x16 {
        super::f32_types::F32x16(self.0.cast::<f32>())
    }

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(i32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 16]) -> Self {
        Self(i32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 16, "I32x16::from_slice needs ≥16 elements");
        Self(i32x16::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 16, "I32x16::copy_to_slice needs ≥16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u16` matching the 16-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }

    /// Per-lane signed greater-than as a packed 16-bit bitmask, LSB-first —
    /// the name the agnostic mask surface (`simd_masking_ops`) calls on every
    /// backend; identical to [`Self::cmpgt_mask`].
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I32x16;
    /// let mut a = [0i32; 16];
    /// a[0] = 1;
    /// a[15] = i32::MAX;
    /// a[7] = i32::MIN;
    /// assert_eq!(I32x16::from_array(a).gt_bitmask(I32x16::splat(0)), 0b1000_0000_0000_0001);
    /// # }
    /// ```
    #[inline(always)]
    pub fn gt_bitmask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }

    /// Load 16 × `i16` and sign-extend to 16 × `i32` (the `VPMOVSXWD`
    /// widening the other backends perform; here `Simd::cast`).
    ///
    /// # Panics
    ///
    /// Panics if `s.len() < 16`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I32x16;
    /// let s: [i16; 16] = core::array::from_fn(|i| (i as i16 - 8) * 1000);
    /// let v = I32x16::from_i16_slice(&s);
    /// assert_eq!(v.to_array()[0], -8000);
    /// assert_eq!(v.to_i16_array(), s);
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_i16_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16, "I32x16::from_i16_slice needs ≥16 elements");
        Self(i16x16::from_slice(s).cast::<i32>())
    }

    /// Narrow 16 × `i32` to 16 × `i16` by truncation (the `VPMOVDW` the
    /// other backends perform; `Simd::cast` truncates like `as i16`).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I32x16;
    /// assert_eq!(I32x16::splat(0x1_0005).to_i16_array()[0], 5); // truncates like `as i16`
    /// # }
    /// ```
    #[inline(always)]
    pub fn to_i16_array(self) -> [i16; 16] {
        self.0.cast::<i16>().to_array()
    }

    /// Lane-wise absolute value, WRAPPING at `i32::MIN` (`|i32::MIN|` stays
    /// `i32::MIN`) — the same semantics as `VPABSD` on the other backends.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I32x16;
    /// assert_eq!(I32x16::splat(-9).abs().to_array()[0], 9);
    /// assert_eq!(I32x16::splat(i32::MIN).abs().to_array()[0], i32::MIN);
    /// # }
    /// ```
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Bit `i` set where lane `i >= 0` (LSB-first) — the sign-bit-clear test
    /// the other backends expose under this name.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I32x16;
    /// let mut a = [-1i32; 16];
    /// a[0] = 0;
    /// a[2] = 5;
    /// assert_eq!(I32x16::from_array(a).cmpge_zero_mask(), 0b101);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpge_zero_mask(self) -> u16 {
        self.0.simd_ge(i32x16::splat(0)).to_bitmask() as u16
    }
}

impl PartialEq for I32x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

/// Lane-wise WRAPPING multiply (low 32 bits of the product) — the `VPMULLD`
/// semantics of the other backends; `core::simd`'s `*` wraps identically.
impl core::ops::Mul for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl core::ops::MulAssign for I32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl core::fmt::Display for I32x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I32x16({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I64x8 — 8-lane signed 64-bit integer
// ════════════════════════════════════════════════════════════════════

/// 8-lane `i64` SIMD vector backed by `core::simd::i64x8`.
///
/// API mirrors `simd_avx512::I64x8`. Miri can execute every method.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I64x8(pub i64x8);

impl I64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(i64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 8]) -> Self {
        Self(i64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 8, "I64x8::from_slice needs ≥8 elements");
        Self(i64x8::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 8, "I64x8::copy_to_slice needs ≥8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all lanes.
    #[inline(always)]
    pub fn reduce_sum(self) -> i64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min / max ───────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Comparisons → bitmask ─────────────────────────────────────

    /// Per-lane equality. Bit i set iff `self[i] == other[i]`.
    /// Returns a `u8` matching the 8-lane count.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane signed greater-than. Bit i set iff `self[i] > other[i]`.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl PartialEq for I64x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl core::fmt::Display for I64x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I64x8({:?})", &self.to_array()[..])
    }
}

// ════════════════════════════════════════════════════════════════════
// I32x8 — 8-lane i32 (256-bit, added 2026-05-20 missing-lanes sweep)
// ════════════════════════════════════════════════════════════════════

/// 8-lane `i32` SIMD vector backed by `core::simd::i32x8`.
///
/// API mirrors `simd_avx512::I32x16` at half-width. Miri-executable.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I32x8(pub i32x8);

impl I32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(i32x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 8, "I32x8::from_slice needs >=8 elements");
        Self(i32x8::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 8]) -> Self {
        Self(i32x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 8, "I32x8::copy_to_slice needs >=8 elements");
        self.0.copy_to_slice(s);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i32 {
        self.0.reduce_sum()
    }
    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        self.0.reduce_min()
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        self.0.reduce_max()
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for I32x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

impl PartialEq for I32x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// I64x4 — 4-lane i64 (256-bit, added 2026-05-20 missing-lanes sweep)
// ════════════════════════════════════════════════════════════════════

/// 4-lane `i64` SIMD vector backed by `core::simd::i64x4`.
///
/// API mirrors `simd_avx512::I64x8` at half-width. Miri-executable.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I64x4(pub i64x4);

impl I64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(i64x4::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 4, "I64x4::from_slice needs >=4 elements");
        Self(i64x4::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 4]) -> Self {
        Self(i64x4::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 4] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 4, "I64x4::copy_to_slice needs >=4 elements");
        self.0.copy_to_slice(s);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i64 {
        self.0.reduce_sum()
    }
    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        self.0.reduce_min()
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        self.0.reduce_max()
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for I64x4 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

impl PartialEq for I64x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}
