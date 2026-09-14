//! W1a consumer-contract primitives on the portable-simd backend.
//!
//! `I8x16` / `U16x8` / `U8x8` / `palette_lookup_u8x8` / `prefetch_read_t*` /
//! `batch_packed_i4_16` exist on every intrinsics backend (`simd_avx512`,
//! `simd_avx2`, `simd_neon`, `simd_wasm`, `simd_scalar`); until 2026-09-14 the
//! `nightly-simd` realization had none of them, so a consumer that compiled
//! against `crate::simd::I8x16` broke the moment the feature was turned on.
//! The polyfill law is that every backend file realizes the whole facade, so
//! they land here as thin `core::simd` bodies — never as a `#[cfg]` that hides
//! the facade tests under this realization.
//!
//! Semantics are the scalar backend's, bit for bit: `from_i4_packed_u64`
//! sign-extends nibble `0x8` to `-8`, `saturating_abs(i8::MIN) == i8::MAX`
//! (the VPABSB correction in `vertical-simd-consumer-contract.md`), the
//! gathers are bounds-checked in debug and return `0` for an out-of-range
//! index in release, and the prefetches are documented no-ops (a hint has no
//! observable result, and `core::simd` carries no prefetch).
#![cfg(feature = "nightly-simd")]

use core::fmt;
use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::{SimdInt, SimdUint};
use core::simd::{i8x16 as core_i8x16, u16x8 as core_u16x8, u64x16, u8x8 as core_u8x8, Simd};

// ── W1a-#1: I8x16 + lane_i8 + from_i4_packed_u64 ────────────────────────────

/// 16-lane `i8` vector backed by `core::simd::i8x16`.
///
/// Mirrors `simd_scalar::I8x16` / `simd_neon::I8x16` so consumer code is
/// backend-agnostic; every method executes under miri.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::I8x16;
/// let v = I8x16::from_i4_packed_u64(0x8);
/// assert_eq!(v.lane_i8::<0>(), -8);
/// assert_eq!(v.lane_i8::<1>(), 0);
/// # }
/// ```
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I8x16(pub core_i8x16);

impl I8x16 {
    /// Number of `i8` lanes.
    pub const LANES: usize = 16;

    /// Broadcast a single `i8` value to all 16 lanes.
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(core_i8x16::splat(v))
    }

    /// Load from a slice (at least 16 elements required; panics otherwise).
    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 16);
        Self(core_i8x16::from_slice(s))
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [i8; 16]) -> Self {
        Self(core_i8x16::from_array(arr))
    }

    /// Extract all 16 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        self.0.to_array()
    }

    /// Copy lanes into a slice (must have at least 16 elements).
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 16);
        self.0.copy_to_slice(&mut s[..16]);
    }

    /// Unpack 16 signed i4 nibbles from a `u64` into 16 sign-extended `i8`
    /// lanes: `lane[i] = sign_extend_i4((packed >> (4*i)) & 0xf)`, so
    /// `0x0..=0x7 → 0..=7` and `0x8..=0xf → -8..=-1`.
    ///
    /// The sign extension is the `(x << 4) >> 4` arithmetic-shift identity on
    /// the i8 lane, which is exactly what the scalar backend's
    /// `if nibble > 7 { nibble - 16 }` computes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I8x16;
    /// assert_eq!(I8x16::from_i4_packed_u64(0).to_array(), [0i8; 16]);
    /// assert_eq!(I8x16::from_i4_packed_u64(u64::MAX).to_array(), [-1i8; 16]);
    /// let v = I8x16::from_i4_packed_u64(0x8_7);
    /// assert_eq!((v.lane_i8::<0>(), v.lane_i8::<1>()), (7, -8));
    /// # }
    /// ```
    #[inline(always)]
    pub fn from_i4_packed_u64(packed: u64) -> Self {
        const SHIFTS: u64x16 = Simd::from_array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]);
        let nibbles = (u64x16::splat(packed) >> SHIFTS) & u64x16::splat(0xf);
        let lanes = nibbles.cast::<i8>();
        Self((lanes << core_i8x16::splat(4)) >> core_i8x16::splat(4))
    }

    /// Extract lane `N` as an `i8`. `N` must be in `0..16`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I8x16;
    /// let v = I8x16::from_array(core::array::from_fn(|i| i as i8 * 3));
    /// assert_eq!(v.lane_i8::<5>(), 15);
    /// # }
    /// ```
    #[inline(always)]
    pub fn lane_i8<const N: usize>(self) -> i8 {
        self.0[N]
    }

    /// Lane-wise saturating absolute value: `saturating_abs(i8::MIN) == i8::MAX`.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I8x16;
    /// assert_eq!(I8x16::splat(i8::MIN).saturating_abs().to_array(), [i8::MAX; 16]);
    /// assert_eq!(I8x16::splat(-4).saturating_abs().to_array(), [4; 16]);
    /// # }
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    /// Lane-wise minimum.
    ///
    /// # Examples
    /// See [`Self::cmpeq_mask`] — one example exercises the four compare / min / max methods.
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Lane-wise maximum.
    ///
    /// # Examples
    /// See [`Self::cmpeq_mask`] — one example exercises the four compare / min / max methods.
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Per-lane `self == other`, as a 16-bit mask (bit `i` = lane `i`).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::I8x16;
    /// let mut a = [0i8; 16];
    /// a[0] = 9;
    /// a[15] = 9;
    /// let v = I8x16::from_array(a);
    /// assert_eq!(v.cmpeq_mask(I8x16::splat(9)), 0b1000_0000_0000_0001);
    /// assert_eq!(v.cmpgt_mask(I8x16::splat(0)), 0b1000_0000_0000_0001);
    /// assert_eq!(v.simd_min(I8x16::splat(3)).to_array()[0], 3);
    /// assert_eq!(v.simd_max(I8x16::splat(3)).to_array()[1], 3);
    /// # }
    /// ```
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane signed `self > other`, as a 16-bit mask.
    ///
    /// # Examples
    /// See [`Self::cmpeq_mask`] — one example exercises the four compare / min / max methods.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }
}

impl PartialEq for I8x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl fmt::Debug for I8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I8x16({:?})", &self.to_array()[..])
    }
}

// ── W1a-#3: U16x8 / U8x8 / palette_lookup_u8x8 ─────────────────────────────

/// 8-lane `u16` vector backed by `core::simd::u16x8`.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::U16x8;
/// let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
/// let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
/// assert_eq!(U16x8::gather_u16(idx, &table).to_array(), [10, 30, 50, 70, 20, 40, 60, 80]);
/// # }
/// ```
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U16x8(pub core_u16x8);

impl U16x8 {
    /// Number of `u16` lanes.
    pub const LANES: usize = 8;

    /// Broadcast a single `u16` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(core_u16x8::splat(v))
    }

    /// Load from a slice (at least 8 elements required; panics otherwise).
    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8);
        Self(core_u16x8::from_slice(s))
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u16; 8]) -> Self {
        Self(core_u16x8::from_array(arr))
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        self.0.to_array()
    }

    /// Gather 8 `u16` values from `table` at the indices in `indices`.
    ///
    /// Panics in debug if any index is `>= table.len()`; in release an
    /// out-of-range index yields `0` (the scalar backend's rule, kept so the
    /// two realizations never disagree on the same input).
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U16x8;
    /// let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
    /// let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
    /// assert_eq!(U16x8::gather_u16(idx, &table).to_array(), [10, 30, 50, 70, 20, 40, 60, 80]);
    /// assert_eq!(U16x8::gather_u16(idx, &table).lane(7), 80);
    /// # }
    /// ```
    #[inline(always)]
    pub fn gather_u16(indices: U16x8, table: &[u16]) -> Self {
        let idx = indices.to_array();
        #[cfg(debug_assertions)]
        for &i in &idx {
            assert!((i as usize) < table.len(), "gather_u16: index {} OOB (len={})", i, table.len());
        }
        let mut out = [0u16; 8];
        for k in 0..8 {
            out[k] = table.get(idx[k] as usize).copied().unwrap_or(0);
        }
        Self::from_array(out)
    }

    /// Extract lane `k` as a `u16`.
    #[inline(always)]
    pub fn lane(self, k: usize) -> u16 {
        self.0[k]
    }

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    /// Horizontal wrapping sum of all 8 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U16x8;
    /// let v = U16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
    /// assert_eq!(v.reduce_sum(), 36);
    /// assert_eq!(v.simd_min(U16x8::splat(4)).to_array(), [1, 2, 3, 4, 4, 4, 4, 4]);
    /// assert_eq!(v.simd_max(U16x8::splat(4)).to_array(), [4, 4, 4, 4, 5, 6, 7, 8]);
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_sum(self) -> u16 {
        self.0.reduce_sum()
    }
}

impl PartialEq for U16x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl fmt::Debug for U16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U16x8({:?})", &self.to_array()[..])
    }
}

/// 8-lane `u8` vector backed by `core::simd::u8x8` — the return type of
/// [`palette_lookup_u8x8`].
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U8x8(pub core_u8x8);

impl U8x8 {
    /// Number of `u8` lanes.
    pub const LANES: usize = 8;

    /// Broadcast a single `u8` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(core_u8x8::splat(v))
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u8; 8]) -> Self {
        Self(core_u8x8::from_array(arr))
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 8] {
        self.0.to_array()
    }

    /// Horizontal wrapping sum of all 8 lanes.
    ///
    /// # Examples
    /// ```rust
    /// # #[cfg(feature = "nightly-simd")] {
    /// use ndarray::simd_nightly::U8x8;
    /// assert_eq!(U8x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]).reduce_sum(), 36);
    /// assert_eq!(U8x8::splat(255).reduce_sum(), 255u8.wrapping_mul(8));
    /// # }
    /// ```
    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        self.0.reduce_sum()
    }
}

impl PartialEq for U8x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl fmt::Debug for U8x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U8x8({:?})", &self.to_array()[..])
    }
}

/// Look up 8 bytes from a `u8` LUT by `u16` indices.
///
/// Panics in debug on an out-of-range index; returns `0` for it in release —
/// identical to the scalar backend.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::{palette_lookup_u8x8, U16x8};
/// let lut: Vec<u8> = (0..=255u8).rev().collect();
/// let idx = U16x8::from_array([0, 1, 2, 255, 100, 200, 3, 4]);
/// assert_eq!(palette_lookup_u8x8(idx, &lut).to_array(), [255, 254, 253, 0, 155, 55, 252, 251]);
/// # }
/// ```
#[inline(always)]
pub fn palette_lookup_u8x8(idx_v: U16x8, lut: &[u8]) -> U8x8 {
    let idx = idx_v.to_array();
    #[cfg(debug_assertions)]
    for &i in &idx {
        assert!((i as usize) < lut.len(), "palette_lookup_u8x8: index {} OOB (len={})", i, lut.len());
    }
    let mut out = [0u8; 8];
    for k in 0..8 {
        out[k] = lut.get(idx[k] as usize).copied().unwrap_or(0);
    }
    U8x8::from_array(out)
}

// ── W1a-#4: prefetch_read_t0/t1/t2 ──────────────────────────────────────────

/// Hint that `ptr` will be read soon. A deliberate no-op on this backend:
/// `core::simd` carries no prefetch, and the contract is a hint with no
/// observable result. `ptr` may be invalid; it is never dereferenced.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::{prefetch_read_t0, prefetch_read_t1, prefetch_read_t2};
/// let buf = [0u8; 64];
/// prefetch_read_t0(buf.as_ptr());
/// prefetch_read_t1(core::ptr::null()); // a hint never dereferences
/// prefetch_read_t2(buf.as_ptr());
/// # }
/// ```
#[inline(always)]
pub fn prefetch_read_t0(_ptr: *const u8) {}

/// Hint to load into L2 (T1) cache — no-op on this backend, see
/// [`prefetch_read_t0`].
#[inline(always)]
pub fn prefetch_read_t1(_ptr: *const u8) {}

/// Hint to load into L3 (T2) cache — no-op on this backend, see
/// [`prefetch_read_t0`].
#[inline(always)]
pub fn prefetch_read_t2(_ptr: *const u8) {}

// ── W1a-#1: batch_packed_i4_16 ──────────────────────────────────────────────

/// Closure-parameterised batch over packed i4 data.
///
/// Iterates `min(packed.len(), out.len())` times; each iteration unpacks
/// `packed[i]` into an [`I8x16`] and passes it with `aux[i]` to `f`, storing
/// the result in `out[i]`. Panics if `packed.len() != aux.len()`.
///
/// # Examples
/// ```rust
/// # #[cfg(feature = "nightly-simd")] {
/// use ndarray::simd_nightly::batch_packed_i4_16;
/// let packed = [0x7777_7777_7777_7777u64, 0x8888_8888_8888_8888];
/// let aux = [1i8, 2];
/// let mut out = [0i32; 2];
/// batch_packed_i4_16(&packed, &aux, &mut out, |v, a| {
///     v.to_array().iter().map(|&x| x as i32).sum::<i32>() * a as i32
/// });
/// assert_eq!(out, [16 * 7, 16 * -8 * 2]);
/// # }
/// ```
#[inline]
pub fn batch_packed_i4_16<E, F>(packed: &[u64], aux: &[i8], out: &mut [E], f: F)
where
    F: Fn(I8x16, i8) -> E + Sync + Send,
    E: Copy,
{
    assert_eq!(packed.len(), aux.len(), "batch_packed_i4_16: packed and aux must be same length");
    let n = packed.len().min(out.len());
    for i in 0..n {
        out[i] = f(I8x16::from_i4_packed_u64(packed[i]), aux[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar backend's rule, restated independently of `core::simd` so
    /// the portable body is checked against something that is not itself.
    fn reference_unpack(packed: u64) -> [i8; 16] {
        let mut lanes = [0i8; 16];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let nibble = ((packed >> (4 * i)) & 0xf) as i8;
            *lane = if nibble > 7 { nibble - 16 } else { nibble };
        }
        lanes
    }

    #[test]
    fn i4_unpack_sign_extends_every_nibble_value_in_every_lane() {
        // Every nibble value at every lane position, plus the two all-same
        // words. A body that forgot the arithmetic shift passes `0..=7` and
        // fails `0x8..=0xf`; one that sign-extended the wrong width fails the
        // per-lane placement.
        for lane in 0..16 {
            for nib in 0u64..16 {
                let packed = nib << (4 * lane);
                assert_eq!(
                    I8x16::from_i4_packed_u64(packed).to_array(),
                    reference_unpack(packed),
                    "lane {lane} nibble {nib:#x}"
                );
            }
        }
        assert_eq!(I8x16::from_i4_packed_u64(u64::MAX).to_array(), [-1i8; 16]);
        assert_eq!(I8x16::from_i4_packed_u64(0x8888_8888_8888_8888).to_array(), [-8i8; 16]);
        let mixed = 0xfedc_ba98_7654_3210u64;
        let got = I8x16::from_i4_packed_u64(mixed);
        assert_eq!(got.to_array(), reference_unpack(mixed));
        assert_eq!(got.lane_i8::<0>(), 0);
        assert_eq!(got.lane_i8::<7>(), 7);
        assert_eq!(got.lane_i8::<8>(), -8);
        assert_eq!(got.lane_i8::<15>(), -1);
    }

    #[test]
    fn saturating_abs_saturates_i8_min_and_leaves_the_rest_exact() {
        let mut arr = [0i8; 16];
        arr[0] = i8::MIN;
        arr[1] = -127;
        arr[2] = -1;
        arr[3] = 0;
        arr[4] = 1;
        arr[5] = i8::MAX;
        let got = I8x16::from_array(arr).saturating_abs().to_array();
        assert_eq!(got[0], i8::MAX, "|i8::MIN| must saturate to 127, not wrap to -128");
        assert_eq!(&got[1..6], &[127, 1, 0, 1, 127]);
    }

    #[test]
    fn gather_and_palette_lookup_index_by_lane() {
        let table: Vec<u16> = (0..300).map(|i| i as u16 * 3).collect();
        let idx = U16x8::from_array([0, 299, 1, 298, 2, 297, 3, 296]);
        assert_eq!(U16x8::gather_u16(idx, &table).to_array(), [0, 897, 3, 894, 6, 891, 9, 888]);
        let lut: Vec<u8> = (0..=255u8).rev().collect();
        assert_eq!(
            palette_lookup_u8x8(idx.simd_min(U16x8::splat(255)), &lut).to_array(),
            [255, 0, 254, 0, 253, 0, 252, 0]
        );
    }

    #[test]
    fn batch_unpacks_each_word_and_stops_at_the_shorter_output() {
        let packed = [0x0u64, u64::MAX, 0x8888_8888_8888_8888, 0x7777_7777_7777_7777];
        let aux = [1i8, 2, 3, 4];
        let mut out = [i32::MIN; 3];
        batch_packed_i4_16(&packed, &aux, &mut out, |v, a| {
            v.to_array().iter().map(|&x| x as i32).sum::<i32>() * a as i32
        });
        assert_eq!(out, [0, -16 * 2, -128 * 3]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn batch_rejects_mismatched_packed_and_aux() {
        let mut out = [0u8; 2];
        batch_packed_i4_16(&[0u64, 0], &[0i8], &mut out, |_, _| 0);
    }

    #[test]
    fn prefetch_hints_accept_any_pointer_without_dereferencing() {
        prefetch_read_t0(core::ptr::null());
        prefetch_read_t1(usize::MAX as *const u8);
        prefetch_read_t2(core::ptr::dangling());
    }
}
