//! U16x32 / U32x16 / U32x8 / U64x8 / U64x4 portable-simd wrappers — round-3-portable-simd agent #4.
#![cfg(feature = "nightly-simd")]

use core::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use core::simd::num::SimdUint;
use core::simd::{u16x16, u16x32, u32x16, u32x8, u64x4, u64x8};

// ════════════════════════════════════════════════════════════════════
// U64x8 — 8-lane u64
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u64` SIMD vector backed by `core::simd::u64x8`.
///
/// Also used as the return type of `F64x8::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U64x8(pub u64x8);

impl U64x8 {
    pub const LANES: usize = 8;

    /// Lane-wise left-rotate by `n` bits — the u64 ARX rotate (BLAKE2b /
    /// argon2 lane). `n` is taken mod 64; `n == 0` returns `self`, since
    /// `x >> 64` is UB on `u64`.
    ///
    /// `core::simd` has no rotate, so this is the same per-lane loop the
    /// scalar backend uses. The codegen oracle measured that such a loop does
    /// **not** vectorize on stable (0 packed, one `rorq` per lane, across
    /// three spellings); whether this backend's codegen does better is
    /// **unmeasured** — the oracle runs on stable and cannot see it. The
    /// native `VPROLVQ`/`VPRORVQ` override lives on `simd_avx512`'s `U64x8`.
    /// See `.claude/knowledge/crypto-lane-status.md`.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let n = n % 64;
        if n == 0 {
            return self;
        }
        let a = self.to_array();
        let mut o = [0u64; 8];
        for i in 0..8 {
            o[i] = a[i].rotate_left(n);
        }
        Self::from_array(o)
    }

    /// Lane-wise right-rotate by `n` bits — BLAKE2b's direction.
    /// `rotr(n) == rotl(64 - n)` exactly; kept distinct because BLAKE2b and
    /// argon2 are specified in terms of right rotation.
    #[inline(always)]
    pub fn rotate_right(self, n: u32) -> Self {
        let n = n % 64;
        if n == 0 {
            return self;
        }
        let a = self.to_array();
        let mut o = [0u64; 8];
        for i in 0..8 {
            o[i] = a[i].rotate_right(n);
        }
        Self::from_array(o)
    }

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 8, "U64x8::from_slice needs >=8 elements");
        Self(u64x8::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self(u64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 8, "U64x8::copy_to_slice needs >=8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U64x4 — 4-lane u64 (companion for F64x4::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 4-lane `u64` SIMD vector backed by `core::simd::u64x4`.
///
/// Return type of `F64x4::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U64x4(pub u64x4);

impl U64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(u64x4::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 4, "U64x4::from_slice needs >=4 elements");
        Self(u64x4::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 4]) -> Self {
        Self(u64x4::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 4] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 4, "U64x4::copy_to_slice needs >=4 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u64 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u64 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit value (4 bits used; bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit value (4 bits used).
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U64x4 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x8 — 8-lane u32 (companion for F32x8::to_bits)
// ════════════════════════════════════════════════════════════════════

/// 8-lane `u32` SIMD vector backed by `core::simd::u32x8`.
///
/// Return type of `F32x8::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U32x8(pub u32x8);

impl U32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 8, "U32x8::from_slice needs >=8 elements");
        Self(u32x8::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self(u32x8::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 8, "U32x8::copy_to_slice needs >=8 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns an 8-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u8 {
        self.0.simd_eq(other.0).to_bitmask() as u8
    }

    /// Per-lane unsigned greater-than. Returns an 8-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u8 {
        self.0.simd_gt(other.0).to_bitmask() as u8
    }
}

impl Default for U32x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x16 — 16-lane u32
// ════════════════════════════════════════════════════════════════════

/// 16-lane `u32` SIMD vector backed by `core::simd::u32x16`.
///
/// Also used as the return type of `F32x16::to_bits`.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U32x16(pub u32x16);

/// BLAKE3 `hash_many` shuffle surface — the transpose network's four unpacks
/// and two half-concatenations, at DEGREE 16.
///
/// **Why these live on `U32x16` and not on a half-width type.** BLAKE3's
/// AVX2 backend is DEGREE 8 over `__m256i`. Two of those fit in one
/// `U32x16`, and every operation in `hash_many` is either lane-wise
/// (add / xor / rotate) or confined *within* a 128- or 256-bit lane — which
/// is exactly what the methods below preserve. The two 8-lane groups
/// therefore never interact, and the algorithm runs on both at once as
/// DEGREE 16 with no cross-talk. A `U32x8` is neither needed nor wanted
/// (operator ruling, 2026-07-28: a half-width type standing in for the lane
/// the substrate actually uses is an absolute no-go).
///
/// Semantics are x86's, applied to **each 256-bit half independently**:
/// `interleave_*_u32` / `interleave_*_u64` reproduce
/// `_mm256_unpack{lo,hi}_epi{32,64}` within each 128-bit quad, and
/// `concat_{lo,hi}_halves` reproduce
/// `_mm256_permute2x128_si256(_, _, 0x20 / 0x31)` within each 256-bit half.
/// The per-lane structure is load-bearing: BLAKE3's transpose is defined in
/// terms of it, so a "helpful" whole-vector interleave would compute a
/// different permutation and produce wrong hashes with no compile error.
///
/// Every body is a plain index loop. The codegen oracle
/// (`.claude/knowledge/simd-codegen-oracle/`) measured this exact shape:
/// fixed two-source permutations written as index loops compile to real
/// packed shuffles, and a transpose composed from them emits
/// `vpunpcklqdq` / `vpermq` / `vinserti128`. No `unsafe`, no `core::arch`,
/// no intrinsic override earned. See
/// `.claude/knowledge/blake3-on-ndarray-simd.md`.
impl U32x16 {
    /// `_mm256_unpacklo_epi32` per 256-bit half: within each 128-bit quad,
    /// interleave the low two `u32` of each operand.
    #[inline(always)]
    pub fn interleave_lo_u32(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for q in 0..4 {
            let i = 4 * q;
            o[i] = a[i];
            o[i + 1] = b[i];
            o[i + 2] = a[i + 1];
            o[i + 3] = b[i + 1];
        }
        Self::from_array(o)
    }

    /// `_mm256_unpackhi_epi32` per 256-bit half: within each 128-bit quad,
    /// interleave the high two `u32` of each operand.
    #[inline(always)]
    pub fn interleave_hi_u32(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for q in 0..4 {
            let i = 4 * q;
            o[i] = a[i + 2];
            o[i + 1] = b[i + 2];
            o[i + 2] = a[i + 3];
            o[i + 3] = b[i + 3];
        }
        Self::from_array(o)
    }

    /// `_mm256_unpacklo_epi64` per 256-bit half: within each 128-bit quad,
    /// the low `u64` of each operand.
    #[inline(always)]
    pub fn interleave_lo_u64(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for q in 0..4 {
            let i = 4 * q;
            o[i] = a[i];
            o[i + 1] = a[i + 1];
            o[i + 2] = b[i];
            o[i + 3] = b[i + 1];
        }
        Self::from_array(o)
    }

    /// `_mm256_unpackhi_epi64` per 256-bit half: within each 128-bit quad,
    /// the high `u64` of each operand.
    #[inline(always)]
    pub fn interleave_hi_u64(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for q in 0..4 {
            let i = 4 * q;
            o[i] = a[i + 2];
            o[i + 1] = a[i + 3];
            o[i + 2] = b[i + 2];
            o[i + 3] = b[i + 3];
        }
        Self::from_array(o)
    }

    /// `_mm256_permute2x128_si256(a, b, 0x20)` per 256-bit half: the low
    /// 128-bit lane of each operand, concatenated.
    ///
    /// Only the two immediates BLAKE3's transpose uses are exposed, as named
    /// methods rather than a generic `const IMM` permute: the remaining
    /// immediates would have no caller and no parity test.
    #[inline(always)]
    pub fn concat_lo_halves(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for h in 0..2 {
            let i = 8 * h;
            o[i..i + 4].copy_from_slice(&a[i..i + 4]);
            o[i + 4..i + 8].copy_from_slice(&b[i..i + 4]);
        }
        Self::from_array(o)
    }

    /// `_mm256_permute2x128_si256(a, b, 0x31)` per 256-bit half: the high
    /// 128-bit lane of each operand, concatenated.
    #[inline(always)]
    pub fn concat_hi_halves(self, other: Self) -> Self {
        let (a, b) = (self.to_array(), other.to_array());
        let mut o = [0u32; 16];
        for h in 0..2 {
            let i = 8 * h;
            o[i..i + 4].copy_from_slice(&a[i + 4..i + 8]);
            o[i + 4..i + 8].copy_from_slice(&b[i + 4..i + 8]);
        }
        Self::from_array(o)
    }

    /// One butterfly exchange at block granularity `G` elements — the general
    /// form of the whole unpack / lane-exchange family, parameterized by
    /// granularity instead of one method per width.
    ///
    /// `G = 1` is the 32-bit unpack, `G = 2` the 64-bit unpack, `G = 4` the
    /// 128-bit lane exchange, and `G = 8` the 256-bit half exchange that a
    /// 512-bit lane additionally needs and for which no AVX2 intrinsic exists.
    /// `G` is a const parameter, so every shuffle pattern is compile-time
    /// constant — the same property a hand-written intrinsic has, and the
    /// precondition for LLVM to select a shuffle rather than an indexed copy.
    ///
    /// Four stages over this (`G` = 1, 2, 4, 8, pairing row `r` with `r | G`)
    /// compose a complete 16x16 transpose. Measured as
    /// `transpose_16x16_composed`: **79 packed / 0 scalar-lane-arith**, 19 of
    /// them real shuffles. The same transpose written as one monolithic index
    /// loop measures **0 packed** — 1088 bytes of stack and a 256-iteration
    /// scalar copy. The spelling is the entire difference. See
    /// `.claude/knowledge/blake3-on-ndarray-simd.md`.
    ///
    /// No intrinsic override is earned: the generic form does not fail.
    #[inline(always)]
    pub fn exchange<const G: usize>(self, other: Self) -> (Self, Self) {
        let (l, h) = (self.to_array(), other.to_array());
        let mut nl = [0u32; 16];
        let mut nh = [0u32; 16];
        for c in 0..16 {
            nl[c] = if c & G == 0 { l[c] } else { h[c ^ G] };
            nh[c] = if c & G != 0 { h[c] } else { l[c ^ G] };
        }
        (Self::from_array(nl), Self::from_array(nh))
    }
}

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(u32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 16, "U32x16::from_slice needs >=16 elements");
        Self(u32x16::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(u32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 16, "U32x16::copy_to_slice needs >=16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns a 16-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    /// Per-lane unsigned greater-than. Returns a 16-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }

    /// Lane-wise left-rotate by `n` bits — the ARX rotate (matches
    /// `u32::rotate_left`), completing `Add` + `BitXor` for ChaCha20/BLAKE.
    /// `core::simd` has no bit-rotate, so this is the shift-or composition with
    /// the `n % 32 == 0` guard (a `>> 32` on a `u32` lane is UB), matching
    /// `u32::rotate_left`'s wrap-by-32 semantics.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let n = n % 32;
        if n == 0 {
            return self;
        }
        Self((self.0 << u32x16::splat(n)) | (self.0 >> u32x16::splat(32 - n)))
    }
}

impl Default for U32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U16x32 — 32-lane u16
// ════════════════════════════════════════════════════════════════════

/// 32-lane `u16` SIMD vector backed by `core::simd::u16x32`.
///
/// API mirrors `simd_avx512::U16x32`. Miri-executable.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U16x32(pub u16x32);

impl U16x32 {
    pub const LANES: usize = 32;

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(u16x32::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 32, "U16x32::from_slice needs >=32 elements");
        Self(u16x32::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u16; 32]) -> Self {
        Self(u16x32::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 32] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 32, "U16x32::copy_to_slice needs >=32 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    /// Wrapping horizontal sum of all 32 lanes (result is u16, wraps on overflow).
    #[inline(always)]
    pub fn reduce_sum(self) -> u16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u16 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max ─────────────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
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

    // ── Compare -> bitmask ────────────────────────────────────────

    /// Per-lane equality. Returns a 32-bit bitmask (bit i = 1 iff lane i equal).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        self.0.simd_eq(other.0).to_bitmask() as u32
    }

    /// Per-lane unsigned greater-than. Returns a 32-bit bitmask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        self.0.simd_gt(other.0).to_bitmask() as u32
    }
}

impl Default for U16x32 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// U16x16 — 16-lane u16 (256-bit, added 2026-05-20 missing-lanes sweep)
// ════════════════════════════════════════════════════════════════════

/// 16-lane `u16` SIMD vector backed by `core::simd::u16x16`.
///
/// API mirrors `simd_avx512::U16x32` at half-width. Miri-executable.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct U16x16(pub u16x16);

impl U16x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(u16x16::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 16, "U16x16::from_slice needs >=16 elements");
        Self(u16x16::from_slice(s))
    }

    #[inline(always)]
    pub fn from_array(arr: [u16; 16]) -> Self {
        Self(u16x16::from_array(arr))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 16, "U16x16::copy_to_slice needs >=16 elements");
        self.0.copy_to_slice(s);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> u16 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> u16 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> u16 {
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
    pub fn cmpeq_mask(self, other: Self) -> u16 {
        self.0.simd_eq(other.0).to_bitmask() as u16
    }

    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u16 {
        self.0.simd_gt(other.0).to_bitmask() as u16
    }
}

impl Default for U16x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16x32_splat_reduce() {
        let v = U16x32::splat(3);
        assert_eq!(v.reduce_min(), 3);
        assert_eq!(v.reduce_max(), 3);
    }

    #[test]
    fn u16x32_cmpeq_mask_all_equal() {
        let a = U16x32::splat(10);
        let b = U16x32::splat(10);
        assert_eq!(a.cmpeq_mask(b), u32::MAX);
    }

    #[test]
    fn u16x32_saturating_add_clamps() {
        let a = U16x32::splat(60000u16);
        let b = U16x32::splat(10000u16);
        let c = a.saturating_add(b);
        assert!(c.to_array().iter().all(|&v| v == u16::MAX));
    }

    #[test]
    fn u32x16_splat_reduce() {
        let v = U32x16::splat(7);
        assert_eq!(v.reduce_sum(), 112u32); // 16 * 7
        assert_eq!(v.reduce_min(), 7);
        assert_eq!(v.reduce_max(), 7);
    }

    #[test]
    fn u32x16_cmpgt_mask() {
        let mut arr = [0u32; 16];
        for (i, x) in arr.iter_mut().enumerate() {
            *x = i as u32;
        }
        let v = U32x16::from_array(arr);
        let threshold = U32x16::splat(7);
        // Lanes 8..15 are > 7 -> bits 8..15 set -> 0xFF00
        assert_eq!(v.cmpgt_mask(threshold), 0xFF00u16);
    }

    #[test]
    fn u32x8_splat_reduce() {
        let v = U32x8::splat(5);
        assert_eq!(v.reduce_sum(), 40u32); // 8 * 5
        assert_eq!(v.reduce_min(), 5);
        assert_eq!(v.reduce_max(), 5);
    }

    #[test]
    fn u32x8_cmpeq_mask() {
        let a = U32x8::from_array([1, 2, 1, 2, 1, 2, 1, 2]);
        let b = U32x8::splat(1);
        // Lanes 0,2,4,6 equal -> bits 0,2,4,6 -> 0b01010101 = 0x55
        assert_eq!(a.cmpeq_mask(b), 0x55u8);
    }

    #[test]
    fn u64x8_splat_reduce() {
        let v = U64x8::splat(100);
        assert_eq!(v.reduce_sum(), 800u64); // 8 * 100
        assert_eq!(v.reduce_min(), 100);
        assert_eq!(v.reduce_max(), 100);
    }

    #[test]
    fn u64x8_cmpeq_mask_all() {
        let a = U64x8::splat(42);
        let b = U64x8::splat(42);
        assert_eq!(a.cmpeq_mask(b), 0xFFu8);
    }

    #[test]
    fn u64x4_splat_reduce() {
        let v = U64x4::splat(9);
        assert_eq!(v.reduce_sum(), 36u64); // 4 * 9
        assert_eq!(v.reduce_min(), 9);
        assert_eq!(v.reduce_max(), 9);
    }

    #[test]
    fn u64x4_cmpeq_mask_partial() {
        let a = U64x4::from_array([1, 2, 1, 2]);
        let b = U64x4::splat(1);
        // Lanes 0,2 equal -> bits 0,2 -> 0b0101 = 5
        assert_eq!(a.cmpeq_mask(b), 0x05u8);
    }

    #[test]
    fn u64x4_cmpgt_mask() {
        let a = U64x4::from_array([10, 1, 10, 1]);
        let b = U64x4::splat(5);
        // Lanes 0,2 > 5 -> bits 0,2 -> 0b0101 = 5
        assert_eq!(a.cmpgt_mask(b), 0x05u8);
    }
}
