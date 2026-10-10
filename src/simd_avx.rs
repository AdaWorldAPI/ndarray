//! AVX-without-AVX2 realization (Sandy Bridge / Ivy Bridge, AMD Bulldozer to
//! Jaguar, VMs that hide AVX2). Plan: `.claude/plans/simd-avx1-backend-v1.md`.
//!
//! Compiled only on `all(target_feature = "avx", not(target_feature = "avx2"))`
//! (`target-cpu=native` on such a CPU, or `.cargo/config-avx.toml`). It is not a
//! second set of types: `simd.rs` exports the same types on this arm as on the
//! AVX2 arm. What differs is the realization of the methods that would need
//! AVX2. `simd_avx2.rs` and `simd_avx512.rs` gate exactly those methods with
//! `cfg(not(<this arm>))`, and this file supplies them with the same signatures
//! and the same results, built from:
//!
//! * two 128-bit SSE2/SSSE3/SSE4.1 halves for 256-bit integer arithmetic, and
//! * AVX1's float-domain forms where they are bit-exact (`and`/`or`/`xor` via
//!   `_ps`, `permute2f128`, `blend_ps`, 64-bit unpacks via `_pd`).
//!
//! Selection is compile-time only; nothing here checks the CPU at run time.
//!
//! # Safety footing (every `unsafe` block below)
//!
//! On this arm `avx` is a compiled-in target feature, and rustc's feature
//! implications make `sse2`, `ssse3`, `sse4.1` and `sse4.2` compiled in with it.
//! `cpu_guard` refuses to start a binary whose compiled-in features the CPU
//! lacks, so every intrinsic used here is supported at run time. Each block
//! states anything beyond that (memory bounds, shift-count ranges).

use core::arch::x86_64::*;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Not, Shl, Shr, Sub, SubAssign,
};

use crate::simd_avx2::{I32x16, U16x16, U64x8, U8x32};
use crate::simd_avx512::{F32x8, I16x16, I8x32};

// ---------------------------------------------------------------------------
// 256 ⇄ 2 × 128
// ---------------------------------------------------------------------------

/// Low and high 128-bit halves of a 256-bit integer register.
#[inline(always)]
fn split(v: __m256i) -> (__m128i, __m128i) {
    // SAFETY: AVX is compiled in (module docs); register-only ops.
    unsafe { (_mm256_castsi256_si128(v), _mm256_extractf128_si256::<1>(v)) }
}

/// Inverse of [`split`].
#[inline(always)]
fn join(lo: __m128i, hi: __m128i) -> __m256i {
    // SAFETY: AVX is compiled in (module docs); register-only op.
    unsafe { _mm256_set_m128i(hi, lo) }
}

/// Applies a 128-bit binary op to both halves of two 256-bit registers.
macro_rules! halves2 {
    ($op:ident, $a:expr, $b:expr) => {{
        let (al, ah) = split($a);
        let (bl, bh) = split($b);
        // SAFETY: the SSE op is compiled in on this arm (module docs);
        // register-only op.
        unsafe { join($op(al, bl), $op(ah, bh)) }
    }};
}

/// Applies an AVX1 float-domain bitwise op to two 256-bit integer registers.
/// Bit-exact: `and`/`or`/`xor` do not look at the bits as floats.
macro_rules! bits_ps {
    ($op:ident, $a:expr, $b:expr) => {{
        // SAFETY: AVX is compiled in (module docs); casts and a bitwise op on
        // registers only.
        unsafe { _mm256_castps_si256($op(_mm256_castsi256_ps($a), _mm256_castsi256_ps($b))) }
    }};
}

/// `movemask_epi8` over a 256-bit register: low half → bits 0..16, high half →
/// bits 16..32, the same order `_mm256_movemask_epi8` produces.
#[inline(always)]
fn movemask_epi8_256(v: __m256i) -> u32 {
    let (lo, hi) = split(v);
    // SAFETY: SSE2 is compiled in (module docs); register-only op.
    unsafe { (_mm_movemask_epi8(lo) as u32 & 0xFFFF) | ((_mm_movemask_epi8(hi) as u32) << 16) }
}

// ---------------------------------------------------------------------------
// Free functions (re-exported from `simd_avx2` on this arm)
// ---------------------------------------------------------------------------

/// Popcount of a byte slice: SSSE3 `pshufb` nibble lookup over 16-byte chunks,
/// `psadbw` to fold. Same result as the AVX2 realization.
pub fn popcount(a: &[u8]) -> u64 {
    let chunks = a.len() / 16;
    // SAFETY: SSE2/SSSE3 are compiled in (module docs). Every load reads the
    // 16 bytes at `i * 16` with `i < chunks = len / 16`, so it is in bounds;
    // `loadu` needs no alignment.
    let mut sum = unsafe {
        let low_mask = _mm_set1_epi8(0x0f);
        let lookup = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        let mut total = _mm_setzero_si128();
        let mut i = 0;
        while i < chunks {
            // Byte counters reach at most 8 per chunk; 31 chunks stay below
            // 256, so fold before they could wrap.
            let end = (i + 31).min(chunks);
            let mut local = _mm_setzero_si128();
            while i < end {
                let v = _mm_loadu_si128(a.as_ptr().add(i * 16) as *const __m128i);
                let lo = _mm_and_si128(v, low_mask);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), low_mask);
                let cnt = _mm_add_epi8(_mm_shuffle_epi8(lookup, lo), _mm_shuffle_epi8(lookup, hi));
                local = _mm_add_epi8(local, cnt);
                i += 1;
            }
            total = _mm_add_epi64(total, _mm_sad_epu8(local, _mm_setzero_si128()));
        }
        let mut lanes = [0u64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, total);
        lanes[0] + lanes[1]
    };
    for &byte in &a[chunks * 16..] {
        sum += byte.count_ones() as u64;
    }
    sum
}

/// Signed int8 dot product, `Σ a[i] as i8 × b[i] as i8`, exact, over the first
/// `min(a.len(), b.len())` bytes. SSE4.1 `pmovsxbw` sign-extends each 16-byte
/// chunk to i16 and SSE2 `pmaddwd` multiplies pairwise into i32, exact for
/// every i8 pair; the i32 lanes are folded into an i64 every
/// `crate::simd_avx2::DOT_I8_FOLD` chunks, so no length can wrap them.
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    let fold = crate::simd_avx2::DOT_I8_FOLD;
    let mut total = 0i64;
    for (ba, bb) in a.chunks(16 * fold).zip(b.chunks(16 * fold)) {
        let (ca, cb) = (ba.chunks_exact(16), bb.chunks_exact(16));
        let (ta, tb) = (ca.remainder(), cb.remainder());
        // SAFETY: SSE2/SSE4.1 are compiled in (module docs). Every chunk is
        // exactly 16 bytes, so each 16-byte `loadu` is in bounds; `loadu`
        // needs no alignment.
        let lanes = unsafe {
            let mut acc = _mm_setzero_si128();
            for (xa, xb) in ca.zip(cb) {
                let av = _mm_loadu_si128(xa.as_ptr() as *const __m128i);
                let bv = _mm_loadu_si128(xb.as_ptr() as *const __m128i);
                let lo = _mm_madd_epi16(_mm_cvtepi8_epi16(av), _mm_cvtepi8_epi16(bv));
                let hi = _mm_madd_epi16(
                    _mm_cvtepi8_epi16(_mm_srli_si128::<8>(av)),
                    _mm_cvtepi8_epi16(_mm_srli_si128::<8>(bv)),
                );
                acc = _mm_add_epi32(acc, _mm_add_epi32(lo, hi));
            }
            let mut v = [0i32; 4];
            _mm_storeu_si128(v.as_mut_ptr() as *mut __m128i, acc);
            v
        };
        total += lanes.iter().map(|&v| v as i64).sum::<i64>();
        total += ta
            .iter()
            .zip(tb)
            .map(|(&x, &y)| (x as i8 as i64) * (y as i8 as i64))
            .sum::<i64>();
    }
    total
}

// ---------------------------------------------------------------------------
// U64x8 (array-backed; only these methods need AVX2 on the AVX2 arm)
// ---------------------------------------------------------------------------

impl U64x8 {
    /// `(x << n) | (x >> (64 - n))` per 64-bit lane of one 256-bit half, with
    /// `1 <= n <= 63` guaranteed by the callers (see the AVX2 realization).
    #[inline(always)]
    pub(crate) fn rotl_half(v: __m256i, n: u32) -> __m256i {
        debug_assert!((1..=63).contains(&n));
        let (lo, hi) = split(v);
        // SAFETY: SSE2 is compiled in (module docs). Counts are in `1..=63`,
        // so neither shift is by 64 or more.
        unsafe {
            let l = _mm_cvtsi32_si128(n as i32);
            let r = _mm_cvtsi32_si128((64 - n) as i32);
            join(
                _mm_or_si128(_mm_sll_epi64(lo, l), _mm_srl_epi64(lo, r)),
                _mm_or_si128(_mm_sll_epi64(hi, l), _mm_srl_epi64(hi, r)),
            )
        }
    }

    /// Lane-wise `lo32(self) × lo32(rhs)` as an exact `u64` (`pmuludq`, four
    /// per vector). The high 32 bits of every input lane are ignored.
    #[inline(always)]
    pub fn mul_lo32(self, rhs: Self) -> Self {
        let mut o = [0u64; 8];
        // SAFETY: SSE2 is compiled in (module docs). `self.0`, `rhs.0` and `o`
        // are `[u64; 8]` (64 bytes); the four 16-byte loads and stores at byte
        // offsets 0, 16, 32, 48 are in bounds.
        unsafe {
            for k in 0..4 {
                let a = _mm_loadu_si128(self.0.as_ptr().add(2 * k) as *const __m128i);
                let b = _mm_loadu_si128(rhs.0.as_ptr().add(2 * k) as *const __m128i);
                _mm_storeu_si128(o.as_mut_ptr().add(2 * k) as *mut __m128i, _mm_mul_epu32(a, b));
            }
        }
        Self(o)
    }

    /// 8×8 transpose of `u64` words across eight registers:
    /// `out[i]` lane `j` == `rows[j]` lane `i`. AVX1 forms of the AVX2
    /// realization: `unpack{lo,hi}_pd` and `permute2f128`, which move the same
    /// 64-bit words without inspecting them.
    #[inline(always)]
    pub fn transpose8(rows: [Self; 8]) -> [Self; 8] {
        let load = |v: &Self, half: usize| -> __m256d {
            // SAFETY: AVX is compiled in (module docs); `v.0` is `[u64; 8]`
            // (64 bytes), so the 32-byte load at byte offset 0 or 32 is in
            // bounds; `loadu` needs no alignment.
            unsafe { _mm256_loadu_pd(v.0.as_ptr().add(4 * half) as *const f64) }
        };
        let mut out = [Self([0u64; 8]); 8];
        for bi in 0..2 {
            for bj in 0..2 {
                let h = |k: usize| load(&rows[4 * bi + k], bj);
                // SAFETY: AVX is compiled in (module docs); register-only ops.
                let c = unsafe {
                    let t0 = _mm256_unpacklo_pd(h(0), h(1));
                    let t1 = _mm256_unpackhi_pd(h(0), h(1));
                    let t2 = _mm256_unpacklo_pd(h(2), h(3));
                    let t3 = _mm256_unpackhi_pd(h(2), h(3));
                    [
                        _mm256_permute2f128_pd::<0x20>(t0, t2),
                        _mm256_permute2f128_pd::<0x20>(t1, t3),
                        _mm256_permute2f128_pd::<0x31>(t0, t2),
                        _mm256_permute2f128_pd::<0x31>(t1, t3),
                    ]
                };
                for (k, v) in c.into_iter().enumerate() {
                    // SAFETY: AVX is compiled in (module docs); `out[..].0` is
                    // `[u64; 8]`, so the 32-byte store at byte offset 0 or 32
                    // is in bounds.
                    unsafe { _mm256_storeu_pd(out[4 * bj + k].0.as_mut_ptr().add(4 * bi) as *mut f64, v) };
                }
            }
        }
        out
    }
}

/// Lane-wise variable left shift. Counts of 64 or more zero the lane, as
/// `VPSLLVQ` does on the AVX2 arm; callers keep counts below 64 (see the AVX2
/// realization for the contract). No SSE/AVX1 form exists, so this is per lane.
impl Shl<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        debug_assert!(rhs.0.iter().all(|&n| n < 64), "U64x8 shift counts are a caller contract: < 64");
        Self(core::array::from_fn(|i| if rhs.0[i] < 64 { self.0[i] << rhs.0[i] } else { 0 }))
    }
}

/// Lane-wise variable right shift; see the `Shl<Self>` impl above.
impl Shr<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        debug_assert!(rhs.0.iter().all(|&n| n < 64), "U64x8 shift counts are a caller contract: < 64");
        Self(core::array::from_fn(|i| if rhs.0[i] < 64 { self.0[i] >> rhs.0[i] } else { 0 }))
    }
}

// ---------------------------------------------------------------------------
// I32x16 (array-backed)
// ---------------------------------------------------------------------------

impl I32x16 {
    /// Four 128-bit quarters of the 16 lanes.
    #[inline(always)]
    fn quarters(self) -> [__m128i; 4] {
        // SAFETY: SSE2 is compiled in (module docs); `self.0` is `[i32; 16]`
        // (64 bytes), so the 16-byte loads at byte offsets 0, 16, 32, 48 are in
        // bounds; `loadu` needs no alignment.
        core::array::from_fn(|k| unsafe { _mm_loadu_si128(self.0.as_ptr().add(4 * k) as *const __m128i) })
    }

    /// Horizontal signed minimum (`pminsd` tree, SSE4.1). Exact.
    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        let [a, b, c, d] = self.quarters();
        // SAFETY: SSE2/SSE4.1 are compiled in (module docs); register-only ops.
        unsafe {
            let m4 = _mm_min_epi32(_mm_min_epi32(a, b), _mm_min_epi32(c, d));
            let m2 = _mm_min_epi32(m4, _mm_shuffle_epi32::<0b01_00_11_10>(m4));
            let m1 = _mm_min_epi32(m2, _mm_shuffle_epi32::<0b00_00_00_01>(m2));
            _mm_cvtsi128_si32(m1)
        }
    }

    /// Horizontal signed maximum, the `pmaxsd` twin of [`Self::reduce_min`].
    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        let [a, b, c, d] = self.quarters();
        // SAFETY: SSE2/SSE4.1 are compiled in (module docs); register-only ops.
        unsafe {
            let m4 = _mm_max_epi32(_mm_max_epi32(a, b), _mm_max_epi32(c, d));
            let m2 = _mm_max_epi32(m4, _mm_shuffle_epi32::<0b01_00_11_10>(m4));
            let m1 = _mm_max_epi32(m2, _mm_shuffle_epi32::<0b00_00_00_01>(m2));
            _mm_cvtsi128_si32(m1)
        }
    }

    /// Lane-wise signed greater-than as a 16-bit mask, LSB-first (lane 0 is
    /// bit 0). Same contract as the AVX2 realization.
    #[inline(always)]
    pub fn gt_bitmask(self, other: Self) -> u16 {
        let a = self.quarters();
        let b = other.quarters();
        let mut m = 0u32;
        for k in 0..4 {
            // SAFETY: SSE/SSE2 are compiled in (module docs); register-only ops.
            let q = unsafe { _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a[k], b[k]))) as u32 };
            m |= q << (4 * k);
        }
        m as u16
    }
}

// ---------------------------------------------------------------------------
// U16x16 (native __m256i)
// ---------------------------------------------------------------------------

impl U16x16 {
    /// Logical right shift of each 16-bit lane by `imm` (any count; 16 or more
    /// gives 0, as on the AVX2 arm).
    #[inline(always)]
    pub fn shr(self, imm: u32) -> Self {
        let (lo, hi) = split(self.0);
        // SAFETY: SSE2 is compiled in (module docs); register-only ops.
        unsafe {
            let c = _mm_cvtsi32_si128(imm as i32);
            Self(join(_mm_srl_epi16(lo, c), _mm_srl_epi16(hi, c)))
        }
    }

    /// Logical left shift of each 16-bit lane by `imm` (any count).
    #[inline(always)]
    pub fn shl(self, imm: u32) -> Self {
        let (lo, hi) = split(self.0);
        // SAFETY: SSE2 is compiled in (module docs); register-only ops.
        unsafe {
            let c = _mm_cvtsi32_si128(imm as i32);
            Self(join(_mm_sll_epi16(lo, c), _mm_sll_epi16(hi, c)))
        }
    }

    /// Multiply, keep the low 16 bits (wrapping).
    #[inline(always)]
    pub fn mullo(self, other: Self) -> Self {
        Self(halves2!(_mm_mullo_epi16, self.0, other.0))
    }

    /// Cross-128-bit-lane permute; `IMM` has `_mm256_permute2x128_si256`'s
    /// meaning. AVX1's `permute2f128` moves the same 128-bit halves.
    #[inline(always)]
    pub fn permute2x128<const IMM: i32>(self, other: Self) -> Self {
        // SAFETY: AVX is compiled in (module docs); register-only op.
        Self(unsafe { _mm256_permute2f128_si256::<IMM>(self.0, other.0) })
    }

    /// Blend 32-bit dwords from `self`/`other` per `IMM`
    /// (`_mm256_blend_epi32`'s meaning; bit `i` set takes dword `i` from
    /// `other`). AVX1's `blend_ps` selects the same dwords.
    #[inline(always)]
    pub fn blend_epi32<const IMM: i32>(self, other: Self) -> Self {
        // SAFETY: AVX is compiled in (module docs); casts and a blend on
        // registers only.
        Self(unsafe {
            _mm256_castps_si256(_mm256_blend_ps::<IMM>(_mm256_castsi256_ps(self.0), _mm256_castsi256_ps(other.0)))
        })
    }

    /// Zero-extend the low 8 × u16 lanes to f32.
    #[inline(always)]
    pub fn to_f32x8_lo(self) -> F32x8 {
        F32x8(widen_u16_to_f32(split(self.0).0))
    }

    /// Zero-extend the high 8 × u16 lanes to f32.
    #[inline(always)]
    pub fn to_f32x8_hi(self) -> F32x8 {
        F32x8(widen_u16_to_f32(split(self.0).1))
    }
}

/// 8 × u16 (one xmm) → 8 × f32 (one ymm), exact.
#[inline(always)]
fn widen_u16_to_f32(v: __m128i) -> __m256 {
    // SAFETY: SSE2/SSE4.1 and AVX are compiled in (module docs); register-only
    // ops. `pmovzxwd` widens the low four lanes; the byte shift moves the high
    // four into place first.
    unsafe {
        let lo = _mm_cvtepu16_epi32(v);
        let hi = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(v));
        _mm256_cvtepi32_ps(join(lo, hi))
    }
}

impl Add for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(halves2!(_mm_add_epi16, self.0, rhs.0))
    }
}
impl Sub for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(halves2!(_mm_sub_epi16, self.0, rhs.0))
    }
}
impl Mul for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mullo(rhs)
    }
}
impl AddAssign for U16x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for U16x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl BitAnd for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_and_ps, self.0, rhs.0))
    }
}
impl BitOr for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_or_ps, self.0, rhs.0))
    }
}
impl BitXor for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_xor_ps, self.0, rhs.0))
    }
}
impl BitAndAssign for U16x16 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}
impl BitOrAssign for U16x16 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}
impl BitXorAssign for U16x16 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}
impl Not for U16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        // SAFETY: AVX is compiled in (module docs); register-only op.
        let ones = unsafe { _mm256_set1_epi16(-1) };
        Self(bits_ps!(_mm256_xor_ps, self.0, ones))
    }
}

// ---------------------------------------------------------------------------
// U8x32 (native __m256i)
// ---------------------------------------------------------------------------

impl U8x32 {
    /// Horizontal byte sum as u64 (does not wrap at 2^8).
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        let (lo, hi) = split(self.0);
        let mut t = [0u64; 4];
        // SAFETY: SSE2 is compiled in (module docs); `t` is `[u64; 4]`
        // (32 bytes), so the two 16-byte stores at byte offsets 0 and 16 are
        // in bounds.
        unsafe {
            let z = _mm_setzero_si128();
            _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, _mm_sad_epu8(lo, z));
            _mm_storeu_si128(t.as_mut_ptr().add(2) as *mut __m128i, _mm_sad_epu8(hi, z));
        }
        t[0] + t[1] + t[2] + t[3]
    }

    /// Lane-wise unsigned min.
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(halves2!(_mm_min_epu8, self.0, other.0))
    }

    /// Lane-wise unsigned max.
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(halves2!(_mm_max_epu8, self.0, other.0))
    }

    /// Per-lane equality as a 32-bit mask (bit `i` set iff `self[i] == other[i]`).
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u32 {
        movemask_epi8_256(halves2!(_mm_cmpeq_epi8, self.0, other.0))
    }

    /// Per-lane unsigned greater-than as a 32-bit mask (signed compare after
    /// flipping the sign bit of both operands, as on the AVX2 arm).
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u32 {
        // SAFETY: AVX is compiled in (module docs); register-only op.
        let bias = unsafe { _mm256_set1_epi8(i8::MIN) };
        let a = bits_ps!(_mm256_xor_ps, self.0, bias);
        let b = bits_ps!(_mm256_xor_ps, other.0, bias);
        movemask_epi8_256(halves2!(_mm_cmpgt_epi8, a, b))
    }

    /// MSB of each lane as a 32-bit mask.
    #[inline(always)]
    pub fn movemask(self) -> u32 {
        movemask_epi8_256(self.0)
    }

    /// Per-lane saturating unsigned add.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(halves2!(_mm_adds_epu8, self.0, other.0))
    }

    /// Per-lane saturating unsigned sub.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(halves2!(_mm_subs_epu8, self.0, other.0))
    }

    /// Per-lane unsigned rounded average, `(a + b + 1) >> 1`.
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        Self(halves2!(_mm_avg_epu8, self.0, other.0))
    }

    /// Right shift each 16-bit lane by `imm` bits (any count).
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        let (lo, hi) = split(self.0);
        // SAFETY: SSE2 is compiled in (module docs); register-only ops.
        unsafe {
            let c = _mm_cvtsi32_si128(imm as i32);
            Self(join(_mm_srl_epi16(lo, c), _mm_srl_epi16(hi, c)))
        }
    }

    /// Left shift each 16-bit lane by `imm` bits (any count).
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        let (lo, hi) = split(self.0);
        // SAFETY: SSE2 is compiled in (module docs); register-only ops.
        unsafe {
            let c = _mm_cvtsi32_si128(imm as i32);
            Self(join(_mm_sll_epi16(lo, c), _mm_sll_epi16(hi, c)))
        }
    }

    /// Within-128-bit-lane byte shuffle with `_mm256_shuffle_epi8`'s meaning:
    /// `pshufb` on each half is exactly that.
    #[inline(always)]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        Self(halves2!(_mm_shuffle_epi8, self.0, idx.0))
    }

    /// Interleave the low 8 bytes of each 128-bit half.
    #[inline(always)]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        Self(halves2!(_mm_unpacklo_epi8, self.0, other.0))
    }

    /// Interleave the high 8 bytes of each 128-bit half.
    #[inline(always)]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        Self(halves2!(_mm_unpackhi_epi8, self.0, other.0))
    }

    /// Select `a` where the mask lane's MSB is set, else `b`.
    #[inline(always)]
    pub fn mask_blend(mask: Self, a: Self, b: Self) -> Self {
        let (ml, mh) = split(mask.0);
        let (al, ah) = split(a.0);
        let (bl, bh) = split(b.0);
        // SAFETY: SSE4.1 is compiled in (module docs); register-only ops.
        Self(unsafe { join(_mm_blendv_epi8(bl, al, ml), _mm_blendv_epi8(bh, ah, mh)) })
    }
}

impl BitAnd for U8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_and_ps, self.0, rhs.0))
    }
}
impl BitOr for U8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_or_ps, self.0, rhs.0))
    }
}
impl BitXor for U8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(bits_ps!(_mm256_xor_ps, self.0, rhs.0))
    }
}
/// Wrapping add (use `saturating_add` to clamp).
impl Add for U8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(halves2!(_mm_add_epi8, self.0, rhs.0))
    }
}
/// Wrapping sub (use `saturating_sub` to clamp).
impl Sub for U8x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(halves2!(_mm_sub_epi8, self.0, rhs.0))
    }
}

// ---------------------------------------------------------------------------
// I8x32 / I16x16 (native __m256i, defined in simd_avx512.rs)
// ---------------------------------------------------------------------------

impl I8x32 {
    /// Wrapping lane-wise add.
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(halves2!(_mm_add_epi8, self.0, other.0))
    }

    /// Wrapping lane-wise sub.
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(halves2!(_mm_sub_epi8, self.0, other.0))
    }

    /// Lane-wise signed min (SSE4.1).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(halves2!(_mm_min_epi8, self.0, other.0))
    }

    /// Lane-wise signed max (SSE4.1).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(halves2!(_mm_max_epi8, self.0, other.0))
    }

    /// Lane-wise signed `self > other` as a 32-bit mask (lane 0 is bit 0).
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        movemask_epi8_256(halves2!(_mm_cmpgt_epi8, self.0, other.0))
    }

    /// Lane-wise `|x|` clamped to `i8::MAX` (`i8::MIN` → `i8::MAX`), via
    /// SSSE3 `pabsb` then an unsigned min with `0x7f`.
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        let (lo, hi) = split(self.0);
        // SAFETY: SSE2/SSSE3 are compiled in (module docs); register-only ops.
        unsafe {
            let cap = _mm_set1_epi8(0x7f);
            Self(join(_mm_min_epu8(_mm_abs_epi8(lo), cap), _mm_min_epu8(_mm_abs_epi8(hi), cap)))
        }
    }
}

impl Add for I8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        I8x32::add(self, rhs)
    }
}
impl Sub for I8x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        I8x32::sub(self, rhs)
    }
}
impl AddAssign for I8x32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = I8x32::add(*self, rhs);
    }
}
impl SubAssign for I8x32 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = I8x32::sub(*self, rhs);
    }
}

impl I16x16 {
    /// Wrapping lane-wise add.
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(halves2!(_mm_add_epi16, self.0, other.0))
    }

    /// Wrapping lane-wise sub.
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(halves2!(_mm_sub_epi16, self.0, other.0))
    }

    /// Lane-wise signed min.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(halves2!(_mm_min_epi16, self.0, other.0))
    }

    /// Lane-wise signed max.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(halves2!(_mm_max_epi16, self.0, other.0))
    }

    /// Lane-wise signed `self > other` as a 16-bit mask (lane 0 is bit 0).
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        let (al, ah) = split(self.0);
        let (bl, bh) = split(other.0);
        // SAFETY: SSE2 is compiled in (module docs); register-only ops.
        // `packs` narrows each all-ones/all-zeros 16-bit lane to one byte,
        // low half's lanes first, so the byte mask is the lane mask in order.
        unsafe {
            let packed = _mm_packs_epi16(_mm_cmpgt_epi16(al, bl), _mm_cmpgt_epi16(ah, bh));
            _mm_movemask_epi8(packed) as u16
        }
    }
}

impl Add for I16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        I16x16::add(self, rhs)
    }
}
impl Sub for I16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        I16x16::sub(self, rhs)
    }
}
impl AddAssign for I16x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = I16x16::add(*self, rhs);
    }
}
impl SubAssign for I16x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = I16x16::sub(*self, rhs);
    }
}
