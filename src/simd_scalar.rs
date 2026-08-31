//! Pure-Rust scalar fallback backend for `crate::simd::*`.
//!
//! Selected by `src/simd.rs` dispatch on non-x86_64 / non-aarch64
//! targets (wasm32, riscv, thumbv6m, etc.) when `feature =
//! "nightly-simd"` is OFF. Mirrors the API of `simd_avx512`,
//! `simd_avx2`, and `simd_neon::aarch64_simd` so consumer code reading
//! `use crate::simd::F32x16` compiles and runs uniformly across all
//! supported targets.
//!
//! Storage is plain `[$elem; $lanes]` arrays aligned to 64 bytes; the
//! arithmetic is loop-unrolled scalar Rust. No SIMD intrinsics — the
//! point is a correct fallback, not performance.
//!
//! The file was extracted from `simd.rs` in Phase 4 of the integration
//! plan in `.claude/knowledge/simd-dispatch-architecture.md` (split out
//! 1271 inline lines so the dispatcher reads as a re-export catalog
//! rather than 1.6k LoC of macro expansions).

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign,
    Neg, Not, Shl, Shr, Sub, SubAssign,
};

// ── Macros for scalar fallback boilerplate ────────────────────────

macro_rules! impl_float_type {
    ($name:ident, $elem:ty, $lanes:expr, $mask:ident, $mask_prim:ty) => {
        #[derive(Copy, Clone)]
        #[repr(align(64))]
        pub struct $name(pub [$elem; $lanes]);

        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self([0.0; $lanes])
            }
        }

        impl $name {
            pub const LANES: usize = $lanes;

            #[inline(always)]
            pub fn splat(v: $elem) -> Self {
                Self([v; $lanes])
            }

            #[inline(always)]
            pub fn from_slice(s: &[$elem]) -> Self {
                assert!(s.len() >= $lanes);
                let mut arr = [0.0 as $elem; $lanes];
                arr.copy_from_slice(&s[..$lanes]);
                Self(arr)
            }

            #[inline(always)]
            pub fn from_array(arr: [$elem; $lanes]) -> Self {
                Self(arr)
            }

            #[inline(always)]
            pub fn to_array(self) -> [$elem; $lanes] {
                self.0
            }

            #[inline(always)]
            pub fn copy_to_slice(self, s: &mut [$elem]) {
                assert!(s.len() >= $lanes);
                s[..$lanes].copy_from_slice(&self.0);
            }

            #[inline(always)]
            pub fn reduce_sum(self) -> $elem {
                self.0.iter().sum()
            }

            #[inline(always)]
            pub fn reduce_min(self) -> $elem {
                self.0.iter().copied().fold(<$elem>::INFINITY, <$elem>::min)
            }

            #[inline(always)]
            pub fn reduce_max(self) -> $elem {
                self.0
                    .iter()
                    .copied()
                    .fold(<$elem>::NEG_INFINITY, <$elem>::max)
            }

            #[inline(always)]
            pub fn simd_min(self, other: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].min(other.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_max(self, other: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].max(other.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
                self.simd_max(lo).simd_min(hi)
            }

            #[inline(always)]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].mul_add(b.0[i], c.0[i]);
                }
                Self(out)
            }

            #[inline(always)]
            pub fn sqrt(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].sqrt();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn round(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].round();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn floor(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].floor();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn abs(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].abs();
                }
                Self(out)
            }

            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] < other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_le(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] <= other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> $mask {
                other.simd_lt(self)
            }

            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> $mask {
                other.simd_le(self)
            }

            #[inline(always)]
            pub fn simd_eq(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] == other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }

            #[inline(always)]
            pub fn simd_ne(self, other: Self) -> $mask {
                let mut bits: $mask_prim = 0;
                for i in 0..$lanes {
                    if self.0[i] != other.0[i] {
                        bits |= 1 << i;
                    }
                }
                $mask(bits)
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] + rhs.0[i];
                }
                Self(out)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] - rhs.0[i];
                }
                Self(out)
            }
        }
        impl Mul for $name {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] * rhs.0[i];
                }
                Self(out)
            }
        }
        impl Div for $name {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] / rhs.0[i];
                }
                Self(out)
            }
        }
        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] += rhs.0[i];
                }
            }
        }
        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] -= rhs.0[i];
                }
            }
        }
        impl MulAssign for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] *= rhs.0[i];
                }
            }
        }
        impl DivAssign for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] /= rhs.0[i];
                }
            }
        }
        impl Neg for $name {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = -self.0[i];
                }
                Self(out)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        // Mask type
        #[derive(Copy, Clone, Debug)]
        pub struct $mask(pub $mask_prim);

        impl $mask {
            #[inline(always)]
            pub fn select(self, true_val: $name, false_val: $name) -> $name {
                let mut out = [0.0 as $elem; $lanes];
                for i in 0..$lanes {
                    out[i] = if (self.0 >> i) & 1 == 1 {
                        true_val.0[i]
                    } else {
                        false_val.0[i]
                    };
                }
                $name(out)
            }
        }
    };
}

macro_rules! impl_int_type {
    ($name:ident, $elem:ty, $lanes:expr, $zero:expr) => {
        #[derive(Copy, Clone)]
        #[repr(align(64))]
        pub struct $name(pub [$elem; $lanes]);

        impl Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self([$zero; $lanes])
            }
        }

        impl $name {
            pub const LANES: usize = $lanes;

            #[inline(always)]
            pub fn splat(v: $elem) -> Self {
                Self([v; $lanes])
            }

            #[inline(always)]
            pub fn from_slice(s: &[$elem]) -> Self {
                assert!(s.len() >= $lanes);
                let mut arr = [$zero; $lanes];
                arr.copy_from_slice(&s[..$lanes]);
                Self(arr)
            }

            #[inline(always)]
            pub fn from_array(arr: [$elem; $lanes]) -> Self {
                Self(arr)
            }

            #[inline(always)]
            pub fn to_array(self) -> [$elem; $lanes] {
                self.0
            }

            #[inline(always)]
            pub fn copy_to_slice(self, s: &mut [$elem]) {
                assert!(s.len() >= $lanes);
                s[..$lanes].copy_from_slice(&self.0);
            }

            #[inline(always)]
            pub fn reduce_sum(self) -> $elem {
                let mut s: $elem = $zero;
                for i in 0..$lanes {
                    s = s.wrapping_add(self.0[i]);
                }
                s
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].wrapping_add(rhs.0[i]);
                }
                Self(out)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i].wrapping_sub(rhs.0[i]);
                }
                Self(out)
            }
        }
        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] = self.0[i].wrapping_add(rhs.0[i]);
                }
            }
        }
        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] = self.0[i].wrapping_sub(rhs.0[i]);
                }
            }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] & rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] | rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = self.0[i] ^ rhs.0[i];
                }
                Self(out)
            }
        }
        impl BitAndAssign for $name {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] &= rhs.0[i];
                }
            }
        }
        impl BitOrAssign for $name {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] |= rhs.0[i];
                }
            }
        }
        impl BitXorAssign for $name {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self.0[i] ^= rhs.0[i];
                }
            }
        }
        impl Not for $name {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self {
                let mut out = [$zero; $lanes];
                for i in 0..$lanes {
                    out[i] = !self.0[i];
                }
                Self(out)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
    };
}

// ── Instantiate all 11 types ─────────────────────────────────────

// 512-bit float types
impl_float_type!(F32x16, f32, 16, F32Mask16, u16);
impl_float_type!(F64x8, f64, 8, F64Mask8, u8);

// 256-bit AVX2 float types
// The macro `impl_float_type!` already emits `pub struct $mask(pub $mask_prim);`,
// so calling it with `F32Mask8Scalar` / `F64Mask4Scalar` defines those mask
// structs. The previous explicit re-declaration was a duplicate that
// tripped E0428 + 6× E0119 on i686-unknown-linux-gnu (where this scalar
// module compiles — `#[cfg(not(target_arch = "x86_64"))]`).
impl_float_type!(F32x8, f32, 8, F32Mask8Scalar, u8);
impl_float_type!(F64x4, f64, 4, F64Mask4Scalar, u8);

// 512-bit integer types
impl_int_type!(U8x64, u8, 64, 0u8);
impl_int_type!(I32x16, i32, 16, 0i32);
impl_int_type!(I64x8, i64, 8, 0i64);
impl_int_type!(U16x32, u16, 32, 0u16);
impl_int_type!(U32x16, u32, 16, 0u32);
impl_int_type!(U64x8, u64, 8, 0u64);

/// u64 ARX rotate — the BLAKE2b / argon2 lane.
///
/// Scalar per-lane loops, and **measured not to vectorize**: the codegen
/// oracle tried three spellings (`u64::rotate_right`, an explicit shift-or
/// with a runtime amount, and the same with BLAKE2b's constants 32/24/16/63)
/// and every one came back 0 packed, one `rorq` per lane. LLVM declines the
/// 64-bit *operation*, not the rotate *idiom* — it folded two of the probes
/// into byte-identical code.
///
/// So unlike every other lane-wise op in this crate, the scalar spec is NOT
/// the implementation here. The native `VPROLVQ`/`VPRORVQ` override lives on
/// `simd_avx512`'s `U64x8`, which is a real `__m512i`; these arms are the
/// correct-but-unvectorized fallback, and that is a known cost rather than an
/// oversight. See `.claude/knowledge/crypto-lane-status.md`.
impl U64x8 {
    /// Lane-wise left-rotate by `n` bits. `n` is taken mod 64.
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
}

// I8/I16 SIMD types (scalar fallback)
impl_int_type!(I8x64, i8, 64, 0i8);
impl_int_type!(I8x32, i8, 32, 0i8);
impl_int_type!(I16x32, i16, 32, 0i16);
impl_int_type!(I16x16, i16, 16, 0i16);

// 256-bit int lanes — scalar polyfills filling the gap surfaced by the
// 2026-05-20 matrix audit. Mirror the additions in `src/simd_avx2.rs`
// (via the `avx2_int_type!` macro) so consumers on every backend reach
// the same type names through `crate::simd::*`.
impl_int_type!(U16x16, u16, 16, 0u16);
impl_int_type!(U32x8, u32, 8, 0u32);
impl_int_type!(U64x4, u64, 4, 0u64);
impl_int_type!(I32x8, i32, 8, 0i32);
impl_int_type!(I64x4, i64, 4, 0i64);

// ── U32x8 shuffle + rotate surface — scalar mirror of `simd_avx2.rs` ────────
//
// Same six methods, same semantics, same doc contract. The AVX2 arm's bodies
// are ALSO plain index loops (see the long comment there and
// `.claude/knowledge/blake3-on-ndarray-simd.md`), so this is not a "fallback"
// that behaves differently — the two arms are the same source shape, and the
// parity tests in `simd.rs` bind them to the real x86 intrinsics on the one
// backend where those exist.
//
// Semantics are x86's, INCLUDING the per-128-bit-lane split, on every
// backend. That is not an x86 leak: BLAKE3's transpose network is defined in
// terms of it, so a backend that "helpfully" used whole-vector interleave
// would compute a different permutation and produce wrong hashes.
impl U32x8 {
    /// Lane-wise `u32::rotate_left(n)`. See the AVX2 arm for the
    /// `rotr(n) == rotate_left(32 - n)` note.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let mut out = [0u32; 8];
        for i in 0..8 {
            out[i] = self.0[i].rotate_left(n);
        }
        Self(out)
    }

    /// `_mm256_unpacklo_epi32`: `[a0,b0,a1,b1, a4,b4,a5,b5]`.
    #[inline(always)]
    pub fn interleave_lo_u32(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]])
    }

    /// `_mm256_unpackhi_epi32`: `[a2,b2,a3,b3, a6,b6,a7,b7]`.
    #[inline(always)]
    pub fn interleave_hi_u32(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]])
    }

    /// `_mm256_unpacklo_epi64`: `[a0,a1,b0,b1, a4,a5,b4,b5]`.
    #[inline(always)]
    pub fn interleave_lo_u64(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[0], a[1], b[0], b[1], a[4], a[5], b[4], b[5]])
    }

    /// `_mm256_unpackhi_epi64`: `[a2,a3,b2,b3, a6,a7,b6,b7]`.
    #[inline(always)]
    pub fn interleave_hi_u64(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[2], a[3], b[2], b[3], a[6], a[7], b[6], b[7]])
    }

    /// `_mm256_permute2x128_si256(a, b, 0x20)`: `[a0..a3, b0..b3]`.
    #[inline(always)]
    pub fn concat_lo_halves(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]])
    }

    /// `_mm256_permute2x128_si256(a, b, 0x31)`: `[a4..a7, b4..b7]`.
    #[inline(always)]
    pub fn concat_hi_halves(self, other: Self) -> Self {
        let (a, b) = (self.0, other.0);
        Self([a[4], a[5], a[6], a[7], b[4], b[5], b[6], b[7]])
    }
}

// I8x64 / I8x32 / I16x32 / I16x16 — AVX-512BW-style methods (scalar shape)
impl I8x64 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i8; 64])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i8; 64];
        for i in 0..64 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] > other.0[i] {
                m |= 1u64 << i;
            }
        }
        m
    }
}
impl I8x32 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i8; 32])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        let mut m: u32 = 0;
        for i in 0..32 {
            if self.0[i] > other.0[i] {
                m |= 1u32 << i;
            }
        }
        m
    }
}
impl I16x32 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i16; 32])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i16; 32];
        for i in 0..32 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u32 {
        let mut m: u32 = 0;
        for i in 0..32 {
            if self.0[i] > other.0[i] {
                m |= 1u32 << i;
            }
        }
        m
    }
}
impl I16x16 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([0i16; 16])
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].wrapping_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].min(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        let mut m: u16 = 0;
        for i in 0..16 {
            if self.0[i] > other.0[i] {
                m |= 1u16 << i;
            }
        }
        m
    }
}

// Extra methods for U16x32 (widen/narrow, shift, multiply)
impl U16x32 {
    #[inline(always)]
    pub fn from_u8x64_lo(v: U8x64) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = v.0[i] as u16;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn from_u8x64_hi(v: U8x64) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = v.0[32 + i] as u16;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn pack_saturate_u8(self, other: Self) -> U8x64 {
        let mut out = [0u8; 64];
        for i in 0..32 {
            out[i] = self.0[i].min(255) as u8;
        }
        for i in 0..32 {
            out[32 + i] = other.0[i].min(255) as u8;
        }
        U8x64(out)
    }
    #[inline(always)]
    pub fn shr(self, imm: u32) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = if imm < 16 { self.0[i] >> imm } else { 0 };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn shl(self, imm: u32) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = if imm < 16 { self.0[i] << imm } else { 0 };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn mullo(self, other: Self) -> Self {
        let mut out = [0u16; 32];
        for i in 0..32 {
            out[i] = self.0[i].wrapping_mul(other.0[i]);
        }
        Self(out)
    }
}

// Extra methods for I32x16 that float types have via the macro
impl I32x16 {
    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cast_f32(self) -> F32x16 {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i] as f32;
        }
        F32x16(out)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].abs();
        }
        Self(out)
    }
    #[inline(always)]
    pub fn from_i16_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 16);
        let mut o = [0i32; 16];
        for i in 0..16 {
            o[i] = s[i] as i32;
        }
        Self(o)
    }
    #[inline(always)]
    pub fn to_i16_array(self) -> [i16; 16] {
        let mut o = [0i16; 16];
        for i in 0..16 {
            o[i] = self.0[i] as i16;
        }
        o
    }
    #[inline(always)]
    pub fn cmpge_zero_mask(self) -> u16 {
        let mut mask = 0u16;
        for i in 0..16 {
            if self.0[i] >= 0 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Lane-wise **signed** greater-than as a packed 16-bit bitmask.
    ///
    /// Bit `i` of the result is set iff `self.lane(i) > other.lane(i)` under
    /// two's-complement signed ordering. Bit order is **LSB-first**: lane `0`
    /// occupies bit `0`. Same convention as [`Self::cmpge_zero_mask`].
    ///
    /// Edge cases (all exact; no saturation, wrapping, or clamping):
    /// * `i32::MIN` as the threshold is set for every lane strictly greater
    ///   than it, and clear for lanes equal to `i32::MIN`.
    /// * `i32::MAX` as the threshold yields `0` — no `i32` exceeds it.
    /// * Comparison is signed, *not* bit-pattern: `-1 > 0` is `false`.
    ///
    /// This is the **scalar correctness anchor** for the primitive: the
    /// AVX-512 arm (`VPCMPGTD` → `__mmask16`) and the AVX2 / NEON / wasm
    /// index-loop arms are all required to agree with this body bit-for-bit.
    #[inline(always)]
    pub fn gt_bitmask(self, other: Self) -> u16 {
        let mut mask = 0u16;
        for i in 0..16 {
            if self.0[i] > other.0[i] {
                mask |= 1 << i;
            }
        }
        mask
    }
}

impl Mul for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for I32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Neg for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

// Extra for F32x16: to_bits/from_bits/cast_i32
impl F32x16 {
    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i].to_bits();
        }
        U32x16(out)
    }
    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = f32::from_bits(bits.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cast_i32(self) -> I32x16 {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = self.0[i] as i32;
        }
        I32x16(out)
    }
}

// Extra for F64x8: to_bits/from_bits
impl F64x8 {
    #[inline(always)]
    pub fn to_bits(self) -> U64x8 {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i].to_bits();
        }
        U64x8(out)
    }
    #[inline(always)]
    pub fn from_bits(bits: U64x8) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 {
            out[i] = f64::from_bits(bits.0[i]);
        }
        Self(out)
    }
}

// Extra for I64x8
impl I64x8 {
    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].abs();
        }
        Self(out)
    }
}

impl Mul for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for I64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Neg for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

// Shift operators for U32x16
impl Shr<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i] >> rhs.0[i];
        }
        Self(out)
    }
}
impl Shl<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i] << rhs.0[i];
        }
        Self(out)
    }
}

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
    /// Lane-wise left-rotate by `n` bits — the ARX rotate (matches
    /// `u32::rotate_left`). The completeness tier: a full, correct
    /// implementation held to the same bar as the SIMD tiers, and the bit-exact
    /// reference they are parity-checked against. Delegates to `u32::rotate_left`
    /// per lane (`rotate_left(0)` and `rotate_left(32)` both no-op, matching std).
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i].rotate_left(n);
        }
        Self(out)
    }

    /// Lane-wise equality as a packed 16-bit bitmask.
    ///
    /// Bit `i` of the result is set iff `self.lane(i) == other.lane(i)`. Bit
    /// order is **LSB-first**: lane `0` occupies bit `0`. Same convention as
    /// [`I32x16::cmpge_zero_mask`] and [`I32x16::gt_bitmask`].
    ///
    /// Edge cases: equality is exact bitwise comparison over the full 32-bit
    /// range, so `u32::MAX` and `0` behave like any other value — no
    /// saturation, wrapping, or signedness question arises.
    ///
    /// This is the **scalar correctness anchor** for the primitive: the
    /// AVX-512 arm (`VPCMPEQD` → `__mmask16`) and the AVX2 / NEON / wasm
    /// index-loop arms are all required to agree with this body bit-for-bit.
    #[inline(always)]
    pub fn eq_bitmask(self, other: Self) -> u16 {
        let mut mask = 0u16;
        for i in 0..16 {
            if self.0[i] == other.0[i] {
                mask |= 1 << i;
            }
        }
        mask
    }
}

// Shift operators for U64x8
impl Shr<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i] >> rhs.0[i];
        }
        Self(out)
    }
}
impl Shl<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i] << rhs.0[i];
        }
        Self(out)
    }
}

// Mul for U8x64 (wrapping)
impl Mul for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}
impl MulAssign for U8x64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// U8x64 extra methods — byte-level operations for palette codec, nibble, byte scan
impl U8x64 {
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        *self.0.iter().min().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        *self.0.iter().max().unwrap_or(&0)
    }
    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if self.0[i] == other.0[i] {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    pub fn shr_epi16(self, imm: u32) -> Self {
        let mut out = [0u8; 64];
        for i in (0..64).step_by(2) {
            let val = u16::from_le_bytes([self.0[i], self.0[i + 1]]);
            let shifted = val >> imm;
            let bytes = shifted.to_le_bytes();
            out[i] = bytes[0];
            out[i + 1] = bytes[1];
        }
        Self(out)
    }
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].saturating_sub(other.0[i]);
        }
        Self(out)
    }
    // ── Tier 1: seismon rasterizer primitives (scalar fallbacks) ──
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = ((self.0[i] as u16 + other.0[i] as u16 + 1) >> 1) as u8;
        }
        Self(out)
    }
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] > other.0[i] {
                m |= 1 << i;
            }
        }
        m
    }
    #[inline(always)]
    pub fn mask_blend(mask: u64, a: Self, b: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = if mask & (1 << i) != 0 { b.0[i] } else { a.0[i] };
        }
        Self(out)
    }
    #[inline(always)]
    pub fn shl_epi16(self, imm: u32) -> Self {
        let mut out = [0u8; 64];
        for i in (0..64).step_by(2) {
            let v = u16::from_le_bytes([self.0[i], self.0[i + 1]]);
            let s = if imm < 16 { v << imm } else { 0 };
            let b = s.to_le_bytes();
            out[i] = b[0];
            out[i + 1] = b[1];
        }
        Self(out)
    }
    // ── Tier 2: sprite blit + palette remap (scalar fallbacks) ──
    #[inline(always)]
    pub unsafe fn mask_store(self, ptr: *mut u8, mask: u64) {
        for i in 0..64 {
            if mask & (1 << i) != 0 {
                *ptr.add(i) = self.0[i];
            }
        }
    }
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[i].saturating_add(other.0[i]);
        }
        Self(out)
    }
    #[inline(always)]
    pub fn permute_bytes(self, idx: Self) -> Self {
        let mut out = [0u8; 64];
        for i in 0..64 {
            out[i] = self.0[(idx.0[i] & 63) as usize];
        }
        Self(out)
    }
    #[inline(always)]
    pub fn movemask(self) -> u64 {
        let mut m: u64 = 0;
        for i in 0..64 {
            if self.0[i] & 0x80 != 0 {
                m |= 1 << i;
            }
        }
        m
    }
    #[inline(always)]
    pub fn unpack_lo_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..8 {
                out[b + i * 2] = self.0[b + i];
                out[b + i * 2 + 1] = other.0[b + i];
            }
        }
        Self(out)
    }
    #[inline(always)]
    pub fn unpack_hi_epi8(self, other: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..8 {
                out[b + i * 2] = self.0[b + 8 + i];
                out[b + i * 2 + 1] = other.0[b + 8 + i];
            }
        }
        Self(out)
    }
    /// Byte-wise shuffle: use `self` as a LUT, `idx` selects bytes within each 128-bit (16-byte) lane.
    #[inline(always)]
    pub fn shuffle_bytes(self, idx: Self) -> Self {
        let mut out = [0u8; 64];
        for lane in 0..4 {
            let b = lane * 16;
            for i in 0..16 {
                out[b + i] = self.0[b + (idx.0[b + i] & 0x0F) as usize];
            }
        }
        Self(out)
    }
    /// Sum all 64 bytes into a single `u64` without wrapping.
    #[inline(always)]
    pub fn sum_bytes_u64(self) -> u64 {
        self.0.iter().map(|&b| b as u64).sum()
    }
    /// Build a nibble-popcount lookup table (replicated across 4 x 16-byte lanes).
    #[inline(always)]
    pub fn nibble_popcount_lut() -> Self {
        let lane: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
        let mut arr = [0u8; 64];
        for l in 0..4 {
            arr[l * 16..(l + 1) * 16].copy_from_slice(&lane);
        }
        Self(arr)
    }
}

// Mul for U32x16
impl Mul for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        Self(out)
    }
}

// ============================================================================
// W1a SIMD primitives — scalar backend
// ============================================================================
//
// The scalar backend is the correctness anchor for all W1a primitives.
// All implementations here are pure safe Rust with no intrinsics.

// ── W1a-#1: I8x16 + lane_i8 + from_i4_packed_u64 (scalar) ──────────────────

/// 16-lane `i8` vector — scalar fallback for non-NEON, non-x86_64 targets.
///
/// On x86_64 this type comes from `simd_avx512.rs`; on aarch64 from
/// `simd_neon.rs`.  This scalar version covers wasm32, riscv, and any other
/// target that falls through to the scalar dispatch arm.
#[derive(Copy, Clone, PartialEq)]
#[repr(align(16))]
pub struct I8x16(pub [i8; 16]);

impl I8x16 {
    pub const LANES: usize = 16;

    /// Broadcast a single `i8` value to all 16 lanes.
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self([v; 16])
    }

    /// Load from a slice (at least 16 elements required).
    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 16);
        let mut a = [0i8; 16];
        a.copy_from_slice(&s[..16]);
        Self(a)
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [i8; 16]) -> Self {
        Self(arr)
    }

    /// Extract all 16 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        self.0
    }

    /// Copy lanes into a slice (must have at least 16 elements).
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 16);
        s[..16].copy_from_slice(&self.0);
    }

    /// Unpack 16 signed i4 nibbles from a `u64` into 16 sign-extended `i8` lanes.
    ///
    /// Nibble layout: `lane[i] = sign_extend_i4((packed >> (4*i)) & 0xf)`.
    /// Values `0x0..=0x7` → `0..=7`; values `0x8..=0xf` → `-8..=-1`.
    ///
    /// Edge cases:
    /// - `from_i4_packed_u64(0)` → all lanes `0`.
    /// - All nibbles `0xf` → all lanes `-1`.
    /// - Nibble `0x8` → lane value `-8` (minimum i4 value).
    ///
    /// # Example
    /// ```rust,ignore
    /// let z = I8x16::from_i4_packed_u64(0);
    /// assert!(z.to_array().iter().all(|&x| x == 0));
    /// let neg = I8x16::from_i4_packed_u64(u64::MAX);
    /// assert!(neg.to_array().iter().all(|&x| x == -1));
    /// ```
    #[inline(always)]
    pub fn from_i4_packed_u64(packed: u64) -> Self {
        let mut lanes = [0i8; 16];
        for i in 0..16 {
            let nibble = ((packed >> (4 * i)) & 0xf) as i8;
            lanes[i] = if nibble > 7 { nibble - 16 } else { nibble };
        }
        Self(lanes)
    }

    /// Extract lane `N` as an `i8`.  `N` must be in `0..16`.
    #[inline(always)]
    pub fn lane_i8<const N: usize>(self) -> i8 {
        self.0[N]
    }

    /// Lane-wise saturating absolute value.
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX` (127).  Uses `i8::saturating_abs`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x16::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        let mut o = [0i8; 16];
        for i in 0..16 {
            o[i] = self.0[i].saturating_abs();
        }
        Self(o)
    }
}

impl core::fmt::Debug for I8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I8x16({:?})", &self.0[..])
    }
}

// ── W1a-#2: I8x32::saturating_abs (scalar) ───────────────────────────────────

impl I8x32 {
    /// Lane-wise saturating absolute value.
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX`.  All 32 lanes via `i8::saturating_abs`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x32::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].saturating_abs();
        }
        Self(o)
    }
}

// ── W1a-#3: U16x8 / U8x8 / palette_lookup_u8x8 (scalar) ─────────────────────

/// 8-lane `u16` vector — scalar fallback.
///
/// On aarch64 this type is backed by `uint16x8_t`; on x86_64 it is a scalar-
/// storage polyfill in `simd_avx512.rs`.  This version covers all other targets.
#[derive(Copy, Clone, PartialEq)]
#[repr(align(16))]
pub struct U16x8(pub [u16; 8]);

impl U16x8 {
    pub const LANES: usize = 8;

    /// Broadcast a single `u16` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self([v; 8])
    }

    /// Load from a slice (at least 8 elements required).
    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8);
        let mut a = [0u16; 8];
        a.copy_from_slice(&s[..8]);
        Self(a)
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u16; 8]) -> Self {
        Self(arr)
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        self.0
    }

    /// Gather 8 `u16` values from `table` at the indices in `self`.
    ///
    /// In debug panics if any index `>= table.len()`.  In release, OOB
    /// indices return 0 safely.
    ///
    /// # Example
    /// ```rust,ignore
    /// let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
    /// let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
    /// let r = U16x8::gather_u16(idx, &table);
    /// assert_eq!(r.to_array(), [10, 30, 50, 70, 20, 40, 60, 80]);
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
        Self(out)
    }

    /// Extract lane `k` as a `u16`.
    #[inline(always)]
    pub fn lane(self, k: usize) -> u16 {
        self.0[k]
    }
}

impl fmt::Debug for U16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U16x8({:?})", &self.0[..])
    }
}

/// 8-lane `u8` vector — scalar fallback.  Used as the return type of
/// `palette_lookup_u8x8`.
#[derive(Copy, Clone, PartialEq)]
#[repr(align(8))]
pub struct U8x8(pub [u8; 8]);

impl U8x8 {
    pub const LANES: usize = 8;

    /// Broadcast a single `u8` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self([v; 8])
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u8; 8]) -> Self {
        Self(arr)
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 8] {
        self.0
    }
}

impl fmt::Debug for U8x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U8x8({:?})", &self.0[..])
    }
}

/// Look up 8 bytes from a `u8` LUT by `u16` indices (scalar fallback).
///
/// Panics in debug on OOB; returns 0 safely in release.
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
    U8x8(out)
}

// ── W1a-#4: prefetch_read_t0/t1/t2 (scalar / other arch) ────────────────────

/// Hint that `ptr` will be read soon (scalar / unknown-arch no-op).
///
/// On `x86_64` the real implementation in `simd_avx512.rs` emits `PREFETCHT0`.
/// On `aarch64` the real implementation in `simd_neon.rs` emits `prfm`.
/// On all other targets (wasm, riscv, …) this is a deliberate no-op because
/// the prefetch contract is a hint — silent no-op is correct per the spec.
///
/// `ptr` may be invalid; it is never dereferenced.
#[inline(always)]
pub fn prefetch_read_t0(_ptr: *const u8) {
    // no-op on unknown/scalar targets
}

/// Hint to load into L2 (T1) cache (scalar / unknown-arch no-op).
#[inline(always)]
pub fn prefetch_read_t1(_ptr: *const u8) {
    // no-op on unknown/scalar targets
}

/// Hint to load into L3 (T2) cache (scalar / unknown-arch no-op).
#[inline(always)]
pub fn prefetch_read_t2(_ptr: *const u8) {
    // no-op on unknown/scalar targets
}

// ── W1a-#5: U64x8::popcnt / xor_popcount + U64x4::popcnt (scalar) ───────────

impl U64x8 {
    /// Lane-wise population count (scalar).  Each lane → set-bit count (0..=64).
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = U64x8::splat(u64::MAX);
    /// assert!(v.popcnt().to_array().iter().all(|&x| x == 64));
    /// let z = U64x8::splat(0);
    /// assert!(z.popcnt().to_array().iter().all(|&x| x == 0));
    /// ```
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        let mut out = [0u64; 8];
        for i in 0..8 {
            out[i] = self.0[i].count_ones() as u64;
        }
        Self(out)
    }

    /// XOR two vectors lane-wise, popcount each lane, sum all 8 lanes.
    ///
    /// # Example
    /// ```rust,ignore
    /// let a = U64x8::splat(u64::MAX);
    /// let b = U64x8::splat(0);
    /// assert_eq!(a.xor_popcount(b), 512); // 64 bits × 8 lanes
    /// assert_eq!(a.xor_popcount(a), 0);   // same inputs → Hamming distance 0
    /// ```
    #[inline(always)]
    pub fn xor_popcount(self, other: Self) -> u64 {
        let mut sum = 0u64;
        for i in 0..8 {
            sum += (self.0[i] ^ other.0[i]).count_ones() as u64;
        }
        sum
    }
}

impl U64x4 {
    /// Lane-wise population count (scalar).  Each lane → set-bit count (0..=64).
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = U64x4::from_array([u64::MAX, 0, 1, !1]);
    /// assert_eq!(v.popcnt().to_array(), [64, 0, 1, 63]);
    /// ```
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = self.0[i].count_ones() as u64;
        }
        Self(out)
    }
}

// ── W1a-#1: batch_packed_i4_16 (scalar backend) ──────────────────────────────

/// Closure-parameterised batch over packed i4 data (scalar backend).
///
/// Iterates `min(packed.len(), aux.len(), out.len())` times.  Each iteration
/// unpacks `packed[i]` into an `I8x16` (16 sign-extended nibbles) and passes
/// it together with `aux[i]` to `f`, storing the result in `out[i]`.
///
/// Panics if `packed.len() != aux.len()`.
#[inline]
pub fn batch_packed_i4_16<E, F>(packed: &[u64], aux: &[i8], out: &mut [E], f: F)
where
    F: Fn(I8x16, i8) -> E + Sync + Send,
    E: Copy,
{
    assert_eq!(packed.len(), aux.len(), "batch_packed_i4_16: packed and aux must be same length");
    let n = packed.len().min(out.len());
    for i in 0..n {
        let lanes = I8x16::from_i4_packed_u64(packed[i]);
        out[i] = f(lanes, aux[i]);
    }
}

// ── Lowercase aliases ─────────────────────────────────────────────────────────
#[allow(non_camel_case_types)]
pub type i8x16 = I8x16;
#[allow(non_camel_case_types)]
pub type u16x8 = U16x8;
#[allow(non_camel_case_types)]
pub type u8x8 = U8x8;

// Lowercase aliases
#[allow(non_camel_case_types)]
pub type f32x16 = F32x16;
#[allow(non_camel_case_types)]
pub type f64x8 = F64x8;
#[allow(non_camel_case_types)]
pub type u8x64 = U8x64;
#[allow(non_camel_case_types)]
pub type i32x16 = I32x16;
#[allow(non_camel_case_types)]
pub type i64x8 = I64x8;
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;
#[allow(non_camel_case_types)]
pub type u64x8 = U64x8;
#[allow(non_camel_case_types)]
pub type f32x8 = F32x8;
#[allow(non_camel_case_types)]
pub type f64x4 = F64x4;
#[allow(non_camel_case_types)]
pub type i8x64 = I8x64;
#[allow(non_camel_case_types)]
pub type i8x32 = I8x32;
#[allow(non_camel_case_types)]
pub type i16x32 = I16x32;
#[allow(non_camel_case_types)]
pub type i16x16 = I16x16;
// Lowercase aliases for the 256-bit polyfills added in the 2026-05-20
// missing-lanes sweep.
#[allow(non_camel_case_types)]
pub type u16x16 = U16x16;
#[allow(non_camel_case_types)]
pub type u32x8 = U32x8;
#[allow(non_camel_case_types)]
pub type u64x4 = U64x4;
#[allow(non_camel_case_types)]
pub type i32x8 = I32x8;
#[allow(non_camel_case_types)]
pub type i64x4 = I64x4;

// ── W1a-#9: U64x8 / U32x16 :: andnot + ternlog (portable backend) ───────────
//
// Masked projection, never traversal. The geometry is fixed and identical on
// every architecture, so these are whole-register operations composed from the
// `BitAnd` / `BitOr` / `Not` this type already carries — there is no lane
// index anywhere below. LLVM lowers the same source to `vpand`/`vpandn` on
// ymm (v3), `vandq_u64`/`vbicq_u64` on NEON, and `v128_and`/`v128_andnot` on
// wasm; the `repr(align(64))` backing is what earns the aligned moves.
//
// `IMM` is a const generic, so each `if IMM & bit` folds at compile time and
// only the minterms the truth table names survive. `AND3` (0x80) reduces to
// two ANDs of the whole register.

impl U64x8 {
    /// Set difference: `self & !other`.
    ///
    /// **Argument order differs from the raw Intel intrinsic.**
    /// `_mm*_andnot_si*(a, b)` computes `!a & b`; this computes
    /// `self & !other` — "self minus other". Every backend, same direction.
    ///
    /// Total function: no saturation, no overflow, no UB. `x.andnot(x)` is
    /// zero; `x.andnot(U64x8::splat(0))` is `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::U64x8;
    /// let a = U64x8::splat(0b1100);
    /// let b = U64x8::splat(0b1010);
    /// assert_eq!(a.andnot(b).to_array()[0], 0b0100); // a & !b
    /// ```
    #[inline(always)]
    pub fn andnot(self, other: Self) -> Self {
        self & !other
    }

    /// Any 3-input boolean function of `self`, `b` and `c`, selected by the
    /// const truth-table immediate `IMM`.
    ///
    /// Per bit position: `index = (self << 2) | (b << 1) | c`, result bit =
    /// `(IMM >> index) & 1` — Intel's VPTERNLOG convention, matched exactly by
    /// every backend. `IMM` is `i32` to mirror the intrinsic's signature; only
    /// `0..=255` is legal, enforced at compile time on EVERY backend (here by
    /// an inline const assert, on AVX-512 by the intrinsic's own static
    /// assert). Named immediates live in `crate::simd::ternlog`. Within that
    /// domain: total function, no lane interaction.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::{ternlog, U64x8};
    /// let (a, b, c) = (U64x8::splat(0b1100), U64x8::splat(0b1010), U64x8::splat(0b1001));
    /// let maj = a.ternlog::<{ ternlog::MAJ3 }>(b, c); // two-of-three majority
    /// assert_eq!(maj.to_array()[0], 0b1000);
    /// ```
    #[inline(always)]
    pub fn ternlog<const IMM: i32>(self, b: Self, c: Self) -> Self {
        const { assert!(IMM >= 0 && IMM <= 255, "ternlog IMM is an 8-bit truth table") }
        let (a, z) = (self, Self::splat(0));
        let mut r = z;
        if IMM & 0x01 != 0 {
            r = r | !a & !b & !c;
        }
        if IMM & 0x02 != 0 {
            r = r | !a & !b & c;
        }
        if IMM & 0x04 != 0 {
            r = r | !a & b & !c;
        }
        if IMM & 0x08 != 0 {
            r = r | !a & b & c;
        }
        if IMM & 0x10 != 0 {
            r = r | a & !b & !c;
        }
        if IMM & 0x20 != 0 {
            r = r | a & !b & c;
        }
        if IMM & 0x40 != 0 {
            r = r | a & b & !c;
        }
        if IMM & 0x80 != 0 {
            r = r | a & b & c;
        }
        r
    }
}

impl U32x16 {
    /// Set difference: `self & !other`. See [`U64x8::andnot`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::U32x16;
    /// let a = U32x16::splat(0b1100);
    /// let b = U32x16::splat(0b1010);
    /// assert_eq!(a.andnot(b).to_array()[0], 0b0100); // a & !b
    /// ```
    #[inline(always)]
    pub fn andnot(self, other: Self) -> Self {
        self & !other
    }

    /// Any 3-input boolean function, 32-bit lanes. See [`U64x8::ternlog`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::{ternlog, U32x16};
    /// let (a, b, c) = (U32x16::splat(0b1100), U32x16::splat(0b1010), U32x16::splat(0b1001));
    /// let maj = a.ternlog::<{ ternlog::MAJ3 }>(b, c); // two-of-three majority
    /// assert_eq!(maj.to_array()[0], 0b1000);
    /// ```
    #[inline(always)]
    pub fn ternlog<const IMM: i32>(self, b: Self, c: Self) -> Self {
        const { assert!(IMM >= 0 && IMM <= 255, "ternlog IMM is an 8-bit truth table") }
        let (a, z) = (self, Self::splat(0));
        let mut r = z;
        if IMM & 0x01 != 0 {
            r = r | !a & !b & !c;
        }
        if IMM & 0x02 != 0 {
            r = r | !a & !b & c;
        }
        if IMM & 0x04 != 0 {
            r = r | !a & b & !c;
        }
        if IMM & 0x08 != 0 {
            r = r | !a & b & c;
        }
        if IMM & 0x10 != 0 {
            r = r | a & !b & !c;
        }
        if IMM & 0x20 != 0 {
            r = r | a & !b & c;
        }
        if IMM & 0x40 != 0 {
            r = r | a & b & !c;
        }
        if IMM & 0x80 != 0 {
            r = r | a & b & c;
        }
        r
    }
}
