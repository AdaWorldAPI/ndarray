//! Portable-SIMD polyfill backend — `core::simd::*` adapter (NIGHTLY ONLY).
//!
//! **Draft.** Not wired into `simd.rs` yet. Gated behind the
//! `nightly-simd` feature (which itself requires nightly Rust because
//! `core::simd` is unstable per <https://github.com/rust-lang/rust/issues/86656>).
//!
//! # Why this exists
//!
//! The default polyfill (`simd_avx512.rs` / `simd_avx2.rs` / `simd_neon.rs`)
//! uses architecture intrinsics (`_mm512_*`, `_mm256_*`, `vaddq_*`). Miri
//! cannot execute those — it treats them as opaque foreign-function calls
//! and skips them. That means miri-runs of consumer code (`hpc/byte_scan`,
//! `compose_neo4j`, the bevy plugin's tick path) never exercise the SIMD
//! branches, so UB inside them (out-of-bounds loads, misaligned reads,
//! aliasing violations) goes undetected.
//!
//! `core::simd` is portable — miri executes the semantics directly. With
//! this backend swapped in via `--features nightly-simd`, the same consumer
//! code becomes miri-checkable on every SIMD path. Bugs that the
//! intrinsics-based backend would silently emit still trip miri here.
//!
//! # Tradeoffs
//!
//! - **Nightly only.** `core::simd` is gated on `#![feature(portable_simd)]`.
//!   The default build (stable 1.95) continues to use the intrinsics
//!   backends per the existing `simd.rs` cfg dispatch.
//! - **Codegen quality varies.** `core::simd` lowers to platform SIMD via
//!   LLVM's portable-simd lowering. On AVX-512 with `-C target-cpu=v4`,
//!   the codegen is competitive (LLVM picks `_mm512_*`). On weaker targets
//!   it may produce scalar code where the intrinsics path picks AVX2.
//! - **Mask shapes differ.** AVX-512 uses 64-bit bitmask registers
//!   (`__mmask64`); `core::simd::Mask<i8, 64>` is a vector mask. The
//!   `cmpeq_mask` adapter bridges via `to_bitmask()` so consumer code
//!   sees the same `u64` shape.
//!
//! # Status
//!
//! Coverage in this draft:
//! - `F32x16`, `F64x8` — full method set (splat, FMA, reductions,
//!   comparisons → bitmask, simd_clamp, sqrt/floor/round/abs).
//! - `U8x64` — load/store + reductions + `cmpeq_mask → u64` +
//!   `saturating_add/sub` + `pairwise_avg`.
//! - Operator impls (`Add`, `Sub`, `Mul`, `Div`, `BitAnd`, `BitOr`,
//!   `BitXor`) via macro.
//!
//! NOT covered (will deny-compile if a consumer reaches for them under
//! `--features nightly-simd`):
//! - `permute_bytes`, `shuffle_bytes` — `core::simd::Simd::swizzle` is
//!   `const N` so it can't take a runtime `idx` vector. Needs a scalar
//!   fallback like the AVX-512F-without-VBMI path in `simd_avx512.rs`.
//! - `BF16x16`, `BF16x8`, `F16x16` — `core::simd` has no half-precision
//!   types; needs a manual `bf16_to_f32_batch` adapter.
//! - AMX tile types — out of scope for portable simd entirely.
//! - 32-/64-/128-bit integer SIMD types (`I32x16`, `U32x16`, `I64x8`,
//!   `U64x8`, `U16x32`) — straightforward to add via the same macro
//!   pattern; left for a follow-up commit.
//!
//! # Wiring into simd.rs (sketch — not done in this draft)
//!
//! ```rust,ignore
//! // In src/simd.rs, add a high-priority cfg branch:
//! #[cfg(feature = "nightly-simd")]
//! pub use crate::simd_nightly::{F32x16, F64x8, U8x64};
//!
//! #[cfg(all(not(feature = "nightly-simd"), target_arch = "x86_64",
//!           target_feature = "avx512f"))]
//! pub use crate::simd_avx512::{F32x16, F64x8, U8x64};
//! // ... existing branches
//! ```
//!
//! # Cargo.toml (sketch — not done in this draft)
//!
//! ```toml
//! [features]
//! nightly-simd = []   # requires nightly rustc; enables core::simd backend
//! ```

#![cfg(feature = "nightly-simd")]
#![feature(portable_simd)]

use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::num::{SimdFloat, SimdInt, SimdUint};
use core::simd::{f32x16 as core_f32x16, f64x8 as core_f64x8, u8x64 as core_u8x64};
use core::simd::{Mask, Simd, ToBitMask};

// ════════════════════════════════════════════════════════════════════
// F32x16 — 16-lane single-precision float
// ════════════════════════════════════════════════════════════════════

/// 16-lane `f32` SIMD vector backed by `core::simd::f32x16`.
///
/// API mirrors `simd_avx512::F32x16` so consumer code is identical.
/// Miri can execute every method below — unlike the intrinsics
/// backend, where SIMD paths are opaque to miri.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32x16(pub core_f32x16);

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(core_f32x16::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self(core_f32x16::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16, "F32x16::from_slice needs ≥16 elements");
        Self(core_f32x16::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16, "F32x16::copy_to_slice needs ≥16 elements");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        self.0.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        self.0.reduce_min()
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        self.0.reduce_max()
    }

    // ── Lane-wise min/max/clamp ───────────────────────────────────

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(self.0.simd_min(other.0))
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(self.0.simd_max(other.0))
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    // ── FMA + math ────────────────────────────────────────────────

    /// Fused multiply-add: `self * b + c`. Maps to `_mm512_fmadd_ps` on
    /// AVX-512 builds via LLVM's portable-simd lowering, scalar
    /// `f32::mul_add` per lane otherwise.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    // ── Bit reinterpretation ──────────────────────────────────────

    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        U32x16(self.0.to_bits())
    }

    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        Self(core_f32x16::from_bits(bits.0))
    }

    // ── Comparisons → mask ────────────────────────────────────────
    //
    // The intrinsics backend exposes `simd_eq` / `simd_lt` / etc.
    // returning `F32Mask16(__mmask16)` (a 16-bit bitmask). We mirror
    // that by wrapping `core::simd::Mask<i32, 16>` and providing
    // `to_bitmask()` for consumers that want the u16 shape.

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_eq(other.0))
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_ne(other.0))
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_lt(other.0))
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_le(other.0))
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_gt(other.0))
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask16 {
        F32Mask16(self.0.simd_ge(other.0))
    }
}

/// 16-lane mask for `F32x16` comparisons.
#[derive(Copy, Clone, Debug)]
pub struct F32Mask16(pub Mask<i32, 16>);

impl F32Mask16 {
    /// Convert to a 16-bit packed bitmask (matches the AVX-512 `__mmask16`
    /// shape). Bit i set iff lane i of the mask is true.
    #[inline(always)]
    pub fn to_bitmask(self) -> u16 {
        self.0.to_bitmask() as u16
    }

    /// Per-lane select: returns `true_val[i]` where mask[i] is set,
    /// else `false_val[i]`.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        F32x16(self.0.select(true_val.0, false_val.0))
    }
}

// ════════════════════════════════════════════════════════════════════
// F64x8 — 8-lane double-precision float
// ════════════════════════════════════════════════════════════════════

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64x8(pub core_f64x8);

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(core_f64x8::splat(v))
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self(core_f64x8::from_array(arr))
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8, "F64x8::from_slice needs ≥8 elements");
        Self(core_f64x8::from_slice(s))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8, "F64x8::copy_to_slice needs ≥8 elements");
        self.0.copy_to_slice(s);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        self.0.reduce_sum()
    }
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        self.0.reduce_min()
    }
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
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
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.simd_clamp(lo.0, hi.0))
    }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

// ════════════════════════════════════════════════════════════════════
// U8x64 — 64-lane unsigned-byte (the rasterizer / palette / NBT width)
// ════════════════════════════════════════════════════════════════════

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U8x64(pub core_u8x64);

impl U8x64 {
    pub const LANES: usize = 64;

    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(core_u8x64::splat(v))
    }
    #[inline(always)]
    pub fn from_array(arr: [u8; 64]) -> Self {
        Self(core_u8x64::from_array(arr))
    }
    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 64, "U8x64::from_slice needs ≥64 bytes");
        Self(core_u8x64::from_slice(s))
    }
    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        self.0.to_array()
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 64, "U8x64::copy_to_slice needs ≥64 bytes");
        self.0.copy_to_slice(s);
    }

    // ── Reductions ────────────────────────────────────────────────

    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        self.0.reduce_sum()
    }
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        self.0.reduce_min()
    }
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
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

    // ── Saturating arithmetic ─────────────────────────────────────

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Per-lane unsigned rounded average: `(a + b + 1) >> 1`.
    /// `core::simd` has no native `avg_epu8` equivalent, so we compute
    /// it from the standard identity. LLVM may still lower to
    /// `_mm512_avg_epu8` on AVX-512 builds; verify with --emit asm.
    #[inline(always)]
    pub fn pairwise_avg(self, other: Self) -> Self {
        // Promote to u16 to avoid overflow on `a + b + 1`.
        let a16 = self.0.cast::<u16>();
        let b16 = other.0.cast::<u16>();
        let avg = (a16 + b16 + core::simd::Simd::splat(1)) >> core::simd::Simd::splat(1);
        Self(avg.cast::<u8>())
    }

    // ── Comparison → 64-bit bitmask ───────────────────────────────

    /// Per-lane equality. Returns a 64-bit mask (bit i = 1 iff
    /// `self[i] == other[i]`). Matches the AVX-512 `__mmask64` shape
    /// of `simd_avx512::U8x64::cmpeq_mask`.
    #[inline(always)]
    pub fn cmpeq_mask(self, other: Self) -> u64 {
        self.0.simd_eq(other.0).to_bitmask()
    }

    /// Per-lane unsigned greater-than. Returns a 64-bit mask.
    #[inline(always)]
    pub fn cmpgt_mask(self, other: Self) -> u64 {
        self.0.simd_gt(other.0).to_bitmask()
    }

    /// Extract MSB of each byte as a 64-bit mask. Equivalent to
    /// `_mm512_movepi8_mask`.
    #[inline(always)]
    pub fn movemask(self) -> u64 {
        // MSB-set ⇔ value ≥ 128 ⇔ value > 127 in unsigned cmp.
        self.0.simd_gt(core_u8x64::splat(0x7F)).to_bitmask()
    }
}

// ════════════════════════════════════════════════════════════════════
// U32x16 — companion type for F32x16::to_bits / from_bits
// ════════════════════════════════════════════════════════════════════

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct U32x16(pub Simd<u32, 16>);

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(Simd::splat(v))
    }
    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(Simd::from_array(arr))
    }
    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        self.0.to_array()
    }
}

// ════════════════════════════════════════════════════════════════════
// Operator impls — `Add`, `Sub`, `Mul`, `Div` for floats; bitwise for
// ints. Macro keeps the boilerplate from spreading.
// ════════════════════════════════════════════════════════════════════

macro_rules! impl_fp_ops {
    ($name:ident) => {
        impl core::ops::Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
        impl core::ops::Mul for $name {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }
        impl core::ops::Div for $name {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }
        impl core::ops::Neg for $name {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
    };
}

macro_rules! impl_int_ops {
    ($name:ident) => {
        impl core::ops::Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
        impl core::ops::BitAnd for $name {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
        impl core::ops::BitOr for $name {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl core::ops::BitXor for $name {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }
    };
}

impl_fp_ops!(F32x16);
impl_fp_ops!(F64x8);
impl_int_ops!(U8x64);
impl_int_ops!(U32x16);

// ════════════════════════════════════════════════════════════════════
// Default impls — `Self::splat(0)` shape.
// ════════════════════════════════════════════════════════════════════

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0.0)
    }
}
impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0.0)
    }
}
impl Default for U8x64 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}
impl Default for U32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests — small parity sanity for the methods consumers rely on.
// Real test surface (miri-runnable) lives in `tests/` once wired.
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_fma_lane_exact() {
        let a = F32x16::splat(0.5);
        let b = F32x16::splat(2.0);
        let c = F32x16::splat(1.0);
        // 0.5 * 2.0 + 1.0 = 2.0 per lane
        assert!(a.mul_add(b, c).to_array().iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn f32x16_mask_bitmask_roundtrip() {
        let v = F32x16::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let threshold = F32x16::splat(8.5);
        let m = v.simd_lt(threshold);
        // First 8 lanes < 8.5, rest aren't.
        assert_eq!(m.to_bitmask(), 0b0000_0000_1111_1111);
    }

    #[test]
    fn u8x64_cmpeq_mask_matches_scalar() {
        let a: [u8; 64] = core::array::from_fn(|i| (i & 7) as u8);
        let b: [u8; 64] = core::array::from_fn(|i| (i & 6) as u8);
        let m = U8x64::from_array(a).cmpeq_mask(U8x64::from_array(b));
        for i in 0..64 {
            assert_eq!(((m >> i) & 1) == 1, a[i] == b[i], "lane {i}");
        }
    }

    #[test]
    fn u8x64_saturating_add_clamps_to_255() {
        let a = U8x64::splat(200);
        let b = U8x64::splat(100);
        assert_eq!(a.saturating_add(b).to_array(), [255u8; 64]);
    }

    #[test]
    fn u8x64_pairwise_avg_rounds_up() {
        let a = U8x64::splat(7);
        let b = U8x64::splat(8);
        // (7 + 8 + 1) / 2 = 8
        assert_eq!(a.pairwise_avg(b).to_array(), [8u8; 64]);
    }

    #[test]
    fn integrate_simd_shape_via_polyfill() {
        // The hot loop in hpc::renderer::integrate_simd looks like:
        //   for chunk in positions.chunks_exact_mut(16) {
        //       let p = F32x16::from_slice(chunk);
        //       let v = F32x16::from_slice(velocity_chunk);
        //       let dt_v = F32x16::splat(dt);
        //       let p_new = v.mul_add(dt_v, p);
        //       p_new.copy_to_slice(chunk);
        //   }
        // Verify the same shape works through the nightly backend.
        let mut p = [0.0f32; 16];
        let v = F32x16::splat(1.0);
        let dt = F32x16::splat(1.0 / 60.0);
        let p0 = F32x16::from_slice(&p);
        v.mul_add(dt, p0).copy_to_slice(&mut p);
        for x in p {
            assert!((x - 1.0 / 60.0).abs() < 1e-6);
        }
    }
}
