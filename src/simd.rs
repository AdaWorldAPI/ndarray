//! SIMD polyfill — `crate::simd::F32x16` dispatches via LazyLock<Tier>.
//!
//! Same pattern as `backend/native.rs`: detect once, dispatch forever.
//! AVX-512 → AVX2 → Scalar. Consumer writes `crate::simd::F32x16`. Period.
//!
//! When `std::simd` stabilizes: swap this file. Zero consumer changes.

#[cfg(feature = "std")]
use std::sync::LazyLock;

// On i686 / wasm32 / etc. only the `Scalar` variant is constructed —
// `detect_tier()`'s feature-detection blocks are `target_arch = "x86_64"`
// or `"aarch64"` gated, both false on i686. Without `dead_code` allowance
// the `-D warnings` build fails with `variants ... are never constructed`.
// Note: this `Tier` enum is *runtime* dispatch only. On `wasm32 +
// target_feature = "simd128"` the SIMD *types* are NOT scalar — they come
// from the compile-time `simd_wasm::wasm32_simd` v128 backend (re-exported
// below); `detect_tier()` simply has no wasm arm, so the runtime tier stays
// `Scalar`.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
enum Tier {
    Avx512 = 1,
    Avx2 = 2,
    /// ARM NEON 128-bit + dotprod (Pi 5 / A76+). 4× int8 throughput.
    NeonDotProd = 3,
    /// ARM NEON 128-bit baseline (Pi 3/4 / A53/A72). Pure float SIMD.
    Neon = 4,
    Scalar = 5,
}

impl Tier {
    /// Inverse of `as u8` — used by the no_std `critical_section`
    /// polyfill below so we can stash a `Tier` into an `AtomicU8`.
    #[allow(dead_code)]
    #[inline(always)]
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Tier::Avx512,
            2 => Tier::Avx2,
            3 => Tier::NeonDotProd,
            4 => Tier::Neon,
            _ => Tier::Scalar,
        }
    }
}

/// Detect the best SIMD tier the current CPU supports.
///
/// Pulled out of the original `LazyLock::new` closure so it can be
/// reused by both the `std` and `no_std` cache implementations below.
#[allow(dead_code)]
fn detect_tier() -> Tier {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Tier::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return Tier::Avx2;
        }
    }
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    {
        // NEON is mandatory on aarch64 — always available.
        // dotprod (ARMv8.2+) distinguishes Pi 5 from Pi 3/4.
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Tier::NeonDotProd;
        }
        return Tier::Neon;
    }
    #[cfg(all(not(feature = "std"), target_arch = "aarch64"))]
    {
        // No runtime feature detection available without std — fall back
        // to whatever the compile-time target features advertise.
        #[cfg(target_feature = "dotprod")]
        return Tier::NeonDotProd;
        #[cfg(not(target_feature = "dotprod"))]
        return Tier::Neon;
    }
    #[cfg(all(not(feature = "std"), target_arch = "x86_64"))]
    {
        // No `is_x86_feature_detected!` without std — pick the highest
        // tier whose features were enabled at compile time.
        #[cfg(target_feature = "avx512f")]
        return Tier::Avx512;
        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
        return Tier::Avx2;
    }
    #[allow(unreachable_code)]
    Tier::Scalar
}

// ── std path: original `LazyLock`-backed cache ───────────────────────
#[cfg(feature = "std")]
static TIER: LazyLock<Tier> = LazyLock::new(detect_tier);

#[cfg(feature = "std")]
#[inline(always)]
#[allow(dead_code)]
fn tier() -> Tier {
    *TIER
}

// ── no_std path: portable-atomic + critical-section polyfill ────────
#[cfg(all(not(feature = "std"), feature = "portable-atomic-critical-section"))]
use portable_atomic::{AtomicU8, Ordering};

#[cfg(all(not(feature = "std"), feature = "portable-atomic-critical-section"))]
static TIER_INIT: AtomicU8 = AtomicU8::new(0);

#[cfg(all(not(feature = "std"), feature = "portable-atomic-critical-section"))]
#[inline]
#[allow(dead_code)]
fn tier() -> Tier {
    let cached = TIER_INIT.load(Ordering::Relaxed);
    if cached != 0 {
        return Tier::from_u8(cached);
    }
    critical_section::with(|_| {
        let detected = detect_tier();
        TIER_INIT.store(detected as u8, Ordering::Relaxed);
        detected
    })
}

// ── no_std path with no polyfill: compile-time fallback ──────────────
#[cfg(all(not(feature = "std"), not(feature = "portable-atomic-critical-section")))]
#[inline(always)]
#[allow(dead_code)]
fn tier() -> Tier {
    detect_tier()
}

// BF16 tier detection happens inline in bf16_to_f32_batch() via
// is_x86_feature_detected!("avx512bf16") — no LazyLock needed.
// The check is cheap (reads a cached cpuid result) and the batch
// function uses as_chunks::<16>() + as_chunks::<8>() for SIMD widths.

// ============================================================================
// Preferred SIMD lane widths — compile-time constants for array_windows
// ============================================================================
//
// Consumer code uses these to select array_windows size at compile time:
//
//   for window in data.array_windows::<{crate::simd::PREFERRED_F64_LANES}>() {
//       let v = F64x8::from_array(*window);   // AVX-512: native 8-wide
//       // or
//       let v = F64x4::from_array(*window);   // AVX2: native 4-wide
//   }
//
// generic_const_exprs is nightly, so consumers must #[cfg] branch on window size.
// These constants document the preferred width per tier.

/// Preferred f64 SIMD width (elements per register).
/// AVX-512: 8 lanes (__m512d). AVX2: 4 lanes (__m256d). NEON: 2 lanes (float64x2_t).
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_F64_LANES: usize = 8;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_F64_LANES: usize = 4;
#[cfg(target_arch = "aarch64")]
pub const PREFERRED_F64_LANES: usize = 2; // NEON: float64x2_t = 2 × f64
#[cfg(target_arch = "wasm32")]
pub const PREFERRED_F64_LANES: usize = 2; // WASM SIMD128: f64x2 = 2 × f64
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub const PREFERRED_F64_LANES: usize = 4; // scalar fallback: same as AVX2 shape

/// Preferred f32 SIMD width.
/// AVX-512: 16 lanes (__m512). AVX2: 8 lanes (__m256). NEON: 4 lanes (float32x4_t).
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_F32_LANES: usize = 16;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_F32_LANES: usize = 8;
#[cfg(target_arch = "aarch64")]
pub const PREFERRED_F32_LANES: usize = 4; // NEON: float32x4_t = 4 × f32
#[cfg(target_arch = "wasm32")]
pub const PREFERRED_F32_LANES: usize = 4; // WASM SIMD128: f32x4 = 4 × f32
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub const PREFERRED_F32_LANES: usize = 8;

/// Preferred u64 SIMD width.
/// AVX-512: 8 lanes. AVX2: 4 lanes. NEON: 2 lanes (uint64x2_t).
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_U64_LANES: usize = 8;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_U64_LANES: usize = 4;
#[cfg(target_arch = "aarch64")]
pub const PREFERRED_U64_LANES: usize = 2; // NEON: uint64x2_t
#[cfg(target_arch = "wasm32")]
pub const PREFERRED_U64_LANES: usize = 2; // WASM SIMD128: i64x2 = 2 × u64
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub const PREFERRED_U64_LANES: usize = 4;

/// Preferred i16 SIMD width (for Base17 L1 on i16[17]).
/// AVX-512: 32 lanes (__m512i via epi16). AVX2: 16 lanes (__m256i).
/// NEON: 8 lanes (int16x8_t). Base17 has 17 dims — NEON needs 3 loads
/// (8+8+1), A72 dual pipeline hides latency on the third.
#[cfg(target_feature = "avx512f")]
pub const PREFERRED_I16_LANES: usize = 32;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub const PREFERRED_I16_LANES: usize = 16;
#[cfg(target_arch = "aarch64")]
pub const PREFERRED_I16_LANES: usize = 8; // NEON: int16x8_t
#[cfg(target_arch = "wasm32")]
pub const PREFERRED_I16_LANES: usize = 8; // WASM SIMD128: i16x8 = 8 × i16
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub const PREFERRED_I16_LANES: usize = 16;

// ============================================================================
// x86_64: re-export based on tier
// ============================================================================

// Compile-time SIMD dispatch via target_feature. The cargo config
// chosen at build (.cargo/config.toml = v3 default / config-avx512.toml
// = v4 / config-native.toml = native) sets the `target_feature` flags
// that select exactly one arm below.
//   * v3 / GitHub-CI default → `target_feature = "avx2"` only →
//     simd_avx2 backend (F32x16 = two-half (f32x8, f32x8), int wrappers
//     are scalar polyfills via the `avx2_int_type!` macro).
//   * v4 (or native on AVX-512 host) → `target_feature = "avx512f"` →
//     simd_avx512 backend with native __m512 / __m512d / __m512i.
//   * aarch64 → simd_neon backend.
//   * everything else (wasm32, riscv, etc.) → scalar fallback.

// Nightly-simd dispatch — when `feature = "nightly-simd"` is on, the
// `crate::simd_nightly` portable backend (wrapping `core::simd::*`)
// REPLACES the intrinsics arms below. This is a compile-time-dispatch
// choice: opt in via `cargo +nightly --features nightly-simd ...` and
// the same `use crate::simd::F32x16` call sites become miri-runnable.
// No target_arch constraint — `core::simd` is portable, so this arm
// is the one true backend on wasm32 / riscv / aarch64 / x86_64 alike
// as soon as `nightly-simd` is on.
#[cfg(feature = "nightly-simd")]
pub use crate::simd_nightly::{
    f32x16, f32x8, f64x4, f64x8, i16x16, i16x32, i32x16, i32x8, i64x4, i64x8, i8x32, i8x64, u16x16, u16x32, u32x16,
    u32x8, u64x4, u64x8, u8x32, u8x64, BF16x16, BF16x8, F16x16, F32Mask16, F32Mask8, F32x16, F32x8, F64Mask4, F64Mask8,
    F64x4, F64x8, I16x16, I16x32, I32x16, I32x8, I64x4, I64x8, I8x32, I8x64, U16x16, U16x32, U32x16, U32x8, U64x4,
    U64x8, U8x32, U8x64,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
pub use crate::simd_avx512::{
    batch_packed_i4_16,
    f32x16,
    f32x8,
    f64x4,
    f64x8,
    i16x16,
    i16x32,
    i32x16,
    i32x8,
    i64x4,
    i64x8,
    i8x16,
    i8x32,
    i8x64,
    palette_lookup_u8x8,
    prefetch_read_t0,
    prefetch_read_t1,
    prefetch_read_t2,
    u16x16,
    u16x8,
    u32x16,
    u32x8,
    u64x4,
    u64x8,
    u8x64,
    u8x8,
    F32Mask16,
    // 512-bit (native AVX-512, __m512/__m512d/__m512i)
    F32x16,
    // 256-bit (AVX2 baseline, __m256/__m256d/__m256i)
    F32x8,
    F64Mask8,
    F64x4,
    F64x8,
    I16x16,
    I16x32,
    I32x16,
    // 256-bit int polyfills surfaced 2026-05-20 (re-exported from
    // `simd_avx2` via `simd_avx512`'s re-export at line ~2260).
    I32x8,
    I64x4,
    I64x8,
    I8x16,
    I8x32,
    I8x64,
    U16x16,
    U16x32,
    U16x8,
    U32x16,
    U32x8,
    U64x4,
    U64x8,
    U8x64,
    U8x8,
};

// BF16 types + batch conversion (always available — scalar fallback built in)
#[cfg(target_arch = "x86_64")]
pub use crate::simd_avx512::{bf16_to_f32_batch, bf16_to_f32_scalar, f32_to_bf16_batch, f32_to_bf16_scalar};

// BF16 RNE (round-to-nearest-even) path — pure AVX-512-F, byte-exact vs
// hardware `_mm512_cvtneps_pbh` on Sapphire Rapids+ (verified on 1M inputs
// in ndarray::simd_avx512::tests). Consumer code should call
// `f32_to_bf16_batch_rne` in hot loops (500-20000× faster than the scalar
// path via AMX / AVX-512 tiles); `f32_to_bf16_scalar_rne` is exposed only
// as a unit-test reference implementation and MUST NOT be called in hot
// loops per the workspace-wide "never scalar ever" rule for F32→BF16.
// See lance-graph/CLAUDE.md § Certification Process.
#[cfg(target_arch = "x86_64")]
pub use crate::simd_avx512::{f32_to_bf16_batch_rne, f32_to_bf16_scalar_rne};
// BF16 SIMD types only available when avx512bf16 is enabled at compile time
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16", not(feature = "nightly-simd")))]
pub use crate::simd_avx512::{BF16x16, BF16x8};

// AVX2 baseline arm — selected by the `x86-64-v3` cargo default. The
// predicate is `not(avx512f)` rather than `avx2 + not(avx512f)`: the
// inner intrinsics in `simd_avx2.rs` use per-function `#[target_feature
// (enable = "avx,avx2,fma")]` annotations, so the OPERATIONS gate
// themselves at the symbol level even when the consumer build target
// is x86-64 baseline. The struct-field types (`__m256` / `__m256i`)
// are core::arch declarations and don't require AVX/AVX2 at the type
// level — only execution does. Keeps GitHub CI green (it runs with
// `RUSTFLAGS="-D warnings"` env, which overrides our v3 config.toml,
// landing on x86-64 baseline → the previous tighter `avx2` predicate
// left no matching arm).
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(feature = "nightly-simd")
))]
pub use crate::simd_avx512::{
    batch_packed_i4_16, f32x8, f64x4, i16x16, i8x16, i8x32, palette_lookup_u8x8, prefetch_read_t0, prefetch_read_t1,
    prefetch_read_t2, u16x8, u8x8, F32x8, F64x4, I16x16, I8x16, I8x32, U16x8, U8x8,
};

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(feature = "nightly-simd")
))]
pub use crate::simd_avx2::{
    f32x16, f64x8, i16x32, i32x16, i32x8, i64x4, i64x8, i8x64, u16x16, u32x16, u32x8, u64x4, u64x8, u8x64, F32Mask16,
    F32x16, F64Mask8, F64x8, I16x32, I32x16, I32x8, I64x4, I64x8, I8x64, U16x16, U16x32, U32x16, U32x8, U64x4, U64x8,
    U8x64,
};

// U8x32 — native AVX2 byte width (one __m256i = 32 bytes). Available on
// both AVX-512 and AVX2 builds: it's the natural width for byte-level
// AVX2 ops, and on AVX-512 builds it's the half-register companion to
// U8x64. Lives in simd_avx2.rs (single source of truth) and is re-exported
// from both tier branches.
#[cfg(all(target_arch = "x86_64", not(feature = "nightly-simd")))]
pub use crate::simd_avx2::{u8x32, U8x32};

// ============================================================================
// Non-x86: scalar fallback types with identical API
// ============================================================================

// Scalar backend lives in its own file (`src/simd_scalar.rs`), declared
// here with `#[path]` so the internal module name stays `scalar` and
// the existing `pub use scalar::{...}` re-exports below don't need to
// change. Extracted from this file in Phase 4 of the integration plan
// (1271 LoC of macro expansions out of the dispatcher).
#[cfg(all(not(target_arch = "x86_64"), not(feature = "nightly-simd")))]
#[path = "simd_scalar.rs"]
pub(crate) mod scalar;

// aarch64: F32x16/F64x8 come from the real NEON paired-load implementation
// in simd_neon::aarch64_simd (verified 2026-04-30, agent A7 — burn parity item 9).
// Integer + 256-bit float types still come from the scalar fallback; they're
// not on the critical path for f32 BLAS-1 / VML kernels.
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use crate::simd_neon::aarch64_simd::{f32x16, f64x8, F32Mask16, F32x16, F64Mask8, F64x8};
// W1a NEON-native types + free functions
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use crate::simd_neon::{
    batch_packed_i4_16, i8x16, i8x32, palette_lookup_u8x8, prefetch_read_t0, prefetch_read_t1, prefetch_read_t2, u8x8,
    I8x16, I8x32, U8x8,
};
// U16x8 on aarch64 comes from simd_neon (backed by uint16x8_t)
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use crate::simd_neon::{u16x8, U16x8};
// U32x16 (native `[U32x4; 4]`) — the ARX lane the ChaCha20 backend rides. Comes
// from simd_neon, not the scalar fallback, so it carries Add/BitXor/rotate_left.
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use crate::simd_neon::{u32x16, U32x16};
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use scalar::{
    f32x8, f64x4, i32x16, i32x8, i64x4, i64x8, u16x16, u32x8, u64x4, u64x8, u8x64, F32x8, F64x4, I32x16, I32x8, I64x4,
    I64x8, U16x16, U16x32, U32x8, U64x4, U64x8, U8x64,
};

// wasm32 + simd128: the native v128 float hot path (F32x16 / F64x8 + masks)
// and native I8x16 come from `simd_wasm::wasm32_simd`; the long-tail integer
// and 256-bit-shaped types come from the scalar fallback. Same split
// `simd_neon` uses on aarch64 (native float kernels, scalar for the rest).
// The `wasm32_simd` module only exists under `target_feature = "simd128"`,
// so this arm is gated identically.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "nightly-simd")))]
pub use crate::simd_wasm::wasm32_simd::{
    f32x16, f64x8, i8x16, u32x16, F32Mask16, F32x16, F64Mask8, F64x8, I8x16, U32x16,
};
// `u32x16`/`U32x16` now come from the native `wasm32_simd` arm above (the ARX
// lane the ChaCha20 backend rides), so they are dropped from this scalar list.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "nightly-simd")))]
pub use scalar::{
    batch_packed_i4_16, f32x8, f64x4, i16x16, i16x32, i32x16, i32x8, i64x4, i64x8, i8x32, i8x64, palette_lookup_u8x8,
    prefetch_read_t0, prefetch_read_t1, prefetch_read_t2, u16x16, u16x8, u32x8, u64x4, u64x8, u8x64, u8x8, F32x8,
    F64x4, I16x16, I16x32, I32x16, I32x8, I64x4, I64x8, I8x32, I8x64, U16x16, U16x32, U16x8, U32x8, U64x4, U64x8,
    U8x64, U8x8,
};

// Other non-x86 targets — wasm32 without simd128, riscv, etc.: full scalar
// fallback. Excludes the wasm32+simd128 case handled by the native arm above.
#[cfg(all(
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64"),
    not(all(target_arch = "wasm32", target_feature = "simd128")),
    not(feature = "nightly-simd")
))]
pub use scalar::{
    batch_packed_i4_16, f32x16, f32x8, f64x4, f64x8, i16x16, i16x32, i32x16, i32x8, i64x4, i64x8, i8x16, i8x32, i8x64,
    palette_lookup_u8x8, prefetch_read_t0, prefetch_read_t1, prefetch_read_t2, u16x16, u16x8, u32x16, u32x8, u64x4,
    u64x8, u8x64, u8x8, F32Mask16, F32x16, F32x8, F64Mask8, F64x4, F64x8, I16x16, I16x32, I32x16, I32x8, I64x4, I64x8,
    I8x16, I8x32, I8x64, U16x16, U16x32, U16x8, U32x16, U32x8, U64x4, U64x8, U8x64, U8x8,
};

// Scalar BF16 conversion — always available on all platforms
#[cfg(not(target_arch = "x86_64"))]
pub fn bf16_to_f32_scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_bf16_scalar(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
#[cfg(not(target_arch = "x86_64"))]
pub fn bf16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    for (i, &b) in input.iter().enumerate() {
        if i < output.len() {
            output[i] = bf16_to_f32_scalar(b);
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) {
    for (i, &v) in input.iter().enumerate() {
        if i < output.len() {
            output[i] = f32_to_bf16_scalar(v);
        }
    }
}

// ============================================================================
// SIMD math functions — ndarray additions (not in std::simd)
// ============================================================================

/// Fast exp(x) for F32x16 — Remez polynomial on [-87, 87].
///
/// Max error ~2 ULP in [-10, 10]. Uses the standard range-reduction
/// approach: exp(x) = 2^n * exp(r) where r = x - n*ln(2).
///
/// Domain: clamps input to [-87.336, 88.722] before reduction so that the
/// integer exponent `n` stays within the IEEE 754 f32 representable range.
/// Beyond the upper bound we'd hit `i32` overflow in `pow2n_from_int` and
/// silently return ~0.5 instead of +Inf (release) or panic (debug).
///
/// NaN handling: `simd_clamp` is `max(lo).min(hi)`, and `_mm512_max_ps` /
/// `_mm512_min_ps` return the SECOND operand when the first is NaN (per
/// Intel SDM § MAXPS/MINPS). That would silently clamp NaN inputs to `lo`
/// (-87.336) producing `exp(-87.336) ≈ 1.4e-38` — a finite tiny value
/// masquerading as valid output. Caught by codex review on PR #142.
///
/// Fix: capture NaN lanes via `x.simd_ne(x)` (NaN ≠ itself per IEEE 754)
/// before the clamp, then mask-select NaN back into those lanes after
/// the polynomial. NaN lanes propagate as NaN; finite lanes are unchanged.
#[inline(always)]
#[allow(dead_code)]
pub fn simd_exp_f32(x: F32x16) -> F32x16 {
    let ln2 = F32x16::splat(core::f32::consts::LN_2);
    let inv_ln2 = F32x16::splat(1.0 / core::f32::consts::LN_2);
    let one = F32x16::splat(1.0);

    // NaN-preservation mask: bit set wherever x is NaN. IEEE 754: NaN ≠ NaN.
    // Captured BEFORE the clamp because simd_clamp destroys NaN lanes.
    let nan_mask = x.simd_ne(x);

    // Pre-clamp to the safe domain. Outside this band exp() is non-representable
    // anyway (overflow → +Inf at ~88.7, underflow → +0 at ~-87.3) so the clamp
    // is observable only at the saturation boundary.
    let x = x.simd_clamp(F32x16::splat(-87.336_f32), F32x16::splat(88.722_f32));

    // Range reduction: n = round(x / ln2), r = x - n * ln2
    let n = (x * inv_ln2).round();
    let r = x - n * ln2;

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    let c2 = F32x16::splat(0.5);
    let c3 = F32x16::splat(1.0 / 6.0);
    let c4 = F32x16::splat(1.0 / 24.0);
    let c5 = F32x16::splat(1.0 / 120.0);

    let poly = one + r * (one + r * (c2 + r * (c3 + r * (c4 + r * c5))));

    // Reconstruct: exp(x) = 2^n * poly
    let result = poly * pow2n_from_int(n);

    // Restore NaN in lanes where the input was NaN (clamp had destroyed them).
    nan_mask.select(F32x16::splat(f32::NAN), result)
}

/// Compute 2^n where n is an integer stored as f32.
///
/// Uses the IEEE 754 trick: set the exponent field directly.
///
/// The `ni` is clamped to [-126, 127] before adding the 127 bias so that
/// `(ni + 127) as u32` stays in [1, 254] (valid normal-number exponent
/// field). Without this clamp, an `Inf` input from `simd_exp_f32` would
/// saturate to `i32::MAX`, then `+ 127` would panic in debug or wrap in
/// release, producing a garbage IEEE bit pattern (was: silent ~0.5 result).
/// Caller `simd_exp_f32` already pre-clamps the domain so this is defense
/// in depth.
#[inline(always)]
#[allow(dead_code)]
fn pow2n_from_int(n: F32x16) -> F32x16 {
    let arr = n.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        let ni = (arr[i] as i32).clamp(-126, 127);
        let bits = ((ni + 127) as u32) << 23;
        out[i] = f32::from_bits(bits);
    }
    F32x16::from_array(out)
}

/// Fast natural log for F32x16.
#[inline(always)]
#[allow(dead_code)]
pub fn simd_ln_f32(x: F32x16) -> F32x16 {
    let arr = x.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = arr[i].ln();
    }
    F32x16::from_array(out)
}

// ============================================================================
// Cognitive shader foundation re-exports
// ============================================================================

// HPC re-exports — only available when the hpc module is compiled.
// Without `hpc-extras`, consumers still get the SIMD polyfill types above
// (F32x16, I8x32, etc.) but NOT the domain-specific functions below.

pub use crate::hpc::bitwise::{hamming_distance_raw, popcount_raw};
pub use crate::hpc::bnn_cross_plane::CollapseGate;
pub use crate::hpc::fft::{wht_f32, wht_f32_new};
pub use crate::hpc::fingerprint::{
    vector_config, Fingerprint, Fingerprint1K, Fingerprint2K, Fingerprint64K, VectorConfig, VectorWidth,
};

// PR-X1 — SoA carrier + const-size slice helpers, dispatched from their
// respective `simd_{type}.rs` modules. The W1a consumer contract forbids
// reaching past `crate::simd::*` into the implementation modules directly.
//
// `array_chunks` (non-overlapping) and `array_windows` (overlapping) are
// the stable-Rust foundation primitives for SIMD-staged kernels — together
// with `add_mul_f32` / `add_mul_f64` below, they reach within a few %
// of a Cranelift-JIT'd inner loop on the BLAS-graph GEMM path and are
// the reason the JIT-native option was deemed unnecessary. See the
// "Foundation primitives — do not remove" notice in `src/simd_ops.rs`.
pub use crate::simd_ops::{array_chunks, array_chunks_checked, array_windows, array_windows_checked};
// Crate-native tiled f64 GEMM (`C := α·A·B + β·C`) with a bit-exactness
// contract: unfused mul+add in ascending-k order per element → bit-identical
// on every backend (AVX-512/AVX2/NEON/WASM/scalar) and, at α=1 β=0,
// bit-identical to the naive triple-loop reference. This is the in-crate
// ground-truth GEMM for probes/certification AND the engine behind
// `backend::native::gemm_f64` (own Rust in the f64 BLAS path; the f32
// sibling still delegates to the external `matrixmultiply` crate).
// `gemm_f64_tiled_fma` is the fast fused tier (same tiling/order, one
// rounding per step) for consumers on FMA-pinned targets — not the
// backend engine, because its scalar polyfill can lower to libm `fma()`
// on baseline builds. Both kernels are alloc-free, but `pub mod
// simd`/`simd_ops` are std-gated in lib.rs, so they are reachable only
// in `std` builds today.
pub use crate::simd_ops::{gemm_f64_tiled, gemm_f64_tiled_fma};
pub use crate::simd_soa::MultiLaneColumn;

pub use crate::hpc::quantized::{
    dequantize_i2_to_f32, dequantize_i4_to_f32, dequantize_i8_to_f32, quantize_f32_to_i2, quantize_f32_to_i4,
    quantize_f32_to_i8, QuantParams,
};

// Half-precision SIMD vectors (BF16x16, F16x16) — portable scalar impl, always
// available. Note: when `target_feature = "avx512bf16"` is active a separate
// hardware-native `BF16x16` is also exported above from `simd_avx512`; in that
// case we only re-export F16x16 + slice ops to avoid name collisions.
//
// On all other targets (including avx512f-without-bf16, NEON, scalar) the
// portable `simd_half::BF16x16` is the canonical 16-lane BF16 vector.

// Always re-export F16x16 + all slice-level ops (no naming conflict).
#[cfg(feature = "std")]
pub use crate::simd_half::{
    add_bf16_inplace, add_f16_inplace, cast_bf16_to_f32_batch, cast_f16_to_f32_batch, cast_f32_to_bf16_batch,
    cast_f32_to_f16_batch, mul_bf16_inplace, mul_f16_inplace, F16x16,
};

// Re-export portable BF16x16 only when the hardware-native avx512bf16 variant
// is NOT active (otherwise `simd_avx512::BF16x16` already occupies the name).
#[cfg(all(feature = "std", not(all(target_arch = "x86_64", target_feature = "avx512bf16"))))]
pub use crate::simd_half::BF16x16;

// K-means + L2 distance

pub use crate::hpc::cam_pq::{kmeans, squared_l2};

// SIMD cosine

pub use crate::hpc::heel_f64x8::cosine_f32_to_f64_simd;

// Dispatched integer matmul — the polyfill entry for batched int8 scoring.
// `matmul_i8_to_i32` runtime-selects AMX `TDPBUSD` tiles (byte-asm, 16384
// MAC/instr, Sapphire Rapids+) → AVX-512 VPDPBUSD → AVX-VNNI → scalar, and
// is bit-identical across tiers. Surfaced here so a consumer reaches the
// whole AMX ladder through the canonical `ndarray::simd::*` import (W1a)
// without dipping into `crate::hpc::amx_matmul` directly. `amx_available()`
// exposes the runtime tier check for reporting.
// AMX is x86_64-only (the `amx_matmul` / `simd_amx` modules are
// `#[cfg(target_arch = "x86_64")]`), so these re-exports are arch-gated.
// Off x86 the cross-platform entry points are `backend::gemm_i8` /
// `backend::gemm_bf16` (portable scalar / NEON / wasm-SIMD paths).
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub use crate::hpc::amx_matmul::{amx_available, matmul_i8_to_i32};
// Runtime-dispatch trampolines (`simd_runtime`, feature = "runtime-dispatch")
// surfaced through the canonical `ndarray::simd::*` namespace — the W1a
// consumer invariant is "all SIMD from `ndarray::simd`", so consumers that
// were importing `ndarray::simd_runtime::{matmul_*, gemm_u8_i8, ...}`
// directly (e.g. the tesseract-rs recognizer's int8 GEMM) can now stay on
// the one polyfill import path. These are thin `#[inline(always)]` aliases
// of the SAME underlying tier ladders (`hpc::amx_matmul`, `simd_int_ops`),
// so switching import paths is bit-identical by construction.
#[cfg(feature = "runtime-dispatch")]
pub use crate::simd_runtime::{gemm_u8_i8, matmul_bf16_to_f32, matmul_f32, vnni_dot_u8_i8};
// `matmul_i8_to_i32` already has its canonical `simd::` name on
// x86_64 + std (the `hpc::amx_matmul` re-export above — the identical
// function `simd_runtime::matmul_i8_to_i32` wraps). Off x86_64 the name
// only exists via the runtime-dispatch trampoline, re-exported here so the
// import path is arch-uniform; the cfg keeps the two aliases from
// colliding when both features hold on x86_64.
#[cfg(all(feature = "runtime-dispatch", not(target_arch = "x86_64")))]
pub use crate::simd_runtime::matmul_i8_to_i32;
// Tile-dispatching sibling of the polyfill `bf16_tile_gemm_16x16` below:
// AMX TDPBF16PS → AVX-512 VDPBF16PS → the same FMA polyfill kernel, selected
// at runtime. Same W1a rationale as `matmul_i8_to_i32` — consumers reach the
// tile ladder through `ndarray::simd::*`; the `_amx` suffix keeps the
// pure-polyfill kernel and the tile-dispatching wrapper distinguishable at
// the call site. `bf16_tile_gemm_16x16_packed` + `PackedBf16B` hoist the
// VNNI pack (and its allocation) out of hot loops — `PackedBf16B::vnni_index`
// additionally supports staging B directly in VNNI layout (zero pack cost).
// `bf16_tile_gemm_tier()` names the tier that will run, for Gotcha-9-style
// run reports.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub use crate::hpc::bf16_tile_gemm::{
    bf16_tile_gemm_16x16 as bf16_tile_gemm_16x16_amx, bf16_tile_gemm_16x16_packed, bf16_tile_gemm_tier, PackedBf16B,
};
// CPU-generation detection (cached): SPR / EMR / GNR / Sierra Forest. Lets a
// consumer report which silicon a run landed on and distinguish "no AMX
// silicon" from "AMX present but not OS-enabled" — both surface via `amx_report`.
#[cfg(target_arch = "x86_64")]
pub use crate::simd_amx::{amx_report, cpu_model, CpuModel};

// Elementwise slice ops — polyfill-dispatched (F32x16/F64x8 chunks + scalar tail).
#[cfg(feature = "std")]
pub use crate::simd_ops::{
    add_f32, add_f32_inplace, add_f64, add_f64_inplace, add_mul_f32, add_mul_f64, add_scalar_f32, bf16_tile_gemm_16x16,
    div_f32, div_f32_inplace, mul_f32, mul_f32_inplace, mul_f64, scale_f32, scale_f32_inplace, sub_f32,
    sub_f32_inplace,
};

// ChaCha20 keystream is NO LONGER an `ndarray::simd` surface: the AdaWorldAPI
// `chacha20` fork (`vendor/chacha20/`) carries the accelerated backend, riding
// the `U32x16` ARX lane above, and is `[patch]`ed transitively under the
// `encryption` AEAD. RustCrypto owns the cipher; ndarray exposes only the lane.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_splat_reduce_sum() {
        let v = F32x16::splat(3.0);
        assert!((v.reduce_sum() - 48.0).abs() < 1e-6);
    }

    #[test]
    fn f32x16_from_array_roundtrip() {
        let data: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let v = F32x16::from_array(data);
        assert_eq!(v.to_array(), data);
    }

    /// ARX triple parity: `U32x16`'s `Add` / `BitXor` / `rotate_left` must be
    /// bit-identical to per-lane `u32` semantics — the ChaCha20/BLAKE primitive
    /// set. Runs on whichever tier this build compiled (`super::*` re-exports the
    /// dispatched `U32x16`), so it gates avx512 / avx2 / neon / wasm / scalar
    /// alike. `rotate_left` is the newly-added op; add/xor are locked with it so
    /// the whole quarter-round vocabulary is proven together.
    #[test]
    fn u32x16_arx_ops_match_scalar() {
        let a_arr: [u32; 16] = [
            0x0000_0000, 0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x1234_5678, 0x9ABC_DEF0, 0xDEAD_BEEF, 0xCAFE_BABE,
            0x0F0F_0F0F, 0xF0F0_F0F0, 0x5555_5555, 0xAAAA_AAAA, 0x0000_00FF, 0xFF00_0000, 0x0101_0101, 0x8080_8080,
        ];
        let b_arr: [u32; 16] = [
            0x9E37_79B9, 0x1111_1111, 0xDEAD_C0DE, 0x0BAD_F00D, 0x7FFF_FFFF, 0x0000_0000, 0xFFFF_FFFF, 0x1357_9BDF,
            0x2468_ACE0, 0xFEDC_BA98, 0x0000_0010, 0x0000_001F, 0xABCD_EF01, 0x1020_4080, 0x0F0F_F0F0, 0xC0DE_CAFE,
        ];
        let a = U32x16::from_array(a_arr);
        let b = U32x16::from_array(b_arr);

        let add = (a + b).to_array();
        let xor = (a ^ b).to_array();
        for i in 0..16 {
            assert_eq!(add[i], a_arr[i].wrapping_add(b_arr[i]), "lane {i} add");
            assert_eq!(xor[i], a_arr[i] ^ b_arr[i], "lane {i} xor");
        }

        // The ARX rotate — ChaCha20 uses 16/12/8/7; edges included.
        for n in [0u32, 1, 7, 8, 12, 16, 24, 31] {
            let got = a.rotate_left(n).to_array();
            for i in 0..16 {
                assert_eq!(got[i], a_arr[i].rotate_left(n), "lane {i} rotate_left({n})");
            }
        }
    }

    /// The u64 ARX rotate — `U64x8::rotate_left` / `rotate_right`, the lane
    /// BLAKE2b (and therefore argon2) needs.
    ///
    /// Runs on whichever tier this build compiled, so it gates the native
    /// AVX-512 `VPROLVQ`/`VPRORVQ` override and the portable scalar arms
    /// against the *same* reference — per-lane `u64::rotate_left`. That
    /// matters more here than for the u32 lane, because unlike every other
    /// lane-wise op in this crate the two implementations are genuinely
    /// different code, not one source form compiled twice.
    ///
    /// Covers BLAKE2b's own amounts (32/24/16/63) plus the edges, and asserts
    /// `rotr(n) == rotl(64 - n)` for each — the identity the two methods'
    /// equivalence rests on.
    #[test]
    fn u64x8_arx_rotate_matches_scalar() {
        use super::U64x8;

        let a_arr: [u64; 8] = [
            0x0123_4567_89AB_CDEF, 0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0001, 0x8000_0000_0000_0000,
            0xDEAD_BEEF_CAFE_BABE, 0x0000_0000_FFFF_FFFF, 0xAAAA_AAAA_5555_5555, 0x0F0F_0F0F_F0F0_F0F0,
        ];
        let a = U64x8::from_array(a_arr);

        // BLAKE2b uses 32/24/16/63; 0 and 63 are the edges, and 64 must wrap
        // to the identity rather than shifting by the full width (UB on u64).
        // 65/127/128/191 pin the documented mod-64 contract: an implementation
        // that only special-cased the width would pass at 64 and fail here.
        // The per-lane oracle is std's own `u64::rotate_left`, which is
        // specified to take its count mod 64, so this asserts agreement with
        // the language rather than with our own restatement of the rule.
        for n in [0u32, 1, 7, 16, 24, 31, 32, 63, 64, 65, 127, 128, 191] {
            let l = a.rotate_left(n).to_array();
            let r = a.rotate_right(n).to_array();
            for i in 0..8 {
                assert_eq!(l[i], a_arr[i].rotate_left(n), "lane {i} rotate_left({n})");
                assert_eq!(r[i], a_arr[i].rotate_right(n), "lane {i} rotate_right({n})");
            }
            // rotr(n) == rotl(64 - n): the identity the pair rests on.
            let back = a.rotate_left((64 - (n % 64)) % 64).to_array();
            assert_eq!(r, back, "rotate_right({n}) != rotate_left(64 - {n})");
        }

        // Round-trip: rotating out and back is the identity for every amount.
        for n in [1u32, 16, 24, 32, 63] {
            assert_eq!(
                a.rotate_left(n).rotate_right(n).to_array(),
                a_arr,
                "rotate_left({n}) then rotate_right({n}) is not the identity"
            );
        }
    }

    /// The BLAKE3 shuffle surface on `U32x16`, checked against the REAL x86
    /// intrinsics it reproduces — applied to each 256-bit half.
    ///
    /// `U32x16` holds two `__m256i` worth of lanes, and BLAKE3's `hash_many`
    /// is either lane-wise or confined within a 128-/256-bit lane, so the two
    /// 8-lane groups run independently in one vector at DEGREE 16. This test
    /// proves that equivalence rather than assuming it: each half of the
    /// `U32x16` result must equal the corresponding 256-bit intrinsic applied
    /// to that half's inputs.
    ///
    /// Load-bearing because the bodies are index loops. Nothing about them is
    /// self-evidently `_mm256_unpacklo_epi32`, and a subtly-wrong interleave
    /// yields wrong hashes, not a compile error.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn u32x16_blake3_shuffles_match_x86_intrinsics_per_half() {
        use core::arch::x86_64::*;

        // Distinct value per lane, so any misplaced lane is visible.
        let a_arr: [u32; 16] = core::array::from_fn(|i| 0x1000_0000 | (i as u32) << 8 | i as u32);
        let b_arr: [u32; 16] = core::array::from_fn(|i| 0x2000_0000 | (i as u32) << 8 | i as u32);
        let (a, b) = (U32x16::from_array(a_arr), U32x16::from_array(b_arr));

        // SAFETY: guarded by `target_feature = "avx2"` on this test.
        unsafe {
            let load = |src: &[u32; 16], half: usize| _mm256_loadu_si256(src.as_ptr().add(half * 8) as *const __m256i);
            let read = |v: __m256i| -> [u32; 8] {
                let mut out = [0u32; 8];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
                out
            };
            // Assert `got`'s two halves against the 256-bit intrinsic applied
            // to each half's own inputs.
            let check = |got: U32x16, want: fn(__m256i, __m256i) -> __m256i, name: &str| {
                let g = got.to_array();
                for half in 0..2 {
                    let expect = read(want(load(&a_arr, half), load(&b_arr, half)));
                    assert_eq!(&g[half * 8..half * 8 + 8], &expect[..], "{name}, half {half}");
                }
            };

            check(a.interleave_lo_u32(b), |x, y| _mm256_unpacklo_epi32(x, y), "interleave_lo_u32");
            check(a.interleave_hi_u32(b), |x, y| _mm256_unpackhi_epi32(x, y), "interleave_hi_u32");
            check(a.interleave_lo_u64(b), |x, y| _mm256_unpacklo_epi64(x, y), "interleave_lo_u64");
            check(a.interleave_hi_u64(b), |x, y| _mm256_unpackhi_epi64(x, y), "interleave_hi_u64");
            check(a.concat_lo_halves(b), |x, y| _mm256_permute2x128_si256(x, y, 0x20), "concat_lo_halves");
            check(a.concat_hi_halves(b), |x, y| _mm256_permute2x128_si256(x, y, 0x31), "concat_hi_halves");

            // BLAKE3 rotates RIGHT by 16/12/8/7 -> rotl(32 - n). Assert against
            // upstream's exact `srli | slli` form, not a reformulation of ours.
            // Unrolled because the shift intrinsics take const immediates --
            // which also mirrors upstream's four separate rot fns one-to-one.
            let rot = |got: U32x16, want: fn(__m256i) -> __m256i, name: &str| {
                let g = got.to_array();
                for half in 0..2 {
                    let expect = read(want(load(&a_arr, half)));
                    assert_eq!(&g[half * 8..half * 8 + 8], &expect[..], "{name}, half {half}");
                }
            };
            rot(a.rotate_left(16), |x| _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 16)), "rot16");
            rot(a.rotate_left(20), |x| _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20)), "rot12");
            rot(a.rotate_left(24), |x| _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(x, 24)), "rot8");
            rot(a.rotate_left(25), |x| _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)), "rot7");
        }
    }

    /// The same six methods pinned lane-by-lane with no intrinsics, so every
    /// backend — scalar / avx512 / neon / wasm / nightly — is held to the
    /// identical permutation on machines where the x86 oracle cannot run.
    #[test]
    fn u32x16_blake3_shuffles_are_lane_exact() {
        let a: [u32; 16] = core::array::from_fn(|i| 10 + i as u32);
        let b: [u32; 16] = core::array::from_fn(|i| 30 + i as u32);
        let (va, vb) = (U32x16::from_array(a), U32x16::from_array(b));

        // Per 128-bit quad q: lanes 4q..4q+4, independently in each quad.
        assert_eq!(
            va.interleave_lo_u32(vb).to_array(),
            [10, 30, 11, 31, 14, 34, 15, 35, 18, 38, 19, 39, 22, 42, 23, 43]
        );
        assert_eq!(
            va.interleave_hi_u32(vb).to_array(),
            [12, 32, 13, 33, 16, 36, 17, 37, 20, 40, 21, 41, 24, 44, 25, 45]
        );
        assert_eq!(
            va.interleave_lo_u64(vb).to_array(),
            [10, 11, 30, 31, 14, 15, 34, 35, 18, 19, 38, 39, 22, 23, 42, 43]
        );
        assert_eq!(
            va.interleave_hi_u64(vb).to_array(),
            [12, 13, 32, 33, 16, 17, 36, 37, 20, 21, 40, 41, 24, 25, 44, 45]
        );
        // Per 256-bit half h: lanes 8h..8h+8.
        assert_eq!(
            va.concat_lo_halves(vb).to_array(),
            [10, 11, 12, 13, 30, 31, 32, 33, 18, 19, 20, 21, 38, 39, 40, 41]
        );
        assert_eq!(
            va.concat_hi_halves(vb).to_array(),
            [14, 15, 16, 17, 34, 35, 36, 37, 22, 23, 24, 25, 42, 43, 44, 45]
        );
    }

    /// `exchange::<G>` composes a complete 16x16 transpose over four stages.
    ///
    /// This is the control the codegen oracle's driver applies to
    /// `transpose_16x16_composed`, brought into the test suite so the library
    /// method is checked and not merely its probe-local twin. A packed-but-
    /// wrong shuffle network would pass a codegen histogram and fail here.
    ///
    /// Note the semantics differ from `interleave_*` / `concat_*` above: those
    /// mirror x86's per-128-bit-lane `unpack`, whereas `exchange` pairs lane
    /// `c` with lane `c ^ G` across the whole vector. They are different
    /// permutations and neither substitutes for the other.
    #[test]
    fn u32x16_exchange_stages_compose_a_transpose() {
        fn stage<const G: usize>(m: &mut [U32x16; 16]) {
            for r in 0..16 {
                if r & G == 0 {
                    let (lo, hi) = m[r].exchange::<G>(m[r | G]);
                    m[r] = lo;
                    m[r | G] = hi;
                }
            }
        }

        // Row r, lane c = r*16 + c. A transpose must yield row r, lane c = c*16 + r.
        let mut m: [U32x16; 16] =
            core::array::from_fn(|r| U32x16::from_array(core::array::from_fn(|c| (r * 16 + c) as u32)));

        stage::<1>(&mut m);
        stage::<2>(&mut m);
        stage::<4>(&mut m);
        stage::<8>(&mut m);

        for (r, row) in m.iter().enumerate() {
            let want: [u32; 16] = core::array::from_fn(|c| (c * 16 + r) as u32);
            assert_eq!(row.to_array(), want, "row {r} after four exchange stages");
        }
    }

    /// Each granularity in isolation, pinned lane-by-lane, so a backend that
    /// gets one stage wrong is not masked by the composition above.
    #[test]
    fn u32x16_exchange_is_lane_exact_per_granularity() {
        let a: [u32; 16] = core::array::from_fn(|i| 10 + i as u32);
        let b: [u32; 16] = core::array::from_fn(|i| 30 + i as u32);
        let (va, vb) = (U32x16::from_array(a), U32x16::from_array(b));

        // lo[c] = a[c] when c & G == 0 else b[c ^ G]
        // hi[c] = b[c] when c & G != 0 else a[c ^ G]
        for g in [1usize, 2, 4, 8] {
            let (lo, hi) = match g {
                1 => va.exchange::<1>(vb),
                2 => va.exchange::<2>(vb),
                4 => va.exchange::<4>(vb),
                _ => va.exchange::<8>(vb),
            };
            let want_lo: [u32; 16] = core::array::from_fn(|c| if c & g == 0 { a[c] } else { b[c ^ g] });
            let want_hi: [u32; 16] = core::array::from_fn(|c| if c & g != 0 { b[c] } else { a[c ^ g] });
            assert_eq!(lo.to_array(), want_lo, "exchange::<{g}> lo");
            assert_eq!(hi.to_array(), want_hi, "exchange::<{g}> hi");
        }
    }

    /// `U32x8`'s shuffle + rotate surface, checked against the REAL x86
    /// intrinsics it claims to reproduce.
    ///
    /// This is the load-bearing test for the BLAKE3 `rust_avx2.rs` port. Those
    /// six methods have plain index-loop bodies, so nothing about them is
    /// self-evidently equal to `_mm256_unpacklo_epi32` and friends — and the
    /// per-128-bit-lane semantics are exactly the kind of thing an "obvious"
    /// whole-vector implementation gets wrong while still looking reasonable.
    /// A transpose built on a subtly-wrong interleave produces wrong hashes,
    /// not a compile error.
    ///
    /// Gated on `target_feature = "avx2"` because it calls the intrinsics
    /// directly as the oracle. The methods themselves are backend-agnostic;
    /// `u32x8_shuffle_surface_is_lane_exact` below covers them everywhere.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn u32x8_shuffles_match_x86_intrinsics() {
        use super::U32x8;
        use core::arch::x86_64::*;

        // Distinct values in every lane, so any misplaced lane is visible.
        let a_arr: [u32; 8] = [
            0x1000_0000, 0x1100_0001, 0x1200_0002, 0x1300_0003, 0x1400_0004, 0x1500_0005, 0x1600_0006, 0x1700_0007,
        ];
        let b_arr: [u32; 8] = [
            0x2000_0000, 0x2100_0001, 0x2200_0002, 0x2300_0003, 0x2400_0004, 0x2500_0005, 0x2600_0006, 0x2700_0007,
        ];
        let (a, b) = (U32x8::from_array(a_arr), U32x8::from_array(b_arr));

        // SAFETY: guarded by `target_feature = "avx2"` on this test.
        unsafe {
            let va = _mm256_loadu_si256(a_arr.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b_arr.as_ptr() as *const __m256i);
            let read = |v: __m256i| -> [u32; 8] {
                let mut out = [0u32; 8];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
                out
            };

            assert_eq!(
                a.interleave_lo_u32(b).to_array(),
                read(_mm256_unpacklo_epi32(va, vb)),
                "interleave_lo_u32 != _mm256_unpacklo_epi32"
            );
            assert_eq!(
                a.interleave_hi_u32(b).to_array(),
                read(_mm256_unpackhi_epi32(va, vb)),
                "interleave_hi_u32 != _mm256_unpackhi_epi32"
            );
            assert_eq!(
                a.interleave_lo_u64(b).to_array(),
                read(_mm256_unpacklo_epi64(va, vb)),
                "interleave_lo_u64 != _mm256_unpacklo_epi64"
            );
            assert_eq!(
                a.interleave_hi_u64(b).to_array(),
                read(_mm256_unpackhi_epi64(va, vb)),
                "interleave_hi_u64 != _mm256_unpackhi_epi64"
            );
            assert_eq!(
                a.concat_lo_halves(b).to_array(),
                read(_mm256_permute2x128_si256(va, vb, 0x20)),
                "concat_lo_halves != _mm256_permute2x128_si256(_, _, 0x20)"
            );
            assert_eq!(
                a.concat_hi_halves(b).to_array(),
                read(_mm256_permute2x128_si256(va, vb, 0x31)),
                "concat_hi_halves != _mm256_permute2x128_si256(_, _, 0x31)"
            );

            // BLAKE3 rotates RIGHT by 16/12/8/7, expressed as rotl(32 - n).
            // Upstream implements each as `srli | slli`; assert against that
            // exact form, not against a reformulation of our own. Written out
            // four times rather than looped because the shift intrinsics take
            // const immediates -- which also makes this mirror upstream's four
            // separate `rot16` / `rot12` / `rot8` / `rot7` functions one-to-one.
            assert_eq!(
                a.rotate_left(16).to_array(),
                read(_mm256_or_si256(_mm256_srli_epi32(va, 16), _mm256_slli_epi32(va, 16))),
                "rotate_left(16) != upstream rot16"
            );
            assert_eq!(
                a.rotate_left(20).to_array(),
                read(_mm256_or_si256(_mm256_srli_epi32(va, 12), _mm256_slli_epi32(va, 20))),
                "rotate_left(20) != upstream rot12"
            );
            assert_eq!(
                a.rotate_left(24).to_array(),
                read(_mm256_or_si256(_mm256_srli_epi32(va, 8), _mm256_slli_epi32(va, 24))),
                "rotate_left(24) != upstream rot8"
            );
            assert_eq!(
                a.rotate_left(25).to_array(),
                read(_mm256_or_si256(_mm256_srli_epi32(va, 7), _mm256_slli_epi32(va, 25))),
                "rotate_left(25) != upstream rot7"
            );
        }
    }

    /// The same six methods, asserted lane-by-lane against their documented
    /// index formulas — on EVERY backend, with no intrinsics involved.
    ///
    /// The intrinsic test above can only run where AVX2 exists. This one
    /// pins the contract on scalar / avx512 / any future arm, so a backend
    /// cannot quietly diverge on a machine where the oracle is unavailable.
    #[test]
    fn u32x8_shuffle_surface_is_lane_exact() {
        use super::U32x8;

        let a: [u32; 8] = [10, 11, 12, 13, 14, 15, 16, 17];
        let b: [u32; 8] = [20, 21, 22, 23, 24, 25, 26, 27];
        let (va, vb) = (U32x8::from_array(a), U32x8::from_array(b));

        assert_eq!(va.interleave_lo_u32(vb).to_array(), [10, 20, 11, 21, 14, 24, 15, 25]);
        assert_eq!(va.interleave_hi_u32(vb).to_array(), [12, 22, 13, 23, 16, 26, 17, 27]);
        assert_eq!(va.interleave_lo_u64(vb).to_array(), [10, 11, 20, 21, 14, 15, 24, 25]);
        assert_eq!(va.interleave_hi_u64(vb).to_array(), [12, 13, 22, 23, 16, 17, 26, 27]);
        assert_eq!(va.concat_lo_halves(vb).to_array(), [10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!(va.concat_hi_halves(vb).to_array(), [14, 15, 16, 17, 24, 25, 26, 27]);

        for n in [0u32, 1, 7, 16, 20, 24, 25, 31] {
            let got = va.rotate_left(n).to_array();
            for i in 0..8 {
                assert_eq!(got[i], a[i].rotate_left(n), "lane {i} rotate_left({n})");
            }
        }
    }

    #[test]
    fn f32x16_add_sub_mul_div() {
        let a = F32x16::splat(6.0);
        let b = F32x16::splat(2.0);
        assert!(((a + b).reduce_sum() - 128.0).abs() < 1e-4);
        assert!(((a - b).reduce_sum() - 64.0).abs() < 1e-4);
        assert!(((a * b).reduce_sum() - 192.0).abs() < 1e-4);
        assert!(((a / b).reduce_sum() - 48.0).abs() < 1e-4);
    }

    #[test]
    fn f32x16_mul_add_fma() {
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        let r = a.mul_add(b, c);
        assert!((r.reduce_sum() - 112.0).abs() < 1e-4);
    }

    #[test]
    fn f32x16_mask_select() {
        let a =
            F32x16::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let threshold = F32x16::splat(8.5);
        let mask = a.simd_lt(threshold);
        let result = mask.select(F32x16::splat(1.0), F32x16::splat(0.0));
        assert!((result.reduce_sum() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn f64x8_splat_reduce_sum() {
        let v = F64x8::splat(3.0);
        assert!((v.reduce_sum() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn f64x8_from_array_roundtrip() {
        let data: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let v = F64x8::from_array(data);
        assert_eq!(v.to_array(), data);
    }

    #[test]
    fn f64x8_mul_add() {
        let a = F64x8::splat(2.0);
        let b = F64x8::splat(3.0);
        let c = F64x8::splat(1.0);
        let r = a.mul_add(b, c);
        assert!((r.reduce_sum() - 56.0).abs() < 1e-10);
    }

    #[test]
    fn f32x16_abs_neg() {
        let a = F32x16::splat(-5.0);
        assert!((a.abs().reduce_sum() - 80.0).abs() < 1e-4);
        let b = F32x16::splat(3.0);
        assert!(((-b).reduce_sum() - (-48.0)).abs() < 1e-4);
    }

    #[test]
    fn f32x16_from_slice_to_slice() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        let mut out = vec![0.0f32; 20];
        v.copy_to_slice(&mut out);
        assert_eq!(&out[..16], &data[..16]);
    }

    #[test]
    fn simd_exp_f32_basic() {
        let zero = F32x16::splat(0.0);
        let result = simd_exp_f32(zero);
        assert!((result.reduce_sum() / 16.0 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn simd_exp_f32_handles_positive_infinity() {
        // Pre-fix: pow2n_from_int saturated f32::INFINITY to i32::MAX,
        // (i32::MAX + 127) panicked in debug / wrapped in release to a
        // garbage exponent, and simd_exp_f32(+Inf) silently returned ~0.5.
        // Post-fix: input is clamped to 88.722 → exp(88.722) ≈ 3.4e38,
        // representable but near f32::MAX. Saturated, not garbage.
        let inf = F32x16::splat(f32::INFINITY);
        let result = simd_exp_f32(inf);
        let arr = result.to_array();
        for &v in &arr {
            assert!(v.is_finite(), "exp(+Inf) must saturate to finite, got {}", v);
            assert!(v > 1e30, "exp(+Inf) must saturate to a large value, got {}", v);
        }
    }

    #[test]
    fn simd_exp_f32_handles_negative_infinity() {
        // -Inf → clamped to -87.336 → exp ≈ 1.4e-38, near zero but representable.
        let neg_inf = F32x16::splat(f32::NEG_INFINITY);
        let result = simd_exp_f32(neg_inf);
        let arr = result.to_array();
        for &v in &arr {
            assert!(v.is_finite(), "exp(-Inf) must saturate to finite, got {}", v);
            assert!(v >= 0.0 && v < 1e-30, "exp(-Inf) must saturate near 0, got {}", v);
        }
    }

    #[test]
    fn simd_exp_f32_propagates_nan() {
        // simd_clamp is max(lo).min(hi); _mm512_max_ps returns the SECOND
        // operand on NaN, so without the nan_mask save/restore, NaN would
        // be silently clamped to -87.336 → exp ≈ 1.4e-38 (a tiny finite
        // value pretending to be valid). With the mask, NaN propagates.
        // Per codex review on PR #142.
        let nan = F32x16::splat(f32::NAN);
        let result = simd_exp_f32(nan);
        let arr = result.to_array();
        for &v in &arr {
            assert!(v.is_nan(), "exp(NaN) must propagate NaN, got {}", v);
        }
    }

    #[test]
    fn simd_exp_f32_propagates_nan_per_lane() {
        // Mixed input: lanes 0,4,8,12 are NaN; rest are 0.0. Verify that
        // NaN propagates only in those lanes; the others compute exp(0)=1.
        let mut data = [0.0f32; 16];
        for i in (0..16).step_by(4) {
            data[i] = f32::NAN;
        }
        let result = simd_exp_f32(F32x16::from_array(data));
        let arr = result.to_array();
        for (i, &v) in arr.iter().enumerate() {
            if i % 4 == 0 {
                assert!(v.is_nan(), "lane {} should be NaN, got {}", i, v);
            } else {
                assert!((v - 1.0).abs() < 1e-4, "lane {} should be exp(0)=1, got {}", i, v);
            }
        }
    }

    #[test]
    fn simd_exp_f32_handles_large_positive() {
        // Without the clamp, x = 200 produced n = 288, ni + 127 = 415 which
        // is still in u32 range so didn't panic, but the resulting bits were
        // outside valid f32 exponent range, producing garbage that masqueraded
        // as a "valid" answer.
        let big = F32x16::splat(200.0);
        let result = simd_exp_f32(big);
        let arr = result.to_array();
        for &v in &arr {
            assert!(v.is_finite(), "exp(200) must saturate, got {}", v);
        }
    }
}
