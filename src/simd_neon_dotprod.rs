//! NEON + dotprod + FP16 tier — ARMv8.2-A `+dotprod,+fp16`.
//!
//! Builds on `simd_neon_baseline.rs`. This tier adds two ISA features
//! that landed together in ARMv8.2-A and ship as a pair on every
//! consumer aarch64 chip from ~2019 onward.
//!
//! # Silicon
//!
//! - **Raspberry Pi 5** (BCM2712, Cortex-A76) — ARMv8.2-A. The first
//!   Pi with dotprod + fp16. 4× int8 dot-product throughput vs Pi 4.
//! - **Cortex-A55r2+** — ARMv8.2-A revision 2 silicon. Mid-range
//!   Android, low-power servers (Ampere Altra little cores).
//! - **Cortex-A75 / A76 / A77 / A78 / A78AE / X1 / X2 / X3** — all
//!   ARMv8.2-A or later, all have dotprod + fp16.
//! - **Apple A11 / A12 / A13 / M1** — ARMv8.3-A or later. (Note: Apple
//!   exposes `bf16` from M2 onward; see `simd_neon_bf16.rs`.)
//! - **Snapdragon 8 Gen 1+** — ARMv8.4-A+. dotprod + fp16 always on.
//!
//! # How to detect at runtime
//!
//! - **Linux**: read `/proc/cpuinfo`'s `Features:` line. dotprod is
//!   reported as `asimddp`; fp16 as `asimdhp` (asimd half-precision)
//!   and `fphp` (scalar fp16). Both should be present on this tier.
//!   Or use `std::arch::is_aarch64_feature_detected!("dotprod")` /
//!   `("fp16")` — already wired into `simd.rs::detect_tier()` at
//!   line 63: `if std::arch::is_aarch64_feature_detected!("dotprod")
//!   { return Tier::NeonDotProd; }`.
//! - **macOS** (Apple silicon): all M-series chips have dotprod + fp16.
//!   `sysctl hw.optional.arm.FEAT_DotProd` returns 1.
//! - **Android NDK**: `getauxval(AT_HWCAP) & HWCAP_ASIMDDP` (bit 20).
//!
//! # How to detect at compile time
//!
//! Cargo config flags `-Ctarget-feature=+dotprod,+fp16` (or
//! `-Ctarget-cpu=cortex-a76` which includes both as defaults). Inside
//! Rust:
//!
//! ```ignore
//! #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
//! pub use crate::simd_neon_dotprod::dot_i8x16;
//!
//! #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
//! pub use crate::simd_neon_dotprod::F16x16;
//! ```
//!
//! # What you get
//!
//! ## DotProd (already wired in `simd_neon.rs:190-203`)
//!
//! - `vdotq_s32(acc, a, b)` — 4×(4×i8·4×i8) → 4×i32 in ONE instruction.
//! - `vdotq_u32` — unsigned variant.
//! - `vdotq_laned_s32` — broadcast lane of `b` across all 4 dotproducts.
//! - Already exposed: `dot_i8x16_neon`, `codebook_gather_i8_dotprod`.
//!
//! ## FP16 (TODO — stubs below)
//!
//! - `vaddq_f16`, `vsubq_f16`, `vmulq_f16`, `vfmaq_f16` — 8×fp16 per
//!   `float16x8_t` register. 2× the throughput of fp32 for inference.
//! - `vcvt_f32_f16` / `vcvt_f16_f32` — pack-into / unpack-from fp32.
//!   Already implemented as scalar bit-twiddle in `simd_neon.rs:324-420`
//!   (`f16_to_f32_scalar`, `f32_to_f16_scalar`); the SIMD versions
//!   need the `target_feature = "fp16"` gate.
//!
//! # Composed wrapper shape
//!
//! `F16x16` = `[float16x8_t; 2]` — two 128-bit FP16 registers compose
//! 16 lanes, matching the AVX-512 / nightly width.
//!
//! # Cargo config
//!
//! ```toml
//! # .cargo/config-pi5.toml
//! [build]
//! target = "aarch64-unknown-linux-gnu"
//! [target.aarch64-unknown-linux-gnu]
//! rustflags = ["-Ctarget-cpu=cortex-a76", "-Ctarget-feature=+dotprod,+fp16"]
//! ```
//!
//! Cross-build for Pi 5 from x86_64:
//! `cargo --config .cargo/config-pi5.toml build`.
//!
//! Apple build (Mac M1+ runs this natively under `aarch64-apple-darwin`):
//! ```toml
//! # .cargo/config-apple.toml
//! [build]
//! target = "aarch64-apple-darwin"
//! [target.aarch64-apple-darwin]
//! rustflags = ["-Ctarget-cpu=apple-m1", "-Ctarget-feature=+dotprod,+fp16"]
//! ```

#![cfg(all(target_arch = "aarch64", feature = "std"))]

// ─── F16 stubs ────────────────────────────────────────────────────────
//
// Target intrinsic family: `float16x8_t` arithmetic, gated on
// `target_feature = "fp16"`. Rust stable 1.95 has `float16x8_t` in
// `core::arch::aarch64` but the intrinsics like `vaddq_f16` are nightly
// only (issue #112800). On stable we have three options:
//
//   1. Use `vfmaq_f16` etc. via `unsafe asm!` with the literal
//      ARM64 instruction (e.g. `asm!("fmla v0.8h, v1.8h, v2.8h",
//      in("v1") a, in("v2") b, inout("v0") acc)`). This is the AMX
//      precedent — see `src/simd_amx.rs::amx_available()` for the
//      pattern of byte-encoded asm on stable Rust.
//   2. Round-trip through f32 using the existing scalar conversions
//      in `simd_neon.rs:324-420`. Loses throughput; only useful as a
//      correctness baseline.
//   3. Wait for `core::arch::aarch64::vaddq_f16` to stabilize.
//
// Phase 3 implementation should pick (1) — it's the only path that
// realizes the 2× throughput claim for inference. The asm-byte
// encoding for `fmla v0.8h, v1.8h, v2.8h` is `0x0e40cc20` (verify with
// `aarch64-linux-gnu-objdump`).

/// Placeholder for the FP16 16-lane composed wrapper.
///
/// Real implementation: `pub struct F16x16(pub [float16x8_t; 2])`.
/// Add methods mirroring `F32x16` (splat, from_slice, add, sub, mul,
/// mul_add, sqrt, reduce_sum, simd_min/max/eq/lt/le/gt/ge/ne).
///
/// Intrinsics map:
/// - `splat` → `vdupq_n_f16`
/// - `from_slice` → `vld1q_f16`
/// - `to_array` → `vst1q_f16`
/// - `Add` → `vaddq_f16`
/// - `Sub` → `vsubq_f16`
/// - `Mul` → `vmulq_f16`
/// - `mul_add` → `vfmaq_f16` (fused multiply-add)
/// - `sqrt` → `vsqrtq_f16`
/// - `reduce_sum` → `vaddvq_f16`
/// - mask compare → `vcgtq_f16`, `vceqq_f16`, ... yielding `uint16x8_t`
///
/// Until implemented, consumers reach `crate::simd::F16x16` via
/// `simd_avx2::F16Scaler` (scalar polyfill) or `simd_nightly::F16x16`
/// (`core::simd::f16x16`).
pub struct F16x16Stub;

impl F16x16Stub {
    /// Placeholder — real impl needs `target_feature = "fp16"` + the
    /// asm-byte / intrinsic path documented above.
    pub fn unimplemented() -> ! {
        unimplemented!(
            "F16x16 NEON dotprod-tier implementation TODO. See \
             src/simd_neon_dotprod.rs module docs for the intrinsic \
             map and stable-Rust asm-byte path."
        )
    }
}

// ─── DotProd: already implemented in src/simd_neon.rs ────────────────
//
// TODO(Phase-3): move `dot_i8x16_neon` and `codebook_gather_i8_dotprod`
// from `src/simd_neon.rs:191-237` to this file. Then re-export from
// `simd_neon.rs` for backwards compatibility during migration. Both
// already carry `#[target_feature(enable = "dotprod")]` so the gating
// is correct — the move is purely structural.
