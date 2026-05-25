//! NEON + BF16 tier — ARMv8.6-A `+bf16` (or ARMv8.2-A + optional `+bf16`).
//!
//! Builds on `simd_neon_dotprod.rs`. Adds the BF16 instruction family:
//! BFDOT, BFMMLA, BFMLALB, BFMLALT, BFCVT. These are the bf16 cousins
//! of dotprod — same 4× int8 throughput shape, but for the half-the-
//! width bfloat16 type that LLM inference standardized on.
//!
//! # Silicon
//!
//! - **Apple M2 / M3 / M4** (Avalanche/Blizzard, Everest/Sawtooth,
//!   Tupai/Donan) — ARMv8.6-A+. BF16 always on. `sysctl
//!   hw.optional.arm.FEAT_BF16` returns 1. M1 does NOT have BF16 — it's
//!   ARMv8.5-A.
//! - **Snapdragon X Elite / X Plus** (Cortex-X4/X3 cores, Oryon
//!   prime) — ARMv8.7-A. BF16 always on.
//! - **Cortex-A510 / A520 / A710 / A720 / X2 / X3 / X4 / X925** —
//!   ARMv9.0-A+. BF16 always on.
//! - **NVIDIA Grace** (Neoverse V2) — ARMv9-A. BF16 on.
//! - **AWS Graviton 3 / 3E / 4** (Neoverse V1/V2) — V1 added BF16 as
//!   optional ARMv8.4-A extension; V2 makes it mandatory.
//! - **Ampere One (M-series)** — ARMv8.6-A+. BF16 on.
//!
//! # NOT in this tier
//!
//! - Apple M1 (ARMv8.5-A, no BF16) — falls back to `simd_neon_dotprod.rs`
//! - Raspberry Pi 5 (Cortex-A76, ARMv8.2-A, no BF16) — `simd_neon_dotprod.rs`
//! - Any Pi 3/4 / Cortex-A53/A72 — `simd_neon_baseline.rs`
//!
//! # How to detect at runtime
//!
//! - **Linux**: `/proc/cpuinfo` Features line should show `bf16`.
//!   `getauxval(AT_HWCAP2) & HWCAP2_BF16` (bit 14).
//!   `std::arch::is_aarch64_feature_detected!("bf16")` — recommended.
//! - **macOS**: `sysctl hw.optional.arm.FEAT_BF16` → `1` means yes.
//!   On M2+ it's always 1; on M1 it's 0.
//! - **Windows ARM64**: `IsProcessorFeaturePresent(PF_ARM_V83_BF16)`
//!   (constant added in Win11 24H2 SDK).
//!
//! # How to detect at compile time
//!
//! Cargo config flags:
//! - `-Ctarget-feature=+bf16` — enables BF16 intrinsics + cfg gate.
//! - `-Ctarget-cpu=apple-m2` — implies bf16 + everything else.
//! - `-Ctarget-cpu=neoverse-v2` — Graviton 4 baseline.
//! - `-Ctarget-cpu=cortex-x4` — Snapdragon X Elite / Cortex-X4 cores.
//!
//! Inside Rust:
//!
//! ```ignore
//! #[cfg(all(target_arch = "aarch64", target_feature = "bf16"))]
//! pub use crate::simd_neon_bf16::{BF16x8, BF16x16, bfdot, bfmmla};
//! ```
//!
//! # What you get
//!
//! ## BF16 dot-product / matrix-multiply
//!
//! - `vbfdotq_f32(acc, a, b)` — 2×(2×bf16·2×bf16) → 2×f32, accumulated
//!   into 4×f32 register. The bf16 analogue of `vdotq_s32`.
//! - `vbfmmlaq_f32(acc, a, b)` — 2×2 outer product BFMMLA. The crown
//!   jewel for transformer GEMM — accumulates a full 2×2 f32 tile per
//!   instruction. 8 bf16 mults + 4 f32 adds per cycle on M2.
//! - `vbfmlalbq_f32` / `vbfmlaltq_f32` — bottom / top half multiply-
//!   accumulate, lane-by-lane variant of BFDOT.
//! - `vbfmlalbq_laned_f32` — broadcast one lane across all bf16
//!   multiplications. Useful for matvec.
//!
//! ## BF16 conversion
//!
//! - `vcvt_bf16_f32` / `vcvtq_low_bf16_f32` / `vcvtq_high_bf16_f32` —
//!   pack 4×f32 → 4×bf16. Hardware rounding (no manual RNE needed
//!   like the AVX-512BF16 `_mm512_cvtne2ps_pbh` path in
//!   `simd_avx512.rs`).
//! - Scalar f32 ↔ bf16: trivial high-16-bit slice (the scalar paths in
//!   `src/simd.rs:1604-1626` work everywhere, including this tier).
//!
//! # Composed wrapper shapes
//!
//! - `BF16x8` = `bfloat16x8_t` — native 128-bit register, 8 bf16 lanes.
//!   Matches AVX-512BF16 `BF16x8 = __m128bh` in shape.
//! - `BF16x16` = `[bfloat16x8_t; 2]` — two 128-bit registers, 16 bf16
//!   lanes. Matches AVX-512BF16 `BF16x16 = __m256bh` in shape.
//!
//! # Cargo configs
//!
//! ```toml
//! # .cargo/config-apple-m2.toml — Apple M2/M3/M4
//! [build]
//! target = "aarch64-apple-darwin"
//! [target.aarch64-apple-darwin]
//! rustflags = ["-Ctarget-cpu=apple-m2", "-Ctarget-feature=+bf16,+dotprod,+fp16"]
//! ```
//!
//! ```toml
//! # .cargo/config-graviton.toml — AWS Graviton 3/4
//! [build]
//! target = "aarch64-unknown-linux-gnu"
//! [target.aarch64-unknown-linux-gnu]
//! rustflags = ["-Ctarget-cpu=neoverse-v2", "-Ctarget-feature=+bf16"]
//! ```
//!
//! ```toml
//! # .cargo/config-snapdragon-x.toml — Snapdragon X Elite (Win/Linux)
//! [build]
//! target = "aarch64-pc-windows-msvc"   # or aarch64-unknown-linux-gnu
//! rustflags = ["-Ctarget-cpu=cortex-x4", "-Ctarget-feature=+bf16,+i8mm"]
//! ```
//!
//! # Stable-Rust constraint
//!
//! Same as the FP16 tier: `bfloat16x8_t` exists in `core::arch::aarch64`
//! on stable, but the intrinsics (`vbfdotq_f32`, `vbfmmlaq_f32`, ...)
//! are nightly-only (issue #117222). Two paths on stable 1.95:
//!
//!   1. **asm! byte encoding** — same pattern as `src/simd_amx.rs`
//!      uses for AMX. Example:
//!      ```ignore
//!      // BFDOT v0.4s, v1.8h, v2.8h
//!      asm!(".inst 0x4e41ec00", inout("v0") acc, in("v1") a, in("v2") b);
//!      // BFMMLA v0.4s, v1.8h, v2.8h
//!      asm!(".inst 0x6e42ec01", inout("v0") acc, in("v1") a, in("v2") b);
//!      ```
//!      Verify the encoding with `aarch64-linux-gnu-objdump --disassemble`
//!      on a reference compile.
//!   2. **Round-trip through f32** — convert bf16 → f32 (scalar bit-
//!      shift), use the existing `vfmaq_f32` from baseline NEON, convert
//!      back. Loses the 4× throughput; only as a correctness anchor for
//!      the asm path.
//!
//! Path (1) is the only one worth shipping. The asm-byte fallback IS
//! how `simd_amx.rs` ships AMX on stable Rust today — same pattern.

#![cfg(all(target_arch = "aarch64", feature = "std"))]

// ─── BF16 stubs ──────────────────────────────────────────────────────

/// Placeholder for the BF16 8-lane native wrapper.
///
/// Real implementation: `pub struct BF16x8(pub bfloat16x8_t)`. API
/// surface mirrors `simd_avx512::BF16x8`:
/// - `splat(bits: u16) -> Self` (broadcast bf16 bit pattern across 8 lanes)
/// - `from_slice(s: &[u16]) -> Self` (load 8 raw bf16 bits as u16s)
/// - `to_array(self) -> [u16; 8]`
/// - `dot_f32(self, other: Self, acc: F32x4) -> F32x4` — wraps BFDOT
/// - `cvt_to_f32_lo(self) -> F32x4`, `cvt_to_f32_hi(self) -> F32x4`
///
/// Without `target_feature = "bf16"`, this falls back to round-trip
/// through f32 (slow). With the feature on, it uses asm-byte BFDOT.
pub struct BF16x8Stub;

/// Placeholder for the BF16 16-lane composed wrapper.
///
/// Real implementation: `pub struct BF16x16(pub [bfloat16x8_t; 2])`.
/// API mirror of `simd_avx512::BF16x16`. The 16-lane variant is the
/// natural width for matmul tile rows in transformer attention.
pub struct BF16x16Stub;

impl BF16x8Stub {
    pub fn unimplemented() -> ! {
        unimplemented!(
            "BF16x8 NEON bf16-tier implementation TODO. See \
             src/simd_neon_bf16.rs module docs for the BFDOT / BFMMLA \
             asm-byte encoding (stable Rust 1.95 can't reach the \
             nightly-only vbfdotq_f32 intrinsic). Reference: \
             src/simd_amx.rs's `.byte` pattern."
        )
    }
}

impl BF16x16Stub {
    pub fn unimplemented() -> ! {
        unimplemented!(
            "BF16x16 NEON bf16-tier implementation TODO. Two-half \
             composed wrapper [bfloat16x8_t; 2] — see module docs."
        )
    }
}

// ─── BFMMLA: the prize intrinsic ─────────────────────────────────────
//
// BFMMLA is the most important instruction this tier unlocks. It
// computes a 2×2 outer-product matrix multiply of bf16 inputs,
// accumulating into a 2×2 f32 tile. One instruction = 8 bf16 mults +
// 4 f32 adds. On Apple M2 the throughput is ~32 GFLOP/s per core in
// bf16-matmul-bound kernels.
//
// Encoding for `BFMMLA Vd.4s, Vn.8h, Vm.8h`: 0x6e40_ec00 | (Vm << 16)
// | (Vn << 5) | Vd. Use a `bfmmla!` macro to emit the asm-byte for any
// (acc, a, b) v-register triple.
//
// TODO(Phase-3): implement `bfmmla(acc: F32x4, a: BF16x8, b: BF16x8)
// -> F32x4` as the primary export. The rest of the BF16 API builds on
// it (BFDOT is BFMMLA's diagonal, BFMLALB/T are its half-slices).

// ─── BFDOT: same shape as DotProd, but bf16 ──────────────────────────
//
// Where `vdotq_s32(acc, a, b)` does 4×(4×i8·4×i8) → 4×i32, BFDOT does
// 2×(2×bf16·2×bf16) → 2×f32 accumulated into 4×f32. The bf16 analogue
// is HALF the lane count per output (2 vs 4) because bf16 is twice as
// wide as i8.
//
// TODO(Phase-3): implement `bfdot(acc: F32x4, a: BF16x8, b: BF16x8)
// -> F32x4`. Asm-byte for `BFDOT Vd.4s, Vn.8h, Vm.8h`:
//   0x4e40_ec00 | (Vm << 16) | (Vn << 5) | Vd
