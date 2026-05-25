//! NEON baseline tier — ARMv8.0-A `+neon` only.
//!
//! # Silicon
//!
//! Every aarch64 CPU since ARMv8.0-A ships NEON unconditionally. This
//! tier is the lowest common denominator — the floor every other tier
//! builds on. Concretely:
//!
//! - **Raspberry Pi 3** (BCM2837, Cortex-A53) — ARMv8.0-A
//! - **Raspberry Pi 4** (BCM2711, Cortex-A72) — ARMv8.0-A
//! - **Pi CM3 / CM4 (4 GB)** — same A72 silicon
//! - Anything reporting `Features: ... asimd ...` in `/proc/cpuinfo`
//!   without `asimddp`, `asimdfhm`, `asimdhp`, or `bf16`.
//!
//! # What you get
//!
//! Native 128-bit lanes: `float32x4_t`, `float64x2_t`, `int8x16_t`,
//! `uint8x16_t`, `int16x8_t`, `uint16x8_t`, `int32x4_t`, `int64x2_t`,
//! `uint32x4_t`, `uint64x2_t`. Standard NEON arithmetic — `vaddq_*`,
//! `vsubq_*`, `vmulq_*`, `vfmaq_*`, `vminq_*`, `vmaxq_*`, gather /
//! scatter via `vld1q_*` / `vst1q_*`, lane select, reduce.
//!
//! # What you do NOT get
//!
//! - **dotprod** (SDOT/UDOT) → see `simd_neon_dotprod.rs`
//! - **fp16 arithmetic** (`vfmlaq_f16`, `vaddq_f16`) → see
//!   `simd_neon_dotprod.rs`
//! - **bf16** (`vbfdotq_f32`, `vbfmlalbq_f32`) → see `simd_neon_bf16.rs`
//! - **SVE2** (variable-length vectors) → not in any current tier file.
//!
//! # 512-bit composed wrappers
//!
//! `crate::simd::F32x16` / `U8x64` etc. on aarch64 compose 4× 128-bit
//! NEON registers into one logical wrapper, e.g.
//! `pub struct F32x16(pub [float32x4_t; 4])`. The four loads/stores
//! pipeline well on dual-issue cores (A72, A76, M-series).
//!
//! # Cargo config
//!
//! No special flags needed. `aarch64-unknown-linux-gnu` / `aarch64-
//! apple-darwin` already enable NEON. Pi 3/4 cross-builds:
//! `cargo build --target=aarch64-unknown-linux-gnu` from any host.
//!
//! # Status
//!
//! Scaffold only — placeholder for Phase 3 implementation. The actual
//! 128-bit native wrappers (I8x16, I16x8, U8x16, U16x8, U32x4, U64x2,
//! I32x4, I64x2) currently live in `src/simd_neon.rs::aarch64_simd`.
//! That code moves here once the tier split lands.
//!
//! Composed 512-bit wrappers (`F32x16` = `[float32x4_t; 4]` etc.) for
//! the 8 missing int types (U8x64, I8x64, I16x32, I32x16, I64x8,
//! U16x32, U32x16, U64x8) are TODO — currently dispatched to
//! `simd_scalar.rs` via the `scalar::*` fallback at `simd.rs:1593-95`.

#![cfg(all(target_arch = "aarch64", feature = "std"))]

// TODO(Phase-3): move the existing `pub mod aarch64_simd` block from
// `src/simd_neon.rs` (lines 463-1126 of master @ 3c20392f) into this
// file. Then re-export from `simd_neon.rs` for backwards compatibility
// during the migration window. Same pattern as Phase 4 used to extract
// `simd_scalar.rs` via `#[path]` declaration.

// TODO(Phase-3): add the 8 missing 512-bit composed wrappers as
// `[neon_native; 4]`. Apply the `avx2_int_type!`-equivalent macro
// pattern to generate them mechanically — name it `neon_int_type!` and
// keep the API surface identical to the AVX-512 / AVX2 / nightly arms.
//
//   neon_int_type!(U8x64,  u8,  64, uint8x16_t, vaddq_u8,  vsubq_u8);
//   neon_int_type!(I8x64,  i8,  64, int8x16_t,  vaddq_s8,  vsubq_s8);
//   neon_int_type!(U16x32, u16, 32, uint16x8_t, vaddq_u16, vsubq_u16);
//   neon_int_type!(I16x32, i16, 32, int16x8_t,  vaddq_s16, vsubq_s16);
//   neon_int_type!(U32x16, u32, 16, uint32x4_t, vaddq_u32, vsubq_u32);
//   neon_int_type!(I32x16, i32, 16, int32x4_t,  vaddq_s32, vsubq_s32);
//   neon_int_type!(U64x8,  u64, 8,  uint64x2_t, vaddq_u64, vsubq_u64);
//   neon_int_type!(I64x8,  i64, 8,  int64x2_t,  vaddq_s64, vsubq_s64);

// TODO(Phase-3): copy the existing F32x16 / F64x8 paired-load impls
// from `src/simd_neon.rs::aarch64_simd::{F32x16, F64x8, F32Mask16,
// F64Mask8}` here. They already use the composed `[float32x4_t; 4]` /
// `[float64x2_t; 4]` layout this tier expects.
