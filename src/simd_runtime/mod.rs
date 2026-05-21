//! Runtime SIMD dispatch — release-binary distribution path.
//!
//! **Alternative** to the compile-time cascade in `crate::simd::*` /
//! `crate::simd_ops::*`. **Additive**: gated under
//! `--features runtime-dispatch`, does not change any existing path.
//!
//! Use case: ship ONE binary that adapts across heterogeneous deployment
//! silicon — AVX-512 server + AVX2-only laptop + Arrow Lake desktop +
//! Sapphire Rapids workstation — all from the same artifact. The
//! existing compile-time `v3` / `v4` / `native` / `nightly-simd`
//! configs target a single class of CPU per build; the runtime layer
//! targets the union.
//!
//! # Dispatch model
//!
//! One `LazyLock<fn ptr>` per public surface. First call fires the
//! closure which reads [`crate::hpc::simd_caps::simd_caps`] and selects
//! a backend; every subsequent call is one pointer-deref + indirect
//! call. Per-call overhead: ~2-3 ns (the indirect call is a branch
//! target predict; LazyLock deref is one atomic-acquire load that's
//! cache-resident after first hit). Invisible against any SIMD op's
//! actual work (~100+ cycles).
//!
//! ```text
//! consumer site:        crate::simd_runtime::vnni_dot_u8_i8(&a, &b)
//!                          ↓ inline
//!                       unsafe { (*VNNI_DOT_U8_I8_DISPATCH)(a, b) }
//!                          ↓ (indirect call, target predicted after first hit)
//!                       vnni_dot_u8_i8_avx512_with_tail  (or vnni2 or scalar)
//!                          ↓ delegate to existing kernel
//!                       crate::simd_amx::vnni_dot_u8_i8(aligned_slice)
//!                          ↓ _mm512_dpbusd_epi32 × ceil(aligned/64)
//!                       _mm512_reduce_add_epi32
//! ```
//!
//! # Backend reuse — no kernel duplication
//!
//! Every dispatch arm in this module delegates to a kernel that already
//! exists in tree. The runtime layer is just the trampoline; the
//! kernels were added by:
//!
//! | Op | Backends (from where) |
//! |---|---|
//! | `vnni_dot_u8_i8` | `simd_amx::{vnni_dot_u8_i8, vnni2_dot_u8_i8, vnni_dot_u8_i8_scalar}` (PR #143) |
//! | `matmul_i8_to_i32` | `hpc::int8_tile_gemm::{int8_gemm_amx_tiled, int8_gemm_vpdpbusd_zmm, int8_gemm_vpdpbusd_ymm}` (PR #184/#185) |
//! | `matmul_bf16_to_f32` | `hpc::amx_matmul::matmul_bf16_to_f32` (already runtime-dispatched internally, PR #182) |
//! | `gemm_u8_i8` | `simd_int_ops::gemm_u8_i8` (already runtime-dispatched internally, PR #185) |
//! | `cast_*_f16` | `simd_half::{cast_f16_to_f32_batch, cast_f32_to_f16_batch}` (already runtime-dispatched, PR #183) |
//! | `add_mul_f32` / `add_mul_f64` | NEW direct-FMA backends in `add_mul.rs` (no pre-existing slice kernel) |
//!
//! The only NEW kernel code in this module is `add_mul_f32` / `add_mul_f64` —
//! the rest is trampolines around existing functions.
//!
//! # Mutually exclusive with `nightly-simd`
//!
//! The portable-SIMD polyfill (`--features nightly-simd`) replaces the
//! architecture-specific intrinsics in `simd_avx2.rs` / `simd_avx512.rs`
//! / `simd_neon.rs` with `core::simd::*`. The runtime trampolines in
//! this module select BETWEEN those architecture-specific kernels,
//! so the two features are semantically incompatible. The compile-time
//! gate in `Cargo.toml` enforces this.
//!
//! # When to use which dispatch model
//!
//! - **Compile-time `crate::simd::*` / `crate::simd_ops::*`** (default):
//!   benchmarking, library that gets `target-cpu`-tuned per deployment,
//!   embedded / fixed-target builds. Direct monomorphized calls; no
//!   runtime branch on the SIMD op.
//! - **Runtime `crate::simd_runtime::*`** (this module): production
//!   release binary that ships once and runs on heterogeneous silicon.
//!   One indirect call per op; ~2-3 ns overhead.
//!
//! See `.claude/knowledge/simd-dispatch-architecture.md` § 7.1 / Phase 5
//! for the full design rationale.

#![cfg(feature = "runtime-dispatch")]

#[cfg(all(feature = "runtime-dispatch", feature = "nightly-simd"))]
compile_error!(
    "features `runtime-dispatch` and `nightly-simd` are mutually exclusive — \
     the portable-SIMD polyfill (nightly-simd) replaces the architecture-specific \
     intrinsics that runtime-dispatch selects between."
);

pub mod add_mul;
pub mod casts;
pub mod cpu_ops;
pub mod matmul;
pub mod vnni_dot;

// Re-export the public trampoline entry points at the module root so
// consumers can `use crate::simd_runtime::*` and get every op flat.
pub use add_mul::{add_mul_f32, add_mul_f64};
pub use casts::{bf16_to_f32_batch, cast_f16_to_f32_batch, cast_f32_to_f16_batch, f32_to_bf16_batch_rne};
pub use cpu_ops::{cpu_ops, cpu_ops_for_cpu, cpu_ops_for_tier, CpuOps};
pub use matmul::{gemm_u8_i8, matmul_bf16_to_f32, matmul_f32, matmul_i8_to_i32};
pub use vnni_dot::vnni_dot_u8_i8;
