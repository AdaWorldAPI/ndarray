# Plan: x86-64-v2 SSE tier (v1)

Operator decision 2026-10-10: a compile-time SIMD tier for **x86-64-v2**
builds (SSE2 through SSE4.2, no AVX). Baseline x86-64 builds (no SSE4.2) are
unchanged: they keep the AVX2 arm and the `cpu_guard` AVX2 startup floor.
No runtime feature checks in the SIMD code.

## Selection

* Arm predicate: `all(target_arch = "x86_64", target_feature = "sse4.2",
  not(target_feature = "avx"))`, written out at each gate.
* Reached by `target-cpu=native` on a v2-only CPU (Nehalem/Westmere, Atom
  Silvermont-class, VMs that hide AVX), or the new `.cargo/config-v2.toml`
  (`-Ctarget-cpu=x86-64-v2`).
* Unchanged: AVX-512, AVX2, AVX-without-AVX2 (`simd_avx.rs`), and baseline.

## Shape: route to the scalar realization, not a new intrinsic file

Every type in `simd_avx2.rs` / `simd_avx512.rs` wraps a 256- or 512-bit
register (`U8x32(pub __m256i)`, …), so none of them can be the v2 realization:
unlike the AVX arm, there is no impl-block split that works. `simd.rs` instead
routes the v2 arm to `simd_scalar.rs` — plain `[T; N]` arrays that LLVM
vectorizes to SSE. Measured earlier (existing `simd_scalar.rs` compiled at
x86-64 / v2 / sandybridge / v3): element-wise ops ~2× the AVX2 instruction
count, SSE2/SSSE3/SSE4.2 produce identical code, reductions stay scalar, and
`mul_add` becomes a software `fmaf` call on any tier without FMA.

Only the `simd_*` backends import the x86 modules directly; every consumer
goes through `crate::simd`, so the switch is confined to `simd.rs`.

## Work items

- [ ] P0 TESTS FIRST: `v2-qemu` parity arm (`--config .cargo/config-v2.toml`,
      run under `qemu-x86_64-static -cpu Nehalem`) in `scripts/masking-parity.sh`
      and a `realization/v2` row in `simd-matrix.yaml`, with an objdump witness
      (no AVX instruction at all: no `%ymm`, no VEX `v`-prefixed op). Expect it
      to fail before the arm exists.
- [ ] P1 `.cargo/config-v2.toml`; `simd.rs` routing; `cpu_guard` floor gated off
      the v2 arm.
- [ ] P2 close the API gap: compile v2, list every missing item, add each to
      `simd_scalar.rs` (types `U8x32`, `F32Mask16`, `F64Mask8` and the x86-only
      methods). Safe Rust only.
- [ ] P3 gates: lib tests on v2 under qemu Nehalem; parity on every arm; clippy
      v2/v3/v4/avx/baseline; codegen witness avx2/avx512 unchanged.
- [ ] P4 docs: CLAUDE.md realization list, blackboard, `simd.rs` dispatch comment.
