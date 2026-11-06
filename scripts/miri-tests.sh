#!/bin/sh
#
# Miri test runner — ephemeral nightly, scoped to this script ONLY.
#
# Rules of the road (do not violate):
#   * The repo's default toolchain is stable (see rust-toolchain.toml).
#     `cargo build`, `cargo test`, `cargo clippy`, CI's clippy / tests jobs
#     all use stable. Nothing else opts into nightly.
#   * Miri requires nightly because `src/simd_nightly/` is gated on
#     `#![feature(portable_simd)]` (unstable issue #86656), and Miri itself
#     ships only on nightly. This script invokes nightly via `+nightly`,
#     which is an ephemeral, per-invocation switch — it does NOT change
#     the default toolchain.
#   * The `nightly-simd` cargo feature is enabled here ONLY. It routes
#     `crate::simd::*` through `core::simd::*` (the std polyfill) instead
#     of the architecture-specific `_mm*_*` intrinsics, so Miri can
#     actually execute the SIMD code paths. Production builds (and CI's
#     clippy / tests on stable) keep using the intrinsics backend.
#   * `blas` is excluded because Miri cannot FFI into `cblas_gemm`.
#
# If Miri stays clean, the matching CI job at `.github/workflows/ci.yaml`
# § miri promotes this from optional → required.

set -x
set -e

# Idempotent install of the miri component on nightly. No-op when already
# present (rustup short-circuits). Safe in CI fresh checkouts.
rustup component add miri --toolchain nightly >/dev/null 2>&1 || \
    rustup +nightly component add miri

# Layout randomisation — catches missing `#[repr(transparent)]` and similar
# layout-dependent UB. Cheap; always on.
export RUSTFLAGS="-Zrandomize-layout"

# Miri reports a stacked borrow violation deep within rayon's
# crossbeam-epoch. Upstream fix: crossbeam PR #871.
# Tree-borrow mode resolves it but trips a different rayon issue
# (rust-lang/miri#1371). Left disabled until both upstream stories close.
# export MIRIFLAGS="-Zmiri-tree-borrows"

# Miri adds random error to float operations such as exp by default. The
# suite's oracles compare bit-exact across two code paths (facade vs hpc home,
# tiled vs reference GEMM), which that error breaks: sigmoid_f32 differed by
# 3 ulp in tests/simd_facade_reexports.rs on the 2026-11-06 master run.
export MIRIFLAGS="-Zmiri-deterministic-floats"

# Architectural limit on the Miri sweep:
#
# `crate::simd::*` (the production dispatch in `src/simd.rs`) re-exports
# from `simd_avx512` / `simd_avx2` / `simd_neon`, which call `_mm*_*` /
# `vget*` intrinsics directly. Miri rejects those with "calling a
# function that requires unavailable target features: avx" because the
# Miri target doesn't enable AVX/AVX2/AVX-512/NEON target features.
#
# The `nightly-simd` feature ships a parallel module `crate::simd_nightly`
# (the 24-type `core::simd` polyfill, at full parity with the 24 types
# defined across `simd_avx2.rs` + `simd_avx512.rs` — landed in PR #146)
# which IS Miri-checkable. But the default `crate::simd::*` dispatch is
# NOT routed through it; consumer modules that import `crate::simd::F32x16`
# (most of `hpc::*` + the `simd::tests::*` suite) go through intrinsics.
# The polyfill is no longer the bottleneck — the missing piece is a
# `cfg(miri)` switch in `src/simd.rs` that re-exports from `simd_nightly`
# instead of `simd_avx*` under Miri.
# CI shards the run (MIRI_PARTITION=count:K/N, see ci.yaml § miri): under Miri
# each test is its own interpreted process, so the whole suite takes about
# 4-5 hours on one runner. Unset, the full suite runs in one go.
cargo +nightly miri nextest run -v --profile miri \
    --no-fail-fast \
    ${MIRI_PARTITION:+--partition "$MIRI_PARTITION"} \
    -p ndarray -p ndarray-rand \
    --features approx,serde,nightly-simd \
    -E '!(
            test(/^hpc::/) - test(/^hpc::byte_scan/)
        ) and !test(/^simd::tests::/)
          and !test(/^hpc::framebuffer::pyramid_tests::/)
          and !test(/^cpu_guard::/)
          and !test(/^simd_avx2::/)
          and !test(/^simd_amx::/)
          and !test(/^simd_int_ops::tests::gemm_u8_i8_/)
       '
#
# Tests that are correct but take hours under interpretation (the exhaustive
# 511 x 511 kernel sweeps, the flush-bound worst-case runs) carry
# `#[cfg_attr(miri, ignore = "...")]` in their own file, and spatial_hash's
# k-NN stops once every entry is visited. The `miri` nextest profile stops any
# test after 20 minutes, so a new slow test fails by name.
#
# Filter rationale (4-clause AND):
#
# 1. `!(test(/^hpc::/) - test(/^hpc::byte_scan/))`
#    Skip everything in `hpc::*` EXCEPT `hpc::byte_scan` (the scalar-fallback
#    path validated against the `cfg(miri)` SimdCaps bypass).
#
# 2. `!test(/^simd::tests::/)`
#    Skip the `simd::tests::*` suite. These exercise `crate::simd::F32x16`
#    etc. directly — types that re-export AVX/AVX2/AVX-512 intrinsics. Miri
#    rejects every one with "calling a function that requires unavailable
#    target features: avx". Same architectural class as `hpc::*`. Will
#    become miri-runnable when `crate::simd::*` gains a cfg(miri) dispatch
#    through `simd_nightly`.
#
# 3. `!test(/^hpc::framebuffer::pyramid_tests::/)`
#    The 3 pyramid tests take 19+ minutes EACH under Miri (large 2D scan
#    loops over SIMD-shaped data). Not a UB signal — pure runtime cost.
#    Re-enable once the test fixtures are sized down or the loops are
#    cfg(miri)-shortened.
#
# 4. `!test(/^cpu_guard::/)`
#    `cpu_guard` reads the CPU with `cpuid`, which is inline assembly; Miri
#    rejects it ("inline assembly is not supported"). The pre-`main` hook is
#    `cfg(not(miri))` for the same reason, so test binaries start under Miri.
#
# 5. `!test(/^simd_avx2::/)`, `!test(/^simd_amx::/)`,
#    `!test(/^simd_int_ops::tests::gemm_u8_i8_/)`
#    Same classes as 2 and 4: `simd_avx2` calls AVX intrinsics ("requires
#    unavailable target features: avx"), `simd_amx` and the u8 x i8 GEMM
#    dispatch read `cpuid` (inline assembly). They first showed up on the
#    2026-11-06 sharded run; the single unsharded run never got that far.
