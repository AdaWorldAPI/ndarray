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

# Architectural limit on the Miri sweep:
#
# `crate::simd::*` (the production dispatch in `src/simd.rs`) re-exports
# from `simd_avx512` / `simd_avx2` / `simd_neon`, which call `_mm*_*` /
# `vget*` intrinsics directly. Miri rejects those with "calling a
# function that requires unavailable target features: avx" because the
# Miri target doesn't enable AVX/AVX2/AVX-512/NEON target features.
#
# The `nightly-simd` feature ships a parallel module `crate::simd_nightly`
# (the 30-type `core::simd` polyfill) which IS Miri-checkable, but the
# default `crate::simd::*` dispatch is NOT routed through it — that's a
# deliberate decision noted in `src/simd.rs:212`. Consumer modules that
# import `crate::simd::F32x16` (most of `hpc::*`) therefore go through
# intrinsics and abort under Miri.
#
# Until the polyfill expands to full type-parity (5/30 today) AND
# `crate::simd::*` gains a `cfg(miri)` dispatch through it, we constrain
# this sweep to:
#   * `simd_nightly::tests::*`  — the polyfill itself, full coverage.
#   * `hpc::byte_scan::*`       — already exercises scalar fallback paths
#                                  through `simd_caps()` cfg(miri) bypass.
#   * everything else ndarray ships that does NOT pull `_mm*` intrinsics.
#
# `--no-fail-fast` so a single regression doesn't mask the rest. CI
# nextest profile `default-miri` already configures isolated retry-on-flake.
cargo +nightly miri nextest run -v \
    --no-fail-fast \
    -p ndarray -p ndarray-rand \
    --features approx,serde,nightly-simd \
    -E '!(
            test(/^hpc::/) - test(/^hpc::byte_scan/)
        ) and !test(/^simd::tests::/)
          and !test(/^hpc::framebuffer::pyramid_tests::/)
       '
#
# Filter rationale (3-clause AND):
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
