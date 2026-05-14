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

# Run via nextest for stable test isolation under Miri. The `+nightly`
# prefix is the ephemeral switch — this command, and only this command,
# runs on nightly.
cargo +nightly miri nextest run -v \
    -p ndarray -p ndarray-rand \
    --features approx,serde,nightly-simd
