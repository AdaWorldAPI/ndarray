#!/usr/bin/env bash
# Masking parity — build `crates/simd-masking-parity` (the ONE facade-only
# parity program) under a realization selector and run it. The program has no
# idea which backend `simd.rs` picked; this script only chooses the build.
#
#   scripts/masking-parity.sh native        host triple, whatever .cargo/config* selects
#   scripts/masking-parity.sh nightly       `cargo +nightly --features nightly-simd`
#   scripts/masking-parity.sh wasm          wasm32 +simd128 under node (simd_wasm arm)
#   scripts/masking-parity.sh wasm-scalar   wasm32 WITHOUT simd128 under node = the scalar arm
#   scripts/masking-parity.sh neon-qemu     aarch64 cross-build run under qemu-aarch64-static
#
# Extra cargo arguments (e.g. `--config .cargo/config-v4.toml` for the v4
# realization on the native row) pass through CARGO_ARGS, see
# scripts/codegen-witness.sh for why that form and not the env var.
set -euo pipefail
ARM="${1:?native|nightly|wasm|wasm-scalar|neon-qemu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/crates/simd-masking-parity/Cargo.toml"
TD="${CARGO_TARGET_DIR:-$ROOT/crates/simd-masking-parity/target}"
export CARGO_TARGET_DIR="$TD"
cd "$ROOT"

# shellcheck disable=SC2086
case "$ARM" in
  native)
    cargo ${CARGO_ARGS:-} build --release --manifest-path "$MANIFEST" --bin simd-masking-parity
    "$TD/release/simd-masking-parity"
    ;;
  nightly)
    cargo +nightly ${CARGO_ARGS:-} build --release --manifest-path "$MANIFEST" --bin simd-masking-parity --features nightly-simd
    "$TD/release/simd-masking-parity"
    ;;
  wasm|wasm-scalar)
    FLAGS=""; [ "$ARM" = wasm ] && FLAGS="-C target-feature=+simd128"
    RUSTFLAGS="$FLAGS" cargo ${CARGO_ARGS:-} build --release --lib --manifest-path "$MANIFEST" --target wasm32-unknown-unknown
    echo "==> $ARM (RUSTFLAGS='$FLAGS') under node"
    node "$ROOT/crates/simd-masking-parity/run.mjs" "$TD/wasm32-unknown-unknown/release/simd_masking_parity.wasm"
    ;;
  neon-qemu)
    QEMU="${QEMU_AARCH64:-qemu-aarch64-static}"
    SYSROOT="${AARCH64_SYSROOT:-/usr/aarch64-linux-gnu}"
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-aarch64-linux-gnu-gcc}" \
      cargo ${CARGO_ARGS:-} build --release --manifest-path "$MANIFEST" --bin simd-masking-parity --target aarch64-unknown-linux-gnu
    "$QEMU" -L "$SYSROOT" "$TD/aarch64-unknown-linux-gnu/release/simd-masking-parity"
    ;;
  *) echo "unknown arm: $ARM"; exit 2 ;;
esac
echo "masking parity ($ARM): PASS"
