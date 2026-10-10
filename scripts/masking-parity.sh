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
#   scripts/masking-parity.sh avx-qemu      AVX-without-AVX2 build (config-avx) under qemu -cpu SandyBridge
#   scripts/masking-parity.sh v2-qemu       x86-64-v2 build (config-v2, scalar realization) under qemu -cpu Nehalem
#
# Extra cargo arguments (e.g. `--config .cargo/config-v4.toml` for the v4
# realization on the native row) pass through CARGO_ARGS, see
# scripts/codegen-witness.sh for why that form and not the env var.
set -euo pipefail
ARM="${1:?native|nightly|wasm|wasm-scalar|neon-qemu|avx-qemu|v2-qemu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/crates/simd-masking-parity/Cargo.toml"
TD="${CARGO_TARGET_DIR:-$ROOT/crates/simd-masking-parity/target}"
export CARGO_TARGET_DIR="$TD"
cd "$ROOT"

# shellcheck disable=SC2086
case "$ARM" in
  native)
    # `env -u RUSTFLAGS`: a workflow-global RUSTFLAGS (CI sets "-D warnings")
    # REPLACES every cargo-config rustflags entry, so a `--config
    # .cargo/config-v4.toml` passed through CARGO_ARGS would silently lose its
    # `-Ctarget-cpu=x86-64-v4` — the exact trap the tier4 CI job hit. Clearing
    # it lets the config win.
    #
    # What you get INSTEAD is not v3 (corrected 2026-09-16, coderabbit on #314;
    # this comment said "would measure v3"). RUSTFLAGS replaces EVERY entry,
    # including the DEFAULT `.cargo/config.toml`'s own `-Ctarget-cpu`, so no
    # target-cpu reaches rustc at all and the build lands on rustc's generic
    # `x86-64` baseline — SSE2, BELOW v3, and the tier `simd_avx2.rs`'s
    # intrinsics SIGILL on. Measured on one unit: `RUSTFLAGS="-D warnings"`
    # emitted ZERO `-Ctarget-cpu` flags against 65 with the env unset.
    #
    # This arm NAMES NO TIER by design: it builds with whatever config wins,
    # which by default is `target-cpu=native` (the host). Read the program's
    # own `avx512f=` header line for the tier; pin `.cargo/config-v3.toml`
    # through CARGO_ARGS when you specifically want AVX2.
    env -u RUSTFLAGS cargo ${CARGO_ARGS:-} build --release --manifest-path "$MANIFEST" --bin simd-masking-parity
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
  avx-qemu)
    # The `simd_avx.rs` arm: AVX but no AVX2. Built for Sandy Bridge and run on
    # an emulated Sandy Bridge, so an AVX2 instruction anywhere in the binary
    # faults (SIGILL) instead of silently passing on an AVX2 host.
    QEMU="${QEMU_X86_64:-qemu-x86_64-static}"
    env -u RUSTFLAGS cargo ${CARGO_ARGS:-} --config "$ROOT/.cargo/config-avx.toml" build --release --manifest-path "$MANIFEST" --bin simd-masking-parity
    "$QEMU" -cpu SandyBridge "$TD/release/simd-masking-parity"
    ;;
  v2-qemu)
    # The x86-64-v2 arm: SSE4.2, no AVX, routed to the scalar realization.
    # Built for x86-64-v2 and run on an emulated Nehalem, so an AVX
    # instruction anywhere in the binary faults instead of passing silently.
    QEMU="${QEMU_X86_64:-qemu-x86_64-static}"
    env -u RUSTFLAGS cargo ${CARGO_ARGS:-} --config "$ROOT/.cargo/config-v2.toml" build --release --manifest-path "$MANIFEST" --bin simd-masking-parity
    "$QEMU" -cpu Nehalem "$TD/release/simd-masking-parity"
    ;;
  *) echo "unknown arm: $ARM"; exit 2 ;;
esac
echo "masking parity ($ARM): PASS"
