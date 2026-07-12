#!/usr/bin/env bash
# NEON SIMD parity gate — cross-build the real ndarray::simd aarch64 types (via
# the `neon-simd-parity` bin) for aarch64-unknown-linux-gnu, then run the numeric
# selfcheck under qemu-aarch64. Exercises BOTH:
#   - correctness/integration: the full ndarray lib compiles for aarch64 on stable
#     (the harness path-deps ndarray, so building it builds ndarray for aarch64);
#   - parity: each NEON SIMD lane (U32x16 ARX / F32x16 / I8x16) is bit-identical
#     to scalar, run under qemu.
# The x86 `cargo test` suite never touches simd_neon.rs, so this is the only
# automated guard for the NEON SIMD tier (the twin of scripts/wasm-parity.sh).
#
# Requirements (Ubuntu CI): gcc-aarch64-linux-gnu (cross linker + sysroot),
# qemu-user-static (qemu-aarch64-static), and `rustup target add
# aarch64-unknown-linux-gnu`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/crates/neon-simd-parity/Cargo.toml"
TARGET="aarch64-unknown-linux-gnu"
QEMU="${QEMU_AARCH64:-qemu-aarch64-static}"
SYSROOT="${AARCH64_SYSROOT:-/usr/aarch64-linux-gnu}"

echo "==> building neon-simd-parity (ndarray::simd aarch64 types) for $TARGET"
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-linux-gnu-gcc" \
  cargo build --release --manifest-path "$MANIFEST" --target "$TARGET"

BIN="$ROOT/crates/neon-simd-parity/target/$TARGET/release/neon-simd-parity"
# Excluded crate → its own target dir under the crate; fall back to workspace target.
[ -f "$BIN" ] || BIN="$ROOT/target/$TARGET/release/neon-simd-parity"

echo "==> running selfcheck under $QEMU (sysroot $SYSROOT): $BIN"
"$QEMU" -L "$SYSROOT" "$BIN"
echo "==> NEON SIMD parity: PASS"
