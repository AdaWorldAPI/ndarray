#!/usr/bin/env bash
# wasm SIMD parity gate — build the real ndarray::simd wasm types (via the
# `wasm-simd-parity` cdylib) with +simd128, then run the numeric selfcheck
# under node. Exercises BOTH:
#   - correctness/integration: the full ndarray lib compiles for wasm32+simd128
#     (the harness path-deps ndarray, so building it builds ndarray for wasm);
#   - parity: each wasm SIMD lane is bit-identical to scalar, run under node.
# The x86 `cargo test` suite never touches simd_wasm.rs, so this is the only
# automated guard for the wasm SIMD tier.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/crates/wasm-simd-parity/Cargo.toml"
TARGET="wasm32-unknown-unknown"

echo "==> building wasm-simd-parity (ndarray::simd wasm types) for $TARGET +simd128"
RUSTFLAGS="-C target-feature=+simd128" cargo build --release \
  --manifest-path "$MANIFEST" --target "$TARGET"

WASM="$ROOT/crates/wasm-simd-parity/target/$TARGET/release/wasm_simd_parity.wasm"
# Excluded crate → its own target dir under the crate; fall back to workspace target.
[ -f "$WASM" ] || WASM="$ROOT/target/$TARGET/release/wasm_simd_parity.wasm"

echo "==> running selfcheck under node: $WASM"
node "$ROOT/crates/wasm-simd-parity/run.mjs" "$WASM"
