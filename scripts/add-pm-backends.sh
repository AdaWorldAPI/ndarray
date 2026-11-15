#!/usr/bin/env bash
# Runs tests/add_pm.rs (the Codec9 P x M add/sub probe) on one SIMD tier, so
# the exhaustive codeword oracle and the mandatory sign cases are checked on
# that backend, not only on the host tier.
#
#   scripts/add-pm-backends.sh avx2       x86-64-v3 build, native run
#   scripts/add-pm-backends.sh avx-qemu   sandybridge build, qemu -cpu SandyBridge
#   scripts/add-pm-backends.sh v2-qemu    x86-64-v2 build (scalar backend), qemu -cpu Nehalem
#   scripts/add-pm-backends.sh neon-qemu  aarch64 cross build, qemu-aarch64
#   scripts/add-pm-backends.sh wasm         wasm32-wasip1 +simd128 (simd_wasm), node WASI
#   scripts/add-pm-backends.sh wasm-scalar  wasm32-wasip1 without simd128 (scalar), node WASI
#
# A run that executes zero tests is a failure: the file is gated on `std`.
set -euo pipefail
ARM="${1:?avx2|avx-qemu|v2-qemu|neon-qemu|wasm|wasm-scalar}"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
build() { # $1 = cargo config file or "", rest = test args; prints the test binary
  local cfg=(); [ -n "$1" ] && cfg=(--config "$1"); shift
  env -u RUSTFLAGS cargo "${cfg[@]}" test --release --features std --test add_pm --no-run --message-format=json "$@" \
    | jq -r 'select(.executable != null and .target.name == "add_pm") | .executable'
}
case "$ARM" in
  avx2) bin=$(build .cargo/config-v3.toml); run=("$bin") ;;
  avx-qemu) bin=$(build .cargo/config-avx.toml); run=(qemu-x86_64-static -cpu SandyBridge "$bin") ;;
  v2-qemu) bin=$(build .cargo/config-v2.toml); run=(qemu-x86_64-static -cpu Nehalem "$bin") ;;
  neon-qemu)
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    bin=$(build "" --target aarch64-unknown-linux-gnu)
    run=(qemu-aarch64-static -L /usr/aarch64-linux-gnu "$bin") ;;
  wasm|wasm-scalar)
    # RUSTFLAGS selects the realization here, as in masking-parity.sh: with
    # +simd128 simd.rs picks simd_wasm, without it the scalar backend.
    flags=""; [ "$ARM" = wasm ] && flags="-C target-feature=+simd128"
    bin=$(RUSTFLAGS="$flags" cargo test --release --features std --test add_pm --no-run \
      --target wasm32-wasip1 --message-format=json \
      | jq -r 'select(.executable != null and .target.name == "add_pm") | .executable')
    run=(node scripts/wasi-test-run.mjs "$bin") ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
[ -n "$bin" ] || { echo "::error::no add_pm test binary built" >&2; exit 1; }
out=$("${run[@]}" --test-threads=1)
echo "$out"
n=$(echo "$out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p')
[ "${n:-0}" -gt 0 ] || { echo "::error::add_pm ran zero tests on $ARM" >&2; exit 1; }
