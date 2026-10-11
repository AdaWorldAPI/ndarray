#!/usr/bin/env bash
# cpu_guard in a library host, two-sided (coresearch 2026-10-11, X1).
#
# Builds, for x86-64-v4, (a) a cdylib that links ndarray and is dlopen'ed by
# a small C program, and (b) ndarray's own `cpu_guard_probe` executable.
# Both run under `qemu-x86_64-static -cpu Haswell` (AVX2, no AVX-512):
#
#   * the library must NOT end its host: the C program prints `survived`
#     and exits 0, and stderr carries the guard's `warning:` line;
#   * the executable must still exit 132 with the guard's `error:` line.
#
# Before the library rule existed, (a) ended the host with status 132.
# Requires: cc, qemu-x86_64-static. Every compile uses debug 0 and ignores
# any ambient RUSTFLAGS (which would REPLACE the pinned v4 config).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU="${QEMU:-qemu-x86_64-static}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
CFG="$ROOT/.cargo/config-v4.toml"

mkdir -p "$WORK/lib/src"
cp "$ROOT/rust-toolchain.toml" "$WORK/lib/"
cat > "$WORK/lib/Cargo.toml" <<TOML
[package]
name = "cpu_guard_library_probe"
version = "0.0.0"
edition = "2021"
publish = false
[lib]
crate-type = ["cdylib"]
[dependencies]
ndarray = { path = "$ROOT", default-features = false, features = ["std"] }
[workspace]
TOML
cat > "$WORK/lib/src/lib.rs" <<'RS'
#[no_mangle]
pub extern "C" fn probe_xor(a: u64, b: u64) -> u64 {
    let x = ndarray::simd::U64x8::splat(a) ^ ndarray::simd::U64x8::splat(b);
    x.to_array()[0]
}
RS
cat > "$WORK/host.c" <<'C'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 2; }
    printf("survived\n");
    return 0;
}
C
cc -O1 -o "$WORK/host" "$WORK/host.c" -ldl

(cd "$WORK/lib" && CARGO_TARGET_DIR="$WORK/target" env -u RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
    cargo --config "$CFG" build --release --quiet)
(cd "$ROOT" && CARGO_TARGET_DIR="$WORK/target-exe" env -u RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
    cargo --config "$CFG" build --release --quiet --example cpu_guard_probe)

fail=0

set +e
lib_out=$("$QEMU" -cpu Haswell "$WORK/host" "$WORK/target/release/libcpu_guard_library_probe.so" 2>&1)
lib_rc=$?
exe_out=$("$QEMU" -cpu Haswell "$WORK/target-exe/release/examples/cpu_guard_probe" 2>&1)
exe_rc=$?
set -e

echo "--- library host: exit $lib_rc"; echo "$lib_out" | grep -v 'TCG doesn' || true
echo "--- executable:   exit $exe_rc"; echo "$exe_out" | grep -v 'TCG doesn' || true

[ "$lib_rc" -eq 0 ] || { echo "FAIL: the library ended its host (exit $lib_rc)"; fail=1; }
echo "$lib_out" | grep -q '^survived$' || { echo "FAIL: host did not reach 'survived'"; fail=1; }
echo "$lib_out" | grep -q '^warning: this binary needs CPU features' || { echo "FAIL: no guard warning from the library"; fail=1; }
[ "$exe_rc" -eq 132 ] || { echo "FAIL: the v4 executable exited $exe_rc, expected 132"; fail=1; }
echo "$exe_out" | grep -q '^error: this binary needs CPU features' || { echo "FAIL: no guard error from the executable"; fail=1; }

[ "$fail" -eq 0 ] && echo "cpu_guard library probe: PASS"
exit "$fail"
