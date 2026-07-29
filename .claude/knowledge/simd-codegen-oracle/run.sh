#!/bin/sh
# SIMD codegen oracle -- measures what actually vectorizes, in both directions:
# Group A probes must show packed SIMD, Group B probes must show none, Group C
# probes are reported without a pass/fail verdict.
#
# This is an ON-DEMAND instrument, not a CI job. Run it when a codegen question
# is genuinely open, record the answer in a doc, and stop. See README.md.
#
# Self-contained: builds a throwaway crate from probes.rs against the ndarray
# checkout this file lives in, so nothing needs to exist under crates/.
#
# Usage: sh run.sh [target-triple] [--verbose]
#   target-triple defaults to the host triple.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# .claude/knowledge/simd-codegen-oracle -> repo root is three levels up.
REPO="$(cd "$HERE/../../.." && pwd)"
if [ ! -f "$REPO/Cargo.toml" ]; then
    echo "==> cannot locate the ndarray repo root from $HERE" >&2
    exit 91
fi

TARGET="${1:-}"
case "$TARGET" in
    "" | -*) TARGET="$(rustc -vV | sed -n 's/^host: //p')" ;;
    *)       shift ;;
esac

case "$TARGET" in
    x86_64-*) BASELINE="$HERE/baseline-x86_64-v3.toml"; CPU="x86-64-v3" ;;
    *)        BASELINE="$HERE/baseline-$TARGET.toml";   CPU="" ;;
esac
if [ ! -f "$BASELINE" ]; then
    echo "==> no baseline for $TARGET at $BASELINE" >&2
    echo "    (run with --verbose and record the observed counts to make one)" >&2
    exit 90
fi

SCRATCH="${TMPDIR:-/tmp}/simd-codegen-oracle-$$"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/src"
cp "$HERE/probes.rs" "$SCRATCH/src/main.rs"
cat > "$SCRATCH/Cargo.toml" <<EOF
[package]
name = "simd-codegen-oracle"
version = "0.0.0"
edition = "2021"

[dependencies]
ndarray = { path = "$REPO", default-features = false, features = ["std"] }

[profile.release]
debug = false
EOF

# The measured baseline is DECLARED, never inherited. Two independent reasons,
# and the first alone is NOT sufficient:
#
#   1. cargo resolves .cargo/config.toml from the CURRENT WORKING DIRECTORY,
#      not from --manifest-path. Running from elsewhere silently misses the
#      repo's config. Hence the `cd "$REPO"` below.
#   2. Even from the right directory, cargo's RUSTFLAGS env var REPLACES
#      `[target.'cfg(...)'].rustflags` rather than merging with it. CI sets
#      RUSTFLAGS="-D warnings" at workflow level, which drops the target-cpu
#      pin entirely:
#        $ cargo build -v | grep target-cpu                  -> x86-64-v3
#        $ RUSTFLAGS="-D warnings" cargo build -v | grep ...  -> (nothing)
#
# Only passing `-C target-cpu` on the final rustc invocation survives both.
# An oracle that inherits its baseline measures a different machine depending
# on where it runs -- exactly the class of error it exists to catch.
if [ -n "$CPU" ]; then
    CPU_FLAG="-C target-cpu=$CPU"
    echo "==> baseline: $TARGET @ target-cpu=$CPU (declared, not inherited)"
else
    CPU_FLAG=""
    echo "==> baseline: $TARGET @ target default (no target-cpu override)"
fi

cd "$REPO"
echo "==> building probes (--emit asm) for $TARGET"
if [ "$TARGET" = "$(rustc -vV | sed -n 's/^host: //p')" ]; then
    # shellcheck disable=SC2086
    cargo rustc --release --manifest-path "$SCRATCH/Cargo.toml" -- \
        --emit asm -C debuginfo=0 $CPU_FLAG
else
    # shellcheck disable=SC2086
    cargo rustc --release --manifest-path "$SCRATCH/Cargo.toml" --target "$TARGET" -- \
        --emit asm -C debuginfo=0 $CPU_FLAG
fi

ASM="$(find "$SCRATCH/target" -name 'simd_codegen_oracle-*.s' 2>/dev/null | head -1)"
if [ -z "$ASM" ]; then
    echo "==> no emitted assembly found under $SCRATCH/target" >&2
    exit 92
fi

echo "==> analyzing $(basename "$ASM") against $(basename "$BASELINE")"
python3 "$HERE/analyze.py" "$ASM" "$BASELINE" "$@"
