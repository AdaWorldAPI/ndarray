#!/bin/sh
# SIMD codegen oracle -- proves what actually vectorizes (crates/simd-codegen-oracle),
# in both directions: Group A probes must show packed AVX2 instructions, Group B
# probes must show none, Group C probes are reported without a pass/fail verdict
# (the open question this oracle was extended to answer). Mirrors the build/locate
# shape of scripts/neon-parity.sh / scripts/wasm-parity.sh, but the analysis itself
# (instruction classification, baseline comparison) lives in the Python helper
# scripts/codegen_oracle_analyze.py -- see its module docstring for the exact
# packed-vector / scalar-lane-arith / loop-control / memory / other classification
# rules and the documented honesty rule (a loop-counter decl is not lane
# arithmetic).
#
# Usage: scripts/codegen-oracle.sh [target-triple] [-- --verbose]
#   target-triple defaults to the host triple (`rustc -vV | grep ^host`).
#   Everything after the target is forwarded to the analyzer (e.g. --verbose
#   to print the raw instruction list per bucket).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/crates/simd-codegen-oracle/Cargo.toml"

TARGET="${1:-}"
case "$TARGET" in
    "" | -*)
        TARGET="$(rustc -vV | sed -n 's/^host: //p')"
        ;;
    *)
        shift
        ;;
esac
# Remaining args (an optional `--` and/or analyzer flags like --verbose) forward as-is.
if [ "${1:-}" = "--" ]; then
    shift
fi

BASELINE="$ROOT/crates/simd-codegen-oracle/baselines/$TARGET.toml"
if [ ! -f "$BASELINE" ]; then
    echo "==> no baseline for target $TARGET at $BASELINE" >&2
    exit 90
fi

echo "==> building simd-codegen-oracle (--emit asm) for $TARGET"
if [ "$TARGET" = "$(rustc -vV | sed -n 's/^host: //p')" ]; then
    cargo rustc --release --manifest-path "$MANIFEST" -- --emit asm -C debuginfo=0
else
    cargo rustc --release --manifest-path "$MANIFEST" --target "$TARGET" -- --emit asm -C debuginfo=0
fi

# Excluded crate -> cargo places `deps/` either under the crate's own target
# dir or falls back to the workspace target dir, exactly like neon/wasm-parity.
ASM=""
for CAND_ROOT in "$ROOT/crates/simd-codegen-oracle/target" "$ROOT/target"; do
    if [ "$TARGET" = "$(rustc -vV | sed -n 's/^host: //p')" ]; then
        CAND_DIR="$CAND_ROOT/release/deps"
    else
        CAND_DIR="$CAND_ROOT/$TARGET/release/deps"
    fi
    if [ -d "$CAND_DIR" ]; then
        FOUND="$(ls -t "$CAND_DIR"/simd_codegen_oracle-*.s 2>/dev/null | head -n1 || true)"
        if [ -n "$FOUND" ]; then
            ASM="$FOUND"
            break
        fi
    fi
done

if [ -z "$ASM" ]; then
    echo "==> could not locate emitted simd_codegen_oracle-*.s under $ROOT" >&2
    exit 91
fi

echo "==> analyzing $ASM against $BASELINE"
python3 "$ROOT/scripts/codegen_oracle_analyze.py" "$ASM" "$BASELINE" "$@"
