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

# Baseline selection is EXACT-MATCH, never a family glob. A baseline records
# instruction counts measured on one triple at one microarchitecture level;
# another triple in the same family has different assembly syntax, a different
# ABI, and different symbol names, so comparing against it produces either a
# meaningless diff or a missing-symbol failure. An earlier version matched
# `x86_64-*`, which silently handed x86_64-pc-windows-msvc the Linux baseline.
#
# `baseline-x86_64-v3.toml` is named for the microarch level rather than the
# triple, so it gets one explicit exact-triple arm. Any other target needs its
# own `baseline-<triple>.toml`, exactly as README.md promises.
case "$TARGET" in
    x86_64-unknown-linux-gnu)
        BASELINE="$HERE/baseline-x86_64-v3.toml"
        CPU="x86-64-v3"
        ;;
    *)
        BASELINE="$HERE/baseline-$TARGET.toml"
        CPU=""
        ;;
esac
if [ ! -f "$BASELINE" ]; then
    echo "==> no baseline for $TARGET at $BASELINE" >&2
    echo "    A baseline is per-TRIPLE; there is deliberately no family fallback." >&2
    echo "    (run with --verbose and record the observed counts to make one)" >&2
    exit 90
fi

# `mktemp -d` and not "$TMPDIR/...-$$": a PID-derived name is predictable and
# `mkdir -p` happily accepts a path that already exists, so a local attacker
# could pre-create the directory (with symlinked children) and redirect the
# `cp` and heredoc writes below. mktemp creates atomically or fails.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/simd-codegen-oracle.XXXXXX")" || exit 91
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

# Seed the scratch crate with the REPOSITORY's lockfile. Two reasons, and the
# second is the one that matters for an instrument:
#
#   1. A fresh scratch package has no lock, so cargo goes to the crates.io
#      index before it will compile anything -- which fails outright in an
#      offline or cold-cache environment even though the repo has a committed
#      Cargo.lock sitting right there.
#   2. Resolving afresh means two oracle runs can measure two different
#      dependency graphs. A tool whose whole job is reproducible measurement
#      must not let its own inputs drift.
#
# The scratch package's dep closure is a subset of the repo's (one path dep on
# ndarray), so cargo amends the copied lock with the new root rather than
# re-resolving. `--offline` is attempted first and falls back to a normal
# build, so a cold cache still works.
if [ -f "$REPO/Cargo.lock" ]; then
    cp "$REPO/Cargo.lock" "$SCRATCH/Cargo.lock"
fi

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

# Try the locked/offline build first (see the Cargo.lock copy above); fall back
# to a network build only if the local cache cannot satisfy the graph.
# --target-dir is PINNED to the scratch tree. Without it an ambient
# CARGO_TARGET_DIR / CARGO_BUILD_TARGET_DIR / build.target-dir would place the
# artifacts somewhere else entirely, so the `find "$SCRATCH/target"` below
# would come up empty and the scratch cleanup would leave real output behind.
build() {
    # shellcheck disable=SC2086
    if [ "$TARGET" = "$(rustc -vV | sed -n 's/^host: //p')" ]; then
        cargo rustc "$@" --release --manifest-path "$SCRATCH/Cargo.toml" \
            --target-dir "$SCRATCH/target" -- \
            --emit asm -C debuginfo=0 $CPU_FLAG
    else
        cargo rustc "$@" --release --manifest-path "$SCRATCH/Cargo.toml" \
            --target "$TARGET" --target-dir "$SCRATCH/target" -- \
            --emit asm -C debuginfo=0 $CPU_FLAG
    fi
}

if [ -f "$SCRATCH/Cargo.lock" ] && build --offline 2>/dev/null; then
    echo "==> built offline against the repository lockfile"
    OFFLINE="--offline"
else
    echo "==> offline build unavailable; resolving online"
    OFFLINE=""
    build
fi

# EXECUTE the probes as well, on the host only. `--emit asm` replaces the
# default emit, so the build above links no binary and any runtime assertion
# inside a probe never runs. Some probes are self-checking -- notably
# `transpose_16x16_composed`, which verifies its shuffle network against a
# naive nested-loop transpose -- and a packed-but-WRONG network reported as a
# success would be precisely the error this tool exists to catch.
#
# Skipped when cross-compiling, where the host cannot run the artifact.
if [ "$TARGET" = "$(rustc -vV | sed -n 's/^host: //p')" ]; then
    echo "==> executing probes (runtime self-checks)"
    # shellcheck disable=SC2086
    if ! cargo run $OFFLINE --release --quiet --manifest-path "$SCRATCH/Cargo.toml" \
        --target-dir "$SCRATCH/target"; then
        echo "==> a probe's runtime self-check FAILED -- codegen results are moot" >&2
        exit 93
    fi
else
    echo "==> cross-compiling; runtime self-checks skipped"
fi

ASM="$(find "$SCRATCH/target" -name 'simd_codegen_oracle-*.s' 2>/dev/null | head -1)"
if [ -z "$ASM" ]; then
    echo "==> no emitted assembly found under $SCRATCH/target" >&2
    exit 92
fi

echo "==> analyzing $(basename "$ASM") against $(basename "$BASELINE")"
python3 "$HERE/analyze.py" "$ASM" "$BASELINE" "$@"
