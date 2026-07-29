#!/bin/sh
# A/B: ndarray's in-tree BLAKE3 vs the external `blake3` crate.
#
# On-demand instrument, same posture as ../simd-codegen-oracle: run it when the
# question is open, record the answer, stop. NOT a CI job.
#
# Requires the external `blake3` crate to still be a dependency -- once it is
# dropped, this bench can no longer build, which is by design: at that point
# there is nothing left to compare against.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
mkdir -p "$REPO/examples"
trap 'rm -f "$REPO/examples/blake3_ab.rs"; rmdir "$REPO/examples" 2>/dev/null || true' EXIT
cp "$HERE/blake3_ab.rs" "$REPO/examples/blake3_ab.rs"
cd "$REPO"
cargo run --release --quiet --example blake3_ab
