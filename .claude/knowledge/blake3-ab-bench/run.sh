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
# The bench needs its source under $REPO/examples/ for cargo to see it, and
# removes it afterwards. Both halves must refuse to touch anything they did not
# create: a fixed destination plus an unconditional `rm` in the EXIT trap would
# clobber a pre-existing examples/blake3_ab.rs and then delete it.
EXAMPLE_DIR="$REPO/examples"
EXAMPLE="$EXAMPLE_DIR/blake3_ab.rs"
created_example_dir=0
if [ -e "$EXAMPLE" ] || [ -L "$EXAMPLE" ]; then
    echo "refusing to overwrite $EXAMPLE" >&2
    exit 1
fi
if [ ! -d "$EXAMPLE_DIR" ]; then
    mkdir "$EXAMPLE_DIR"
    created_example_dir=1
fi
# Only remove the directory if this script created it.
trap 'rm -f "$EXAMPLE"; [ "$created_example_dir" -eq 0 ] || rmdir "$EXAMPLE_DIR" 2>/dev/null || true' EXIT
cp "$HERE/blake3_ab.rs" "$EXAMPLE"
cd "$REPO"
cargo run --release --quiet --example blake3_ab
