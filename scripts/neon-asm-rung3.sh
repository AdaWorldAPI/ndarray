#!/usr/bin/env bash
# AArch64 acceptance ladder, rung 3: does the cross-compiled harness assembly
# SELECT NEON vector logical instructions for the mask lane, or scalarise?
#
# Rung 1 (compile) and rung 2 (qemu parity) can both pass on a body that
# LLVM scalarised — measured 2026-09-13 on the first generated NEON ternlog
# (a per-lane loop through to_array/from_array): 536 scalar vs 4 vector ops.
# This script is that inspection, made re-runnable and SYMMETRIC: the same
# mnemonic set is counted on vector registers (v*.16b / v*.8b) and on
# general-purpose registers (w*/x*), and every count is attributed to the
# symbol it appears in so "scaffolding" is a measurement, not a claim.
# Attribution is per FUNCTION symbol: LLVM's local `.LBB*` basic-block labels
# are deliberately not symbol boundaries — treating them as such fragmented a
# 684-vector-op ternlog body into hundreds of 4-op fragments and misreported
# the function as scalarised (measured 2026-09-14, first run of this script).
#
# Requires: `rustup target add aarch64-unknown-linux-gnu`. No linker or qemu
# needed — `--emit=asm` stops before the link step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/crates/neon-simd-parity/Cargo.toml"
TARGET="aarch64-unknown-linux-gnu"
TD="${CARGO_TARGET_DIR:-$ROOT/target-aarch64}"

echo "==> emitting aarch64 assembly of neon-simd-parity (link step skipped)"
CARGO_TARGET_DIR="$TD" cargo rustc --release --manifest-path "$MANIFEST" --target "$TARGET" \
  -- --emit=asm -C debuginfo=0 >/dev/null 2>&1 || true
ASM="$(ls -t "$TD/$TARGET/release/deps/"neon_simd_parity-*.s | head -1)"
[ -f "$ASM" ] || { echo "no .s produced"; exit 2; }
echo "asm: $ASM"

LOGIC='(and|orr|eor|bic|orn|eon|mvn|not|bif|bit|bsl)'
VEC=$(grep -cE "^\s+${LOGIC}\s+v[0-9]+\.(16b|8b)" "$ASM" || true)
SCA=$(grep -cE "^\s+${LOGIC}\s+[wx][0-9]+,"          "$ASM" || true)
echo "vector logical ops (v-regs): $VEC"
echo "scalar logical ops (w/x-regs): $SCA"

echo "==> per-symbol attribution (top 12 by vector count, then any symbol with scalar ops)"
awk -v logic="$LOGIC" '
  /^[A-Za-z_$][A-Za-z0-9_.$]*:/ { sym=$1 }   # NOT .L*: LLVM local (basic-block) labels stay inside their function
  $1 ~ "^"logic"$" && $2 ~ /^v[0-9]+\.(16b|8b)/ { v[sym]++ }
  $1 ~ "^"logic"$" && $2 ~ /^[wx][0-9]+,/       { s[sym]++ }
  END { for (k in v) printf "%6d vec %6d sca  %s\n", v[k], s[k]+0, k; for (k in s) if (!(k in v)) printf "%6d vec %6d sca  %s\n", 0, s[k], k }
' "$ASM" | sort -rn | head -12

# Gate: the ternlog / mask kernels must be vector-dominant. Symbols whose
# demangled name contains "ternlog" must carry more vector than scalar logic —
# EXCEPT the scalar reference oracles (`*reference*`), which are scalar by
# design: the parity check is SIMD-vs-scalar, so a vectorised oracle would
# compare the backend against itself.
echo "==> gate: ternlog symbols vector-dominant"
BAD=$(awk -v logic="$LOGIC" '
  /^[A-Za-z_$][A-Za-z0-9_.$]*:/ { sym=$1 }   # NOT .L*: LLVM local (basic-block) labels stay inside their function
  $1 ~ "^"logic"$" && $2 ~ /^v[0-9]+\.(16b|8b)/ { v[sym]++ }
  $1 ~ "^"logic"$" && $2 ~ /^[wx][0-9]+,/       { s[sym]++ }
  END { for (k in s) if (k ~ /ternlog/ && k !~ /reference/ && s[k] > v[k]+0) print k }
' "$ASM")
if [ -n "$BAD" ]; then echo "SCALARISED ternlog symbols:"; echo "$BAD"; exit 1; fi
echo "rung 3: PASS ($VEC vector / $SCA scalar logical ops)"
