#!/usr/bin/env bash
# Codegen witness — does the backend the build selected actually REALIZE the
# mask family in its own instructions? Builds `examples/ternlog_codegen_probe`
# under the optimized `ci-codegen` profile with `--emit=asm`, runs the probe's
# own self-check (a packed-but-wrong body must fail before its assembly is
# trusted), then asserts per expectation:
#
#   avx512  every ternlog probe symbol contains vpternlogq/vpternlogd
#   avx2    no vpternlog anywhere; every ternlog probe symbol carries packed
#           ymm logic and ZERO GPR logic on lane data
#   neon    every ternlog probe symbol carries vector logic on v*.16b and ZERO
#           GPR logic
#
# Usage: scripts/codegen-witness.sh <avx512|avx2|neon> [target-triple]
# The CPU comes from the caller, never from this script — it only checks what
# the chosen CPU produced. Pass cargo's own selector through CARGO_ARGS, e.g.
#   CARGO_ARGS='--config .cargo/config-v4.toml' scripts/codegen-witness.sh avx512
#
# Why `--config` and NOT `CARGO_TARGET_<TRIPLE>_RUSTFLAGS`: cargo JOINS every
# matching target.<triple>.rustflags and target.<cfg>.rustflags entry, and the
# last `-Ctarget-cpu` wins. Measured 2026-09-14 with `cargo -v`: the env var
# form passes `-Ctarget-cpu=x86-64-v4` and THEN `.cargo/config.toml`'s
# cfg-keyed `x86-64-v3`, so the build is v3 and this witness reports 0
# vpternlog on a "v4" build. `--config .cargo/config-v4.toml` is the same
# cfg key at higher precedence, so it is placed LAST and v4 wins. Plain
# RUSTFLAGS would win too, but it also reaches host build scripts, which is
# the SIGILL the CI job avoided by moving off it in the first place.
set -euo pipefail
EXPECT="${1:?expect: avx512|avx2|neon}"
TRIPLE="${2:-$(rustc -vV | sed -n 's/^host: //p')}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TD="${CARGO_TARGET_DIR:-$ROOT/target}"
OUT="$TD/$TRIPLE/ci-codegen/examples"

cd "$ROOT"
rm -f "$OUT"/ternlog_codegen_probe-*.s
# Deleting the old .s is only half of "never grade stale assembly": a cached
# build re-emits nothing, so the glob below would find no file at all (it did,
# on the first run after the rm landed). Touching the probe source forces the
# ONE crate that produces the .s to recompile; the library stays cached.
touch "$ROOT/examples/ternlog_codegen_probe.rs"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
# A cross target emits assembly only: linking would need the foreign linker,
# and the self-check is a semantic claim the parity arms already carry for
# that platform. Natively the binary is linked AND run, so a packed-but-wrong
# body fails before its assembly is trusted.
if [ "$TRIPLE" = "$HOST" ]; then EMIT="asm,link"; else EMIT="asm"; fi
echo "==> building ternlog_codegen_probe (profile ci-codegen, --emit=$EMIT) for $TRIPLE, expecting $EXPECT"
# shellcheck disable=SC2086  # CARGO_ARGS is deliberately word-split
cargo ${CARGO_ARGS:-} rustc --profile ci-codegen --example ternlog_codegen_probe --target "$TRIPLE" -- --emit="$EMIT" -C debuginfo=0
ASM="$(ls -t "$OUT"/ternlog_codegen_probe-*.s | head -1)"
[ -f "$ASM" ] || { echo "no .s produced under $OUT"; exit 2; }

if [ "$TRIPLE" = "$HOST" ]; then
  echo "==> running the probe's self-check natively"
  "$TD/$TRIPLE/ci-codegen/examples/ternlog_codegen_probe"
else
  echo "==> cross target; self-check skipped (semantic parity is a separate arm)"
fi

# Per-symbol attribution: function labels only (never LLVM's .L* basic-block
# labels — counting those fragments a body into 4-op stubs and misreports it
# as scalarised; measured on scripts/neon-asm-rung3.sh 2026-09-14).
#
# Two rules, because the probes have two shapes:
#   * register probes (probe_ternlog_u64x8 / _u32x16 / probe_andnot_u64x8) —
#     straight-line, no loop: packed logic present AND ZERO GPR logic.
#   * the slice probe (probe_mask_ternlog_slice) — a loop over 64 words:
#     packed logic present AND GPR logic bounded by SLICE_GPR_CAP. The GPR
#     ops that legitimately remain are index arithmetic on LENGTHS, never on
#     lane data: `andl $7` (tail length = len % lanes) and `andq $-64`-style
#     chunk-byte rounding from `as_chunks`. There is NO scalar tail peel any
#     more — `simd_masking_ops` zero-pads the tail into one register and
#     runs it through the same packed op (`pad_tail`), so a peel showing up
#     here is a regression. `xorl %r, %r` register clears are excluded from
#     the count. Measured 2026-09-14 with every operand field inspected:
#     4 on v3 and 2 on aarch64 (down from 9 / 16 — the 16 was LLVM fully
#     unrolling the old scalar tail into 7 × (and, orr) on aarch64, which is
#     exactly what the cap exists to catch and what the padding removed).
#     A scalarised body fails the "packed present" half, which is the
#     discriminating one — the cap only stops a body from quietly growing a
#     second scalar loop; 12 leaves headroom for LLVM's index-math choices
#     without admitting a 7-word unrolled peel.
SLICE_GPR_CAP=12
# The scalar rule inspects EVERY operand field ($2..$NF), not just the first:
# `andq $-8, %r10` has its register in $3 and `andq (%rdi,%rax,8), %r9` its
# register in $3 too, so a `$2`-only rule undercounted exactly the two forms
# the slice cap exists to bound (CodeRabbit, PR #306). Every required probe
# is always printed (the loops below fail on a missing row rather than
# skipping it), so a symbol the compiler folded away cannot pass by absence.
REQUIRED_PROBES="probe_ternlog_u64x8 probe_ternlog_u32x16 probe_andnot_u64x8 probe_mask_ternlog_slice"
report() {
  awk -v vec="$1" -v sca="$2" -v req="$REQUIRED_PROBES" '
    BEGIN { n = split(req, r, " "); for (i = 1; i <= n; i++) want[r[i]] = 1 }
    /^[A-Za-z_$][A-Za-z0-9_.$]*:/ { sym=$1; for (k in want) if (index(sym, k)) seen[k] = sym }
    sym ~ /probe_/ && $1 ~ vec { v[sym]++ }
    sym ~ /probe_/ && $1 ~ sca {
      # `xorl %eax, %eax` is the register-clear idiom, not logic on data:
      # identical source and destination means the value is discarded.
      a = $2; sub(/,$/, "", a)
      if ($1 ~ /^xor/ && NF == 3 && a == $3) next
      hit = 0
      for (f = 2; f <= NF; f++) if ($f ~ /%[re][a-z0-9]+|^[wx][0-9]+,?$/) hit = 1
      if (hit) s[sym]++
    }
    END {
      for (k in want) if (!(k in seen)) printf "%6d vec %6d sca  MISSING:%s\n", 0, 0, k
      for (k in seen) { sym = seen[k]; printf "%6d vec %6d sca  %s\n", v[sym]+0, s[sym]+0, sym }
    }
  ' "$ASM" | sort -k5
}

fail=0
case "$EXPECT" in
  avx512)
    echo "==> ternlog symbols must select vpternlog{q,d}"
    for sym in probe_ternlog_u64x8 probe_ternlog_u32x16 probe_mask_ternlog_slice; do
      n=$(awk -v s="$sym" '/^[A-Za-z_$][A-Za-z0-9_.$]*:/ { sym=$1 } sym ~ s && $1 ~ /^vpternlog[qd]$/ { c++ } END { print c+0 }' "$ASM")
      echo "   $sym: $n vpternlog"
      [ "$n" -ge 1 ] || { echo "   FAIL: $sym has no vpternlog on an AVX-512 build"; fail=1; }
    done
    ;;
  avx2)
    echo "==> AVX2 build must not contain vpternlog anywhere"
    n=$(grep -cE '^\s+vpternlog' "$ASM" || true)
    [ "$n" -eq 0 ] || { echo "   FAIL: $n vpternlog instructions on an AVX2 build (wrong arm selected)"; fail=1; }
    echo "==> per-symbol packed (ymm logic) vs GPR logic on lane data"
    report '^(vpand|vpandn|vpor|vpxor|vandps|vandnps|vorps|vxorps|vandpd|vandnpd|vorpd|vxorpd)$' '^(and[lq]?|or[lq]?|xor[lq]?|andn[lq]?|not[lq]?)$'
    while read -r v _ s _ sym; do
      case "$sym" in
        MISSING:*)
          echo "   FAIL: required probe $sym has no symbol in the assembly"; fail=1 ;;
        *probe_mask_ternlog_slice*)
          [ "$v" -ge 2 ] || { echo "   FAIL: $sym has no packed logic (facade layer scalarised)"; fail=1; }
          [ "$s" -le "$SLICE_GPR_CAP" ] || { echo "   FAIL: $sym carries $s GPR logic ops (cap $SLICE_GPR_CAP: loop control + one tail peel)"; fail=1; } ;;
        *probe_ternlog*|*probe_andnot*)
          [ "$v" -ge 1 ] || { echo "   FAIL: $sym has no packed logic"; fail=1; }
          [ "$s" -eq 0 ] || { echo "   FAIL: $sym carries $s GPR logic ops on lane data"; fail=1; } ;;
      esac
    done < <(report '^(vpand|vpandn|vpor|vpxor|vandps|vandnps|vorps|vxorps|vandpd|vandnpd|vorpd|vxorpd)$' '^(and[lq]?|or[lq]?|xor[lq]?|andn[lq]?|not[lq]?)$')
    ;;
  neon)
    echo "==> per-symbol vector (v*.16b) vs GPR logic"
    report '^(and|orr|eor|bic|orn|eon|mvn|not|bif|bit|bsl)$' '^(and|orr|eor|bic|orn|eon|mvn)$'
    while read -r v _ s _ sym; do
      case "$sym" in
        MISSING:*)
          echo "   FAIL: required probe $sym has no symbol in the assembly"; fail=1 ;;
        *probe_mask_ternlog_slice*)
          [ "$v" -ge 2 ] || { echo "   FAIL: $sym has no vector logic (facade layer scalarised)"; fail=1; }
          [ "$s" -le "$SLICE_GPR_CAP" ] || { echo "   FAIL: $sym carries $s GPR logic ops (cap $SLICE_GPR_CAP: loop control + one tail peel)"; fail=1; } ;;
        *probe_ternlog*|*probe_andnot*)
          [ "$v" -ge 1 ] || { echo "   FAIL: $sym has no vector logic"; fail=1; }
          [ "$s" -eq 0 ] || { echo "   FAIL: $sym carries $s GPR logic ops"; fail=1; } ;;
      esac
    done < <(report '^(and|orr|eor|bic|orn|eon|mvn|not|bif|bit|bsl)$' '^(and|orr|eor|bic|orn|eon|mvn)$')
    ;;
  *) echo "unknown expectation: $EXPECT"; exit 2 ;;
esac
[ "$fail" -eq 0 ] && echo "codegen witness ($EXPECT): PASS" || { echo "codegen witness ($EXPECT): FAIL"; exit 1; }
