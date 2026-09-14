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
#     ops that legitimately remain are loop control (`andq $-8` / `andl $3`
#     index rounding, `xorl` counter zeroing) plus the one scalar tail peel
#     for len % lanes; measured 9 on the v3 build 2026-09-14. A scalarised
#     body fails the "packed present" half, which is the discriminating one —
#     the cap only stops a body from quietly growing a second scalar loop.
SLICE_GPR_CAP=12
report() {
  awk -v vec="$1" -v sca="$2" '
    /^[A-Za-z_$][A-Za-z0-9_.$]*:/ { sym=$1 }
    sym ~ /probe_/ && $1 ~ vec { v[sym]++ }
    sym ~ /probe_/ && $1 ~ sca && $2 ~ /%[re][a-z0-9]+|^[wx][0-9]+,/ { s[sym]++ }
    END { for (k in v) printf "%6d vec %6d sca  %s\n", v[k], s[k]+0, k;
          for (k in s) if (!(k in v)) printf "%6d vec %6d sca  %s\n", 0, s[k], k }
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
