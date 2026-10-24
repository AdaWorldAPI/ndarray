#!/usr/bin/env bash
# Gate G5 of .claude/plans/quantized-codec8-v2.md: compare the assembly of each generic
# codec8 instantiation in examples/codec8_asm_probe.rs with its hand-written twin.
# Usage: scripts/codec8_asm_diff.sh [cargo --config file], e.g. .cargo/config-v4.toml
set -euo pipefail
cfg=()
[ $# -ge 1 ] && cfg=(--config "$1")
env -u RUSTFLAGS cargo "${cfg[@]}" rustc --release --example codec8_asm_probe -- --emit asm -C codegen-units=1 >/dev/null
asm=$(ls -t target/release/examples/codec8_asm_probe-*.s | head -1)
cut() { awk -v f="$1" '$0 ~ "^"f":" {p=1} p {print} p && /\.cfi_endproc/ {exit}' "$asm" \
  | sed -E 's/\.(LBB|LCPI|Ltmp|Lfunc_end)[0-9_]+/.L/g; s/\<'"$1"'\>/FN/g'; }
status=0
for pair in "g_lane_even d_lane_even" "g_tflite_act d_tflite_act" "g_onnx d_tflite_act"; do
  set -- $pair
  a=$(cut "$1"); b=$(cut "$2")
  [ -n "$a" ] && [ -n "$b" ] || { echo "MISSING $1 or $2"; status=1; continue; }
  if [ "$a" = "$b" ]; then r=IDENTICAL; else r=DIFFERS; fi
  echo "$1 vs $2: $r ($(wc -l <<<"$a") / $(wc -l <<<"$b") lines)"
done
exit $status
