#!/usr/bin/env bash
# Read-only inspection of a built u64x64 probe binary (D-U64X64-LLVM-1, plan §4).
# Prints one row per kernel symbol (u64x64_o*_*): code bytes, instruction
# count, vector register use, and the instruction classes the plan's decision
# rule and FACET STRIDE checks need. Runs no cargo.
#
#   scripts/u64x64-inspect.sh <binary> [symbol-substring]
#
# Columns:
#   bytes   code size from nm -S
#   insn    instructions
#   zmm/ymm/xmm  instructions naming that register width
#   vdata   vector data ops (zmm/ymm/xmm, excluding plain moves)
#   sdata   scalar GPR ALU ops (add/sub/and/or/xor/shl/shr/imul/cmp/test/lea)
#   popcnt  popcnt / vpopcnt*
#   div     div / idiv
#   mul     vpmull* / vpmulhuw / imul / mulx
#   shuf    byte-shuffle family (vpshufb, vpermb, vpermi2b, vpermt2b) — a FACET R1 candidate
#   kmask   mask-register ops (k*)
#   ld/st   memory loads / stores (operand form)
#   bback   backward branches (loops)
#   call    calls (the plan requires 0 in a kernel)
set -euo pipefail

bin=${1:?usage: u64x64-inspect.sh <binary> [symbol-substring]}
filter=${2:-}
objdump=${LLVM_OBJDUMP:-llvm-objdump}

printf '%-26s %6s %5s %4s %4s %4s %5s %5s %6s %3s %3s %4s %5s %4s %4s %5s %4s\n' \
  symbol bytes insn zmm ymm xmm vdata sdata popcnt div mul shuf kmask ld st bback call

nm -S "$bin" | awk '$3 ~ /^[tT]$/ && $4 ~ /u64x64_o[0-9a]+_/ {print $2, $4}' |
while read -r size mangled; do
  short=$(sed -E 's/.*(u64x64_o[0-9a]+_[a-z0-9_]+)[0-9]{2,}.*/\1/' <<<"$mangled")
  short=${short%17h}
  [[ -n $filter && $short != *"$filter"* ]] && continue
  dis=$("$objdump" -d --no-show-raw-insn --disassemble-symbols="$mangled" "$bin" |
        awk -F'\t' 'NF >= 2 && $1 ~ /^ *[0-9a-f]+:/ {print $2 "\t" $3}')
  count() { grep -cE "$1" <<<"$dis" || true; }
  insn=$(grep -c . <<<"$dis" || true)
  zmm=$(count 'zmm'); ymm=$(count 'ymm'); xmm=$(count 'xmm')
  vdata=$(grep -E 'mm[0-9]' <<<"$dis" | grep -vcE '^v?mov' || true)
  sdata=$(grep -vE 'mm[0-9]' <<<"$dis" | grep -cE '^(add|sub|and|or|xor|shl|shr|sar|imul|cmp|test|lea|not|neg|inc|dec|bt|andn|shlx|shrx|blsr|tzcnt|lzcnt)' || true)
  popcnt=$(count 'popcnt'); div=$(count '^i?div')
  mul=$(count '(vpmull|vpmulhuw|^imul|mulx)'); shuf=$(count '(vpshufb|vpermb|vpermi2b|vpermt2b)')
  kmask=$(count '^k[a-z]+');
  # dis lines are "<mnemonic>\t<operands>" in AT&T order (destination last).
  # A store is a mov whose last operand is memory; any other memory operand
  # (lea and nop excluded) is a load.
  ld=$(awk -F'\t' '$1 !~ /^(lea|nop)/ && $2 ~ /\(/ {
          n2 = split($2, o, ", "); st = ($1 ~ /mov/ && o[n2] ~ /\(/);
          if (!st) n++ } END{print n+0}' <<<"$dis")
  st=$(awk -F'\t' '$1 ~ /mov/ && $2 ~ /\(/ { n2 = split($2, o, ", "); if (o[n2] ~ /\(/) n++ }
         END{print n+0}' <<<"$dis")
  bback=$("$objdump" -d --no-show-raw-insn --disassemble-symbols="$mangled" "$bin" |
          awk -F'\t' 'function hex(h,  i,c,v) { v=0; h=tolower(h); sub(/^0x/,"",h);
                         for (i=1;i<=length(h);i++) { c=index("0123456789abcdef",substr(h,i,1))-1; v=v*16+c } return v }
               $1 ~ /^ *[0-9a-f]+:/ && $2 ~ /^j/ { a=$1; gsub(/[ :]/,"",a); t=$3; sub(/ .*/,"",t);
                 if (t ~ /^0x[0-9a-f]+$/ && hex(t) <= hex(a)) n++ }
               END {print n+0}')
  call=$(count '^call')
  printf '%-26s %6d %5d %4d %4d %4d %5d %5d %6d %3d %3d %4d %5d %4d %4d %5d %4d\n' \
    "$short" "$((16#$size))" "$insn" "$zmm" "$ymm" "$xmm" "$vdata" "$sdata" "$popcnt" "$div" \
    "$mul" "$shuf" "$kmask" "$ld" "$st" "$bback" "$call"
done | sort
