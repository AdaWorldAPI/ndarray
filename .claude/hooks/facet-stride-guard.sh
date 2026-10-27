#!/usr/bin/env bash
# facet-stride-guard.sh — catches violations of the FACET STRIDE shape.
#
#   '{0}..{15}' -f $128.split()
#
# A 128-bit value IS sixteen u8 positions. Reading position k costs the one
# load (mov) that brought the 128 bits in; nothing is moved to read it. A byte
# shuffle only MOVES bytes between positions. Using shuffles to compute a
# result (a lookup assembled from 16 shuffles and masks, a fold across rows,
# a reduction) where the positions can simply be read is the violation.
#
# Rules, applied to ADDED lines only (existing code is audited with --all):
#   R1 SHUFFLE-AS-COMPUTE   byte shuffle / permute / swizzle / table-lookup
#                           instruction in new code.
#   R2 SCALAR-IN-SIMD       a per-lane scalar loop in a non-scalar SIMD
#                           backend (simd_avx2/avx512/avx/neon/wasm/amx,
#                           simd_nightly/). Scalar belongs in simd_scalar.rs
#                           only; on AVX2 the split (2 x U8x32, 128-bit
#                           halves) is the realization.
#   R3 U8-I8-MIX            u8 has no sign: `as i8` in a SIMD file, or a line
#                           that converts between u8 and i8.
#
# Escape hatch, on the line itself or the line above, with a reason:
#   // facet-stride: ok <why this line is a genuine byte MOVE / scalar backend / not a mix>
# A marker without a reason does not count.
#
# Modes:
#   --hook            PostToolUse hook: reads the tool JSON on stdin, checks the
#                     edited file's added lines against HEAD. Exit 2 = blocked.
#   --diff <rev>      check lines added in <rev>..HEAD plus the working tree.
#   --all             audit every line of the tree (report; exit 1 if any hit).
#   --file <path>...  check the whole of the given files.
set -uo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 0

in_scope() {            # rust sources only; test files may use scalar oracles by design
  case "$1" in
    */tests.rs|*/tests/*|tests/*) return 1 ;;
    src/*.rs|src/*/*.rs|src/*/*/*.rs|crates/*.rs|crates/*/*.rs|crates/*/*/*.rs|crates/*/*/*/*.rs|examples/*.rs) return 0 ;;
    *) return 1 ;;
  esac
}
simd_backend() {        # non-scalar SIMD realizations
  case "$1" in
    src/simd_avx2.rs|src/simd_avx512.rs|src/simd_avx.rs|src/simd_neon.rs|src/simd_wasm.rs|src/simd_amx.rs|src/simd_nightly/*.rs) return 0 ;;
    *) return 1 ;;
  esac
}
simd_file() {
  case "$1" in src/simd*.rs|src/simd*/*.rs) return 0 ;; *) return 1 ;; esac
}

R1='\.shuffle_bytes\(|\.permute_bytes\(|_shuffle_epi8|permutexvar_epi8|permutex2var_epi8|swizzle_dyn|vqtbl[1-4]q?|vtbl[1-4]|i8x16_swizzle|u8x16_swizzle'
R2='for +[a-z_]+ +in +0 *\.\. *(8|16|32|64)\b|for +lane +in|for +\(?[a-z_, &]*\)? +in +[a-z_.]*\.iter(_mut)?\(\)\.zip\('
R3='as +i8\b|i8::from\(|u8::from\([^)]*i8|as +u8\b.*\bi8\b|\bi8\b.*as +u8\b'

HITS_FILE="$(mktemp)"; trap 'rm -f "$HITS_FILE"' EXIT

# check_lines FILE  <stdin: "LINENO<TAB>TEXT" for each line to check>
# One python pass per file (a grep per line made --all take minutes).
check_lines() {
  R1="$R1" R2="$R2" R3="$R3" BACKEND="$(simd_backend "$1" && echo 1)" SIMD="$(simd_file "$1" && echo 1)" \
  python3 -c '
import os, re, sys
f = sys.argv[1]
r1, r2, r3 = (re.compile(os.environ[k]) for k in ("R1", "R2", "R3"))
backend, simd = os.environ.get("BACKEND") == "1", os.environ.get("SIMD") == "1"
ok = re.compile(r"facet-stride: ok \S")
comment = re.compile(r"^\s*//")
prev, hits = "", 0
for raw in sys.stdin:
    n, _, t = raw.rstrip("\n").partition("\t")
    waived = ok.search(prev) or ok.search(t)
    prev = t
    if waived or comment.match(t):
        continue
    for rule, rx, on in (("R1 SHUFFLE-AS-COMPUTE", r1, True), ("R2 SCALAR-IN-SIMD", r2, backend), ("R3 U8-I8-MIX", r3, simd)):
        if on and rx.search(t):
            hits += 1
            print(f"{f}:{n}  {rule}  {t.strip()[:140]}", file=sys.stderr)
print(hits)
' "$1" >> "$HITS_FILE"
}

# added lines of FILE relative to REV (working tree included); untracked = whole file
added_lines() {
  local f="$1" rev="$2"
  if ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    awk '{printf "%d\t%s\n", NR, $0}' "$f"; return
  fi
  git diff -U0 "$rev" -- "$f" | awk '
    /^@@/ { split($3, a, ","); ln = substr(a[1], 2) + 0; next }
    /^\+\+\+/ { next }
    /^\+/ { printf "%d\t%s\n", ln, substr($0, 2); ln++ ; next }'
}

whole_file() { awk '{printf "%d\t%s\n", NR, $0}' "$1"; }

# Skip `#[cfg(test)] mod tests { ... }` regions in whole-file mode: tests may
# use scalar oracles by design.
strip_tests() {
  awk -F'\t' '
    $2 ~ /^#\[cfg\(test\)\]/ { skip = 1 }
    skip && $2 ~ /^}/ { skip = 0; next }
    !skip { print }'
}

banner() {
  cat >&2 <<'MSG'

  ███ FACET STRIDE VIOLATION ███   '{0}..{15}' -f $128.split()
  A 128-bit value is 16 u8 positions; one mov reads them all. Shuffles only
  MOVE bytes; never assemble a result from them. Scalar loops live only in
  simd_scalar.rs (on AVX2 use the 2 x U8x32 / 128-bit-half split). u8 has no
  sign: do not mix u8 and i8.
  Genuine exception: `// facet-stride: ok <reason>` on or above the line.
  Rule text: CLAUDE.md § FACET STRIDE.
MSG
}

mode="${1:---all}"; shift || true
case "$mode" in
  --hook)
    payload="$(cat)"
    f="$(printf '%s' "$payload" | python3 -c 'import json,sys
d=json.load(sys.stdin); ti=d.get("tool_input",{})
print(ti.get("file_path") or ti.get("notebook_path") or "")' 2>/dev/null)"
    [ -z "$f" ] && exit 0
    f="${f#"$ROOT"/}"
    in_scope "$f" || exit 0
    [ -f "$f" ] || exit 0
    added_lines "$f" HEAD | check_lines "$f"
    ;;
  --diff)
    rev="${1:?--diff needs a revision}"
    while IFS= read -r f; do
      in_scope "$f" && [ -f "$f" ] && added_lines "$f" "$rev" | check_lines "$f"
    done < <( { git diff --name-only "$rev"; git ls-files --others --exclude-standard; } | sort -u)
    ;;
  --file)
    for f in "$@"; do f="${f#"$ROOT"/}"; [ -f "$f" ] && whole_file "$f" | strip_tests | check_lines "$f"; done
    ;;
  --all)
    while IFS= read -r f; do whole_file "$f" | strip_tests | check_lines "$f"; done \
      < <(git ls-files 'src/*.rs' 'crates/*.rs' 'examples/*.rs' | while IFS= read -r f; do in_scope "$f" && echo "$f"; done)
    ;;
  *) echo "usage: $0 --hook | --diff <rev> | --all | --file <path>..." >&2; exit 64 ;;
esac

hits=$(awk '{s += $1} END {print s + 0}' "$HITS_FILE")
if [ "$hits" -gt 0 ]; then
  banner
  echo "  $hits violation(s)." >&2
  [ "$mode" = "--hook" ] && exit 2
  exit 1
fi
exit 0
