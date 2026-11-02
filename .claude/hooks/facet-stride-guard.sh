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

GUARD_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$GUARD_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ "${1:-}" = "--root" ]; then ROOT="$(cd "${2:?--root needs a directory}" && pwd)"; shift 2; fi
cd "$ROOT" || exit 0
NDARRAY=0; [ -f src/simd_scalar.rs ] && [ -f src/simd_masking_ops.rs ] && NDARRAY=1
# Exclusions: <root>/.claude/hooks/facet-stride-exclude.txt, lines
#   <path-glob> | <rules, e.g. R1 or R1,R3> | <reason>
# for code whose byte shuffle IS the defined operation (cipher rounds,
# permutation maps) or a deliberate measurement of the anti-pattern.
# A line without a reason is ignored.
EXCLUDE_FILE="$ROOT/.claude/hooks/facet-stride-exclude.txt"

in_scope() {            # rust sources only; test files may use scalar oracles by design
  case "$1" in
    */tests.rs|*/tests/*|tests/*) return 1 ;;
    */benches/*|benches/*) return 1 ;;
    *.rs) return 0 ;;
    *) return 1 ;;
  esac
}
simd_backend() {        # non-scalar SIMD realizations (ndarray only)
  [ "$NDARRAY" = 1 ] || return 1
  case "$1" in
    src/simd_avx2.rs|src/simd_avx512.rs|src/simd_avx.rs|src/simd_neon.rs|src/simd_wasm.rs|src/simd_amx.rs|src/simd_nightly/*.rs) return 0 ;;
    *) return 1 ;;
  esac
}
simd_file() {
  if [ "$NDARRAY" = 1 ]; then
    case "$1" in src/simd*.rs|src/simd*/*.rs) return 0 ;; *) return 1 ;; esac
  fi
  # Elsewhere: any file that touches SIMD at all.
  grep -qE 'core::arch|std::arch|ndarray::simd' "$1" 2>/dev/null
}

R1='\.shuffle_bytes\(|\.permute_bytes\(|_shuffle_epi8|permutexvar_epi8|permutex2var_epi8|swizzle_dyn|vqtbl[1-4]q?|vtbl[1-4]|i8x16_swizzle|u8x16_swizzle'
R2='for +[a-z_]+ +in +0 *\.\. *(8|16|32|64)\b|for +lane +in|for +\(?[a-z_, &]*\)? +in +[a-z_.]*\.iter(_mut)?\(\)\.zip\('
R3='as +i8\b|i8::from\(|u8::from\([^)]*i8|as +u8\b.*\bi8\b|\bi8\b.*as +u8\b'

HITS_FILE="$(mktemp)"; trap 'rm -f "$HITS_FILE"' EXIT

# check_lines FILE  <stdin: "LINENO<TAB>TEXT" for the lines in scope>
# The checker reads the WHOLE file for context (test-module brace depth, the
# enclosing impl type and fn) and reports only the line numbers it was given.
check_lines() {
  R1="$R1" R2="$R2" BACKEND="$(simd_backend "$1" && echo 1)" SIMD="$(simd_file "$1" && echo 1)" \
  EXCLUDE_FILE="$EXCLUDE_FILE" python3 -c '
import fnmatch, os, re, sys
f = sys.argv[1]
want = set()
for raw in sys.stdin:
    n, _, _ = raw.partition("\t")
    if n.strip().isdigit():
        want.add(int(n))
r1, r2 = re.compile(os.environ["R1"]), re.compile(os.environ["R2"])
backend, simd = os.environ.get("BACKEND") == "1", os.environ.get("SIMD") == "1"
excluded = set()
try:
    for line in open(os.environ["EXCLUDE_FILE"]):
        parts = [x.strip() for x in line.split("|")]
        if len(parts) == 3 and parts[2] and not parts[0].startswith("#") and fnmatch.fnmatch(f, parts[0]):
            excluded |= {r.strip() for r in parts[1].split(",")}
except OSError:
    pass
# A byte-shuffle wrapper may contain the instruction: it IS the move.
shuffle_def = re.compile(r"^(shuffle_bytes|permute_bytes|permute_bytes_vbmi|swizzle\w*|shuffle\w*|permute\w*)$")
ok = re.compile(r"facet-stride: ok \S")
comment = re.compile(r"^\s*//")
impl_re = re.compile(r"^\s*impl\b[^{;]*?\b(?:for\s+)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^{]*>)?\s*\{")
fn_re = re.compile(r"\bfn\s+([A-Za-z0-9_]+)")
lines = open(f, encoding="utf-8", errors="replace").read().split("\n")
depth, test_until, pending_test, attr = 0, None, False, None
impl_name, impl_depth, fn_name = "", None, ""
hits = 0
for i, t in enumerate(lines, 1):
    code = t.split("//")[0]
    # cfg attributes may span lines: collect until the closing bracket.
    if attr is None and re.match(r"^\s*#\[cfg\(", t):
        attr = ""
    if attr is not None:
        attr += code
        if "]" in code:
            if re.search(r"\btest\b", attr):
                pending_test = True
            attr = None
    if re.match(r"^\s*(pub(\(\w+\))?\s+)?mod\s+\w*tests?\b", t):
        pending_test = True
    m = impl_re.match(t)
    if m:
        impl_name, impl_depth = m.group(1), depth
    fm = fn_re.search(code)
    if fm:
        fn_name = fm.group(1)
    in_test = test_until is not None
    if pending_test and "{" in code:
        if test_until is None:
            test_until = depth
        pending_test = False
        in_test = True
    if i in want and not in_test and not comment.match(t):
        prev = lines[i - 2] if i >= 2 else ""
        if not (ok.search(prev) or ok.search(t)):
            found = []
            if "R1" not in excluded and r1.search(code) and not shuffle_def.match(fn_name):
                found.append("R1 SHUFFLE-AS-COMPUTE")
            if "R2" not in excluded and backend and r2.search(code):
                found.append("R2 SCALAR-IN-SIMD")
            if "R3" not in excluded and simd:
                u8_ty, i8_ty = impl_name.startswith("U8"), impl_name.startswith("I8")
                to_i8, to_u8 = re.search(r"as\s+i8\b", code), re.search(r"as\s+u8\b", code)
                mix = (u8_ty and to_i8) or (i8_ty and to_u8) \
                    or (to_i8 and re.search(r"\bu8\b", code)) or (to_u8 and re.search(r"\bi8\b", code))
                if mix:
                    found.append("R3 U8-I8-MIX")
            for rule in found:
                hits += 1
                print(f"{f}:{i}  {rule}  {t.strip()[:140]}", file=sys.stderr)
    depth += code.count("{") - code.count("}")
    if test_until is not None and depth <= test_until:
        test_until = None
    if impl_depth is not None and depth <= impl_depth:
        impl_name, impl_depth = "", None
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
    for f in "$@"; do f="${f#"$ROOT"/}"; [ -f "$f" ] && whole_file "$f" | check_lines "$f"; done
    ;;
  --all)
    # A tracked path can be a dangling symlink (crates/burn/src/*.rs point
    # into an absent ../upstream checkout); skip anything that is not a file.
    while IFS= read -r f; do [ -f "$f" ] && whole_file "$f" | check_lines "$f"; done \
      < <(git ls-files '*.rs' | while IFS= read -r f; do in_scope "$f" && echo "$f"; done)
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
