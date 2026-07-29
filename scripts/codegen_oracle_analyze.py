#!/usr/bin/env python3
"""Instruction-histogram analyzer for `scripts/codegen-oracle.sh`.

Parses the `--emit asm` output of `simd-codegen-oracle`, extracts each probe
kernel's assembly block by its `.type <sym>,@function` label, classifies each
instruction, and compares the result against the committed baseline TOML.

# Classification buckets

  packed-vector      AVX/AVX2 instructions operating on a full ymm/zmm width
                      lane group: `vp*` (packed integer), `v*ps`/`v*pd`
                      (packed single/double), `vfmadd*ps`/`vfmadd*pd` etc.,
                      `vmovaps`/`vmovups`/`vmovdqa`/`vmovdqu`, broadcast/
                      permute/blend/extract-128/insert-128 forms. Explicitly
                      EXCLUDES the AVX *scalar* forms (`v*ss`/`v*sd`, single
                      lane despite the `v` prefix) and single-lane extract/
                      insert forms (`vmovq`/`vmovd`/`vpextr*`/`vpinsr*`) --
                      those move or compute exactly one lane, not a packed
                      width, and are bucketed as scalar-lane-arith / memory
                      respectively.

  scalar-lane-arith  Scalar arithmetic that is NOT part of the loop-control
                      idiom -- i.e. GPR add/sub/and/or/xor/shl/shr/sar/rol/
                      ror/rorx/imul/neg/not/inc/dec (or the AVX *scalar*
                      ss/sd arithmetic forms) operating on data, not on an
                      index/pointer/trip-count register. See "Loop-control
                      vs lane-arith" below for exactly how the two are told
                      apart -- this is the CRITICAL HONESTY distinction the
                      oracle exists to get right: a `decl` that decrements a
                      loop counter is scalar-ALU but does not touch lane
                      data, and must not be counted here.

  loop-control       Conditional/unconditional jumps, `cmp`/`test`, `lea`
                      (address computation), and any GPR arithmetic
                      instruction identified as operating on an index /
                      pointer / trip-count register (see below).

  memory             Plain `mov`-family (register<->register, register<->
                      memory, zero/sign-extending loads), `push`/`pop`, and
                      single-lane vector<->GPR data movement (`vmovq`,
                      `vmovd`, `vpextr*`, `vpinsr*`) that is not itself an
                      arithmetic op.

  other              Everything else (`vzeroupper`, `.cfi_*` already
                      stripped at extraction, prefetch/nop, etc).

# Loop-control vs lane-arith (the honesty-critical rule)

`cmp`/`test`/`j*`/`lea` are ALWAYS loop-control -- across every probe in this
oracle, no kernel performs a per-element scalar *comparison* or *branch* on
lane data (they are pure straight-line ARX/gather/widen/reverse kernels), so
this is a safe rule for this specific probe set, not a general-purpose
disassembler heuristic.

For the remaining GPR arithmetic mnemonics, a register is classified as an
"index/pointer/bookkeeping register" for the whole block if, ANYWHERE in the
block, it is (a) used as a base or index register inside a memory operand
`offset(%base[,%index[,scale]])`, or (b) an operand of any `cmp`/`test`
instruction. An arithmetic instruction whose destination register is in that
set is loop-control; otherwise it is scalar-lane-arith.

Two exceptions, both load-bearing:

  - `inc`/`dec` are ALWAYS loop-control (the spec's own canonical example:
    a loop-counter decrement is scalar-ALU but never touches lane data).
  - `rol`/`ror`/`rorx`/`rolx` (rotate, in any width) are ALWAYS
    scalar-lane-arith. No probe in this oracle uses rotate for index
    bookkeeping -- every rotate instruction that appears is a real ARX/
    BLAKE2b/ChaCha rotate on lane data, so treating rotate as
    "index-adjacent" would hide exactly the finding this oracle exists to
    surface.

KNOWN LIMITATION (stated here, not hidden): this is a whole-block, name-based
heuristic, not a real dataflow/liveness analysis. A register that is
reused across two unrelated logical roles within one block (e.g. briefly
holding a copied pointer, then later reused -- after being overwritten --
to hold a scalar data value) can be misclassified, because the heuristic
does not track *when* a register held which role, only *whether* it ever
played an address/compare role anywhere in the block. Measured impact: this
under-counts scalar-lane-arith in `blake2b_g_u64x8` specifically (a straight-
line, register-heavy kernel with real register reuse) by a handful of
instructions out of several dozen; the AGGREGATE verdict (substantial
scalar-lane-arith present, zero packed-vector coverage for the rotate-
dependent chain) is unaffected and was cross-checked by hand against the raw
disassembly. Every other probe in this oracle has no register-reuse-across-
roles and is classified exactly.
"""
import argparse
import re
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - CI runners pin 3.11+, this is a courtesy fallback
    tomllib = None


PACKED_RE = re.compile(
    r"^v("
    r"p(?!extr|insr)[a-z0-9]+"  # vp* packed-integer, but not vpextr*/vpinsr* (single-lane)
    r"|mova?p[sd]"
    r"|movu?p[sd]"
    r"|movdqa|movdqu"
    r"|broadcast[a-z0-9]*"
    r"|perm[a-z0-9]*"
    r"|extracti[0-9]+"
    r"|inserti[0-9]+"
    r"|blendvb|blendps|blendpd"
    r"|add[ps][sd]?p[sd]|addp[sd]|subp[sd]|mulp[sd]|divp[sd]"
    r"|xorp[sd]|andnp[sd]|andp[sd]|orp[sd]"
    r"|cmpp[sd]"
    r"|f(n?m)?add[0-9]+p[sd]|f(n?m)?sub[0-9]+p[sd]"
    r"|cvt[a-z0-9]*p[sd]|cvttp[sd]2[a-z0-9]+"
    r"|roundp[sd]|maxp[sd]|minp[sd]|sqrtp[sd]"
    r"|unpcklp[sd]|unpckhp[sd]|shufp[sd]|movmskp[sd]"
    r")$"
)
# Single-lane AVX forms: NOT packed, despite the `v` prefix.
SCALAR_VECTOR_ARITH_RE = re.compile(r"^v(f(n?m)?add[0-9]+s[sd]|adds[sd]|subs[sd]|muls[sd]|divs[sd])$")
SCALAR_VECTOR_MOVE_RE = re.compile(r"^v(movs[sd]|movq|movd|pextr[bwdq]|pinsr[bwdq])$")

ROTATE_RE = re.compile(r"^(rol|ror|rorx|rolx)[lqwb]?$")
INCDEC_RE = re.compile(r"^(inc|dec)[lqwb]?$")
GPR_ARITH_RE = re.compile(r"^(add|adc|sub|sbb|and|or|xor|shl|sal|shr|sar|imul|mul|neg|not)[lqwb]?$")
CMP_TEST_RE = re.compile(r"^(cmp|test)[lqwb]?$")
JUMP_RE = re.compile(r"^j[a-z]*$")
LEA_RE = re.compile(r"^lea[qwl]?$")
MOV_RE = re.compile(r"^(mov(z|s)?[bwlq]{0,2}|movabs[qlwb]?|push[qwl]?|pop[qwl]?)$")
NOP_RE = re.compile(r"^(nop[lwq]?|endbr(32|64)|ud2)$")

REG_TOKEN_RE = re.compile(r"%([a-z][a-z0-9]*)")
IMM_RE = re.compile(r"\$-?(\d+)")


def normalize_reg(tok: str) -> str:
    """Collapse register width aliases to one canonical family name."""
    t = tok.lower()
    if re.fullmatch(r"r(\d+)[bwd]?", t):
        return re.match(r"r\d+", t).group(0)  # r8b/r8w/r8d -> r8
    table = {
        "al": "ax", "ah": "ax", "ax": "ax", "eax": "ax", "rax": "ax",
        "bl": "bx", "bh": "bx", "bx": "bx", "ebx": "bx", "rbx": "bx",
        "cl": "cx", "ch": "cx", "cx": "cx", "ecx": "cx", "rcx": "cx",
        "dl": "dx", "dh": "dx", "dx": "dx", "edx": "dx", "rdx": "dx",
        "sil": "si", "si": "si", "esi": "si", "rsi": "si",
        "dil": "di", "di": "di", "edi": "di", "rdi": "di",
        "bpl": "bp", "bp": "bp", "ebp": "bp", "rbp": "bp",
        "spl": "sp", "sp": "sp", "esp": "sp", "rsp": "sp",
    }
    return table.get(t, t)


def split_operands(rest: str):
    """Split an AT&T operand list on top-level commas (parens protect memory operands)."""
    depth = 0
    field = []
    out = []
    for ch in rest:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(field).strip())
            field = []
        else:
            field.append(ch)
    if field:
        out.append("".join(field).strip())
    return out


def mem_operand_regs(operand: str):
    """Registers used as base/index inside a memory operand, e.g. `16(%rdx,%rax,4)`."""
    m = re.search(r"\(([^)]*)\)", operand)
    if not m:
        return set()
    return {normalize_reg(r) for r in REG_TOKEN_RE.findall(m.group(1))}


class Instr:
    __slots__ = ("raw", "mnemonic", "operands")

    def __init__(self, raw: str):
        self.raw = raw.strip()
        parts = self.raw.split(None, 1)
        self.mnemonic = parts[0].lower() if parts else ""
        self.operands = split_operands(parts[1]) if len(parts) > 1 else []


def is_instruction_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("//"):
        return False
    if s.startswith("."):
        return False
    if s.endswith(":"):
        return False
    first = s.split(None, 1)[0]
    return bool(re.match(r"^[a-z]", first, re.IGNORECASE))


def extract_probe_block(lines, sym_needle: str):
    """Return the raw instruction lines between `.type <sym>,@function` and the
    matching `ret`/`retq`, falling back to the next `.type ...,@function` label
    (a different function starting) as a safety bound if no ret is found first."""
    start = None
    for i, line in enumerate(lines):
        if ".type" in line and "@function" in line and sym_needle in line:
            start = i
            break
    if start is None:
        return None
    body = []
    i = start + 1
    # Skip the symbol's own label line (`SYM:`) if present.
    if i < len(lines) and lines[i].strip().endswith(":"):
        i += 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if re.match(r"^ret[q]?$", stripped, re.IGNORECASE):
            body.append(line)
            break
        if ".type" in line and "@function" in line:
            # Next function started; no ret found (shouldn't happen for our
            # probes, but bound the extraction defensively).
            break
        body.append(line)
        i += 1
    return body


def classify_block(lines):
    instrs = [Instr(l) for l in lines if is_instruction_line(l)]

    # Pass 1: collect the whole-block "bookkeeping register" set (addressing
    # base/index registers, and any register compared/tested).
    bookkeeping = set()
    for ins in instrs:
        for op in ins.operands:
            bookkeeping |= mem_operand_regs(op)
        if CMP_TEST_RE.match(ins.mnemonic):
            for op in ins.operands:
                bookkeeping |= {normalize_reg(r) for r in REG_TOKEN_RE.findall(op)}

    counts = {"packed-vector": 0, "scalar-lane-arith": 0, "loop-control": 0, "memory": 0, "other": 0}
    detail = {k: [] for k in counts}

    for ins in instrs:
        m = ins.mnemonic
        bucket = None

        if PACKED_RE.match(m):
            bucket = "packed-vector"
        elif SCALAR_VECTOR_ARITH_RE.match(m):
            bucket = "scalar-lane-arith"
        elif SCALAR_VECTOR_MOVE_RE.match(m):
            bucket = "memory"
        elif JUMP_RE.match(m) or CMP_TEST_RE.match(m) or LEA_RE.match(m):
            bucket = "loop-control"
        elif INCDEC_RE.match(m):
            bucket = "loop-control"  # canonical loop-counter idiom, never lane data
        elif ROTATE_RE.match(m):
            bucket = "scalar-lane-arith"  # never index bookkeeping in this probe set
        elif GPR_ARITH_RE.match(m):
            dest = normalize_reg(REG_TOKEN_RE.findall(ins.operands[-1])[0]) if ins.operands and REG_TOKEN_RE.findall(ins.operands[-1]) else None
            bucket = "loop-control" if dest in bookkeeping else "scalar-lane-arith"
        elif MOV_RE.match(m):
            bucket = "memory"
        elif NOP_RE.match(m) or m in ("vzeroupper", "vzeroall"):
            bucket = "other"
        else:
            bucket = "other"

        counts[bucket] += 1
        detail[bucket].append(ins.raw)

    return counts, detail


def load_baseline(path):
    with open(path, "rb") as f:
        if tomllib is not None:
            return tomllib.load(f)
        return _parse_toml_fallback(f.read().decode("utf-8"))


def _parse_toml_fallback(text: str):
    """Minimal TOML subset parser (only what baselines/*.toml actually uses):
    `[section.name]` headers and `key = value` where value is a bare word,
    quoted string, or integer. Used only if `tomllib` is unavailable."""
    data = {}
    section = data
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            keys = line[1:-1].split(".")
            node = data
            for k in keys:
                node = node.setdefault(k, {})
            section = node
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            elif re.fullmatch(r"-?\d+", v):
                v = int(v)
            section[k] = v
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm_path")
    ap.add_argument("baseline_path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.asm_path, "r", errors="replace") as f:
        lines = f.readlines()

    baseline = load_baseline(args.baseline_path)
    probes = baseline.get("probe", {})
    # Stable order: declaration order in the baseline file isn't preserved by
    # a naive dict in older parsers; sort by name for determinism instead.
    names = sorted(probes.keys())

    print("=" * 100)
    print("simd-codegen-oracle -- instruction histogram")
    print(
        "NOTE: 'scalar arithmetic on lane data' EXCLUDES loop-control-idiom ops\n"
        "      (cmp/test/jumps/lea, and any GPR arithmetic on an index/pointer/\n"
        "      trip-count register, including inc/dec). A loop-counter decl does\n"
        "      NOT count as lane arithmetic. See this script's module docstring\n"
        "      for the exact rule and its documented limitation."
    )
    print("=" * 100)
    header = f"{'probe':<26}{'packed':>8}{'lane-arith':>12}{'loop-ctl':>10}{'memory':>8}{'other':>8}  verdict"
    print(header)
    print("-" * len(header))

    exit_mask = 0
    any_missing = False
    for idx, name in enumerate(names):
        spec = probes[name]
        needle = f"{len(name)}{name}"
        block = extract_probe_block(lines, needle)
        if block is None:
            print(f"{name:<26}  MISSING FROM ASM (symbol not found)")
            any_missing = True
            exit_mask |= 1 << idx
            continue
        counts, detail = classify_block(block)

        expect = spec.get("expect", "unknown")
        min_packed = int(spec.get("min_packed", 0))
        max_scalar = spec.get("max_scalar_lane_arith")
        max_packed = spec.get("max_packed")

        ok = True
        reasons = []
        if expect == "vectorized":
            if counts["packed-vector"] < min_packed:
                ok = False
                reasons.append(f"packed={counts['packed-vector']} < min_packed={min_packed}")
            if max_scalar is not None and counts["scalar-lane-arith"] > int(max_scalar):
                ok = False
                reasons.append(f"lane-arith={counts['scalar-lane-arith']} > max_scalar_lane_arith={max_scalar}")
        elif expect == "scalar":
            cap = int(max_packed) if max_packed is not None else 0
            if counts["packed-vector"] > cap:
                ok = False
                reasons.append(f"packed={counts['packed-vector']} > max_packed={cap}")
        elif expect == "unknown":
            pass  # Group C: report only, no pass/fail assertion.
        else:
            ok = False
            reasons.append(f"unrecognized expect={expect!r} in baseline")

        if expect == "unknown":
            verdict = "OBSERVE"
        elif ok:
            verdict = "PASS"
        else:
            verdict = "FAIL: " + "; ".join(reasons)
            exit_mask |= 1 << idx

        print(
            f"{name:<26}{counts['packed-vector']:>8}{counts['scalar-lane-arith']:>12}"
            f"{counts['loop-control']:>10}{counts['memory']:>8}{counts['other']:>8}  {verdict}"
        )
        if args.verbose:
            for bucket in ("packed-vector", "scalar-lane-arith", "loop-control"):
                if detail[bucket]:
                    print(f"    [{bucket}]")
                    for raw in detail[bucket]:
                        print(f"        {raw}")

    print("-" * len(header))
    if any_missing:
        print("ERROR: one or more probes were not found in the emitted assembly.")
    if exit_mask == 0:
        print("simd-codegen-oracle: ALL PROBES MATCH BASELINE EXPECTATIONS")
        return 0

    # The bitmask is DIAGNOSTIC OUTPUT ONLY, never the exit status.
    #
    # Unix truncates an exit status to its low 8 bits. Returning the raw
    # mask means a failure confined to probe index >= 8 exits 0x100 / 0x200
    # / ..., which the shell observes as 0: the script prints "FAILURES"
    # and reports success. With probes sorted alphabetically that silently
    # swallowed regressions in rot_u64x4, rot_u64x8, saturating_abs_i8x32,
    # serial_dependent_chain and widening_u16_to_f32 -- including both
    # u64-rotate probes, the finding this tool exists to protect.
    print(f"simd-codegen-oracle: FAILURES (probe bitmask 0x{exit_mask:x}) -- see FAIL rows above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
