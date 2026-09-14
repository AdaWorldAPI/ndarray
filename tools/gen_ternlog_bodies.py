#!/usr/bin/env python3
"""Generate the backend-local `ternlog::<IMM>` lowering bodies.

THE BACKEND LAW (operator, 2026-09-13): ndarray exposes ONE architecture-
agnostic semantic API, selected at compile time into COMPLETE PEER
implementations in simd_avx512.rs / simd_avx2.rs / simd_neon.rs /
simd_wasm.rs / simd_scalar.rs. No shared runtime polyfill sits under them.
Shared *tests* and shared *generated truth-table logic* are fine; a common
implementation function the backends delegate into is not.

This script is that shared logic. It derives, once, the Shannon lowering of
an 8-bit truth table into a minimal Boolean DAG, checks it against a
bit-serial reference for all 256 tables, and then PRINTS each backend's body
in that backend's own vocabulary (operator traits on the array-backed lane
types; `vandq/vorrq/veorq/vbicq_u{32,64}` NEON intrinsics per 128-bit quad —
NOT a per-lane `u32` loop, which LLVM scalarised (536 scalar / 4 vector ops
measured); `v128_*` intrinsics for WASM).
The emitted text is pasted into the backend file between GEN markers by
`--apply`; it is committed source, and the generator is its provenance.

Lowering: with index `(a << 2) | (b << 1) | c`, the even bits of IMM are the
2-input table T0(a,b) (c = 0) and the odd bits T1(a,b) (c = 1);
    f = (!c & T0) | (c & T1)
with every 2-input table a <= 2-op closed form and six collapse shapes of the
outer combination. Worst case 7 ops where the vocabulary has a native and-not
(NEON `vbic`, WASM `v128.andnot`), 8 where and-not is spelled `x & !y` (the
avx2/scalar operator vocabularies); the naive 8-minterm form was up to 36.
The count is ASSERTED below (`max_ops`), not just stated.
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"

# 2-input table (bit k = value at index (a<<1)|b) -> expression AST
TWO = {
    0x0: ("zero",), 0x1: ("not", ("or", "a", "b")), 0x2: ("and", ("not", "a"), "b"),
    0x3: ("not", "a"), 0x4: ("andnot", "a", "b"), 0x5: ("not", "b"), 0x6: ("xor", "a", "b"),
    0x7: ("not", ("and", "a", "b")), 0x8: ("and", "a", "b"), 0x9: ("not", ("xor", "a", "b")),
    0xA: "b", 0xB: ("or", ("not", "a"), "b"), 0xC: "a", 0xD: ("or", "a", ("not", "b")),
    0xE: ("or", "a", "b"), 0xF: ("ones",),
}

def ev(ast, a, b, c=None):
    if isinstance(ast, str):
        return {"a": a, "b": b, "c": c, "g0": None, "g1": None}[ast]
    k = ast[0]
    if k == "zero": return 0
    if k == "ones": return 0xFFFFFFFFFFFFFFFF
    if k == "not": return ev(ast[1], a, b, c) ^ 0xFFFFFFFFFFFFFFFF
    x, y = ev(ast[1], a, b, c), ev(ast[2], a, b, c)
    return {"and": x & y, "or": x | y, "xor": x ^ y, "andnot": x & (y ^ 0xFFFFFFFFFFFFFFFF)}[k]

def reference(imm, a, b, c):
    r = 0
    for bit in range(64):
        idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        r |= ((imm >> idx) & 1) << bit
    return r

def halves(imm):
    t0 = sum(((imm >> (2 * k)) & 1) << k for k in range(4))
    t1 = sum(((imm >> (2 * k + 1)) & 1) << k for k in range(4))
    return t0, t1

# self-check: every table, adversarial operands covering all 8 index combos
A, B, C = 0xF0F0F0F0F0F0F0F0, 0xCCCCCCCCCCCCCCCC, 0xAAAAAAAAAAAAAAAA
def lowered_value(imm):
    t0, t1 = halves(imm)
    g0, g1 = ev(TWO[t0], A, B), ev(TWO[t1], A, B)
    NOT = lambda x: x ^ 0xFFFFFFFFFFFFFFFF
    if t0 == t1: return g0
    if t0 == 0: return C & g1
    if t1 == 0: return g0 & NOT(C)
    if t1 == (t0 ^ 0xF): return C ^ g0
    if t0 == 0xF: return NOT(C) | g1
    if t1 == 0xF: return C | g0
    return (g0 & NOT(C)) | (g1 & C)
for imm in range(256):
    assert lowered_value(imm) == reference(imm, A, B, C), f"lowering wrong at {imm:#04x}"

# per-backend printers: (name, vocabulary) — vocabulary maps AST node -> source
def op_printer(kind_and, kind_or, kind_xor, kind_not, kind_andnot, zero, ones):
    def p(ast):
        if isinstance(ast, str): return ast
        k = ast[0]
        if k == "zero": return zero
        if k == "ones": return ones
        if k == "not": return kind_not(p(ast[1]))
        x, y = p(ast[1]), p(ast[2])
        return {"and": kind_and, "or": kind_or, "xor": kind_xor, "andnot": kind_andnot}[k](x, y)
    return p

OPS = op_printer(lambda x,y: f"({x} & {y})", lambda x,y: f"({x} | {y})", lambda x,y: f"({x} ^ {y})",
                 lambda x: f"!{x}", lambda x,y: f"({x} & !{y})", "ZERO", "ONES")
NEON = op_printer(lambda x,y: f"vandq_u32({x}, {y})", lambda x,y: f"vorrq_u32({x}, {y})",
                  lambda x,y: f"veorq_u32({x}, {y})", lambda x: f"vmvnq_u32({x})",
                  lambda x,y: f"vbicq_u32({x}, {y})", "vdupq_n_u32(0)", "vdupq_n_u32(!0)")
NEON64 = op_printer(lambda x,y: f"vandq_u64({x}, {y})", lambda x,y: f"vorrq_u64({x}, {y})",
                    lambda x,y: f"veorq_u64({x}, {y})", lambda x: f"veorq_u64({x}, vdupq_n_u64(!0))",
                    lambda x,y: f"vbicq_u64({x}, {y})", "vdupq_n_u64(0)", "vdupq_n_u64(!0)")
WASM = op_printer(lambda x,y: f"v128_and({x}, {y})", lambda x,y: f"v128_or({x}, {y})",
                  lambda x,y: f"v128_xor({x}, {y})", lambda x: f"v128_not({x})",
                  lambda x,y: f"v128_andnot({x}, {y})", "u32x4_splat(0)", "u32x4_splat(!0)")

def top(src):
    """Strip one redundant outer paren layer (rustc `unused_parens` is a warning)."""
    if src.startswith("(") and src.endswith(")"):
        depth = 0
        for i, ch in enumerate(src):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(src) - 1:
                return src
        return src[1:-1]
    return src

def two_input_fn(name, ty, printer, zero, ones, indent="", attrs=(), unsafe_reason=None):
    lines = [f"{indent}/// GENERATED by `tools/gen_ternlog_bodies.py` — a 2-input Boolean function",
             f"{indent}/// by its 4-bit table (bit `k` = value at index `(a << 1) | b`), at most",
             f"{indent}/// two operations. `#[inline]` (not `always`): the 256-table test would",
             f"{indent}/// otherwise carry every arm's temporaries in one debug frame."]
    lines += [f"{indent}{a}" for a in attrs]
    lines += [f"{indent}#[inline]",
              f"{indent}fn {name}(t: u8, a: {ty}, b: {ty}) -> {ty} {{"]
    inner = indent + ("    " if unsafe_reason else "")
    if unsafe_reason:
        for k, part in enumerate(unsafe_reason.split("\n")):
            lines.append(f"{indent}    // {'SAFETY: ' if k == 0 else ''}{part}")
        lines.append(f"{indent}    unsafe {{")
    lines.append(f"{inner}    match t & 0xF {{")
    for k in range(16):
        src = top(printer(TWO[k]).replace("ZERO", zero).replace("ONES", ones))
        lines.append(f"{inner}        {'_' if k == 15 else f'{k:#03x}'} => {src},")
    lines.append(f"{inner}    }}")
    if unsafe_reason:
        lines.append(f"{indent}    }}")
    lines.append(f"{indent}}}")
    return "\n".join(lines)

def ladder(g, a, b, c, and_, or_, xor_, not_, andnot, indent, bind_indent=None):
    """The outer Shannon combination, branching only on IMM-derived values (let-bound: a `const` item
    cannot name the enclosing fn's IMM; after monomorphization these fold identically).
    Returns (bindings, body) so a backend can place the two plain-integer `let`s OUTSIDE
    its intrinsic `unsafe` block — the block then wraps intrinsic calls and nothing else."""
    bi = indent if bind_indent is None else bind_indent
    B = [f"{bi}let t0: u8 = ((IMM & 1) | ((IMM >> 1) & 2) | ((IMM >> 2) & 4) | ((IMM >> 3) & 8)) as u8;",
         f"{bi}let t1: u8 = (((IMM >> 1) & 1) | ((IMM >> 2) & 2) | ((IMM >> 3) & 4) | ((IMM >> 4) & 8)) as u8;"]
    L = []
    L.append(f"{indent}if t0 == t1 {{")
    L.append(f"{indent}    {g}(t0, {a}, {b})")
    L.append(f"{indent}}} else if t0 == 0 {{")
    L.append(f"{indent}    {top(and_(c, f'{g}(t1, {a}, {b})'))}")
    L.append(f"{indent}}} else if t1 == 0 {{")
    L.append(f"{indent}    {top(andnot(f'{g}(t0, {a}, {b})', c))}")
    L.append(f"{indent}}} else if t1 == (t0 ^ 0xF) {{")
    L.append(f"{indent}    {top(xor_(c, f'{g}(t0, {a}, {b})'))}")
    L.append(f"{indent}}} else if t0 == 0xF {{")
    L.append(f"{indent}    {top(or_(not_(c), f'{g}(t1, {a}, {b})'))}")
    L.append(f"{indent}}} else if t1 == 0xF {{")
    L.append(f"{indent}    {top(or_(c, f'{g}(t0, {a}, {b})'))}")
    L.append(f"{indent}}} else {{")
    L.append(f"{indent}    {top(or_(andnot(f'{g}(t0, {a}, {b})', c), and_(f'{g}(t1, {a}, {b})', c)))}")
    L.append(f"{indent}}}")
    return "\n".join(B), "\n".join(L)

def count_ops(src):
    """Operator/intrinsic count of one emitted expression (the metric the docs quote)."""
    return len(re.findall(r"[&|^]|!(?=[a-z(])|\bv(?:and|orr|eor|bic)q_u(?:32|64)\b|\bv128_(?:and|or|xor|not|andnot)\b|vdupq_n_u64\(!0\)", src))

OPL = dict(and_=lambda x,y: f"({x} & {y})", or_=lambda x,y: f"({x} | {y})", xor_=lambda x,y: f"({x} ^ {y})",
           not_=lambda x: f"!{x}", andnot=lambda x,y: f"({x} & !{y})")
NEONL = dict(and_=lambda x,y: f"vandq_u32({x}, {y})", or_=lambda x,y: f"vorrq_u32({x}, {y})",
             xor_=lambda x,y: f"veorq_u32({x}, {y})", not_=lambda x: f"vmvnq_u32({x})",
             andnot=lambda x,y: f"vbicq_u32({x}, {y})")
NEON64L = dict(and_=lambda x,y: f"vandq_u64({x}, {y})", or_=lambda x,y: f"vorrq_u64({x}, {y})",
               xor_=lambda x,y: f"veorq_u64({x}, {y})", not_=lambda x: f"veorq_u64({x}, vdupq_n_u64(!0))",
               andnot=lambda x,y: f"vbicq_u64({x}, {y})")
WASML = dict(and_=lambda x,y: f"v128_and({x}, {y})", or_=lambda x,y: f"v128_or({x}, {y})",
             xor_=lambda x,y: f"v128_xor({x}, {y})", not_=lambda x: f"v128_not({x})",
             andnot=lambda x,y: f"v128_andnot({x}, {y})")

BEGIN = "// GEN-TERNLOG-BEGIN (tools/gen_ternlog_bodies.py — regenerate, do not hand-edit)"
END = "// GEN-TERNLOG-END"

def body_lane_type(ty):
    """avx2 / scalar: the array-backed lane types own & | ^ ! and splat."""
    helper = f"ternlog_two_input_{ty.lower()}"
    return ladder(helper, "self", "b", "c", indent="        ", **OPL), \
           two_input_fn(helper, ty, OPS, f"{ty}::splat(0)", f"{ty}::splat(!0)")

MARK = re.compile(r"^( *)// GENERATED lowering \((?:tools/gen_ternlog_bodies\.py|regenerating)\)[^\n]*\n", re.M)

# Where each generated body ENDS (the last line the generator itself emits), by
# shape. A freshly-written `(regenerating)` stub always ends at its
# `Self::from_array(o)` line, whatever the backend.
ARRAY_BODY_END = r"\n        \}\n"                      # avx2/scalar: the ladder's final `}` (the fn's own `}` is kept)
NEON_BODY_END  = r"\n        \}\)\)\n"                  # `Self(core::array::from_fn(|p| { ... }))`
WASM_BODY_END  = r"\n            \}\)\)\n"              # same shape, one module level deeper
STUB_END       = r"\n *Self::from_array\(o\)\n"

def apply(path, replacements, appendix, inside_module=None, write=True):
    s = path.read_text()
    for (sig_re, end_re), new in replacements:
        m = sig_re.search(s)
        assert m, (path.name, sig_re.pattern[-80:])
        ca = s.index("\n", s.index("const { assert!(IMM", m.end())) + 1
        mk = MARK.search(s, ca)
        assert mk and mk.start() < ca + 400, (path.name, "no GENERATED marker after the const assert")
        is_stub = "(regenerating)" in mk.group(0)
        endm = re.compile(STUB_END if is_stub else end_re).search(s, mk.end())
        assert endm, (path.name, "body end not found")
        # Replace from the marker through the END of the matched closer; `new`
        # carries its own closer. Text before the marker (nothing but the
        # const assert) and after the closer (the fn's own `}` where the body
        # did not include it) is kept verbatim.
        s = s[:mk.start()] + new + s[endm.end():]
    s = re.sub(r"\nimpl crate::simd_ternlog_lower::TernlogLanes for \w+ \{.*?\n\}\n", "\n", s, flags=re.S)
    if BEGIN in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        s = s[:i].rstrip("\n") + "\n" + s[j:].lstrip("\n")
        s = s.rstrip("\n") + "\n"
    block = BEGIN + "\n" + appendix + "\n" + END
    if inside_module is None:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    else:
        head = s.index(inside_module)
        close = s.index("\n}\n", head)
        s = s[:close].rstrip("\n") + "\n\n" + block + "\n" + s[close:]
    if write:
        path.write_text(s)
    return s

def fn_sig(ty_impl_re, fn_indent):
    """Regex for the `ternlog` signature INSIDE the given impl: from `impl <Ty> {` to the first
    ternlog signature, with no other `impl ` line in between (so an earlier `impl <Ty>` block
    without a ternlog cannot capture a later type's fn)."""
    return re.compile(ty_impl_re + r"(?:(?!\nimpl |\n    impl ).)*?" + re.escape(fn_indent + "pub fn ternlog<const IMM: i32>(self, b: Self, c: Self) -> Self {"), re.S)

def main(write, check=False):
    out = {}
    worst = {}
    # ── avx2 + scalar: U64x8 and U32x16 (array lanes, operator vocabulary) ──
    for fname in ("simd_avx2.rs", "simd_scalar.rs"):
        reps, helpers = [], []
        for ty in ("U64x8", "U32x16"):
            helper = f"ternlog_two_input_{ty.lower()}"
            binds, lad = ladder(helper, "self", "b", "c", indent="        ", **OPL)
            body = ("        // GENERATED lowering (tools/gen_ternlog_bodies.py): Shannon-expand on `c`\n"
                    "        // into two 2-input tables; <= 8 ops for any table in this vocabulary\n"
                    "        // (and-not is `x & !y`, two ops), folded at compile time.\n" + binds + "\n" + lad + "\n")
            reps.append(((fn_sig(rf"impl {ty} \{{", "    "), ARRAY_BODY_END), body))
            helpers.append(two_input_fn(helper, ty, OPS, f"{ty}::splat(0)", f"{ty}::splat(!0)"))
            worst[(fname, ty)] = 8
        out[fname] = (reps, "\n\n".join(helpers), None)
    # ── neon: U32x16 per-quad uint32x4_t and U64x8 per-quad uint64x2_t ──
    NEON_SAFETY = ("NEON is a baseline feature of every aarch64 target this module compiles\n"
                   "for; these are pure register operations on values already in `uint32x4_t`.")
    NEON_SAFETY64 = NEON_SAFETY.replace("uint32x4_t", "uint64x2_t")
    def neon_body(helper, quad_ty, vocab, safety):
        binds, lad = ladder(helper, "x", "y", "z", indent="                ", bind_indent="            ", **vocab)
        return ("        // GENERATED lowering (tools/gen_ternlog_bodies.py), per 128-bit quad (NEON).\n"
                "        Self(core::array::from_fn(|p| {\n"
                "            let (x, y, z) = (self.0[p].0, b.0[p].0, c.0[p].0);\n" + binds + "\n"
                "            // SAFETY: " + safety.replace("\n", "\n            // ") + "\n"
                f"            {quad_ty}(unsafe {{\n" + lad + "\n            })\n        }))\n")
    out["simd_neon.rs"] = ([
        ((fn_sig(r"impl U32x16 \{", "    "), NEON_BODY_END), neon_body("ternlog_two_input_u32x4", "U32x4", NEONL, NEON_SAFETY)),
        ((fn_sig(r"impl U64x8 \{", "    "), NEON_BODY_END), neon_body("ternlog_two_input_u64x2", "U64x2", NEON64L, NEON_SAFETY64)),
    ], "\n\n".join([
        two_input_fn("ternlog_two_input_u32x4", "uint32x4_t", NEON, "vdupq_n_u32(0)", "vdupq_n_u32(!0)",
                     attrs=('#[cfg(target_arch = "aarch64")]',), unsafe_reason=NEON_SAFETY),
        two_input_fn("ternlog_two_input_u64x2", "uint64x2_t", NEON64, "vdupq_n_u64(0)", "vdupq_n_u64(!0)",
                     attrs=('#[cfg(target_arch = "aarch64")]',), unsafe_reason=NEON_SAFETY64),
    ]), None)
    worst[("simd_neon.rs", "U32x16")] = 7; worst[("simd_neon.rs", "U64x8")] = 8  # NOT via veor(x, all-ones) costs one more
    # ── wasm: U32x16 and U64x8, both per-quad v128 through the one lane-agnostic helper ──
    def wasm_body(quad_ty):
        binds, lad = ladder("ternlog_two_input_v128", "x", "y", "z", indent="                    ", bind_indent="                ", **WASML)
        return ("            // GENERATED lowering (tools/gen_ternlog_bodies.py), per 128-bit quad.\n"
                "            Self(core::array::from_fn(|p| {\n"
                "                let (x, y, z) = (self.0[p].0, b.0[p].0, c.0[p].0);\n" + binds + "\n"
                f"                {quad_ty}({{\n" + lad + "\n                })\n            }))\n")
    out["simd_wasm.rs"] = ([
        ((fn_sig(r"impl U32x16 \{", "        "), WASM_BODY_END), wasm_body("U32x4")),
        ((fn_sig(r"impl U64x8 \{", "        "), WASM_BODY_END), wasm_body("U64x2")),
    ], two_input_fn("ternlog_two_input_v128", "v128", WASM, "u32x4_splat(0)", "u32x4_splat(!0)", indent="    "),
       "pub mod wasm32_simd {")
    worst[("simd_wasm.rs", "U32x16")] = 7; worst[("simd_wasm.rs", "U64x8")] = 7
    # Op-count assertion: the number the docs quote is measured on the emitted text.
    for (fname, ty), bound in worst.items():
        reps, _h, _m = out[fname]
        body = [n for ((sig, _e), n) in reps if f"impl {ty} " in sig.pattern.replace("\\{", "{").replace("\\", "")][0]
        last_else = [l for l in body.splitlines() if "(t0, x, y)" in l or "(t0, self, b)" in l][-1]
        ops = count_ops(last_else) + 2 * 2
        assert ops <= bound, (fname, ty, ops, bound)
    if check:
        # Compare AFTER rustfmt: the committed files are formatted, the emitted
        # text is not (long ladder lines get wrapped), so raw bytes would
        # always drift. Formatting is not content; a hand-edited arm still is.
        import subprocess, tempfile, difflib
        drift = []
        for fname, (reps, app, mod) in out.items():
            path = ROOT / fname
            regenerated = apply(path, reps, app, inside_module=mod, write=False)
            with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False, dir=str(ROOT.parent / "target") if (ROOT.parent / "target").exists() else None) as tf:
                tf.write(regenerated); tmp = tf.name
            subprocess.run(["rustfmt", "--edition", "2021", "--config-path", str(ROOT.parent), tmp], check=True)
            formatted = pathlib.Path(tmp).read_text(); pathlib.Path(tmp).unlink()
            if formatted != path.read_text():
                drift.append(fname)
                for line in difflib.unified_diff(path.read_text().splitlines(), formatted.splitlines(), "committed", "regenerated", lineterm="", n=1):
                    print(line)
        if drift:
            print("DRIFT: generated bodies differ from the generator's output in:", ", ".join(drift))
            sys.exit(1)
        print("check: all generated bodies current")
        return
    if not write:
        for k, (reps, app, _m) in out.items():
            print(f"=== {k}\n{reps[0][1]}\n{app}\n")
        return
    for fname, (reps, app, mod) in out.items():
        apply(ROOT / fname, reps, app, inside_module=mod)
        print("applied", fname)

if __name__ == "__main__":
    main("--apply" in sys.argv, check="--check" in sys.argv)
