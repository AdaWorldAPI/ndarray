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
types; plain `u32` for NEON's per-lane loop; `v128_*` intrinsics for WASM).
The emitted text is pasted into the backend file between GEN markers by
`--apply`; it is committed source, and the generator is its provenance.

Lowering: with index `(a << 2) | (b << 1) | c`, the even bits of IMM are the
2-input table T0(a,b) (c = 0) and the odd bits T1(a,b) (c = 1);
    f = (!c & T0) | (c & T1)
with every 2-input table a <= 2-op closed form and six collapse shapes of the
outer combination. Worst case 7 ops; the naive 8-minterm form was up to 36.
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

def ladder(g, a, b, c, and_, or_, xor_, not_, andnot, indent):
    """The outer Shannon combination, branching only on IMM-derived values (let-bound: a `const` item
    cannot name the enclosing fn's IMM; after monomorphization these fold identically)."""
    L = []
    L.append(f"{indent}let t0: u8 = ((IMM & 1) | ((IMM >> 1) & 2) | ((IMM >> 2) & 4) | ((IMM >> 3) & 8)) as u8;")
    L.append(f"{indent}let t1: u8 = (((IMM >> 1) & 1) | ((IMM >> 2) & 2) | ((IMM >> 3) & 4) | ((IMM >> 4) & 8)) as u8;")
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
    return "\n".join(L)

OPL = dict(and_=lambda x,y: f"({x} & {y})", or_=lambda x,y: f"({x} | {y})", xor_=lambda x,y: f"({x} ^ {y})",
           not_=lambda x: f"!{x}", andnot=lambda x,y: f"({x} & !{y})")
NEONL = dict(and_=lambda x,y: f"vandq_u32({x}, {y})", or_=lambda x,y: f"vorrq_u32({x}, {y})",
             xor_=lambda x,y: f"veorq_u32({x}, {y})", not_=lambda x: f"vmvnq_u32({x})",
             andnot=lambda x,y: f"vbicq_u32({x}, {y})")
WASML = dict(and_=lambda x,y: f"v128_and({x}, {y})", or_=lambda x,y: f"v128_or({x}, {y})",
             xor_=lambda x,y: f"v128_xor({x}, {y})", not_=lambda x: f"v128_not({x})",
             andnot=lambda x,y: f"v128_andnot({x}, {y})")

# The NEON `ternlog` body region: from the lane unpack to the repack. Matched by
# regex so the generator can replace either the pre-generator per-lane loop or
# an earlier generated scalar-loop body.
NEON_SCALAR_LOOP_RE = re.compile(
    r"        let \(a, b, c\) = \(self\.to_array\(\), b\.to_array\(\), c\.to_array\(\)\);\n.*?        Self::from_array\(o\)"
    r"|        // GENERATED lowering \(regenerating\)\n.*?        \}\)\)", re.S)

BEGIN = "// GEN-TERNLOG-BEGIN (tools/gen_ternlog_bodies.py — regenerate, do not hand-edit)"
END = "// GEN-TERNLOG-END"

def body_lane_type(ty):
    """avx2 / scalar: the array-backed lane types own & | ^ ! and splat."""
    helper = f"ternlog_two_input_{ty.lower()}"
    return ladder(helper, "self", "b", "c", indent="        ", **OPL), \
           two_input_fn(helper, ty, OPS, f"{ty}::splat(0)", f"{ty}::splat(!0)")

def apply(path, replacements, appendix, inside_module=None):
    s = path.read_text()
    # Replacements are applied IN ORDER, one occurrence each: a backend file
    # may carry the same call line under two lane types (U64x8 first).
    for old, new in replacements:
        # Idempotent re-run: the body was already generated (and possibly
        # re-indented by `cargo fmt`), recognised by its marker comment.
        if new.strip().splitlines()[0].strip() in s:
            continue
        if isinstance(old, re.Pattern):
            assert old.search(s), (path.name, old.pattern[:60])
            s = old.sub(lambda _m: new, s, count=1)
            continue
        assert s.count(old) >= 1, (path.name, old[:60])
        s = s.replace(old, new, 1)
    # Drop any earlier trait-impl residue from the retired generic module.
    s = re.sub(r"\nimpl crate::simd_ternlog_lower::TernlogLanes for \w+ \{.*?\n\}\n", "\n", s, flags=re.S)
    # Strip a previous GEN block wherever it sits, then re-insert at the anchor.
    if BEGIN in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        s = s[:i].rstrip("\n") + "\n" + s[j:].lstrip("\n")
        s = s.rstrip("\n") + "\n"
    block = BEGIN + "\n" + appendix + "\n" + END
    if inside_module is None:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    else:
        # The helper must live INSIDE the cfg-gated backend module (wasm's
        # `pub mod wasm32_simd`), or it compiles on every host and fails to
        # resolve the v128 intrinsics. The module's own closing brace is the
        # first column-0 `}` after its opening line.
        head = s.index(inside_module)
        close = s.index("\n}\n", head)
        s = s[:close].rstrip("\n") + "\n\n" + block + "\n" + s[close:]
    path.write_text(s)

def main(write):
    out = {}
    # ── avx2 + scalar: U64x8 and U32x16 ──
    for fname in ("simd_avx2.rs", "simd_scalar.rs"):
        reps, helpers = [], []
        for ty in ("U64x8", "U32x16"):
            lad, helper = body_lane_type(ty)
            reps.append((f"        crate::simd_ternlog_lower::ternlog_lowered::<Self, IMM>(self, b, c)\n    }}",
                         f"        // GENERATED lowering (tools/gen_ternlog_bodies.py): Shannon-expand on `c`\n        // into two 2-input tables; <= 7 ops for any table, folded at compile time.\n{lad}\n    }}"))
            helpers.append(helper)
        out[fname] = (reps, "\n\n".join(helpers))
    # ── neon: U32x16 per-quad uint32x4_t (vandq/vorrq/veorq/vbicq/vmvnq) ──
    # A per-u32-lane loop through to_array()/from_array() SCALARIZED (measured
    # on the cross-compiled harness: 536 scalar vs 4 vector logic ops), which
    # fails rung 3 of the aarch64 ladder ("the assembly contains the expected
    # NEON operations and no unexpected scalarization"). The body must be
    # written in the backend's own intrinsic vocabulary, one quad at a time.
    # Why `unsafe` (measured 2026-09-13, not assumed): the NEON intrinsics are
    # safe `#[target_feature(enable = "neon")]` fns since Rust 1.87, but rustc
    # (1.98.1) still requires the CALLER to carry `#[target_feature(enable =
    # "neon")]` — "the neon target feature being enabled in the build
    # configuration does not remove the requirement to list it" (E0133). Putting
    # that attribute on the pub `ternlog` would propagate the same requirement
    # to every safe caller (simd_masking_ops, mask-risc), so the intrinsic
    # boundary is the ONE place unsafe lives, as narrowly as an expression, with
    # the same SAFETY reasoning every other intrinsic call in simd_neon.rs
    # carries. Everything above the backend stays `forbid(unsafe_code)`.
    NEON_SAFETY = ("NEON is a baseline feature of every aarch64 target this module compiles\n"
                   "for; these are pure register operations on values already in `uint32x4_t`.")
    lad = ladder("ternlog_two_input_u32x4", "x", "y", "z", indent="                ", **NEONL)
    neon_body = ("        // GENERATED lowering (tools/gen_ternlog_bodies.py), per 128-bit quad (NEON).\n"
                 "        Self(core::array::from_fn(|p| {\n"
                 "            let (x, y, z) = (self.0[p].0, b.0[p].0, c.0[p].0);\n"
                 "            // SAFETY: " + NEON_SAFETY.replace("\n", "\n            // ") + "\n"
                 "            U32x4(unsafe {\n" + lad + "\n            })\n        }))")
    out["simd_neon.rs"] = ([(NEON_SCALAR_LOOP_RE, neon_body)],
        two_input_fn("ternlog_two_input_u32x4", "uint32x4_t", NEON, "vdupq_n_u32(0)", "vdupq_n_u32(!0)",
                     attrs=('#[cfg(target_arch = "aarch64")]',), unsafe_reason=NEON_SAFETY))
    # ── wasm: U32x16 per-quad v128 ──
    lad = ladder("ternlog_two_input_v128", "x", "y", "z", indent="                ", **WASML)
    out["simd_wasm.rs"] = ([(
        "            for p in 0..4 {\n                let (x, y, z) = (V128Lanes(self.0[p].0), V128Lanes(b.0[p].0), V128Lanes(c.0[p].0));\n                parts[p] = crate::simd_ternlog_lower::ternlog_lowered::<V128Lanes, IMM>(x, y, z).0;\n            }",
        "            // GENERATED lowering (tools/gen_ternlog_bodies.py), per 128-bit quad.\n            for p in 0..4 {\n                let (x, y, z) = (self.0[p].0, b.0[p].0, c.0[p].0);\n                parts[p] = {\n" + lad.replace("\n", "\n    ") + "\n                };\n            }")],
        two_input_fn("ternlog_two_input_v128", "v128", WASM, "u32x4_splat(0)", "u32x4_splat(!0)", indent="    "))
    if not write:
        for k, (reps, app) in out.items():
            print(f"=== {k}\n{reps[0][1]}\n{app}\n")
        return
    for fname, (reps, app) in out.items():
        apply(ROOT / fname, reps, app,
              inside_module="pub mod wasm32_simd {" if fname == "simd_wasm.rs" else None)
        print("applied", fname)

if __name__ == "__main__":
    main("--apply" in sys.argv)
