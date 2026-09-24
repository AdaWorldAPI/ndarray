#!/usr/bin/env python3
"""Generate the per-CPU SIMD feature inventory from LLVM's own TableGen source.

WHY THIS EXISTS. ndarray dispatches at compile time on `cfg(target_feature)`.
The set of features a given `-C target-cpu=X` turns on is not ours to decide:
it is LLVM's processor table, compiled into rustc. That table is public source
in llvm-project (`X86.td`, `AArch64Processors.td`, `AArch64Features.td`), so
it can be READ instead of maintained by hand or re-derived by compiling probes.

This script reads those `.td` files at the `llvmorg-*` tag that matches the
LLVM rustc itself embeds (from `rustc -vV`, so `rust-toolchain.toml` pins it),
resolves every processor's feature list (the `!listconcat` inheritance chains,
the `foreach P = [...]` alias blocks, `ProcessorAlias`, and each feature's
transitive implied features), and writes:

  tools/llvm-inventory/inventory.json        full per-CPU feature sets
  .claude/knowledge/llvm-cpu-inventory.md    the table a human reads

For each CPU it also records which ndarray backend the `simd.rs` cfg ladder
selects, and whether each SIMD primitive that the crypto references
(curve25519-dalek, poly1305, argon2) need has its instruction on that CPU.

The primitive table below names, per primitive, the LLVM instruction def and
the LLVM predicate/feature that gates it. Those names are ASSERTED to exist in
the fetched `.td` files, so an LLVM rename fails this script loudly instead of
silently leaving a stale row.

Like `gen_ternlog_bodies.py`, the output is committed and this script is its
provenance.

  python3 tools/gen_llvm_inventory.py --write          regenerate the outputs
  python3 tools/gen_llvm_inventory.py                  check they are current (exit 1 if stale)
  python3 tools/gen_llvm_inventory.py --verify-rustc   compare EVERY CPU against rustc's own
                                                       `--print cfg -C target-cpu=...`

Sources are fetched from raw.githubusercontent.com into
`target/llvm-inventory/<tag>/` (gitignored); `--llvm-src DIR` reads a local
llvm-project checkout instead.

CAVEAT. rustc builds against rust-lang's LLVM fork, not the upstream tag. The
fork carries patches, so the upstream tag is an approximation of what rustc
embeds. `--verify-rustc` is the check that the approximation holds for the
features rustc exposes.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "tools" / "llvm-inventory" / "inventory.json"
OUT_MD = ROOT / ".claude" / "knowledge" / "llvm-cpu-inventory.md"
CACHE = ROOT / "target" / "llvm-inventory"

TD = {
    "x86": [
        "llvm/lib/Target/X86/X86.td",
        "llvm/lib/Target/X86/X86InstrAVX512.td",
        "llvm/lib/Target/X86/X86InstrSSE.td",
        "llvm/lib/Target/X86/X86InstrPredicates.td",
    ],
    "aarch64": [
        "llvm/lib/Target/AArch64/AArch64Features.td",
        "llvm/lib/Target/AArch64/AArch64Processors.td",
        "llvm/lib/Target/AArch64/AArch64InstrInfo.td",
    ],
}
RUST_TARGET = {"x86": "x86_64-unknown-linux-gnu", "aarch64": "aarch64-unknown-linux-gnu"}

# Primitives the crypto references need, and what gates each one in LLVM.
#   insn: (file basename, regex that must match the instruction's def)
#   gate: (file basename, regex that must match the predicate/feature def)
#   needs: LLVM feature names that must ALL be present on the CPU
# Several rows per primitive = alternative encodings; a CPU has the primitive
# if any row's `needs` is satisfied.
PRIMITIVES = [
    ("mul_lo32", "x86", "PMULUDQ (xmm, SSE2)", ("X86InstrSSE.td", r"defm PMULUDQ\b"), None, ["sse2"]),
    ("mul_lo32", "x86", "VPMULUDQ (ymm, AVX2)", ("X86InstrSSE.td", r"defm PMULUDQ\b"), None, ["avx2"]),
    ("mul_lo32", "x86", "VPMULUDQ (zmm, AVX-512F)", ("X86InstrAVX512.td", r"defm VPMULUDQ\b"), None, ["avx512f"]),
    ("mul_lo32", "aarch64", "UMULL / UMULL2", ("AArch64InstrInfo.td", r"defm UMULL\s*:"), None, ["neon"]),
    ("shift_var_u32", "x86", "VPSLLVD / VPSRLVD", ("X86InstrSSE.td", r"defm VPSLLVD\b"), None, ["avx2"]),
    ("shift_var_u32", "aarch64", "USHL", ("AArch64InstrInfo.td", r"defm USHL\s*:"), None, ["neon"]),
    ("permute_var_u32", "x86", "VPERMD", ("X86InstrSSE.td", r"defm VPERMD\b"), None, ["avx2"]),
    ("permute_var_u32", "aarch64", "TBL", ("AArch64InstrInfo.td", r"defm TBL\s*:"), None, ["neon"]),
    ("ifma52", "x86", "VPMADD52LUQ/HUQ (EVEX)", ("X86InstrAVX512.td", r"defm VPMADD52LUQ\b"),
     ("X86InstrPredicates.td", r"def HasIFMA\b"), ["avx512ifma"]),
    ("ifma52", "x86", "VPMADD52LUQ/HUQ (VEX)", ("X86InstrSSE.td", r"defm VPMADD52LUQ\b"),
     ("X86InstrPredicates.td", r"def HasAVXIFMA\b"), ["avxifma"]),
    ("clmul", "x86", "PCLMULQDQ", ("X86InstrSSE.td", r"PCLMULQDQ"),
     ("X86InstrPredicates.td", r"def HasPCLMUL\b"), ["pclmul"]),
    ("clmul", "x86", "VPCLMULQDQ (zmm)", ("X86InstrAVX512.td", r"defm VPCLMULQDQZ\b"),
     ("X86InstrPredicates.td", r"def HasVPCLMULQDQ\b"), ["vpclmulqdq", "avx512f"]),
    ("clmul", "aarch64", "PMULL (64x64->128)", ("AArch64InstrInfo.td", r"defm PMULL\s*:"),
     ("AArch64Features.td", r'def FeatureAES\b[^;]*FEAT_PMULL'), ["aes"]),
    ("aes_round", "x86", "AESENC", ("X86InstrSSE.td", r"defm VAESENC\b"),
     ("X86InstrPredicates.td", r"def HasAES\b"), ["aes"]),
    ("aes_round", "x86", "VAESENC (zmm)", ("X86InstrAVX512.td", r"defm VAESENC\b"),
     ("X86InstrPredicates.td", r"def HasVAES\b"), ["vaes", "avx512f"]),
    ("aes_round", "aarch64", "AESE + AESMC", ("AArch64InstrInfo.td", r"def AESErr\b"),
     ("AArch64Features.td", r"def FeatureAES\b"), ["aes"]),
]
PRIM_ORDER = ["mul_lo32", "shift_var_u32", "permute_var_u32", "ifma52", "clmul", "aes_round"]

# Features reported in the summary columns.
SHOWN = {"x86": ["avx2", "avx512f", "avx512vl", "avx512ifma", "avxifma", "pclmul",
                 "vpclmulqdq", "aes", "vaes", "gfni"],
         "aarch64": ["neon", "dotprod", "i8mm", "aes", "sha2", "sve2"]}


def rustc(*args):
    return subprocess.run(["rustc", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def llvm_tag():
    m = re.search(r"LLVM version: (\d+\.\d+\.\d+)", rustc("-vV"))
    if not m:
        sys.exit("cannot read the LLVM version from `rustc -vV`")
    return "llvmorg-" + m.group(1)


def load(tag, src_dir):
    out = {}
    for arch, files in TD.items():
        for rel in files:
            name = pathlib.PurePosixPath(rel).name
            if src_dir:
                text = (pathlib.Path(src_dir) / rel).read_text()
            else:
                cached = CACHE / tag / name
                if not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    url = f"https://raw.githubusercontent.com/llvm/llvm-project/{tag}/{rel}"
                    with urllib.request.urlopen(url, timeout=60) as r:
                        cached.write_bytes(r.read())
                text = cached.read_text()
            out[name] = text
    return out


def strip_comments(t):
    return re.sub(r"//[^\n]*", "", t)


def expand_foreach(t):
    """`foreach P = ["a", "b"] in { def : ProcModel<P, ...>; }` -> one def per name."""
    extra = []
    for m in re.finditer(r"foreach\s+(\w+)\s*=\s*\[([^\]]*)\]\s*in\s*\{(.*?)\n\}", t, re.S):
        var, body = m.group(1), m.group(3)
        for name in re.findall(r'"([^"]+)"', m.group(2)):
            extra.append(re.sub(r"<\s*" + var + r"\s*,", f'<"{name}",', body))
    return t + "\n" + "\n".join(extra)


def split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def resolve(text):
    """Every processor -> set of LLVM feature names (implied features included)."""
    t = expand_foreach(strip_comments(text))
    features = {}
    # Every feature-like def: the NAME is its first string argument, and the
    # IMPLIED features are its first plain list argument. That holds for
    # SubtargetFeature<name, attr, value, desc, [implied]>, Extension*<name,
    # spelling, FEAT_, desc, [implied]> and Architecture64<major, minor,
    # profile, name, [implied], !listconcat(defaults)>. Taking the LAST list
    # instead reads Architecture64's -march defaults as implied features.
    for m in re.finditer(r"def\s+(\w+)\s*:\s*(\w+)<(.*?)>\s*;", t, re.S):
        rec, args = m.group(1), split_top(m.group(3))
        name = next((a.strip()[1:-1] for a in args if re.fullmatch(r'\s*"[^"]*"\s*', a)), None)
        if name is None:
            continue
        lst = next((a for a in args if a.strip().startswith("[")), "")
        features.setdefault(rec, (name, re.findall(r"\w+", lst)))
    # AArch64 architecture levels also carry DEFAULT EXTENSIONS (their last
    # argument). They are not implied features, but rustc's cfg for
    # `-C target-cpu` includes them: without them apple-m4 misses `ssbs`
    # (a v8.5 default) and oryon-1 misses `fp16`/`fhm`, and --verify-rustc
    # reports exactly those rows.
    arch_defaults = {m.group(1): split_top(m.group(2))[-1] for m in
                     re.finditer(r"def\s+(\w+)\s*:\s*Architecture64<(.*?)>\s*;", t, re.S)}

    def defaults(expr):
        expr = expr.strip()
        if expr.startswith("!listconcat("):
            return [r for p in split_top(expr[len("!listconcat("):-1]) for r in defaults(p)]
        if expr.startswith("!listremove("):
            base, drop = split_top(expr[len("!listremove("):-1])
            gone = set(defaults(drop))
            return [r for r in defaults(base) if r not in gone]
        if expr.endswith(".DefaultExts"):
            return defaults(arch_defaults[expr[:-len(".DefaultExts")]])
        if expr.startswith("["):
            return re.findall(r"\w+", expr)
        # Scraping words out of an unknown expression is how `!listremove`
        # once leaked the architecture def's own name (and its whole implied
        # chain) into cortex-r82. Refuse instead.
        sys.exit(f"unsupported TableGen expression in DefaultExts: {expr[:60]}")

    body = re.search(r"def\s+ProcessorFeatures\s*\{(.*?)\n\}", t, re.S).group(1)
    lists = {m.group(1): m.group(2) for m in
             re.finditer(r"list<SubtargetFeature>\s+(\w+)\s*=\s*(.*?);", body, re.S)}

    def ev(expr):
        expr = expr.strip()
        if expr.startswith("!listconcat("):
            return [r for p in split_top(expr[len("!listconcat("):-1]) for r in ev(p)]
        if expr.startswith("!listremove("):
            base, drop = split_top(expr[len("!listremove("):-1])
            gone = set(ev(drop))
            return [r for r in ev(base) if r not in gone]
        if expr.startswith("!"):
            sys.exit(f"unsupported TableGen operator in a processor list: {expr[:60]}")
        if expr.startswith("["):
            return re.findall(r"\w+", expr)
        return ev(lists[expr.replace("ProcessorFeatures.", "")])

    def closure(recs):
        """Transitive implied features. Tune records (`Tune*`, `Tuning*`) are
        followed but not reported: they are scheduling knobs, yet some imply
        real ISA features (TuneOryon implies FeatureFullFP16/FP16FML/SPE)."""
        seen, stack = set(), list(recs)
        while stack:
            r = stack.pop()
            if r in seen or r not in features:
                continue
            seen.add(r)
            stack += features[r][1]
        return {features[r][0] for r in seen if not r.startswith("Tun")}

    cpus = {}
    for m in re.finditer(r"def\s*:\s*Proc(?:essor)?Model<\s*\"([^\"]+)\"\s*,\s*\w+\s*,(.*?)>\s*;", t, re.S):
        args = split_top(m.group(2))
        recs = ev(args[0])
        recs += [d for r in recs if r in arch_defaults for d in defaults(arch_defaults[r])]
        # The tune list (next argument) is applied too, and its records'
        # implied ISA features are enabled; see closure().
        if len(args) > 1 and args[1].strip().startswith(("[", "!", "ProcessorFeatures.")):
            recs += ev(args[1])
        cpus[m.group(1)] = closure(recs)
    for m in re.finditer(r'ProcessorAlias<\s*"([^"]+)"\s*,\s*"([^"]+)"', t):
        if m.group(2) in cpus:
            cpus[m.group(1)] = cpus[m.group(2)]
    return cpus


def backend(arch, feats):
    """Mirror of the `simd.rs` cfg ladder (read by hand, not parsed)."""
    if arch == "x86":
        if "avx512f" in feats:
            return "simd_avx512"
        return "simd_avx2" if "avx2" in feats else "simd_avx2 (no avx2: array polyfill)"
    return "simd_neon" + (" +dotprod" if "dotprod" in feats else "")


def check_primitive_sources(src):
    for prim, arch, label, insn, gate, _ in PRIMITIVES:
        for where in (insn, gate):
            if where and not re.search(where[1], strip_comments(src[where[0]]), re.S):
                sys.exit(f"LLVM source no longer matches {prim}/{label}: /{where[1]}/ not in {where[0]}")


def primitives_for(arch, feats):
    out = {}
    for prim in PRIM_ORDER:
        rows = [r for r in PRIMITIVES if r[0] == prim and r[1] == arch]
        hit = [r[2] for r in rows if all(n in feats for n in r[5])]
        out[prim] = hit
    return out


def build(tag, src):
    check_primitive_sources(src)
    inv = {"llvm_tag": tag, "arch": {}}
    for arch in TD:
        text = "\n".join(src[pathlib.PurePosixPath(f).name] for f in TD[arch]
                         if "Processors" in f or f.endswith("X86.td") or "Features" in f)
        cpus = resolve(text)
        inv["arch"][arch] = {
            cpu: {"features": sorted(f), "backend": backend(arch, f),
                  "primitives": primitives_for(arch, f)}
            for cpu, f in sorted(cpus.items())
        }
    return inv


def render_md(inv):
    tag = inv["llvm_tag"]
    lines = [
        "# Per-CPU SIMD inventory — generated from LLVM's TableGen source",
        "",
        f"> **GENERATED** by `tools/gen_llvm_inventory.py` from llvm-project `{tag}`",
        "> (the tag matching `rustc -vV` under this repo's pinned toolchain). Do not edit by hand:",
        "> run `python3 tools/gen_llvm_inventory.py --write`. `--verify-rustc` checks every row",
        "> against rustc's own `--print cfg -C target-cpu=...`.",
        "",
        "`backend` mirrors the `simd.rs` cfg ladder. Primitive columns list the instruction(s)",
        "that implement the primitive on that CPU; `—` means the CPU has none and a portable",
        "fallback is required.",
        "",
    ]
    for arch, cpus in inv["arch"].items():
        shown = SHOWN[arch]
        lines += [f"## {arch} ({len(cpus)} CPUs)", ""]
        head = ["cpu", "backend"] + shown + PRIM_ORDER
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for cpu, row in cpus.items():
            feats = set(row["features"])
            cells = [f"`{cpu}`", row["backend"]] + ["✓" if f in feats else "" for f in shown]
            cells += ["<br>".join(row["primitives"][p]) or "—" for p in PRIM_ORDER]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def rust_to_llvm_names(llvm_util_rs):
    """rustc's own Rust-name -> LLVM-name table, parsed from `to_llvm_features`."""
    body = llvm_util_rs[llvm_util_rs.index("fn to_llvm_features"):]
    table = {}
    for arch, marker in (("aarch64", "Arch::AArch64"), ("x86", "Arch::X86")):
        start = body.index(marker)
        end = body.index("Arch::", start + len(marker) + 20)
        section = body[start:end]
        table[arch] = {r: l for r, l in re.findall(
            r'"([^"]+)"\s*=>\s*Some\(LLVMFeature::(?:new|with_dependencies)\(\s*"([^"]+)"', section)}
    return table


def verify_rustc(inv, names):
    """Compare every CPU with rustc's `--print cfg`, over the features stable rustc prints."""
    bad = 0
    for arch, cpus in inv["arch"].items():
        to_rust = {}
        for r, l in names[arch].items():
            to_rust.setdefault(l, set()).add(r)
        cfg = {}
        for cpu in cpus:
            try:
                out = rustc("--print", "cfg", "--target", RUST_TARGET[arch], "-C", f"target-cpu={cpu}")
            except subprocess.CalledProcessError:
                print(f"  {arch}/{cpu}: rustc rejects this cpu name (skipped)")
                continue
            cfg[cpu] = set(re.findall(r'target_feature="([^"]+)"', out))
        exposed = set().union(*cfg.values())      # features stable rustc can print at all
        # Features rustc prints for EVERY cpu are target-level, not per-cpu:
        # the x86_64 ABI requires sse/sse2, so rustc reports them even for
        # LLVM's featureless `generic`. They come from the target, not from
        # any processor list, so they are added to both sides.
        baseline = set.intersection(*cfg.values())
        for cpu, rust in cfg.items():
            mine = ({r for f in cpus[cpu]["features"] for r in to_rust.get(f, {f})} & exposed) | baseline
            if mine != rust:
                bad += 1
                print(f"  MISMATCH {arch}/{cpu}: llvm-only={sorted(mine - rust)} rustc-only={sorted(rust - mine)}")
        print(f"{arch}: {len(cfg)} CPUs compared against rustc over {len(exposed)} exposed features")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the generated outputs")
    ap.add_argument("--verify-rustc", action="store_true", help="compare every CPU with rustc's table")
    ap.add_argument("--tag", help="llvm-project tag (default: from rustc -vV)")
    ap.add_argument("--llvm-src", help="local llvm-project checkout instead of fetching")
    a = ap.parse_args()

    tag = a.tag or llvm_tag()
    inv = build(tag, load(tag, a.llvm_src))
    js = json.dumps(inv, indent=1, sort_keys=True) + "\n"
    md = render_md(inv)

    if a.verify_rustc:
        release = re.search(r"release: (\S+)", rustc("-vV")).group(1)
        cached = CACHE / f"rust-{release}" / "llvm_util.rs"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            url = ("https://raw.githubusercontent.com/rust-lang/rust/"
                   f"{release}/compiler/rustc_codegen_llvm/src/llvm_util.rs")
            with urllib.request.urlopen(url, timeout=60) as r:
                cached.write_bytes(r.read())
        bad = verify_rustc(inv, rust_to_llvm_names(cached.read_text()))
        print("rustc agrees on every compared CPU" if not bad else f"{bad} CPU(s) disagree")
        if bad:
            sys.exit(1)
    if a.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(js)
        OUT_MD.write_text(md)
        n = {k: len(v) for k, v in inv["arch"].items()}
        print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)} ({tag}, {n})")
    elif not a.verify_rustc:
        stale = [p for p, want in ((OUT_JSON, js), (OUT_MD, md))
                 if not p.exists() or p.read_text() != want]
        if stale:
            sys.exit("stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale)
                     + " (run with --write)")
        print(f"inventory is current ({tag})")


if __name__ == "__main__":
    main()
