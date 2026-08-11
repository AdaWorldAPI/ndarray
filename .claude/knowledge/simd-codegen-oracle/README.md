# SIMD codegen oracle — an on-demand instrument

Answers one question with evidence: **does this lane operation actually
vectorize, or does the source only look like it should?**

Lives here, not in `crates/` or `scripts/`, because it is an investigation
instrument rather than a product artifact. It ships no code, adds no
workspace member, and runs no CI job.

## When to use it

When a codegen question is genuinely open — before hand-writing intrinsics,
before "lowering" a scalar-looking polyfill, before claiming a lane is slow.
Answer it once, record the answer in a doc, move on.

**Not a standing check.** A permanent job that re-answers a settled question
on every commit is machinery guarding a lapse the design step should
prevent. Measure during design instead.

## Files

| file | role |
|---|---|
| `probes.rs` | 13 `#[inline(never)]` kernels over `ndarray::simd` types |
| `analyze.py` | extracts per-symbol instruction histograms from emitted asm, classifies, compares to baseline |
| `run.sh` | builds with `--emit asm`, locates the `.s`, invokes the analyzer |
| `baseline-x86_64-v3.toml` | expected class assertions per probe |

## Running it

```sh
sh .claude/knowledge/simd-codegen-oracle/run.sh                 # host, x86-64-v3 baseline
sh .claude/knowledge/simd-codegen-oracle/run.sh <target-triple> # cross-target
sh .claude/knowledge/simd-codegen-oracle/run.sh --verbose        # per-probe instruction detail
```

No setup: the script locates the ndarray checkout it lives in, builds a
throwaway crate from `probes.rs` under `$TMPDIR`, emits assembly, and hands
the `.s` to `analyze.py`. The scratch tree is removed on exit.

Adding a target means adding a `baseline-<triple>.toml`; the script refuses
to guess and exits 90 if one is missing.

## Two properties that make the result trustworthy

**The baseline is declared, never inherited.** `-C target-cpu=x86-64-v3` is
passed on the final rustc invocation. This is load-bearing: cargo's
`RUSTFLAGS` env var *replaces* `[target.'cfg(…)'].rustflags` from
`.cargo/config.toml` — they do not merge — so a tool that inherits its
baseline measures a different machine depending on where it runs. Verified:

```sh
$ cargo build -p ndarray --lib -v | grep -o 'target-cpu=[a-z0-9-]*'
target-cpu=x86-64-v3
$ RUSTFLAGS="-D warnings" cargo build -p ndarray --lib -v | grep -o 'target-cpu=[a-z0-9-]*'
          # (empty — silently dropped)
```

**Scalar arithmetic on lane data is separated from loop control**, so a
trip-counter `decl` is never miscounted as scalar lane work. The distinction
matters: "zero scalar instructions" is almost always false (a loop counter
exists); "no scalar op touches lane data" is the claim that means something.

## Recorded results

Measured on x86_64 + `x86-64-v3`, rustc 1.95.0. Full narrative in
`../td-t22-asm-investigation.md` and `../crypto-lane-status.md`.

| probe | packed | scalar (lane data) | lowering |
|---|---|---|---|
| `arx_rounds_u32x16` | 52 | 0 | `vpaddd`/`vpxor`/`vpshufb`; 8 `vpaddd` per round = the AVX2 floor for 64 lanes |
| `arx_u32x16` | 11 | 0 | `rotate_left(16)` → `vpshufb` |
| `reduce_u32x16` | 8 | 0 | logarithmic reduction tree |
| `fma_f32x16` | 28 | 0 | `vfmadd213ps` |
| `bitwise_u8x64` | 12 | 0 | packed |
| `saturating_abs_i8x32` | 4 | 0 | `vpxor`/`vpsubsb`/`vpblendvb` — synthesized the abs+clamp trick unprompted |
| `widening_u16_to_f32` | 6 | 0 | `vpmovzxwd` + `vcvtdq2ps` |
| `cross_lane_reverse_u8x64` | 9 | 0 | `vbroadcasti128`/`vpshufb`/`vpermq` — invented a cross-lane permute from a scalar index loop |
| **`rot_u64x8`** | **0** | 8 | scalar `rorq %cl`, one per lane |
| **`rot_u64x4`** | **0** | 4 | scalar `rorq %cl`, one per lane |
| **`blake2b_g_u64x8`** | 22 | ~88 | packed leading add; scalar through all four rotates |
| `gather_lookup_u8` | 0 | 0 | `movzbl` chain, no arithmetic |
| `serial_dependent_chain` | 0 | 27 | loop-carried dependency |

**The headline:** LLVM vectorizes far more than intuition suggests —
including cross-lane permutes, widening converts, and saturating
arithmetic, all from plain scalar loops. It does **not** vectorize u64
rotates, even for byte-granular amounts that fold to `vpshufb` at u32
width, and even though `vpsllq`/`vpsrlq` are available and it uses exactly
that shift-or for u32 `rotate_left(12)`/`(7)`.

That single row is why the tool was worth building: it is the one place a
hand-written intrinsic is currently justified, and it is what argon2 needs
(BLAKE2b is a 64-bit ARX cipher).

## A bug worth remembering

`analyze.py` originally returned its failure bitmask as the process exit
status. Unix truncates exit status to the low 8 bits, so a failure confined
to probe index ≥ 8 exits `0x100`/`0x200`/… and the shell observes **0** —
printing `FAILURES` while reporting success. Alphabetically that covered
both u64-rotate probes. Now it exits 1 on any failure, with the bitmask
kept as diagnostic text.
