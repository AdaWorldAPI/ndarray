# One spec, N backends — collapsing the SIMD type surface

> **Status: DESIGN.** Not implemented. The enabling measurement is done
> (TD-T22, merged 2026-07-28); the migration is a staged epic, not a PR.

## READ BY:
- Anyone about to hand-write a lane type in `src/simd_<arch>.rs`
- Anyone proposing to "add the missing types" to a backend
- `simd-savant`, `truth-architect`

---

## The measurement that makes this possible

TD-T22 (`.claude/knowledge/td-t22-asm-investigation.md`) established, with
assembly evidence: **under a pinned `target-cpu` baseline, LLVM compiles a
scalar-shaped lane loop to optimal packed SIMD.** On the ChaCha20 ARX triple
over the scalar-storage `U32x16`, the emitted code hits the AVX2 instruction
floor — 8 `vpaddd` for 64 u32 lanes — with no scalar op touching lane data,
and `rotate_left(16)` strength-reduced to `vpshufb`, which is *cheaper* than
the `shl|shr|or` triple a hand-written intrinsic emits.

The consequence is not "the polyfill is fine." It is: **for lane-wise
operations, the scalar spec IS the implementation, on every backend.**

## What exists today

| backend | types from a macro | hand-written structs |
|---|---|---|
| `simd_avx2` | 12 | 8 |
| `simd_avx512` | 0 | 21 |
| `simd_scalar` | 19 | 6 |
| `simd_neon` | 0 | 15 |
| `simd_wasm` | 0 | 7 |
| **total** | **31** | **57** |

Plus 72 `impl_bin_op!`-family invocations in `simd_avx512` layered on top of
its hand-written structs. **13,253 LoC across five files, three different
authoring strategies, and the same logical type written five times.**

Three strategies for one problem:
1. `avx2` / `scalar` — type-generating macros (`avx2_int_type!`, `impl_int_type!`)
2. `avx512` — hand-written struct + operator-generating macros
3. `neon` / `wasm` — fully hand-written

This is why adding one lane type is a five-file change, why ten AVX2 int
types are still "unlowered," and why the same boilerplate was hand-typed
twice for `U16x16` and then again for `U32x8`.

## The design

One declaration per logical type. Backends are *generated*, not authored.

```rust
simd_type! {
    name: U32x16, elem: u32, lanes: 16, repr: align(64),

    // Lane-wise ops. Emitted as the scalar loop form for EVERY backend.
    // LLVM vectorizes them under the pinned target-cpu baseline; the
    // an on-demand codegen probe proves it, per target, when the question
    // is actually open. NOT a standing CI job — see "On tooling" below.
    lanewise: [
        add(wrapping), sub(wrapping), mul(wrapping),
        and, or, xor, not,
        rotate_left, shl(zero_on_overshift), shr(zero_on_overshift),
        reduce_sum(wrapping),
    ],

    // ESCAPE HATCH. A per-backend intrinsic override may be added ONLY
    // with an oracle probe showing the generic form does not vectorize,
    // or a measured win the generic form cannot reach.
    intrinsic: {
        avx512: { rotate_left: "_mm512_rolv_epi32" },  // VPROLVD, 1 instr
    },
}
```

**The entry criterion is the whole point.** Today "should this be an
intrinsic?" is answered by intuition, and intuition said yes to a case where
LLVM was already at the instruction floor. Under this design the question is
answered by the oracle (`.claude/knowledge/simd-codegen-oracle/`): if the generic form vectorizes, the
override is rejected; if it doesn't, the override is justified and the probe
that justified it is committed alongside.

## What must NOT be generated — MEASURED, and my predictions were wrong

I predicted five classes LLVM could not synthesize from scalar source. The
oracle ran them. **Three of the five vectorized anyway.** Recorded here
because the wrong list is more instructive than the right one: the intuition
that "cross-lane / widening / saturating obviously needs intrinsics" is
exactly the intuition that produced a 700-line PR against a nonexistent gap.

| predicted scalar | actual | what LLVM emitted |
|---|---|---|
| `saturating_abs_i8x32` | **VECTORIZED** (4 packed / 0 scalar) | `vpxor` → `vpsubsb` (saturating 0−x) → `vpblendvb` — the exact abs+clamp trick the VPABSB correction documents, synthesized on its own |
| `widening_u16_to_f32` | **VECTORIZED** (6 packed / 0 scalar) | `vpmovzxwd` + `vcvtdq2ps` |
| `cross_lane_reverse_u8x64` | **VECTORIZED** (9 packed / 0 scalar) | `vbroadcasti128` + `vpshufb` + `vpermq` — it invented a cross-lane permute from a scalar index loop |
| `serial_dependent_chain` | scalar, as predicted | GPR `rorxl`/`addl`/`xorl` chain — a loop-carried dependency cannot vectorize |
| `gather_lookup_u8` | scalar, as predicted | pure `movzbl`/`movb`; no arithmetic at all |

So the genuine "cannot be generated" list is much shorter than assumed:

- **Loop-carried dependencies.** Structural; no compiler escapes them.
- **Gather / table lookup.** No contiguous load to widen.
- **u64 lane rotates** — see below. The one case where LLVM has the
  mechanism and declines to use it.

**Everything else measured so far is free.** `U16x16`'s hand-written
`permute2x128`/`blend_epi32` may still be justified — they are *explicit API
surface* consumers call directly, not something to be synthesized — but the
claim that cross-lane work inherently requires intrinsics is false.

## The u64 rotate — the first earned intrinsic override

Measured (`rot_u64x8`, `rot_u64x4`, `blake2b_g_u64x8`):

- `rot_u64x8` / `rot_u64x4`: **0 packed.** Each lane's `u64::rotate_right(n)`
  becomes a scalar GPR `rorq %cl, reg`, one per lane.
- `blake2b_g_u64x8`: the leading `a = a + b` vectorizes (`vpaddq` over both
  ymm halves); the moment a rotate appears LLVM extracts every lane
  (`vmovq`/`vpextrq`) and stays scalar (`rorxq`/`addq`/`xorq`) through all
  four rotate stages, reassembling only at the return.

The striking part: **the byte-granular amounts 32/24/16 stay scalar here**,
while the same class of amounts (16, 8) fold to `vpshufb` for u32. And AVX2
has `vpsllq`/`vpsrlq` — the exact shift-or mechanism LLVM *does* apply to
u32's rotate-by-12 and rotate-by-7. It has the tools and does not reach for
them on u64.

This is the crate's first intrinsic override that meets the entry criterion:
a probe showing the generic form does not vectorize. `_mm512_rorv_epi64`
(`VPROLVQ`) is a single instruction on AVX-512; AVX2/NEON/wasm get the
shift-or composition written explicitly.

## Staged migration (not one PR)

1. **Answer the codegen question once, in the design.** Run
   the oracle (`.claude/knowledge/simd-codegen-oracle/`) for the type family under consideration and
   record the result. This is a design activity, not a standing check.
2. **Characterize.** Run the oracle across x86-64-v3 / v4 / aarch64 / wasm32.
   Produce the per-target table of what vectorizes and what doesn't. That
   table *is* the specification of which intrinsic overrides are legitimate.
3. **Pilot on one type family.** The u32 lanes (`U32x8`, `U32x16`) — smallest
   blast radius, best-understood semantics, already measured.
4. **Migrate the 31 macro-generated types.** Mechanical; the macros already
   prove the shape is regular.
5. **Migrate the 57 hand-written types**, keeping every intrinsic the oracle
   justifies and deleting the rest. Expect the survivors to be concentrated
   in the cross-lane / widening / saturating families above.

## Invariants the design must preserve

- **`repr(align(64))` on every lane type.** Nine sites across
  `scalar`/`neon`/`wasm` carry it; it is a cacheline guarantee, not an
  accident. A `repr(transparent)` wrapper over `__m256i` LOSES it (measured:
  `U32x8` size 64→32, `U32x16` align 64→32). The spec must emit `align(64)`
  by default.
- **One API on every backend.** The generated surface is identical by
  construction — which structurally eliminates the class of bug where a
  method exists on x86_64 and nowhere else.
- **`U32x8` must not be `U32x16`'s building block** (operator ruling,
  2026-07-28). Composition, where needed, is an implementation detail of the
  generated backend, never a public half-width type standing in for the lane
  the substrate actually uses.
- **No `core::simd` or `hpc::` in a public signature; consumers only ever
  name `crate::simd::*`.** Backend-internal construction uses concrete
  backend types — a backend file is compiled even when its dispatch arm is
  not selected.

## What this buys

- Adding a lane type: one declaration instead of a five-file change.
- Ten currently-unlowered AVX2 int types: free.
- The "is this fast enough?" argument: answered by a twenty-minute
  measurement during design, instead of debated.
- ~13k LoC of hand-maintained backend code: substantially reduced, with the
  remainder being exactly the intrinsics that earn their place.

## On tooling — why this is NOT a CI job

`.claude/knowledge/simd-codegen-oracle/` is an **on-demand instrument**, deliberately
not wired into CI (operator ruling, 2026-07-29). A standing job that parses
assembly to catch a designer skipping the measurement is machinery guarding
against a lapse the design step should prevent — it institutionalizes the
lapse instead of fixing it, and it carries permanent cost: an asm parser, a
per-target baseline, and brittleness that already bit once (the job's first
run failed because it inherited its baseline from the ambient environment
rather than declaring it).

The question "does this vectorize?" is answered **once, during design**, and
the answer goes in a doc. It is not re-answered on every commit.

Run it when a codegen question is genuinely open:

```sh
sh .claude/knowledge/simd-codegen-oracle/run.sh          # host, x86-64-v3 baseline
```

Cross-targets take a triple argument, but each needs its own
`baseline-<triple>.toml` first — only `baseline-x86_64-v3.toml` exists today,
so `run.sh aarch64-unknown-linux-gnu` currently exits 90 by design. Producing
that baseline is step 2 of the migration below, not a prerequisite anyone has
met yet.
