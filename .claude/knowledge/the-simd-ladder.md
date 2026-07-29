# The ladder — one audited SIMD surface, nothing above it carrying its own

> **Status: PLAN, with measured rungs marked.** Every "proven" row below cites
> the measurement. Every "assumed" row says so.

## READ BY:
- Anyone continuing the BLAKE3 / chacha20 / dalek port work
- Anyone about to add a dependency that carries `core::arch`
- `simd-savant`, `truth-architect`

## The invariant

**All SIMD lives once, audited, inside `ndarray::simd`.** A dependency that
ships its own intrinsics is a second unaudited surface — which is what the
matryoshka pattern exists to prevent, and what `.cargo/config.toml` already
neutralizes by hand for one crate (`curve25519_dalek_backend = "serial"`).

The ladder is the ordered work of making that true rather than aspirational.

---

## The cycle map — the thing that orders everything else

Whether a dependency can consume `ndarray::simd` is decided by **which
ndarray package pulls it**, and this was measured, not assumed:

| dependency | entry point | `X → ndarray::simd` cycle? |
|---|---|---|
| `chacha20` | `crates/encryption` → `chacha20poly1305` → `chacha20` | **no** |
| `curve25519-dalek` | `crates/encryption` → `ed25519-dalek` → `curve25519-dalek` | **no** |
| `blake3` | **root `ndarray`** (`std` feature, 11 files in `src/hpc/`) | **YES** |

The evidence is the **positive** reverse tree, not an error message. Asking
for the full reverse-dependency tree of `curve25519-dalek` shows every path
to it, and the tree terminates at `encryption` — root `ndarray` is nowhere in
it:

```console
$ cargo tree -p encryption -i curve25519-dalek
curve25519-dalek v4.1.3
└── ed25519-dalek v2.2.0
    └── encryption v0.1.0 (crates/encryption)
```

**Control — the same command shape does produce a hit when the edge exists**,
so the method can discriminate:

```console
$ cargo tree -p ndarray -i blake3
blake3 v1.8.4
└── ndarray v0.17.2 (/workspace/ndarray)
```

An earlier version rested this on `cargo tree -p ndarray -i curve25519-dalek`
returning `error: package ID specification ... did not match any packages`.
That is **weak evidence and was corrected** (CodeRabbit, #268): a mistyped
name produces the byte-identical message —

```console
$ cargo tree -p ndarray -i curve25519-dalekk
error: package ID specification `curve25519-dalekk` did not match any packages
```

— so the error cannot distinguish "no such edge" from "no such package". Use
the positive tree, and keep a known-hit control beside it.

**Only blake3 has a cycle**, because only blake3 is pulled by the *root*
package. `cargo update -p blake3` reports it by naming the chain — the patched
`blake3` satisfies ndarray's dependency, and ndarray satisfies blake3's path
dependency.

Do **not** read `[[patch.unused]]` as that diagnostic. It only means the patch
was not selected, and the ordinary causes are a version that does not satisfy
the requirement or a stale lockfile. The two were conflated in an earlier
draft (codex, #268); treating the unused patch as a cycle signature teaches
the next reader to misdiagnose a plain stale patch.

The other two ride a pattern that **already works in this repo today**:
`crates/encryption` → `chacha20` → `ndarray(root)` is exactly that shape.

---

## Rungs

### 1. `U32x16` ARX — PROVEN, merged

Add / BitXor / `rotate_left`. Measured at the AVX2 instruction floor: 8
`vpaddd` for 64 u32 lanes, no scalar op touching lane data,
`rotate_left(16)` strength-reduced to `vpshufb`
(`td-t22-asm-investigation.md`).

### 2. `U32x16` shuffle surface — PROVEN, merged (#267)

`interleave_{lo,hi}_u{32,64}`, `concat_{lo,hi}_halves`. Semantics are x86's
per 256-bit half, parity-checked against the real intrinsics and
control-tested. Present on all six backends.

**The insight that made it right:** two `__m256i` fit in one `U32x16`, and
every operation in BLAKE3's `hash_many` is either lane-wise or confined
*within* a 128-/256-bit lane, so two 8-lane groups never interact and the
algorithm runs at DEGREE 16 with no cross-talk. An earlier attempt built this
on `U32x8` and was wrong (operator-corrected; see EPIPHANIES).

### 3. BLAKE3 — IN FLIGHT, and the only rung that must cut a cycle

**Scoping finding:** ndarray's own blake3 usage is *entirely single-input* —
`hash`, `Hasher::{new, new_keyed, update, finalize, finalize_xof().fill()}`.
**No `hash_many`.** So cutting the cycle needs only the **portable** core
(scalar `[u32; 16]` compress + the chunk/tree/XOF state machine), which uses
no SIMD at all.

That splits the rung:

- **3a — cut the cycle.** Transcribe portable BLAKE3 into ndarray, drop the
  external crate. Removes the cycle and ~2,910 lines of second-surface
  `core::arch`. Verified against the official `test_vectors.json`. *No SIMD
  involved — correctness only.*

  **It does NOT remove a C build.** `Cargo.toml:213` already sets
  `default-features = false, features = ["pure"]`, which removed all C/ASM
  compilation in #264 — 33 `.o` objects and `libblake3_avx512_assembly.a`,
  measured at the time. Earlier wording here credited 3a with removing a C
  build that a previous PR had already removed; codex caught it on #268.
  Overstating the benefit matters especially here, because the thing being
  weighed against it is transcribing a cryptographic implementation.
- **3b — throughput, optional.** Port `hash_many` onto the rung-2 shuffle
  surface for multi-chunk inputs. **This is where a benchmark matters**, and
  it is not on the critical path.

### 4. chacha20 — the wasm arm is live and CI-guarded; the AVX-512 arm is not

`vendor/chacha20`'s `ndarray_simd` backend has two arms, and they are in
very different states:

- **wasm32 + `simd128` — compiled and guarded.** `ci.yaml:141-142` builds
  `vendor/chacha20` for `wasm32-unknown-unknown` with
  `RUSTFLAGS="-C target-feature=+simd128"`, which selects `ndarray_simd`.
  The job comment calls it "the wasm matryoshka" guard, and it sits directly
  after the node parity step that proves the same `U32x16` lane bit-exact.
- **x86_64 + `avx512f` — compiled by nothing.** No image and no CI job
  (`chacha20-vendoring-blast-radius.md`).

An earlier version of this section said "no CI job compiles it" without
qualification. That was **false** — caught by codex on #268 — and is a fourth
instance of the scope-quantifier error this repo keeps hitting: the x86 path
was checked and the conclusion generalized to all targets.

Upstream 0.10.1 now ships its own `avx512.rs` in the same niche, unmeasured
against ours. **The AVX-512 arm is blocked on a benchmark, then on the
fork-vs-vendored ruling. The wasm arm is already working.**

### 5. curve25519-dalek — the largest untouched surface

57 `_mm*` intrinsic calls under 52 `unsafe` occurrences across
`backend/vector/avx2/field.rs` and `packed_simd.rs`. Currently neutralized
rather than ported: `.cargo/config.toml` sets
`curve25519_dalek_backend = "serial"`, which cfg's the whole `vector` module
out at `backend/mod.rs:42`.

That is the correct *reachability* answer and costs nothing today (X25519's
Montgomery ladder never touches the vector backend; the vector path serves
only Edwards multi-scalar work this crate doesn't do). Porting it onto
`ndarray::simd` is a real rung, has **no cycle**, and is gated on nothing
except appetite.

### 6. u64 ARX — the one *earned* intrinsic override, unbuilt

Measured: `rotate_left`/`rotate_right` do not exist on `U64x8`/`U64x4` on any
of the six backends, and a scalar u64 rotate loop does **not** vectorize —
0 packed, one scalar `rorq` per lane, even for byte-granular amounts, even
with compile-time-constant counts, on a target that has `vpsllq`/`vpsrlq` and
uses exactly that shift-or for u32 (`crypto-lane-status.md`).

This is the only place in the crate where a hand-written intrinsic currently
meets `simd-one-spec-design.md`'s entry criterion. BLAKE2b needs it, and
argon2 needs BLAKE2b. Nothing is built on it.

---

## Is 4096 bit / 512 byte viable as the DEFAULT unit? — MEASURED, yes

The substrate's canonical node is 4096 bit = 512 byte
(`key(16) | edges(16) | value(480)`). `U32x16` is 512 **bit** — one eighth of
it. So: does a node-wide unit, `[U32x16; 8]` = 128 u32 lanes, stay packed, or
does it drown in spill?

It cannot be a register type — 512 byte is 8 zmm / 16 ymm — so a node-wide
lane is necessarily a **tile** over the existing lane, exactly as `U32x16` is
already a polyfill over 2 ymm on avx2. The real question is whether the wider
unit costs anything.

Measured (`arx_node4096` vs `arx_lane512_x8`, same work, asserted bit-for-bit
identical in the driver):

| form | packed | scalar | **memory** | arithmetic emitted |
|---|---|---|---|---|
| `[U32x16; 8]` node-wide | 98 | 0 | **0** | 16 `vpaddd`, 16 `vpxor`, 16 `vpshufb`, 33 `vmovdqa` |
| `U32x16` × 8 | 81 | 0 | **0** | *identical* |

**The arithmetic is the same instruction-for-instruction, and neither spills.**
The 17-instruction delta is `16 vmovaps + 1 vxorps` — the dead zero-init of
`[U32x16::splat(0); 8]` in the loop form, which an implementation building the
array by value does not emit.

**So the node width is free — for the right shape of work.** The
discriminating property is *liveness*, not width:

- **Streaming / elementwise** (ARX, bitwise, add-mul): each of the 8 vectors
  is independent, LLVM processes them one at a time, and 128 lanes of state
  never need to be live simultaneously. **Zero spill.**
- **Whole-node-live** (a transpose): all vectors must be live at once.
  `transpose_16x16_composed` pays 41 memory ops for exactly this reason.

That is the rule to design against: a node-wide default unit is viable, and
the ops that must stay lane-width are the ones that need the whole node
resident.

## Order

**There is almost no ordering.** An earlier version drew arrows from 3a to
everything, which contradicted this section's own next sentence and would
have serialized independent work; codex caught it on #268.

The only real dependency is 3a → 3b, because `hash_many` needs an in-tree
BLAKE3 to live in. Everything else is genuinely parallel: chacha20's backend
already depends on `ndarray`, dalek has no cycle (established above), and the
u64 lane is implemented *inside* `ndarray`. None of the three is unblocked by
anything 3a does.

```text
3a (cut the cycle) ──> 3b (hash_many throughput)   [needs a bench]

4  (chacha20 AVX-512 arm)   [independent — needs a bench + a ruling]
5  (dalek)                  [independent — no blocker but appetite]
6  (u64 ARX / argon2)       [independent — no blocker but appetite]
```

## What is NOT claimed here

- **Not that any of this is faster.** Every rung is measured at
  *instruction class*, never at time. "Emits packed shuffles" is not "beats
  the intrinsic backend," and no benchmark of any rung against its upstream
  equivalent has been run.
- **Not that rung 5 is a defect today.** `serial` is a correct, deliberate,
  documented choice, and the vector backend serves no operation this crate
  performs.
- **Not that the ladder must be completed.** Rungs 4–6 are optional; only 3a
  removes something (a cycle, a C build, an unaudited surface).

## Open, needing a ruling rather than work

Carried from `chacha20-vendoring-blast-radius.md` so they are visible in one
place:

1. Fork vs vendored copy for `vendor/chacha20` — `AdaWorldAPI/stream-ciphers`
   is a bare upstream mirror at 0.10.1, vendored is 0.9.1 + one backend, and
   moving crosses a `cipher` major.
2. Should the Docker images build `encryption` at all? Neither does today, so
   the AEAD path ships in no image.
3. Should CI cover the chacha20 v4 arm? The exact command is recorded; adding
   the job is a product change.
4. `Cargo.toml:481`'s comment, whose main clause holds only for builds that
   select `encryption`.
5. `blake3` and `curve25519-dalek` both resolve from crates.io while
   AdaWorldAPI forks exist — two P0 violations. Rung 3a dissolves the blake3
   half by removing the dependency entirely.
