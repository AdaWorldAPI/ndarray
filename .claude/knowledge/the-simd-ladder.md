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

```console
$ cargo tree -p encryption -i curve25519-dalek
curve25519-dalek v4.1.3
└── ed25519-dalek v2.2.0
    └── encryption v0.1.0 (crates/encryption)

$ cargo tree -p ndarray -i curve25519-dalek
error: package ID specification `curve25519-dalek` did not match any packages
```

**Only blake3 has a cycle**, because only blake3 is pulled by the *root*
package. Cargo names it explicitly and then silently falls back to the
registry crate with `[[patch.unused]]`, so a naive port compiles and has zero
effect — the same silent-no-op class as the chacha20 patch (see
`chacha20-vendoring-blast-radius.md`).

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
  external crate. Removes the cycle, the C question, and 2,910 lines of
  second-surface `core::arch` in one move. Verified against the official
  `test_vectors.json`. *No SIMD involved — correctness only.*
- **3b — throughput, optional.** Port `hash_many` onto the rung-2 shuffle
  surface for multi-chunk inputs. **This is where a benchmark matters**, and
  it is not on the critical path.

### 4. chacha20 — backend exists, effectively unreachable

`vendor/chacha20` already has an `ndarray_simd` backend. Measured: no image
and no CI job compiles it (`chacha20-vendoring-blast-radius.md`). And
upstream 0.10.1 now ships its own `avx512.rs` in the same niche, unmeasured
against ours. **Blocked on a benchmark, then on the fork-vs-vendored ruling.**

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

## Order

3a is the only rung that unblocks anything else, and it needs no measurement
to justify — it removes a dependency edge, a C build, and a second SIMD
surface regardless of throughput. Everything after it is independent:

```
3a (cut the cycle)  ──> 3b (hash_many throughput)   [needs a bench]
                    ──> 4  (chacha20)               [needs a bench + a ruling]
                    ──> 5  (dalek)                  [no blocker but appetite]
                    ──> 6  (u64 ARX / argon2)       [no blocker but appetite]
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
