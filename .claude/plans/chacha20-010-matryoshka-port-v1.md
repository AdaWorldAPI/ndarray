# chacha20 0.9.1 → 0.10.1 matryoshka port — spec v1

> **Status: sources read, not built.** Structural claims cite a file:line that
> exists in this container. Points that could not be verified from source here
> are marked **UNVERIFIED** and are not guessed.

## 0. Verification ledger

| artefact | where | status |
|---|---|---|
| the fork | `vendor/chacha20/` | read, 10 files |
| pristine chacha20 0.9.1 | `~/.cargo/registry/cache/*/chacha20-0.9.1.crate` | read via `tar -xzO`, diffed |
| upstream chacha20 0.10.1 | `~/.cargo/registry/src/*/chacha20-0.10.1/` | read, full source |
| `cipher` 0.5.2 | fetched `.crate` | read `src/lib.rs`, `src/stream/core_api.rs`, manifest |
| `cipher` 0.4.4 | registry | read `src/stream_core.rs` |
| `chacha20poly1305` 0.10.1 / 0.11.0 | registry + index + `.crate` | manifests + `src/lib.rs` |
| `poly1305` 0.9.0 | fetched `.crate` | `src/backend.rs`, manifest |
| `aead` 0.6.1 | fetched `.crate` | grep of exports only — **partial** |
| MedCare-rs / OGAR / a2ui-rs lockfiles | `/home/user/*/Cargo.lock` | read |

**UNVERIFIED, left open:** (a) `aead` 0.6's exact `encrypt`/`decrypt` signatures;
(b) whether cargo accepts the "patch same crate twice under a rename" form
(Option B, §5.4) — settle with `cargo metadata`, not by reading; (c) throughput
of `ndarray_simd` vs upstream's new `backends/avx512.rs` — unmeasured on both
sides, and it decides whether the x86 arm is worth porting at all.

## 1. The delta to re-apply

Diffed against pristine 0.9.1: the AdaWorldAPI delta is **4 source hunks + 1 new
file + a manifest rewrite**. `legacy.rs`, `xchacha.rs`, `backends/{soft,sse2,avx2,neon}.rs`,
`tests/mod.rs` are **byte-identical** to upstream.

- **Hunk 1** `src/lib.rs:110-113` — `#![allow(unexpected_cfgs)]`. **Dies in the
  port**; 0.10 declares cfgs via `check-cfg` in the manifest instead.
- **Hunk 2** `src/backends.rs:6-23` — declares `ndarray_simd` under
  `any(all(x86_64, avx512f, …), all(wasm32, simd128))`. **Structural problem:**
  this arm *replaces* the x86 subtree, so `soft`/`avx2`/`sse2` are not compiled
  at all. Survivable at 0.9.1; **not** at 0.10 (see Hunk 6).
- **Hunk 3** `src/lib.rs:161-172` — `type Tokens = ();` (the "no runtime probe" property).
- **Hunk 4** `src/lib.rs:233-242` — the mirror: `let tokens = ();`.
- **Hunk 5** `src/lib.rs:273-290` — dispatch `f.call(&mut backends::ndarray_simd::Backend(self))`.
- **New file** `src/backends/ndarray_simd.rs` (110 lines) — vertical layout, 16
  blocks in parallel, word `w` of every block across the 16 lanes of one
  `U32x16`, counter word 12 carrying lane index (`:76-81`). No cross-lane
  shuffle, no `unsafe`, no intrinsics. `ParBlocksSize = U16` (`:30-32`).
- **Manifest** — `rust-version` 1.95, empty `[workspace]`, and the one
  non-upstream dependency: `ndarray` as a target-dep with `features = ["std"]`
  (required — `ndarray::simd` is `#[cfg(feature = "std")]`, `src/lib.rs:239-241`).
- **Hunk 6 (NEW at 0.10)** `src/rng.rs:49-90` carries a **second independent copy
  of the backend dispatch tree**, including `let (avx2_token, sse2_token) = self.tokens;`.
  With `Tokens = ()` and `feature = "rng"` active this **fails to compile**. The
  delta grows from four hunks to five, in a file that did not exist before.

## 2. `cipher 0.4.4 → 0.5.2`, as it touches the fork

**The backend trait's shape is unchanged** (`cipher-0.5.2/src/stream/core_api.rs:10-31`
vs `cipher-0.4.4/src/stream_core.rs:10-31`): same three methods, same defaults,
`ParBlocksSizeUser`/`BlockSizeUser` verbatim. The `ks16` algorithm ports with
**zero changes**.

Renames: `StreamBackend` → `StreamCipherBackend`; `StreamClosure` →
`StreamCipherClosure`; `generic_array::GenericArray` → `cipher::array::Array`
(hybrid-array); `crate::Block` → `crate::chacha::Block`.

Semantic changes:

- **(a)** `R: Unsigned` → `R: Rounds` with `const COUNT` (`lib.rs:53-80`):
  `ks16<R: Unsigned>` → `<R: Rounds>`, `R::USIZE` → `R::COUNT`.
- **(b)** `ChaChaCore` gains `V: Variant` (`lib.rs:119-127`, `variants.rs:9-84`):
  every impl becomes `impl<R: Rounds, V: Variant> … for Backend<'_, R, V>`.
- **(c) THE CORRECTNESS TRAP — the counter is 64-bit for `Variant = Legacy`**
  (`variants.rs:66`). It occupies `state[12..14]`. `ndarray_simd.rs:41`, `:50`,
  `:76-81` only ever touch `state[12]`. Consequences: `orig[13]` can no longer be
  a `splat` (lanes straddle a carry within 15 of a 2^32 boundary); both advance
  sites need the 64-bit carry under a `size_of::<V::Counter>() == 8` guard.
  **Invisible to RFC 8439** (IETF vectors are `Counter = u32`) — caught only by
  legacy KATs. Upstream's shape: `backends/avx512.rs:47-56`, `:66-79`.
- **(d) Opportunity:** with `ParBlocksSize = U16` the default `gen_tail_blocks`
  wastes up to 240 block computations per tail. Override it (upstream does:
  `avx512.rs:315-338`).

`cipher 0.5` has **no `std` feature** — the fork's `std = ["cipher/std"]` cannot
survive (0.10 removed `std` outright).

## 3. The wasm question — the premise needs correcting

**`cpufeatures` is NOT unconditional at 0.10, and never was at 0.9.** It is
`[target.'cfg(any(target_arch = "x86_64", target_arch = "x86"))']`-scoped at
**both** versions. A `wasm32-unknown-unknown` build of stock upstream 0.10.1
pulls no `cpufeatures` and compiles no per-arch intrinsics — the cfg chain
(`backends.rs:5-31`) falls through to `soft`.

So the framing "stock upstream pulls cpufeatures, which wasm cannot use" is
**not correct as a compilability claim**. What the fork actually buys on wasm is
**throughput**: a 16-wide `U32x16` backend that `ndarray::simd` lowers to the
native `[U32x4; 4]` simd128 lane (`src/simd.rs:404-406`) instead of the
1-block-at-a-time `soft` backend.

Consequence for the port: keeping the wasm path free of `cpufeatures` costs
**nothing** — there is nothing to keep it free of. The wasm work is exactly one
thing: preserve the `all(wasm32, simd128)` arm in both cfg chains and keep the CI
gate green.

Also: 0.10 raises edition to 2024 / MSRV 1.85. Repo pins 1.97.1, so satisfied —
but the fork manifest's `edition = "2021"` / `rust-version = "1.95"` must move.

## 4. The new dependencies — all three optional

| dep | req | optional | gate |
|---|---|---|---|
| `cipher` | `^0.5` (`stream-wrapper`) | yes | default feature `cipher` |
| `rand_core` | `^0.10` | yes | feature `rng` |
| `zeroize` | `^1.8.1` | yes | implicit `dep:` feature |
| `cpufeatures` | `^0.3` | no | **x86 target-dep only** |

**`rand 0.10`'s `std_rng` is a default feature and enables `chacha20/rng`.** So
the moment the fork carries 0.10 in a graph with default-featured `rand`,
`src/rng.rs` compiles — which promotes Hunk 6 from theoretical to mandatory.
Already true in MedCare-rs (`Cargo.lock`: `rand 0.10.2 → chacha20 0.10.1`).

## 5. The resolution problem, and the trap in the obvious fix

### 5.1 The half-applied patch is in MedCare-rs, not ndarray

MedCare's lock carries **two** chacha20 nodes: `0.9.1` from the ndarray fork
(via `chacha20poly1305 0.10.1`) and `0.10.1` from the **registry** (via
`rand 0.10.2`). No warning.

**ndarray's own workspace cannot reproduce it** — its only `^0.10` requirer
(`rand` via `quickcheck`) resolves without `std_rng`, and `crates/burn`, which
does declare it, is excluded. That shapes falsifier F3.

### 5.2 THE TRAP — the obvious fix inverts the bug

`chacha20poly1305 0.10.1` requires **`chacha20 ^0.9`**. So naively bumping the
fork to 0.10.1 produces:

- `chacha20poly1305` no longer matches the patch → resolves **registry 0.9.1** →
  the AEAD, the actual production consumer, **loses the fork entirely**;
- `rand` now matches → the fork accelerates a CSPRNG nobody asked to accelerate.

Strictly worse than today. **The version carry and the AEAD bump are one atomic
change.** Target: `chacha20poly1305 0.11.0` (requires `chacha20 ^0.10` with
`xchacha`, `aead ^0.6`, `cipher ^0.5`, `poly1305 ^0.9`).

### 5.3 Three second-order consequences of the AEAD bump

- **(a) `zeroize` stops being automatic.** 0.10.1 forced `chacha20/zeroize`;
  0.11.0 makes it opt-in. `crates/encryption` passes `default-features = false`,
  so after the bump the ChaCha state is **no longer zeroized on drop** unless
  `"zeroize"` is added. Silent security regression.
- **(b) `poly1305_force_soft` becomes a no-op.** poly1305 0.9 renamed the cfg to
  `poly1305_backend="soft"`. The old flag is silently ignored and the
  424-intrinsic AVX2 surface returns — the exact second unaudited SIMD surface
  the matryoshka pattern exists to prevent. `.cargo/config.toml` must change in
  the same commit.
- **(c)** `crates/encryption/src/aead.rs` import migration; `from_slice` maps to
  `hybrid_array::Array::from_slice`. **UNVERIFIED:** `aead 0.6` method signatures.

### 5.4 Two options

**Option A (recommended)** — one fork carrying 0.10.1, AEAD bumped with it,
poly1305 cfg renamed. One chacha20 in every consumer graph.

**Option B** — two forks, one per major, both patched. Doubles the maintenance
surface of a crypto fork against two `cipher` majors, forever. **UNVERIFIED**
that cargo accepts the rename form here.

## 6. Falsifiers, each with the disable that proves it real

- **F1a — RFC 8439 through the matryoshka backend.** `RUSTFLAGS="-Ctarget-cpu=x86-64-v4"
  cargo test --all-features` in `vendor/chacha20`.
  *Disable:* change a rotate constant in `ndarray_simd.rs:64-70`. If still green,
  the vectors are not reaching `ndarray_simd` and F2 is what is broken.
  *Port note:* `tests/mod.rs:174-223` is a currently-dead `legacy` module that
  will start compiling at 0.10 and fail — replace with upstream's `tests/kats.rs`.
- **F1b — the 64-bit counter, which F1a cannot see.** Seek `ChaCha20Legacy` to
  `block_pos = 2^32 - 8`, compare 16 blocks against a `soft` build.
  *Disable:* drop the `state[13]` write. F1a stays green; F1b must go red.
- **F2 — two-sided proof `ndarray_simd` is selected.** Positive: at v4, `ndarray`
  is a compiled dep of the fork. **Negative: at the default v3 it is NOT** — the
  arm that proves the gate is a gate.
  *Disable:* delete `target_feature = "avx512f"`; the negative side must go red.
- **F3 — exactly ONE chacha20 resolves.** `cargo tree -d | grep -c '^chacha20'` == 0.
  Needs a **canary dev-dep** (`rand` with `std_rng`) in `crates/encryption`, since
  ndarray's own graph cannot currently fail this.
  *Disable:* revert the fork to 0.9.1 with the canary in place → two nodes.
- **F4 — wasm gate.** Extend `ci.yaml:141-142` with `--all-features` (so `rng`,
  i.e. Hunk 6, compiles for wasm) and a `cargo tree --target wasm32 | grep -c
  cpufeatures` == 0 assertion.
  *Disable:* removing the wasm arm still builds (falls to `soft`), so the build
  alone is **not** a falsifier — needs a numeric parity assertion.
- **F5 — the poly1305 cfg rename did not disarm.** `cargo build -p encryption -v
  | grep poly1305_backend` must hit.
  *Disable:* leave the old flag; grep comes back empty.
- **F6 — `encryption`'s 38 tests on both tiers** (default v3 and v4 matryoshka).

## 7. Risk register and sequencing

Highest-severity rows: `zeroize` silently dropped (§5.3a); poly1305 AVX2 surface
silently re-armed (§5.3b); the 64-bit counter carry (§2c); and **cross-repo** —
MedCare's `[patch]` is a *branch pointer*, so it picks up the new version on the
next `cargo update` and its `chacha20poly1305` must move to 0.11 in the same
window or its AEAD silently drops to the registry crate.

**The unmeasured question this port does not answer:** upstream 0.10 now ships
`backends/avx512.rs` — 16-parallel-block, `ParBlocksSize = U16`, the exact niche
`ndarray_simd.rs` was written to fill, opt-in via `--cfg chacha20_avx512`.
Neither side is benchmarked. The *architectural* justification (no raw intrinsics
in the crypto crate) still holds; the *performance* one is unmeasured. **Measure
before porting** — a "lose" verdict shrinks the port to the wasm arm alone and
removes Hunks 3, 4 and 6 entirely.

**Order:** 1. measure (all else is conditional) → 2. vendor pristine 0.10.1 and
gate it green with *zero* delta applied (isolates "faithful copy" from "correct
delta") → 3. port `ndarray_simd.rs` (F1a+F1b) → 4. re-apply cfg hunks, keeping
`soft`/`avx2`/`sse2` compiled so `rng.rs` resolves (F2) → 5. bump `encryption` to
chacha20poly1305 0.11 with `zeroize` (F6) → 6. rename the poly1305 cfg (F5) →
7. canary dev-dep + `cargo tree -d` gate (F3) → 8. extend the wasm CI gate (F4) →
9. update the prose that would otherwise be wrong → 10. coordinate MedCare-rs.

Steps 2-8 are independently revertible. The one irreversible-in-practice coupling
is 5 → 10 (the AEAD bump crossing repos): prepare MedCare's side unmerged so the
window is minutes.
