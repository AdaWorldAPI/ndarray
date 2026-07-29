# `vendor/chacha20` — what it actually reaches

> **Status: MEASURED, 2026-07-29.** Every row below is read from `Cargo.lock`,
> a manifest on disk, `rustc --print cfg`, or a git object hash. No estimates.

## READ BY:
- Anyone about to re-sync, extend, or delete `vendor/chacha20`
- Anyone citing "the AdaWorldAPI chacha20 fork" as an accelerated dependency
- Anyone applying the P0 fork rule to a crate that also has a vendored copy

## P0 TRIGGER
About to state that a consumer's ChaCha20 keystream is accelerated because
ndarray patches `chacha20`? **Read the two reach tables first. The patch and
the acceleration have different, and both narrower-than-assumed, reach.**

---

## Two different questions, two different answers

"Does the fork reach consumer X?" is really two questions, and conflating
them is what produced the overstated claims in the manifest comments.

1. **Does the `[patch.crates-io]` apply** — i.e. does the build link the
   vendored source at all?
2. **Does `ndarray_simd` compile** — i.e. does the added backend get
   selected, or does the vendored crate fall through to RustCrypto's own?

A build can answer yes to (1) and no to (2). Most do.

## Reach of the patch

`[patch.crates-io]` takes effect **only from the top-level workspace
manifest**. It is ignored in a dependency's manifest, so ndarray's patch
does not travel with ndarray — each consumer must declare its own.

| repo | declares a chacha20 patch | links the vendored source |
|---|---|---|
| ndarray | yes — `Cargo.toml:488` | **yes** |
| MedCare-rs | yes — `vendor/ndarray/vendor/chacha20` | **yes** |
| lance-graph | no | no — registry crate |
| OGAR | no | no — registry crate |
| tesseract-rs | no | no — registry crate |
| stockfish-rs | no | no — registry crate |
| a2ui-rs | no | no — registry crate |

Evidence for the two "yes" rows: ndarray's `Cargo.lock` carries
`chacha20 0.9.1` with **no `source` line**, which is how a path override
appears; MedCare-rs's root manifest declares
`chacha20 = { path = "vendor/ndarray/vendor/chacha20" }`, and its
`vendor/ndarray` symlink canonicalizes to the same checkout, so one
`ndarray` node resolves.

Evidence for the five "no" rows: `grep 'chacha20'` and `grep '\[patch'`
across each root manifest. lance-graph's manifest contains only a comment
*about* patching, explicitly recording that no `[patch]` block exists there.

## Reach of the `ndarray_simd` backend

`vendor/chacha20/src/backends.rs` selects the added backend under

```rust
all(target_arch = "x86_64", target_feature = "avx512f", …)   // or
all(target_arch = "wasm32",  target_feature = "simd128")
```

`avx512f` is a **compile-time** feature cfg, so it is present only if the
build pins a target-cpu that includes it. Measured:

```console
$ rustc --print cfg -Ctarget-cpu=x86-64-v3 | grep avx
target_feature="avx2"            # <- no avx512f
$ rustc --print cfg -Ctarget-cpu=x86-64-v4 | grep avx512f
target_feature="avx512f"
```

| build | target-cpu | `avx512f` | backend compiled |
|---|---|---|---|
| ndarray, default | `x86-64-v3` (`.cargo/config.toml`) | absent | RustCrypto soft/avx2/sse2 |
| ndarray, `--config .cargo/config-avx512.toml` | `x86-64-v4` | present | **`ndarray_simd`** |
| MedCare-rs, default | none — no `.cargo/config.toml` at all | absent | RustCrypto soft/avx2/sse2 |
| wasm32 + `simd128` | — | n/a | **`ndarray_simd`** |
| the five unpatched repos | — | — | not present in the source they link |

**No default build of any repo in the workspace runs `ndarray_simd`.** Two
opt-in configurations do: an explicitly-selected AVX-512 x86_64 build, and a
wasm32 build with `simd128`.

This is not a defect — the fallthrough is correct and the comments in
MedCare-rs's manifest describe it accurately ("non-avx512 builds fall through
to RustCrypto's own backends"). It is a scope fact that keeps being stated
one size too large.

## One manifest comment overstates it

`Cargo.toml:481-482` reads:

> This transitively accelerates the `encryption` crate's XChaCha20-Poly1305
> keystream … **on any x86_64+avx512f build (the workspace's
> `target-cpu=x86-64-v4`)**.

The workspace's pinned target-cpu is **v3**, in `.cargo/config.toml`. `v4`
is the opt-in `.cargo/config-avx512.toml`. The conditional half of the
sentence is right; the parenthetical asserts the default build satisfies it,
and it does not.

**Not edited here.** Manifest string changes in this repo need an operator
ruling; the finding is recorded instead of applied.

## The fork the P0 rule names is not this tree

`AdaWorldAPI/stream-ciphers` exists and is now cloned in-session.

| | `AdaWorldAPI/stream-ciphers` | `ndarray/vendor/chacha20` |
|---|---|---|
| what it is | mirror of RustCrypto upstream | hand-copied source tree |
| head | `5f3430b` "Release chacha20 v0.10.1 (#574)" | — |
| branches | `master` only | — |
| chacha20 version | **0.10.1** | **0.9.1** |
| `cipher` | 0.5 | 0.4.4 |
| edition / rust-version | 2024 / 1.85 | 2021 / 1.95 |
| `backends/ndarray_simd.rs` | **absent** | present |
| `backends/avx512.rs` | **present (upstream's own)** | absent |
| `repository` field | RustCrypto | rewritten to `AdaWorldAPI/ndarray` |

Three consequences worth stating plainly:

1. **The fork carries no AdaWorldAPI delta**, and this is checkable rather
   than inferred from a branch count. Against `RustCrypto/stream-ciphers`
   fetched as `upstream`:

   ```console
   $ git rev-parse HEAD
   5f3430b7531a33aa14957b6bd407b46687635124
   $ git rev-parse upstream/master
   5f3430b7531a33aa14957b6bd407b46687635124
   $ git rev-list --count upstream/master..HEAD
   0
   $ git rev-parse HEAD:chacha20 upstream/master:chacha20
   3f58304be34b79a3a38430a5752fb8db223d1a20
   3f58304be34b79a3a38430a5752fb8db223d1a20
   ```

   The fork's head *is* upstream's head — same commit hash, zero commits
   ahead, and the `chacha20/` subtree hashes to the same object. Git tree
   hashes cover content recursively, so identical tree ids mean byte equality
   for that directory. "Depend on the fork" and "depend on upstream" are
   therefore the same bytes today — as a verified fact, not an inference.
   (Repeat the check after any upstream sync; it is one command.)
2. **The vendored tree is not a checkout of the fork**, and is one minor
   release behind it. It cannot be re-synced by `git pull`; the re-sync
   procedure its own header describes (bump the source, re-apply the backend
   and the four cfg branches, re-run the vectors) is a manual port, and
   0.9.1 → 0.10.1 crosses a `cipher` major (0.4 → 0.5).
3. **Upstream now ships its own AVX-512 backend.** `backends/avx512.rs`
   at 0.10.1 occupies the exact niche `ndarray_simd.rs` was written to fill.
   Whether the ndarray-lane version still earns its place is a measurement
   question — and the instrument for it already exists
   (`.claude/knowledge/simd-codegen-oracle/`), with the relevant lane already
   measured: the u32 ARX triple on `U32x16` hits the AVX2 instruction floor
   (`td-t22-asm-investigation.md`).

## What is NOT claimed here

- Not that the vendored backend is wrong. It is untested by default builds,
  which is a different statement.
- Not that the patch should be removed. That is a decision, and it depends on
  (3) above plus the AVX-512 deployment question, neither of which this audit
  settles.
- Not that MedCare-rs's wiring is incorrect. It is the one consumer that
  declared its patch deliberately and documented the fallthrough honestly.

## Open, for an operator ruling

1. **Point `vendor/chacha20` at the fork, or keep the copy?** The P0 rule
   says depend on the AdaWorldAPI fork. Today the tree is neither — a copy of
   a version the fork does not carry, with `repository` rewritten to point at
   ndarray itself.
2. **Does `ndarray_simd` still beat upstream's own `avx512.rs`?** Answerable
   with the oracle. Unmeasured today, on both sides.
3. **The v3/v4 manifest comment** — correct in place, or leave the record and
   annotate?
