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

### The tiers, and what each one actually builds

**Two independent axes decide this, and reading either one alone gets it
wrong.** Axis one is `target-cpu`, which decides *which backend* is selected.
Axis two is **package selection**, which decides whether `chacha20` is
compiled at all.

| tier | target-cpu | builds `encryption`? | chacha20 backend |
|---|---|---|---|
| **CI** (`ci.yaml`) | none globally | no | not compiled |
| CI `tier4-avx512-check` | v4, per-job | no — `-p ndarray` only | not compiled |
| **`Dockerfile`** | v3 | **no** — bare `cargo build` | **not compiled** |
| **`Dockerfile.avx512`** | v4 | **no** — bare `cargo build` | **not compiled** |
| dev `cargo build -p encryption` | v3 (`.cargo/config.toml`) | yes | RustCrypto avx2/sse2 |
| …with `--config .cargo/config-avx512.toml` | sapphirerapids | yes | **`ndarray_simd`** |
| wasm32 + `simd128` | — | if selected | **`ndarray_simd`** |

The Dockerfile rows are the ones that surprise. Both images run bare
`cargo build --release`, which selects `default-members`
(`Cargo.toml:436-443`): `.`, `ndarray-rand`, `crates/ndarray-gen`,
`crates/numeric-tests`, `crates/serialization-tests`. **`crates/encryption`
is not among them**, and nothing else in that set depends on chacha20 —
verified per-package:

```console
$ for p in ndarray ndarray-rand ndarray-gen numeric-tests serialization-tests; do
      cargo tree -p "$p" -i chacha20; done
error: package ID specification `chacha20` did not match any packages   # x5

$ cargo tree -p encryption -i chacha20
chacha20 v0.9.1 (/workspace/ndarray/vendor/chacha20)
└── chacha20poly1305 v0.10.1
    └── encryption v0.1.0 (/workspace/ndarray/crates/encryption)
```

`encryption` *is* a workspace member (via `crates/*`), so `--workspace` or
`-p encryption` reaches it — neither Dockerfile passes either.

**So `ndarray_simd` is reached only by an explicit `-p encryption` /
`--workspace` build under an AVX-512 config, or by a wasm32+`simd128`
build.** No image in this repo compiles it.

### Correction history — three passes on one paragraph

Worth keeping visible, because the failure mode repeated:

1. **First claim:** "no default build runs `ndarray_simd`", reasoned purely
   from `.cargo/config.toml` pinning v3. Right answer, incomplete reason.
2. **First correction:** the operator pointed out *cargo is CI is github
   needs V3; dockerfile is V4*, so I concluded `Dockerfile.avx512` compiles
   and ships the backend. **Wrong** — I fixed the `target-cpu` axis and
   introduced a new error on the package-selection axis I still had not
   checked. (Caught by codex on PR #266.)
3. **This version:** both axes checked with `cargo tree`, per package.

The operator's tier statement was correct throughout; what it does not imply
is that either image builds the crate that pulls chacha20.

Two things none of this changes:

- The **patch-reach** table above stands. It is about which repos link the
  vendored source when they *do* build it.
- The gate is **compile-time**, not runtime. `Dockerfile`'s own comment notes
  that ndarray's `simd.rs` detects AVX-512 at run time via `LazyLock<Tier>`
  even in a v3 build — but `vendor/chacha20/src/backends.rs` keys on
  `#[cfg(target_feature = "avx512f")]`, resolved by the compiler with no such
  fallback. So even in a hypothetical v4 image that *did* build `encryption`,
  the two subsystems would dispatch by different mechanisms.

### The v4 arm is covered in CI, per-job

CI's global env carries no `target-cpu`, and `ci.yaml:17-22` records why:
it collides with the `cross_test` matrix (`i686` is 32-bit, `s390x` is not
x86 at all) and contradicts the one-binary + runtime-dispatch design intent.
Jobs needing a higher tier opt in individually — `tier4-avx512-check` sets
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS: "-D warnings
-Ctarget-cpu=x86-64-v4"`, deliberately the per-target form rather than plain
`RUSTFLAGS`, because `RUSTFLAGS` also applies to host build scripts and those
SIGILL on a runner without AVX-512 silicon.

**That job covers ndarray's v4 arm and nothing else.** Both its steps are
`cargo check --target=x86_64-unknown-linux-gnu -p ndarray --features
approx,serde,rayon` (the second adding `hpc-extras`) — package-scoped to
`ndarray`, so `crates/encryption` and therefore `vendor/chacha20` are outside
it. **No CI job compiles the chacha20 AVX-512 backend**, and none of the
coverage claimed here extends to it. Closing that gap would take an explicit
step such as:

```console
$ CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-Ctarget-cpu=x86-64-v4" \
      cargo check --target=x86_64-unknown-linux-gnu -p encryption
```

Not added here — adding a CI job is a product change, and this is a docs PR.

## The manifest comment, re-read

`Cargo.toml:481-482` reads:

> This transitively accelerates the `encryption` crate's XChaCha20-Poly1305
> keystream … **on any x86_64+avx512f build (the workspace's
> `target-cpu=x86-64-v4`)**.

An earlier version of this audit filed that parenthetical as a factual error,
on the grounds that the workspace pins v3; a later version withdrew the
filing, on the grounds that `Dockerfile.avx512` is a v4 build of this
workspace. **Both were partly wrong.**

Settled: "the workspace's `target-cpu=x86-64-v4`" does name a real tier, so
the parenthetical is imprecise rather than false — it is not the tier
`cargo build` selects. But the sentence's *main* clause is the weaker part.
"This transitively accelerates the `encryption` crate's … keystream" holds
only for a build that actually selects `encryption`, and **no image in this
repo does**. The acceleration is real and reachable; it is not automatic, and
nothing ships with it today.

Still **not edited**, for the same reason as before: manifest string changes
need a ruling. Recorded here so the next reader is not misled by it.

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

- Not that the vendored backend is wrong, or unused. `Dockerfile.avx512`
  compiles it; it is the shipped path on v4 silicon.
- Not that the patch should be removed. That depends on (3) above, which is a
  measurement nobody has taken.
- Not that MedCare-rs's wiring is incorrect. It is the one consumer that
  declared its patch deliberately and documented the fallthrough honestly.

## Still open

1. **Point `vendor/chacha20` at the fork, or keep the copy?** The P0 rule says
   depend on the AdaWorldAPI fork. Today the tree is neither — a hand copy of
   a version the fork does not carry, with `repository` rewritten to point at
   ndarray itself. The fork is provably a bare upstream mirror (same head sha,
   same `chacha20/` tree hash), so "track the fork" currently means "track
   upstream 0.10.1", which crosses a `cipher` major.
2. **Does `ndarray_simd` still beat upstream's own `avx512.rs`?** This is the
   load-bearing question now that upstream ships an AVX-512 backend of its
   own, and it is the one that decides (1). Unmeasured on both sides. The
   oracle answers the codegen half; the throughput half needs a bench.
3. **Should `vendor/chacha20`'s cfg gate get a runtime arm?** Today it is
   compile-time only, so a v3 image on AVX-512 silicon runs RustCrypto's
   backends for the keystream while ndarray's own kernels upgrade via
   `LazyLock<Tier>`. Whether that asymmetry is worth closing is a design call,
   not a bug report — and it only matters if (2) says the ndarray lane wins.
