# Revert vs Migrate — decision memo for PR #256 (crates/encryption sealed-channel primitives)

> Read-only investigation. All claims cite commit SHAs / file paths verified
> in this repo on 2026-07-25. No git state was changed; nothing was reverted,
> merged, or built.

## 0. Verified situation

- `0e5ebc02` (`git log -1`) is a merge commit, parents `2850886e` (master
  tip before) + `6e6f5b6e` (PR #256 tip). Merge message: "encryption:
  X25519, HKDF-SHA384 and Ed25519-signed bundles". Confirmed via
  `git show --stat 0e5ebc02`.
- `backup/chacha20-matryoshka-0.9.1` resolves to **exactly** `0e5ebc02`
  (`git rev-parse` both = `0e5ebc0215...`) — it is a safety snapshot at the
  merge point, not a divergent branch.
- `claude/chacha20-matryoshka-0.10.1` has `0e5ebc02` as an ancestor
  (`git merge-base --is-ancestor` confirms) and adds 7 commits on top
  (`1476c4be`..`3264dd4f`), entirely under `vendor/chacha20/` +
  `src/simd_avx512.rs` (native `U32x8`) + the migration plan doc. **Zero
  overlap** with `kx.rs` / `hkdf_sha384.rs` / `bundle.rs`
  (`git diff master...claude/chacha20-matryoshka-0.10.1 --stat -- <those
  paths>` returns empty). Cargo.lock diff between the two branches is
  7 insertions / 2 deletions (the vendored chacha20 0.10.1 registry
  entry) — **no changes to the `x25519-dalek` / `hkdf` / `curve25519-dalek`
  lock entries**, which are byte-identical on both branches.
  **A revert of #256 would not conflict with the chacha20 branch.**

## 1. Correction to the stated premise — read this first

The prompt frames PR #256 as the thing that "pulled `curve25519-dalek`,
whose build.rs selects its own AVX2 backend." **That's not what the repo
shows.** `git show 2850886e:Cargo.lock` (the commit immediately BEFORE
PR #256 merged) already contains:

```
[[package]]
name = "curve25519-dalek"
version = "4.1.3"
...
[[package]]
name = "ed25519-dalek"
version = "2.2.0"
dependencies = [ "curve25519-dalek", "ed25519", "serde", "sha2", "subtle", "zeroize" ]
```

`ed25519-dalek` was already a crates.io dependency of `crates/encryption`
**before** #256 (confirmed also in `.claude/FORK_MIGRATION_PLAN.md` line 19,
which lists `ed25519-dalek 2.x`, `sha2 0.10`, `chacha20poly1305 0.10`,
`argon2 0.5` all as pre-existing crates.io deps, separate from the two
#256 added: `x25519-dalek` and `hkdf`). `curve25519-dalek`'s `build.rs`
(`/root/.cargo/registry/.../curve25519-dalek-4.1.3/build.rs` lines 58-71)
auto-selects the `simd` (AVX2/AVX512-IFMA) backend on any x86_64/64-bit
target unless a `--cfg curve25519_dalek_backend` override is present — and
no such override exists anywhere in this repo's `.cargo/*.toml` today.

**Consequence: the foreign-AVX2-in-the-crypto-crate problem predates #256
and is NOT fixed by reverting #256.** `ed25519-dalek` alone already builds
`curve25519-dalek`'s hand-written AVX2 `unsafe` backend on master today,
merge or no merge. Revert removes the *newest* two crates.io deps
(`x25519-dalek`, `hkdf`) and ~1228 lines of new code; it does not touch the
`curve25519-dalek` build-time backend selection at all, because
`ed25519-dalek` keeps pulling it in either way.

This reframes both paths: **"fix the fork-rule violation growth" (revert)
and "fix the AVX2-intrinsics problem" (Path B's cfg flag or matryoshka)
are two separate fixes, and only one of the two candidate actions
addresses the second one.**

## 2. Path A — revert the merge

**What lands (i.e. what disappears):** per `git show --stat 0e5ebc02`,
exactly 8 files / 1228 insertions:

| File | Lines | Public API | Inline tests |
|---|---|---|---|
| `crates/encryption/src/bundle.rs` | 475 | `BundleKind`, `BundleFlags`, `BundleHeader`, `BundleError`, `sign()`, `verify()` | 11 |
| `crates/encryption/src/hkdf_sha384.rs` | 299 | `Prk`, `HkdfError`, `extract()`, `expand()`, `derive()`, `SessionKeys`, `session_keys()` | 8 |
| `crates/encryption/src/kx.rs` | 295 | `KxError`, `PublicKey`, `EphemeralSecret`, `StaticSecret` | 5 |
| `crates/encryption/src/wasm.rs` | 91 | `seal_envelope`/`open_envelope`/`generate_signing_seed`/`sign_message`/`verify_signature`/`VerifiedBundle`/`verify_bundle`/`sha384` wasm-bindgen exports | 0 (wasm-bindgen bindings, no inline `#[test]`) |
| `crates/encryption/src/lib.rs` | +3 | `pub mod bundle; pub mod hkdf_sha384; pub mod kx;` | — |
| `crates/encryption/Cargo.toml` | +3 | adds `x25519-dalek`, `hkdf` deps | — |
| `Cargo.lock` | +31 | new lock entries | — |
| `.claude/blackboard.md` | +31 | session notes | — |

Total of the crate's 55 `#[test]` functions today, **24 (44%) belong to
the three files a revert removes** (bundle 11 + hkdf_sha384 8 + kx 5).
The other 31 (aead 4, envelope 10, hash 3, kdf 9, sign 5) are pre-existing
and untouched — they don't import `kx`/`hkdf_sha384`/`bundle`.

**Dependents:** grepped the ndarray workspace (`*.rs`) and `lance-graph`
for `kx::`, `hkdf_sha384`, `SealedBundle`, `encryption::kx`,
`encryption::bundle`, `x25519-dalek` — **zero hits** outside
`crates/encryption/` itself. (MedCare-rs and smb-office-rs grep timed out
under load in this session — not re-verified; treat as unverified for
those two repos, though nothing in their `CLAUDE.md`s references this
crate and it's `publish = false` / not a listed cross-repo dependency, so
low prior.) No downstream consumer breaks that I could confirm.

**Conflict with `claude/chacha20-matryoshka-0.10.1`:** none — confirmed in
§0, zero file overlap, identical lockfile entries for the three deps in
question.

**What breaks:** the sealed-channel key-exchange + signed-bundle feature
ships zero this cycle. Whoever asked for PR #256 (X25519 KX for the
wasm/browser envelope path per `wasm.rs`'s `seal_envelope`/`verify_bundle`)
loses it until re-landed via the fork.

**What stays violated:** per §1, `curve25519-dalek`'s AVX2 backend via
`ed25519-dalek` — unchanged. `sha2`, `chacha20poly1305`, `argon2`,
`ed25519-dalek` all stay on crates.io (never fork-compliant to begin
with, per `FORK_MIGRATION_PLAN.md`'s own table). **Revert fixes 2 of 6
crates.io violations and 0 of 1 AVX2 violations.**

**Time:** trivial — `git revert -m 1 0e5ebc02` (not run), likely clean
given zero overlap with the chacha20 branch. Under an hour including
verifying the revert compiles.

## 3. Path B — migrate to forks (full plan) vs. minimal violation fix

`.claude/FORK_MIGRATION_PLAN.md` (read in full) proposes a 6-step,
5-crate-family jump (`ed25519-dalek`→3.0, `x25519-dalek`→3.0,
`curve25519-dalek`→5.0, `hkdf`→0.13, `sha2`→0.11,
`chacha20poly1305`→0.11) because the shared `digest`/`crypto-common`
generation can't be split. Plan's own S0-S6 order puts the curve backend
question (serial vs matryoshka) at **S6, gated on an operator decision**,
and `argon2` (S5) as an **open blocker** — no confirmed fork path yet.
This is not a small change; it's the "full generation jump" the prompt
alludes to.

**Minimal fix that removes the AVX2 violation only, verified reachable
today, independent of the fork migration or the revert decision:**

`curve25519-dalek`'s `build.rs` (lines 53-72) reads
`CARGO_CFG_CURVE25519_DALEK_BACKEND`, which Cargo populates from a
`--cfg curve25519_dalek_backend="serial"` rustflag — **not** a plain env
var (confirmed by reading the build script: it calls
`std::env::var("CARGO_CFG_CURVE25519_DALEK_BACKEND")`, which is only ever
set by Cargo when a matching `--cfg` was passed to rustc). The crate's own
README (read at
`.../curve25519-dalek-4.1.3/README.md` lines 103-113) documents exactly
this:

```
RUSTFLAGS='--cfg curve25519_dalek_backend="BACKEND"'
# or, in .cargo/config.toml:
rustflags = ['--cfg=curve25519_dalek_backend="BACKEND"']
```

This repo's `.cargo/config.toml`, `.cargo/config-avx512.toml`, and
`.cargo/config-native.toml` (the three x86_64 profiles; `-graviton`,
`-apple-m2`, `-pi5`, `-wasm` are non-x86_64 and irrelevant here) each
already carry a `[target.'cfg(target_arch = "x86_64")']` `rustflags`
array (`-Ctarget-cpu=x86-64-v3` / `sapphirerapids` / `native`
respectively). Because Cargo does not merge `rustflags` from multiple
sources for the same target key, the fix is to **append** to those three
existing arrays, e.g.:

```toml
rustflags = ["-Ctarget-cpu=x86-64-v3", "--cfg", "curve25519_dalek_backend=\"serial\""]
```

3 files, 1 line each. No dependency bump, no version jump, no `argon2`
blocker, no digest-generation coordination. Applies regardless of whether
#256 is reverted (it silences the AVX2 backend for `ed25519-dalek`'s
existing pre-#256 use of `curve25519-dalek` too).

**Cost:** the README states `serial` is "an optimized, non-parallel
implementation" vs. `simd` "Intel AVX2/AVX512 IFMA accelerated backend" —
no quantified benchmark number is in the README (only qualitative
naming); I did not find a benchmark table in this crate's docs, so any
"2-4x" figure would be a guess — flagging as **unverified, do not cite a
number**. The honest cost statement is: gives up whatever speedup the
crate's own SIMD backend provides on X25519/Ed25519 scalar-mult
operations, in exchange for zero foreign `unsafe` AVX2 in the built
binary. Given this crate is doing key exchange + signing (small, infrequent
ops, not a hot loop), the perf cost is very plausibly irrelevant — but
that's a judgment call, not a measurement I have.

**What stays violated:** crates.io as the *source* (not the fork) for
`curve25519-dalek`/`ed25519-dalek`/`x25519-dalek`/`hkdf`/`sha2`/
`chacha20poly1305`/`argon2` — the P0 fork rule is still broken, just with
its most dangerous symptom (foreign AVX2 `unsafe`) neutralized. This is a
stopgap, not a fix of the CLAUDE.md P0 rule.

**Time:** the 3-line cfg change: under an hour, no version coordination.
The full fork migration (S0-S6): plan explicitly scopes S0 as "grindwork
over five cloned repos... exactly what a Sonnet fleet is for" — i.e. days,
not hours, and S5/S6 are explicitly blocked on operator decisions the
plan itself declines to make.

## 4. Path C — combine, don't choose

Nothing else in the sources suggests a materially different third path,
but the two paths above are not mutually exclusive and the plan doesn't
frame them as an either/or:

1. **Now:** apply the 3-file `--cfg curve25519_dalek_backend="serial"`
   change (§3 minimal fix) — kills the actual AVX2-`unsafe` problem
   immediately, on master, unconditional on the revert decision.
2. **Separately, operator's call:** revert #256 or keep it. Since it
   doesn't conflict with the chacha20 branch and doesn't touch the AVX2
   question either way, this becomes purely a product decision ("do we
   want the sealed-channel/X25519-KX feature now, on a temporarily
   fork-noncompliant crates.io dep, or not") — not a technical blocker.
3. **Later:** the full fork migration per `FORK_MIGRATION_PLAN.md`,
   S0-S6, still needed eventually to actually close the P0 rule violation
   (crates.io source, not just the AVX2 symptom) — argon2 (S5) remains
   an open unknown that needs its own answer before this can complete.

## 5. Recommendation

Do the §3 cfg fix immediately (small, mechanical, unblocks nothing else,
removes the one violation everyone agrees is dangerous — foreign
`unsafe` AVX2 — regardless of what happens to #256). Then let the
operator decide Path A vs. "keep #256 and accept its two crates.io deps
as an addition to a debt list the fork-migration plan already owns" —
because reverting #256 does NOT get master to fork-compliant (§1), it
only prevents the debt from growing by two more crates.io deps while the
other six (pre-existing) stay exactly as violating as before.

**The one question for the operator:** given that `curve25519-dalek`
already violates the P0 fork rule via `ed25519-dalek` *before* #256, and
reverting #256 does not change that — is the goal (a) stop the bleeding
now (§3 cfg fix, ships today, unconditional) plus decide #256's fate as a
feature-scope question, or (b) hold everything for the full 6-crate fork
migration in `FORK_MIGRATION_PLAN.md` before shipping anything further,
accepting that `argon2`'s fork reachability (S5) is still unresolved and
could block indefinitely?

## 6. Uncomfortable part, stated plainly

The violating code (crates.io `curve25519-dalek` with its own AVX2
`unsafe` backend) is on master **today**, and was on master **before**
PR #256 too — via `ed25519-dalek`, which nobody has flagged until this
memo. "Do nothing" is a decision that leaves that AVX2 code running,
and it was already true yesterday. Reverting #256 does not change this
fact even by one bit; it only reverts an unrelated, temporally-adjacent,
also-still-crates.io addition (`x25519-dalek` + `hkdf`) that happened to
land in the same session. Conflating "revert #256" with "fix the AVX2
problem" is the mistake baked into the situation as originally framed —
worth surfacing to the operator explicitly rather than let it stand.
