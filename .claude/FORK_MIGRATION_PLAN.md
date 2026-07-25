# crates/encryption → AdaWorldAPI forks (Ed25519, X25519, HKDF, SHA-2, AEAD)

> Status: PLAN. 2026-07-25.
> Trigger: `x25519-dalek` and `hkdf` were added from **crates.io** in #256, which
> breaks the fork rule. Reverting is one option; migrating the whole crate to the
> forks is the other, and it fixes four pre-existing violations at the same time.

> **CORRECTED 2026-07-25 after the S0 measurements** (`.claude/s0-findings/`).
> Two claims below were wrong when written and are struck through in place:
> §0's "all five move together or not at all", and §3's implication that the
> foreign AVX2 arrived with PR #256. See §7.

## 0. Why this is one move and not five

`crates/encryption` today pulls every primitive from crates.io. The forks exist
for all of them, but they sit **one generation ahead**, and that generation is
shared: `digest` / `crypto-common` / `cipher` / `aead` are the traits every one
of these crates builds on. Bumping one alone puts two versions of those traits
in the graph — it compiles, and then `Sha384` from one generation cannot be
handed to `Argon2` from the other. So the five deps move together or not at all.

| Dep | today (crates.io) | fork | fork repo |
|---|---|---|---|
| `ed25519-dalek` | 2.x | **3.0.0** | `curve25519-dalek` (monorepo) |
| `x25519-dalek` | 2.0.1 ← #256 | **3.0.0** | `curve25519-dalek` (monorepo) |
| `curve25519-dalek` | 4.1.3 (transitive) | **5.0.0** | `curve25519-dalek` |
| `hkdf` | 0.12.4 ← #256 | **0.13.0** | `KDFs` |
| `sha2` | 0.10 | **0.11.0** | `hashes` |
| `chacha20poly1305` | 0.10 | **0.11.0** | `AEADs` |
| `chacha20` | 0.9.1 (vendored) | **0.10.1** | `stream-ciphers` + our vendor |
| `argon2` | 0.5 | ? | `password-hashes` — NOT reachable yet |

**`argon2` is the open blocker.** It depends on the same `digest` generation.
If `password-hashes` cannot be reached, the choice is: keep `argon2` on the old
generation and accept a split graph, or hold the migration. Not a judgement
call to make silently — flagged for the operator.

## 1. Ed25519 — smallest surface, biggest jump

Used in exactly one file (`src/sign.rs`) and through exactly three names:
`SigningKey`, `VerifyingKey`, `Signature::from_bytes`. Everything else in the
crate touches Ed25519 only through our own `Keypair` wrapper — which is the
point of having it, and why this migration is bounded.

What must be verified against the 3.0.0 source, not assumed:

1. `SigningKey::from_bytes(&[u8; 32])` — still infallible in 3.0?
2. `VerifyingKey::from_bytes` — still `Result`, same error type?
3. `Signature::from_bytes(&[u8; 64])` — 2.x made this infallible (it was
   fallible in 1.x); 3.0 may have moved again.
4. `Signer` / `Verifier` trait paths — `signature` crate generation bump.
5. **Strict verification.** `verify_strict` vs `verify` differ on small-order
   keys and on the ZIP-215 rules. Whatever `sign.rs` uses today must keep the
   same semantics, or signatures that verified yesterday stop verifying — and
   the audit witness is merkle-chained over exactly those signatures.

(5) is the one that can silently invalidate stored data. It gets a test that
signs with the old version's known vector and verifies with the new.

## 2. The other four, in dependency order

**`sha2` 0.10 → 0.11 (`hashes` fork).** `Digest` trait generation bump. Our
surface is one function (`sha384(&[u8]) -> [u8; 48]`) so the call site is
trivial; the risk is entirely in whether every other dep accepts the new
`digest`. This one goes first because everything else depends on it.

**`hkdf` 0.12 → 0.13 (`KDFs` fork).** Depends on `hmac` 0.13, which depends on
the new `digest`. Our surface: `Hkdf::<Sha384>::new / from_prk / expand /
expand_multi_info`. Verify each still exists with the same signature.

**`chacha20poly1305` 0.10 → 0.11 (`AEADs` fork).** Depends on `aead` and
`cipher` 0.5. Our surface: `XChaCha20Poly1305`, `KeyInit`, `Payload`,
`encrypt_in_place_detached`-shaped calls. **This is the one that unblocks the
vendored `chacha20` 0.10.1** — until it lands, `[patch.crates-io]` does not
apply and the matryoshka is silently bypassed (the "patch not used" warning is
a policy alarm, not noise).

**`x25519-dalek` 2 → 3 + `curve25519-dalek` 4 → 5 (monorepo).** Only reached
through `kx.rs` from #256. Note the API constraint found earlier: with
`default-features = false, features = ["static_secrets", "zeroize"]` there is
no reachable `EphemeralSecret::random`, so the ephemeral is a `StaticSecret`
consumed by value. Re-verify against 3.0.0 — it may have changed.

## 3. The AVX2 question, which does NOT go away by migrating

`curve25519-dalek` selects `curve25519_dalek_backend="simd"` by default on
x86_64 and compiles **its own AVX2 backend with its own `unsafe`** — 350 lines
of `packed_simd.rs` carrying 37 `unsafe` blocks, under 1 558 lines of
`field.rs` + `edwards.rs` built on top.

The fork is a clean upstream mirror; it does **not** carry an `ndarray::simd`
backend. So migrating to the fork fixes the *source* rule and leaves the
*polyfill* rule unmet. Two ways, and they are not equivalent:

* **(a) `curve25519_dalek_backend="serial"`** — one line, no foreign
  intrinsics, costs the vector speedup on the curve.
* **(b) Matryoshka `packed_simd.rs` over `ndarray::simd`** — same shape as the
  ChaCha20 work: replace one shim file, leave the 1 558 lines above it
  untouched. Needs the same vocabulary audit that ChaCha20 needed (it uses
  `u32x8`/`u64x4` with shuffles and `madd`-style ops; the five primitives added
  for ChaCha20 are a start, not the whole set).

## 4. Order of work

| Step | Content | Gate |
|---|---|---|
| S0 | API-break inventory per crate, read from the fork sources | this plan's assumptions verified or corrected |
| S1 | `sha2` 0.11 | crate builds, hash vectors unchanged |
| S2 | `hkdf` 0.13 + `chacha20poly1305` 0.11 | `[patch.crates-io]` applies again; RFC 8439 + envelope round-trips |
| S3 | `ed25519-dalek` 3.0 | §1 checks, incl. the strict-verification test |
| S4 | `x25519-dalek` 3.0 | RFC 7748 vector, low-order rejection |
| S5 | `argon2` decision | operator: split graph or hold |
| S6 | curve backend: (a) serial or (b) matryoshka | operator |

S0 is grindwork over five cloned repos: mechanical, bounded, and exactly what
a Sonnet fleet is for. S1–S4 are edits gated on S0's findings. S5 and S6 are
operator decisions and are not started without one.

---

## 7. Corrections from S0 (measured, not assumed)

**§0 was wrong: the five deps do NOT have to move together.** Two trait
generations coexist in one graph — Cargo carries `crypto-common` 0.1 and 0.2
side by side. They only conflict where a *typed* value crosses the seam, and in
`crates/encryption` nothing does: every boundary hands over raw `&[u8; N]`.
argon2 (via blake2) and sha2 share `digest 0.10.7` today; the moment
`ed25519-dalek 3` lands it demands `sha2 0.11`, and argon2 stays behind on 0.10.
That is a cost — compile time, binary size, two audit surfaces — not a blocker.

**§3 implied the foreign AVX2 came in with PR #256. It did not.**
`curve25519-dalek 4.1.3` is in `Cargo.lock` at `2850886e`, the commit *before*
that merge, pulled by `ed25519-dalek`, which this crate has always used.
#256 added a **source** violation (x25519-dalek and hkdf from crates.io); the
**polyfill** violation predates it. Reverting #256 therefore fixes 2 of 6
registry dependencies and 0 of 1 AVX2 problems.

**The AVX2 problem has a one-line fix, independent of everything else.**
`curve25519-dalek` honours `--cfg curve25519_dalek_backend="serial"` via
RUSTFLAGS (its own README, lines 107/113). Appended to the x86_64 blocks in the
three `.cargo/config*.toml` files it removes the crate's 37 `unsafe` intrinsic
sites from the crypto path with no version bump, no fork question, no revert.
Cost: the curve loses its vector acceleration.

**The curve polyfill (§3 option b) IS viable — it is just bigger, and it was
wrong to relay it as "not viable."** What the S0 agent actually established is
narrower: the *one-shim* framing ("replace `packed_simd.rs`, consumers
untouched") is false, because `field.rs` also reaches the intrinsics directly
at 35 sites across 7 of them (17× `blend_epi32`, 8× `shuffle_epi32`, 2×
`permutevar8x32_epi32`, plus `srlv`/`mul_epu32`/`unpack`). Those 35 sites are
the work, not a wall: each becomes a `U32x8`/`U64x4` method, exactly as the
five ChaCha20 primitives were added.

The cross-lane ops are not a blocker either. The polyfill already carries two
tiers, and both can express them bit-exactly:
- scalar-array tier: `permutevar8x32` is `out[i] = self[idx[i] & 7]`, `blend`
  is a mask loop, `unpack` is an interleave loop — trivially correct, no design
  question;
- AVX-512 tier: the native `vpermd` / `vpblendd` / `vpunpck` intrinsics.

The one genuinely open item is **performance on the AVX2-only tier**, where a
cross-lane permute lowers to a scalar gather. That is a speed question, not a
correctness or viability one — and bit-exactness, the crypto requirement, holds
in both tiers regardless. ChaCha20 sidestepped it by redesigning to a transpose
layout; `FieldElement2625x4` cannot (the shuffles ARE the cross-term
arithmetic), so here the cross-lane vocabulary gets built rather than avoided.

Scope, stated honestly: ~500–600 lines over `packed_simd.rs` + `field.rs`, plus
~7 new cross-lane primitives on `U32x8`/`U64x4` in all three SIMD tiers. Larger
than ChaCha20, same pattern. Every other fork this migration pulls in needs its
own polyfill pass too (`sha2`'s `x86_avx2.rs`, `blake2`'s `simd/`) — that is the
real cost of the migration, and calling any single one "not viable" was a way
of not stating that total.

**Fork coverage is incomplete above the primitives.** `hmac`, `digest` and
`hybrid-array` have no reachable AdaWorldAPI fork, so even a completed
migration pulls them transitively from crates.io. Per the P0 rule this is a
STOP-and-ask, not a silent fallback. (`password-hashes` likewise unreachable —
which, given replication lag, is not evidence of absence.)

**One silent failure mode found, worth a guard.** `Cargo.lock` carried a
`[[patch.unused]]` entry for `chacha20` while `Cargo.toml` claimed the patch was
exercised by the test suite. The matryoshka was simply not in the build, and
nothing said so. MedCare-rs is set up to repeat this with its own vendored copy.
A test that fails on `[[patch.unused]]` costs nothing and would have caught it.

## 8. What this leaves the operator, as three separate calls

1. **Foreign AVX2 out of the crypto path** — the one-line `serial` cfg buys time
   TODAY (costs curve speed); the durable fix is the curve polyfill pass, which
   is viable and large, not "not viable" as an earlier draft of this plan said.
2. **#256: keep or revert** — a scope question (do we need `kx` / `hkdf_sha384`
   / `bundle` now?), no longer a safety question.
3. **Fork migration** — multi-day, and the honest total is: every fork with a
   SIMD backend needs a polyfill pass (curve25519, sha2, blake2), gated on
   argon2 and on whether the fork rule binds trait crates that have no fork.
