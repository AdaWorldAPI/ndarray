# S0 findings — encryption crate crypto-trait-generation bump blast radius

Scope: `ndarray/crates/encryption` bump (sha2 0.10→0.11, chacha20poly1305
0.10→0.11, ed25519-dalek 2→3, hkdf 0.12→0.13, x25519-dalek 2→3). Chain
verified: `encryption` → `OGAR/crates/ogar-encryption` (thin re-export) →
`ogar-auth` / `medcare-core::sealed`.

## Consumer table

| file:line | how it consumes | breaks on bump? | pins its own version? |
|---|---|---|---|
| `OGAR/crates/ogar-encryption/src/lib.rs:60` `pub use encryption::{aead, envelope, hash, kdf, sign};` | re-exports **modules**, not concrete third-party types | survives — pass-through | no; git dep on ndarray `branch = "master"` (`OGAR/crates/ogar-encryption/Cargo.toml:16`) |
| `OGAR/crates/ogar-encryption/src/lib.rs:65,69` `pub use encryption::{EnvelopeError, KdfParams, open, seal}; pub use encryption::RngError;` | re-exports `encryption`'s **own** wrapper types (`KdfParams` defined at `ndarray/crates/encryption/src/kdf.rs:12`, `RngError` at `.../src/lib.rs:90`, `AeadError`/`SignError` similarly own-defined) — NOT re-exports of `chacha20poly1305::*` / `sha2::*` / `ed25519_dalek::*` / `hkdf::Hkdf` / `x25519_dalek::*` | survives, **provided** `encryption`'s own wrapper API shape is unchanged after the bump (ndarray's responsibility, not the consumer's) | no |
| `OGAR/crates/ogar-auth/src/lib.rs:58` `pub use ogar_encryption::{aead, envelope, hash, kdf, sign};` | re-export of a re-export, same modules | survives (pass-through of pass-through) | no; path dep `ogar-encryption = { path = "../ogar-encryption" }` (`ogar-auth/Cargo.toml`) |
| `OGAR/crates/ogar-auth/src/password.rs:15-16` `use argon2::Argon2; use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};` | names **argon2** crate types directly in its own logic | **not in this bump's scope** (argon2 isn't one of the 5 crates being bumped) — orthogonal coupling, unaffected here | yes, `argon2 = "0.5"` directly in `ogar-auth/Cargo.toml`, matches `encryption`'s own `argon2 = "0.5"` |
| `MedCare-rs/crates/medcare-core/src/sealed.rs:55` `pub use ogar_encryption::KdfParams;` | re-exports the wrapper type again | survives, same condition as above | no |
| `MedCare-rs/crates/medcare-core/src/sealed.rs:84,112` `ogar_encryption::seal(...)` / `ogar_encryption::open(...)` | calls wrapper **functions** only | survives — no third-party type named in medcare-core's own signatures | no |
| `MedCare-rs/crates/medcare-core/src/crypto.rs:6,60`, `document_kv.rs:40,54`, `medcare-server/src/client_ip.rs:40,120` | `use sha2::{Digest, Sha256}` — **direct** dependency, NOT through `encryption`/`ogar_encryption` chain at all | independent dependency edge; already resolves to `sha2 = "0.11"` (root `Cargo.toml:274`) — i.e. MedCare's own direct sha2 dep is already ahead of what `encryption` crate currently needs (`encryption/Cargo.toml` still declares `sha2 = "0.10"`) | yes, root workspace `Cargo.toml:274` `sha2 = "0.11"` |
| `smb-office-rs`, `lance-graph` | grepped `ogar_encryption`, `encryption::`, `ed25519`, `chacha20poly1305`, `XChaCha20Poly1305`, `Sha384` across both trees | **no hits** — neither repo consumes this crate/chain at all | n/a |
| `OGAR/crates/ogar-action-handler/src/lib.rs:483,487` | `"ed25519"` hit is an SSH key **filename string** (`/keys/id_ed25519`) in a test, not the crypto crate | false positive, no coupling | n/a |

No consumer found anywhere (MedCare-rs, OGAR, lance-graph, smb-office-rs)
that names `chacha20poly1305::XChaCha20Poly1305`, `ed25519_dalek::{SigningKey,
VerifyingKey}`, `hkdf::Hkdf`, `x25519_dalek::{PublicKey, StaticSecret}`, or
raw `sha2` types *through* the `encryption`/`ogar_encryption` chain. Every
hop insulates via its own wrapper type or a straight function call. **The
Rust-type-level blast radius of this bump, through the verified chain, is
effectively zero** — assuming `encryption`'s own public wrapper API
(`KdfParams`, `EnvelopeError`, `RngError`, `AeadError`, `SignError`, fn
signatures of `seal`/`open`) is unchanged shape after the version jump.

## What breaks first, and where the failure will be reported

Not at the Rust type-checker. It breaks at **Cargo dependency resolution /
patch application**, and it may already be silently broken right now:

1. `ogar-encryption` and `a2ui`-adjacent deps are **git deps on ndarray
   `branch = "master"`** (`OGAR/crates/ogar-encryption/Cargo.toml:16`), so
   the moment the bump lands on ndarray's master, every consumer that does a
   fresh `cargo update` / clean lock picks up the new `encryption` crate
   with no version pin to hold it back. First failure surface: whichever
   consumer's `encryption`-internal code fails to build against the new
   trait generation — that is entirely inside `ndarray/crates/encryption/src/*.rs`
   (kdf.rs, aead.rs, sign.rs, hkdf_sha384.rs, envelope.rs, kx.rs), i.e. it
   reports as a **compile error inside the `encryption` crate itself**, in
   ndarray's own CI/build, before it ever reaches OGAR or MedCare-rs.
2. If `encryption` compiles clean against the bumped crates, the next place
   it can surface is `cargo build`/`clippy` in `OGAR` (ogar-encryption /
   ogar-auth) since those are on `branch = "master"` with no lock pin
   isolating them.
3. MedCare-rs is the most insulated (path-deps + `[patch]`, not git-branch
   floating) but is also the one carrying the actual, already-real
   divergence below.

## The divergence finding: MedCare-rs's vendored `chacha20` patch is a plain directory copy, and it has ALREADY silently drifted ahead of MedCare's lockfile

**Nature of the vendored copy:** `MedCare-rs/vendor/ndarray` is a **symlink**
(`vendor/ndarray -> ../../ndarray`, confirmed via `ls -la`), so
`vendor/ndarray/vendor/chacha20` resolves to the real path
`/home/user/ndarray/vendor/chacha20`. That directory is **not a git
submodule** (no `.git` inside it, not listed in `ndarray/.gitmodules`,
which only lists `crates/burn/upstream`) — it is a **plain directory of
files tracked directly inside ndarray's own git repository**
(`git ls-files vendor/chacha20` lists `Cargo.toml`, `src/*.rs`, etc. as
ordinary tracked blobs of the `ndarray` repo).

**Version, verified, NOT what the comment in MedCare's Cargo.toml says:**

- `MedCare-rs/Cargo.toml:383` (the `[patch.crates-io]` comment) currently
  reads: *"ndarray ships a fork of `chacha20` **0.9.1** whose ONE added
  backend expresses the keystream double-round over `ndarray::simd::U32x16`..."*
- But `/home/user/ndarray/vendor/chacha20/Cargo.toml:19` currently reads
  `version = "0.10.1"` — **not 0.9.1**.
- `git -C /home/user/ndarray log --oneline -5 -- vendor/chacha20/Cargo.toml`
  shows this is not stale documentation catching up later — it is two
  **already-committed** ndarray-repo commits that moved the vendor copy
  forward:
  ```
  43a22665 vendor/chacha20: replay the ndarray::simd polyfill onto 0.10.1 (step 3 of 3)
  4d2cfcce vendor/chacha20: refresh to upstream 0.10.1 (step 2 of 3 — polyfill absent on purpose)
  ```
- `git -C /home/user/ndarray status --porcelain vendor/chacha20` is clean —
  this is landed, not a work-in-progress diff sitting in the tree.

**The lockfile evidence that the drift is already live:**

- `/home/user/ndarray/Cargo.lock` (regenerated `17:29`, after the bump
  commits) shows plain **unpatched** registry `chacha20 0.9.1` (ndarray's
  own workspace doesn't apply the `[patch.crates-io]` to itself — that only
  exists in MedCare-rs's root manifest) and registry `chacha20poly1305
  0.10.1` depending on it. Unrelated to the fork question — this is just
  ndarray building its own `encryption` crate normally.
- `/home/user/MedCare-rs/Cargo.lock` (regenerated `16:30`, **before** the
  vendor bump commits' lock effects propagated — i.e. it is now stale
  relative to the on-disk vendor source) shows the **patched** local
  package as:
  ```
  [[package]]
  name = "chacha20"
  version = "0.9.1"
  dependencies = [ "cfg-if 1.0.4", "cipher", "cpufeatures 0.2.17", "ndarray 0.17.2" ]
  ```
  (no `source =` line ⇒ this is the local path package satisfying
  `[patch.crates-io] chacha20 = { path = "vendor/ndarray/vendor/chacha20" }`;
  the `ndarray 0.17.2` dependency is the SIMD-polyfill backend, confirming
  it IS the patched fork, not a registry entry that happens to share the
  version). **This locked version (0.9.1) no longer matches what
  `vendor/chacha20/Cargo.toml` declares on disk (0.10.1) right now.**
  Separately in the same lockfile, registry `chacha20poly1305 0.10.1`
  still depends on `chacha20 0.9.1` (its own manifest requirement, `^0.9`).

**Why this is a live risk, not just a cosmetic mismatch:** Cargo's
`[patch.crates-io]` only substitutes a patched source for a dependency edge
when the patched package's own declared version is semver-compatible with
what the dependent requires. `chacha20poly1305` (as shipped, still at
0.10.x, unless it also lands the 0.10→0.11 leg of this bump) requires
`chacha20 ^0.9`. The vendor fork now declares itself `0.10.1`. Under
Cargo's 0.x semver rules, `0.9` and `0.10` are **not** compatible ranges.
The instant MedCare-rs's lockfile is regenerated (`cargo update` / deleting
`Cargo.lock` / any dependency-graph-touching change) with the vendor tree
in its current state, Cargo has two options and neither is silent-safe:
either (a) it can no longer satisfy the patch for that edge and falls back
to fetching **real, unpatched, non-SIMD crates.io `chacha20` 0.9.1**,
silently defeating the entire point of the vendored fork (the exact P0
violation lance-graph's `CLAUDE.md` names: *"Patch `<crate>` was not used
in the crate graph" is a policy alert*), or (b) the resolver fails outright
if a strict/frozen mode is in effect. Right now MedCare-rs's *lockfile*
still shows the old, internally-consistent 0.9.1 state, which is exactly
why nothing has visibly broken yet — the drift is real on disk but has not
yet been forced through a re-resolve.

**Most likely explanation:** this looks like a two-step migration already
in flight on the ndarray side (bump the vendor `chacha20` fork to 0.10.1
first, to pre-stage for `chacha20poly1305` 0.10→0.11 — which itself would
require `chacha20 ^0.10`, realigning the patch once `encryption/Cargo.toml`
also bumps its `chacha20poly1305 = "0.10"` pin to `"0.11"`, which it has
**not** done yet — `ndarray/crates/encryption/Cargo.toml:18` still reads
`chacha20poly1305 = { version = "0.10", ... }` as of this read). Until that
second leg lands, the vendor tree and MedCare's patch declaration are in an
inconsistent intermediate state, and `MedCare-rs/Cargo.toml:383`'s "0.9.1"
comment is already factually wrong about the file it is describing.

**Recommendation (for whoever lands the bump):** land the `chacha20poly1305`
0.10→0.11 leg in `ndarray/crates/encryption/Cargo.toml` in the same change
that the vendor `chacha20` fork moves to 0.10.1 (already done), then
immediately regenerate `MedCare-rs/Cargo.lock` and grep the fresh lock for
a *single* `chacha20` entry (patched, no `source =` line) rather than two —
two entries post-bump means the patch silently stopped applying and MedCare
lost the SIMD-accelerated fork without any compile error announcing it.
Also fix the stale "0.9.1" wording in `MedCare-rs/Cargo.toml:383`'s comment
once the version is confirmed stable.

All claims above are backed by direct file reads / `git log` / `git status`
/ `grep` output at the paths cited; nothing here is inferred without
evidence. No cargo build/check/test was run (out of scope for this
read-only pass).
