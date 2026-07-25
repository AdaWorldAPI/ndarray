# argon2 fork-probe findings — S0

Scope: `crates/encryption` migration from crates.io to AdaWorldAPI forks.
`argon2` is the crate with no reachable fork this session. READ-ONLY probe;
no cargo run, no commit.

---

## TASK 1 — Probe: is `AdaWorldAPI/password-hashes` reachable?

Command run exactly as specified:

```
GIT_TERMINAL_PROMPT=0 timeout 60 git ls-remote \
  "https://x-access-token:${TOKEN}@github.com/AdaWorldAPI/password-hashes.git"
```

(`TOKEN` built via `T=$(printf '%s' "${GITHUB_TOKEN:-}" | tr -d '"'"'"'')`;
token never printed or embedded in displayed output — only the sanitized
length, 40, was echoed for sanity.)

**Verbatim output:**

```
exit code: 128
remote: Repository not found.
fatal: repository 'https://github.com/AdaWorldAPI/password-hashes.git/' not found
```

**Verdict: UNVERIFIED — not "does not exist."** Per the brief's own framing,
GitHub returns this identical message for "repo absent" and "repo exists but
token/session has no access yet (replication lag 2–25 min)". This probe
proves only "not reachable at $(date -u) this session", nothing stronger.
Re-probe later in this session or in a fresh session before treating the
fork as confirmed absent. I did NOT clone anything (nothing to clone) and
did NOT inventory upstream `argon2`'s version there — the repo was not
reachable.

---

## TASK 2 — The coupling question

### Current state (today, before any migration) — NOT duplicated

Read `/home/user/ndarray/crates/encryption/Cargo.toml`:
- L17: `argon2 = { version = "0.5", default-features = false, features = ["alloc"] }`
- L20: `sha2 = { version = "0.10", default-features = false }`

`argon2`'s own manifest (`~/.cargo/registry/src/…/argon2-0.5.3/Cargo.toml`)
declares:
- `blake2 = "0.10.6"` (default-features = false)
- `password-hash = "0.5"` (optional; pulled in by our `alloc` feature via
  argon2's own `alloc = ["password-hash/alloc"]`, Cargo.toml L69)
- `base64ct = "1"`

`blake2-0.10.6`'s own manifest: `digest = "0.10.3"` (features = ["mac"]),
which itself requires `crypto-common = "0.1.3"`.

**Resolved versions in `/home/user/ndarray/Cargo.lock` today** (grep
`^name = "(sha2|digest|crypto-common|blake2|argon2|password-hash|base64ct)"`):

| crate | resolved version | Cargo.lock line |
|---|---|---|
| argon2 | 0.5.3 | 76-77 |
| blake2 | 0.10.6 | 139-140 |
| sha2 | 0.10.9 | 1910-1911 |
| digest | **0.10.7** (single entry — one node in the graph) | 700-701 |
| crypto-common | **0.1.7** (single entry) | 637-638 |
| base64ct | 1.8.3 | 118-119 |
| password-hash | 0.5.0 | 1391-1392 |

**`argon2` (via `blake2`) and `sha2` share the exact same `digest 0.10.7` /
`crypto-common 0.1.7` node today.** One `digest` in the lockfile, not two.
So RIGHT NOW the graph is not split — argon2 staying on crates.io while
`sha2` stays on `0.10.x` is genuinely harmless: same generation, same
compiled crate, deduped by Cargo.

### Where they actually meet, and the trap in the "migrate sha2" plan

They meet at the `digest`/`crypto-common` trait-generation boundary, not
inside our own code — grepped `crates/encryption/src/*.rs`: `hash.rs:7`
(`use sha2::{Digest, Sha384}`) and `hkdf_sha384.rs:37,96,…`
(`Hkdf::<Sha384>`) use `sha2`'s `Digest` trait directly; `kdf.rs` (the
argon2 call site) never takes a `Digest`/generic hash parameter — it calls
concrete `Argon2::new(...).hash_password_into(...)` (kdf.rs:205-209). **No
function in this crate hands a `sha2::Sha384` value into an argon2 API or
vice versa** — so there is no *compile-error*-level coupling, only a
*dependency-graph* coupling: both crates need SOME version of `digest` to
exist, and today it's the same one.

**The trap:** crates.io already has `sha2 0.11.0`, and it is a **generation
break**, confirmed by extracting the cached `.crate` tarballs directly
(`/root/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/`):

- `digest-0.11.3.crate` → `Cargo.toml`: depends on `crypto-common 0.2`
  (package name `common`), `block-buffer 0.12` (vs `0.10` on the 0.10 line),
  edition 2024, dev-dep `sha2 = "0.11"`.
- `digest-0.10.7`'s own manifest (extracted): `crypto-common = "0.1.3"`,
  `block-buffer = "0.10"` (optional).
- Cache also holds `crypto-common-0.2.2.crate` and `sha2-0.11.0.crate`
  alongside the `0.10.9`/`0.1.7`/`0.10.7` set already in the lockfile —
  i.e., crates.io itself already forked this generation once.

**So: IF a future migration bumps `sha2` to an AdaWorldAPI fork tracking
the `0.11` generation (or any digest-0.11-based fork) while `argon2` stays
pinned at crates.io `0.5.3` (→ `blake2 0.10.6` → `digest 0.10.7` →
`crypto-common 0.1.7` → `block-buffer 0.10`), the graph WOULD carry BOTH
generations simultaneously:** `digest` 0.10 **and** 0.11, `crypto-common`
0.1 **and** 0.2, `block-buffer` 0.10 **and** 0.12. That is a real split
graph, not merely an aesthetic one — it is exactly the "two generations"
scenario the task asked me to check for, and it is TRUE only in the
sha2-jumps-generation branch, not in the status quo.

**Second-order finding (widens the blast radius beyond argon2):**
`ed25519-dalek 2.2.0`'s own manifest (extracted from
`~/.cargo/registry/src/…/ed25519-dalek-2*/Cargo.toml`) **also directly
depends on `sha2 = "0.10"`** (default-features = false), independent of
argon2 entirely. So bumping `sha2` to a `0.11`-generation fork creates a
split graph **even if argon2 is ignored**, unless `ed25519-dalek` (and
`hkdf 0.12.4`, which also needs `digest 0.10` per its own manifest) are
migrated in lockstep. This means: **the "migrate sha2 alone" sub-plan is
not actually available** — sha2's generation is entangled with
ed25519-dalek and hkdf too, not just argon2. Whoever owns the sha2 fork
question needs to confirm the fork tracks `0.10.x` (dedups cleanly with
argon2/ed25519-dalek/hkdf as today) or `0.11.x` (forces a 3-crate ripple,
not a 1-crate one).

**Answer to the decisive question, precisely:** independent in the current
manifests (no shared API), but coupled at the dependency-graph level
through `digest`/`crypto-common`, and that coupling is currently a
non-issue (same generation) — it only becomes a real, multi-crate split if
whatever `sha2` fork is chosen tracks the `0.11` generation while argon2
(and ed25519-dalek, and hkdf) stay on crates.io `0.10`-generation. Not
"broken" in the sense of a compile error either way — Cargo will happily
build two `digest` generations side by side; it's wasted compile units and
lockfile noise, not a build failure.

---

## TASK 3 — Fallback options with real costs

**(a) Hold the whole migration until `password-hashes` is reachable.**
Cost: open-ended — Task 1's probe is inconclusive by construction (GitHub's
"not found" is indistinguishable from "not yet replicated"), so there is no
known ETA to hold against. If the repo genuinely does not exist, this
option blocks forever on a fork that will never appear, with no timeout
condition specified anywhere I could find. Safest per the P0 rule
(never fall back to crates.io as a convenience) but has zero forward
progress guarantee.

**(b) Migrate the other crypto deps, leave `argon2` on crates.io.**
Exact duplicate crates that would appear in `Cargo.lock` — conditional on
which generation the other forks target:
  - If the other forks (sha2, ed25519-dalek/x25519-dalek family, hkdf) all
    stay on the **same `digest 0.10` generation** as today (i.e., forks are
    same-version mirrors, not upgrades): **zero duplication.** `argon2` on
    crates.io 0.5.3 dedups cleanly against forked `sha2`/`hkdf`/etc as long
    as they all resolve to `digest 0.10.7` / `crypto-common 0.1.7`. This is
    the cheap, low-risk sub-case.
  - If any fork jumps to the **`digest 0.11` generation** (sha2 0.11.x
    exists on crates.io right now, so this is plausible): duplicated crates
    are `digest` (0.10.7 + 0.11.3), `crypto-common` (0.1.7 + 0.2.2),
    `block-buffer` (0.10.x + 0.12.x) — 3 extra small crate compiles, each a
    few hundred to ~1500 LOC of mostly-generic/trait code. Compile-time
    cost: on the order of a handful of extra incremental crate builds
    (roughly comparable to compiling `base64ct` or `subtle` twice) — low
    single-digit seconds in a cold build, not a structural slowdown.
    Binary-size cost: these are trait/no_std-friendly crates with little
    codegen of their own (the actual hash-loop code lives in `sha2`/
    `blake2` themselves, which would ALSO be duplicated across the two
    generations if both are linked in) — the true cost driver is
    duplicating `sha2`'s or `blake2`'s compiled hash loops themselves, not
    the thin trait crates. Still small in absolute terms for a crate this
    size, but it is exactly the kind of graph noise `CLAUDE.md`'s patch
    warning (`"Patch <crate> ... was not used in the crate graph"`) exists
    to catch — worth flagging, not worth blocking on, IF the fork owner
    confirms generation before wiring it.
  - **This option is silent on P0 compliance**: leaving `argon2` on
    crates.io is explicitly against the "Always depend on the AdaWorldAPI
    fork ... NEVER use the upstream crates.io version" rule unless treated
    as a documented, time-boxed exception (the rule's own escape hatch is
    "STOP and ask" when fork coordinates are unknown — this is that case).

**(c) Something else found in the sources: request/confirm the sha2 fork's
target generation BEFORE wiring it, independent of the argon2 question.**
Since `ed25519-dalek` and `hkdf` already independently pin `digest 0.10`
(see Task 2), the actual lowest-risk sequencing is: migrate `sha2` (and any
other RustCrypto-family crate) to a fork **pinned to the `0.10.x`
generation** first — this keeps today's zero-duplication state intact
regardless of what happens with `argon2` — and treat a `0.11`-generation
jump as a **separate, deliberate** decision requiring `ed25519-dalek` +
`hkdf` forks to move in the same PR, not sha2 alone. This turns option (b)
from "duplication risk depends on luck" into "duplication risk is a chosen
generation, verified before merge." Recommend surfacing this to the
operator alongside the password-hashes reachability question rather than
guessing at a fork's version.

---

## Bottom line

- Probe: **unverified** — `password-hashes` not reachable this session,
  cannot distinguish absent-vs-not-yet-replicated. Retry later.
- Coupling: **not broken today** (argon2/sha2 share `digest 0.10.7`
  cleanly); **becomes a real split** only if a future fork wiring jumps
  `sha2` (or ed25519-dalek/hkdf) to the `digest 0.11` generation while
  argon2 stays on crates.io 0.5.3. No shared API call site exists between
  them in this crate's own code (`kdf.rs` vs `hash.rs`/`hkdf_sha384.rs`),
  so any duplication would be compile-graph noise, not a build break.
- Recommend: pursue (c) — confirm/pin the sha2-family fork generation to
  `0.10.x` now, re-probe `password-hashes` later, treat a `0.11` jump as
  its own explicit, operator-approved decision.
