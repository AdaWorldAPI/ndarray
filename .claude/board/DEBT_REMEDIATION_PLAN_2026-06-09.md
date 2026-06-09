# Debt Remediation Plan — 2026-06-09 (ndarray)

> **Companion to** `DEBT_REVIEW_2026-06-09.md` (the findings). This doc is the
> **roadmap**: P0-core vs low-hanging fruit, the structural fix, and the
> scoped-agent wave sequencing.
> **Branch:** `claude/quirky-volta-m2r6ak`. **Toolchain:** Rust 1.95.0.
> **Cross-repo twin:** `lance-graph/.claude/board/DEBT_REMEDIATION_PLAN_2026-06-09.md`.

## Hard execution constraints (non-negotiable — set by the user 2026-06-09)

1. **No autofix.** No `cargo clippy --fix`, no blanket mechanical rewrites.
   Autofix breaks more than it fixes here (e.g. `clippy --fix` mangled
   `reader_state.rs` in lance-graph's #479). Work happens via **tightly-scoped
   Sonnet agents that reason and write**, one scope each.
2. **No deletion of unused code or dependencies — ever — without explicit
   per-item confirmation.** All "unused" findings live in a **propose-and-confirm
   queue** (bottom of this doc), never an autonomous action list.
3. **Every wave ends at a named review gate** — `sentinel-qa` for unsafe,
   `product-engineer` for clippy/API/feature-gates — per
   `.claude/rules/agent-cargo-hygiene.md` (fleet without worktrees, edit-only,
   the Opus orchestrator compiles + lints **once** in the shared `target/`).

## Core diagnosis

Debt accumulated because **the gate can't see it**, not because the code is
buggy. Two mechanisms:
- `#![allow(clippy::all, …)]` blankets the entire `hpc/` tree (`src/hpc/mod.rs:1`)
  + all 12 `simd_*` modules (13 `#[allow(clippy::all,…)]` on `src/lib.rs` mod
  decls) — clippy is **off** across the new surface.
- CI clippy runs the **root package only** (`.github/workflows/ci.yaml:59,60,172`)
  — never `--workspace` — so member crates drift ungated.

**Decisive measurement** (force-warn, read-only): lifting the suppression
surfaces **397 default-group warnings**, but their composition is **style, not
bugs** — 199 `needless_range_loop`, 34 `manual_div_ceil`, 29
`too_many_arguments`, 27 `assign_op_pattern`, 18 `should_implement_trait`, …
A targeted scan finds **zero** high-risk correctness/suspicious lints
(`out_of_bounds_indexing`, `not_unsafe_ptr_arg_deref`, `eq_op`,
`transmute_null`, …). The only safety-relevant lint in the pile is **17
`missing_safety_doc`**.

→ Therefore P0 is **restore visibility surgically**, NOT a 397-warning bulk-fix.

## P0 — Core (structural; small, surgical, high-leverage)

| # | Item | Why P0 | Effort | Owner / gate |
|---|---|---|---|---|
| **C1** | Un-suppress **only** `missing_safety_doc`: add `#![warn(clippy::missing_safety_doc)]` to `src/hpc/mod.rs` + the `simd_*` modules *alongside* the existing blanket allow (named `warn` overrides the group `allow`). | Surfaces the 17 lib + 4 `blas-mock-tests` public-unsafe-fn doc gaps **without** eating the 397 style warnings. Makes the "every unsafe needs a contract" hard rule enforceable. | XS (attrs) + agent doc-writing | `sentinel-qa` |
| **C2** | Widen CI clippy from root-package to `--workspace` (select a BLAS backend, or `--exclude blas-tests numeric-tests` which carry an intentional backend `compile_error!`). | Exactly the hole the cesium `erasing_op` RED slipped through. Permanently closes member-crate drift. | XS (CI YAML) | `product-engineer` |

(C3 = lance-graph `protoc` in CI — see the twin doc.)

## Low-hanging fruit (additive, safe, scoped — no deletion)

- ✅ **cesium `erasing_op`** — DONE & pushed (`c68596a`): named `(k, ch)` indices,
  no suppression. Verified `cargo clippy -p cesium --all-targets` exit 0.
- **`blas-mock-tests`: write 4 `# Safety` sections** (`crates/blas-mock-tests/src/lib.rs:17,39,61,83`)
  — purely additive doc; pairs with C1.

## Leave alone / gate (do NOT touch autonomously)

- **397 style lints + 411 pedantic `cast_possible_truncation`** — expected
  SIMD/numeric noise. **Do not bulk-fix.** Structural treatment is W2: convert
  the blanket `clippy::all` allow into an *enumerated, commented* allow-list so
  the debt is **named, not hidden** (documentation, ~zero churn, no autofix).
- **Unused code/deps** → **propose-and-confirm queue** (bottom). Nothing removed
  without sign-off.

## Structural fix (prevents recurrence)

1. **Every member crate is gated** (C2) — no ungated corner where a RED hides.
2. **Suppression is enumerated + justified, never blanket** (W2) — a *future*
   correctness lint then fires instead of being swallowed.
3. **Unsafe enforced at the level the project claims.** C1 enforces
   `missing_safety_doc` now. The stricter hard-rule (`// SAFETY:` on **all 460**
   `hpc/`+`simd_*` blocks; only **148** present) is an **optional P1
   "raise-the-bar"** via `clippy::undocumented_unsafe_blocks` — ~300 reasoned
   comments, a real commitment, only if you want it.
4. **Unused is tracked-and-confirmed, not silently allowed** — the queue below
   moves to "removed" only on your tick.

## Wave sequencing (ndarray slice)

| Wave | Scope | Agent(s) | Gate | Acceptance |
|---|---|---|---|---|
| **W0** | C2 (CI YAML) + blas-mock 4 safety docs | `product-engineer` (1 scope each) | Opus lints `--workspace` once | workspace clippy GREEN w/ backend; 4 docs land |
| **W1** | C1: re-enable `missing_safety_doc`, then write the 17 lib + 4 mock SAFETY docs — **one agent per module/file** | `sentinel-qa`-reviewed Sonnet fleet | `sentinel-qa` per doc | `clippy -W missing_safety_doc` clean; no logic change |
| **W2** | Enumerate the blanket `clippy::all` allow → explicit commented list (names the 397) | `product-engineer` | Opus lint | `-D warnings` still green; debt now *named* |
| **W3** | Present the propose-and-confirm queue → act ONLY on confirmed items | (review) | **user sign-off per item** | nothing deleted without a tick |

### Scoped-agent mission template (every wave)
> Scope: ONE file/module. Model: Sonnet. **Edit-only; no worktree; no
> `cargo build`/`check`; NO `clippy --fix`; NO deletion of any item.** Read
> `DEBT_REVIEW_2026-06-09.md` + this plan first. Make the additive/rewrite
> change in-scope, leave a one-line justification on any retained `#[allow]`.
> The Opus orchestrator compiles + lints centrally. End at the named gate.

## Propose-and-confirm queue (ndarray) — NOTHING removed without your sign-off

**Dead-code / suppression (from `UNUSED_INVENTORY_1.95.md` A-series):**
- A1 `src/impl_owned_array.rs:7`, A2 `src/iterators/mod.rs:24` — stale "Rust 1.64"
  `unused_imports` suppressions; *test* if the import is still needed, propose
  removal of the **suppression** (not necessarily the import).
- A3 `src/hpc/simd_dispatch.rs:47` — phantom `SimdTier::{Sse2,WasmSimd128}`
  (never selected in `detect()`). **Decision needed: delete OR wire** — not auto.
- A4 `src/hpc/clam.rs:928`, A5 `src/hpc/clam_compress.rs:141`, A6
  `src/hpc/jitson/scan_config.rs:144`, A7 `src/impl_ref_types.rs:364`, A8
  `src/backend/native.rs:376,423`, A9 `src/hpc/packed.rs:29,42` — dead helpers /
  unused vars; **domain-owner review each** before any change.

**Unused deps (`cargo-machete`) — main `ndarray` lib is CLEAN; aux crates only:**
- `crates/burn` → 14 (vendored crate — likely partial vendor; confirm intent).
- `crates/p64` → `fractal`.
- `crates/blas-tests` `{blis-src,netlib-src,openblas-src}`, `crates/numeric-tests`
  `openblas-src` — **false positives** (`-src` link crates; keep).

Cross-ref: `DEBT_REVIEW_2026-06-09.md`, `UNUSED_INVENTORY_1.95.md`.
