# Consolidation Arc Codex P0 Audit — Verdict

Auditor: Opus codex P0 auditor (Phase 11 of autoattended sprint protocol)
Branch audited: `claude/pr-x4-splat-cascade-design` @ `5e266d19`
Compared against: `origin/master` (37 commits ahead)
Diff scale: 79 files, +18,736 / −5,705 lines (~6,000 lines new Rust + ~900 lines TTL)

Verdict: **NEEDS-FIX**

P0 count: **3**
P1 count: 4 (advisory)
P2 count: 2 (defer to P2 savant)

## P0 findings (must fix before ready-for-review)

### P0-1 — `hpc::linalg::hilbert::hilbert3d_encode` produces wrong index at level 4 boundary

`src/hpc/linalg/hilbert.rs:135-154`. The Butz / Lam-Shapiro 3D Hilbert encoder is broken at the maximum-coordinate boundary. Unit test `max_position_maps_to_max_index_level4` (file line 232) asserts that encoding `[15, 15, 15]` at level 4 must yield `4095` (the maximum 12-bit index), but the implementation returns `2925`. The test FAILS under `cargo test -p ndarray --lib --features std,linalg,ogit_bridge,pillar hpc::linalg`.

Root cause: the state-transition table `NEXT_STATE` and/or `H_TO_XYZ` permutations at lines 71-116 do not satisfy the bijection property at the maximum-corner orbit. The author's claim that "decode(encode(pos, level), level) == pos for all pos and level" (line 23) is contradicted by the 4095-index endpoint check.

**Patch language**:
- (i) Either rederive `H_TO_XYZ` / `NEXT_STATE` from a verified reference (Hamilton 2006 Table 2 or Skilling 2004 "Programming the Hilbert curve") and regenerate; or
- (ii) Scope-cut Hilbert-3D out of this PR. The commit message for `c043cf1e` already documents "Hilbert-3D scope-cut" as an earlier scope decision; the restart commit `59082f70` reintroduces the broken code. Reverting `59082f70` would close the gap without a rederivation.

This blocks ready-for-review absolutely — a failing `cargo test` on the trunk-equivalent gates is a P0 by every PR-X3 / W3-W6 audit precedent in this workspace.

### P0-2 — `cargo fmt --all -- --check` fails with 141 format violations

Running `cargo fmt --all -- --check` exits non-zero with 141 distinct `Diff in ...` reports spanning the bulk of the new linalg / pillar / ogit_bridge files (e.g. `src/hpc/linalg/attention.rs` 23 violations, `src/hpc/linalg/conv.rs` ~10, `src/hpc/pillar/temporal_sandwich.rs` multiple). Gate 7d is a hard fail.

**Patch language**: run `cargo fmt --all` once on the branch; all 141 diffs are auto-applied stylistic changes (collapsed array-of-arrays, `assert!` formatting, line breaks). No semantic risk.

### P0-3 — `cargo test --doc -p ndarray ... hpc::linalg` fails on Hilbert doctest

`src/hpc/linalg/hilbert.rs:162-165`. The doctest

```rust
# use ndarray::hpc::linalg::{hilbert3d_encode, hilbert3d_decode};
assert_eq!(hilbert3d_decode(0, 1), [0, 0, 0]);
```

fails to compile under the crate's `#![deny(warnings)]` lint level — `hilbert3d_encode` is imported but unused in this snippet, and `unused_imports` is denied via the workspace lint group. Gate 7c FAIL.

**Patch language**: drop `hilbert3d_encode` from the import line in the decode doctest, or expand the body to exercise round-trip. The accompanying encode doctest at line 130-134 imports both and exercises both, so the encode side is fine; only the decode-side doctest at lines 162-165 needs the unused-import trimmed.

## P1 findings (advisory — coordinator should apply before Phase 13)

### P1-1 — Missing `# Example` doctests on many public fns

Gate 4 spot-check found multiple files where `pub fn` count exceeds `# Example` block count, and module-level `#![allow(missing_docs)]` is set to suppress the missing-docs lint. Concrete shortfalls (all in new files):

- `src/hpc/linalg/matrix.rs` — 23 pub fns, 7 examples (16 short). Affected: `MatN::{get, set, row, col, trace}`, `Spd2::{new, trace, det, frobenius_sq, is_symmetric_pd, to_mat_n, from_mat_n_symmetric}`, `Spd3::{new, trace, det, frobenius_sq, is_symmetric_pd, to_mat_n, from_mat_n_symmetric}` (under `cfg(not(feature="splat3d"))`).
- `src/hpc/linalg/sh.rs` — 4 pub fns, 0 examples. `sh_coeffs_per_channel`, `sh_coeffs_per_gaussian`, `sh_eval`, `sh_eval_rgb` all lack working doctests.
- `src/hpc/linalg/wasserstein.rs` — 3 pub fns, 0 examples. `sinkhorn_knopp_f32`, `hungarian_f32`, `wasserstein_1_f32` all lack working doctests.
- `src/hpc/ogit_bridge/cognitive_bridge.rs` — 6 pub fns, 0 examples. `CamCodebook::{len, is_empty}`, `CognitiveBridge::{load_embedded, family_of, nearest_basin, codebook}`.
- `src/hpc/ogit_bridge/embedded.rs` — 1 pub fn (`cognitive_ttls`), 0 examples.
- `src/hpc/ogit_bridge/turtle_parser.rs` — 4 pub fns, 2 examples. `TurtleLexer::{new, offset, next_token}`, `TurtleParser::parse` — only two have examples.
- `src/hpc/pillar/ewa_sandwich_3d.rs` — 2 pub fns, 1 example. `ewa_sandwich_3d` itself lacks an example.

The `#![allow(missing_docs)]` at module heads in 15 of the new files is the proximate cause — it silenced the standard ndarray hard rule. Per CLAUDE.md "All public APIs need `///` doc comments with examples". Recommend either removing the allow + adding the missing examples, or downgrading these to P0 if the rule is treated as load-bearing.

### P1-2 — Inconsistent `Spd3` import path between code and doctests in `pillar::koestenberger`

`src/hpc/pillar/koestenberger.rs:34` imports `crate::hpc::linalg::Spd3`, but the doctest at line 71 uses `ndarray::hpc::splat3d::spd3::Spd3`. Both resolve to the same type under `feature = "splat3d"` (which is implied by `feature = "pillar"`), so both compile. The inconsistency is cosmetic but confusing for readers.

**Patch language**: standardise on `ndarray::hpc::linalg::Spd3` everywhere in the new code so PR-X10 is the canonical import path. This matches the design doc's "linalg = the consolidating middle layer".

### P1-3 — `pub fn next_u64 / next_f32 / next_f64 / next_normal_f32` on `SplitMix64` lack Data-flow Rule #3 docstring citation

`src/hpc/pillar/prove_runner.rs:64, 76, 86, 108`. The audit gate (per pr-x3 template) requires every `&mut self` method on a non-builder / non-constructor type to carry a `# Data-flow rule` docstring citing `.claude/rules/data-flow.md` verbatim. `SplitMix64` is a stateful RNG — by definition a generator, not strictly a builder.

The other `&mut self` methods in scope are all genuinely builder / parser internals (`CamCodebook::push`, `TurtleLexer::*`), but `SplitMix64::next_*` is arguably the compute path inside probes.

**Patch language**: add a four-line `# Data-flow rule` block on each `next_*` method noting "RNG is an explicit state-machine generator (PRNG carve-out) — citation: `.claude/rules/data-flow.md` Rule #3, builders / constructors clause." Same pattern as the `RansEncoder::encode_symbol` carve-out from `c043cf1e` P0-1.

### P1-4 — `MatN::set(&mut self, ...)` is a setter on a non-builder foundation type, no rule citation

`src/hpc/linalg/matrix.rs:119`. `MatN<N>::set` exposes mutability on the foundation matrix carrier — neither a builder nor a constructor. Per the same Rule #3 audit gate, this is the strongest case in the diff. Patch: either add the `# Data-flow rule` block (with a "value-class setter" carve-out), or downgrade `set` to `with(row, col, v) -> Self` (functional-update style), which would be cleaner against the rule's spirit.

## P2 findings (deferred to P2 savant)

### P2-1 — 15 `#![allow(missing_docs)]` module-level suppressions

Across `linalg/{eig_sym, matfn, polar, wasserstein, conv, hilbert, svd, rope, sh, attention}.rs`, `pillar/{cov_high_d, pflug}.rs`, `ogit_bridge/{mod, cognitive_bridge, schema}.rs`. Each is a workaround for the missing-`# Example` gap above. P2 savant should rule on whether the policy should be enforced repo-wide (removing all 15 allows + filling in examples) or whether new modules get a 30-day grace window.

### P2-2 — `pub` surface on `CamCodebook` / `BasinAtom` not downscoped

`src/hpc/ogit_bridge/cognitive_bridge.rs` exposes `CamCodebook` and `BasinAtom` as `pub`, but they appear to be implementation details of `CognitiveBridge`. Could be `pub(crate)` or `pub(super)`. P2 savant ruling needed on the public-API minimisation pattern.

## Audit gates — pass/fail summary

| # | Gate | Result |
|---|---|---|
| 1 | Zero per-arch surface (target_feature / cfg / intrinsics / per-arch imports) in linalg / pillar / ogit_bridge | PASS (only two doc-comment mentions; zero actual annotations) |
| 2 | Data-flow Rule #3 docstring on every `&mut self` method on non-builder / non-constructor types | NEEDS-FIX (P1-3 on `SplitMix64::next_*`, P1-4 on `MatN::set`) |
| 3 | Zero distance-aware API surface (`Box<dyn Distance>`, `enum DistanceMetric`, `fn distance<T>`) | PASS (zero matches) |
| 4 | Every public fn / type has a working `# Example` doctest | FAIL (P1-1: ~30+ missing examples; 15 `allow(missing_docs)` suppress the lint) |
| 5 | Cross-PR API consistency (koestenberger ⇄ linalg::Spd3, cognitive_bridge ⇄ turtle_parser+schema, linalg::Spd3 ⇄ splat3d::Spd3) | MOSTLY-PASS (one cosmetic inconsistency, P1-2) |
| 6 | Every `unsafe { }` has `// SAFETY:` comment | PASS (zero `unsafe` blocks in any new file) |
| 7a | `cargo check -p ndarray --features std,linalg,ogit_bridge,pillar,splat3d` | PASS |
| 7b | `cargo test -p ndarray --lib --features std,linalg,ogit_bridge,pillar hpc::linalg` | FAIL (P0-1: hilbert level-4 boundary test) |
| 7c | `cargo test --doc -p ndarray --features std,linalg,ogit_bridge,pillar hpc::linalg` | FAIL (P0-3: hilbert decode doctest unused-import under deny(warnings)) |
| 7d | `cargo fmt --all -- --check` | FAIL (P0-2: 141 format diffs) |
| 7e | `cargo clippy -p ndarray --features std,linalg,ogit_bridge,pillar -- -D warnings` | PASS (no warnings) |

## Net call

The consolidation arc lands ~6,000 lines of Rust + ~900 lines of TTL spanning 22 workers' output, with strong architectural discipline on the headline gates: zero per-arch surface, zero distance-aware API, zero `unsafe`, zero `Box<dyn>` in hot paths, clippy clean. The trunk is structurally healthy.

But three gate-7 failures and one critical correctness bug block ready-for-review:

1. **The Hilbert-3D encoder is mathematically wrong** (P0-1) — the level-4 max-corner test fails outright, falsifying the bijection property the implementation claims. This is the showstopper; the rest of the diff is well-formed.
2. `cargo fmt` was never run on the integrated branch (P0-2) — 141 format violations, all auto-fixable.
3. The Hilbert decode doctest trips `unused_imports` under `deny(warnings)` (P0-3) — one-line fix.

After applying the 3 P0 patches (or scope-cutting Hilbert-3D back out as commit `c043cf1e` originally did), the branch becomes ready-for-review. P1 doc-coverage gaps and the `#![allow(missing_docs)]` policy question should be handed to Phase 13 (P2 savant pre-merge review) before flipping to ready-for-merge.

**Recommended next action**: coordinator decides between (a) rederiving Hilbert-3D tables from a verified reference, or (b) reverting commit `59082f70` to honour the earlier scope-cut. Either path, then `cargo fmt --all`, then fix the decode doctest, then re-run gate 7 — at that point the verdict flips to READY-FOR-PR.

## Sentinel: codex-p0-completed
