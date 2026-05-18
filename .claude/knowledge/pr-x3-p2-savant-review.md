# PR-X3 P2 Savant Pre-Merge Review — Verdict

Reviewer: Sonnet P2 savant (Phase 13 of PR-X3 sprint)
PR: AdaWorldAPI/ndarray#158
Branch reviewed: `claude/pr-x3-cognitive-grid-design` @ `01a70edb`

Verdict: **SHIP-WITH-FOLLOWUPS**

P2 count: 4 (3 applied pre-merge, 1 deferred to PR-X3.1)

## Highest-leverage tightenings (rank-ordered)

1. **P2-3 — drop stray `T: Copy` from `from_grid` + iterator impls** (APPLIED) — `src/hpc/blocked_grid/base.rs:334, 482` and `iter.rs:55, 87, 144, 187, 193`
2. **P2-1 — downscope four helpers to `pub(crate)`** (APPLIED) — `src/hpc/blocked_grid/base.rs:413, 421, 557, 563`
3. **P2-4 — macro deferral wording strengthened with PR-X3.1 marker** (APPLIED) — `src/hpc/blocked_grid/grid_struct_macro.rs:12-18`
4. **P2-2 — typed `field_grid::<I, FieldT>()` accessor** (DEFERRED → PR-X3.1)

## Detailed findings + rulings

### P2-1: `pub` helpers on `GridBlock` / `GridBlockMut` → `pub(crate)` (APPLIED)

Worker A4 added `data_slice()` / `padded_cols_stride()` on `GridBlock` and `data_mut()` / `padded_cols()` on `GridBlockMut` as `#[doc(hidden)] pub` to enable sibling-module access. Leaving them `pub` means downstream consumers can call `blk.data_slice()` and bypass the `# Footgun` guard on `as_padded_slice`. Downscoped all four to `pub(crate)` and dropped the `#[doc(hidden)]` attribute (no longer needed once visibility is tightened).

### P2-2: `field_n::<I>()` type erasure (DEFERRED → PR-X3.1)

Returns `&dyn FieldGridRef` (erased type), matching the W3-W6 `soa_struct!` pattern. Adding a typed `field_grid::<I, FieldT>()` accessor is additive but requires either a `FromFieldGridRef` downcast trait or a macro-generated per-field method — neither is trivially additive without a new trait or extra macro emit arm. The erased form is sufficient for the PR-X3 use case (dimension parity checks); the typed accessor would unlock `let edge: &BlockedGrid<u64, 64, 64> = g.field_grid::<0, u64>()` but no current consumer needs it. Queue for PR-X3.1 alongside macro L2/L3/L4 deferral.

### P2-3: Stray `T: Copy` bound on iterator surface (APPLIED)

`GridBlock::from_grid` / `GridBlockMut::from_grid` carried `where T: Copy` even though their bodies only compute index arithmetic and slice `&grid.data[start..end]` — no `T` value is ever copied. This bound propagated into `Iterator for BaseBlockIter` / `BaseBlockIterMut` / `ExactSizeIterator` / the `impl<T: Copy> BlockedGrid<T, BR, BC>` block holding `blocks_base` / `blocks_base_mut`. A consumer with `BlockedGrid<MyNonCopyType, 8, 8>` could only `get` / `set` cells, not iterate. Removed the bound from all six sites. `BlockedGrid::get` / `set` still correctly require `T: Copy` (they actually copy values).

### P2-4: Macro L1-only deferral wording (APPLIED)

The v1 macro emits `map_l1` / `bulk_apply_l1` / `blocks_l1` on the generated struct; L2/L3/L4 are deferred. The deferral itself is the right call (emitting lockstep `{Name}L2Block` for a four-field struct requires `paste!`-generated types with `N=4` const generics — non-trivial without regression risk). But the v1 deferral note was low-visibility, risking callers cementing per-field workarounds. Strengthened the wording: explicit PR-X3.1 ticket reference + `TODO(PR-X3.1)` marker + dedicated "per-field workaround warning" subsection alerting readers that per-field call sites won't auto-migrate when PR-X3.1 lands.

## CI signal

No fragile tests in the new modules: no timing-dependent, no env-dependent, no `#[ignore]`-gated tests. The `BaseBlockIterMut` raw-pointer lending-iterator carries three `// SAFETY:` annotations accounting for the aliasing invariant — appropriate level of annotation for this pattern. No CI concern.

The `paste = "1"` dep (P1-2 from codex audit) is already in the workspace lock and has zero binary impact.

## Net call

Three P2 tightenings applied as a same-day follow-up commit on this branch. P2-2 (typed `field_grid` accessor) correctly post-merge — queued for PR-X3.1 alongside the macro L2/L3/L4 emission.

After this commit lands, PR #158 flips draft → ready-for-review and advances to the merge ladder.

## PR-X3.1 follow-up backlog

Queued for a small same-week follow-up PR:
1. Emit lockstep `{Name}L{2,3,4}Block` block view types + `map_l{2,3,4}` + `bulk_apply_l{2,3,4}` methods on the macro-generated SoA-of-grids struct
2. Add `field_grid::<I, FieldT>()` typed accessor alongside the existing `field_n::<I>()` erased accessor
3. Naming consistency: rename `GridBlockMut::padded_cols` → `padded_cols_stride` to match `GridBlock::padded_cols_stride`
