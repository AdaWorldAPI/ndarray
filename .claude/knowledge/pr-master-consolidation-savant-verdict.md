# Joint Plan-Review Savant Verdict — Master Consolidation Arc

Reviewer: Sonnet joint plan-review savant
Branch reviewed: `claude/pr-x4-splat-cascade-design` @ `190dcbe7`
Docs reviewed: 10
Date: 2026-05-18

Verdict: **READY-WITH-DOC-FIXES**

P0 count: **4** (must fix before any sprint spawns)
P1 count: **5** (advisory; sprints can spawn with these unresolved)
P2 count: **3** (defer to per-sprint P2 savants)

## P0 findings (must fix)

### P0-1 — PR-X12 §ANS: `RansEncoder::encode_symbol(&mut self)` missing data-flow rule docstring

`pr-x12-codec-x265-design.md` ~L141 shows `pub fn encode_symbol(&mut self, ...)` with no `# Data-flow rule` docstring section. Rule #3 binds. The rANS encoder legitimately IS a streaming builder (accumulates output bytes); the method is correct in principle but the design doc must carry the builder-exemption justification.

**Patch**: add to `encode_symbol`:
```
/// # Data-flow rule
///
/// `RansEncoder` is a streaming byte-stream builder per
/// `.claude/rules/data-flow.md` Rule #3's builder/constructor exemption.
/// `encode_symbol` accumulates into `self.output`; no shared data is
/// mutated during computation. Caller holds the encoder exclusively per
/// encoding session.
```

### P0-2 — PR-X12 §Core types: `Box<CtuPartition>` heap allocation on hot path

Violates invariant 1 (zero-cost on hot path). RDO loop runs per-cell; allocating a quad-tree node per CTU split blows the cache.

**Patch**: replace `Split([Box<CtuPartition>; 4])` with stack-arena pattern. Quad-tree depth ≤ 3 levels (64→32→16→8), so total CU count per CTU ≤ 1+4+16+64 = 85 nodes. Use `tinyvec::ArrayVec<CtuPartition, 85>` OR a pre-allocated `Vec<CtuPartition>` indexed by `u16`. Document the stack-arena pattern in the doc body.

### P0-3 — PR-X13 §Why: `include_bytes!` vs `include_str!` inconsistency

PR-X13 mixes references to `include_bytes!` (body text) and `include_str!` (Q1 ruling). The two have different SAFETY profiles: `include_str!` validates UTF-8 at compile time; `include_bytes!` requires runtime UTF-8 validation by the Turtle parser.

**Patch**: commit to `include_str!` throughout. Replace all `include_bytes!` references in PR-X13 body text. Remove the UTF-8-boundary item from the SAFETY audit gate (it's no longer a runtime concern). The ~150 KB size estimate stays accurate.

### P0-4 — PR-X9 vs PR-X12 cross-sprint API split

PR-X9 §A5 `encode.rs` independently defines `RdoConfig` + mode-encoder logic. PR-X12 §A2 `mode.rs` + §A6 `rdo.rs` define the canonical version. Two parallel codec surfaces will diverge at integration time.

**Patch**: PR-X9 A5 `encode.rs` MUST import `ndarray::hpc::codec::{CellMode, MergeDir, rdo_cell, RdoConfig}` from PR-X12. PR-X9 A5 scope narrows from "RDO loop + mode picker" to "LazyBlockedGrid encoder using PR-X12 codec primitives". Feature flag dependency already correct (`cognitive = [..., codec, ...]`); only the worker-scope text needs patching.

## P1 findings (advisory)

### P1-1 — PR-X10 §jc consolidation: internal contradiction on invariant 12

PR-X10 Q4 leans "(a) keep jc zero-dep" while master + PR-X11 break it. Confusing.

**Patch**: delete the (a) lean in PR-X10 Q4. State that the master ruling is **invariant 12 + path (b)** for the PR-X11 consolidation. PR-X10 doesn't need to decide; it just ships the ndarray surface.

### P1-2 — PR-X11 Pillar-8: σ_temporal PASS gate currently arbitrary

Pre-staging Pillar-8 without echocardiography calibration risks a PASS gate that always passes.

**Patch**: ship Pillar-8 with placeholder σ_temporal. PASS gate documented as `report.psd_rate >= PILLAR_8_PSD_THRESHOLD` where `PILLAR_8_PSD_THRESHOLD = TODO_CALIBRATE_FROM_ECHOCARDIOGRAPHY`. Tracking issue + `// TODO(calibrate-pillar-8-σ_temporal)` comment in the const block.

### P1-3 — PR-X4 module path inconsistency

PR-X4 says `src/hpc/splat3d/v2/` (interim worktree path) but master maps to `ndarray::hpc::splat4d::*` (final public path).

**Patch**: clarify in PR-X4 that `src/hpc/splat3d/v2/` is the interim worktree location during the migration; public module path is `crate::hpc::splat4d::*` from day one via `mod.rs` re-export. Both paths resolve to the same code.

### P1-4 — PR-X12 worker parallelism overstated

"A2-A7 parallel" is wrong: A6 (RDO loop) has hard dependencies on A2 (CellMode) + A3 (predict) + A4 (transform) + A5 (quantize). True max fan-out is A2-A5 (4-way).

**Patch**: revise to "A2-A5 parallel, then A6+A7 parallel, then A8 sequential." Maximum 4-way effective parallelism, not 6-way.

### P1-5 — PR-X9 `GridStorage` trait: associated const in const-generic position fails on stable 1.94

`pr-x9-design.md` L157 defines `trait GridStorage<T> { const BR: usize; const BC: usize; ... }` and uses `{ Self::BR }` in associated type bounds. Generic const expressions are NOT stable. CLAUDE.md mandates Rust 1.94 Stable.

**Patch**: switch to type-param generics: `trait GridStorage<T: Copy, const BR: usize, const BC: usize>`. Impls become `impl<T, const BR, const BC> GridStorage<T, BR, BC> for BlockedGrid<T, BR, BC>`. Compiles on stable 1.94.

## Rulings on the 10 joint decisions

| # | Decision | Ruling | Note |
|---|---|---|---|
| 1 | OGIT integration | **(c) embedded TTL bundle** | PR-X13 subsumes Z1+Z2 cleanly |
| 2 | jc zero-dep | **break + invariant 12** | feature-flag isolation preserves the property jc isolation was meant to give |
| 3 | Codec coder | **rANS** | cognitive symbol skew (70% skip) compresses better than video luma/chroma |
| 4 | Cross-sprint ordering | **concurrent** | DAG verified, no cycles |
| 5 | 12 workers/sprint | **confirmed as max ceiling** | PR-X10 hits 12; others hit natural fan-out limit (6/4/4/3/2) |
| 6 | Phase 2 Protocol A | **confirmed** | 6 specialist savants — overlap is feature, not bug |
| 7 | Backward compat for splat3d | **full type aliases** | Rust monomorphization works across aliases (same type, not new type) |
| 8 | Pillar count PR-X11 | **6 (Pillar-8 with placeholder σ)** | per P1-2 patch |
| 9 | Codec mode count | **4** | 5th basin-shift collapses into escape mode |
| 10 | PR-X10 closed-form + general-N coexist | **confirmed** | one-paragraph routing guide eliminates fork-in-the-road |

## Critical scope concerns

### Scope-creep — PR-X10 is exceptionally large

Tier 1+2+3 = ~5,000 LoC across 14 files. "12-worker parallel" conceals that each worker owns 300-600 LoC of non-trivial numerical code.

**Recommendation**: treat PR-X10 **Tier 3 as in-sprint optional** (RNG distributions, FFT extensions, sparse GEMM, banded solvers). Ship only if Tier 1+2 finish within the 2-week window. Tier 3 has no downstream sprint blockers.

### Scope-cut — Hilbert-3D missing from all sprint docs (CRITICAL)

Required for `CascadeAddr::from_position` in splat4d cascade. NOT in any sprint doc.

**Recommendation**: assign Hilbert-3D encode/decode to **PR-X10 A12** as a MANDATORY (not optional) Tier-3 item. ~200 LoC, Butz/Skilling algorithm, pure integer, no precision concerns. New file `src/hpc/linalg/hilbert.rs` OR fold into `hpc::vml`.

## Audit gate results

| Gate | Result | Notes |
|---|---|---|
| A: Data-flow Rule #3 | **PARTIAL FAIL** | P0-1 — rANS encode_symbol missing docstring exemption |
| B: Layering rule | **PASS** | zero `#[target_feature]` in example code |
| C: Distance-typing | **PASS** | no `Box<dyn Distance>`, no umbrella enum |
| D: SAFETY-claim | **PARTIAL FAIL** | P0-3 — TTL parser SAFETY ambiguous |
| E: Cross-PR API consistency | **PARTIAL FAIL** | P0-4 — PR-X9 + PR-X12 codec surface split |

## Net call

**Apply 4 P0 patches + the 5 P1 patches + add Hilbert-3D to PR-X10 A12. Then advance to Phase 2 preflight per Protocol A.**

The dependency DAG is correct, the 10 joint decisions are sound, and the overall architecture is coherent. Sprint ordering (W1-W2 PR-X10, W3 PR-X11+X13, W4-W5 PR-X12+X4, W6-W7 PR-X9, W8 integration) is confirmed. All 44 sprint workers can be spawned per the **corrected** parallelism counts (12/6/4/4/3/2 = 31 effective parallel slots; remaining 13 workers run sequentially within their sprints).
