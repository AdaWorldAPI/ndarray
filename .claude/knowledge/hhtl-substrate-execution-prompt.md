# HHTL Substrate Execution Prompt — Phase 2 Protocol A, 8 Weeks, 6 Sprints

Master execution prompt for the 8-week / 6-sprint substrate build that takes
PR-X4 + PR-X9 (and their dependencies PR-X10/X11/X12/X13) from **design to
substrate**, culminating in the NARS-revision canary defined in
`hhtl-canary-inhabitance-plan.md`.

Companion docs (read first):
- `pr-master-consolidation.md` — sprint plan + dependency DAG
- `pr-master-consolidation-savant-verdict.md` — Phase 1 verdict (READY-WITH-DOC-FIXES, all 10 patches applied)
- `pr-x4-design.md`, `pr-x9-design.md`, `pr-x10-linalg-core-design.md`, `pr-x11-jc-consolidation-design.md`, `pr-x12-codec-x265-design.md`, `pr-x13-ogit-bridge-design.md` — per-sprint specs
- `hhtl-canary-inhabitance-plan.md` — the integration deliverable
- `vertical-simd-consumer-contract.md` — SIMD primitives W1a contract
- `.claude/rules/data-flow.md` — Rule #3

This prompt is the **copy-paste-into-fresh-session** artifact that spawns
each sprint per Protocol A. It is NOT a single Claude Code session — each
sprint kickoff is its own session (Protocol A semantics make sprints
parallelism-bounded, not session-bounded).

---

## How to use this prompt

For each sprint window in the W1-W8 schedule, copy the relevant **§ Sprint
kickoff** block below into a fresh Claude Code session. Authorize the listed
tools. The session runs the sprint per Protocol A: preflight → 6 savants →
workers → P0 fix → P2 review → merge.

Sessions in different windows are independent and can run on different
days. Sessions within the same window (e.g. PR-X11 + PR-X13 in W3) are
independent and can run in parallel. Each sprint produces its own PR off
`claude/pr-x4-splat-cascade-design` (or successor session branches per
session policy).

---

## Phase 2 Protocol A — the cadence each sprint follows

Every sprint kickoff in the schedule below runs the same 7-step Protocol A:

1. **Preflight skeleton** — coordinator agent writes commented-out Rust:
   all impl blocks `unimplemented!()`, all types stubbed, all doc-comment
   data-flow rules in place, no bodies. ~200-400 LoC depending on sprint
   surface. Goal: get the API shape on the page before bodies exist.
2. **Parallel-savant fan-out (6 specialists, same skeleton, no collision)**:
   - `savant-architect` — layering, target_feature isolation, SoA shape
   - `sentinel-qa` — SAFETY claims, `unsafe` block audit
   - **data-flow-savant** — Rule #3, builder exemption, &mut/&self split
   - **distance-typing-savant** — typed-distance discipline (no `Box<dyn>`)
   - **naming-collision-savant** — symbol clashes with shipped crates
   - **test-coverage-savant** — parity/property/integration test plan
   Each writes a verdict against the preflight skeleton. Verdicts can be
   PASS, BLOCK, or ADVISORY. BLOCK halts the sprint until resolved.
3. **Workers fill bodies** — N workers (per-sprint count below), each
   owning one file, parallel where the dependency graph permits. Workers
   import the preflight types; they do not edit type signatures unless a
   savant explicitly demanded it.
4. **Codex P0 audit on combined diff** — runs against the whole sprint
   diff once all workers report green. Codex is invoked via the existing
   audit harness; output committed to `.claude/knowledge/pr-x{N}-codex-audit.md`.
5. **Coordinator fixes P0s** — every P0 must be resolved before P2 review.
6. **P2 savant pre-merge review** — joint plan-review savant with full
   diff context. Output: SHIP / DO-NOT-SHIP / SHIP-AFTER-X. Committed to
   `.claude/knowledge/pr-x{N}-p2-savant-review.md`.
7. **Merge, integration test, signal next sprint** — merge gates: green
   `cargo test --workspace --features <sprint-feature>`, green
   `cargo clippy -- -D warnings`, SHIP verdict from P2 savant.

PR-X3 had a post-merge UB finding (overlapping `&mut [T]`). Protocol A's
data-flow-savant at step 2 would have caught that. The protocol is
load-bearing — do not skip steps.

---

## Sprint sequencing — the 8-week schedule

| Week | Sprints | Workers | Parallel? |
|---|---|---|---|
| W1-W2 | PR-X10 (linalg-core foundation) | 12 (max fan-out: A1 → A2-A12 parallel) | Internal max-parallel |
| W3 | PR-X11 (jc consolidation) + PR-X13 (OGIT bridge) | 6 + 4 | Yes (independent sprints) |
| W4-W5 | PR-X12 (codec) + PR-X4 (splat cascade) | 8 (effective 4-way per P1-4) + 5 | Yes (independent sprints) |
| W6-W7 | PR-X9 (basin-codebook) | 6 | Single sprint (depends on X12 + X13) |
| W8 | Integration + canary | 3 (canary build, harness, recording) | Single sprint (canary deliverable) |

Total: 44 sprint workers + 6 coordinators + 6 specialist savants (re-used
across sprints — stateless re-roles) over 8 weeks.

---

## § Sprint kickoff — W1-W2: PR-X10 (linalg-core foundation)

```text
You are coordinator for PR-X10 (linalg-core), the foundation sprint of the
HHTL substrate arc. 12 max-fan-out workers; 2-week window; produces the
`ndarray::hpc::linalg::*` surface that every downstream sprint consumes.

READ FIRST:
- `.claude/knowledge/pr-x10-linalg-core-design.md` — the per-worker A1-A12
  decomposition; A12 is MANDATORY Hilbert-3D per joint savant scope-cut
- `.claude/knowledge/pr-master-consolidation.md` — sprint plan + DAG
- `.claude/knowledge/pr-master-consolidation-savant-verdict.md` — P0/P1
  applied state; invariant 12 governs (master ruling: path (b))
- `.claude/knowledge/vertical-simd-consumer-contract.md` — SIMD W1a gate
- `.claude/rules/data-flow.md` — Rule #3

WORKER DECOMPOSITION (12 max-fan-out):
- A1 (sequential) — `linalg/mod.rs` + `MatN<const N>` foundation
- A2 (parallel) — `linalg/quat.rs` (Quat algebra)
- A3 (parallel) — `linalg/spd.rs` (Spd2/Spd3/SpdN, sandwich ops)
- A4 (parallel) — `linalg/eig.rs` (eig_sym_3 closed-form + Jacobi general-N)
- A5 (parallel) — `linalg/svd.rs` (Golub-Reinsch + one-sided Jacobi)
- A6 (parallel) — `linalg/polar.rs` (polar decomposition)
- A7 (parallel) — `linalg/mat_exp.rs` (matrix exponential, Padé)
- A8 (parallel) — `linalg/sh.rs` (spherical harmonics deg 0..=7)
- A9 (parallel) — `linalg/conv.rs` (Conv1d/2d/3d typed wrappers)
- A10 (parallel) — `linalg/attention.rs` (naive + flash, both ship)
- A11 (parallel) — `linalg/norm.rs` + `activations_ext.rs` + `rope.rs`
- A12 (parallel, MANDATORY) — `linalg/hilbert.rs` (Butz/Skilling 3D Hilbert
  encode/decode, ~200 LoC; consumed by PR-X4 splat4d::cascade::CascadeAddr)
- Tier 3 OPTIONAL (rng/vml ext/fft ext/sparse/banded) — ship only if Tier
  1+2 finish in window; defer otherwise

PROTOCOL A — execute the 7 steps in `hhtl-substrate-execution-prompt.md`.
The 6 specialist savants for the preflight review are listed there.

ACCEPTANCE GATES:
- All A1-A12 mandatory items merged with green tests, green clippy, green
  codex P0 audit, SHIP verdict from P2 savant
- `cargo test --workspace --features linalg` passes
- W1a consumer contract honored for every new public SIMD-touching fn
- Type aliases preserve splat3d::Spd3 for backward compat (invariant: full
  type aliases ruling)
- Closed-form + general-N coexist per invariant 12

PR FORMAT: open one PR per worker (A1..A12), all targeting a single
integration branch `pr-x10/linalg-core`. Coordinator merges the
integration branch as one PR to master after Protocol A step 7.

BUDGET: 2 weeks. If A1 slips, all 12 workers slip — coordinator's first
job is unblocking A1 within 48 hours.

NEXT SPRINTS: W3 spawns PR-X11 + PR-X13 in parallel once PR-X10 merges.
```

---

## § Sprint kickoff — W3: PR-X11 (jc consolidation) + PR-X13 (OGIT bridge)

These two sprints run **in parallel**; spawn one session each. They share
no files and have no inter-sprint dependencies.

### PR-X11 (jc consolidation, 6 workers, 1 week)

```text
You are coordinator for PR-X11 (jc consolidation). 6 workers; 1-week
window; moves jc's Spd2/Spd3/Wasserstein/signature/cov_high_d math into
`ndarray::hpc::pillar::*` per invariant 12.

READ FIRST:
- `.claude/knowledge/pr-x11-jc-consolidation-design.md` (Pillar-8 with
  placeholder σ_temporal per joint savant P1-2)
- `.claude/knowledge/pr-master-consolidation.md`
- `.claude/knowledge/pr-master-consolidation-savant-verdict.md`
- The relevant `lance-graph/crates/jc/src/*.rs` files that move

WORKER DECOMPOSITION (6 workers):
- B1 — `pillar/mod.rs` + Pillar-6 (Spd2 ewa_sandwich_2d, from jc)
- B2 — Pillar-7 (Spd3 ewa_sandwich_3d + koestenberger, from jc)
- B3 — Pillar-10 (Pflug Wasserstein-1, from jc/src/pflug.rs)
- B4 — Pillar-8 (temporal_sandwich, NEW; placeholder σ_temporal +
  `TODO(calibrate-pillar-8-σ_temporal)` per P1-2)
- B5 — Pillar-9 (Cov16384 / cov_high_d, Düker-Zoubouloglou CLT)
- B6 — Pillar-11 (Hambly-Lyons signature transform)

PROTOCOL A — 7 steps.

ACCEPTANCE GATES:
- All 6 pillars implemented + probe runners shipped
- Probe PASS gates: PSD rate ≥ 0.999, log-norm concentration verifiable
- `#[deprecated]` markers added to `lance-graph/crates/jc/src/{ewa_sandwich,
  ewa_sandwich_3d,koestenberger,pflug}.rs` with 1-cycle transition note
- `ndarray::hpc::pillar::*` is the canonical home; jc becomes a thin
  probe-runner that imports pillar
- Pillar-8 ships with documented-arbitrary placeholder σ_temporal +
  tracking issue link

PARALLELISM: B1-B6 run in parallel after Protocol A step 1 (preflight)
lands — none of them depend on each other. Hard fan-out = 6.

BUDGET: 1 week. The user's "12 agenten" cadence is the ceiling; this
sprint hits 6 effective because pillars are file-scoped independent.
```

### PR-X13 (OGIT bridge, 4 workers, 1 week)

```text
You are coordinator for PR-X13 (OGIT embedded TTL bundle). 4 workers;
1-week window; replaces the lance-graph-ontology hop with embedded TTL
files via `include_str!` per joint savant P0-3.

READ FIRST:
- `.claude/knowledge/pr-x13-ogit-bridge-design.md` (include_str! confirmed
  per P0-3)
- The 26 OGIT TTL files (mirror PR-Z1's spec)

WORKER DECOMPOSITION (4 workers):
- D1 — `ogit_bridge/mod.rs` + the trait surface
- D2 — `ogit_bridge/cognitive.rs` (per-namespace bridge for cognitive)
- D3 — `ogit_bridge/parser.rs` (Turtle parser over `include_str!` strings)
- D4 — `assets/cognitive/*.ttl` + `embedded.rs` (the 26 TTL files +
  include_str! wiring; ~50 LoC + 900 lines TTL)

PROTOCOL A — 7 steps.

ACCEPTANCE GATES:
- `include_str!` validated UTF-8 at compile time (P0-3 ruling)
- No `include_bytes!` references anywhere in the bridge code
- TTL files baked into the binary (~150 KB compressed)
- Bridge exposes `cognitive_ttls()` returning `&'static [(name, str)]`
- Zero-startup-cost lookup (no runtime parsing for the embedded path)
- `ndarray::hpc::ogit_bridge::*` is the canonical home; lance-graph-ontology
  bridge pattern deprecated

PARALLELISM: D1 sequential (mod.rs foundation), then D2/D3/D4 parallel.

BUDGET: 1 week.
```

---

## § Sprint kickoff — W4-W5: PR-X12 (codec) + PR-X4 (splat cascade)

These two sprints run **in parallel**; spawn one session each. They share
no files but PR-X9 (W6-W7) depends on both.

### PR-X12 (codec, 8 workers / 4-way effective parallel, 2 weeks)

```text
You are coordinator for PR-X12 (x265-style codec for cognitive basin
compression). 8 workers; 4-way effective parallel per joint savant P1-4;
2-week window.

READ FIRST:
- `.claude/knowledge/pr-x12-codec-x265-design.md` — RansEncoder docstring
  per P0-1; tinyvec::ArrayVec<[CtuPartition; 85]> per P0-2; A2-A5 parallel
  then A6-A7 parallel then A8 sequential per P1-4
- `.claude/knowledge/pr-master-consolidation-savant-verdict.md`

WORKER DECOMPOSITION (8 workers, max effective 4-way):
- A1 (sequential) — `codec/ctu.rs` (Ctu carrier + CtuPartition + quad-tree)
- A2 (parallel after A1) — `codec/mode.rs` (CellMode enum, 4 modes per
  P1-4 ruling: skip/merge/delta/escape)
- A3 (parallel) — `codec/predict.rs` (per-mode prediction)
- A4 (parallel) — `codec/transform.rs` (DCT-like spatial xform on cell
  deltas)
- A5 (parallel) — `codec/quantize.rs` (quantization with `RdoConfig`)
- A6 (parallel after A2-A5) — `codec/rdo.rs` (λ-RDO loop + `rdo_cell`)
- A7 (parallel after A2-A5) — `codec/rans.rs` (rANS encoder; encode_symbol
  has the builder-exemption docstring per P0-1)
- A8 (sequential after A7) — `codec/stream.rs` (pack/unpack stream format)

PROTOCOL A — 7 steps.

ACCEPTANCE GATES:
- `RansEncoder::encode_symbol(&mut self)` carries Rule #3 builder
  exemption docstring (P0-1)
- `CtuPartition` quad-tree uses stack-arena pattern (tinyvec::ArrayVec or
  pre-allocated Vec indexed by u16); no `Box<CtuPartition>` heap allocs
  on the RDO loop hot path (P0-2)
- 4 codec modes (skip/merge/delta/escape); 5th mode (basin-shift)
  collapsed into escape
- rANS chosen over CABAC (cognitive symbol skew justifies)
- `cargo test --workspace --features codec` passes
- Compression ratio ≥ 5:1 on synthetic basin codebook fixtures

PARALLELISM: per P1-4 ruling, **4-way max** (A2-A5), not 6-way. A1 → 
[A2,A3,A4,A5] → [A6,A7] → A8.

BUDGET: 2 weeks.
```

### PR-X4 (splat cascade, 5 workers, 1 week)

```text
You are coordinator for PR-X4 (splat4d temporal cascade onto BlockedGrid).
5 workers; 1-week window. Interim worktree path `src/hpc/splat3d/v2/` per
P1-3; public module path `crate::hpc::splat4d::*` from day one via
mod.rs re-export.

READ FIRST:
- `.claude/knowledge/pr-x4-design.md` — module path clarification per P1-3
- `.claude/knowledge/pr-x10-linalg-core-design.md` — A12 Hilbert-3D
  consumed by `CascadeAddr::from_position`
- `.claude/knowledge/pr-master-consolidation-savant-verdict.md`

WORKER DECOMPOSITION (5 workers):
- C1 (sequential) — `splat4d/mod.rs` + `CascadeAddr` type (4 bytes, cache-
  aligned, parent/children via shift-mask)
- C2 (parallel after C1) — `splat4d/cascade.rs` (L1-L4 cascade hops; XOR
  projection; consumes `linalg::hilbert::Hilbert3D::encode` from PR-X10)
- C3 (parallel) — `splat4d/pyramid.rs` (SplatPyramid<T, S: GridStorage<T>,
  BR, BC>; storage is generic over PR-X9's GridStorage trait, defaults
  to BlockedGrid for v1)
- C4 (parallel) — `splat4d/temporal_sandwich.rs` (Pillar-8 consumer +
  temporal drift sandwich)
- C5 (parallel) — `splat4d/raster.rs` (cascade-aware rasterization;
  backward-compat shim wrapping splat3d::tile.rs)

PROTOCOL A — 7 steps.

ACCEPTANCE GATES:
- `crate::hpc::splat4d::*` reachable from day one via mod.rs re-export
  (P1-3)
- CascadeAddr is 4 bytes, deterministic XOR cascade
- L1-L4 hop traversal in <400ns p99 (cache-resident path) — see
  `hhtl-canary-inhabitance-plan.md` performance gate 3
- splat3d::tile.rs becomes a shim, deprecated 1-cycle
- SplatPyramid storage-polymorphic over GridStorage<T> (PR-X9 trait)
- `cargo test --workspace --features splat4d` passes

PARALLELISM: C1 sequential, then C2-C5 parallel. 4-way effective.

BUDGET: 1 week.

NEXT SPRINT: W6-W7 spawns PR-X9 once both PR-X12 and PR-X4 merge.
```

---

## § Sprint kickoff — W6-W7: PR-X9 (basin-codebook, 6 workers, 1.5 weeks)

```text
You are coordinator for PR-X9 (lazy basin-codebook with LazyBlockedGrid).
6 workers; 1.5-week window. Depends on PR-X12 (codec primitives) and
PR-X13 (OGIT bridge). Per P0-4, PR-X9 A5 uses PR-X12's codec primitives
verbatim (no codec re-implementation in this sprint).

READ FIRST:
- `.claude/knowledge/pr-x9-design.md` — GridStorage trait with
  `T: Copy, const BR, const BC` type params per P1-5; A5 narrowed scope
  per P0-4
- `.claude/knowledge/pr-x12-codec-x265-design.md` — the codec surface
  PR-X9 A5 consumes
- `.claude/knowledge/pr-x13-ogit-bridge-design.md` — the OGIT cognitive
  namespace PR-X9 attaches basins to

WORKER DECOMPOSITION (6 workers):
- E1 (sequential) — `cognitive/storage.rs` (GridStorage<T: Copy, const BR,
  const BC> trait + impl for BlockedGrid per P1-5 stable-1.94 fix)
- E2 (parallel after E1) — `cognitive/lazy_grid.rs` (LazyBlockedGrid<T, BR,
  BC>: present-cells in BlockedGrid, absent-cells materialized on demand
  under Rubicon write-back gate)
- E3 (parallel) — `cognitive/codebook.rs` (BasinCodebook: per-cell rANS-
  encoded payload + decode-on-access cache; bounded LRU)
- E4 (parallel) — `cognitive/revise.rs` (NARS revision lifted to
  GridStorage<T>; consumes `ndarray::hpc::pillar::Pillar-7` for
  certification)
- E5 (parallel) — `cognitive/encode.rs` (encode_from_dense using
  `ndarray::hpc::codec::{CellMode, MergeDir, rdo_cell, RdoConfig}` per
  P0-4 — no codec re-impl)
- E6 (parallel) — `cognitive/parity.rs` (BlockedGrid ↔ LazyBlockedGrid
  cell-by-cell parity test harness; integration target)

PROTOCOL A — 7 steps.

ACCEPTANCE GATES:
- GridStorage trait compiles on stable Rust 1.94 (no generic const
  expressions) per P1-5
- LazyBlockedGrid implements GridStorage<T, BR, BC> with on-demand
  materialization under Rubicon write-back gate (single-target gated XOR
  semantics per data-flow.md Rule #3)
- Codec surface imported from PR-X12, not re-implemented (P0-4)
- BlockedGrid ↔ LazyBlockedGrid parity: per-cell L1 distance ≤
  `epsilon_floor` for any RdoConfig
- `cargo test --workspace --features cognitive` passes
- Codebook hit rate target ≥ 95% on warmed-up workload (canary gate
  performance #4)

PARALLELISM: E1 sequential (GridStorage foundation), then E2-E6 parallel.
5-way effective.

BUDGET: 1.5 weeks.

NEXT SPRINT: W8 integration + canary.
```

---

## § Sprint kickoff — W8: Integration + Canary (3 workers, 1 week)

```text
You are coordinator for the integration sprint. 3 workers; 1-week window;
delivers the **NARS-revision canary** defined in
`hhtl-canary-inhabitance-plan.md`.

This sprint is where the substrate stops being parts and becomes a system.

READ FIRST:
- `.claude/knowledge/hhtl-canary-inhabitance-plan.md` — THE canary spec
  (workload, 11 substrate steps, correctness gates, performance gates,
  inhabitance gates)
- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` — Rubicon
  model, zone boundaries, three-legged stool
- `.claude/knowledge/bardioc-weekend-rebuild-prompt.md` — the baseline
  the canary measures against

WORKER DECOMPOSITION (3 workers):
- F1 — `lance-graph/cognitive/nars_actor.rs` (~200 LoC)
  - Ractor actor with mailbox = `PerceptualSurface`
  - Handler = Rubicon crossing: cascade route → basin lookup →
    materialize-on-cold → NARS revise → write-back via gated XOR
  - Per-thought bindspace owned by the message lifetime
  - `&mut self` ONLY in the handler, and only for the gated commit
- F2 — `lance-graph/cognitive/nars_persist.rs` (~200 LoC)
  - Zone-1→zone-2 boundary: typed surface in (NarsBeliefRevision),
    SurrealDB ACID-tx out
  - Typed surface defined in ndarray::hpc::*; no DTO layer
  - Zone-2→zone-3 (sea-orm SQL egress) optional and behind a feature flag
- F3 — `lance-graph/examples/nars_canary.rs` + `lance-graph/benches/nars_canary.rs`
  - End-to-end binary: ingest 1M synthetic perceptual surfaces, route
    through HHTL cascade, revise, commit, measure
  - Bench harness: p50/p95/p99 latency (warm + cold), throughput,
    codebook hit rate
  - 30-second screen recording committed to repo

PROTOCOL A — 7 steps (lightweight — small surface, 3 workers).

CANARY ACCEPTANCE GATES (from hhtl-canary-inhabitance-plan.md):

Correctness (binary, all must pass):
1. Revision output bit-exact (Fingerprint) and ULP ≤ 4 (TruthValue) vs
   `src/hpc/nars.rs::revise` scalar reference, 10,000 seeded revisions,
   zero divergences
2. Cascade routing deterministic across 100 runs
3. No `&mut self` in compute paths (clippy + sentinel-qa audit)
4. No static/lazy_static carrying mutable cognitive state in zone 1
5. Typed surfaces at zone boundaries (no serde_json::Value, no DTOs)

Performance (numeric, all must pass on Zen4 or SPR 8-core, AVX-512):
1. p99 revision latency warm: ≤ 1.5 µs
2. p99 revision latency cold: ≤ 15 µs
3. Cascade-only latency: ≤ 400 ns p99
4. Codebook hit rate after 1M warmup: ≥ 95%
5. Throughput saturated: ≥ 1M revisions/sec per core sustained 10s
6. Working set per worker: ≤ 1 MB
7. ndarray::simd primitive coverage: 100% of hot-path SIMD

Inhabitance (qualitative):
1. Canary code reads like the architecture document — 11 substrate steps
   traceable to 11 specific function calls
2. No Bardioc-shaped code in canary path (no SQL builders, no ES DSL,
   no JanusGraph traversals, no ClickHouse aggregations)
3. Sentinel-qa P0 SAFETY findings on new code: zero
4. 30-second screen recording committed (canary running end-to-end, p99
   on screen, hit rate climbing during warmup)

DELIVERABLE: `.claude/knowledge/pr-x4-x9-canary-results.md` — measured
numbers per gate; SHIP / RE-MEASURE / RE-ARCHITECT decision; comparison
against the Bardioc baseline from bardioc-weekend-rebuild-prompt.md (if
the baseline has been run); next-steps recommendations.

BUDGET: 1 week. If a gate fails, document the failure in the results
doc, then decide:
- Performance fail → re-examine cascade depth, codebook materialization
  cost, or SIMD primitive coverage; patch and re-measure
- Correctness fail → P0; block dependent sprint work until resolved
- Inhabitance fail → re-write the wiring (F1/F2), not the substrate

CANARY OUTCOME:
PASS → HHTL is operationally proved; per-workload Bardioc cutover becomes
    mechanically composable; analytic-tier paths (A: trojan horse, C:
    Databend) can be executed with confidence.
FAIL → HHTL claim is not yet validated; the next session decides whether
    to debug the substrate or revisit the architecture.
```

---

## Cross-sprint operational notes

### Specialist savant rotation

The 6 specialist savants are **stateless re-roles**, not per-sprint
incarnations. The same `data-flow-savant` reviews PR-X10 preflight, then
PR-X11 preflight, then PR-X12 preflight, etc. Reduces savant context-switch
overhead per joint savant decision 6 ruling.

### Codex P0 audits

Run codex on the combined sprint diff at step 4, not per-worker. Output
goes to `.claude/knowledge/pr-x{N}-codex-audit.md`. Coordinators must
resolve every P0 before P2 review at step 6.

### Branch hygiene

Each sprint uses an integration branch (`pr-x{N}/integration`); per-worker
PRs target the integration branch, coordinator merges integration to
master after Protocol A step 7. Avoids 12 simultaneous PRs to master.

### Deprecation cycle

PR-X11 marks jc files `#[deprecated(since="0.X", note="moved to
ndarray::hpc::pillar")]` for one cycle. Removal in cycle N+2. PR-X13
supersedes lance-graph-ontology bridge pattern with the same cadence.

### Feature gate matrix (additive)

```toml
# Default
default = ["std", "linalg"]

# Per-sprint
splat3d        = ["dep:..."]
splat4d        = ["splat3d", "linalg"]
blocked_grid   = ["std"]
linalg         = ["std"]
pillar         = ["linalg"]
codec          = ["std", "blocked_grid"]
ogit_bridge    = ["std"]
cognitive      = ["blocked_grid", "linalg", "codec", "ogit_bridge"]

# Aggregates
cognitive_full = ["cognitive", "splat4d", "pillar"]
```

Default builds stay small; canary opts in to `cognitive_full`.

### Backward compat for splat3d consumers

`pub use crate::hpc::linalg::Spd3 as Spd3;` etc — Rust monomorphizes
across type aliases (same type, not new type). Existing splat3d
consumers compile unchanged after PR-X10 lands.

---

## What this prompt does NOT do

- It does not run the 4-prompt analytic-tier arc (Bardioc baseline,
  trojan horse, Databend). Those are independent and can run in parallel
  with the substrate arc. The canary measures against the Bardioc
  baseline if it has been run; absent that, the canary measures absolute
  numbers.
- It does not migrate Bardioc workloads. The canary proves
  *inhabitability* of one workload; per-workload migration is a follow-on
  multi-month effort.
- It does not address HHTL theory or paper-writing. The canary is the
  operational proof; theory artifacts are downstream.
- It does not contain code. It contains the kickoff prompts for each
  sprint session; code is written inside those sessions.

---

## Done criteria (substrate arc, 8 weeks)

The substrate arc is "done" when:

- All 6 sprints land per the W1-W8 schedule (44 sprint workers + 6
  coordinators + 6 specialist savants)
- `ndarray::hpc::*` 10-submodule layout is the canonical structure
- jc deprecated 1 cycle; lance-graph-ontology bridge pattern superseded
- The NARS-revision canary passes all 3 gate classes (correctness +
  performance + inhabitance)
- 30-second screen recording committed showing canary running end-to-end
- `.claude/knowledge/pr-x4-x9-canary-results.md` written with measured
  numbers and SHIP / RE-MEASURE / RE-ARCHITECT decision

If all six criteria hit on schedule: HHTL is inhabited. Bardioc cognitive-
tier cutover is now a mechanical per-workload migration; the analytic
tier follows path A or path C per the four-prompt arc. The architecture
that started as a strategic document is now an operational substrate.
