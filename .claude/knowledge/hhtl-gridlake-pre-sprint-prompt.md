# HHTL Substrate — GridLake Pre-Sprint Prompt (PR-X1 + PR-X2)

> Date: 2026-05-19 (drafted post-PR #162 merge at master tip `c8f4af68`)
> Status: Pre-sprint kickoff prompt — slots before / inside PR-X10 in the
> 8-week Phase 2 substrate arc. Companion to
> `hhtl-substrate-execution-prompt.md` (W1-W8 master schedule).
>
> **Q-NEW-1 marker present** — the plan-review savant decides path (a) vs
> path (b) at preflight. See § "Schedule slot" below.

## Why this exists — the column-substrate identity is load-bearing

GridLake is the in-process materialisation of the Lance column. The
`MultiLaneColumn` carrier (`Arc<[u8]>`-backed byte storage with typed
multi-width lane views) is not a convenience type; it is the substrate
identity from PR #162 made concrete at the Rust ownership layer.

**Verbatim, from `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md`
§ "Column-substrate identity — Lance ≡ Arrow ≡ ndarray SoA"** (master
commit `c8f4af68`, lines 126-220):

```
Lance dataset (single physical store)
     │
     ▼
Lance column  ≡  Arrow column buffer  ≡  ndarray SoA
                  (one representation, all the way down)
     │
     ├──→ lance-graph: XOR-cascade lookups, cognitive-shader cycles
     │       (ndarray SIMD ops directly over the column bytes;
     │        no copy, no serde, no marshal — the "in-RAM Thought"
     │        IS the Lance column slot)
     │
     ├──→ SurrealDB: SurrealQL parses → reads the same column
     │       LIVE subscription = a watch on column-state predicates
     │
     ├──→ sea-orm: SQL via Lance backend → reads the same column
     │       (Zone-3 egress is materialise-once into PG-shape for the
     │        legacy surface; the source bytes are unchanged)
     │
     ├──→ Databend: analytic SQL → reads the same column
     │       (ndarray::simd kernel swap → operates on the same bytes
     │        the cognitive cascade just operated on)
     │
     └──→ Tantivy: FTS index → built over the same column
```

> **One physical representation, end to end.** The Lance column layout, the
> Arrow column buffer layout, and the ndarray SoA layout are the same bytes
> viewed through three names. The four dialect surfaces (lance-graph cascade,
> SurrealDB, sea-orm, Databend, Tantivy) all parse their respective query
> languages down to operations on those same bytes.

> **ndarray amortises the SIMD primitive across the whole stack.** The same
> kernel that runs the cognitive cascade, that Databend's filter pushdown
> invokes, that Tantivy's indexer reads, that sea-orm projects to legacy
> egress — they are the same kernel on the same bytes. ndarray pays for the
> SIMD primitive once and the entire stack collects rent. No transcode tier,
> no copy boundary, no format conversion at any zone.

And the load-bearing summary (consolidation doc lines 211-215):

> **The column IS the SoA IS the ndarray buffer.** The cognitive cascade,
> the analytic scan, the FTS index build, and the graph traversal all
> operate on the same bytes through the same SIMD kernels. ndarray::simd
> is the common substrate because the substrate is genuinely one thing,
> not four parallel things wearing the same uniform.

`MultiLaneColumn` is what makes this literal in Rust — a single `Arc<[u8]>`
that the cascade reads as `F32x16` slices, that the codec reads as
`U64x8` slices, that the SurrealDB FFI reads as Arrow buffers, that
sea-orm projects to PG-shape rows. **Same bytes. Different lane width.
No copy.** If this primitive doesn't exist before PR-X4 / PR-X9 / PR-X12
ship, those sprints either (a) reintroduce a per-consumer copy boundary
that the consolidation doc explicitly forbids, or (b) reach for raw
`std::slice::from_raw_parts` violating the W1a consumer contract.

That is the gap this pre-sprint fills.

## What GridLake is, mechanically

```rust
// PR-X1: MultiLaneColumn carrier
pub struct MultiLaneColumn {
    bytes: Arc<[u8]>,         // 64-byte-aligned, padded to lane boundary
    elem_count: usize,         // logical length (pre-padding)
    elem_width_bytes: u8,      // 1, 2, 4, 8 (u8/u16/u32/u64/f32/f64)
}

impl MultiLaneColumn {
    pub fn iter_u8x64(&self) -> impl Iterator<Item = U8x64> + '_ { ... }
    pub fn iter_u16x32(&self) -> impl Iterator<Item = U16x32> + '_ { ... }
    pub fn iter_u32x16(&self) -> impl Iterator<Item = U32x16> + '_ { ... }
    pub fn iter_u64x8(&self)  -> impl Iterator<Item = U64x8>  + '_ { ... }
    pub fn iter_f32x16(&self) -> impl Iterator<Item = F32x16> + '_ { ... }
    pub fn iter_f64x8(&self)  -> impl Iterator<Item = F64x8>  + '_ { ... }
    pub fn iter_bf16x32(&self) -> impl Iterator<Item = Bf16x32> + '_ { ... }

    pub fn len(&self) -> usize { self.elem_count }      // logical, NOT byte-padded
    pub fn byte_capacity(&self) -> usize { self.bytes.len() }  // padded
    pub fn as_bytes(&self) -> &[u8] { &self.bytes }     // for Arrow / Lance FFI
}

// PR-X2: #[soa(pad_to_lanes=N)] proc-macro derive
#[derive(SoA)]
#[soa(pad_to_lanes = 64)]   // pad each column to multiple of 64 elements
struct Belief {
    fingerprint: u64,
    truth_f: f32,
    truth_c: f32,
    committed: bool,
}
// derives: BeliefSoa { fingerprint: MultiLaneColumn, truth_f: MultiLaneColumn, ... }
//   plus per-column typed accessors and a `slice(range)` returning row-tuples
```

The `Arc<[u8]>` is the column. The lane iterators are the views. The
derive emits column-typed accessors at compile time (which is what the
"What survives — JITson / Cranelift, cleaner than before" passage in the
consolidation doc names as the load-bearing compile-time pipeline).

## Schedule slot — Q-NEW-1 for plan-review savant

The GridLake sprint is not currently in the 8-week schedule from
`hhtl-substrate-execution-prompt.md` (master commit `c8f4af68`, lines
76-91):

| Week | Sprints | Workers |
|---|---|---|
| W1-W2 | PR-X10 (linalg-core foundation) | 12 (max fan-out: A1 → A2-A12) |
| W3 | PR-X11 (jc consolidation) + PR-X13 (OGIT bridge) | 6 + 4 |
| W4-W5 | PR-X12 (codec) + PR-X4 (splat cascade) | 8 + 5 |
| W6-W7 | PR-X9 (basin-codebook) | 6 |
| W8 | Integration + canary | 3 |

PR-X1 and PR-X2 are needed before PR-X4 / PR-X9 / PR-X12's SIMD-staged
inner loops, because those consumers expect the typed column surface to
already exist. Two viable insertion points exist; the plan-review savant
must choose one at preflight:

### Path (a): W2.5 prerequisite slot — clean but extends schedule

GridLake lands as a dedicated 0.5-week sprint after PR-X10 merges and
before W3 spawns PR-X11 + PR-X13. The new schedule:

| Week | Sprints |
|---|---|
| W1-W2 | PR-X10 (linalg-core, 12 workers) |
| **W2.5** | **PR-X1 + PR-X2 (GridLake, 4 workers, ~3 days)** |
| W3 | PR-X11 + PR-X13 |
| W4-W5 | PR-X12 + PR-X4 |
| W6-W7 | PR-X9 |
| W8 | Integration + canary |

- ✅ Clean scope — GridLake's surface (`MultiLaneColumn` + `#[derive(SoA)]`)
  is the entire PR, no entanglement with linalg primitives
- ✅ Locks `crate::simd::*` surface before X11 / X13 spawn
- ✅ Q-NEW-1 trade-off is visible to W3+ coordinators
- ❌ Extends the 8-week arc to 8.5 weeks
- ❌ Adds a coordinator + specialist savant cycle (extra ~0.5 day overhead)

### Path (b): X10-A13/A14 absorption — cheaper but pollutes PR-X10

GridLake workers become A13 (MultiLaneColumn) and A14 (#[derive(SoA)])
inside PR-X10. The PR-X10 sprint becomes 14 max-fan-out workers instead
of 12; A13/A14 spawn parallel with A2-A12 after A1 lands.

- ✅ No schedule extension — fits inside W1-W2
- ✅ Single coordinator, single specialist-savant preflight covering A1-A14
- ❌ Pollutes PR-X10's scope (linalg primitives + storage carrier in one PR)
- ❌ A13/A14 don't share dependency surface with A2-A12 — they're a
  different concern under the same coordinator
- ❌ Already exists tension: A6 absorbs heel_f64x8 distance kernels per
  the merged salvage note; adding A13/A14 doubles the absorption load
  on PR-X10's review surface

**Recommendation (informational, plan-review savant decides):** Path (a).
The scope cleanliness matters more than 0.5 weeks of schedule when the
deliverable is a typed substrate surface every downstream sprint
consumes. Path (b) is acceptable if the W1-W2 window has slack after A1
unblocks (rare — coordinators historically use the slack for tier-3
optional workers).

**Q-NEW-1**: choose path (a) or (b) before sprint kickoff. Recorded as an
open question for the plan-review savant; do not start workers until the
answer is committed to this doc.

## Worker decomposition (4 workers, max fan-out)

```
A1 (sequential) — `simd_soa/multi_lane.rs`     — MultiLaneColumn carrier + lane iterators
A2 (parallel)   — `simd_soa_derive/`           — #[derive(SoA)] proc-macro crate
A3 (parallel)   — `simd_soa/bench/`            — zero-copy / no-alloc bench harness
A4 (parallel)   — `simd_soa/integration/`      — splat3d + nars consumer probes
```

### A1 — `crate::simd_soa::MultiLaneColumn`

Surface (mandatory):
- `MultiLaneColumn::new_from_slice<T: Pod>(slice: &[T], pad_to_lanes: usize) -> Self`
  — allocates one `Arc<[u8]>`, 64-byte-aligned, padded to lane-multiple
- `iter_u8x64`, `iter_u16x32`, `iter_u32x16`, `iter_u64x8`, `iter_f32x16`,
  `iter_f64x8`, `iter_bf16x32` — each returns `impl Iterator<Item = LaneT> + '_`
- `as_bytes(&self) -> &[u8]` — for Arrow / Lance FFI
- `len()` returns LOGICAL element count (pre-padding); `byte_capacity()`
  returns padded byte count
- All lane iterators are zero-copy: no `Vec` allocation, no `collect`,
  no temporary buffers — verified by A3 bench harness

Backing the W1a consumer contract:
- All lane types are `crate::simd::*` re-exports (`U8x64`, `F32x16`, …)
- All three backends (AVX-512, AVX2, scalar) work because the underlying
  types are already polyfilled
- Every `unsafe` block in `iter_*` carries a `// SAFETY:` comment that
  cites the multiple-of-64 alignment invariant + lane-multiple
  padding invariant

Out of scope for A1 (defer to PR-X4 / PR-X9 / PR-X12 consumers):
- Sliced row-tuple iteration (the `#[derive(SoA)]` macro generates this)
- Cross-column joins or column-level math (consumers compose iterators)
- Lance dataset I/O (PR-X13 OGIT bridge owns FFI; GridLake exposes the
  Arrow-compatible byte buffer, doesn't read or write Lance files)

### A2 — `simd-soa-derive` proc-macro crate

```rust
#[derive(SoA)]
#[soa(pad_to_lanes = N)]      // N ∈ {8, 16, 32, 64}, default 64
struct Belief { fingerprint: u64, truth_f: f32, truth_c: f32, committed: bool }
```

Generates:
- `struct BeliefSoa { fingerprint: MultiLaneColumn, truth_f: MultiLaneColumn,
   truth_c: MultiLaneColumn, committed: MultiLaneColumn }`
- `impl BeliefSoa { pub fn from_aos(rows: &[Belief]) -> Self; pub fn len(&self) -> usize; }`
- `pub fn slice(&self, range: Range<usize>) -> BeliefSoaSlice<'_>`
  returning a `&` to each underlying `MultiLaneColumn` over the range
- Per-column zero-copy accessors typed to the column's lane width

Mandatory invariants:
- `#[soa(pad_to_lanes = N)]` MUST NOT change `BeliefSoa::len()` semantics —
  `len()` always returns the logical row count, not the padded byte count
  (gates Correctness.1 "bit-exact" — padded slots cannot leak into
  consumer output)
- Compile error if a field type is not `Pod` or not `repr(C)`
- Compile error if `bool` columns are used without explicit byte-packing
  hint (`#[soa(bool_repr = "byte" | "bitpack")]`)

Out of scope for A2:
- Schema-evolution hooks (OGIT bridge / DeriveEntityModel in PR-X13)
- Cranelift JIT specialisation (named as "what survives" in PR #162 doc;
  out of scope here, future work)

### A3 — bench harness `simd_soa/bench/`

Mandatory benches (criterion-based):
- `bench_no_alloc` — runs lane iterators over a 1M-row column, asserts
  zero heap allocations under `dhat::Profiler` or equivalent
- `bench_f32x16_throughput` — measures `iter_f32x16` rate vs raw
  `std::slice::chunks_exact(16)`; required ratio ≥ 0.98 (i.e., the
  typed wrapper costs less than 2% vs raw slices)
- `bench_lane_width_swap` — same column, iterated as `iter_u64x8` then
  as `iter_f64x8` in alternation; measures cache friendliness of
  the `Arc<[u8]>`-as-shared-substrate model
- `bench_arc_clone_cost` — `MultiLaneColumn::clone()` (which clones the
  `Arc`, not the bytes) must be O(1) — assert ≤ 50 ns

These benches gate the W2.5 → W3 handoff.

### A4 — integration probes `simd_soa/integration/`

Probe 1 — splat3d backward compat: `splat3d::tile::TileBinning`'s
existing 16×16 fixed-tile arrays can be expressed as `MultiLaneColumn`
slices without changing splat3d's public API. Probe ships as a feature
flag (`simd-soa-splat3d-probe`), opt-in.

Probe 2 — nars revise compat: `hpc::nars::Belief` (the canary's
operand type) can be `#[derive(SoA)]`-ified. Probe shows the SoA layout
delivers the same `revise()` output as the AoS layout, bit-exact, on
10,000 randomly-seeded inputs (mirrors the canary Correctness gate 1).

Probe 3 — Arrow round-trip: A `MultiLaneColumn::as_bytes()` slice can
be reconstituted into an `arrow::array::PrimitiveArray<T>` (zero-copy)
and back. Validates the "Lance column ≡ Arrow column buffer" claim at
the Rust type-system layer.

## Forbidden (negative constraints)

1. **Do NOT re-introduce L1 / L2 / L∞ distance kernels.** PR-X10 A6
   absorbs them under `crate::hpc::linalg::distance` per the merged
   salvage note in `pr-x10-linalg-core-design.md` § "Distance kernels"
   (master commit `c8f4af68`):

   > Absorbs the `heel_f64x8::l1/l2/linf` kernels from PR #160 (lance-graph)
   > — the code is correct, the framing was wrong (it was filed as "Sprint
   > 0a of a four-repo integration arc"; the right home is here, alongside
   > polar / matfn in the linalg core). Bench parity vs the PR #160
   > implementation is part of the A6 acceptance gate, not a separate
   > worker.

   If a GridLake SIMD-staged inner loop wants a distance, **call
   `crate::hpc::linalg::distance::l2_f64_simd`** (which lands as part of
   PR-X10 A6 in W2). Do not implement distance primitives in
   `crate::simd_soa::*`.

2. **Do NOT introduce a new SIMD type.** All lane types are
   `crate::simd::*` re-exports. If GridLake's iterators need a lane
   width that doesn't exist (unlikely — `crate::simd` covers
   u8x64/u16x32/u32x16/u64x8/f32x16/f64x8/bf16x32 already), file an
   extension proposal against `vertical-simd-consumer-contract.md`
   before adding it. PR-X1 must not be the introduction point for a
   new SIMD type.

3. **Do NOT add Lance file I/O.** The GridLake column surface is the
   in-process materialisation. Lance dataset open / commit / scan
   belongs to PR-X13 OGIT bridge. GridLake exposes `as_bytes()` for
   FFI; the FFI itself is downstream.

4. **Do NOT add cross-column compute.** A `MultiLaneColumn` is one
   column. Joins, projections, predicates over multiple columns are
   composed by consumers iterating multiple columns in lockstep.
   GridLake does not own a query layer.

5. **Do NOT add a runtime schema registry.** The `#[derive(SoA)]` macro
   is compile-time. The "ontology evolution → next compile cycle"
   pipeline from the consolidation doc lives at the OGIT-schema-compile
   layer, not at runtime in GridLake.

## Acceptance gates — inherit canary gates, map to GridLake primitives

The canary gates from `hhtl-canary-inhabitance-plan.md` (master commit
`c8f4af68`, lines 64-127) are the substrate's final-form gates. GridLake's
done criteria are the per-primitive contributions that *let* those gates
pass.

### Correctness (binary, from canary lines 69-89)

> **1. Revision output matches scalar reference**: `Fingerprint` (u64)
> bit-exact match against `src/hpc/nars.rs::revise`; `TruthValue` (f, c)
> within ULP ≤ 4 of scalar reference; 10,000 randomly-seeded revisions,
> zero divergences allowed.

GridLake contribution:
- A2's `#[soa(pad_to_lanes=N)]` MUST NOT change `len()` semantics —
  padded slots cannot leak into `revise()` input
- A4 integration probe 2 runs the 10,000-seed scalar-vs-SoA parity test
  on the canary's `Belief` type, asserts bit-exact match before
  PR-X1/PR-X2 ship

> **5. Typed surfaces at zone boundaries**

GridLake contribution: `MultiLaneColumn::as_bytes()` is the Zone-1↔Zone-2
typed surface (Arrow buffer = Lance column = ndarray SoA = same bytes,
per the column-substrate identity). No `serde_json::Value`, no
`HashMap<String, Box<dyn Any>>` — the buffer is the contract.

### Performance (numeric, from canary lines 91-109)

> **6. Working set per worker thread**: ≤ **1 MB** (fits L2 cache on Zen4/SPR)

GridLake contribution: `MultiLaneColumn::iter_u8x64()` zero-copy
semantics verified by A3 `bench_no_alloc` — no hidden allocations on
the hot path. The 1 MB budget assumes the SoA layout fits in L2; if
iterators allocate, the budget is blown.

> **7. ndarray::simd primitive coverage**: 100% of hot-path SIMD ops route
> through `ndarray::simd::*` — zero raw intrinsics in the cognitive path
> (enforced by clippy lint and the W1a consumer contract gate).

GridLake contribution: every lane iterator returns a `crate::simd::*`
type. Consumers cannot route around `ndarray::simd` because
`MultiLaneColumn` doesn't expose typed-slice-of-T accessors — only
lane iterators. This is the *enforcement mechanism* for gate 7 across
PR-X4 / PR-X9 / PR-X12.

### Inhabitance (qualitative, from canary lines 110-126)

> **3. The canary survives a sentinel-qa audit with zero P0 SAFETY
> findings on the new code.**

GridLake contribution: all `unsafe` blocks in `MultiLaneColumn::iter_*`
audited by sentinel-qa with `// SAFETY:` comments citing the
multiple-of-64 invariant + lane-multiple-padding invariant. PR-X1 must
not merge with any P0 finding.

> **2. No "Bardioc-shaped" code in the canary path.**

GridLake contribution: the `#[derive(SoA)]` macro emits direct lane
accessors, not a runtime query DSL. A consumer that wants to filter a
column writes `column.iter_f32x16().filter(|lane| lane.gt(threshold))`,
not `query("SELECT * WHERE x > ?")`. The substrate identity is preserved.

## Cross-references

- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` § "Column-substrate identity" (lines 126-220, master commit `c8f4af68`) — load-bearing architectural justification
- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` § "Salvage from the 2026-05-19 cross-repo rollback" — heel_f64x8 absorption note (forbidden #1)
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` § "Sprint sequencing" (lines 76-91) — 8-week schedule that Q-NEW-1 must patch
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` § "W1-W2: PR-X10 kickoff" (lines 91-144) — kickoff block format mirrored below
- `.claude/knowledge/hhtl-canary-inhabitance-plan.md` § "Measurement gates" (lines 64-127) — canary correctness / performance / inhabitance gates GridLake inherits
- `.claude/knowledge/pr-x10-linalg-core-design.md` § "Distance kernels — `linalg::distance`" (master commit `c8f4af68`) — A6 owns L1/L2/L∞, GridLake must not duplicate
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a contract every `crate::simd_soa::*` public fn obeys
- `.claude/rules/data-flow.md` — Rule #3 (no `&mut self` during computation)

---

## § Sprint kickoff — W2.5 (path a) or W1-W2 inline (path b): PR-X1 + PR-X2 (GridLake substrate)

```text
You are coordinator for PR-X1 + PR-X2 (GridLake substrate), the
in-process column-substrate-identity pre-sprint of the HHTL substrate
arc. 4 max-fan-out workers; 0.5-week window (path a) or 1-week inline
window (path b); produces the `crate::simd_soa::*` surface that every
downstream sprint consumes alongside `crate::hpc::linalg::*`.

PREFLIGHT Q-NEW-1: plan-review savant MUST choose path (a) or path (b)
and commit the answer to `hhtl-gridlake-pre-sprint-prompt.md` before
spawning workers. Default recommendation: path (a). Do not start
without an answer.

READ FIRST:
- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — this doc
- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` § "Column-substrate identity" — load-bearing reason GridLake exists
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` — Protocol A 7-step cadence + sprint sequencing GridLake patches
- `.claude/knowledge/hhtl-canary-inhabitance-plan.md` § "Measurement gates" — gates GridLake inherits
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a gate
- `.claude/knowledge/pr-x10-linalg-core-design.md` § "Distance kernels" — A6 absorption (do not duplicate)
- `.claude/rules/data-flow.md` — Rule #3

WORKER DECOMPOSITION (4 max-fan-out):
- A1 (sequential) — `simd_soa/multi_lane.rs` MultiLaneColumn carrier + 7 lane iterators
- A2 (parallel) — `simd-soa-derive/` proc-macro crate (#[derive(SoA)] + #[soa(pad_to_lanes=N)])
- A3 (parallel) — `simd_soa/bench/` no-alloc + throughput + lane-swap + arc-clone benches
- A4 (parallel) — `simd_soa/integration/` splat3d probe + nars probe + arrow round-trip

PROTOCOL A — execute the 7 steps in `hhtl-substrate-execution-prompt.md`.
The 6 specialist savants for the preflight review are listed there.
Specialist savants of particular interest for this sprint:
- data-flow (Rule #3 enforcement on MultiLaneColumn API)
- SAFETY (every unsafe block in iter_* needs justified // SAFETY:)
- W1a-consumer-contract (every public fn obeys the contract)
- naming-collision (simd_soa vs simd; multi_lane vs lane_view)

ACCEPTANCE GATES:
- A1-A4 mandatory items merged with green tests, green clippy, green
  codex P0 audit, SHIP verdict from P2 savant
- `cargo test --workspace --features simd-soa` passes
- A3 bench_no_alloc passes (zero hidden allocations on iter_* hot path)
- A4 nars-probe passes 10,000-seed scalar-vs-SoA parity test bit-exact
- A4 arrow round-trip passes zero-copy (no buffer allocation)
- W1a consumer contract honored for every new public SIMD-touching fn
- All unsafe blocks audited zero-P0 by sentinel-qa
- L1/L2/L∞ distance kernels NOT present in simd_soa::* (forbidden #1)

PR FORMAT: open one PR per worker (A1..A4), all targeting a single
integration branch `pr-x1-x2/gridlake-substrate`. Coordinator merges the
integration branch as one PR to master after Protocol A step 7.

BUDGET (path a): 0.5 weeks. If A1 slips past day 2, A2-A4 slip — A2
needs MultiLaneColumn types, A3 needs lane iterators to bench, A4 needs
the carrier to probe.

BUDGET (path b): inline with PR-X10's 2-week window. A1 spawns parallel
with A2-A12 after PR-X10's A1 (MatN) lands. A3/A4 spawn after GridLake's
A1 (MultiLaneColumn) lands.

NEXT SPRINTS:
- (path a) W3 spawns PR-X11 + PR-X13 — both must use crate::simd_soa::*
  for any column-shape work
- (path b) W3 spawns as normal; GridLake done-state goes in PR-X10's
  merge message
- W4 PR-X12 codec uses MultiLaneColumn for rANS / leaf storage
- W5 PR-X4 splat cascade uses #[derive(SoA)] for tile / splat columns
- W6 PR-X9 basin codebook uses MultiLaneColumn for the lookup table
- W8 canary uses #[derive(SoA)] on Belief; the canary's bit-exact
  parity test against scalar revise is GridLake's final acceptance
```
