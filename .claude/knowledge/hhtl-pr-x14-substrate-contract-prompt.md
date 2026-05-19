# HHTL Substrate — PR-X14′ Pre-Sprint Prompt (lance-graph-contract::column + bridge)

> Date: 2026-05-19 (drafted alongside `hhtl-gridlake-pre-sprint-prompt.md`
> on branch `claude/gridlake-pre-sprint-prompt`, parent commit `ade8edb2`)
>
> Status: Pre-sprint kickoff — slots concurrent with PR-X1 + PR-X2 (GridLake)
> at W2.5, OR immediately after at W3. Companion to:
> - `hhtl-gridlake-pre-sprint-prompt.md` — the carrier (`MultiLaneColumn` +
>   `#[derive(SoA)]`) that this contract wraps
> - `hhtl-substrate-execution-prompt.md` — W1-W8 master schedule
> - `stack-consolidation-bardioc-to-hhtl.md` — § "Column-substrate identity"
>
> **Q-NEW-2 marker present** — placement choice (concurrent W2.5 vs sequential
> W3 after GridLake lands) is the plan-review savant's call at preflight.

## Why this exists — four duplicate column-access patterns

Hot paths across the lance-graph workspace each implement their own column
access today. Four distinct patterns coexist; none share a substrate carrier.
**The carrier the GridLake prompt names (`MultiLaneColumn`) needs a contract
crate as its consumer-facing surface, or each consumer rebuilds the same
bridge.**

### Pattern 1 — flat owned buffers (3× `Box<[u64]>` + 1× `Box<[f32]>`, BindSpace, SIMD-hottest reader)

`cognitive-shader-driver/src/bindspace.rs`:36-44 (verbatim):

```rust
pub struct FingerprintColumns {
    pub content: Box<[u64]>,      // len * 256 u64 = flat SoA
    pub cycle:   Box<[f32]>,       // len * 16_384 f32 = Vsa16kF32 carrier
    pub topic:   Box<[u64]>,
    pub angle:   Box<[u64]>,
    pub sigma:   Box<[u8]>,        // 1 byte per row
}
```

Comment at line 51 explicit: *"row-major `Box<[u64]>` gives O(1)
`chunks_exact(256)` iteration which LLVM autovectorises cleanly."*

This pattern is **already MultiLaneColumn-shaped** — flat byte storage,
multi-row chunked SIMD access, separate column per field. But it is owned not
`Arc`-shared, and bindspace rolled it by hand because no shared carrier
exists in the workspace.

### Pattern 2 — `HashMap<String, RecordBatch>` (lance-graph query + SQL path)

`lance-graph/src/query.rs`:243-245 (verbatim):

```rust
pub async fn execute(
    &self,
    datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ...
) -> Result<arrow::record_batch::RecordBatch>
```

`lance-graph/src/sql_query.rs`:69 mirror:

```rust
pub async fn execute(&self, datasets: HashMap<String, RecordBatch>) -> Result<RecordBatch>
```

Cypher path normalizes via `normalize_record_batch` at line 97; SQL path runs
through datafusion's `SessionContext`. **The datafusion dep lives here
(`lance-graph/Cargo.toml`:35-39) and stays untouched** — SQL is deferred per
the 2026-05-19 architecture decision.

Every consumer downcasts `Arc<dyn Array>` from `batch.columns()` itself —
duplicate downcast/extract logic in each Cypher operator and each
DataFusion-backed SQL caller.

### Pattern 3 — `Morsel { columns: Vec<ColumnData> }` placeholder enum (the planner)

`lance-graph-planner/src/physical/mod.rs`:37-58 (verbatim) — **explicit
placeholder, not yet wired to Arrow**:

```rust
/// A morsel of data (batch of rows) flowing through the pipeline.
/// In the real implementation, this wraps Arrow RecordBatch.
#[derive(Debug, Clone)]
pub struct Morsel {
    pub num_rows: usize,
    /// Column data (placeholder — real impl uses Arrow arrays).
    pub columns: Vec<ColumnData>,
}

/// Column data in a morsel (placeholder for Arrow integration).
#[derive(Debug, Clone)]
pub enum ColumnData {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    String(Vec<String>),
    Fingerprint(Vec<Vec<u64>>),
    TruthValue(Vec<(f64, f64)>),
}
```

The planner has the IR (`ir/{expr.rs, logical_op.rs, schema.rs}`) and the
cost model (`plan/{cost.rs, dp_enumerator.rs}`); its physical executor is
awaiting a real Arrow bridge. **The comment "placeholder for Arrow
integration" is the gap PR-X14′ closes.**

### Pattern 4 — `lance::Dataset.scan()` direct (optional-feature consumers)

Two feature-gated consumers each call lance directly:

- `crates/holograph/Cargo.toml`:38 — `lance = { version = "=4.0.0", optional = true, default-features = false }`
- `crates/lance-graph-ontology/Cargo.toml`:35-47 — `lance = { version = "=4.0.0", optional = true }` behind feature `lance-cache`

Each opens `lance::Dataset` and calls `.scan()` returning
`Stream<RecordBatch>`. Each rolls its own scan-bridge per feature flag. A
third candidate (`bgz-tensor` with optional `lance-graph-contract` dep) is
positioned to need the same bridge.

### ndarray's dep direction (clarification)

`grep` confirms **ndarray has zero direct dep on `lance` or `lancedb`**. The
"ndarray crates depending on lance" you named are lance-graph crates (jc,
lance-graph-planner, cognitive-shader-driver, …) that depend on **ndarray
↑** (for SIMD primitives via `crate::hpc::*` + `crate::simd::*`). The
dep direction is **ndarray-as-substrate**, not ndarray-needs-lance.

PR-X14′ preserves this direction. The contract crate lives in the lance-graph
workspace; the bridge crate lives in the lance-graph workspace; ndarray
remains substrate-pure with zero lance/arrow surface.

## The contract crate already exists — but lacks a column module

`crates/lance-graph-contract/src/lib.rs`:1-12 (verbatim):

```
//! # lance-graph-contract — The Single Source of Truth
//!
//! Zero-dependency trait crate that defines the contract between:
//! - **lance-graph-planner** (implements these traits)
//! - **ladybug-rs** (calls Planner + CamPq + OrchestrationBridge)
//! - **crewai-rust** (calls ThinkingStyleContract + MulContract)
//! - **n8n-rs** (calls JitContract + OrchestrationBridge)
```

10+ workspace crates already depend on it (jc, sigma-tier-router,
lance-graph-ontology, lance-graph-archetype, lance-graph-planner,
cognitive-shader-driver, bgz-tensor, lance-graph-rbac, lance-graph-cognitive,
lance-graph-callcenter).

But its 44 modules are **cognitive contracts** (`thinking`, `mul`, `cam`,
`jit`, `nars`, `qualia`, `collapse_gate`, `cycle_accumulator`, `splat`,
`vsa`, …). **There is no `column`, no `multi_lane_column`, no `soa`, no
`lance_source`.** The column-substrate contract is the missing module.

## What PR-X14′ is, mechanically

Two pieces. One module set added to the existing `lance-graph-contract`
crate, plus one new sibling bridge crate.

### Piece 1 — `lance-graph-contract::column` module set (ADD)

Submission target: existing crate `crates/lance-graph-contract/src/`.
Preserves the "zero-dependency trait crate" invariant from lib.rs:1.

```
crates/lance-graph-contract/src/                    (existing, ADD column/)
└── column/
    ├── mod.rs                       // re-exports
    ├── multi_lane_column.rs         // MultiLaneColumn carrier — owns the surface
    │                                //   PR-X1 (GridLake) emits this type
    ├── source.rs                    // GridLakeSource trait
    │                                //   pub trait GridLakeSource {
    │                                //       fn column(&self, name: &str) -> Result<MultiLaneColumn>;
    │                                //       fn columns(&self) -> impl Iterator<Item = (&str, MultiLaneColumn)>;
    │                                //       fn schema(&self) -> &SchemaRef;
    │                                //   }
    ├── soa.rs                       // SoaSource<S: SoaSchema> trait surface
    │                                //   PR-X2 (#[derive(SoA)]) emits impls to this
    ├── lane_view.rs                 // typed lane iterator surfaces
    │                                //   F32x16Iter / U64x8Iter / Bf16x32Iter handles
    │                                //   (delegating to ndarray::simd types)
    └── schema.rs                    // SchemaRef alias + minimal column-metadata
                                     //   (kept thin; do NOT pull arrow-schema here —
                                     //    re-export a slim view-shaped type that the
                                     //    bridge crate adapts to/from arrow_schema)
```

Crate-level Cargo.toml change (`crates/lance-graph-contract/Cargo.toml`):
- Add `ndarray = { path = "../../../ndarray", default-features = false, features = ["std", "hpc-extras"] }` (already a transitive dep via lance-graph-planner; here it becomes direct because `lane_view` needs `ndarray::simd::F32x16` etc.)
- Do NOT add `arrow`, `arrow-array`, `arrow-schema`, `lance` — those live in the bridge crate.

**Zero-dependency invariant**: the contract crate after this change depends
on `ndarray` (path) + `serde` (already present) + `thiserror` (already
present) + `glob` (already present, build-script only). No arrow, no lance,
no datafusion.

### Piece 2 — `lance-graph-contract-bridge` (NEW sibling crate)

```
crates/lance-graph-contract-bridge/                 (NEW)
├── Cargo.toml                       (deps: lance-graph-contract,
│                                          lance = "=4.0.0",
│                                          arrow = "57",
│                                          arrow-array = "57",
│                                          arrow-schema = "57",
│                                          ndarray (path))
└── src/
    ├── lib.rs
    ├── record_batch_to_mlc.rs       // RecordBatch → MultiLaneColumn
    │                                //   Zero-copy where alignment permits:
    │                                //   if Arrow buffer is 64-byte aligned
    │                                //   and length is a lane-multiple, wrap
    │                                //   Arc<[u8]> over the existing buffer.
    │                                //   Otherwise: one-shot pad+align copy.
    ├── lance_dataset_to_mlc.rs      // lance::Dataset.scan() → MultiLaneColumn stream
    │                                //   pub async fn scan_as_mlc(
    │                                //       dataset: &Dataset,
    │                                //       columns: &[&str],
    │                                //   ) -> impl Stream<Item = Result<HashMap<String, MultiLaneColumn>>>
    ├── arrow_array_view.rs          // Arc<dyn Array> → typed lane view
    │                                //   Type-by-type bridge: PrimitiveArray<Float32Type>
    │                                //   → MultiLaneColumn { width: 4 } with F32x16 lane iterator
    └── soa_record_batch.rs          // impl<S: SoaSchema> SoaSource<S> for RecordBatch
                                     //   (the dual: take a RecordBatch shaped per the
                                     //    derive macro's schema declaration, project
                                     //    to typed SoA struct)
```

Crate ownership: lives at `crates/lance-graph-contract-bridge/` in the
lance-graph repo. Opt-in by anyone needing Lance/Arrow access; pure-cognitive
consumers (cognitive-shader-driver hot path, jc math kernels) do not pull
lance/arrow through.

## Migration — four patterns collapse to one

| Current pattern | Location | After PR-X1/X2/X14′ |
|---|---|---|
| **1.** `FingerprintColumns { content: Box<[u64]>, cycle: Box<[f32]>, ... }` | `cognitive-shader-driver/src/bindspace.rs`:36-44 | `FingerprintColumns { content: MultiLaneColumn, cycle: MultiLaneColumn, ... }` — same SIMD-hot reader, but bytes can now come from BindSpace-owned (via `MultiLaneColumn::from_box`) OR from a Lance scan (via bridge) OR from a `RecordBatch` (via bridge). Bindspace stops being a source-of-truth duplicate. |
| **2.** `HashMap<String, RecordBatch>` (Cypher + SQL execute) | `lance-graph/src/query.rs`:243, `sql_query.rs`:69 | **Cypher path**: `HashMap<String, MultiLaneColumn>` via bridge crate; Cypher executor gets typed lane views directly. **SQL path: UNCHANGED** — keeps `RecordBatch + datafusion::SessionContext`. Two paths, one substrate. |
| **3.** `Morsel { columns: Vec<ColumnData /* placeholder enum */> }` | `lance-graph-planner/src/physical/mod.rs`:37-58 | `Morsel { columns: Vec<MultiLaneColumn> }`. Drop the placeholder enum entirely. The planner is finally wired to real Arrow bytes via the bridge crate; physical operators receive typed lane iterators directly. |
| **4.** `lance::Dataset.scan()` rolled per consumer | `holograph` (`lancedb` feature), `lance-graph-ontology` (`lance-cache` feature), `bgz-tensor` (proposed) | All three call `lance_dataset_to_mlc::scan_as_mlc()` from the bridge crate. Single scan-bridge impl; the three feature-gated callers share one path. |

**Net clutter-replacement count**: 4 duplicate column-access bridges → 1
typed surface (in the contract crate) + 1 reusable bridge impl (in the new
sibling crate). The SQL/datafusion surface stays untouched as a deliberate
deferred decision; the cognitive/SIMD-hot/planner/optional-Lance surfaces all
share one substrate.

## Schedule slot — Q-NEW-2 for plan-review savant

The 8-week schedule from `hhtl-substrate-execution-prompt.md`:76-91 has not
been amended to include PR-X14′. Two viable insertion points:

### Path (α): Concurrent with GridLake at W2.5

PR-X1 + PR-X2 + PR-X14′ all land in the same 0.5-week slot. Workers fan out:
- A1 (chain dep): `MultiLaneColumn` type signature + `Arc<[u8]>` + alignment
  invariants — must land first, parsing target for all others
- A2-A4 (parallel): SoA derive macro, lane iterator implementations,
  alignment+padding tests
- A5-A7 (parallel, blocked on A1 only): `column/source.rs` + `column/soa.rs`
  + `column/lane_view.rs` in lance-graph-contract
- A8-A10 (parallel, blocked on A1 + A5): bridge crate impls
  (`record_batch_to_mlc`, `lance_dataset_to_mlc`, `arrow_array_view`)

Pros:
- Single coordinator + savant cycle covering GridLake + contract + bridge
- All three pieces land together as one substrate "block"
- W3+ consumers (X11, X12, X13, X4, X9) start with the full substrate
  already in place

Cons:
- 10 workers in W2.5 (heavier than the GridLake-only 4)
- Coordinator's fan-out math is tighter; A1 critical-path slip cascades
- A 0.5-week slot becomes ~5 days at heavy fan-out

### Path (β): Sequential at W3 (after GridLake lands)

PR-X1 + PR-X2 ship at W2.5 (4 workers). PR-X14′ takes W3's first half as a
dedicated 3-day sprint with 4 workers, blocking W3's PR-X11 + PR-X13 to
W3.5+.

Pros:
- Clean dependency: PR-X1 + PR-X2 land before X14′ starts; no chain-dep risk
- 4-worker fan-out matches GridLake shape (one coordinator, one cycle)
- A1 critical-path stays inside GridLake; X14′ workers have a stable target

Cons:
- W3's PR-X11 + PR-X13 shift right by 3 days → 8-week arc becomes 8.5+ weeks
- Two preflight cycles (GridLake savant cycle + X14′ savant cycle) instead
  of one

**The plan-review savant decides α vs β at preflight.** Both paths reach the
same end-state; the choice is fan-out heat vs schedule tightness.

## Worker spawn shape — path (α) detail

If path α is chosen, the worker DAG:

```
A1: MultiLaneColumn carrier (chain dep)
    │
    ├──→ A2: #[derive(SoA)] proc-macro
    ├──→ A3: lane iterators (F32x16/U64x8/Bf16x32 typed handles)
    ├──→ A4: alignment + padding test harness
    │
    ├──→ A5: lance-graph-contract::column::source (GridLakeSource trait)
    ├──→ A6: lance-graph-contract::column::soa (SoaSource trait)
    ├──→ A7: lance-graph-contract::column::lane_view (typed iterator surfaces)
    │       │
    │       └──→ A8: bridge::record_batch_to_mlc
    │       └──→ A9: bridge::lance_dataset_to_mlc
    │       └──→ A10: bridge::arrow_array_view + soa_record_batch
```

A1 is the only chain dep. A2-A10 are all parallel after A1 lands. Critical
path is A1 → (A5 ‖ A7) → A8/A9/A10; typical depth ~3 days.

## Done criteria

The sprint is done when ALL of the following hold:

1. **Contract module set** lands in `crates/lance-graph-contract/src/column/`
   - All five files present (`mod.rs`, `multi_lane_column.rs`, `source.rs`,
     `soa.rs`, `lane_view.rs`)
   - **Zero new external deps** in `lance-graph-contract/Cargo.toml` except
     `ndarray` (path)
   - `cargo build -p lance-graph-contract --no-default-features` passes
     (proves zero-dep invariant preserved)

2. **Bridge crate** lands at `crates/lance-graph-contract-bridge/`
   - `record_batch_to_mlc::wrap_zero_copy` returns `MultiLaneColumn` from an
     already-64-byte-aligned, lane-padded Arrow buffer with **0 allocations**
     (verified by `dhat` or `cargo bench` heap-counter)
   - `record_batch_to_mlc::pad_and_align` returns `MultiLaneColumn` from an
     unaligned/short Arrow buffer with exactly **1 allocation** (the padded
     `Arc<[u8]>`)
   - `lance_dataset_to_mlc::scan_as_mlc` round-trips a Lance dataset → MLC
     stream → reassembled `RecordBatch` byte-identical to the source

3. **Parity tests against the four current patterns**:
   - **BindSpace migration parity**: `FingerprintColumns` rebuilt over
     `MultiLaneColumn` produces byte-identical SIMD sweep output to the
     current `Box<[u64]>` impl on a 1M-row fixture
   - **lance-graph Cypher parity**: a representative Cypher query over a
     dataset routed through `MultiLaneColumn` (via bridge) produces identical
     result to the current `RecordBatch`-HashMap path
   - **Planner Morsel parity**: a representative `Morsel`-shaped pipeline
     (BROADCAST → SCAN → ACCUMULATE → COLLAPSE) executes against
     `Vec<MultiLaneColumn>` and produces identical row counts + identical
     fingerprint output to the placeholder-enum path
   - **`lance::Dataset.scan()` parity**: holograph + ontology call
     `scan_as_mlc` and produce identical row data to the current per-feature
     scan implementation

4. **Benchmarks**:
   - Zero-copy MLC wrap of pre-aligned Arrow buffer: **<50 ns** (one Arc
     atomic + struct construction)
   - Lane iterator over `MultiLaneColumn::iter_f32x16` on 16K-row column:
     **within 5%** of the equivalent `chunks_exact(16).map(F32x16::from)`
     loop over `&[f32]` (proves no abstraction tax)
   - BindSpace SIMD sweep over `MultiLaneColumn::iter_u64x4` for fingerprint
     hamming: **within 5%** of the current `Box<[u64]>` chunks_exact path
     (proves bindspace migration is free)

5. **Workspace compile**: `cargo check --workspace` clean. All 10+ existing
   `lance-graph-contract` consumers compile without source changes (the
   `column` module is purely additive).

6. **Out of scope (must NOT be touched by this sprint)**:
   - `lance-graph/src/sql_query.rs` and `sql_catalog.rs` — SQL path stays on
     `RecordBatch + datafusion` per the 2026-05-19 deferred-SQL decision
   - `lance-graph/Cargo.toml`'s datafusion deps (lines 35-39) — left
     untouched per the 2026-05-19 "don't evict datafusion" decision
   - The Cypher parser (`lance-graph/src/parser.rs`) — works as-is; only the
     execution path's data interface shifts to MLC
   - ndarray crate — receives zero changes; this PR is consumer-side only

## Forbidden constraints

Two invariants the sprint MUST NOT violate:

1. **The zero-dependency invariant of `lance-graph-contract`** stated in
   lib.rs:1: *"Zero-dependency trait crate that defines the contract
   between [...]"*. Adding `ndarray` (path dep, already transitively
   present) is acceptable because ndarray is the substrate; adding `arrow`,
   `arrow-schema`, `arrow-array`, `lance`, `lancedb`, or `datafusion` to
   this crate **violates the invariant and gates the sprint**.

2. **No write to lance-graph upstream storage format**. The bridge crate
   reads `lance::Dataset` and `arrow::RecordBatch` as-is. It does NOT define
   new Arrow extension types, NOT register custom Lance encoders, NOT
   modify schemas in place. Wrapping via `Arc<[u8]>` over existing buffers
   is allowed; in-place mutation of Arrow/Lance buffers is forbidden.

## Cross-references

- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — PR-X1 + PR-X2
  carrier (`MultiLaneColumn` + `#[derive(SoA)]`) that this contract wraps
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` — W1-W8 master
  schedule (PR-X14′ slots at W2.5-α or W3-β)
- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` § "Column-
  substrate identity — Lance ≡ Arrow ≡ ndarray SoA" lines 126-220
- `.claude/knowledge/pr-arithmetic-inventory.md` § "Shopping-list addendum"
  — needs 2026-05-19 update with PR-X14′ entry
- File evidence cited above:
  - `crates/lance-graph-contract/src/lib.rs`:1-12 (zero-dep claim)
  - `crates/cognitive-shader-driver/src/bindspace.rs`:36-44 (Pattern 1)
  - `crates/lance-graph/src/query.rs`:243-245 (Pattern 2 Cypher)
  - `crates/lance-graph/src/sql_query.rs`:69 (Pattern 2 SQL)
  - `crates/lance-graph-planner/src/physical/mod.rs`:37-58 (Pattern 3 placeholder)
  - `crates/holograph/Cargo.toml`:38 (Pattern 4 holograph)
  - `crates/lance-graph-ontology/Cargo.toml`:35-47 (Pattern 4 ontology)
  - `crates/lance-graph/Cargo.toml`:35-39 (datafusion deps to leave alone)

## TL;DR

PR-X14′ adds a `column/` module set to the existing `lance-graph-contract`
crate (preserving its zero-dep invariant) and a new sibling
`lance-graph-contract-bridge` crate that bridges Lance datasets + Arrow
RecordBatches into `MultiLaneColumn`. Four current duplicate column-access
patterns (BindSpace `Box<[u64]>` + `Box<[f32]>`, lance-graph query `HashMap<String,
RecordBatch>`, planner `Morsel` placeholder enum, `lance::Dataset.scan()`
rolled per feature) collapse to one. SQL path + lance-graph's datafusion
deps are deliberately untouched. The plan-review savant chooses path α
(concurrent with GridLake at W2.5, 10 workers) vs path β (sequential at W3,
4 workers + half-week schedule extension) at preflight.
