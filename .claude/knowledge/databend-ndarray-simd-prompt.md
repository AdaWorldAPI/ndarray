# Databend + ndarray::simd — Claude Code Flex Prompt

Adopt Databend as the Rust-native ClickHouse successor and inject `ndarray::simd`
into its hot kernel paths. This is the **recommended ClickHouse-tier
migration target** per `stack-consolidation-bardioc-to-hhtl.md` (path C: 0
transcode cost, weeks not years to OLAP parity).

Companion to:
- `ndarray-simd-trojan-horse-prompt.md` (path A — FFI into stock ClickHouse,
  buys time during cutover)
- `bardioc-weekend-rebuild-prompt.md` (the baseline measurement target)

Copy the block below into a fresh Claude Code session. Authorize
`--allowed-tools '*'`, Rust 1.94, Docker.

Budget: 24 hours wall-clock (half the trojan horse — Databend is already
Rust-native, no FFI bridge to build).

---

```text
You are integrating `ndarray::simd` (from adaworldapi/ndarray, AVX-512 default,
`target-cpu=x86-64-v4`) into Databend (datafuselabs/databend, Rust columnar
OLAP on Arrow + DataFusion + Tokio, MIT licensed). The deliverable is a
fork that swaps Databend's SIMD code paths for ndarray::simd primitives,
benchmarks against stock Databend AND stock ClickHouse, and produces a
report comparing all three.

This is path C from the consolidation: Databend is the recommended
ClickHouse successor for the AdaWorldAPI stack's analytic tier. Bardioc's
ClickHouse decommissions when Databend + ndarray::simd reaches parity on
the OLAP workloads that matter.

Spawn 8 parallel workers + 1 coordinator. Git worktrees per worker. Branch:
`databend-simd/{role}-{id}`. Integration via docker-compose stand-up of
three OLAP engines side-by-side.

## Why Databend, not transcode ClickHouse

Full ClickHouse transcode is 5–10 engineer-years. Databend is:
- Rust-native (no FFI bridge needed)
- Arrow + DataFusion + Tokio (compatible with the wider Rust ecosystem)
- ClickHouse-shape SQL dialect (much of TPC-H ports unchanged)
- MIT licensed (clean integration with AdaWorldAPI codebase)
- Already maintained by a funded team (datafuselabs)
- Smaller hot kernel surface than ClickHouse — fewer kernels to swap

Trade-off accepted: Databend's storage format is not ClickHouse-wire-compatible.
The migration plan is workload-by-workload re-ingestion from Bardioc Cassandra
into Databend, not in-place storage swap. Acceptable because Bardioc cutover
already involves dual-write phases (see bardioc-weekend-rebuild-prompt.md).

## Databend SIMD injection targets

Fork Databend at the current stable tag. Add ndarray as a workspace dep.
Replace target SIMD paths with ndarray::simd calls. Tests stay; benches add.

Priority order (most-impact kernels first):

1. **`src/query/expression/src/kernels/filter.rs`** — column filter
   `mask & column` and packed-int boolean evaluation →
   `ndarray::simd::filter_apply_mask`
2. **`src/query/functions/src/aggregates/aggregate_sum.rs`** + `avg.rs` +
   `min_max.rs` → `ndarray::simd::reduce_{sum,min,max,mean}` for all
   numeric types (f32, f64, i32, i64, u32, u64)
3. **`src/query/expression/src/kernels/hash.rs`** — hash-table probing for
   joins and group-by → `ndarray::simd::hash_xxh3_batch`
4. **`src/query/functions/src/scalars/comparison.rs`** — column-vs-column and
   column-vs-literal `< == >` → `ndarray::simd::compare_{lt,eq,gt}`
5. **`src/query/expression/src/kernels/take.rs`** — gather operations for
   selection vectors → `ndarray::simd::gather_{f32,f64,u32,u64}`
6. **`src/common/storage/parquet/`** — parquet decode hot path (bitpack +
   RLE) → `ndarray::simd::{bitpack_decode,rle_decode}`
7. **`src/query/functions/src/scalars/string/`** — substring / position
   functions → `ndarray::simd::substring_find`

Databend test suite is comprehensive — `cargo test --workspace` must pass
unchanged after each swap. SIMD primitives that don't exist yet in
ndarray::simd: document the gap and skip the kernel (becomes a follow-on
ndarray PR under the W1a consumer contract).

## Worker split (8 + coordinator)

| Worker | Target | Role |
|---|---|---|
| W1 | Fork + dep wiring | Fork Databend at stable tag; add ndarray dep; CI setup; bench harness skeleton |
| W2 | Kernel 1 (filter) | Filter / mask kernel swap + parity tests + bench vs stock |
| W3 | Kernel 2 (aggregates) | Sum/avg/min/max for all numeric types + bench |
| W4 | Kernel 3 (hash) | Hash-table probing + group-by + join hash + bench |
| W5 | Kernel 4 (comparison) | Comparison ops + bench |
| W6 | Kernel 5 + 6 (take + parquet) | Gather + parquet decode + bench |
| W7 | Kernel 7 (string) | Substring / position + bench |
| W8 | Three-way bench | docker-compose: stock ClickHouse + stock Databend + ndarray-Databend; identical workload; report generator |

Coordinator: integration testing, cherry-pick to main branch, docker-compose
orchestration, REPORT.md generation.

## Benchmark workload

Run THREE engines against the SAME workload:
- **Stock ClickHouse** (reference performance — the bar to beat or match)
- **Stock Databend** (current Rust-native baseline)
- **ndarray-Databend** (the fork from this prompt)

Workloads:
1. **TPC-H scale factor 10** — Q1, Q3, Q6, Q14 (these stress the kernels
   we swapped: filter, agg, join, group-by). Standard benchmark, comparable
   across the industry.
2. **ClickBench** — datafuselabs' adapted ClickHouse benchmark, ~40 queries
   on a real web-analytics dataset. Directly designed for ClickHouse-vs-X
   comparison.
3. **Cognitive analytics mini-workload** — 100 ad-hoc queries over a
   synthetic NARS-revision log (joins, time-bucketing, top-K aggregation).
   This represents the actual operational-analytics queries the AdaWorldAPI
   stack will run against egressed cognitive state.

Report per engine:
- p50 / p95 / p99 query latency per query
- Cold-cache vs warm-cache latency
- CPU instructions retired (`perf stat`)
- Peak memory
- Indexing/ingestion throughput

Output: `./benchmarks/REPORT.md` with three-column comparison tables.

## Acceptance criteria

Per kernel swap:
1. Bit-exact parity for integer, ULP-bounded for float
2. Within 5% of stock Databend OR faster
3. Existing Databend test suite passes (`cargo test --workspace`)

Per engine:
1. All TPC-H + ClickBench queries return correct results on all three
   engines (cross-validate ClickHouse ↔ Databend ↔ ndarray-Databend)
2. ndarray-Databend ≥ stock Databend on geomean latency
3. ndarray-Databend within 2× of stock ClickHouse on geomean latency (the
   migration story is "Rust-native parity at acceptable cost", not
   "beat ClickHouse on every query")

If ndarray-Databend beats ClickHouse on ANY query: that's a major signal,
call it out in REPORT.md.

## Anti-goals

- Do NOT add new ndarray::simd primitives this weekend. If a kernel needs a
  missing primitive, document the gap and skip the kernel. The gap becomes
  a follow-on ndarray PR.
- Do NOT submit upstream PRs to Databend this weekend. The deliverable is
  the validated fork + benchmark report. Upstream contribution is a
  separate follow-on after numbers are clean and reviewed.
- Do NOT introduce nightly Rust. Databend builds on stable; keep it that way.
- Do NOT optimize Databend's planner / SQL parser / catalog. The point is
  kernel-level SIMD swap, not architecture work.
- Do NOT touch HHTL substrate (PR-X4, PR-X9). This is independent OLAP-tier
  work; HHTL is the cognitive-tier work.

## Time budget (24 hours)

| Hour 0-2 | W1: fork + dep wiring + bench harness skeleton |
| Hour 2-12 | W2-W7 in parallel: kernel swaps + per-kernel benches |
| Hour 12-18 | W8: three-way docker-compose stack + ClickBench run |
| Hour 18-22 | Cognitive mini-workload + report generation |
| Hour 22-24 | REPORT.md write-up + handoff |

If a kernel doesn't reach parity in its allotted window, document the gap
and skip. Honest negatives are also data — they tell us which ndarray::simd
primitives need follow-on work.

## Strategic outcomes (what the report unlocks)

1. **Migration target validated**: if ndarray-Databend reaches Databend
   parity AND is within 2× of ClickHouse on TPC-H + ClickBench, the
   consolidation doc's "Databend is the ClickHouse successor" claim is
   evidenced rather than asserted.

2. **Three-engine reference point**: future Databend or ClickHouse PRs can
   re-run this exact harness and see whether ndarray::simd injection is
   still worth it. Living benchmark, not a one-shot report.

3. **Cognitive-tier evidence**: the cognitive mini-workload demonstrates
   that Databend handles the actual operational-analytics queries the
   AdaWorldAPI stack will issue (post-cognitive egress to SQL). If those
   queries are sub-second on ndarray-Databend, the analytics tier is
   solved without further work.

4. **ndarray::simd cross-validation**: kernels validated against TWO
   engines (Databend benchmarks plus the trojan-horse ClickHouse-via-FFI
   benchmarks) is much stronger evidence than either alone. The
   intersection set (kernels both engines stress the same way) becomes the
   ndarray::simd "battle-tested" subset.

5. **Decommission timeline**: Bardioc ClickHouse can be decommissioned
   per-workload when ndarray-Databend passes the relevant cognitive
   mini-workload subset, not all at once. Risk-bounded cutover.

Begin. Report progress every 4 hours with kernel done / in-progress /
blocked + parity pass-fail + perf delta vs stock Databend AND stock
ClickHouse.
```

---

## Notes for using this prompt

- Databend builds clean on Rust 1.94 stable. ~10 min full build, ~30s
  incremental. No CMake, no JVM, no FFI bridge — pure Cargo.
- ClickHouse stand-up via official docker image (`clickhouse/clickhouse-server`).
- Databend has an official docker image too (`datafuselabs/databend`).
- ClickBench dataset is ~14GB compressed; provision disk accordingly.
- TPC-H generation via `dbgen`; scale factor 10 produces ~10GB.
- The cognitive mini-workload is the most important — it's the only one
  that's actually shaped like AdaWorldAPI's real future queries.

## Composition with other prompts

This prompt sits inside the four-prompt strategic arc:

1. **`bardioc-weekend-rebuild-prompt.md`** — build the OLD stack honest
   (migration baseline measurement target)
2. **`stack-consolidation-bardioc-to-hhtl.md`** — the architectural reframe
   doc (why the NEW stack wins, four-tier picture)
3. **`ndarray-simd-trojan-horse-prompt.md`** — path A: inject ndarray::simd
   INTO the legacy stack (ClickHouse + Tantivy via FFI) — buys time during
   cutover, accelerates legacy
4. **`databend-ndarray-simd-prompt.md`** (this) — path C: adopt the
   Rust-native CLICKHOUSE-shape successor with ndarray::simd injection —
   the actual migration TARGET

Combined timeline:
- Weekend 1: prompt 1 (Bardioc baseline)
- Weekend 2: this prompt (Databend integration)
- Weekend 3: prompt 3 (trojan horse — optional, buys cutover time)
- Ongoing: HHTL development (PR-X4 + PR-X9), workload-by-workload cutover

## Follow-on opportunities (NOT this weekend)

- Upstream PR cadence to Databend: 1 PR per parity-or-better kernel; faster
  cycle than ClickHouse because Rust-native (no FFI review burden)
- Polars integration: same ndarray::simd primitives plug into Polars
  DataFrame ops; weekend follow-on
- DataFusion integration: arrow-rs has SIMD for filter/take/aggregate;
  ndarray::simd could plug in there too, benefiting the entire
  DataFusion-derived ecosystem (Databend, GreptimeDB, InfluxDB IOx, Ballista)
- Quickwit integration: combines Tantivy trojan horse + Databend analytics
  in one operational stack
