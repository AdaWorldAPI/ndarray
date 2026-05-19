# HHTL Canary Inhabitance Plan

Date: 2026-05-19
Status: Phase 2 entry condition — names the canary workload for the 6-sprint substrate arc
Companion docs:
- `stack-consolidation-bardioc-to-hhtl.md` (architectural frame)
- `pr-master-consolidation.md` (6-sprint plan)
- `pr-master-consolidation-savant-verdict.md` (Phase 1 verdict — READY-WITH-DOC-FIXES, all patches applied)
- `hhtl-substrate-execution-prompt.md` (Phase 2 execution flex prompt — sibling to this doc)

## Why this doc exists

The strategic arc proves the new architecture wins **on paper**. The 6-sprint
plan moves PR-X4 + PR-X9 from **design to substrate**. Neither artifact answers
the question the substrate has to answer to count as **inhabited**: when does
one specific cognitive query path *run end-to-end on the new architecture using
the new idioms*?

This doc names the canary. The canary is what closes the gap between
"substrate exists" and "substrate is lived in."

## The canary: NARS revision routed through HHTL cascade

**Workload**: a NARS belief revision triggered by a perceptual surface, routed
through the splat4d cascade to the relevant basin, materializing the basin
codebook entry on demand, returning a revised `TruthValue` via the Rubicon
commit gate, persisted to SurrealDB through a typed-surface adapter.

**Why this workload**:
- It is **architecturally pure** — exercises every load-bearing piece of the
  new substrate (cascade, codebook, Rule #3, Rubicon, per-thought bindspace,
  typed surfaces, zone-1↔2 boundary, ndarray::simd kernels)
- It is **real** — NARS revision is a primary cognitive workload, not a
  synthetic benchmark; the existing Bardioc stack runs it constantly
- It is **measurable** — has a scalar reference implementation in
  `src/hpc/nars.rs` to compare against for correctness
- It is **scoped** — one query path, not a system migration; can be
  retracted without affecting parallel sprint work
- It is **representative** — the result generalizes: if revision-via-HHTL
  works, every other cascade-routed cognitive op works the same way

## What "routed through HHTL" concretely means

Each step exercises a specific substrate primitive. This is the inhabitance
checklist — not the implementation order:

| Step | Substrate piece | Rule / discipline |
|---|---|---|
| 1. Perceptual surface arrives at a Ractor mailbox | Ractor as Rubicon gate (not Erlang) | Per-thought bindspace begins on mailbox entry |
| 2. Surface → `Base17` typed wrapper | ndarray::hpc::cognitive (PR-X9) | Typed surface, not DTO |
| 3. `CascadeAddr::from_position` Hilbert-3D encode | PR-X10 A12 hilbert.rs | Deterministic, no shared state |
| 4. Cascade L1 XOR projection | PR-X4 splat4d cascade | Single XOR + table-addressing, no scan |
| 5. Cascade L2-L4 hops | PR-X4 splat4d cascade | Each hop = 1 XOR; total ≤ 4 hops |
| 6. Basin lookup at leaf address | PR-X9 LazyBlockedGrid | Lazy: codebook present → return; absent → materialize |
| 7. Basin materialization (cold path only) | PR-X12 codec (rANS decode) | Decode under the Rubicon write-back gate, not during cascade |
| 8. NARS revision over (existing truth, new evidence) | hpc::nars existing | Pure function: returns new `TruthValue`, no `&mut self` |
| 9. Rubicon commit | Ractor handler `&mut self` is the legitimate gated write | Single committed outcome per mailbox message |
| 10. Zone-1↔2 boundary crossing | sea-orm at zone 3 (only if egressing); SurrealDB at zone 2 | Typed surface in, ACID-tx out, materialization once |
| 11. Per-thought bindspace dies | Message lifetime | No global registry retained |

Eleven steps, one query path, four hops, sub-microsecond worst case (claimed).
The canary either reaches that envelope or the architecture is wrong.

## Measurement gates

The canary passes Phase 2 when **all** of the following hold on a Zen4 or
Sapphire Rapids 8-core box, AVX-512 enabled (`target-cpu=x86-64-v4`):

### Correctness gates (binary)

1. **Revision output matches scalar reference**:
   - `Fingerprint` (u64) bit-exact match against `src/hpc/nars.rs::revise`
   - `TruthValue` (f, c) within ULP ≤ 4 of scalar reference
   - 10,000 randomly-seeded revisions, zero divergences allowed
2. **Cascade routing is deterministic**:
   - Same `(Base17, position)` → same `CascadeAddr` across runs
   - Same `CascadeAddr` → same basin entry (warm cache or cold-materialized)
   - Bit-exact reproducibility across 100 runs
3. **No `&mut self` during compute** (compile-time enforcement):
   - `ndarray::hpc::cognitive::*` engines have `revise(&self, ...) -> Result`
   - Only Ractor handlers carry `&mut self` and only for commit, never compute
   - Clippy lint `clippy::needless_pass_by_ref_mut` clean
4. **Per-thought bindspace is per-thought**:
   - No `static`/`lazy_static`/`OnceLock` carrying mutable cognitive state
     inside zone 1 — audited by grep + sentinel-qa review
5. **Typed surfaces at zone boundaries**:
   - Zone 1 → zone 2: `ndarray::hpc::*` types, no `serde_json::Value`, no
     `HashMap<String, Box<dyn Any>>`, no DTO layer
   - Zone 2 → zone 3: `sea-orm` ActiveModel, materialization exactly once

### Performance gates (numeric)

1. **p99 revision latency** (warm cache, cascade depth ≤ 4):
   ≤ **1.5 µs** (target 700 ns mean per the HHTL claim; allow 2× headroom on p99)
2. **p99 revision latency** (cold cache, includes basin materialization):
   ≤ **15 µs** (codec decode + cascade + revision; rANS decode dominates)
3. **Cascade-only latency** (excluding revision math):
   ≤ **400 ns p99** (4 XOR hops + 4 table addressings)
4. **Codebook hit rate after 1M revisions warmup**:
   ≥ **95%** (sparse basins not pre-materialized; popular cells warm fast)
5. **Throughput, saturated**:
   ≥ **1M revisions/sec** per core sustained over 10 seconds (~1 µs amortized)
6. **Working set per worker thread**:
   ≤ **1 MB** (fits L2 cache on Zen4/SPR)
7. **ndarray::simd primitive coverage**:
   100% of hot-path SIMD ops route through `ndarray::simd::*` — zero raw
   intrinsics in the cognitive path (enforced by clippy lint and the W1a
   consumer contract gate)

### Inhabitance gates (qualitative)

1. **The canary path reads like the architecture document.** A new reader
   should be able to trace each of the 11 steps above to a specific function
   in the codebase. If the code is more complex than the architecture
   description, the architecture didn't get inhabited — a translation
   layer got built.
2. **No "Bardioc-shaped" code in the canary path.** No SQL builders for
   the lookup, no Elasticsearch-shaped query DSL, no JanusGraph-shaped
   traversal, no ClickHouse-shaped aggregation. The cascade is the lookup;
   the codebook is the storage; the Rubicon is the commit. If any
   step reaches for a legacy idiom, the canary has not inhabited.
3. **The canary survives a sentinel-qa audit** with zero P0 SAFETY findings
   on the new code (existing scalar reference is grandfathered).
4. **The integration sprint produces a 30-second screen recording** showing
   the canary running end-to-end, p99 latency on screen, codebook hit
   rate climbing during warmup. Recording is committed to the repo.

## What is NOT the canary

Explicit anti-scope so the canary doesn't drift into a system migration:

- **Not**: a full Bardioc → HHTL stack swap
- **Not**: a multi-workload benchmark suite
- **Not**: a SQL or graph-query analog of NARS revision
- **Not**: production cutover from Bardioc
- **Not**: a UI demo
- **Not**: a research artifact about HHTL theory — the canary is the
  *operational* proof, not a paper

If the canary works, Bardioc cutover is a follow-on per-workload migration
that can take months. The canary just has to demonstrate inhabitability of
*one* path.

## Where the canary lives

| Component | Crate / path | Sprint |
|---|---|---|
| `Base17` + `Fingerprint` + `TruthValue` types | `ndarray::hpc::{nars,fingerprint,base17}` (existing) | — (pre-existing) |
| `Hilbert3D::{encode,decode}` | `ndarray::hpc::linalg::hilbert` | PR-X10 A12 |
| `CascadeAddr` + `from_position` + `XorProjection` | `ndarray::hpc::splat4d::cascade` | PR-X4 |
| `SplatPyramid<T, S: GridStorage<T>, BR, BC>` | `ndarray::hpc::splat4d::pyramid` | PR-X4 + PR-X9 (GridStorage is PR-X9) |
| `BasinCodebook` + `LazyBlockedGrid` | `ndarray::hpc::cognitive::{codebook,storage}` | PR-X9 |
| rANS encode/decode + `CellMode` + `rdo_cell` | `ndarray::hpc::codec::*` | PR-X12 |
| Per-pillar PASS gates (revision math certified) | `ndarray::hpc::pillar::*` | PR-X11 |
| OGIT cognitive namespace bridge | `ndarray::hpc::ogit_bridge::*` | PR-X13 |
| Ractor Rubicon gate (`RevisionHandler`) | `lance-graph::cognitive::nars_actor` (new) | Integration sprint |
| SurrealDB egress (zone 2 typed surface) | `lance-graph::cognitive::nars_persist` (new) | Integration sprint |
| End-to-end canary binary | `lance-graph/examples/nars_canary.rs` (new) | Integration sprint |
| Measurement harness | `lance-graph/benches/nars_canary.rs` (new) | Integration sprint |

The integration sprint produces the two `lance-graph::cognitive::*` modules
that wire the substrate pieces together. The wiring is small (~200 LoC each);
the substrate pieces are the work.

## Composition with the 4-prompt strategic arc

| Strategic prompt | Role | Canary relationship |
|---|---|---|
| `bardioc-weekend-rebuild-prompt.md` | Baseline measurement (legacy) | Produces the **NARS-revision-on-Bardioc** number the canary beats |
| `ndarray-simd-trojan-horse-prompt.md` | Path A: ClickHouse + Tantivy FFI inject | **Independent** — analytic tier, not cognitive |
| `databend-ndarray-simd-prompt.md` | Path C: Rust-native ClickHouse successor | **Independent** — analytic tier, not cognitive |
| **THIS DOC + `hhtl-substrate-execution-prompt.md`** | Cognitive tier — the actual architectural win | Canary measures **revision-on-HHTL** vs the Bardioc baseline |

The four-prompt arc handles the **analytic tier** (where ClickHouse used to
live). This canary handles the **cognitive tier** (where HHTL lives). They
compose: the analytic tier is Bardioc's escape hatch; the cognitive tier is
the architecture's reason to exist.

Both must work for the consolidation to be real. The cognitive canary is
the harder and more important one.

## Pass/fail decision

If the canary passes all gates: HHTL is **inhabited**. Bardioc cognitive-tier
cutover is a per-workload migration; analytic-tier cutover follows path A
(buy time) or path C (replace). The consolidation arc is operationally
proved.

If the canary fails **performance gates** (latency/throughput): the
architecture's algorithmic regime claim ("two orders of magnitude") is
wrong. Re-examine the cascade depth, the codebook materialization cost,
or the SIMD primitive coverage. Patch and re-measure.

If the canary fails **correctness gates** (ULP/bit-exact): a substrate bug
exists. P0 — block all dependent sprint work until resolved.

If the canary fails **inhabitance gates** (qualitative): the substrate
exists but isn't being lived in — the integration sprint built a
translation layer instead of using the substrate primitives. Re-write
the wiring, not the substrate.

## Sequencing

The canary cannot be implemented until the 6 substrate sprints land (the
canary depends on PR-X4 + PR-X9 + PR-X10 A12 + PR-X11 + PR-X12 + PR-X13).
**The canary is the integration sprint deliverable**, not a parallel track.

The 6 sprints run per the master schedule (W1-W8 in
`pr-master-consolidation.md`). Integration sprint = W8 = canary build +
measure + record + write report.

## What changes if the canary passes

Three things become true that aren't true today:

1. **The architecture document stops being a claim and becomes a measurement.**
   The "700ns at depth 4" claim is now a number with confidence intervals.
2. **Per-workload Bardioc cutover becomes mechanically composable.** Each
   subsequent cognitive workload follows the canary pattern: typed surface
   in, cascade lookup, codebook materialization, Rubicon commit, zone
   boundary crossing. No new architectural decisions per workload.
3. **The four strategic prompts can be executed with confidence.** Today
   they read as "buy time + measure baseline + adopt successor." After
   the canary passes, they read as "execute the cutover" with the cognitive
   tier already proven.

If the canary doesn't pass, those three things stay false — and the next
session has to decide whether to debug the substrate or revisit the
architecture.
