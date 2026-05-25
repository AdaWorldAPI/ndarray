# Stack Consolidation: Bardioc → HHTL Substrate

Date: 2026-05-19
Status: Architectural reframe — load-bearing for PR-X4 + PR-X9 + four-repo demo
Companion docs:
- `pr-master-consolidation.md` (PR sprint plan)
- `pr-x4-design.md` (Gaussian splat cascade)
- `pr-x9-design.md` (lazy basin codebook)
- `pr-master-consolidation-savant-verdict.md` (joint plan-review verdict)

## What clicked (one paragraph)

Bardioc is **heterogeneous specialization**: 6 runtimes, 5 consistency models, 3 query
languages, glued together so each layer plays to its strength. The new stack is
**homogeneous consolidation**: one language (Rust 1.94 stable), one type system
(monomorphization across repo boundaries), one async runtime (Tokio), one
distribution primitive (TiKV ranges). The load-bearing reframe is that
**HHTL collapses the OLAP question entirely** — ClickHouse's scan-and-aggregate
regime isn't replaced, it's *made unnecessary* because cognitive queries are
project-and-lookup, not aggregate-scan. Two orders of magnitude latency drop
(700ns vs ms) at any cascade depth that fits in working memory, and
distribution is free because XOR-projection is deterministic.

## Old stack: Bardioc

| Layer | Runtime | Role | Consistency model |
|---|---|---|---|
| Cassandra | JVM | distributed wide-column KV | tunable (LWW, quorum, ALL) |
| JanusGraph | JVM | graph index over Cassandra | inherited from Cassandra |
| ClickHouse | C++ | columnar OLAP, vectorized scan | linearizable per shard |
| Elasticsearch + Lucene | JVM | full-text + inverted index | refresh-interval bounded staleness |
| OTP / BEAM / Erlang | BEAM VM | distributed actors, supervision | actor-local; cluster via mesh |
| (application) | mixed | typed surfaces (ad hoc per service) | service-by-service |

5 different consistency regimes welded together at runtime. Every cross-layer
join had to translate between consistency models — that's where cognitive
coherence was leaking.

## New stack: HHTL substrate

| Layer | Runtime | Role | Consistency model |
|---|---|---|---|
| TiKV | Rust | distributed KV, Raft, MVCC, range scans | linearizable + snapshot isolation |
| SurrealDB | Rust | zone-2 multi-model store (graph + doc + FTS) | per-tx ACID |
| Tantivy | Rust | full-text + inverted index (under SurrealDB FTS) | document-update visibility |
| sea-orm | Rust | zone-3 outbound legacy adapter | SQL transactional |
| Ractor | Rust | actors, Rubicon-model commitment gates | per-thought (no shared state) |
| ndarray::hpc::\* | Rust | typed cognitive substrate, SIMD, HHTL leaves | per-thought bindspace |
| lance-graph | Rust | HHTL orchestration, cascade routing | per-thought + write-back-on-commit |

**One ownership model end-to-end.** Typed surfaces cross all layer boundaries
inside zones 1+2. Materialization happens exactly once, at the zone-2↔3 sea-orm
edge, on purpose.

## Translation by job

| Bardioc job | Bardioc layer | HHTL stack | Notes |
|---|---|---|---|
| distributed KV | Cassandra | TiKV | Raft consensus, MVCC, identical primitives |
| graph index | JanusGraph | lance-graph + ndarray typed surfaces over TiKV | no separate index process; graph semantics in-language |
| OLAP scan | ClickHouse | **HHTL (PR-X4 + PR-X9)** | regime change: project-and-lookup, not aggregate-scan |
| full-text search | Elasticsearch / Lucene | Tantivy under SurrealDB FTS | Rust port of Lucene; mature enough |
| distributed actors | OTP / BEAM | Ractor + Rubicon model | conceptually tighter; operationally younger |
| supervision trees | OTP | Ractor supervisors | same shape, Rust ergonomics |
| legacy SQL egress | (custom adapters per service) | sea-orm at zone 3 | materialization happens here, on purpose |
| application typed surfaces | (ad hoc per service) | ndarray::hpc::\* | monomorphized across repo boundaries |

## The HHTL reframe (why ClickHouse doesn't move)

ClickHouse stays in Bardioc. It is not ported. **HHTL makes the ClickHouse-shaped
question disappear.**

Mechanism:
- 16,384-column row = 2¹⁴ orthogonal features, cache-aligned 2KB
- 90°-rotated vector = Walsh-Hadamard / Reed-Muller basis projection
- One XOR + table-addressing = 20-170ns fixed-address lookup (no scan, no
  comparison loop)
- Cascade composition: each hop reduces search space by 16,384×
- 3 hops = 2⁴² ≈ 4.4 trillion addressable cells
- 4 hops = 2⁵⁶ ≈ 72 quadrillion addressable cells
- End-to-end at depth 4: ~700ns worst case

ClickHouse's fastest point-lookup is milliseconds (granule scan).
HHTL is sub-microsecond at any cascade depth that fits in working memory.
**Two orders of magnitude isn't competitive; it's a different algorithmic regime.**

The cognitive workload only ever issues project-and-lookup queries
("given this perceptual surface, what basin matches?"). It never issues
aggregate-scan queries. So the ClickHouse strength is irrelevant, not absent.

## Zone model

| Zone | Column-state phase | Surface that watches | What "being in this zone" means |
|---|---|---|---|
| **Zone 1** (hot) | `committed = false`, currently held in mailbox-cycle scope | lance-graph cascade ops | the row is being deliberated; cascade compute is in-flight against the same bytes a future Zone-2 reader will see |
| **Zone 2** (warm) | `committed = true`, Lance-versioned | SurrealDB LIVE subscriptions, lance-graph reads | the row's truth-value crossed the Rubicon; any LIVE watcher with a matching predicate observes the flip as a column-state transition |
| **Zone 3** (cold) | `egressed_at IS NOT NULL`, mirrored once | sea-orm to legacy RDBMS | the row has been materialised into PG-shape for the legacy surface; the source Lance bytes are unchanged |

**Zones are temporal phases of column state on a single Lance dataset, not
storage tiers.** Same physical bytes throughout. A row does not "move" from
zone 1 to zone 2; a column flips from `committed = false` to `true`, and
the LIVE watchers notice. There is no serialise / marshal / wire-format
step between strata because there are no strata — there is one Lance
dataset, multiple state-flag columns, and multiple dialect surfaces reading
the same buffers.

This is the right framing for the Rubicon model: the crossing is a *column
flip*, not a write event. There is no "mailbox in RAM commits to
SurrealDB" — SurrealDB always saw the row, the row just changed state. The
mailbox-cycle still governs the commit (the handler decides when to flip
the flag, and `&mut self` there is the gated write), but the flip itself
is a state transition on bytes that didn't move.

What stays true from earlier framings:
- The cascade inside a single handler body is pure function composition over
  typed surfaces (Rule #3 territory)
- The `&mut self` in the handler IS the gated write — legitimate because it
  IS the Rubicon crossing (the column flip), not "during computation"
- Typed surfaces at the dialect interfaces (SurrealQL parses to column
  predicates; sea-orm projects to legacy DTOs; Databend pushes filters to
  column kernels) — but these are *type-level* contracts on how each
  dialect reads the same bytes, not perimeters around different stores

See § "Column-substrate identity" below for the full unification.

## Column-substrate identity — Lance ≡ Arrow ≡ ndarray SoA

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

**One physical representation, end to end.** The Lance column layout, the
Arrow column buffer layout, and the ndarray SoA layout are the same bytes
viewed through three names. The four dialect surfaces (lance-graph cascade,
SurrealDB, sea-orm, Databend, Tantivy) all parse their respective query
languages down to operations on those same bytes.

**ndarray amortises the SIMD primitive across the whole stack.** The same
kernel that runs the cognitive cascade, that Databend's filter pushdown
invokes, that Tantivy's indexer reads, that sea-orm projects to legacy
egress — they are the same kernel on the same bytes. ndarray pays for the
SIMD primitive once and the entire stack collects rent. No transcode tier,
no copy boundary, no format conversion at any zone.

**Rubicon = column-state flip, not write event.** A Thought is a Lance row
from the moment it is allocated to the moment it is queried by any surface.
"Crossing the Rubicon" means flipping (e.g.) `committed: false → true` —
versioned natively by Lance, observed by any LIVE watcher with a matching
predicate, no serialisation involved.

### What this dissolves

| Earlier framing (wrong) | Why it's wrong |
|---|---|
| "Mailbox writes to SurrealDB on Rubicon crossing" | There is no write — SurrealDB always saw the row; the row just changed state |
| "MvccProvider::snapshot_ts threads across engines" | There is one Lance dataset with one version chain; all readers see the same version |
| "surrealdb-ractor as cf-event router" | No cf-event-as-message needed; mailboxes already share the same column slice that SurrealDB watches |
| "sea-orm-ractor entity-actor dispatch by PK" | The mailbox IS the row; no separate dispatch layer |
| "Zone 1 in-process vs Zone 2 durable" (as storage tiers) | Same physical bytes; zones are temporal phases of column state, not storage tiers |
| "TiKV as routing / coordination layer" | TiKV ranges are Lance dataset shards under the XOR cascade — substrate, not routing |
| "kv-lance translates records into Lance rows for SurrealDB" | No translation; SurrealQL parses directly against Lance columns that lance-graph already owns |

### What survives — JITson / Cranelift, cleaner than before

The compile-time → JIT pipeline does not collapse with the framing — it
sharpens:

- **ndarray SoA layout = Lance column layout = known at OGIT-schema-compile time.**
  The schema fixes the column shape; everything downstream specialises against it.
- **`DeriveEntityModel` (or equivalent) emits column-typed accessors at Rust
  compile time** — typed handles into the same bytes for each dialect surface.
- **Cranelift JITs hot-path kernels specialised for the OGIT-derived column
  types at first call** — predicate compilation, projection compilation,
  cascade-step compilation, all against the typed column shape.
- **"Sinkin becomes compile next time"** — when a new column shape enters the
  substrate (ontology evolution), the next compile cycle regenerates the typed
  accessors and the JIT re-specialises against the new shape.
- **All four dialect surfaces automatically inherit the new kernels** because
  they all operate on the same column layout. Add a column → all surfaces
  see it. Specialise a kernel → all surfaces use it.

### Implication for the four-tier picture

The four-tier picture earlier in this doc names `ndarray::simd` as "the
common SIMD substrate across all four tiers". That claim is correct, but
its load-bearing reason is the column-substrate identity, not "we happen
to use the same SIMD library in four places". The deeper fact:

> **The column IS the SoA IS the ndarray buffer.** The cognitive cascade,
> the analytic scan, the FTS index build, and the graph traversal all
> operate on the same bytes through the same SIMD kernels. ndarray::simd
> is the common substrate because the substrate is genuinely one thing,
> not four parallel things wearing the same uniform.

This is the actually-clean Foundry-aspiring shape: one physical store, one
column layout, one kernel set, multiple dialect surfaces. The "same data,
different syntax" claim is finally literal — not "same schema across
translation layers" but **same bytes, period.**

## Rule #3 ⊕ Rubicon ⊕ Per-thought bindspace (three-legged stool)

The new stack works because three principles compose:

1. **Rule #3 (`.claude/rules/data-flow.md`):** no `&mut self` during computation.
   Engines return results; mutations are gated write-back only.
2. **Rubicon model (Ractor):** actors are cognitive commitment gates.
   Mailbox = deliberation buffer; handler = Rubicon crossing (single committed
   outcome); reply = orchestrated propagation. The `&mut self` in the handler
   is legitimate because it IS the gated write, not "during computation".
3. **Per-thought bindspace:** no static / global bindspace. Each actor's
   bindspace is ephemeral, owned, scoped to one Rubicon crossing. Lifetime =
   message lifetime. Drop the message, the bindspace dies.

Together: no shared mutable state at any layer, ever. Lock contention vanishes
structurally. GC is trivial (Rust ownership does the work). Source-of-truth is
obvious (the actor IS the truth for its thought). Distribution is free (XOR
projections are local).

## PR-X4 + PR-X9 ARE the HHTL implementation

Re-reading the splat cascade + lazy basin codebook designs through HHTL:

- **PR-X4 splat cascade** = HHTL projection mechanism. Each cascade level L is
  one XOR-projection hop. `CascadeAddr_{L+1} = XOR(parent_addr, rotation_L(query))`.
  Gaussian splat is the basis-vector kernel of the orthogonal projection at that
  level. The 64×256×4096×16384 hierarchy is the four cascade levels = 2⁴⁰ leaf
  positions per root.
- **PR-X9 lazy basin codebook** = HHTL leaf layer. Basins aren't pre-materialized
  across all 2⁵⁶ cells (1 exabyte). The cascade routes O(1) to the right leaf
  address; if present return it, if absent the generative function constructs it
  on demand under the Rubicon write-back gate. Dense index, sparse leaves.
- **lance-graph** = HHTL orchestration. Which queries to project, which basins
  to materialize, when to evict cold leaves, how to shard the address space
  across TiKV ranges.
- **TiKV ranges** = HHTL address-space shards. Shard owner for any query is
  computable from the query itself (XOR is deterministic). No coordinator
  lookup, no consistent-hash-ring rebalancing pain.

**PR-X4 and PR-X9 stopped being "two designs we'll get to" and became the
actual product.** Everything else in the master consolidation (PR-X10 linalg,
PR-X11 jc pillars, PR-X12 codec, PR-X13 OGIT bridge) is *infrastructure for*
HHTL.

## Integration plan (PR ordering, updated)

The master-consolidation sprint order stands, but **the destination is HHTL**:

| Week | Sprint | What ships | Why HHTL needs it |
|---|---|---|---|
| W1-W2 | PR-X10 linalg-core | MatN, Spd3, Hilbert-3D, eig_sym, FFT | basis projections, 3D address space curves |
| W3 | PR-X11 jc pillars | 6 pillars + 2 placeholder σ | numerical certification of cascade ops |
| W3 | PR-X13 OGIT bridge | embedded TTL bundle, ndarray-native codebook | ontology grounding for basin labels |
| W4-W5 | PR-X12 codec | x265-style CTU quad-tree, rANS | compressed leaf storage in zone 2 |
| W4-W5 | **PR-X4 splat cascade** | **HHTL projection mechanism** | **THE PRODUCT** |
| W6-W7 | **PR-X9 lazy basin codebook** | **HHTL leaf layer** | **THE PRODUCT** |
| W8 | integration | four-repo demo end-to-end | proves the stack |

The four-repo demo (lance-graph + ndarray + ractor + tikv) is no longer
"SumShader" or any toy aggregate. It is **a NARS revision projected through the
HHTL cascade, committing a basin assignment via Rubicon gate, persisted to
SurrealDB through Ractor, with sea-orm egress to a legacy Postgres row**. Five
zones touched, zero Rule #3 violations, sub-microsecond zone-1 latency,
sub-millisecond zone-2 commit, observable end-to-end.

## Migration plan (Bardioc weekend rebuild → cutover)

The Bardioc weekend rebuild (see `bardioc-weekend-rebuild-prompt.md`) provides:

1. **Migration baseline** — the same cognitive workload running on both stacks,
   so HHTL's claimed advantages are measurable, not theoretical.
2. **Nostalgia / honesty** — a forcing function to remember that the heterogeneous
   stack worked, just at higher operational cost. Avoids straw-manning Bardioc.
3. **Workload-by-workload port** — each cognitive workload migrates from
   Bardioc to HHTL one at a time, with both stacks running, so cutover is
   risk-bounded.

Cutover sequence (after Bardioc baseline + HHTL substrate both green):
1. **Read-only mirror**: HHTL reads from TiKV; Bardioc still writes to Cassandra;
   periodic ETL keeps TiKV current. Validate HHTL latency claims on real
   workload.
2. **Dual write**: writes go to both Bardioc and HHTL through a fork in the
   ingestion path. Validate consistency.
3. **HHTL-primary**: reads switch to HHTL; Bardioc demoted to disaster-recovery
   mirror. Validate operational stability.
4. **Bardioc decommission**: shut down Cassandra, JanusGraph, ClickHouse, ES,
   BEAM cluster. Recover infra cost.

Estimated duration per workload: 2-4 weeks. Estimated total cutover: 3-6 months
depending on workload count.

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| ClickHouse-shaped query slips through | Med | Audit every read path; reject scan-aggregate queries at the API layer; force callers to project |
| BEAM operational maturity gap (Ractor) | Med-High | Run Ractor with explicit supervisor trees from day one; replicate the OTP supervision patterns; chaos-test |
| Tantivy FTS depth vs Lucene | Low-Med | Tantivy is mature; benchmark against ES on real query mix; fall back to embedding ES if a specific feature is missing |
| TiKV operational footprint (Raft, PD) | Med | Operate at small cluster first (3-5 nodes); use TiUP for cluster lifecycle; treat PD as the critical path |
| sea-orm vs custom-written SQL | Low | sea-orm is mature; for hot legacy paths, drop to sqlx if codegen is too verbose |
| HHTL distribution math is wrong | High | This is the load-bearing claim; numerical certification (PR-X11 pillars) covers cascade ops; add formal proofs for the XOR-projection bijectivity property before zone-2 commit |
| 90° vector / Walsh-Hadamard basis breaks for non-projectable queries | High | API enforces "queries must be expressible in basis"; queries that aren't are bounced back to the caller with a typed error, not silently scanned |

## Click-moments inventory (the four architectural dissolutions)

These are the moments where a perceived problem turned out to not be a problem:

1. **SurrealDB ⊕ sea-orm overlap was source-of-truth ambiguity** → **Zone model
   shows they're stratified, not overlapping.** SurrealDB is zone-2 native
   persistence; sea-orm is zone-3 legacy egress. No overlap.

2. **Ractor `&mut self` violated Rule #3** → **Rubicon model shows actors are
   commitment gates, not shared-state mutators.** The handler body IS the
   Rubicon crossing; `&mut self` there is the gated write, not "during
   computation". Dual to Rule #3, not opposed. **Refinement** (2026-05-19,
   post-PR #404 rollback): the mailbox carries the commitment responsibility
   implicitly, so there is no physical boundary between zones 1/2/3 for
   actors to "live at" — Rubicon is per-mailbox-commit-cycle, distributed
   everywhere there is a handler.

3. **ClickHouse OLAP gap blocked the new stack** → **HHTL shows the cognitive
   workload doesn't need OLAP, just project-and-lookup.** ClickHouse stays in
   Bardioc and is decommissioned when the last scan-aggregate query is ported
   (which is never, because cognitive queries don't have that shape).

4. **Multi-store consistency / cross-zone messaging looked like the hard
   coordination problem** → **Column-substrate identity shows there is no
   cross-zone messaging.** Lance column ≡ Arrow buffer ≡ ndarray SoA, same
   bytes for every dialect surface (lance-graph, SurrealDB, sea-orm, Databend,
   Tantivy). Rubicon is a column-state flip, not a write event. SurrealDB
   LIVE subscriptions watch column predicates on bytes they were already
   reading. The hard problem dissolves because there were never multiple
   stores to keep consistent. See § "Column-substrate identity" above.

All four dissolutions are structural — they don't require new code, they
require seeing the existing architecture through the correct frame. That's why
they "click hard": the answer was already in the design; it just needed the
right name. Dissolutions 1–3 are workload-shape dissolutions; #4 is the
substrate-identity dissolution and is the deepest of the four — it makes
the other three's "no copy, no marshal, no coordination" claims literal.

## What's NOT covered by this consolidation

Honesty roster — things that genuinely don't fit and need separate stories:

- **Time-series telemetry / metrics**: ClickHouse-ish workload that the
  cognitive stack DOESN'T have but operational monitoring DOES. Solution:
  Prometheus + Grafana for ops; not a Bardioc replacement, separate concern.
- **Cold archival** (>1 year, rarely accessed cognitive state): TiKV with
  cheap storage tier OR object storage (S3-compatible) with on-demand recall.
  Not yet designed; tracking issue.
- **Cross-DC replication / geo-distribution**: TiKV supports it; not designed
  for the cognitive workload yet. Tracking issue.
- **Schema evolution at zone 2**: SurrealDB schemaless mode handles it; but
  the typed-surface contract from zone 1 means schema changes ripple through
  Rust types. Migration discipline TBD.

## Four-tier picture: HHTL only has to win the cognitive layer

The synthesis that compresses the whole consolidation arc to one diagram.
Bardioc had four CPU-heavy specialty layers. **Three of them already have
Rust-native successors that aren't HHTL.** HHTL only has to win at the
cognitive layer it was designed for.

| Tier | Workload shape | Bardioc layer | Rust-native successor | Acceleration |
|---|---|---|---|---|
| **Cognitive** (hot path, sub-µs) | project-and-lookup, cascade routing | (no Bardioc analog — application code) | **HHTL** = TiKV + SurrealDB + Ractor + ndarray + lance-graph | ndarray::simd (native), HHTL projection (2 orders of magnitude vs scan) |
| **Analytic** (cold path, ms) | scan-and-aggregate, OLAP, ad-hoc SQL | ClickHouse | **Databend** (Arrow + DataFusion + Tokio, MIT, ClickHouse-shape) | ndarray::simd injection (filter / aggregate / hash kernels) |
| **Search** (full-text) | inverted index, BM25, ngram, faceting | Elasticsearch / Lucene | **Tantivy** (under SurrealDB FTS, also via Quickwit) | ndarray::simd injection (bitpack decode / BM25 / skip-list intersection) |
| **Graph** (traversal) | BFS/DFS, edge-label filter, frontier expansion | JanusGraph | **lance-graph native** (typed surfaces over TiKV ranges) | ndarray::simd (frontier bitsets), no JNI |

This is the load-bearing reframe for the migration argument:

- **HHTL is genuinely new IP** — nothing exists like it; the cognitive layer
  is where the architecture earns its keep.
- **The other three are inheritances** — Databend / Tantivy / lance-graph are
  pre-existing Rust-native engines that already do what their Bardioc
  counterparts did, just in Rust with Tokio and Arrow.
- **ndarray::simd is the common SIMD substrate across all four tiers** —
  injection target for Databend + Tantivy (the trojan-horse prompts);
  native for HHTL + lance-graph (the hot-path cognitive substrate).

Migration scope shrinks proportionally. The total work is:
1. Build HHTL (PR-X4 + PR-X9, the genuinely new piece).
2. Adopt Databend, Tantivy, lance-graph (existing, just integrate).
3. Inject ndarray::simd into Databend + Tantivy (trojan horse prompts,
   1–2 engineer-weeks per target).
4. Cutover from Bardioc one workload at a time (read-only mirror → dual
   write → primary-flip → decommission, the existing migration plan).

**No transcode of ClickHouse, ES, or JanusGraph is required, ever.**

## Why we don't transcode ClickHouse (cheap escape hatches)

A full ClickHouse transcode is one of the hardest software undertakings in
modern infrastructure: ~1.2M LOC C++ core, ~150 vendored libraries, ~1000
hand-tuned aggregation/scalar functions, decades of SIMD/cache/JIT
optimization. Realistic cost: **5–10 engineer-years**. Reference points:
TiKV's Rust rewrite took ~5 years with the original team; Servo's
C++→Rust port took ~10 years and ended partial; the Postgres→CockroachDB
conceptual port is still incomplete after a decade.

Three cheaper escape hatches, in order of cost:

| Approach | Cost | Outcome |
|---|---|---|
| **A. FFI inject ndarray::simd into ClickHouse** (trojan horse prompt) | 1–2 engineer-weeks | ClickHouse stays C++, hot kernels are Rust; legacy stack faster, Bardioc cutover urgency reduced |
| **B. Transcode only the vectorized executor** (~50–100k LOC) | 1–2 engineer-years | Hybrid C++ shell + Rust executor core; deep IP investment, narrow scope |
| **C. Adopt Databend + ndarray::simd injection** (databend prompt) | 0 transcode | Rust-native, ClickHouse-shape, MIT licensed, already maintained, rides upstream |

**Recommended: C.** Databend already covers ClickHouse-shape workloads in
Rust on Arrow + DataFusion + Tokio. ndarray::simd injection earns the
"hand-tuned" performance parity. Combined cost is engineer-weeks, not
engineer-years, with zero transcode debt.

A is also valuable in parallel — it accelerates Bardioc during the cutover
window and creates upstream contribution opportunities. B is rarely worth
it; only justified if you need ClickHouse-storage-format wire-compatibility
in a Rust-native engine, which the cognitive stack does not.

The C# ecosystem analog (asked separately): RavenDB is the closest
single-binary-vendor-everything analog to ClickHouse in .NET, with
EventStoreDB second. Neither is performance-competitive with ClickHouse on
OLAP scan, but they share the operational philosophy. Notable because the
ClickHouse design pattern (full vendoring + native compilation +
obsessive SIMD/cache tuning + willingness to patch upstream) is rare —
ClickHouse may be the only OSS database that does all four. Yandex
heritage is what made it possible.

## Salvage from the 2026-05-19 cross-repo rollback (PR #404 / PR #160)

The four-repo demo PR #404 in lance-graph (and its companion ndarray PR #160)
was reverted via PR #405 on 2026-05-19 — the architectural intent is
preserved as a next-cycle target, the code attempt was withdrawn. Two pieces
of that work are NOT dead and have their re-entry points named here so the
next-cycle implementation doesn't lose them:

### 1. `heel_f64x8::{l1, l2, linf}_f64_simd` → PR-X10 A6 `linalg::distance`

The distance kernels themselves are correct; the framing was wrong (filed as
"Sprint 0a of a four-repo integration arc" with cross-repo coupling that
made the rollback unavoidable). The same code re-emerges as
`ndarray::hpc::linalg::distance::{l1, l2, linf}_f64_simd` under worker A6
in PR-X10 — the polar.rs / matfn.rs neighbourhood. Bench parity vs the
PR #160 implementation is part of A6's acceptance gate. See
`pr-x10-linalg-core-design.md` § "Distance kernels — `linalg::distance`".

### 2. `lance-graph-contract::{ir, provider, actor}` → mostly redundant, except…

The IR / provider types (`Operator`, `Cardinality`, `EngineHint`,
`MvccProvider`) duplicate work the HHTL arc covers natively — they don't
re-emerge. They're correctly dead.

**Exception: `SupervisableShader` + `RestartBackoff`** have a future as
*mailbox-cycle commitment-gate primitives* on Ractor actors. **Important
framing refinement** (2026-05-19, post-rollback session): with the Rubicon
model, the mailbox itself carries the commitment responsibility, so the
gate fires *per-message-commit-cycle*, not at a physical zone-1↔zone-2
boundary. The earlier framing ("they only fire at zone 1↔2 transitions")
was a category error — zones 1/2/3 are *logical* stratification of where
state physically lives, not perimeter walls actors cross. There is no
physical boundary because the mailbox IS the Rubicon.

What this means concretely for the next-cycle implementation, **under the
column-substrate-identity framing** (see § "Column-substrate identity"
above):

- `SupervisableShader` is the supervisor-aware wrapper around a Ractor
  handler that owns a *column-flip cycle* (read column → compute → flip
  state-flag → reply / drop). Its "supervision boundary" is the
  flip-cycle, not a perimeter between stores — because there is no second
  store. SurrealDB / sea-orm / Databend / Tantivy are dialect surfaces on
  the same Lance column the handler is operating on.
- `RestartBackoff` governs how the supervisor responds when a flip-cycle
  panics or returns an error before the flag is set. It gates *retry
  attempts on the same column flip*, not retries across physical
  infrastructure. The Lance version chain provides the natural retry
  semantics (the flip either landed-and-committed in the version chain or
  it didn't; SurrealDB LIVE watchers only see committed flips).
- Both primitives are stateless types that live in `lance-graph` (the
  thinking layer); they don't belong in ndarray (the hardware layer).
- Re-entry point: a future PR-X14 or sibling sprint in the lance-graph
  repo that introduces `LanceActor`-shaped wrappers for the canary's
  commit path (see `hhtl-canary-inhabitance-plan.md` step 9 — the natural
  first consumer is the NARS-revision handler that flips a `revised: false →
  true` column on the belief row).

The "no physical boundary 1/2/3 — and no second store either" insight is
captured as the **fourth click-moment** in the Click-moments inventory
above. Click-moments 1–3 were workload-shape dissolutions; #4 is the
substrate-identity dissolution and is the deepest of the four. The
SupervisableShader + RestartBackoff primitives can be small (~50 LoC each)
because they encode column-flip-cycle semantics, not cross-store
plumbing.

### Lesson for future cross-repo arcs

PR #404's failure mode was not bad code — it was a four-repo coupling
filed as a single arc, which made the rollback inherently cross-repo and
the merge-window inherently fragile. The architecturally-equivalent work
re-enters as multiple single-repo PRs across the Phase 2 schedule
(PR-X10 absorbs the distance kernels; a future PR-X14 absorbs the
column-flip-cycle primitives). The next-cycle architectural target — the
four-repo integration demo — happens *after* the canary lands in W8,
not before. Integration depends on substrate, not vice versa. And under
the column-substrate-identity framing, "integration" mostly means "wire
the dialect surfaces to read the columns the canary writes" — there is
no marshal layer to build.

## References

- `pr-master-consolidation.md` — sprint plan, 10-submodule layout
- `pr-master-consolidation-savant-verdict.md` — READY-WITH-DOC-FIXES verdict
- `pr-x4-design.md` — splat cascade (HHTL projection)
- `pr-x9-design.md` — lazy basin codebook (HHTL leaves)
- `pr-x10-linalg-core-design.md` — linalg primitives (basis projections live here)
- `pr-x11-jc-consolidation-design.md` — numerical certification (cascade ops)
- `pr-x12-codec-x265-design.md` — compressed leaf storage
- `pr-x13-ogit-bridge-design.md` — OGIT TTL bundle (ontology grounding)
- `bardioc-weekend-rebuild-prompt.md` — migration baseline prompt (build the old stack honest)
- `ndarray-simd-trojan-horse-prompt.md` — inject ndarray::simd into ClickHouse + Tantivy (path A)
- `databend-ndarray-simd-prompt.md` — adopt Databend + ndarray::simd as ClickHouse successor (path C, recommended)
- `hhtl-canary-inhabitance-plan.md` — Phase 2 entry condition: names the NARS-revision canary + correctness/performance/inhabitance gates
- `hhtl-substrate-execution-prompt.md` — Phase 2 Protocol A execution prompt (8 weeks, 6 sprints, 44 workers; per-sprint kickoff blocks for W1-W8)
- `.claude/rules/data-flow.md` — Rule #3 source
- lance-graph PR #404 — four-repo demo (architectural target; merge reverted via PR #405 in 2026-05-19 cross-repo rollback — intent preserved as next-cycle target, code attempt withdrawn)
