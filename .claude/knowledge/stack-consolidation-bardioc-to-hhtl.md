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

| Zone | Layer | Role | Boundary contract |
|---|---|---|---|
| **Zone 1** (hot / in-process) | lance-graph + ndarray + Ractor | cognitive shader stack, Rubicon gates, HHTL cascade | typed surfaces, no serde, Rule #3 territory |
| **Zone 2** (warm / persistence) | SurrealDB (+ Tantivy FTS) | cognitive system's own state — committed outcomes only | typed surfaces in, ACID-tx out |
| **Zone 3** (cold / egress) | sea-orm | legacy SQL bridge (PostgreSQL, MySQL, host org's DB) | DTOs / SQL rows — materialization happens here |

**Ractor lives at zone boundaries**, never inside the zone-1 cascade. Actors are
the gates between deliberation and persistence (1↔2) and between persistence
and legacy egress (2↔3). Inside zone 1, the cascade is pure function composition
over typed surfaces.

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

## Click-moments inventory (the three architectural dissolutions)

These are the moments where a perceived problem turned out to not be a problem:

1. **SurrealDB ⊕ sea-orm overlap was source-of-truth ambiguity** → **Zone model
   shows they're stratified, not overlapping.** SurrealDB is zone-2 native
   persistence; sea-orm is zone-3 legacy egress. No overlap.

2. **Ractor `&mut self` violated Rule #3** → **Rubicon model shows actors are
   commitment gates, not shared-state mutators.** The handler body IS the
   Rubicon crossing; `&mut self` there is the gated write, not "during
   computation". Dual to Rule #3, not opposed.

3. **ClickHouse OLAP gap blocked the new stack** → **HHTL shows the cognitive
   workload doesn't need OLAP, just project-and-lookup.** ClickHouse stays in
   Bardioc and is decommissioned when the last scan-aggregate query is ported
   (which is never, because cognitive queries don't have that shape).

All three dissolutions are structural — they don't require new code, they
require seeing the existing architecture through the correct frame. That's why
they "click hard": the answer was already in the design; it just needed the
right name.

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

## References

- `pr-master-consolidation.md` — sprint plan, 10-submodule layout
- `pr-master-consolidation-savant-verdict.md` — READY-WITH-DOC-FIXES verdict
- `pr-x4-design.md` — splat cascade (HHTL projection)
- `pr-x9-design.md` — lazy basin codebook (HHTL leaves)
- `pr-x10-linalg-core-design.md` — linalg primitives (basis projections live here)
- `pr-x11-jc-consolidation-design.md` — numerical certification (cascade ops)
- `pr-x12-codec-x265-design.md` — compressed leaf storage
- `pr-x13-ogit-bridge-design.md` — OGIT TTL bundle (ontology grounding)
- `bardioc-weekend-rebuild-prompt.md` — migration baseline prompt
- `.claude/rules/data-flow.md` — Rule #3 source
- lance-graph PR #404 — four-repo demo (architectural target)
