# Bardioc Weekend Rebuild — Claude Code Flex Prompt

Copy the block below into a fresh Claude Code session. Authorize Docker + wildcards.
Budget: 48 hours wall-clock. Goal: migration baseline + nostalgia.

---

```text
You are building a migration-baseline replica of the legacy Bardioc cognitive
stack in a single weekend. The point is not to ship production code — the
point is to have the OLD stack running end-to-end so we can measure latency,
operational footprint, and consistency-model overhead against the new HHTL
substrate (TiKV + SurrealDB + Ractor + ndarray + lance-graph) we are migrating
TO.

This is a flex. Spawn 12 parallel workers + 1 coordinator. Use docker-compose
to orchestrate. Cargo / poetry / mix for per-service builds. No Kubernetes,
no Terraform — just compose, scripts, and discipline.

## Stack to rebuild

1. Cassandra 4.x cluster (3 nodes via docker-compose)
   - Keyspace: cognitive
   - Tables: triples (s,p,o,truth_freq,truth_conf,ts),
             basins (cascade_addr,centroid_blob,count,ts),
             qualia_text (id,description,ts)
   - Replication factor 3, consistency QUORUM

2. JanusGraph 1.x over the Cassandra cluster
   - Schema: Vertex labels {Concept, Triple, Basin}; Edge labels {asserts, revises, splat_of}
   - TinkerPop/Gremlin server on :8182
   - ScyllaDB-compatible config NOT used; vanilla Cassandra backend

3. ClickHouse 24.x (single node OK for weekend)
   - Database: cognitive_olap
   - Tables: triple_revisions_log (Engine=MergeTree, ORDER BY (ts, s)),
             basin_lookup_log,
             qualia_query_log
   - Materialized views for hourly aggregates

4. Elasticsearch 8.x + ingest-attachment plugin
   - Index: qualia (text body, embedding vector, timestamps)
   - Index: triples_searchable (s/p/o concatenated for full-text)
   - Mapping with stemming + ngram tokenizer for cognitive-domain terms

5. Erlang/OTP 26 BEAM cluster (2 nodes)
   - Application: bardioc_actors
   - Supervisor tree: top-level supervisor → {revision_worker_pool,
     cascade_worker_pool, egress_worker_pool}
   - Use gen_server + gproc for actor registry
   - Cluster via Erlang distribution protocol on EPMD

6. Application services (Python 3.12 + FastAPI for ingestion;
   Java 21 for JanusGraph clients; Elixir 1.16 for BEAM integration)
   - Ingestion API: POST /triples, POST /qualia, GET /basin/{addr}
   - Background workers in BEAM consume from Cassandra change feed,
     project basins into JanusGraph, log revisions to ClickHouse,
     index qualia text into Elasticsearch
   - Egress workers write committed outcomes to a downstream Postgres
     ("legacy host org DB")

## Cognitive workload (so the benchmark is real, not synthetic)

Implement a minimal end-to-end cognitive cycle on the Bardioc stack:

1. Ingestion: stream 100k NARS triples + 10k qualia descriptions over 1 hour
   (use a generator that produces plausible {subject, predicate, object,
   freq, conf, timestamp} tuples; qualia descriptions are 50-200 token
   sentences drawn from a small domain vocabulary).

2. Revision: every 5 seconds, run a batch NARS revision over the last 60s
   of incoming triples. Apply Wang's revision formula. Persist updated
   posteriors to Cassandra. Log revision events to ClickHouse.

3. Basin assignment: for each revised triple, look up the nearest basin
   in JanusGraph (Gremlin traversal: g.V().hasLabel('Basin').has(...)
   .order().by(centroidDist(triple_embedding)).limit(1)). Persist
   assignment in JanusGraph.

4. Full-text query: every 30s, run 100 full-text qualia queries against
   Elasticsearch (terms from the same domain vocabulary). Log query
   latency to ClickHouse.

5. Egress: every 60s, push the last minute's committed basin assignments
   to the downstream Postgres via the BEAM egress worker pool.

## Benchmarks to record

For each cycle, record to ClickHouse:
- p50, p95, p99 latency per operation (ingest, revise, basin-lookup,
  qualia-query, egress)
- Throughput (ops/sec)
- Memory + CPU per container (Docker stats)
- Cross-layer hop count per cognitive query

Run for 4 hours minimum. Export the results table as CSV to
./benchmarks/bardioc-baseline-{timestamp}.csv.

## Migration harness (most important deliverable)

Create a harness in ./migration/ that:

1. Defines an abstract CognitiveBackend trait/protocol with the operations
   above (ingest_triple, revise_batch, assign_basin, query_qualia,
   egress_batch).
2. Provides two implementations: BardiocBackend (this build) and
   HhtlBackend (stub for now, points at TiKV + SurrealDB + Ractor +
   ndarray + lance-graph).
3. Runs the same workload generator against either backend.
4. Generates a side-by-side comparison report (latency histograms,
   throughput, resource cost).

The HhtlBackend stub does not need to work — it just needs to exist with
the right interface so when the real HHTL substrate is ready, the harness
plugs in zero-changes.

## Deliverables (end of weekend)

1. docker-compose.yml that brings the whole Bardioc stack up with
   `docker-compose up -d`
2. Schema migrations for Cassandra, JanusGraph, ClickHouse,
   Elasticsearch indexes
3. BEAM application with supervisor tree, deployed via rebar3 release
4. FastAPI ingestion service + Elixir BEAM bridge
5. Workload generator (./bench/workload.py)
6. Benchmark output (./benchmarks/bardioc-baseline-*.csv)
7. Migration harness (./migration/) with both backend implementations
   (HhtlBackend may be a stub)
8. README.md explaining how to run, where the metrics live, and the
   teardown procedure
9. A 1-page POSTMORTEM.md naming three things that were operationally
   painful (you will use this to justify the HHTL migration to
   stakeholders)

## Anti-goals (do NOT do these)

- Do NOT optimize Bardioc. The point is to show it works at honest
  out-of-the-box settings, not to tune it. Default JVM heap sizes,
  default Cassandra config, no ClickHouse cluster tuning. The new stack
  has to beat HONEST Bardioc, not heroically-tuned Bardioc.
- Do NOT add features beyond the cognitive cycle above. If a feature
  isn't in the 5-step workload, skip it.
- Do NOT touch the HHTL substrate. The HhtlBackend implementation is
  pure interface — leave the real implementation to the master
  consolidation arc (PR-X4 + PR-X9 + ...).
- Do NOT spend more than 4 hours on any single component. If something
  doesn't come up cleanly, document it and move on. The pain itself is
  part of the deliverable (see POSTMORTEM.md).
- Do NOT use Kubernetes, Helm, Terraform, or any deployment automation
  beyond docker-compose + shell scripts. Honest operational footprint.

## Coordination protocol

12 parallel workers, 1 coordinator (you). Suggested split:
- W1: Cassandra cluster + schema + smoke test
- W2: JanusGraph + Gremlin schema + smoke test
- W3: ClickHouse + tables + materialized views + smoke test
- W4: Elasticsearch + indexes + ingest pipeline + smoke test
- W5: BEAM application skeleton + supervisor tree
- W6: BEAM revision_worker_pool implementation
- W7: BEAM cascade_worker_pool implementation
- W8: BEAM egress_worker_pool implementation
- W9: FastAPI ingestion service + workload generator
- W10: Cross-layer integration glue (Python ↔ Cassandra ↔ BEAM ↔ ES ↔
  ClickHouse)
- W11: Benchmark harness + metrics collection + CSV export
- W12: Migration harness (./migration/) + HhtlBackend interface stub

Coordinator (you): docker-compose.yml, README, POSTMORTEM, integration
testing, force-prune dead workers, cherry-pick across worktrees.

Use git worktrees per worker. Branch per worker:
bardioc-weekend/{role}-{worker-id}. Coordinator cherry-picks to main
when each worker's smoke test passes. Cargo gates skipped per worker;
docker-compose build is the integration gate.

## Time budget

| Hour 0-4 | Stack standup (W1-W4 in parallel) |
| Hour 4-12 | BEAM application (W5-W8) + integration (W9-W10) |
| Hour 12-24 | Cognitive workload + first benchmark run |
| Hour 24-36 | Tuning out the worst failures + second benchmark run |
| Hour 36-44 | Migration harness + HhtlBackend stub |
| Hour 44-48 | POSTMORTEM + README + final benchmark + handoff |

If you hit 44 hours and nothing works end-to-end, ship the postmortem
anyway — the "what broke" data is also migration baseline.

## Why this matters

When the HHTL substrate (TiKV + SurrealDB + Ractor + ndarray +
lance-graph) is ready to demo, we will run the IDENTICAL cognitive
workload through the migration harness against BOTH backends. The
numbers will tell us whether the homogeneous-consolidation bet pays out
in practice, not just in architecture diagrams.

If HHTL is 100× faster at the same workload on 1/10 the operational
footprint, the migration justifies itself. If HHTL is only 2× faster,
we have a much harder conversation. Either way, we need the baseline
to know.

Begin. Report progress every 4 hours with a status table per worker.
```

---

## Notes for using this prompt

- Drop into a fresh Claude Code session with `--allowed-tools '*'` and
  Docker + docker-compose installed on the machine.
- The 12-worker spawn pattern matches the master-consolidation protocol:
  brainstorm/scaffolding/review split by model (Opus / Sonnet / Opus).
- Expect ~30-50 GB disk usage (JVM-heavy stack, multiple data volumes).
  Prune aggressively between runs.
- The POSTMORTEM.md is the most under-rated deliverable. Stakeholder
  conversations about "should we migrate?" hinge on that one page.
- Once the Bardioc baseline + HHTL substrate are both running, the
  migration harness becomes the cutover instrument: dual-write phase,
  read-mirror phase, primary-flip phase, decommission phase. Each phase
  is one harness reconfiguration.
