# ndarray::simd Trojan Horse — Claude Code Flex Prompt

Inject `ndarray::simd` into the most CPU-hungry layers of the legacy Bardioc
stack (ClickHouse + Tantivy/Quickwit), measured against stock, in a single
weekend. Goals: real-world validation of ndarray::simd against the gold-standard
SIMD code on earth + strategic dependency injection that softens the eventual
HHTL migration.

Copy the block below into a fresh Claude Code session. Authorize `--allowed-tools '*'`,
Docker, Rust 1.94, CMake, Clang ≥ 17, GCC ≥ 13.

Budget: 48 hours wall-clock.

---

```text
You are injecting `ndarray::simd` (from adaworldapi/ndarray, AVX-512 default
target via .cargo/config.toml `target-cpu=x86-64-v4`) into the hot SIMD paths
of two CPU-hungry data systems: ClickHouse and Tantivy (which Quickwit
inherits). The deliverable is a measurable speed/parity report against stock,
not production patches.

Spawn 12 parallel workers + 1 coordinator (you). Use git worktrees per worker.
Branch per worker: `simd-trojan/{role}-{id}`. Cargo gates skipped per worker;
integration is the gate. Coordinator cherry-picks to main when smoke tests pass.

## Why this matters

ndarray::simd has been validated against synthetic micro-benchmarks but not
against real-world OLAP/FTS workloads. ClickHouse C++ SIMD and Tantivy's
packed-integer codecs are the gold standard. If ndarray::simd matches or
beats them via direct integration, the result is:

1. Real-world validation against the most demanding SIMD workloads on earth.
2. Upstream contribution opportunities (ClickHouse `rust/` workspace, Tantivy
   crate ecosystem).
3. Strategic dependency injection: the legacy Bardioc stack now depends on
   ndarray::simd, so the eventual HHTL migration is "completing a dependency
   you've already accepted" rather than "rip-and-replace".
4. A Trojan horse: ndarray::simd embedded in OSS infrastructure earns
   ecosystem trust independent of the AdaWorldAPI cognitive stack.

## Targets in scope (and explicitly OUT)

IN:
- ClickHouse (C++ via existing `rust/` cargo workspace integration)
- Tantivy (direct Rust dependency injection)
- Quickwit (gets it for free via Tantivy)

OUT (do not waste worker-hours here):
- Elasticsearch / Lucene: JNI overhead too high; bypass via Quickwit instead.
- TinkerPop / Gremlin: mostly scalar traversal; JNI kills SIMD gains.
- ScyllaDB: noted as a follow-up; not this weekend.

## ClickHouse injection plan

ClickHouse already has a Rust workspace at `rust/` (prql, skim_str, parquet,
blake3, etc.) — use it. Add a new crate `rust/ndarray_simd_kernels/` that
exposes C-ABI wrappers around ndarray::simd primitives. Wire into ClickHouse's
function registration via the same pattern existing Rust crates use.

Target kernels (priority order — pick the ones with the most existing C++
SIMD hand-tuning to make the comparison fair):

1. `src/Functions/sum.cpp` → `ndarray::simd::reduce_sum_f32/f64/i32/i64`
2. `src/AggregateFunctions/AggregateFunctionAvg.cpp` → sum + count combined
3. `src/Functions/array/arrayMin.cpp` + `arrayMax.cpp` →
   `ndarray::simd::reduce_{min,max}`
4. `src/Functions/like.cpp` (substring match) → `ndarray::simd::substring_find`
   (W1a closure-parameterized batch primitive)
5. `src/Common/HashTable/Hash.h` (hash function batching) →
   `ndarray::simd::hash_xxh3_batch`
6. `src/Functions/comparison.cpp` (`==`, `<`, `>` on numeric columns) →
   `ndarray::simd::compare_lt/eq/gt`

Function-call FFI overhead amortizes over `DEFAULT_BLOCK_SIZE = 65536` rows.
Per-block FFI is fine; per-row FFI is not. Design the C ABI accordingly:
batch-in, batch-out, no per-element callbacks across the FFI boundary.

Build setup:
- ClickHouse build via `cmake -DENABLE_RUST=1 -DENABLE_TESTS=1`
- `rust/ndarray_simd_kernels/Cargo.toml` depends on `ndarray = { git = "..." }`
- `cbindgen` to generate the C header
- Static link into ClickHouse server binary
- Runtime dispatch: ClickHouse's `CpuFlags` decides which backend; ndarray
  exposes separate AVX-512 / AVX2 / NEON / scalar entry points

## Tantivy injection plan

Tantivy is already Rust — direct dependency. Fork Tantivy at the current tag,
add `ndarray = { git = "..." }` to its `Cargo.toml`, replace its SIMD code
paths with ndarray::simd calls.

Target paths (look for `#[cfg(target_feature)]` and packed-int decoding):

1. `src/postings/compression/` — bitpacked posting list decode →
   `ndarray::simd::bitpack_decode_u32`
2. `src/aggregation/bucket/range.rs` — range bucketing →
   `ndarray::simd::bucketize_f64`
3. `src/query/term_query.rs` — term frequency scoring (BM25) →
   `ndarray::simd::bm25_score_batch`
4. `src/postings/skip.rs` — skip list intersection →
   `ndarray::simd::intersect_sorted_u32`
5. `src/columnar/` — columnar reads → `ndarray::simd::gather_f32/u32`

Tantivy has a comprehensive test suite. The bar is: all Tantivy tests pass
with the ndarray::simd backend, AND the bench suite shows parity or better.

## Worker split (12 + coordinator)

| Worker | Target | Role |
|---|---|---|
| W1 | ClickHouse | Build setup + `rust/ndarray_simd_kernels/` crate skeleton + cbindgen |
| W2 | ClickHouse | Kernels 1+2 (sum, avg) + benches |
| W3 | ClickHouse | Kernels 3+6 (min/max + comparison) + benches |
| W4 | ClickHouse | Kernel 4 (substring match) + benches |
| W5 | ClickHouse | Kernel 5 (hash batching) + benches |
| W6 | Tantivy | Fork setup + ndarray dep wiring + paths 1+5 (bitpack + gather) |
| W7 | Tantivy | Path 2 (range bucketing) + benches |
| W8 | Tantivy | Path 3 (BM25 scoring) + benches |
| W9 | Tantivy | Path 4 (skip list intersection) + benches |
| W10 | ClickHouse + Tantivy | C ABI parity tests (same kernel from both sides returns same result) |
| W11 | Both | Combined benchmark harness (docker-compose: ClickHouse + Quickwit + workload) |
| W12 | Both | Report generator: stock-vs-ndarray-simd latency tables in markdown + plots |

Coordinator: integration testing, cherry-pick, docker-compose orchestration,
final REPORT.md.

## Benchmarks (deliverable)

Run BOTH stock and ndarray::simd-injected versions against the SAME workloads:

ClickHouse workload:
- TPC-H scale factor 10 (~10 GB)
- Q1, Q3, Q6, Q14 (these stress the kernels we replaced)
- Report: p50/p95/p99 query latency, CPU instructions retired (`perf stat`),
  IPC, cache miss rate

Tantivy/Quickwit workload:
- StackOverflow dataset (~10M docs, full body)
- 1000 queries from a realistic mix: term, phrase, range, aggregation
- Report: p50/p95/p99 query latency, indexing throughput, cold-cache vs
  warm-cache latency

Output: `./benchmarks/REPORT.md` with side-by-side tables.

## Acceptance criteria

Per kernel:
1. Correctness: parity with stock (bit-exact for integer, ULP-bounded for
   float)
2. Performance: within 5% of stock OR faster. If slower by >5%, document
   why (function-call overhead per call, larger batch needed, missing AVX-512
   primitive, etc.)
3. Test coverage: existing test suites pass unchanged.

Per system:
- ClickHouse: full `ctest` passes with `ENABLE_RUST=1`
- Tantivy: `cargo test --all-features` passes
- Benchmarks reproduce in a clean docker-compose stand-up

## Anti-goals

- Do NOT optimize ndarray::simd to win these specific benchmarks. The point
  is to measure what ndarray::simd is TODAY against the gold standard, not
  to fake a win.
- Do NOT introduce nightly-only code paths (Rust 1.94 stable; portable_simd
  is gated, intrinsics via `core::arch::*` only).
- Do NOT upstream patches this weekend. The deliverable is the validated fork
  + benchmark report, NOT merged PRs. Upstream contribution is a separate
  follow-on after the numbers are clean.
- Do NOT touch HHTL substrate (PR-X4, PR-X9, etc.). This is independent
  validation of ndarray::simd, not HHTL development.
- Do NOT add new SIMD primitives to ndarray::simd to plug gaps. If a kernel
  needs a primitive that doesn't exist, document the gap and skip that
  kernel — the gap becomes a follow-on ndarray::simd PR with W1a consumer
  contract.

## Time budget

| Hour 0-4 | Build setups (W1, W6) + worker bootstrap |
| Hour 4-16 | Per-kernel implementation (W2-W5, W7-W9) in parallel |
| Hour 16-24 | C ABI parity testing (W10) + first benchmark pass (W11) |
| Hour 24-36 | Tune the worst regressions; iterate kernel-by-kernel |
| Hour 36-44 | Final benchmark pass + report generation (W12) |
| Hour 44-48 | REPORT.md write-up + identified upstream-PR opportunities + handoff |

If a kernel doesn't reach parity in its allotted window, document the gap
(missing ndarray primitive, FFI overhead too high, layout mismatch) and
move on. Honest negatives are also data.

## Strategic outcomes (what the report unlocks)

1. **Validation**: ndarray::simd benchmarked against ClickHouse C++ SIMD
   (decades of hand-tuning) and Tantivy's bitpacked codecs (Lucene-class
   FTS). This is the bar.

2. **Upstream PR pipeline**: each kernel that hits parity-or-better becomes
   a candidate upstream contribution. ClickHouse `rust/` workspace is the
   natural channel; Tantivy crate ecosystem the other. Earns ecosystem
   credibility independent of AdaWorldAPI.

3. **Migration pressure relief**: if the Bardioc stack itself gets faster
   via ndarray::simd injection, the cutover urgency decreases. That's
   actually GOOD — it lets HHTL ship on its own merits rather than under
   "we have to migrate, Bardioc is too slow" pressure. Honest migration
   conversation.

4. **Dependency Trojan horse**: when HHTL is ready, ClickHouse and Tantivy
   already depend on ndarray::simd. The migration is "completing a
   dependency you've already accepted" rather than "abandoning everything".
   Softer organizational change.

5. **Cross-team signal**: this weekend ships a benchmark report that any
   ClickHouse / Tantivy team can read and respond to. Opens conversations
   that pure cognitive-stack work doesn't.

Begin. Report progress every 4 hours with a status table per worker (kernel
done / in-progress / blocked + correctness pass-fail + perf delta vs stock).
```

---

## Notes for using this prompt

- Drop into a fresh Claude Code session on a build machine with Rust 1.94 +
  Clang ≥ 17 + GCC ≥ 13 + CMake ≥ 3.27 + Docker.
- ClickHouse build is heavy: ~40 GB disk, ~30 min full build. Plan accordingly.
- Tantivy build is light: ~5 min.
- Quickwit is the operational shell for Tantivy benchmarks — easier than
  running raw Tantivy bench harness.
- The 12-worker pattern matches the master-consolidation protocol. brainstorm
  (Opus) for kernel design, scaffolding (Sonnet) for FFI wrappers, review
  (Opus) for correctness gates.
- If you only have 24 hours (half-flex), cut Tantivy entirely and focus on
  ClickHouse kernels 1+2+4 (sum, avg, substring) — these are the most
  ClickHouse-celebrated SIMD paths and the most impressive parity comparison.
- The REPORT.md should be writable as a blog post — that's the strategic
  amplification angle that turns a benchmark exercise into ecosystem signal.

## Follow-on opportunities (NOT this weekend)

- **Upstream PR cadence**: 1 ClickHouse PR per 2 weeks for each parity-or-better
  kernel. Tantivy PRs faster (no JVM in the build pipeline).
- **ScyllaDB Rust driver SIMD**: hash-function family swap. Similar shape.
- **cudf / Polars** (Rust DataFrame): Polars already uses ndarray-style
  vectorization; check if ndarray::simd primitives can replace its
  hand-rolled ones.
- **Apache Arrow Rust**: arrow-rs has SIMD for filter/take/aggregate;
  ndarray::simd could plug in there too.
- **DataFusion** (Rust SQL engine): similar to Arrow path; the SIMD layer
  is generic enough to swap.

The pattern generalizes: any Rust-or-Rust-FFI-able data system with hot
SIMD paths is a candidate. ndarray::simd as the canonical Rust SIMD
substrate is a multi-year strategic position; this weekend is the proof
of concept.
