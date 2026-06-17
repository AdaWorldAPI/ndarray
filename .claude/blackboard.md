# Current epoch (2026-05-26) — splat / palette / pillar / 3DGS

> **Read this first.** The "Polyglot Notebook" architecture below is a
> separate/older program, not the current epoch.

## 2026-06-17 — DECISION: HHTL fork ladder coded in `hpc::entropy_ladder` (CONJECTURE)

Reified the operator's standing idea — *if the orthogonal (helix/CAM-PQ)
leaf residue is strong enough, free energy forks into another domain
(HHTL shift = new exploration)* — as pure functions beside the existing
entropy/quadrant code. Unifies four vocabularies as one 2-axis structure:
`entropy_ladder::Quadrant(entropy,energy)` ≡ `lance-graph-contract::mul::
FlowState(challenge,skill)` (Csikszentmihalyi) ≡ Friston model-vs-surprise
≡ Staunen↔Wisdom.

- `residue_surprise(mag, noise_floor, sigma_k) → [0,1]` — orthogonal residue
  magnitude (prediction error the in-domain centroid codebook fails to
  explain) → challenge axis. Below floor = quantization (≈0); linear ramp
  over `sigma_k·noise_floor`. Threshold provenance per `I-NOISE-FLOOR-JIRAK`
  (Berry-Esseen wrong under CAM-PQ weak dependence); ramp is an honest proxy
  pending Jirak calibration, **not** a claimed bound.
- `ForkAction {Commit, DescendDeeper, ForkBasin, ForkDomain}` + `fork_decision`
  — bands challenge−skill on the shipped `mul::flow_state_from` boundaries
  (Anxiety δ>0.2, Boredom δ<-0.2; the matched middle |δ|≤0.2, which
  flow_state_from splits into Flow/Transition, collapses to one in-domain
  branch here); HHTL depth decides descend-vs-
  fork. `ForkDomain` (mint a new classid = the Friston model-switch) requires
  BOTH leaf depth AND challenge≫skill — the "strong enough AT THE LEAF" invariant.

Layering kept honest: the `FlowState` *assessment* stays in lance-graph
(thinking); the fork *math* lives here in ndarray (substrate, where residue +
energy physically are) — per the Architecture Rule. Pure fns + one enum, no
struct/layer, composes with `Quadrant`. 5 lib + 2 doctests, clippy clean.
Branch `claude/jirak-math-theorems-harvest-rfii13`.

**Loose ends (CONJECTURE → gated):** (a) feed the *real* `edge_codec::
CoarseResidue` magnitude from the live codec into `fork_decision` (currently
caller-supplied); (b) `ForkDomain` vs `ForkBasin` should be arbitrated by
residue *orthogonality* (⊥ all in-domain centroids = genuinely new), not the
depth+delta proxy; (c) Jirak-derived σ threshold to replace the `sigma_k`
proxy. This driver-side wire merges with lance-graph `materialize`'s
`ThoughtCtx::from_live` step (same call-site).


## 2026-06-10 — DECISION: GUID prefix→shape routing crystallized (docs-only)

The operator-pinned canonical GUID (`OGAR/CLAUDE.md`: hex dash-groups =
`classid(8)-HEEL(4)-HIP(4)-TWIG(4)-[basin·leaf+id]`; 3×4 tiers, `>> 2`)
now has its ndarray-side contract at
`.claude/knowledge/guid-prefix-shape-routing.md`: ndarray = MECHANISM
(layout-only `PrefixShapeTable`, opaque `ShapeId(u16)`, longest-prefix,
L1/L2-resident, no distance API — no-umbrella honored), consumer =
POLICY (lance-graph registers the table). GridLake continuation: key
selects grid family + pyramid level; value stays one byte-store
(column-substrate identity). φ-quorum anti-eigenvalue-theater contract
pinned with the PP-13 casebook as failure catalog; probes named
(ROUTE-1, QUORUM-1, PHI-1, PYR-1, CODEBOOK-44; HILBERT-L4 = existing
P0-4 blocker for any L4 cascade claim). CONJECTURE until coded — no
.rs touched in this commit.

## Evidence model (binding — from PR #200)
- **L0** = source · passing tests · ratified standards (ground truth).
- **L1** = `.claude/PR-X12-docs-audit.md` (#200) + `.claude/knowledge/plans-alignment-triage.md` — claims-about-source; **spot-check, never inherit**.
- **L2** = `.claude/plans/*` + `pr-x12-*` perspective docs = **inspiration, NOT evidence**.
- **Whole-file reads only** — no `grep`/`sed`/`head`/`tail` (`ls` to locate).
- Build/bench locally at `target-cpu=x86-64-v4`; committed `.cargo/config.toml` stays **v3** (GitHub/CI).

## Settled architecture (grounded this epoch, whole-read)
- **Cognitive similarity/cosine = Palette256 + Fisher-z**, integer: `hpc::cam_pq`
  squared-L2 ADC (u8 codebook indices) gated by θ = `distance::similarity_z` (atanh).
  Validated 10k×10k @ θ≈cos-0.90. **No float MAC in the distance kernel.**
- The cognitive **"splat"** = `lance-graph-contract::splat::CamPlaneSplat` (q8) →
  `AwarenessPlane16K` (16 384-bit OR deposition). **Sibling of, not the same as,** the
  graphics `splat3d` EWA renderer (per `splat3d/mod.rs`).
- EWA float-Σ sandwich (`splat3d::spd3`, Pillar 6/7) = uncertainty propagation +
  certification, **not** similarity. Pillar suite (6–17) certifies the substrate;
  **Pflug-10 certifies the CAM-PQ palette**.
- Typed distance (`cognitive-distance-typing.md`): one named fn per metric, newtype
  outputs, **no `fn distance<T>` umbrella**, conversions explicit. `palette→fisher→
  cosine→hamming` roundtrip is the named anti-pattern.

## Outstanding (per triage + #200)
- **#4** pr-x12 doc-fixes + evidence-policy + archive fabrication-heavy plans (Geo/Gov).
- **#5** ASG-leaf canon spec (Gov) — prerequisite for #7.
- **#7** ASG-leaf impl (Kernel) — must **extend `CamPlaneSplat`**, not reinvent; trails #5.
- `cam-pq-production-wiring` (UNOWNED, lance-graph) — route `cam_pq` through `CamCodecContract`.
- `UNUSED_INVENTORY_1.95` A1–A9 dead-code (phantom `SimdTier::{Sse2,WasmSimd128}`, stale 1.64 imports).

## Consolidation-sprint debt (PR-X program; ground-truthed `ls src/hpc/` 2026-05-27)
> Shipped-state vs `pr-master-consolidation.md`. Landed: ✅ **PR-X10** `linalg/`,
> ✅ **PR-X11** `pillar/`, ✅ **PR-X13** `ogit_bridge/`, ✅ **PR-X3** `blocked_grid/`.
- **PR-X12 codec ⚠️ v1 NEAR-COMPLETE** — `ctu/mode/predict` + now **`rdo` (A6, λ-RDO,
  integer fixed-point λ_q8 — no float) + `ans` (A7, static-table rANS over the 4-symbol
  mode alphabet, bit-exact round-trip)**. Remaining: `transform` (A4, deferred to v2 per
  design Q2) + `stream` (A8, framing over `ans`). 81 lib + 20 doctests green, clippy clean.
- **PR-X4 splat4d ❌ OUTSTANDING** — no `src/hpc/splat4d/`. Unbuilt.
- **PR-X9 cognitive ❌ OUTSTANDING** — no `src/hpc/cognitive/`. Unbuilt; must **consume**
  `lance-graph-contract::splat::CamPlaneSplat` (q8), never redefine it (contract is sacred).

## Merged / closed this epoch
- ✅ #201 triage · #205 `3dgs-tiles` (cesium tileset) · #206 + #208 render-depth cert.
- ❌ #207 EWA-SYRK bench **closed** (wrong regime — category error).
- 🗑 `phi_spiral.rs` abandoned (float, wrong manifold). Net new usable code this
  session = 0 (see `board/EPIPHANIES.md` grounding-discipline entry).

---

# Polyglot Notebook — Single Binary Architecture

> Separate/older program — NOT the current epoch (see top of file).

## The Binary

One `cargo build`. Ships as one executable. Contains:

```
reactive runtime     (transcoded from marimo Python)
graph query engines  (transcoded from graph-notebook Python)
kernel protocol      (Rust-native ZMQ, from kernel-protocol spec)
document publisher   (transcoded from quarto TS/Deno)
local graph database (lance-graph, already Rust)
SIMD kernels         (ndarray, already Rust)
graph compiler       (rs-graph-llm, already Rust)
web frontend         (marimo's JS/React, served by the binary)
```

External process: R only (Bardioc/almato). Speaks Arrow IPC to the binary.

## Repos → Crates

| Repo (source) | Becomes | Work |
|------|---------|------|
| marimo | `crate::runtime` + `crate::server` | Transcode Python→Rust |
| graph-notebook | `crate::query::{cypher,gremlin,sparql,nars}` | Transcode Python→Rust |
| kernel-protocol | `crate::kernel` | Implement from spec in Rust |
| quarto | `crate::publish` | Transcode TS→Rust |
| quarto-r | external R process | Stays R, Arrow IPC bridge |
| lance-graph | `crate::graph` | Already Rust, integrate |
| ndarray | `crate::simd` + `crate::linalg` | Already Rust, integrate |
| rs-graph-llm | `crate::compiler` | Already Rust, fix build |

## Scopes (parallel, non-overlapping)

### SCOPE A: Reactive Runtime (marimo → Rust)
Transcode marimo's reactive cell execution model to Rust.
The core insight: cells have dependencies, when a cell's input changes,
downstream cells re-execute. That's a DAG scheduler — natural in Rust.

### SCOPE B: Query Engines (graph-notebook → Rust)
Transcode graph-notebook's Cypher/Gremlin/SPARQL executors to Rust.
Bolt protocol client, WebSocket client, HTTP client — all Rust-native.
Add local path: Cypher → lance-graph semiring (no network).

### SCOPE C: Kernel Protocol (kernel-protocol spec → Rust)
Implement Jupyter kernel wire protocol in Rust.
Only needed for R (IRkernel) — everything else runs in-process.
ZMQ via zeromq-rs. Connection file parsing. Message ser/de.

### SCOPE D: Publisher (quarto TS → Rust)
Transcode Quarto's document rendering pipeline to Rust.
Pandoc AST manipulation. Markdown → PDF/HTML.
Custom graph visualization extension.

### SCOPE E: Integration (lance-graph + ndarray + rs-graph-llm)
Wire the existing Rust crates into the binary.
Fix rs-graph-llm build. SIMD kernels for graph ops.
This is mostly Cargo.toml workspace wiring + API surface.

## Decisions
[DECISION] One binary, no Python runtime
[DECISION] marimo's JS frontend served by Rust HTTP server (axum/actix)
[DECISION] R is the ONLY external process (Arrow IPC bridge)
[DECISION] Cypher executes locally via lance-graph semiring by default
[DECISION] Remote DB connections (Neo4j, FalkorDB) via native Bolt client
[DECISION] vis.js graph rendering served as static assets by the binary

## Architecture Decisions

### 2026-06-13 — GEMM-dispatch routing fixes (savant-architect)
Branch `claude/wonderful-hawking-lodtql`. Three public GEMM entry points
were not routing to the accelerated kernels.

- **`backend::gemm_bf16` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Now routes to
  `hpc::amx_matmul::matmul_bf16_to_f32` (AMX `TDPBF16PS` → AVX-512
  `VDPBF16PS` → scalar). Slice→ArrayView2 wrapping mirrors the call shape
  in `simd_runtime::matmul`; inputs sliced to exact `m*k`/`k*n`/`m*n`.
  Bit-equivalent on non-AMX/non-AVX512BF16 hosts because the dispatcher's
  scalar fallback is the same `quantized::bf16_gemm_f32(a,b,c,m,n,k,1.0,0.0)`
  the old direct call used (alpha=1, beta=0 preserved).
- **`backend::gemm_i8` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Routes to `simd_int_ops::gemm_u8_i8`
  (4-tier: AMX `TDPBUSD` → VNNI-zmm → AVX-VNNI-ymm → scalar).
  [DECISION] Deliberately NOT routed to `amx_matmul::matmul_i8_to_i32` as
  the literal task text asked: `gemm_i8` is **u8×i8→i32**, but
  `matmul_i8_to_i32` is **i8×i8→i32** and would reinterpret A-bytes ≥128
  as negative — NOT bit-equivalent. `gemm_u8_i8`'s scalar fallback is the
  same `quantized::int8_gemm_i32` the old `vnni_gemm::int8_gemm_vnni`
  used → bit-identical on scalar hosts; VNNI-zmm arm calls the same
  `int8_gemm_vnni_avx512` kernel as before. All tiers integer-exact.
- **`native::gemv_f32` / `gemv_f64` (src/backend/native.rs)** — FIXED
  THIS TURN (was calling `scalar::gemv_*` unconditionally). Now matches
  on `tier()`: Scalar tier → unchanged `scalar::gemv_*` (byte-identical);
  Avx2/Avx512 tiers → per-row `dot_f32`/`dot_f64` (the existing
  dispatched, parity-tested SIMD dot). GEMV = stack of row dots; each A
  row is row-major-contiguous so contiguous `dot_*` loads apply. Leading
  `n` of each `lda`-wide row taken via `&a[i*lda..i*lda+n]`; no new bounds
  requirement vs scalar ref. SIMD tiers carry the module's documented
  1-2 ULP reduce-order drift (within BLAS tol; `test_gemv_f32` uses 1e-5,
  no byte-exact consumer asserts gemv).

[UNSAFE-AUDIT] gemv fix added **zero** new `unsafe` — it reuses the
already-audited `dot_*` kernels. No new sentinel-qa surface from this turn.
The two mod.rs fixes contain `unsafe` repr(transparent) slice reinterprets
(BF16/u16) that were landed earlier this session and warrant the standard
sentinel-qa pass if not already covered.

[LOOSE END] Repo references modules that exist on disk but the Glob/Grep
index was transiently stale this session (returned empty for
`simd_int_ops.rs`, `vnni_gemm.rs`, `bf16_gemm_f32`); Bash ground-truth
confirmed all present. Orchestrator should `cargo fmt`/`clippy`/`test`
centrally (edits were edit-only, no compile performed here).
