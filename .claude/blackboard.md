# Current epoch (2026-05-26) — splat / palette / pillar / 3DGS

> **Read this first.** The "Polyglot Notebook" architecture below is a
> separate/older program, not the current epoch.

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
