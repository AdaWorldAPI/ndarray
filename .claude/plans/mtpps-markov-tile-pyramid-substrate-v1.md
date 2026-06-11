# MTPPS — Markov Tile-Pyramid as SoA Envelope: ndarray substrate plan (v1)

> **Status:** PROPOSAL / integration plan. Design-spec only; **no code in this plan**.
> **Authored:** 2026-06-10. Branch: `claude/medcare-gaussian-splat-8z76jc`.
> **Trigger:** "would it make sense to create in ndarray the ability to render or
> encode the Markov tile stacked pyramid as an optimized SoA envelope of a stacked
> grid pyramid — the north star for encoding, rendering, cognition, and x266 — as
> a reusable pattern with a sibling crate for AMX/AVX-512 acceleration, maybe a
> `simd_shader.rs`?"
>
> **Verdict: YES — and ~70% of the substrate already exists.** The plan below is
> a composition + two gaps, not a green-field build. Iron mandate (same as the
> identity plan): **compose existing committed modules, do NOT re-invent.**
>
> **Cornerstone framing:** `cesium/docs/GENERIC_RENDERER_CORNERSTONE.md` (MTPPS
> definition — Markov cascade routing over a stacked tile pyramid with
> per-Gaussian perturbation; geo / graph / splat payloads, one substrate).

---

## 1. What already exists (grounded inventory — do not rebuild)

| Need | Existing module | Evidence |
|---|---|---|
| **Stacked grid pyramid** | `src/hpc/blocked_grid/` (PR-X3) — `BlockedGrid<T, BR, BC>` with L1–L4 tiers as iteration patterns (`super_block.rs`, `TierBlockIter`, `blocks_tier`), zero-dim compile-fail guard | mod.rs header |
| **SoA envelope** | `src/simd_soa.rs` (PR-X1) — `MultiLaneColumn` carriers, layout-only, re-exported via `crate::simd::*` per W1a; PLUS `blocked_grid_struct!` SoA-of-grids macro (worker B) | simd_soa.rs, grid_struct_macro.rs |
| **AMX acceleration** | `src/simd_amx.rs` — AMX-TILE/INT8/BF16 **hardware-confirmed via inline asm on stable Rust 1.94** (TDPBF16PS, TDPBUSD; 256 MACs/instr); `hpc/amx_matmul.rs`, `bf16_tile_gemm.rs`, `int8_tile_gemm.rs`, `vnni_gemm.rs` | simd_amx.rs header |
| **Grid↔AMX shape unity** | `AmxBf16Grid = BlockedGrid<u16, 16, 16>`, `AmxInt8Grid = BlockedGrid<u8, 16, 64>` — **the AMX tile IS a grid tier alias already** | blocked_grid/aliases.rs:92,109 |
| **Cascade precedent (search, NOT routing)** | `src/hpc/cascade.rs` — 3-stroke HDR adaptive cascade (Hamming NN + precision tiers VNNI/BF16/DeltaXor). Precedent for tiered skip-machinery; the Markov *transition-table router* itself does NOT exist yet — it is the core of D-MTP-1, not inventory | cascade.rs header |
| **Render sink** | `src/hpc/renderer.rs` — double-buffer SIMD renderer (front/back `RenderFrame`, atomic swap, F32x16 FMA; AVX-512/AVX2/**AMX**/NEON/scalar tiers; explicitly "Neo4j-style visual rendering") + `src/hpc/framebuffer.rs` — palette-indexed framebuffer ("ndarray IS the graphics card"; 4-bit nibble wire format, tier-adaptive palette) | renderer.rs, framebuffer.rs headers |
| **Tile streaming envelope** | `crates/cesium/` — `implicit_tiling.rs`, `hlod.rs`, `sse.rs`, `tileset.rs`, `khr_gs.rs`, `spz.rs`, `to_cam_soa.rs`, `osm_pbf.rs` | crate listing |
| **Splat SIMD math (queued)** | D-SPLAT-2 five primitives (`batched_cholesky_3x3`, `batched_mahalanobis`, `batched_opacity_blend`, `batched_sh_eval_l3`, `batched_se3_transform`) per `splat-native-ultrasound-simd-substrate-v1.md` | ndarray plans |

**The headline finding:** the "optimized SoA envelope of a stacked grid pyramid"
is not hypothetical — `BlockedGrid` L1–L4 + `blocked_grid_struct!` + `simd_soa`
carriers ARE that envelope. And the AMX alignment is strong, with one honest
caveat: a 16×16 u16 grid block (512 B = 16 rows × 32 B) loads directly as a
**half-width AMX tile** (max tile is 16 rows × 64 B), and `TDPBF16PS` computes a
16×16 f32 accumulator from **K=32** BF16 operands with the B operand in
**VNNI pair-interleaved layout**. So the transition table needs **one build-time
repack** (row-pairing into VNNI order, amortized across all queries) — after
which the hot path runs with zero per-query adaptation. "Same shape" is true at
the accumulator (16×16 f32 out); the operand layout is a documented repack, not
free.

## 2. What's missing (the two real gaps)

1. **No `src/simd_shader.rs`** — there is no kernel family that *composes*
   pyramid-walk + cascade-route + perturb-paint as one W1a-conformant surface.
   Today the pieces live apart (blocked_grid = layout, cascade = search,
   renderer = sink).
2. **No sibling crate that packages the pattern** — `crates/cesium` is
   ingest/streaming; `crates/p64` is convergence; nothing owns "tile-pyramid as
   reusable encode/render/cognition pattern".

## 3. Deliverables

### D-MTP-1 — `src/simd_shader.rs` (the Markov-tile kernel family)

**Owner:** ndarray `src/`. **~600 LOC + tests + bench.** **Risk: MED.**
**Governed by the W1a consumer contract** (struct methods on typed wrappers,
closure-parameterized batch primitives, all three backends AVX-512/NEON/scalar
implemented, parity test mandatory, saturating/overflow semantics documented).
AMX is an *additional* fast path behind `simd_caps()`, never the only path.

Four kernels (names indicative):

- `tile_frontier_route` — frontier vector × 16×16 transition tile → **raw
  child scores only**. The Skip/Attend/Compose/Escalate decision is NOT made
  here: thresholds/policy arrive as **closure parameters** (the W1a
  closure-parameterized-primitive pattern), so the `RouteAction` semantics stay
  upstream where they live today (`bgz-tensor::hhtl_cache` owns the enum).
  Scores are hardware; routing policy is thinking — the ndarray=hardware rule
  holds. Hot path: gather + table-lookup + popcount; AMX `TDPBF16PS` on
  VNNI-repacked BF16 transition tiles when available (one build-time repack,
  §1). This is the **Markov** step — Markov in the plain first-order sense
  (level-(k+1) routing depends only on the level-k frontier and the table).
- `tile_pyramid_predict` — upsample tier-k block → tier-(k+1) prediction
  (bilinear/nearest per payload). This is the **x266-style intra-pyramid
  prediction** step: predict the finer level from the coarser, so only
  residuals need encoding.
- `tile_residual_code` — encode/decode the predict-residual through the
  existing palette path (Palette256/ADR-024 + PhiSpiral256 residual-location
  atoms). Encode = quantize residual to palette index; decode = const-table
  lookup, zero-allocation. This is the **codec** step — render and encode share
  one walk: a renderer is a decoder that paints, an encoder is a renderer that
  remembers.
- `tile_perturb_paint` — per-leaf Gaussian perturbation (Σ shaping, opacity,
  SH ℓ≤3) blended into `hpc::framebuffer`/`renderer` targets, reusing D-SPLAT-2
  primitives once they land (soft dep; scalar reference path first). This is
  the **perturbation** step.

**Layering law (from simd_soa.rs precedent):** `simd_shader.rs` dips ONLY into
`crate::simd::*` typed wrappers — never raw per-arch intrinsics; layout stays in
`blocked_grid`/`simd_soa`; this file is *operations*.

**Tests:** parity across backends; AMX-vs-scalar equivalence on 16×16 tiles;
predict+residual round-trip lossless at palette resolution; route determinism.

### D-MTP-2 — sibling crate `crates/mtpps` (the reusable pattern)

**Owner:** new `crates/mtpps`. **~800 LOC + tests.** **Risk: MED.**

The packaging crate that composes — not re-implements — the pattern:

```text
TilePyramidSoA<T>                       (the SoA envelope, via blocked_grid_struct!)
  ├ tiers: BlockedGrid L1..L4           (stacked grid pyramid — exists)
  ├ lanes: bind_source/state/signal/time/cert   (simd_soa MultiLaneColumn — exists)
  ├ transitions: AmxBf16Grid per tier   (Markov tables as DATA — exists as alias)
  └ walk(): cascade route → predict → residual-code → perturb-paint
            (the four D-MTP-1 kernels, one traversal)
```

Three consumer facades, one walk: `encode()` (walk that remembers residuals),
`render()` (walk that paints into framebuffer), `traverse()` (walk that returns
routed candidates — the cognition/HHTL use). **This is the north-star claim
made concrete: encoding, rendering, and cognition are the same pyramid walk
with different sinks.**

**Architecture rule (CLAUDE.md, non-negotiable):** ndarray = hardware. The
Markov *transition tables are inputs* (data); their *semantics* (NARS, thinking
styles, truth) stay upstream in lance-graph. `crates/mtpps` has zero
thinking logic — it is mechanism, like `crates/cesium`.

**Tests:** the §8 cornerstone fixture (one tile, three payload kinds —
geo/graph/splat — through one walk); encode→decode→render round-trip;
`cargo clippy -- -D warnings` clean.

### D-MTP-3 — x266-style codec probe (falsifiable north-star gate)

**Owner:** `crates/mtpps/examples/`. **~200 LOC.** **Risk: LOW.**

The example IS the probe (workspace convention). Run predict+residual coding
over a synthetic pyramid (then a real OSM tile when D-OSM-2 lands) and report:

- **Reconstruction-error budget** for the residual codec path — a per-primitive
  error bound (max + p95, payload-appropriate units), NOT rank correlation.
  ρ measures ordering fidelity of palette *distances*; codec fidelity is
  reconstruction error. Both are reported, each gating its own claim:
- **ρ-vs-reference ≥ 0.99** (ADR-024 contract) — only where the palette is used
  as a *distance surrogate* (similarity/routing), not as the codec-fidelity gate.
- **bits/primitive vs flat encoding** — the pyramid-prediction win must be
  measured, not asserted (truth-architect is mandatory reviewer per Hard Rule:
  no performance claims without bench).
- **AMX payoff** — `tile_frontier_route` AMX vs F32x16 vs scalar at N=1M
  primitives; gate: AMX ≥ 2× F32x16 on confirmed hardware, else document.

**Scope honesty:** "x266" here means *the coding pattern* (hierarchical
intra-pyramid prediction + residual quantization, as VVC-class codecs use) —
NOT a bitstream-compatible H.266 implementation. That would be a multi-year
arc; the pattern is the north star, the bitstream is out of scope.

## 4. Sequencing + gates

```text
D-MTP-1 (simd_shader.rs)  ── gates on: nothing hard (D-SPLAT-2 soft, scalar-first)
        │
        ▼
D-MTP-2 (crates/mtpps)    ── gates on: D-MTP-1
        │
        ▼
D-MTP-3 (x266 probe)      ── gates on: D-MTP-2; real-data run gates on D-OSM-2
```

First brick: **D-MTP-1 `tile_frontier_route` + `tile_pyramid_predict`** with
scalar + AVX-512 + NEON parity — the AMX fast path lands in the same PR only if
the inline-asm path stays clean; otherwise it follows.

## 5. Constraints (iron rules binding this plan)

- **W1a consumer contract** on every new `pub fn` in `src/simd_*.rs` — all
  three backends, parity tests, documented saturation semantics.
- **Agent cargo hygiene** — fleet edits in the shared checkout; orchestrator
  compiles once.
- **ndarray = hardware** — no NARS/truth/thinking semantics in D-MTP-1/2/3;
  transition tables are opaque data.
- **blocked_grid layering** — no SIMD in blocked_grid; no layout in
  simd_shader; the macro composes them.
- **Measurement before claims** — D-MTP-3 is the gate; no "faster" assertions
  without its numbers.

## 6. Open questions

- **OQ-MTP-1:** crate name — `mtpps` vs `tile-pyramid` vs folding into
  `crates/cesium` behind a `pyramid` feature. (Recommend: separate `crates/mtpps`;
  cesium stays ingest/streaming-shaped.)
- **OQ-MTP-2:** transition-tile dtype default — BF16 (AMX TDPBF16PS, more
  range) vs u8×i8 (TDPBUSD, 4× density)? (Recommend: BF16 default, i8 feature.)
- **OQ-MTP-3:** does `tile_perturb_paint` write into `hpc::framebuffer`
  (palette-indexed, wire-ready) or `hpc::renderer` (RenderFrame double-buffer)?
  (Recommend: framebuffer for encode-parity — paint and code share the palette.)
- **OQ-MTP-4:** `BlockedGrid` is **2-D** by design (PR-X3 header). Quadtree
  pyramids (maps, images, screen-space) are native. The 3-D *addressing* is **not
  a blocker** — `hilbert3d_encode` is **verified green** (#215, 2026-06-10:
  13/13 tests, `level4_all_indices_unique` bijective onto [0,4096), the exact
  property cascade addressing needs; the earlier `encode([15,15,15],4)==4095`
  "red" was a wrong *orientation expectation*, not an encoder bug — 2925 is a
  valid endpoint). So a 3-D address space exists and is tested; the remaining gap
  is only the *storage container*. Octree *volume storage* (3-D splat fields,
  the `ok:` implicit-tiling variant) is NOT yet covered by `BlockedGrid` —
  options: (a) z-sliced stack of 2-D grids per level (cheap, anisotropic),
  (b) a future `BlockedGrid3<T, BR, BC, BD>` (real work, own plan),
  (c) project-to-2-D at paint time and keep volumes upstream. v1 scopes storage
  to quadtree; the octree-storage decision is deferred until a volumetric
  consumer demands it. The L4 Hilbert suite is a **standing regression gate**
  (any table change must keep it green before L-deep addressing claims), no
  longer a blocker. Do NOT claim the 2-D grid "is" the 3-D pyramid — but DO note
  the 3-D address math is already sound.

---

_End of plan v1. Companion: `cesium/docs/GENERIC_RENDERER_CORNERSTONE.md` (MTPPS
definition), `3DGS-SIMD-forward-renderer-plan.md`, `3DGS-HHTL-CPU-cascade-plan.md`,
`3DGS-columnar-splat-codec-plan.md`, `splat-native-ultrasound-simd-substrate-v1.md`._
