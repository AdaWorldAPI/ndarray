# Plans Alignment Triage — ChatGPT 48h corpus vs ground truth

> **STATUS: NOT AUTHORITATIVE.** This document ranks *inspiration* — the
> ChatGPT-seeded `.claude/plans/` corpus committed 2026-05-24/25 across
> `ndarray` and `lance-graph` — against **ground truth**. It is **not a spec
> to implement** and **may not be cited as evidence** for any claim about
> current code.
>
> `USE-NOW` means only: "the idea coincides with verified code or a ratified
> standard." Build from that ground truth, citing the plan as the idea's
> *origin*, never as its authority. Where a plan diverges from code/canon/
> standard, **the divergence is the correction list** below — ground truth
> wins every time.

## Evidence model (from PR #200 Tier-4, = "inspiration ≠ authority")

| Tier | What | May be cited as evidence? |
|---|---|---|
| **Evidence** | source files · W1a contracts · ratified standards · the merged `.claude/PR-X12-docs-audit.md` (#200) | yes |
| **Inspiration** | the `.claude/plans/` corpus · the `pr-x12-*` perspective docs · the R-1..R-15 series (forward-conditional, audit-flagged) | **no** |

## Ground-truth anchors (verified this session, whole-file)

- **code (ndarray):** `src/hpc/splat3d/{spd3,project,raster,sh,tile,gaussian}.rs`,
  `src/hpc/pillar/{ewa_sandwich_2d,ewa_sandwich_3d,splat_invariants}.rs`,
  `src/hpc/cam_pq.rs` (847 lines), `src/hpc/bgz17_bridge.rs`
  (`BASE_DIM=17`, `GOLDEN_STEP=11`, `Base17=[i16;17]`)
- **code (lance-graph):** `crates/cognitive-shader-driver/*` (bindspace,
  mailbox_soa, token_agreement, codec_bridge, decode_kernel), `crates/jc`
  (Pillar 10 `pflug`, Pillar 11 `hambly_lyons`), `crates/sigker`,
  `crates/bgz17`, `crates/lance-graph/.../blasgraph`
- **audit:** `.claude/PR-X12-docs-audit.md` (PR #200, merged) — the
  authority over the `pr-x12-*` doc layer
- **standards:** `KHR_gaussian_splatting` (RC Feb 2026), OGC 3D Tiles 2.0,
  Cesium HLOD-for-splats (Apr 2026), ArcGIS `ESRI_crs`

## Correction list (plan ⟂ ground truth — ground truth wins)

Sourced from PR #200 (`PR-X12-docs-audit.md`) + first-party reads:

1. **blasgraph is the canonical bit-exact kernel; bgz17 is a *lossy sibling*.**
   The plans/docs that present bgz17 as "the actual kernel home" with
   blasgraph as "eventual abstraction" have it **inverted** (#200 §1, #1/#4).
   Substituting bgz17 for blasgraph is a soundness violation.
2. The symbol `bgz17::scalar_sparse::tropical_spmv` **does not exist**; the
   real kernel is the method `ScalarCsr::spmv_min_plus` (#200 #2).
3. `ndarray::hpc::blas_level2::batched_ssd_search` **does not exist** (#200 #5).
4. `sigker::signature_kernel_pde` is **not buggy** — its own passing tests
   prove convergence to `I₀(2·√⟨u,v⟩)` (#200 #9). The "use truncated because
   pde is buggy" framing is wrong.
5. Per-arch DCT crossover constants (SPR=64, ICX=32, Zen4=96, Apple=256,
   Graviton=128) are **uncalibrated estimates**, not commitments (#200 #6/#7).
   Mark `[uncalibrated]` until a bench measures them.
6. `simd_soa.rs` is a **373-line PR-X1 layout carrier**, not a codec hub
   (#200 #24).
7. The **R-1..R-15 series is forward-conditional planning**, not present-tense
   canon (#200 §2 truth #1).
8. Manifold: the directional lane is **spherical** (KHR ships
   `SH_DEGREE_n_COEF_i`; `sh.rs` is degree-3 SH). Poincaré is reserved for the
   LOD tile-tree only — *not* the directional lane.
9. GS payload = a glTF carrying the `KHR_gaussian_splatting` extension, not a
   peer `ContentKind::GaussianSplat3d`. SPZ = interop tier; PR-X12 = internal.

## Supersedes-note — this thread's own in-conversation errors

Recorded for honesty; the inventory below supersedes them:

- **Gov (this session):** repeatedly asserted the R-7 inversion
  (blasgraph-aspirational / `bgz17::scalar_sparse::tropical_spmv`-is-real), the
  sigker-PDE "bug", and `batched_ssd_search` — all falsified by #200.
- **Geo session:** the `SplatShaderBlas` "blasgraph-aspirational / bgz17-is-the-
  kernel" inversion (retracted).
- **Kernel session:** premature `phi_spiral.rs` on the wrong (flat-disk)
  manifold + the unverified "BGZ17 offset 17-27 / stride 2,4" annotation
  (corrected in `fc27775c`; real values `GOLDEN_STEP=11`).

## Legend

- **HOP** — 1 (coincides with verified code/standard) · 2 (one experiment
  away) · 3 (speculative / fenced)
- **CV** — ✓ cross-verified (whole-read by ≥2 sessions) · ◑ single-session
- **VERDICT** — USE-NOW / KEEP-EPIPHANY / GROUND-FIRST / ARCHIVE / OUT-OF-SCOPE

## Hop 1 — USE-NOW (idea coincides with verified code/standard)

| Plan | Repo | CV | Grounds onto |
|---|---|---|---|
| 3DGS-3D-Tiles-runtime | lance-graph | ✓ | OGC 3D Tiles 1.1 + KHR |
| 3DGS-Lance-Arrow-storage | lance-graph | ✓ | Arrow/Lance schemas (foundation) |
| 3DGS-HHTL-datalake-traversal | lance-graph | ✓ | Lance/DataFusion pruning |
| 3DGS-ArcGIS-Cesium-ingestion | lance-graph | ✓ | real ArcGIS/Cesium + `ESRI_crs` |
| 3DGS-certified-query-render | lance-graph | ✓ | auditable skip/refine/hydrate |
| 3DGS-integration-wiring | lance-graph | ✓ | the lance-graph↔ndarray boundary |
| 3DGS-Cesium-feature-mapping | lance-graph | ✓ | 3D Tiles spec (gold-standard transfer) |
| 3DGS-SIMD-forward-renderer | ndarray | ◑ | real `splat3d/{project,raster,sh}` |
| 3DGS-columnar-splat-codec | ndarray | ◑ | f16 carrier + `splat3d` block views |
| 3DGS-error-certification-pillars | ndarray | ◑ | real Pillar-6/7/12 |
| 3DGS-validation-benchmark | ndarray | ◑ | real renderer/codec/pillar paths |
| 3DGS-HHTL-CPU-cascade | ndarray | ◑ | `hpc::{cascade,clam,cam_pq}` |
| 3DGS-EWA-SYRK-BLAS-MKL | ndarray | ◑ | `spd3::sandwich` + Pillar-7; **backend gated behind bench (PR-3)** |
| cam-pq-production-wiring | lance-graph | ◑ | real `cam_pq` (self-corrects 1300×→128×) · **UNOWNED cross-repo pick-up** |
| codec-sweep-via-lab-infra | lance-graph | ◑ | real `cognitive-shader-driver`; honest negative controls |
| categorical-algebraic-inference | lance-graph | ◑ | real grammar/VSA surface (categorical framing = labeled conjecture) |
| archetype-scaffold | lance-graph | ◑ | self-contained crate, runnable acceptance |
| burn-ndarray-parity-sprint | lance-graph | ◑ | real ndarray parity, pinned base SHA |
| jc-pillars-runtime-wiring **+ ERRATUM** | lance-graph | ◑ | real 12-pillar `jc`; ERRATUM de-risks layer attribution |
| cognitive-substrate-convergence-v3 | lance-graph | ◑ | canonical; tracks merged PRs #383–389 |

### Idea USE-NOW / doc RE-GROUND
| Plan | Repo | CV | Note |
|---|---|---|---|
| PR-X12-tensor-container-expansion-capstone | ndarray | ✓ | block grammar sound (R-1/R-2/R-9), but carries PR #200 #1–#16 debt. Do not promote until PR-4 re-grounds. |

## Hop 2 — KEEP-EPIPHANY (true direction, preserve as vision, not build)

| Plan | Repo | Note |
|---|---|---|
| 3DGS-epiphany-roadmap | lance-graph | **source of C/HLOD** (now standard-backed by Cesium HLOD) |
| 3DGS-blast-radius-application-map | lance-graph | Ring 1 grounded; Rings 2–4 speculative |
| 3DGS-Cesium-BindSpace4-headstone | lance-graph | real anchors + unbuilt markdown |
| tetrahedral-epiphany-splat-integration | lance-graph | "no runtime claimed"; idea source |
| oxigraph-arigraph-cognitive-shader-soa-merge | lance-graph | core splat-thread (aspirational, in-program) |
| ogit-g-context-bundle | lance-graph | the ontology↔codec "semantic exponent" bridge |
| unified-ogit-architecture | lance-graph | anatomy north-star wires FMA + splat pillars |
| thought-cycle-soa-awareness | lance-graph | ⚠ redefines `CausalEdge64` — collides with v3 canon; reconcile first |
| 3DGS-4x4-cognitive-shader-integration | lance-graph | geospatial slice buildable; cross-domain lanes not |

## Hop 2/3 — GROUND-FIRST (needs an experiment/fixture before build)

| Plan | Repo | Note |
|---|---|---|
| 3DGS-4x4-cognitive-shader-SoA | ndarray | `(4×4)⁴` fanout speculative |
| PhiSpiral256-SoA-cross-system | lance-graph | core lane markdown-only; needs the kernel |
| 2026-05-06-splat-osint-ingestion | lance-graph | real EWA/jc spine; markdown-only SPLAT-1 types |
| 3DGS-Blender-transcode-crosspollination | lance-graph | real bridges, speculative mappings |
| 3DGS-SplatShaderBlas-BLASGraph | lance-graph | **plan inverts blasgraph/bgz17 (see correction #1); blasgraph is canonical** |
| 3DGS-domain-adapter-strategy | lance-graph | trait stays plan-level until 2 adapters exist |
| bindspace-columns | lance-graph | cols A–D real; E–H proposals |
| causaledge64-mailbox-rename-soa | lance-graph | real crates; draft compartment layer |
| anatomy-realtime | lance-graph | strong floor; depends on unmerged tiers + fixtures |
| 3DGS-ultrasound-SaMD | lance-graph | no in-repo fixture; SaMD trajectory |
| lance-graph-rdf-fma-snomed | lance-graph | grounds the ontology bridge; itself ontology ingest |
| sprint-5-through-9-roadmap | lance-graph | only Sprint 7/9 touch splat/EWA/jc; rest compliance |
| 3DGS-render-depth-certification | ndarray | point-cloud fallback per KHR · *pending whole-read* |

## Hop 3 — ARCHIVE (superseded / fabrication-heavy)

| Plan | Repo | Note |
|---|---|---|
| cognitive-substrate-convergence-v1 | lance-graph | superseded by v3 |
| cognitive-substrate-convergence-v2 | lance-graph | superseded by v3 |
| Palette256-3DSB-PhiSpiral-attention-integration | lance-graph | **most fabrication-heavy** — 5+ markdown-only primitives, no deliverables |

## Out of scope — a separate program (real/shipping, not codec/3DGS)

- **graph-DB / ontology:** lance-graph-ontology-v5, sql-spo-ontology-bridge,
  lf-integration-mapping, unified-integration, compile-time-consumer-binding
- **business / SaaS / RBAC:** super-domain-rbac-tenancy, callcenter-membrane,
  supabase-subscriber, foundry-consumer-parity, foundry-roadmap-unified-smb-medcare,
  palantir-parity-cascade, q2-foundry-integration, ogit-cascade-supabase-callcenter,
  grammar-foundry-followup, elegant-herding-rocket

These reuse the shared VSA/CAM-PQ/bgz17 fingerprint substrate as a *storage
layer* but are not about splatting.

## Headline concept: C/HLOD

`C/HLOD` (from `3DGS-epiphany-roadmap`) = **Cesium HLOD (shipping, Apr 2026) +
error-certificate + query-relevance overlay.** The HLOD half is now a real,
shipped Cesium feature (`GaussianSplat3DTileContent`, back-to-front radix
`GaussianSplatSorter`); the contribution is the certified/queryable overlay,
which maps onto `tile.rs` radix-sort + `project.rs` conic + `raster.rs`
alpha-blend that ndarray already has.

## Forward program (7 PRs, 3 sessions — this doc is PR #1)

```
W1 (independent):  #1 triage [Gov] · #2 ada-3dtiles [Geo] · #3 EWA-SYRK bench [Kernel]
W2 (spine):        #4 pr-x12 fixes+archive [Geo] / evidence-policy [Gov] ── #5 canon specs (ASG·BGZ17·C/HLOD) [Gov]
W3 (impl):         #6 ErrorCertificate [Kernel] · #7 ASG lobe+recovery-gate [Kernel]   (both trail #5)
```

**Parked (deliberately unbuilt):** EWA-SYRK *backend* (gated behind #3's
measurement; likely a 3×3 pessimization) · PR-X12 tensor capstone build
(debt-laden until #4) · `phi_spiral.rs` (wrong manifold, superseded by #7).

## Scope boundary — not whole-read this pass

Honest boundary so the next session knows where to pick up:
`3DGS-certified-field-kernel-substrate` and `3DGS-render-depth-certification`
(ndarray, chat-added) were not in the 7-agent whole-read batches; their rows
above are provisional. The `pr-x12-*` perspective docs are graded by PR #200,
not re-read here.

## Convergence — comment surface

This PR is the comment surface. CodeRabbit, Codex, and the parallel sessions:
please flag (1) mis-graded rows, (2) missed fabrications, (3) corrections to
the correction-list. Hop-2/3 build work lands in PRs #4–#7.
