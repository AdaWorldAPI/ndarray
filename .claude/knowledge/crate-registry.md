# Crate registry + naming convention (3dgs-* family)

> **AUTHORITATIVE policy** (a decision, not inspiration — unlike the plans triage).
> Anticipates a battery of splat/tiles crates; this is the index + the rules that
> keep them tracked and stop the battery from becoming empty shells.

## Posture (read this first)

**We do not *use* ArcGIS / Cesium.** No SDK, no service integration, no
API/wire compatibility, no runtime dependency. The relationship is exactly two
things, and nothing else:

- **Reinvent** — build native CAM SoA from the data's first principles. This is
  the goal.
- **Reverse-engineer** — one-way extraction of data out of their static files,
  rebuilt as CAM SoA. This is a *donkey-bridge*: transitional grounding,
  acceptable **only** when a native refactor isn't yet feasible, retired the
  moment it is.

Using-ArcGIS vs reinventing-and-reverse-engineering is a completely different
animal. We are the latter, never the former.

## Hard rules

1. **Single crate family: `3dgs-*`.** No `cesium-*`, no `ada-*` (deprecated).
   Naming is `3dgs-<role>` (`3dgs-tiles`, `3dgs-ewa-syrk`, future `3dgs-render`,
   `3dgs-codec`, …).

2. **CAM SoA is the mandatory representation.** CAM SoA =
   - **content-addressable** — addressed by `cam_pq` (CAM-PQ) codebook index, not by position;
   - **structure-of-arrays** — `simd_soa::MultiLaneColumn` / `cognitive-shader-driver::bindspace` columns / lane carriers;
   - **binary planes** — `AwarenessPlane16K` (16384-bit), q8 lanes, `#[repr(C)]`.

   Whatever a source contains is **reverse-engineered** into CAM SoA — and
   **ONLY** reverse-engineered: extract and rebuild, never depend on the source
   format as a library, never round-trip it, never emit it. No AoS, no
   source-native structs, no intermediate format survives past the import
   boundary.

   **Refactor > reverse-engineer.** The CAM SoA layout is *refactored natively*
   from first principles, not derived from the source's structure.
   Reverse-engineering is the donkey-bridge (see Posture) — a crutch, never the
   design. Never let an imported format's structure calcify into ours.

3. **NO JSON IN THE HOTPATH. EVER.** Restates the existing W1a invariant
   (`no serde`, "no JSON in types", `#[repr(C)]` — see
   `lance-graph-contract::splat`). JSON is permitted **only** at the cold,
   one-time, read-only *import boundary* (reverse-engineering someone else's
   file); it is rebuilt into CAM SoA immediately and **never enters the hotpath,
   is never persisted, and is never emitted**.

4. **Import adapters are read-once, one-way, output-CAM-SoA.** The 3D Tiles
   reader (in `3dgs-tiles`) reads `tileset.json` + `KHR_gaussian_splatting` glTF
   + `ESRI_crs` once, reverse-engineers into CAM SoA, and discards the source.
   No `cesium-*` family; nothing emits or serves their wire format.

5. **Rethink clause.** Our CAM SoA is canonical. An external format is
   reconsidered **only** on falsifiable proof it is strictly better — not by
   default, not for interop convenience.

6. **Creation gate.** A crate exists **only with ≥1 real consumer** (≥2 for a
   generic abstraction — the `certified-field-kernel-substrate` self-fence).
   Until then it stays a **module inside an existing crate**. A `3dgs-*` crate
   with no consumer is the code version of a markdown-only plan.

## Registry (seed — verified first-party only)

`status`: shipped / planned / module-until-earned. Only crates read first-party
this session carry a description; the rest are the catalogue checklist below.

| Crate / module | Repo | Family | Status | Purpose (verified) | Lane |
|---|---|---|---|---|---|
| `3dgs-tiles` | lance-graph | `3dgs-*` | planned | 3D Tiles reader. **Reverse-engineers** KHR_gaussian_splatting glTF + ESRI_crs → CAM SoA (donkey-bridge, transitional); JSON read-once at the cold boundary only | Geo |
| `3dgs-ewa-syrk` | ndarray (`benches/`) | `3dgs-*` | planned | EWA-SYRK crossover bench (`project_batch` vs cblas-batched sandwich) | Kernel |
| `splat3d` (module) | ndarray `src/hpc/` | — (pre-conv.) | shipped | CPU-SIMD forward renderer (`spd3`/`project`/`raster`/`sh`/`tile`/`gaussian`) | — |
| `cam_pq` (module) | ndarray `src/hpc/` | — | shipped | 6×256 CAM-PQ codebook — the **CAM** in CAM SoA | — |
| `simd_soa` (module) | ndarray `src/` | — | shipped | `MultiLaneColumn` lane carrier — the **SoA** in CAM SoA | — |
| `bgz17` | lance-graph | — (bare) | shipped | LOSSY golden-step recoverable sampling (BASE_DIM=17, GOLDEN_STEP=11); `ScalarCsr::spmv_min_plus` | — |
| `blasgraph` (module) | lance-graph `…/graph/` | — | shipped | **canonical bit-exact** 16384-bit GraphBLAS substrate (7 HDR semirings) | — |
| `jc` | lance-graph | — (bare) | shipped | 12-pillar proof-in-code (Pillar 10 `pflug`, 11 `hambly_lyons`) | — |
| `sigker` | lance-graph | — (bare) | shipped | path-signature kernels (`signature_truncated`, `signature_kernel_pde`) | — |
| `cognitive-shader-driver` | lance-graph | — (bare) | shipped | BindSpace **SoA** / mailbox / token_agreement / codec bridge | — |
| `lance-graph-contract` | lance-graph | — | shipped | SPLAT-1 CAM SoA contract types (`CamPlaneSplat`/`SplatPlaneSet`/`CamSplatCertificate`/`AwarenessPlane16K`) | — |

**Convention note:** existing bare/`lance-graph-*` crates predate the `3dgs-*`
convention and are **not force-renamed** (churn). The convention binds **new**
splat-pipeline crates. CAM SoA (rule 2) and the no-JSON-hotpath rule (rule 3)
bind **everyone** — old and new.

## Catalogue checklist (not yet first-party read — do not guess purposes)

lance-graph crates to add rows for, from each `lib.rs`, by lane owners:
`bge-m3`, `bgz-tensor`, `causal-edge`, `deepnsm`, `highheelbgz`, `holograph`,
`lance-graph-archetype`, `lance-graph-benches`, `lance-graph-callcenter`,
`lance-graph-catalog`, `lance-graph-codec-research`, `lance-graph-cognitive`,
`lance-graph-consumer-conformance`, `lance-graph-ontology`, `lance-graph-osint`,
`lance-graph-planner`, `lance-graph-python`, `lance-graph-rbac`,
`lance-graph-supervisor`, `lance-graph` (main), `learning`, `neural-debug`,
`p64-bridge`, `reader-lm`, `sigma-tier-router`, `thinking-engine`.

## Why this exists (governance parallel)

Same shape as #200 Tier-4, one layer down: docs proliferated → needed "2
canonical sources + perspective-docs-aren't-evidence"; crates will proliferate →
need "single `3dgs-*` family + registry + creation gate + CAM SoA as the one
representation." Naming + registry *track* the battery; the creation gate
*prevents* the part that shouldn't exist; **CAM SoA + no-JSON-hotpath keep every
crate speaking the same content-addressed SoA substrate instead of a battery of
private formats — and the reinvent/reverse-engineer posture keeps us from ever
becoming an ArcGIS/Cesium consumer.**
