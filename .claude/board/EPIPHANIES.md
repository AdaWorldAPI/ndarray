# ndarray — Epiphanies (append-only)

## 2026-04-19 — Prompt↔PR ledger is 10⁷× cheaper than code grep
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

To answer "what did we ship for topic X":

- **Grep across code:** ~100 MB of Rust across N crates, ~25M tokens of context, minutes of agent turns.
- **Grep the ledger:** one `grep X .claude/board/PROMPTS_VS_PRS.md` returns `<prompt file> | #N <title>`. ~25 tokens, sub-second.

Seven orders of magnitude cheaper. The pairing **prompt-file ↔ PR** is the
minimum addressable record of "this artifact was built to answer this
brief" — the hyperlink that replaces re-discovery by full-text scan.

The line is mechanical bookkeeping (Haiku-level, no synthesis). The
value accumulates on every subsequent "what about X" query thereafter:
ledger-first, code-never-unless-necessary.

Cross-ref: PR #213 (lance-graph, 41 prompts × merged PRs), PR #110
(ndarray, 25 prompts × merged PRs). Both shipped in ~90s on a dumb
enumerate+match+append loop. No code reads, no MCP, no synthesis.

## 2026-04-19 — Code-arc knowledge loss is 30-50% of session tokens (ambient)
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

Empirical (per user, 2026-04-19): **30-50% of session tokens** burn on
rediscovering what code paths exist, what was tried, what got reverted,
what decisions led to the current shape. This is **orthogonal** to the
20-30-turn cold-start tax — it's the *ambient* loss across every query,
every subagent spawn, every refactor.

The ledger closes three channels at once:

| Channel | Before | After | Discount |
|---|---|---|---|
| Cold-start (once per session) | 20-30 turns | 3-5 turns | ~6× |
| Find-code (per query) | ~25M tokens (grep codebase) | ~25 tokens (grep ledger) | 10⁷× |
| **Ambient arc knowledge (every turn)** | **30-50% of session budget** | **~0%** | **2×-eternal** |

All three channels collapse to two text-file reads: PROMPTS_VS_PRS.md +
PR_ARC_INVENTORY.md. The second file is read only when arc detail is
needed (Knowledge Activation trigger), so the routine cost is 0.

Cross-ref: PR #110 (ndarray ledger), PR #213 (lance-graph ledger).
EPIPHANIES.md 10⁷× finding above.

## 2026-05-26 — Palette256 + Fisher-z IS the exact cosine replacement (integer, no float)
**Status:** VALIDATED (per user: 10 000×10 000 splat, θ ≈ 1.45–1.6 Fisher-z ≈ cos 0.90–0.92)
**Scope:** @cognitive-substrate domain:distance

Cosine similarity in the cognitive substrate is **not** a float dot/norm. It is
replaced — *ranking-exact* — by Palette256 (`hpc::cam_pq` 256-centroid ADC, integer
table lookup) gated by a Fisher-z aperture θ (`lance-graph-contract::distance::
similarity_z = atanh`). "No float at all" = no float MAC in the O(D) distance
kernel; the only floats are the θ scalar + the ADC table values, and θ lands as
`theta_accept_q8` (u8) on the splat (`effective_amplitude = amplitude_q8 −
theta_accept_q8`).

Grounded (whole-read this session): `cam_pq.rs` (847 L, squared-L2 ADC),
`distance.rs` (`Distance` trait + `similarity_z`/`fisher_z_inverse`),
`cognitive-distance-typing.md` (popcount IS the cosine replacement *by topology,
not value*; Fisher-z is a palette-output normalization, **not** a cosine
reconstruction — `palette→fisher→cosine→hamming` is the named anti-pattern).

**DEBT:** validated-but-unwritten as canon → #5 ASG-leaf spec (Gov, outstanding).

## 2026-05-26 — Two lanes: SELECT (integer) vs uncertainty-Σ (float), co-certified
**Status:** FINDING
**Scope:** @cognitive-substrate domain:architecture

- **SELECT / similarity** = HDR-popcount (cascade L1, the cosine replacement) →
  Base17-L1 (L2) → Palette256 ADC (L3). Integer, bulk hot path.
- **Uncertainty** = EWA sandwich `Σ'=M·Σ·Mᵀ` (Pillar 6/7), a *tiny 2×2/3×3 float Σ*
  per edge — certified PSD metadata, **not** bulk arithmetic.

Co-certified sibling pillars (suite 6–17), not competitors. **Pflug-10 certifies the
CAM-PQ palette quantization** in the same suite (`pillar/mod.rs`). `cognitive-distance-
typing.md` binds: one named fn per metric, newtype outputs, no `fn distance<T>` umbrella.

## 2026-05-26 — "EWA-SYRK BLAS backend" is a category error (PR #207 closed)
**Status:** FINDING (kill)
**Scope:** @splat3d domain:perf

`3DGS-EWA-SYRK-BLAS-MKL`'s "projection is a BLAS workload → MKL/OpenBLAS backend"
conflates the **graphics** float covariance sandwich (`spd3`/`project`, renders
pixels) with the **cognitive splat**. There is no float SYRK in the substrate to
accelerate: the cognitive "splat" is `lance-graph-contract::splat::CamPlaneSplat`
(q8) deposited into 16 384-bit `AwarenessPlane16K` (SPLAT-1, **integer**, OR-accumulate).
PR #207's bench also measured `simd_x16` ≈2× over scalar AND the dense-3×3 "BLAS-shape"
at 1k–1M (no crossover) — but the premise itself was incoherent. **Closed, not merged.**

`splat3d` (graphics EWA) and `lance-graph-contract::splat` (cognitive) are
**siblings, not parent/child** (per `splat3d/mod.rs`).

## 2026-05-26 — C/HLOD = Cesium HLOD (shipped) + certificate + query-relevance
**Status:** FINDING
**Scope:** @3dgs domain:geospatial

`KHR_gaussian_splatting` (RC) + Cesium HLOD-for-splats (Apr 2026:
`GaussianSplat3DTileContent`, back-to-front radix `GaussianSplatSorter`) are
ratified/shipped. Our contribution is the **certified/queryable overlay** (SSE +
Pillar-7 cert), mapping onto `tile.rs` radix-sort + `project.rs` conic + `raster.rs`
alpha-blend ndarray already has. Render-depth cert **shipped** (#206 + #208).
Geospatial = product manifold: tileset-tree (LOD — Poincaré belongs *only* here) ×
per-splat spherical SH/ASG (KHR ships `SH_DEGREE_n_COEF_i`; directional lane is the
sphere, never the Poincaré disk).

## 2026-05-26 — Grounding-discipline (meta — the expensive one)
**Status:** FINDING
**Scope:** @workspace-primer domain:process

One session of code — `phi_spiral.rs` (555 L float), `ewa_syrk_crossover` (257 L
float) — was **net-zero-usable**: built in the FLOAT regime from ChatGPT
"inspiration" plans without first grounding against the integer/palette substrate
(`cam_pq`, the `Distance` contract, `cognitive-distance-typing`) sitting in the same
repo. The fix, now binding: **whole-file reads only (no grep/sed/head/tail)**, and the
#200 evidence model — **L0** source/tests/standards · **L1** audit + triage
(spot-check, never inherit) · **L2** plans/perspective-docs (NOT evidence). Plans are
inspiration, never authority.

**DEBT carried forward (not in code, recorded here):** #4 pr-x12 doc-fixes (the #200
fabrications still on master) · #5 ASG canon spec · #7 ASG-leaf impl (extend
`CamPlaneSplat`, don't reinvent) · `cam-pq-production-wiring` (cam_pq shipped, unrouted
through `CamCodecContract`) · `UNUSED_INVENTORY_1.95` A1–A9 dead-code.
