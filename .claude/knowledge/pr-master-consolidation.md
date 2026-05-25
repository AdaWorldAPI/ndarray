# Master Consolidation — ndarray as the universal CPU-shape-aware substrate

> READ BY: every agent in the AdaWorldAPI stack
> (savant-architect, l3-strategist, cascade-architect, splat3d-architect,
> cognitive-architect, jc-architect, codec-architect, ogit-architect,
> training-architect, arm-neon-specialist, sentinel-qa, product-engineer,
> truth-architect, vector-synthesis).
>
> Status: strategic v1 — drafted 2026-05-18.
>
> **This doc inverts the existing repo split.** Previously: ndarray was "CPU
> primitives", lance-graph was "cognitive graph", jc was "self-certifying
> math probes", lance-graph-ontology was "OGIT consumer". After this
> consolidation: ndarray becomes the universal CPU-shape-aware substrate
> for ALL math, ALL cascades, ALL codecs, ALL ontology hydration. The
> other crates become thin domain orchestrators.

## Strategic frame

The architectural truth surfaced in this session:

**Spd3 is not generic — it's shaped for AVX-512.** 32-byte align fits one half-zmm. The 6 + 8 layout reads as one 256-bit FMA chain. `sandwich_x16` maps to 16 packed FMA pipelines = 256 mul-adds per cycle on Zen4/Sapphire Rapids.

**CascadeAddr is not generic — it's shaped for cache.** 4 bytes total, 16 addresses per cache line, parent/children extraction via single shift-mask.

**GaussianBatch is not generic — it's shaped for SoA SIMD.** Every field 64-byte aligned, padded to PREFERRED_F32_LANES so no scalar tails.

**These structs encode CPU throughput limits in their byte layouts.** The shape IS the CPU contract.

What IS agnostic isn't the data — it's the algorithms over the data:
- `sandwich(M, N)` is agnostic; it just happens to run best on a CPU-shaped Spd3
- Splat cascade is agnostic; it runs best on cache-shaped CascadeAddr
- NARS revision is agnostic; runs best on f32 SoA cells
- x265 RDO is agnostic; runs best on quad-tree CTU partition (which IS the same shape as splat cascade)

**The shape-aware substrate IS the CPU contract. The agnostic algorithm is the math contract. Both live in the same crate.** That crate is `ndarray::hpc::*`.

## The 10-submodule layout

```
ndarray::hpc::
├── simd/              ← SIMD polyfill (shipped) — F32x16 / U64x8 / NEON / AMX
├── blas_level{1,2,3}/ ← BLAS (shipped)
├── lapack/            ← LAPACK FFI (shipped)
├── linalg/            ← PR-X10 — MatN / Quat / SVD / polar / mat_exp / Conv / Attention
├── blocked_grid/      ← PR-X3 (shipped) — generic 2-D block-padded grid
├── splat3d/           ← shipped — Spd3 + Gaussian + EWA + tile bin + raster
├── splat4d/           ← PR-X4 — temporal sandwich + L1-L4 cascade
├── cognitive/         ← PR-X9 — BasinCodebook + LazyBlockedGrid + NARS revise
├── pillar/            ← PR-X11 — jc-style certification probes (Pillar-6..N)
├── codec/             ← PR-X12 — x265-style CTU/CU + skip/merge/delta/escape + RDO + ANS
└── ogit_bridge/       ← PR-X13 — embedded TTL → in-memory schema (replaces lance-graph-ontology hop)
```

11 submodules, each feature-gated. Default build ships only `simd + blas + lapack + linalg`. Cognitive shader work opts in via `--features cognitive,codec,ogit_bridge,splat4d,pillar`.

## What moves where

| Previously | New home | Replaces / supersedes |
|---|---|---|
| `lance-graph/crates/jc/src/ewa_sandwich.rs` | `ndarray::hpc::pillar::ewa_sandwich_2d` | Pillar-6 — Spd2 |
| `lance-graph/crates/jc/src/ewa_sandwich_3d.rs` | `ndarray::hpc::pillar::ewa_sandwich_3d` | Pillar-7 — Spd3 (was duplicated with splat3d) |
| `lance-graph/crates/jc/src/koestenberger.rs` | `ndarray::hpc::pillar::koestenberger` | another Spd3 copy |
| `lance-graph/crates/jc/src/pflug.rs` | `ndarray::hpc::pillar::pflug` + `ndarray::hpc::linalg::wasserstein` | Pillar-10 |
| (proposed) Pillar-8 temporal_sandwich | `ndarray::hpc::pillar::temporal_sandwich` | drafted in splat4d cascade PR 1 |
| (proposed) signature transform Pillar-11 | `ndarray::hpc::pillar::signature` | Hambly-Lyons |
| (proposed) Pillar-9 Cov16384 | `ndarray::hpc::pillar::cov_high_d` | Düker-Zoubouloglou CLT |
| `lance-graph-ontology` bridge pattern | `ndarray::hpc::ogit_bridge` with per-namespace bridges | dispenses with the hop |
| (no source today) x265 RDO / CTU / CABAC | `ndarray::hpc::codec::{ctu,rdo,ans}` | new |
| `splat3d::sh.rs` (deg-3 only) | `ndarray::hpc::linalg::sh` (deg 0..=7) | PR-X10 |
| `splat3d::Spd3` | `ndarray::hpc::linalg::Spd3` (type alias preserves splat3d::Spd3) | PR-X10 |
| `lance-graph::*` inline RMSNorm / SiLU / RoPE / attention | `ndarray::hpc::linalg::{norm,activations_ext,rope,attention}` | PR-X10 |

## The invariant that breaks (intentionally)

**Old invariant** (from splat3d sprint, doc reference: `splat3d_sprint_prompt.md:38`):
> "10. **The cognitive `splat.rs` is sacred.** ... The graphics splat is a NEW module that happens to share *math* (EWA-sandwich, SPD push-forward) with the cognitive splat. They are siblings, not parent/child."

This invariant said: contract types stay contract types, math lives in two places (graphics + cognitive), no cross-coupling.

**Also breaks**: `jc`'s zero-dep on ndarray rule. Was sound when jc was 2 files; with 4-way Spd2/Spd3 duplication appearing, the cost of self-certification exceeds the benefit.

**Replacement invariant** (binding from PR-X10 onward):

> **12. Certification is about determinism and inspectability of the math, not about repo separation.** `ndarray::hpc::pillar::prove_pillar_7()` calls `ndarray::hpc::linalg::eig_sym_3()`. The certification holds because:
> - the probe SEED is documented (`0x_EDA_5A_DC_5A_DD` style constants)
> - the implementation is git-tracked and inspectable
> - the bench output is committed to `RESULTS.md`
> - the PASS gate (PSD rate ≥ 0.999, log-norm concentration) is verifiable from probe output alone
>
> The math contract is preserved by the probe's PASS gate, not by repo isolation.

**Old invariant** (from jc design):
> "lance-graph-contract/src/splat.rs is sacred."

**Stays in force.** The cognitive contract type (q8-only, no floats) lives in `lance-graph-contract` and is never modified. `ndarray::hpc::cognitive::*` CONSUMES the contract types but does not modify them.

## Six-sprint plan with dependencies

```
                    ┌─────────────────────┐
                    │ PR-X10 (linalg-core)│   ← foundation, max-fan-out 12 workers
                    │ A1 MatN → A2-A12 ∥  │   ← 2 wks parallel
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────┐
        │ PR-X11   │   │ PR-X13       │   │ PR-X12   │
        │ jc → pi- │   │ ogit_bridge  │   │ codec    │
        │ llar/*   │   │ (embedded TTL)│   │ (x265)   │
        │ 6 workers│   │ 4 workers    │   │ 8 workers│
        │ 1 wk     │   │ 1 wk         │   │ 2 wks    │
        └─────┬────┘   └──────┬───────┘   └─────┬────┘
              │               │                 │
              │  parallel     │   parallel      │
              │  (no overlap) │                 │
              ▼               ▼                 ▼
                       ┌──────────┐
                       │ PR-X4    │   ← splat cascade onto BlockedGrid
                       │ 5 workers│   ← needs splat3d (shipped) + linalg
                       │ 1 wk     │
                       └─────┬────┘
                             │
                             ▼
                       ┌──────────┐
                       │ PR-X9    │   ← basin-codebook lazy storage
                       │ 6 workers│   ← needs codec (PR-X12) + ogit_bridge (PR-X13)
                       │ 1.5 wks  │
                       └─────┬────┘
                             │
                             ▼
                       ┌──────────┐
                       │ Integ.   │   ← e2e demo + bench + docs
                       │ 3 workers│
                       │ 1 wk     │
                       └──────────┘
```

**Total: 8 weeks** if concurrent sprints honor their independence. **44 sprint workers** + 6 coordinators (one per sprint) + savants. The 12-agent cadence per sprint matches the user's preferred protocol.

### Concrete sprint schedule

| Weeks | Sprints running concurrently |
|---|---|
| W1-W2 | **PR-X10** (linalg-core foundation) |
| W3 | **PR-X11** (jc consolidation) + **PR-X13** (OGIT bridge) in parallel |
| W4-W5 | **PR-X12** (codec) + **PR-X4** (splat cascade) in parallel |
| W6-W7 | **PR-X9** (basin-codebook) — needs PR-X12 + PR-X13 |
| W8 | **Integration sprint** — e2e demo, docs, bench, recording |

## Phase 1 / Phase 2 protocol acknowledgment

User's confirmed cadence (from `12 agenten + 1 Koordinator` proposal):

**Phase 1 (DESIGN) — Protocol B:**
```
plan → savant review → correct → ...
```
**WE ARE HERE.** All 6 sprints have design docs drafted:
- `pr-x3-cognitive-grid-design.md` (shipped as PR #158)
- `pr-x4-design.md` (drafted on `claude/pr-x4-splat-cascade-design`)
- `pr-x9-design.md` (drafted; OGIT correction applied)
- `pr-z1-ogit-cognitive-bootstrap.md` (drafted; superseded by PR-X13 below)
- `pr-arithmetic-inventory.md` (drafted)
- `pr-x10-linalg-core-design.md` (drafted; max-fan-out 12 workers)
- `pr-master-consolidation.md` — THIS DOC
- `pr-x11-jc-consolidation-design.md` — drafted alongside
- `pr-x12-codec-x265-design.md` — drafted alongside
- `pr-x13-ogit-bridge-design.md` — drafted alongside (subsumes PR-Z1)

**Next step**: spawn the **joint plan-review savant** on ALL 10 docs at once. It rules on:
- The OGIT integration path (PR-X13 embedded TTL bundle vs PR-Z1 + PR-Z2 inter-repo coordination)
- The invariant inversion (does breaking jc's zero-dep rule survive scrutiny)
- The codec choice (CABAC vs ANS in PR-X12)
- The 7-question lists in each sprint doc
- The cross-sprint ordering above (concurrent vs sequential where there's any ambiguity)

Output: a single verdict file ruling on the architectural surface.

**Phase 2 (IMPLEMENTATION) — Protocol A:**
```
preflight (commented-out Rust skeleton) →
  → parallel-savant review fan-out (data-flow, layering, distance-typing,
    SAFETY-claim, naming-collision, test-coverage) →
  → workers fill bodies →
  → fix P0 → review → commit → repeat
```

**Coming next, per sprint.** Each sprint kickoff:
1. One agent writes preflight Rust skeleton for that sprint (all impl blocks `unimplemented!()`, all types stubbed)
2. 6 parallel specialist savants review the skeleton (different concerns, same skeleton, no collision)
3. Sprint workers fill bodies (file-scoped, parallel where dependency graph permits)
4. Codex P0 audit on combined diff
5. Coordinator fixes P0s
6. P2 savant pre-merge review
7. Merge, integration test, repeat for next sprint

Why Protocol A in Phase 2: the PR-X3 sprint's post-merge `GridBlockMut::data: &'a mut [T]` UB would have been caught at preflight by the SAFETY-claim savant. Protocol A catches latent UB earlier than codex audit at end-of-sprint.

## Deprecation timeline

| Sprint | Crate effects | Deprecation action |
|---|---|---|
| PR-X10 lands | `ndarray::hpc::linalg::*` becomes canonical | none yet (additive) |
| PR-X11 lands | `lance-graph/crates/jc/src/{ewa_sandwich,ewa_sandwich_3d,koestenberger,pflug}.rs` deprecated | `#[deprecated(since="0.X", note="moved to ndarray::hpc::pillar")]` + 1-cycle transition |
| jc consumers migrate | jc becomes a thin probe-runner that imports `ndarray::hpc::pillar::*` | follow-on PR after PR-X11 |
| PR-X13 lands | `lance-graph/crates/lance-graph-ontology/` becomes thin Bardioc REST client | bridges move to `ndarray::hpc::ogit_bridge::*` |
| PR-X12 lands | x265-style codec available crate-wide | no deprecations (new code only) |
| PR-X4 lands | splat3d/tile.rs becomes a backward-compat shim for the new `ndarray::hpc::splat4d::cascade` | shim deprecated after splat3d::raster.rs migrates |
| PR-X9 lands | lazy basin-codebook storage available | none (new code only) |

## Feature-flag matrix

```toml
# Default minimal build
default = ["std", "linalg"]

# Add submodules as needed
splat3d         = ["dep:..."]                          # shipped
splat4d         = ["splat3d", "linalg"]
blocked_grid    = ["std"]
linalg          = ["std"]                              # foundation
pillar          = ["linalg"]                           # jc consolidation
codec           = ["std", "blocked_grid"]              # x265-style
ogit_bridge     = ["std"]                              # embedded TTL
cognitive       = ["blocked_grid", "linalg", "codec", "ogit_bridge"]

# Aggregates
cognitive_full  = ["cognitive", "splat4d", "pillar"]   # the full cognitive shader stack
medical         = ["splat4d", "blocked_grid"]          # the medical imaging atlas (skeleton-anchored)
training        = ["linalg", "splat3d"]                # splat3d backward / training sprint
```

Default builds stay small. Consumer crates pick the cocktail they need. No mandatory cognitive surface for splat3d consumers; no mandatory splat surface for jc consumers.

## What the joint plan-review savant rules on

| Decision | Options | Lean |
|---|---|---|
| OGIT integration path | (a) sequential OGIT/Cognitive bootstrap → lance-graph CognitiveBridge → PR-X9 / (b) parallel with stubs / (c) embedded TTL bundle in ndarray | **(c)** — subsumed by PR-X13 |
| jc zero-dep invariant | keep / break | **break** — invariant 12 replaces it |
| Codec entropy coder | CABAC (industry-standard, complex) / ANS (simpler, cache-friendlier) | **ANS** for v1, CABAC follow-on if compression ratio insufficient |
| Cross-sprint ordering | strict sequential / aggressive concurrent | **concurrent** per dependency graph above |
| Worker decomposition (per sprint) | 5-10 / 12 / more | **12 per sprint** (the "12 agenten" cadence) |
| Phase 2 protocol | Protocol A preflight / Protocol B direct sprint | **Protocol A** for all implementation phases |
| Backward compat for splat3d consumers | full / partial / break | **full** via type aliases in PR-X10 |
| jc deprecation cycle | 0-cycle / 1-cycle / 2-cycle | **1-cycle** (one release of `#[deprecated]` before removal) |
| Pillar count in PR-X11 | 4 (Pillar-6,7,10 + signature) / 5 (+ Pillar-8 temporal) / 6 (+ Pillar-9 high-D) | **6** — pre-stage Pillar-8 and Pillar-9 for the splat4d temporal sandwich |
| codec mode count | 2 (skip/explicit) / 4 (skip/merge/delta/escape) / 6 (+ multi-merge variants) | **4** — matches x265 medium-preset complexity |

## Open questions (joint savant ruling)

1. **Does PR-X10's `Quat` belong in `linalg` or in a new `geom` submodule?** Lean: in `linalg`. Quaternions are linalg primitives (4-vec on the 3-sphere, multiplication is a Lie group op).

2. **Should `ndarray::hpc::pillar::*` re-export jc's old API verbatim for the deprecation cycle?** Lean: yes — `pub use crate::hpc::linalg::Spd3 as Spd3;` etc., so jc's existing consumers compile unchanged.

3. **Does `ndarray::hpc::codec::*` need a streaming API (frame-by-frame ingest) or only a batch API (whole-frame encode)?** Lean: both — batch for v1, streaming added in PR-X12.1.

4. **Does `ndarray::hpc::ogit_bridge::*` ship the OGIT TTL files as `include_bytes!` (compiled into the binary) or as build-time-cached files?** Lean: `include_bytes!` for v1 (zero-startup, ~50 KB compressed), build-time-cached as opt-in for larger ontologies.

5. **Phase 2 Protocol A — does each sprint share the SAME 6 specialist savants, or does each sprint get its own set?** Lean: same 6 (reduces savant context-switch overhead); the savants are stateless w.r.t. each sprint.

6. **Where does AriGraph / SPO-bundle / NARS-engine orchestration live AFTER this consolidation?** Lean: stays in `lance-graph/*` — those are domain orchestrators, not substrate math. `lance-graph::cognitive::*` consumes `ndarray::hpc::cognitive::*`.

7. **Does the master consolidation imply renaming `ndarray` → `ada-substrate` or similar?** Lean: NO — `ndarray` is established, downstream-stable, has CI history. The name's slightly misleading after this consolidation (it's no longer just N-dimensional arrays) but the rename cost exceeds the clarity benefit.

## Sprint-by-sprint headlines

(Full design docs adjacent on the branch.)

- **PR-X10** — 12-worker max-fan-out linalg foundation. A1 MatN → A2-A12 all parallel. The Spd3 closed-form fast path and Jacobi general-N coexist (invariant 12).
- **PR-X11** — jc Spd2/Spd3/Wasserstein/signature/cov_high_d consolidation. 6 workers. jc becomes a probe-runner; math lives in `ndarray::hpc::pillar::*`.
- **PR-X12** — x265-style codec for cognitive basin-codebook compression. CTU quad-tree (4 levels matching splat4d cascade), skip/merge/delta/escape modes, λ-RDO, ANS entropy coder. 8 workers.
- **PR-X13** — OGIT embedded TTL bundle + Cognitive namespace bridge. 4 workers. Replaces the 3-repo coordination of PR-Z1 + PR-Z2 + PR-X9; one self-contained ndarray sprint.

## Done criteria (master consolidation)

Master consolidation is "done" when:
- All 6 sprints land per the schedule above (~8 weeks)
- `ndarray::hpc::*` 10-submodule layout is the canonical structure
- jc deprecated 1 cycle and removed in cycle N+2
- lance-graph-ontology becomes thin Bardioc REST client (or deprecated entirely if no live-schema use case)
- `lance-graph` consumers migrate to `ndarray::hpc::cognitive::*` as the substrate
- e2e demo on a Zen4/Sapphire Rapids 8-core: cognitive cascade ingesting at 30+ fps, splat atlas rendering at 50+ fps, all CPU
- Codex P0 audits across all 6 sprints pass 0 P0s
- P2 savant pre-merge reviews across all 6 sprints deliver SHIP verdicts
- Final integration sprint produces 30-second screen recording: cognitive shader stack end-to-end on CPU, no GPU

## Token-reset safety notes

If you're picking up after a token reset:
1. Read this doc first.
2. Read the 9 sibling design docs on `claude/pr-x4-splat-cascade-design` branch.
3. The conversation context: after PR #158 (PR-X3 BlockedGrid) merged on 2026-05-18, the user pushed for maximal consolidation — `ndarray::hpc::*` becomes the universal CPU-shape-aware substrate. Five additional sprints proposed: PR-X10 (linalg foundation), PR-X11 (jc consolidation), PR-X12 (x265 codec), PR-X13 (OGIT bridge), and the cognitive layer (PR-X4 + PR-X9 already drafted). Invariant 12 replaces the jc zero-dep rule.
4. The Phase 1 / Phase 2 protocol is explicit: Phase 1 (Protocol B — plan → savant review → correct) is happening now; Phase 2 (Protocol A — preflight Rust skeleton → parallel-savant fan-out → workers fill bodies) starts after the joint savant verdict.
5. The 12-worker max-fan-out shape (PR-X10's A1 MatN → A2-A12 parallel) is the gold-standard for sprint composition; subsequent sprints follow the same shape where the dependency graph permits.
6. The deprecation timeline is staged: PR-X11 marks jc files `#[deprecated]` for 1 cycle; PR-X13 supersedes lance-graph-ontology bridge pattern; both fully removed in cycle N+2.
