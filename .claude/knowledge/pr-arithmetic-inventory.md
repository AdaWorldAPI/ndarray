# Arithmetic Inventory — splat3d / splat4d / cognitive shader stack

> READ BY: all agents touching `crate::hpc::*` math kernels
> (savant-architect, splat3d-architect, cascade-architect, cognitive-architect,
> arm-neon-specialist, sentinel-qa, truth-architect, vector-synthesis).
>
> Status: review v1 — drafted 2026-05-18 in response to the three uploaded
> sprint prompts (`splat3d_sprint_prompt.md`, `splat4d_cascade_sprint.md`,
> `splat4d_skeleton_anchored_sprint.md`).
>
> Purpose: enumerate every arithmetic primitive required by the
> splat3d → splat4d → cognitive-shader stack, tag each `shipped | drafted |
> gap`, flag precision-class concerns, and identify the ordering blockers
> for the joint sprint.
>
> Parallel docs:
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid substrate (shipped, PR #158)
> - `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade onto BlockedGrid
> - `.claude/knowledge/pr-x9-design.md` — lazy basin-codebook storage
> - `.claude/knowledge/pr-z1-ogit-cognitive-bootstrap.md` — OGIT Cognitive namespace bootstrap

## TL;DR

| Layer | Primitives | Status |
|---|---|---|
| **L0 SPD substrate** (Pillar-6/7) | Smith-1961 eig, Σ^t, sandwich, Spd3, sandwich_x16 | ✅ shipped (splat3d PR #153) |
| **L1 Projection / EWA** | W·Σ·Wᵀ, J·Σ·Jᵀ, 2D conic inverse, 3σ radius | ✅ shipped |
| **L2 SH eval** (deg-3) | 16-basis × 3-channel, Inria convention | ✅ shipped |
| **L3 Tile bin + rasterize** | radix sort, Mahalanobis², fast_exp_x16, alpha compose | ✅ shipped |
| **L4 Cascade addressing** | CascadeAddr bit-pack, parent/children, **Hilbert-3D** | ⬜ **GAP** — Hilbert-3D not specified anywhere |
| **L5 Gaussian-mixture moment-match** | Σ_parent = (1/n)·Σ(Σ_i + Δμ·Δμᵀ) | ⬜ drafted in splat4d cascade PR 1 |
| **L6 Pillar-8 temporal sandwich** | Σ_{t+1} = M·Σ_t·Mᵀ with M = sqrt(σ_temporal) | ⬜ drafted in splat4d cascade |
| **L7 Mesh→splat fitting** | PCA over vertex positions, fiber-direction Σ alignment | ⬜ drafted in skeleton-anchored PR 2a/2b |
| **L8 Cognitive overlay** | INT4×N dot, NARS revision, basin XOR-popcount, CTU modes | ⬜⬜ **GAP** — not in any splat doc |

**Five concrete gaps gate the joint sprint:**
1. Hilbert-3D curve encode/decode (~16 ops/coord, needed by L4 cascade addressing)
2. INT4×32 packed dot product (needed by cognitive cell signature)
3. NARS truth-revision kernel + precision class (replaces alpha-compositing in W7)
4. x265-style CTU mode encoder (skip/merge/delta/escape — needed by PR-X9 lazy storage)
5. fast_exp_x16 precision audit (3% relative error — OK for alpha, **suspect for NARS confidence**)

## Layer-by-layer detail

### L0 — SPD substrate (Pillar-6 2D / Pillar-7 3D)

**Shipped in splat3d PR #153**. The single most-reused arithmetic primitive in the stack — the temporal-sandwich (Pillar-8), the splat-cascade aggregate-up, and the EWA projection ALL reduce to `M·Σ·Mᵀ` on Spd3. No new SPD machinery needed downstream; only new semantic interpretations of M.

```
Smith-1961 closed-form eigendecomp:    O(1) per matrix, ~30 ops, scalar
Σ^t = V · diag(λᵢᵗ) · Vᵀ:               O(eig) + 3 pow, scalar inner
sandwich(M, N): M·N·Mᵀ symmetric:       21 mul + 12 add for 3×3 sym
sandwich_x16:                            AVX-512 batched, 10× over scalar
from_scale_quat: Σ = R·diag(s²)·Rᵀ:     9 mul + 6 add (R) + 9 mul (sandwich)
is_spd, frobenius², det, log_spd:        constant-fold scalar
```

**Precision class: EXACT** for all downstream compute. No approximations. `Spd3::eig` uses branchless acos clamp + Gram-Schmidt orthonormalization on near-degenerate covariances.

### L1 — Projection + EWA (3D world → 2D conic)

**Shipped in splat3d PR #153** (the math heat of PR 3).

```
μ_cam = V · μ_world:                     16 FMA/gaussian (3×4 mat-vec)
Frustum cull (depth + AABB):             F32x16 mask, branchless
J = [[fx/z, 0, -fx·x/z²], [0, fy/z, -fy·y/z²]]:  6 div by z (vrcp14ps)
Σ_image = J · W · Σ_world · Wᵀ · Jᵀ:    ~50 FMA/gaussian — THE hottest single op
2D conic = Σ_image⁻¹:                    4 mul + 1 div + 3 mul
3σ radius = 3·sqrt(λ_max(Σ_image)):     closed-form 2×2 root + sqrt
View dir = normalize(μ - cam_pos):       3-vec norm (vrsqrt14ps)
```

**Precision class: FAST OK** for graphics (Inria parity SSIM ≥ 0.97 with vrcp14/vrsqrt14). **VERIFY** if reused for cognitive distance — the perspective division ε accumulates over the cascade.

### L2 — Spherical harmonics evaluation (deg 0–3)

**Shipped in splat3d PR #153**.

```
SH_C0, SH_C1, SH_C2[5], SH_C3[7]:        baked f32 constants
sh_eval_deg3(sh[48], d):                 17 mul-add per channel × 3 = 51 FMA per gaussian
sh_eval_deg3_x16:                        AVX-512 batched, ~6× over scalar
Inria convention: (v + 0.5).clamp(0, 1)
```

**Cognitive reframe**: same math gives "appearance under different cognitive inquiries" — `vocab_idx × thinking_style` projection per PR-X4's `SplatCell`. **Drafted but not shipped** in cognitive form.

### L3 — Tile binner + rasterizer (alpha-compositing)

**Shipped in splat3d PR #153**. Three precision-class flags:

```
Mahalanobis² power = -0.5·(ca·dx² + 2·cb·dx·dy + cc·dy²):  4 FMA/pixel/splat — EXACT
fast_exp_x16 (Schraudolph 1999):                          1 cast + 1 FMA — **3% rel err**
alpha = min(0.99, op · fast_exp(power)):                  EXACT after exp
T·alpha compose, T *= (1−α):                              4 FMA/pixel/splat — EXACT
Saturation early-exit T < 1e-4:                           single compare
```

**Precision class: FAST OK for alpha-compositing. FLAG: fast_exp's 3% relative error is graphics-suitable but breaks NARS truth-revision convergence** — see L8 gap analysis.

Radix sort on packed u64 (tile_id << 32 | depth_bits) → 2M instances in ≤8 ms. **EXACT** by construction (integer key).

### L4 — Cascade addressing (splat4d cascade PR 1) — **PARTIAL GAP**

```
CascadeAddr::level(l): (bits >> (l*4)) & 0xF:            1 shift + 1 and
parent(): bits & !0xF000:                                 1 and
children(): [parent | (i<<12) for i in 0..16]:           16 ors
from_position(p, bbox, level): **Hilbert-3D encode**     ⬜ GAP
to_position_center(addr, bbox): **Hilbert-3D decode**    ⬜ GAP
```

**The Hilbert-3D curve at 4 bits per axis per level isn't specified in any of the three docs.** Splat4d cascade PR 1 says "Hilbert-3D order at L4 for cache locality" but doesn't sketch the math. We need:
- `position → (l0, l1, l2, l3)` nibble path: ~16 conditional swaps per coordinate per level (Butz's algorithm)
- Inverse decode: same shape
- Possibly a precomputed 16-entry rotation table for Gray-code-order branching

**Precision class: EXACT** (integer-only encode/decode; no float ops). Estimated ~64 ops per address conversion.

### L5 — Gaussian-mixture moment-match (cascade aggregate-up) — **DRAFTED**

```
Σ_parent = (1/n) · Σᵢ(Σᵢ + Δμᵢ · Δμᵢᵀ)  where Δμᵢ = μᵢ − μ_parent
μ_parent = (1/n) · Σᵢ μᵢ
opacity_parent = mean or alpha-composite of children
```

For N=16 children at each tier:
```
16 outer products Δμᵢ·Δμᵢᵀ:              16 × 6 mul = 96 mul/parent (3×3 sym)
16 Spd3 additions:                       16 × 6 = 96 add/parent
1 scalar division by 16:                 1 div (or shift)
```

≈ 200 ops/parent at L3, ≈ 16 × 200 = 3200 ops/parent at L2, etc. Total cascade aggregate-up across all 65,536 leaves: ~5 M ops, well within frame budget.

**Drafted in splat4d cascade PR 1.** Same math is PR-X4's `compose_cascade` operator from our prior conversation. **EXACT** precision class — no approximations.

### L6 — Pillar-8 temporal sandwich (Σ_{t+1} = M·Σ_t·Mᵀ) — **DRAFTED**

**Reuses Spd3 + sandwich_x16. Only the σ_temporal table is new.** Three motion bands stratify:

| Band | Frequency | Amplitude | σ_temporal (Frobenius) |
|---|---|---|---|
| Cardiac | ~6 Hz | ~5 mm | needs literature value |
| Respiratory | ~0.3 Hz | ~20 mm | needs literature value |
| Micro-motion | ~120 Hz | ~0.1 mm | needs literature value |

**Cross-cutting research item #5 from splat4d cascade.** PASS gate is arbitrary until echocardiography literature pins these down.

**For the cognitive shader path**: σ_temporal represents NARS truth-confidence decay across frames, NOT physical motion. Same math, different calibration. Both interpretations share the substrate.

### L7 — Mesh → splat fitting (skeleton-anchored PR 2a/2b) — **DRAFTED**

```
Per-bone-segment PCA:
  μ_seg = (1/n) · Σ vᵢ                  Welford accumulator, ~3n FMA
  Σ_seg = (1/n) · Σ (vᵢ − μ)(vᵢ − μ)ᵀ   single-pass via Welford, ~6n FMA
Quaternion from rotation matrix:        ~30 ops (Shepperd's method, sign-tracking)
Fiber-direction Σ alignment (muscles):
  axis = normalize(insertion − origo)   3-vec normalize
  Σ_major = axis · axisᵀ · σ_length²    3 mul + outer product
  Σ_minor from cross_section_mm²        2 scalar fills
```

**Precision class: EXACT** for the offline mesh→splat conversion (cached in build.rs); doesn't run on the hot path.

### L8 — Cognitive overlay — **THE GAP CATEGORY**

The five primitives that the three splat docs DON'T cover but our PR-X4/X9/Z1 docs require:

#### G1 — INT4×N packed dot product

For thinking-style (32-dim INT4) and qualia (16-dim INT4) cell signatures.

```
thinking_dot(a: [u8; 16], b: [u8; 16]) -> i32:
  unpack u4 pairs → i8 (table lookup or shift-and-mask)
  vpdpbusd-style i8×u8 → i32 accumulator (AVX-512 VNNI / NEON dotprod)
```

**Hardware path:**
- AVX-512 VNNI `vpdpbusd`: i8 × u8 → i32 accumulator, 64 ops per instruction → 2 instructions for 32-dim INT4
- ARM NEON `sdot`: i8.4 dot u8.4 → i32, 4 ops per instruction → 8 instructions for 32-dim
- AMX BF16 tile op handles INT8 not INT4 directly — would need software unpacking
- Scalar fallback: 32-way unrolled

**Precision class: EXACT** (integer dot product). **GAP** — not in any sprint doc; needs to land before PR-X7 typed cell-DSL.

#### G2 — NARS truth-revision kernel

Replaces alpha-compositing in W7's PR-X4 closure swap.

```
revise(T1, T2) = (
  freq:  (f1·c1 + f2·c2) / (c1 + c2),
  conf:  (c1 + c2) / (c1 + c2 + k),     where k = 1 by NARS convention
)
```

4 FMA + 1 div per cell pair. **Precision class: NEEDS AUDIT**. The confidence numerator/denominator near c1+c2≈0 is the precision risk; `vrcp14ps` (14-bit mantissa, used by splat3d for perspective division) is likely insufficient — Newton-Raphson refinement (one step → 28 bits) probably required.

**GAP** — designed in PR-X4 §"W7 closure swap" but not implemented.

#### G3 — Basin XOR-popcount (OGIT-schema-driven)

Per-cell basin matching against the 4096-atom CAM codebook, gated by the OGIT family bitmap.

```
For cell with edge u64:
  family = ogit_schema.family_of(cell.basin_hint)        // O(1)
  for basin_idx in family_bitmap.iter_ones():            // ~16-64 candidates
    delta = cell.edge XOR codebook[basin_idx].edge       // 1 XOR
    dist = delta.count_ones()                            // 1 popcnt
    track min dist + idx
  return best_basin_idx
```

**Hardware**: `popcnt` is single-cycle on AVX-512 (`vpopcntq`) and on NEON (`cnt` + `addv`). **EXACT** by construction. **GAP** — drafted in PR-X9 §"Encoding modes" but not implemented; needs OGIT-rs hydrate path (PR-Z1 + PR-Z2).

#### G4 — x265-style CTU mode encoder (skip/merge/delta/escape)

The per-cell rate-distortion loop that picks encoding mode in PR-X9's lazy storage.

```
For each cell (basin_idx, true_value):
  skip_cost  = 0 if true_value == basin else INF
  merge_cost = 2 if delta(neighbor) decodes to true_value within ε else INF
  delta_cost = 8 + |true_value - basin - decode(quantized_delta)|·λ
  escape_cost = 64 (always available)

  pick min(skip, merge, delta, escape)
```

Per-cell: ~4 compares + 1 subtract + 1 quantize. Inner loop of PR-X9's `encode_from_dense`. **EXACT** integer arithmetic. **GAP** — drafted in PR-X9 §"Encoding modes" but not implemented.

#### G5 — fast_exp precision audit for NARS

The splat3d `fast_exp_x16` (Schraudolph 1999) has 3% relative error. Acceptable for alpha attenuation (visual ε); **probably unacceptable for NARS confidence-cascade convergence** because the error compounds multiplicatively across tier propagation.

Decision needed:
- **(a)** Add `precise_exp_x16` path with 4th-order Padé polynomial (~7 FMA, accurate to 1e-7) and use it inside NARS revise closures
- **(b)** Re-derive NARS revise as a closed-form rational that avoids exp entirely (truth-revision is fractional, not exponential — exp only enters if we use confidence-as-exponential-decay)
- **(c)** A/B test 3% fast_exp against precise_exp on a synthetic NARS cascade and measure convergence drift

**Lean: (b)** — NARS truth-revision doesn't actually need exp; it's a weighted average + saturation. The exp came in via splat alpha-compositing. If W7 closure-swap replaces alpha with NARS, the exp call goes away with it.

## Precision-class summary

| Class | Definition | Primitives |
|---|---|---|
| **EXACT** | Bit-exact across reorderings | Spd3 ops, Mahalanobis², radix sort, CascadeAddr, NARS revise (after G5 (b)), basin XOR-popcount, CTU mode encoder, Hilbert-3D |
| **FAST OK** | 1-5% relative error acceptable | fast_exp_x16 (alpha only), vrcp14ps (perspective div), vrsqrt14ps (view-dir norm), SH eval (deg-3 truncation) |
| **VERIFY** | A/B audit before cognitive use | fast_exp_x16 in NARS context (G5), `from_scale_quat` near-degenerate cases, near-singular Σ_image conic inverse |

## Cross-cutting research items (consolidating from all 3 docs + this review)

From the three splat sprint docs:
1. BodyParts3D coverage (skeleton-anchored)
2. Muscle attachment table (skeleton-anchored)
3. Clarius access (now optional via SyntheticBinding)
4. FMA license (CC-BY-SA implications)
5. Pillar-8 σ_temporal calibration (cardiac/respiratory/micro literature values)

From this arithmetic review (NEW):
6. **Hilbert-3D encode/decode algorithm choice** (Butz's algorithm vs Skilling's algorithm vs precomputed rotation table)
7. **INT4×N packed dot product strategy** (VNNI vs software unpack vs AMX with INT8 widening)
8. **NARS revise precision class decision** (G5 (a) / (b) / (c) above)
9. **CTU mode encoder λ-RDO calibration** (borrow x265 medium-preset λ table vs NARS-confidence-derived)
10. **Codebook size const-generic strategy** (PR-X9 Q7: u8 vs u16 basin_idx)

## Recommended ordering

**Phase 0 (substrate, parallel-safe):**
- L4 Hilbert-3D encode/decode — single-worker, ~3 days, ~200 LoC. Unblocks splat4d cascade PR 1 AND PR-X4 cascade addressing.
- G1 INT4×32 packed dot product — single-worker, ~3 days, ~150 LoC. Unblocks PR-X7 typed cell-DSL.

**Phase 1 (medical + cognitive co-substrate, sequential):**
- L6 Pillar-8 temporal sandwich — needs σ_temporal literature values first
- L5 Gaussian-mixture moment-match — needs L6
- L7 Mesh→splat fitting — needs L0 (shipped)

**Phase 2 (cognitive-only):**
- G3 Basin XOR-popcount — needs PR-Z1 + PR-Z2 OR embedded-TTL escape hatch
- G4 CTU mode encoder — needs G3
- G2 NARS truth-revision — needs G5 decision

**Phase 3 (closure swap):**
- W7: replace splat alpha-compose closure with NARS revise. Single-PR scope. Drops fast_exp from the cognitive path entirely.

The CRITICAL observation: **Phase 0 unblocks both the medical sprint (splat4d skeleton-anchored) AND the cognitive sprint (PR-X4 + PR-X9)** because Hilbert-3D + INT4×N are dependencies of both. Build them first; both downstream stacks accelerate together.

## Recommended workshop

Before the joint plan-review savant, do ONE 30-min math workshop that:
1. Confirms σ_temporal values from echo/respiratory literature (item 5)
2. Picks the Hilbert-3D algorithm (item 6 — recommend Butz/Skilling table-driven)
3. Decides G5 NARS precision class (recommend (b) — drop exp from NARS path)
4. Drafts the precise_exp_x16 path (FOR splat alpha; the cognitive path doesn't need exp after G5(b))

After that workshop, the joint savant has a clean math surface to rule on. Without it, the savant will surface these as open questions per design doc and slow the sprint.

## Shopping-list addendum (2026-05-18 cross-cutting gap analysis)

Beyond the cognitive-shader-stack-specific gaps above, a broader inventory
identifies the **shared-linalg-below-LAPACK** gap as the highest-leverage
non-cognitive sprint in the queue. Three downstream stacks all hand-roll
math that should live in one canonical module:

- **splat3d** ships its own `Spd3` (Smith-1961, shipped PR #153)
- **lance-graph jc** has **three** separate Spd2/Spd3 copies (`ewa_sandwich.rs`,
  `ewa_sandwich_3d.rs`, `koestenberger.rs`)
- **`hpc::{gpt2, openchat, stable_diffusion}`** inline RMSNorm, SiLU, RoPE,
  attention because there's no canonical fn

Consolidating into one `crate::hpc::linalg::*` unblocks three sprints
simultaneously. **See `.claude/knowledge/pr-x10-linalg-core-design.md`** —
the consolidating sprint (12-worker max-fan-out, A1 MatN as the only chain
dependency, ~5 weeks sequential or ~2 weeks parallel).

The PR-X10 shopping list (Tier 1 blocks splat3d training; Tier 2 blocks
the inference modules; Tier 3 nice-to-have):

| Tier | Primitives |
|---|---|
| **T1** | MatN const-generic carrier · Quat algebra (mul, conjugate, slerp, from_axis_angle) · 3×3/4×4 inverse · symmetric eig N≥4 (Jacobi + QR) · SVD (Golub-Reinsch + one-sided Jacobi) · polar decomposition · mat_exp / mat_log (Padé + s&s) · SH deg 4–7 · splat3d backward API freeze |
| **T2** | Conv1D / Conv2D · batched gemm · LayerNorm / RMSNorm / GroupNorm · GELU / SiLU / Swish / Mish · RoPE · fused attention (naive + flash) · cross-entropy + softmax-backward |
| **T3** | SIMD RNG distributions (Gaussian / exp / beta) · special fns (erf / gamma / beta / Bessel) · einsum · Bluestein FFT · irfft · DCT-II/IV + Daubechies wavelets · sparse GEMM · tridiagonal / banded solvers |

The **jc consolidation** is a follow-on (`jc-X1..X4`):
- Consolidate the three Spd2/Spd3 copies into a private `jc::hadamard` (keeps jc zero-dep on ndarray; mirrors PR-X10's canonical surface)
- `Cov16384` carrier for Pillar 8 (Düker-Zoubouloglou CLT in ℝ^16384)
- Wasserstein-1 / nested distance solver (Sinkhorn-Knopp + Hungarian) for Pillar 10
- Signature transform for Pillar 11 (Hambly-Lyons)
- SPD-cone ops: log-Euclidean mean, affine-invariant Riemannian (Karcher/Fréchet), Bures-Wasserstein geodesic
- Manifold log/exp maps: SO(n), Grassmannian, Stiefel (unblocks Pillar 2 Cartan-Kuranishi)

PR-X10 ships the ndarray-side canonical surface; jc agents pick up the
consolidation against it. PR-X10 is **independent** of PR-X4 / PR-X9 / PR-Z1
(no file overlap), can ship concurrently from a separate branch.

## Shopping-list addendum (2026-05-19 substrate-pair update)

The 2026-05-18 entry above (PR-X10 + jc consolidation) is the **arithmetic**
substrate. A 2026-05-19 cross-cutting design session identified the
**storage-and-contract** substrate as the missing companion piece. Two
pre-sprint prompts capture it:

- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — **PR-X1 + PR-X2**
  (GridLake carrier: `MultiLaneColumn` + `#[derive(SoA)]` proc-macro)
- `.claude/knowledge/hhtl-pr-x14-substrate-contract-prompt.md` — **PR-X14′**
  (`lance-graph-contract::column` module set + new
  `lance-graph-contract-bridge` sibling crate)

### Cost-of-ownership framework adopted this session

Cost-of-ownership is calculated only by **how many cluttered parts a choice
replaces**. By that metric the substrate-pair is the highest-leverage move
in the Phase-2 arc:

| Adoption | Cluttered parts replaced |
|---|---|
| PR-X1 + PR-X2 (GridLake carrier) | **5 implicit per-consumer column-buffer copies** (Databend, Tantivy, lance-graph, sea-orm, SurrealDB each maintain their own) collapse to one |
| PR-X14′ (contract + bridge) | **4 duplicate column-access patterns** (BindSpace `Box<[u64]>`, lance-graph query `HashMap<String, RecordBatch>`, planner `Morsel` placeholder enum, `lance::Dataset.scan()` rolled per feature) collapse to one |
| Databend (future, when integrated through the contract) | ClickHouse (1 JVM/C++ service decommissioned) |
| Tantivy (future, when integrated through the contract) | Elasticsearch + Lucene (1 JVM service decommissioned) |
| lance-graph + ndarray-over-TiKV (existing) | JanusGraph + TinkerPop/Gremlin (2 JVM services decommissioned) |

### Decisions that pinned the scope

Three architectural decisions narrowed the prompt's scope from earlier
proposals in the session:

1. **Skip SQL for now**. All SQL-frontend work (sea-orm column contract,
   sqlparser-rs standalone, PostgREST front, the Phase-3 "fusion parser"
   sprint) is deferred. The lance-graph SQL path keeps its `RecordBatch +
   DataFusion::SessionContext` pipeline unchanged.

2. **Don't evict datafusion**. `lance-graph/Cargo.toml`:35-39 has hard deps
   on `datafusion = "52"` + four sibling `datafusion-*` crates; lance-graph
   upstream depends on them. PR-X14′ does NOT touch these — the datafusion
   1400-module tail stays contained inside the lance-graph crate.

3. **Use storage format as-is**. PR-X14′ reads `lance::Dataset` and
   `arrow::RecordBatch` as-is; it does NOT define Arrow extension types,
   NOT register custom Lance encoders, NOT modify schemas in place.

### Decisions superseded

- ❌ Earlier-session proposal: PR-X14 evicts datafusion from lance-graph and
  builds a parsendes Fusionskraftwerk at 240ns/parse. **Superseded** —
  lance-graph upstream dep makes eviction out of scope; the planner IR +
  Cypher nom parser + NSM/FSM parser already exist and PR-X14′ is the
  consolidation, not a fusion-parser sprint.
- ❌ Earlier-session proposal: two new top-level crates `gridlake-contract`
  + `gridlake-bridge`. **Superseded** — `lance-graph-contract` already
  exists with 10+ workspace consumers and is the right home for the column
  module; only a sibling bridge crate is genuinely new.

### Two Q-markers carried forward (plan-review savant decides at preflight)

- **Q-NEW-1** (from GridLake prompt): GridLake at W2.5 prerequisite slot
  vs absorbed into PR-X10 as A13/A14
- **Q-NEW-2** (from PR-X14′ prompt): PR-X14′ concurrent with GridLake at
  W2.5 (path α, 10 workers) vs sequential at W3 (path β, 4 workers + 0.5
  week schedule extension)

### Forbidden constraints (preserved from the prompts)

- PR-X14′ must NOT add `arrow`, `arrow-schema`, `arrow-array`, `lance`,
  `lancedb`, or `datafusion` to `lance-graph-contract`'s Cargo.toml —
  preserves the lib.rs:1 zero-dependency invariant
- PR-X14′ must NOT touch `lance-graph/src/sql_query.rs`,
  `lance-graph/src/sql_catalog.rs`, or `lance-graph/Cargo.toml`'s
  datafusion deps — these are deliberately out of scope per decision (2)
  above
- ndarray crate receives zero changes from PR-X14′ — the substrate-pair is
  consumer-side only

## Cross-references

- `/root/.claude/uploads/.../7b0ea082-splat3d_sprint_prompt.md` — splat3d sprint (shipped as ndarray PR #153, 2026-05-18)
- `/root/.claude/uploads/.../cdcb7d3d-splat4d_cascade_sprint.md` — splat4d cascade sprint (proposed, superseded by skeleton-anchored)
- `/root/.claude/uploads/.../7071b77a-splat4d_skeleton_anchored_sprint.md` — splat4d skeleton-anchored sprint (current proposal)
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid substrate, shipped
- `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade onto BlockedGrid
- `.claude/knowledge/pr-x9-design.md` — lazy basin-codebook storage
- `.claude/knowledge/pr-z1-ogit-cognitive-bootstrap.md` — OGIT Cognitive namespace bootstrap
- `.claude/knowledge/pr-x10-linalg-core-design.md` — **the consolidating linalg sprint** that unblocks splat3d training + inference modules + jc Pillars simultaneously
- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — **PR-X1 + PR-X2** GridLake carrier (`MultiLaneColumn` + `#[derive(SoA)]`); 2026-05-19 session output, branch `claude/gridlake-pre-sprint-prompt`
- `.claude/knowledge/hhtl-pr-x14-substrate-contract-prompt.md` — **PR-X14′** `lance-graph-contract::column` module + `lance-graph-contract-bridge` sibling crate; 2026-05-19 session output, same branch
