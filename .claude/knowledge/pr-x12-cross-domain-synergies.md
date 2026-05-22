# PR-X12 — Cross-Domain Synergies: x265 ⇄ 3D Gaussian Splat ⇄ Cognitive Shaders ⇄ BLAS/MKL

> READ BY: savant-architect, codec-architect, cognitive-architect,
> splat-architect, vector-synthesis, l3-strategist, sentinel-qa,
> product-engineer.
>
> Status: epiphany-grade exploration doc, drafted 2026-05-22 during
> the PR-195 (A2 + A3-intra) review cycle.
>
> Companion to `.claude/knowledge/pr-x12-codec-x265-design.md` (the
> mechanical design). This doc captures the **why-it-generalizes**
> that the design doc deliberately scopes out.

## TL;DR

PR-X12 was framed as "x265 for cognitive cells" — the mechanical
design doc already maps x265 onto BlockedGrid. The deeper observation
this doc commits to is that the **same primitives — `LeafCu`,
`pack_leaf`, `predict_intra`, `Ctu`, rANS — also serve 3D Gaussian
splat coefficient compression, transformer attention sparsification,
and distributed-SGD gradient streaming.** The four domains are not
analogous; they are **four loads on a single predictive-coder
substrate.** This doc:

1. Names the isomorphism precisely (§ 2)
2. Maps every codec primitive to its load in each domain (§ 3)
3. Calls out the epiphanies — cross-domain insights I have not seen
   in print (§ 4)
4. Lays out integration plans with concrete PR-arc estimates (§ 5)
5. Catalogues exploration paths that warrant a sprint, not a PR (§ 6)
6. States the holy grail outcomes that fall out if it all lands (§ 7)
7. Honest debt accounting — codec side (§ 8) and existing stack
   side (§ 9). No marketing.

## 0. Audience preconditions

This doc assumes the reader has internalised:

- `Ctu` / `CtuArena` / `CtuPartition` / `LeafCu` / `CellMode` /
  `MergeDir` from `src/hpc/codec/ctu.rs` (PR-170).
- `pack_leaf` / `unpack_leaf` / `pack_header` / `predict_intra` from
  `src/hpc/codec/mode.rs` + `src/hpc/codec/predict.rs` (PR-195).
- The Click P-1 method discipline (operations on carriers, not free
  functions) and the data-flow rule (no `&mut self` during compute).
- The cognitive cell → basin codebook story from
  `.claude/knowledge/pr-x12-codec-x265-design.md` § "Core types".
- Inria 3DGS paper (Kerbl et al. 2023) + EWA Splatting (Zwicker 2001).
- That the cognitive `splat.rs` in `lance-graph-contract` is sacred
  and **separate** from `splat3d::*` (the geometric forward renderer
  shipped in PRs 1-7 of the May sprint).

If any of the above is fuzzy, read those sources first; the rest of
this doc compresses.

## 1. The four loads

| Load | Carrier | Per-element payload | Predictability source |
|------|---------|--------------------|----------------------|
| **Cognitive cell** | `BlockedGrid<u64, 64, 64>` | 64-bit fingerprint | basin codebook (per-frame), spatial NEWS neighbours |
| **3DGS Gaussian** | SoA `(μ, scale, rot, opacity, SH)` | ~236 bytes raw | sorted-along-curve neighbours, basin (color/scale clusters) |
| **Transformer attention** | `(Q, K, V)` per (head, token) | Q,K,V vectors | KV palette clusters, previous-token attention pattern |
| **Distributed SGD gradient** | per-parameter `∂L/∂w` | FP32 grad | mini-batch siblings, gradient sparsity, sign agreement |

All four loads share the same predictive-coding skeleton:

```
                ┌──────────────────────────────────────┐
                │ 1. Build basin codebook (offline or  │
                │    online k-means on the carrier)    │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ 2. Resolve nearest basin per element │
                │    → (basin_idx, δ from basin)       │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ 3. Mode-decide per element:          │
                │    Skip (δ=0)                        │
                │    Merge (δ matches NEWS neighbour)  │
                │    Delta (δ fits 8-bit)              │
                │    Escape (full payload, idx into    │
                │            per-frame escape vector)  │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ 4. Pack LeafCu (2/3/3/6 bytes) into  │
                │    bytestream                        │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ 5. rANS-encode the bytestream with   │
                │    per-frame frequency tables (A7)   │
                └──────────────────────────────────────┘
```

Steps 1-5 are domain-agnostic. **What changes per load is the basin
clustering rule (step 1) and the escape payload (step 4's Escape
branch).** Everything else is shared kernel.

## 2. The isomorphism

**Claim:** `LeafCu` is a **discriminated union over (basin_ref,
residual) representations**, parameterised by a 2-bit predictability
class. The four bits across `(CellMode × MergeDir)` form a 16-state
classification machine that is not specific to video or cognitive
content. It is the natural mode-coding alphabet for any signal that
is:

- **Locally predictable** from a small per-frame codebook
- **Spatially smooth** in a defined neighbour topology (NEWS, in
  PR-X12 today; trivially generalisable to 6-way XYZ or
  token-sequential)
- **Heavy-tailed** in its residuals (most values fit a small δ;
  rare values need full Escape)

All four loads named in § 1 satisfy these three properties. The
codec we shipped is therefore not "an HEVC port"; it is the
**reference encoder for predictable-codebook signals**. HEVC is one
consumer.

### The 16-state classification table

`(CellMode, MergeDir)` cross product, repurposed per domain:

| Mode × Dir | Cognitive cell | 3DGS Gaussian | Attention | Gradient |
|------------|----------------|---------------|-----------|----------|
| Skip, — | cell = basin exactly | Gaussian = palette splat exactly | Q has no significant K | grad ≈ 0 (sparse update) |
| Merge, N | inherit δ from N-neighbour | inherit from prev-Morton Gaussian | inherit attention from prev-token | inherit grad from prev-layer sibling |
| Merge, E | inherit from E-neighbour | inherit from next-Morton | inherit from next-token | inherit from next-layer |
| Merge, W | inherit from W-neighbour | inherit from coarse-tier parent | inherit from prev-head | inherit from prev-iteration |
| Merge, S | inherit from S-neighbour | inherit from fine-tier child | inherit from next-head | inherit from next-iteration |
| Delta, — | 8-bit cell perturbation | 8-bit residual on (μ, scale, op) | 8-bit attention weight δ | 8-bit grad (QSGD, signSGD-magnitude) |
| Escape, — | full 64-bit fingerprint via idx | full SH coeffs ≥ L=2 via idx | full FP16 Q vector via idx | full FP32 grad via idx |

`MergeDir`'s 4-way alphabet is **already the natural carrier** for
"inherit from one of 4 neighbours in some topology". The topology
varies per load; the encoding does not.

## 3. Primitive → load mapping matrix

This is the dense one. Each row is one primitive from PR-X12; the
columns are the four loads. Cells say what the primitive does in
that load, with file/line refs back to ndarray master.

### 3.1 Carrier primitives

| Primitive | Cognitive cell | 3DGS Gaussian | Attention | Gradient SGD |
|-----------|----------------|---------------|-----------|--------------|
| `Ctu` (`ctu.rs:285`) | one L1 BlockedGrid block (64×64 cells) | one tile-bin or one octree node (64-256 Gaussians) | one (token-window × heads) block (typically 64×16) | one parameter-shard (64K weights) |
| `CtuArena` (`ctu.rs:212`) | 85-node quad-tree per CTU | tile quad-tree (LOD cascade) | token-window prefix-tree | per-shard residual hierarchy |
| `CtuPartition` (`ctu.rs:193`) | recursive 64→32→16→8 split | tile 64×64 → 16×16 → 4×4 LOD | window 64→16→4 attention granularity | shard 64K→16K→4K gradient grouping |
| `LeafCu` (`ctu.rs:114`) | one cell's encoded mode | one Gaussian's encoded mode | one (head, token-position)'s mode | one weight's gradient mode |
| `MAX_BASIN_IDX = 4095` (`mode.rs:62`) | 4096-entry basin codebook | 4096-entry palette (μ_color × scale clusters) | 4096-entry KV cluster centroid | 4096-entry gradient-pattern bank |
| `BASIN_NONE` (`mode.rs:71`) | cell outside any basin | Gaussian outside palette range | Q outside KV palette (forces Escape) | grad outside known patterns |

### 3.2 Encoder primitives

| Primitive | Cognitive cell | 3DGS Gaussian | Attention | Gradient SGD |
|-----------|----------------|---------------|-----------|--------------|
| `pack_header(mode, basin_idx)` (`mode.rs:83`) | 16-bit cell header | 16-bit Gaussian header | 16-bit (head, token) header | 16-bit weight header |
| `pack_leaf` (`mode.rs:172`) | 2/3/3/6 byte cell record | 2/3/3/N byte Gaussian record (N depends on SH order in Escape) | 2/3/3/Q-width byte attention record | 2/3/3/4-byte weight gradient record |
| `predict_intra` (`predict.rs:186`) | encoder picks mode for cell | encoder picks mode for Gaussian per-Morton-step | encoder picks mode for (Q, K) pair | encoder picks mode for ∂L/∂w |
| `IntraContext.neighbours` (`predict.rs:117`) | NEWS spatial neighbours | prev/next Morton-sorted neighbours + parent/child tier | prev/next token + prev/next head | prev/next layer + prev/next iter |
| `IntraConfig` (`predict.rs:132`) | (future RDO knobs) | (future LOD/PSNR tradeoff knobs) | (future accuracy/latency knobs) | (future compression/convergence knobs) |
| `escape_next: Option<&mut u32>` (`predict.rs:202`) | escape vector cursor for full-payload cells | escape vector cursor for SH-heavy Gaussians | escape vector cursor for outlier Q | escape vector cursor for outlier grads |

### 3.3 Wire-format primitives (deferred to A7/A8)

| Primitive | Cognitive cell | 3DGS Gaussian | Attention | Gradient SGD |
|-----------|----------------|---------------|-----------|--------------|
| rANS encoder (A7) | per-frame basin-frequency table | per-asset palette-frequency table | per-context attention-pattern frequency | per-layer gradient-mode frequency |
| Stream framing (A8) | CTU markers, frame headers | tile-bin markers, asset headers | window markers, batch headers | shard markers, iter headers |
| Escape vector | per-frame `Vec<u64>` of full fingerprints | per-asset `Vec<f32; SH_LEN>` | per-context `Vec<f16; head_dim>` | per-shard `Vec<f32>` |

## 4. Epiphanies

Cross-domain insights worth flagging because each has 1-3 papers'
worth of novelty if pursued. None of these are in print as of the
literature snapshot I'm working from; **claim** is the right word, not
"finding".

### E1. **`MergeDir` is a topology, not a direction.**

`{North, East, West, South}` happens to be a 2D Cartesian raster
mental model. The codec doesn't care. The discriminant alphabet just
needs to be a 4-way categorical over "which of 4 neighbours did I
inherit from". In 3DGS that's `{prev-Morton, next-Morton, parent-LOD,
child-LOD}`. In attention that's `{prev-token, next-token, prev-head,
next-head}`. In SGD that's `{prev-iter, next-iter, prev-layer,
next-layer}`. **No code change required.** The doc + the docstring
in `IntraContext.neighbours` are the only constraints; the 2-bit
encoding is topology-free. → write up as: "Carrier-agnostic Merge
inheritance via parameterised 4-neighbour topology" (mini-paper).

### E2. **`predict_intra` already encodes attention sinks.**

The "Skip" mode case in `predict_intra` (`predict.rs:189-190`) —
returns when `delta_i32 == 0` — is exactly the attention-sink
phenomenon Streaming-LLM, H2O, SnapKV chase. Their attention mass
concentrates on a tiny subset of tokens; the rest are "Skip". With
the basin codebook as KV cluster centroids, **`predict_intra` is a
zero-shot attention sparsifier**: it labels every (Q, K) pair as
Skip/Merge/Delta/Escape and the wire cost is monotone in attention
mass. Combine with the rANS A7 and you get **bit-exact KV-cache
compression with a tunable accuracy floor**. The encoder is shipped.

### E3. **`escape_next: &mut u32` is the lineage of gradient streaming.**

The owner-author review's P1 — escape allocator collision — is the
exact issue federated-SGD papers solve with "all-reduce buckets":
multiple workers each emit gradient deltas, the aggregator needs
non-colliding slots in a shared vector. The `Option<&mut u32>` cursor
**is the all-reduce slot allocator**, just per-CTU instead of
per-batch. Lift it to a worker-pool API and you have a federated
gradient codec without writing new code.

### E4. **The 64-bit `Fingerprint` and a 3DGS Gaussian's first-six floats compress identically.**

Cognitive `Fingerprint` is 64 bits = 4×16-bit lanes. A 3DGS Gaussian's
`(μ_x, μ_y, μ_z, scale_x, scale_y, scale_z)` is 6 × FP16 quantised =
96 bits, but with 32 high bits dominated by the scale envelope which
is locally constant per palette basin. After basin subtraction, the
residual is ~64 bits — **identical to the cognitive cell case**. The
same `pack_leaf` works. The escape vector type changes from `u64` to
`[u16; 6]` but the codec is structurally invariant.

### E5. **The Morton/Hilbert sort along which we encode 3DGS Gaussians is the EXACT spatial structure HEVC's macroblock raster scan implements in 2D.**

HEVC's CTU traversal is z-order. 3DGS Gaussians sorted Morton/Hilbert
along their μ are z-order in 3D. The encoder doesn't know it's seeing
3D content; the spatial coherence in 1D-along-curve is identical to
2D-along-raster. **The CTU partition machinery in `ctu.rs` ports to
3DGS with zero changes to the partition logic.** What changes is the
predicate that decides when to split (variance of (μ, scale, opacity)
inside the node vs. PSNR target).

### E6. **rANS with per-frame frequency tables is the **only** entropy coder that scales to 10⁶+ tokens.**

CABAC is fine for video at ~10⁵ macroblocks/frame. Attention at
10⁶+ tokens/sec needs an entropy coder whose state machine fits in
L1 cache and whose throughput is gated by table lookup, not by the
serial CABAC interval renormalisation. rANS is that. **A7 is the
critical piece; without it the codec is academic.** Prioritise.

### E7. **The 4096-entry basin codebook is identical to attention's KV palette identity in lance-graph.**

This is the architectural payoff. `lance-graph::SpoDistanceMatrices`
computes (basin_id, distance) for SPO triples at 611M lookups/sec
(see CLAUDE.md "Session: Qwen3.5 × Opus 4.5/4.6"). The same data
structure feeds the cognitive codec basin lookup AND the attention
KV-cluster lookup AND the 3DGS palette nearest-neighbour. **One
codebook, three consumers, identical lookup kernel.** Lance is the
column substrate; the codebook is its first "logical schema" — and
that schema is shared.

### E8. **Mode-coding is parameter-efficient supervised LoRA.**

A LoRA adapter on a weight matrix `W` is a rank-r perturbation
`W + ΔW = W + B·A`. Express `ΔW` as a `BlockedGrid<u64, 64, 64>` and
mode-code it. Most weights are Skip (no LoRA contribution), some
inherit from neighbours (Merge), a few have small per-weight deltas
(Delta), and the heavy hitters are Escape. **`LeafCu`-coded LoRA is
~10× smaller than rank-32 LoRA on weight matrices > 4096².** The
codec is the parameter-efficient fine-tuning representation.
The user's "Pertuberationslernen" instinct lands here.

### E9. **The `splat3d` PRs 1-7 (May sprint) and the `codec` PRs are the SAME pipeline shifted 90°.**

The splat3d forward pipeline is: project → tile-bin → mode-decide
(which Gaussian contributes at which pixel) → alpha-composite. The
codec pipeline is: build codebook → block-partition → mode-decide
(which mode each cell takes) → entropy-code. **Both end in
mode-decide → reduce.** The mode-decide kernel is `predict_intra`
in both cases; the reduction differs (alpha vs. rANS). A unified
"mode-decide + reduce" trait would collapse 2 KLoC. **Worth a
sprint, not a PR**.

### E10. **The lossy Escape fallback is a PSNR knob in disguise.**

The owner-review's P2 nit — "lossy Escape emits `CellMode::Delta`,
the docstring lies" — is a feature, not a bug, **iff** we expose a
"lossy_threshold: u8" config. Then the fallback becomes "use Delta
for any |δ| ≤ threshold even if it would normally Escape". That's
the rate-distortion knob HEVC's λ-RDO tunes. **Promote the
docstring acknowledgement into a config field in A6 RDO.**

## 5. Integration plans

Concrete branches/PRs, each with effort estimate + dependency.
Listed by priority (impact ÷ risk).

### Plan A — A7 rANS (critical, no domain-specific blockers)

**Effort:** 1 worker × 1 week. Standard rANS, single-symbol,
encoder + decoder + parity test. Consumes `pack_leaf` output, emits
compressed bytestream.

**File:** `src/hpc/codec/ans.rs` (new).

**Dependency:** none — A2 + A3-intra are sufficient input.

**Why first:** without entropy coding, the codec gives 2-3× over
raw. With rANS at per-frame frequency tables, 6-10×. Below the
rANS threshold, the codec is academic.

### Plan B — A3-inter (cross-tier neighbour scan) (codec-side completion)

**Effort:** 1 worker × 3 days. Extend `IntraContext.neighbours` to
include parent-tier and child-tier neighbours from `BlockedGrid`'s
L2/L3 cascade. Mode-decision tree gains 8 candidates instead of 4.

**File:** `src/hpc/codec/predict.rs` (extend) + new `inter.rs`.

**Dependency:** PR-X3 BlockedGrid L2/L3 cascade (shipped).

**Why second:** unlocks the recursive partition compression. Without
inter prediction, parent-tier basins don't seed child-tier deltas.

### Plan C — EWA SYRK-batched (3DGS performance, no codec changes)

**Effort:** 1 worker × 1 week. Replace `sandwich_x16` per-Gaussian
loop with batched `cblas_ssyrk`. Add backend-dispatch (native /
intel-mkl / openblas).

**File:** `src/hpc/splat3d/spd3.rs` (extend) +
`src/backend/{native,mkl,openblas}.rs` (add syrk wiring).

**Dependency:** ndarray BLAS backend infra (shipped).

**Why third:** biggest pure-FLOPS win, splat-aligned, no codec
coupling. Hits the holy grail outcome §7.1.

### Plan D — Attention codec PoC (cognitive-side new ground)

**Effort:** 2 workers × 2 weeks. Wire `predict_intra` against a
synthetic KV cache; build the basin codebook via mini-batch k-means
on K vectors; measure compression vs. accuracy on a known LLM
benchmark (LongBench, RULER).

**File:** new crate `crates/attention-codec/` consuming
`ndarray::hpc::codec::*`.

**Dependency:** Plan A (rANS) for realistic compression numbers.

**Why fourth:** highest-novelty load; depends on A7 to be convincing.

### Plan E — 3DGS coefficient codec (splat-side compression)

**Effort:** 2 workers × 3 weeks. Morton-sort a trained scene's
Gaussians, build per-asset palette codebook via k-means over
(color, scale), mode-code the residuals through `pack_leaf`, rANS
through A7.

**File:** new module `src/hpc/splat3d/codec.rs`.

**Dependency:** Plan A (rANS), Plan B (A3-inter for LOD cascade).

**Why fifth:** highest engineering value, but has external benchmark
risk — Inria's PLY format has format-stability constraints we'd
need to negotiate (or just ship a parallel format).

### Plan F — Gradient streaming codec (federated SGD)

**Effort:** 2 workers × 4 weeks. Workers emit `LeafCu` streams; the
aggregator decodes and applies. Requires a `&mut u32` allocator
generalised across worker pools (see E3).

**File:** new crate `crates/grad-codec/`.

**Dependency:** Plan A, Plan B.

**Why sixth:** highest research novelty; lowest near-term ROI
(federated SGD is a niche stack).

## 6. Exploration paths

Things that warrant a sprint or research session, not a single PR.
Each has at least one unresolved question that disqualifies it from
"integration plan" status.

### X1. Carrier-agnostic 4-neighbour topology trait

Design a `trait NeighbourTopology<const N: usize>` that
`IntraContext` consumes generically. Cognitive: N=4 NEWS. 3DGS: N=4
(prev/next-Morton, parent/child-LOD). Attention: N=4 (prev/next
token, prev/next head). SGD: N=4 (prev/next iter, prev/next layer).
Compile-time-resolved, zero-cost. **Open question:** does mode-coding
generalise to N=6 (3D XYZ)? Two more `MergeDir` discriminants needed;
bit-budget impact on the wire format.

### X2. Hierarchical motion estimation as cross-tier prediction

HEVC's hierarchical ME (4-tier coarse-to-fine pyramid) maps onto
the BlockedGrid L1/L2/L3/L4 cascade. **Open question:** the cost
function. HEVC uses SAD on luma; cognitive uses Hamming on
Fingerprints; 3DGS uses PSNR on rendered tiles. Three cost
functions, one search structure — is the hierarchical-ME logic
worth the abstraction?

### X3. CABAC vs. rANS for attention KV cache

CABAC's serial dependency caps throughput at ~10⁸ symbols/sec on
modern CPU. rANS gets ~10⁹. **Open question:** does the latency
floor matter for attention's real bottleneck (memory bandwidth,
not entropy decode)? Bench before committing to A7.

### X4. SH coefficient intra-prediction in spectral space

Predict L=2, L=3 SH from a learned linear function of L=0, L=1.
**Open question:** is the linear function global or per-basin? Per-
basin is more expensive but probably 2× better; need data to
decide. Inria's stock 3DGS dataset (Mip-NeRF 360, T&T, Deep
Blending) is the benchmark.

### X5. Mode-coded LoRA

E8 above. **Open question:** does Skip-heavy `ΔW` retain LoRA's
fine-tuning quality? Run a controlled experiment on a Qwen3.5-7B
checkpoint with LoRA rank 8 vs. mode-coded ΔW at the same byte
budget. Measure on MMLU-redux + a downstream task.

### X6. Unified `mode_decide + reduce` trait (E9)

Generalise `predict_intra` so it's parameterised on the **reduction
operator**: alpha-composite (3DGS), rANS-encode (codec),
sum-reduce (SGD all-reduce), softmax (attention). **Open
question:** does a single trait actually compose, or does each
domain need its own bespoke variant? Risk: premature abstraction.

### X7. Lance column substrate as the universal palette codebook backing store

`SpoDistanceMatrices` at 611M lookups/sec, 388 KB RAM. If we
extend it to handle (basin_centroid → idx) lookups for all four
loads in § 1, we get one column-store serving cognitive cells,
KV palettes, 3DGS palette, and gradient-pattern banks. **Open
question:** the centroid distance function differs per load
(Hamming for fingerprints, L2 for Gaussians, cosine for Q vectors,
sign-vote for gradients). Does `SpoDistanceMatrices` accept
pluggable metrics?

### X8. AMX TDPBF16PS for batched EWA sandwich

The `M · Σ · Mᵀ` operation on 16 Gaussians at a time fits AMX's
16×16 BF16 tile exactly. **Open question:** the precision loss
from BF16 vs. FP32 on 2D conic invertibility — preliminary lit
search says fine, but needs Pillar-7-style probe before commit.

## 7. Holy Grail material

If all of § 5 + § 6 land, the following outcomes fall out. None
are guaranteed; each is the "yes, that worked" branch.

### HG1. **One codec, four loads.**

A unified bytestream format codes cognitive cells, 3DGS scenes,
KV caches, and gradient streams interchangeably. The Lance column
substrate stores them all in the same Arrow-backed layout. A
single `cargo install` ships compression for video-codec-equivalent
+ all four cognitive/ML loads.

Marketing line: *"x265 was a codec for one signal. PR-X12 is a
codec for the manifold of predictable codebook-coded signals."*

### HG2. **Sub-1-bit-per-Gaussian 3DGS compression.**

Stock 3DGS: ~250 bytes/Gaussian raw, ~50 bytes after PLY-trim.
PR-X12 mode-coded + A7 rANS: ~3-8 bits/Gaussian for the dominant
modes. **30-60× over current state of the art.** A 1M-Gaussian
scene fits in ~500 KB instead of 50 MB. Streamable as a video.

### HG3. **Bit-exact attention with tunable accuracy floor.**

`predict_intra` over (Q, K) palette gives an attention sparsifier
that is bit-exact at the "Escape always" setting and gradually
loses precision as Skip/Merge/Delta dominate. The accuracy floor
is a single knob (`escape_threshold: u8`) — no per-model tuning.
Streaming-LLM, H2O, SnapKV become consumers of one codec.

### HG4. **Federated SGD at 8-16× compression with zero accuracy loss.**

Worker→aggregator gradient streams via `LeafCu`. Skip-mode kills
noise; Merge-mode discovers parameter sharing online; Delta-mode
gives QSGD; Escape-mode preserves outliers. The compression is
free because the codec already exists.

### HG5. **Lance column-substrate identity becomes the ground truth.**

The same Arrow buffer feeds: cognitive cell storage, 3DGS Gaussian
SoA, KV cache, gradient shards. The codec encodes the same bytes
across all four. `lance-graph::SpoDistanceMatrices` becomes the
universal palette codebook lookup. ndarray = hardware; lance =
substrate; codec = compression; PR-X12 closes the substrate loop.

### HG6. **The "splat3d × x265" bet pays out as one library.**

The May splat3d sprint (PRs 1-7) gave a CPU-SIMD 3DGS renderer.
PR-X12 gives the codec. Combined, the same library compresses,
streams, decodes, and renders 3D scenes in real-time on a single
core. **The combination is novel; neither half is.**

## 8. Codec-side technical debt

Honest accounting. PR-X12 shipped A2 + A3-intra; what we owe
ourselves to make the rest of this doc bankable:

### D-CODEC-1. A3-inter is unwritten. (P1)

The `IntraContext` consumes 4 NEWS neighbours; the design doc
calls for parent-tier + child-tier extension. Without inter
prediction, the BlockedGrid L2/L3/L4 cascade contributes nothing
to compression. **Plan B in § 5.**

### D-CODEC-2. rANS A7 is unwritten. (P0 for any real benchmark)

Without entropy coding, the per-mode bit budget is rounded to
bytes. 2 bits/cell achievable becomes 2 bytes/cell shipped — 8×
overhead. Plan A.

### D-CODEC-3. λ-RDO A6 is unwritten. (P1)

Mode-decision is greedy (cheapest-fit wire cost). Real codecs
trade bits for distortion via λ-RDO. Without it, the codec
cannot be tuned for accuracy/compression trade-off — the lossy
Escape fallback is the only knob and it's binary.

### D-CODEC-4. Stream framing A8 is unwritten. (P1)

`pack_leaf` writes raw `LeafCu` records back-to-back. No frame
boundaries, no CTU markers, no error recovery. Live streaming
needs all three.

### D-CODEC-5. The basin codebook is **not built**. (P1, blocks all loads)

The codec assumes `basin_idx` comes from somewhere. For cognitive
cells the somewhere is `OgitBridge` (downstream). For 3DGS,
attention, SGD — the codebook construction is per-load, k-means
over the carrier, no shared infra yet.

### D-CODEC-6. The lossy Escape fallback is a footgun. (P3)

Owner-review noted the docstring acknowledges the lie. Long-term:
promote to a config field (E10). Short-term: docstring is fine.

### D-CODEC-7. NEWS topology is hard-coded. (P2)

`merge_dir_from_index` in `predict.rs:281` is a 4-way match. The
codec is not generic over topology yet. Plan X1 — exploration.

### D-CODEC-8. No SIMD-batched CTU sweep. (P2)

`predict_intra` is scalar; per-CTU at 64×64 = 4096 cells, the
SIMD opportunity is obvious (16 cells per `F32x16` lane). Deferred
until reference + reconstruction parity test land.

### D-CODEC-9. No `Result`-shaped error variant. (P3)

`pack_leaf` returns `Option<usize>`. Real errors (buffer too
short, mode-decision inconsistency) lose semantics. Promote to
a typed `enum CodecError`.

### D-CODEC-10. The mode 2-bit encoding pins us to ≤4 modes. (P3, architectural)

`pack_header` puts 2 bits at bits 12-13 of u16, leaving 2 reserved
high bits. Future "mode 5" (e.g., a 16-bit Delta variant for
splat) needs to claim bit 14. **Plan the upgrade path in the
design doc before shipping A7.**

## 9. Stack-side technical debt when combining synergies

The harder accounting. PR-X12 fits cleanly into ndarray. But when
we wire the synergies of §§ 4-7, the **existing stack** has debts
that get worse, not better, under multi-load pressure. Honest
catalogue:

### D-STACK-1. `BlockedGrid` block size is fixed at 64×64. (P1 if 3DGS lands)

3DGS tiles in the splat3d crate are 16×16. The codec assumes 64×64
CTUs. The pre-sprint prompt for `pr-x12` aligns them at L1 = 64×64
of cognitive cells. **For 3DGS coefficient compression**, the
natural CTU is one tile = 16×16. Mismatch: either generalise
`Ctu` over block size (preferred, low cost) or maintain two block
formats (technical debt). Decide before Plan E (3DGS codec).

### D-STACK-2. The basin codebook lookup has no SIMD path. (P1)

`SpoDistanceMatrices` at 611M lookups/sec is sequential; the codec
needs **batched** lookup (1 CTU = 4096 cells × 4096 basins = 16M
distance computes). Without SIMD, the encoder is lookup-bound at
~10⁵ CTU/sec. With AVX-512 + AMX, 10⁷ CTU/sec achievable. **Bench
before A6 RDO.**

### D-STACK-3. `MergeDir`'s 4-way alphabet is wire-pinned. (P1 for X1)

`cell_mode_discriminants_match_wire_codes` test pins MergeDir to
`{N=0, E=1, W=2, S=3}` on the wire. If X1 generalises topology to
N=6 or N=8, the wire format breaks. Plan the upgrade with a
version byte in A8 stream framing.

### D-STACK-4. `Fingerprint` is 64-bit only. (P2 for 3DGS)

3DGS basin residual is 96 bits (6 × FP16). Either widen
`Fingerprint` (touches truth/cascade/bf16_truth modules) or
introduce a sibling type for splat (better — keep cognitive
cells fingerprint-typed). The codec is type-generic enough to
not care, but consumers will.

### D-STACK-5. The `splat3d` PRs do not consume `codec`. (P2)

Currently independent. Combining E9's "mode-decide + reduce" trait
requires either (a) a shared trait crate or (b) a refactor of
both. Decide before committing to Plan E.

### D-STACK-6. Lance column substrate exists in `lance-graph`, not `ndarray`. (P1 for HG5)

The HG5 "Lance is the substrate" outcome requires `ndarray::hpc::codec`
to depend on `lance-graph::SpoDistanceMatrices`. Currently ndarray
is the **dependency-bottom** of the stack. Two options: invert
(ndarray depends on lance — wrong, breaks the layering rule from
CLAUDE.md "Architecture Rule"), or introduce a third crate that
both depend on. Probably the latter; needs a sprint.

### D-STACK-7. The cognitive `splat.rs` in `lance-graph-contract` is sacred. (P0, do not touch)

Per the sprint setup: that file is the contract. PR-X12 must never
import or refactor it. If E4 (Fingerprint ≡ 3DGS first-6-floats)
becomes provable, it'll be **tempting** to fold them. Don't. The
abstraction boundary is load-bearing for the cognitive
architecture, even if the bit patterns rhyme.

### D-STACK-8. No backend dispatch in the codec. (P2)

`pack_leaf` is one implementation. EWA, BLAS, MKL all have backend
dispatch (`native` / `intel-mkl` / `openblas` features). The codec
will need: scalar / SIMD / AMX backends for the SIMD-batched CTU
sweep (D-CODEC-8). Plan when D-CODEC-8 lands.

### D-STACK-9. The 4096-basin codebook size assumes "per-frame, reset between frames". (P3)

For attention's KV cache, the "frame" is the (context-window,
batch-element) tuple. For 3DGS, the "frame" is the entire trained
scene (codebook is static after training). For SGD, the "frame"
is one mini-batch. **Three different lifetimes, one type.** Either
generalise lifetime (preferred) or document the discipline (likely).

### D-STACK-10. The current PR-arc cadence is one PR per worker per day. (P2, organisational)

The synergies in §§ 5-7 will require multi-worker coordinated
sprints (e.g., Plan D = 2 workers × 2 weeks). The autoattended
multi-agent protocol scales worker count, but the coordinator's
state machine doesn't currently model multi-week dependencies.
**Update the coordinator agent prompt before kicking off Plan D.**

### D-STACK-11. AVX-512 is mandatory in `.cargo/config.toml`. (P1 for portability)

CLAUDE.md: `target-cpu=x86-64-v4`. Plan F (federated SGD) implies
multi-architecture (NEON workers, AVX2 workers). Either drop the
mandatory AVX-512 or scope federated SGD to AVX-512 nodes only.

### D-STACK-12. The cognitive `Base17` / `NarsTruth` / `TripleModel` types live in `lance-graph`. (P1 for HG3)

HG3 (attention codec) wants to consume cognitive truth values
(NarsTruth) to gate the Escape-or-Skip decision. Same dependency
inversion as D-STACK-6.

### D-STACK-13. No multi-domain benchmark harness. (P0 if we want to claim HG1)

We have splat3d bench, codec tests, SpoDistanceMatrices bench
separately. A combined "single-codec-four-loads" benchmark — one
build, one binary, four scenarios — does not exist. Without it,
HG1 is a claim, not a demonstration. **Build the harness before
the marketing.**

## 10. Sequencing summary

If we commit to all of this, the order matters:

```
                                       Plan A (rANS A7)
                                          │
                                          ▼
              ┌─────────────────┬────────────────────┬─────────────────┐
              ▼                 ▼                    ▼                 ▼
        Plan B (A3-inter)  Plan C (EWA SYRK)   Plan X8 (AMX BF16)   D-STACK-2 (SIMD lookup)
              │                                                       │
              ▼                                                       ▼
         Plan E (3DGS codec) ◄──────────────────────────────── D-STACK-1 (block size)
              │
              ▼
         Plan D (attention codec) ◄──────────────────────────── D-STACK-6/12 (third crate)
              │
              ▼
         Plan F (gradient codec)
              │
              ▼
         HG1-HG6 unlocked
```

Critical path: **A7 rANS** → everything else. Without it, no
benchmark is convincing. Plan A is one worker for one week. Ship
that next; the rest of this doc is just inventory until A7 lands.

## 11. Compaction-preservation note

Per CLAUDE.md § Compaction Preservation, this doc must survive
summarisation. The blackboard entry should reference this file by
path; do not inline the matrix. Key facts to retain across
compaction:

1. PR-X12 A2 + A3-intra shipped in PR-195 (master commits b39a5769,
   b44fe59f). All review comments resolved or outdated.
2. The four-load isomorphism (§ 2) is the architectural claim;
   everything else is sequencing.
3. The critical path is A7 rANS — without it, the codec is academic.
4. The Lance column substrate identity (HG5) is the convergence
   highway; both ndarray and lance-graph land there.
5. The sacred file is `lance-graph-contract/src/splat.rs`. Never
   touch even if the bit patterns rhyme (D-STACK-7).
