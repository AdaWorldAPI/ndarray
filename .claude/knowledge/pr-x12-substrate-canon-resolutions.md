# PR-X12 — Substrate Canon Resolutions

> Date: 2026-05-22
> Status: **canon supplement** — resolves the eighteen open items raised in
> the review of `pr-x12-substrate-merged-canon.md` (PR #196). Additive to
> the canon, not a replacement.
> Reads after the canon. Cite from this doc as `R-N` (resolution N).

---

## 0. How to read this doc

The merged canon (`pr-x12-substrate-merged-canon.md`, master commit
`bc9da4ad`) is the single point of architectural truth. It successfully
fuses session A (`pr-x12-codec-cognitive-substrate-mapping.md`) and
session B (`pr-x12-cross-domain-synergies.md`). What it does NOT yet do
is commit to concrete shapes for the load-bearing pieces. Eighteen items
were raised in review:

- **§3** — five things the merge merged well (confirmed, one-liners)
- **§4** — four items the merge raised in abstraction but did not commit
  (R-1 through R-4 resolutions)
- **§5** — three pieces of detail from session A the merge underrepresented
  (R-5 through R-7 restorations)
- **§6** — three pieces of detail from session B the merge underrepresented
  (R-8 through R-10 restorations)
- **§7** — five commitments missing from both originals and from the
  merge: R-11 through R-13 (latency, flush granularity, federated
  codebook) plus R-14 (formal correctness via `jc` pillars) and R-15
  (`SignatureBasis<DEPTH>` as fifth Plan G lane), the latter two
  surfaced post-merge by the substrate-binding docs

Then five integration pieces that make the resolutions actionable:

- **§8** — the canonical contracts (trait signatures for `PredictiveSignal`,
  `LinearReduce<Basis>`, `Basis<T>`, `CurveOrder<const N>`, `RdoMetric`)
- **§9** — falsifiability matrix (every claim → criterion → test → pass)
- **§10** — sequencing diagram with named gates
- **§11** — end-state + trajectory (think it from the end)
- **§12** — compaction-preservation contract

Citation IDs: `R-1` through `R-15` for resolutions (R-14, R-15
appended post-merge from the substrate-binding doc; numbering remains
append-only). Canon IDs (`M:E-*`, `A:E-*`, `B:E-*`, `M:H-*`, `M:T-*`)
remain stable; this doc adds, does not renumber.

Sister docs (read order):

1. `pr-x12-codec-x265-design.md` — mechanical spec
2. `pr-x12-substrate-merged-canon.md` — architectural fusion (THE canon)
3. **this doc** — resolutions of opens in the canon
4. `pr-x12-codec-cognitive-substrate-mapping.md` — session A archeology
5. `pr-x12-cross-domain-synergies.md` — session B archeology

---

## 1. The end state — think it from the end

Where this lands if every plan ships:

```text
                       ┌────────────────────────────────────────┐
                       │  $ codec-bench --mode video    --input scene.y4m   │
                       │  $ codec-bench --mode splat    --input scene.ply   │
                       │  $ codec-bench --mode kv-cache --input kv.bin      │
                       │  $ codec-bench --mode gradient --input grad.lance  │
                       │                                                    │
                       │  → all four emit compressed Lance columns          │
                       │  → all four meet their threshold (§9)              │
                       │  → all four share ~1.5 KLoC generic codec body     │
                       │  → each ships ~200 LoC of trait impl               │
                       └────────────────────────────────────────┘
```

**Five-category architecture, codec is its own layer:**

```text
ndarray         = hardware       (SIMD, Palette, Base17, SpoDistanceMatrices)
ndarray-codec   = compression substrate  ← extracted via Plan H
                  (Ctu, LeafCu, PredictiveSignal, LinearReduce, CurveOrder, rANS, RDO)
lance-graph     = thinking       (NarsTruth, TripleModel, AutocompleteCache)
causal-edge     = protocol       (CausalEdge64, NarsTables)
p64             = convergence    (where ndarray + lance-graph meet)
```

**Three plug-points factor everything domain-specific out of the codec**
(per M:E-E + R-1 below): Transform basis, Curve order, Escape payload.
Anything domain-specific that does not fit one of these three is a sign
that the abstraction is wrong, not that the codec needs growth.

**Single binary `codec-bench`** is the falsifiability proof of M:H-NEW-1.
The binary, not an argument, demonstrates HG1 / HG6 / M:H-NEW-1. Plan G
(§5 of canon, §10 here) builds it before A7 rANS ships.

This is the end state. The trajectory in §11 is how we get there.

---

## 2. The trajectory

```text
T+0 weeks    Phase 0 starts — substrate gates
   T+0    : Plan H (extract ndarray-codec, 3 days, parallel)
   T+0    : Plan I (PredictiveSignal trait, 3 days, parallel)
   T+0    : Plan A4-design (Transform trait shape, 1 day, parallel)
   T+0    : Plan G (multi-domain bench, 2 weeks, the gate)

T+2 weeks  Phase 0 closes. Plan G binary exists, runs on 4 inputs.
   T+2    : Plan A7 starts (1.5 weeks, CRITICAL PATH)

T+3.5 weeks  Plan A7 lands. Compression-ratio thresholds testable.
   T+3.5  : Plan A4-impl (1 week, parallel)
   T+3.5  : Plan B / A3-inter (1 week, parallel)
   T+3.5  : Plan C / EWA SYRK-batched (1 week, parallel)
   T+3.5  : Plan A6 (1 week, parallel)
   T+3.5  : Plan A8 (1 week, parallel)

T+4.5 weeks  Phase 1 closes. Codec mechanism complete.
   T+4.5  : Plan E (3DGS coefficient codec, 3 weeks, 2 workers)
   T+4.5  : Plan D (attention codec, 2 weeks, 2 workers, can run parallel)
   T+6.5  : Plan F (federated SGD, 4 weeks, 2 workers, after Plan D)

T+10.5 weeks  All four consumer integrations land.
   T+10.5 : Plan G thresholds re-run against all four loads.
   T+10.5 : M:H-1 through M:H-9 all unlocked (or falsified — see §9).
```

Critical path: **Plan G → Plan A7**. Everything else parallelises after
Plan A7. Total ~10.5 weeks of wall-clock work; ~2 workers steady-state
through Phases 0 and 1, ramping to 6 workers in Phase 2.

---

## 3. What the merge merged well (preserved)

Five pieces of synthesis that genuinely emerge from putting A and B
side by side. None appear in either original. Confirmed as canon:

- **M:E-A** — `LinearReduce<Basis>` unifies α-composite / rANS /
  sum-reduce / softmax as the *same* matrix-vector reduce. A's E-4
  (transform = optimizer) and B's E9 (mode-decide + reduce = same
  kernel) collapse into one trait.
- **M:E-D** — Fifth crate category `ndarray-codec`. Both originals
  saw the dep-cycle; neither named the resolution.
- **M:E-G** — `Ctu<const N: usize>` reconciles 64×64 (cognitive) and
  16×16 (splat) at the type level. A treated as invariant; B as debt;
  merge factors via const-generic.
- **M:E-I** — Trait isomorphism (`PredictiveSignal`) over code-folding
  for `splat.rs` vs `Fingerprint`. Shared interface, not shared types.
- **M:E-F** — A7-first critical path BUT commit A4-design trait shape
  first (1 day). Resolves A-vs-B sequencing dispute correctly.

These five are the canon's load-bearing pieces. R-1 through R-13
below resolve what these five did not yet commit.

---

## 4. Resolutions: items the merge raised but did not commit

### R-1 — `LinearReduce<Basis>` and `Basis<T>` trait signatures

**Problem.** M:E-A and M:H-NEW-2 invoke `trait LinearReduce<Basis>` as
the unifying surface but the canon never gives the signature. Without
it, Plan A7 is written against an unknown shape.

**Resolution.** Commit the trait pair at Plan A4-design time (1 day,
Phase 0). The shape:

```rust
/// A basis for a linear reduction. Implementors define a small dense
/// (or sparse) matrix and how to apply it to a `[T; dim()]` input.
///
/// Concrete impls land in their natural homes:
/// - `IdentityBasis<T>`         in `ndarray-codec::basis`
/// - `DctIIBasis<const N>`      in `ndarray::hpc::fft`
/// - `AdamPrecondBasis`         in `burn-codec` (consumer)
/// - `KFACBlockBasis`           in `burn-codec` (consumer)
/// - `ShSpectralBasis<L>`       in `ndarray::hpc::splat3d`
pub trait Basis<T: Copy> {
    /// Dimension of the basis (square: dim()×dim()).
    fn dim(&self) -> usize;

    /// Apply the basis: `dst = B · src`. Caller pre-allocates dst.
    /// Length contract: `src.len() == dst.len() == self.dim()`.
    fn apply(&self, src: &[T], dst: &mut [T]);

    /// Inverse: `dst = B⁻¹ · src`. Same length contract.
    /// For orthogonal bases (DCT, Hadamard) this is `Bᵀ · src`.
    fn invert(&self, src: &[T], dst: &mut [T]);
}

/// A linear reduction over a sequence of symbols against a basis,
/// producing a single output. The kind of reduction depends on impl:
/// - alpha-composite (3DGS rasterizer): RGB blending
/// - rANS-encode (codec A7): state-machine accumulation
/// - sum-reduce (SGD all-reduce): cross-worker summation
/// - softmax (attention): exp-normalize-multiply
pub trait LinearReduce {
    type Symbol: Copy;
    type Output;
    type Basis: Basis<Self::Symbol>;

    /// Reduce a single sequence of symbols against the basis.
    fn reduce(&self, src: &[Self::Symbol], basis: &Self::Basis) -> Self::Output;

    /// Batched reduction: each row is one sequence. Returns one output
    /// per row. Implementors may dispatch to BLAS GEMM for large batch.
    fn reduce_batch(
        &self,
        src: &[&[Self::Symbol]],
        basis: &Self::Basis,
    ) -> Vec<Self::Output>;
}
```

**Why two traits, not one.** The basis is data; the reduction is logic.
Same basis (e.g. DCT-II 8×8) is used by both the transform path (codec
A4) and the EWA splat path (matrix-vector product). Separating lets a
basis ship once and serve many reductions.

**Why no const generic on `Basis::dim()`.** The codec needs to handle
4×4, 8×8, 16×16, 32×32 DCT-II blocks at runtime per CTU split depth.
A const-generic basis would force depth at the type level — wrong
factoring. The compile-time win comes from monomorphising over the
*reduction* type (which is single per consumer); the basis dim is a
runtime knob.

**Falsifies.** If a consumer needs to subclass `LinearReduce` to make
their reduction work (e.g. splat-rasterizer demands access to depth
buffer), the trait factoring is wrong and Plan A7 will accumulate
domain-specific code. Plan G's bench harness is the gate that catches
this — it runs all four reductions through the same trait.

**Cite as R-1 in PR descriptions touching A4 or A7.**

---

### R-2 — Bits 14-15 of the leaf header: cross-load contention

**Problem.** M:E-J claims bits 14-15 of the 16-bit leaf header for
cognitive Pearl-rung metadata (`{Observation, Intervention, Counter-
factual, inter-tier-link}`). A:E-15 had reserved the same bits for
inter-tier reference. The canon does not say what video / splat /
gradient consumers do with these bits, and Plan E (3DGS) ships in
Phase 2 before the reservation is pinned in Plan A8.

**Resolution.** Split the two reserved bits asymmetrically.

```text
bit 15  ──  UNIVERSAL: "has inter-tier reference"
            0 = leaf is self-contained
            1 = leaf refers to a parent-tier `LeafCu` (A3-inter)
            All four consumers respect this bit identically.

bit 14  ──  CONSUMER-TYPED: semantic owned by `ConsumerProfile`
            cognitive: bit 14 = Pearl rung high bit
                       (combined with mode bits 12-13 if rung 4
                        wanted; today rungs 1-3 + reserved = 2-bit
                        encoding using just bit 14)
            video    : bit 14 = 0 (reserved)
            splat    : bit 14 = LOD-cascade-source flag
            gradient : bit 14 = worker-shard parity (for FRC)
```

**Frame header carries the `ConsumerProfile` tag** (Plan A8). 2-bit
field at frame boundary. Decoders route bit-14 interpretation per
profile. Cognitive consumer gets the Pearl-rung high bit; others
reuse bit 14 for their own semantic without protocol break.

**Why not put causal metadata in the frame header instead.** Per-leaf
granularity matters: causal direction can change per cell in a
cognitive scene, but profile is per-frame. Bit 14 must be leaf-local.

**Why not consume both bits per profile.** Bit 15 must stay universal
because A3-inter (cross-tier reference) is generic across consumers —
the LOD cascade applies to all four loads.

**Plan A8 implementation note.** The 2-bit `ConsumerProfile` lives in
the frame header alongside the per-frame basin codebook ref + rANS
frequency table. Decoders mask bit 14 of every leaf header through a
profile-specific demultiplexer before exposing to the consumer.

**Cite as R-2 in A3-inter and Plan A8 PR descriptions.**

---

### R-3 — M:H-NEW-2 LoC budget: actual current count + commitment

**Problem.** M:H-NEW-2 claims `<1.5 KLoC generic codec glue + <200 LoC
per domain consumer`. The canon does not state current LoC nor pin
the budget envelope.

**Resolution.** Measure now, commit the budget envelope, audit per PR.

**Current LoC on master commit `bc9da4ad`** (post PR #195 + PR #196):

| File | Total LoC | Approximate breakdown |
|------|-----------|-----------------------|
| `src/hpc/codec/ctu.rs` | 771 | partition machinery + LeafCu types |
| `src/hpc/codec/mode.rs` | 518 | bit-pack/unpack helpers |
| `src/hpc/codec/predict.rs` | 511 | intra-prediction decision tree |
| `src/hpc/codec/mod.rs` | 38 | re-exports |
| **Total** | **1838** | with tests + doctests + comments |

Of the 1838 total, my read of the files: **~600 lines is non-test,
non-doc-comment generic code**, ~800 lines is inline tests, ~400 lines
is doc-comment / doctest blocks, ~38 lines is mod.rs glue.

**Generic-code LoC currently ~600.** M:H-NEW-2's `<1.5 KLoC generic
glue` budget allows another ~900 lines for A4 (transform), A6 (RDO),
A7 (rANS), A8 (stream), A3-inter (cross-tier).

**Per-sub-card LoC envelope (committed):**

| Sub-card | Generic-glue LoC envelope | Rationale |
|----------|---------------------------|-----------|
| A4 transform | ≤200 | DCT-II + Identity + Transform trait |
| A6 RDO | ≤150 | λ-RDO + RdoMetric trait |
| A7 rANS | ≤300 | encoder + decoder + per-frame freq table |
| A8 stream | ≤200 | framing + ConsumerProfile demux (R-2) |
| A3-inter | ≤100 | extend IntraContext with parent-tier slot |
| **Total budget** | **≤950** | leaves ~50 LoC margin |

**Per-consumer LoC envelope (committed):**

| Consumer | Generic-glue LoC envelope | What ships |
|----------|---------------------------|------------|
| `splat3d::codec` (Plan E) | ≤200 | `impl PredictiveSignal for GaussianSplat` + Morton `CurveOrder` |
| `attention-codec` (Plan D) | ≤200 | `impl PredictiveSignal for AttentionSlot` + token-seq curve |
| `grad-codec` (Plan F) | ≤200 | `impl PredictiveSignal for GradientWeight` + layer-seq curve |
| `video` (Plan G consumer side) | ≤200 | `impl PredictiveSignal for VideoCell` + raster curve |
| **Per-consumer total** | **≤800** | sum across four consumers |

**Audit rule.** Every PR introducing or modifying generic-codec code
must include a one-line generic-LoC delta in the body. If the cumulative
delta exceeds the envelope, the PR escalates to architectural review
(not a CR-style nit; a real "is the abstraction wrong?" question).

**Falsifies M:H-NEW-2 if.** Generic-glue LoC exceeds 1500 after A4-A8
land + at least one consumer integration. That's the falsifiability
condition; tracked in Plan G's metrics report.

**Cite as R-3 in any PR body modifying `src/hpc/codec/`.**

---

### R-4 — Plan G falsifiability thresholds

**Problem.** Plan G ships "a single binary that ingests video / 3DGS /
KV cache / gradient stream and emits compressed Lance columns + ratio
+ reconstruction error". The canon does not name a pass threshold per
load.

**Resolution.** Commit four threshold pairs (compression ratio + quality
floor). Failure to clear any threshold blocks the corresponding consumer
PR landing.

| Load | Reference baseline | Compression target | Quality floor |
|------|-------------------|--------------------|--------------|
| **Video** | x265 `--preset ultrafast` at CRF 23 on Big Buck Bunny 1080p | ≥0.95× reference ratio | PSNR within ±0.1 dB of reference |
| **3DGS** | Inria stock PLY-trim on Mip-NeRF 360 (garden scene) | ≥30× over PLY-trim raw | SSIM ≥ ref − 0.005 at same SH-order |
| **KV cache** | FP16 raw cache, Llama-3-8B-Instruct, 64K context, RULER benchmark | ≥4× over raw FP16 | downstream RULER score loss ≤0.5 % |
| **Gradient** | BERT-large fine-tune on GLUE-MNLI, signSGD baseline | ≥8× over signSGD raw | final validation-loss delta ≤0.5 % |

**Three-way pass criterion** per load:

1. **Ratio threshold cleared** — measured during Plan G run
2. **Quality floor cleared** — measured during reconstruction
3. **Per-consumer LoC envelope respected** — per R-3 audit

All three must pass for the consumer's holy-grail claim to count as
demonstrated rather than asserted.

**Sub-threshold = blocker.** If any of (ratio, quality, LoC) fails for
a load, the corresponding consumer plan (D / E / F / video) cannot
claim "complete". The merged canon's M:H-1 through M:H-9 are then
provably partial; only the cleared loads count.

**Why these thresholds and not stricter.** Conservative initial bars:
- Video at parity with x265 ultrafast is meaningful (PR-X12 is supposed
  to *generalise* x265, not beat it at its specialty)
- 30× over Inria PLY-trim is the floor for "this changes 3DGS streaming"
- 4× KV-cache compression at <0.5% accuracy = passes the smell test
  against StreamingLLM / H2O / SnapKV
- 8× gradient over signSGD = roughly the rANS theoretical floor for
  heavy-tail-distributed gradients

**Stretch targets** (recorded separately, not blocking):
- Video at 1.5× x265 ultrafast at same PSNR (would justify HG1 strongly)
- 3DGS at sub-1-bit/Gaussian (M:H-6 / B:HG2 — see R-10 for math)
- KV cache at 8× (matches the FlashAttention-3 ceiling)
- Gradient at 16× (peer-reviewed federated-SGD upper bound)

**Cite as R-4 in Plan G's PR description; the binary's `--threshold`
flag must enforce all four pass criteria.**

---

## 5. Restored detail from session A

### R-5 — DCT-II vs GEMM crossover at 64 blocks (from A:§5.3)

**Problem.** The merge punts to Plan A4-impl without preserving the
operational decision rule for transform dispatch.

**Resolution.** Pin the crossover number in Plan A4-impl's spec.

**Decision rule for A4 transform dispatch:**

```text
N = number of contiguous transform blocks to apply

if N <  64:  per-block butterfly path
             ~80 ops/block for 32×32 DCT-II via Loeffler/Lengwehasatit
             Fits L1 trivially; no batching cost

if N >= 64:  batched GEMM path
             ~32K ops/block (matrix form) but 256 blocks/cycle in AMX bf16
             ~128 KB working-set, fits L1
             Amortises hardware fusion + reduces dispatch overhead

Crossover empirically at ~64 blocks on Sapphire Rapids; calibrate
per architecture during A4-impl.
```

**Why crossover at 64.** AMX TDPBF16PS does one 16×16 BF16 tile per
cycle. 64 blocks at 32×32 → 256 tile operations → ~256 cycles for
batched GEMM. The per-block butterfly at 80 ops/block × 64 blocks =
5120 ops, which at ~4 IPC = 1280 cycles. Crossover is approximate;
real measurement during A4-impl pins per-arch.

**Per-architecture override matrix (Plan A4-impl deliverable):**

> **[UNCALIBRATED ESTIMATES]** (audit #6, marked 2026-07-16): the crossover
> numbers below are pre-bench heuristics with no measurement source. Plan
> A4-impl / Plan G produce the real numbers; until then these are hypotheses,
> not commitments.

| Architecture | Per-block path | Crossover N | Batched path |
|--------------|----------------|-------------|--------------|
| Sapphire Rapids (AMX-BF16) | Loeffler 1D + transpose | ~64 | AMX TDPBF16PS via `bf16_tile_gemm` |
| Skylake-X / Ice Lake (AVX-512F) | Loeffler 1D + transpose | ~32 | AVX-512 ZMM batched DCT |
| Zen 4 (AVX-512) | Loeffler 1D + transpose | ~96 | AVX-512 ZMM (no AMX) |
| Apple Silicon (NEON) | Loeffler 1D | ~256 | NEON 4×4 GEMM via `bf16_tile_gemm` NEON stub |

**Cite as R-5 in A4-impl PR descriptions.**

---

### R-6 — SSD reformulation for VNNI block-match ME (from A:E-7)

**Problem.** Merge cites "Block-matched ME via i8gemm" without the
SSD reformulation math. That math is what *proves* ME goes through
BLAS at all; without it the BLAS-synergy claim is decorative.

**Resolution.** Restore the math and the speedup citation.

**SAD (HEVC native) — not a GEMM:**

```text
SAD(A, B) = Σ_{ij} |A_{ij} - B_{ij}|
```

The absolute-value inside the sum has no matrix shape.

**SSD (PR-X12 reformulation) — has a GEMM:**

```text
SSD(A, B) = Σ_{ij} (A_{ij} - B_{ij})²
         = Σ A_{ij}² - 2·Σ A_{ij}·B_{ij} + Σ B_{ij}²
         = ||A||² - 2·(A·B) + ||B||²
                       ▲
                       │
                       └── this term IS a GEMM
```

**For N motion-vector candidates** at one reference block:

```text
Candidates  A_1, A_2, ..., A_N     each 16×16 = 256 pixels = 256-d vector
Reference   B                       16×16 = 256-d vector

Middle term: A_batch @ B            (N×256) @ (256×1) = N×1
                                    one GEMV; or for batched ME
                                    over multiple reference blocks,
                                    N×K matrix.

||A_i||²    precomputed once per candidate window
||B||²      precomputed once per reference
```

**VNNI VPDPBUSD throughput:** 64 i8·i8 → i32 dot-product ops per cycle
on Cascade Lake+ . One 256-element dot product = 4 VPDPBUSD ops = ~4
cycles. Vs hand-tuned SAD via VPSADBW: ~8 cycles per 16-pixel row, so
~128 cycles per 16×16 SAD. **Speedup: ~32× to ~50× depending on
batch dispatch.**

**Implication for PR-X12 E-7 (block-matched ME via i8gemm):** ME path
in A4 or A5 ships as a `batched_ssd_search` primitive in `ndarray::hpc::
blas_level2` that downstream consumers (video, splat scene flow) call
into. **Not a codec-specific function** — landing in BLAS L2 keeps the
factoring clean (codec uses the math; BLAS owns the math).
**[PLANNED symbol — `blas_level2.rs` today exports only the 8 classical
BLAS-L2 methods; `batched_ssd_search` does not exist yet (audit #5,
re-verified 2026-07-16). Do not cite as an existing API.]**

**Cite as R-6 in any ME-path or splat scene-flow PR description.**

---

### R-7 — CTU partition as tropical-GEMM (from A:§13.3)

**Problem.** Merge mentions "tropical-GEMM" in §11 Phase 3 but drops
the `O(4^d) → O(d²)` complexity bound. That bound is the architectural
justification for the `lance-graph::blasgraph` dependency.

**Resolution.** Restore the complexity argument and pin the algorithm.

**HEVC's recursive partition RDO:**

```text
For each CTU at depth d:
  for each of 4 children:
    recursive RDO at depth d+1
  combine children's costs

Time: O(4^d) where d = max split depth (4 in PR-X12, giving 256 nodes
worst case per CTU)
```

**Tropical-semiring formulation (R-7 commitment):**

```text
1. Represent the 85-node tree as a DAG (parent → child edges).
2. Edge weights W[parent, child] = ΔRDO cost of choosing child.
3. Compute shortest-path costs to every node via matrix relaxation:

     D ← min(D, D + W)     ← tropical-GEMM iteration

   Repeat for d iterations where d = depth.
4. Optimal partition = argmin_n D[root, n] for n in leaf nodes.

Time: O(d² × |nodes|) using batched tropical-GEMM on `lance-graph::
blasgraph`. For d=4, |nodes|=85: O(16 × 85) = O(1360) ops per CTU.
Vs. O(4^4 × |nodes|) = O(21,760) ops for the naive recursive RDO.
```

**Speedup: ~16×.** For a 4K frame at ~2,040 CTUs (corrected 2026-07-16 —
the earlier "~132K CTUs" was the 8×8 leaf count), this is the difference
between ~2.8 ms and ~44 ms per frame just for partition RDO at ~1 op/ns.
At 60 fps, that's the difference between fitting and missing the budget.

**Why this targets `lance-graph::blasgraph`:** Standard BLAS GEMM uses
(× , +) semiring. Tropical uses (+ , min) semiring. blasgraph is the
**canonical, bit-exact** kernel home for semiring algebra. *(Corrected
2026-07-16, audit #1-#3: blasgraph today exports 7 HDR semirings over
16384-bit BitVec — XorBundle, BindFirst, HammingMin, SimilarityMax,
Resonance, Boolean, XorField — none of which is a numerical min-plus
over weighted f32 edges. The claim "blasgraph already ships
tropical-GEMM kernels" was wrong; the f32 tropical-GEMM kernel is
UNWRITTEN and lands in blasgraph when A6 wires it.)* Cross-repo dep
from ndarray-codec → lance-graph::blasgraph (after Plan H extraction,
this is dep-allowed because ndarray-codec is a sibling, not the bottom).

**Shipped min-plus today (corrected).** The only shipped min-plus
primitive is the method `bgz17::ScalarCsr::spmv_min_plus`
(`fn(&self, x: &[f32]) -> Vec<f32>`, `crates/bgz17/src/scalar_sparse.rs:98`).
The earlier citation `bgz17::scalar_sparse::tropical_spmv(edge_weights, dag)`
named a free function that **does not exist** (audit #2). bgz17 is a lossy
sibling encoding stack — its CSR may serve as an A6 prototype adapter, but
substituting it for the bit-exact blasgraph canon is a soundness violation,
not a re-targeting (audit ground-truth #5).

**Plan A6 RDO (1 week) ships this.** The λ-RDO knob (per A:§10.3) and
the tropical-GEMM partition solver are the same kernel: λ scales the
edge weights, the relaxation computes the optimal mode tree.

**Cite as R-7 in Plan A6 PR description; required reading for anyone
touching `RdoConfig` or `predict_intra` policy.**

---

## 6. Restored detail from session B

### R-8 — Plan G framing: confidence-degradation gate

**Problem.** Merge promoted my B:D-STACK-13 (no multi-domain bench
harness) to Plan G + M:E-H but lost the rationale for *why* it goes
in Phase 0 vs Phase 1.

**Resolution.** Make the framing explicit in canon and in this doc.

**46 debt items across A's T-1..T-23 and B's D-CODEC-1..10 + D-STACK-
1..13. 45 of them degrade either performance or correctness:**

- A:T-1, T-2, T-7: correctness (already-fixed CodeRabbit findings) or
  performance (SIMD-batched encode)
- B:D-CODEC-1..10: correctness (cross-tier, RDO, stream framing) or
  performance (no SIMD batch)
- B:D-STACK-1..12: performance (block-size mismatch, SIMD lookup) or
  correctness (sacred file, mandatory AVX-512)

**One debt item — B:D-STACK-13 — degrades *confidence*:**

> Without a single-binary four-loads benchmark, the entire architectural
> claim is unproven. Every other debt item degrades performance or
> correctness; this one degrades **confidence**. (B's original framing.)

**Implication for sequencing.** Performance/correctness debt is
incremental and recoverable; confidence debt is foundational and
self-reinforcing. A single performance regression makes the codec
slow; a single confidence gap makes every other resolution
unverifiable. Plan G must precede A7 because:

1. If A7's trait shape is wrong, fixing it after A7 ships is 4-8x
   the cost of getting it right under bench pressure
2. If the architectural claim is wrong, no amount of A7 perf work
   makes it right
3. "Two weeks of bench-harness work front-loaded saves six months of
   trait-shape rework" — original B framing, preserved.

**Plan G is the unfalsifiability gate.** Without it, M:H-1 through
M:H-9 are claims. With it, they are demonstrably true or
demonstrably false against the R-4 thresholds.

**Cite as R-8 in Plan G's PR body; the framing belongs in the body,
not buried in commit messages.**

---

### R-9 — `MergeDir` is topology-FREE, not just topology-generic

**Problem.** Merge folds B:E1 into M:E-B (`trait CurveOrder`) but
weakens the claim. M:E-B says "different curve, same kernel" — implies
a curve still exists at the codec layer. B:E1's stronger claim: the
4-way alphabet has *no spatial semantics at all* at the codec layer.

**Resolution.** Pin the topology-free contract on `PredictiveSignal`.

**The codec layer sees neighbours as `(slot_0, slot_1, slot_2, slot_3)`.
Period.** No `MergeDir::North/East/West/South` semantic labels exist
inside the codec. Consumers attach semantic labels *outside* the codec
boundary.

**`PredictiveSignal::neighbours` contract:**

```rust
pub trait PredictiveSignal {
    /// Returns the 4 neighbour slots in implementation-defined order.
    /// The codec NEVER interprets "slot 0" as "north" or any direction.
    ///
    /// Implementor semantic:
    /// - cognitive: slot 0 = N, slot 1 = E, slot 2 = W, slot 3 = S
    /// - splat:     slot 0 = prev-Morton, slot 1 = next-Morton,
    ///              slot 2 = parent-LOD, slot 3 = child-LOD
    /// - attention: slot 0 = prev-token, slot 1 = next-token,
    ///              slot 2 = prev-head,  slot 3 = next-head
    /// - gradient:  slot 0 = prev-iter,  slot 1 = next-iter,
    ///              slot 2 = prev-layer, slot 3 = next-layer
    ///
    /// The codec writes `MergeDir = slot index (0..=3)`. Consumers
    /// reinterpret on decode. No spatial semantic crosses the boundary.
    fn neighbours(&self) -> [Option<Self::NeighbourRef<'_>>; 4];

    type NeighbourRef<'a> where Self: 'a;
}
```

**Implication for Plan I (PredictiveSignal trait, 3 days, Phase 0).**

- The codec body never has "`if dir == North { ... }`" anywhere
- The 4-slot neighbour array is treated as an opaque categorical
- `MergeDir` enum becomes a *consumer-side* name for slot indices,
  exposed via `mode.rs::pack_merge_dir(MergeDir) -> u8` but never used
  in the predict / RDO / stream / rANS paths

**Why this is stronger than M:E-B.** `CurveOrder` says "different curve,
same kernel" — the curve is an attribute of the consumer's data layout.
Topology-free goes further: even *with* a curve, the codec doesn't see
it. The curve exists only in `nearest_basin` resolution (consumer-side)
and `escape_vector_decode` (consumer-side).

**Falsifies if.** Any codec-body code references slot 0 / 1 / 2 / 3 by
semantic name (north / east / etc.). The grep for that pattern is the
audit. Currently `predict.rs` does this in tests but never in code; the
production path is already topology-free. Keep it that way through A6 /
A7 / A8.

**Cite as R-9 in Plan I PR description and in any future codec-body PR
that touches `predict_intra`.**

---

### R-10 — Sub-1-bit/Gaussian math breakdown (from B:HG2)

**Problem.** B:HG2 / M:H-6 claim sub-1-bit/Gaussian 3DGS compression.
Neither my original nor the merge back-of-envelopes this. The claim
floats without justification.

**Resolution.** Commit the factor breakdown; mark sub-1-bit as
*stretch*, ~4 bits/Gaussian as the *floor* (R-4 quality floor).

**Stock 3DGS-PLY baseline (Inria trim):** ~50 bytes/Gaussian.

**Factor 1: k-means palette mode coding (≈10×)**

Most Gaussians in a trained scene cluster around a few hundred
"archetype" (color, scale, opacity) tuples. After k-means basin
assignment + Skip-heavy mode coding (flat regions all Skip):

- Stock: 50 bytes/Gaussian = 400 bits
- After mode coding: ~40 bits/Gaussian average (Skip=16, Merge=24,
  Delta=24, Escape=48; with 60% Skip, 20% Merge, 15% Delta, 5% Escape):

```text
0.60 × 16 + 0.20 × 24 + 0.15 × 24 + 0.05 × 48 = 9.6 + 4.8 + 3.6 + 2.4 = 20.4 bits
```

After this factor: **~20 bits/Gaussian = 2.5 bytes/Gaussian = 20× over PLY.**

**Factor 2: rANS entropy coding (≈3×)**

Mode-distribution is heavy-tailed (60% Skip, 20% Merge, etc.). rANS
entropy of that distribution:

```text
H = -(0.60 log₂ 0.60 + 0.20 log₂ 0.20 + 0.15 log₂ 0.15 + 0.05 log₂ 0.05)
  = -(0.60 × -0.737 + 0.20 × -2.322 + 0.15 × -2.737 + 0.05 × -4.322)
  = 0.442 + 0.464 + 0.411 + 0.216
  = 1.533 bits per mode tag
```

Vs 2 bits flat for the mode tag. Savings on the mode field: 2 → 1.5 bits.
Savings on the basin field (heavy-tail): 12 → ~6 bits. Savings on the
8-bit delta (also heavy-tail): 8 → ~5 bits.

Per-Gaussian average after rANS: ~7 bits.

After this factor: **~7 bits/Gaussian = 5.7× over factor-1 = ~57× over PLY.**

**Factor 3: SH-residual cross-LOD prediction (≈2×)**

L=2 and L=3 SH coefficients are highly predictable from L=0 and L=1.
A linear basis (R-1's `Basis<T>`) for SH spectral prediction reduces
L=2/L=3 residuals to near-zero in flat regions. Skip-mode dominates
SH ≥ L=2 coefficients in trained scenes.

Per-Gaussian average after SH cross-prediction: ~4 bits.

After this factor: **~4 bits/Gaussian = ~100× over PLY.**

**Where the stretch comes from (sub-1-bit):**

- **Factor 4a (≈2×)**: Per-asset codebook training (offline). Today
  the basin codebook builds per-frame. For 3DGS, a single trained
  scene = one asset = one codebook. Offline-trained codebooks
  eliminate per-frame codebook overhead in the wire format. Gets to
  ~2 bits/Gaussian.
- **Factor 4b (≈2×)**: Higher-order rANS context modeling (CABAC-style
  or tiny-transformer per A:E-9). Per-mode-given-neighbour-mode
  probabilities are far more concentrated than per-mode marginals.
  Gets to ~1 bit/Gaussian.
- **Factor 4c (≈2×)**: Inter-frame coding for video-of-3DGS scenes
  (Plan E2, post-MVP). Per-frame delta from previous frame's
  reconstruction. Gets to ~0.5 bit/Gaussian.

**Honest near-term target: ~4 bits/Gaussian (factor 1+2+3).** That's
**100× over PLY trim, 12.5× over the R-4 floor of 30×.**

**Stretch target: ~1 bit/Gaussian.** Requires factor 4a (offline
codebook training, ~1 week) + 4b (CABAC-style context, ~2 weeks) =
3 weeks beyond Plan E baseline.

**Sub-1-bit target: ~0.5 bit/Gaussian.** Requires factor 4c (inter-frame
coding) which is a Plan E2 or later.

**Cite as R-10 in Plan E PR description and Plan G's `--mode splat`
threshold doc.**

---

## 7. New commitments missing from both originals and from the merge

### R-11 — Per-CTU encoder latency budget

**Problem.** Neither doc nor the merge states ms-per-CTU at 60 fps 4K.
Without it, B:D-CODEC-8 / A:T-7 (no SIMD-batched encode) have no
falsifiability criterion.

**Resolution.** Commit the budget; pin the SIMD-batched-encode debt
to the budget.

**4K @ 60 fps frame budget:**

```text
4K = 3840 × 2160 = 8.3 M pixels
60 fps = 16.67 ms/frame
At 8×8 leaf granularity (HEVC's smallest CU; the unit at which the
encoder's inner-loop work is paid):
                              129,600 leaves/frame (exact: 3840·2160/64)
                              (padded 64×64 accounting: 60×34 = 2,040
                               CTUs/frame → 130,560 leaves at max split)
Per-leaf budget: 16.67 ms / 129,600 = ~129 ns/leaf
(Corrected 2026-07-16: the earlier 132,710 figure — 130,560 plus an
unsourced "~1.6 % chroma alignment bias" — was not numerically
grounded; use exact 129,600 or padded 130,560.)
```

**Encoder per-leaf breakdown (scalar reference, current):**

| Stage | Scalar cost | SIMD-batched target |
|-------|-------------|---------------------|
| basin lookup (4096 entries, Hamming dist) | ~800 ns | ~50 ns (SIMD batched) |
| mode decide (Skip → Merge → Delta → Escape) | ~80 ns | ~80 ns (already cheap) |
| header pack (`pack_header`) | ~5 ns | ~5 ns |
| transform (A4, 8×8 DCT-II butterfly) | ~30 ns | ~30 ns |
| quantize (i8 round) | ~5 ns | ~5 ns |
| rANS encode (A7) | ~40 ns | ~40 ns |
| **Total per-leaf** | **~960 ns** | **~210 ns** |

**At scalar reference (960 ns/leaf): 4K @ 60 fps requires 129,600 ×
960 ns = 124 ms/frame. Misses 60 fps by ~7.5×.**

**At SIMD-batched (210 ns/leaf): 129,600 × 210 ns = 27 ms/frame. Misses
60 fps by ~1.6×; needs further work but in the same order of magnitude.**

**To hit 60 fps 4K real-time** requires the SIMD-batched-encode path
to land. **This pins B:D-CODEC-8 / A:T-7 from P2 to P1.** Plan A4-impl
and Plan A6 should both ship with SIMD-batched paths, not scalar
reference only.

**Implication for Plan G.** The `--mode video` threshold (R-4)
includes a latency assertion: total encode time for the Big Buck Bunny
1080p clip must complete within (clip duration × 0.5). At 1080p that's
~32,400 leaves/frame × 210 ns × 30 fps = ~204 ms/sec, well within
budget. 4K is the stretch target.

**Cite as R-11 in any encoder-path PR description; the latency
budget is the gate that determines whether SIMD-batched encode is P0
or P1.**

---

### R-12 — Streaming-buffer flush granularity

**Problem.** Neither doc nor the merge says: per-CTU? per-frame? per-GOP?
Different answers make Plan A8 substantially different shapes.

**Resolution.** Commit per-CTU as the default; per-bucket for Plan F.

**Per-CTU flush (committed default; CTU = 64×64 cells, so 4096 cells/CTU,
2,040 CTUs/frame at 4K and ~510 CTUs/frame at 1080p):**

```text
Buffer size:   ~12 KB per CTU
                 = 4096 cells × avg 3 bytes (mode-distribution per R-10)
Flush rate:    ~122,400 flushes/sec at 4K 60 fps  (2,040 CTUs/frame × 60)
               ~30,600 flushes/sec at 1080p 60 fps (510 CTUs/frame × 60)
Latency:       sub-ms per CTU; consumer can start decoding the first
               CTU before encoder finishes the frame
```

**Why per-CTU and not per-frame:**

- per-frame buffer = ~1.5 MB; latency cost = 16.67 ms (one frame
  latency added to encode-decode pipeline)
- per-GOP buffer = ~25 MB at 16-frame GOP; latency = 267 ms,
  unacceptable for live attention / KV-cache use cases
- per-CTU = ~12 KB; latency = ~125 ns

**Per-bucket override for Plan F (federated SGD):**

```text
Bucket = 4096 weights (one BlockedGrid L1 block of gradients)
Buffer size: ~12 KB per bucket (same envelope as per-CTU)
Flush rate:  per-iteration, per-bucket
Latency:     bucket-local; all-reduce happens after bucket flush
```

**Wire format implication:** A8 frame header has a `FlushUnit` tag
(2-bit field):

```text
FlushUnit::Ctu      → 00 (default, video / splat / attention)
FlushUnit::Bucket   → 01 (gradient SGD)
FlushUnit::Frame    → 10 (offline batch encode)
FlushUnit::Reserved → 11
```

**Plan A8 implementation note:** Flush granularity lives in the frame
header alongside `ConsumerProfile` (R-2) and the per-frame basin
codebook ref. Stream readers route on `FlushUnit` for buffer
allocation.

**Cite as R-12 in Plan A8 PR description and Plan F PR description.**

---

### R-13 — Basin codebook distribution policy for Plan F

**Problem.** Plan F is 2 weeks × 2 workers; the merge doesn't address
whether the 4096-entry codebook is replicated across workers or
partitioned. Either answer is fine; not deciding makes Plan F
undefined.

**Resolution.** Commit Option A (per-shard codebook) for Plan F v1;
list alternatives as Phase 3 exploration.

**Option A — Per-shard codebook (Plan F v1, committed):**

```text
Each worker holds 1 parameter shard, builds its own 4096-entry codebook
over its shard, encodes its gradients against its own codebook.
Wire format: each LeafCu carries (worker_id, basin_idx) in the per-frame
escape vector lookup. No cross-worker comm during codebook build.

Pro:  zero cross-worker codebook-build comm
      worker independence
      no global codebook drift
Con:  loses cross-shard correlation (Merge-mode never fires across shards)
      may compress worse than Option B by 1.5-2× per parameter
```

**Wire format extension for Option A:**

```text
Frame header (per worker, per iteration):
  FlushUnit::Bucket
  ConsumerProfile::Gradient
  WorkerId: u16              ← NEW: per-shard codebook index
  CodebookHash: u64          ← integrity check
  rANS frequency table
```

**Option B — Replicated codebook (alternative, Phase 3):**

```text
One global 4096-entry codebook, all workers consume identical codebook.
Cross-worker codebook-build comm: one all-reduce per epoch.

Pro:  Merge-mode fires across shards (cross-parameter correlation)
      better compression by 1.5-2×
Con:  cross-worker codebook-build comm cost
      codebook stale-ness if epoch boundary misses a parameter
      complex resync after worker failure
```

**Option C — Hierarchical codebook (Phase 3+):**

```text
Per-shard codebook + global "override" codebook (256 entries) for the
heavy-hitters that cross shards.
LeafCu first checks global override; falls through to per-shard.

Pro:  best compression in expectation (combines A and B)
Con:  complex protocol; requires global hot-set tracking
      worker-failure recovery non-trivial
```

**Plan F v1 commits Option A.** v2 (post-stability) evaluates Option B
empirically; v3 (research-grade) tries Option C.

**Falsifies if.** Option A on BERT-large fine-tune fails to clear the
R-4 gradient threshold (8× compression at <0.5% loss delta). At that
point, Plan F v1 escalates to Option B in a follow-up PR.

**Implementation primitives (current substrate, no new code required):**

| Concern | Crate / module |
|---------|----------------|
| Codebook training (k-means + CAM-PQ) | `ndarray::hpc::cam_pq::CamCodebook` (`train_geometric` / `train_semantic` / `train_hybrid`) |
| Deployed encoding format (per-shard) | `lance-graph::bgz-tensor::Codebook4096` and the `bgz-hhtl-d` shared-palette variant |
| Online plastic updates (`SharedClusterWide`) | `ndarray::hpc::dn_tree` (quaternary plastic memory, partial-Hamming descent) |
| Integrity proof for distributed updates | `ndarray::hpc::merkle_tree` (Blake3-48-bit, 1 KB root, `xor_diff` panCAKES compression) |
| Gossip protocol (cluster-wide) | `q2` (external — implements the wire protocol) |

The four policy modes (`LocalEphemeral` / `SharedClusterWide` /
`SharedRegional` / `PretrainedStatic`) compose these primitives
differently; the codec body exposes a `CodebookHandle` trait, and the
primitives plug in via that trait. **PR-X12 contributes the wire format
+ trait + Option A; the primitives above already exist.**

**Cite as R-13 in Plan F PR description.**

---

### R-14 — Formal correctness via `lance-graph::jc` pillars

**Problem.** Canon and resolutions describe the codec's empirical
behaviour (R-4 thresholds, R-11 latency) but never name the formal
correctness proofs the substrate already carries. Without a citation,
"the codec is correct" is unverifiable; with citations, the codec
inherits machine-checked guarantees from existing crates.

**Resolution.** Pin both pillars and what each proves.

**Two formal proofs in `lance-graph::jc`:**

- **Quantization correctness (Pillar 10, Pflug-Pichler):**
  nested-distance Lipschitz on Sigma DN-trees. Proves that CAM-PQ tree
  quantization preserves the FreeEnergy functional within a Lipschitz
  factor Lε. **This is the proof PR-X12 cites for "wire-format
  quantization is faithful."** Implementation: `jc::pflug` (active in
  default build, zero-dep).
- **Path-signature correctness (Pillar 11, Hambly-Lyons):**
  signature uniqueness on tree-quotient. Proves that any path of
  bounded variation is uniquely determined by its truncated signature
  up to tree-like equivalence (Annals of Mathematics 171(1):109–167,
  arXiv:math/0507536). **This is the proof PR-X12 cites for the
  `SignatureBasis<DEPTH>` lane (R-15).** Implementation:
  `jc::hambly_lyons` (active under `--features hambly-lyons`, since
  PR #348 landed on 2026-05-07).

**What the codec inherits.** Both pillars exist; the codec cites them
and does not reprove. R-4's "Quality floor" rows for video / KV /
gradient inherit Pillar 10's Lipschitz bound automatically. R-15's
signature-lane gates on Pillar 11.

**Status.**

- Pillar 10: active in default zero-dep build.
- Pillar 11: active under `--features hambly-lyons`; passes its probe
  (forward < 1e-9, converse > 0.05, discrimination ratio ≥ 1e6 over
  N=100 random pairs in d=3 at depth-2).
- Production-scale benchmarking remains open — see Gap G-4 in
  `pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md`. *(Corrected
  2026-07-16, audit #9: the "PR #350 Goursat-PDE math correction"
  claim is withdrawn — `signature_kernel_pde`'s own tests prove
  convergence to `I_0(2·√⟨u,v⟩)` at `rel<1e-3` with O(1/N) refinement;
  there is no known bug. Pillar 11's probe uses `signature_truncated`
  as a design choice, not a workaround.)*

**Falsifies if.** Pillar 10 ever flips state (a regression in the
Pflug-Pichler proof bound) — Plan G's video / KV / gradient quality
floors lose their formal underwriting and become empirical-only.

**Cite as R-14 in any PR claiming "codec output is faithful to
input" or wiring `SignatureBasis` (R-15).**

---

### R-15 — `SignatureBasis<const DEPTH: usize>` as `Basis<f32>` impl

**Problem.** R-1 commits the `Basis<T>` shape; the canon lists three
concrete impls (`DctIIBasis<N>` for video, `EwaSplatBasis` for 3DGS,
`ShSpectralBasis<L>` for splat SH). No `Basis<T>` impl targets
*streams* — audio waveforms, time-series, gesture/handwriting paths.
Plan G has only four lanes; path-structured signals are unaddressed.

**Resolution.** Commit `SignatureBasis<const DEPTH: usize>: Basis<f32>`
as the fifth concrete impl, wrapping the path-signature kernel from
the external `lance-graph::sigker` crate.

```rust
// Concrete impl, lives in ndarray::hpc::signature (new module, ~1 wk)
impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH> {
    fn dim(&self) -> usize { /* truncated tensor-algebra dim at DEPTH */ }
    fn apply(&self, path: &[f32], signature: &mut [f32]) {
        // iterated-integral truncation against sigker::signature_truncated
    }
    fn invert(&self, _sig: &[f32], _path: &mut [f32]) {
        // signature → path is many-to-one (tree-quotient); document as N/A
        unimplemented!("signature inversion is N/A — path unique only up to \
                        tree-like equivalence per R-14 / Pillar 11")
    }
}
```

**Why `signature_truncated` and not `signature_kernel_pde`.** Design
choice: the tensor-algebra path (`signature_truncated`) is what jc
Pillar 11 cites directly, so R-15 wraps it. *(Corrected 2026-07-16,
audit #9: the earlier "known math bug — Goursat-PDE form diverges"
rationale was false. `sigker/src/kernel.rs`'s tests
`pde_kernel_converges_to_closed_form_for_linear_paths` and
`pde_and_truncated_agree_on_linear_paths_in_the_limit` pass; both
forms compute the same kernel. Either is usable.)*

**Plan G gets a fifth lane.** "Stream signal" mode:

- Input: audio waveform / time-series / gesture stream
- Codec: `SignatureBasis<DEPTH=3>` truncates path signature, residuals
  go through standard rANS via the four-mode taxonomy
- Quality floor: signature-uniqueness preservation per Pillar 11
- Compression target: ~10× over raw f32 path samples (estimate;
  calibrate during Plan G)

**Falsifies if.** `SignatureBasis<DEPTH=3>` plus rANS fails to
reconstruct the path within ε under Pillar 11's discrimination ratio.
At that point, raise DEPTH or fall back to per-block DCT-II for the
stream lane.

**Cost.** ~1 week wrapper around `sigker::signature_truncated` +
basis-trait plumbing + Plan G fifth-lane wiring.

**Cite as R-15 in any PR adding a stream-signal codec lane or
wiring `SignatureBasis`.**

---

## 8. The canonical contracts — concrete trait signatures

All three plug-points (per M:E-E) get concrete signatures here. These
are the contracts Plans G / H / I / A4-design commit to in Phase 0.

```rust
// ────────────────────────────────────────────────────────────────────
// Plug-point 1: PredictiveSignal — what the consumer ships
// ────────────────────────────────────────────────────────────────────

/// Implemented by each domain's per-element data type:
/// - cognitive cell `Fingerprint`
/// - 3D Gaussian splat tuple
/// - attention slot `(Q, K)` pair
/// - gradient weight `(param_id, ∂L/∂w)`
///
/// Single trait surface; ~50 LoC per consumer impl.
pub trait PredictiveSignal: Copy + Eq {
    /// Basin codebook entry type. Often the same as `Self` (e.g.
    /// cognitive: Fingerprint ↔ Fingerprint), but consumers like
    /// gradient may use a tuple like `(GradientPattern, magnitude)`.
    type Basin: Copy + Eq;

    /// Residual after subtracting the nearest basin. Should fit
    /// in i8 when "Delta-mode worthy".
    type Residual: Copy;

    /// What lives in the per-frame escape vector. Stock 3DGS:
    /// `[f16; 48]` for SH≥L=2 + (μ, scale, rot, opacity, color).
    /// Cognitive: `u64` Fingerprint.
    /// Attention: `[f16; head_dim]`.
    /// Gradient: `f32`.
    type Escape: Copy;

    /// Find the nearest basin in the codebook.
    /// Returns (basin_idx, residual). basin_idx must be ≤ MAX_BASIN_IDX.
    fn nearest_basin(&self, codebook: &[Self::Basin]) -> (u16, Self::Residual);

    /// Is this residual small enough for Delta mode (fits i8)?
    fn fits_delta(residual: &Self::Residual) -> bool;

    /// Encode residual as the u8 byte that goes into the LeafCu.
    fn pack_residual(residual: &Self::Residual) -> u8;

    /// Type-erased neighbour reference (consumer-defined topology).
    /// Codec NEVER interprets slot semantics — per R-9.
    type NeighbourRef<'a>: Copy where Self: 'a;
    fn neighbours(&self) -> [Option<Self::NeighbourRef<'_>>; 4];

    /// Convert self into the escape payload (for Escape-mode encode).
    fn to_escape(&self) -> Self::Escape;
}

// ────────────────────────────────────────────────────────────────────
// Plug-point 2: Basis<T> + LinearReduce — per R-1
// ────────────────────────────────────────────────────────────────────

pub trait Basis<T: Copy> {
    fn dim(&self) -> usize;
    fn apply(&self, src: &[T], dst: &mut [T]);
    fn invert(&self, src: &[T], dst: &mut [T]);
}

pub trait LinearReduce {
    type Symbol: Copy;
    type Output;
    type Basis: Basis<Self::Symbol>;

    fn reduce(&self, src: &[Self::Symbol], basis: &Self::Basis) -> Self::Output;
    fn reduce_batch(
        &self,
        src: &[&[Self::Symbol]],
        basis: &Self::Basis,
    ) -> Vec<Self::Output>;
}

// Concrete impls (each ~30-80 LoC, lives in consumer crate):
//   - IdentityBasis<T>           in ndarray-codec
//   - DctIIBasis<const N>        in ndarray::hpc::fft
//   - HadamardBasis<const N>     in ndarray::hpc::fft
//   - AdamPrecondBasis           in burn-codec
//   - KFACBlockBasis             in burn-codec
//   - ShSpectralBasis<const L>   in ndarray::hpc::splat3d
//
//   - AlphaCompositeReduce       in ndarray::hpc::splat3d
//   - RansEncodeReduce           in ndarray-codec::ans
//   - SumReduce                  in ndarray-codec::reduce
//   - SoftmaxReduce              in ndarray::hpc::activations

// ────────────────────────────────────────────────────────────────────
// Plug-point 3: CurveOrder — per M:E-B
// ────────────────────────────────────────────────────────────────────

/// Space-filling curve that linearises a multi-dim consumer payload
/// into 1D for codec processing. The codec sees only the 1D stream.
///
/// Concrete impls (each ~20-40 LoC):
///   - RasterScan<const W, const H>  for cognitive cells
///   - MortonOrder<const D>          for 3DGS in 3D
///   - HilbertOrder<const D>         for splat in 3D (alternative)
///   - TokenSequence                 for attention
///   - LayerSequence                 for gradient
pub trait CurveOrder<const N: usize> {
    /// Total points on the curve.
    fn len(&self) -> usize;
    /// (i+1)-th neighbour of point i along the curve, or None at boundary.
    fn next(&self, i: usize) -> Option<usize>;
    /// Per-point coordinate (in consumer's native dimensionality).
    fn coord(&self, i: usize) -> [i32; N];
}

// ────────────────────────────────────────────────────────────────────
// Plug-point 4 (lower priority, M:E new): RdoMetric
// ────────────────────────────────────────────────────────────────────

pub trait RdoMetric {
    type Distortion: Copy + PartialOrd;
    fn distortion(&self, reconstructed: &[u8], original: &[u8]) -> Self::Distortion;
    fn rate(&self, bits_used: usize) -> f32;
    fn cost(&self, d: Self::Distortion, r: f32, lambda: f32) -> f32;
}

// Concrete impls (consumer crate):
//   - PsnrMetric       for video
//   - SsimMetric       for splat
//   - LossDeltaMetric  for gradient
//   - KlDivergence     for attention
```

**The trait surface is the contract.** Plan I (3 days, Phase 0)
implements `PredictiveSignal` for cognitive cells as the reference
consumer. Plan A4-design (1 day) commits the `Basis<T>` + `LinearReduce`
shapes. Plans D / E / F each ship one `impl PredictiveSignal for ...`
plus their `CurveOrder` / `Basis` / `RdoMetric` impls.

---

## 9. Falsifiability matrix

Every load-bearing claim from the canon and from this doc has a
test, a metric, and a pass condition. The matrix is the audit
that decides whether each holy-grail claim is demonstrated.

| Claim | Source | Test | Metric | Pass condition |
|-------|--------|------|--------|----------------|
| M:H-1 / HG1 (4 loads → 1 codec) | canon | Plan G binary runs all 4 modes | 4 Lance columns emitted | All 4 emit successfully |
| M:H-2 / H-2 (transform = optimizer) | canon | A4 + burn-codec ship | AdamPrecondBasis impls LinearReduce | Bench Adam-as-codec on BERT-glue |
| M:H-3 / HG3 (bit-exact attention) | canon | Plan D ships | KV cache compresses + RULER score | ≥4× ratio, ≤0.5% accuracy loss |
| M:H-4 / H-4 (Shannon-optimal grad) | canon | Plan F + signSGD bench | rANS frequency-table entropy match | Empirical entropy within 5% of H(p) |
| M:H-5 / HG4 (ZeRO generalisation) | canon | Plan F + DeepSpeed bench | 8-16× compression at 16+ workers | ≥8× at ≤0.5% loss delta |
| M:H-6 / HG2 (sub-1-bit/Gaussian) | canon + R-10 | Plan E + offline codebook | bits/Gaussian on Mip-NeRF 360 | Near: ≤4 bit; stretch: ≤1 bit |
| M:H-7 / HG5 (Lance substrate) | canon | Plan H + Plan I land | 4-load Lance columns same schema | Schema check; per-load `read_codec_lance` |
| M:H-8 / H-6 (64×64 universal) | canon | M:E-G `Ctu<const N>` | Compiles for N ∈ {16, 32, 64} | All 3 sizes pass codec tests |
| M:H-9 / HG6 (splat3d × x265 one lib) | canon | Plan E + ndarray-codec | 1 binary, 1 dep tree | Binary size <10 MB; deps tree-clean |
| M:H-NEW-1 (single binary, 4 loads) | canon | Plan G binary | `codec-bench --mode {video,splat,kv,grad}` | Executes each in <60s on ref data |
| M:H-NEW-2 (~2 KLoC stack) | canon + R-3 | LoC audit per PR | Generic-codec LoC | <1500 LoC; per-consumer <200 LoC |
| R-1 (LinearReduce shape correct) | this doc | Plan A7 builds against the trait | Trait isn't subclassed by A7 | A7 uses public trait surface only |
| R-2 (bit 14 consumer-typed) | this doc | Plan A8 ships ConsumerProfile demux | All 4 profile decoders run | 4 profile-specific tests pass |
| R-3 (LoC envelope) | this doc | LoC audit per PR | Cumulative generic LoC | <1500 after A4-A8 |
| R-4 (Plan G thresholds) | this doc | `codec-bench --threshold` flag | Ratio + quality + LoC | All 4 thresholds clear |
| R-5 (DCT crossover at 64) | this doc | A4-impl bench at varying N | Per-block vs batched dispatch time | Crossover within [32, 96] empirically |
| R-6 (SSD via VNNI ≥30×) | this doc | `batched_ssd_search` micro-bench | Cycles per 16×16 ME candidate | ≤4 cycles per 256-d dot (VNNI) |
| R-7 (tropical-GEMM ≥10×) | this doc | Plan A6 partition bench | Per-CTU partition RDO time | ≥10× over naive recursive on Zen 4 |
| R-8 (Plan G is confidence gate) | this doc | Phase order | Plan G ships before A7 | A7 PR doesn't merge until Plan G binary green |
| R-9 (topology-free) | this doc | grep audit | Codec body has no spatial-semantic refs | `grep -rE 'North\|East\|West\|South' src/hpc/codec/*.rs` returns only test/doc |
| R-10 (4 bit/Gaussian floor) | this doc | Plan E bench | bits/Gaussian on Mip-NeRF 360 | ≤4 bits/Gaussian without offline codebook |
| R-11 (4K 60fps SIMD-batched) | this doc | Plan G video latency assert | Per-leaf encode time | ≤210 ns/leaf on Sapphire Rapids (≡ ≤13.4 µs per fully-split 64×64 CTU = 64 leaves; corrected 2026-07-16 — the earlier "≤210 ns/CTU" mislabeled the per-leaf breakdown total) |
| R-12 (per-CTU flush) | this doc | A8 frame-header parse + decode | First-CTU latency | First CTU decodable before frame complete |
| R-13 (Option A per-shard) | this doc | Plan F on BERT-glue | 8× compression + accuracy | Holds; else escalate to Option B |
| R-14 (Pillar 10 active) | this doc | `cargo test -p jc` (default features) | Pflug-Pichler Lipschitz bound | Pillar 10 probe green |
| R-14 (Pillar 11 active) | this doc | `cargo test -p jc --features hambly-lyons` | Signature uniqueness probe | forward < 1e-9, converse > 0.05, ratio ≥ 1e6 |
| R-15 (SignatureBasis lane) | this doc | Plan G stream-signal lane | signature-space discrimination under Pillar 11 (forward-only — path inversion is N/A per R-15) | forward < 1e-9, converse > 0.05, ratio ≥ 1e6 (or agreed DEPTH-specific floor) |

**Every row of this matrix is a test — FORWARD-CONDITIONAL.** Plan G's
bench harness binary emits a JSON report containing the actual
measurement for each row; the merge job for Phase 2 consumer PRs reads
that report and gates on pass-fail.

> **[Status tag, 2026-07-16 per audit #19]:** the Plan G bench-harness
> binary does not exist yet. Every row above is a *planned* test, not a
> passed one — this matrix is a falsification contract, never a
> passed-tests dashboard. Rows citing "Plan G binary" as the test are
> circular until that binary ships.

---

## 10. Sequencing diagram (canonical)

```text
                                     T+0
            ┌─────────────────────────────────────────────┐
            │            PHASE 0 — substrate gates         │
            │                                             │
            │   Plan H — extract ndarray-codec (3d)        │
            │   Plan I — PredictiveSignal trait (3d)       │
            │   Plan A4-design — Basis<T> + LinearReduce (1d)
            │   Plan G — multi-domain bench (2w) ★ GATE    │
            └─────────────────────┬───────────────────────┘
                                  │
                                  ▼
                    Plan G binary green; thresholds testable
                                  │
                                  ▼
        ╔════════════════════════════════════════════════╗
        ║   Plan A7 — rANS  (1.5 w)  CRITICAL PATH       ║
        ║   gates on Plan G; ships against R-1 trait     ║
        ╚════════════════════════╦═══════════════════════╝
                                 │
              ┌──────┬───────┬──┴───┬──────┬──────────┐
              ▼      ▼       ▼      ▼      ▼          ▼
        Plan B   Plan A4  Plan A6  Plan A8  Plan C   (R-11 SIMD
        (inter) (impl)   (RDO)   (stream) (EWA)      batch path
        1 wk    1 wk     1 wk    1 wk     1 wk       lands in
              └──────┴───────┴──┬───┴──────┘         each)
                                ▼
              ┌─────────────────────────────────┐
              │  PHASE 1 closes — codec mech    │
              │  complete; thresholds re-run    │
              └────────────────┬────────────────┘
                               │
              ┌────────┬───────┴────────┐
              ▼        ▼                ▼
          Plan E   Plan D           Plan F (after D)
          (3DGS    (attention       (federated SGD,
          3 wk×2)   2 wk×2)          4 wk×2, R-13)
              │        │                │
              └────────┴──────┬─────────┘
                              ▼
                    Plan G runs all 4 thresholds
                              │
                              ▼
                    HG1 / HG6 / M:H-NEW-1 demonstrated
                    (or specific claims falsified)
```

★ **Gate semantics.** Plan G is a *blocking* gate: Plan A7 cannot
merge until Plan G's bench-harness binary is green (i.e., runs all 4
modes end-to-end, even if compression ratios are below threshold —
those calibrate in Phase 1). The threshold pass-fail bind on Phase 2
consumer PRs, not on Phase 1 codec PRs.

---

## 11. End-state recap and exit conditions

**The end state, recapped from §1:**

After ~10.5 weeks of trajectory work:

1. One binary `codec-bench` runs four modes end-to-end (HG1 demonstrated).
2. Generic codec LoC ≤1500 (R-3 / M:H-NEW-2 demonstrated).
3. Each consumer ≤200 LoC of trait impl (R-3 demonstrated).
4. Compression ratios meet R-4 thresholds for all four loads:
   - Video: ≥0.95× x265 ultrafast at parity PSNR
   - Splat: ≥30× over Inria PLY-trim at SSIM parity
   - KV cache: ≥4× over FP16 raw at ≤0.5% RULER loss
   - Gradient: ≥8× over signSGD at ≤0.5% loss delta
5. `ndarray-codec` crate extracted (M:E-D / Plan H demonstrated).
6. Three traits land at type-erased boundaries (Plan I + A4-design).
7. CLAUDE.md "Architecture Rule" lists 5 categories (M:T-3 closed).

**Exit conditions per claim:**

- **M:H-1 met** when `codec-bench --mode {video, splat, kv, grad}`
  emits 4 Lance columns within the LoC envelope.
- **M:H-2 met** when AdamPrecondBasis impl ships in burn-codec and
  reduces BERT-glue training to within 5% of stock Adam loss curve
  using the same `LinearReduce` trait surface as A7 rANS.
- **M:H-3 met** when Plan D ships and Llama-3 inference on RULER 64K
  passes R-4 threshold.
- **M:H-4 met** when Plan F + signSGD bench shows rANS frequency-table
  entropy within 5% of empirical H(p) for ≥3 layer types.
- **M:H-5 met** when 16-worker BERT fine-tune via Plan F clears 8×
  compression at ≤0.5% loss delta on GLUE.
- **M:H-6 met** when Mip-NeRF 360 garden scene compresses to ≤4
  bits/Gaussian (near target per R-10).
- **M:H-7 met** when each of the 4 loads writes to a Lance column
  with identical schema (one read path serves all).
- **M:H-8 met** when `Ctu<16>`, `Ctu<32>`, `Ctu<64>` all pass codec
  tests in a single build.
- **M:H-9 met** when one binary `codec-bench` ships <10 MB with all
  4 modes wired (no `--feature splat` gating; everything compiled in).

**If any claim fails its exit condition**, the corresponding consumer
PR scopes down (e.g., M:H-6 stretch sub-1-bit/Gaussian fails → ship
4-bit/Gaussian as near-term reality, mark sub-1-bit as Plan E2). The
falsifiability is the point; not every claim has to hold for the
architecture to be valuable.

**Worst-case fallback.** If even Plan G itself fails (e.g., trait
shape from R-1 turns out wrong under bench pressure), the canon's
M:E-A factoring is wrong. Recovery: revise traits before A7 ships.
Cost: ~3 days extra in Phase 0. **This is exactly why Plan G goes
first — to catch this before A7 wastes 1.5 weeks against the wrong
shape.**

---

## 12. Compaction-preservation contract

When this doc is summarised across context windows, preserve at minimum:

1. **The five "merged well" items** from §3 (M:E-A / M:E-D / M:E-G /
   M:E-I / M:E-F) — these are the canon's load-bearing synthesis.
2. **The fifteen resolutions** R-1 through R-15 with their summary
   commitments:
   - R-1: `LinearReduce<Basis>` two-trait shape
   - R-2: bit 15 universal, bit 14 consumer-typed
   - R-3: ≤1500 LoC generic, ≤200 LoC per consumer
   - R-4: 4 threshold pairs (video, splat, kv, grad)
   - R-5: DCT crossover ~64 blocks
   - R-6: SSD via VNNI ≥30× over SAD
   - R-7: tropical-GEMM partition O(4^d) → O(d²); canonical home
     `lance-graph::blasgraph` (kernel unwritten); shipped min-plus is
     `bgz17::ScalarCsr::spmv_min_plus` [lossy sibling, prototype only —
     corrected 2026-07-16 per audit]
   - R-8: Plan G is confidence gate
   - R-9: topology-FREE codec layer
   - R-10: ~4 bits/Gaussian near target, ~1 bit stretch
   - R-11: 210 ns/leaf SIMD-batched encode (per-leaf, not per-CTU)
   - R-12: per-CTU flush default; per-bucket Plan F
   - R-13: Option A (per-shard codebook) for Plan F v1; primitives are
     `cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`
   - R-14: formal correctness via `jc::pflug` (Pillar 10) +
     `jc::hambly_lyons` (Pillar 11, feature-gated)
   - R-15: `SignatureBasis<DEPTH>: Basis<f32>` as fifth Plan G lane
     (stream signal)
3. **The trajectory** from §2 — Phase 0 → A7 → parallelise → Phase 2
4. **The five-category architecture** including `ndarray-codec`
5. **The four traits** as the canonical contracts:
   `PredictiveSignal`, `Basis<T>`, `LinearReduce`, `CurveOrder<const N>`
   (plus `RdoMetric` for A6)
6. **Plan G as the gate** — A7 cannot merge until Plan G binary green
7. **The falsifiability matrix in §9** — every claim has a test;
   not every claim will pass; that's the design

**Citation IDs in this doc** (R-1 .. R-15) are stable. Canon IDs
(M:E-*, M:H-*, M:H-NEW-*, M:T-*, A:E-*, A:H-*, A:T-*, B:E-*, B:HG-*,
B:D-*) remain stable per canon's §10. Append, never renumber.

---

## 13. The single load-bearing paragraph

If you read nothing else:

> *The merged canon committed to the right architectural synthesis
> (M:E-A, M:E-D, M:E-G, M:E-I) but left the load-bearing contracts
> unsigned. This doc commits them: `Basis<T>` + `LinearReduce` are
> two traits not one (R-1); bit 14 of the leaf header is consumer-
> typed and bit 15 universal (R-2); generic codec body ≤1500 LoC
> with ≤200 LoC per consumer (R-3); four threshold pairs gate
> Plan G's pass criteria (R-4); the trajectory is Plan G (2 wks) →
> Plan A7 critical path (1.5 wks) → Phase 2 consumers parallel
> (3 wks); end state is one binary, four loads, ~2 KLoC stack
> demonstrating M:H-NEW-1 in ~10.5 weeks of wall-clock. Every claim
> in §9 has a test; Plan G's bench-harness binary is the audit. The
> falsifiability is the point.*

---

_Last edit: 2026-05-22 — companion to merged canon `bc9da4ad`.
Edit when an R-N resolves to ship, when a falsifiability test pin
shifts, or when an exit condition closes. Renumber only by appending._
