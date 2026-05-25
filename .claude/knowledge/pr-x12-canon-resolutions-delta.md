# PR-X12 — Canon Resolutions Delta

> Date: 2026-05-22
> Status: **extract** — distills the content from PR #197's `pr-x12-substrate-canon-resolutions.md` (1281 lines) that is NOT already covered by the four prior PR-X12 docs (`codec-x265-design`, `codec-cognitive-substrate-mapping`, `cross-domain-synergies`, `substrate-merged-canon`).
>
> Read this when you want only the new commitments; read the full canon-resolutions doc when you want the full chain-of-reasoning that produced them.

---

## 0. What's actually new

The merged canon (`bc9da4ad`) argued the architecture; canon-resolutions makes it falsifiable. Six categories of novel content survive the delta filter:

1. **Concrete trait signatures** — R-1 (`Basis<T>` + `LinearReduce` split), §8 surface (`PredictiveSignal`, `CurveOrder<const N>`, `RdoMetric`)
2. **Quantified budgets** — R-3 LoC envelope per sub-card / per consumer + audit rule; R-4 four Plan G thresholds; R-11 4K@60fps latency budget
3. **Math identities** — R-6 SSD-via-VNNI (`||A||² - 2A·B + ||B||²`), R-7 tropical-GEMM partition (`O(4^d) → O(d²)`, kernel at `bgz17::scalar_sparse::tropical_spmv`)
4. **Type-level invariants** — R-2 bit-15/bit-14 split, R-9 topology-FREE codec
5. **Phasing patterns** — R-8 confidence-gate framing, R-13 Option-A-then-B for federated codebook (primitives: `cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`)
6. **Formal-correctness + stream lane (post-merge)** — R-14 (`jc::pflug` Pillar 10 + `jc::hambly_lyons` Pillar 11), R-15 (`SignatureBasis<DEPTH>` as fifth Plan G lane)

Plus the synthesis layer: §9 falsifiability matrix (24+3 rows including R-14/R-15), §10 sequencing with named gates, §12 compaction-preservation contract.

---

## 1. The trait signatures (R-1 + §8)

The merge cited `trait LinearReduce<Basis>` but never gave the shape. Canon-resolutions commits it:

```rust
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
    fn reduce_batch(&self, src: &[&[Self::Symbol]], basis: &Self::Basis) -> Vec<Self::Output>;
}
```

**Two traits, not one. Why:** Basis is data; LinearReduce is logic. Same `DctIIBasis<8>` feeds the codec transform path (`A4`) and the EWA splat rasterizer (Plan E). Single-trait conflation loses that reuse.

**No const generic on `dim()`. Why:** codec dispatches 4×4 / 8×8 / 16×16 / 32×32 at runtime per CTU split depth. Const-generic basis forces depth at type level — wrong factoring. Compile-time win comes from monomorphising the *reduction* type (single per consumer), not the basis dim.

**Concrete impls list:**

| Impl | Home crate |
|---|---|
| `IdentityBasis<T>` | `ndarray-codec::basis` |
| `DctIIBasis<const N>` | `ndarray::hpc::fft` |
| `HadamardBasis<const N>` | `ndarray::hpc::fft` |
| `AdamPrecondBasis` | `burn-codec` (consumer) |
| `KFACBlockBasis` | `burn-codec` (consumer) |
| `ShSpectralBasis<const L>` | `ndarray::hpc::splat3d` |
| `AlphaCompositeReduce` | `ndarray::hpc::splat3d` |
| `RansEncodeReduce` | `ndarray-codec::ans` |
| `SumReduce` | `ndarray-codec::reduce` |
| `SoftmaxReduce` | `ndarray::hpc::activations` |

**`PredictiveSignal` (Plan I, 3 days):**

```rust
pub trait PredictiveSignal: Copy + Eq {
    type Basin: Copy + Eq;
    type Residual: Copy;
    type Escape: Copy;
    type NeighbourRef<'a>: Copy where Self: 'a;

    fn nearest_basin(&self, codebook: &[Self::Basin]) -> (u16, Self::Residual);
    fn fits_delta(residual: &Self::Residual) -> bool;
    fn pack_residual(residual: &Self::Residual) -> u8;
    fn neighbours(&self) -> [Option<Self::NeighbourRef<'_>>; 4];
    fn to_escape(&self) -> Self::Escape;
}
```

~50 LoC per consumer impl. Reference impl is the cognitive cell `Fingerprint`.

**`CurveOrder<const N: usize>`** — space-filling curve over consumer's native dim:

```rust
pub trait CurveOrder<const N: usize> {
    fn len(&self) -> usize;
    fn next(&self, i: usize) -> Option<usize>;
    fn coord(&self, i: usize) -> [i32; N];
}
```

Concrete impls: `RasterScan<W,H>` (cognitive), `MortonOrder<D>` (3DGS), `HilbertOrder<D>` (alternative splat), `TokenSequence` (attention), `LayerSequence` (gradient). Each ~20-40 LoC.

**`RdoMetric`** (Plan A6):

```rust
pub trait RdoMetric {
    type Distortion: Copy + PartialOrd;
    fn distortion(&self, reconstructed: &[u8], original: &[u8]) -> Self::Distortion;
    fn rate(&self, bits_used: usize) -> f32;
    fn cost(&self, d: Self::Distortion, r: f32, lambda: f32) -> f32;
}
```

Consumer impls: `PsnrMetric` (video), `SsimMetric` (splat), `LossDeltaMetric` (gradient), `KlDivergence` (attention).

---

## 2. The quantified budgets (R-3 + R-4 + R-11)

### 2.1 LoC envelope (R-3)

Current state on master commit `bc9da4ad`:

| File | Total | Of which generic glue |
|---|---|---|
| `ctu.rs` | 771 | ~280 |
| `mode.rs` | 518 | ~180 |
| `predict.rs` | 511 | ~140 |
| `mod.rs` | 38 | ~38 |
| **Total** | **1838** | **~600** |

The remaining ~1240 lines are tests / doctests / docstrings.

**Budget envelope:**

| Sub-card | Generic-glue LoC ceiling |
|---|---|
| A4 (transform) | ≤200 |
| A6 (RDO) | ≤150 |
| A7 (rANS) | ≤300 |
| A8 (stream) | ≤200 |
| A3-inter | ≤100 |
| Sum | ≤950 (with ~50 LoC margin to the 1500 ceiling) |

Per-consumer (4 consumers): ≤200 LoC each = ≤800 total trait-impl glue.

**Audit rule (load-bearing):** every PR introducing or modifying generic-codec code must include a one-line generic-LoC delta in the body. Exceeding the envelope triggers architectural review, not CR nits.

**Falsifies M:H-NEW-2 if:** cumulative generic LoC exceeds 1500 after A4-A8 land + at least one consumer integration.

### 2.2 Plan G thresholds (R-4)

| Load | Reference baseline | Compression target | Quality floor |
|---|---|---|---|
| Video | x265 `--preset ultrafast` CRF 23 on Big Buck Bunny 1080p | ≥0.95× reference ratio | PSNR ±0.1 dB |
| 3DGS | Inria stock PLY-trim on Mip-NeRF 360 garden | ≥30× over PLY-trim raw | SSIM ≥ ref − 0.005 |
| KV cache | FP16 raw, Llama-3-8B-Instruct, 64K context, RULER | ≥4× over raw FP16 | RULER loss ≤0.5% |
| Gradient | BERT-large fine-tune on GLUE-MNLI, signSGD baseline | ≥8× over signSGD raw | validation-loss delta ≤0.5% |

Three-way pass per load: (ratio + quality + LoC). Sub-threshold on any one = blocker.

**Stretch (recorded, not blocking):** video 1.5× x265, 3DGS sub-1-bit/Gaussian, KV 8×, gradient 16×.

### 2.3 4K@60fps latency budget (R-11)

| Constraint | Value |
|---|---|
| 4K resolution | 3840 × 2160 = 8.3 M pixels |
| 60 fps | 16.67 ms/frame |
| 64×64 CTU | 132,710 CTUs/frame |
| **Per-CTU budget** | **125 ns/CTU** |

Encoder per-CTU breakdown:

| Stage | Scalar reference | SIMD-batched target |
|---|---|---|
| basin lookup (4096-entry Hamming dist) | ~800 ns | ~50 ns |
| mode decide (Skip→Merge→Delta→Escape) | ~80 ns | ~80 ns |
| header pack | ~5 ns | ~5 ns |
| transform (A4, 8×8 DCT-II) | ~30 ns | ~30 ns |
| quantize (i8 round) | ~5 ns | ~5 ns |
| rANS encode (A7) | ~40 ns | ~40 ns |
| **Total** | **~960 ns** | **~210 ns** |

Scalar misses 60 fps by 7.6×; SIMD-batched misses by 1.7× (same OoM). **Pins B:D-CODEC-8 / A:T-7 from P2 → P1** — A4-impl and A6 must ship SIMD-batched, not scalar-then-vectorize.

---

## 3. Math identities (R-6 + R-7)

### 3.1 SSD via VNNI (R-6)

```text
SAD(A,B) = Σ |A_{ij} - B_{ij}|              ← no matrix shape
SSD(A,B) = Σ (A_{ij} - B_{ij})²
        = ||A||² - 2·(A·B) + ||B||²          ← middle term IS a GEMM
```

For N motion-vector candidates against one 16×16 reference block:

```text
Candidates  A_1..A_N : (N × 256) matrix
Reference   B        : 256-d vector
A_batch @ B          : N×256 @ 256×1 → N×1 GEMV
```

**Throughput:** VNNI VPDPBUSD = 64 i8·i8→i32 dot-products per cycle on Cascade Lake+. One 256-elem dot = 4 VPDPBUSD ops = ~4 cycles. Hand-tuned SAD via VPSADBW = ~128 cycles per 16×16 block. **Speedup: 30-50×.**

**Layering:** lands as `batched_ssd_search` in `ndarray::hpc::blas_level2`. Not codec-specific. Codec uses the math; BLAS owns the math.

### 3.2 Tropical-GEMM partition RDO (R-7)

HEVC's recursive partition: `O(4^d)` per CTU at depth d.

Tropical-semiring (+, min) formulation:

```text
1. 85-node quad-tree as DAG with edge weights W[parent, child] = ΔRDO
2. Matrix relaxation:  D ← min(D, D + W)     ← tropical-GEMM iteration
3. Repeat for d iterations
4. Optimal partition = argmin_n D[root, n] over leaf nodes
```

**Complexity:** `O(d² × |nodes|)`. For d=4, |nodes|=85: 1360 ops/CTU vs 21,760 naive. **~16× speedup.**

At 4K 132K CTUs/frame: ~4 ms vs ~64 ms just for partition RDO. At 60 fps, the difference between fitting and missing budget.

**Dep direction:** `ndarray-codec → lance-graph::blasgraph` (tropical-GEMM kernels nominally live in blasgraph). Allowed post-Plan-H because ndarray-codec is a sibling crate, not the bottom.

**Actual kernel home (current):** `lance-graph::bgz17::scalar_sparse::tropical_spmv`. The `blasgraph` namespace is the eventual abstraction; until that lands, ndarray-codec depends on bgz17 directly. Cite the symbol when wiring A6, not the namespace.

**Plan A6 (1 week) ships this.** λ-RDO knob scales edge weights; tropical-GEMM relaxation computes optimal mode tree.

---

## 4. Type-level invariants (R-2 + R-9)

### 4.1 Header bit-14/bit-15 split (R-2)

```text
bit 15  UNIVERSAL   "has inter-tier reference" (A3-inter)
                    0 = self-contained leaf
                    1 = refers to parent-tier LeafCu
                    Same semantic for all four consumers.

bit 14  CONSUMER    multiplexed via ConsumerProfile in frame header (Plan A8)
                    cognitive : Pearl rung high bit
                    video     : reserved 0
                    splat     : LOD-cascade-source flag
                    gradient  : worker-shard parity (for FRC)
```

Frame header carries 2-bit `ConsumerProfile` tag. Decoder routes bit-14 interpretation per profile. Per-leaf granularity matters: causal direction can change per cell in a cognitive scene, but profile is per-frame.

### 4.2 Topology-FREE codec (R-9)

Stronger than topology-generic. The codec body never knows N/E/W/S.

```rust
// PredictiveSignal::neighbours -> [Option<NeighbourRef>; 4]
//   slot 0, slot 1, slot 2, slot 3 — codec sees indices, not directions
//
// Consumer attaches the semantic:
//   cognitive : slot 0 = N, slot 1 = E, slot 2 = W, slot 3 = S
//   splat     : slot 0 = prev-Morton, slot 1 = next-Morton,
//               slot 2 = parent-LOD,  slot 3 = child-LOD
//   attention : slot 0 = prev-token,  slot 1 = next-token,
//               slot 2 = prev-head,   slot 3 = next-head
//   gradient  : slot 0 = prev-iter,   slot 1 = next-iter,
//               slot 2 = prev-layer,  slot 3 = next-layer
```

**`MergeDir` enum is a consumer-side name for slot indices**, exposed via `pack_merge_dir(MergeDir) -> u8` at the boundary. Never used inside predict / RDO / stream / rANS paths.

**Audit:** `grep -rE 'North|East|West|South' src/hpc/codec/*.rs` must return only test/doc, never production paths.

This is what makes "~200 LoC per consumer" plausible: the consumer attaches all semantic labels outside the codec boundary.

---

## 5. Phasing patterns (R-8 + R-13)

### 5.1 Plan G as confidence gate (R-8)

46 debt items across A:T-1..T-23, B:D-CODEC-1..10, B:D-STACK-1..13. **45 of them degrade perf or correctness.** One — B:D-STACK-13 (no bench harness) — degrades **confidence**.

Confidence debt ≠ perf debt ≠ correctness debt. It's foundational and self-reinforcing: a perf regression makes the codec slow; a confidence gap makes every other resolution unverifiable.

**Plan G must precede A7 because:**
- If A7's trait shape is wrong, fixing it after ship is 4-8× the cost
- If the architectural claim is wrong, no A7 perf work makes it right
- Two weeks of bench-harness work front-loaded saves six months of trait-shape rework

### 5.2 Decision-deferral pattern for federated codebook (R-13)

| Option | Compression | Cross-worker comm | Verdict |
|---|---|---|---|
| A (per-shard codebook) | baseline | zero | **Plan F v1** |
| B (replicated codebook) | 1.5-2× better | one all-reduce/epoch | Phase 3 if v1 fails R-4 |
| C (hierarchical) | best | complex protocol | Research-grade, Phase 3+ |

Pattern: ship simplest-that-works, measure, escalate. Don't pick best-in-theory upfront.

Wire-format hook for Option A: `WorkerId: u16` + `CodebookHash: u64` in frame header.

**Implementation primitives** (already exist; PR-X12 only adds the wire format + `CodebookHandle` trait):

| Concern | Crate / module |
|---|---|
| Codebook training (k-means + CAM-PQ) | `ndarray::hpc::cam_pq::CamCodebook` |
| Deployed encoding format | `lance-graph::bgz-tensor::Codebook4096` / `bgz-hhtl-d` |
| Online plastic updates (SharedClusterWide) | `ndarray::hpc::dn_tree` |
| Integrity proof (Blake3-48 Merkle root, xor_diff) | `ndarray::hpc::merkle_tree` |
| Gossip protocol | `q2` (external) |

### 5.3 Streaming flush granularity (R-12)

Per-CTU default. `FlushUnit` 2-bit tag in frame header:

```text
FlushUnit::Ctu      00  default — video / splat / attention
FlushUnit::Bucket   01  gradient SGD (per-bucket 4096 weights)
FlushUnit::Frame    10  offline batch encode
FlushUnit::Reserved 11
```

**Why per-CTU:** ~12 KB buffer, ~125 ns latency, ~80K flushes/sec at 4K 60fps. Per-frame = ~1.5 MB buffer, ~16.67 ms latency (one frame added to pipeline). Per-GOP = ~25 MB / 267 ms — unacceptable for live attention / KV-cache.

---

## 6. Cross-architecture DCT-II crossover (R-5)

DCT-II vs GEMM dispatch crossover varies by architecture. Plan A4-impl calibrates per arch:

| Architecture | Crossover N | Per-block path | Batched path |
|---|---|---|---|
| Sapphire Rapids (AMX-BF16) | ~64 | Loeffler 1D + transpose | AMX TDPBF16PS |
| Skylake-X / Ice Lake (AVX-512F) | ~32 | Loeffler 1D + transpose | AVX-512 ZMM batched |
| Zen 4 (AVX-512) | ~96 | Loeffler 1D + transpose | AVX-512 ZMM (no AMX) |
| Apple Silicon (NEON) | ~256 | Loeffler 1D | NEON 4×4 GEMM stub |

**Why crossover at 64 on SPR:** AMX TDPBF16PS = one 16×16 BF16 tile per cycle. 64 blocks × 32×32 → 256 tile ops → ~256 cycles batched. Per-block butterfly = 80 ops × 64 = 5120 ops → at 4 IPC = 1280 cycles. Crossover within order of magnitude.

---

## 7. Sub-1-bit/Gaussian factor breakdown (R-10)

Stock 3DGS-PLY: ~50 bytes/Gaussian = 400 bits.

| Factor | Reduction | Mechanism | Cumulative |
|---|---|---|---|
| 1 | ≈10× → 20 bits | k-means basin + Skip-heavy mode coding (60% Skip / 20% Merge / 15% Delta / 5% Escape) | 20× over PLY |
| 2 | ≈3× → 7 bits | rANS entropy coding (mode entropy = 1.53 bits; basin/delta entropy similarly heavy-tailed) | 57× over PLY |
| 3 | ≈2× → 4 bits | SH-residual cross-LOD prediction (L=2/L=3 SH highly predictable from L=0/L=1) | **100× over PLY = near target** |
| 4a | ≈2× → 2 bits | Offline per-asset codebook training (stretch, +1 wk) | 200× over PLY |
| 4b | ≈2× → 1 bit | CABAC-style context modeling (per-mode-given-neighbour-mode probs) | 400× over PLY |
| 4c | ≈2× → 0.5 bit | Inter-frame coding for video-of-3DGS (Plan E2) | 800× over PLY |

**Honest near-term target: ~4 bits/Gaussian** (factors 1+2+3). Clears R-4's 30× threshold by 3.3×.

**Stretch: ~1 bit** = factors 4a+4b, +3 weeks beyond Plan E baseline.

**Sub-1-bit: ~0.5 bit** = factor 4c, requires Plan E2.

---

## 8. Falsifiability matrix (§9 of canon-resolutions)

24 rows mapping every M:H-N and R-N to (test, metric, pass condition). Plan G's bench harness emits a JSON report; merge job for Phase 2 consumer PRs reads it and gates pass-fail.

Highlights of falsifiers — the canary tests:

| Row | If this fails | Then |
|---|---|---|
| M:H-NEW-1 | `codec-bench` doesn't run 4 modes in <60s on ref data | The single-binary claim is unproven; architectural synthesis was wrong |
| R-1 | A7 has to subclass `LinearReduce` to make rANS work | Trait factoring wrong; A7 wastes 1.5 wks |
| R-3 | Cumulative generic LoC > 1500 after A4-A8 | M:H-NEW-2 falsified; the abstraction grew domain-specific code |
| R-9 | `grep -E 'North|East|West|South' src/hpc/codec/*.rs` returns production paths | Topology-free contract broken; consumer semantics leaked into codec |
| R-11 | SIMD-batched encode > 210 ns/CTU on SPR | Plan G video threshold can't pass; 4K real-time falsified |

---

## 9. Sequencing with named gates (§10)

```text
Phase 0 (T+0 .. T+2 wks)   substrate gates
   Plan H    (3d)   extract ndarray-codec
   Plan I    (3d)   PredictiveSignal trait
   A4-design (1d)   Basis<T> + LinearReduce shapes
   Plan G    (2w)   multi-domain bench   ★ BLOCKING GATE

Phase 0 → Phase 1   GATE: Plan G binary runs all 4 modes end-to-end

Phase 1 (T+2 .. T+4.5 wks)   codec mechanism
   Plan A7  (1.5w)  rANS — CRITICAL PATH
   then parallel:
     Plan B  / A3-inter (1w)
     Plan A4-impl       (1w)
     Plan A6 (RDO)      (1w)
     Plan A8 (stream)   (1w)
     Plan C  (EWA SYRK) (1w)

Phase 2 (T+4.5 .. T+10.5 wks)   consumer integrations
   Plan E (3DGS)      3 wks × 2 workers
   Plan D (attention) 2 wks × 2 workers (parallel to E)
   Plan F (gradient)  4 wks × 2 workers (after D)
```

**Critical path: Plan G → Plan A7.** Everything post-A7 parallelises. Total: ~10.5 wks wall-clock; 2 workers steady-state through Phases 0/1, ramping to 6 in Phase 2.

---

## 10. Compaction-preservation contract (§12)

When this doc family is summarised across context windows, these 7 items must survive:

1. **Five "merged well"** items from canon §3 (M:E-A, M:E-D, M:E-G, M:E-I, M:E-F)
2. **Thirteen R-resolutions** with one-line summaries
3. **The trajectory** Phase 0 → A7 → parallelise → Phase 2
4. **The five-category architecture** including `ndarray-codec`
5. **The four traits** as canonical contracts: `PredictiveSignal`, `Basis<T>`, `LinearReduce`, `CurveOrder<const N>` (+ `RdoMetric` for A6)
6. **Plan G as the gate** — A7 cannot merge until Plan G binary green
7. **The falsifiability matrix** in §9 — every claim has a test

Citation IDs (R-1..R-13) stable. Canon IDs (M:E-*, M:H-*, M:H-NEW-*, M:T-*, A:E-*, A:H-*, A:T-*, B:E-*, B:HG-*, B:D-*) preserved. Append, never renumber.

---

## 11. Formal-correctness layer (R-14) — post-merge addition

The substrate-binding doc (`pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md`) surfaced two formal proofs in `lance-graph::jc` that the codec inherits without re-proving:

| Pillar | Crate / module | What it proves | Status |
|---|---|---|---|
| **Pillar 10** (Pflug-Pichler) | `jc::pflug` | Nested-distance Lipschitz on Sigma DN-trees: CAM-PQ tree quantization preserves FreeEnergy within Lε | Active in default zero-dep build |
| **Pillar 11** (Hambly-Lyons) | `jc::hambly_lyons` | Signature uniqueness on tree-quotient: any path of bounded variation is uniquely determined by its truncated signature up to tree-like equivalence (Annals 171(1), arXiv:math/0507536) | Active under `--features hambly-lyons` (PR #348, 2026-05-07); probe passes (forward<1e-9, converse>0.05, ratio≥1e6) |

R-4's quality-floor rows for video / KV / gradient inherit Pillar 10's Lipschitz bound. R-15's signature lane gates on Pillar 11.

**Open work (G-4):** PR #350 corrects `sigker::signature_kernel_pde`'s known Goursat-PDE math bug; Pillar 11's probe deliberately uses `signature_truncated` (tensor-algebra) until PR #350 lands. Production-scale benchmarking pending.

---

## 12. Stream-signal codec lane (R-15) — post-merge addition

`SignatureBasis<const DEPTH: usize>: Basis<f32>` is the fifth concrete `Basis<T>` impl, complementing the four lanes in §1's table:

```rust
// New: ndarray::hpc::signature (~1 wk, wraps sigker::signature_truncated)
impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH> {
    fn dim(&self) -> usize { /* truncated tensor-algebra dim */ }
    fn apply(&self, path: &[f32], signature: &mut [f32]) {
        // iterated-integral truncation via sigker::signature_truncated
    }
    fn invert(&self, _sig: &[f32], _path: &mut [f32]) {
        unimplemented!("path-from-signature is unique only up to tree-like \
                        equivalence per R-14 Pillar 11")
    }
}
```

**Plan G gets a fifth lane: "stream signal"** — audio waveforms / time-series / gesture / handwriting paths. Codec is `SignatureBasis<DEPTH=3>` + standard rANS over the four-mode taxonomy; quality floor inherits from Pillar 11 (R-14); compression target ~10× over raw f32 path samples (calibrate during Plan G).

**Why `signature_truncated` not `signature_kernel_pde`:** the PDE form ships a known divergence bug (PR #350). The tensor-algebra path is correct today and is what Pillar 11 cites.

---

## 13. The single load-bearing paragraph (canon-resolutions §13)

> *The merged canon committed to the right architectural synthesis (M:E-A, M:E-D, M:E-G, M:E-I) but left the load-bearing contracts unsigned. Canon-resolutions commits them: `Basis<T>` + `LinearReduce` are two traits not one (R-1); bit 14 of the leaf header is consumer-typed and bit 15 universal (R-2); generic codec body ≤1500 LoC with ≤200 LoC per consumer (R-3); four threshold pairs gate Plan G's pass criteria (R-4); the trajectory is Plan G (2 wks) → Plan A7 critical path (1.5 wks) → Phase 2 consumers parallel (3 wks); end state is one binary, four loads, ~2 KLoC stack demonstrating M:H-NEW-1 in ~10.5 weeks of wall-clock. Every claim in §9 has a test; Plan G's bench-harness binary is the audit. The falsifiability is the point. The substrate-binding follow-up (R-14, R-15) adds a formal-correctness layer via `jc` pillars and a fifth stream-signal lane via `SignatureBasis<DEPTH>`.*

---

## Cross-references

- **Full source:** `pr-x12-substrate-canon-resolutions.md` (PR #197, when merged)
- **Architecture canon:** `pr-x12-substrate-merged-canon.md`
- **Companion lenses (this PR):**
  - `pr-x12-x265-blasgraph-gemm.md` — codec primitives re-read through pure GEMM
  - `pr-x12-x266-3dgs-spacetime-upscaling.md` — next-gen codec with 3DGS as upscaling primitive
  - `pr-x12-cognitive-shader-gridlake-soa.md` — splat-spacetime mapping into cognitive shaders + GridLake SoA
  - `pr-x12-nesw-risc-soa-unification.md` — NESW as the agnostic reusable SoA DTO

_Last edit: 2026-05-22._
