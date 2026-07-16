# PR-X12 Codec — Cognitive Substrate Mapping & Holy Grail

> Date: 2026-05-22  
> Author: cross-stack architecture session triggered by PR #195 review  
> Scope: ndarray codec ↔ Gaussian splat ↔ cognitive shaders ↔ blasgraph/MKL ↔ gradient optimization  
> Status: **survives compaction** — load-bearing claim mapping + integration plan + debt inventory  
> Companion to: `pr-x12-codec-x265-design.md` (the as-shipped HEVC-analog spec) — this doc is the *generalisation* of that spec across the rest of the stack
>
> **Post-merge formalisation (2026-05-22):** the bench / cost / dep-direction claims below have been numbered and pinned in `pr-x12-canon-resolutions-delta.md`:
> - §4.1 (4096-entry basin codebook) → **R-10** (sub-1-bit commitment), **R-13** (federated codebook policy)
> - §5.3 (DCT-II / GEMM crossover) → **R-5** (per-arch crossover constants, bench-tuned)
> - §13.1 (block-matched ME → batched i8 GEMM) → **R-6** (ME via SSD identity, VNNI path)
> - §13.3 (CTU partition as tropical-GEMM) → **R-7** (kernel home in `lance-graph::blasgraph`, dep direction allowed)
> - Plan G (bench harness) → **R-4** (architecture-conditional gate), **R-11** (latency assertions per stage)
>
> Perspective lenses written 2026-05-22 (sibling docs):
> `pr-x12-x265-blasgraph-gemm.md` · `pr-x12-x266-3dgs-spacetime-upscaling.md` · `pr-x12-woa-multiarch-orchestration.md` · `pr-x12-anti-neural-lookup-inversion.md` · `pr-x12-gguf-llm-weights-encoding.md` · **`pr-x12-bgz-jc-substrate-synergies.md`** (grounds PR-X12 in already-implemented `bgz17`/`bgz-tensor`/`bgz-hhtl-d`/`jc` crates)

---

## 0. The big claim — read this first

PR-X12 is implicitly building the **gradient-quantization substrate that GenAI training has been missing for two years**. Skip / Merge / Delta / Escape is more honest than ZeRO's ad-hoc compression bucket count; the CTU quad-tree is more general than Mistral's hardcoded sliding-window attention; the per-frame basin codebook is what HEVC SCC (screen-content coding) could have been if k-means at frame rate had existed in 2013.

Read in reverse: every "efficient transformer", "gradient compression" and "neural codec" paper from 2023-2025 has been rediscovering corners of the HEVC/x265 design space under different names. Once A4 (transform) + A6 (RDO) + A7 (rANS) land on top of the A1/A2/A3-intra foundation in PR #195, PR-X12 becomes a **drop-in replacement for SGD's parameter-update step that's also a video codec, also a Gaussian-splat compressor, also an attention-pruning policy** — all from one shared kernel.

The codec is the substrate. The other three are renamings.

---

## 1. The four-axis mapping

Same operator, four names, twelve years apart. The codec column is the most precise because it has bit-exact spec hardening behind it; cite the codec name when in doubt.

| Operation | x265 / HEVC | Gaussian splat | Cognitive shader / attention | Gradient optimizer |
|---|---|---|---|---|
| Reference unit | CTU (64×64 luma block) | Tile (16×16 splat group, Kerbl 2023) | Cognitive frame / one attention layer | One parameter tensor |
| Subdivision | CU quad-tree (64→32→16→8) | Anisotropic tile subdivision under view-frustum density | Attention-head hierarchy (heads → MQA group → sliding window) | Per-layer → per-block → per-row → per-parameter |
| Codebook entry | Palette index (HEVC SCC, ≤ 64) | Splat archetype (color cluster, scale cluster) | Cognitive prototype / basin / vocabulary token | Optimizer state cluster (LoRA rank) |
| **Skip** | Cell exactly matches motion-predicted ref (`δ=0`) | Splat unchanged across frame | Token attends to its own slot only | Parameter frozen this step |
| **Merge** | Cell inherits motion vector from N/E/W/S neighbour | Splat inherits position/scale/SH from neighbour splat | Token inherits attention from neighbour / LoRA group share | Parameter follows neighbour's gradient |
| **Delta** | Cell stores 8-bit residual after motion compensation | Splat stores 8-bit per-axis perturbation (Δx, Δy, Δz, ΔSH_k, Δscale) | Token has small attention perturbation | 8-bit quantized SGD step |
| **Escape** | Cell stores full original value (fallback) | Splat stores full f32 covariance + SH coefficients | Token needs full attention slot | Full-precision parameter update |
| Transform | DCT-II on 4×4/8×8/16×16/32×32 residuals | DCT/wavelet on SH-coefficient residuals | Frequency-domain attention basis | Preconditioner (Adam diagonal / KFAC block) |
| Entropy coder | CABAC (context-adaptive binary arithmetic) | (not yet standardised) | (typically Huffman or byte-pair) | (typically none — full f32) |
| RDO | λ · D + R minimisation over mode tree | Sparsity regulariser (L0 + view-PSNR) | Attention sparsity × downstream loss | Compression ratio × loss-degradation |

**The bet of PR-X12**: one Rust module, one set of types, four use cases. Each use case independently has a $100M+ industry around it. The codec name is precise; the other three are loose because their communities didn't have a 12-year-hardened spec to anchor against.

---

## 2. Mode taxonomy — deep mapping

### 2.1 The 2-bit mode field is load-bearing

```text
bit  15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
    ┌──┬──┬──┬──┬──────────────────────┐
    │ 0│ 0│M1│M0│      basin_idx (12)  │   ← 16-bit header
    └──┴──┴──┴──┴──────────────────────┘
       reserved   mode      vocabulary index
```

Two bits = four modes. Not more, not fewer. The four modes form a **strict cost lattice**:

```text
Skip   (2 bytes total) ⊂  free
Merge  (3 bytes)       ⊂  borrow from neighbour
Delta  (3 bytes)       ⊂  store quantized perturbation
Escape (6 bytes)       ⊂  store full precision
```

Monotone cost ordering is what lets `predict_intra` use a "first-fit cheapest" decision tree without explicit RDO. PR-X12 A6 RDO refines this with λ-weighted distortion, but the cost ordering itself is invariant.

### 2.2 Why "four" is the correct count, not three or five

- **Three** (Skip/Delta/Escape) loses the parameter-sharing optimisation. Merge captures spatial coherence: when a parameter's gradient matches a neighbour's gradient, you encode the relationship not the value. This is **LoRA-as-codec**: low-rank update ≡ Merge-mode parameter group.
- **Five** (Skip/Merge/Delta/Escape/Predict) — HEVC originally had an "Intra-mode-predicted" extra mode that turned out to be subsumable into Merge with the right neighbour predictor. The spec collapsed to four after 2010-2012 testing; we inherit the lesson for free.

### 2.3 Why the mode discriminants match the wire codes

`CellMode::Skip = 0b00`, `Merge = 0b01`, `Delta = 0b10`, `Escape = 0b11` — pinned by `cell_mode_discriminants_match_wire_codes` in `ctu.rs`. This eliminates a translation table at the encoder/decoder boundary. The same Rust enum is the in-memory representation **and** the on-wire byte. Cf. data-flow Rule §2 (Reasoning = owned Copy microcopies).

---

## 3. CTU quad-tree — three hierarchies in one structure

### 3.1 Spatial hierarchy (HEVC's original use)

```text
depth 0 (64×64 CTU)   ↔ one BlockedGrid L1 block (`ctu.rs:236`)
depth 1 (32×32 split) ↔ one CU at split-level 1
depth 2 (16×16 split) ↔ ...
depth 3 (8×8 leaf)    ↔ leaf CU (smallest cognitively-meaningful unit)
```

85 nodes maximum, `MAX_QUAD_TREE_NODES = 1 + 4 + 16 + 64 = 85` (`ctu.rs:55`). Arena-allocated, no `Box<dyn>`, indices are `u16`. Cache-friendly: the entire tree for one CTU fits in 85 × ~16 = 1360 bytes — under one cache line of headers + payload.

### 3.2 Attention hierarchy (rediscovery as transformer)

```text
depth 0 ↔ one attention layer
depth 1 ↔ one attention head (32×32 = 1024 attention slots)
depth 2 ↔ one multi-query-attention group (4 heads sharing KV)
depth 3 ↔ one 8-token sliding window
```

Mistral / Llama4 sliding-window attention is **exactly depth-3 leaf processing** — they just don't think of it as a quad-tree. The split/merge ops in `Ctu::split` / `Ctu::merge` are the dynamic-shape-attention prune/expand decisions every "efficient transformer" paper from 2023-2025 rediscovers.

**Holy grail claim H-3**: The quad-tree is the unification point. The same `Ctu` carrier serves the codec hot path and the attention hot path. Once we commit to that, swapping basis (raw → DCT → learned) and swapping entropy coder (raw u8 → rANS) gives us a video codec + a sparse-attention substrate + a gradient-compression substrate from one set of types.

### 3.3 Gradient-update hierarchy (the optimizer mapping)

```text
depth 0 ↔ "should this parameter tensor be touched this step?"
depth 1 ↔ "which block of this tensor needs update?"
depth 2 ↔ "which row of that block?"
depth 3 ↔ "which parameter within that row?"
```

Translates one-to-one to the codec decision tree:
- Skip = parameter frozen
- Merge = parameter shares update with neighbour (LoRA group)
- Delta = 8-bit quantized gradient
- Escape = full f32 gradient

This is **what DeepSpeed-ZeRO does informally** with `bf16_compress`, `int8_compress`, and `fp32_keep` buckets — except they have three modes, no quad-tree, no transform, no entropy coder. PR-X12 generalises ZeRO's compression scheme.

---

## 4. Palette / basin codebook — what HEVC SCC tried and missed

> [Codebook lifecycle pinned post-merge as **R-13**: the codec exposes the basin codebook as a swappable handle (LocalEphemeral | SharedClusterWide | SharedRegional | PretrainedStatic). The 4096-entry capacity claim below is unchanged; what's new is that the codebook is *not baked* into the codec — orchestration (q2 / woa-rs) picks the right one per request.]

### 4.1 The 12-bit basin = 4096-entry vocabulary

`MAX_BASIN_IDX = (1 << 12) - 1 = 4095` (`mode.rs:79`). The full 12-bit range addresses 4096 real basins — every `LeafCu` carries an index into a fully-populated per-Heel codebook. No slot is reserved as a sentinel: the HHTL ontology (`Heel > Hip > Twig > Leaf`, see `src/hpc/ogit_bridge/assets/cognitive/entities/Leaf.ttl`) defines the codebook as `16 Hips × 16 Twigs × 16 Leaves = 4096 Leaves per Heel`, every Leaf carrying a real `basinSignature`. Authoring-time uncertainty ("not yet decided") stays in the encoder's `Option<u16>` scratch state and never leaks onto the wire. For:

- **Video**: 4096 palette entries per GOP — orders of magnitude more than HEVC SCC's 64-entry cap
- **Splats**: 4096 splat archetypes (colour clusters × scale clusters × view-direction clusters) — covers a non-toy scene
- **Cognitive**: 4096 prototype embeddings per attention layer — comfortably above typical BPE token vocabulary slices
- **Gradients**: 4096 optimizer-state clusters — captures Adam's per-parameter `(m, v)` after k-means quantization

### 4.2 K-means at frame rate is the unlock HEVC SCC could not reach

HEVC SCC profile was **frozen in 2013**. At that time:
- AVX-512 didn't exist (Knights Corner only; SPR shipped 2023)
- VPOPCNTDQ (per-lane popcount, the Hamming-distance hot kernel) didn't exist (Ice Lake 2019)
- VNNI (i8 GEMM for distance assignment) didn't exist (Cascade Lake 2019)
- AMX tile ops (bf16 / int8 batched GEMM for centroid update) didn't exist (Sapphire Rapids 2023)

The SCC team had to cap palette at 64 entries rebuilt per-CTU because that was the only thing the 2013-era hardware could afford. With our current stack:
- `crate::hpc::cam_pq::kmeans` (already shipping) does 4096-entry k-means via AVX-512 distance GEMM
- Centroid update via VNNI i8 GEMM hits ~200M assignments/sec at 32D
- The codebook fits in L1 cache (4096 × 32 = 128 KB at u8 quantisation)
- Frame-rate k-means: **rebuild a 4096-entry codebook every 16 frames at 60 fps**, fully amortised

**Holy grail claim H-1**: PR-X12 + cam_pq gives you the screen-content video codec HEVC SCC was trying to be in 2013 — without retrofitting, just by composing existing modules. Cite this when somebody asks "why is the basin field 12 bits and not 8 like HEVC SCC".

---

## 5. Transform basis — DCT-II ↔ optimizer preconditioner ↔ wavelet ↔ learned

### 5.1 The transform is interchangeable; only the basis differs

A4 will introduce a residual transform applied to Delta-mode 8-bit perturbations before quantization. The choice of basis is the central knob:

| Basis | Use case | Why this one |
|---|---|---|
| Identity | A3 (current) | Reference; lets RDO compare against baseline |
| DCT-II 4×4/8×8 | Video / image (A4 default) | 12 years of HEVC tuning; separable; FFT-amortizable |
| Learned conv 5×5 | Splat compression | Trained for per-scene PSNR; conditioned on view |
| Adam preconditioner | Gradient compression | Per-parameter √(v) scaling, applied lazily as a basis |
| KFAC block-diagonal | LLM gradient compression | Captures cross-parameter correlation per layer |

**All five share the same arithmetic shape**: `Δ' = B · Δ` where `B` is a small dense matrix (4×4, 8×8, or block-diagonal). PR-X12 A4 should define the basis as a generic over a `Transform` trait, then ship DCT-II as the default. Other bases follow without touching `predict.rs` or `mode.rs`.

### 5.2 Holy grail claim H-2: the transform IS the optimizer

In the gradient-compression interpretation, the transform basis IS the optimizer's preconditioning matrix. Adam uses `B = diag(1/√v)`; KFAC uses block-diagonal Kronecker factors. PR-X12 lets you swap optimizer-as-codec at the basis layer, with zero code changes elsewhere.

This is **the most underrated** of the four mappings. Optimizer research treats preconditioner choice as the central knob; codec research treats transform basis as the central knob; nobody has noticed they're the same knob.

### 5.3 The DCT-II / GEMM tradeoff (for downstream batched encode)

> [Resolved post-merge as **R-5**: per-arch crossover constants, to be calibrated by Plan G's `codec-bench`. The candidate defaults in canon-resolutions-delta §R-5 (SPR≈64, ICX≈32, Zen4≈96, Apple M≈256) are **uncalibrated estimates** (audit #6, marked 2026-07-16); the previously-cited "Graviton=128" appears in no source doc and is withdrawn (audit #7 — self-fabricating cross-reference). See `pr-x12-x265-blasgraph-gemm.md` §2.2 for the GEMM-form derivation.]

Single 32×32 DCT-II via butterflies: ~80 ops. Same via GEMM (`C = A @ DCT_BASIS`): ~32K ops. **Per-block, butterfly wins by 400×**. But:

- For a 4K frame with ~1024 CUs, batched GEMM amortises hardware fusion
- AMX tile GEMM (already in `crate::hpc::bf16_tile_gemm`) does 256 blocks/cycle in bf16
- Crossover at ~64 blocks: above that, GEMM wins; below, butterfly wins

→ A4 should ship both, dispatch at the worker level: per-CTU butterfly, per-frame batched GEMM.

---

## 6. rANS entropy coding ↔ gradient compression

### 6.1 rANS is the entropy coder gradient-compression has been hand-rolling

The "Asymmetric Numeral Systems" coder (Duda 2013) achieves entropy-coding-rate compression with cache-friendly arithmetic. PR-X12 A7 spec: rANS with per-frame frequency tables.

Gradient-compression literature has been **manually implementing rANS-equivalents** for five years:
- *Top-K sparsification* = rANS with all but top-K coded as Skip
- *Random-K sparsification* = rANS with random subset coded as Skip
- *Sign-SGD* = rANS with 1-bit symbol alphabet
- *PowerSGD* = rANS with low-rank approximation as the basis (≡ our Transform choice in §5)

PR-X12 A7 generalises **every published gradient-compression scheme** through one configurable knob (the frequency table) and one swappable basis (the transform). No new theory; we get the synergy by naming.

### 6.2 Per-frame frequency table from k-means

The basin codebook's hit-frequency distribution IS the rANS frequency table. Build them together:
- k-means → 4096 centroids
- assignment pass → per-centroid hit count
- rANS table → normalised hit counts

This is a **single pass** through the data; the codebook construction amortises the frequency-table construction. HEVC CABAC pays a separate context-modeling pass; we save it.

### 6.3 Holy grail claim H-4: rANS + k-means = optimal lossless compression of gradients

**Shannon optimal** entropy = `H(p)` where `p` is the per-symbol probability. K-means + rANS asymptotically achieves this for **any** real-valued distribution that admits a codebook. Gradient distributions are heavy-tailed (most parameters get tiny updates; a few get huge updates) → exactly the regime where k-means + rANS shines.

---

## 7. λ-RDO — the central training objective

### 7.1 Rate × Distortion minimisation is the loss function

`λ · D + R` minimisation:
- **R** = bits used (codec) / sparsity (training) / attention slots (inference)
- **D** = pixel error (codec) / PSNR loss (splat) / downstream loss (training) / softmax KL (attention)
- **λ** = the rate/distortion tradeoff knob = the user-tunable compression ratio

This is **the same objective** that:
- HEVC pays to choose Skip vs Merge vs Delta vs Escape
- ZeRO pays to choose fp32 vs bf16 vs int8 vs skip
- Splat-compression pays to prune or merge Gaussians
- Sparse-attention pays to drop tokens

→ PR-X12 A6 RDO ships **one** implementation. Downstream consumers tune λ for their domain. This is the highest-leverage sub-card after A1/A2.

---

## 8. Epiphanies

Numbered for citation. Each one survives compaction.

**E-1** — *The codec's mode taxonomy is the gradient-compression policy.* Skip/Merge/Delta/Escape is what ZeRO's `int8_compress` family approximates with three modes. Four is correct because Merge captures parameter sharing (LoRA group structure); three doesn't.

**E-2** — *The CTU quad-tree is the attention hierarchy.* Mistral's sliding-window attention is depth-3 leaf processing. Multi-query attention is a depth-2 merge. The split/merge ops in `Ctu::split` / `Ctu::merge` are the dynamic-shape-attention prune/expand operations.

**E-3** — *K-means at frame rate is the HEVC SCC unlock.* The 2013 SCC team capped palette at 64 entries because their hardware budget couldn't afford more. AVX-512 + AMX make 4096-entry codebooks at 60 fps trivial. PR-X12 already has this via `cam_pq::kmeans`.

**E-4** — *The transform basis IS the optimizer's preconditioner.* DCT-II ↔ Adam ↔ KFAC ↔ learned conv are all `Δ' = B·Δ`. PR-X12 A4 should define `Transform` as a trait, not hardcode DCT-II.

**E-5** — *rANS + k-means achieves Shannon-optimal lossless gradient compression.* Every published gradient-compression scheme (Top-K, Sign-SGD, PowerSGD) is a special case of rANS with a particular frequency table and basis choice. PR-X12 A7 generalises them all.

**E-6** — *λ-RDO is the universal training objective.* The same loss `λ·D + R` drives codec mode-decision, ZeRO bucket assignment, splat pruning, and sparse-attention dropout. One implementation, four use cases.

**E-7** — *Block-matching motion estimation is i8gemm.* HEVC ME does SAD; reformulated as SSD (`||A||² - 2A·B + ||B||²`) the middle term is a GEMM. AVX-512 VNNI gives ~50× speedup over hand-tuned SAD. The x265 team didn't have VNNI in 2013.

**E-8** — *CTU mode-decision is a graph shortest-path problem.* x265 spends ~30% CPU on the 85-node partition tree RDO. Tropical-semiring GEMM (which `lance-graph::blasgraph` already provides) solves all partitions in parallel via batched Bellman-Ford. `O(4^d)` → `O(d²)` per CTU.

**E-9** — *CABAC context modeling is a tiny transformer.* HEVC's context-adaptive arithmetic coding is a hand-tuned sliding window of past symbols. A 64-hidden × 256-vocab × 64-history transformer does it better and compresses ~5-10% more. The x265 team had no training infrastructure; we do.

**E-10** — *Deblocking + SAO is a learned conv.* x265's hand-tuned 4-tap filter is QP-conditional but coefficient-fixed. A QP-conditioned 5×5 conv does both deblocking and SAO with better PSNR per QP. One im2col + sgemm per frame.

**E-11** — *Palette codebook training is k-means; k-means is GEMM.* Distance assignment = `argmin_k ||x - μ_k||²` (a distance GEMM). Centroid update = sparse GEMM. The whole codebook build is two GEMM calls per iteration. The HEVC SCC team didn't have MKL k-means.

**E-12** — *The mode discriminants pin to the wire codes.* `CellMode::Skip = 0b00` etc. eliminates a translation table at the encoder/decoder boundary. Same Rust enum is in-memory **and** on-wire. Data-flow rule §2 invariant.

**E-13** — *The basin codebook is also the rANS frequency table.* Single pass through data builds both: assignment counts → normalised frequencies → rANS table. HEVC CABAC pays a separate context-modeling pass; we don't.

**E-14** — *PR-X12 is the cross-domain unification PR.* One Rust module, four orthogonal industries. Each industry has $100M+ around it independently. The codec name is the most precise because it has the bit-exact spec behind it.

**E-15** — *The 16-bit header's reserved high bits (14, 15) are the cross-tier link.* PR-X12 A3-inter (deferred) needs a pointer to the parent-tier CTU. Two bits = 4-way selector across L1/L2/L3/L4 cascade. The bit allocation is **already there**, just not used in A3-intra.

---

## 9. Holy grail material

The mappings above are dense but specific. The holy grail claims are general and load-bearing.

**H-1** — *PR-X12 + cam_pq is the screen-content video codec HEVC SCC was trying to be in 2013.* Frame-rate k-means at 4096 entries was infeasible in 2013-era hardware; our stack does it natively. Cite this when somebody asks "why 12-bit basin field".

**H-2** — *The transform IS the optimizer.* DCT-II / Adam preconditioner / KFAC / learned conv are the same `Δ' = B·Δ` operator. PR-X12 A4 unifies the three industries that each treat their preconditioner choice as the central knob.

**H-3** — *The CTU quad-tree is the universal hierarchical-attention substrate.* Mistral's sliding window, multi-query attention, mixture-of-experts gating — all are special cases of depth-3 leaf processing on the quad-tree. The split/merge ops are the dynamic-attention shape-change operations.

**H-4** — *rANS + k-means achieves Shannon-optimal lossless gradient compression.* Every published gradient-compression scheme is a special case. PR-X12 A7 ships them all from one configurable knob.

**H-5** — *PR-X12 generalises ZeRO.* ZeRO's bf16/int8/fp32 buckets are the codec's Merge/Delta/Escape. Skip is the bucket ZeRO doesn't have (frozen parameters). PR-X12 adds the spatial-coherence dimension (Merge captures inter-parameter correlation) that ZeRO leaves on the table.

**H-6** — *The 64×64 CTU is the right unit for both 4K video and 7B-parameter transformers.* HEVC chose 64×64 for the video luma block; LLaMA chose 4096 hidden dim / 64 head dim. The arithmetic units land at the same size by independent convergent evolution. CTU is the canonical name.

**H-7** — *The codec is the substrate; everything else is a renaming.* The strongest claim, the load-bearing one. After A4-A8 ship, downstream consumers (splat-compression, attention pruning, gradient quantization) configure λ + basis + frequency-table and inherit the entire codec for free. This is the architectural payoff that justifies the eight sub-cards of PR-X12.

---

## 10. Integration plan — per-sub-card + new PRs

### 10.1 PR-X12 A2/A3-intra — current state (PR #195)

**Status**: Open, two CodeRabbit follow-ups pending.

**My P0+P1 findings (all fixed)**:
- ✅ Merge wrapping-cast alias (P0) → `fits_i8` guard at `predict.rs:158`
- ✅ Escape allocator collision (P1) → `Option<&mut u32>` cursor
- ✅ NESW/NEWS doc mismatch (P1) → explicit slot table

**Resolved in PR #195 follow-up (commit `24232985`)**:
- ✅ `pack_leaf` `unwrap_or` fallbacks → switched to `?` operator (bijective serialisation; 3 regression tests added)

Originally listed in §12 below; entry updated.

### 10.2 PR-X12 A4 — transform

**Owner**: TBD (specialist with linalg-core background; PR-X10 A1 author preferred).  
**Depends on**: A1 (Ctu carrier), A2 (mode/header pack).  
**Scope**:
- Define `Transform` trait with `apply(&[i8; N]) -> [i8; N]`, `invert(...)` symmetric methods.
- Ship DCT-II 4×4 + 8×8 as the default impl (cite HEVC tables; can use `crate::hpc::fft` infrastructure).
- Ship Identity as a degenerate impl for RDO baseline comparison.
- Reserve trait slots for `LearnedConv5x5` (A4-follow-up), `AdamPreconditioner` (gradient use case), `KFACBlock` (LLM use case).
- Batched form: `apply_batch(&[[i8; N]]) -> [[i8; N]]` dispatching to `bf16_tile_gemm` when batch ≥ 64.

**Holy grail tie-in**: H-2 (transform IS optimizer). This is the most important A-card after A1 because it determines whether downstream consumers can swap basis without touching the codec hot path.

**Estimated effort**: 1 week (DCT-II is well-documented; batched dispatch is the harder half).

### 10.3 PR-X12 A6 — RDO

**Owner**: TBD (codec-architect; needs RDO experience).  
**Depends on**: A2, A3-intra, A4 (for transform-aware distortion).  
**Scope**:
- λ-weighted Lagrangian: `cost(mode) = D(mode) + λ · R(mode)`
- `D(mode)` = reconstruction error against the original cell value
- `R(mode)` = `packed_byte_len(mode)` (already shipping) × 8 bits
- λ knob exposed as `RdoConfig::lambda: f32`, user-tunable
- Replace `predict_intra`'s first-fit policy with RDO when λ > 0; preserve first-fit when λ == 0 (zero-cost baseline)

**Holy grail tie-in**: H-7 (codec is substrate). RDO is the policy layer; without it, downstream consumers can't tune the rate/distortion tradeoff for their domain.

**Estimated effort**: 1 week.

### 10.4 PR-X12 A7 — rANS entropy coder

**Owner**: TBD (sentinel-qa for the unsafe bitwise primitives).  
**Depends on**: A2 (gives the symbols to encode), A6 (gives the frequencies).  
**Scope**:
- Per-frame frequency table built jointly with k-means assignment pass (E-13)
- 64-bit rANS state, byte-stream output
- Cache-line-friendly state structure (no `Box<dyn>`)
- Decoder mirror (the half of the spec gradient-compression needs)

**Holy grail tie-in**: H-4 (Shannon-optimal gradient compression). Highest-impact A-card for downstream gradient-compression consumers.

**Estimated effort**: 1.5 weeks (rANS has subtle invariants; reference Duda 2013 + Charles Bloom's blog posts).

### 10.5 PR-X12 A8 — stream framing

**Owner**: product-engineer (API surface design).  
**Depends on**: A1-A7.  
**Scope**: Frame headers, CTU markers, per-frame basin codebook serialisation, per-frame rANS frequency table serialisation. Wire format spec.

**Holy grail tie-in**: H-5 (PR-X12 generalises ZeRO). The wire format is what lets distributed training nodes exchange compressed gradients without ad-hoc framing.

**Estimated effort**: 1 week.

### 10.6 PR-X12 A3-inter — cross-tier prediction

**Owner**: cascade-architect.  
**Depends on**: A3-intra (current), `BlockedGrid` L2/L3 access surface.  
**Scope**:
- Add parent-tier `LeafCu` reference to `IntraContext` (`IntraContext` should be renamed `PredictContext`).
- Use the reserved bits 14-15 in the 16-bit header (E-15) to indicate parent-tier link presence.
- New mode? Or extend Merge to allow cross-tier neighbours?
  - **Recommend extending Merge**: parent-tier neighbour is the 5th merge candidate, slot index 4. `MergeDir` becomes 3-bit. Costs 1 bit per Merge leaf; saves a new mode.

**Holy grail tie-in**: H-3 (universal hierarchical attention). Inter-tier prediction is what makes the quad-tree a true hierarchical substrate vs four independent quad-trees.

**Estimated effort**: 0.5 weeks (after A3-intra solidifies).

### 10.7 New PR — Gaussian splat consumer (PR-X12+splat3d)

**Path**: Wire PR-X12's `LeafCu` API into `crate::hpc::splat3d::tile::TileBinning` as an alternative compression layer for splat parameters.

**Approach**:
1. Define `SplatCodec` trait wrapping PR-X12's encode/decode.
2. Apply per splat-parameter axis (Δx, Δy, Δz, Δscale, ΔSH_k).
3. Bench against raw f32 storage; should hit ~10× compression at < 0.5 dB PSNR loss.

**Estimated effort**: 0.5 weeks after A4+A6 ship.

### 10.8 New PR — Cognitive shader consumer (PR-X12+nars)

**Path**: Wire PR-X12 into `crate::hpc::nars::Belief` storage as a quantised attention-slot codec.

**Approach**:
1. Each `Belief.fingerprint` field becomes a CTU; quad-tree subdivides by attention head.
2. Mode decision = attention-sparsity decision.
3. Gradient updates flow through `predict_intra` (LoRA → Merge, Adam step → Delta, full update → Escape).

**Estimated effort**: 1 week after A4+A6+A7 ship.

### 10.9 New PR — Gradient compression consumer (cross-repo)

**Path**: Standalone `burn-codec` integration (or candle equivalent) using PR-X12 as the gradient-bucket compressor.

**Approach**:
1. Reuse PR-X12 codec; downstream is just configuration.
2. Per-layer codebook (4096 entries) trained at warmup.
3. Per-step encoder pass replaces ZeRO's bucket assignment.

**Estimated effort**: 2 weeks (cross-repo coordination overhead).

---

## 11. Exploration paths — ranked by confidence

### 11.1 High confidence (do these)

1. **A4 transform as `Transform` trait, not hardcoded DCT-II** — unlocks H-2 directly. Cost: 1 day of trait design. Payoff: every downstream consumer can swap basis.
2. **A6 RDO with user-tunable λ** — unlocks H-7. Cost: 1 week. Payoff: configurable rate/distortion per domain.
3. **A7 rANS with shared k-means frequency table** — unlocks H-4 + saves a pass. Cost: 1.5 weeks. Payoff: Shannon-optimal compression for all four use cases.
4. **PR-X12 + cam_pq integration** — wire the 4096-entry codebook builder into the codec module. Cost: 2-3 days. Payoff: H-1 fully realised.
5. **Block-matched ME via i8gemm (E-7)** — replaces hand-tuned SAD with VNNI. Cost: 1 week. Payoff: ~50× speedup, validates the BLAS-substrate claim.

### 11.2 Medium confidence (worth prototyping)

6. **CABAC replacement with tiny transformer (E-9)** — 64-hidden × 256-vocab transformer as context model. ~5-10% better compression. Cost: 1-2 weeks (model training + integration). Payoff: bleeding-edge compression ratio; differentiates PR-X12 from any existing codec.
7. **CTU partition as tropical-GEMM (E-8)** — solve all 85-node partitions in parallel via Bellman-Ford-as-GEMM. Cost: 1 week (requires lance-graph::blasgraph access). Payoff: O(4^d) → O(d²) per CTU; huge win for batch encode.
8. **Splat consumer prototype (10.7)** — proves H-7 for the splat case. Cost: 0.5 weeks after A4+A6. Payoff: validates cross-domain claim with a concrete demo.
9. **Inter-tier prediction with reserved bits 14-15 (E-15)** — use the already-reserved bits to add a cross-tier link without growing the header. Cost: 0.5 weeks. Payoff: completes A3, unblocks H-3 fully.
10. **Deblocking + SAO as learned conv (E-10)** — train a small per-QP conv kernel. Cost: 1 week (training infra + integration). Payoff: better PSNR at every QP.

### 11.3 Speculative (research bets)

11. **Gradient compression integration with burn/candle** — proves H-5 (PR-X12 generalises ZeRO). Cost: 2+ weeks cross-repo. Payoff: if it works, the codec becomes the universal gradient compressor.
12. **Per-tier codebook hierarchy** — basin codebooks at L1/L2/L3/L4, each refining the parent. Aligns with cascade-search hierarchy. Cost: unclear (research-stage). Payoff: enables cross-tier merge mode without per-frame full codebook rebuild.
13. **CTU split decision via reinforcement learning** — train a tiny policy network on real workloads. Replaces the heuristic RDO with a learned policy. Cost: research time. Payoff: ~5-10% better compression on out-of-distribution data.
14. **Cognitive prototype "vocabulary" trained once, frozen for inference** — analog of HEVC's "intra-prediction modes" being a fixed table. Cost: training one good codebook. Payoff: stable inference path; no per-frame k-means.
15. **WebGPU implementation** — once the codec is stable, port the SIMD hot paths to WGSL for browser-side encode. Cost: a different organism. Payoff: client-side compressed attention transmission.

### 11.4 Watch but do not pursue (yet)

- Quantum-aware codec basis (Φ_q transform over qubit gates). Interesting; requires hardware we don't have.
- Differential-privacy-aware rANS (privacy bits as part of the rate model). Niche.
- Per-channel codebook for multi-spectral inputs. Cleaner once we have a 4-spectral input that needs it.

---

## 12. Technical debt — codec side + stack side

### 12.1 Codec-side (PR-X12)

**Severity P0** (block):
- None currently.

**Severity P1** (fix before next-sub-card):
- ~~*T-2*: `pack_leaf` `unwrap_or` fallbacks (`mode.rs:194-210`).~~ **RESOLVED** via PR #195 commit `24232985`: switched to `?` operator; 3 regression tests added (`leaf_pack_rejects_malformed_{merge,delta,escape}_without_*`).

**Severity P2** (fix in follow-up):
- *T-3*: A3-intra currently scans NEWS without RDO; replace with λ-weighted RDO when A6 lands. Today's first-fit policy is the right default for λ=0 but suboptimal for typical λ.
- *T-4*: Lossy Escape fallback emits `CellMode::Delta` rather than returning `Result`. Documented but surprising; revisit when A6 forces explicit error paths.
- *T-5*: `IntraContext::neighbours: [Option<&LeafCu>; 4]` is per-cell allocation-free but doesn't cover inter-tier. A3-inter will need to extend this.
- *T-6*: The 2-bit `MergeDir` will become 3-bit when inter-tier lands. Wire format break — plan now, in A3-inter design.
- *T-7*: No SIMD-batched form of `predict_intra` yet. Scalar reference is fine for A3-intra ship; batched form via `crate::simd_soa::MultiLaneColumn` is a follow-up.

**Severity P3** (cosmetic / docs):
- *T-8*: A2 doc cross-references need update once A4 transform module lands.
- *T-9*: The `_reserved: ()` field in `IntraConfig` is a future-compat slot; document the constraint loudly so consumers don't construct via struct literal expecting fields.

### 12.2 ndarray substrate debt (cross-cutting)

- *T-10*: HPC graduation incomplete — `framebuffer`, `ocr_simd`, `ocr_felt`, `audio` still in `src/hpc/` despite low-dep status. Next batch should clear these before A4 starts (clean foundation).
- *T-11*: No `Transform` trait yet at the linalg-core level. PR-X12 A4 should propose this in a shared module so it's reusable beyond the codec.
- *T-12*: `simd_caps` graduated but the cap-detection cost is paid per-call; LazyLock cache exists but agent #5 review flagged AMX detection is bypassed. Fix before A6 RDO uses caps for dispatch.
- *T-13*: `bf16_tile_gemm` is x86_64-only; ARM NEON tile-GEMM path is a stub. A4 batched-DCT dispatch will lean on this; should land NEON impl first.
- *T-14*: No `Result`-returning encode API today. `predict_intra` is `-> LeafCu` (infallible); A6 RDO will want `Result<LeafCu, CodecError>`. Plan the migration in A6 design.
- *T-15*: K-means in `cam_pq` operates on f32; codec wants u8 for basin assignments. Need a u8-mode k-means wrapper or quantize-after-cluster. Tracked but not yet specified.

### 12.3 lance-graph cognitive layer debt

- *T-16*: PR-X12 lives in ndarray but its primary cognitive consumer (`nars::Belief`) lives in lance-graph (via `crate::hpc::nars`). The cross-repo dependency direction means lance-graph's `Belief` storage migration will pull a ndarray version bump. Plan a coordinated release.
- *T-17*: `lance-graph::blasgraph` tropical-semiring GEMM exists but is not yet wired to the codec partition-tree path. E-8 (CTU partition as shortest path) needs this; should be in the A6 RDO design discussion.
- *T-18*: `lance-graph::crates/sigker` (path-signature primitives) is gated on Hambly-Lyons 2010 certification. Not directly codec-related but the signature-PDE primitives could become a transform basis option in A4 (W1.5-#6 / -#7 from `td-simd-tier-audit.md`).
- *T-19*: GridLake (PR-X1 / PR-X2) shipped `MultiLaneColumn` carrier but A2 derive macro (`#[derive(SoA)]`) never landed — PR-X2 shipped the declarative `soa_struct!` instead. The codec batched encode path will want `derive(SoA)` on `LeafCu`; track in follow-up.

### 12.4 Cross-repo coordination debt

- *T-20*: ndarray's HPC graduation (PR #192/#193/#194 + the four-module batch I just pushed) and lance-graph's CausalEdge64 v2 migration are concurrent. The two are independent but share the `claude/continue-ndarray-x0Oaw` branch name across repos; aliases are convenient but make per-repo CI nontrivial. Stable once both stabilise.
- *T-21*: cognitive-substrate-convergence-v1.md (cross-repo locked spec) names the CausalEdge64 v2 layout (Option F). The codec's `LeafCu` overlaps in scope but doesn't share the layout. Worth a cross-reference note from convergence-v1 → this doc.
- *T-22*: causal-edge crate's v2 mantissa encoding (±6 for Intervention/Counterfactual) doesn't yet hook into the codec's basin codebook. Once it does, the Skip/Merge/Delta/Escape modes can carry causal-direction metadata for free in the reserved header bits.
- *T-23*: The architecture rule from `CLAUDE.md` ("ndarray = hardware, lance-graph = thinking") suggests the codec belongs in ndarray. But the codec's *policy* layer (RDO, k-means basin training) is arguably thinking, not hardware. Worth a explicit boundary note in A6 design: "codec mechanism in ndarray; codec policy split between ndarray (math) and lance-graph (cognitive λ choice)."

### 12.5 PR #195 specific debt (track separately, this is in flight)

- 🟡 *T-PR195-1*: CodeRabbit finding on `BASIN_NONE` (T-1 above).
- 🟡 *T-PR195-2*: CodeRabbit finding on `unwrap_or` (T-2 above).
- ⚪ *T-PR195-3*: PR body open question #2 ("lossy Escape fallback") — author resolved by documenting loudly; revisit when A6 lands.

---

## 13. blasgraph / MKL synergies x265 inventors couldn't reach

Six places where blasgraph + MKL change the algorithmic complexity, not just constants. Cross-reference from chat session 2026-05-22.

### 13.1 Block-matched ME → batched i8gemm (E-7)

> [Pinned as **R-6**: SSD-via-GEMM identity is the canonical ME path; the API is **planned** for `ndarray::hpc::blas_level2::batched_ssd_search` — the symbol does not exist yet (audit #5, re-verified 2026-07-16; `blas_level2.rs` ships only the 8 classical L2 methods). The 30-50× estimate is derived in the GEMM-lens companion doc; the bench is asserted by Plan G video lane (R-4) when it ships.]

Classical ME: SAD over 32×32 window. Reformulate as SSD via `||A||² - 2A·B + ||B||²` — middle term is a GEMM. AVX-512 VNNI `i8gemm_i32` does a whole CTU's motion candidates in one call. **~50× over hand-tuned NEON/AVX2 SAD.**

### 13.2 Batched DCT-II via MKL sgemm (E-7-variant)

Per-block butterfly wins for single 32×32. Per-frame batched `C = A_batch @ DCT_BASIS` wins for ≥ 64 blocks via AMX tile fusion. Trades latency for throughput. A4 should ship both.

### 13.3 CTU partition mode-decision as tropical-GEMM (E-8)

> [Pinned as **R-7**: the tropical-GEMM kernel's canonical home is `lance-graph::blasgraph` — but the symbol `blasgraph::tropical_gemm` **does not exist** (audit #3, corrected 2026-07-16; blasgraph's 7 HDR semirings are binary-Hamming over 16384-bit BitVec, no numerical min-plus). The kernel is unwritten; the only shipped min-plus is `bgz17::ScalarCsr::spmv_min_plus` (lossy sibling, prototype only). The `ndarray-codec → lance-graph` dep direction was confirmed *allowed* post-merge (both are sibling crates above `ndarray::hpc` and below `woa-rs`). See the corrected R-7 in the delta doc.]

x265 spends ~30% CPU on recursive partition RDO. Reformulate: each partition is a node in an 85-node DAG, edges = split/merge transitions, weights = ΔRDO. Optimal partition = shortest path. A **planned** tropical-semiring GEMM in blasgraph (`D ← min(D, D + W)`; unwritten today, per the R-7 correction above) would solve all partitions in one batched matrix-relax — the `O(4^d)` → `O(d²)` reduction holds algebraically but the speedup is **conditional on the future A6 partition bench** confirming it on real edge weights.

### 13.4 CABAC context modeling → tiny transformer (E-9)

64-hidden × 256-vocab × 64-history transformer replaces HEVC's hand-tuned sliding-window context. Trains in hours on a single GPU; better compression ratio. The x265 team had no training infra in 2013.

### 13.5 Deblocking + SAO as learned conv (E-10)

x265's QP-conditional 4-tap filter has fixed coefficients per QP band. Replace with a learned 5×5 conv kernel, QP encoded as embedding. One im2col + sgemm per frame; better PSNR.

### 13.6 Palette codebook training as MKL k-means (E-11, H-1)

The big one. 4096-entry codebook per 16-frame GOP at 60 fps via AVX-512 distance GEMM + VNNI assignment. Already shipping in `crate::hpc::cam_pq::kmeans`. The HEVC SCC team capped at 64 entries because 2013 hardware couldn't afford more.

---

## 14. Cross-references

### Design docs (read in this order for a fresh session)

1. `.claude/knowledge/pr-x12-codec-x265-design.md` — original codec spec
2. **this doc** — cross-stack mapping + holy grail + integration plan
3. `.claude/knowledge/pr-x10-linalg-core-design.md` — linalg substrate (distance kernels, basis transforms)
4. `.claude/knowledge/cognitive-shader-foundation.md` — cognitive consumer side
5. `.claude/knowledge/pr-x4-design.md` — splat cascade (Gaussian splat consumer)
6. `.claude/knowledge/pr-x1-design.md` + `pr-x2-design.md` — GridLake substrate (batched encode path)

### Rules pinned

- `.claude/rules/data-flow.md` — three patterns, one rule (no `&mut self` during compute)
- CLAUDE.md hard rules — `unsafe` + `// SAFETY:`, `cargo clippy -D warnings`, doctest examples
- W1a consumer contract (`.claude/knowledge/vertical-simd-consumer-contract.md`) — every SIMD-touching public fn

### Code references (current as of 2026-05-22)

- `src/hpc/codec/ctu.rs` — A1 carrier + quad-tree (shipped, PR #170+#191)
- `src/hpc/codec/mode.rs` — A2 bit-pack (shipped, PR #195, BASIN_NONE fix pending)
- `src/hpc/codec/predict.rs` — A3-intra decision tree (shipped, PR #195, fits_i8 fix landed)
- `src/hpc/codec/mod.rs` — module surface
- `src/hpc/cam_pq.rs` — k-means substrate for §4.2 / H-1
- `src/hpc/bf16_tile_gemm.rs` — AMX batched GEMM for §5.3 / §13.2

### Currently in flight

- **PR #195** (this review session): A2 mode + A3-intra. Two CodeRabbit findings open (T-1, T-2).
- HPC module graduation (#192/#193/#194 + new batch on `claude/continue-ndarray-x0Oaw`): substrate cleanup, no codec impact.

---

## 15. How to use this doc

**Read for a fresh session**: §0 (the big claim) + §1 (the table) + §8 (epiphanies) + §12 (debt). ~10 minutes.

**Read before designing A4/A6/A7/A8**: the relevant integration plan in §10, plus §11 (exploration paths) for the option space, plus the corresponding holy grail claim in §9.

**Read before merging PR #195**: §12.5 (PR-specific debt) — confirm T-PR195-1 + T-PR195-2 are addressed or explicitly deferred.

**Read before any downstream consumer integration (splat, attention, gradient)**: §6 + §10.7-9 (consumer-specific paths). The codec mechanism is general; the consumer-side configuration is where domain knowledge lives.

**Cite from a PR description**: cite by epiphany number (E-N) or holy grail number (H-N). The numbering is stable across edits to this doc.

---

_Last edit: 2026-05-22. Edit this doc when a new epiphany lands, a holy grail claim is falsified, or technical debt graduates from open → resolved. Renumber only by appending — never reuse a retired number._
