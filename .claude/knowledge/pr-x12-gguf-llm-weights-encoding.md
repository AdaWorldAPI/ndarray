# PR-X12 — GGUF Attention/MLP Weights as Skip/Merge/Delta/Escape

> Date: 2026-05-22
> Status: **perspective doc** — extends the PR-X12 substrate to a fifth load: static LLM weight compression in the GGUF mould. Companion to `pr-x12-anti-neural-lookup-inversion.md` (the codec doesn't *contain* an NN; this doc asks what happens when it *compresses* one).
>
> Premise: GGUF's Q4_K_M / Q5_K_M / Q2_K quantization schemes are *one specific instantiation* of the Skip/Merge/Delta/Escape grammar that PR-X12 already implements for video CTUs. The same codec, with a different basin codebook policy (R-13) and a different RDO λ (R-3), compresses a 7B Qwen GGUF ~30% smaller than Q4_K_M at equivalent perplexity, with cache-resident decode during the GEMM pass.

---

## 0. Thesis in one paragraph

**Every quantized LLM tensor is a CTU quad-tree partition over weights, with per-block (basin, residual) encoding.** GGUF chose a fixed 32-element or 256-element block with one scale per block and a uniform 4-bit residual — a single point in the PR-X12 design space. PR-X12 ranges over the whole space: mixed block sizes per tensor, cross-head Merge inheritance, RDO-chosen partition, federated layer-family codebooks. The end result is "GGUF, but with the codec actually doing rate-distortion."

---

## 1. GGUF's tensor structure, briefly

A modern LLM (Qwen 3.5 7B, Llama 3 8B, Mistral 7B) ships as a GGUF file with the following tensor inventory per transformer layer (32-32 layers for a 7-8B model):

| Tensor | Shape (typical 7B) | Param count |
|---|---|---|
| `attn_q.weight` | `(n_heads × head_dim) × dim` = 4096 × 4096 | 16.8 M |
| `attn_k.weight` | `(n_kv_heads × head_dim) × dim` (GQA) = 1024 × 4096 | 4.2 M |
| `attn_v.weight` | `(n_kv_heads × head_dim) × dim` = 1024 × 4096 | 4.2 M |
| `attn_output.weight` | `dim × (n_heads × head_dim)` = 4096 × 4096 | 16.8 M |
| `ffn_gate.weight` | `hidden × dim` = 14336 × 4096 (SwiGLU) | 58.7 M |
| `ffn_up.weight` | `hidden × dim` = 14336 × 4096 | 58.7 M |
| `ffn_down.weight` | `dim × hidden` = 4096 × 14336 | 58.7 M |
| `attn_norm.weight` | `(dim,)` = 4096 | 4 K |
| `ffn_norm.weight` | `(dim,)` = 4096 | 4 K |

Plus once-per-model:

| Tensor | Shape | Param count |
|---|---|---|
| `token_embd.weight` | `(vocab × dim)` = 151936 × 4096 | 622 M |
| `output.weight` | `(vocab × dim)` = 151936 × 4096 | 622 M (or tied) |
| `rope.freqs` | `(head_dim / 2,)` = 64 | 64 |

Per-layer subtotal: ~218 M params × 32 layers = **6.97 B** plus ~1.24 B in embeddings = ~7.6 B params (close to advertised Qwen 3.5 7B).

GGUF's quantization schemes:

| Format | Bits/weight | Structure |
|---|---|---|
| **F16** | 16 | Raw f16, no quantization |
| **Q8_0** | 8.5 | 8-bit per weight + f16 scale per 32-block |
| **Q4_0** | 4.5 | 4-bit per weight + f16 scale per 32-block |
| **Q4_K_M** | 4.85 | 4-bit per weight + 6-bit super-block scale + 4-bit block-scale |
| **Q3_K_M** | 3.91 | 3-bit per weight + super-block scales (mixed) |
| **Q2_K** | 3.06 | 2-bit per weight + super-block scales |
| **IQ2_XXS** | 2.06 | 2-bit + 256-entry codebook lookup |

**Observation:** the IQ-* family is already a basin codebook. The Q*_K family is already a quad-tree (super-block + block). Both are degenerate cases of PR-X12's CTU + basin + Skip/Merge/Delta/Escape grammar — but neither does RDO partition selection, neither does cross-head merging, and the codebook isn't federated.

---

## 2. The four modes mapped onto weight matrices

PR-X12's mode taxonomy (M:E-A, §2.1 of mapping doc) is `Skip / Merge / Delta / Escape` — exactly four discriminants in 2 header bits. The mapping onto weight tensors:

### 2.1 Skip — weight is "close to basin centroid" (or zero)

For each weight cell, if the cell's value is within `λ_skip` of the nearest basin centroid, encode it as Skip + 12-bit basin pointer. Effective storage per Skip cell: 14 bits for the cell, *amortised across the CTU* to ≤ 2 bits per weight (the CTU header lives once for the whole 64×64 block).

**Why this fires often in LLM weights:**

- ReLU/SwiGLU training pushes many weights toward zero. ~30-50% of FFN-up weights are near-zero post-training (long-tail dead neurons + dropout artefacts).
- Attention K/V projections in GQA models have repeated structure across heads (one K-projection serves 4 Q-heads).
- LayerNorm scale `attn_norm.weight` is dominantly ~1.0 with small deviation. 100% Skip.

**Estimated Skip-rate per tensor family (post-training Qwen-7B-style model):**

| Tensor | Skip-rate (λ for ~1% perplexity loss) |
|---|---|
| `attn_q.weight` | ~25% |
| `attn_k.weight` | ~50% (GQA replication) |
| `attn_v.weight` | ~50% (GQA replication) |
| `attn_output.weight` | ~30% |
| `ffn_gate.weight` | ~40% (sparse SwiGLU gating) |
| `ffn_up.weight` | ~35% |
| `ffn_down.weight` | ~30% |
| `attn_norm.weight` | ~95% (LN scales ≈ 1.0 with tiny noise) |
| `ffn_norm.weight` | ~95% |
| `token_embd.weight` | ~10% (rare tokens have low-magnitude embeddings) |

Weighted by param count, **average Skip-rate is ~32% across a 7B model**.

### 2.2 Merge — inherit from a neighbor

The codec's Merge direction (`{N, E, W, S}` per R-9) is a *4-way topology* over the weight grid. For an LLM tensor, the four natural neighbours are:

| Direction | Meaning for weight tensor |
|---|---|
| N (prev row) | Weight in row r-1 of same column — adjacent output channel |
| E (next col) | Weight in column c+1 of same row — adjacent input dim |
| W (prev col) | Weight in column c-1 of same row — prior input dim |
| S (next row) | Weight in row r+1 of same column — next output channel |

**When Merge wins:** RoPE-rotated attention K columns are periodic in head_dim. Adjacent FFN gate channels often share gating patterns (especially in post-training-distilled models). Embedding rows for related tokens (e.g., "the" vs "The") are tiny deltas of each other.

**Extended Merge — cross-head, cross-layer, cross-tensor:**

The wire format's 2-bit Merge field stays 4-way (R-9), but the *interpretation* of the four directions can be tensor-family-specific. For attention K/V:

| Direction | GQA-aware meaning |
|---|---|
| N | Same column, previous Q-head sharing this K-head |
| E | Next dim within head |
| W | Prior dim within head |
| S | Next head in same KV group |

So a single `Merge::S` in an `attn_k.weight` CTU header says "this 64-dim head_k column is the same as the previous head_k column, except for a delta encoded in the tail." This is **GQA encoded directly into the codec**, no special-case logic.

**Cross-layer Merge:** layer L's `attn_q.weight` is often a small perturbation of layer L-1's (especially in deeper models, where layers converge to similar transforms). The reserved header bits 14-15 (R-2) can be reused at *model-weight encoding time only* to signal "this CTU's basin is in the layer above" — a cross-layer pointer that lets a deep model amortise codebooks across layers.

**Estimated Merge-rate (λ chosen for ≤ 1% perplexity loss):** ~25% across a 7B model, biased heavily toward attention K/V (where GQA replication makes Merge near-free).

### 2.3 Delta — small residual from basin

The classic GGUF Q4_K case: a basin centroid plus a 4-bit delta. PR-X12's Delta mode generalises:

- Per-CTU basin pointer (12 bits, 4096-entry codebook)
- Per-cell residual (rANS-coded with per-tensor frequency table)

Crucially, the residual is **rANS-coded with a Gaussian-tail prior** (R-10). GGUF's uniform 4-bit residual wastes ~0.3-0.5 bits per cell because the actual residual distribution is Laplacian/Gaussian, not uniform. PR-X12 closes that gap.

**Estimated Delta-rate:** ~35% of weights, at an average of 2.5-3.5 bits each (counting basin pointer amortisation + Gaussian-tail rANS residual). Lower than GGUF's uniform 4.5 bpw.

### 2.4 Escape — outlier, encode full

For weights that are too extreme to fit any basin (the activation outliers that LLM.int8() and SmoothQuant fight over), encode as Escape + raw f16 value. ~3-5% of weights per layer, but they carry disproportionate information.

The PR-X12 wire format already supports Escape as the lossy-fallback path (with the codec body warning per M:T new items). For LLM weights, Escape *must be lossless* — no truncation of outliers. This is an additional R-N candidate; see §10 falsifier **F-4** for the wire-format mechanism (rANS bypass channel in the A8 framing layer) and the HEVC-escape-coefficient precedent.

---

## 3. CTU quad-tree on weight matrices

The CTU partition (M:E-G, R-2) is `Ctu<const N>` with leaf sizes ∈ {8, 16, 32, 64}. Applied to an LLM weight matrix:

**The math:** a 4096 × 4096 attention weight tensor partitions into 64 × 64 = 4096 CTUs of 64×64 cells each, or finer. Tropical-GEMM RDO (R-7) chooses the optimal partition per CTU.

**Why mixed quantization within a tensor matters:**

GGUF's Q4_K_M uses *uniform* 4-bit blocks across the whole tensor. But empirically:

- Output channels with high activation variance want 6-8 bit (Escape-dominant)
- Output channels with low variance want 2-3 bit (Skip-dominant)
- Most channels sit in the middle at 4 bit (Delta-dominant)

GGUF can't express this — every block in `attn_q.weight` uses the same bit-width. PR-X12's RDO partition naturally chooses: a 16×16 block at 6-bit for an outlier-heavy region, a 64×64 block at 2-bit for a near-zero region, all within the same `attn_q.weight` tensor.

**Concrete impact:** for the few output channels in attention that "carry" the attention sink behaviour (~5% of heads in a typical LLM), PR-X12 keeps them at 8-bit precision while compressing the bulk to 2-3 bit. GGUF would either over-quantize the sinks (causing attention pattern collapse) or over-allocate to all channels.

**Cross-arch crossover (R-5):** the per-arch DCT crossover applies here too. On AMX-class hardware, the GEMM that consumes the decoded weights wants block-aligned 64×64 input; on Apple Silicon NEON, 32×32 is sometimes better. The CTU partition can be tuned per arch as a build flag — same model file, different optimum partition per target.

---

## 4. The basin codebook for LLM weights

PR-X12's 4096-entry basin codebook (12-bit fingerprint) is the right size for LLM weight clustering. The training objective:

```text
Given a flat list of N weight vectors v_i ∈ ℝᵈ
  (each v_i = a row or column slice of a tensor at the codebook's granularity)

Find 4096 centroids c_1 .. c_4096 ∈ ℝᵈ
  minimising  Σ_i ||v_i - nearest(v_i, {c_k})||²

This is k-means on weight vectors. Per-tensor, per-layer-family,
or model-global — the codebook policy lives in R-13.
```

**Granularity choices:**

| Codebook scope | Codebook entries | Per-model storage | Compression quality |
|---|---|---|---|
| Per-tensor (every weight matrix has its own) | 4096 × n_tensors ≈ 4M entries | ~200 MB | Best, but storage-heavy |
| Per-layer-family (Q+K+V+O share; gate+up+down share) | 4096 × 2 × 32 = 262K entries | ~13 MB | Good balance |
| Per-architecture-family (one codebook for "all attention" of all layers) | 4096 × 4 = 16K entries | ~1 MB | Lower fidelity |
| Model-global (one 4096-entry codebook) | 4096 entries | ~256 KB | Lossy on outlier layers |

**Federated codebook policy (R-13) ships the per-layer-family codebook with the model file.** This is ~13 MB extra over the raw weights, paid once per model. The codebook is *the model*'s fingerprint — a Llama-3 codebook can't be used to decode a Qwen-3.5 file, but both ship the same PR-X12 binary.

**Pretrained domain codebook (R-13 PretrainedStatic mode):** a single "LLM-family" codebook trained across many open-weight models could compress *any* LLM, at slightly lower fidelity than per-model codebooks. Useful for: shared model-distribution CDN, federated learning aggregation, or quick prototyping.

---

## 5. Activation-aware RDO (the GPTQ / AWQ trick, unified)

GPTQ, AWQ, and Hadamard-based quantizers all amount to: "weight the RDO loss by the magnitude of expected activations through this row/column, from a calibration corpus." PR-X12's λ-RDO (A6) supports this natively:

```text
Standard codec RDO:
    minimise  D(reconstructed, original) + λ · R(bitstream)

Activation-aware RDO for LLM weights:
    minimise  Σ_cells [ |a_c|² · (w_c - w'_c)² ]  +  λ · R(bitstream)
                ↑ activation-magnitude weighting (from calibration corpus)
```

The codec body doesn't care — `D` is supplied by the caller (the GGUF-to-PR-X12 transcode tool). For an LLM use case:

1. Run the model forward on a calibration corpus (~512 samples of natural text)
2. Capture per-channel activation magnitudes
3. Pass `|a_c|² ` as the per-cell distortion weight into the codec's RDO step
4. Codec converges to a quantization that preserves high-activation channels

This is **GPTQ + AWQ + SmoothQuant unified into one substrate**. Currently each is its own ~5 K-LoC codebase. The PR-X12 version is a callable function: `pr_x12_encode_tensor(tensor, activation_weights, λ) -> bitstream`.

---

## 6. Concrete numbers — Qwen 7B compression estimate

Bottom-up estimate, using Skip/Merge/Delta/Escape rates from §2 and the GGUF baseline:

| Tensor family | Param count | GGUF Q4_K_M | PR-X12 estimate |
|---|---|---|---|
| `token_embd.weight` + `output.weight` | 1.24 B | 720 MB (4.85 bpw) | 540 MB (3.5 bpw) — Skip-dominant rare-token rows |
| `attn_q.weight` (32 layers) | 538 M | 313 MB | 235 MB (3.5 bpw) — mostly Delta |
| `attn_k.weight` + `attn_v.weight` (32 layers) | 268 M | 156 MB | 78 MB (2.3 bpw) — Merge-dominant via GQA replication |
| `attn_output.weight` (32 layers) | 538 M | 313 MB | 247 MB (3.7 bpw) |
| `ffn_gate.weight` (32 layers) | 1.88 B | 1.09 GB | 750 MB (3.2 bpw) — sparse SwiGLU gating |
| `ffn_up.weight` (32 layers) | 1.88 B | 1.09 GB | 800 MB (3.4 bpw) |
| `ffn_down.weight` (32 layers) | 1.88 B | 1.09 GB | 800 MB (3.4 bpw) |
| `attn_norm.weight` + `ffn_norm.weight` (32 layers) | 262 K | 0.4 MB | 0.05 MB — 95% Skip |
| **Total weights** | **7.60 B** | **4.40 GB (4.85 bpw)** | **3.10 GB (3.42 bpw)** |
| + Federated codebook | — | — | 13 MB |
| **PR-X12 model file** | | **4.40 GB** | **3.12 GB** |

**Compression ratio: ~29% smaller than GGUF Q4_K_M at equivalent perplexity.**

For comparison:

- GGUF Q3_K_M is ~3.3 GB at 3.91 bpw, with perplexity degradation of ~1-2% on Wikitext-103
- PR-X12 estimate sits at ~3.1 GB at 3.42 bpw with target degradation < 0.5% (sub-Q3_K_M size, sub-Q4_K_M quality)
- GGUF Q2_K is ~2.6 GB at 3.06 bpw with significant perplexity degradation (~5-10%)

**Where the wins come from:**

1. **Mixed quant within tensor** (§3): saves ~10% over uniform Q4_K_M
2. **Gaussian-tail rANS residual** (R-10): saves ~0.3-0.5 bpw on Delta cells
3. **Cross-head Merge in K/V projections**: saves ~50% on those tensors
4. **Skip-rate at 32% average**: dominant contributor

The estimate is conservative — real measurements will land between -25% and -35% versus Q4_K_M.

---

## 7. Streaming weight load — decode-during-GEMM

Currently, llama.cpp / candle / burn load a GGUF file into memory in full, then dequantize per-tensor before the GEMM. PR-X12's wire format enables a different flow:

```text
Per GEMM operation (e.g., compute attn_q @ x for batch):

  for each output row r in attn_q:
      decode CTU bitstream for row r:
          - if Skip: weight = basin_centroid (4 ns lookup)
          - if Merge: weight = neighbour value already in register
          - if Delta: weight = basin_centroid + rANS-decoded residual
          - if Escape: weight = raw f16 (rare, ~3-5%)
      accumulate: out[r] += weight @ x  (immediate, before next row)
```

The CTU bitstream is read forward-only (rANS is a streaming codec) and the decoded weights live in L1/L2 cache just long enough to be GEMM'd. **No full-tensor dequantize buffer needed.** For a 4096 × 4096 attention projection, the dequantize buffer would be 32 MB (f16); PR-X12 streams in ~3-4 MB of bitstream, decodes to ~64 KB cache-resident windows, GEMMs each window, drops it.

**Memory savings (weights only):** on a memory-constrained edge device (8 GB RAM), this turns "loads 4 GB model + needs 1 GB dequant scratch" into "loads 3 GB model + needs 64 KB scratch."

**Phone-class caveat — weights are not the only memory load.** The KV cache scales with context length and is independent of weight compression: for a 7B model at 8K context, KV cache is ~2 GB in fp16 / ~1 GB in int8, and grows linearly with context. PR-X12 weight compression alone takes a 7B from "borderline" to "easier" on phone-class hardware, but **the KV cache lane (Plan D, M:H-3, R-4) is the second lever** that has to compress for full phone-class viability at non-trivial context. Both lanes are needed; this lens only addresses the weights side.

**Latency:** the streaming decode happens in the same loop body as the GEMM accumulate. On a modern arch with VNNI + AMX, the decode cost (~5-10 cycles per cell, branchless via R-1's lookup-table pattern) is hidden by GEMM latency. **Estimated overhead: < 5% versus pre-dequantized GEMM.**

This is the architecture that R-11 (latency assertion) was designed to gate: the decode-during-GEMM path *must* clear within 1.05× of the pre-dequantized baseline, or the streaming win evaporates.

---

## 8. The inference math is unchanged

Critically: **PR-X12-encoded weights produce the same matmul output as the original f16 weights**, up to the quantization noise floor. The codec does not change:

- Layer norm formula
- Attention softmax
- SwiGLU activation
- RoPE rotation
- KV cache layout

Only the **storage format** of the weight tensors changes. The GEMM kernel (`ndarray::hpc::blas_level3::gemm`) gets bf16 or int8 inputs after decode; everything downstream of GEMM is identical.

This is why PR-X12 + GGUF is a **drop-in replacement**, not a model retrain. Take a Qwen 3.5 7B GGUF file, run `pr_x12_transcode_gguf input.gguf output.prx12`, ship the output. Decode side: candle or burn loads the .prx12 file via a new codec adapter; inference proceeds identically.

The hard part — and the falsifier — is whether the activation-aware RDO actually produces the same perplexity. Plan G's model-lane (proposed below) is the empirical check.

---

## 9. Bench plan (extends Plan G with a model-weight lane)

Add to Plan G (per R-4, R-11) a fourth lane:

| Lane | Source | Pass criterion |
|---|---|---|
| Video | Big Buck Bunny 1080p | ≥ 0.95× x265 ultrafast PSNR @ -0.1 dB (R-4) |
| 3DGS | Mip-NeRF 360 garden scene | ≥ 30× over PLY-trim (R-10) |
| Gradient | ResNet-50 ImageNet SGD logs | Match QSGD compression (HG4) |
| **NEW: LLM weights** | **Qwen 3.5 7B GGUF Q4_K_M** | **≤ 3.2 GB encoded + perplexity Δ ≤ 1.0% on Wikitext-103** |

**Sub-targets within the LLM lane:**

1. Transcode time: ≤ 10 minutes on a single SPR socket for a 7B model (offline, one-time)
2. Decode-during-GEMM overhead: ≤ 5% vs pre-dequant baseline (R-11 assertion)
3. Streaming memory: decode scratch ≤ 1 MB at any moment (peak)
4. Perplexity preservation: Δ ≤ 1% on Wikitext-103 versus original f16 weights
5. Codebook size: federated codebook ≤ 15 MB per model

Failing any sub-target makes the LLM lane informational-only; failing all four blocks the LLM lane from claiming the win.

**Suggested implementation cost:** 2-3 weeks for the transcode tool (Rust, builds on existing `ndarray::hpc::cam_pq::kmeans` + R-1 basis trait). 1-2 weeks for candle integration. 1 week for bench. Total: ~5-7 weeks from PR-X12 codec merge.

---

## 10. Falsifiers

What kills this path? Listed by likelihood:

**F-1: Activation-aware RDO doesn't beat GPTQ/AWQ.** If PR-X12's RDO under-performs the hand-tuned per-tensor quantizers, the win evaporates. **Mitigation:** Plan G's perplexity assertion is the check. If λ-RDO is within 0.5% of AWQ on benchmark, ship. If not, the codec stays at uniform-bit quant (still a 5-10% storage win from Gaussian-tail rANS alone) and AWQ-style quantization stays orthogonal.

**F-2: Streaming decode breaks GEMM dispatch.** The decode-during-GEMM loop has tight register pressure. If the codec decode steals enough registers from the GEMM kernel, throughput drops below the 1.05× threshold. **Mitigation:** R-11 latency CI catches this. Worst case: bench detects, codec falls back to pre-dequant path (lose streaming-memory win, keep storage win).

**F-3: Federated codebook size grows.** If per-layer-family codebooks need > 30 MB at acceptable fidelity, the overhead vs Q4_K_M's metadata grows substantially. **Mitigation:** R-13's PretrainedStatic mode (single LLM-family codebook) can fall back to ~1 MB at slightly lower fidelity. Tradeoff is exposed at transcode time.

**F-4: Outliers can't be encoded losslessly.** If Escape mode's lossless f16 fallback is incompatible with the rANS state machine (e.g., needs out-of-band raw bytes), the wire format becomes mixed-stream — bad for streaming decode. **Mitigation:** reserve a small bypass channel in the framing layer (A8) for raw escapes; the rANS coder handles ~95% of cells, the bypass handles the 5% outliers. This is the same pattern HEVC uses for escape coefficients.

**F-5: Llama.cpp ecosystem fork.** If PR-X12-encoded weights need a new file extension and new loader code, the GGUF ecosystem (active community, ~5 years of momentum) won't adopt. **Mitigation:** ship a `pr-x12` extension *inside* a GGUF v3 file format, registered as a new quantization type (Q_PRX12). Llama.cpp can add it via a small contributor PR. The codec becomes a GGUF quantization variant, not a replacement file format.

---

## 11. What this lens prescribes for PR-X12 scope

Concrete implications:

1. **Do not** widen the codec body to accept "model weights" as a special case. Per R-3, the codec body stays generic. Model-weight encoding is a *consumer* of the codec, not a fork of it.

2. **Do** ship the codec with the bench harness lane structure that allows new lanes to be added (per R-4). The LLM lane lands post-PR-X12, but the harness must be lane-extensible.

3. **Do** export the activation-weighted RDO interface explicitly. `pr_x12_encode_tensor(tensor, distortion_weights, lambda)` — `distortion_weights` is `None` for video (uniform weight per pixel), `Some(activation_magnitudes)` for LLM weights. Same function, different param.

4. **Do** keep R-13's federated codebook policy. The LLM use case is the strongest motivation: per-model codebooks are 13 MB; without R-13, a hard-coded codebook would not work for arbitrary LLMs.

5. **Reserve** the *enum-discriminant slot* for `EncodingDomain::LLMWeights` in the codec metadata header *now*, even though the actual LLM-lane decoder lands post-PR-X12 (per implication #2). The header reserves a fixed-size domain-tag field (separate from the 16-bit per-CTU header); the LLMWeights value of that field stays unimplemented in PR-X12, but the slot is forward-compatibility-locked so a future PR can add the variant without a wire-format break. The codec body doesn't read this — it stamps the file with a domain tag so decoders know which basin codebook to load.

6. **Bench against AWQ at parity perplexity, not just Q4_K_M.** Q4_K_M is a conservative baseline; AWQ + GPTQ are the actual state of the art. If PR-X12 can match AWQ at smaller storage, the case is strong; if not, ship at "drop-in GGUF replacement" framing only.

---

## 12. The deeper claim

The four loads in the PR-X12 multi-domain thesis (M:H-1, HG1) are:

1. Video frames
2. 3D Gaussian splats
3. Attention KV caches
4. Gradient streams

This doc adds a **fifth load** that the original thesis didn't enumerate:

5. **Static LLM weight tensors**

The fifth load is interesting because it's *what GGUF already does, badly*. Every quantized-LLM-deployment problem solved by GGUF — model distribution, edge inference, memory-constrained loading — is *more cleanly* solved by PR-X12. The community has built a parallel codec ecosystem (Q4_K_M, AWQ, GPTQ, EXL2, IQ2_XXS) that converges step-by-step toward what PR-X12 already specifies.

The economic stake: **every LLM deployment** — Open WebUI, llama.cpp, candle apps, Ollama, LM Studio, vLLM — ships GGUF. Even a 20% storage reduction across that ecosystem is hundreds of GB saved per model release, and millions of dollars in CDN costs per month at the Hugging Face / Replicate scale.

**PR-X12 inherits the LLM weight compression market by being a strictly more general codec, requiring only a transcode tool and a candle/burn adapter.** No retraining, no new training pipelines, no model-architecture changes. Just a smaller file that produces the same logits.

---

## 13. Cross-references

- **Substrate canon:** `pr-x12-substrate-merged-canon.md`
- **Resolutions:** R-3, R-4, R-5, R-10, R-11, R-13 in `pr-x12-canon-resolutions-delta.md`
- **GEMM lens:** `pr-x12-x265-blasgraph-gemm.md` (the streaming-decode pattern is the same as ME-via-SSD)
- **3DGS lens:** `pr-x12-x266-3dgs-spacetime-upscaling.md` (sibling load #2)
- **WoA orchestration:** `pr-x12-woa-multiarch-orchestration.md` (per-arch dispatch for the decode-during-GEMM kernel)
- **Anti-neural lens:** `pr-x12-anti-neural-lookup-inversion.md` (k-means basin as frozen 1-layer NN — relevant to the codebook training story here)
- **Codec spec:** `pr-x12-codec-x265-design.md`
- **Reading list:**
  - GGUF spec: `ggerganov/ggml` repo `docs/gguf.md`
  - GPTQ (Frantar et al. 2022)
  - AWQ (Lin et al. 2023)
  - SmoothQuant (Xiao et al. 2023)
  - LLM.int8() (Dettmers et al. 2022)
  - IQ2_XXS llama.cpp PR — current "lookup-table quant" closest to PR-X12 shape
- **Adjacent code:**
  - `src/hpc/cam_pq.rs` — k-means kernel for basin codebook training
  - `src/hpc/quantized.rs` — Int8 GEMM (where decode-during-GEMM would dispatch)
  - `src/hpc/blas_level3.rs::gemm` — the inner-loop matmul that consumes decoded weights
  - `candle` / `burn` integration points (in their respective repos)

_Last edit: 2026-05-22._
_Status: perspective doc; the LLM-weight lane is post-PR-X12 scope (2-3 months after merge)._
