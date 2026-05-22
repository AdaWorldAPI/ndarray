# PR-X12 — The Anti-Neural Codec: Lookup-Table Inversion of NN Inner Loops

> Date: 2026-05-22
> Status: **wildcard perspective doc** — the most interesting reframe I can articulate of PR-X12's substrate. Companion to the GEMM lens (`pr-x12-x265-blasgraph-gemm.md`), 3DGS lens (`pr-x12-x266-3dgs-spacetime-upscaling.md`), and orchestration lens (`pr-x12-woa-multiarch-orchestration.md`).
>
> Premise: every "neural codec" primitive in current research — VQ-VAE, neural RDO, neural rendering, learned wavelets — has a **frozen lookup-table dual** that achieves the same information-theoretic compression at 50-1000× lower inner-loop cost. PR-X12 systematically picks the lookup-table dual for every inner-loop op, then proves it converges to within an information-theory-bounded ε of the neural codec's compression ratio. The codec has **zero NN forward passes in the inner loop**, by design.

---

## 0. Thesis in one paragraph

**A 4096-entry codebook indexed by a 12-bit fingerprint is structurally equivalent to a 1-layer 12-bit-input MLP that has been frozen and tabulated.** Any neural codec whose inner loop is "embed → match → score" can be replaced by "fingerprint → table lookup → score" for the same expressive power, at table-lookup latency (~3-10 ns) vs NN-forward-pass latency (~3-30 µs). PR-X12 makes this systematic: every primitive that *could* be an NN inner-loop op is instead a lookup table. The result is a codec that has the compression of a neural codec but the inner-loop cost of x265.

This is not anti-NN. It is **anti-NN-in-the-inner-loop**. NNs train the tables. Once trained, the table replaces the NN.

---

## 1. The current research direction: NN-in-loop codecs

Recent codec research direction (2020-2026):

| Codec | NN role | Inner-loop cost |
|---|---|---|
| **Lyra** (Google, 2021) | Neural vocoder decoder | ~3 ms per 20 ms audio frame on a phone |
| **SoundStream** (Google, 2021) | VQ-VAE encoder + neural decoder | ~10 ms per 20 ms audio frame |
| **EnCodec** (Meta, 2022) | Residual VQ-VAE + transformer prior | ~30 ms per 20 ms audio frame on GPU |
| **NVIDIA Maxine** (2020+) | Latent-space face encoding | ~16 ms per 1080p video frame on a 4090 |
| **AOMedia ML-AV1** (research) | Per-CTU NN-based RDO | ~5-20 ms per CTU |
| **Google ML-Image** (2023) | Learned transform + entropy model | ~100 ms per image on GPU |

All of these share a common shape:
- Encoder: input pixels → embedding network → quantize → bitstream
- Decoder: bitstream → embedding → decoder network → output pixels
- Inner loop has *at least one* NN forward pass per emit operation

The compression results are excellent. Lyra hits ~3 kbps speech at 16 kHz quality. EnCodec matches MP3 at ~12× lower bitrate. The inner-loop latency cost is *catastrophic*: ~3-100 ms per emit, vs ~0.1-1 µs for x265's per-block inner loop.

**The structural problem with NN-in-loop:**

1. Each forward pass = thousands to millions of MAC operations
2. Tensor framework overhead (PyTorch / candle / burn) = 50-200 µs per dispatch
3. Model version drift across decoders breaks playback
4. Quantization sensitivity: int8 NN weights vs f16 activations have numerical determinism issues
5. Cannot run inside L1 cache; needs L3 / HBM for weights

---

## 2. The PR-X12 inversion: pre-baked lookup tables

Every NN-in-loop primitive in §1 has a frozen-table dual. PR-X12 is the systematic instantiation of those duals:

| NN-in-loop primitive | PR-X12 lookup-table dual | Inner-loop cost |
|---|---|---|
| VQ-VAE encoder embedding | k-means basin codebook (R-10, M:H-6) | ~10 ns per cell (L1-resident) |
| VQ-VAE decoder | Same codebook, reverse lookup | ~3 ns per cell |
| Neural RDO scoring | Tropical-GEMM partition (R-7) | ~1.4 K ops per CTU |
| Neural rendering | EWA splat rasterizer (Plan E) | ~5-15 ms per 4K frame |
| Learned transform | DCT-II batched GEMM (R-5) | ~256 cycles per 32×32 block |
| Transformer prior / entropy model | Gaussian-tail rANS (R-10) | ~10 ns per symbol |

The codec's inner loop **never** dispatches to a tensor framework. The basin codebook is a fixed `[Fingerprint; 4096]` slice (~256 KB, fits L2). The tropical-GEMM partition runs over an 85-node DAG (~1 KB working set). The DCT basis is a `[i16; N*N]` array (~8 KB for 64×64). All resident in cache, all branchless on the hot path.

**The total NN-flops in the codec's inner loop: zero.** NNs trained the codebooks; the codebooks live in the bitstream / metadata; the decoder does table lookups, not forward passes.

---

## 3. The math: every NN-in-loop primitive has a lookup-table dual

### 3.1 Basin codebook ≡ frozen 1-layer 12-bit MLP

A VQ-VAE encoder's job: map continuous input embedding `x ∈ ℝᵈ` to discrete index `k ∈ {0..K-1}`, where centroid `c_k ∈ ℝᵈ` is the nearest among K learned centroids.

```text
VQ-VAE encoder:    k = argmin_j ||x - c_j||²
VQ-VAE decoder:    x' = c_k
```

**PR-X12 basin codebook (R-10, M:H-6):** same algebraic operation, with the embedding step pre-computed by an OFFLINE training run (k-means over a corpus), then frozen into a 4096-entry codebook indexed by a 12-bit fingerprint.

```text
PR-X12 encoder:    fp = compute_fingerprint(x)    [~10 ns, deterministic hash]
                   k  = codebook.nearest_index(fp) [~3 ns, table lookup]
PR-X12 decoder:    x' = codebook[k]               [~3 ns]
```

**Why this is equivalent in expressive power:**

- A 4096-entry lookup table on 12-bit input is structurally a `[4096]` array — i.e., a 12-bit-input 4096-output discrete function
- Any 12-bit-input network has at most 2^12 = 4096 distinct outputs
- A `Linear(12, K) → argmax` with frozen weights is structurally an array lookup
- The codebook IS the trained network, materialized as data

**Why this is faster:**

- 4096-entry lookup: 1 memory ref (table is in L2 cache, 64 ns p99)
- 1-layer 12-bit-input 4096-output Linear: ≥ 49,152 MACs + softmax + argmax ≈ 3-30 µs
- **Speedup: 1000-5000×** per inner-loop emit

The compression ratio (R) is bounded by Shannon's source coding theorem: R ≥ H(cells). The codebook achieves H(cells) up to a log-factor of K=4096 entries' overhead. A neural encoder achieves the same H(cells) (assuming optimal training). Compression is asymptotically equivalent; latency is not.

### 3.2 Tropical-GEMM RDO ≡ frozen GNN

Neural RDO research (AOMedia ML-AV1, others 2022-2025): train a graph neural network to score quad-tree partition candidates. Each CU is a node; edges are split decisions; node features include local pixel statistics; the GNN outputs a scalar RDO score.

The GNN's expressiveness for this problem maps directly onto tropical-semiring arithmetic:

```text
GNN forward pass on RDO graph:
    h_v^(l+1) = σ(W · aggregate(h_u^(l) for u in N(v)) + b)

where aggregate = sum or max, σ = ReLU, ...

Tropical semiring (R-7) on the same graph:
    h_v^(l+1) = min_u (h_u^(l) + W_{uv})    [identity on min-plus algebra]
```

**Identity:** if the GNN's aggregator is `max` and σ is identity-on-positive, then the GNN forward pass on the RDO graph **is** a tropical-GEMM iteration over the negative semiring. The neural RDO research community has spent ~3 years arriving back at min-plus algebra, the way Bellman-Ford has always solved this.

**PR-X12's tropical-GEMM:**

- O(d²) iterations of `D ← min(D, D ⊕ W)` over 85-node DAG
- Hand-tuned `W` edge weights (or one offline calibration run)
- ~1.4 K ops per CTU (R-7 estimate)

**Neural RDO:**

- Per-CTU GNN forward pass with ~30-50 K parameters
- ~5-20 ms per CTU (10,000× slower)
- Same algebraic information content for the partition problem

**Why frozen wins:** the partition problem is small (85 nodes, d=4 depth). The hand-tuned W matrix has ~340 weights. A learned GNN trained on the same partition problem has 30-50K parameters but the optimum is *low-dimensional*. PR-X12 picks the low-dim solution directly.

### 3.3 EWA splat ≡ frozen 1-layer projection

Neural rendering (NeRF, Mip-NeRF, Instant-NGP): MLPs that map (pos, viewdir) → (RGB, density). Forward pass per pixel during render.

```text
NeRF:           per-pixel MLP forward pass, ~10-100 µs per pixel on GPU
3DGS rasterize: per-Gaussian closed-form EWA projection, ~30-100 ns per Gaussian
```

The 3DGS render *is* the discretized, frozen, closed-form solution that NeRF's MLP was trying to approximate. The 200K Gaussians in a scene are a non-parametric discrete representation of what a NeRF MLP encodes implicitly.

**PR-X12's EWA splat basis (Plan E, future x266):**

- Per-Gaussian: 1 projection (4 MAC), 1 covariance evaluation (6 MAC), 1 tile-binning lookup
- Per-pixel: sort + alpha-blend (already optimized in published 3DGS code)

**Neural rendering equivalent:** ~10,000× slower at comparable visual quality. The compression ratio (scene MB per pixel rendered) is approximately equivalent — within a factor of 2 — because both encode the same 3D scene at the same fidelity. The latency gap is the win.

### 3.4 DCT-II basis ≡ 1-layer linear projection

This one is too well-known to belabor: an N-point DCT-II is a fixed `(N × N)` matrix multiplied against the input. A "learned transform" research codec uses gradient descent to find a (close-to-DCT) transform that's slightly better at the training distribution. The information gain is bounded: most natural images have a near-DCT eigenbasis, and the learned transforms typically beat DCT by <0.1 dB PSNR.

For 0.1 dB PSNR you pay:

- Per-block matrix multiply with the learned weights (~256 cycles, same as DCT)
- *PLUS* the model versioning / training framework / per-arch dispatch headache

PR-X12 chooses DCT-II (R-5) because the gain from a learned transform is below the noise floor of arch-dependent rounding.

---

## 4. Why frozen lookups win at codec inner-loop scale

The four core arguments:

### 4.1 Determinism

Lookup tables produce bit-exact outputs across:
- Compiler version (gcc 12 vs 13 vs clang 18)
- SIMD width (AVX-512 vs SVE2 vs NEON)
- Float rounding mode
- Tensor framework version (PyTorch 2.3 vs 2.4 vs torch.compile)

NN inner loops do not. The 2024 "neural codec evaluation" papers regularly report ±0.5 dB PSNR variation across runs of the *same model* on the *same input* due to non-determinism in CUDA reductions. For a codec, this is a non-starter.

### 4.2 L1 / L2 cache fit

A 4096-entry × 8-byte codebook = 32 KB (fits L1 on most archs). A 100-element tropical-GEMM working set = ~1 KB. An 85-node partition DAG = ~1 KB. Everything in the codec's inner loop fits in L1 + L2.

A neural codec's NN weights (~10-100 MB) sit in L3 or HBM. Per-pixel inner loop fetches from L3 = ~30-50 ns per fetch. Even before MACs, you're paying L3 latency PR inner-loop iteration.

### 4.3 No tensor framework dependency

The codec runs in pure Rust + `ndarray::hpc` SIMD. No PyTorch. No candle (the codec doesn't depend on candle; the inverse is also true). No CUDA dependency for CPU encode. No ROCm.

This matters for deployment: PR-X12 ships in a 5 MB stripped binary; a neural codec needs 50-500 MB of model weights + framework dependencies. For edge / mobile / embedded, this is the difference between "ships" and "doesn't."

### 4.4 No model versioning

A neural codec is essentially a versioned shared model state. Decoder must have the *exact* version that encoded the stream. Cross-vendor decoder interop is impossible without standards bodies (which take years; cf. JPEG XL's ~7-year ratification story).

A frozen-lookup codec's wire format is fully specified by the byte-level layout. The "model" — the codebook — is part of the bitstream or part of the static codec spec. Decoder vendors interop by reading the spec. The codec is *intrinsically* an open standard.

### 4.5 Patentability around ML monopolies

The neural codec space is full of patents on specific model architectures. Encoder using "VQ-VAE + residual transformer prior" is patent-encumbered by Meta (EnCodec), Google (SoundStream), and others. Decoder using "MLP for neural rendering" overlaps with NeRF patents.

A k-means basin codebook + tropical-GEMM RDO + EWA splat codec sits in **mathematically-prior-art** territory. k-means (1957), tropical algebra (1990s applied codec literature), EWA splat (2001). All decades-old, all in the public domain or expired. PR-X12's substrate is intrinsically patent-free.

This is not a small consideration. The H.265 / HEVC patent pool charges $0.02 per device sold; the codec ecosystem pays ~$1B/yr in HEVC royalties. PR-X12's substrate sidesteps this by construction.

---

## 5. The Hutter information-theoretic bound

Marcus Hutter's compression thesis ("Universal AI is compression"): for a stationary source X with entropy H(X), the optimal compression ratio is bounded by H(X). Any codec — neural or frozen-lookup — achieving R = H(X) is *information-theoretically optimal*. There is no further compression to extract.

**Claim:** for the source distributions that PR-X12 targets (video frames, audio waveforms, text streams), the basin codebook + tropical-GEMM partition + DCT transform achieves R within ε of H(X). The ε is bounded by:

- The log-of-codebook-size overhead: log₂(4096) / cell ≈ 12 bits / cell
- The basis approximation gap: DCT vs Karhunen-Loève optimal transform ≈ 0.05 dB PSNR
- The quad-tree partition granularity: 8×8 leaf vs continuous ≈ 0.1 dB PSNR

**Total ε: ~0.2 dB PSNR.** Within the JND (just-noticeable-difference) threshold for human perception.

A neural codec can theoretically close this gap, but only by learning the exact optimal codebook + transform + partition for the *specific* source distribution. The cost: per-source training (hours to days), large model storage (MB to GB), per-inference forward pass (ms per emit). The information gain: ~0.2 dB.

**PR-X12 buys ~0.2 dB of PSNR for 1000-5000× faster inner loop.** That's a Pareto-dominant trade for any deployment where latency matters more than the 0.2 dB.

---

## 6. When NN-in-loop wins

The honest answer: **ultra-low-bitrate, perceptually-tuned, generative codecs.**

For bitrates < 1 kbps (e.g., Lyra speech, neural face codecs at 256 bps), the source distribution is so undersampled that any frozen codebook leaves obvious quality on the table. A neural model can "hallucinate" plausible content from the few bits transmitted, beating a frozen codec by 5-15 dB PSNR equivalent.

This is **codec-as-generative-model** territory, not codec-as-source-coding. The hallucinated content may not match the original (PR-X12's failure-of-completeness vs failure-of-fidelity discussion in the 3DGS doc — same distinction).

For these use cases, the right architecture is a **layered codec**:

1. **Base layer:** PR-X12 frozen-lookup codec for the bits-actually-transmitted
2. **Enhancement layer:** NN generative refinement at the decoder (optional, off by default)

The base layer guarantees fidelity bounded by Shannon. The enhancement layer provides perceptual hallucination when the user opts in. PR-X12's wire format reserves a single bit (M:E-J bit 14 currently used for leaf_size; one of the reserved bits in future revisions) for the "enhancement layer available" flag.

This is also the right architecture for high-stakes content (legal, medical, scientific): always run the base layer, never run the enhancement layer. Determinism preserved.

---

## 7. PR-X12 is the floor; NN can layer on top

The architectural commitment:

```text
              ┌───────────────────────────────────────┐
              │ Optional enhancement layer (NN)        │
              │ - Generative refinement                │
              │ - Off by default; opt-in per use case  │
              │ - Lives in burn/candle, NOT in codec   │
              └───────────────────┬───────────────────┘
                                  │ standardized API:
                                  │ decoded_frame → enhanced_frame
                                  ▼
              ┌───────────────────────────────────────┐
              │ PR-X12 base codec (lookup-table only) │
              │ - k-means basin codebook              │
              │ - Tropical-GEMM RDO                   │
              │ - DCT-II / EWA splat basis            │
              │ - Gaussian-tail rANS entropy          │
              │ - Zero NN forward passes              │
              │ - Deterministic across archs          │
              └───────────────────────────────────────┘
```

**Why this layering matters for PR-X12 scope:** the base layer is what's IN PR-X12. The enhancement layer is what `burn`/`candle` consumers may build *later*, taking PR-X12's decoded frames as input. The boundary is clean. The base layer never imports NN code; the enhancement layer takes pixels and produces pixels.

R-10's commitment to sub-1-bit-per-token + Gaussian-tail rANS is the *base layer's* extreme limit. If a use case needs lower bitrate than R-10 supports, layer NN on top — don't push NN into the base codec.

---

## 8. Falsifiers — what would invalidate this thesis

Be specific:

**F-1: Neural codecs close the latency gap.** If by 2028, neural codecs ship at < 100 µs per emit on commodity CPUs, the latency argument weakens. **Likelihood: low.** Forward-pass cost scales with model parameters; even ternary-quantized 1M-parameter models need ~3-5 µs per pass on AMX. The 50-1000× gap is structural, not implementation-dependent.

**F-2: Codebook adaptation breaks fixed lookup.** If real-world content distributions drift such that a 4096-entry codebook can't capture them, R-13's federated codebook update mechanism is required. **Mitigation:** R-13 is in scope. The codebook is swappable, not frozen-forever.

**F-3: PSNR gap exceeds 0.2 dB on real content.** If §5's ε estimate is wrong on real video clips, the Pareto argument weakens. **Mitigation:** Plan G video lane (R-4, R-11) is the empirical check. If PR-X12's PSNR vs x265 ultrafast is < 0.95× on Bbb 1080p, R-4 blocks the merge. The test is in CI.

**F-4: NN forward-pass becomes free on next-gen hardware.** If by 2030, all consumer hardware has 50 TFLOP/s of int8 throughput, NN inner-loop cost drops to lookup-table levels. **Mitigation:** even if NN cost drops, frozen lookup is still simpler and more deterministic. The Pareto argument doesn't reverse; only the slope changes.

**F-5: The basin codebook can't fit a streaming bitstream's symbol distribution online.** If R-10's sub-1-bit-per-token rANS path requires per-stream codebook training (slow), the codec stalls on stream init. **Mitigation:** federated codebook (R-13) ships pretrained codebooks for {video, audio, text, image} domains. New streams use the pretrained codebook; per-stream fine-tuning is optional and out-of-loop.

None of these falsifiers are decisive against PR-X12's thesis. They constrain its parameter choices, not its fundamental architecture.

---

## 9. What this lens prescribes for PR-X12 scope

Concrete implications:

1. **Do not** introduce any NN dependency in `ndarray-codec`. No `candle` or `burn` imports. No PyTorch FFI. Codec is dependency-free below `ndarray::hpc`.

2. **Do** ship the codebook as data, not as code. A 32-KB `[Fingerprint; 4096]` slice in the binary's `.rodata` section, not a `lazy_static` of a constructed object. Faster to load, simpler to swap (R-13).

3. **Do** keep tropical-GEMM in `lance-graph::blasgraph` and call it from the codec. Don't inline the algorithm into the codec; the kernel is a reusable substrate primitive (other consumers — `lance-graph`'s graph queries — already use it).

4. **Do** commit to the 0.2 dB PSNR Pareto-tradeoff publicly. Plan G's video bench (R-4, R-11) is the proof. If we miss it, we fall back to "compression-equivalent-to-x265-ultrafast-faster" instead of "compression-near-best-in-class."

5. **Reserve** a bitstream flag for the enhancement-layer hook (§7). One bit, in a reserved field of the 16-bit header. Decoder logs it; consumer crates may use it; codec doesn't.

6. **Document** the patent-free posture explicitly in `pr-x12-codec-x265-design.md`. Cite k-means (1957), tropical algebra (1990s), EWA splat (2001), DCT (1974), rANS (2014, patent-expired). Make the IP story unambiguous.

---

## 10. The deeper claim

**Neural codecs are not the future of codecs.** They are *one* future of codecs, narrowly applicable to generative ultra-low-bitrate use cases.

The other future — the much larger one — is **frozen-lookup codecs with NN-trained tables and an optional NN enhancement layer**. PR-X12 is a working prototype of this future. The substrate (R-1 basis trait, R-3 LoC envelope, R-11 latency assertions, R-13 federated codebook) makes it composable, deterministic, and patent-free.

The neural codec research community will arrive at this conclusion in 5-10 years, after burning through the latency and determinism walls. PR-X12 skips that detour.

---

## 11. Cross-references

- **Substrate canon:** `pr-x12-substrate-merged-canon.md`
- **Resolutions:** R-1, R-3, R-7, R-10, R-11, R-13 in `pr-x12-canon-resolutions-delta.md`
- **GEMM lens:** `pr-x12-x265-blasgraph-gemm.md` (companion analysis of the inner-loop math)
- **3DGS lens:** `pr-x12-x266-3dgs-spacetime-upscaling.md` (the EWA splat case study extended)
- **Multi-arch lens:** `pr-x12-woa-multiarch-orchestration.md` (why determinism matters fleet-wide)
- **Codec spec:** `pr-x12-codec-x265-design.md`
- **Reading list:**
  - Hutter (2005) "Universal AI"
  - Shannon (1948) source coding theorem
  - Hartigan (1975) k-means clustering
  - Zwicker et al. (2001) EWA Splatting
  - Duda (2014) Asymmetric Numeral Systems
  - Lyra (2021), SoundStream (2021), EnCodec (2022) papers for context

_Last edit: 2026-05-22._
_Status: opinionated perspective doc; the thesis is sharper than the rest of PR-X12 canon by design._
