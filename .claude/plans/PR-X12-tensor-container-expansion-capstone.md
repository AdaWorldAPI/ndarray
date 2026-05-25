# PR-X12 Tensor Container Expansion Capstone

## Purpose

Capture the capstone epiphany that the PR-X12 codec line is not only an x265 replacement and not only a future x266 / 3DGS scene codec.

It also expands naturally into a universal compressed tensor substrate for:

```text
GGUF
safetensors
Lance / Arrow tensor chunks
KV cache streams
gradient streams
3DGS scene anchors
model-weight distribution
```

This document connects the existing perspective docs:

```text
.claude/knowledge/pr-x12-x265-blasgraph-gemm.md
.claude/knowledge/pr-x12-x266-3dgs-spacetime-upscaling.md
.claude/knowledge/pr-x12-gguf-llm-weights-encoding.md
.claude/knowledge/pr-x12-anti-neural-lookup-inversion.md
```

## Capstone thesis

```text
PR-X12 is not a video codec.
PR-X12 is a hierarchical block grammar for structured tensors.
```

The video path is the first consumer:

```text
x265 / HEVC
  -> CTU blocks
  -> Skip / Merge / Delta / Escape
  -> BLAS/GEMM inner loops
  -> PR-X12
```

The 3DGS path is the second consumer:

```text
x266-style scene codec
  -> scene anchor
  -> EWA splat basis
  -> deterministic space-time rendering
  -> codec-native upscaling
```

The model-weight path is the third consumer:

```text
GGUF / safetensors
  -> tensor CTUs
  -> activation-aware RDO
  -> basin codebooks
  -> decode-during-GEMM
```

The datalake path is the fourth consumer:

```text
Lance / Arrow tensor chunks
  -> HHTL fragment traversal
  -> certified skip / refine / hydrate
  -> exact rows, vectors, weights, or graph edges only when needed
```

## General grammar

PR-X12 gives every structured payload this shape:

```text
hierarchical block
  -> mode: Skip / Merge / Delta / Escape
  -> basin or codebook pointer
  -> residual or bypass payload
  -> optional inter-tier reference
  -> entropy-coded tail
  -> certified decode / reconstruction path
```

The payload differs, but the grammar remains stable.

## Payload mapping

| Payload | Block | Basis / kernel | RDO distortion | Decode consumer |
|---|---|---|---|---|
| Video | CTU / CU | DCT / SSD / deblock | PSNR / rate-distortion | frame decoder |
| 3DGS scene | splat block / scene anchor | EWA splat basis | raster error / view error | scene renderer |
| GGUF weights | tensor CTU | GEMM / codebook lookup | activation-weighted error | inference matmul |
| safetensors | raw tensor block | GEMM / quant decode | task or tensor error | training/inference loader |
| KV cache | temporal tensor block | merge/delta stream | attention-logit error | transformer runtime |
| gradients | shard / bucket | delta / sketch / codebook | optimizer-step error | distributed training |
| Lance tensor chunks | fragment / row group | HHTL / field kernel | query confidence error | datalake executor |

## GGUF path

GGUF is the deployment-facing path.

Recommended compatibility strategy:

```text
GGUF v3/vNext
  -> add Q_PRX12 quantization type
  -> keep GGUF metadata / tokenizer / architecture fields
  -> store PR-X12 encoded tensor blocks as the tensor payload
```

Advantages:

```text
llama.cpp ecosystem can adopt incrementally
Ollama / LM Studio / Open WebUI can load through existing GGUF discovery
model distributors can ship one familiar container
PR-X12 becomes a quantization variant, not a format war
```

The codec layer should not become GGUF-specific. GGUF is an adapter.

## Safetensors path

Safetensors is the training/checkpoint-facing path.

Recommended strategies:

```text
1. sidecar mode
   model.safetensors
   model.prx12.index
   model.prx12.blocks

2. transcode mode
   model.safetensors
        ->
   model.prx12.safetensorpack

3. Lance mode
   safetensors shards
        ->
   Lance / Arrow tensor chunks
        ->
   PR-X12 HHTL block encoding
```

Safetensors should be treated as the clean source format. PR-X12 supplies compression, hierarchy, and decode-during-GEMM behavior.

## Lance / Arrow tensor chunk path

This is the most native path for the AdaWorld stack.

```text
checkpoint shard or GGUF tensor
        ->
Arrow tensor chunk metadata
        ->
Lance block store
        ->
PR-X12 encoded payload
        ->
HHTL traversal and decode scheduling
```

This enables:

```text
partial model loading
layer-family codebook sharing
tensor-block search
activation-aware block retention
streaming decode-during-GEMM
model-datalake queries
cross-model weight comparison
```

## Decode-during-GEMM

The most important runtime idea from the GGUF doc is preserved here:

```text
encoded tensor block
  -> decode a small cache-resident window
  -> immediately consume it in GEMM
  -> discard decoded window
```

This avoids full-tensor dequantization buffers.

Target behavior:

```text
storage smaller than GGUF Q4_K_M
perplexity near Q4_K_M / AWQ parity
scratch memory near cache-window size
GEMM overhead within a strict latency budget
```

## Anti-neural lookup inversion

The anti-neural doc supplies the runtime discipline:

```text
NNs may train tables.
NNs must not sit in the codec hot loop.
```

The runtime prefers:

```text
k-means basin codebook
frozen lookup table
Gaussian-tail rANS
DCT / EWA / BLAS basis
HHTL traversal
```

Optional neural enhancement belongs above the deterministic base layer, not inside it.

## Relationship to 3DGS plans

The 3DGS work is not separate from tensor compression.

The common shape is:

```text
3DGS splat block
  position / covariance / color / opacity
  -> EWA basis
  -> render reconstruction

LLM tensor block
  weight / scale / residual / codebook
  -> GEMM basis
  -> logit reconstruction
```

Both are basis-selected reconstruction from compressed blocks.

## Proposed follow-up docs

Potential future documents:

```text
PR-X12-GGUF-Q_PRX12-format-sketch.md
PR-X12-safetensors-sidecar-format-sketch.md
PR-X12-Lance-tensor-chunk-layout.md
PR-X12-decode-during-GEMM-benchmark-plan.md
PR-X12-activation-aware-RDO-for-weights.md
```

## Implementation trajectory

```text
Phase 0: preserve existing PR-X12 docs and add this capstone
Phase 1: prototype tensor block encoder over small f16 matrix
Phase 2: add activation-weighted RDO input
Phase 3: add decode-during-GEMM microbench
Phase 4: transcode one small safetensors tensor into PR-X12 block payload
Phase 5: wrap one GGUF tensor as Q_PRX12-like experimental payload
Phase 6: Lance/Arrow tensor chunk storage and HHTL traversal
```

## Acceptance criteria

- PR-X12 remains domain-neutral.
- GGUF and safetensors are adapters, not special cases in the codec core.
- Decode-during-GEMM remains a first-class benchmark target.
- The anti-neural lookup discipline remains intact.
- 3DGS scene anchors and LLM tensor blocks are documented as sibling consumers of the same grammar.

## Wall sentence

```text
PR-X12 starts as x265-through-BLAS, grows into x266-through-3DGS, and then becomes a universal compressed tensor substrate for model weights, scene fields, datalakes, and streaming inference.
```
