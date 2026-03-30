# SESSION: HiDream-I1 DiT+MoE — First Diffusion Model Indexing

## MISSION

Index HiDream-I1-Full (17B DiT+MoE, MIT license) through the bgz17 pipeline.
First cross-domain validation: do image generation MoE experts show the same
structural redundancy as LLM MoE experts (Maverick's 123,000×)?

Also diff HiDream's Llama-3.1-8B text encoder against base Llama-3.1-8B
to see what "learning to see" does to a language model's attention patterns.

## READ FIRST

```bash
cat src/hpc/safetensors.rs     # read_safetensors_header, stream_index_safetensors_bf16
cat src/hpc/gguf_indexer.rs    # stream_index_gguf_bf16_with_header (shared core)
cat src/hpc/causal_diff.rs     # causal_diff, find_reasoning_scaffold
```

## MODEL MAP

```
HiDream-ai/HiDream-I1-Full (MIT, ungated)

Transformer (DiT + MoE):
  transformer/diffusion_pytorch_model-{00001..00007}-of-00007.safetensors
  Shard 1:  4.99 GB
  Shard 2:  4.98 GB
  Shard 3:  4.99 GB
  Shard 4:  4.98 GB
  Shard 5:  4.99 GB
  Shard 6:  4.99 GB
  Shard 7:  4.29 GB
  Total:   35.21 GB

Text Encoders:
  text_encoder/model.safetensors          0.49 GB  (CLIP-L)
  text_encoder_2/model.safetensors        2.77 GB  (CLIP-G/OpenCLIP ViT-bigG)
  text_encoder_3/model-00001-of-00002     4.99 GB  (Llama-3.1-8B shard 1)
  text_encoder_3/model-00002-of-00002     4.53 GB  (Llama-3.1-8B shard 2)
  Total:                                 12.78 GB

VAE:
  vae/diffusion_pytorch_model.safetensors 0.16 GB

Grand total: ~48.15 GB
```

## PHASE 1: Index Transformer (35 GB, ~1 hour)

The DiT+MoE transformer is the main target. Architecture:
- DiT blocks with self-attention (Q/K/V/O projections)
- MoE expert layers (gate + expert FFN)
- Cross-attention (text conditioning)
- Time-step embeddings

```bash
cargo test test_stream_index_hidream_transformer \
    --release -- --ignored --nocapture 2>&1 | tee /tmp/hidream_transformer.log
```

Expected compression:
- MoE expert weights: 50,000-100,000× (if similar to Maverick)
- Attention Q/K/V/O: 500-2,000×
- Cross-attention: unknown — this is NEW (text→image conditioning)
- Time embedding MLP: unknown — sinusoidal structure may compress differently

## PHASE 2: Index Text Encoders (13 GB, ~30 min)

Index all three text encoders. The Llama-3.1-8B encoder is especially
interesting — it's a known architecture fine-tuned for image conditioning.

```bash
cargo test test_stream_index_hidream_text_encoders \
    --release -- --ignored --nocapture
```

## PHASE 3: Diff Llama-3.1-8B (what "seeing" adds to "reading")

Compare HiDream's Llama-3.1-8B (text_encoder_3) against base Llama-3.1-8B
(unsloth/Llama-3.1-8B, ungated safetensors).

```bash
# Index base Llama-3.1-8B
cargo test test_stream_index_llama31_8b_base \
    --release -- --ignored --nocapture

# Diff
cargo test test_hidream_llama_diff \
    --release -- --ignored --nocapture
```

This diff tells us: which attention heads re-routed when a language model
learned to condition image generation? The Q/K/V/O shift pattern reveals
what "visual grounding" looks like in weight space.

Cross-reference with the Qwen3.5 reasoning scaffold:
- Qwen3.5 diff: what "structured reasoning" looks like (Claude distillation)
- HiDream diff: what "visual grounding" looks like (image conditioning)
- Same NARS pipeline, different capability injection
- Do they share attention heads? If yes → multimodal reasoning is routing

## PHASE 4: Cross-Domain MoE Comparison

Compare HiDream's MoE expert compression against Maverick's:

```
Maverick (LLM):     128 experts, 123,000× on gate/up_exps
HiDream (diffusion): N experts, ???× on expert layers

If similar ratios → MoE structural redundancy is architecture-level,
                     not domain-level. Experts are commodity everywhere.
If different      → image generation experts specialize more than
                     language experts (domain shapes expert identity).
```

## EXPECTED RESULTS

```
HiDream DiT+MoE transformer (35 GB):
  Conservative: 5-10 MB (3,500-7,000×)
  If MoE-heavy:  1-3 MB (12,000-35,000×)
  
CLIP-L (0.49 GB):         ~100 KB (5,000×)
CLIP-G (2.77 GB):         ~500 KB (5,500×)
Llama-3.1-8B (9.52 GB):   ~2 MB (5,000×)

Total ~48 GB:  →  ~3-13 MB
```

## CRITICAL NOTES

1. Use safetensors path: stream_index_safetensors_bf16 (BF16 precision)
2. Tensor names will differ from GGUF conventions — classify_tensor and
   classify_projection may need HiDream-specific patterns
3. Check tensor names in shard 1 header first: the naming convention
   determines whether classify_tensor catches attention/FFN/MoE correctly
4. If MoE expert tensors are named differently than llama.cpp convention,
   add patterns to classify_tensor BEFORE running (or they'll be classified
   as generic Attention and compress at lower ratios)

## DELIVERABLES

1. bgz7 indexes: /tmp/hidream_transformer_shard{01-07}.bgz7
2. bgz7 indexes: /tmp/hidream_clip_l.bgz7, hidream_clip_g.bgz7
3. bgz7 indexes: /tmp/hidream_llama_enc.bgz7 (combined shards)
4. bgz7 indexes: /tmp/llama31_8b_base.bgz7
5. Diff results: .claude/knowledge/hidream_results.md
6. Cross-domain MoE comparison: .claude/knowledge/moe_cross_domain.md
