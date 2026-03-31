# SESSION: Structural Restoration — Q8_0 + Palette3D Attention Overlay

## MISSION

Prove that a 196KB Palette3D extracted from BF16 weight diffs can restore
Q8_0 quantization loss in the attention heads that carry meaning.

This is not fine-tuning. This is not LoRA. This is **structural restoration**:
the Palette tells the quantized model WHERE and WHY to attend, compensating
for the routing precision that Q8_0 uniformly destroyed.

## THE INSIGHT

Q8_0 quantizes uniformly — every weight gets 8 bits regardless of importance.
The 4 causal diffs across 5 Qwen models reveal which weights are **volatile**
(changed across model versions = carry meaning) vs **ballast** (stable = structural).

```
BF16 → Q8_0 loss map:
  Stable weights (ballast):    Q8_0 sufficient, no precision loss
  Volatile weights (attention): Q8_0 destroys fine differences
                                 exactly where models diverge

4 diffs × 5 models = cross-validated volatility map
  High NARS freq across all diffs → architecture, not noise
  Low NARS freq across all diffs  → ballast
```

The Palette3D from `scaffold_to_palette3d_layers()` encodes this volatility
topology in 196KB (8 layers × 64 rows × 64 bits × 2 bytes overhead).

## PREREQUISITE

The bgz7 indexes from SESSION_QWEN_CLAUDE_REVERSE_ENG must exist:

```
/tmp/qwen35_27b_base_shard{01..11}.bgz7
/tmp/qwen35_27b_v1_shard{01..11}.bgz7
/tmp/qwen35_27b_v2_shard{01..11}.bgz7
/tmp/qwen35_9b_base_shard{01..04}.bgz7
/tmp/qwen35_9b_dist_shard{01..04}.bgz7
```

If not, run the safetensors indexing tests first:
```bash
cargo test test_index_qwen35_27b_base --release -- --ignored --nocapture
# ... (all 5 models)
```

## PHASE 1: Extract Palette3D from Diffs

```rust
// Run all 4 diffs (already wired in test_qwen35_claude_reasoning_diff)
let (edges_v1, stats_v1) = causal_diff_sharded("qwen35_27b_base", "qwen35_27b_v1", 11, 100);
let (edges_v2, stats_v2) = causal_diff_sharded("qwen35_27b_base", "qwen35_27b_v2", 11, 100);
let (edges_v1v2, _)      = causal_diff_sharded("qwen35_27b_v1", "qwen35_27b_v2", 11, 100);
let (edges_9b, _)        = causal_diff_sharded("qwen35_9b_base", "qwen35_9b_dist", 4, 100);

// Build the 8-layer reasoning circuit
let layers = scaffold_to_palette3d_layers(&edges_v1, &edges_v2, &edges_v1v2, &edges_9b);

// Construct Palette3D (from p64 crate)
use p64::{Palette64, Palette3D, ThinkingStyle};
let palette_layers: [Palette64; 8] = layers.map(|rows| Palette64 { rows });
let palette3d = Palette3D::new(palette_layers, ThinkingStyle::ANALYTICAL);

// Serialize: 8 × 64 × 8 bytes = 4096 bytes core + metadata
serialize_palette3d(&palette3d, "/tmp/qwen35_claude_palette3d.bin");
```

## PHASE 2: Volatility Scoring

For each weight tensor, compute a **volatility score** from all 4 diffs:

```rust
// NARS revision across all 4 diffs gives integrated truth per projection
let revised = revise_across_diffs(&[
    ("27B base→v1", &stats_v1),
    ("27B base→v2", &stats_v2),
    ("27B v1→v2",   &stats_v1v2),
    ("9B base→dist", &stats_9b),
]);

// Per-head volatility: cluster_by_head gives (block, proj) → (count, total, mean_L1)
// Heads with high count/total ratio across ALL 4 diffs = architectural attention
// Heads with low ratio = ballast
```

The volatility map IS the Palette3D bit pattern:
- bit=1 → volatile → Q8_0 damaged this → Palette compensates
- bit=0 → stable → Q8_0 is fine → no overlay needed

## PHASE 3: Inference with Palette Overlay

The overlay works at attention routing time, not at weight level:

```text
Standard Q8_0 inference:
  Q_q8 × K_q8^T → attention_scores → softmax → V_q8

With Palette overlay:
  Q_q8 × K_q8^T → attention_scores
  scores[i][j] *= palette3d.infer(block).attention[j] ? 1.0 : decay_factor
  → softmax → V_q8

Where:
  block = current transformer block (0..63)
  j = target token position (mapped to 64-bin palette coordinate)
  decay_factor = 0.8 (palette says "this connection is ballast, dampen it")
                 vs 1.0 (palette says "this connection matters, keep full strength")
```

This is O(1) per attention head per token: one Palette3D::infer() call (POPCNT).

## PHASE 4: Measure Divergence

```text
Inputs: 100 diverse prompts (reasoning, code, creative, factual)

For each prompt, generate 256 tokens with:
  A: BF16 full model (ground truth)
  B: Q8_0 alone
  C: Q8_0 + Palette3D overlay

Metrics:
  1. Token agreement: % of tokens matching BF16 output
  2. Logit KL divergence: KL(BF16 || Q8_0) vs KL(BF16 || Q8_0+Palette)
  3. Perplexity gap: PPL(BF16) vs PPL(Q8_0) vs PPL(Q8_0+Palette)
  4. Attention pattern cosine similarity: cos(attn_BF16, attn_Q8_0) vs cos(attn_BF16, attn_Q8_0+P)

Expected result:
  KL(BF16 || Q8_0+Palette) < KL(BF16 || Q8_0)
  "196KB of structural information restores precision lost by uniform quantization"
```

## PHASE 5: Scale Test (9B)

Repeat Phase 3-4 with Qwen3.5-9B:
- 9B palette from `heels_9b` (4 shards, smaller)
- If the palette works at BOTH 27B and 9B → scale-invariant structural restoration
- The BECOMES layer (predicate 7) captures exactly this

## WHAT THIS PROVES

If successful, this demonstrates:

1. **Selective precision recovery**: 196KB beats uniform 8-bit by knowing WHERE precision matters
2. **Cross-model structural discovery**: 5 models → volatility map → architecture, not training
3. **Scale invariance**: same palette works at 9B and 27B
4. **NARS-grounded evidence**: every bit in the palette has a measured truth value (f, c)
5. **O(1) inference cost**: POPCNT per head, not matrix multiply

The Palette3D is not a model. It's a **structural prior** — it tells any quantized model
"these are the attention connections that matter, preserved at 1-bit granularity."

## RUN COMMAND

```bash
# After all 5 models are indexed:
cargo test test_palette3d_structural_restoration --release -- --ignored --nocapture
```

## CRITICAL CONSTRAINTS

1. BF16 safetensors for indexing (not GGUF Q8_0 — that's the degraded version)
2. The Palette3D is extracted from DIFFS, not from a single model
3. 4 diffs minimum for cross-validation (single diff = noise)
4. Overlay is multiplicative on attention scores, not additive on weights
5. Do NOT modify existing production code — only add test functions
