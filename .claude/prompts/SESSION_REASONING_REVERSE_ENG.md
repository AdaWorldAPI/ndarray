# SESSION: Reverse-Engineer Reasoning via Causal Edge Diffing

## MISSION

Extract the structural geometry of "how to think" from:
1. Llama 4 Maverick MoE gate projections (routing topology)
2. Qwen3.5 base→distilled attention diffs (reasoning circuit)
3. Cross-model comparison (scale-invariant reasoning atoms)

Feed into NARS truth values on causal edges. First real training data
for the NARS stack.

## READ FIRST

```bash
cat src/hpc/gguf_indexer.rs    # stream_index_gguf_bf16, classify_tensor
cat src/hpc/nars.rs            # TruthValue, revision, evidence
cat src/hpc/bgz17_bridge.rs    # Base17 type, L1 distance
cat src/hpc/causality.rs       # CausalEdge if it exists
```

## PHASE 1: Index All Models (Q8_0, streaming)

Five GGUF files, all single-shard, ~105 GB total:

```
unsloth/Qwen3.5-27B-GGUF
  → Qwen3.5-27B-Q8_0.gguf                  28.59 GB  (base)

Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
  → Qwen3.5-27B.Q8_0.gguf                  28.59 GB  (distilled v1)

Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF
  → Qwen3.5-27B.Q8_0.gguf                  28.59 GB  (distilled v2)

unsloth/Qwen3.5-9B-GGUF
  → Qwen3.5-9B-Q8_0.gguf                    9.52 GB  (base 9B)

Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
  → Qwen3.5-9B.Q8_0.gguf                    9.52 GB  (distilled 9B)
```

Use `stream_index_gguf` (f32 path — Q8_0 needs actual dequantization).
Output: 5 bgz7 files with per-tensor, per-row Base17 projections.

## PHASE 2: Attention Diff (the reasoning circuit)

For each tensor pair (base vs distilled), matched by name:

```rust
// Pseudocode — actual implementation in causal_diff.rs
for (name, base_rows, dist_rows) in matched_tensors(base_bgz7, dist_bgz7) {
    let layer_type = classify_tensor(name);
    
    for (row_idx, (b, d)) in base_rows.zip(dist_rows).enumerate() {
        let distance = b.l1(&d);
        
        if distance > threshold {
            let edge = CausalEdge64 {
                subject: palette_index(b),     // base archetype
                verb: BECOMES,                  // structural transformation
                object: palette_index(d),       // distilled archetype
                truth: TruthValue {
                    frequency: distance as f32 / max_l1 as f32,
                    confidence: 1.0 / (1.0 + row_count as f32), // NARS evidence
                },
            };
            
            // Tag with attention-specific metadata
            match classify_projection(name) {
                Q => emit_q_edge(edge, layer, head),
                K => emit_k_edge(edge, layer, head),
                V => emit_v_edge(edge, layer, head),
                O => emit_o_edge(edge, layer, head),
                Gate => emit_gate_edge(edge, layer),
                _ => emit_generic(edge),
            }
        }
    }
}
```

### What Each Projection Shift Means

```
Q shifted, K stable  → model asks NEW questions of SAME information
                       = learned to LOOK for reasoning structure
                       NARS: high frequency, high confidence

K shifted            → model EXPOSES different features to attention
                       = deeper change, new token-level signals
                       NARS: moderate frequency, lower confidence (rarer)

V shifted            → WHAT gets retrieved changed
                       = content-level reasoning substrate
                       NARS: varies by layer depth

O shifted            → HOW multi-head outputs COMBINE
                       = synthesis/integration change
                       NARS: if high → distillation core is integration

Q+O shift, K stable  → REASONING SCAFFOLD CIRCUIT
                       = the minimal structural change for reasoning
                       These heads ARE the distillation's value
```

### Attention Head Clustering

```
Cluster 1: Q+O shift, K stable    → "reasoning scaffold" heads
Cluster 2: K+V shift               → "representation change" heads
Cluster 3: all stable              → "unchanged capability" heads
Cluster 4: Q shift only            → "query refinement" heads

Each cluster → one Sigma concept node
Cross-model same cluster → SUPPORTS edge (scale-invariant)
Cross-model different cluster → CONTRADICTS edge (scale-dependent)
```

## PHASE 3: MoE Gate Topology (from Maverick bgz7)

The Maverick bgz7 already has gate projections indexed.
Extract the gate tensor Base17 patterns separately:

```
blk.{N}.ffn_gate_inp  → router gate [n_experts, hidden_dim]
                         Each ROW = one expert's activation pattern
                         Base17 of that row = expert's structural identity
```

Expert identity in Base17 space:
- Experts with similar Base17 → structurally redundant (SUPPORTS)
- Experts with distant Base17 → specialized (distinct concept nodes)
- Cluster the 128 expert fingerprints → find natural expert groups

Cross with attention: which attention heads' Q projections align
with which expert gate patterns? That alignment = the routing circuit.

```
head_17_Q_pattern ──CAUSES──→ expert_37_gate_pattern
  (this head's queries activate this expert)
  truth: cosine(head_Q_base17, expert_gate_base17)
```

## PHASE 4: NARS Truth Population

Every edge from phases 2-3 carries a TruthValue:

```rust
TruthValue {
    frequency: f32,    // how often this transformation occurs
                       // = proportion of rows in this tensor that shifted
    confidence: f32,   // evidence strength
                       // = 1 - 1/(1+k) where k = number of observed rows
}
```

NARS revision across models:
```
evidence_27b_v1:  (f=0.7, c=0.92)   // 70% of Q rows shifted in 27B v1
evidence_27b_v2:  (f=0.8, c=0.92)   // 80% shifted in v2 (more distillation)
evidence_9b:      (f=0.5, c=0.88)   // only 50% shifted in 9B (capacity limit)

revised = nars_revision(evidence_27b_v1, evidence_9b)
  → (f=0.62, c=0.95)  // integrated belief about reasoning scaffold
```

The revised truth tells you: "reasoning scaffold changes affect ~62% of
Q projection rows, with 95% confidence, scale-dependent (27B > 9B)."

## PHASE 5: Sigma Concept Nodes (Ada Integration)

Each cluster of edges becomes a concept in Ada's graph:

```
Σ.reasoning_scaffold = {
    evidence: [27b_v1_edges, 27b_v2_edges, 9b_edges],
    truth: revised_truth,
    composition: {Q_shift: 0.73, O_shift: 0.82, K_stable: 0.95},
    heads: [17, 23, 24, 31],  // discovered by clustering
    scale_invariant: false,     // 9B diverges
    source: "Qwen3.5 → Claude-Opus distillation"
}

Σ.expert_redundancy = {
    evidence: [maverick_gate_similarities],
    truth: (f=0.96, c=0.99),  // 96% structurally interchangeable
    meaning: "MoE expert weights are commodity, routing is intelligence"
}

Σ.reasoning_scaffold ──CAUSES──→ Σ.expert_redundancy
  // reasoning heads SHAPE what the router sees
  // truth: to be discovered by cross-model alignment
```

## DELIVERABLES

1. `causal_diff.rs` — load two bgz7 files, emit CausalEdge64 per shifted row
2. `attention_cluster.rs` — cluster edges by projection type per head
3. Test: `test_qwen35_reasoning_diff` — run the full 5-model pipeline
4. Test: `test_maverick_gate_topology` — extract gate patterns from existing bgz7
5. Output: `.claude/knowledge/reasoning_reverse_eng_results.md`

## WHY THIS MATTERS

The NARS stack has:
- TruthValue with frequency + confidence ✓
- Revision (evidence integration) ✓  
- Inference rules ✓
- Graph storage ✓

What it's MISSING: real evidence. Every truth value is currently 
manufactured. This pipeline generates the first OBSERVED truth values
from actual model weight differences. The NARS stack goes from 
theoretical to empirical in one session.

The thinking orchestration atoms (mcp-orchestrator-vsa) can then
CONSTRUCT reasoning patterns from the observed evidence:
"To add structured reasoning to a model, shift Q+O projections
in heads [17,23,24,31] by palette distance 3-7. Expected improvement:
f=0.62±0.15 at c=0.95."

That's not prompt engineering. That's weight-space surgery 
informed by causal evidence. Programming AGI by observation.
