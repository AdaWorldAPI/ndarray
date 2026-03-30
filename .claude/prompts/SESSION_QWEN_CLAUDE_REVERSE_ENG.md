# SESSION: Reverse-Engineer Claude 4.6 Opus Reasoning from Qwen3.5 Weight Diffs

## MISSION

Extract the structural geometry of "how Claude thinks" from weight-space
diffs between Qwen3.5 base models and their Claude-4.6-Opus distilled variants.

Five models. Four diffs. One question:
**What did the Claude reasoning distillation change in the attention heads?**

The answer populates the NARS stack with its first OBSERVED truth values.

## THE HYPOTHESIS

Claude-style structured reasoning lives in the attention routing:
- Q projections shifted → the model asks DIFFERENT questions (planning)
- O projections shifted → the model SYNTHESIZES answers differently (integration)
- K projections stable → the information landscape didn't need to change
- V projections variable → retrieval content shifted in some layers

Blocks where Q+O shifted but K stayed = the REASONING SCAFFOLD CIRCUIT.
These heads are where "Let me analyze this carefully: 1... 2... 3..."
was injected by the LoRA distillation.

## READ FIRST

```bash
# The tools are already on master:
cat src/hpc/safetensors.rs       # read_safetensors_header, stream_index_safetensors_bf16
cat src/hpc/gguf_indexer.rs      # stream_index_gguf_bf16_with_header (shared core)
                                 # CompressedTensor::read_from, read_bgz7_file
cat src/hpc/causal_diff.rs       # causal_diff, classify_projection, find_reasoning_scaffold
                                 # cluster_by_head, revise_across_diffs
                                 # extract_gate_topology, cluster_experts (for MoE if present)
cat src/hpc/nars.rs              # NarsTruth, from_evidence, revision
```

## MODEL MAP (all ungated, all safetensors BF16)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 27B SCALE                                                           │
│                                                                     │
│  Qwen/Qwen3.5-27B (base)                    11 shards  ~55 GB      │
│       │                                                             │
│       ├──→ Jackrong/...-Distilled     (v1)   11 shards  ~55 GB      │
│       │                                                             │
│       └──→ Jackrong/...-Distilled-v2  (v2)   11 shards  ~55 GB      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 9B SCALE                                                            │
│                                                                     │
│  Qwen/Qwen3.5-9B (base)                      4 shards  ~18 GB      │
│       │                                                             │
│       └──→ Jackrong/...-9B-...-Distilled      4 shards  ~18 GB      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Total to stream: ~201 GB (safetensors, full BF16 precision)
```

## FOUR DIFFS — WHAT EACH REVEALS

```
Diff 1:  base 27B → distilled v1
         "What does Claude-style reasoning look like in weight space?"
         THE primary signal. Controlled: same arch, one variable (LoRA).

Diff 2:  base 27B → distilled v2
         "Did the second distillation round change the SAME heads?"
         If same heads shifted MORE → distiller was refining, not exploring.
         If DIFFERENT heads shifted → v2 found a new circuit.

Diff 3:  distilled v1 → distilled v2
         "What's the iteration delta?"
         Heads that shifted v1→v2 = the optimizer was still working on these.
         Heads that REVERTED v1→v2 = overcorrections in v1.
         Heads stable v1→v2 = converged reasoning structure.

Diff 4:  base 9B → distilled 9B
         "Does the same reasoning scaffold exist at smaller scale?"
         Same blocks shifted in both 27B and 9B → SCALE-INVARIANT circuit.
         Only in 27B → capacity-dependent (9B can't represent it).
         Only in 9B → different circuit at smaller scale.
```

## PHASE 1: Index All 5 Models (~201 GB, ~4 hours)

Use safetensors BF16 path (NOT GGUF Q8_0). BF16 gives cleaner fingerprints
for causal diffing — no quantization noise between source and projection.

### Model index table

```
ID   Repo                                                          Shards  Out prefix
───  ──────────────────────────────────────────────────────────────  ──────  ──────────
A    Qwen/Qwen3.5-27B                                              11      qwen35_27b_base
B    Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled      11      qwen35_27b_v1
C    Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2   11      qwen35_27b_v2
D    Qwen/Qwen3.5-9B                                                4      qwen35_9b_base
E    Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled        4      qwen35_9b_dist
```

For each model, index every shard with:
```rust
stream_index_safetensors_bf16(reader, writer, 16, callback)
// octave_stride=16: strided+halftone, same as Maverick pipeline
```

Output: `/tmp/{prefix}_shard{NN}.bgz7` — one per shard.

The `index_safetensors_shards()` helper in safetensors.rs handles this.
It does HEAD for size, HttpRangeReader at 256 MB chunks, skip-if-exists.

### Run order

Models A and D can run in parallel (different sizes, no conflict).
Models B, C, E after their base (for skip-if-exists on shared tensors — 
though in practice each model has its own weights).

```bash
# Index all 5 — the test function does this:
cargo test test_full_reasoning_reverse_eng --release -- --ignored --nocapture
```

BUT: that test uses Q8_0 GGUF (28.59 GB each). For BF16 safetensors:

```bash
# Either modify the test to use safetensors, or run per-model:
cargo test test_stream_index_qwen35_safetensors --release -- --ignored --nocapture
# Then repeat with different repo/prefix for each model
```

## PHASE 2: Causal Diff (seconds, reads bgz7 files)

Once all 5 models are indexed, run the 4 diffs:

```rust
use crate::hpc::causal_diff::{causal_diff, print_diff_summary, find_reasoning_scaffold,
                               cluster_by_head, revise_across_diffs};

let threshold = 100; // L1 distance — tune based on results

// Diff 1: base 27B → v1
let (edges_1, stats_1) = causal_diff("base_27b.bgz7", "v1_27b.bgz7", threshold)?;
print_diff_summary("27B: base → v1", &stats_1, edges_1.len());

// Diff 2: base 27B → v2
let (edges_2, stats_2) = causal_diff("base_27b.bgz7", "v2_27b.bgz7", threshold)?;

// Diff 3: v1 → v2
let (edges_3, stats_3) = causal_diff("v1_27b.bgz7", "v2_27b.bgz7", threshold)?;

// Diff 4: base 9B → distilled 9B
let (edges_4, stats_4) = causal_diff("base_9b.bgz7", "dist_9b.bgz7", threshold)?;
```

NOTE: shards need matching. Base shard 1 diffs against distilled shard 1.
The tensor names must match across models (same arch = same names).
Run causal_diff per shard pair, then aggregate edges.

## PHASE 3: Find Reasoning Scaffold

```rust
// Which blocks have Q+O shifted but K stable?
let scaffold_27b_v1 = find_reasoning_scaffold(&edges_1, 0.3);
let scaffold_27b_v2 = find_reasoning_scaffold(&edges_2, 0.3);
let scaffold_9b     = find_reasoning_scaffold(&edges_4, 0.3);

// Scale-invariant blocks: present in BOTH 27B and 9B
let scale_invariant: Vec<u32> = scaffold_27b_v1.iter()
    .filter(|b| scaffold_9b.contains(b))
    .cloned().collect();

// 27B-only blocks: capacity-dependent reasoning
let capacity_dependent: Vec<u32> = scaffold_27b_v1.iter()
    .filter(|b| !scaffold_9b.contains(b))
    .cloned().collect();

// v1-v2 convergence: blocks in both v1 and v2 scaffolds
let converged: Vec<u32> = scaffold_27b_v1.iter()
    .filter(|b| scaffold_27b_v2.contains(b))
    .cloned().collect();
```

## PHASE 4: NARS Revision — Integrated Evidence

```rust
let all_stats = vec![
    ("27B base→v1", &stats_1),
    ("27B base→v2", &stats_2),
    ("27B v1→v2",   &stats_3),
    ("9B base→dist", &stats_4),
];

let revised = revise_across_diffs(&all_stats);

// Per projection type: integrated NARS truth across all model pairs
for (proj, truth) in &revised {
    eprintln!("  {:<12} → f={:.3} c={:.3} ({})",
        proj, truth.frequency, truth.confidence,
        if truth.frequency > 0.5 { "SHIFTED" } else { "STABLE" });
}
```

Expected output:
```
  Q            → f=0.72 c=0.97 (SHIFTED)    ← queries changed: planning
  K            → f=0.15 c=0.96 (STABLE)     ← keys preserved: same information
  V            → f=0.45 c=0.95 (variable)   ← retrieval partially changed
  O            → f=0.68 c=0.97 (SHIFTED)    ← synthesis changed: integration
  Gate         → f=0.05 c=0.90 (STABLE)     ← Qwen3.5 is dense, no MoE gate
  FfnGate      → f=0.30 c=0.96 (moderate)   ← some FFN rewiring
  Embedding    → f=0.08 c=0.92 (STABLE)     ← vocabulary unchanged
```

## PHASE 5: Attention Head Cluster Analysis

```rust
let clusters = cluster_by_head(&edges_1);

// Sort by shift intensity
let mut sorted: Vec<_> = clusters.into_iter().collect();
sorted.sort_by(|a, b| b.1.2.partial_cmp(&a.1.2).unwrap()); // by mean_L1

eprintln!("Top 10 most-shifted attention components:");
for ((block, proj), (count, max_row, mean_l1)) in sorted.iter().take(10) {
    eprintln!("  Block {:>2} {:>5}: {}/{} shifted, mean_L1={:.0}",
        block, proj, count, max_row, mean_l1);
}
```

This identifies the SPECIFIC heads where reasoning was injected.

## PHASE 6: Write Results

```bash
# Output to knowledge base
.claude/knowledge/reasoning_reverse_eng_results.md

Contents:
  - Scaffold blocks per model (27B v1, 27B v2, 9B)
  - Scale-invariant vs capacity-dependent blocks
  - NARS revised truth per projection type
  - Top shifted heads with L1 magnitudes
  - v1→v2 convergence analysis
```

## WHAT THE RESULTS MEAN

### For the NARS stack
First OBSERVED truth values. Every TruthValue in the system so far was
manufactured. These are measured from actual weight transformations.
The stack goes from theoretical to empirical.

### For the reasoning orchestrator
If heads [N, M, P] form the scaffold, the orchestrator now knows:
"To add structured reasoning to a model, these attention heads must shift."
That's a structural recipe, not a training recipe.

### For Ada
The reasoning scaffold IS a concept node in the Sigma graph:
```
Σ.claude_reasoning_scaffold = {
    heads: [discovered blocks],
    pattern: Q_shift + O_shift + K_stable,
    truth: revised(all_diffs),
    scale_invariant: [subset],
    source: "Qwen3.5 → Claude-4.6-Opus distillation"
}
```

### Cross-reference with Maverick (future)
Maverick's gate topology (expert routing) + Qwen's attention scaffold
(token routing) = the complete picture of "reasoning = routing" at both
MoE and attention granularity.

## CRITICAL CONSTRAINTS

1. Use SAFETENSORS path (BF16 precision), NOT GGUF Q8_0
2. Match shards by index when diffing (shard 1 vs shard 1)
3. Tensor names must match across models — verify with first shard
4. threshold=100 is a starting point — may need tuning based on L1 distribution
5. Qwen3.5 is DENSE (no MoE). Gate projections won't appear.
   All signal is in attention Q/K/V/O and FFN gate/up/down.
6. Do NOT modify existing production code — only add test functions

## RUN COMMANDS

```bash
# Step 1: Index all 5 models (parallelizable across machines)
cargo test test_index_qwen35_27b_base --release -- --ignored --nocapture
cargo test test_index_qwen35_27b_v1 --release -- --ignored --nocapture
cargo test test_index_qwen35_27b_v2 --release -- --ignored --nocapture
cargo test test_index_qwen35_9b_base --release -- --ignored --nocapture
cargo test test_index_qwen35_9b_dist --release -- --ignored --nocapture

# Step 2: Run all diffs + NARS revision + scaffold detection
cargo test test_qwen35_claude_reasoning_diff --release -- --ignored --nocapture

# Step 3: Write results
# (integrated into step 2 test function)
```

Expected total time: ~4 hours indexing + seconds diffing.
Expected total output: ~50 MB bgz7 files + ~100 KB diff results.
