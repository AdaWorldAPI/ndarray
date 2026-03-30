# SESSION: Wire the Synapses — 5 Bridges Between Existing Modules

## MISSION

Everything is built. Nothing is wired. This session connects the existing
modules WITHOUT creating new abstractions. Pure function calls between
code that already compiles and passes tests.

**P0 RULE: READ → UNDERSTAND → WIRE. Do NOT rewrite, refactor, or "improve"
any module. Add bridge functions, add tests, add comments. Nothing else.**

## CONTEXT: What Another Session Discovered

A review of all ndarray hpc/ modules found 7 "unwired synapses" — places
where Module A produces exactly what Module B needs, but the call is missing.
This session wires the 5 that are pure plumbing (no research needed).

Reference prompts that describe the target architecture:
- `.claude/prompts/session_deepnsm_cam.md` — NSM tokenizer → SPO → ThinkingGraph
- `.claude/prompts/session_unified_vector_search.md` — RaBitQ × Cascade × bgz17
- `.claude/prompts/01_clam_qualiacam.md` — CLAM tree → Cascade candidate set
- `.claude/prompts/04_lance_graph_integration.md` — DataFusion UDFs
- `.claude/prompts/05_cross_repo_map.md` — Module dependency map

READ those prompts. Then read the CODE they reference. Then come back here.

## READ FIRST (the code, not the prompts)

```bash
# The 5 modules to wire:
cat src/hpc/jina/runtime.rs      # ModelRuntime, pack_spo_edge(), cam_fingerprint()
cat src/hpc/jina/causal.rs       # CausalEdge64, pack_edge(), causal_distance()
cat src/hpc/gpt2/inference.rs    # Gpt2Engine, causal_edges: Vec<AttentionEdge>
cat src/hpc/openchat/inference.rs # OpenChatEngine, causal_edges
cat src/hpc/models/router.rs     # ModelRouter, ModelBackend enum, dispatch
cat src/hpc/crystal_encoder.rs   # external embedding → Node SPO planes
cat src/hpc/substrate.rs         # SubstrateRoute, thinking style references
cat src/hpc/graph.rs             # VerbCodebook, find_non_causal_edges
cat src/hpc/vml.rs               # HEEL/HIP archetypes, image compression (tests)
cat src/hpc/cascade.rs           # Cascade 3-stroke search
cat src/hpc/clam.rs              # ClamTree, rho_nn()

# The palette/compose infrastructure:
cat src/hpc/palette_codec.rs     # Palette encoding
cat src/hpc/palette_distance.rs  # 256×256 distance table
```

## BRIDGE 1: Jina Palette → VerbCodebook Graph Ops
**~30 LOC addition to `graph.rs`**

### What exists:
- `jina/runtime.rs`: `ModelRuntime::pack_spo_edge(s, p, o) → CausalEdge64`
- `jina/runtime.rs`: `ModelRuntime::cam_fingerprint(token) → [u8; 6]`  
- `graph.rs`: `VerbCodebook` maps verb names → binary hypervectors
- `jina/causal.rs`: `causal_distance(a, b, mask) → PearlDecomposition`

### What's missing:
VerbCodebook uses string-based verb lookup. It should ALSO accept palette indices
for O(1) verb distance via the 256×256 table.

### What to add:

```bash
# In graph.rs, ADD (don't replace existing methods):
```

```rust
// ADD to impl VerbCodebook — palette-accelerated verb lookup
// 
// CONTEXT: The Jina codec compresses any token to a palette index (1 byte).
// Instead of looking up verbs by string → hash → binary vector → Hamming,
// this path does: palette_index → 256×256 distance table → u16 distance.
// Same result, O(1) instead of O(verb_name.len() + popcount).
//
// See: .claude/prompts/session_unified_vector_search.md (compose tables)
// See: src/hpc/jina/runtime.rs (cam_fingerprint, pack_spo_edge)

/// Look up the nearest verb by palette index.
/// Returns (verb_name, distance) using the precomputed distance matrix.
/// 
/// This is the palette-accelerated equivalent of `nearest_verb()` — 
/// instead of computing Hamming distance to all verb vectors, 
/// it does a single row lookup in the 256×256 table.
pub fn nearest_verb_by_palette(
    &self, 
    palette_idx: u8, 
    distance_table: &[[u16; 256]; 256],
) -> Option<(&str, u16)> {
    // TODO: implement — iterate self.verbs, find min distance_table[palette_idx][verb_palette_idx]
    // REQUIRES: each verb in the codebook needs a cached palette_idx
    // ADD a field: verb_palette_indices: Vec<u8> parallel to self.verbs
    todo!("Bridge 1: palette-accelerated verb lookup")
}
```

### Inline comments to ADD to existing code:

```bash
# In jina/runtime.rs, find pack_spo_edge() and ADD above it:
```
```rust
// BRIDGE NOTE: This function produces CausalEdge64 that can be stored
// parallel to any graph's triplet list. See graph.rs VerbCodebook for
// the verb-lookup side. The palette index from cam_fingerprint() feeds
// into VerbCodebook::nearest_verb_by_palette() for O(1) verb matching.
// Wiring status: pack_spo_edge EXISTS, VerbCodebook palette path PENDING.
```

---

## BRIDGE 2: GPT-2/OpenChat Attention Edges → Thinking Metrics
**~50 LOC addition to `substrate.rs`**

### What exists:
- `gpt2/inference.rs`: `pub causal_edges: Vec<AttentionEdge>` — emitted during inference
- `openchat/inference.rs`: same pattern, `pub causal_edges: Vec<AttentionEdge>`
- `substrate.rs`: `SubstrateRoute` with cognitive state, references ThinkingStyle

### What's missing:
Nobody reads causal_edges after inference. They're computed, stored, ignored.
Three metrics can be derived that map directly to thinking styles.

### What to add:

```bash
# In substrate.rs, ADD a new function (don't modify SubstrateRoute):
```

```rust
/// Derive thinking-style signals from transformer attention edges.
///
/// Maps attention patterns to cognitive indicators:
/// - `attention_entropy`: Shannon entropy of attention weights (focused vs diffuse)
/// - `layer_depth_used`: which layers fired strongly (early=pattern, late=reasoning)
/// - `cross_attention_density`: cross-document edges (novel associations)
///
/// These feed into thinking style classification:
/// - Analytical: high entropy in late layers (deliberate exploration)
/// - Intuitive: low entropy in early layers (System 1 pattern match)
/// - Creative: high cross-attention (lateral connections)
///
/// CONTEXT: See epiphany #2 from session review.
/// SOURCE: gpt2/inference.rs (AttentionEdge), openchat/inference.rs (same)
/// TARGET: ThinkingStyle selection in crewai-rust (upstream)
///
/// BRIDGE NOTE: This function is PURE — takes edges in, returns metrics out.
/// No mutation, no side effects. The upstream ThinkingStyle mapper calls this.
pub struct AttentionMetrics {
    /// Shannon entropy of attention distribution. High = exploring, Low = focused.
    pub attention_entropy: f32,
    /// Average layer depth weighted by attention strength. High = deep reasoning.
    pub mean_layer_depth: f32,
    /// Fraction of attention edges crossing document boundaries. High = creative.
    pub cross_attention_density: f32,
    /// Total number of edges analyzed.
    pub edge_count: usize,
}

/// Compute attention metrics from a batch of causal edges.
///
/// `edges`: the `causal_edges` vector from Gpt2Engine or OpenChatEngine
/// `num_layers`: total transformer layers (GPT-2=12, OpenChat=32)
///
/// # Example
/// ```rust
/// let engine = Gpt2Engine::new(weights);
/// engine.emit_causal_edges = true;
/// engine.forward(&input);
/// let metrics = compute_attention_metrics(&engine.causal_edges, 12);
/// // metrics.attention_entropy → thinking style signal
/// ```
pub fn compute_attention_metrics(
    edges: &[super::jina::causal::CausalEdge64],
    num_layers: usize,
) -> AttentionMetrics {
    if edges.is_empty() {
        return AttentionMetrics {
            attention_entropy: 0.0,
            mean_layer_depth: 0.0,
            cross_attention_density: 0.0,
            edge_count: 0,
        };
    }

    // TODO: implement
    // 1. Decode each CausalEdge64 → extract layer_id, attention_weight, source_pos, target_pos
    // 2. attention_entropy = -Σ p(w) log2(p(w)) over attention weights
    // 3. mean_layer_depth = Σ (layer_id × weight) / Σ weight
    // 4. cross_attention_density = count(|source_pos - target_pos| > context_window/2) / total
    todo!("Bridge 2: attention edges → thinking metrics")
}
```

### Inline comments to ADD to existing code:

```bash
# In gpt2/inference.rs, find `pub causal_edges` and ADD:
```
```rust
// BRIDGE NOTE: These edges are consumed by substrate::compute_attention_metrics()
// to derive thinking-style signals (entropy, depth, cross-attention density).
// Enable with: self.emit_causal_edges = true before forward().
// Wiring status: emission EXISTS, metric computation PENDING in substrate.rs.
```

```bash
# In openchat/inference.rs, find `pub causal_edges` and ADD same comment.
```

---

## BRIDGE 3: Crystal Encoder → Jina Learned Embeddings  
**~20 LOC addition to `crystal_encoder.rs`**

### What exists:
- `crystal_encoder.rs`: `absorb_external_embedding(embedding: &[f32]) → Node`
- `crystal_encoder.rs`: hash-based encoding via 65-entry NSM codebook
- `jina/runtime.rs`: `ModelRuntime` produces learned f32 embeddings

### What's missing:
Crystal encoder's `absorb_external_embedding` takes any `&[f32]`. Jina produces `&[f32]`.
The call is trivial but nobody documented that they connect.

### What to add:

```bash
# In crystal_encoder.rs, ADD a convenience function + doc comments:
```

```rust
/// Create an SPO Node from Jina-learned embeddings for subject, predicate, object.
///
/// This replaces the hash-based path (`from_content()`) with semantically
/// meaningful embeddings. The resulting Node has the same API but its
/// Hamming distances reflect SEMANTIC similarity, not hash similarity.
///
/// CONTEXT: See .claude/prompts/02_crystal_encoder.md
/// SOURCE: jina/runtime.rs (ModelRuntime::encode_token → f32 embedding)
/// TARGET: Any code that creates Nodes (graph.rs, substrate.rs)
///
/// # Example
/// ```rust
/// let jina = ModelRuntime::new(weights);
/// let s_emb = jina.encode_token("Alice");
/// let p_emb = jina.encode_token("knows");  
/// let o_emb = jina.encode_token("Bob");
/// let node = from_jina_spo(&s_emb, &p_emb, &o_emb);
/// // node.distance(&other) now reflects Jina's learned similarity
/// ```
pub fn from_jina_spo(
    subject_embedding: &[f32],
    predicate_embedding: &[f32],
    object_embedding: &[f32],
) -> super::node::Node {
    // absorb_external_embedding already does the projection.
    // We just need to call it three times with the right role bindings.
    // 
    // TODO: implement using existing absorb_external_embedding()
    // - Project each embedding to the corresponding S/P/O plane
    // - Bind with role vectors (same as nsm_encoder.rs SPO binding)
    // - Bundle into a single Node
    todo!("Bridge 3: Jina embeddings → crystal encoder → Node")
}
```

### Inline comments to ADD:

```bash
# In crystal_encoder.rs, find absorb_external_embedding and ADD:
```
```rust
// BRIDGE NOTE: This function is the generic entry point for ANY external embedding.
// For Jina-specific convenience, see from_jina_spo() below.
// For hash-based encoding (no external model), see from_content().
// The choice between these IS the System 1 vs System 2 tradeoff:
//   from_content()     = O(1), hash-based, no semantic understanding
//   from_jina_spo()    = O(embed_time), learned, semantic distances
//   palette_index only = O(1), 1 byte, coarse but instant
```

---

## BRIDGE 4: ModelRouter × ThinkingStyle → Model Selection
**~40 LOC addition to `models/router.rs`**

### What exists:
- `models/router.rs`: `ModelRouter` with `ModelBackend::{Gpt2, OpenChat, Jina, Bert, StableDiffusion}`
- `substrate.rs`: References `ThinkingStyle` (defined upstream in crewai-rust)

### What's missing:
ModelRouter dispatches by explicit `ModelBackend` enum. Nobody maps
thinking styles to model selection automatically.

### What to add:

```bash
# In models/router.rs, ADD (below existing dispatch methods):
```

```rust
/// Map a thinking style cluster to the optimal model backend.
///
/// This is the automatic model selection based on cognitive state:
/// - Analytical → OpenChat (Mistral-7B, GQA, good at multi-hop reasoning)
/// - Creative → GPT-2 (autoregressive, finds unexpected continuations)
/// - Focused → Jina (embedding model, best single-vector retrieval)
/// - Exploratory → BERT (bidirectional, sees context from both directions)
///
/// CONTEXT: See epiphany #3 and #7 from session review.
/// The planner's ThinkingStyle::cluster() already maps to 4 behaviors.
/// This function maps those behaviors to concrete model backends.
///
/// NOTE: ThinkingStyle lives in crewai-rust (upstream). This function
/// takes a u8 cluster ID (0-3) to avoid the cross-crate dependency.
/// Mapping: 0=Analytical, 1=Creative, 2=Focused, 3=Exploratory
///
/// # Elevation integration (see epiphany #7):
/// The thinking style also sets the STARTING codec resolution:
/// - Analytical (0) → start at L2 (Cascade, 34 bytes/token)
/// - Creative (1)   → start at L5 (full transformer forward pass)
/// - Focused (2)    → start at L0 (palette index, 1 byte/token)
/// - Exploratory (3) → start at L1 (Base17 scan, 34 bytes/token)
pub fn backend_for_thinking_cluster(&self, cluster_id: u8) -> ModelBackend {
    match cluster_id {
        0 => { // Analytical → deliberate reasoning
            if self.openchat.is_some() { ModelBackend::OpenChat }
            else if self.gpt2.is_some() { ModelBackend::Gpt2 }
            else { ModelBackend::Jina } // fallback
        }
        1 => { // Creative → generative continuation
            if self.gpt2.is_some() { ModelBackend::Gpt2 }
            else if self.openchat.is_some() { ModelBackend::OpenChat }
            else { ModelBackend::Jina }
        }
        2 => { // Focused → retrieval
            ModelBackend::Jina // always Jina for retrieval
        }
        3 => { // Exploratory → bidirectional context
            if self.openchat.is_some() { ModelBackend::Bert } // prefer bidirectional
            else { ModelBackend::Jina }
        }
        _ => ModelBackend::Jina, // safe default
    }
}

/// Recommended starting elevation level for a thinking cluster.
///
/// See: session_unified_vector_search.md for elevation levels.
/// L0 = palette (1B, O(1)), L1 = Base17 (34B, scan), L2 = Cascade (3-stroke),
/// L3 = Batch, L4 = IVF, L5 = full transformer forward pass.
pub fn starting_elevation_for_cluster(cluster_id: u8) -> u8 {
    match cluster_id {
        0 => 2, // Analytical → start at Cascade (enough precision for reasoning)
        1 => 5, // Creative → start at full transformer (need generative)
        2 => 0, // Focused → start at palette (fastest retrieval)
        3 => 1, // Exploratory → start at Base17 scan (broad but cheap)
        _ => 1, // default: scan
    }
}
```

---

## BRIDGE 5: CLAM → Cascade Candidate Prefilter
**~30 LOC addition to `cascade.rs`**

### What exists:
- `clam.rs`: `ClamTree::rho_nn(query, k) → Vec<(usize, u32)>` — sublinear NN search
- `cascade.rs`: `Cascade::query(fingerprint) → RankedHit[]` — 3-stroke verification

### What's missing:
Cascade does a FULL SCAN of all candidates in stage 1. CLAM can prefilter
to O(k·2^LFD·log n) candidates BEFORE stage 1 starts.

### What to add:

```bash
# In cascade.rs, ADD (don't modify existing query method):
```

```rust
/// Query with CLAM prefiltering — sublinear candidate generation.
///
/// Instead of scanning ALL entries in stage 1 (O(n)), use the CLAM tree
/// to generate a candidate set first (O(k·2^LFD·log n)), then run the
/// 3-stroke cascade only on candidates.
///
/// CONTEXT: See .claude/prompts/01_clam_qualiacam.md
/// SOURCE: clam.rs ClamTree::rho_nn()
/// TARGET: Cascade stage 1 input
///
/// For n=1M entries and LFD=3: full scan = 1M comparisons.
/// CLAM prefilter = ~8k candidates. Cascade on 8k = 800 (at 90% rejection).
/// Total: 8800 comparisons vs 1M. 113× speedup.
///
/// # Example
/// ```rust
/// let tree = ClamTree::build(&data, &distance_fn);
/// let cascade = Cascade::new(&data, &config);
/// let results = cascade.query_with_clam(&fingerprint, &tree, 100);
/// ```
pub fn query_with_clam(
    &self,
    query: &super::fingerprint::Fingerprint,
    clam_tree: &super::clam::ClamTree,
    top_k: usize,
) -> Vec<super::node::RankedHit> {
    // Step 1: CLAM generates candidate indices (sublinear)
    // Step 2: Cascade runs 3-stroke ONLY on candidates (not full corpus)
    //
    // TODO: implement
    // - Call clam_tree.rho_nn(query, top_k * 10) for 10× oversampling
    // - Convert candidate indices to the format Cascade expects
    // - Run self.query_subset(query, &candidates) instead of self.query(query)
    // 
    // NOTE: query_subset() doesn't exist yet. Add it as a private method
    // that takes a slice of candidate indices and only checks those.
    todo!("Bridge 5: CLAM prefilter → Cascade 3-stroke")
}
```

### Inline comments to ADD:

```bash
# In clam.rs, find rho_nn and ADD:
```
```rust
// BRIDGE NOTE: The output of rho_nn() feeds into Cascade::query_with_clam()
// as a prefilter. This turns the Cascade from O(n) to O(k·2^LFD·log n).
// See: cascade.rs query_with_clam(), .claude/prompts/01_clam_qualiacam.md
```

```bash
# In cascade.rs, find the existing query() method and ADD:
```
```rust
// BRIDGE NOTE: For sublinear search, use query_with_clam() instead of query().
// query() does a full scan — correct but O(n).
// query_with_clam() uses the CLAM tree to prefilter — same results, O(k·log n).
// Both methods return the same RankedHit[] type.
```

---

## REVIEW EXISTING PROMPTS: Corrections and Expansions

After wiring the bridges, review these prompts and ADD inline comments
where the prompts describe something that is now wired or partially wired.

### `session_deepnsm_cam.md`

Find the Architecture diagram and ADD below it:
```markdown
<!-- WIRING STATUS (updated YYYY-MM-DD):
  nsm_tokenizer.rs → nsm_parser.rs → nsm_encoder.rs: EXISTS, compiles
  nsm_encoder.rs → SpoBase17: EXISTS, tested
  SpoBase17 → VerbCodebook: BRIDGE 1 added (palette path)
  VerbCodebook → ThinkingGraph: UNWIRED (needs ThinkingGraph in rs-graph-llm)
  
  NEW BRIDGE: VerbCodebook::nearest_verb_by_palette() added in graph.rs
  This replaces the string-lookup path with palette-index O(1) lookup.
-->
```

### `session_unified_vector_search.md`

Find "READ FIRST" section and ADD:
```markdown
<!-- WIRING STATUS (updated YYYY-MM-DD):
  RaBitQ prefilter: EXISTS in lance-graph Cascade
  Cascade 3-stroke: EXISTS in ndarray cascade.rs
  bgz17 palette compose: EXISTS in palette_distance.rs
  CLAM → Cascade prefilter: BRIDGE 5 added (query_with_clam)
  Generative decompression: EXISTS in bgz17 generative.rs
  SimilarityTable: EXISTS in palette_distance.rs
  
  REMAINING GAP: The compose() call between RaBitQ binary codes and
  bgz17 palette indices. RaBitQ outputs binary, palette expects u8 index.
  The SimHash → palette projection is in jina/codec.rs but not exposed
  as a UDF for the query pipeline.
-->
```

### `01_clam_qualiacam.md`

Find the Architecture diagram and ADD:
```markdown
<!-- WIRING STATUS (updated YYYY-MM-DD):
  ClamTree::rho_nn(): EXISTS, tested
  Cascade::query(): EXISTS, tested
  ClamTree → Cascade: BRIDGE 5 added (query_with_clam in cascade.rs)
  QualiaCAM integration: UNWIRED (qualia.rs exists, not connected to CLAM)
-->
```

### `05_cross_repo_map.md`

Find the module table and ADD a new section:
```markdown
## Bridge Functions (added YYYY-MM-DD)

| Bridge | Source | Target | Status |
|--------|--------|--------|--------|
| 1. Palette verb lookup | jina/runtime.rs | graph.rs VerbCodebook | TODO stub |
| 2. Attention metrics | gpt2+openchat inference | substrate.rs | TODO stub |
| 3. Jina → Crystal → Node | jina/runtime.rs | crystal_encoder.rs | TODO stub |
| 4. ThinkingStyle → Model | substrate.rs | models/router.rs | TODO stub |
| 5. CLAM → Cascade | clam.rs | cascade.rs | TODO stub |

These bridges connect existing modules. Each is <50 LOC.
See: .claude/prompts/SESSION_WIRE_SYNAPSES.md for full spec.
```

---

## TESTS TO ADD

One test per bridge. Each test proves the wiring compiles and the
type signatures match. NOT integration tests — just bridge smoke tests.

```rust
#[cfg(test)]
mod bridge_tests {
    #[test]
    fn test_bridge1_verb_palette_signature() {
        // Verify VerbCodebook has nearest_verb_by_palette method
        // Verify it accepts (u8, &[[u16; 256]; 256])
        // Verify it returns Option<(&str, u16)>
    }

    #[test]
    fn test_bridge2_attention_metrics_signature() {
        // Verify compute_attention_metrics accepts (&[CausalEdge64], usize)
        // Verify it returns AttentionMetrics with 4 fields
    }

    #[test]
    fn test_bridge3_jina_crystal_signature() {
        // Verify from_jina_spo accepts (&[f32], &[f32], &[f32])
        // Verify it returns Node
    }

    #[test]
    fn test_bridge4_thinking_cluster_dispatch() {
        // Verify backend_for_thinking_cluster maps 0-3 to ModelBackend
        // Verify starting_elevation_for_cluster maps 0-3 to u8
        let router = ModelRouter::new();
        assert!(matches!(router.backend_for_thinking_cluster(2), ModelBackend::Jina));
        assert_eq!(ModelRouter::starting_elevation_for_cluster(2), 0);
    }

    #[test]
    fn test_bridge5_clam_cascade_signature() {
        // Verify Cascade has query_with_clam method
        // Verify it accepts (&Fingerprint, &ClamTree, usize)
        // Verify it returns Vec<RankedHit>
    }
}
```

---

## EXECUTION ORDER

```
1. READ all source files listed in "READ FIRST"              ~15 min
2. READ the 5 referenced prompts                             ~10 min
3. ADD Bridge 4 (router.rs — smallest, standalone)           ~10 min
4. ADD Bridge 2 (substrate.rs — pure function, no deps)      ~15 min
5. ADD Bridge 3 (crystal_encoder.rs — thin wrapper)          ~10 min
6. ADD Bridge 1 (graph.rs — needs palette field addition)    ~15 min
7. ADD Bridge 5 (cascade.rs — needs query_subset helper)     ~20 min
8. ADD bridge_tests module                                   ~15 min
9. ADD inline comments to existing code (BRIDGE NOTE: ...)   ~15 min
10. UPDATE referenced prompts with WIRING STATUS comments    ~10 min
11. cargo test --lib                                         ~5 min
12. Commit + push                                            ~2 min
```

## FINAL RULE

**Add. Comment. Test. Never subtract.**

Every `todo!()` is intentional. It marks WHERE the implementation goes
without GUESSING what it should be. A future session fills in the body
after reading the adjacent code. A `todo!()` that compiles is infinitely
more valuable than an implementation that doesn't.

Commit message: `wire: 5 bridges between jina/gpt2/openchat/crystal/clam/cascade/router`
