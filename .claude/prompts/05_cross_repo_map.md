# Cross-Repository Integration Map

> Transcoded from rustynum `09_cross_repo_integration_map.md`.
> Adapted for ndarray as the unified compute engine.

## Overview

ndarray serves as the unified compute engine, replacing rustynum's split
across rustynum-core, rustynum-bnn, rustynum-clam, and rustynum-arrow.

## ndarray HPC Module Map

| Module | Origin | Core Responsibility |
|--------|--------|---------------------|
| `hpc/bitwise.rs` | ndarray native | SIMD Hamming, popcount dispatch |
| `hpc/fingerprint.rs` | rustynum-core | Const-generic binary vectors |
| `hpc/plane.rs` | rustynum-core | 16384-bit i8 accumulator |
| `hpc/node.rs` | rustynum-core | SPO cognitive atom |
| `hpc/seal.rs` | rustynum-core | Blake3 merkle integrity |
| `hpc/cascade.rs` | rustynum-core (hdr) | 3-stroke adaptive search |
| `hpc/bf16_truth.rs` | rustynum-core | BF16 weighted Hamming |
| `hpc/causality.rs` | rustynum-core | Causality decomposition |
| `hpc/blackboard.rs` | rustynum-core | Zero-copy arena allocator |
| `hpc/bnn.rs` | rustynum-bnn | Binary neural network inference |
| `hpc/clam.rs` | rustynum-clam | CLAM tree + search |
| `hpc/arrow_bridge.rs` | rustynum-arrow | Arrow zero-copy bridge |
| `hpc/projection.rs` | ndarray native | Random projection |
| `hpc/cogrecord.rs` | ndarray native | 4-channel cognitive units |
| `hpc/graph.rs` | ndarray native | VerbCodebook graph ops |
| `hpc/packed.rs` | ndarray native | Packed memory layouts |

## Integration Flow

```
External embeddings (Jina, etc.)
    │
    ▼
Projection → Fingerprint<256>
    │
    ▼
Node::absorb() → 3 × Plane
    │
    ▼
ClamTree::build() → hierarchical index
    │
    ▼
rho_nn() + Cascade::query() → RankedHit[]
    │
    ▼
causality_decompose() → CausalityDecomposition
    │
    ▼
NarsTruthValue::revision() → accumulated evidence
    │
    ▼
Arrow bridge → Lance storage
```

## Consumer Crates

Downstream crates that use ndarray's HPC modules:

| Crate | Uses | Purpose |
|-------|------|---------|
| lance-graph | arrow_bridge, cascade, bitwise | Graph query engine |
| ladybug-rs | node, plane, seal, causality | Cognitive substrate |
| qualia-cam | clam, cascade, bnn | Real-time recognition |

## Verification Hierarchy

| Level | Claim | Evidence |
|-------|-------|----------|
| Rung 1 (proven) | SIMD dispatch correct | Bit-exact tests vs scalar |
| Rung 1 (proven) | XOR group properties | Algebraic property tests |
| Rung 2 (statistical) | Cascade bands separate signal | Distributional tests |
| Rung 3 (needs theorem) | SPO recovers causal structure | Formal proof required |
