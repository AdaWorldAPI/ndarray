# Lance-Graph DataFusion Integration

> Transcoded from rustynum `07_lance_graph_notime_integration.md`.
> Adapted for ndarray's `hpc/` module infrastructure.

## Goal

Package ndarray's HPC operations as DataFusion scalar UDFs for use in
lance-graph query pipelines.

## Layer 1: DataFusion UDFs

Expose `hpc/` functions as DataFusion scalar UDFs:

| UDF Name | Source | Signature |
|----------|--------|-----------|
| `hamming` | `bitwise::hamming_distance_raw` | (Binary, Binary) → UInt64 |
| `spo_distance` | `node::Node::distance` | (Binary×3, Binary×3) → Struct |
| `nars_revision` | `causality::NarsTruthValue::revision` | (UInt16×2, UInt16×2) → UInt16×2 |
| `sigma_classify` | `cascade::Cascade::expose` | (UInt32, UInt64) → Utf8 |
| `bf16_hamming` | `bf16_truth::bf16_hamming_scalar` | (Binary, Binary, Binary) → Float64 |

## Layer 2: Causal Query Operators

New logical operators for the query compiler:

| Operator | Description |
|----------|-------------|
| `Factorize` | SPO 8-term interaction decomposition |
| `NarsAccumulate` | Evidence accumulation over windows |
| `SigmaClassify` | Cascade band classification |
| `CausalEdge` | Emit causality direction from decomposition |

## Layer 3: Validation

Use formal causal identifiability tests:
1. Binarization preserves causal structure (Hamming ↔ cosine rank correlation)
2. SPO factorization recovers known causal edges
3. NARS accumulation converges to ground truth
4. Cascade bands separate true matches from noise

## Implementation Notes

- All UDFs use `hpc/bitwise.rs` SIMD dispatch (no separate SIMD paths)
- Arrow buffers flow through `hpc/arrow_bridge.rs` zero-copy
- UDF registration is external to ndarray (in lance-graph or consumer crate)
- ndarray provides the compute kernels; integration crate provides the glue
