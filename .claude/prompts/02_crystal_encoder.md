# Crystal Encoder Strategy

> Transcoded from rustynum `05_crystal_encoder_strategy.md`.
> Adapted for ndarray's `hpc/` module infrastructure.

## Goal

SPO 2³ factorization + crystal encoding as a transformer-free sentence
representation pipeline. Uses ndarray's SIMD Hamming infrastructure for
238× faster similarity than cosine distance.

## Core Insight

A `Node` (S/P/O triple of `Plane` accumulators) can encode sentence meaning
through factored binary projections. The 8-term SPO interaction
(∅, S, P, O, SP, PO, SO, SPO) provides Pearl Rung 1-3 causal decomposition
as a free byproduct.

## Architecture

```
Input text
    │
    ▼
┌──────────────┐
│  Projection  │  hpc/projection.rs
│  random_proj  │  dense → binary
└──────┬───────┘
       │ Fingerprint<256>
       ▼
┌──────────────┐
│    Node      │  hpc/node.rs
│  absorb()    │  accumulate into S/P/O planes
└──────┬───────┘
       │ 3 × Plane (16384-bit each)
       ▼
┌──────────────┐
│   Seal       │  hpc/seal.rs
│  merkle()    │  integrity hash (blake3)
└──────┬───────┘
       │ sealed Node
       ▼
┌──────────────┐
│  Cascade     │  hpc/cascade.rs
│  query()     │  nearest-neighbour search
└──────────────┘
```

## Key Types

| Type | Module | Role |
|------|--------|------|
| `Plane` | `plane.rs` | 16384-bit i8 accumulator, L1 resident |
| `Node` | `node.rs` | 3 × Plane (S/P/O) with attention mask |
| `Fingerprint<256>` | `fingerprint.rs` | 2048-byte binary vector |
| `Seal` | `seal.rs` | Blake3 merkle verification |
| `PackedQualia` | `bf16_truth.rs` | 16-dim i8 resonance + BF16 scalar |
| `NarsTruthValue` | `causality.rs` | Integer frequency/confidence |

## Three-Phase Pipeline

### Phase 1: External Embeddings (bootstrap)
- Use external embeddings (e.g., Jina) as input
- Project to binary via `hpc/projection.rs` random projection
- Accumulate into Node's S/P/O planes

### Phase 2: Distillation
- Train projection weights to minimize SPO structural loss
- Loss = Hamming(student_SPO, teacher_SPO) + λ·causal_divergence
- Use `bf16_hamming_scalar()` for weighted distance during training

### Phase 3: Pure Crystal Encoding
- Direct text → Node encoding without external embeddings
- 65 NSM semantic primes as codebook entries
- Each word maps to a `VerbCodebook` entry → binary fingerprint

## Performance Targets

| Metric | Transformer | Crystal |
|--------|------------|---------|
| Similarity | cosine ~10ms | Hamming ~20μs |
| Speedup | 1× | 238× (target, via VPOPCNTDQ) |
| Memory | 384-dim f32 | 2048-byte binary |
| Causal info | none | 8-term SPO decomposition |

## Verification
- `cargo test --lib` — Node absorb/distance tests pass
- Hamming distance via `bitwise::hamming_distance_raw()` matches reference
- Seal verification round-trips correctly
- CausalityDecomposition produces valid direction labels
