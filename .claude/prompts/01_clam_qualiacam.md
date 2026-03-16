# CLAM + QualiaCAM Integration

> Transcoded from rustynum `01_clam_qualiacam_and_stable_simd.md`.
> SIMD task is already complete in ndarray (AVX-512 → AVX2 → scalar dispatch in `hpc/bitwise.rs`).

## Goal

Wire the CLAM tree (`hpc/clam.rs`) for sublinear O(k·2^LFD·log n) nearest-neighbour
search, replacing brute-force O(N) scans in the cascade (`hpc/cascade.rs`).

## Architecture

```
Query fingerprint
       │
       ▼
  ┌─────────────┐
  │  ClamTree   │  triangle-inequality pruning
  │  rho_nn()   │  O(k · 2^LFD · log n)
  └──────┬──────┘
         │ candidate set (δ⁻ ≤ ρ)
         ▼
  ┌─────────────┐
  │  Cascade    │  3-stroke verification
  │  query()    │  Stroke 1 → 2 → 3
  └──────┬──────┘
         │ verified RankedHit[]
         ▼
  ┌─────────────┐
  │  Node SPO   │  harvest S/P/O distances
  │  distance() │  → NarsTruthValue
  └─────────────┘
```

## Key Types (all in `hpc/`)

| Type | Module | Role |
|------|--------|------|
| `ClamTree` | `clam.rs` | Divisive hierarchical clustering |
| `Cluster` | `clam.rs` | Tree node with center, radius, LFD |
| `Lfd` | `clam.rs` | Local fractal dimension |
| `rho_nn()` | `clam.rs` | ρ-nearest-neighbour via triangle inequality |
| `Cascade` | `cascade.rs` | 3-stroke adaptive search |
| `PackedDatabase` | `cascade.rs` | Stroke-aligned memory layout |
| `Node` | `node.rs` | 3-plane SPO cognitive atom |
| `Fingerprint<N>` | `fingerprint.rs` | Const-generic binary vector |

## Implementation Phases

### Phase 1: CLAM → Cascade Bridge
1. Add `ClamTree::rho_nn_cascade()` that feeds survivors into `Cascade::query()`
2. The cascade's stroke-1 becomes redundant when CLAM provides tight candidates
3. Benchmark: CLAM pruning should reduce stroke-1 scan by 10-100×

### Phase 2: SPO Distance Harvest
1. After cascade verification, decompose each hit via `Node::distance()`
2. Feed S/P/O distances into `NarsTruthValue` accumulation
3. Use `CausalityDecomposition` to extract directional relationships

### Phase 3: panCAKES Compression
1. Use `compress()` in `clam.rs` for XOR-diff encoding
2. Leverage cluster centers as codebook entries
3. Target: 4-8× compression on binary fingerprint databases

### Phase 4: CHAODA Anomaly Detection
1. Compute cluster-based anomaly scores from LFD distribution
2. Flag outlier concepts (high LFD = complex local geometry)
3. Feed anomaly scores into awareness thresholds (`bf16_truth.rs`)

## Verification
- `cargo test --lib` — all CLAM + cascade tests pass
- `cargo clippy -- -D warnings` — clean
- Benchmark: `rho_nn()` vs `knn_brute()` on 10K fingerprints, measure distance calls saved
- Bit-exact: CLAM search must find same results as brute-force (within ρ radius)
