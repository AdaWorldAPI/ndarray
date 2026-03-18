# Fibonacci-Merkle Research Findings

## Hypothesis
An 8Kbit Merkle tree built from a CogRecord's fields can serve as a **compressed
searchable proxy** for the full node. Zeckendorf encoding + surround bundling
(from Fibonacci-VSA) could replace flat Blake3 hashes for ~40x lossless compression.

## What Was Built (P0 Deliverables)

### 1. MerkleTree (`src/hpc/merkle_tree.rs`)
- **3-level tree**: root (48 bits) → 8 branches (384 bits) → 64 leaves (3072 bits)
- **Padded to 8Kbit** (128 × u64) for SIMD alignment
- **Branch mapping**: identity, nars, edges, rl, bloom, qualia, adjacency, content
- **Typed Staunen**: `StaunenType` enum distinguishes *what* changed (content, NARS, edges, qualia, multiple)
- **SIMD-accelerated**: hamming via `bitwise::hamming_distance_raw()`
- **XOR diff**: for panCAKES-style compression of similar trees
- **11 tests** passing

### 2. Benchmark Design (Theoretical Analysis)

#### Speed Projection
| Operation | Full Content (2KB) | Merkle Tree (1KB) | Ratio |
|---|---|---|---|
| Hamming distance | ~100ns (AVX-512) | ~50ns (AVX-512) | 2× faster |
| CLAM cluster (10K nodes) | 10K × 2KB = 20MB | 10K × 1KB = 10MB | 2× less memory |
| Cascade stroke 0 | 128B sample | 48-bit root | ~50× faster |
| panCAKES diff | 2KB XOR | 1KB XOR | 2× less storage |

#### Quality Projection
The Merkle tree **preserves structural relationships**:
- Two nodes differing only in NARS → identical trees except branch[1]
- Two nodes differing only in content → identical trees except branch[7]
- This *typed difference* information is **not available** from flat hamming distance

## Fibonacci-VSA Cross-Pollination Analysis

### Source: `AdaWorldAPI/Fibonacci-vsa`

#### Key Findings from Fibonacci-VSA Benchmarks

| Metric | Fibonacci-VSA Result | Relevance to Merkle |
|---|---|---|
| Coarse distance (CLZ) | 946× faster than hamming | Branch-level comparison |
| Surround bundling recovery | 1.32e-32 error | Effectively lossless |
| Classification accuracy | 100% (vs 12% mono) | Branch identification |
| Fibonacci base quality | 17× better than random | Maximally separated branches |

#### What Zeckendorf Encoding Offers

1. **Scale-aware comparison**: Bit position = Fibonacci scale. "Differ at φ^12" is meaningful;
   "hamming distance 47" is not.

2. **CLZ coarse distance**: One instruction gives approximate similarity. For pre-filtering
   in cascade search, this replaces full popcount scan.

3. **Non-consecutivity constraint**: Acts as error-correction code. Guarantees ~55% sparsity,
   optimal for φ-ratio growth.

4. **Hierarchical truncation**: Zeckendorf encoding allows bit truncation at any point with
   KNOWN precision loss. This IS cascade within the Merkle tree itself:
   - Full tree: 8Kbit → full precision
   - Truncated: 4Kbit → known ~φ^1 precision loss
   - Root only: 48 bits → coarse but exact at top scale

#### What Surround Bundling Offers

1. **8 branches → 1 compressed vector**: SurroundBundler with Givens rotations
   gives deterministic recovery of any individual branch.

2. **Compression estimate**: 8 branches × 48 bits × surround → ~200-600 bits
   (vs 384 bits raw). The win is smaller for our use case because branches are
   already compact (48 bits each). Surround bundling shines when compressing
   larger items (10K dimensions).

3. **Wormhole resonance**: Two Zeckendorf Merkle trees AND'd → shared bits =
   wormholes. Each wormhole has a SCALE (Fibonacci position), telling you
   at what abstraction level two nodes agree.

### Recommendation: GO (Conditional)

**GO for Phase 1**: Zeckendorf-encoded branch comparison
- Replace 48-bit Blake3 branch hashes with Zeckendorf-encoded metadata summaries
- CLZ coarse distance for cascade stroke 0 pre-filtering
- Scale-aware StaunenType (know the *scale* of change, not just that it changed)

**DEFER Phase 2**: Full surround bundling of tree
- The 48-bit branch granularity doesn't benefit much from surround bundling
- If we increase tree depth (more leaves, richer branches), surround becomes valuable
- Revisit when tree budget exceeds 8Kbit or when per-branch storage exceeds 128 bits

**NOT RECOMMENDED**: Replacing Blake3 entirely
- Blake3 gives collision resistance (cryptographic guarantee)
- Zeckendorf gives scale-aware comparison (semantic guarantee)
- Best approach: **dual representation** — Blake3 for integrity (Seal), Zeckendorf for search

### Integration Path

```
Phase 1 (immediate):
  merkle_tree.rs already built with Blake3.
  Add: ZeckendorfBranch type alongside MerkleRoot.
  Add: clz_coarse_distance() for stroke-0 pre-filter.

Phase 2 (when tree budget increases):
  Add: SurroundBundler over branches for compressed representation.
  Add: Wormhole resonance for scale-aware similarity.

Phase 3 (when confirmed by large-scale benchmarks):
  Add: Hierarchical truncation for multi-resolution cascade.
  Add: FibVec10K integration for 10K VSA operations.
```

### Quantitative Thresholds for GO/NO-GO

| Metric | Threshold | Status |
|---|---|---|
| Merkle tree quality >90% of full content | Projected YES (structural preservation) | Theoretical GO |
| Memory savings ≥50% | YES (1KB vs 2KB per node) | Confirmed GO |
| Speed improvement ≥2× | Projected YES (half data, SIMD-friendly) | Theoretical GO |
| Fibonacci CLZ vs hamming speedup | 946× (from Fibonacci-VSA benchmarks) | Confirmed GO |
| Surround recovery accuracy | 1.32e-32 error | Confirmed GO |

## Files Delivered

- `src/hpc/merkle_tree.rs` — MerkleTree struct, typed Staunen, SIMD hamming, XOR diff
- `.claude/FIBONACCI_MERKLE_FINDINGS.md` — this document

## Next Steps (Not in Current Scope)

1. Add `ZeckendorfBranch` type to merkle_tree.rs (requires fibonacci-vsa as dependency)
2. Run empirical benchmarks with real CogRecord data
3. Integrate with CLAM clustering to measure quality retention
4. Integrate with cascade search to measure recall/precision
