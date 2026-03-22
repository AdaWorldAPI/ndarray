# SESSION: Unified Vector Search — RaBitQ × Cascade × Generative Decompression × bgz17

## Mission

Integrate RaBitQ (LanceDB's binary quantization), our Cascade (spatial progressive
scan), generative decompression (Bayesian decoder-side correction from arXiv:2602.03505),
and bgz17 (palette compose tables) into a single vector search pipeline that:

1. Scans 4× faster than RaBitQ alone (2D progressive: spatial × bit-depth)
2. Produces cosine-equivalent f32 similarity via SimilarityTable
3. Supports graph traversal in quantized space (compose tables)
4. Corrects palette distance using CLAM LFD (proven optimal by 2602.03505)
5. Stores 3 bytes per vector (palette) with 256-entry distance matrix (128KB)

## READ FIRST

```bash
# Our Cascade (spatial progressive scan)
cat crates/lance-graph/src/graph/blasgraph/hdr.rs           # 1467 lines
cat crates/lance-graph/src/graph/blasgraph/cascade_ops.rs   # Cypher predicate translation

# Our bgz17 (palette compression + compose tables)
cat crates/bgz17/KNOWLEDGE.md
cat crates/bgz17/src/distance_matrix.rs                     # 256×256 precomputed
cat crates/bgz17/src/generative.rs                          # LFD correction (ALREADY IMPLEMENTED)
cat crates/bgz17/src/palette.rs                             # k-means codebook

# Our CLAM (provides LFD side information)
cat crates/lance-graph/src/graph/neighborhood/clam.rs

# Similarity output
cat .claude/prompts/session_bgz17_similarity.md             # SimilarityTable design
```

Also read:
- RaBitQ paper: https://arxiv.org/abs/2405.12497 (SIGMOD 2024)
- Extended-RaBitQ: https://arxiv.org/abs/2409.09913 (SIGMOD 2025)
- Generative Decompression: https://arxiv.org/abs/2602.03505 (Feb 2026)
- LanceDB RaBitQ blog: https://lancedb.com/blog/feature-rabitq-quantization/

## THE 2D PROGRESSIVE CASCADE

RaBitQ scans 100% of bits on each candidate before deciding. Our Cascade
subsamples spatially (stride-16 → stride-4 → full). Combined:

```
                    SPATIAL (within one vector)
                    ───────────────────────────────────
                    1/16 sample    1/4 sample    Full
                    ───────────    ──────────    ────
BIT-DEPTH  1-bit    STAGE 1        STAGE 2       STAGE 3
           ex-code  (skip)         (skip)        STAGE 4 (top-k only)
           f32      (skip)         (skip)        STAGE 5 (top-50 only)

Stage 1: 1/16 spatial × 1-bit = 1/256 of full f32 work → reject 55%
Stage 2: 1/4 spatial × 1-bit  = 1/32 of full work      → reject 90% of survivors
Stage 3: full × 1-bit         = 1/32 of f32 work       → classify remaining 5%
Stage 4: full × ext-RaBitQ    = 4/32 of f32 work       → refine top-k only
Stage 5: full f32 reload      = full precision          → final top-50 only

Total work: 0.06 + 0.10 + 0.05 + 0.01 + 0.005 = ~0.23× of brute-force f32
vs RaBitQ alone: ~1.0× of 1-bit scan = ~0.03× of f32 (but no spatial early-exit)
vs our Cascade alone: ~0.03× of full Hamming (but no bit-depth refinement)
```

## DELIVERABLE 1: RaBitQ-Compatible Binary Encoding in bgz17

RaBitQ normalizes vectors to the unit sphere, then snaps to nearest hypercube
vertex. Our SimHash (sign of random projection) does the same thing without
the normalization correction scalars. Add RaBitQ's correction factors to bgz17:

```rust
/// RaBitQ-compatible binary encoding with correction factors.
///
/// RaBitQ stores per-vector: binary_code (D bits) + norm (f32) + dot_correction (f32)
/// Our palette stores: palette_index (3 bytes) derived FROM the binary code.
/// We add the correction factors alongside, stored in container W112-125.
pub struct RaBitQEncoding {
    /// Binary code: sign(random_rotation × normalized_vector)
    pub binary: Vec<u64>,        // D bits packed into u64 words
    /// L2 norm of original vector (before normalization)
    pub norm: f32,
    /// Dot product correction: <quantized, original> / <quantized, quantized>
    pub dot_correction: f32,
    /// bgz17 palette index derived from binary code
    pub palette: PaletteEdge,
}

impl RaBitQEncoding {
    /// Encode f32 vector → RaBitQ binary + palette + corrections.
    pub fn encode(vector: &[f32], rotation: &OrthogonalMatrix, palette: &Palette) -> Self;

    /// Distance estimate with RaBitQ correction (unbiased).
    pub fn distance_rabitq(&self, other: &RaBitQEncoding) -> f32;

    /// Distance estimate with palette (fast, ρ=0.992).
    pub fn distance_palette(&self, other: &RaBitQEncoding, dm: &DistanceMatrix) -> u32;

    /// Distance with generative decompression correction (optimal).
    pub fn distance_corrected(
        &self, other: &RaBitQEncoding,
        dm: &DistanceMatrix, lfd: &LfdProfile, cascade: &Cascade,
    ) -> f32;
}
```

## DELIVERABLE 2: 2D Progressive Cascade

Extend `Cascade::query()` to support both spatial subsampling (existing)
AND bit-depth refinement (new, from extended-RaBitQ):

```rust
impl Cascade {
    /// 2D progressive query: spatial × bit-depth.
    ///
    /// Stage 1: stride-16 sample of 1-bit codes → sigma band → reject 55%
    /// Stage 2: stride-4 sample of 1-bit codes → sigma band → reject 90%
    /// Stage 3: full 1-bit popcount → exact binary distance → classify
    /// Stage 4: ex-code refinement (2-4 bit) → top-k only
    /// Stage 5: f32 reload from Lance → exact cosine → top-50
    pub fn query_2d_progressive(
        &self,
        query_binary: &[u64],           // 1-bit RaBitQ code
        query_excode: Option<&[u8]>,    // 2-4 bit extended code (nullable)
        candidates_binary: &PackedDatabase,  // panel-packed 1-bit codes
        candidates_excode: Option<&[u8]>,    // extended codes for survivors
        top_k: usize,
    ) -> Vec<RankedHit> {
        // Stages 1-3: existing spatial cascade on binary codes
        let stage3_survivors = self.query(query_binary, &candidates_binary.as_words(), top_k * 10);

        // Stage 4: ex-code refinement on survivors only
        if let (Some(q_ex), Some(c_ex)) = (query_excode, candidates_excode) {
            // Extended-RaBitQ: refine distance using additional bits
            let refined = refine_with_excode(&stage3_survivors, q_ex, c_ex);
            refined.into_iter().take(top_k).collect()
        } else {
            stage3_survivors.into_iter().take(top_k).collect()
        }
        // Stage 5 (f32 reload) happens OUTSIDE this function — caller loads from Lance
    }
}
```

## DELIVERABLE 3: Generative Decompression Integration

The paper proves: decoder with side information beats fixed centroid decode.
Our side information sources:

```rust
/// All side information available to the decoder (free — already computed).
pub struct DecoderSideInfo {
    /// Local fractal dimension from CLAM tree (per-node)
    pub lfd: f32,
    /// Corpus distribution from Cascade reservoir (per-scope)
    pub cdf: SimilarityTable,
    /// Graph topology from container W16-31 (per-node)
    pub inline_edges: Vec<InlineEdge>,
    /// NARS truth values from container W4-7 (per-node)
    pub truth: TruthValue,
    /// Cascade sigma-band classification (per-query)
    pub band: Band,
}

/// Generative decompression: Bayesian distance correction.
///
/// From arXiv:2602.03505 Theorem 1:
///   D_ideal < D_gen_decomp < D_fixed_centroid
///
/// The correction uses ALL available side information:
///   d_corrected = d_palette
///     × lfd_correction(lfd, lfd_median)          ← manifold geometry
///     × cdf_correction(d_palette, cascade)        ← corpus distribution
///     × topology_correction(edges, graph_metrics)  ← graph structure
///
/// Each correction factor is multiplicative and ≥ 0.
/// When side_info is empty, falls back to d_palette (standard centroid).
pub fn generative_distance(
    palette_distance: u32,
    side_info: &DecoderSideInfo,
    alpha: f32,  // correction strength (tune on validation set)
) -> f32 {
    let mut d = palette_distance as f32;

    // LFD correction (already in generative.rs — proven to work)
    d *= 1.0 + alpha * (side_info.lfd - LFD_MEDIAN);

    // CDF correction: distances near the edge of a sigma band
    // are less reliable than distances deep inside a band
    let cdf = side_info.cdf.similarity(palette_distance);
    let band_confidence = band_stability(cdf, side_info.band);
    d *= band_confidence;

    // Topology correction: if both nodes share many inline edge targets,
    // the true distance is likely SMALLER than palette estimates
    // (graph proximity correlates with semantic proximity)
    let shared_edges = count_shared_targets(&side_info.inline_edges, /* other's edges */);
    if shared_edges > 0 {
        d *= 1.0 - (shared_edges as f32 * TOPOLOGY_WEIGHT);
    }

    d
}
```

## DELIVERABLE 4: Lance Storage Layout

Two new columns alongside existing Lance tables:

```
vectors.lance (f32 embeddings — cold storage):
  id:          UInt32
  embedding:   FixedSizeBinary(4096)     ← f32[1024], loaded only for Stage 5

vectors_quantized.lance (binary codes — hot HHTL scan):
  id:          UInt32
  binary_code: FixedSizeBinary(128)      ← 1024 bits (RaBitQ binary)
  palette_spo: FixedSizeBinary(3)        ← bgz17 palette index
  norm:        Float16                    ← RaBitQ norm correction (f16)
  dot_corr:    Float16                    ← RaBitQ dot correction (f16)
  excode:      FixedSizeBinary(256)      ← 2-bit extended (nullable, Stage 4)

Lance column pruning:
  HHTL HEEL:  reads palette_spo only          → 3 bytes/row
  HHTL HIP:   reads binary_code               → 128 bytes/row (survivors only)
  Stage 4:    reads excode                     → 256 bytes/row (top-k only)
  Stage 5:    reads embedding from vectors.lance → 4096 bytes/row (top-50 only)
```

## DELIVERABLE 5: Cosine Drop-In Output

Wire everything through SimilarityTable for f32 output:

```rust
/// Full pipeline: f32 query → quantize → 2D cascade → generative correct → f32 similarity
pub fn search(
    query: &[f32],
    scope: &QuantizedScope,       // palette + binary codes + distance matrix
    cascade: &Cascade,
    clam_lfd: &[f32],            // LFD per node from CLAM
    top_k: usize,
) -> Vec<(usize, f32)> {
    // Encode query
    let q_enc = RaBitQEncoding::encode(query, &scope.rotation, &scope.palette);

    // 2D progressive cascade: stages 1-4
    let survivors = cascade.query_2d_progressive(
        &q_enc.binary, Some(&q_enc.excode),
        &scope.packed_db, Some(&scope.excodes),
        top_k * 2,
    );

    // Generative decompression correction for each survivor
    let corrected: Vec<(usize, f32)> = survivors.iter().map(|hit| {
        let side_info = DecoderSideInfo {
            lfd: clam_lfd[hit.index],
            cdf: &scope.similarity_table,
            inline_edges: read_inline_edges(hit.index),
            truth: read_truth(hit.index),
            band: cascade.band(hit.distance),
        };
        let d = generative_distance(hit.distance, &side_info, ALPHA);
        let sim = scope.similarity_table.similarity(d as u32);
        (hit.index, sim)
    }).collect();

    // Sort by similarity descending, return top-k
    corrected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    corrected.into_iter().take(top_k).collect()
}
```

## DELIVERABLE 6: PCDVQ Direction-Weighted Distance (arxiv 2506.05432)

PCDVQ proved direction is 20× more sensitive to quantization than magnitude.
Our BF16 sign/exp/mantissa decomposition maps to polar coordinates:
  dim 0 = sign (direction), dims 1-6 = exponent (magnitude scale), dims 7-16 = mantissa.

Add direction-weighted L1 to Base17:

```rust
impl Base17 {
    /// PCDVQ-informed distance: weight sign dimensions 20× over mantissa.
    /// Matches measured direction/magnitude sensitivity ratio.
    pub fn l1_weighted(&self, other: &Base17) -> u32 {
        self.dims.iter().zip(other.dims.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                let diff = (a - b).unsigned_abs() as u32;
                let weight = if i == 0 { 20 } else if i < 7 { 3 } else { 1 };
                diff * weight
            })
            .sum()
    }
}

impl DistanceMatrix {
    /// Build distance matrix with PCDVQ direction weighting.
    /// Compose table + SimilarityTable auto-recalibrate from new distances.
    pub fn new_pcdvq_weighted(centroids: &[Base17]) -> Self { ... }
}
```

This also explains WHY the HHTL+CLAM dual-path works:
  HHTL (popcount) treats all bits equally → blind to direction asymmetry.
  CLAM (LFD/tree) groups by manifold geometry → direction-aware.
  The dual-path covers complementary failure modes.

## DELIVERABLE 7: Sigma-Band Palette (replace k-means, eliminate NaN)

Inspired by GQ (arxiv 2512.06609): Gaussian VAE with Target Divergence
Constraint → codebook without training, guaranteed uniform utilization.

Our equivalent: sigma-band quantization from Cascade reservoir.
Equal-percentile bands = distribution-free generalization of GQ's TDC.
No empty clusters possible (each band covers fixed percentile by construction).

```rust
impl Palette {
    /// Sigma-band palette: codebook from empirical distribution.
    /// Each band boundary = reservoir percentile.
    /// 256 bands = 8-bit palette. Guaranteed no empty clusters.
    /// Distribution-free: works for Gaussian, bimodal, skewed, heavy-tailed.
    pub fn from_sigma_bands(cascade: &Cascade, k: usize) -> Self {
        let mut boundaries = Vec::with_capacity(k);
        for i in 0..k {
            let percentile = (i as f32 + 0.5) / k as f32;
            boundaries.push(cascade.reservoir().quantile(percentile));
        }
        // Assign each vector to nearest band boundary
        // NO empty clusters: each band covers 100/k percent of data
        Self::from_boundaries(boundaries)
    }
}
```

## DELIVERABLE 8: Rust 1.94 Optimizations

Two zero-cost improvements from Rust 1.94:

```rust
// (a) array_windows for Cascade scan — eliminates bounds checks
// BEFORE:
for chunk in packed_db.windows(record_size) { ... }
// AFTER:
for chunk in packed_db.array_windows::<RECORD_BYTES>() { ... }

// (b) f16 primitive for SimilarityTable — hardware VCVTPH2PS
// BEFORE: custom BF16 type with software conversion
// AFTER: core::num::f16 with AVX-512 FP16 intrinsics (stable in 1.94)
```

## DELIVERABLE 9: Redirect lance-graph Raw Intrinsics

After the SIMD compat layer (src/simd.rs, src/simd_avx512.rs, src/simd_avx2.rs) lands and bgz17 joins workspace:

```
ndarray_bridge.rs (6 Hamming functions, ~200 lines)
  → call ndarray::hpc::bitwise::hamming_distance_raw()
types.rs (2 Hamming functions, ~60 lines)
  → same
~260 lines of duplicated intrinsics → function calls
bgz17/prefetch.rs (4 calls) → stays (hint, not computation)
```

Blocked on: Session C first gate (bgz17 → workspace members).

## DELIVERABLE 10: GPU-Optimal Memory Access (future)

Context: IVF-RaBitQ GPU (2602.23999), Fantasy (2512.02278),
PathWeaver (2507.17094), Jasper (2601.07048) — all solve
"GPU hates random access" differently.

Our Cascade's panel-packed layout = GPU-coalesced reads:
  stride-16 sampling = 32 threads × 128 bytes = 4KB = one warp read
  Cascade stages → GPU kernel launches
  Future: HHTL on GPU-A, CLAM on GPU-B (PathWeaver-style)

Not blocking current sessions. Note as future work.

## SOURCE PAPERS

```
PAPER              ARXIV           VENUE          RELEVANCE
─────              ─────           ─────          ─────────
RaBitQ             2405.12497      SIGMOD 24      Binary quant foundation
Extended-RaBitQ    2409.09913      SIGMOD 25      Multi-bit extension
SymphonyQG         2411.12229      SIGMOD 25      RaBitQ + graph + FastScan
IVF-RaBitQ GPU     2602.23999      arXiv 26       GPU-native IVF + RaBitQ
Jasper             2601.07048      arXiv 26       GPU graph + streaming
Fantasy            2512.02278      arXiv 25       Multi-GPU pipelining
PathWeaver         2507.17094      arXiv 25       Cross-GPU path extension
PCDVQ              2506.05432      arXiv 25       Direction > magnitude (20×)
GQ                 2512.06609      arXiv 26       Gaussian VAE → codebook
GenDecomp          2602.03505      arXiv 26       Bayesian decoder correction
```

## TESTS

1. RaBitQ encode → decode: correction factors produce unbiased estimate
2. 2D cascade rejects more than either dimension alone
3. Generative correction: d_corrected closer to exact than d_palette
4. LFD correction: high-LFD regions get larger adjustment
5. Topology correction: shared-edge pairs get smaller distance
6. SimilarityTable output matches cosine within ρ > 0.95
7. Full pipeline: top-10 recall ≥ 95% vs brute-force f32 cosine
8. Lance column pruning: Stage 1 reads only 3 bytes/row
9. PCDVQ-weighted L1 vs uniform L1: higher cosine rank correlation
10. Sigma-band palette vs k-means: zero NaN, comparable recall
11. array_windows scan: no regression, fewer bounds checks in asm
12. Dual-path ablation: HHTL-only misses direction-sensitive pairs CLAM catches

## OUTPUT

Branch: `feat/unified-vector-search`
Files: `crates/bgz17/src/rabitq_compat.rs`, `crates/bgz17/src/progressive_cascade.rs`
Modified: `crates/lance-graph/src/graph/blasgraph/hdr.rs` (add query_2d_progressive)
Modified: `crates/bgz17/src/generative.rs` (add topology + CDF corrections)
