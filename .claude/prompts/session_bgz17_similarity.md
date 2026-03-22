# SESSION: bgz17 Similarity Score — Drop-In Cosine Replacement

## Mission

Add `similarity() -> f32` to bgz17 that any consumer expecting cosine can use.
Not a band classification (Foveal/Near/Good). Not a raw distance (u32).
A continuous f32 in [0.0, 1.0] (or [-1.0, 1.0]) calibrated from the corpus
distribution, with 256-point resolution, stored in BF16, batch-computed via SIMD.

## Why This Works

For binary vectors, cosine similarity = `1 - 2×hamming/N`. Exact linear relationship.
bgz17's palette distance is L1 on i16[17], a compressed proxy for Hamming (ρ=0.992).
The Cascade already calibrates μ and σ from the corpus. The reservoir already has
the empirical CDF via `quantile()`. We just need to invert it.

```
cosine 0.9    ≈  "3σ below random"  ≈  bgz17 similarity 0.95
cosine 0.75   ≈  "2σ below random"  ≈  bgz17 similarity 0.88
cosine 0.5    ≈  "1σ below random"  ≈  bgz17 similarity 0.73
cosine 0.0    ≈  "at random"        ≈  bgz17 similarity 0.50
```

The DECISION BOUNDARIES are equivalent. The values aren't identical but a consumer
who thresholds at cosine > 0.8 gets the same matches from bgz17 similarity > 0.88.

## READ FIRST

```bash
cat crates/lance-graph/src/graph/blasgraph/hdr.rs    # Cascade, ReservoirSample, quantile()
cat crates/bgz17/src/base17.rs                       # Base17::l1(), SpoBase17::l1()
cat crates/bgz17/src/distance_matrix.rs              # DistanceMatrix::distance(), spo_distance()
cat crates/bgz17/src/bridge.rs                       # Bgz17Distance trait, Precision enum
```

## DELIVERABLE 1: SimilarityTable (new: similarity.rs in bgz17)

A 256-entry BF16 lookup table built from the Cascade's reservoir.

```rust
use core::num::f16;  // Rust 1.94: f16 is a primitive type (IEEE 754 half-precision)

/// 256-entry empirical CDF lookup table for distance → similarity conversion.
///
/// Built from the Cascade's ReservoirSample at calibration time.
/// Each entry maps a distance bucket to f32 similarity in [0.0, 1.0].
/// Stored in f16 for cache efficiency (512 bytes total = fits in 8 cache lines).
///
/// Resolution: 256 buckets across the distance range [0, 2×μ].
/// Distances beyond 2×μ map to similarity 0.0 (pure noise).
/// Distances at 0 map to similarity 1.0 (identical).
///
/// f16 precision: 10-bit mantissa → ~0.001 resolution in [0, 1].
/// Good enough for threshold decisions. f32 computed on read via hardware VCVTPH2PS.
pub struct SimilarityTable {
    /// 256 similarity values in f16, indexed by distance bucket.
    /// table[0] = similarity for distance 0 (= 1.0)
    /// table[255] = similarity for distance ≥ 2×μ (= 0.0)
    table: [f16; 256],
    /// Distance range: bucket_width = (2 * mu) / 256.
    bucket_width: u32,
    /// Maximum distance that maps to bucket 255.
    max_distance: u32,
}

impl SimilarityTable {
    /// Build from Cascade's calibrated statistics.
    ///
    /// Uses the reservoir's empirical CDF for distribution-free calibration.
    /// Works for normal, bimodal, skewed, heavy-tailed distributions.
    pub fn from_cascade(cascade: &Cascade) -> Self {
        let mu = cascade.mu();
        let max_distance = 2 * mu; // Beyond 2×μ = pure noise
        let bucket_width = max_distance / 256;

        let mut table = [f16::from_f32(0.0); 256];

        // Build empirical CDF from reservoir
        // For each bucket center, compute: similarity = 1.0 - CDF(distance)
        // CDF(d) = fraction of reservoir samples ≤ d
        for i in 0..256 {
            let bucket_center = (i as u32 * bucket_width) + bucket_width / 2;
            let cdf = cascade.cdf_at(bucket_center); // fraction of pairs closer than this
            let similarity = 1.0 - cdf;
            table[i] = f16::from_f32(similarity);
        }

        Self { table, bucket_width, max_distance }
    }

    /// Build from Cascade using sigma-band interpolation (fallback when reservoir is small).
    ///
    /// Uses quarter-sigma cascade thresholds for 8 calibration points,
    /// plus band thresholds for 4 more. Piecewise linear between points.
    pub fn from_cascade_parametric(cascade: &Cascade) -> Self {
        let mu = cascade.mu();
        let sigma = cascade.sigma();
        let max_distance = 2 * mu;
        let bucket_width = max_distance / 256;

        let mut table = [f16::from_f32(0.0); 256];

        // 12 calibration points from cascade:
        // cascade[0..7] = μ-1σ to μ-3σ in quarter-sigma steps
        // bands[0..3] = μ-3σ, μ-2σ, μ-σ, μ
        // Map each to similarity via sigmoid(z) where z = (μ - d) / σ
        for i in 0..256 {
            let distance = (i as u32 * bucket_width) + bucket_width / 2;
            let z = (mu as f32 - distance as f32) / sigma.max(1) as f32;
            let similarity = 1.0 / (1.0 + (-z).exp()); // sigmoid
            table[i] = f16::from_f32(similarity);
        }

        Self { table, bucket_width, max_distance }
    }

    /// Lookup similarity for a raw distance. O(1).
    #[inline(always)]
    pub fn similarity(&self, distance: u32) -> f32 {
        if distance >= self.max_distance {
            return 0.0;
        }
        let bucket = (distance / self.bucket_width).min(255) as usize;
        f32::from(self.table[bucket])  // f16 → f32, hardware VCVTPH2PS if available
    }

    /// Batch similarity for multiple distances. SIMD-accelerated.
    ///
    /// On AVX-512 FP16: VGATHERDPS from f16 table + VCVTPH2PS → 16 similarities/instruction.
    /// On AVX2: scalar gather + software f16→f32 → still fast (table fits L1).
    pub fn similarity_batch(&self, distances: &[u32], out: &mut [f32]) {
        assert_eq!(distances.len(), out.len());
        for (i, &d) in distances.iter().enumerate() {
            out[i] = self.similarity(d);
        }
        // TODO: SIMD path with VGATHERDPS + VCVTPH2PS when the SIMD compat layer lands
    }

    /// Raw table access (for inspection/debugging).
    pub fn table(&self) -> &[f16; 256] {
        &self.table
    }

    /// Bucket width (distance units per bucket).
    pub fn bucket_width(&self) -> u32 {
        self.bucket_width
    }
}
```

## DELIVERABLE 2: Cascade::cdf_at() method (hdr.rs addition)

The reservoir has `quantile(q) -> distance` (percentile → distance).
We need the inverse: `cdf_at(distance) -> f32` (distance → percentile).

```rust
impl Cascade {
    /// Empirical CDF: fraction of reservoir samples ≤ distance.
    ///
    /// Returns a value in [0.0, 1.0] where:
    /// - 0.0 means no samples are this close (very unusual)
    /// - 0.5 means half the samples are closer (median distance)
    /// - 1.0 means all samples are closer (very far, noise)
    pub fn cdf_at(&self, distance: u32) -> f32 {
        if self.reservoir.is_empty() {
            // Fallback: use parametric normal CDF approximation
            let z = (distance as f32 - self.mu as f32) / self.sigma.max(1) as f32;
            return 1.0 / (1.0 + (-1.7 * z).exp()); // logistic CDF approx
        }
        // Count samples ≤ distance / total samples
        let count = self.reservoir.count_below(distance);
        count as f32 / self.reservoir.len() as f32
    }
}

impl ReservoirSample {
    /// Count samples ≤ threshold. Linear scan on sorted copy.
    /// Called once at calibration time, not on hot path.
    pub fn count_below(&self, threshold: u32) -> usize {
        self.samples.iter().filter(|&&s| s <= threshold).count()
    }
}
```

## DELIVERABLE 3: Similarity Methods on bgz17 Types

Wire SimilarityTable into Base17, PaletteEdge, and SpoBase17:

```rust
impl Base17 {
    /// Similarity score: f32 in [0.0, 1.0], calibrated from corpus distribution.
    /// Drop-in replacement for cosine_similarity on binary vectors.
    pub fn similarity(&self, other: &Base17, table: &SimilarityTable) -> f32 {
        table.similarity(self.l1(other))
    }
}

impl SpoBase17 {
    /// SPO similarity: average of per-plane similarities.
    pub fn similarity(&self, other: &SpoBase17, table: &SimilarityTable) -> f32 {
        table.similarity(self.l1(other))
    }

    /// Per-plane similarities: (subject, predicate, object).
    pub fn similarity_per_plane(
        &self, other: &SpoBase17, table: &SimilarityTable
    ) -> (f32, f32, f32) {
        let (ds, dp, do_) = self.l1_per_plane(other);
        (table.similarity(ds), table.similarity(dp), table.similarity(do_))
    }
}

impl PaletteEdge {
    /// Palette similarity: O(1) distance matrix lookup + table lookup.
    pub fn similarity(
        &self, other: &PaletteEdge,
        dm: &SpoDistanceMatrices, table: &SimilarityTable
    ) -> f32 {
        let dist = dm.spo_distance(
            self.s_idx, self.p_idx, self.o_idx,
            other.s_idx, other.p_idx, other.o_idx,
        );
        table.similarity(dist)
    }
}
```

## DELIVERABLE 4: Batch Similarity with SIMD (future, after SIMD compat layer)

When the SIMD compat layer lands:

```rust
impl SimilarityTable {
    /// Batch palette similarity: 16 lookups per AVX-512 instruction.
    ///
    /// Pipeline:
    /// 1. 16 palette distances via VGATHERDPS from distance_matrix (existing)
    /// 2. 16 bucket indices via integer divide (SIMD reciprocal multiply)
    /// 3. 16 f16 lookups via VGATHERDPS from similarity table
    /// 4. 16 f16→f32 via VCVTPH2PS (Rust 1.94 AVX-512 FP16, stable)
    ///
    /// Total: 4 SIMD instructions per 16 similarity scores.
    pub fn similarity_batch_simd(
        &self,
        distances: &[u32],
        out: &mut [f32],
    ) {
        // With crate::simd types:
        // let bucket_width = U32x16::splat(self.bucket_width);
        // for chunk in distances.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        //     let d = U32x16::from_slice(chunk.0);
        //     let buckets = d / bucket_width;  // SIMD integer divide
        //     let f16_vals = self.table.simd_gather(buckets);  // VGATHERDPS
        //     let f32_vals = f16_vals.to_f32();  // VCVTPH2PS
        //     f32_vals.copy_to_slice(chunk.1);
        // }
    }
}
```

## DELIVERABLE 5: Arrow-Compatible Similarity Column

For cold-path DataFusion queries, the similarity score is stored as Float16
in Lance (same column type as edge weight):

```rust
/// Compute similarity column for a batch of edges.
/// Stores as Float16 for Lance column storage (Arrow DataType::Float16).
/// Consumer reads as f32 via Arrow cast or hardware VCVTPH2PS.
pub fn compute_similarity_column(
    edges: &[SpoBase17],
    query: &SpoBase17,
    table: &SimilarityTable,
) -> Vec<f16> {
    edges.iter()
        .map(|e| f16::from_f32(e.similarity(query, table)))
        .collect()
}
```

This integrates with the existing `EdgeSchema` which already has `weight: Float16`.
The similarity score IS the weight — or a parallel column alongside it.

## Precision Hierarchy

```
LAYER         INPUT        LOOKUP         OUTPUT    RESOLUTION
──────        ─────        ──────         ──────    ──────────
Palette       3 bytes      2 lookups      f32       256 buckets (0.004 resolution)
              (s,p,o idx)  (dm + table)

Base17        102 bytes    1 lookup       f32       256 buckets (0.004 resolution)
              (L1 dist)    (table)

Full Hamming  2KB × 3      1 lookup       f32       256 buckets (0.004 resolution)
              (popcount)   (table)

ALL produce the same f32 similarity via the same 256-entry table.
The table is calibrated once from the Cascade's reservoir.
Palette is fastest (O(1) cache load), full Hamming is most accurate.
```

## Consumer Migration

```python
# BEFORE (cosine):
similarity = cosine_similarity(vec_a, vec_b)  # f32, [-1, 1] or [0, 1]
if similarity > 0.8:
    match()

# AFTER (bgz17):
similarity = palette_a.similarity(palette_b, &dm, &table)  # f32, [0, 1]
if similarity > 0.88:  # equivalent threshold (2σ above random)
    match()

# OR: expose a threshold translator
threshold_bgz17 = table.translate_cosine_threshold(0.8)  # → 0.88
```

## TESTS

1. SimilarityTable::from_cascade: table[0] ≈ 1.0, table[255] ≈ 0.0
2. Monotonicity: if distance_a < distance_b then similarity_a > similarity_b
3. Self-similarity: base17.similarity(&base17, table) ≈ 1.0 (distance 0)
4. Random pair: similarity ≈ 0.5 (near μ)
5. Band equivalence: Foveal distance → similarity > 0.95
6. f16 precision: |f32_from_f16 - f32_exact| < 0.002 across full range
7. Batch matches scalar: similarity_batch produces same results
8. cdf_at: cdf_at(0) ≈ 0.0, cdf_at(μ) ≈ 0.5, cdf_at(2μ) ≈ 1.0
9. Parametric fallback: from_cascade_parametric ≈ from_cascade (ρ > 0.99)

## OUTPUT

Branch: `feat/bgz17-similarity`
Files created: `crates/bgz17/src/similarity.rs`
Files modified: `crates/bgz17/src/lib.rs`, `crates/lance-graph/src/graph/blasgraph/hdr.rs`
Add to Session B remaining work.
