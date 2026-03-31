//! Palette distance matrix and search.
//!
//! Precomputed k x k distance matrix for O(1) palette distance lookups.
//! After building a palette, compute ALL pairwise L1 distances once.
//! Every subsequent distance lookup becomes a single u16 array load.
//!
//! Self-contained re-implementation of lance-graph's bgz17 palette and
//! distance_matrix modules for interoperability.

use super::bgz17_bridge::{Base17, PaletteEdge, SpoBase17};

const MAX_PALETTE_SIZE: usize = 256;
const BASE_DIM: usize = 17;

/// A palette codebook: up to 256 archetypal Base17 patterns.
#[derive(Clone, Debug)]
pub struct Palette {
    /// The archetype entries.
    pub entries: Vec<Base17>,
}

/// Precomputed pairwise distance matrix for one plane's palette.
///
/// `data[i * k + j]` = scaled L1 distance between palette entries i and j.
/// Symmetric: `data[i * k + j] == data[j * k + i]`.
/// Diagonal: `data[i * k + i] == 0`.
#[derive(Clone, Debug)]
pub struct DistanceMatrix {
    /// Flat storage: k x k u16 values.
    pub data: Vec<u16>,
    /// Palette size (k). `data.len() == k * k`.
    pub k: usize,
}

/// Three distance matrices: one per S/P/O plane.
#[derive(Clone, Debug)]
pub struct SpoDistanceMatrices {
    pub subject: DistanceMatrix,
    pub predicate: DistanceMatrix,
    pub object: DistanceMatrix,
}

impl Palette {
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the palette is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Byte size of the codebook.
    pub fn codebook_bytes(&self) -> usize {
        self.entries.len() * Base17::BYTE_SIZE
    }

    /// Find the nearest palette entry to a given base pattern. Returns index.
    pub fn nearest(&self, query: &Base17) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let d = query.l1(entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i as u8;
            }
        }
        best_idx
    }

    /// Encode an SpoBase17 edge to palette indices.
    pub fn encode_edge(&self, edge: &SpoBase17) -> PaletteEdge {
        PaletteEdge {
            s_idx: self.nearest(&edge.subject),
            p_idx: self.nearest(&edge.predicate),
            o_idx: self.nearest(&edge.object),
        }
    }

    /// Decode palette indices back to Base17 patterns (lossy).
    pub fn decode_edge(&self, pe: PaletteEdge) -> SpoBase17 {
        SpoBase17 {
            subject: self.entries[pe.s_idx as usize].clone(),
            predicate: self.entries[pe.p_idx as usize].clone(),
            object: self.entries[pe.o_idx as usize].clone(),
        }
    }

    /// Build a palette from a collection of Base17 patterns using k-means.
    ///
    /// `k`: target palette size (max 256).
    /// `max_iter`: k-means iterations (typically converges in 1-3).
    ///
    /// Initialization: k-means++ style (first pattern, then farthest-first).
    pub fn build(patterns: &[Base17], k: usize, max_iter: usize) -> Self {
        let k = k.min(MAX_PALETTE_SIZE).min(patterns.len());
        if k == 0 {
            return Palette { entries: Vec::new() };
        }

        // Initialize centroids: k-means++ style (first = first pattern, rest = farthest)
        let mut centroids: Vec<Base17> = Vec::with_capacity(k);
        centroids.push(patterns[0].clone());

        for _ in 1..k {
            let mut best_idx = 0;
            let mut best_dist = 0u64;
            for (i, p) in patterns.iter().enumerate() {
                let min_d: u32 = centroids.iter().map(|c| p.l1(c)).min().unwrap_or(u32::MAX);
                if min_d as u64 > best_dist {
                    best_dist = min_d as u64;
                    best_idx = i;
                }
            }
            centroids.push(patterns[best_idx].clone());
        }

        // K-means iterations
        for _iter in 0..max_iter {
            // Assign each pattern to nearest centroid
            let mut assignments = vec![0usize; patterns.len()];
            for (i, p) in patterns.iter().enumerate() {
                let mut best = 0;
                let mut best_d = u32::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = p.l1(centroid);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // Recompute centroids
            let mut new_centroids: Vec<[i64; BASE_DIM]> = vec![[0i64; BASE_DIM]; k];
            let mut counts = vec![0u32; k];

            for (i, p) in patterns.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for d in 0..BASE_DIM {
                    new_centroids[c][d] += p.dims[d] as i64;
                }
            }

            let mut changed = false;
            for c in 0..k {
                if counts[c] == 0 {
                    continue;
                }
                let mut new_dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    new_dims[d] = (new_centroids[c][d] / counts[c] as i64) as i16;
                }
                let new_base = Base17 { dims: new_dims };
                if new_base != centroids[c] {
                    changed = true;
                    centroids[c] = new_base;
                }
            }

            if !changed {
                break;
            }
        }

        Palette { entries: centroids }
    }

    /// Build three palettes (one per S/P/O plane) from a set of SpoBase17 edges.
    pub fn build_spo(edges: &[SpoBase17], k: usize, max_iter: usize) -> (Self, Self, Self) {
        let s_patterns: Vec<Base17> = edges.iter().map(|e| e.subject.clone()).collect();
        let p_patterns: Vec<Base17> = edges.iter().map(|e| e.predicate.clone()).collect();
        let o_patterns: Vec<Base17> = edges.iter().map(|e| e.object.clone()).collect();

        (
            Palette::build(&s_patterns, k, max_iter),
            Palette::build(&p_patterns, k, max_iter),
            Palette::build(&o_patterns, k, max_iter),
        )
    }
}

impl DistanceMatrix {
    /// Build from a palette. O(k^2) pairwise comparisons.
    ///
    /// Distances are scaled to u16 range: `d / max_l1 * 65535` where
    /// max_l1 = 17 * 65535 = 1,114,095.
    pub fn build(palette: &Palette) -> Self {
        let k = palette.len();
        let mut data = vec![0u16; k * k];

        for i in 0..k {
            for j in (i + 1)..k {
                let d = palette.entries[i].l1(&palette.entries[j]);
                let max_l1 = 17u64 * 65535;
                let scaled = ((d as u64 * 65535) / max_l1).min(65535) as u16;
                data[i * k + j] = scaled;
                data[j * k + i] = scaled;
            }
        }

        DistanceMatrix { data, k }
    }

    /// Look up distance between two palette indices. O(1).
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.data[a as usize * self.k + b as usize]
    }

    /// Byte size of the matrix.
    pub fn byte_size(&self) -> usize {
        self.k * self.k * 2
    }
}

impl SpoDistanceMatrices {
    /// Build from three palettes.
    pub fn build(s_pal: &Palette, p_pal: &Palette, o_pal: &Palette) -> Self {
        SpoDistanceMatrices {
            subject: DistanceMatrix::build(s_pal),
            predicate: DistanceMatrix::build(p_pal),
            object: DistanceMatrix::build(o_pal),
        }
    }

    /// Combined S+P+O distance from palette indices. O(1): 3 array loads.
    #[inline]
    pub fn spo_distance(&self, a_s: u8, a_p: u8, a_o: u8, b_s: u8, b_p: u8, b_o: u8) -> u32 {
        self.subject.distance(a_s, b_s) as u32
            + self.predicate.distance(a_p, b_p) as u32
            + self.object.distance(a_o, b_o) as u32
    }

    /// Total byte size of all three matrices.
    pub fn byte_size(&self) -> usize {
        self.subject.byte_size() + self.predicate.byte_size() + self.object.byte_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_patterns(n: usize) -> Vec<Base17> {
        (0..n)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    dims[d] = ((i * 7 + d * 13) % 256) as i16 - 128;
                }
                Base17 { dims }
            })
            .collect()
    }

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        Palette { entries }
    }

    #[test]
    fn test_build_palette() {
        let patterns = make_patterns(100);
        let palette = Palette::build(&patterns, 16, 10);
        assert_eq!(palette.len(), 16);
    }

    #[test]
    fn test_build_palette_empty() {
        let patterns: Vec<Base17> = Vec::new();
        let palette = Palette::build(&patterns, 16, 10);
        assert_eq!(palette.len(), 0);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_build_palette_k_larger_than_n() {
        let patterns = make_patterns(5);
        let palette = Palette::build(&patterns, 16, 10);
        assert_eq!(palette.len(), 5);
    }

    #[test]
    fn test_nearest_self() {
        let patterns = make_patterns(50);
        let palette = Palette::build(&patterns, 50, 1);
        for p in &patterns {
            let idx = palette.nearest(p);
            let dist = p.l1(&palette.entries[idx as usize]);
            assert!(dist < 1000, "nearest distance {} too large", dist);
        }
    }

    #[test]
    fn test_encode_decode() {
        let patterns = make_patterns(100);
        let palette = Palette::build(&patterns, 32, 5);
        let edge = SpoBase17 {
            subject: patterns[10].clone(),
            predicate: patterns[20].clone(),
            object: patterns[30].clone(),
        };
        let encoded = palette.encode_edge(&edge);
        let decoded = palette.decode_edge(encoded);
        assert!(edge.subject.l1(&decoded.subject) < 2000);
    }

    #[test]
    fn test_convergence() {
        let patterns = make_patterns(200);
        let p1 = Palette::build(&patterns, 32, 1);
        let p5 = Palette::build(&patterns, 32, 5);
        let p20 = Palette::build(&patterns, 32, 20);

        let total_dist = |pal: &Palette| -> u64 {
            patterns
                .iter()
                .map(|p| {
                    let idx = pal.nearest(p);
                    p.l1(&pal.entries[idx as usize]) as u64
                })
                .sum::<u64>()
        };

        let d1 = total_dist(&p1);
        let d5 = total_dist(&p5);
        let d20 = total_dist(&p20);
        assert!(d5 <= d1, "5 iters should be <= 1 iter: {} vs {}", d5, d1);
        assert!(d20 <= d5, "20 iters should be <= 5 iters: {} vs {}", d20, d5);
    }

    #[test]
    fn test_distance_self_zero() {
        let pal = make_palette(32);
        let dm = DistanceMatrix::build(&pal);
        for i in 0..32 {
            assert_eq!(dm.distance(i, i), 0, "self-distance must be 0 for entry {}", i);
        }
    }

    #[test]
    fn test_distance_symmetric() {
        let pal = make_palette(32);
        let dm = DistanceMatrix::build(&pal);
        for i in 0..32u8 {
            for j in 0..32u8 {
                assert_eq!(dm.distance(i, j), dm.distance(j, i));
            }
        }
    }

    #[test]
    fn test_spo_distance_self_zero() {
        let pal = make_palette(16);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);
        assert_eq!(spo.spo_distance(5, 5, 5, 5, 5, 5), 0);
    }

    #[test]
    fn test_cache_friendliness() {
        let pal = make_palette(256);
        let dm = DistanceMatrix::build(&pal);
        assert_eq!(dm.byte_size(), 256 * 256 * 2); // 128 KB
        assert!(dm.byte_size() <= 131072);
    }

    #[test]
    fn test_codebook_bytes() {
        let pal = make_palette(64);
        assert_eq!(pal.codebook_bytes(), 64 * 34);
    }

    #[test]
    fn test_spo_distance_triangle_inequality() {
        let pal = make_palette(16);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);
        // d(a, c) <= d(a, b) + d(b, c) for the scaled metric
        let d_ac = spo.spo_distance(0, 0, 0, 2, 2, 2);
        let d_ab = spo.spo_distance(0, 0, 0, 1, 1, 1);
        let d_bc = spo.spo_distance(1, 1, 1, 2, 2, 2);
        assert!(
            d_ac <= d_ab + d_bc,
            "triangle inequality violated: d(a,c)={} > d(a,b)={} + d(b,c)={}",
            d_ac, d_ab, d_bc
        );
    }

    #[test]
    fn test_build_spo() {
        let patterns = make_patterns(100);
        let edges: Vec<SpoBase17> = (0..30)
            .map(|i| SpoBase17 {
                subject: patterns[i].clone(),
                predicate: patterns[i + 30].clone(),
                object: patterns[i + 60].clone(),
            })
            .collect();
        let (s, p, o) = Palette::build_spo(&edges, 16, 5);
        assert_eq!(s.len(), 16);
        assert_eq!(p.len(), 16);
        assert_eq!(o.len(), 16);
    }

    #[test]
    fn test_spo_byte_size() {
        let pal = make_palette(32);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);
        assert_eq!(spo.byte_size(), 3 * 32 * 32 * 2);
    }

    #[test]
    fn test_4096_head_spo_throughput() {
        // Build 256-entry palette
        let pal = make_palette(256);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);

        // 4096 heads = 64×64, each with S/P/O palette index
        let mut heads_s = [0u8; 4096];
        let mut heads_p = [0u8; 4096];
        let mut heads_o = [0u8; 4096];
        for i in 0..4096 {
            heads_s[i] = (i % 256) as u8;
            heads_p[i] = ((i * 7) % 256) as u8;
            heads_o[i] = ((i * 13) % 256) as u8;
        }

        // Benchmark: 4096 × 64 SPO lookups (one row attending to 64 targets)
        let start = std::time::Instant::now();
        let mut total_dist = 0u64;
        let iterations = 100;
        for _ in 0..iterations {
            for row in 0..64 {
                for col in 0..64 {
                    let i = row * 64 + col;
                    for target in 0..64 {
                        let j = row * 64 + target;
                        total_dist += spo.spo_distance(
                            heads_s[i], heads_p[i], heads_o[i],
                            heads_s[j], heads_p[j], heads_o[j],
                        ) as u64;
                    }
                }
            }
        }
        let elapsed = start.elapsed();
        let total_lookups = 64u64 * 64 * 64 * iterations as u64;
        let lookups_per_sec = total_lookups as f64 / elapsed.as_secs_f64();
        let ns_per_lookup = elapsed.as_nanos() as f64 / total_lookups as f64;

        // Pearl 2³: multiply by 8 projections
        let pearl_ns = ns_per_lookup * 8.0 / 3.0; // each projection uses 1-3 planes
        let tokens_per_sec_spo = 1e9 / (ns_per_lookup * 64.0 * 64.0); // one token = full 64×64 pass
        let tokens_per_sec_pearl = 1e9 / (pearl_ns * 64.0 * 64.0);

        eprintln!();
        eprintln!("═══ Qwen3.5 + Opus 4.6: 4096-Head SPO Benchmark ═══");
        eprintln!("  Palette: 256 entries, SPO matrices: {} KB", spo.byte_size() / 1024);
        eprintln!("  Lookups: {} total ({} iterations × 64×64×64)", total_lookups, iterations);
        eprintln!("  Time:    {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        eprintln!("  Rate:    {:.0} M lookups/sec", lookups_per_sec / 1e6);
        eprintln!("  Latency: {:.1} ns/lookup (SPO, 3 planes)", ns_per_lookup);
        eprintln!("  Pearl:   {:.1} ns/lookup (8 projections avg)", pearl_ns);
        eprintln!();
        eprintln!("  Token throughput:");
        eprintln!("    SPO only:       {:.0} tokens/sec (64×64 attention per token)", tokens_per_sec_spo);
        eprintln!("    Pearl 2³:       {:.0} tokens/sec (8 projections per head)", tokens_per_sec_pearl);
        eprintln!("    Triple model:   {:.0} tokens/sec (self+user+impact)", tokens_per_sec_pearl / 3.0);
        eprintln!();
        eprintln!("  Memory: {} KB SPO tables + 4 KB head indices = {} KB total",
            spo.byte_size() / 1024, spo.byte_size() / 1024 + 4);
        eprintln!("  (blackhole: {})", total_dist); // prevent optimizer from eliding
    }
}
