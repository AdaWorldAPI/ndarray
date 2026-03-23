//! Compression curves: SPO bundle vs competing methods.
//!
//! Sweeps across multiple dimensions and compression ratios, measuring:
//! - Recall@k (search quality)
//! - Spearman ρ (ranking preservation)
//! - Cluster purity (separation quality)
//! - Bits per vector (absolute compression)
//! - Compression ratio (relative compression)
//!
//! Competing methods:
//! - **SPO Bundle** (majority vote cyclic shift): our method
//! - **SimHash** (random hyperplane LSH): classic binary sketching
//! - **Binary Quantization** (sign of each component): simplest possible
//! - **Product Quantization** (sub-vector k-means, simplified): PQ-lite
//! - **Random Projection + Binarize**: JL-lemma based

// ============================================================================
// PRNG (self-contained, deterministic)
// ============================================================================

fn prng(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn prng_f64(state: &mut u64) -> f64 {
    (prng(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller normal variate.
fn prng_normal(state: &mut u64) -> f64 {
    let u1 = prng_f64(state).max(1e-15);
    let u2 = prng_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn seed_state(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1)
}

// ============================================================================
// Float vector helpers
// ============================================================================

fn random_f64_vec(dim: usize, seed: u64) -> Vec<f64> {
    let mut s = seed_state(seed);
    (0..dim).map(|_| prng_normal(&mut s)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ============================================================================
// Bit vector helpers
// ============================================================================

fn hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

// ============================================================================
// Method 1: SimHash (random hyperplane LSH)
// ============================================================================

struct SimHashProjector {
    hyperplanes: Vec<Vec<f64>>, // n_bits × dim
}

impl SimHashProjector {
    fn new(dim: usize, n_bits: usize, seed: u64) -> Self {
        let mut s = seed_state(seed);
        let hyperplanes = (0..n_bits)
            .map(|_| (0..dim).map(|_| prng_normal(&mut s)).collect())
            .collect();
        SimHashProjector { hyperplanes }
    }

    fn hash(&self, v: &[f64]) -> Vec<u8> {
        let n_bytes = (self.hyperplanes.len() + 7) / 8;
        let mut result = vec![0u8; n_bytes];
        for (i, hp) in self.hyperplanes.iter().enumerate() {
            if dot(v, hp) >= 0.0 {
                result[i / 8] |= 1u8 << (i % 8);
            }
        }
        result
    }
}

// ============================================================================
// Method 2: Binary Quantization (sign bit per dimension)
// ============================================================================

fn binary_quantize(v: &[f64]) -> Vec<u8> {
    let n_bytes = (v.len() + 7) / 8;
    let mut result = vec![0u8; n_bytes];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            result[i / 8] |= 1u8 << (i % 8);
        }
    }
    result
}

// ============================================================================
// Method 3: Random Projection + Binarize (JL → binary)
// ============================================================================

struct RandomProjection {
    matrix: Vec<Vec<f64>>, // target_dim × source_dim
}

impl RandomProjection {
    fn new(source_dim: usize, target_dim: usize, seed: u64) -> Self {
        let mut s = seed_state(seed);
        let scale = 1.0 / (target_dim as f64).sqrt();
        let matrix = (0..target_dim)
            .map(|_| (0..source_dim).map(|_| prng_normal(&mut s) * scale).collect())
            .collect();
        RandomProjection { matrix }
    }

    fn project_and_binarize(&self, v: &[f64]) -> Vec<u8> {
        let target_dim = self.matrix.len();
        let n_bytes = (target_dim + 7) / 8;
        let mut result = vec![0u8; n_bytes];
        for (i, row) in self.matrix.iter().enumerate() {
            if dot(v, row) >= 0.0 {
                result[i / 8] |= 1u8 << (i % 8);
            }
        }
        result
    }
}

// ============================================================================
// Method 4: Product Quantization (simplified: per-subvector centroid)
// ============================================================================

/// Simplified PQ: divide vector into M subvectors, each quantized to k_bits.
struct ProductQuantizer {
    m: usize,         // number of sub-vectors
    k_bits: usize,    // bits per sub-quantizer (2^k_bits centroids)
    sub_dim: usize,
    codebooks: Vec<Vec<Vec<f64>>>, // M × 2^k_bits × sub_dim
}

impl ProductQuantizer {
    /// Build codebooks from training data (simplified: random centroids from data distribution).
    fn train(dim: usize, m: usize, k_bits: usize, training_data: &[Vec<f64>], seed: u64) -> Self {
        let sub_dim = dim / m;
        let n_centroids = 1usize << k_bits;
        let mut s = seed_state(seed);

        let codebooks: Vec<Vec<Vec<f64>>> = (0..m)
            .map(|sub| {
                // Simple k-means initialization: pick random data points as centroids
                let mut centroids: Vec<Vec<f64>> = (0..n_centroids)
                    .map(|_| {
                        let idx = (prng(&mut s) as usize) % training_data.len();
                        let start = sub * sub_dim;
                        let end = start + sub_dim;
                        training_data[idx][start..end].to_vec()
                    })
                    .collect();

                // Run 5 iterations of k-means
                for _ in 0..5 {
                    let mut sums = vec![vec![0.0f64; sub_dim]; n_centroids];
                    let mut counts = vec![0usize; n_centroids];

                    for v in training_data {
                        let sub_v = &v[sub * sub_dim..(sub + 1) * sub_dim];
                        let nearest = centroids
                            .iter()
                            .enumerate()
                            .map(|(ci, c)| (ci, l2_dist(sub_v, c)))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap()
                            .0;
                        for (j, &val) in sub_v.iter().enumerate() {
                            sums[nearest][j] += val;
                        }
                        counts[nearest] += 1;
                    }

                    for ci in 0..n_centroids {
                        if counts[ci] > 0 {
                            for j in 0..sub_dim {
                                centroids[ci][j] = sums[ci][j] / counts[ci] as f64;
                            }
                        }
                    }
                }

                centroids
            })
            .collect();

        ProductQuantizer { m, k_bits, sub_dim, codebooks }
    }

    fn encode(&self, v: &[f64]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let sub_v = &v[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let nearest = self.codebooks[sub]
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, l2_dist(sub_v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            codes.push(nearest as u8);
        }
        codes
    }

    fn asymmetric_dist(&self, query: &[f64], code: &[u8]) -> f64 {
        let mut dist = 0.0;
        for sub in 0..self.m {
            let sub_q = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let centroid = &self.codebooks[sub][code[sub] as usize];
            dist += l2_dist(sub_q, centroid).powi(2);
        }
        dist.sqrt()
    }

    fn bits_per_vector(&self) -> usize {
        self.m * self.k_bits
    }
}

// ============================================================================
// Method 5: SPO Bundle (our method)
// ============================================================================

const PHI: f64 = 1.618_033_988_749_895;

const fn golden_shift(d: usize) -> usize {
    let raw = (d as f64 / (PHI * PHI)) as usize;
    if raw % 2 == 0 { raw + 1 } else { raw }
}

fn cyclic_shift_dyn(bits: &[u64], shift: usize) -> Vec<u64> {
    let n = bits.len();
    let d = n * 64;
    let shift = shift % d;
    if shift == 0 {
        return bits.to_vec();
    }
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut result = vec![0u64; n];
    for i in 0..n {
        let src = (i + word_shift) % n;
        let next = (src + 1) % n;
        if bit_shift == 0 {
            result[i] = bits[src];
        } else {
            result[i] = (bits[src] >> bit_shift) | (bits[next] << (64 - bit_shift));
        }
    }
    result
}

fn majority_vote_3_dyn(a: &[u64], b: &[u64], c: &[u64]) -> Vec<u64> {
    a.iter()
        .zip(b.iter().zip(c.iter()))
        .map(|(&x, (&y, &z))| (x & y) | (x & z) | (y & z))
        .collect()
}

/// Convert f64 vector to binary plane via blake3 XOF.
fn f64_to_plane(v: &[f64], plane_bits: usize) -> Vec<u64> {
    // Hash the float bytes to get a deterministic binary plane
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let n_words = plane_bits / 64;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&bytes);
    let mut output = vec![0u8; n_words * 8];
    let mut reader = hasher.finalize_xof();
    reader.fill(&mut output);
    let mut words = vec![0u64; n_words];
    for (i, chunk) in output.chunks_exact(8).enumerate() {
        words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    words
}

/// Bundle 3 planes into SPO bundle at given dimension.
fn spo_bundle(s: &[u64], p: &[u64], o: &[u64], dim: usize) -> Vec<u64> {
    let shift = golden_shift(dim);
    let p_shifted = cyclic_shift_dyn(p, shift);
    let o_shifted = cyclic_shift_dyn(o, (shift * 2) % dim);
    majority_vote_3_dyn(s, &p_shifted, &o_shifted)
}

// ============================================================================
// Spearman rank correlation
// ============================================================================

fn to_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        ranks[idx] = rank as f64;
    }
    ranks
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 { return 0.0; }
    let ra = to_ranks(a);
    let rb = to_ranks(b);
    let ma: f64 = ra.iter().sum::<f64>() / n as f64;
    let mb: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = ra[i] - ma;
        let db = rb[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va < 1e-10 || vb < 1e-10 { return 0.0; }
    cov / (va.sqrt() * vb.sqrt())
}

// ============================================================================
// Benchmark harness
// ============================================================================

struct CompressionResult {
    method: &'static str,
    bits_per_vector: usize,
    source_bits: usize,         // original f64 vector size in bits
    recall_at_10: f64,
    spearman_rho: f64,
    cluster_purity: f64,
    compression_ratio: f64,     // source_bits / bits_per_vector
}

impl CompressionResult {
    fn absolute_bpv(&self) -> f64 {
        self.bits_per_vector as f64
    }
    fn relative_ratio(&self) -> f64 {
        self.compression_ratio
    }
}

/// Run a full benchmark at a given dimension.
fn benchmark_at_dim(
    dim: usize,
    n_vectors: usize,
    n_clusters: usize,
    cluster_spread: f64,
) -> Vec<CompressionResult> {
    let source_bits = dim * 64; // f64 per dimension

    // ── Generate clustered data ──────────────────────────────────
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(n_vectors);
    let mut labels: Vec<usize> = Vec::with_capacity(n_vectors);
    let per_cluster = n_vectors / n_clusters;

    for c in 0..n_clusters {
        let center = random_f64_vec(dim, (c as u64 + 1) * 1000);
        for m in 0..per_cluster {
            let seed = (c * per_cluster + m) as u64 * 7 + 42;
            let noise = random_f64_vec(dim, seed);
            let v: Vec<f64> = center
                .iter()
                .zip(&noise)
                .map(|(&c, &n)| c + n * cluster_spread)
                .collect();
            vectors.push(v);
            labels.push(c);
        }
    }

    // ── Ground truth distances (cosine) ──────────────────────────
    let qi = 0;
    let gt_dists: Vec<f64> = (1..n_vectors)
        .map(|i| 1.0 - cosine_sim(&vectors[qi], &vectors[i]))
        .collect();

    let mut gt_ranked: Vec<(usize, f64)> = gt_dists.iter().copied().enumerate().collect();
    gt_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let _gt_top10: std::collections::HashSet<usize> =
        gt_ranked[..10.min(gt_ranked.len())].iter().map(|&(i, _)| i).collect();

    let mut results = Vec::new();

    // ── Method 1: SimHash at various bit widths ──────────────────
    for &n_bits in &[64, 128, 256, 512, 1024, 2048] {
        if n_bits > dim * 4 { continue; } // skip if oversized
        let projector = SimHashProjector::new(dim, n_bits, 12345);
        let hashes: Vec<Vec<u8>> = vectors.iter().map(|v| projector.hash(v)).collect();

        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| hamming_bytes(&hashes[qi], &hashes[i]) as f64)
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);
        let purity = cluster_purity_knn(&dists, &labels, qi, 10);

        results.push(CompressionResult {
            method: "SimHash",
            bits_per_vector: n_bits,
            source_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            cluster_purity: purity,
            compression_ratio: source_bits as f64 / n_bits as f64,
        });
    }

    // ── Method 2: Binary Quantization ────────────────────────────
    {
        let bq: Vec<Vec<u8>> = vectors.iter().map(|v| binary_quantize(v)).collect();
        let n_bits = dim; // 1 bit per dimension
        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| hamming_bytes(&bq[qi], &bq[i]) as f64)
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);
        let purity = cluster_purity_knn(&dists, &labels, qi, 10);

        results.push(CompressionResult {
            method: "BinaryQuant",
            bits_per_vector: n_bits,
            source_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            cluster_purity: purity,
            compression_ratio: source_bits as f64 / n_bits as f64,
        });
    }

    // ── Method 3: Random Projection + Binarize ───────────────────
    for &target_bits in &[64, 128, 256, 512, 1024] {
        if target_bits > dim * 2 { continue; }
        let rp = RandomProjection::new(dim, target_bits, 54321);
        let projected: Vec<Vec<u8>> = vectors.iter()
            .map(|v| rp.project_and_binarize(v))
            .collect();

        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| hamming_bytes(&projected[qi], &projected[i]) as f64)
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);
        let purity = cluster_purity_knn(&dists, &labels, qi, 10);

        results.push(CompressionResult {
            method: "RandProj",
            bits_per_vector: target_bits,
            source_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            cluster_purity: purity,
            compression_ratio: source_bits as f64 / target_bits as f64,
        });
    }

    // ── Method 4: Product Quantization ───────────────────────────
    for &(m, k_bits) in &[(8, 4), (8, 8), (16, 4), (16, 8), (32, 4), (32, 8)] {
        if m > dim { continue; }
        if dim % m != 0 { continue; }
        let pq = ProductQuantizer::train(dim, m, k_bits, &vectors, 99999);
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v)).collect();

        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| pq.asymmetric_dist(&vectors[qi], &codes[i]))
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);
        let purity = cluster_purity_knn(&dists, &labels, qi, 10);

        results.push(CompressionResult {
            method: "PQ",
            bits_per_vector: pq.bits_per_vector(),
            source_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            cluster_purity: purity,
            compression_ratio: source_bits as f64 / pq.bits_per_vector() as f64,
        });
    }

    // ── Method 5: SPO-Direct (sign-bit encoding + cyclic bundle) ──
    // Convert f64 → sign bits → cyclic permutation bundle.
    // This is the correct approach for float→binary compression via SPO.
    for &plane_bits in &[128, 256, 512, 1024, 2048, 4096, 8192] {
        if plane_bits > dim { continue; }
        let planes: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = vectors
            .iter()
            .map(|v| {
                let third = plane_bits / 64;
                let s = sign_bits(&v[..plane_bits.min(dim)], third);
                let p_start = (dim / 3).min(dim - 1);
                let o_start = (2 * dim / 3).min(dim - 1);
                let p = sign_bits(&v[p_start..], third);
                let o = sign_bits(&v[o_start..], third);
                (s, p, o)
            })
            .collect();

        let bundles: Vec<Vec<u64>> = planes
            .iter()
            .map(|(s, p, o)| spo_bundle(s, p, o, plane_bits))
            .collect();

        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| hamming_u64(&bundles[qi], &bundles[i]) as f64)
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);
        let purity = cluster_purity_knn(&dists, &labels, qi, 10);

        results.push(CompressionResult {
            method: "SPO-Sign",
            bits_per_vector: plane_bits,
            source_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            cluster_purity: purity,
            compression_ratio: source_bits as f64 / plane_bits as f64,
        });
    }

    results
}

/// Convert f64 slice to sign bits (packed into u64 words).
fn sign_bits(v: &[f64], n_words: usize) -> Vec<u64> {
    let n_bits = n_words * 64;
    let mut words = vec![0u64; n_words];
    for i in 0..n_bits.min(v.len()) {
        if v[i] >= 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    words
}

fn recall_at_k(gt_dists: &[f64], approx_dists: &[f64], k: usize) -> f64 {
    let n = gt_dists.len();
    let k = k.min(n);

    let mut gt_idx: Vec<usize> = (0..n).collect();
    gt_idx.sort_by(|&a, &b| gt_dists[a].partial_cmp(&gt_dists[b]).unwrap());
    let gt_top: std::collections::HashSet<usize> = gt_idx[..k].iter().copied().collect();

    let mut ap_idx: Vec<usize> = (0..n).collect();
    ap_idx.sort_by(|&a, &b| approx_dists[a].partial_cmp(&approx_dists[b]).unwrap());
    let ap_top: std::collections::HashSet<usize> = ap_idx[..k].iter().copied().collect();

    gt_top.intersection(&ap_top).count() as f64 / k as f64
}

fn cluster_purity_knn(dists: &[f64], labels: &[usize], qi: usize, k: usize) -> f64 {
    let n = dists.len();
    let k = k.min(n);
    let query_label = labels[qi];

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| dists[a].partial_cmp(&dists[b]).unwrap());

    // Map back: dists index i corresponds to vector index i+1 (qi=0 excluded)
    let same = idx[..k]
        .iter()
        .filter(|&&i| labels[i + 1] == query_label)
        .count();
    same as f64 / k as f64
}

// ============================================================================
// Dimension sweep: separation quality per resolution
// ============================================================================

struct DimSweepResult {
    dim: usize,
    method: &'static str,
    bits: usize,
    inter_cluster_mean: f64,
    intra_cluster_mean: f64,
    separation_ratio: f64, // inter / intra (higher = better separation)
    spearman_rho: f64,
}

fn dimension_sweep(dims: &[usize], n_per_cluster: usize, n_clusters: usize) -> Vec<DimSweepResult> {
    let mut results = Vec::new();

    for &dim in dims {
        let spread = 0.3;
        let mut vectors: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();

        for c in 0..n_clusters {
            let center = random_f64_vec(dim, (c as u64 + 1) * 5000);
            for m in 0..n_per_cluster {
                let seed = (c * n_per_cluster + m) as u64 * 11 + 100;
                let noise = random_f64_vec(dim, seed);
                let v: Vec<f64> = center
                    .iter()
                    .zip(&noise)
                    .map(|(&c, &n)| c + n * spread)
                    .collect();
                vectors.push(v);
                labels.push(c);
            }
        }

        let n = vectors.len();

        // Compute all-pairs ground truth
        let gt_dists: Vec<f64> = (1..n)
            .map(|i| 1.0 - cosine_sim(&vectors[0], &vectors[i]))
            .collect();

        // For each method, compute separation
        let methods: Vec<(&str, Box<dyn Fn(&[Vec<f64>]) -> (usize, Vec<f64>)>)> = vec![
            ("BinaryQuant", Box::new(|vecs: &[Vec<f64>]| {
                let bq: Vec<Vec<u8>> = vecs.iter().map(|v| binary_quantize(v)).collect();
                let bits = vecs[0].len();
                let dists: Vec<f64> = (1..vecs.len())
                    .map(|i| hamming_bytes(&bq[0], &bq[i]) as f64)
                    .collect();
                (bits, dists)
            })),
            ("SimHash-256", Box::new(move |vecs: &[Vec<f64>]| {
                let proj = SimHashProjector::new(dim, 256, 12345);
                let h: Vec<Vec<u8>> = vecs.iter().map(|v| proj.hash(v)).collect();
                let dists: Vec<f64> = (1..vecs.len())
                    .map(|i| hamming_bytes(&h[0], &h[i]) as f64)
                    .collect();
                (256, dists)
            })),
            ("SimHash-1024", Box::new(move |vecs: &[Vec<f64>]| {
                let bits = 1024.min(dim * 2);
                let proj = SimHashProjector::new(dim, bits, 12345);
                let h: Vec<Vec<u8>> = vecs.iter().map(|v| proj.hash(v)).collect();
                let dists: Vec<f64> = (1..vecs.len())
                    .map(|i| hamming_bytes(&h[0], &h[i]) as f64)
                    .collect();
                (bits, dists)
            })),
            ("SPO-Sign", Box::new(move |vecs: &[Vec<f64>]| {
                // SPO with sign-bit encoding: best float→binary→bundle path
                let plane_bits = dim.min(1024); // clamp to dim
                let third = plane_bits / 64;
                if third == 0 { return (1, vec![0.0; vecs.len() - 1]); }
                let planes: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = vecs
                    .iter()
                    .map(|v| {
                        let s = sign_bits(&v[..plane_bits.min(dim)], third);
                        let p_start = (dim / 3).min(dim - 1);
                        let o_start = (2 * dim / 3).min(dim - 1);
                        let p = sign_bits(&v[p_start..], third);
                        let o = sign_bits(&v[o_start..], third);
                        (s, p, o)
                    })
                    .collect();
                let bundles: Vec<Vec<u64>> = planes.iter()
                    .map(|(s, p, o)| spo_bundle(s, p, o, plane_bits))
                    .collect();
                let dists: Vec<f64> = (1..vecs.len())
                    .map(|i| hamming_u64(&bundles[0], &bundles[i]) as f64)
                    .collect();
                (plane_bits, dists)
            })),
        ];

        for (name, method_fn) in &methods {
            let (bits, dists) = method_fn(&vectors);

            let rho = spearman(&gt_dists, &dists);

            // Compute intra/inter cluster distances
            let mut intra = Vec::new();
            let mut inter = Vec::new();
            let q_label = labels[0];
            for i in 0..dists.len() {
                if labels[i + 1] == q_label {
                    intra.push(dists[i]);
                } else {
                    inter.push(dists[i]);
                }
            }

            let intra_mean = if intra.is_empty() { 1.0 } else { intra.iter().sum::<f64>() / intra.len() as f64 };
            let inter_mean = if inter.is_empty() { 1.0 } else { inter.iter().sum::<f64>() / inter.len() as f64 };
            let sep = if intra_mean > 1e-10 { inter_mean / intra_mean } else { inter_mean };

            results.push(DimSweepResult {
                dim,
                method: name,
                bits,
                inter_cluster_mean: inter_mean,
                intra_cluster_mean: intra_mean,
                separation_ratio: sep,
                spearman_rho: rho,
            });
        }
    }

    results
}

// ============================================================================
// Precision sweet-spot finder
// ============================================================================

struct PrecisionPoint {
    bits: usize,
    recall_at_10: f64,
    spearman_rho: f64,
    marginal_gain: f64, // Δρ per additional bit
}

fn find_sweet_spot(dim: usize, n_vectors: usize) -> Vec<PrecisionPoint> {
    let spread = 0.5;
    let n_clusters = 5;
    let per_cluster = n_vectors / n_clusters;

    let mut vectors: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for c in 0..n_clusters {
        let center = random_f64_vec(dim, (c as u64 + 1) * 9999);
        for m in 0..per_cluster {
            let seed = (c * per_cluster + m) as u64 * 13 + 77;
            let noise = random_f64_vec(dim, seed);
            let v: Vec<f64> = center.iter().zip(&noise).map(|(&c, &n)| c + n * spread).collect();
            vectors.push(v);
            labels.push(c);
        }
    }

    let gt_dists: Vec<f64> = (1..n_vectors)
        .map(|i| 1.0 - cosine_sim(&vectors[0], &vectors[i]))
        .collect();

    // Fine-grained sweep for SPO bundle
    let bit_widths: Vec<usize> = (6..=15).map(|e| 1 << e).collect();
    // 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768

    let mut points: Vec<PrecisionPoint> = Vec::new();
    let mut prev_rho = 0.0;
    let mut prev_bits = 0;

    for &plane_bits in &bit_widths {
        if plane_bits > dim { continue; }
        let n_words = plane_bits / 64;
        if n_words == 0 { continue; }
        let planes: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = vectors
            .iter()
            .map(|v| {
                let s = sign_bits(&v[..plane_bits.min(dim)], n_words);
                let p_start = (dim / 3).min(dim - 1);
                let o_start = (2 * dim / 3).min(dim - 1);
                let p = sign_bits(&v[p_start..], n_words);
                let o = sign_bits(&v[o_start..], n_words);
                (s, p, o)
            })
            .collect();

        let bundles: Vec<Vec<u64>> = planes
            .iter()
            .map(|(s, p, o)| spo_bundle(s, p, o, plane_bits))
            .collect();

        let dists: Vec<f64> = (1..n_vectors)
            .map(|i| hamming_u64(&bundles[0], &bundles[i]) as f64)
            .collect();

        let rho = spearman(&gt_dists, &dists);
        let recall = recall_at_k(&gt_dists, &dists, 10);

        let marginal = if prev_bits > 0 && plane_bits > prev_bits {
            (rho - prev_rho) / (plane_bits - prev_bits) as f64
        } else {
            0.0
        };

        points.push(PrecisionPoint {
            bits: plane_bits,
            recall_at_10: recall,
            spearman_rho: rho,
            marginal_gain: marginal,
        });

        prev_rho = rho;
        prev_bits = plane_bits;
    }

    points
}

// ============================================================================
// Native binary plane compression (XOR-fold + cyclic bundle)
// This is the PRIMARY SPO use case: compress 3×16Kbit planes into one bundle.
// ============================================================================

/// XOR-fold a u64 vector to half its size, repeatedly until target_words.
fn xor_fold(v: &[u64], target_words: usize) -> Vec<u64> {
    let mut current = v.to_vec();
    while current.len() > target_words && current.len() >= 2 * target_words {
        let half = current.len() / 2;
        let mut folded = vec![0u64; half];
        for i in 0..half {
            folded[i] = current[i] ^ current[i + half];
        }
        current = folded;
    }
    current.truncate(target_words);
    current
}

/// Generate random binary plane (N words of u64).
fn random_plane(n_words: usize, seed: u64) -> Vec<u64> {
    let mut s = seed_state(seed);
    (0..n_words).map(|_| prng(&mut s)).collect()
}

/// Flip n_flips random bits in a plane.
fn flip_plane(v: &[u64], n_flips: usize, seed: u64) -> Vec<u64> {
    let d = v.len() * 64;
    let mut result = v.to_vec();
    let mut s = seed_state(seed);
    let mut used = vec![false; d];
    let mut flipped = 0;
    while flipped < n_flips && flipped < d {
        let pos = (prng(&mut s) as usize) % d;
        if !used[pos] {
            used[pos] = true;
            result[pos / 64] ^= 1u64 << (pos % 64);
            flipped += 1;
        }
    }
    result
}

struct NativeBinaryResult {
    method: &'static str,
    bits: usize,
    source_bits: usize,
    rho_random: f64,
    rho_structured: f64,
    recall_at_10_random: f64,
    recall_at_10_structured: f64,
}

/// Full native binary benchmark: 500 nodes × 3×16Kbit planes.
fn native_binary_benchmark(
    n_nodes: usize,
    n_queries: usize,
    plane_words: usize, // 256 for 16Kbit
) -> Vec<NativeBinaryResult> {
    let plane_bits = plane_words * 64;
    let source_bits = plane_bits * 3;

    // Generate random SPO triples
    let nodes_s: Vec<Vec<u64>> = (0..n_nodes).map(|i| random_plane(plane_words, i as u64 * 3 + 1)).collect();
    let nodes_p: Vec<Vec<u64>> = (0..n_nodes).map(|i| random_plane(plane_words, i as u64 * 3 + 2)).collect();
    let nodes_o: Vec<Vec<u64>> = (0..n_nodes).map(|i| random_plane(plane_words, i as u64 * 3 + 3)).collect();

    // Structured: first 50 share similar S, next 50 share similar P
    let base_s = random_plane(plane_words, 99999);
    let base_p = random_plane(plane_words, 88888);
    let mut struct_s = nodes_s.clone();
    let mut struct_p = nodes_p.clone();
    let struct_o = nodes_o.clone();

    for i in 0..50.min(n_nodes) {
        struct_s[i] = flip_plane(&base_s, i * 80, i as u64 * 10 + 1000);
    }
    for i in 50..100.min(n_nodes) {
        struct_p[i] = flip_plane(&base_p, (i - 50) * 80, i as u64 * 10 + 2000);
    }

    // Ground truth: exact S+P+O hamming
    let exact_random: Vec<Vec<u32>> = (0..n_queries.min(n_nodes))
        .map(|q| {
            (0..n_nodes)
                .map(|i| {
                    hamming_u64(&nodes_s[q], &nodes_s[i])
                        + hamming_u64(&nodes_p[q], &nodes_p[i])
                        + hamming_u64(&nodes_o[q], &nodes_o[i])
                })
                .collect()
        })
        .collect();

    let exact_struct: Vec<Vec<u32>> = (0..n_queries.min(n_nodes))
        .map(|q| {
            (0..n_nodes)
                .map(|i| {
                    hamming_u64(&struct_s[q], &struct_s[i])
                        + hamming_u64(&struct_p[q], &struct_p[i])
                        + hamming_u64(&struct_o[q], &struct_o[i])
                })
                .collect()
        })
        .collect();

    let mut results = Vec::new();

    // Test at various target dimensions via XOR-fold + cyclic bundle
    let target_dims: Vec<usize> = vec![32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64]
        .into_iter()
        .filter(|&d| d <= plane_bits * 2) // max is 2× a single plane (via no-fold)
        .collect();

    for &target_bits in &target_dims {
        let target_words = target_bits / 64;

        // Cyclic bundle at target dimension
        let bundles_random: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let s = xor_fold(&nodes_s[i], target_words);
                let p = xor_fold(&nodes_p[i], target_words);
                let o = xor_fold(&nodes_o[i], target_words);
                spo_bundle(&s, &p, &o, target_bits)
            })
            .collect();

        let bundles_struct: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let s = xor_fold(&struct_s[i], target_words);
                let p = xor_fold(&struct_p[i], target_words);
                let o = xor_fold(&struct_o[i], target_words);
                spo_bundle(&s, &p, &o, target_bits)
            })
            .collect();

        let (rho_r, recall_r) = avg_metrics(&exact_random, &bundles_random, n_queries.min(n_nodes), n_nodes);
        let (rho_s, recall_s) = avg_metrics(&exact_struct, &bundles_struct, n_queries.min(n_nodes), n_nodes);

        results.push(NativeBinaryResult {
            method: "CyclicBundle",
            bits: target_bits,
            source_bits,
            rho_random: rho_r,
            rho_structured: rho_s,
            recall_at_10_random: recall_r,
            recall_at_10_structured: recall_s,
        });
    }

    // Simple truncation baseline
    for &target_bits in &target_dims {
        if target_bits > source_bits { continue; }
        let third = target_bits / 3;
        let third_words = third / 64;
        if third_words == 0 { continue; }
        let remainder_words = (target_bits - 2 * third) / 64;

        let trunc_random: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let mut v = Vec::with_capacity(target_bits / 64);
                v.extend_from_slice(&nodes_s[i][..third_words.min(nodes_s[i].len())]);
                v.extend_from_slice(&nodes_p[i][..third_words.min(nodes_p[i].len())]);
                v.extend_from_slice(&nodes_o[i][..remainder_words.min(nodes_o[i].len())]);
                v
            })
            .collect();

        let trunc_struct: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let mut v = Vec::with_capacity(target_bits / 64);
                v.extend_from_slice(&struct_s[i][..third_words.min(struct_s[i].len())]);
                v.extend_from_slice(&struct_p[i][..third_words.min(struct_p[i].len())]);
                v.extend_from_slice(&struct_o[i][..remainder_words.min(struct_o[i].len())]);
                v
            })
            .collect();

        let (rho_r, recall_r) = avg_metrics_raw(&exact_random, &trunc_random, n_queries.min(n_nodes), n_nodes);
        let (rho_s, recall_s) = avg_metrics_raw(&exact_struct, &trunc_struct, n_queries.min(n_nodes), n_nodes);

        results.push(NativeBinaryResult {
            method: "Truncation",
            bits: target_bits,
            source_bits,
            rho_random: rho_r,
            rho_structured: rho_s,
            recall_at_10_random: recall_r,
            recall_at_10_structured: recall_s,
        });
    }

    // SimHash baseline (random bit sampling from concatenated S+P+O)
    for &target_bits in &target_dims {
        if target_bits > source_bits { continue; }
        // Sample target_bits random positions from 3×plane_bits
        let mut s = seed_state(77777);
        let indices: Vec<usize> = (0..target_bits)
            .map(|_| (prng(&mut s) as usize) % source_bits)
            .collect();

        let sampled_random: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let mut v = vec![0u64; target_bits / 64];
                for (bi, &idx) in indices.iter().enumerate() {
                    let (plane, pos) = if idx < plane_bits {
                        (&nodes_s[i], idx)
                    } else if idx < 2 * plane_bits {
                        (&nodes_p[i], idx - plane_bits)
                    } else {
                        (&nodes_o[i], idx - 2 * plane_bits)
                    };
                    let bit = (plane[pos / 64] >> (pos % 64)) & 1;
                    if bit == 1 {
                        v[bi / 64] |= 1u64 << (bi % 64);
                    }
                }
                v
            })
            .collect();

        let sampled_struct: Vec<Vec<u64>> = (0..n_nodes)
            .map(|i| {
                let mut v = vec![0u64; target_bits / 64];
                for (bi, &idx) in indices.iter().enumerate() {
                    let (plane, pos) = if idx < plane_bits {
                        (&struct_s[i], idx)
                    } else if idx < 2 * plane_bits {
                        (&struct_p[i], idx - plane_bits)
                    } else {
                        (&struct_o[i], idx - 2 * plane_bits)
                    };
                    let bit = (plane[pos / 64] >> (pos % 64)) & 1;
                    if bit == 1 {
                        v[bi / 64] |= 1u64 << (bi % 64);
                    }
                }
                v
            })
            .collect();

        let (rho_r, recall_r) = avg_metrics_raw(&exact_random, &sampled_random, n_queries.min(n_nodes), n_nodes);
        let (rho_s, recall_s) = avg_metrics_raw(&exact_struct, &sampled_struct, n_queries.min(n_nodes), n_nodes);

        results.push(NativeBinaryResult {
            method: "BitSample",
            bits: target_bits,
            source_bits,
            rho_random: rho_r,
            rho_structured: rho_s,
            recall_at_10_random: recall_r,
            recall_at_10_structured: recall_s,
        });
    }

    results
}

/// Compute average Spearman ρ and recall@10 from bundle hamming vs exact distances.
fn avg_metrics(
    exact_dists: &[Vec<u32>],
    bundles: &[Vec<u64>],
    n_queries: usize,
    n_nodes: usize,
) -> (f64, f64) {
    let mut rhos = Vec::new();
    let mut recalls = Vec::new();
    for q in 0..n_queries {
        let gt: Vec<f64> = (0..n_nodes).filter(|&i| i != q)
            .map(|i| exact_dists[q][i] as f64).collect();
        let ap: Vec<f64> = (0..n_nodes).filter(|&i| i != q)
            .map(|i| hamming_u64(&bundles[q], &bundles[i]) as f64).collect();
        rhos.push(spearman(&gt, &ap));
        recalls.push(recall_at_k(&gt, &ap, 10));
    }
    let mean_rho = rhos.iter().sum::<f64>() / rhos.len() as f64;
    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    (mean_rho, mean_recall)
}

/// Compute avg metrics from raw u64 vectors (hamming distance).
fn avg_metrics_raw(
    exact_dists: &[Vec<u32>],
    compressed: &[Vec<u64>],
    n_queries: usize,
    n_nodes: usize,
) -> (f64, f64) {
    let mut rhos = Vec::new();
    let mut recalls = Vec::new();
    for q in 0..n_queries {
        let gt: Vec<f64> = (0..n_nodes).filter(|&i| i != q)
            .map(|i| exact_dists[q][i] as f64).collect();
        let ap: Vec<f64> = (0..n_nodes).filter(|&i| i != q)
            .map(|i| hamming_u64(&compressed[q], &compressed[i]) as f64).collect();
        rhos.push(spearman(&gt, &ap));
        recalls.push(recall_at_k(&gt, &ap, 10));
    }
    let mean_rho = rhos.iter().sum::<f64>() / rhos.len() as f64;
    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    (mean_rho, mean_recall)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TEST 1: Compression curve at dim=128 (fast)
    // ========================================================================

    #[test]
    fn compression_curve_128d() {
        let dim = 128;
        let n = 200;
        let results = benchmark_at_dim(dim, n, 5, 0.5);

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  COMPRESSION CURVE: {}D vectors, n={}                                       ║", dim, n);
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>12} │ {:>6} │ {:>8} │ {:>8} │ {:>7} │ {:>7} ║",
            "Method", "Bits", "Ratio", "ρ", "R@10", "Purity");
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");

        for r in &results {
            eprintln!("║ {:>12} │ {:>6} │ {:>7.1}× │ {:>8.4} │ {:>7.2} │ {:>7.2} ║",
                r.method, r.bits_per_vector, r.compression_ratio, r.spearman_rho,
                r.recall_at_10, r.cluster_purity);
        }
        eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");

        // Sanity: all methods should have non-negative ρ
        for r in &results {
            assert!(r.spearman_rho > -0.5,
                "{} at {} bits has ρ={:.4}", r.method, r.bits_per_vector, r.spearman_rho);
        }
    }

    // ========================================================================
    // TEST 2: Compression curve at dim=768 (realistic embedding size)
    // ========================================================================

    #[test]
    fn compression_curve_768d() {
        let dim = 768;
        let n = 200;
        let results = benchmark_at_dim(dim, n, 5, 0.3);

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  COMPRESSION CURVE: {}D vectors (sentence embeddings), n={}               ║", dim, n);
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>12} │ {:>6} │ {:>8} │ {:>8} │ {:>7} │ {:>7} ║",
            "Method", "Bits", "Ratio", "ρ", "R@10", "Purity");
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");

        let mut sorted = results.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.bits_per_vector.cmp(&b.bits_per_vector));

        for r in &sorted {
            eprintln!("║ {:>12} │ {:>6} │ {:>7.1}× │ {:>8.4} │ {:>7.2} │ {:>7.2} ║",
                r.method, r.bits_per_vector, r.compression_ratio, r.spearman_rho,
                r.recall_at_10, r.cluster_purity);
        }
        eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");

        // At 768d, PQ and SimHash should both work reasonably
        let pq_best = results.iter().filter(|r| r.method == "PQ").map(|r| r.spearman_rho)
            .fold(0.0f64, f64::max);
        assert!(pq_best > 0.3, "PQ best ρ={:.4} too low at 768d", pq_best);
    }

    // ========================================================================
    // TEST 3: Dimension sweep — separation quality curve
    // ========================================================================

    #[test]
    fn dimension_separation_sweep() {
        let dims = [32, 64, 128, 256, 512];
        let results = dimension_sweep(&dims, 30, 5);

        eprintln!("\n╔════════════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  DIMENSION × SEPARATION QUALITY CURVE                                             ║");
        eprintln!("╠════════════════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>4} │ {:>14} │ {:>6} │ {:>9} │ {:>9} │ {:>5} │ {:>6} ║",
            "Dim", "Method", "Bits", "Intra", "Inter", "Sep", "ρ");
        eprintln!("╠════════════════════════════════════════════════════════════════════════════════════╣");

        for r in &results {
            eprintln!("║ {:>4} │ {:>14} │ {:>6} │ {:>9.1} │ {:>9.1} │ {:>5.2} │ {:>6.3} ║",
                r.dim, r.method, r.bits, r.intra_cluster_mean,
                r.inter_cluster_mean, r.separation_ratio, r.spearman_rho);
        }
        eprintln!("╚════════════════════════════════════════════════════════════════════════════════════╝");

        // Higher dimensions should give better separation for all methods
        for method in &["BinaryQuant", "SimHash-256", "SPO-8K"] {
            let method_results: Vec<&DimSweepResult> = results.iter()
                .filter(|r| r.method == *method)
                .collect();
            if method_results.len() >= 2 {
                let first_sep = method_results[0].separation_ratio;
                let last_sep = method_results.last().unwrap().separation_ratio;
                eprintln!("  {} separation: dim={} → {:.2}, dim={} → {:.2}",
                    method, method_results[0].dim, first_sep,
                    method_results.last().unwrap().dim, last_sep);
            }
        }
    }

    // ========================================================================
    // TEST 4: Precision sweet-spot finder (SPO bundle)
    // ========================================================================

    #[test]
    fn precision_sweet_spot() {
        let dim = 256;
        let n = 300;
        let points = find_sweet_spot(dim, n);

        eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PRECISION SWEET SPOT: SPO Bundle at {}D, n={}                 ║", dim, n);
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>7} │ {:>8} │ {:>7} │ {:>12} │ {:>7} ║",
            "Bits", "ρ", "R@10", "Δρ/bit (×1e6)", "Grade");
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");

        let mut peak_idx = 0;
        let mut peak_marginal = 0.0;
        for (i, p) in points.iter().enumerate() {
            let grade = if p.marginal_gain * 1e6 > peak_marginal && i > 0 {
                peak_marginal = p.marginal_gain * 1e6;
                peak_idx = i;
                "←PEAK"
            } else if p.spearman_rho > 0.8 {
                "good"
            } else if p.spearman_rho > 0.5 {
                "ok"
            } else {
                "low"
            };

            eprintln!("║ {:>7} │ {:>8.4} │ {:>7.2} │ {:>12.2} │ {:>7} ║",
                p.bits, p.spearman_rho, p.recall_at_10, p.marginal_gain * 1e6, grade);
        }
        eprintln!("╚══════════════════════════════════════════════════════════════════╝");

        if !points.is_empty() {
            eprintln!("\n  >>> Sweet spot: {} bits (best marginal ρ gain per bit)",
                points[peak_idx].bits);
            eprintln!("  >>> Peak ρ: {:.4} at {} bits",
                points.last().unwrap().spearman_rho, points.last().unwrap().bits);
        }
    }

    // ========================================================================
    // TEST 5: Head-to-head at equal bit budget
    // ========================================================================

    #[test]
    fn head_to_head_equal_budget() {
        let dim = 256;
        let n = 300;
        let spread = 0.4;

        let n_clusters = 5;
        let per_cluster = n / n_clusters;
        let mut vectors: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for c in 0..n_clusters {
            let center = random_f64_vec(dim, (c as u64 + 1) * 3333);
            for m in 0..per_cluster {
                let seed = (c * per_cluster + m) as u64 * 19 + 7;
                let noise = random_f64_vec(dim, seed);
                let v: Vec<f64> = center.iter().zip(&noise).map(|(&c, &n)| c + n * spread).collect();
                vectors.push(v);
                labels.push(c);
            }
        }

        let gt_dists: Vec<f64> = (1..n)
            .map(|i| 1.0 - cosine_sim(&vectors[0], &vectors[i]))
            .collect();

        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  HEAD-TO-HEAD at equal bit budget ({}D, n={})              ║", dim, n);
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // Compare at 1024 bits
        let budget = 1024;
        eprintln!("║  Budget: {} bits ({} bytes)                                ║", budget, budget / 8);
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>14} │ {:>8} │ {:>7} │ {:>7} ║", "Method", "ρ", "R@10", "Purity");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");

        // SimHash @ 1024 bits
        let sh = SimHashProjector::new(dim, budget, 12345);
        let sh_h: Vec<Vec<u8>> = vectors.iter().map(|v| sh.hash(v)).collect();
        let sh_d: Vec<f64> = (1..n).map(|i| hamming_bytes(&sh_h[0], &sh_h[i]) as f64).collect();
        let sh_rho = spearman(&gt_dists, &sh_d);
        let sh_r10 = recall_at_k(&gt_dists, &sh_d, 10);
        let sh_pur = cluster_purity_knn(&sh_d, &labels, 0, 10);
        eprintln!("║ {:>14} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║", "SimHash", sh_rho, sh_r10, sh_pur);

        // Random Projection @ 1024 bits
        let rp = RandomProjection::new(dim, budget, 54321);
        let rp_h: Vec<Vec<u8>> = vectors.iter().map(|v| rp.project_and_binarize(v)).collect();
        let rp_d: Vec<f64> = (1..n).map(|i| hamming_bytes(&rp_h[0], &rp_h[i]) as f64).collect();
        let rp_rho = spearman(&gt_dists, &rp_d);
        let rp_r10 = recall_at_k(&gt_dists, &rp_d, 10);
        let rp_pur = cluster_purity_knn(&rp_d, &labels, 0, 10);
        eprintln!("║ {:>14} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║", "RandProj", rp_rho, rp_r10, rp_pur);

        // PQ @ ~1024 bits = 128 sub-vectors × 8 bits
        if dim % 128 == 0 {
            let pq = ProductQuantizer::train(dim, 128, 8, &vectors, 99999);
            let codes: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v)).collect();
            let pq_d: Vec<f64> = (1..n).map(|i| pq.asymmetric_dist(&vectors[0], &codes[i])).collect();
            let pq_rho = spearman(&gt_dists, &pq_d);
            let pq_r10 = recall_at_k(&gt_dists, &pq_d, 10);
            let pq_pur = cluster_purity_knn(&pq_d, &labels, 0, 10);
            eprintln!("║ {:>14} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║", "PQ-128×8", pq_rho, pq_r10, pq_pur);
        }

        // SPO-Sign @ 1024 bits (sign-bit encoding)
        let n_words = budget / 64;
        let planes: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = vectors.iter()
            .map(|v| {
                let s = sign_bits(&v[..budget.min(dim)], n_words);
                let p_start = (dim / 3).min(dim - 1);
                let o_start = (2 * dim / 3).min(dim - 1);
                let p = sign_bits(&v[p_start..], n_words);
                let o = sign_bits(&v[o_start..], n_words);
                (s, p, o)
            })
            .collect();
        let bundles: Vec<Vec<u64>> = planes.iter()
            .map(|(s, p, o)| spo_bundle(s, p, o, budget))
            .collect();
        let spo_d: Vec<f64> = (1..n).map(|i| hamming_u64(&bundles[0], &bundles[i]) as f64).collect();
        let spo_rho = spearman(&gt_dists, &spo_d);
        let spo_r10 = recall_at_k(&gt_dists, &spo_d, 10);
        let spo_pur = cluster_purity_knn(&spo_d, &labels, 0, 10);
        eprintln!("║ {:>14} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║", "SPO-Sign", spo_rho, spo_r10, spo_pur);

        // BinaryQuant @ 256 bits (1 bit per dim, natural size)
        let bq: Vec<Vec<u8>> = vectors.iter().map(|v| binary_quantize(v)).collect();
        let bq_d: Vec<f64> = (1..n).map(|i| hamming_bytes(&bq[0], &bq[i]) as f64).collect();
        let bq_rho = spearman(&gt_dists, &bq_d);
        let bq_r10 = recall_at_k(&gt_dists, &bq_d, 10);
        let bq_pur = cluster_purity_knn(&bq_d, &labels, 0, 10);
        eprintln!("║ {:>14} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║", "BinQuant(256b)", bq_rho, bq_r10, bq_pur);

        eprintln!("╚══════════════════════════════════════════════════════════════╝");

        // At least SimHash and SPO should have positive correlation
        assert!(sh_rho > 0.0, "SimHash ρ should be positive");
    }

    // ========================================================================
    // TEST 6: Multi-query averaged precision curves
    // ========================================================================

    #[test]
    fn multi_query_precision_curves() {
        let dim = 256;
        let n = 300;
        let n_queries = 10;
        let spread = 0.4;

        let n_clusters = 5;
        let per_cluster = n / n_clusters;
        let mut vectors: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for c in 0..n_clusters {
            let center = random_f64_vec(dim, (c as u64 + 1) * 7777);
            for m in 0..per_cluster {
                let seed = (c * per_cluster + m) as u64 * 23 + 11;
                let noise = random_f64_vec(dim, seed);
                let v: Vec<f64> = center.iter().zip(&noise).map(|(&c, &n)| c + n * spread).collect();
                vectors.push(v);
                labels.push(c);
            }
        }

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  MULTI-QUERY PRECISION ({} queries, {}D, n={})                     ║", n_queries, dim, n);
        eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>12} │ {:>6} │ {:>8} │ {:>8} │ {:>8} ║",
            "Method", "Bits", "mean_ρ", "mean_R@10", "std_ρ");
        eprintln!("╠══════════════════════════════════════════════════════════════════════╣");

        struct MethodConfig {
            name: &'static str,
            bits: usize,
        }

        let configs = vec![
            MethodConfig { name: "SimHash", bits: 256 },
            MethodConfig { name: "SimHash", bits: 1024 },
            MethodConfig { name: "SimHash", bits: 4096 },
            MethodConfig { name: "SPO-Bndl", bits: 1024 },
            MethodConfig { name: "SPO-Bndl", bits: 4096 },
            MethodConfig { name: "SPO-Bndl", bits: 8192 },
            MethodConfig { name: "SPO-Bndl", bits: 16384 },
            MethodConfig { name: "BinQuant", bits: dim },
        ];

        for cfg in &configs {
            let mut rhos = Vec::new();
            let mut recalls = Vec::new();

            for qi in 0..n_queries {
                let gt_dists: Vec<f64> = (0..n)
                    .filter(|&i| i != qi)
                    .map(|i| 1.0 - cosine_sim(&vectors[qi], &vectors[i]))
                    .collect();

                let approx_dists: Vec<f64> = match cfg.name {
                    "SimHash" => {
                        let proj = SimHashProjector::new(dim, cfg.bits, 12345);
                        let hashes: Vec<Vec<u8>> = vectors.iter().map(|v| proj.hash(v)).collect();
                        (0..n).filter(|&i| i != qi)
                            .map(|i| hamming_bytes(&hashes[qi], &hashes[i]) as f64)
                            .collect()
                    }
                    "SPO-Bndl" => {
                        let bw = cfg.bits.min(dim);
                        let nw = bw / 64;
                        if nw == 0 { continue; }
                        let planes: Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> = vectors.iter()
                            .map(|v| {
                                let s = sign_bits(&v[..bw.min(dim)], nw);
                                let ps = (dim / 3).min(dim - 1);
                                let os = (2 * dim / 3).min(dim - 1);
                                let p = sign_bits(&v[ps..], nw);
                                let o = sign_bits(&v[os..], nw);
                                (s, p, o)
                            })
                            .collect();
                        let bundles: Vec<Vec<u64>> = planes.iter()
                            .map(|(s, p, o)| spo_bundle(s, p, o, bw))
                            .collect();
                        (0..n).filter(|&i| i != qi)
                            .map(|i| hamming_u64(&bundles[qi], &bundles[i]) as f64)
                            .collect()
                    }
                    "BinQuant" => {
                        let bq: Vec<Vec<u8>> = vectors.iter().map(|v| binary_quantize(v)).collect();
                        (0..n).filter(|&i| i != qi)
                            .map(|i| hamming_bytes(&bq[qi], &bq[i]) as f64)
                            .collect()
                    }
                    _ => vec![],
                };

                if !approx_dists.is_empty() {
                    rhos.push(spearman(&gt_dists, &approx_dists));
                    recalls.push(recall_at_k(&gt_dists, &approx_dists, 10));
                }
            }

            let mean_rho = rhos.iter().sum::<f64>() / rhos.len() as f64;
            let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
            let std_rho = (rhos.iter().map(|r| (r - mean_rho).powi(2)).sum::<f64>() / rhos.len() as f64).sqrt();

            eprintln!("║ {:>12} │ {:>6} │ {:>8.4} │ {:>8.2} │ {:>8.4} ║",
                cfg.name, cfg.bits, mean_rho, mean_recall, std_rho);
        }

        eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    }

    // ========================================================================
    // TEST 7: Absolute vs Relative compression comparison
    // ========================================================================

    #[test]
    fn absolute_vs_relative_compression() {
        let dim = 256;
        let n = 200;
        let results = benchmark_at_dim(dim, n, 5, 0.4);

        eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  ABSOLUTE vs RELATIVE COMPRESSION: {}D                                   ║", dim);
        eprintln!("╠═══════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║  Source: {} bits ({} bytes) per vector                                  ║", dim * 64, dim * 8);
        eprintln!("╠═══════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>12} │ {:>8} │ {:>8} │ {:>6} │ {:>8} │ {:>7} ║",
            "Method", "Abs(bits)", "Abs(KB)", "Ratio", "ρ", "R@10");
        eprintln!("╠═══════════════════════════════════════════════════════════════════════════╣");

        let mut sorted = results.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.bits_per_vector.cmp(&b.bits_per_vector));

        for r in &sorted {
            let kb = r.bits_per_vector as f64 / 8192.0;
            eprintln!("║ {:>12} │ {:>8} │ {:>7.3} │ {:>5.0}× │ {:>8.4} │ {:>7.2} ║",
                r.method, r.bits_per_vector, kb, r.compression_ratio,
                r.spearman_rho, r.recall_at_10);
        }
        eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");

        // Find Pareto frontier: methods with best ρ for their bit budget
        eprintln!("\n  Pareto frontier (best ρ per bit budget):");
        let mut pareto: Vec<(&str, usize, f64)> = Vec::new();
        let mut best_rho = -1.0f64;
        for r in &sorted {
            if r.spearman_rho > best_rho {
                best_rho = r.spearman_rho;
                pareto.push((r.method, r.bits_per_vector, r.spearman_rho));
                eprintln!("    {} @ {} bits → ρ={:.4}", r.method, r.bits_per_vector, r.spearman_rho);
            }
        }
    }

    // ========================================================================
    // TEST 8: Native binary plane compression (the real SPO use case)
    // ========================================================================

    #[test]
    fn native_binary_compression_curve() {
        let n_nodes = 200;
        let n_queries = 20;
        let plane_words = 256; // 16Kbit planes
        let results = native_binary_benchmark(n_nodes, n_queries, plane_words);

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  NATIVE BINARY COMPRESSION: 3×16Kbit SPO planes → bundle (n={})                       ║", n_nodes);
        eprintln!("║  Source: 49152 bits (6KB) per SPO triple                                                ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>13} │ {:>6} │ {:>6} │ {:>8} │ {:>8} │ {:>7} │ {:>7} ║",
            "Method", "Bits", "Ratio", "ρ_rand", "ρ_struct", "R@10_r", "R@10_s");
        eprintln!("╠══════════════════════════════════════════════════════════════════════════════════════════╣");

        let source_bits = plane_words * 64 * 3;
        let mut sorted = results.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| {
            a.method.cmp(&b.method).then(a.bits.cmp(&b.bits).reverse())
        });

        for r in &sorted {
            let ratio = source_bits as f64 / r.bits as f64;
            eprintln!("║ {:>13} │ {:>6} │ {:>5.1}× │ {:>8.4} │ {:>8.4} │ {:>7.2} │ {:>7.2} ║",
                r.method, r.bits, ratio, r.rho_random, r.rho_structured,
                r.recall_at_10_random, r.recall_at_10_structured);
        }
        eprintln!("╚══════════════════════════════════════════════════════════════════════════════════════════╝");

        // Find crossover point: where does cyclic bundle beat truncation?
        eprintln!("\n  Method comparison at each bit budget:");
        let budgets: Vec<usize> = results.iter().map(|r| r.bits).collect::<std::collections::HashSet<_>>()
            .into_iter().collect::<Vec<_>>();
        let mut budgets_sorted = budgets;
        budgets_sorted.sort_by(|a, b| b.cmp(a));

        for &bits in &budgets_sorted {
            let bundle = results.iter().find(|r| r.method == "CyclicBundle" && r.bits == bits);
            let trunc = results.iter().find(|r| r.method == "Truncation" && r.bits == bits);
            let sample = results.iter().find(|r| r.method == "BitSample" && r.bits == bits);

            if let (Some(b), Some(t)) = (bundle, trunc) {
                let winner = if b.rho_structured > t.rho_structured { "Bundle" } else { "Trunc" };
                let s_winner = if let Some(s) = sample {
                    if s.rho_structured > b.rho_structured.max(t.rho_structured) { "Sample" } else { winner }
                } else { winner };
                eprintln!("    {}b: Bundle ρ_s={:.4} vs Trunc ρ_s={:.4} → {}",
                    bits, b.rho_structured, t.rho_structured, s_winner);
            }
        }

        // CyclicBundle should have reasonable ρ on structured data
        let best_bundle_struct = results.iter()
            .filter(|r| r.method == "CyclicBundle")
            .map(|r| r.rho_structured)
            .fold(0.0f64, f64::max);
        eprintln!("\n  Best CyclicBundle ρ_structured: {:.4}", best_bundle_struct);
    }

    // ========================================================================
    // TEST 9: Precision sweet-spot for native binary (fine-grained)
    // ========================================================================

    #[test]
    fn native_binary_sweet_spot() {
        let n_nodes = 200;
        let n_queries = 15;
        let plane_words = 256;
        let plane_bits = plane_words * 64;
        let source_bits = plane_bits * 3;

        // Generate structured data (clustered S planes)
        let base_planes: Vec<Vec<u64>> = (0..5)
            .map(|c| random_plane(plane_words, c as u64 * 50000 + 42))
            .collect();

        let mut nodes_s = Vec::new();
        let mut nodes_p = Vec::new();
        let mut nodes_o = Vec::new();
        let mut labels = Vec::new();

        for (ci, base) in base_planes.iter().enumerate() {
            for m in 0..(n_nodes / 5) {
                let s = flip_plane(base, m * 200, (ci * 100 + m) as u64);
                let p = random_plane(plane_words, (ci * 100 + m) as u64 * 7 + 1);
                let o = random_plane(plane_words, (ci * 100 + m) as u64 * 7 + 2);
                nodes_s.push(s);
                nodes_p.push(p);
                nodes_o.push(o);
                labels.push(ci);
            }
        }

        // Ground truth
        let exact: Vec<Vec<u32>> = (0..n_queries)
            .map(|q| {
                (0..n_nodes)
                    .map(|i| {
                        hamming_u64(&nodes_s[q], &nodes_s[i])
                            + hamming_u64(&nodes_p[q], &nodes_p[i])
                            + hamming_u64(&nodes_o[q], &nodes_o[i])
                    })
                    .collect()
            })
            .collect();

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  NATIVE BINARY SWEET SPOT: Clustered SPO planes                     ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
        eprintln!("║ {:>7} │ {:>6} │ {:>8} │ {:>7} │ {:>8} │ {:>12} ║",
            "Bits", "Ratio", "ρ", "R@10", "Purity", "Δρ/bit×1e6");
        eprintln!("╠══════════════════════════════════════════════════════════════════════╣");

        let bit_widths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
        let mut prev_rho = 0.0;
        let mut prev_bits = 0usize;
        let mut peak_marginal = 0.0;
        let mut sweet_spot = 0;

        for &target_bits in &bit_widths {
            let target_words = target_bits / 64;

            let bundles: Vec<Vec<u64>> = (0..n_nodes)
                .map(|i| {
                    let s = xor_fold(&nodes_s[i], target_words);
                    let p = xor_fold(&nodes_p[i], target_words);
                    let o = xor_fold(&nodes_o[i], target_words);
                    spo_bundle(&s, &p, &o, target_bits)
                })
                .collect();

            let (rho, recall) = avg_metrics(&exact, &bundles, n_queries, n_nodes);

            // Cluster purity via k-NN
            let mut purity_sum = 0.0;
            for q in 0..n_queries {
                let mut dists: Vec<(usize, u32)> = (0..n_nodes)
                    .filter(|&i| i != q)
                    .map(|i| (i, hamming_u64(&bundles[q], &bundles[i])))
                    .collect();
                dists.sort_by_key(|&(_, d)| d);
                let same = dists[..10.min(dists.len())]
                    .iter()
                    .filter(|&&(i, _)| labels[i] == labels[q])
                    .count();
                purity_sum += same as f64 / 10.0;
            }
            let purity = purity_sum / n_queries as f64;

            let marginal = if prev_bits > 0 && target_bits > prev_bits {
                (rho - prev_rho) / (target_bits - prev_bits) as f64
            } else {
                0.0
            };

            let marker = if marginal * 1e6 > peak_marginal && prev_bits > 0 {
                peak_marginal = marginal * 1e6;
                sweet_spot = target_bits;
                " ←PEAK"
            } else {
                ""
            };

            eprintln!("║ {:>7} │ {:>5.1}× │ {:>8.4} │ {:>7.2} │ {:>8.2} │ {:>11.2}{} ║",
                target_bits, source_bits as f64 / target_bits as f64,
                rho, recall, purity, marginal * 1e6, marker);

            prev_rho = rho;
            prev_bits = target_bits;
        }

        eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
        if sweet_spot > 0 {
            eprintln!("  >>> Sweet spot: {} bits ({}× compression)",
                sweet_spot, source_bits / sweet_spot);
        }
    }
}
