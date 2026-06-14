//! Edge-codec flavors — deterministic palette code ↔ residue, at three byte
//! budgets that share ONE 16-byte edge block by *interpretation*, not layout.
//!
//! The canon node's edge block is 16 bytes (`canonical_node.rs`). A class picks
//! how those bytes (plus an optional value-slab residue) are read:
//!
//! | flavor          | bytes/vector        | interpretation                       |
//! |-----------------|---------------------|--------------------------------------|
//! | [`CoarseOnly`]  | 1 (palette index)   | one of K=256 centroids — the EdgeBlock byte as-is |
//! | [`CoarseResidue`]| 1 + D/2             | centroid + per-dim signed-4-bit residue (value slab) |
//! | [`Pq32x4`]      | 16 (32 × 4-bit)     | the 16 edge bytes read as a 32-subquantizer PQ code  |
//!
//! All three are *quantizers of the same D-dim vector* measured by reconstruction
//! fidelity, so they are directly comparable (feed each output to
//! [`crate::hpc::reliability::FidelityReport`]). The deterministic part (the
//! nearest-centroid assignment) is recomputable — on x86 it runs on the AMX
//! `matmul_i8_to_i32` tile path (see `examples/edge_codec_compare.rs`); here the
//! assignment is plain scalar so the codec is portable and bit-identical to the
//! accelerated path.
//!
//! Nibbles are packed `lo = code[2t]`, `hi = code[2t+1]` (FastScan/turbovec
//! convention). Signed-4-bit residue codes are two's-complement in the low 4
//! bits (range −8..=7).
//!
//! [`CoarseOnly`]: reconstruct_coarse
//! [`CoarseResidue`]: CoarseResidueCodec
//! [`Pq32x4`]: ProductQuantizer

/// SplitMix64 — deterministic seeding for k-means init (no external rng dep).
#[inline]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// A flat k-means codebook: `k` centroids of `dim` f32 each, row-major.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Number of centroids.
    pub k: usize,
    /// Dimensionality of each centroid.
    pub dim: usize,
    /// `k * dim` centroid coordinates, row-major.
    pub centroids: Vec<f32>,
}

impl Codebook {
    /// Train `k` centroids on `n` row-major vectors of `dim` via Lloyd's
    /// algorithm (`iters` passes, deterministic `seed`). Empty clusters are
    /// re-seeded to a pseudo-random data point so every centroid stays live.
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::edge_codec::Codebook;
    /// // Two well-separated blobs → 2 centroids recover them.
    /// let data = [0.0, 0.0,  0.1, 0.0,  9.0, 9.0,  9.1, 9.0];
    /// let cb = Codebook::train(&data, 4, 2, 2, 10, 1);
    /// assert_eq!(cb.k, 2);
    /// assert_ne!(cb.assign(&[0.05, 0.0]), cb.assign(&[9.05, 9.0]));
    /// ```
    pub fn train(data: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(dim > 0 && k > 0 && n >= k, "need n >= k >= 1 and dim > 0");
        assert_eq!(data.len(), n * dim, "data length must be n * dim");
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        // Init: distinct random data points as centroids.
        let mut centroids = vec![0.0f32; k * dim];
        {
            let mut chosen = vec![false; n];
            for c in 0..k {
                let mut idx = (splitmix(&mut state) % n as u64) as usize;
                let mut guard = 0;
                while chosen[idx] && guard < n {
                    idx = (idx + 1) % n;
                    guard += 1;
                }
                chosen[idx] = true;
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
            }
        }
        let mut assign = vec![0u32; n];
        for _ in 0..iters {
            // Assignment step.
            for i in 0..n {
                let v = &data[i * dim..(i + 1) * dim];
                let mut best = f32::INFINITY;
                let mut bj = 0u32;
                for c in 0..k {
                    let d = sq_dist(v, &centroids[c * dim..(c + 1) * dim]);
                    if d < best {
                        best = d;
                        bj = c as u32;
                    }
                }
                assign[i] = bj;
            }
            // Update step.
            let mut sums = vec![0.0f32; k * dim];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let c = assign[i] as usize;
                counts[c] += 1;
                let v = &data[i * dim..(i + 1) * dim];
                for d in 0..dim {
                    sums[c * dim + d] += v[d];
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    // Re-seed dead cluster to a random point.
                    let idx = (splitmix(&mut state) % n as u64) as usize;
                    centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..dim {
                        centroids[c * dim + d] = sums[c * dim + d] * inv;
                    }
                }
            }
        }
        Codebook { k, dim, centroids }
    }

    /// Index of the nearest centroid to `v` (scalar; identical result to the
    /// AMX `matmul_i8_to_i32` assignment path). Panics if `v.len() != dim`.
    #[inline]
    pub fn assign(&self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let mut best = f32::INFINITY;
        let mut bj = 0u32;
        for c in 0..self.k {
            let d = sq_dist(v, &self.centroids[c * self.dim..(c + 1) * self.dim]);
            if d < best {
                best = d;
                bj = c as u32;
            }
        }
        bj
    }

    /// Borrow centroid `c` as a slice.
    #[inline]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }
}

/// Pack signed nibbles (`-8..=7`) two-per-byte: `lo = codes[2t]`, `hi = codes[2t+1]`.
/// `codes.len()` must be even.
fn pack_nibbles_signed(codes: &[i8]) -> Vec<u8> {
    debug_assert!(codes.len().is_multiple_of(2));
    codes
        .chunks_exact(2)
        .map(|p| ((p[0] as u8) & 0x0F) | (((p[1] as u8) & 0x0F) << 4))
        .collect()
}

/// Unpack `n` signed nibbles from `bytes` (inverse of [`pack_nibbles_signed`]).
fn unpack_nibbles_signed(bytes: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let byte = bytes[t / 2];
        let nib = if t.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 };
        // Sign-extend the low 4 bits.
        out.push(if nib >= 8 { nib as i8 - 16 } else { nib as i8 });
    }
    out
}

// ── Flavor 1: coarse only ────────────────────────────────────────────────────

/// Reconstruct a vector from its coarse palette index alone (1 byte/vector).
/// This is the canonical EdgeBlock byte read literally as a centroid pointer.
///
/// # Examples
/// ```
/// use ndarray::hpc::edge_codec::{reconstruct_coarse, Codebook};
/// let data = [0.0, 0.0, 5.0, 5.0]; // 2 points × dim 2
/// let cb = Codebook::train(&data, 2, 2, 2, 5, 1);
/// let idx = cb.assign(&[5.0, 5.0]);
/// assert_eq!(reconstruct_coarse(&cb, idx).len(), 2);
/// ```
#[inline]
pub fn reconstruct_coarse(cb: &Codebook, index: u32) -> Vec<f32> {
    cb.centroid(index as usize).to_vec()
}

// ── Flavor 2: coarse + per-dim signed-4-bit residue (value-slab) ─────────────

/// Coarse palette + a shared-scale per-dimension signed-4-bit residue.
///
/// The residue scale is computed once over the training set (class-level), so
/// each vector costs exactly `1 + dim/2` bytes (no per-vector scale word). The
/// residue lands in the node's reserved value slab — no edge-block layout change.
#[derive(Debug, Clone)]
pub struct CoarseResidueCodec {
    /// The coarse palette.
    pub cb: Codebook,
    /// Shared residue quantization step (max|residue| / 7 over the train set).
    pub res_scale: f32,
}

/// One encoded vector: coarse index + packed `dim/2` residue bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoarseResidueCode {
    /// Nearest-centroid index (the coarse 1-byte code).
    pub index: u32,
    /// `dim/2` bytes of packed signed nibbles (the value-slab residue).
    pub residue: Vec<u8>,
}

impl CoarseResidueCodec {
    /// Fit the palette (k-means) and the shared residue scale on the data.
    /// `dim` must be even (nibbles pack two-per-byte).
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::edge_codec::CoarseResidueCodec;
    /// let data: Vec<f32> = (0..8 * 4).map(|i| (i % 5) as f32 - 2.0).collect();
    /// let codec = CoarseResidueCodec::fit(&data, 8, 4, 4, 6, 1);
    /// let code = codec.encode(&data[0..4]);
    /// assert_eq!(code.residue.len(), 2); // dim/2 packed residue bytes
    /// assert_eq!(codec.reconstruct(&code).len(), 4);
    /// ```
    pub fn fit(data: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(dim.is_multiple_of(2), "dim must be even for nibble packing");
        let cb = Codebook::train(data, n, dim, k, iters, seed);
        let mut rmax = 0.0f32;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let c = cb.centroid(cb.assign(v) as usize);
            for d in 0..dim {
                rmax = rmax.max((v[d] - c[d]).abs());
            }
        }
        let res_scale = (rmax / 7.0).max(1e-12);
        CoarseResidueCodec { cb, res_scale }
    }

    /// Encode one vector to coarse index + packed residue nibbles.
    pub fn encode(&self, v: &[f32]) -> CoarseResidueCode {
        let dim = self.cb.dim;
        let index = self.cb.assign(v);
        let c = self.cb.centroid(index as usize);
        let codes: Vec<i8> = (0..dim)
            .map(|d| {
                let q = ((v[d] - c[d]) / self.res_scale).round();
                q.clamp(-8.0, 7.0) as i8
            })
            .collect();
        CoarseResidueCode {
            index,
            residue: pack_nibbles_signed(&codes),
        }
    }

    /// Reconstruct `centroid + dequantized residue`.
    pub fn reconstruct(&self, code: &CoarseResidueCode) -> Vec<f32> {
        let dim = self.cb.dim;
        let c = self.cb.centroid(code.index as usize);
        let res = unpack_nibbles_signed(&code.residue, dim);
        (0..dim)
            .map(|d| c[d] + res[d] as f32 * self.res_scale)
            .collect()
    }
}

// ── Flavor 3: product quantizer — 32 subspaces × 4-bit (the 16-byte block) ───

/// Product quantizer: split `dim` into `m` subspaces, each with a 16-entry
/// (4-bit) sub-codebook. With `m = 32` the code is exactly `16` bytes — the same
/// budget as the edge block, read as 32 nibble codes (the turbovec PQ model).
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Number of subspaces (nibble codes per vector).
    pub m: usize,
    /// Sub-vector dimensionality (`dim / m`).
    pub sub_dim: usize,
    /// One 16-entry sub-codebook per subspace.
    pub sub: Vec<Codebook>,
}

impl ProductQuantizer {
    /// Fit `m` independent 16-entry sub-codebooks. `dim` must be divisible by
    /// `m`, and `m` must be even (nibbles pack two-per-byte).
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::edge_codec::ProductQuantizer;
    /// // dim 8, m 4 subspaces → 4 nibbles → 2 bytes.
    /// let data: Vec<f32> = (0..32 * 8).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
    /// let pq = ProductQuantizer::fit(&data, 32, 8, 4, 6, 1);
    /// let code = pq.encode(&data[0..8]);
    /// assert_eq!(code.len(), 2);
    /// assert_eq!(pq.reconstruct(&code).len(), 8);
    /// ```
    pub fn fit(data: &[f32], n: usize, dim: usize, m: usize, iters: usize, seed: u64) -> Self {
        assert!(m.is_multiple_of(2), "m must be even for nibble packing");
        assert!(dim.is_multiple_of(m), "dim must be divisible by m");
        let sub_dim = dim / m;
        let ksub = 16; // 4-bit
        let mut sub = Vec::with_capacity(m);
        for s in 0..m {
            // Gather subspace `s` columns into a contiguous n × sub_dim buffer.
            let mut buf = vec![0.0f32; n * sub_dim];
            for i in 0..n {
                let src = &data[i * dim + s * sub_dim..i * dim + (s + 1) * sub_dim];
                buf[i * sub_dim..(i + 1) * sub_dim].copy_from_slice(src);
            }
            sub.push(Codebook::train(&buf, n, sub_dim, ksub, iters, seed ^ (s as u64).wrapping_mul(0x9E37_79B9)));
        }
        ProductQuantizer { m, sub_dim, sub }
    }

    /// Encode one vector to `m/2` packed nibble bytes (16 bytes when `m = 32`).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let codes: Vec<i8> = (0..self.m)
            .map(|s| {
                let sv = &v[s * self.sub_dim..(s + 1) * self.sub_dim];
                self.sub[s].assign(sv) as i8 // 0..=15 fits the low nibble
            })
            .collect();
        // Unsigned 0..15 codes share the same packing as signed (low 4 bits).
        codes
            .chunks_exact(2)
            .map(|p| ((p[0] as u8) & 0x0F) | (((p[1] as u8) & 0x0F) << 4))
            .collect()
    }

    /// Reconstruct by concatenating the selected sub-centroids.
    pub fn reconstruct(&self, code: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.m * self.sub_dim];
        for s in 0..self.m {
            let byte = code[s / 2];
            let nib = if s.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 } as usize;
            out[s * self.sub_dim..(s + 1) * self.sub_dim].copy_from_slice(self.sub[s].centroid(nib));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic clustered data: `n` vectors of `dim`, drawn from `k` blobs.
    fn make_data(n: usize, dim: usize, k: usize, noise: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let centers: Vec<f32> = (0..k * dim)
            .map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
            .collect();
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            let c = (splitmix(&mut s) % k as u64) as usize;
            for d in 0..dim {
                let z = (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                data[i * dim + d] = centers[c * dim + d] + noise * z;
            }
        }
        data
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let se: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let sn: f32 = a.iter().map(|x| x * x).sum();
        (se.sqrt() / sn.sqrt().max(1e-12)).max(0.0)
    }

    #[test]
    fn nibble_pack_roundtrip_signed() {
        let codes: Vec<i8> = vec![-8, 7, 0, -1, 3, -4];
        let packed = pack_nibbles_signed(&codes);
        assert_eq!(packed.len(), 3);
        let back = unpack_nibbles_signed(&packed, codes.len());
        assert_eq!(back, codes);
    }

    #[test]
    fn kmeans_separates_blobs() {
        let data = make_data(256, 8, 4, 0.05, 11);
        let cb = Codebook::train(&data, 256, 8, 4, 12, 7);
        // All 4 centroids should be used (no degenerate collapse).
        let mut used = [false; 4];
        for i in 0..256 {
            used[cb.assign(&data[i * 8..(i + 1) * 8]) as usize] = true;
        }
        assert!(used.iter().all(|&u| u), "all centroids should attract points");
    }

    #[test]
    fn coarse_residue_roundtrip_is_deterministic() {
        let data = make_data(512, 16, 32, 0.2, 3);
        let codec = CoarseResidueCodec::fit(&data, 512, 16, 32, 10, 5);
        let v = &data[0..16];
        let c1 = codec.encode(v);
        let c2 = codec.encode(v);
        assert_eq!(c1, c2, "encode must be deterministic");
        assert_eq!(c1.residue.len(), 8, "dim/2 = 8 residue bytes");
        let r = codec.reconstruct(&c1);
        assert_eq!(r.len(), 16);
    }

    #[test]
    fn residue_beats_coarse_only() {
        let (n, dim, k) = (1024, 32, 256);
        let data = make_data(n, dim, k, 0.25, 9);
        let codec = CoarseResidueCodec::fit(&data, n, dim, k, 12, 1);
        let mut coarse_err = 0.0f32;
        let mut fine_err = 0.0f32;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let code = codec.encode(v);
            let idx = code.index;
            coarse_err += rel_l2(v, &reconstruct_coarse(&codec.cb, idx));
            fine_err += rel_l2(v, &codec.reconstruct(&code));
        }
        assert!(
            fine_err < coarse_err * 0.5,
            "coarse+residue must roughly halve error: coarse={coarse_err} fine={fine_err}"
        );
    }

    #[test]
    fn pq32x4_code_is_sixteen_bytes_and_reconstructs() {
        // High-dim CONTINUOUS data (not blobs): the realistic embedding regime
        // where a 256-entry coarse codebook can't tile the space but a 32×4-bit
        // product code can. With few blobs, coarse-only memorises them and the
        // comparison is meaningless — PQ's win is a high-dimensionality effect.
        let (n, dim) = (1024, 64);
        let mut s = 0xABCD_1234u64;
        let data: Vec<f32> = (0..n * dim)
            .map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
            .collect();
        let pq = ProductQuantizer::fit(&data, n, dim, 32, 10, 2);
        let code = pq.encode(&data[0..dim]);
        assert_eq!(code.len(), 16, "32 nibbles = 16 bytes (the edge-block budget)");
        assert_eq!(pq.reconstruct(&code).len(), dim);

        let cb = Codebook::train(&data, n, dim, 256, 12, 2);
        let mut pq_err = 0.0f32;
        let mut coarse_err = 0.0f32;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            pq_err += rel_l2(v, &pq.reconstruct(&pq.encode(v)));
            coarse_err += rel_l2(v, &reconstruct_coarse(&cb, cb.assign(v)));
        }
        assert!(
            pq_err < coarse_err,
            "PQ(16B) should beat coarse(1B) in high-D continuous regime: pq={pq_err} coarse={coarse_err}"
        );
    }
}
