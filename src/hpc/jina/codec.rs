//! Jina GGUF → Base17 → Palette codec.
//!
//! Loads embedding matrix from GGUF, projects through golden-step Base17,
//! builds k-means palette. All via `crate::simd` for SIMD acceleration.

use crate::simd::F32x16;
use std::sync::LazyLock;

/// Base17 projection parameters.
pub const BASE_DIM: usize = 17;
pub const GOLDEN_STEP: usize = 11;
pub const FP_SCALE: f32 = 1000.0;
pub const PALETTE_K: usize = 256;

/// A single Base17-projected token embedding.
#[derive(Clone, Copy, Debug)]
pub struct Base17Token {
    pub dims: [i16; BASE_DIM],
}

impl Base17Token {
    /// Project an f32 embedding vector to Base17 via golden-step folding.
    #[inline]
    pub fn from_f32(embedding: &[f32]) -> Self {
        let d = embedding.len();
        let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;
        let mut sum = [0.0f64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];

        for octave in 0..n_octaves {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + ((bi * GOLDEN_STEP) % BASE_DIM);
                if dim < d {
                    sum[bi] += embedding[dim] as f64;
                    count[bi] += 1;
                }
            }
        }

        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            if count[i] > 0 {
                let mean = sum[i] / count[i] as f64;
                dims[i] = (mean * FP_SCALE as f64).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
        Base17Token { dims }
    }

    /// L1 distance between two Base17 tokens.
    ///
    /// 17 dimensions: 16 via `crate::simd::F32x16` + 1 scalar remainder.
    /// Consumer never sees hardware — `F32x16` dispatches to AVX-512/AVX2/scalar.
    #[inline(always)]
    pub fn l1(&self, other: &Base17Token) -> u32 {
        // SIMD path: load 16 dims as f32, compute abs diff, reduce.
        // The 17th dim is scalar.
        let mut a_f32 = [0.0f32; 16];
        let mut b_f32 = [0.0f32; 16];
        for i in 0..16 {
            a_f32[i] = self.dims[i] as f32;
            b_f32[i] = other.dims[i] as f32;
        }
        let va = F32x16::from_slice(&a_f32);
        let vb = F32x16::from_slice(&b_f32);
        let diff = va - vb;
        let abs_diff = diff.abs();
        let simd_sum = abs_diff.reduce_sum() as u32;

        // 17th dimension: scalar
        let d16 = (self.dims[16] as i32 - other.dims[16] as i32).unsigned_abs();
        simd_sum + d16
    }
}

/// Palette: 256 centroids for O(1) token categorization.
#[derive(Clone)]
pub struct JinaPalette {
    /// 256 centroid vectors in Base17 space.
    pub centroids: [Base17Token; PALETTE_K],
    /// Per-token palette assignment (index into centroids).
    pub assignments: Vec<u8>,
    /// Precomputed 256×256 L1 distance table.
    pub distance_table: [[u16; PALETTE_K]; PALETTE_K],
}

impl JinaPalette {
    /// Build palette from Base17 embeddings via k-means.
    pub fn build(tokens: &[Base17Token], max_iter: usize) -> Self {
        let n = tokens.len();
        assert!(n >= PALETTE_K, "Need at least {} tokens", PALETTE_K);

        // Initialize centroids: evenly spaced sample
        let step = n / PALETTE_K;
        let mut centroids = [Base17Token { dims: [0; BASE_DIM] }; PALETTE_K];
        for k in 0..PALETTE_K {
            centroids[k] = tokens[k * step];
        }

        let mut assignments = vec![0u8; n];

        for _iter in 0..max_iter {
            // Assign each token to nearest centroid
            let mut changed = false;
            for i in 0..n {
                let mut best_k = 0u8;
                let mut best_d = u32::MAX;
                for k in 0..PALETTE_K {
                    let d = tokens[i].l1(&centroids[k]);
                    if d < best_d {
                        best_d = d;
                        best_k = k as u8;
                    }
                }
                if assignments[i] != best_k {
                    assignments[i] = best_k;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Update centroids
            let mut sums = [[0i64; BASE_DIM]; PALETTE_K];
            let mut counts = [0u32; PALETTE_K];
            for i in 0..n {
                let k = assignments[i] as usize;
                counts[k] += 1;
                for d in 0..BASE_DIM {
                    sums[k][d] += tokens[i].dims[d] as i64;
                }
            }
            for k in 0..PALETTE_K {
                if counts[k] > 0 {
                    for d in 0..BASE_DIM {
                        centroids[k].dims[d] =
                            (sums[k][d] / counts[k] as i64).clamp(-32768, 32767) as i16;
                    }
                }
            }
        }

        // Build distance table
        let mut distance_table = [[0u16; PALETTE_K]; PALETTE_K];
        for i in 0..PALETTE_K {
            for j in i..PALETTE_K {
                let d = centroids[i].l1(&centroids[j]).min(u16::MAX as u32) as u16;
                distance_table[i][j] = d;
                distance_table[j][i] = d;
            }
        }

        JinaPalette {
            centroids,
            assignments,
            distance_table,
        }
    }

    /// O(1) distance between two tokens via palette lookup.
    #[inline(always)]
    pub fn distance(&self, token_a: usize, token_b: usize) -> u16 {
        self.distance_table[self.assignments[token_a] as usize]
            [self.assignments[token_b] as usize]
    }

    /// Palette index for a token.
    #[inline(always)]
    pub fn palette_index(&self, token: usize) -> u8 {
        self.assignments[token]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base17_projection() {
        let embedding = vec![1.0f32; 2048];
        let token = Base17Token::from_f32(&embedding);
        // All dims should be equal (uniform input)
        assert!(token.dims.iter().all(|&d| (d - token.dims[0]).abs() < 2));
    }

    #[test]
    fn test_base17_l1_self_zero() {
        let t = Base17Token { dims: [100; BASE_DIM] };
        assert_eq!(t.l1(&t), 0);
    }

    #[test]
    fn test_base17_l1_symmetric() {
        let a = Base17Token::from_f32(&vec![1.0; 256]);
        let b = Base17Token::from_f32(&vec![2.0; 256]);
        assert_eq!(a.l1(&b), b.l1(&a));
    }

    #[test]
    fn test_palette_build_small() {
        // Build palette from 512 synthetic tokens
        let tokens: Vec<Base17Token> = (0..512)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    dims[d] = ((i * 17 + d * 31) % 1000) as i16 - 500;
                }
                Base17Token { dims }
            })
            .collect();

        let palette = JinaPalette::build(&tokens, 10);
        assert_eq!(palette.assignments.len(), 512);
        assert_eq!(palette.distance_table[0][0], 0); // self-distance = 0

        // Distance should be symmetric
        let d01 = palette.distance(0, 1);
        let d10 = palette.distance(1, 0);
        assert_eq!(d01, d10);
    }

    #[test]
    fn test_palette_distance_table_symmetric() {
        let tokens: Vec<Base17Token> = (0..256)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                dims[0] = i as i16 * 100;
                Base17Token { dims }
            })
            .collect();

        let palette = JinaPalette::build(&tokens, 5);
        for i in 0..PALETTE_K {
            for j in 0..PALETTE_K {
                assert_eq!(
                    palette.distance_table[i][j],
                    palette.distance_table[j][i],
                    "Asymmetric at [{i}][{j}]"
                );
            }
        }
    }
}
