//! SimHash projection: embedding → binary hypervector.
//!
//! Projects dense float embeddings into binary space using
//! random hyperplane projection (SimHash / locality-sensitive hashing).

use crate::imp_prelude::*;

/// Project a float embedding into a binary hypervector using SimHash.
///
/// Uses a seeded PRNG to generate random hyperplanes,
/// then projects the embedding onto each hyperplane.
/// The sign of the projection determines the output bit.
///
/// # Arguments
/// * `embedding` - Dense float embedding
/// * `container_bits` - Number of bits in the output container
/// * `seed` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ndarray::hpc::projection::simhash_project;
///
/// let embedding = vec![1.0f32, -0.5, 0.3, 0.8];
/// let binary = simhash_project(&embedding, 32, 42);
/// assert_eq!(binary.len(), 4); // 32 bits = 4 bytes
/// ```
pub fn simhash_project(embedding: &[f32], container_bits: usize, seed: u64) -> Array<u8, Ix1> {
    let container_bytes = (container_bits + 7) / 8;
    let dim = embedding.len();
    let mut result = vec![0u8; container_bytes];

    for bit_idx in 0..container_bits {
        // Generate random hyperplane using LCG
        let mut dot = 0.0f32;
        let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(bit_idx as u64);
        for d in 0..dim {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(d as u64 + 1);
            // Convert to [-1, 1] range
            let random_val = ((rng_state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0;
            dot += embedding[d] * random_val;
        }
        if dot > 0.0 {
            result[bit_idx / 8] |= 1 << (bit_idx % 8);
        }
    }

    Array::from_vec(result)
}

/// Batch project multiple embeddings.
///
/// # Arguments
/// * `embeddings` - Flat array of n*d floats (n embeddings of dimension d)
/// * `n` - Number of embeddings
/// * `d` - Embedding dimension
/// * `container_bits` - Number of bits per output container
/// * `seed` - Random seed
///
/// # Example
///
/// ```
/// use ndarray::hpc::projection::simhash_batch_project;
///
/// let embeddings = vec![1.0f32, 0.5, -0.3, 0.8, -1.0, 0.2];
/// let results = simhash_batch_project(&embeddings, 2, 3, 16, 42);
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].len(), 2); // 16 bits = 2 bytes
/// ```
pub fn simhash_batch_project(
    embeddings: &[f32],
    n: usize,
    d: usize,
    container_bits: usize,
    seed: u64,
) -> Vec<Array<u8, Ix1>> {
    (0..n)
        .map(|i| {
            let start = i * d;
            let end = (start + d).min(embeddings.len());
            simhash_project(&embeddings[start..end], container_bits, seed)
        })
        .collect()
}

/// Project an int8 embedding into a binary hypervector.
///
/// Same algorithm as `simhash_project` but for int8 input.
pub fn simhash_int8_project(embedding_i8: &[i8], container_bits: usize, seed: u64) -> Array<u8, Ix1> {
    let container_bytes = (container_bits + 7) / 8;
    let dim = embedding_i8.len();
    let mut result = vec![0u8; container_bytes];

    for bit_idx in 0..container_bits {
        let mut dot = 0i64;
        let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(bit_idx as u64);
        for d in 0..dim {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(d as u64 + 1);
            let random_val = if (rng_state >> 63) == 0 { 1i64 } else { -1i64 };
            dot += embedding_i8[d] as i64 * random_val;
        }
        if dot > 0 {
            result[bit_idx / 8] |= 1 << (bit_idx % 8);
        }
    }

    Array::from_vec(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_project_deterministic() {
        let emb = vec![1.0f32, -0.5, 0.3, 0.8];
        let a = simhash_project(&emb, 64, 42);
        let b = simhash_project(&emb, 64, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_simhash_project_size() {
        let emb = vec![1.0f32; 128];
        let result = simhash_project(&emb, 256, 0);
        assert_eq!(result.len(), 32); // 256 bits = 32 bytes
    }

    #[test]
    fn test_batch_project() {
        let embeddings = vec![1.0f32, 0.5, -0.3, 0.8, -1.0, 0.2];
        let results = simhash_batch_project(&embeddings, 2, 3, 16, 42);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
    }

    #[test]
    fn test_int8_project() {
        let emb = vec![1i8, -1, 2, -2];
        let result = simhash_int8_project(&emb, 32, 42);
        assert_eq!(result.len(), 4);
    }
}
