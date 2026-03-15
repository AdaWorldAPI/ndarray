//! Bitwise operations: Hamming distance, popcount, batch operations.
//!
//! SIMD-accelerated via runtime CPU detection.

use crate::imp_prelude::*;

/// Bitwise distance operations on u8 arrays.
///
/// # Example
///
/// ```
/// use ndarray::prelude::*;
/// use ndarray::hpc::bitwise::BitwiseOps;
///
/// let a = array![0xFFu8, 0x00];
/// let b = array![0x0Fu8, 0xF0];
/// assert_eq!(a.hamming_distance(&b), 8);
/// ```
pub trait BitwiseOps {
    /// Hamming distance: number of differing bits.
    fn hamming_distance(&self, other: &Self) -> u64;

    /// Population count: total number of 1-bits.
    fn popcount(&self) -> u64;

    /// Batch Hamming distance: treats self and other as concatenated vectors
    /// of `vec_len` bytes each, computes Hamming distance for each pair.
    fn hamming_distance_batch(&self, other: &Self, vec_len: usize, count: usize) -> Vec<u64>;

    /// Hamming top-k: find the k nearest vectors by Hamming distance.
    ///
    /// `candidates` is a flat u8 slice of `n_candidates * vec_len` bytes.
    /// Returns `(indices, distances)` sorted by distance ascending.
    fn hamming_top_k(&self, candidates: &[u8], vec_len: usize, k: usize) -> (Vec<usize>, Vec<u64>);
}

fn popcount_scalar(data: &[u8]) -> u64 {
    let mut count = 0u64;
    for &byte in data {
        count += byte.count_ones() as u64;
    }
    count
}

fn hamming_scalar(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let mut count = 0u64;
    for i in 0..n {
        count += (a[i] ^ b[i]).count_ones() as u64;
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut total = 0u64;

    // Lookup table for popcount nibbles
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    let mut inner_count = 0u32;

    while i + 32 <= n {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let xor = _mm256_xor_si256(va, vb);
        let lo = _mm256_and_si256(xor, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);
        let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
        acc = _mm256_add_epi8(acc, _mm256_add_epi8(popcnt_lo, popcnt_hi));
        i += 32;
        inner_count += 1;
        // Prevent overflow of u8 accumulators (max 255/8 ≈ 31 iterations)
        if inner_count >= 30 {
            let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
            let arr: [u64; 4] = core::mem::transmute(sad);
            total += arr[0] + arr[1] + arr[2] + arr[3];
            acc = _mm256_setzero_si256();
            inner_count = 0;
        }
    }

    // Flush accumulator
    if inner_count > 0 {
        let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
        let arr: [u64; 4] = core::mem::transmute(sad);
        total += arr[0] + arr[1] + arr[2] + arr[3];
    }

    // Handle remainder
    while i < n {
        total += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    total
}

fn dispatch_hamming(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: We checked for AVX-512F + VPOPCNTDQ support
            return unsafe { crate::backend::kernels_avx512::hamming_distance(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We checked for AVX2 support
            return unsafe { hamming_avx2(a, b) };
        }
    }
    hamming_scalar(a, b)
}

fn dispatch_popcount(a: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: We checked for AVX-512F + VPOPCNTDQ support
            return unsafe { crate::backend::kernels_avx512::popcount(a) };
        }
    }
    popcount_scalar(a)
}

fn dispatch_hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: We checked for AVX-512F + VPOPCNTDQ support
            return unsafe { crate::backend::kernels_avx512::hamming_batch(query, database, num_rows, row_bytes) };
        }
    }
    // Fallback: per-row dispatch
    (0..num_rows)
        .map(|i| {
            let start = i * row_bytes;
            dispatch_hamming(query, &database[start..start + row_bytes])
        })
        .collect()
}

impl<S> BitwiseOps for ArrayBase<S, Ix1>
where S: Data<Elem = u8>
{
    fn hamming_distance(&self, other: &Self) -> u64 {
        if let (Some(a), Some(b)) = (self.as_slice(), other.as_slice()) {
            dispatch_hamming(a, b)
        } else {
            let n = self.len().min(other.len());
            let mut count = 0u64;
            for i in 0..n {
                count += (self[i] ^ other[i]).count_ones() as u64;
            }
            count
        }
    }

    fn popcount(&self) -> u64 {
        if let Some(s) = self.as_slice() {
            dispatch_popcount(s)
        } else {
            self.iter().map(|&b| b.count_ones() as u64).sum()
        }
    }

    fn hamming_distance_batch(&self, other: &Self, vec_len: usize, count: usize) -> Vec<u64> {
        let a_data = self.as_slice().expect("self must be contiguous");
        let b_data = other.as_slice().expect("other must be contiguous");
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let a_start = i * vec_len;
            let b_start = i * vec_len;
            let a_end = (a_start + vec_len).min(a_data.len());
            let b_end = (b_start + vec_len).min(b_data.len());
            results.push(dispatch_hamming(&a_data[a_start..a_end], &b_data[b_start..b_end]));
        }
        results
    }

    fn hamming_top_k(&self, candidates: &[u8], vec_len: usize, k: usize) -> (Vec<usize>, Vec<u64>) {
        let query = self.as_slice().expect("query must be contiguous");
        let n_candidates = candidates.len() / vec_len;

        // Use batch dispatch for the distance computation
        let distances = dispatch_hamming_batch(query, candidates, n_candidates, vec_len);

        let k = k.min(n_candidates);
        let mut indexed: Vec<(usize, u64)> = distances.into_iter().enumerate().collect();
        indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
        indexed.truncate(k);
        indexed.sort_unstable_by_key(|&(_, d)| d);
        let indices = indexed.iter().map(|&(i, _)| i).collect();
        let dists = indexed.iter().map(|&(_, d)| d).collect();
        (indices, dists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_hamming_distance() {
        let a = array![0xFFu8, 0x00];
        let b = array![0x0Fu8, 0xF0];
        // 0xFF ^ 0x0F = 0xF0 (4 bits), 0x00 ^ 0xF0 = 0xF0 (4 bits)
        assert_eq!(a.hamming_distance(&b), 8);
    }

    #[test]
    fn test_popcount() {
        let a = array![0xFFu8, 0x0F]; // 8 + 4 = 12
        assert_eq!(a.popcount(), 12);
    }

    #[test]
    fn test_hamming_batch() {
        let a = array![0xFFu8, 0x00, 0xAAu8, 0x55];
        let b = array![0x00u8, 0xFF, 0x55u8, 0xAA];
        let dists = a.hamming_distance_batch(&b, 2, 2);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0], 16); // all bits differ
        assert_eq!(dists[1], 16);
    }

    #[test]
    fn test_hamming_top_k() {
        let query = array![0xFFu8, 0xFF];
        let candidates: Vec<u8> = vec![
            0xFF, 0xFF, // dist 0
            0x00, 0x00, // dist 16
            0xFF, 0x00, // dist 8
        ];
        let (indices, dists) = query.hamming_top_k(&candidates, 2, 2);
        assert_eq!(indices, vec![0, 2]);
        assert_eq!(dists, vec![0, 8]);
    }
}
