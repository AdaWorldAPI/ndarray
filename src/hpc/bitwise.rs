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

    /// Query-vs-database batch: compute Hamming distance between `self` (query)
    /// and each row of `database` (flat contiguous slice of `n_candidates * vec_len` bytes).
    ///
    /// This is the hot-path for cascade search — zero allocation, SIMD-accelerated.
    fn hamming_query_batch(&self, database: &[u8], vec_len: usize) -> Vec<u64>;

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

/// AVX-512 BW hamming using 512-bit vpshufb — 64 bytes per iteration.
/// Works on any CPU with avx512bw (no VPOPCNTDQ required).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn hamming_avx512bw(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut total = 0u64;

    // vpshufb LUT: popcount of each nibble (replicated across 64B)
    let lookup = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    let mut inner_count = 0u32;

    while i + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let xor = _mm512_xor_si512(va, vb);

        let lo = _mm512_and_si512(xor, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(xor, 4), low_mask);
        let popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
        acc = _mm512_add_epi8(acc, _mm512_add_epi8(popcnt_lo, popcnt_hi));

        i += 64;
        inner_count += 1;
        // Flush u8 accumulators before overflow (max 255/8 ≈ 31 iterations)
        if inner_count >= 30 {
            // sad_epu8 sums groups of 8 bytes into u64 lanes
            let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
            total += _mm512_reduce_add_epi64(sad) as u64;
            acc = _mm512_setzero_si512();
            inner_count = 0;
        }
    }

    if inner_count > 0 {
        let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
        total += _mm512_reduce_add_epi64(sad) as u64;
    }

    // Remainder
    while i < n {
        total += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    total
}

/// AVX-512 BW popcount using 512-bit vpshufb — 64 bytes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn popcount_avx512bw(a: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut total = 0u64;

    let lookup = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    let mut inner_count = 0u32;

    while i + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let lo = _mm512_and_si512(va, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(va, 4), low_mask);
        let popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
        acc = _mm512_add_epi8(acc, _mm512_add_epi8(popcnt_lo, popcnt_hi));

        i += 64;
        inner_count += 1;
        if inner_count >= 30 {
            let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
            total += _mm512_reduce_add_epi64(sad) as u64;
            acc = _mm512_setzero_si512();
            inner_count = 0;
        }
    }

    if inner_count > 0 {
        let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
        total += _mm512_reduce_add_epi64(sad) as u64;
    }

    while i < n {
        total += a[i].count_ones() as u64;
        i += 1;
    }
    total
}

/// Hamming distance on raw slices — dispatches to VPOPCNTDQ → AVX-512BW → AVX2 → scalar.
///
/// Public API for callers that operate on raw `&[u8]` without ndarray arrays.
pub fn hamming_distance_raw(a: &[u8], b: &[u8]) -> u64 {
    dispatch_hamming(a, b)
}

/// Population count on raw slice.
pub fn popcount_raw(a: &[u8]) -> u64 {
    dispatch_popcount(a)
}

/// Query-vs-database batch Hamming on raw slices — zero allocation.
///
/// `database` is `num_rows * row_bytes` contiguous bytes.
/// Returns a Vec of `num_rows` Hamming distances.
pub fn hamming_batch_raw(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    dispatch_hamming_batch(query, database, num_rows, row_bytes)
}

/// Top-k nearest neighbors by Hamming distance on raw slices.
///
/// Returns (indices, distances) of the k closest rows in the database.
/// Uses `select_nth_unstable` for O(n) partial sort instead of O(n log n).
pub fn hamming_top_k_raw(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    let distances = dispatch_hamming_batch(query, database, num_rows, row_bytes);
    let k = k.min(num_rows);
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut indexed: Vec<(usize, u64)> = distances.into_iter().enumerate().collect();
    indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
    indexed.truncate(k);
    indexed.sort_unstable_by_key(|&(_, d)| d);
    let indices = indexed.iter().map(|&(i, _)| i).collect();
    let dists = indexed.iter().map(|&(_, d)| d).collect();
    (indices, dists)
}

fn dispatch_hamming(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.has_avx512_bw_popcnt() {
            // SAFETY: checked VPOPCNTDQ + BW
            return unsafe { crate::backend::kernels_avx512::hamming_distance(a, b) };
        }
        if caps.avx512bw {
            // SAFETY: checked AVX-512 BW — uses 512-bit vpshufb (64B/iter)
            return unsafe { hamming_avx512bw(a, b) };
        }
        if caps.avx2 {
            // SAFETY: checked AVX2 — uses 256-bit vpshufb (32B/iter)
            return unsafe { hamming_avx2(a, b) };
        }
    }
    hamming_scalar(a, b)
}

fn dispatch_popcount(a: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.avx512vpopcntdq {
            // SAFETY: checked VPOPCNTDQ
            return unsafe { crate::backend::kernels_avx512::popcount(a) };
        }
        if caps.avx512bw {
            // SAFETY: checked AVX-512 BW — uses 512-bit vpshufb
            return unsafe { popcount_avx512bw(a) };
        }
    }
    popcount_scalar(a)
}

fn dispatch_hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.has_avx512_bw_popcnt() {
            // SAFETY: checked VPOPCNTDQ + BW
            return unsafe { crate::backend::kernels_avx512::hamming_batch(query, database, num_rows, row_bytes) };
        }
    }
    // Fallback: per-row dispatch (will pick avx512bw or avx2 per row)
    (0..num_rows)
        .map(|i| {
            let start = i * row_bytes;
            dispatch_hamming(query, &database[start..start + row_bytes])
        })
        .collect()
}

/// Count set bits across an array of u64 words.
/// More efficient than reinterpreting as bytes — works on native u64s directly.
pub fn popcount_batch_u64(words: &[u64]) -> u64 {
    // Use POPCNT instruction if available, else scalar
    words.iter().map(|w| w.count_ones() as u64).sum()
}

/// Per-word popcount: returns count of set bits in each u64.
pub fn popcount_per_word(words: &[u64]) -> Vec<u32> {
    words.iter().map(|w| w.count_ones()).collect()
}

/// Batch AND + popcount: for each word, compute (word & mask).count_ones().
/// Used for "count blocks matching a property mask in each palette group."
pub fn masked_popcount_batch(words: &[u64], mask: u64) -> Vec<u32> {
    words.iter().map(|w| (w & mask).count_ones()).collect()
}

/// Total masked popcount across all words.
pub fn masked_popcount_total(words: &[u64], mask: u64) -> u64 {
    words.iter().map(|w| (w & mask).count_ones() as u64).sum()
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
        // Pairwise: compute hamming(a[i], b[i]) for i in 0..count
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

    fn hamming_query_batch(&self, database: &[u8], vec_len: usize) -> Vec<u64> {
        let query = self.as_slice().expect("query must be contiguous");
        let n_candidates = database.len() / vec_len;
        dispatch_hamming_batch(query, database, n_candidates, vec_len)
    }

    fn hamming_top_k(&self, candidates: &[u8], vec_len: usize, k: usize) -> (Vec<usize>, Vec<u64>) {
        let query = self.as_slice().expect("query must be contiguous");
        let n_candidates = candidates.len() / vec_len;
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

    #[test]
    fn test_hamming_query_batch() {
        let query = crate::Array1::from_vec(vec![0xAAu8; 16]);
        let mut database = vec![0u8; 16 * 4];
        database[..16].fill(0xAA); // row 0: identical → 0
        database[16..32].fill(0x55); // row 1: all diff → 128
        database[32..48].fill(0xAA); // row 2: identical → 0
        database[48..64].fill(0x00); // row 3: half diff → 64
        let dists = query.hamming_query_batch(&database, 16);
        assert_eq!(dists.len(), 4);
        assert_eq!(dists[0], 0);
        assert_eq!(dists[1], 128);
        assert_eq!(dists[2], 0);
        assert_eq!(dists[3], 64);
    }

    #[test]
    fn test_raw_slice_apis() {
        let a = vec![0xFFu8; 64];
        let b = vec![0x0Fu8; 64];
        assert_eq!(super::hamming_distance_raw(&a, &b), 64 * 4); // 4 bits diff per byte
        assert_eq!(super::popcount_raw(&a), 64 * 8);

        let query = vec![0xAAu8; 32];
        let mut db = vec![0xAAu8; 32 * 3];
        db[32] = 0x55; // row 1: 1 byte diff → 8
        let dists = super::hamming_batch_raw(&query, &db, 3, 32);
        assert_eq!(dists[0], 0);
        assert_eq!(dists[1], 8);
        assert_eq!(dists[2], 0);
    }

    // ── Per-tier hamming correctness tests ──────────────────────────
    //
    // These call each kernel directly, bypassing dispatch, to verify
    // all 4 tiers produce identical results.

    /// Generate deterministic pseudo-random test data.
    fn test_data(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| ((i as u8).wrapping_mul(7).wrapping_add(seed).wrapping_mul(13)) ^ (i as u8)).collect()
    }

    /// Scalar reference — always correct, used to verify SIMD tiers.
    fn reference_hamming(a: &[u8], b: &[u8]) -> u64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x ^ y).count_ones() as u64).sum()
    }

    fn reference_popcount(a: &[u8]) -> u64 {
        a.iter().map(|&x| x.count_ones() as u64).sum()
    }

    #[test]
    fn test_tier_scalar_hamming() {
        for &n in &[0, 1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 255, 1024, 8192] {
            let a = test_data(n, 0xAA);
            let b = test_data(n, 0x55);
            let expected = reference_hamming(&a, &b);
            let got = hamming_scalar(&a, &b);
            assert_eq!(got, expected, "scalar hamming failed at n={}", n);
        }
    }

    #[test]
    fn test_tier_scalar_popcount() {
        for &n in &[0, 1, 7, 64, 128, 1024, 8192] {
            let a = test_data(n, 0xBB);
            let expected = reference_popcount(&a);
            let got = popcount_scalar(&a);
            assert_eq!(got, expected, "scalar popcount failed at n={}", n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tier_avx2_hamming() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("SKIP: AVX2 not available");
            return;
        }
        for &n in &[0, 1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1024, 4096, 8192] {
            let a = test_data(n, 0xAA);
            let b = test_data(n, 0x55);
            let expected = reference_hamming(&a, &b);
            let got = unsafe { hamming_avx2(&a, &b) };
            assert_eq!(got, expected, "AVX2 hamming failed at n={}", n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tier_avx512bw_hamming() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("SKIP: AVX-512 BW not available");
            return;
        }
        for &n in &[0, 1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1024, 4096, 8192, 65536] {
            let a = test_data(n, 0xAA);
            let b = test_data(n, 0x55);
            let expected = reference_hamming(&a, &b);
            let got = unsafe { hamming_avx512bw(&a, &b) };
            assert_eq!(got, expected, "AVX-512 BW hamming failed at n={}", n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tier_avx512bw_popcount() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("SKIP: AVX-512 BW not available");
            return;
        }
        for &n in &[0, 1, 7, 63, 64, 65, 128, 1024, 8192, 65536] {
            let a = test_data(n, 0xCC);
            let expected = reference_popcount(&a);
            let got = unsafe { popcount_avx512bw(&a) };
            assert_eq!(got, expected, "AVX-512 BW popcount failed at n={}", n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tier_vpopcntdq_hamming() {
        if !(is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw")) {
            eprintln!("SKIP: VPOPCNTDQ not available");
            return;
        }
        for &n in &[0, 1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1024, 4096, 8192, 65536] {
            let a = test_data(n, 0xAA);
            let b = test_data(n, 0x55);
            let expected = reference_hamming(&a, &b);
            let got = unsafe { crate::backend::kernels_avx512::hamming_distance(&a, &b) };
            assert_eq!(got, expected, "VPOPCNTDQ hamming failed at n={}", n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tier_vpopcntdq_popcount() {
        if !is_x86_feature_detected!("avx512vpopcntdq") {
            eprintln!("SKIP: VPOPCNTDQ not available");
            return;
        }
        for &n in &[0, 1, 7, 63, 64, 65, 128, 1024, 8192, 65536] {
            let a = test_data(n, 0xDD);
            let expected = reference_popcount(&a);
            let got = unsafe { crate::backend::kernels_avx512::popcount(&a) };
            assert_eq!(got, expected, "VPOPCNTDQ popcount failed at n={}", n);
        }
    }

    /// Cross-tier consistency: all available tiers must produce identical results.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_all_tiers_agree() {
        let sizes = [0, 1, 3, 7, 15, 16, 31, 32, 33, 63, 64, 65,
                     127, 128, 129, 255, 256, 512, 1024, 2048, 4096, 8192];

        for &n in &sizes {
            let a = test_data(n, 0x42);
            let b = test_data(n, 0x99);
            let scalar = hamming_scalar(&a, &b);

            if is_x86_feature_detected!("avx2") {
                let avx2 = unsafe { hamming_avx2(&a, &b) };
                assert_eq!(scalar, avx2,
                    "scalar≠avx2 at n={}: {} vs {}", n, scalar, avx2);
            }
            if is_x86_feature_detected!("avx512bw") {
                let bw = unsafe { hamming_avx512bw(&a, &b) };
                assert_eq!(scalar, bw,
                    "scalar≠avx512bw at n={}: {} vs {}", n, scalar, bw);
            }
            if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw") {
                let vpc = unsafe { crate::backend::kernels_avx512::hamming_distance(&a, &b) };
                assert_eq!(scalar, vpc,
                    "scalar≠vpopcntdq at n={}: {} vs {}", n, scalar, vpc);
            }
        }
    }

    /// Cross-tier consistency for popcount.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_all_tiers_agree_popcount() {
        let sizes = [0, 1, 7, 15, 32, 63, 64, 65, 128, 256, 1024, 8192];

        for &n in &sizes {
            let a = test_data(n, 0x77);
            let scalar = popcount_scalar(&a);

            if is_x86_feature_detected!("avx512bw") {
                let bw = unsafe { popcount_avx512bw(&a) };
                assert_eq!(scalar, bw,
                    "popcount scalar≠avx512bw at n={}: {} vs {}", n, scalar, bw);
            }
            if is_x86_feature_detected!("avx512vpopcntdq") {
                let vpc = unsafe { crate::backend::kernels_avx512::popcount(&a) };
                assert_eq!(scalar, vpc,
                    "popcount scalar≠vpopcntdq at n={}: {} vs {}", n, scalar, vpc);
            }
        }
    }

    /// Stress test: large data at all boundaries (catches accumulator overflow bugs).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_large_hamming_all_tiers() {
        // 64KB = fingerprint size. All bits different → max hamming.
        let n = 65536;
        let a = vec![0xAAu8; n];
        let b = vec![0x55u8; n]; // XOR = 0xFF → 8 bits per byte
        let expected = n as u64 * 8;

        assert_eq!(hamming_scalar(&a, &b), expected, "scalar large");

        if is_x86_feature_detected!("avx2") {
            assert_eq!(unsafe { hamming_avx2(&a, &b) }, expected, "avx2 large");
        }
        if is_x86_feature_detected!("avx512bw") {
            assert_eq!(unsafe { hamming_avx512bw(&a, &b) }, expected, "avx512bw large");
        }
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw") {
            assert_eq!(
                unsafe { crate::backend::kernels_avx512::hamming_distance(&a, &b) },
                expected, "vpopcntdq large"
            );
        }
    }

    #[test]
    fn test_popcount_batch_u64() {
        let words = [0xFFFFFFFFFFFFFFFFu64, 0, 0x0F0F0F0F0F0F0F0F];
        assert_eq!(super::popcount_batch_u64(&words), 64 + 0 + 32);
    }

    #[test]
    fn test_popcount_per_word() {
        let words = [0xFFu64, 0xFFFF, 0];
        let counts = super::popcount_per_word(&words);
        assert_eq!(counts, vec![8, 16, 0]);
    }

    #[test]
    fn test_masked_popcount() {
        let words = [0xFFu64, 0xFF00, 0xFFFF];
        let mask = 0xFF;
        assert_eq!(super::masked_popcount_batch(&words, mask), vec![8, 0, 8]);
        assert_eq!(super::masked_popcount_total(&words, mask), 16);
    }

    /// Edge: identical vectors → distance 0 at all tiers.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_identical_all_tiers() {
        for &n in &[0, 1, 64, 128, 8192] {
            let a = test_data(n, 0xEE);
            let b = a.clone();

            assert_eq!(hamming_scalar(&a, &b), 0, "scalar identical n={}", n);
            if is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { hamming_avx2(&a, &b) }, 0, "avx2 identical n={}", n);
            }
            if is_x86_feature_detected!("avx512bw") {
                assert_eq!(unsafe { hamming_avx512bw(&a, &b) }, 0, "bw identical n={}", n);
            }
            if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512bw") {
                assert_eq!(
                    unsafe { crate::backend::kernels_avx512::hamming_distance(&a, &b) },
                    0, "vpc identical n={}", n
                );
            }
        }
    }
}
