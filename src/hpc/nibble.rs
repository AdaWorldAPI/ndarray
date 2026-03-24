//! Nibble batch operations for 4-bit packed data (light levels).
//!
//! Light levels in Minecraft are stored as 4-bit nibbles packed two per byte
//! (low nibble = even index, high nibble = odd index). This module provides
//! SIMD-accelerated batch operations on packed nibble arrays.

/// Unpack 4-bit nibbles from a packed byte array into full `u8` values (0-15).
///
/// Each byte in `packed` holds two nibbles: the low nibble at even index,
/// the high nibble at the subsequent odd index. Exactly `count` values
/// are returned.
///
/// # Panics
/// Panics if `packed.len() < (count + 1) / 2`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::nibble::nibble_unpack;
/// let packed = &[0x3A]; // low=0xA, high=0x3
/// assert_eq!(nibble_unpack(packed, 2), vec![0xA, 0x3]);
/// ```
pub fn nibble_unpack(packed: &[u8], count: usize) -> Vec<u8> {
    assert!(packed.len() >= (count + 1) / 2, "packed buffer too small");

    let mut out = Vec::with_capacity(count);

    #[cfg(target_arch = "x86_64")]
    {
        if count >= 32 && is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected, packed buffer large enough.
            unsafe {
                nibble_unpack_avx2(packed, count, &mut out);
                return out;
            }
        }
    }

    nibble_unpack_scalar(packed, count, &mut out);
    out
}

fn nibble_unpack_scalar(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    for i in 0..count {
        let byte = packed[i / 2];
        let val = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        out.push(val);
    }
}

/// AVX2 nibble unpack: processes 16 packed bytes → 32 nibbles per iteration.
///
/// # Safety
/// Caller must ensure AVX2 is available and `count >= 32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_unpack_avx2(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    use core::arch::x86_64::*;

    let low_mask = _mm_set1_epi8(0x0F);
    let mut i = 0usize; // byte index into packed
    let mut emitted = 0usize;

    // Each iteration: load 16 packed bytes → 32 nibbles
    while emitted + 32 <= count && i + 16 <= packed.len() {
        // SAFETY: i + 16 <= packed.len(), avx2 checked by caller.
        let data = _mm_loadu_si128(packed.as_ptr().add(i) as *const __m128i);

        // Low nibbles (even indices)
        let lo = _mm_and_si128(data, low_mask);
        // High nibbles (odd indices)
        let hi = _mm_and_si128(_mm_srli_epi16(data, 4), low_mask);

        // Interleave: lo[0],hi[0], lo[1],hi[1], ...
        let interleaved_lo = _mm_unpacklo_epi8(lo, hi); // bytes 0-7 → 16 nibbles
        let interleaved_hi = _mm_unpackhi_epi8(lo, hi); // bytes 8-15 → 16 nibbles

        let mut buf = [0u8; 32];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, interleaved_lo);
        _mm_storeu_si128(buf.as_mut_ptr().add(16) as *mut __m128i, interleaved_hi);

        out.extend_from_slice(&buf);
        i += 16;
        emitted += 32;
    }

    // Scalar tail for remaining nibbles
    for idx in emitted..count {
        let byte = packed[idx / 2];
        let val = if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        out.push(val);
    }
}

/// Pack `u8` values (each 0-15) into 4-bit nibble pairs.
///
/// Values are clamped to 0-15. The resulting byte count is `(values.len() + 1) / 2`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::nibble::nibble_pack;
/// let packed = nibble_pack(&[0xA, 0x3]);
/// assert_eq!(packed, vec![0x3A]);
/// ```
pub fn nibble_pack(values: &[u8]) -> Vec<u8> {
    let out_len = (values.len() + 1) / 2;
    let mut out = vec![0u8; out_len];

    for (i, &v) in values.iter().enumerate() {
        let clamped = v & 0x0F;
        let byte_idx = i / 2;
        if i & 1 == 0 {
            out[byte_idx] |= clamped;
        } else {
            out[byte_idx] |= clamped << 4;
        }
    }
    out
}

/// Batch subtract with clamp: every nibble in `packed` has `delta` subtracted,
/// clamping to 0. Used for light propagation BFS decay.
///
/// Operates in-place on the packed representation.
pub fn nibble_sub_clamp(packed: &mut [u8], delta: u8) {
    if delta == 0 {
        return;
    }
    if delta >= 15 {
        for b in packed.iter_mut() {
            *b = 0;
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected, slice is mutable and valid.
            unsafe {
                nibble_sub_clamp_avx512(packed, delta);
                return;
            }
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected, slice is mutable and valid.
            unsafe {
                nibble_sub_clamp_avx2(packed, delta);
                return;
            }
        }
    }

    nibble_sub_clamp_scalar(packed, delta);
}

fn nibble_sub_clamp_scalar(packed: &mut [u8], delta: u8) {
    for b in packed.iter_mut() {
        let lo = (*b & 0x0F).saturating_sub(delta);
        let hi = ((*b >> 4) & 0x0F).saturating_sub(delta);
        *b = lo | (hi << 4);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_sub_clamp_avx2(packed: &mut [u8], delta: u8) {
    use core::arch::x86_64::*;

    let mask_lo = _mm256_set1_epi8(0x0F);
    let mask_hi = _mm256_set1_epi8(0xF0u8 as i8);
    let delta_v = _mm256_set1_epi8(delta as i8);
    // delta shifted into high nibble position for direct subtraction
    let delta_hi = _mm256_set1_epi8((delta << 4) as i8);
    let chunks = packed.len() / 32;

    for c in 0..chunks {
        let ptr = packed.as_mut_ptr().add(c * 32);
        let data = _mm256_loadu_si256(ptr as *const __m256i);

        // Extract low nibbles, subtract with saturation
        let lo = _mm256_and_si256(data, mask_lo);
        let lo_sub = _mm256_subs_epu8(lo, delta_v);

        // Extract high nibbles (keep in high position), subtract with saturation
        let hi = _mm256_and_si256(data, mask_hi);
        let hi_sub = _mm256_subs_epu8(hi, delta_hi);

        // Combine: low nibbles are already clean (0-15), high nibbles already in position
        let result = _mm256_or_si256(
            _mm256_and_si256(lo_sub, mask_lo),
            _mm256_and_si256(hi_sub, mask_hi),
        );

        _mm256_storeu_si256(ptr as *mut __m256i, result);
    }

    // Scalar tail
    nibble_sub_clamp_scalar(&mut packed[chunks * 32..], delta);
}

/// AVX-512 BW nibble sub_clamp: processes 64 bytes (128 nibbles) per iteration.
///
/// # Safety
/// Caller must ensure AVX-512 BW is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn nibble_sub_clamp_avx512(packed: &mut [u8], delta: u8) {
    use core::arch::x86_64::*;

    let mask_lo = _mm512_set1_epi8(0x0F);
    let mask_hi = _mm512_set1_epi8(0xF0u8 as i8);
    let delta_v = _mm512_set1_epi8(delta as i8);
    let delta_hi = _mm512_set1_epi8((delta << 4) as i8);
    let chunks = packed.len() / 64;

    for c in 0..chunks {
        let ptr = packed.as_mut_ptr().add(c * 64);
        // SAFETY: ptr is within bounds (c * 64 + 64 <= packed.len()), avx512bw checked.
        let data = _mm512_loadu_si512(ptr as *const __m512i);

        let lo = _mm512_and_si512(data, mask_lo);
        let lo_sub = _mm512_subs_epu8(lo, delta_v);

        let hi = _mm512_and_si512(data, mask_hi);
        let hi_sub = _mm512_subs_epu8(hi, delta_hi);

        let result = _mm512_or_si512(
            _mm512_and_si512(lo_sub, mask_lo),
            _mm512_and_si512(hi_sub, mask_hi),
        );

        _mm512_storeu_si512(ptr as *mut __m512i, result);
    }

    // Scalar tail
    nibble_sub_clamp_scalar(&mut packed[chunks * 64..], delta);
}

/// Find all nibble indices with value strictly above `threshold`. Returns sorted indices.
pub fn nibble_above_threshold(packed: &[u8], threshold: u8) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if packed.len() >= 16 && is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected, packed buffer large enough.
            return unsafe { nibble_above_threshold_avx2(packed, threshold) };
        }
    }

    nibble_above_threshold_scalar(packed, threshold)
}

fn nibble_above_threshold_scalar(packed: &[u8], threshold: u8) -> Vec<usize> {
    let mut result = Vec::new();
    let count = packed.len() * 2;
    for i in 0..count {
        if nibble_get(packed, i) > threshold {
            result.push(i);
        }
    }
    result
}

/// AVX2 nibble threshold scan: processes 32 packed bytes (64 nibbles) per iteration.
///
/// Splits each byte into lo/hi nibbles, compares against threshold using
/// signed comparison (with bias trick), and extracts matching indices from bitmask.
///
/// # Safety
/// Caller must ensure AVX2 is available and `packed.len() >= 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_above_threshold_avx2(packed: &[u8], threshold: u8) -> Vec<usize> {
    use core::arch::x86_64::*;

    let mut result = Vec::new();
    let low_mask = _mm256_set1_epi8(0x0F);
    // For unsigned comparison via signed: bias both operands by -128
    let bias = _mm256_set1_epi8(-128i8);
    // We want > threshold, which is: (val - 128) > (threshold - 128) via signed cmpgt
    let thresh_lo = _mm256_set1_epi8((threshold as i8).wrapping_add(-128));

    let chunks = packed.len() / 32;
    for c in 0..chunks {
        let base_byte = c * 32;
        // SAFETY: base_byte + 32 <= packed.len(), avx2 checked.
        let data = _mm256_loadu_si256(packed.as_ptr().add(base_byte) as *const __m256i);

        // Extract low nibbles
        let lo = _mm256_and_si256(data, low_mask);
        // Extract high nibbles
        let hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), low_mask);

        // Bias for unsigned compare: add -128 then use signed cmpgt
        let lo_biased = _mm256_add_epi8(lo, bias);
        let hi_biased = _mm256_add_epi8(hi, bias);

        let lo_gt = _mm256_cmpgt_epi8(lo_biased, thresh_lo);
        let hi_gt = _mm256_cmpgt_epi8(hi_biased, thresh_lo);

        let mut lo_mask = _mm256_movemask_epi8(lo_gt) as u32;
        let mut hi_mask = _mm256_movemask_epi8(hi_gt) as u32;

        // Low nibbles are at even indices: byte_index * 2
        while lo_mask != 0 {
            let bit = lo_mask.trailing_zeros() as usize;
            result.push((base_byte + bit) * 2);
            lo_mask &= lo_mask - 1;
        }
        // High nibbles are at odd indices: byte_index * 2 + 1
        while hi_mask != 0 {
            let bit = hi_mask.trailing_zeros() as usize;
            result.push((base_byte + bit) * 2 + 1);
            hi_mask &= hi_mask - 1;
        }
    }

    // Scalar tail
    let tail_start = chunks * 32;
    for byte_idx in tail_start..packed.len() {
        let lo = packed[byte_idx] & 0x0F;
        let hi = packed[byte_idx] >> 4;
        if lo > threshold {
            result.push(byte_idx * 2);
        }
        if hi > threshold {
            result.push(byte_idx * 2 + 1);
        }
    }

    result.sort_unstable();
    result
}

/// Batch BFS decay: subtract `delta` from all nibbles (clamping to 0) and
/// return indices of nibbles that remain non-zero (the propagation frontier).
///
/// This composes `nibble_sub_clamp` and `nibble_above_threshold` — both are
/// SIMD-accelerated when available.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::nibble::{nibble_pack, nibble_propagate_bfs, nibble_unpack};
/// let mut packed = nibble_pack(&[5, 3, 10, 1, 0, 15, 2, 7]);
/// let frontier = nibble_propagate_bfs(&mut packed, 3);
/// // After subtracting 3: [2, 0, 7, 0, 0, 12, 0, 4]
/// // Non-zero indices: 0, 2, 5, 7
/// assert_eq!(frontier, vec![0, 2, 5, 7]);
/// ```
pub fn nibble_propagate_bfs(packed: &mut [u8], delta: u8) -> Vec<usize> {
    nibble_sub_clamp(packed, delta);
    nibble_above_threshold(packed, 0)
}

/// Get a single nibble value at the given index.
///
/// # Panics
/// Panics if `index / 2 >= packed.len()`.
#[inline]
pub fn nibble_get(packed: &[u8], index: usize) -> u8 {
    let byte = packed[index / 2];
    if index & 1 == 0 {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

/// Set a single nibble value at the given index. Value is clamped to 0-15.
///
/// # Panics
/// Panics if `index / 2 >= packed.len()`.
#[inline]
pub fn nibble_set(packed: &mut [u8], index: usize, value: u8) {
    let clamped = value & 0x0F;
    let byte_idx = index / 2;
    if index & 1 == 0 {
        packed[byte_idx] = (packed[byte_idx] & 0xF0) | clamped;
    } else {
        packed[byte_idx] = (packed[byte_idx] & 0x0F) | (clamped << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_pack_unpack() {
        let original: Vec<u8> = (0..16).collect();
        let packed = nibble_pack(&original);
        let unpacked = nibble_unpack(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn test_roundtrip_odd_count() {
        let original = vec![1, 5, 9];
        let packed = nibble_pack(&original);
        let unpacked = nibble_unpack(&packed, 3);
        assert_eq!(unpacked, original);
    }

    #[test]
    fn test_roundtrip_large() {
        let original: Vec<u8> = (0..4096).map(|i| (i % 16) as u8).collect();
        let packed = nibble_pack(&original);
        let unpacked = nibble_unpack(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn test_get_set() {
        let mut packed = vec![0u8; 4]; // 8 nibbles
        for i in 0..8 {
            nibble_set(&mut packed, i, (i as u8) % 16);
        }
        for i in 0..8 {
            assert_eq!(nibble_get(&packed, i), (i as u8) % 16);
        }
    }

    #[test]
    fn test_sub_clamp_basic() {
        // Two bytes: nibbles [5, 3, 10, 1]
        let mut packed = nibble_pack(&[5, 3, 10, 1]);
        nibble_sub_clamp(&mut packed, 3);
        let vals = nibble_unpack(&packed, 4);
        assert_eq!(vals, vec![2, 0, 7, 0]);
    }

    #[test]
    fn test_sub_clamp_zero_delta() {
        let mut packed = nibble_pack(&[5, 3, 10, 1]);
        let original = packed.clone();
        nibble_sub_clamp(&mut packed, 0);
        assert_eq!(packed, original);
    }

    #[test]
    fn test_sub_clamp_large_delta() {
        let mut packed = nibble_pack(&[15, 15, 15, 15]);
        nibble_sub_clamp(&mut packed, 15);
        let vals = nibble_unpack(&packed, 4);
        assert_eq!(vals, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_sub_clamp_large() {
        let original: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        let mut packed = nibble_pack(&original);
        nibble_sub_clamp(&mut packed, 4);
        let result = nibble_unpack(&packed, original.len());
        for (i, (&orig, &res)) in original.iter().zip(result.iter()).enumerate() {
            assert_eq!(res, orig.saturating_sub(4), "mismatch at nibble {}", i);
        }
    }

    #[test]
    fn test_above_threshold() {
        let packed = nibble_pack(&[0, 5, 3, 15, 7, 1, 14, 8]);
        let above_5 = nibble_above_threshold(&packed, 5);
        // Indices with value > 5: index 3 (15), 4 (7), 6 (14), 7 (8)
        assert_eq!(above_5, vec![3, 4, 6, 7]);
    }

    #[test]
    fn test_above_threshold_none() {
        let packed = nibble_pack(&[0, 1, 2, 3]);
        assert!(nibble_above_threshold(&packed, 15).is_empty());
    }

    #[test]
    fn test_above_threshold_all() {
        let packed = nibble_pack(&[15, 15, 15, 15]);
        let above_0 = nibble_above_threshold(&packed, 0);
        assert_eq!(above_0, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_clamping_on_pack() {
        // Values above 15 should be clamped
        let packed = nibble_pack(&[0xFF, 0x1A]);
        let unpacked = nibble_unpack(&packed, 2);
        assert_eq!(unpacked[0], 0x0F);
        assert_eq!(unpacked[1], 0x0A);
    }

    #[test]
    fn test_empty() {
        let packed = nibble_pack(&[]);
        assert!(packed.is_empty());
        let unpacked = nibble_unpack(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_single_nibble() {
        let packed = nibble_pack(&[7]);
        assert_eq!(packed.len(), 1);
        let unpacked = nibble_unpack(&packed, 1);
        assert_eq!(unpacked, vec![7]);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_unpack_too_small() {
        nibble_unpack(&[0x00], 4); // 1 byte can hold 2 nibbles, not 4
    }

    // ---------- AVX2 unpack parity ----------

    #[test]
    fn test_unpack_avx2_matches_scalar() {
        let original: Vec<u8> = (0..4096).map(|i| (i % 16) as u8).collect();
        let packed = nibble_pack(&original);
        let unpacked = nibble_unpack(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn test_unpack_avx2_non_aligned() {
        // Non-multiple of 32 nibbles
        let original: Vec<u8> = (0..47).map(|i| (i % 16) as u8).collect();
        let packed = nibble_pack(&original);
        let unpacked = nibble_unpack(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    // ---------- AVX2 threshold parity ----------

    #[test]
    fn test_above_threshold_avx2_matches_scalar() {
        let original: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        let packed = nibble_pack(&original);
        let result = nibble_above_threshold(&packed, 7);
        let expected: Vec<usize> = original
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 7)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_above_threshold_avx2_all_values() {
        for thresh in 0..16u8 {
            let original: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
            let packed = nibble_pack(&original);
            let result = nibble_above_threshold(&packed, thresh);
            let expected: Vec<usize> = original
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > thresh)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(result, expected, "threshold={thresh}");
        }
    }

    // ---------- AVX-512 sub_clamp parity ----------

    #[test]
    fn test_sub_clamp_avx512_matches_scalar() {
        let original: Vec<u8> = (0..512).map(|i| (i % 16) as u8).collect();
        for delta in 0..16u8 {
            let mut packed = nibble_pack(&original);
            nibble_sub_clamp(&mut packed, delta);
            let result = nibble_unpack(&packed, original.len());
            for (i, (&orig, &res)) in original.iter().zip(result.iter()).enumerate() {
                assert_eq!(
                    res,
                    orig.saturating_sub(delta),
                    "avx512 sub_clamp mismatch at nibble {} (delta={})",
                    i,
                    delta
                );
            }
        }
    }

    // ---------- BFS propagation ----------

    #[test]
    fn test_propagate_bfs_basic() {
        let mut packed = nibble_pack(&[5, 3, 10, 1, 0, 15, 2, 7]);
        let frontier = nibble_propagate_bfs(&mut packed, 3);
        // After subtracting 3: [2, 0, 7, 0, 0, 12, 0, 4]
        assert_eq!(frontier, vec![0, 2, 5, 7]);
    }

    #[test]
    fn test_propagate_bfs_zero_delta() {
        let vals: Vec<u8> = (0..16).collect();
        let mut packed = nibble_pack(&vals);
        let frontier = nibble_propagate_bfs(&mut packed, 0);
        // All non-zero values remain
        let expected: Vec<usize> = (1..16).collect();
        assert_eq!(frontier, expected);
    }

    #[test]
    fn test_propagate_bfs_full_clamp() {
        let mut packed = nibble_pack(&[15, 15, 15, 15]);
        let frontier = nibble_propagate_bfs(&mut packed, 15);
        assert!(frontier.is_empty());
    }
}
