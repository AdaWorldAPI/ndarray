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
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected, slice bounds respected.
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_unpack_avx2(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    use core::arch::x86_64::*;

    let mask_lo = _mm256_set1_epi8(0x0F);
    let full_bytes = count / 2;
    let chunks = full_bytes / 32;

    // Reserve space
    out.reserve(count);
    let dst = out.as_mut_ptr();

    for c in 0..chunks {
        let src = packed.as_ptr().add(c * 32);
        let data = _mm256_loadu_si256(src as *const __m256i);
        let lo = _mm256_and_si256(data, mask_lo);
        let hi = _mm256_srli_epi16(data, 4);
        let hi = _mm256_and_si256(hi, mask_lo);
        // Interleave: lo[i], hi[i] for each byte
        let interleaved_lo = _mm256_unpacklo_epi8(lo, hi);
        let interleaved_hi = _mm256_unpackhi_epi8(lo, hi);
        // Store (note: unpacklo/hi work on 128-bit lanes so need permute)
        let perm_lo = _mm256_permute4x64_epi64(interleaved_lo, 0b11_01_10_00);
        let perm_hi = _mm256_permute4x64_epi64(interleaved_hi, 0b11_01_10_00);
        _mm256_storeu_si256(dst.add(c * 64) as *mut __m256i, perm_lo);
        _mm256_storeu_si256(dst.add(c * 64 + 32) as *mut __m256i, perm_hi);
    }

    let simd_done = chunks * 64;
    out.set_len(simd_done);

    // Scalar tail
    for i in simd_done..count {
        let byte = packed[i / 2];
        let val = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
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
    let delta_v = _mm256_set1_epi8(delta as i8);
    let chunks = packed.len() / 32;

    for c in 0..chunks {
        let ptr = packed.as_mut_ptr().add(c * 32);
        let data = _mm256_loadu_si256(ptr as *const __m256i);

        let lo = _mm256_and_si256(data, mask_lo);
        let hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask_lo);

        let lo_sub = _mm256_subs_epu8(lo, delta_v);
        let hi_sub = _mm256_subs_epu8(hi, delta_v);

        let result = _mm256_or_si256(lo_sub, _mm256_slli_epi16(hi_sub, 4));
        // Clear any bits that leaked from the shift into adjacent nibbles.
        let clean_hi = _mm256_and_si256(result, _mm256_set1_epi8(0xF0u8 as i8));
        let clean_lo = _mm256_and_si256(result, mask_lo);
        let clean = _mm256_or_si256(clean_lo, clean_hi);

        _mm256_storeu_si256(ptr as *mut __m256i, clean);
    }

    // Scalar tail
    nibble_sub_clamp_scalar(&mut packed[chunks * 32..], delta);
}

/// Find all nibble indices with value strictly above `threshold`. Returns sorted indices.
pub fn nibble_above_threshold(packed: &[u8], threshold: u8) -> Vec<usize> {
    let mut result = Vec::new();
    let count = packed.len() * 2;
    for i in 0..count {
        if nibble_get(packed, i) > threshold {
            result.push(i);
        }
    }
    result
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
}
