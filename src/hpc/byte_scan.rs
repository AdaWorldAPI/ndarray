//! Byte pattern scanning for NBT tag detection.
//!
//! SIMD-accelerated search for byte values and short patterns in contiguous
//! buffers. All functions operate on borrowed `&[u8]` slices with zero copies.
//! Scalar fallback is provided for non-x86 targets.

// ---------------------------------------------------------------------------
// SIMD (x86_64 SSE2 / AVX2 / AVX-512) internals
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod simd_impl {
    use core::arch::x86_64::*;

    /// Find all positions of `needle` in `haystack` using AVX2 (32 bytes/iter).
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn byte_find_all_avx2(haystack: &[u8], needle: u8) -> Vec<usize> {
        let mut result = Vec::new();
        let n = haystack.len();
        let ptr = haystack.as_ptr();
        let needle_v = _mm256_set1_epi8(needle as i8);

        let mut i = 0usize;
        while i + 32 <= n {
            let data = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(data, needle_v);
            let mut mask = _mm256_movemask_epi8(cmp) as u32;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                result.push(i + bit);
                mask &= mask - 1; // clear lowest set bit
            }
            i += 32;
        }
        // Scalar tail
        for j in i..n {
            if *ptr.add(j) == needle {
                result.push(j);
            }
        }
        result
    }

    /// Find all positions of `needle` in `haystack` using AVX-512 BW (64 bytes/iter).
    ///
    /// Uses `_mm512_cmpeq_epi8_mask` which returns a `u64` kmask directly,
    /// avoiding the movemask step needed in AVX2.
    ///
    /// # Safety
    /// Caller must ensure AVX-512 BW is available.
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn byte_find_all_avx512(haystack: &[u8], needle: u8) -> Vec<usize> {
        let mut result = Vec::new();
        let n = haystack.len();
        let ptr = haystack.as_ptr();
        let needle_v = _mm512_set1_epi8(needle as i8);

        let mut i = 0usize;
        while i + 64 <= n {
            // SAFETY: ptr.add(i) is within bounds, avx512bw checked by caller.
            let data = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            let mut mask = _mm512_cmpeq_epi8_mask(data, needle_v);
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                result.push(i + bit);
                mask &= mask - 1;
            }
            i += 64;
        }
        // Scalar tail
        for j in i..n {
            if *ptr.add(j) == needle {
                result.push(j);
            }
        }
        result
    }

    /// Count occurrences of `needle` using AVX2.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn byte_count_avx2(haystack: &[u8], needle: u8) -> usize {
        let n = haystack.len();
        let ptr = haystack.as_ptr();
        let needle_v = _mm256_set1_epi8(needle as i8);
        let mut total = 0usize;

        let mut i = 0usize;
        while i + 32 <= n {
            let data = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(data, needle_v);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            total += mask.count_ones() as usize;
            i += 32;
        }
        for j in i..n {
            if *ptr.add(j) == needle {
                total += 1;
            }
        }
        total
    }

    /// Count occurrences of `needle` using AVX-512 BW (64 bytes/iter).
    ///
    /// # Safety
    /// Caller must ensure AVX-512 BW is available.
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn byte_count_avx512(haystack: &[u8], needle: u8) -> usize {
        let n = haystack.len();
        let ptr = haystack.as_ptr();
        let needle_v = _mm512_set1_epi8(needle as i8);
        let mut total = 0usize;

        let mut i = 0usize;
        while i + 64 <= n {
            // SAFETY: ptr.add(i) is within bounds, avx512bw checked by caller.
            let data = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
            let mask = _mm512_cmpeq_epi8_mask(data, needle_v);
            total += mask.count_ones() as usize;
            i += 64;
        }
        for j in i..n {
            if *ptr.add(j) == needle {
                total += 1;
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find all occurrences of a byte value. Returns indices.
pub fn byte_find_all(haystack: &[u8], needle: u8) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_find_all_avx512(haystack, needle) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_find_all_avx2(haystack, needle) };
        }
    }
    // Scalar fallback
    haystack
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b == needle { Some(i) } else { None })
        .collect()
}

/// Find all occurrences of a 2-byte pattern (big-endian u16). Returns indices
/// of the first byte of each match.
pub fn u16_find_all(haystack: &[u8], pattern: u16) -> Vec<usize> {
    let hi = (pattern >> 8) as u8;
    let lo = (pattern & 0xFF) as u8;
    if haystack.len() < 2 {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..haystack.len() - 1 {
        if haystack[i] == hi && haystack[i + 1] == lo {
            result.push(i);
        }
    }
    result
}

/// Count occurrences of a byte value.
pub fn byte_count(haystack: &[u8], needle: u8) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_count_avx512(haystack, needle) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_count_avx2(haystack, needle) };
        }
    }
    // Scalar fallback
    haystack.iter().filter(|&&b| b == needle).count()
}

/// Find first occurrence of a byte value. Returns index or `None`.
pub fn byte_find_first(haystack: &[u8], needle: u8) -> Option<usize> {
    // memchr-style: the compiler will auto-vectorise this well,
    // but we also have a fast-path via the find-all SIMD path.
    haystack.iter().position(|&b| b == needle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_byte_find_all(haystack: &[u8], needle: u8) -> Vec<usize> {
        haystack
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b == needle { Some(i) } else { None })
            .collect()
    }

    fn naive_byte_count(haystack: &[u8], needle: u8) -> usize {
        haystack.iter().filter(|&&b| b == needle).count()
    }

    #[test]
    fn test_byte_find_all_matches_naive() {
        // Use a buffer that exercises both SIMD and scalar tail.
        let buf: Vec<u8> = (0..200).map(|i| (i % 7) as u8).collect();
        for needle in 0..7u8 {
            assert_eq!(
                byte_find_all(&buf, needle),
                naive_byte_find_all(&buf, needle),
                "mismatch for needle {needle}"
            );
        }
    }

    #[test]
    fn test_byte_count_matches_naive() {
        let buf: Vec<u8> = (0..200).map(|i| (i % 7) as u8).collect();
        for needle in 0..7u8 {
            assert_eq!(
                byte_count(&buf, needle),
                naive_byte_count(&buf, needle),
                "mismatch for needle {needle}"
            );
        }
    }

    #[test]
    fn test_u16_find_all() {
        let buf = [0x00, 0x0A, 0x0B, 0x0A, 0x0B, 0xFF];
        let result = u16_find_all(&buf, 0x0A0B);
        assert_eq!(result, vec![1, 3]);
    }

    #[test]
    fn test_u16_find_all_at_boundary() {
        let buf = [0xAB, 0xCD];
        assert_eq!(u16_find_all(&buf, 0xABCD), vec![0]);
    }

    #[test]
    fn test_byte_find_first_found() {
        let buf = [1, 2, 3, 4, 5];
        assert_eq!(byte_find_first(&buf, 3), Some(2));
    }

    #[test]
    fn test_byte_find_first_not_found() {
        let buf = [1, 2, 3, 4, 5];
        assert_eq!(byte_find_first(&buf, 99), None);
    }

    #[test]
    fn test_empty_haystack() {
        let empty: &[u8] = &[];
        assert!(byte_find_all(empty, 0).is_empty());
        assert_eq!(byte_count(empty, 0), 0);
        assert_eq!(byte_find_first(empty, 0), None);
        assert!(u16_find_all(empty, 0x0000).is_empty());
    }

    #[test]
    fn test_single_byte_haystack() {
        assert_eq!(byte_find_all(&[42], 42), vec![0]);
        assert_eq!(byte_find_all(&[42], 0), Vec::<usize>::new());
        assert!(u16_find_all(&[42], 0x2A00).is_empty());
    }

    #[test]
    fn test_u16_not_found() {
        let buf = [0x00, 0x01, 0x02, 0x03];
        assert!(u16_find_all(&buf, 0xFFFF).is_empty());
    }

    #[test]
    fn test_byte_find_all_avx512_matches_scalar() {
        // Use a buffer large enough to exercise AVX-512 (64-byte) + AVX2 + scalar tail.
        let buf: Vec<u8> = (0..500).map(|i| (i % 7) as u8).collect();
        for needle in 0..7u8 {
            let result = byte_find_all(&buf, needle);
            let expected = naive_byte_find_all(&buf, needle);
            assert_eq!(result, expected, "avx512 find_all mismatch for needle {needle}");
        }
    }

    #[test]
    fn test_byte_count_avx512_matches_scalar() {
        let buf: Vec<u8> = (0..500).map(|i| (i % 7) as u8).collect();
        for needle in 0..7u8 {
            let result = byte_count(&buf, needle);
            let expected = naive_byte_count(&buf, needle);
            assert_eq!(result, expected, "avx512 count mismatch for needle {needle}");
        }
    }

    #[test]
    fn test_byte_find_all_exact_64_boundary() {
        // Exactly 64 bytes: one full AVX-512 register, no tail.
        let buf: Vec<u8> = (0..64).map(|i| if i == 17 { 0xFF } else { 0 }).collect();
        assert_eq!(byte_find_all(&buf, 0xFF), vec![17]);
    }

    #[test]
    fn test_byte_count_exact_64_boundary() {
        let buf = vec![0xABu8; 64];
        assert_eq!(byte_count(&buf, 0xAB), 64);
    }
}
