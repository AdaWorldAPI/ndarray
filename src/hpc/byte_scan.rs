//! Byte pattern scanning for NBT tag detection.
//!
//! SIMD-accelerated search for byte values and short patterns in contiguous
//! buffers. All functions operate on borrowed `&[u8]` slices with zero copies.
//! Scalar fallback is provided for non-x86 targets.

// ---------------------------------------------------------------------------
// SIMD (x86_64 SSE2 / AVX2 / AVX-512) internals
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_impl {
    use core::arch::x86_64::*;

    /// Find all positions of `needle` in `haystack` using AVX2 (32 bytes/iter).
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn byte_find_all_avx2(haystack: &[u8], needle: u8) -> Vec<usize> {
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
    pub(crate) unsafe fn byte_find_all_avx512(haystack: &[u8], needle: u8) -> Vec<usize> {
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
    pub(crate) unsafe fn byte_count_avx2(haystack: &[u8], needle: u8) -> usize {
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
    pub(crate) unsafe fn byte_count_avx512(haystack: &[u8], needle: u8) -> usize {
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
        let caps = super::simd_caps::simd_caps();
        if caps.avx512bw {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_find_all_avx512(haystack, needle) };
        }
        if caps.avx2 {
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
        let caps = super::simd_caps::simd_caps();
        if caps.avx512bw {
            // SAFETY: feature detected above.
            return unsafe { simd_impl::byte_count_avx512(haystack, needle) };
        }
        if caps.avx2 {
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
// NBT schema-aware scanning
// ---------------------------------------------------------------------------

/// NBT tag type identifiers (matching Minecraft NBT format).
///
/// Used by the schema scanner to identify tag boundaries in raw NBT data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NbtTagId {
    /// TAG_End (0) — marks the end of a compound tag.
    End = 0,
    /// TAG_Byte (1) — a single signed byte.
    Byte = 1,
    /// TAG_Short (2) — a signed 16-bit integer.
    Short = 2,
    /// TAG_Int (3) — a signed 32-bit integer.
    Int = 3,
    /// TAG_Long (4) — a signed 64-bit integer.
    Long = 4,
    /// TAG_Float (5) — an IEEE 754 single-precision float.
    Float = 5,
    /// TAG_Double (6) — an IEEE 754 double-precision float.
    Double = 6,
    /// TAG_Byte_Array (7) — a length-prefixed array of bytes.
    ByteArray = 7,
    /// TAG_String (8) — a length-prefixed UTF-8 string.
    String = 8,
    /// TAG_List (9) — a typed list of tags.
    List = 9,
    /// TAG_Compound (10) — a set of named tags.
    Compound = 10,
    /// TAG_Int_Array (11) — a length-prefixed array of 32-bit integers.
    IntArray = 11,
    /// TAG_Long_Array (12) — a length-prefixed array of 64-bit integers.
    LongArray = 12,
}

/// A schema entry describing a named NBT tag to locate.
///
/// The scanner searches for the tag name bytes preceded by the tag type byte
/// and a 2-byte big-endian name length.
#[derive(Debug, Clone)]
pub struct NbtSchemaEntry {
    /// Expected tag type.
    pub tag_id: NbtTagId,
    /// Tag name bytes (UTF-8).
    pub name: Vec<u8>,
}

impl NbtSchemaEntry {
    /// Create a schema entry for a named compound tag.
    pub fn compound(name: &str) -> Self {
        Self { tag_id: NbtTagId::Compound, name: name.as_bytes().to_vec() }
    }

    /// Create a schema entry for a named list tag.
    pub fn list(name: &str) -> Self {
        Self { tag_id: NbtTagId::List, name: name.as_bytes().to_vec() }
    }

    /// Create a schema entry for any tag type with given name.
    pub fn new(tag_id: NbtTagId, name: &str) -> Self {
        Self { tag_id, name: name.as_bytes().to_vec() }
    }
}

/// A match from schema scanning: the byte offset where this tag's payload begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NbtSchemaMatch {
    /// Index of the schema entry that matched.
    pub schema_index: usize,
    /// Byte offset of the tag type byte in the buffer.
    pub tag_offset: usize,
    /// Byte offset where the tag's payload begins (after type + name_len + name).
    pub payload_offset: usize,
}

/// Scan a raw NBT byte buffer for multiple named tags simultaneously.
///
/// For each schema entry, searches for the pattern:
/// `[tag_id_byte] [name_len_hi] [name_len_lo] [name_bytes...]`
///
/// Returns all matches found, sorted by offset.
///
/// # Strategy
///
/// 1. Use SIMD `byte_find_all` to locate all occurrences of each unique tag_id byte
/// 2. At each candidate position, verify the name length and name bytes match
/// 3. Record payload offset (position + 1 + 2 + name_len)
///
/// This avoids linear scanning of the entire buffer for each tag.
pub fn nbt_schema_scan(data: &[u8], schema: &[NbtSchemaEntry]) -> Vec<NbtSchemaMatch> {
    let mut matches = Vec::new();

    // Group schema entries by tag_id to avoid redundant SIMD scans.
    // Collect unique tag_id bytes and the schema indices that use each.
    let mut tag_groups: Vec<(u8, Vec<usize>)> = Vec::new();
    for (si, entry) in schema.iter().enumerate() {
        let tid = entry.tag_id as u8;
        if let Some(group) = tag_groups.iter_mut().find(|(t, _)| *t == tid) {
            group.1.push(si);
        } else {
            tag_groups.push((tid, vec![si]));
        }
    }

    for (tid_byte, schema_indices) in &tag_groups {
        // SIMD-accelerated scan for this tag type byte.
        let candidates = byte_find_all(data, *tid_byte);

        for &pos in &candidates {
            // Need at least 3 bytes (tag_id + 2-byte name_len) after pos.
            if pos + 3 > data.len() {
                continue;
            }

            // Read big-endian u16 name length.
            let name_len = u16::from_be_bytes([data[pos + 1], data[pos + 2]]) as usize;

            // Check bounds for the full name.
            if pos + 3 + name_len > data.len() {
                continue;
            }

            let name_slice = &data[pos + 3..pos + 3 + name_len];

            // Check against every schema entry for this tag_id.
            for &si in schema_indices {
                let entry = &schema[si];
                if entry.name.len() == name_len && name_slice == entry.name.as_slice() {
                    matches.push(NbtSchemaMatch {
                        schema_index: si,
                        tag_offset: pos,
                        payload_offset: pos + 3 + name_len,
                    });
                }
            }
        }
    }

    // Sort by tag_offset for deterministic output order.
    matches.sort_by_key(|m| m.tag_offset);
    matches
}

/// Scan multiple NBT buffers against the same schema.
///
/// Returns per-buffer match vectors. Useful for batch region loading
/// where 1024 chunk NBT blobs are processed together.
pub fn nbt_schema_scan_batch(
    buffers: &[&[u8]],
    schema: &[NbtSchemaEntry],
) -> Vec<Vec<NbtSchemaMatch>> {
    buffers.iter().map(|buf| nbt_schema_scan(buf, schema)).collect()
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

    #[test]
    fn test_nbt_schema_scan_basic() {
        // Manually craft an NBT-like buffer with a Compound tag named "Entities"
        // Format: tag_id(1) + name_len(2 BE) + name(N) + payload...
        let mut data = Vec::new();
        // Tag: Compound "Entities"
        data.push(10); // Compound tag id
        data.extend_from_slice(&(8u16).to_be_bytes()); // name length
        data.extend_from_slice(b"Entities"); // name
        data.extend_from_slice(&[0; 10]); // some payload

        let schema = vec![NbtSchemaEntry::compound("Entities")];
        let matches = nbt_schema_scan(&data, &schema);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].schema_index, 0);
        assert_eq!(matches[0].tag_offset, 0);
        assert_eq!(matches[0].payload_offset, 11); // 1 + 2 + 8
    }

    #[test]
    fn test_nbt_schema_scan_multiple_tags() {
        let mut data = Vec::new();
        // Compound "Entities"
        data.push(10);
        data.extend_from_slice(&(8u16).to_be_bytes());
        data.extend_from_slice(b"Entities");
        data.extend_from_slice(&[0; 5]);
        // List "BlockEntities"
        let offset2 = data.len();
        data.push(9); // List
        data.extend_from_slice(&(13u16).to_be_bytes());
        data.extend_from_slice(b"BlockEntities");
        data.extend_from_slice(&[0; 5]);

        let schema = vec![
            NbtSchemaEntry::compound("Entities"),
            NbtSchemaEntry::list("BlockEntities"),
        ];
        let matches = nbt_schema_scan(&data, &schema);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].tag_offset, 0);
        assert_eq!(matches[1].tag_offset, offset2);
    }

    #[test]
    fn test_nbt_schema_scan_no_match() {
        let data = vec![0u8; 100];
        let schema = vec![NbtSchemaEntry::compound("Entities")];
        let matches = nbt_schema_scan(&data, &schema);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_nbt_schema_scan_batch() {
        let buf1 = {
            let mut d = Vec::new();
            d.push(10);
            d.extend_from_slice(&(4u16).to_be_bytes());
            d.extend_from_slice(b"Test");
            d
        };
        let buf2 = vec![0u8; 20]; // no match

        let schema = vec![NbtSchemaEntry::compound("Test")];
        let results = nbt_schema_scan_batch(&[&buf1, &buf2], &schema);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1);
        assert!(results[1].is_empty());
    }

    #[test]
    fn test_nbt_tag_id_values() {
        assert_eq!(NbtTagId::End as u8, 0);
        assert_eq!(NbtTagId::Compound as u8, 10);
        assert_eq!(NbtTagId::LongArray as u8, 12);
    }
}
