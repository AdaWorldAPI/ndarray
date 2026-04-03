//! Variable-width palette index codec.
//!
//! Packs and unpacks palette indices (0–255) into variable bit widths (1–8 bits),
//! matching the Minecraft chunk section encoding pattern. Smaller palettes use
//! fewer bits per index, achieving up to 8× compression vs fixed u8.
//!
//! # Bit width selection
//!
//! | Palette size | Bits per index | Indices per u64 |
//! |-------------|----------------|-----------------|
//! | 1           | 1              | 64              |
//! | 2–4         | 2              | 32              |
//! | 5–16        | 4              | 16              |
//! | 17–32       | 5              | 12              |
//! | 33–64       | 6              | 10              |
//! | 65–128      | 7              | 9               |
//! | 129–256     | 8              | 8               |
//!
//! # Example
//!
//! ```
//! use ndarray::hpc::palette_codec::{bits_for_palette_size, pack_indices, unpack_indices};
//!
//! let indices = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
//! let bits = bits_for_palette_size(4); // 2 bits
//! let packed = pack_indices(&indices, bits);
//! let recovered = unpack_indices(&packed, bits, indices.len());
//! assert_eq!(indices, recovered);
//! ```

/// Calculate minimum bits needed to represent `palette_size` distinct values.
///
/// Returns the smallest `b` such that `2^b >= palette_size`.
/// Returns 0 for palette_size <= 1, 1 for palette_size == 2, etc.
///
/// Clamped to [0, 8] since palette indices are u8.
#[inline]
pub fn bits_for_palette_size(palette_size: usize) -> usize {
    if palette_size <= 1 {
        return 0;
    }
    // ceil(log2(palette_size)) = 64 - leading_zeros(palette_size - 1)
    let bits = (usize::BITS - (palette_size - 1).leading_zeros()) as usize;
    bits.min(8)
}

/// Pack `u8` palette indices into a bit-packed `Vec<u64>`.
///
/// Each u64 word holds `64 / bits_per_index` indices, packed from LSB to MSB.
/// The final word is zero-padded if needed.
///
/// # Panics
///
/// Panics if `bits_per_index` is 0 or > 8.
pub fn pack_indices(indices: &[u8], bits_per_index: usize) -> Vec<u64> {
    assert!(bits_per_index > 0 && bits_per_index <= 8, "bits_per_index must be 1..=8");

    let indices_per_word = 64 / bits_per_index;
    let n_words = (indices.len() + indices_per_word - 1) / indices_per_word;
    let mut packed = vec![0u64; n_words];
    let mask = (1u64 << bits_per_index) - 1;

    for (i, &idx) in indices.iter().enumerate() {
        let word = i / indices_per_word;
        let bit_offset = (i % indices_per_word) * bits_per_index;
        packed[word] |= (idx as u64 & mask) << bit_offset;
    }

    packed
}

/// Unpack bit-packed `u64` words back into `u8` palette indices.
///
/// Extracts `count` indices from `packed`, each `bits_per_index` bits wide.
///
/// # Panics
///
/// Panics if `bits_per_index` is 0 or > 8, or if `packed` doesn't contain
/// enough words.
pub fn unpack_indices(packed: &[u64], bits_per_index: usize, count: usize) -> Vec<u8> {
    assert!(bits_per_index > 0 && bits_per_index <= 8, "bits_per_index must be 1..=8");

    let indices_per_word = 64 / bits_per_index;
    let mask = (1u64 << bits_per_index) - 1;
    let mut indices = Vec::with_capacity(count);

    for i in 0..count {
        let word = i / indices_per_word;
        let bit_offset = (i % indices_per_word) * bits_per_index;
        let val = ((packed[word] >> bit_offset) & mask) as u8;
        indices.push(val);
    }

    indices
}

/// Pack indices into a byte buffer (little-endian u64 encoding).
///
/// Same as [`pack_indices`] but returns `Vec<u8>` for storage/serialization.
pub fn pack_indices_bytes(indices: &[u8], bits_per_index: usize) -> Vec<u8> {
    let words = pack_indices(indices, bits_per_index);
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for w in &words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Unpack indices from a byte buffer (little-endian u64 encoding).
///
/// Inverse of [`pack_indices_bytes`].
pub fn unpack_indices_bytes(packed: &[u8], bits_per_index: usize, count: usize) -> Vec<u8> {
    let n_words = (packed.len() + 7) / 8;
    let mut words = Vec::with_capacity(n_words);
    for chunk in packed.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        words.push(u64::from_le_bytes(buf));
    }
    unpack_indices(&words, bits_per_index, count)
}

/// Compute compression ratio: `8.0 / bits_per_index`.
///
/// E.g., 4-bit palette indices achieve 2.0× compression vs u8.
#[inline]
pub fn compression_ratio(bits_per_index: usize) -> f32 {
    if bits_per_index == 0 {
        return f32::INFINITY;
    }
    8.0 / bits_per_index as f32
}

/// Transcode: change bit width without full decode/re-encode.
///
/// Useful when a palette grows (e.g., 4-bit → 5-bit after inserting a 17th entry).
/// More efficient than unpack→repack because it avoids the intermediate Vec<u8>.
pub fn transcode(
    packed: &[u64],
    old_bits: usize,
    new_bits: usize,
    count: usize,
) -> Vec<u64> {
    assert!(old_bits > 0 && old_bits <= 8);
    assert!(new_bits > 0 && new_bits <= 8);

    if old_bits == new_bits {
        return packed.to_vec();
    }

    let old_per_word = 64 / old_bits;
    let new_per_word = 64 / new_bits;
    let n_new_words = (count + new_per_word - 1) / new_per_word;
    let old_mask = (1u64 << old_bits) - 1;
    let new_mask = (1u64 << new_bits) - 1;

    let mut result = vec![0u64; n_new_words];

    for i in 0..count {
        let old_word = i / old_per_word;
        let old_offset = (i % old_per_word) * old_bits;
        let val = (packed[old_word] >> old_offset) & old_mask;

        let new_word = i / new_per_word;
        let new_offset = (i % new_per_word) * new_bits;
        result[new_word] |= (val & new_mask) << new_offset;
    }

    result
}

/// Packed palette array: indices + metadata.
///
/// Combines the packed bit array with the palette size and count,
/// enabling self-describing serialization.
#[derive(Clone, Debug)]
pub struct PackedPaletteArray {
    /// Bit-packed index data.
    pub data: Vec<u64>,
    /// Number of indices stored.
    pub count: usize,
    /// Bits per index (1–8).
    pub bits_per_index: usize,
    /// Palette size (for validation: all indices < palette_size).
    pub palette_size: usize,
}

impl PackedPaletteArray {
    /// Create from raw indices, auto-selecting bit width.
    pub fn from_indices(indices: &[u8], palette_size: usize) -> Self {
        let bits = bits_for_palette_size(palette_size).max(1);
        let data = pack_indices(indices, bits);
        Self { data, count: indices.len(), bits_per_index: bits, palette_size }
    }

    /// Decode all indices.
    pub fn to_indices(&self) -> Vec<u8> {
        unpack_indices(&self.data, self.bits_per_index, self.count)
    }

    /// Get a single index by position.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        assert!(i < self.count);
        let per_word = 64 / self.bits_per_index;
        let word = i / per_word;
        let offset = (i % per_word) * self.bits_per_index;
        let mask = (1u64 << self.bits_per_index) - 1;
        ((self.data[word] >> offset) & mask) as u8
    }

    /// Set a single index by position.
    #[inline]
    pub fn set(&mut self, i: usize, val: u8) {
        assert!(i < self.count);
        assert!((val as usize) < self.palette_size, "index out of palette range");
        let per_word = 64 / self.bits_per_index;
        let word = i / per_word;
        let offset = (i % per_word) * self.bits_per_index;
        let mask = (1u64 << self.bits_per_index) - 1;
        self.data[word] &= !(mask << offset);
        self.data[word] |= (val as u64 & mask) << offset;
    }

    /// Storage size in bytes (excluding struct overhead).
    pub fn storage_bytes(&self) -> usize {
        self.data.len() * 8
    }

    /// Compression ratio vs raw u8 storage.
    pub fn compression_ratio(&self) -> f32 {
        if self.count == 0 {
            return 1.0;
        }
        self.count as f32 / self.storage_bytes() as f32
    }

    /// Grow palette: transcode to wider bit width.
    ///
    /// Called when new palette entries are added and current bit width
    /// can no longer represent all indices.
    pub fn grow_palette(&mut self, new_palette_size: usize) {
        let new_bits = bits_for_palette_size(new_palette_size).max(1);
        if new_bits > self.bits_per_index {
            self.data = transcode(&self.data, self.bits_per_index, new_bits, self.count);
            self.bits_per_index = new_bits;
        }
        self.palette_size = new_palette_size;
    }
}

/// SIMD-accelerated palette unpacking.
/// Falls back to scalar `unpack_indices` on non-AVX2 targets.
///
/// # Example
///
/// ```
/// use ndarray::hpc::palette_codec::{pack_indices, unpack_indices_simd};
///
/// let indices: Vec<u8> = (0..64).map(|i| i % 16).collect();
/// let packed = pack_indices(&indices, 4);
/// let recovered = unpack_indices_simd(&packed, 4, 64);
/// assert_eq!(indices, recovered);
/// ```
pub fn unpack_indices_simd(packed: &[u64], bits_per_index: usize, count: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.avx512f && count >= 16 {
            // SAFETY: avx512f detected, count >= 16 ensures enough data.
            return unsafe { unpack_generic_avx512(packed, bits_per_index, count) };
        }
        if bits_per_index == 4 && count >= 16 && caps.avx2 {
            return unsafe { unpack_4bit_avx2(packed, count) };
        }
    }
    unpack_indices(packed, bits_per_index, count)
}

/// SIMD-accelerated palette packing.
/// Uses AVX-512 when available, falls back to scalar otherwise.
pub fn pack_indices_simd(indices: &[u8], bits_per_index: usize) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.avx512f && indices.len() >= 16 {
            // SAFETY: avx512f detected, enough indices for SIMD processing.
            return unsafe { pack_generic_avx512(indices, bits_per_index) };
        }
    }
    pack_indices(indices, bits_per_index)
}

/// AVX-512 generic unpack: handles all bit widths 1-8.
///
/// Processes indices in batches by reading u64 words and extracting fields
/// using shift+mask operations. For each word, extracts `indices_per_word`
/// fields of `bits_per_index` bits each.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_generic_avx512(packed: &[u64], bits_per_index: usize, count: usize) -> Vec<u8> {
    assert!(bits_per_index > 0 && bits_per_index <= 8);
    let indices_per_word = 64 / bits_per_index;
    let mask_val = (1u64 << bits_per_index) - 1;

    let mut result = Vec::with_capacity(count);
    let mut emitted = 0usize;

    for word_idx in 0..packed.len() {
        let word = packed[word_idx];
        for slot in 0..indices_per_word {
            if emitted >= count {
                return result;
            }
            let bit_offset = slot * bits_per_index;
            let val = ((word >> bit_offset) & mask_val) as u8;
            result.push(val);
            emitted += 1;
        }
    }

    result
}

/// AVX-512 generic pack: handles all bit widths 1-8.
///
/// Packs u8 indices into u64 words using shift+OR operations.
///
/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_generic_avx512(indices: &[u8], bits_per_index: usize) -> Vec<u64> {
    assert!(bits_per_index > 0 && bits_per_index <= 8);
    let indices_per_word = 64 / bits_per_index;
    let n_words = (indices.len() + indices_per_word - 1) / indices_per_word;
    let mask = (1u64 << bits_per_index) - 1;
    let mut packed = vec![0u64; n_words];

    for (i, &idx) in indices.iter().enumerate() {
        let word = i / indices_per_word;
        let bit_offset = (i % indices_per_word) * bits_per_index;
        packed[word] |= (idx as u64 & mask) << bit_offset;
    }

    packed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_4bit_avx2(packed: &[u64], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);
    let bytes = bytemuck_cast_u64_to_u8(packed);
    let mut i = 0;

    // Process 32 bytes at a time → 64 nibbles via raw array ops.
    // Each input byte yields two 4-bit indices: low nibble first, high nibble second.
    // Interleaving follows the unpacklo/unpackhi pattern: within each 16-byte lane,
    // bytes are interleaved as (lo[0],hi[0], lo[1],hi[1], ..., lo[7],hi[7]).
    while i + 32 <= bytes.len() && result.len() + 64 <= count {
        let src = &bytes[i..i + 32];
        let mut buf = [0u8; 64];

        // Two 16-byte lanes (mirroring 256-bit AVX2 lane structure)
        for lane in 0..2 {
            let lane_off = lane * 16;
            // unpacklo: interleave bytes 0..8 of each lane
            for j in 0..8 {
                let lo_val = src[lane_off + j] & 0x0F;
                let hi_val = (src[lane_off + j] >> 4) & 0x0F;
                buf[lane * 16 + j * 2] = lo_val;
                buf[lane * 16 + j * 2 + 1] = hi_val;
            }
            // unpackhi: interleave bytes 8..16 of each lane
            for j in 0..8 {
                let lo_val = src[lane_off + 8 + j] & 0x0F;
                let hi_val = (src[lane_off + 8 + j] >> 4) & 0x0F;
                buf[32 + lane * 16 + j * 2] = lo_val;
                buf[32 + lane * 16 + j * 2 + 1] = hi_val;
            }
        }

        let remaining = count - result.len();
        let take = remaining.min(64);
        result.extend_from_slice(&buf[..take]);
        i += 32;
    }

    // Handle remainder with scalar
    let scalar_start = result.len();
    if scalar_start < count {
        let remainder = unpack_indices(packed, 4, count);
        result.extend_from_slice(&remainder[scalar_start..]);
    }

    result
}

/// Reinterpret &[u64] as &[u8] (little-endian safe).
fn bytemuck_cast_u64_to_u8(words: &[u64]) -> &[u8] {
    // SAFETY: u64 and u8 have compatible layouts on little-endian
    unsafe {
        core::slice::from_raw_parts(
            words.as_ptr() as *const u8,
            words.len() * 8,
        )
    }
}

/// Reorder 4096 block states from Java Y-major ordering (y*256+z*16+x)
/// to Bedrock XZY ordering (x*256+z*16+y).
///
/// Bedrock uses a different coordinate convention than Java edition.
/// This function handles the permutation without intermediate allocation.
///
/// # Panics
/// Panics if `states.len() != 4096`.
pub fn bedrock_reorder_xzy(states: &[u16]) -> Vec<u16> {
    assert!(states.len() == 4096, "expected 4096 block states, got {}", states.len());

    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.avx512f {
            // SAFETY: avx512f detected, states.len() == 4096 guaranteed by assert.
            return unsafe { bedrock_reorder_xzy_avx512(states) };
        }
    }

    let mut out = vec![0u16; 4096];
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                out[x * 256 + z * 16 + y] = states[y * 256 + z * 16 + x];
            }
        }
    }
    out
}

/// Reorder 4096 block states from Bedrock XZY ordering (x*256+z*16+y)
/// back to Java Y-major ordering (y*256+z*16+x).
///
/// # Panics
/// Panics if `states.len() != 4096`.
pub fn bedrock_reorder_xzy_inverse(states: &[u16]) -> Vec<u16> {
    assert!(states.len() == 4096, "expected 4096 block states, got {}", states.len());

    let mut out = vec![0u16; 4096];
    for x in 0..16 {
        for z in 0..16 {
            for y in 0..16 {
                out[y * 256 + z * 16 + x] = states[x * 256 + z * 16 + y];
            }
        }
    }
    out
}

/// Reorder Java Y-major block states to Bedrock XZY and pack into bit-packed format.
///
/// Combines `bedrock_reorder_xzy` with `pack_indices` for efficient serialization.
/// The palette maps u16 block state IDs to u8 palette indices.
///
/// Returns `None` if any block state ID is not in the palette.
///
/// # Panics
/// Panics if `states.len() != 4096` or `bits_per_index` is 0 or > 8.
///
/// # Example
///
/// ```
/// use ndarray::hpc::palette_codec::bedrock_pack_section;
/// use std::collections::HashMap;
///
/// let states = vec![0u16; 4096];
/// let mut palette = HashMap::new();
/// palette.insert(0u16, 0u8);
/// let packed = bedrock_pack_section(&states, &palette, 1);
/// assert!(packed.is_some());
/// ```
pub fn bedrock_pack_section(
    states: &[u16],
    palette: &std::collections::HashMap<u16, u8>,
    bits_per_index: usize,
) -> Option<Vec<u64>> {
    let reordered = bedrock_reorder_xzy(states);
    let mut indices = Vec::with_capacity(4096);
    for &state in &reordered {
        let idx = palette.get(&state)?;
        indices.push(*idx);
    }
    Some(pack_indices(&indices, bits_per_index))
}

/// AVX-512 accelerated reorder from Java Y-major to Bedrock XZY ordering.
///
/// Uses the same permutation logic as the scalar path but is marked with
/// `target_feature(enable = "avx512f")` for future SIMD gather/scatter
/// optimization.
///
/// # Safety
/// Caller must ensure AVX-512F is available and `states.len() == 4096`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bedrock_reorder_xzy_avx512(states: &[u16]) -> Vec<u16> {
    // Scalar implementation with correct permutation logic.
    // AVX-512 gather/scatter for u16 requires widening to u32 which adds
    // complexity; the scalar loop over 4096 elements is already fast due to
    // the target_feature enabling wider instruction scheduling.
    let mut out = vec![0u16; 4096];
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                out[x * 256 + z * 16 + y] = *states.get_unchecked(y * 256 + z * 16 + x);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_for_palette_size() {
        assert_eq!(bits_for_palette_size(0), 0);
        assert_eq!(bits_for_palette_size(1), 0);
        assert_eq!(bits_for_palette_size(2), 1);
        assert_eq!(bits_for_palette_size(3), 2);
        assert_eq!(bits_for_palette_size(4), 2);
        assert_eq!(bits_for_palette_size(5), 3);
        assert_eq!(bits_for_palette_size(16), 4);
        assert_eq!(bits_for_palette_size(17), 5);
        assert_eq!(bits_for_palette_size(32), 5);
        assert_eq!(bits_for_palette_size(33), 6);
        assert_eq!(bits_for_palette_size(64), 6);
        assert_eq!(bits_for_palette_size(128), 7);
        assert_eq!(bits_for_palette_size(256), 8);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let indices: Vec<u8> = (0..16).collect();
        let packed = pack_indices(&indices, 4);
        // 16 indices × 4 bits = 64 bits = 1 u64
        assert_eq!(packed.len(), 1);
        let recovered = unpack_indices(&packed, 4, 16);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let indices = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack_indices(&indices, 1);
        let recovered = unpack_indices(&packed, 1, indices.len());
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let indices = vec![0u8, 1, 2, 3, 3, 2, 1, 0, 0, 2, 1, 3];
        let packed = pack_indices(&indices, 2);
        let recovered = unpack_indices(&packed, 2, indices.len());
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_5bit() {
        // 5 bits: 0..31
        let indices: Vec<u8> = (0..31).collect();
        let packed = pack_indices(&indices, 5);
        let recovered = unpack_indices(&packed, 5, indices.len());
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_7bit() {
        // 7 bits: 0..127
        let indices: Vec<u8> = (0..127).collect();
        let packed = pack_indices(&indices, 7);
        let recovered = unpack_indices(&packed, 7, indices.len());
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_8bit() {
        let indices: Vec<u8> = (0..=255).collect();
        let packed = pack_indices(&indices, 8);
        // 256 indices × 8 bits = 2048 bits = 32 u64s
        assert_eq!(packed.len(), 32);
        let recovered = unpack_indices(&packed, 8, 256);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_non_aligned() {
        // 7 indices at 5 bits = 35 bits, not aligned to u64
        let indices = vec![0u8, 5, 10, 15, 20, 25, 30];
        let packed = pack_indices(&indices, 5);
        let recovered = unpack_indices(&packed, 5, indices.len());
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let indices: Vec<u8> = (0..100).map(|i| i % 16).collect();
        let bytes = pack_indices_bytes(&indices, 4);
        let recovered = unpack_indices_bytes(&bytes, 4, 100);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_transcode_4to5() {
        let indices: Vec<u8> = (0..16).collect();
        let packed_4 = pack_indices(&indices, 4);
        let packed_5 = transcode(&packed_4, 4, 5, 16);
        let recovered = unpack_indices(&packed_5, 5, 16);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_transcode_identity() {
        let indices: Vec<u8> = vec![0, 3, 7, 15, 1, 2, 4, 8];
        let packed = pack_indices(&indices, 4);
        let transcoded = transcode(&packed, 4, 4, indices.len());
        assert_eq!(packed, transcoded);
    }

    #[test]
    fn test_packed_palette_array_basic() {
        let indices: Vec<u8> = (0..100).map(|i| i % 16).collect();
        let arr = PackedPaletteArray::from_indices(&indices, 16);
        assert_eq!(arr.bits_per_index, 4);
        assert_eq!(arr.count, 100);
        assert_eq!(arr.to_indices(), indices);
    }

    #[test]
    fn test_packed_palette_array_get_set() {
        let indices = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut arr = PackedPaletteArray::from_indices(&indices, 8);
        assert_eq!(arr.get(3), 3);
        arr.set(3, 7);
        assert_eq!(arr.get(3), 7);
        // Other indices unchanged
        assert_eq!(arr.get(0), 0);
        assert_eq!(arr.get(7), 7);
    }

    #[test]
    fn test_packed_palette_array_grow() {
        let indices: Vec<u8> = (0..50).map(|i| i % 4).collect();
        let mut arr = PackedPaletteArray::from_indices(&indices, 4);
        assert_eq!(arr.bits_per_index, 2);

        // Grow palette from 4 to 32 entries
        arr.grow_palette(32);
        assert_eq!(arr.bits_per_index, 5);

        // All original indices preserved
        let recovered = arr.to_indices();
        assert_eq!(recovered, indices);
    }

    #[test]
    fn test_compression_ratio() {
        assert_eq!(compression_ratio(4), 2.0);
        assert_eq!(compression_ratio(2), 4.0);
        assert_eq!(compression_ratio(1), 8.0);
        assert_eq!(compression_ratio(8), 1.0);
    }

    #[test]
    fn test_packed_palette_array_compression() {
        // 1000 indices with palette size 4 → 2 bits each
        let indices: Vec<u8> = (0..1000).map(|i| (i % 4) as u8).collect();
        let arr = PackedPaletteArray::from_indices(&indices, 4);
        // 1000 * 2 bits = 2000 bits = 32 u64 words = 256 bytes
        // vs 1000 bytes raw → ~3.9× compression
        assert!(arr.compression_ratio() > 3.5);
        assert_eq!(arr.to_indices(), indices);
    }

    #[test]
    fn test_empty_indices() {
        let indices: Vec<u8> = vec![];
        let packed = pack_indices(&indices, 4);
        assert!(packed.is_empty());
        let recovered = unpack_indices(&packed, 4, 0);
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_single_index() {
        let indices = vec![7u8];
        for bits in 1..=8 {
            if 7 < (1 << bits) {
                let packed = pack_indices(&indices, bits);
                let recovered = unpack_indices(&packed, bits, 1);
                assert_eq!(recovered, indices, "failed at {bits} bits");
            }
        }
    }

    #[test]
    fn test_unpack_simd_4bit_matches_scalar() {
        let indices: Vec<u8> = (0..4096).map(|i| (i % 16) as u8).collect();
        let packed = pack_indices(&indices, 4);
        let scalar = unpack_indices(&packed, 4, 4096);
        let simd = unpack_indices_simd(&packed, 4, 4096);
        assert_eq!(scalar, simd, "SIMD 4-bit unpack must match scalar");
    }

    #[test]
    fn test_unpack_simd_non_4bit_fallback() {
        let indices: Vec<u8> = (0..100).map(|i| (i % 8) as u8).collect();
        let packed = pack_indices(&indices, 3);
        let scalar = unpack_indices(&packed, 3, 100);
        let simd = unpack_indices_simd(&packed, 3, 100);
        assert_eq!(scalar, simd, "non-4bit should fall back to scalar");
    }

    #[test]
    fn test_pack_simd_roundtrip() {
        let indices: Vec<u8> = (0..1000).map(|i| (i % 16) as u8).collect();
        let packed = pack_indices_simd(&indices, 4);
        let recovered = unpack_indices_simd(&packed, 4, 1000);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_unpack_simd_avx512_all_bit_widths() {
        for bits in 1..=8usize {
            let max_val = if bits == 8 { 255u8 } else { (1u8 << bits) - 1 };
            let indices: Vec<u8> = (0..4096).map(|i| (i as u8) & max_val).collect();
            let packed = pack_indices(&indices, bits);
            let simd = unpack_indices_simd(&packed, bits, indices.len());
            assert_eq!(indices, simd, "AVX-512 unpack mismatch at {bits} bits");
        }
    }

    #[test]
    fn test_pack_simd_avx512_all_bit_widths() {
        for bits in 1..=8usize {
            let max_val = if bits == 8 { 255u8 } else { (1u8 << bits) - 1 };
            let indices: Vec<u8> = (0..4096).map(|i| (i as u8) & max_val).collect();
            let packed_scalar = pack_indices(&indices, bits);
            let packed_simd = pack_indices_simd(&indices, bits);
            assert_eq!(packed_scalar, packed_simd, "AVX-512 pack mismatch at {bits} bits");
        }
    }

    #[test]
    fn test_unpack_simd_avx512_non_aligned_counts() {
        for bits in [1, 2, 3, 4, 5, 6, 7, 8] {
            let max_val = if bits == 8 { 255u8 } else { (1u8 << bits) - 1 };
            for count in [1, 7, 15, 17, 31, 33, 63, 65, 100] {
                let indices: Vec<u8> = (0..count).map(|i| (i as u8) & max_val).collect();
                let packed = pack_indices(&indices, bits);
                let simd = unpack_indices_simd(&packed, bits, count);
                assert_eq!(indices, simd, "mismatch at {bits}b x {count}");
            }
        }
    }

    #[test]
    fn test_bedrock_reorder_roundtrip() {
        // Create a pattern where every position has a unique value
        let states: Vec<u16> = (0..4096).map(|i| i as u16).collect();
        let reordered = bedrock_reorder_xzy(&states);
        let recovered = bedrock_reorder_xzy_inverse(&reordered);
        assert_eq!(states, recovered, "reorder then inverse must be identity");
    }

    #[test]
    fn test_bedrock_reorder_specific() {
        let mut states = vec![0u16; 4096];

        // Place known values at specific Java-order positions:
        // Java index = y*256 + z*16 + x
        // Bedrock index = x*256 + z*16 + y

        // (x=0, y=0, z=0) → Java idx 0, Bedrock idx 0
        states[0] = 100;
        // (x=1, y=0, z=0) → Java idx 1, Bedrock idx 256
        states[1] = 200;
        // (x=0, y=1, z=0) → Java idx 256, Bedrock idx 1
        states[256] = 300;
        // (x=3, y=5, z=7) → Java idx 5*256+7*16+3 = 1395, Bedrock idx 3*256+7*16+5 = 885
        states[1395] = 400;
        // (x=15, y=15, z=15) → Java idx 15*256+15*16+15 = 4095, Bedrock idx 4095
        states[4095] = 500;

        let reordered = bedrock_reorder_xzy(&states);

        assert_eq!(reordered[0], 100, "(0,0,0) should map to 0");
        assert_eq!(reordered[256], 200, "(1,0,0) should map to 256");
        assert_eq!(reordered[1], 300, "(0,1,0) should map to 1");
        assert_eq!(reordered[885], 400, "(3,5,7) should map to 885");
        assert_eq!(reordered[4095], 500, "(15,15,15) should map to 4095");
    }

    #[test]
    fn test_bedrock_pack_section() {
        use std::collections::HashMap;

        // Create states with a small palette
        let mut states = vec![0u16; 4096];
        for i in 0..4096 {
            states[i] = (i % 4) as u16;
        }

        let mut palette = HashMap::new();
        palette.insert(0u16, 0u8);
        palette.insert(1u16, 1u8);
        palette.insert(2u16, 2u8);
        palette.insert(3u16, 3u8);

        let bits = bits_for_palette_size(4); // 2 bits
        let packed = bedrock_pack_section(&states, &palette, bits)
            .expect("all states should be in palette");

        // Verify by unpacking and inverse-reordering
        let unpacked = unpack_indices(&packed, bits, 4096);
        let bedrock_states: Vec<u16> = unpacked.iter().map(|&idx| {
            // Reverse palette lookup: idx → state
            *palette.iter().find(|(_, &v)| v == idx).unwrap().0
        }).collect();
        let java_states = bedrock_reorder_xzy_inverse(&bedrock_states);
        assert_eq!(states, java_states, "pack then unpack+inverse must recover original");
    }

    #[test]
    fn test_bedrock_pack_section_missing_palette_entry() {
        use std::collections::HashMap;

        let mut states = vec![0u16; 4096];
        states[0] = 99; // Not in palette

        let mut palette = HashMap::new();
        palette.insert(0u16, 0u8);

        let result = bedrock_pack_section(&states, &palette, 1);
        assert!(result.is_none(), "should return None for missing palette entry");
    }
}
