//! Block property mask — compiled bitset queries.
//!
//! Compiles boolean property queries (e.g. "waterlogged AND facing_north AND NOT open")
//! into bitmask operations that test a single block state in O(1).
//!
//! With AVX-512 VPTERNLOGD: tests 3 conditions in 1 cycle.

/// A compiled property query on block state bits.
///
/// Tests multiple boolean properties in a single operation:
/// `(block_state & and_mask) == and_expect && (block_state & andn_mask) == 0`
///
/// # Examples
///
/// ```
/// use ndarray::hpc::property_mask::PropertyMask;
///
/// let mask = PropertyMask::new()
///     .require_bit(0)   // bit 0 must be set
///     .forbid_bit(3);   // bit 3 must NOT be set
///
/// assert!(mask.test(0b0001));   // bit 0 set, bit 3 clear
/// assert!(!mask.test(0b1001));  // bit 3 is set — forbidden
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyMask {
    /// Bits to test (AND)
    and_mask: u64,
    /// Expected result after AND
    and_expect: u64,
    /// Bits that must NOT be set
    andn_mask: u64,
}

impl PropertyMask {
    /// Create a new empty mask that matches all block states.
    pub fn new() -> Self {
        Self {
            and_mask: 0,
            and_expect: 0,
            andn_mask: 0,
        }
    }

    /// Require that `bit` is set in the block state.
    ///
    /// # Panics
    /// Panics if `bit >= 64`.
    pub fn require_bit(mut self, bit: usize) -> Self {
        assert!(bit < 64, "bit index out of range");
        let b = 1u64 << bit;
        self.and_mask |= b;
        self.and_expect |= b;
        self
    }

    /// Require that a multi-bit field at `offset` with `width` bits equals `value`.
    ///
    /// # Panics
    /// Panics if the field exceeds 64 bits or `value` does not fit in `width` bits.
    pub fn require_value(mut self, offset: usize, width: usize, value: u64) -> Self {
        assert!(width > 0 && offset + width <= 64, "field out of range");
        let field_mask = ((1u64 << width) - 1) << offset;
        assert!(value < (1u64 << width), "value does not fit in width");
        self.and_mask |= field_mask;
        self.and_expect = (self.and_expect & !field_mask) | (value << offset);
        self
    }

    /// Forbid `bit` from being set in the block state.
    ///
    /// # Panics
    /// Panics if `bit >= 64`.
    pub fn forbid_bit(mut self, bit: usize) -> Self {
        assert!(bit < 64, "bit index out of range");
        self.andn_mask |= 1u64 << bit;
        self
    }

    /// Test a single block state against the compiled mask.
    #[inline(always)]
    pub fn test(&self, block_state: u64) -> bool {
        (block_state & self.and_mask) == self.and_expect
            && (block_state & self.andn_mask) == 0
    }

    /// Batch test up to 4096 block states (one chunk section).
    /// Returns a `Vec<u64>` where each bit indicates whether the
    /// corresponding state matched.
    ///
    /// The returned vector has `ceil(states.len() / 64)` entries.
    pub fn test_section(&self, states: &[u64]) -> Vec<u64> {
        let n = states.len();
        let result_len = (n + 63) / 64;
        let mut result = vec![0u64; result_len];

        #[cfg(target_arch = "x86_64")]
        {
            let caps = super::simd_caps::simd_caps();
            if caps.avx512f {
                // SAFETY: avx512f detected, pointers are within slice bounds.
                unsafe {
                    self.test_section_avx512(states, &mut result);
                    return result;
                }
            }
            if caps.avx2 {
                // SAFETY: we checked avx2 at runtime, pointers are within slice bounds.
                unsafe {
                    self.test_section_avx2(states, &mut result);
                    return result;
                }
            }
        }

        self.test_section_scalar(states, &mut result);
        result
    }

    /// Count the number of matching block states in the slice.
    pub fn count_section(&self, states: &[u64]) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            let caps = super::simd_caps::simd_caps();
            if caps.avx512vpopcntdq && caps.avx512f {
                // SAFETY: feature detected above.
                return unsafe { self.count_section_avx512(states) };
            }
        }
        let bits = self.test_section(states);
        let full_words = states.len() / 64;
        let remainder = states.len() % 64;
        let mut count = 0u32;
        for &w in &bits[..full_words] {
            count += w.count_ones();
        }
        if remainder > 0 {
            // Mask off bits beyond the actual state count.
            let last = bits[full_words] & ((1u64 << remainder) - 1);
            count += last.count_ones();
        }
        count
    }

    // ---------- scalar fallback ----------

    fn test_section_scalar(&self, states: &[u64], result: &mut [u64]) {
        for (i, &state) in states.iter().enumerate() {
            if self.test(state) {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    // ---------- AVX-512 path ----------

    /// Test block states using AVX-512F, processing 8 u64s at a time.
    ///
    /// Uses 512-bit registers with `_mm512_cmpeq_epi64_mask` returning a
    /// `__mmask8` directly, avoiding the movemask+lane-extract dance of AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn test_section_avx512(&self, states: &[u64], result: &mut [u64]) {
        use core::arch::x86_64::*;

        let and_mask_v = _mm512_set1_epi64(self.and_mask as i64);
        let and_expect_v = _mm512_set1_epi64(self.and_expect as i64);
        let andn_mask_v = _mm512_set1_epi64(self.andn_mask as i64);
        let zero = _mm512_setzero_si512();

        let chunks = states.len() / 8;
        for c in 0..chunks {
            let base = c * 8;
            // SAFETY: base + 8 <= states.len(), avx512f checked by caller.
            let vals = _mm512_loadu_si512(states.as_ptr().add(base) as *const __m512i);

            // (vals & and_mask) == and_expect
            let anded = _mm512_and_si512(vals, and_mask_v);
            let eq_and = _mm512_cmpeq_epi64_mask(anded, and_expect_v);

            // (vals & andn_mask) == 0
            let andned = _mm512_and_si512(vals, andn_mask_v);
            let eq_andn = _mm512_cmpeq_epi64_mask(andned, zero);

            // Both conditions: AND the two kmasks
            let both = eq_and & eq_andn;

            // Set bits in the result bitmap
            for lane in 0..8usize {
                if (both >> lane) & 1 != 0 {
                    let idx = base + lane;
                    result[idx / 64] |= 1u64 << (idx % 64);
                }
            }
        }

        // Scalar tail
        for i in (chunks * 8)..states.len() {
            if self.test(states[i]) {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    /// Count matching states using AVX-512 VPOPCNTDQ for direct in-register popcount.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
    unsafe fn count_section_avx512(&self, states: &[u64]) -> u32 {
        use core::arch::x86_64::*;

        let and_mask_v = _mm512_set1_epi64(self.and_mask as i64);
        let and_expect_v = _mm512_set1_epi64(self.and_expect as i64);
        let andn_mask_v = _mm512_set1_epi64(self.andn_mask as i64);
        let zero = _mm512_setzero_si512();
        let mut total = 0u32;

        let chunks = states.len() / 8;
        for c in 0..chunks {
            let base = c * 8;
            // SAFETY: base + 8 <= states.len(), features checked by caller.
            let vals = _mm512_loadu_si512(states.as_ptr().add(base) as *const __m512i);

            let anded = _mm512_and_si512(vals, and_mask_v);
            let eq_and = _mm512_cmpeq_epi64_mask(anded, and_expect_v);

            let andned = _mm512_and_si512(vals, andn_mask_v);
            let eq_andn = _mm512_cmpeq_epi64_mask(andned, zero);

            let both = eq_and & eq_andn;
            total += (both as u32).count_ones();
        }

        // Scalar tail
        for i in (chunks * 8)..states.len() {
            if self.test(states[i]) {
                total += 1;
            }
        }
        total
    }

    // ---------- AVX2 path ----------

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn test_section_avx2(&self, states: &[u64], result: &mut [u64]) {
        // AVX2 processes 4 u64s at a time via 256-bit registers.
        use core::arch::x86_64::*;

        let and_mask_v = _mm256_set1_epi64x(self.and_mask as i64);
        let and_expect_v = _mm256_set1_epi64x(self.and_expect as i64);
        let andn_mask_v = _mm256_set1_epi64x(self.andn_mask as i64);
        let zero = _mm256_setzero_si256();

        let chunks = states.len() / 4;
        for c in 0..chunks {
            let base = c * 4;
            let vals = _mm256_loadu_si256(states.as_ptr().add(base) as *const __m256i);

            // (vals & and_mask) == and_expect
            let anded = _mm256_and_si256(vals, and_mask_v);
            let eq_and = _mm256_cmpeq_epi64(anded, and_expect_v);

            // (vals & andn_mask) == 0
            let andned = _mm256_and_si256(vals, andn_mask_v);
            let eq_andn = _mm256_cmpeq_epi64(andned, zero);

            // both conditions
            let both = _mm256_and_si256(eq_and, eq_andn);

            // Extract per-lane results (each lane is all-1s or all-0s).
            // _mm256_movemask_epi8 gives 32 bits; lanes are 8 bytes each.
            let mask32 = _mm256_movemask_epi8(both) as u32;
            // Lane k matched if bytes [k*8..(k+1)*8] are all 0xFF → bits set.
            for lane in 0..4usize {
                let byte_mask = (mask32 >> (lane * 8)) & 0xFF;
                if byte_mask == 0xFF {
                    let idx = base + lane;
                    result[idx / 64] |= 1u64 << (idx % 64);
                }
            }
        }

        // Scalar tail
        for i in (chunks * 4)..states.len() {
            if self.test(states[i]) {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
}

impl Default for PropertyMask {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of multi-mask counting: per-mask match counts from a single pass.
///
/// Enables "count crops AND count liquids AND count redstone" in one scan,
/// avoiding redundant iteration over 4096 block states.
#[derive(Debug, Clone)]
pub struct MultiMaskResult {
    /// Per-mask match counts, in the same order as the input masks.
    pub counts: Vec<u32>,
}

/// Count matches for multiple masks in a single pass over the data.
///
/// More efficient than calling `count_section()` N times because:
/// - Single pass over the state array (one cache line read per state)
/// - Each state is loaded once and tested against all masks
///
/// # Examples
///
/// ```
/// use ndarray::hpc::property_mask::{PropertyMask, count_section_multi};
///
/// let crops = PropertyMask::new().require_bit(0);
/// let liquids = PropertyMask::new().require_bit(1);
/// let redstone = PropertyMask::new().require_bit(2);
/// let states: Vec<u64> = (0..100).collect();
/// let result = count_section_multi(&[crops, liquids, redstone], &states);
/// assert_eq!(result.counts.len(), 3);
/// ```
pub fn count_section_multi(masks: &[PropertyMask], states: &[u64]) -> MultiMaskResult {
    if masks.is_empty() {
        return MultiMaskResult { counts: vec![] };
    }

    #[cfg(target_arch = "x86_64")]
    {
        let caps = super::simd_caps::simd_caps();
        if caps.avx512f && states.len() >= 8 {
            // SAFETY: avx512f detected above, states.len() >= 8 guaranteed.
            unsafe {
                return count_section_multi_avx512(masks, states);
            }
        }
    }

    // scalar fallback
    count_section_multi_scalar(masks, states)
}

/// Scalar fallback for multi-mask counting.
fn count_section_multi_scalar(masks: &[PropertyMask], states: &[u64]) -> MultiMaskResult {
    let mut counts = vec![0u32; masks.len()];
    for &state in states {
        for (m_idx, mask) in masks.iter().enumerate() {
            if mask.test(state) {
                counts[m_idx] += 1;
            }
        }
    }
    MultiMaskResult { counts }
}

/// AVX-512 multi-mask counting: process 8 states at a time per mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn count_section_multi_avx512(masks: &[PropertyMask], states: &[u64]) -> MultiMaskResult {
    use core::arch::x86_64::*;

    let mut counts = vec![0u32; masks.len()];
    let zero = _mm512_setzero_si512();
    let chunks = states.len() / 8;

    for c in 0..chunks {
        let base = c * 8;
        // SAFETY: base + 8 <= states.len(), avx512f checked by caller.
        let vals = _mm512_loadu_si512(states.as_ptr().add(base) as *const __m512i);

        for (m_idx, mask) in masks.iter().enumerate() {
            let and_mask_v = _mm512_set1_epi64(mask.and_mask as i64);
            let and_expect_v = _mm512_set1_epi64(mask.and_expect as i64);
            let andn_mask_v = _mm512_set1_epi64(mask.andn_mask as i64);

            // (vals & and_mask) == and_expect
            let anded = _mm512_and_si512(vals, and_mask_v);
            let eq_and = _mm512_cmpeq_epi64_mask(anded, and_expect_v);

            // (vals & andn_mask) == 0
            let andned = _mm512_and_si512(vals, andn_mask_v);
            let eq_andn = _mm512_cmpeq_epi64_mask(andned, zero);

            // Both conditions: AND the two kmasks
            let both = eq_and & eq_andn;
            counts[m_idx] += (both as u32).count_ones();
        }
    }

    // Scalar tail
    for i in (chunks * 8)..states.len() {
        let state = states[i];
        for (m_idx, mask) in masks.iter().enumerate() {
            if mask.test(state) {
                counts[m_idx] += 1;
            }
        }
    }

    MultiMaskResult { counts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_require_bit() {
        let m = PropertyMask::new().require_bit(0);
        assert!(m.test(0b0001));
        assert!(m.test(0b1111));
        assert!(!m.test(0b0000));
        assert!(!m.test(0b1110));
    }

    #[test]
    fn test_single_forbid_bit() {
        let m = PropertyMask::new().forbid_bit(2);
        assert!(m.test(0b0001));
        assert!(!m.test(0b0100));
        assert!(!m.test(0b0111));
    }

    #[test]
    fn test_require_and_forbid() {
        let m = PropertyMask::new().require_bit(0).forbid_bit(3);
        assert!(m.test(0b0001));
        assert!(!m.test(0b1001)); // bit 3 forbidden
        assert!(!m.test(0b0000)); // bit 0 required
    }

    #[test]
    fn test_require_value() {
        // bits [2..4] must equal 2 (binary 10)
        let m = PropertyMask::new().require_value(2, 2, 2);
        assert!(m.test(0b1000));       // field = 10 => 2
        assert!(!m.test(0b0100));      // field = 01 => 1
        assert!(!m.test(0b1100));      // field = 11 => 3
        assert!(m.test(0b11111_1000)); // field still 10
    }

    #[test]
    fn test_empty_mask_matches_everything() {
        let m = PropertyMask::new();
        assert!(m.test(0));
        assert!(m.test(u64::MAX));
        assert!(m.test(0xDEADBEEF));
    }

    #[test]
    fn test_batch_section() {
        let m = PropertyMask::new().require_bit(0);
        let states: Vec<u64> = (0..128).collect();
        let bits = m.test_section(&states);
        // Every odd-indexed value in 0..128 has bit 0 set.
        for i in 0..128 {
            let matched = (bits[i / 64] >> (i % 64)) & 1 == 1;
            assert_eq!(matched, i & 1 == 1, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_section_non_multiple() {
        let m = PropertyMask::new().forbid_bit(0);
        // 7 states: 0,1,2,3,4,5,6
        let states: Vec<u64> = (0..7).collect();
        let bits = m.test_section(&states);
        assert_eq!(bits.len(), 1);
        for i in 0..7 {
            let matched = (bits[0] >> i) & 1 == 1;
            assert_eq!(matched, i % 2 == 0, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_count_section() {
        let m = PropertyMask::new().require_bit(0);
        let states: Vec<u64> = (0..100).collect();
        let count = m.count_section(&states);
        // Numbers 1,3,5,...,99 → 50
        assert_eq!(count, 50);
    }

    #[test]
    fn test_count_empty() {
        let m = PropertyMask::new();
        let states: Vec<u64> = (0..256).collect();
        assert_eq!(m.count_section(&states), 256);
    }

    #[test]
    fn test_builder_chain() {
        let m = PropertyMask::new()
            .require_bit(0)
            .require_bit(1)
            .forbid_bit(4)
            .require_value(8, 4, 0xA);

        let state = 0b0000_1010_0000_0011u64; // bits 0,1 set; field [8..12]=0xA; bit 4 clear
        assert!(m.test(state));

        let bad_bit4 = state | (1 << 4);
        assert!(!m.test(bad_bit4));
    }

    #[test]
    fn test_scalar_parity_with_batch() {
        // Ensure scalar single-test agrees with batch for a complex mask.
        let m = PropertyMask::new()
            .require_bit(5)
            .forbid_bit(10)
            .require_value(16, 3, 5);

        let states: Vec<u64> = (0..512u64).map(|i| i.wrapping_mul(0x123456789)).collect();
        let batch = m.test_section(&states);
        for (i, &s) in states.iter().enumerate() {
            let from_batch = (batch[i / 64] >> (i % 64)) & 1 == 1;
            assert_eq!(from_batch, m.test(s), "parity mismatch at index {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "bit index out of range")]
    fn test_require_bit_oob() {
        PropertyMask::new().require_bit(64);
    }

    #[test]
    #[should_panic(expected = "bit index out of range")]
    fn test_forbid_bit_oob() {
        PropertyMask::new().forbid_bit(64);
    }

    #[test]
    #[should_panic(expected = "field out of range")]
    fn test_require_value_oob() {
        PropertyMask::new().require_value(60, 8, 0);
    }

    #[test]
    fn test_default_is_new() {
        assert_eq!(PropertyMask::default(), PropertyMask::new());
    }

    #[test]
    fn test_batch_section_avx512_parity() {
        // Test with enough states to exercise the 8-wide AVX-512 path + tail.
        let m = PropertyMask::new()
            .require_bit(3)
            .forbid_bit(7)
            .require_value(16, 4, 0xB);

        let states: Vec<u64> = (0..1024u64).map(|i| i.wrapping_mul(0xABCDEF01)).collect();
        let batch = m.test_section(&states);
        for (i, &s) in states.iter().enumerate() {
            let from_batch = (batch[i / 64] >> (i % 64)) & 1 == 1;
            assert_eq!(from_batch, m.test(s), "avx512 parity mismatch at index {}", i);
        }
    }

    #[test]
    fn test_count_section_avx512_parity() {
        let m = PropertyMask::new().require_bit(2).forbid_bit(5);
        let states: Vec<u64> = (0..500u64).map(|i| i.wrapping_mul(0x12345)).collect();
        let count = m.count_section(&states);
        let expected = states.iter().filter(|&&s| m.test(s)).count() as u32;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_count_multi_basic() {
        let crops = PropertyMask::new().require_bit(0);
        let liquids = PropertyMask::new().require_bit(1);
        let redstone = PropertyMask::new().require_bit(2).forbid_bit(5);

        let states: Vec<u64> = (0..256).collect();
        let result = count_section_multi(&[crops, liquids, redstone], &states);

        assert_eq!(result.counts.len(), 3);
        assert_eq!(result.counts[0], crops.count_section(&states));
        assert_eq!(result.counts[1], liquids.count_section(&states));
        assert_eq!(result.counts[2], redstone.count_section(&states));
    }

    #[test]
    fn test_count_multi_empty_masks() {
        let states: Vec<u64> = (0..100).collect();
        let result = count_section_multi(&[], &states);
        assert!(result.counts.is_empty());
    }

    #[test]
    fn test_count_multi_single() {
        let m = PropertyMask::new().require_bit(3).forbid_bit(7);
        let states: Vec<u64> = (0..200).collect();
        let result = count_section_multi(&[m], &states);
        assert_eq!(result.counts.len(), 1);
        assert_eq!(result.counts[0], m.count_section(&states));
    }

    #[test]
    fn test_count_multi_avx512_parity() {
        let masks = [
            PropertyMask::new().require_bit(0),
            PropertyMask::new().require_bit(1).forbid_bit(4),
            PropertyMask::new().require_value(8, 4, 0xA),
            PropertyMask::new().forbid_bit(3).forbid_bit(6),
            PropertyMask::new().require_bit(2).require_bit(5),
        ];

        let states: Vec<u64> = (0..1024u64).map(|i| i.wrapping_mul(0xABCDEF01)).collect();
        let result = count_section_multi(&masks, &states);

        assert_eq!(result.counts.len(), masks.len());
        for (m_idx, mask) in masks.iter().enumerate() {
            let expected = mask.count_section(&states);
            assert_eq!(
                result.counts[m_idx], expected,
                "multi-mask parity mismatch for mask index {}",
                m_idx
            );
        }
    }
}
