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
        if count >= 32 && super::simd_caps::simd_caps().avx2 {
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

pub(crate) fn nibble_unpack_scalar(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    for i in 0..count {
        let byte = packed[i / 2];
        let val = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        out.push(val);
    }
}

/// AVX2 nibble unpack: processes 16 packed bytes → 32 nibbles per iteration.
///
/// Uses scalar array operations on 16-byte chunks for portability.
///
/// # Safety
/// Caller must ensure AVX2 is available and `count >= 32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nibble_unpack_avx2(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    let mut i = 0usize; // byte index into packed
    let mut emitted = 0usize;

    // Each iteration: load 16 packed bytes → 32 nibbles
    while emitted + 32 <= count && i + 16 <= packed.len() {
        let mut data = [0u8; 16];
        data.copy_from_slice(&packed[i..i + 16]);

        // Extract low and high nibbles
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for j in 0..16 {
            lo[j] = data[j] & 0x0F;
            hi[j] = (data[j] >> 4) & 0x0F;
        }

        // Interleave: lo[0],hi[0], lo[1],hi[1], ...
        let mut buf = [0u8; 32];
        for j in 0..16 {
            buf[j * 2] = lo[j];
            buf[j * 2 + 1] = hi[j];
        }

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
        let caps = super::simd_caps::simd_caps();
        if caps.avx512bw {
            // SAFETY: avx512bw detected, slice is mutable and valid.
            unsafe {
                nibble_sub_clamp_avx512(packed, delta);
                return;
            }
        }
        if caps.avx2 {
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
    let chunks = packed.len() / 32;

    for c in 0..chunks {
        let offset = c * 32;
        let mut data = [0u8; 32];
        data.copy_from_slice(&packed[offset..offset + 32]);

        for j in 0..32 {
            let lo = (data[j] & 0x0F).saturating_sub(delta);
            let hi = ((data[j] >> 4) & 0x0F).saturating_sub(delta);
            data[j] = lo | (hi << 4);
        }

        packed[offset..offset + 32].copy_from_slice(&data);
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
    use crate::simd::U8x64;

    let mask_lo = U8x64::splat(0x0F);
    let mask_hi = U8x64::splat(0xF0);
    let delta_v = U8x64::splat(delta);
    let delta_hi = U8x64::splat(delta << 4);
    let chunks = packed.len() / 64;

    for c in 0..chunks {
        let offset = c * 64;
        // SAFETY: offset + 64 <= packed.len(), avx512bw checked.
        let data = U8x64::from_slice(&packed[offset..]);

        let lo = data & mask_lo;
        let lo_sub = lo.saturating_sub(delta_v);

        let hi = data & mask_hi;
        let hi_sub = hi.saturating_sub(delta_hi);

        let result = (lo_sub & mask_lo) | (hi_sub & mask_hi);

        result.copy_to_slice(&mut packed[offset..offset + 64]);
    }

    // Scalar tail
    nibble_sub_clamp_scalar(&mut packed[chunks * 64..], delta);
}

/// Find all nibble indices with value strictly above `threshold`. Returns sorted indices.
pub fn nibble_above_threshold(packed: &[u8], threshold: u8) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if packed.len() >= 16 && super::simd_caps::simd_caps().avx2 {
            // SAFETY: avx2 detected, packed buffer large enough.
            return unsafe { nibble_above_threshold_avx2(packed, threshold) };
        }
    }

    nibble_above_threshold_scalar(packed, threshold)
}

pub(crate) fn nibble_above_threshold_scalar(packed: &[u8], threshold: u8) -> Vec<usize> {
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
/// Uses scalar array operations on 32-byte chunks for portability.
///
/// # Safety
/// Caller must ensure AVX2 is available and `packed.len() >= 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nibble_above_threshold_avx2(packed: &[u8], threshold: u8) -> Vec<usize> {
    let mut result = Vec::new();

    let chunks = packed.len() / 32;
    for c in 0..chunks {
        let base_byte = c * 32;
        let chunk = &packed[base_byte..base_byte + 32];

        for j in 0..32 {
            let lo = chunk[j] & 0x0F;
            let hi = (chunk[j] >> 4) & 0x0F;
            if lo > threshold {
                result.push((base_byte + j) * 2);
            }
            if hi > threshold {
                result.push((base_byte + j) * 2 + 1);
            }
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
