//! Qualia system: 16-channel phenomenal coloring of binary containers.
//!
//! Each channel captures a different aspect of the "feel" of a binary fingerprint.
//! Dual use: sparse in metadata for NARS coloring, dense as universal CAM index.

use super::bf16_truth::PackedQualia;

/// Named channel indices (re-export from causality for convenience).
pub use super::causality::qualia_dim;

/// 16-channel qualia vector. Each channel is a quantized value in [0, 65535].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualiaVector {
    pub channels: [u16; 16],
}

impl QualiaVector {
    /// All channels zero.
    pub fn zero() -> Self {
        Self { channels: [0u16; 16] }
    }
}

/// Compute all 16 channels from raw binary data.
///
/// Each channel is computed from bit-pattern analysis of the input bytes.
/// All values are normalized to the `u16` range `[0, 65535]`.
///
/// Returns a zero vector if `data` is empty.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia::qualia_color;
///
/// let data = vec![0xAAu8; 128];
/// let q = qualia_color(&data);
/// // 0xAA has popcount 4 of 8 => luminance ≈ 32767
/// assert!(q.channels[0] > 30000 && q.channels[0] < 35000);
/// ```
pub fn qualia_color(data: &[u8]) -> QualiaVector {
    if data.is_empty() {
        return QualiaVector::zero();
    }

    let mut channels = [0u16; 16];
    let len = data.len();
    let total_bits = len * 8;

    // Per-byte popcounts (reused by several channels).
    let byte_pops: Vec<u32> = data.iter().map(|b| b.count_ones()).collect();
    let total_pop: u64 = byte_pops.iter().map(|&p| p as u64).sum();

    // 0: LUMINANCE — overall bit density
    channels[qualia_dim::LUMINANCE] = scale_u16(total_pop, total_bits as u64);

    // 1: RED_GREEN — asymmetry between first and second half
    {
        let mid = len / 2;
        let pop_first: u64 = byte_pops[..mid].iter().map(|&p| p as u64).sum();
        let pop_second: u64 = byte_pops[mid..].iter().map(|&p| p as u64).sum();
        let max_possible = total_bits as u64;
        let diff = pop_first.abs_diff(pop_second);
        channels[qualia_dim::RED_GREEN] = scale_u16(diff, max_possible);
    }

    // 2: BLUE_YELLOW — asymmetry between odd and even indexed bytes
    {
        let mut pop_even: u64 = 0;
        let mut pop_odd: u64 = 0;
        for (i, &p) in byte_pops.iter().enumerate() {
            if i % 2 == 0 {
                pop_even += p as u64;
            } else {
                pop_odd += p as u64;
            }
        }
        let max_possible = total_bits as u64;
        let diff = pop_even.abs_diff(pop_odd);
        channels[qualia_dim::BLUE_YELLOW] = scale_u16(diff, max_possible);
    }

    // 3: PITCH — frequency of bit transitions (XOR adjacent bytes, sum popcounts)
    {
        let mut transitions: u64 = 0;
        for i in 1..len {
            transitions += (data[i] ^ data[i - 1]).count_ones() as u64;
        }
        // Maximum transitions: (len-1) * 8
        let max_trans = if len > 1 { (len - 1) as u64 * 8 } else { 1 };
        channels[qualia_dim::PITCH] = scale_u16(transitions, max_trans);
    }

    // 4: WARMTH — popcount of middle 50% of data
    {
        let q1 = len / 4;
        let q3 = q1 + len / 2;
        let mid_pop: u64 = byte_pops[q1..q3].iter().map(|&p| p as u64).sum();
        let mid_bits = (q3 - q1) as u64 * 8;
        let max = if mid_bits > 0 { mid_bits } else { 1 };
        channels[qualia_dim::WARMTH] = scale_u16(mid_pop, max);
    }

    // 5: PRESSURE — variance of per-byte popcounts, scaled
    {
        let mean = total_pop as f64 / len as f64;
        let var: f64 = byte_pops
            .iter()
            .map(|&p| {
                let diff = p as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / len as f64;
        // Max variance for bytes is when half are 0 and half are 8: (4^2) = 16
        // Theoretical max variance of popcount(byte) is 16.0
        let normalized = (var / 16.0).min(1.0);
        channels[qualia_dim::PRESSURE] = (normalized * 65535.0) as u16;
    }

    // 6: SOCIAL — correlation of adjacent byte pairs (XOR and inverse)
    {
        if len >= 2 {
            let mut agreement: u64 = 0;
            let pairs = len - 1;
            for i in 0..pairs {
                // Agreement = 8 - popcount(xor) for each pair
                agreement += 8 - (data[i] ^ data[i + 1]).count_ones() as u64;
            }
            let max_agreement = pairs as u64 * 8;
            channels[qualia_dim::SOCIAL] = scale_u16(agreement, max_agreement);
        }
    }

    // 7: TEMPORAL — gradient (first quarter vs last quarter popcount)
    {
        let q_size = len / 4;
        if q_size > 0 {
            let pop_first_q: u64 = byte_pops[..q_size].iter().map(|&p| p as u64).sum();
            let pop_last_q: u64 = byte_pops[len - q_size..].iter().map(|&p| p as u64).sum();
            let max = q_size as u64 * 8;
            // Map difference to [0, 65535] with 32768 as "no gradient"
            let diff = pop_last_q as f64 - pop_first_q as f64;
            let normalized = (diff / max as f64 + 1.0) / 2.0; // map [-1,1] to [0,1]
            channels[qualia_dim::TEMPORAL] = (normalized.clamp(0.0, 1.0) * 65535.0) as u16;
        } else {
            channels[qualia_dim::TEMPORAL] = 32767; // neutral
        }
    }

    // 8: SACREDNESS — entropy of byte value distribution (256 bins)
    {
        let mut histogram = [0u32; 256];
        for &b in data {
            histogram[b as usize] += 1;
        }
        let n = len as f64;
        let mut entropy = 0.0f64;
        for &count in &histogram {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }
        // Max entropy = log2(256) = 8.0 (uniform distribution)
        let normalized = (entropy / 8.0).min(1.0);
        channels[qualia_dim::SACREDNESS] = (normalized * 65535.0) as u16;
    }

    // 9: AROUSAL — same as luminance but with nonlinear scaling (square)
    {
        let ratio = total_pop as f64 / total_bits as f64;
        let squared = ratio * ratio;
        channels[qualia_dim::AROUSAL] = (squared.min(1.0) * 65535.0) as u16;
    }

    // 10: VALENCE — fraction of bytes with popcount > 4 vs < 4
    {
        let mut high = 0u64;
        let mut low = 0u64;
        for &p in &byte_pops {
            if p > 4 {
                high += 1;
            } else if p < 4 {
                low += 1;
            }
        }
        let total_hl = high + low;
        if total_hl > 0 {
            channels[qualia_dim::VALENCE] = scale_u16(high, total_hl);
        } else {
            channels[qualia_dim::VALENCE] = 32768; // neutral when all bytes have popcount == 4
        }
    }

    // 11: AGENCY — number of distinct byte values / 256
    {
        let mut seen = [false; 256];
        for &b in data {
            seen[b as usize] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count() as u64;
        channels[qualia_dim::AGENCY] = scale_u16(distinct, 256);
    }

    // 12: DEPTH — variance of per-block (64-byte) popcounts
    {
        let block_size = 64usize;
        let n_blocks = len / block_size;
        if n_blocks >= 2 {
            let block_pops: Vec<u64> = (0..n_blocks)
                .map(|i| {
                    let start = i * block_size;
                    let end = start + block_size;
                    byte_pops[start..end].iter().map(|&p| p as u64).sum()
                })
                .collect();
            let mean = block_pops.iter().sum::<u64>() as f64 / n_blocks as f64;
            let var: f64 = block_pops
                .iter()
                .map(|&p| {
                    let d = p as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n_blocks as f64;
            // Max variance: (block_size * 8)^2 / 4 = (512)^2 / 4 = 65536
            let normalized = (var / 65536.0).min(1.0);
            channels[qualia_dim::DEPTH] = (normalized * 65535.0) as u16;
        }
    }

    // 13: TEXTURE — count distinct 4-byte patterns / total
    {
        if len >= 4 {
            let n_quads = len - 3;
            let mut seen = std::collections::HashSet::new();
            for i in 0..n_quads {
                let pattern = [data[i], data[i + 1], data[i + 2], data[i + 3]];
                seen.insert(pattern);
            }
            let distinct = seen.len() as u64;
            channels[qualia_dim::TEXTURE] = scale_u16(distinct, n_quads as u64);
        }
    }

    // 14: FAMILIARITY — KL divergence from uniform byte distribution, inverted
    {
        let mut histogram = [0u32; 256];
        for &b in data {
            histogram[b as usize] += 1;
        }
        let n = len as f64;
        let uniform_p = 1.0 / 256.0;
        let mut kl = 0.0f64;
        for &count in &histogram {
            let q = (count as f64 / n).max(1e-10);
            kl += q * (q / uniform_p).ln();
        }
        // KL is 0 for uniform, grows with divergence.
        // Max KL ≈ ln(256) ≈ 5.545 when all mass in one bin.
        let normalized_kl = (kl / 5.545).min(1.0);
        // Invert: high familiarity = close to uniform
        channels[qualia_dim::FAMILIARITY] = ((1.0 - normalized_kl) * 65535.0) as u16;
    }

    // 15: SURPRISE — |actual_total_popcount - expected_50%| / max
    {
        let expected = total_bits as u64 / 2;
        let diff = total_pop.abs_diff(expected);
        channels[qualia_dim::SURPRISE] = scale_u16(diff, expected.max(1));
    }

    QualiaVector { channels }
}

/// Euclidean distance between two qualia vectors, normalized to `[0.0, 1.0]`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia::{QualiaVector, qualia_distance};
///
/// let a = QualiaVector { channels: [0; 16] };
/// let b = QualiaVector { channels: [65535; 16] };
/// let d = qualia_distance(&a, &b);
/// assert!((d - 1.0).abs() < 1e-5);
/// ```
pub fn qualia_distance(a: &QualiaVector, b: &QualiaVector) -> f32 {
    let mut sum_sq: f64 = 0.0;
    for i in 0..16 {
        let diff = a.channels[i] as f64 - b.channels[i] as f64;
        sum_sq += diff * diff;
    }
    let dist = sum_sq.sqrt();
    // Maximum distance: sqrt(16 * 65535^2) = 4 * 65535 = 262140
    let max_dist = (16.0f64 * 65535.0 * 65535.0).sqrt();
    (dist / max_dist) as f32
}

/// Similarity between two qualia vectors: `1.0 - normalized_distance`.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 means identical.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia::{QualiaVector, qualia_similarity};
///
/// let a = QualiaVector { channels: [32768; 16] };
/// let s = qualia_similarity(&a, &a);
/// assert!((s - 1.0).abs() < 1e-6);
/// ```
pub fn qualia_similarity(a: &QualiaVector, b: &QualiaVector) -> f32 {
    1.0 - qualia_distance(a, b)
}

/// Content-addressable key from qualia vector.
///
/// Packs 16 channels into a `u128` by taking the top 8 bits of each channel,
/// yielding a 128-bit key suitable for hash-based lookup.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia::{QualiaVector, qualia_cam_key};
///
/// let q = QualiaVector { channels: [0xFF00; 16] };
/// let key = qualia_cam_key(&q);
/// assert_ne!(key, 0);
/// ```
pub fn qualia_cam_key(q: &QualiaVector) -> u128 {
    let mut key: u128 = 0;
    for i in 0..16 {
        let top8 = (q.channels[i] >> 8) as u128;
        key |= top8 << (i * 8);
    }
    key
}

/// Convert from [`PackedQualia`] (i8 resonance) to [`QualiaVector`] (u16 channels).
///
/// Maps `i8` range `[-128, 127]` to `u16` range `[0, 65535]` via:
/// `u16_value = ((i8_value as i16) + 128) * 256 + 128`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::bf16_truth::PackedQualia;
/// use ndarray::hpc::qualia::qualia_from_packed;
///
/// let pq = PackedQualia::zero();
/// let qv = qualia_from_packed(&pq);
/// // 0 -> (0 + 128) * 256 + 128 = 32896
/// assert_eq!(qv.channels[0], 32896);
/// ```
pub fn qualia_from_packed(packed: &PackedQualia) -> QualiaVector {
    let mut channels = [0u16; 16];
    for i in 0..16 {
        let v = packed.resonance[i] as i16;
        // Map [-128, 127] to [0, 65535]
        // -128 -> 0, 0 -> 32896, 127 -> 65408
        channels[i] = ((v + 128) as u16) * 256 + 128;
    }
    QualiaVector { channels }
}

/// Convert from [`QualiaVector`] (u16 channels) to [`PackedQualia`] (i8 resonance).
///
/// Inverse of [`qualia_from_packed`]. The scalar field is set to zero.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia::{QualiaVector, qualia_to_packed};
///
/// let qv = QualiaVector { channels: [32896; 16] };
/// let pq = qualia_to_packed(&qv);
/// assert_eq!(pq.resonance[0], 0);
/// ```
pub fn qualia_to_packed(q: &QualiaVector) -> PackedQualia {
    let mut resonance = [0i8; 16];
    for i in 0..16 {
        // Reverse: i8 = (u16 - 128) / 256 - 128
        // Simplified: i8 = (u16 / 256) - 128, clamped
        let v = (q.channels[i] / 256) as i16 - 128;
        resonance[i] = v.clamp(-128, 127) as i8;
    }
    PackedQualia {
        resonance,
        scalar: [0u8; 2],
    }
}

/// Scale a ratio `numerator / denominator` to `u16` range `[0, 65535]`.
#[inline]
fn scale_u16(numerator: u64, denominator: u64) -> u16 {
    if denominator == 0 {
        return 0;
    }
    let ratio = numerator as f64 / denominator as f64;
    (ratio.min(1.0) * 65535.0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualia_color_zeros() {
        let data = vec![0u8; 256];
        let q = qualia_color(&data);

        // All zero bytes: popcount=0 everywhere
        assert_eq!(q.channels[qualia_dim::LUMINANCE], 0);
        assert_eq!(q.channels[qualia_dim::RED_GREEN], 0);
        assert_eq!(q.channels[qualia_dim::BLUE_YELLOW], 0);
        assert_eq!(q.channels[qualia_dim::PITCH], 0); // no transitions
        assert_eq!(q.channels[qualia_dim::WARMTH], 0);
        assert_eq!(q.channels[qualia_dim::PRESSURE], 0); // zero variance
        // SOCIAL: all pairs agree perfectly (xor=0, agreement=8)
        assert_eq!(q.channels[qualia_dim::SOCIAL], 65535);
        // TEMPORAL: no gradient => 0.5 * 65535 = 32767
        assert_eq!(q.channels[qualia_dim::TEMPORAL], 32767);
        // SACREDNESS: entropy of all-zero bytes = 0 (only one bin)
        assert_eq!(q.channels[qualia_dim::SACREDNESS], 0);
        // AROUSAL: 0^2 = 0
        assert_eq!(q.channels[qualia_dim::AROUSAL], 0);
        // AGENCY: only 1 distinct value out of 256
        assert!(q.channels[qualia_dim::AGENCY] > 0 && q.channels[qualia_dim::AGENCY] < 512);
        // SURPRISE: |0 - 1024| / 1024 = 1.0
        assert_eq!(q.channels[qualia_dim::SURPRISE], 65535);
    }

    #[test]
    fn test_qualia_color_ones() {
        let data = vec![0xFFu8; 256];
        let q = qualia_color(&data);

        // All 0xFF bytes: popcount=8 everywhere
        assert_eq!(q.channels[qualia_dim::LUMINANCE], 65535);
        assert_eq!(q.channels[qualia_dim::RED_GREEN], 0); // symmetric halves
        assert_eq!(q.channels[qualia_dim::BLUE_YELLOW], 0); // symmetric odd/even
        assert_eq!(q.channels[qualia_dim::PITCH], 0); // no transitions (xor of identical bytes = 0)
        assert_eq!(q.channels[qualia_dim::WARMTH], 65535);
        assert_eq!(q.channels[qualia_dim::PRESSURE], 0); // zero variance
        // SOCIAL: all pairs identical => max agreement
        assert_eq!(q.channels[qualia_dim::SOCIAL], 65535);
        assert_eq!(q.channels[qualia_dim::TEMPORAL], 32767); // no gradient
        // SACREDNESS: only one byte value => entropy = 0
        assert_eq!(q.channels[qualia_dim::SACREDNESS], 0);
        // AROUSAL: 1.0^2 = 1.0
        assert_eq!(q.channels[qualia_dim::AROUSAL], 65535);
        // AGENCY: only 1 distinct value
        assert!(q.channels[qualia_dim::AGENCY] > 0 && q.channels[qualia_dim::AGENCY] < 512);
        // SURPRISE: |2048 - 1024| / 1024 = 1.0
        assert_eq!(q.channels[qualia_dim::SURPRISE], 65535);
    }

    #[test]
    fn test_qualia_color_deterministic() {
        let data: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(37)).collect();
        let q1 = qualia_color(&data);
        let q2 = qualia_color(&data);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_qualia_color_empty() {
        let q = qualia_color(&[]);
        assert_eq!(q, QualiaVector::zero());
    }

    #[test]
    fn test_qualia_distance_self_zero() {
        let data: Vec<u8> = (0..128).map(|i| (i * 3) as u8).collect();
        let q = qualia_color(&data);
        let d = qualia_distance(&q, &q);
        assert!(d.abs() < 1e-6, "self-distance should be 0, got {}", d);
    }

    #[test]
    fn test_qualia_distance_symmetry() {
        let data_a: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
        let data_b: Vec<u8> = (0..128).map(|i| (i * 13 + 42) as u8).collect();
        let a = qualia_color(&data_a);
        let b = qualia_color(&data_b);
        let d_ab = qualia_distance(&a, &b);
        let d_ba = qualia_distance(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "distance should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_qualia_distance_range() {
        let a = QualiaVector { channels: [0; 16] };
        let b = QualiaVector { channels: [65535; 16] };
        let d = qualia_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-5,
            "max distance should be ~1.0, got {}",
            d
        );
    }

    #[test]
    fn test_qualia_similar_closer() {
        // Two nearly identical inputs should be closer than two very different ones
        let base: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let similar: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_add(1)).collect();
        let different = vec![0xFFu8; 256];

        let q_base = qualia_color(&base);
        let q_similar = qualia_color(&similar);
        let q_different = qualia_color(&different);

        let d_similar = qualia_distance(&q_base, &q_similar);
        let d_different = qualia_distance(&q_base, &q_different);

        assert!(
            d_similar < d_different,
            "similar data should have smaller distance: {} vs {}",
            d_similar,
            d_different
        );
    }

    #[test]
    fn test_qualia_similarity_complement() {
        let data_a: Vec<u8> = (0..128).collect();
        let data_b: Vec<u8> = (128..256).map(|i| i as u8).collect();
        let a = qualia_color(&data_a);
        let b = qualia_color(&data_b);
        let d = qualia_distance(&a, &b);
        let s = qualia_similarity(&a, &b);
        assert!(
            (d + s - 1.0).abs() < 1e-6,
            "distance + similarity should = 1.0"
        );
    }

    #[test]
    fn test_qualia_cam_key_deterministic() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let q = qualia_color(&data);
        let k1 = qualia_cam_key(&q);
        let k2 = qualia_cam_key(&q);
        assert_eq!(k1, k2);
        assert_ne!(k1, 0, "cam key should be non-zero for non-trivial input");
    }

    #[test]
    fn test_qualia_cam_key_different_inputs() {
        let a = qualia_color(&[0u8; 256]);
        let b = qualia_color(&[0xFFu8; 256]);
        let ka = qualia_cam_key(&a);
        let kb = qualia_cam_key(&b);
        assert_ne!(ka, kb, "different qualia should produce different cam keys");
    }

    #[test]
    fn test_qualia_roundtrip_packed() {
        // Create a PackedQualia with known values
        let packed = PackedQualia::new(
            [0, 10, -10, 50, -50, 100, -100, 127, -128, 1, -1, 42, -42, 63, -63, 0],
            0.0,
        );
        let qv = qualia_from_packed(&packed);
        let packed2 = qualia_to_packed(&qv);

        // Roundtrip should preserve resonance values approximately
        for i in 0..16 {
            let diff = (packed.resonance[i] as i16 - packed2.resonance[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "channel {}: original={}, roundtripped={}, diff={}",
                i,
                packed.resonance[i],
                packed2.resonance[i],
                diff
            );
        }
    }

    #[test]
    fn test_qualia_from_packed_range() {
        // Minimum i8 -> should map near 0
        let mut packed = PackedQualia::zero();
        packed.resonance[0] = -128;
        let qv = qualia_from_packed(&packed);
        assert_eq!(qv.channels[0], 128); // (-128+128)*256 + 128 = 128

        // Maximum i8 -> should map near 65535
        packed.resonance[0] = 127;
        let qv = qualia_from_packed(&packed);
        assert_eq!(qv.channels[0], 65408); // (127+128)*256 + 128 = 65408
    }

    #[test]
    fn test_qualia_to_packed_extremes() {
        let q = QualiaVector {
            channels: [0, 65535, 32768, 128, 65408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let p = qualia_to_packed(&q);
        assert_eq!(p.resonance[0], -128); // 0/256 - 128 = -128
        assert_eq!(p.resonance[1], 127); // 65535/256 = 255 - 128 = 127
        assert_eq!(p.resonance[3], -128); // 128/256 = 0 - 128 = -128
    }
}
