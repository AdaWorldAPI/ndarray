//! Bridge between ndarray Fingerprint<256> (2KB) and Base17 (34 bytes).
//!
//! Converts flat 16384-bit fingerprint planes to i16[17] base patterns
//! using golden-step octave averaging.
//!
//! This is a self-contained port of the bgz17 crate's `base17` module,
//! ensuring data interoperability without adding an external dependency.

const BASE_DIM: usize = 17;
const FULL_DIM: usize = 16384;
const GOLDEN_STEP: usize = 11;
const FP_SCALE: f64 = 256.0;

/// Golden-step position table.
const GOLDEN_POS: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

/// Number of octaves.
const N_OCTAVES: usize = (FULL_DIM + BASE_DIM - 1) / BASE_DIM;

/// 17-dimensional base pattern. 34 bytes.
///
/// Each dimension is an i16 fixed-point value (scaled by 256) representing
/// the average of golden-step-selected positions from a 16384-element accumulator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Base17 {
    pub dims: [i16; BASE_DIM],
}

/// SPO triple of Base17 patterns. 102 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpoBase17 {
    pub subject: Base17,
    pub predicate: Base17,
    pub object: Base17,
}

/// Palette edge: 3-byte compressed SPO triple (one u8 index per plane).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaletteEdge {
    pub s_idx: u8,
    pub p_idx: u8,
    pub o_idx: u8,
}

impl Base17 {
    /// Byte size of serialized form.
    pub const BYTE_SIZE: usize = BASE_DIM * 2; // 34

    /// Encode i8[16384] accumulator into a Base17 pattern.
    ///
    /// For each of 17 base dimensions, averages the accumulator values at
    /// golden-step-selected positions across all octaves, then scales by
    /// FP_SCALE (256) into fixed-point i16.
    pub fn encode(acc: &[i8]) -> Self {
        assert!(acc.len() >= FULL_DIM);
        let mut sum = [0i64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];

        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                if dim < FULL_DIM {
                    sum[bi] += acc[dim] as i64;
                    count[bi] += 1;
                }
            }
        }

        let mut dims = [0i16; BASE_DIM];
        for d in 0..BASE_DIM {
            if count[d] > 0 {
                let mean = sum[d] as f64 / count[d] as f64;
                dims[d] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
        Base17 { dims }
    }

    /// All-zero pattern (identity for xor_bind).
    pub fn zero() -> Self {
        Base17 { dims: [0i16; BASE_DIM] }
    }

    /// L1 (Manhattan) distance.
    #[inline]
    pub fn l1(&self, other: &Base17) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            d += (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
        }
        d
    }

    /// PCDVQ-informed L1: weight sign dimension 20x over mantissa.
    ///
    /// From arxiv 2506.05432: direction (sign) is 20x more sensitive to
    /// quantization than magnitude. BF16 decomposition maps to polar:
    ///   dim 0 = sign (direction), dims 1-6 = exponent (magnitude scale),
    ///   dims 7-16 = mantissa (fine detail).
    #[inline]
    pub fn l1_weighted(&self, other: &Base17) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            let diff = (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
            let weight = if i == 0 { 20 } else if i < 7 { 3 } else { 1 };
            d += diff * weight;
        }
        d
    }

    /// Sign-bit agreement (out of 17).
    #[inline]
    pub fn sign_agreement(&self, other: &Base17) -> u32 {
        let mut a = 0u32;
        for i in 0..BASE_DIM {
            if (self.dims[i] >= 0) == (other.dims[i] >= 0) {
                a += 1;
            }
        }
        a
    }

    /// XOR bind: path composition in hyperdimensional space.
    ///
    /// Bitwise XOR on each i16 dimension (reinterpreted as u16).
    /// Self-inverse: `a.xor_bind(&b).xor_bind(&b) == a`.
    /// Identity: `a.xor_bind(&Base17::zero()) == a`.
    #[inline]
    pub fn xor_bind(&self, other: &Base17) -> Base17 {
        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            dims[i] = (self.dims[i] as u16 ^ other.dims[i] as u16) as i16;
        }
        Base17 { dims }
    }

    /// Bundle: element-wise majority vote (set union in VSA).
    ///
    /// For each dimension, sums all patterns and takes the average.
    /// Ties (sum == 0) resolve to 0.
    pub fn bundle(patterns: &[&Base17]) -> Base17 {
        if patterns.is_empty() {
            return Base17::zero();
        }
        let mut dims = [0i16; BASE_DIM];
        let mut sums = [0i64; BASE_DIM];
        for p in patterns {
            for d in 0..BASE_DIM {
                sums[d] += p.dims[d] as i64;
            }
        }
        let n = patterns.len() as i64;
        for d in 0..BASE_DIM {
            dims[d] = (sums[d] / n).clamp(-32768, 32767) as i16;
        }
        Base17 { dims }
    }

    /// Permute: cyclic dimension shift (sequence encoding in VSA).
    ///
    /// `result[i] = self[(i + shift) % 17]`.
    #[inline]
    pub fn permute(&self, shift: usize) -> Base17 {
        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            dims[i] = self.dims[(i + shift) % BASE_DIM];
        }
        Base17 { dims }
    }

    /// Serialize to 34 bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut buf = [0u8; Self::BYTE_SIZE];
        for i in 0..BASE_DIM {
            let b = self.dims[i].to_le_bytes();
            buf[i * 2] = b[0];
            buf[i * 2 + 1] = b[1];
        }
        buf
    }

    /// Deserialize from 34 bytes (little-endian).
    pub fn from_bytes(buf: &[u8; Self::BYTE_SIZE]) -> Self {
        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            dims[i] = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
        Base17 { dims }
    }
}

impl SpoBase17 {
    /// Byte size of serialized form.
    pub const BYTE_SIZE: usize = Base17::BYTE_SIZE * 3; // 102

    /// Encode three i8[16384] accumulator planes.
    pub fn encode(s: &[i8], p: &[i8], o: &[i8]) -> Self {
        SpoBase17 {
            subject: Base17::encode(s),
            predicate: Base17::encode(p),
            object: Base17::encode(o),
        }
    }

    /// Combined L1 distance (sum of three planes).
    #[inline]
    pub fn l1(&self, other: &SpoBase17) -> u32 {
        self.subject.l1(&other.subject)
            + self.predicate.l1(&other.predicate)
            + self.object.l1(&other.object)
    }

    /// Per-plane L1 distances.
    #[inline]
    pub fn l1_per_plane(&self, other: &SpoBase17) -> (u32, u32, u32) {
        (
            self.subject.l1(&other.subject),
            self.predicate.l1(&other.predicate),
            self.object.l1(&other.object),
        )
    }
}

impl PaletteEdge {
    /// Serialize to 3 bytes.
    pub fn to_bytes(self) -> [u8; 3] {
        [self.s_idx, self.p_idx, self.o_idx]
    }

    /// Deserialize from 3 bytes.
    pub fn from_bytes(b: &[u8; 3]) -> Self {
        PaletteEdge { s_idx: b[0], p_idx: b[1], o_idx: b[2] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_coverage() {
        let mut seen = [false; BASE_DIM];
        for &p in &GOLDEN_POS { seen[p as usize] = true; }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_l1_self_zero() {
        let a = Base17 { dims: [100, -50, 0, 127, -128, 1, -1, 50, 25, -25, 0, 0, 0, 0, 0, 0, 0] };
        assert_eq!(a.l1(&a), 0);
    }

    #[test]
    fn test_l1_symmetric() {
        let a = Base17 { dims: [100; BASE_DIM] };
        let b = Base17 { dims: [-100; BASE_DIM] };
        assert_eq!(a.l1(&b), b.l1(&a));
    }

    #[test]
    fn test_xor_bind_self_inverse() {
        let a = Base17 { dims: [100, -200, 300, -400, 500, -600, 700, -800, 900, -1000, 1100, -1200, 1300, -1400, 1500, -1600, 1700] };
        let b = Base17 { dims: [-50, 150, -250, 350, -450, 550, -650, 750, -850, 950, -1050, 1150, -1250, 1350, -1450, 1550, -1650] };
        let bound = a.xor_bind(&b);
        let recovered = bound.xor_bind(&b);
        assert_eq!(a, recovered, "xor_bind must be its own inverse");
    }

    #[test]
    fn test_xor_bind_identity() {
        let a = Base17 { dims: [100, -200, 300, -400, 500, -600, 700, -800, 900, -1000, 1100, -1200, 1300, -1400, 1500, -1600, 1700] };
        let zero = Base17::zero();
        assert_eq!(a.xor_bind(&zero), a, "xor_bind with zero must be identity");
    }

    #[test]
    fn test_bundle_single() {
        let a = Base17 { dims: [100; BASE_DIM] };
        let result = Base17::bundle(&[&a]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_bundle_majority() {
        let pos = Base17 { dims: [100; BASE_DIM] };
        let neg = Base17 { dims: [-100; BASE_DIM] };
        let result = Base17::bundle(&[&pos, &pos, &neg]);
        for d in 0..BASE_DIM {
            assert!(result.dims[d] > 0, "dim {} should be positive from majority vote", d);
        }
    }

    #[test]
    fn test_permute_identity() {
        let a = Base17 { dims: [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17] };
        assert_eq!(a.permute(0), a, "permute(0) must be identity");
        assert_eq!(a.permute(BASE_DIM), a, "permute(17) must wrap to identity");
    }

    #[test]
    fn test_permute_cyclic() {
        let a = Base17 { dims: [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17] };
        let shifted = a.permute(1);
        for i in 0..BASE_DIM {
            assert_eq!(shifted.dims[i], a.dims[(i + 1) % BASE_DIM]);
        }
    }

    #[test]
    fn test_byte_roundtrip() {
        let a = Base17 { dims: [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17] };
        let bytes = a.to_bytes();
        let b = Base17::from_bytes(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_encode_all_zeros() {
        let acc = vec![0i8; FULL_DIM];
        let b = Base17::encode(&acc);
        assert_eq!(b, Base17::zero());
    }

    #[test]
    fn test_encode_all_positive() {
        let acc = vec![1i8; FULL_DIM];
        let b = Base17::encode(&acc);
        // Each dim should average to 1.0, scaled by 256 = 256
        for d in 0..BASE_DIM {
            assert_eq!(b.dims[d], 256, "dim {} should be 256", d);
        }
    }

    #[test]
    fn test_spo_l1_self_zero() {
        let edge = SpoBase17 {
            subject: Base17 { dims: [100; BASE_DIM] },
            predicate: Base17 { dims: [-50; BASE_DIM] },
            object: Base17 { dims: [25; BASE_DIM] },
        };
        assert_eq!(edge.l1(&edge), 0);
    }

    #[test]
    fn test_spo_encode() {
        let s = vec![1i8; FULL_DIM];
        let p = vec![-1i8; FULL_DIM];
        let o = vec![0i8; FULL_DIM];
        let spo = SpoBase17::encode(&s, &p, &o);
        assert!(spo.subject.dims[0] > 0);
        assert!(spo.predicate.dims[0] < 0);
        assert_eq!(spo.object.dims[0], 0);
    }

    #[test]
    fn test_palette_edge_roundtrip() {
        let pe = PaletteEdge { s_idx: 42, p_idx: 128, o_idx: 255 };
        let bytes = pe.to_bytes();
        let pe2 = PaletteEdge::from_bytes(&bytes);
        assert_eq!(pe, pe2);
    }

    #[test]
    fn test_l1_weighted_sign_dim_dominates() {
        let a = Base17 { dims: [0; 17] };
        let mut b_sign = Base17 { dims: [0; 17] };
        b_sign.dims[0] = 100;
        let mut b_mant = Base17 { dims: [0; 17] };
        b_mant.dims[10] = 100;

        let d_sign = a.l1_weighted(&b_sign);
        let d_mant = a.l1_weighted(&b_mant);

        assert_eq!(d_sign, 100 * 20);
        assert_eq!(d_mant, 100 * 1);
        assert!(d_sign > d_mant * 10);
    }

    #[test]
    fn test_sign_agreement_self() {
        let a = Base17 { dims: [100, -50, 30, 0, 10, -20, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] };
        assert_eq!(a.sign_agreement(&a), BASE_DIM as u32);
    }

    #[test]
    fn test_sign_agreement_opposite() {
        let a = Base17 { dims: [1; BASE_DIM] };
        let b = Base17 { dims: [-1; BASE_DIM] };
        assert_eq!(a.sign_agreement(&b), 0);
    }
}
