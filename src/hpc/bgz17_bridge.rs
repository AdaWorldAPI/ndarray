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

// ============================================================================
// Multi-versioned L1 kernel: AVX-512 → AVX2 → scalar. One binary, all ISAs.
// ============================================================================

type L1Fn = unsafe fn(&[i16; 17], &[i16; 17]) -> u32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l1_avx512(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    // Load 16 i16 → 16 i32 via sign-extension
    let va = _mm512_cvtepi16_epi32(_mm256_loadu_si256(a.as_ptr() as *const __m256i));
    let vb = _mm512_cvtepi16_epi32(_mm256_loadu_si256(b.as_ptr() as *const __m256i));
    let diff = _mm512_sub_epi32(va, vb);
    let abs_diff = _mm512_abs_epi32(diff);
    let sum16 = _mm512_reduce_add_epi32(abs_diff) as u32;
    // 17th dim scalar
    let d16 = (a[16] as i32 - b[16] as i32).unsigned_abs();
    sum16 + d16
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_avx2(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    // Process 8 dims at a time (2 passes of 8 = 16, + 1 scalar)
    let va0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i));
    let vb0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr() as *const __m128i));
    let diff0 = _mm256_sub_epi32(va0, vb0);
    let abs0 = _mm256_abs_epi32(diff0);

    let va1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a[8..].as_ptr() as *const __m128i));
    let vb1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b[8..].as_ptr() as *const __m128i));
    let diff1 = _mm256_sub_epi32(va1, vb1);
    let abs1 = _mm256_abs_epi32(diff1);

    let sum = _mm256_add_epi32(abs0, abs1);
    // Horizontal sum of 8 i32
    let hi128 = _mm256_extracti128_si256(sum, 1);
    let lo128 = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let sum16 = _mm_extract_epi32(sum32, 0) as u32;
    // 17th dim scalar
    let d16 = (a[16] as i32 - b[16] as i32).unsigned_abs();
    sum16 + d16
}

fn l1_scalar(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    let mut d = 0u32;
    for i in 0..17 {
        d += (a[i] as i32 - b[i] as i32).unsigned_abs();
    }
    d
}

static L1_KERNEL: std::sync::LazyLock<L1Fn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return l1_avx512 as L1Fn;
        }
        if is_x86_feature_detected!("avx2") {
            return l1_avx2 as L1Fn;
        }
    }
    l1_scalar as L1Fn
});

// ============================================================================
// Multi-versioned L1-weighted kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

type L1WeightedFn = unsafe fn(&[i16; 17], &[i16; 17]) -> u32;

const WEIGHT_VEC: [i32; 16] = [20, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l1_weighted_avx512(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    let va = _mm512_cvtepi16_epi32(_mm256_loadu_si256(a.as_ptr() as *const __m256i));
    let vb = _mm512_cvtepi16_epi32(_mm256_loadu_si256(b.as_ptr() as *const __m256i));
    let diff = _mm512_sub_epi32(va, vb);
    let abs_diff = _mm512_abs_epi32(diff);
    let vw = _mm512_loadu_si512(WEIGHT_VEC.as_ptr() as *const __m512i);
    let weighted = _mm512_mullo_epi32(abs_diff, vw);
    let sum16 = _mm512_reduce_add_epi32(weighted) as u32;
    // 17th dim: weight = 1
    let d16 = (a[16] as i32 - b[16] as i32).unsigned_abs();
    sum16 + d16
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_weighted_avx2(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    // First 8 dims
    let va0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i));
    let vb0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr() as *const __m128i));
    let diff0 = _mm256_sub_epi32(va0, vb0);
    let abs0 = _mm256_abs_epi32(diff0);
    let vw0 = _mm256_loadu_si256(WEIGHT_VEC.as_ptr() as *const __m256i);
    let w0 = _mm256_mullo_epi32(abs0, vw0);

    // Dims 8..16
    let va1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a[8..].as_ptr() as *const __m128i));
    let vb1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b[8..].as_ptr() as *const __m128i));
    let diff1 = _mm256_sub_epi32(va1, vb1);
    let abs1 = _mm256_abs_epi32(diff1);
    let vw1 = _mm256_loadu_si256(WEIGHT_VEC[8..].as_ptr() as *const __m256i);
    let w1 = _mm256_mullo_epi32(abs1, vw1);

    let sum = _mm256_add_epi32(w0, w1);
    // Horizontal sum
    let hi128 = _mm256_extracti128_si256(sum, 1);
    let lo128 = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let s = _mm_extract_epi32(sum32, 0) as u32;
    // 17th dim: weight = 1
    let d16 = (a[16] as i32 - b[16] as i32).unsigned_abs();
    s + d16
}

fn l1_weighted_scalar(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    let mut d = 0u32;
    for i in 0..17 {
        let diff = (a[i] as i32 - b[i] as i32).unsigned_abs();
        let weight = if i == 0 { 20 } else if i < 7 { 3 } else { 1 };
        d += diff * weight;
    }
    d
}

static L1_WEIGHTED_KERNEL: std::sync::LazyLock<L1WeightedFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return l1_weighted_avx512 as L1WeightedFn;
        }
        if is_x86_feature_detected!("avx2") {
            return l1_weighted_avx2 as L1WeightedFn;
        }
    }
    l1_weighted_scalar as L1WeightedFn
});

// ============================================================================
// Multi-versioned sign_agreement kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

type SignAgreementFn = unsafe fn(&[i16; 17], &[i16; 17]) -> u32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sign_agreement_avx512(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    let va = _mm512_cvtepi16_epi32(_mm256_loadu_si256(a.as_ptr() as *const __m256i));
    let vb = _mm512_cvtepi16_epi32(_mm256_loadu_si256(b.as_ptr() as *const __m256i));
    // XOR: same sign → non-negative, different sign → negative
    let xor = _mm512_xor_si512(va, vb);
    // Compare >= 0: mask bit set where same sign
    let zero = _mm512_setzero_si512();
    let mask = _mm512_cmpge_epi32_mask(xor, zero);
    let count16 = mask.count_ones();
    // 17th dim
    let same17 = if (a[16] >= 0) == (b[16] >= 0) { 1u32 } else { 0u32 };
    count16 + same17
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sign_agreement_avx2(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    use std::arch::x86_64::*;
    // First 8 dims
    let va0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i));
    let vb0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr() as *const __m128i));
    let xor0 = _mm256_xor_si256(va0, vb0);
    let zero = _mm256_setzero_si256();
    let neg0 = _mm256_cmpgt_epi32(zero, xor0); // -1 where xor < 0
    // movemask_ps on the reinterpreted float gives 8 bits, one per 32-bit lane
    let mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(neg0)) as u32;
    let same0 = 8 - mask0.count_ones();

    // Dims 8..16
    let va1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a[8..].as_ptr() as *const __m128i));
    let vb1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b[8..].as_ptr() as *const __m128i));
    let xor1 = _mm256_xor_si256(va1, vb1);
    let neg1 = _mm256_cmpgt_epi32(zero, xor1);
    let mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(neg1)) as u32;
    let same1 = 8 - mask1.count_ones();

    // 17th dim
    let same17 = if (a[16] >= 0) == (b[16] >= 0) { 1u32 } else { 0u32 };
    same0 + same1 + same17
}

fn sign_agreement_scalar(a: &[i16; 17], b: &[i16; 17]) -> u32 {
    let mut count = 0u32;
    for i in 0..17 {
        if (a[i] >= 0) == (b[i] >= 0) {
            count += 1;
        }
    }
    count
}

static SIGN_AGREEMENT_KERNEL: std::sync::LazyLock<SignAgreementFn> =
    std::sync::LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return sign_agreement_avx512 as SignAgreementFn;
            }
            if is_x86_feature_detected!("avx2") {
                return sign_agreement_avx2 as SignAgreementFn;
            }
        }
        sign_agreement_scalar as SignAgreementFn
    });

// ============================================================================
// Multi-versioned xor_bind kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

type XorBindFn = unsafe fn(&[i16; 17], &[i16; 17]) -> [i16; 17];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xor_bind_avx512(a: &[i16; 17], b: &[i16; 17]) -> [i16; 17] {
    use std::arch::x86_64::*;
    // Load 16 i16 as i32, XOR, store back as i16
    let va = _mm512_cvtepi16_epi32(_mm256_loadu_si256(a.as_ptr() as *const __m256i));
    let vb = _mm512_cvtepi16_epi32(_mm256_loadu_si256(b.as_ptr() as *const __m256i));
    let xored = _mm512_xor_si512(va, vb);
    // Convert back to i16: truncate i32 -> i16 via pmovdw
    let packed = _mm512_cvtepi32_epi16(xored);
    let mut dims = [0i16; 17];
    _mm256_storeu_si256(dims.as_mut_ptr() as *mut __m256i, packed);
    dims[16] = (a[16] as u16 ^ b[16] as u16) as i16;
    dims
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_bind_avx2(a: &[i16; 17], b: &[i16; 17]) -> [i16; 17] {
    use std::arch::x86_64::*;
    // First 8 dims: load as i32, XOR, narrow back
    let va0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i));
    let vb0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr() as *const __m128i));
    let xor0 = _mm256_xor_si256(va0, vb0);

    // Dims 8..16
    let va1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(a[8..].as_ptr() as *const __m128i));
    let vb1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(b[8..].as_ptr() as *const __m128i));
    let xor1 = _mm256_xor_si256(va1, vb1);

    // Extract results back to i16
    let mut dims = [0i16; 17];
    // Pack i32 -> i16 via shuffle + truncation
    // We need the low 16 bits of each i32 lane.
    // Use _mm256_packs_epi32 which saturates — but XOR of two i16 fits in i16,
    // so we use manual extraction instead to avoid saturation issues.
    let arr0: [i32; 8] = core::mem::transmute(xor0);
    let arr1: [i32; 8] = core::mem::transmute(xor1);
    for i in 0..8 {
        dims[i] = arr0[i] as i16;
    }
    for i in 0..8 {
        dims[8 + i] = arr1[i] as i16;
    }
    dims[16] = (a[16] as u16 ^ b[16] as u16) as i16;
    dims
}

fn xor_bind_scalar(a: &[i16; 17], b: &[i16; 17]) -> [i16; 17] {
    let mut dims = [0i16; 17];
    for i in 0..17 {
        dims[i] = (a[i] as u16 ^ b[i] as u16) as i16;
    }
    dims
}

static XOR_BIND_KERNEL: std::sync::LazyLock<XorBindFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return xor_bind_avx512 as XorBindFn;
        }
        if is_x86_feature_detected!("avx2") {
            return xor_bind_avx2 as XorBindFn;
        }
    }
    xor_bind_scalar as XorBindFn
});

// ============================================================================
// Multi-versioned inject_noise kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

type InjectNoiseFn = unsafe fn(&[i16; 17], i16, u64) -> [i16; 17];

/// Deterministic PRNG step (PCG-like LCG).
#[inline(always)]
fn prng_step(state: &mut u64) {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
}

/// Compute noise value from PRNG state.
#[inline(always)]
fn noise_from_state(state: u64, scale: i16) -> i16 {
    ((state >> 33) as i16).wrapping_mul(scale) >> 15
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn inject_noise_avx512(dims: &[i16; 17], scale: i16, seed: u64) -> [i16; 17] {
    use std::arch::x86_64::*;
    // Generate 16 noise values via PRNG
    let mut state = seed;
    let mut noise_vals = [0i32; 16];
    for i in 0..16 {
        prng_step(&mut state);
        noise_vals[i] = noise_from_state(state, scale) as i32;
    }
    // Load dims as i32
    let vd = _mm512_cvtepi16_epi32(_mm256_loadu_si256(dims.as_ptr() as *const __m256i));
    let vn = _mm512_loadu_si512(noise_vals.as_ptr() as *const __m512i);
    // Saturating add: add then clamp to i16 range
    let sum = _mm512_add_epi32(vd, vn);
    let lo = _mm512_set1_epi32(-32768);
    let hi = _mm512_set1_epi32(32767);
    let clamped = _mm512_max_epi32(_mm512_min_epi32(sum, hi), lo);
    let packed = _mm512_cvtepi32_epi16(clamped);
    let mut result = [0i16; 17];
    _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, packed);
    // 17th dim
    prng_step(&mut state);
    let n16 = noise_from_state(state, scale);
    result[16] = dims[16].saturating_add(n16);
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inject_noise_avx2(dims: &[i16; 17], scale: i16, seed: u64) -> [i16; 17] {
    use std::arch::x86_64::*;
    let mut state = seed;
    // First 8 dims
    let mut noise0 = [0i32; 8];
    for i in 0..8 {
        prng_step(&mut state);
        noise0[i] = noise_from_state(state, scale) as i32;
    }
    let vd0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(dims.as_ptr() as *const __m128i));
    let vn0 = _mm256_loadu_si256(noise0.as_ptr() as *const __m256i);
    let sum0 = _mm256_add_epi32(vd0, vn0);

    // Dims 8..16
    let mut noise1 = [0i32; 8];
    for i in 0..8 {
        prng_step(&mut state);
        noise1[i] = noise_from_state(state, scale) as i32;
    }
    let vd1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(dims[8..].as_ptr() as *const __m128i));
    let vn1 = _mm256_loadu_si256(noise1.as_ptr() as *const __m256i);
    let sum1 = _mm256_add_epi32(vd1, vn1);

    // Clamp and extract
    let lo = _mm256_set1_epi32(-32768);
    let hi = _mm256_set1_epi32(32767);
    let c0 = _mm256_max_epi32(_mm256_min_epi32(sum0, hi), lo);
    let c1 = _mm256_max_epi32(_mm256_min_epi32(sum1, hi), lo);

    let arr0: [i32; 8] = core::mem::transmute(c0);
    let arr1: [i32; 8] = core::mem::transmute(c1);
    let mut result = [0i16; 17];
    for i in 0..8 {
        result[i] = arr0[i] as i16;
    }
    for i in 0..8 {
        result[8 + i] = arr1[i] as i16;
    }
    // 17th dim
    prng_step(&mut state);
    let n16 = noise_from_state(state, scale);
    result[16] = dims[16].saturating_add(n16);
    result
}

fn inject_noise_scalar(dims: &[i16; 17], scale: i16, seed: u64) -> [i16; 17] {
    let mut result = [0i16; 17];
    result.copy_from_slice(dims);
    let mut state = seed;
    for d in 0..17 {
        prng_step(&mut state);
        let noise = noise_from_state(state, scale);
        result[d] = result[d].saturating_add(noise);
    }
    result
}

static INJECT_NOISE_KERNEL: std::sync::LazyLock<InjectNoiseFn> =
    std::sync::LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return inject_noise_avx512 as InjectNoiseFn;
            }
            if is_x86_feature_detected!("avx2") {
                return inject_noise_avx2 as InjectNoiseFn;
            }
        }
        inject_noise_scalar as InjectNoiseFn
    });

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

    /// L1 (Manhattan) distance — multi-versioned kernel.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 → AVX2 → scalar.
    /// One binary serves all ISAs.
    #[inline]
    pub fn l1(&self, other: &Base17) -> u32 {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        unsafe { L1_KERNEL(&self.dims, &other.dims) }
    }

    /// PCDVQ-informed L1: weight sign dimension 20x over mantissa.
    ///
    /// From arxiv 2506.05432: direction (sign) is 20x more sensitive to
    /// quantization than magnitude. BF16 decomposition maps to polar:
    ///   dim 0 = sign (direction), dims 1-6 = exponent (magnitude scale),
    ///   dims 7-16 = mantissa (fine detail).
    /// PCDVQ-weighted L1 via SIMD: sign=20x, magnitude=3x, detail=1x.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 -> AVX2 -> scalar.
    #[inline]
    pub fn l1_weighted(&self, other: &Base17) -> u32 {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        unsafe { L1_WEIGHTED_KERNEL(&self.dims, &other.dims) }
    }

    /// Sign-bit agreement (out of 17) — multi-versioned kernel.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 -> AVX2 -> scalar.
    #[inline]
    pub fn sign_agreement(&self, other: &Base17) -> u32 {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        unsafe { SIGN_AGREEMENT_KERNEL(&self.dims, &other.dims) }
    }

    /// XOR bind: path composition in hyperdimensional space.
    /// Self-inverse: `a.xor_bind(&b).xor_bind(&b) == a`.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 -> AVX2 -> scalar.
    #[inline]
    pub fn xor_bind(&self, other: &Base17) -> Base17 {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        let dims = unsafe { XOR_BIND_KERNEL(&self.dims, &other.dims) };
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

    /// #6 Thought Randomization — calibrated noise injection on Base17.
    /// Flip dims with magnitude proportional to coefficient of variation.
    /// Science: Kirkpatrick et al. (1983), Rahimi & Recht (2007).
    ///
    /// Runtime dispatch via LazyLock: AVX-512 -> AVX2 -> scalar.
    pub fn inject_noise(&self, cv: f32, seed: u64) -> Base17 {
        let scale = (cv * 32767.0).min(32767.0) as i16;
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        let dims = unsafe { INJECT_NOISE_KERNEL(&self.dims, scale, seed) };
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
    fn test_inject_noise() {
        let b = Base17 { dims: [100; 17] };
        let noisy = b.inject_noise(0.1, 42);
        assert_ne!(b.dims, noisy.dims); // should be different
        let dist = b.l1(&noisy);
        assert!(dist > 0); // noise injected
        assert!(dist < 17 * 32767); // not totally destroyed
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
