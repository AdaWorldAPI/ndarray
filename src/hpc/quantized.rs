//! Quantized GEMM: BF16, Int8, Int4 (Q4_0 GGUF) matrix multiplication.
//!
//! Provides BF16 (bfloat16) type with conversions, BF16 GEMM with f32 accumulation,
//! int8 quantized GEMM, and GGUF Q4_0 block-quantized helpers.

// Types used only for ndarray integration (Array re-exports)

// ── BF16 ───────────────────────────────────────────────────────────

/// BFloat16: 16-bit floating point with 8-bit exponent (same as f32).
///
/// Provides ~3 decimal digits of precision but same dynamic range as f32.
/// Used for ML inference where full f32 precision isn't needed.
///
/// # Example
///
/// ```
/// use ndarray::hpc::quantized::BF16;
///
/// let val = BF16::from_f32(3.14);
/// let back = val.to_f32();
/// assert!((back - 3.14).abs() < 0.02); // ~1% precision
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct BF16(pub u16);

impl BF16 {
    /// Zero in BF16 representation.
    pub const ZERO: BF16 = BF16(0x0000);
    /// One in BF16 representation (same exponent/mantissa layout as f32).
    pub const ONE: BF16 = BF16(0x3F80);

    /// Convert f32 to BF16 by truncation (drops the lower 16 bits).
    ///
    /// Note: despite the plain name, this method **truncates** rather than
    /// rounding. In rustyblas the truncating variant is called
    /// `from_f32_truncate`; use that alias if you prefer explicit naming.
    pub fn from_f32(v: f32) -> Self {
        BF16((v.to_bits() >> 16) as u16)
    }

    /// Alias for [`from_f32`](Self::from_f32) — truncating conversion.
    ///
    /// Provided so that code following the rustyblas naming convention
    /// (`from_f32_truncate` truncates, `from_f32` rounds) works without
    /// changes.
    #[inline]
    pub fn from_f32_truncate(v: f32) -> Self {
        Self::from_f32(v)
    }

    /// Convert f32 to BF16 with round-to-nearest-even.
    ///
    /// This is the higher-quality conversion; prefer it when precision
    /// matters. In rustyblas the rounding variant is simply called `from_f32`.
    pub fn from_f32_rounded(v: f32) -> Self {
        let bits = v.to_bits();
        let round_bit = (bits >> 15) & 1;
        let sticky = if bits & 0x7FFF != 0 { 1u32 } else { 0 };
        let rounded = (bits >> 16).wrapping_add(round_bit & (sticky | ((bits >> 16) & 1)));
        BF16(rounded as u16)
    }

    /// Convert BF16 to f32.
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// Convert f32 slice to BF16 (truncation).
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [BF16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = BF16::from_f32(src[i]);
    }
}

/// Convert f32 slice to BF16 (round-to-nearest-even).
pub fn f32_to_bf16_rounded(src: &[f32], dst: &mut [BF16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = BF16::from_f32_rounded(src[i]);
    }
}

/// Convert BF16 slice to f32.
pub fn bf16_to_f32_slice(src: &[BF16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = src[i].to_f32();
    }
}

/// Convert f32 vec to BF16 vec.
pub fn f32_vec_to_bf16(src: &[f32]) -> Vec<BF16> {
    src.iter().map(|&v| BF16::from_f32(v)).collect()
}

/// Convert BF16 vec to f32 vec.
pub fn bf16_vec_to_f32(src: &[BF16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

// ── F16 (IEEE 754 binary16) ────────────────────────────────────────

/// IEEE 754 binary16 (half-precision): 1 sign + 5 exponent + 10 mantissa bits.
///
/// Range ±65504, ~3 decimal digits of precision. Smaller dynamic range than
/// [`BF16`] but more mantissa precision; typical for graphics / Apple Neural
/// Engine / NVIDIA tensor cores.
///
/// # Example
///
/// ```
/// use ndarray::hpc::quantized::F16;
///
/// let val = F16::from_f32(1.0);
/// assert_eq!(val, F16::ONE);
/// assert_eq!(val.to_f32(), 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// Zero in F16.
    pub const ZERO: F16 = F16(0x0000);
    /// One in F16.
    pub const ONE: F16 = F16(0x3C00);
    /// Negative one.
    pub const NEG_ONE: F16 = F16(0xBC00);
    /// Positive infinity.
    pub const INFINITY: F16 = F16(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: F16 = F16(0xFC00);
    /// Quiet NaN (canonical bit pattern).
    pub const NAN: F16 = F16(0x7E00);

    /// Convert f32 to F16 with round-to-nearest-even (the default rule for
    /// IEEE 754 binary16).
    ///
    /// Tested against: `0.0 → 0x0000`, `1.0 → 0x3C00`, `-2.0 → 0xC000`,
    /// `+inf → 0x7C00`, NaN → quiet NaN.
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self::from_f32_rounded(v)
    }

    /// Truncating f32 → F16 conversion (drop low mantissa bits, no rounding).
    ///
    /// Faster than [`Self::from_f32_rounded`] but ~0.5 ULP biased.
    #[inline]
    pub fn from_f32_truncate(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        // Inf / NaN
        if exp == 0xFF {
            if mant == 0 {
                return F16(sign | 0x7C00);
            }
            // Preserve NaN-ness; truncate mantissa to top 10 bits, force at
            // least one mantissa bit so it stays NaN (not Inf).
            let mant16 = (mant >> 13) as u16;
            let mant16 = if mant16 == 0 { 1 } else { mant16 };
            return F16(sign | 0x7C00 | mant16);
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1F {
            // Overflow → ±Inf
            return F16(sign | 0x7C00);
        }
        if new_exp <= 0 {
            // Subnormal or underflow to zero.
            if new_exp < -10 {
                return F16(sign);
            }
            // Subnormal: insert implicit leading 1, then shift right.
            let mant_full = mant | 0x0080_0000;
            let shift = 14 - new_exp; // 13 + (1 - new_exp)
            let mant16 = (mant_full >> shift) as u16;
            return F16(sign | mant16);
        }
        let mant16 = (mant >> 13) as u16;
        F16(sign | ((new_exp as u16) << 10) | mant16)
    }

    /// f32 → F16 with round-to-nearest-even.
    pub fn from_f32_rounded(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        if exp == 0xFF {
            // Inf/NaN
            if mant == 0 {
                return F16(sign | 0x7C00);
            }
            let mant16 = (mant >> 13) as u16;
            let mant16 = if mant16 == 0 { 0x200 } else { mant16 | 0x200 };
            return F16(sign | 0x7C00 | mant16);
        }

        let new_exp = exp - 127 + 15;

        // Normal range: 1 ≤ new_exp ≤ 30
        if new_exp >= 0x1F {
            return F16(sign | 0x7C00);
        }
        if new_exp >= 1 {
            // Round-to-nearest-even using the dropped 13 bits.
            let half_bits = mant & 0x0000_1FFF;
            let truncated = (mant >> 13) as u32;
            let mant16 = truncated;
            let half = 0x1000u32;
            let round_up = if half_bits > half {
                1
            } else if half_bits < half {
                0
            } else {
                // tie → round to even
                mant16 & 1
            };
            let mant16 = mant16 + round_up;
            // Mantissa overflow bumps exponent.
            if mant16 == 0x400 {
                let new_exp = new_exp + 1;
                if new_exp >= 0x1F {
                    return F16(sign | 0x7C00);
                }
                return F16(sign | ((new_exp as u16) << 10));
            }
            return F16(sign | ((new_exp as u16) << 10) | (mant16 as u16));
        }

        // Subnormal / underflow.
        if new_exp < -10 {
            // Underflow to zero (still RNE: half-of-min subnormal could round
            // up, but here we treat exponent below -10 as truly tiny).
            return F16(sign);
        }
        // Build the full mantissa with implicit leading 1 then shift.
        let mant_full = mant | 0x0080_0000; // 24-bit
        let shift = (14 - new_exp) as u32; // 13 + (1 - new_exp)
        let truncated = mant_full >> shift;
        let dropped_mask = (1u32 << shift) - 1;
        let dropped = mant_full & dropped_mask;
        let half = 1u32 << (shift - 1);
        let round_up = if dropped > half {
            1
        } else if dropped < half {
            0
        } else {
            truncated & 1
        };
        let mant16 = (truncated + round_up) as u16;
        // mant16 may be 0x400 = 1024 → next exponent (which is 1, normal)
        F16(sign | mant16)
    }

    /// F16 → f32 (lossless, since binary32 strictly subsumes binary16).
    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x03FF;

        let bits = if exp == 0 {
            if mant == 0 {
                // ±0
                sign
            } else {
                // Subnormal: normalize.
                let mut m = mant;
                let mut e: i32 = 1;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                let m = m & 0x03FF;
                let new_exp = (e - 1 + 127 - 14) as u32;
                sign | (new_exp << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            // Inf or NaN.
            if mant == 0 {
                sign | 0x7F80_0000
            } else {
                sign | 0x7F80_0000 | (mant << 13)
            }
        } else {
            let new_exp = exp + (127 - 15);
            sign | (new_exp << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

/// Convert f32 slice to F16 (round-to-nearest-even).
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [F16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = F16::from_f32(src[i]);
    }
}

/// Convert F16 slice to f32.
pub fn f16_to_f32_slice(src: &[F16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = src[i].to_f32();
    }
}

/// Convert f32 vec to F16 vec.
pub fn f32_vec_to_f16(src: &[f32]) -> Vec<F16> {
    src.iter().map(|&v| F16::from_f32(v)).collect()
}

/// Convert F16 vec to f32 vec.
pub fn f16_vec_to_f32(src: &[F16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

// ── Operator + num_traits impls for BF16 + F16 ─────────────────────

macro_rules! impl_half_ops {
    ($t:ident) => {
        impl core::ops::Add for $t {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                $t::from_f32_rounded(self.to_f32() + rhs.to_f32())
            }
        }
        impl core::ops::Sub for $t {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                $t::from_f32_rounded(self.to_f32() - rhs.to_f32())
            }
        }
        impl core::ops::Mul for $t {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $t::from_f32_rounded(self.to_f32() * rhs.to_f32())
            }
        }
        impl core::ops::Div for $t {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                $t::from_f32_rounded(self.to_f32() / rhs.to_f32())
            }
        }
        impl core::ops::Rem for $t {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: Self) -> Self {
                $t::from_f32_rounded(self.to_f32() % rhs.to_f32())
            }
        }
        impl core::ops::Neg for $t {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(self.0 ^ 0x8000)
            }
        }
        impl core::ops::AddAssign for $t {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl core::ops::SubAssign for $t {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl core::ops::MulAssign for $t {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl core::ops::DivAssign for $t {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
        impl core::ops::RemAssign for $t {
            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                *self = *self % rhs;
            }
        }
        impl num_traits::Zero for $t {
            #[inline]
            fn zero() -> Self {
                Self::ZERO
            }
            #[inline]
            fn is_zero(&self) -> bool {
                // Treat both +0 and -0 as zero.
                (self.0 & 0x7FFF) == 0
            }
        }
        impl num_traits::One for $t {
            #[inline]
            fn one() -> Self {
                Self::ONE
            }
        }
        impl Default for $t {
            #[inline]
            fn default() -> Self {
                Self::ZERO
            }
        }
        impl core::fmt::Display for $t {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}", self.to_f32())
            }
        }
        impl crate::ScalarOperand for $t {}
    };
}

impl_half_ops!(BF16);
impl_half_ops!(F16);

/// BF16 GEMM with f32 accumulation: C = alpha * A * B + beta * C
///
/// A and B are BF16, C is f32. Accumulation done in f32 for precision.
pub fn bf16_gemm_f32(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    // Apply beta
    if beta == 0.0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
    }

    // Tiled multiply
    const TILE: usize = 32;
    let mut kk = 0;
    while kk < k {
        let kb = TILE.min(k - kk);
        let mut ii = 0;
        while ii < m {
            let ib = TILE.min(m - ii);
            let mut jj = 0;
            while jj < n {
                let jb = TILE.min(n - jj);
                for i in 0..ib {
                    for p in 0..kb {
                        let a_val = alpha * a[(ii + i) * k + (kk + p)].to_f32();
                        for j in 0..jb {
                            c[(ii + i) * n + (jj + j)] +=
                                a_val * b[(kk + p) * n + (jj + j)].to_f32();
                        }
                    }
                }
                jj += jb;
            }
            ii += ib;
        }
        kk += kb;
    }
}

/// Mixed precision GEMM: f32 inputs, BF16 compute, f32 output.
pub fn mixed_precision_gemm(
    a_f32: &[f32],
    b_f32: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let a_bf16 = f32_vec_to_bf16(a_f32);
    let b_bf16 = f32_vec_to_bf16(b_f32);
    bf16_gemm_f32(&a_bf16, &b_bf16, c, m, n, k, alpha, beta);
}

// ── Int8 Quantization ──────────────────────────────────────────────

/// Quantization parameters.
#[derive(Clone, Debug)]
pub struct QuantParams {
    /// Scale factor.
    pub scale: f32,
    /// Zero point.
    pub zero_point: i32,
    /// Minimum f32 value.
    pub min_val: f32,
    /// Maximum f32 value.
    pub max_val: f32,
}

/// Per-channel quantization parameters.
#[derive(Clone, Debug)]
pub struct PerChannelQuantParams {
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    /// Per-row zero points.
    pub zero_points: Vec<i32>,
}

/// Quantize f32 to u8.
pub fn quantize_f32_to_u8(data: &[f32]) -> (Vec<u8>, QuantParams) {
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let scale = if max_val > min_val {
        (max_val - min_val) / 255.0
    } else {
        1.0
    };
    let zero_point = (-min_val / scale).round() as i32;
    let zero_point = zero_point.clamp(0, 255);

    let quantized: Vec<u8> = data
        .iter()
        .map(|&v| ((v / scale + zero_point as f32).round() as i32).clamp(0, 255) as u8)
        .collect();

    (quantized, QuantParams { scale, zero_point, min_val, max_val })
}

/// Quantize f32 to i8.
pub fn quantize_f32_to_i8(data: &[f32]) -> (Vec<i8>, QuantParams) {
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let abs_max = min_val.abs().max(max_val.abs());
    let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();

    (quantized, QuantParams { scale, zero_point: 0, min_val, max_val })
}

/// Dequantize i8 codes back to f32 using the [`QuantParams`] from
/// [`quantize_f32_to_i8`].
///
/// ndarray's symmetric i8 quantization stores `code = round(val / scale)`,
/// so dequantization is `val ≈ code * scale`. Returns up to `n` values
/// (extra trailing codes are ignored).
///
/// # Example
///
/// ```
/// use ndarray::hpc::quantized::{quantize_f32_to_i8, dequantize_i8_to_f32};
///
/// let data = vec![1.0f32, -1.0, 0.5, -0.5];
/// let (codes, params) = quantize_f32_to_i8(&data);
/// let recovered = dequantize_i8_to_f32(&codes, &params, data.len());
/// for (a, b) in data.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 0.05);
/// }
/// ```
pub fn dequantize_i8_to_f32(codes: &[i8], params: &QuantParams, n: usize) -> Vec<f32> {
    codes
        .iter()
        .take(n)
        .map(|&c| c as f32 * params.scale)
        .collect()
}

/// Per-channel i8 quantization (per row).
pub fn quantize_per_channel_i8(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, PerChannelQuantParams) {
    let mut quantized = vec![0i8; data.len()];
    let mut scales = Vec::with_capacity(rows);
    let mut zero_points = Vec::with_capacity(rows);

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_data = &data[start..end];
        let abs_max = row_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        scales.push(scale);
        zero_points.push(0);

        for (i, &v) in row_data.iter().enumerate() {
            quantized[start + i] = (v / scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (quantized, PerChannelQuantParams { scales, zero_points })
}

/// Int8 GEMM with i32 accumulation: C = A * B
pub fn int8_gemm_i32(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for v in c.iter_mut() {
        *v = 0;
    }
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p] as i32;
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j] as i32;
            }
        }
    }
}

/// Int8 GEMM with f32 dequantization.
pub fn int8_gemm_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    zero_point_a: i32,
    scale_b: f32,
) {
    let mut c_i32 = vec![0i32; m * n];
    int8_gemm_i32(a, b, &mut c_i32, m, n, k);
    let scale = scale_a * scale_b;
    for i in 0..m {
        for j in 0..n {
            // Adjust for zero point
            let mut acc = c_i32[i * n + j];
            // Subtract zero_point_a contribution
            let mut col_sum = 0i32;
            for p in 0..k {
                col_sum += b[p * n + j] as i32;
            }
            acc -= zero_point_a * col_sum;
            c[i * n + j] = acc as f32 * scale;
        }
    }
}

/// Per-channel int8 GEMM with f32 output.
pub fn int8_gemm_per_channel_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    a_scales: &[f32],
    a_zero_points: &[i32],
    b_scales: &[f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += (a[i * k + p] as i32 - a_zero_points[i]) * b[p * n + j] as i32;
            }
            c[i * n + j] = acc as f32 * a_scales[i] * b_scales[j];
        }
    }
}

/// Quantize f32 to i4 (packed: two i4 values per byte).
pub fn quantize_f32_to_i4(data: &[f32]) -> (Vec<u8>, QuantParams) {
    let abs_max = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };

    let packed_len = (data.len() + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for (i, &v) in data.iter().enumerate() {
        let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
        let nibble = (q & 0x0F) as u8;
        if i % 2 == 0 {
            packed[i / 2] |= nibble;
        } else {
            packed[i / 2] |= nibble << 4;
        }
    }

    (
        packed,
        QuantParams {
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
        },
    )
}

/// Dequantize i4 (packed) to f32.
pub fn dequantize_i4_to_f32(packed: &[u8], params: &QuantParams, len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        };
        // Sign-extend from 4 bits
        let val = if nibble & 0x08 != 0 {
            nibble as i8 | !0x0F_u8 as i8
        } else {
            nibble as i8
        };
        result.push(val as f32 * params.scale);
    }
    result
}

/// 2-bit symmetric quantization: maps values to `{-1, 0, +1}` packed 4 per byte.
///
/// Encoding: 2-bit signed value per element, stored in LSB-first order within
/// each byte. Bit pattern: `0b11 = -1`, `0b00 = 0`, `0b01 = +1`. Output
/// length is `ceil(data.len() / 4)` bytes. Use [`dequantize_i2_to_f32`] for
/// the inverse, supplying the original element count.
///
/// `params.scale` is the original `abs_max` (not `abs_max / Q_RANGE`),
/// because the quantized magnitudes are already `{-1, 0, +1}`.
pub fn quantize_f32_to_i2(data: &[f32]) -> (Vec<u8>, QuantParams) {
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let abs_max = min_val.abs().max(max_val.abs());
    let scale = if abs_max > 0.0 { 1.0 / abs_max } else { 1.0 };

    let mut packed = vec![0u8; (data.len() + 3) / 4];
    for (i, &v) in data.iter().enumerate() {
        let q = (v * scale).round().clamp(-1.0, 1.0) as i8;
        let bits = (q & 0x03) as u8;
        packed[i / 4] |= bits << ((i % 4) * 2);
    }
    (
        packed,
        QuantParams {
            scale: abs_max,
            zero_point: 0,
            min_val,
            max_val,
        },
    )
}

/// Dequantize 2-bit packed codes back to f32.
///
/// Inverse of [`quantize_f32_to_i2`]. `n` is the original element count
/// (the packed buffer is implicitly padded to a byte boundary).
///
/// # Panics
/// Panics if `packed.len() * 4 < n` — the buffer is too short to decode `n`
/// elements. This guards against truncated payloads or inconsistent metadata
/// from untrusted sources.
pub fn dequantize_i2_to_f32(packed: &[u8], params: &QuantParams, n: usize) -> Vec<f32> {
    assert!(
        packed.len() * 4 >= n,
        "dequantize_i2_to_f32: packed buffer holds {} elements, requested {}",
        packed.len() * 4,
        n,
    );
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let byte = packed[i / 4];
        let bits = (byte >> ((i % 4) * 2)) & 0x03;
        let val = match bits {
            0b11 => -1.0f32,
            0b01 => 1.0,
            _ => 0.0,
        };
        out.push(val * params.scale);
    }
    out
}

// ── Q4_0 (GGUF-compatible block quantization) ─────────────────────

/// Q4_0 block size (number of f32 elements per block).
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Number of bytes used to pack one Q4_0 block (32 nibbles = 16 bytes).
pub const Q4_0_BYTES_PER_BLOCK: usize = Q4_0_BLOCK_SIZE / 2;

/// Quantize f32 to Q4_0 (GGUF-compatible) per-block 4-bit quantization.
///
/// Each block of [`Q4_0_BLOCK_SIZE`] (32) f32 elements is encoded as
/// [`Q4_0_BYTES_PER_BLOCK`] (16) packed bytes plus one f32 scale `d`.
///
/// Encoding (matches `llama.cpp` / GGUF reference):
///   - `d = max(|x_i|) / -8.0` (negated max-abs, signed)
///   - `q_i = clamp(round(x_i / d) + 8, 0, 15)` (unsigned nibble in `0..=15`)
///   - Within a block, element `j` (`0 <= j < 16`) and element `j + 16`
///     share byte `j`: low nibble holds element `j`, high nibble holds
///     element `j + 16`. (This is the GGUF interleaved layout, NOT the
///     simple "two consecutive elements per byte" layout used by
///     [`quantize_f32_to_i4`].)
///
/// Returns `(packed_bytes, scales)` where `scales.len() == data.len() / 32`
/// and `packed_bytes.len() == scales.len() * 16`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of [`Q4_0_BLOCK_SIZE`].
pub fn quantize_f32_to_q4_0(data: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert!(
        data.len() % Q4_0_BLOCK_SIZE == 0,
        "Q4_0 requires data.len() to be a multiple of {}",
        Q4_0_BLOCK_SIZE
    );

    let n_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let mut packed = vec![0u8; n_blocks * Q4_0_BYTES_PER_BLOCK];
    let mut scales = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let block = &data[b * Q4_0_BLOCK_SIZE..(b + 1) * Q4_0_BLOCK_SIZE];

        // Find signed max-abs (preserving sign of the extreme element).
        let mut amax = 0.0f32;
        let mut max_signed = 0.0f32;
        for &x in block {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
                max_signed = x;
            }
        }

        // d = max_signed / -8 ; if all-zero block, d = 0 and all q = 8.
        let d = if amax > 0.0 { max_signed / -8.0 } else { 0.0 };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);

        let byte_off = b * Q4_0_BYTES_PER_BLOCK;
        for j in 0..Q4_0_BYTES_PER_BLOCK {
            let lo = ((block[j] * id).round() + 8.5).floor().clamp(0.0, 15.0) as u8;
            let hi = ((block[j + Q4_0_BYTES_PER_BLOCK] * id).round() + 8.5)
                .floor()
                .clamp(0.0, 15.0) as u8;
            packed[byte_off + j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
        }
    }

    (packed, scales)
}

/// Dequantize Q4_0 (GGUF-compatible) packed bytes back to f32.
///
/// Inverse of [`quantize_f32_to_q4_0`]. `packed.len()` must equal
/// `scales.len() * 16` and the result has length `scales.len() * 32`.
///
/// # Panics
///
/// Panics if `packed.len() != scales.len() * Q4_0_BYTES_PER_BLOCK`.
pub fn dequantize_q4_0_to_f32(packed: &[u8], scales: &[f32]) -> Vec<f32> {
    assert_eq!(
        packed.len(),
        scales.len() * Q4_0_BYTES_PER_BLOCK,
        "Q4_0 packed length must equal scales.len() * {}",
        Q4_0_BYTES_PER_BLOCK
    );

    let n_blocks = scales.len();
    let mut out = vec![0.0f32; n_blocks * Q4_0_BLOCK_SIZE];

    for b in 0..n_blocks {
        let d = scales[b];
        let byte_off = b * Q4_0_BYTES_PER_BLOCK;
        let elem_off = b * Q4_0_BLOCK_SIZE;
        for j in 0..Q4_0_BYTES_PER_BLOCK {
            let byte = packed[byte_off + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[elem_off + j] = lo as f32 * d;
            out[elem_off + j + Q4_0_BYTES_PER_BLOCK] = hi as f32 * d;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let values = [1.0f32, -1.0, 0.0, 3.14, 1000.0, 0.001];
        for &v in &values {
            let bf = BF16::from_f32(v);
            let back = bf.to_f32();
            assert!((back - v).abs() / v.abs().max(1.0) < 0.02, "BF16 roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_bf16_gemm() {
        // 2x2 * 2x2
        let a = vec![BF16::from_f32(1.0), BF16::from_f32(2.0), BF16::from_f32(3.0), BF16::from_f32(4.0)];
        let b = vec![BF16::from_f32(5.0), BF16::from_f32(6.0), BF16::from_f32(7.0), BF16::from_f32(8.0)];
        let mut c = vec![0.0f32; 4];
        bf16_gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert!((c[0] - 19.0).abs() < 0.5);
        assert!((c[3] - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_quantize_u8() {
        let data = vec![0.0, 0.5, 1.0, -1.0];
        let (q, params) = quantize_f32_to_u8(&data);
        assert_eq!(q.len(), 4);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quantize_i8() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5];
        let (q, _params) = quantize_f32_to_i8(&data);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_int8_gemm() {
        let a: Vec<u8> = vec![128, 128, 128, 128]; // centered at 128
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut c = vec![0i32; 4];
        int8_gemm_i32(&a, &b, &mut c, 2, 2, 2);
        // Row 0: 128*1+128*0=128, 128*0+128*1=128
        assert_eq!(c[0], 128);
        assert_eq!(c[1], 128);
    }

    #[test]
    fn test_bf16_zero_one_constants() {
        assert_eq!(BF16::ZERO.to_f32(), 0.0);
        assert_eq!(BF16::ONE.to_f32(), 1.0);
        // Also verify from_f32_truncate alias works
        assert_eq!(BF16::from_f32_truncate(1.0), BF16::ONE);
    }

    #[test]
    fn test_i4_roundtrip() {
        let data = vec![1.0f32, -1.0, 3.0, -3.0, 7.0, -7.0];
        let (packed, params) = quantize_f32_to_i4(&data);
        let recovered = dequantize_i4_to_f32(&packed, &params, data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.5, "i4 roundtrip: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_i8_roundtrip() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.25, -0.25];
        let (codes, params) = quantize_f32_to_i8(&data);
        let recovered = dequantize_i8_to_f32(&codes, &params, data.len());
        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.02, "i8 roundtrip: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_i2_roundtrip() {
        // Values cluster near {-1, 0, +1} after scaling.
        let data = vec![1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let (packed, params) = quantize_f32_to_i2(&data);
        assert_eq!(packed.len(), 2); // ceil(8/4) = 2 bytes
        let recovered = dequantize_i2_to_f32(&packed, &params, data.len());
        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-5, "i2 roundtrip: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_f16_special_constants() {
        assert_eq!(F16::ZERO.0, 0x0000);
        assert_eq!(F16::ONE.0, 0x3C00);
        assert_eq!(F16::INFINITY.0, 0x7C00);
        assert_eq!(F16::NEG_INFINITY.0, 0xFC00);
    }

    #[test]
    fn test_f16_from_f32_known_bits() {
        assert_eq!(F16::from_f32(0.0).0, 0x0000);
        assert_eq!(F16::from_f32(-0.0).0, 0x8000);
        assert_eq!(F16::from_f32(1.0).0, 0x3C00);
        assert_eq!(F16::from_f32(-1.0).0, 0xBC00);
        assert_eq!(F16::from_f32(2.0).0, 0x4000);
        assert_eq!(F16::from_f32(-2.0).0, 0xC000);
        assert_eq!(F16::from_f32(0.5).0, 0x3800);
        assert_eq!(F16::from_f32(-0.5).0, 0xB800);
    }

    #[test]
    fn test_f16_inf_nan() {
        // ±Inf
        assert_eq!(F16::from_f32(f32::INFINITY).0, 0x7C00);
        assert_eq!(F16::from_f32(f32::NEG_INFINITY).0, 0xFC00);
        assert!(F16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(F16::from_f32(f32::INFINITY).to_f32() > 0.0);
        assert!(F16::from_f32(f32::NEG_INFINITY).to_f32() < 0.0);

        // NaN preserved
        let nan = F16::from_f32(f32::NAN);
        assert!(nan.to_f32().is_nan());
        // Top exponent = all-ones, mantissa nonzero
        assert_eq!(nan.0 & 0x7C00, 0x7C00);
        assert!(nan.0 & 0x03FF != 0);
    }

    #[test]
    fn test_f16_roundtrip_exact() {
        let exact = [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25, 65504.0, -65504.0];
        for &v in &exact {
            let h = F16::from_f32(v);
            assert_eq!(h.to_f32(), v, "F16 lost {}", v);
        }
    }

    #[test]
    fn test_f16_roundtrip_approx() {
        let approx = [3.14f32, -3.14, 0.1, 100.0, 1000.0, 1e-3, 1e-4];
        for &v in &approx {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            assert!(
                (back - v).abs() / v.abs().max(1.0) < 0.001,
                "F16 roundtrip {} → {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_f16_negation() {
        let one = F16::ONE;
        let neg_one = -one;
        assert_eq!(neg_one.to_f32(), -1.0);
        let zero = F16::ZERO;
        let neg_zero = -zero;
        assert_eq!(neg_zero.0, 0x8000);
    }

    #[test]
    fn test_f16_arithmetic() {
        let a = F16::from_f32(3.0);
        let b = F16::from_f32(4.0);
        assert!(((a + b).to_f32() - 7.0).abs() < 1e-3);
        assert!(((b - a).to_f32() - 1.0).abs() < 1e-3);
        assert!(((a * b).to_f32() - 12.0).abs() < 1e-3);
        assert!(((b / a).to_f32() - 4.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_bf16_arithmetic() {
        let a = BF16::from_f32_rounded(3.0);
        let b = BF16::from_f32_rounded(4.0);
        assert!(((a + b).to_f32() - 7.0).abs() < 0.05);
        assert!(((b - a).to_f32() - 1.0).abs() < 0.05);
        assert!(((a * b).to_f32() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_half_zero_one_traits() {
        use num_traits::{One, Zero};
        let z: F16 = F16::zero();
        let o: F16 = F16::one();
        assert_eq!(z, F16::ZERO);
        assert_eq!(o, F16::ONE);
        assert!(z.is_zero());
        let bz: BF16 = BF16::zero();
        let bo: BF16 = BF16::one();
        assert_eq!(bz, BF16::ZERO);
        assert_eq!(bo, BF16::ONE);
    }

    #[test]
    fn test_linalg_scalar_compiles() {
        // Smoke test: F16 / BF16 satisfy LinalgScalar bounds.
        fn assert_linalg<T: crate::linalg_traits::LinalgScalar>() {}
        assert_linalg::<F16>();
        assert_linalg::<BF16>();
    }

    #[test]
    fn test_scalar_operand_compiles() {
        fn assert_scalar<T: crate::ScalarOperand>() {}
        assert_scalar::<F16>();
        assert_scalar::<BF16>();
    }

    #[test]
    fn test_i2_packing_layout() {
        // 4 values per byte, LSB first.
        let data = vec![1.0f32, -1.0, 0.0, 1.0]; // bits: 01, 11, 00, 01
        let (packed, _) = quantize_f32_to_i2(&data);
        assert_eq!(packed.len(), 1);
        // byte = (01) | (11 << 2) | (00 << 4) | (01 << 6)
        //      = 0b01_00_11_01 = 0x4D
        assert_eq!(packed[0], 0b01_00_11_01);
    }

    #[test]
    fn test_i4_boundary_values() {
        // With abs_max=7 -> scale=1.0; the i4 grid maps directly:
        // input  -7  ->  q=-7  -> dequant -7.0
        // input   0  ->  q= 0  -> dequant  0.0
        // input   7  ->  q= 7  -> dequant  7.0
        let data = vec![-7.0f32, -3.0, 0.0, 3.0, 7.0];
        let (packed, params) = quantize_f32_to_i4(&data);
        assert!((params.scale - 1.0).abs() < 1e-6);
        let recovered = dequantize_i4_to_f32(&packed, &params, data.len());
        assert_eq!(recovered, vec![-7.0, -3.0, 0.0, 3.0, 7.0]);

        // Negative-end clamp: with abs_max=8 -> scale=8/7, the value -8
        // maps to q=-7 (since -8 / (8/7) = -7), dequantizing to -8.0
        // exactly. The 8 grid cell hits q=7 -> dequant 8.0 exactly.
        let data2 = vec![-8.0f32, 0.0, 8.0];
        let (packed2, params2) = quantize_f32_to_i4(&data2);
        let s = params2.scale;
        assert!((s - 8.0 / 7.0).abs() < 1e-6);
        let rec2 = dequantize_i4_to_f32(&packed2, &params2, data2.len());
        assert!((rec2[0] - -8.0).abs() < 1e-4);
        assert_eq!(rec2[1], 0.0);
        assert!((rec2[2] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_q4_0_roundtrip_single_block() {
        let mut data = Vec::with_capacity(Q4_0_BLOCK_SIZE);
        for i in 0..Q4_0_BLOCK_SIZE {
            data.push((i as f32) - 16.0); // values in [-16, 15]
        }
        let (packed, scales) = quantize_f32_to_q4_0(&data);
        assert_eq!(packed.len(), Q4_0_BYTES_PER_BLOCK);
        assert_eq!(scales.len(), 1);
        let recovered = dequantize_q4_0_to_f32(&packed, &scales);
        assert_eq!(recovered.len(), data.len());
        // Max abs in block is 16. With 4-bit signed grid (16 levels),
        // expected error <= |d| ≈ 16/8 = 2.0.
        let max_abs = 16.0f32;
        let tol = max_abs / 8.0 + 1e-4;
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!((orig - rec).abs() <= tol, "q4_0 roundtrip[{i}]: {orig} vs {rec} (tol {tol})");
        }
    }

    #[test]
    fn test_q4_0_roundtrip_multi_block() {
        // 3 blocks (96 elements), monotonically varying values.
        let n = 3 * Q4_0_BLOCK_SIZE;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) - 48.0) * 0.25).collect();
        let (packed, scales) = quantize_f32_to_q4_0(&data);
        assert_eq!(scales.len(), 3);
        assert_eq!(packed.len(), 3 * Q4_0_BYTES_PER_BLOCK);
        let recovered = dequantize_q4_0_to_f32(&packed, &scales);
        assert_eq!(recovered.len(), n);
        for (b, &d) in scales.iter().enumerate() {
            let tol = d.abs() + 1e-4;
            for j in 0..Q4_0_BLOCK_SIZE {
                let i = b * Q4_0_BLOCK_SIZE + j;
                assert!(
                    (data[i] - recovered[i]).abs() <= tol,
                    "q4_0 multi[{i}] block={b}: {} vs {} tol={tol}",
                    data[i],
                    recovered[i]
                );
            }
        }
    }

    #[test]
    fn test_q4_0_zero_block() {
        let data = vec![0.0f32; Q4_0_BLOCK_SIZE];
        let (packed, scales) = quantize_f32_to_q4_0(&data);
        assert_eq!(scales[0], 0.0);
        let recovered = dequantize_q4_0_to_f32(&packed, &scales);
        for v in recovered {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_packing_layout_interleaved() {
        // Verify GGUF interleaved layout: byte j carries element j (low)
        // and element j + 16 (high), within one 32-element block.
        let mut data = vec![0.0f32; Q4_0_BLOCK_SIZE];
        // Set element 0 to extreme negative so q=15 (since d<0, x/d>0),
        // and leave element 16 at 0 so its q=8.
        data[0] = -1.0;
        data[16] = 0.0;
        let (packed, scales) = quantize_f32_to_q4_0(&data);
        // d = (-1.0) / -8 = 0.125 ; q[0] = round(-1/0.125)+8 = -8+8 = 0
        // q[16] = 0 + 8 = 8
        assert!(scales[0] > 0.0);
        // byte 0: low nibble = q[0] = 0, high nibble = q[16] = 8
        assert_eq!(packed[0] & 0x0F, 0);
        assert_eq!((packed[0] >> 4) & 0x0F, 8);
    }

    #[test]
    #[should_panic]
    fn test_q4_0_requires_block_aligned() {
        let data = vec![1.0f32; Q4_0_BLOCK_SIZE - 1];
        let _ = quantize_f32_to_q4_0(&data);
    }
}
