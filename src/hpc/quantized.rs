//! Quantized GEMM: BF16 and Int8 matrix multiplication.
//!
//! Provides BF16 (bfloat16) type with conversions, BF16 GEMM with f32 accumulation,
//! and int8 quantized GEMM with various dequantization modes.

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
    /// Convert f32 to BF16 (truncation).
    pub fn from_f32(v: f32) -> Self {
        BF16((v.to_bits() >> 16) as u16)
    }

    /// Convert f32 to BF16 with round-to-nearest-even.
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
    fn test_i4_roundtrip() {
        let data = vec![1.0f32, -1.0, 3.0, -3.0, 7.0, -7.0];
        let (packed, params) = quantize_f32_to_i4(&data);
        let recovered = dequantize_i4_to_f32(&packed, &params, data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.5, "i4 roundtrip: {} vs {}", orig, rec);
        }
    }
}
