//! Shared SIMD-accelerated neural network layers.
//!
//! All ops use `crate::simd::F32x16` — the ONLY SIMD interface consumers see.
//! Used by GPT-2, Stable Diffusion CLIP, BERT, and any future transformer model.

use crate::simd::F32x16;

/// Layer normalization with F32x16 SIMD.
///
/// `x` is modified in-place: `x = (x - mean) / sqrt(var + eps) * weight + bias`
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = x.len();
    let chunks = n / 16;

    // Mean (SIMD)
    let mut sum_acc = F32x16::splat(0.0);
    for c in 0..chunks {
        let off = c * 16;
        sum_acc = sum_acc + F32x16::from_slice(&x[off..off + 16]);
    }
    let mut mean = sum_acc.reduce_sum();
    for i in (chunks * 16)..n {
        mean += x[i];
    }
    mean /= n as f32;

    // Variance (SIMD)
    let mean_vec = F32x16::splat(mean);
    let mut var_acc = F32x16::splat(0.0);
    for c in 0..chunks {
        let off = c * 16;
        let diff = F32x16::from_slice(&x[off..off + 16]) - mean_vec;
        var_acc = diff.mul_add(diff, var_acc);
    }
    let mut var = var_acc.reduce_sum();
    for i in (chunks * 16)..n {
        let d = x[i] - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    // Normalize + scale + shift (SIMD)
    let inv_std_vec = F32x16::splat(inv_std);
    for c in 0..chunks {
        let off = c * 16;
        let val = F32x16::from_slice(&x[off..off + 16]);
        let w = F32x16::from_slice(&weight[off..off + 16]);
        let b = F32x16::from_slice(&bias[off..off + 16]);
        let normed = (val - mean_vec) * inv_std_vec;
        let result = normed * w + b;
        result.copy_to_slice(&mut x[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Group normalization with F32x16 SIMD.
///
/// Used by UNet (Stable Diffusion). Splits channels into `num_groups`,
/// normalizes each group independently.
pub fn group_norm(x: &mut [f32], num_groups: usize, weight: &[f32], bias: &[f32]) {
    let total = x.len();
    let group_size = total / num_groups;

    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group = &mut x[start..end];

        // Mean
        let mut mean = 0.0f32;
        for &v in group.iter() {
            mean += v;
        }
        mean /= group_size as f32;

        // Variance
        let mut var = 0.0f32;
        for &v in group.iter() {
            let d = v - mean;
            var += d * d;
        }
        var /= group_size as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        // Normalize + affine
        for i in 0..group_size {
            let idx = start + i;
            group[i] = (group[i] - mean) * inv_std * weight[idx] + bias[idx];
        }
    }
}

/// GELU activation: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// SIMD-accelerated via F32x16 exp for tanh approximation.
pub fn gelu(x: &mut [f32]) {
    let n = x.len();
    let sqrt_2_over_pi = F32x16::splat(0.7978845608);
    let coeff = F32x16::splat(0.044715);
    let half = F32x16::splat(0.5);
    let one = F32x16::splat(1.0);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        let v3 = v * v * v;
        let inner = sqrt_2_over_pi * (v + coeff * v3);
        let two_inner = inner + inner;
        let exp_2x = crate::simd::simd_exp_f32(two_inner);
        let tanh_v = (exp_2x - one) / (exp_2x + one);
        let result = v * half * (one + tanh_v);
        result.copy_to_slice(&mut x[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        let v = x[i];
        let inner = 0.7978845608 * (v + 0.044715 * v * v * v);
        x[i] = v * 0.5 * (1.0 + inner.tanh());
    }
}

/// SiLU (Swish) activation: `x * sigmoid(x)`.
///
/// Used by Stable Diffusion UNet. SIMD-accelerated via F32x16 exp.
pub fn silu(x: &mut [f32]) {
    let n = x.len();
    let one = F32x16::splat(1.0);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        let neg_v = F32x16::splat(0.0) - v;
        let exp_neg = crate::simd::simd_exp_f32(neg_v);
        let sigmoid = one / (one + exp_neg);
        let result = v * sigmoid;
        result.copy_to_slice(&mut x[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        let v = x[i];
        let sig = 1.0 / (1.0 + (-v).exp());
        x[i] = v * sig;
    }
}

/// Numerically stable softmax (in-place).
pub fn softmax(x: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Matrix-vector multiply: `output = input @ weight + bias`.
///
/// Weight must be PRE-TRANSPOSED to `[out_dim, in_dim]` for contiguous SIMD.
/// Use `models::safetensors::transpose_matrix()` at load time.
pub fn matmul_vec(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32], in_dim: usize, out_dim: usize) {
    let chunks = in_dim / 16;
    let remainder = in_dim % 16;

    for o in 0..out_dim {
        let row_offset = o * in_dim;
        let mut acc = F32x16::splat(0.0);
        for c in 0..chunks {
            let off = c * 16;
            let vi = F32x16::from_slice(&input[off..off + 16]);
            let vw = F32x16::from_slice(&weight[row_offset + off..row_offset + off + 16]);
            acc = vi.mul_add(vw, acc);
        }
        let mut dot = acc.reduce_sum();
        let tail_start = chunks * 16;
        for i in 0..remainder {
            dot += input[tail_start + i] * weight[row_offset + tail_start + i];
        }
        output[o] = dot + bias[o];
    }
}

/// SIMD dot product of two f32 slices (same length).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let mut acc = F32x16::splat(0.0);
    for c in 0..chunks {
        let off = c * 16;
        let va = F32x16::from_slice(&a[off..off + 16]);
        let vb = F32x16::from_slice(&b[off..off + 16]);
        acc = va.mul_add(vb, acc);
    }
    let mut sum = acc.reduce_sum();
    for i in (chunks * 16)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_zero_mean() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        layer_norm(&mut x, &w, &b);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_gelu_zero() {
        let mut x = vec![0.0f32; 16];
        gelu(&mut x);
        assert!(x[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_positive() {
        let mut x = vec![2.0f32; 16];
        gelu(&mut x);
        assert!((x[0] - 1.9545).abs() < 0.01);
    }

    #[test]
    fn test_silu_zero() {
        let mut x = vec![0.0f32; 16];
        silu(&mut x);
        assert!(x[0].abs() < 0.01, "SiLU(0) = 0");
    }

    #[test]
    fn test_silu_positive() {
        let mut x = vec![2.0f32; 16];
        silu(&mut x);
        // SiLU(2) = 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
        assert!((x[0] - 1.7616).abs() < 0.01, "SiLU(2) ≈ 1.76, got {}", x[0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_group_norm_two_groups() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0]; // 2 groups of 2
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        group_norm(&mut x, 2, &w, &b);
        // Each group normalized independently
        let g1_mean = (x[0] + x[1]) / 2.0;
        let g2_mean = (x[2] + x[3]) / 2.0;
        assert!(g1_mean.abs() < 0.01);
        assert!(g2_mean.abs() < 0.01);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = dot_product(&a, &b);
        assert!((d - 32.0).abs() < 1e-5); // 4+10+18 = 32
    }

    #[test]
    fn test_dot_product_simd_path() {
        let a: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..48).map(|i| (i * 2) as f32).collect();
        let d = dot_product(&a, &b);
        let expected: f32 = (0..48).map(|i| (i * i * 2) as f32).sum();
        assert!((d - expected).abs() < 1.0);
    }

    #[test]
    fn test_matmul_vec_identity() {
        // 2×2 identity matrix (pre-transposed = still identity)
        let input = vec![3.0, 7.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [out=2, in=2]
        let bias = vec![0.0, 0.0];
        let mut output = vec![0.0; 2];
        matmul_vec(&input, &weight, &bias, &mut output, 2, 2);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }
}
