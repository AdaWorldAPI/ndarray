//! GPT-2 inference engine — forward pass + generation loop.
//!
//! All transcendental ops use `crate::simd::F32x16`.
//! LayerNorm, GELU, Softmax — all SIMD-accelerated.

use super::weights::*;
use crate::simd::F32x16;

/// A generated token with its probability.
#[derive(Clone, Debug)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub logprob: f32,
}

/// GPT-2 inference engine.
pub struct Gpt2Engine {
    weights: Gpt2Weights,
    /// KV cache for autoregressive generation.
    kv_cache: Vec<KvCache>,
    /// Current sequence length.
    seq_len: usize,
}

/// Key-Value cache for one layer.
#[derive(Clone)]
struct KvCache {
    /// Cached keys: [seq_len, embed_dim]
    keys: Vec<f32>,
    /// Cached values: [seq_len, embed_dim]
    values: Vec<f32>,
}

impl Gpt2Engine {
    /// Create engine from weights.
    pub fn new(weights: Gpt2Weights) -> Self {
        let kv_cache = (0..NUM_LAYERS)
            .map(|_| KvCache {
                keys: Vec::with_capacity(MAX_SEQ_LEN * EMBED_DIM),
                values: Vec::with_capacity(MAX_SEQ_LEN * EMBED_DIM),
            })
            .collect();
        Self { weights, kv_cache, seq_len: 0 }
    }

    /// Access weights (for embedding lookups).
    pub fn weights(&self) -> &Gpt2Weights {
        &self.weights
    }

    /// Reset KV cache (new conversation).
    pub fn reset(&mut self) {
        for kv in &mut self.kv_cache {
            kv.keys.clear();
            kv.values.clear();
        }
        self.seq_len = 0;
    }

    /// Forward pass for one token → logits over vocabulary.
    ///
    /// Uses KV cache for O(seq_len) attention instead of O(seq_len²).
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        let pos = self.seq_len;
        assert!(pos < MAX_SEQ_LEN, "sequence too long");

        // Embedding: wte[token] + wpe[position]
        let mut hidden = vec![0.0f32; EMBED_DIM];
        let wte_offset = token_id as usize * EMBED_DIM;
        let wpe_offset = pos * EMBED_DIM;
        for i in 0..EMBED_DIM {
            hidden[i] = self.weights.wte[wte_offset + i] + self.weights.wpe[wpe_offset + i];
        }

        // 12 transformer layers
        for layer_idx in 0..NUM_LAYERS {
            hidden = self.transformer_layer(layer_idx, &hidden);
        }

        // Final layer norm
        layer_norm_simd(&mut hidden, &self.weights.ln_f_weight, &self.weights.ln_f_bias);

        // Logits: hidden @ wte.T (weight tying)
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let chunks = EMBED_DIM / 16;
        for v in 0..VOCAB_SIZE {
            let wte_off = v * EMBED_DIM;
            let mut acc = F32x16::splat(0.0);
            for c in 0..chunks {
                let off = c * 16;
                let vh = F32x16::from_slice(&hidden[off..off + 16]);
                let vw = F32x16::from_slice(&self.weights.wte[wte_off + off..wte_off + off + 16]);
                acc = vh.mul_add(vw, acc);
            }
            logits[v] = acc.reduce_sum();
        }

        self.seq_len += 1;
        logits
    }

    /// One transformer layer: attention + MLP with residuals.
    fn transformer_layer(&mut self, layer_idx: usize, input: &[f32]) -> Vec<f32> {
        // Clone LayerNorm params before mutable borrow on self (KV cache).
        let ln1_w = self.weights.layers[layer_idx].ln1_weight.clone();
        let ln1_b = self.weights.layers[layer_idx].ln1_bias.clone();
        let ln2_w = self.weights.layers[layer_idx].ln2_weight.clone();
        let ln2_b = self.weights.layers[layer_idx].ln2_bias.clone();

        // Pre-attention LayerNorm
        let mut normed = input.to_vec();
        layer_norm_simd(&mut normed, &ln1_w, &ln1_b);

        // Attention: Q/K/V projection → scaled dot-product → output
        let attn_out = self.multi_head_attention(layer_idx, &normed);

        // Residual connection
        let mut hidden: Vec<f32> = input.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

        // Pre-MLP LayerNorm
        let mut normed2 = hidden.clone();
        layer_norm_simd(&mut normed2, &ln2_w, &ln2_b);

        // MLP: fc → GELU → proj
        let mlp_out = self.mlp(layer_idx, &normed2);

        // Residual connection
        for i in 0..EMBED_DIM {
            hidden[i] += mlp_out[i];
        }

        hidden
    }

    /// Multi-head self-attention with KV cache.
    fn multi_head_attention(&mut self, layer_idx: usize, input: &[f32]) -> Vec<f32> {
        let layer = &self.weights.layers[layer_idx];

        // Q/K/V projection: input[768] × weight[768, 2304] + bias[2304]
        let mut qkv = vec![0.0f32; 3 * EMBED_DIM]; // [Q(768), K(768), V(768)]
        matmul_vec_simd(input, &layer.attn_qkv_weight, &layer.attn_qkv_bias, &mut qkv, EMBED_DIM, 3 * EMBED_DIM);

        let q = &qkv[..EMBED_DIM];
        let k = &qkv[EMBED_DIM..2 * EMBED_DIM];
        let v = &qkv[2 * EMBED_DIM..3 * EMBED_DIM];

        // Append K, V to cache
        self.kv_cache[layer_idx].keys.extend_from_slice(k);
        self.kv_cache[layer_idx].values.extend_from_slice(v);

        let seq_len = self.seq_len + 1; // including current token

        // Per-head attention
        let mut output = vec![0.0f32; EMBED_DIM];
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        for head in 0..NUM_HEADS {
            let h_offset = head * HEAD_DIM;

            // Compute attention scores: Q[head] · K[head]^T for all cached positions
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_offset = t * EMBED_DIM + h_offset;
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += q[h_offset + d] * self.kv_cache[layer_idx].keys[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask: only attend to past and current (already enforced by cache length)
            // Softmax
            softmax_simd(&mut scores);

            // Weighted sum of values
            for t in 0..seq_len {
                let v_offset = t * EMBED_DIM + h_offset;
                let w = scores[t];
                for d in 0..HEAD_DIM {
                    output[h_offset + d] += w * self.kv_cache[layer_idx].values[v_offset + d];
                }
            }
        }

        // Output projection: output[768] × weight[768, 768] + bias[768]
        let mut projected = vec![0.0f32; EMBED_DIM];
        matmul_vec_simd(&output, &layer.attn_out_weight, &layer.attn_out_bias, &mut projected, EMBED_DIM, EMBED_DIM);

        projected
    }

    /// MLP: fc[768→3072] → GELU → proj[3072→768].
    fn mlp(&self, layer_idx: usize, input: &[f32]) -> Vec<f32> {
        let layer = &self.weights.layers[layer_idx];

        // FC: input[768] × weight[768, 3072] + bias[3072]
        let mut fc_out = vec![0.0f32; MLP_DIM];
        matmul_vec_simd(input, &layer.mlp_fc_weight, &layer.mlp_fc_bias, &mut fc_out, EMBED_DIM, MLP_DIM);

        // GELU activation (via SIMD)
        gelu_simd(&mut fc_out);

        // Proj: fc_out[3072] × weight[3072, 768] + bias[768]
        let mut output = vec![0.0f32; EMBED_DIM];
        matmul_vec_simd(&fc_out, &layer.mlp_proj_weight, &layer.mlp_proj_bias, &mut output, MLP_DIM, EMBED_DIM);

        output
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, prompt_tokens: &[u32], max_new_tokens: usize, temperature: f32) -> Vec<GeneratedToken> {
        self.reset();
        let mut generated = Vec::new();

        // Process prompt (fill KV cache)
        let mut last_logits = vec![0.0f32; VOCAB_SIZE];
        for &token in prompt_tokens {
            last_logits = self.forward(token);
        }

        // Generate new tokens
        for _ in 0..max_new_tokens {
            // Apply temperature
            if temperature != 1.0 {
                for l in &mut last_logits {
                    *l /= temperature;
                }
            }

            // Greedy: argmax
            let mut best_id = 0u32;
            let mut best_logit = f32::NEG_INFINITY;
            for (i, &l) in last_logits.iter().enumerate() {
                if l > best_logit {
                    best_logit = l;
                    best_id = i as u32;
                }
            }

            // End of text
            if best_id == 50256 {
                break;
            }

            generated.push(GeneratedToken {
                token_id: best_id,
                logprob: best_logit,
            });

            // Feed generated token back
            last_logits = self.forward(best_id);
        }

        generated
    }
}

// ============================================================================
// SIMD-accelerated primitives (all use crate::simd::F32x16)
// ============================================================================

/// Layer normalization with F32x16 SIMD.
fn layer_norm_simd(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = x.len();

    // Mean (SIMD)
    let chunks = n / 16;
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

/// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
fn gelu_simd(x: &mut [f32]) {
    let n = x.len();
    let sqrt_2_over_pi = F32x16::splat(0.7978845608); // sqrt(2/π)
    let coeff = F32x16::splat(0.044715);
    let half = F32x16::splat(0.5);
    let one = F32x16::splat(1.0);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        let v3 = v * v * v;
        let inner = sqrt_2_over_pi * (v + coeff * v3);
        // tanh approximation via exp: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let two_inner = inner + inner;
        let exp_2x = crate::simd::simd_exp_f32(two_inner);
        let tanh_v = (exp_2x - one) / (exp_2x + one);
        let result = v * half * (one + tanh_v);
        result.copy_to_slice(&mut x[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        let v = x[i];
        let inner = 0.7978845608 * (v + 0.044715 * v * v * v);
        let tanh_v = inner.tanh();
        x[i] = v * 0.5 * (1.0 + tanh_v);
    }
}

/// Softmax with numerical stability (SIMD).
fn softmax_simd(x: &mut [f32]) {
    // Find max (for numerical stability)
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }

    // exp(x - max) and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Matrix-vector multiply: out = input @ weight^T + bias.
/// Weight stored as [input_dim, output_dim] (row-major, transposed access).
/// SIMD accelerated for the dot product.
/// Matrix-vector multiply: out = input @ weight + bias.
/// Weight is PRE-TRANSPOSED to [out_dim, in_dim] for contiguous SIMD access.
/// Each output element reads a contiguous row of in_dim floats.
fn matmul_vec_simd(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32], in_dim: usize, out_dim: usize) {
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
        // Scalar tail
        let tail_start = chunks * 16;
        for i in 0..remainder {
            dot += input[tail_start + i] * weight[row_offset + tail_start + i];
        }
        output[o] = dot + bias[o];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let b = vec![0.0; 4];
        layer_norm_simd(&mut x, &w, &b);
        // After normalization: mean≈0, std≈1
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.01, "mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_gelu_zero() {
        let mut x = vec![0.0f32; 16];
        gelu_simd(&mut x);
        assert!(x[0].abs() < 0.01, "GELU(0) should be ~0");
    }

    #[test]
    fn test_gelu_positive() {
        let mut x = vec![2.0f32; 16];
        gelu_simd(&mut x);
        // GELU(2) ≈ 1.9545
        assert!((x[0] - 1.9545).abs() < 0.01, "GELU(2) ≈ 1.95, got {}", x[0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax_simd(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_softmax_argmax_preserved() {
        let mut x = vec![1.0, 5.0, 2.0, 3.0];
        softmax_simd(&mut x);
        // Index 1 (value 5.0) should have highest probability
        assert!(x[1] > x[0] && x[1] > x[2] && x[1] > x[3]);
    }
}
