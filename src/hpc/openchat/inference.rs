//! OpenChat 3.5 / Mistral-7B forward pass + generation loop.
//!
//! Key differences from GPT-2:
//! - Grouped Query Attention (GQA): 32 Q heads share 8 KV heads
//! - RoPE positional encoding (no learned position embeddings)
//! - RMSNorm instead of LayerNorm
//! - SiLU activation in FFN (gated MLP: gate * up, then down)
//! - All ops via `crate::hpc::models::layers` (shared F32x16 SIMD)

use super::weights::*;
use crate::hpc::jina::{causal, runtime};
use crate::hpc::models::layers;
use crate::simd::F32x16;

/// A generated token with its probability.
#[derive(Clone, Debug)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub logprob: f32,
}

/// CausalEdge64 emitted during attention.
#[derive(Clone, Debug)]
pub struct AttentionEdge {
    pub layer: u8,
    pub head: u8,
    pub edge: u64,
}

/// KV cache for one layer (GQA: only 8 KV heads cached, not 32).
#[derive(Clone)]
struct KvCache {
    /// Cached keys: [seq_len, kv_dim] where kv_dim = 8 × 128 = 1024.
    keys: Vec<f32>,
    /// Cached values: [seq_len, kv_dim].
    values: Vec<f32>,
}

/// OpenChat 3.5 inference engine.
pub struct OpenChatEngine {
    weights: OpenChatWeights,
    kv_cache: Vec<KvCache>,
    seq_len: usize,
    token_history: Vec<u32>,
    /// Accumulated causal edges from attention patterns.
    pub causal_edges: Vec<AttentionEdge>,
    /// Whether to emit CausalEdge64 from attention patterns.
    pub emit_causal_edges: bool,
}

impl OpenChatEngine {
    pub fn new(weights: OpenChatWeights) -> Self {
        let n_layers = weights.layers.len();
        let kv_cache = (0..n_layers)
            .map(|_| KvCache {
                keys: Vec::with_capacity(MAX_SEQ_LEN * KV_DIM),
                values: Vec::with_capacity(MAX_SEQ_LEN * KV_DIM),
            })
            .collect();
        Self {
            weights,
            kv_cache,
            seq_len: 0,
            token_history: Vec::with_capacity(MAX_SEQ_LEN),
            causal_edges: Vec::new(),
            emit_causal_edges: false,
        }
    }

    /// Access weights.
    pub fn weights(&self) -> &OpenChatWeights {
        &self.weights
    }

    /// Reset KV cache.
    pub fn reset(&mut self) {
        for kv in &mut self.kv_cache {
            kv.keys.clear();
            kv.values.clear();
        }
        self.seq_len = 0;
        self.token_history.clear();
        self.causal_edges.clear();
    }

    /// Forward pass for one token → logits over vocabulary.
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        let pos = self.seq_len;
        assert!(pos < MAX_SEQ_LEN, "sequence too long ({} >= {})", pos, MAX_SEQ_LEN);
        self.token_history.push(token_id);

        // Token embedding (no position embedding — RoPE handles that)
        let mut hidden = vec![0.0f32; EMBED_DIM];
        let emb_offset = token_id as usize * EMBED_DIM;
        hidden.copy_from_slice(&self.weights.token_embd[emb_offset..emb_offset + EMBED_DIM]);

        // Transformer layers (may be fewer than NUM_LAYERS for testing/distilled models)
        let n_layers = self.weights.layers.len();
        for layer_idx in 0..n_layers {
            hidden = self.transformer_layer(layer_idx, &hidden, pos);
        }

        // Final RMSNorm
        layers::rms_norm(&mut hidden, &self.weights.output_norm, RMS_EPS);

        // Logits: hidden @ output.T (weight already in [vocab, embed] layout)
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        let chunks = EMBED_DIM / 16;
        for v in 0..VOCAB_SIZE {
            let w_off = v * EMBED_DIM;
            let mut acc = F32x16::splat(0.0);
            for c in 0..chunks {
                let off = c * 16;
                let vh = F32x16::from_slice(&hidden[off..off + 16]);
                let vw = F32x16::from_slice(&self.weights.output[w_off + off..w_off + off + 16]);
                acc = vh.mul_add(vw, acc);
            }
            logits[v] = acc.reduce_sum();
        }

        self.seq_len += 1;
        logits
    }

    /// One transformer layer: GQA attention + gated MLP.
    fn transformer_layer(&mut self, layer_idx: usize, input: &[f32], pos: usize) -> Vec<f32> {
        // Clone norm weights before mutable borrow
        let attn_norm = self.weights.layers[layer_idx].attn_norm.clone();
        let ffn_norm = self.weights.layers[layer_idx].ffn_norm.clone();

        // Pre-attention RMSNorm
        let mut normed = input.to_vec();
        layers::rms_norm(&mut normed, &attn_norm, RMS_EPS);

        // GQA attention
        let attn_out = self.gqa_attention(layer_idx, &normed, pos);

        // Residual
        let mut hidden: Vec<f32> = input.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

        // Pre-FFN RMSNorm
        let mut normed2 = hidden.clone();
        layers::rms_norm(&mut normed2, &ffn_norm, RMS_EPS);

        // Gated MLP: SiLU(gate(x)) * up(x) → down
        let ffn_out = self.gated_mlp(layer_idx, &normed2);

        // Residual
        for i in 0..EMBED_DIM {
            hidden[i] += ffn_out[i];
        }

        hidden
    }

    /// Grouped Query Attention: 32 Q heads, 8 KV heads (4:1 ratio).
    ///
    /// Each KV head is shared by 4 Q heads. RoPE applied to Q and K.
    fn gqa_attention(&mut self, layer_idx: usize, input: &[f32], pos: usize) -> Vec<f32> {
        let layer = &self.weights.layers[layer_idx];
        let zero_bias_q = vec![0.0f32; EMBED_DIM];
        let zero_bias_kv = vec![0.0f32; KV_DIM];

        // Q projection: [4096] → [4096] (32 heads × 128D)
        let mut q = vec![0.0f32; EMBED_DIM];
        layers::matmul_vec(input, &layer.attn_q, &zero_bias_q, &mut q, EMBED_DIM, EMBED_DIM);

        // K projection: [4096] → [1024] (8 heads × 128D)
        let mut k = vec![0.0f32; KV_DIM];
        layers::matmul_vec(input, &layer.attn_k, &zero_bias_kv, &mut k, EMBED_DIM, KV_DIM);

        // V projection: [4096] → [1024] (8 heads × 128D)
        let mut v = vec![0.0f32; KV_DIM];
        layers::matmul_vec(input, &layer.attn_v, &zero_bias_kv, &mut v, EMBED_DIM, KV_DIM);

        // Apply RoPE to Q and K (per-head)
        for qh in 0..NUM_Q_HEADS {
            let kv_h = qh / GQA_RATIO;
            let q_off = qh * HEAD_DIM;
            let k_off = kv_h * HEAD_DIM;
            layers::rope_apply(
                &mut q[q_off..q_off + HEAD_DIM],
                &mut k[k_off..k_off + HEAD_DIM],
                HEAD_DIM,
                pos,
                ROPE_THETA,
            );
        }

        // Append K, V to cache
        self.kv_cache[layer_idx].keys.extend_from_slice(&k);
        self.kv_cache[layer_idx].values.extend_from_slice(&v);

        let seq_len = self.seq_len + 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // Per Q-head attention with GQA (4 Q heads share 1 KV head)
        let mut output = vec![0.0f32; EMBED_DIM];
        let emit = self.emit_causal_edges;

        for qh in 0..NUM_Q_HEADS {
            let kv_h = qh / GQA_RATIO;
            let q_off = qh * HEAD_DIM;

            // Scores: Q[qh] · K[kv_h]^T for all cached positions
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let k_off = t * KV_DIM + kv_h * HEAD_DIM;
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += q[q_off + d] * self.kv_cache[layer_idx].keys[k_off + d];
                }
                scores[t] = dot * scale;
            }

            layers::softmax(&mut scores);

            // CausalEdge64 emission
            if emit {
                let current_token = *self.token_history.last().unwrap_or(&0);
                for t in 0..seq_len {
                    if scores[t] > 0.05 && t < self.token_history.len() {
                        let key_token = self.token_history[t];
                        // Use GPT2 palette for now — OpenChat palette can be built later
                        // Token IDs may differ but the edge structure is the same
                        let edge = causal::pack_edge(
                            (current_token % 256) as u8,
                            (qh % 256) as u8,
                            (key_token % 256) as u8,
                            scores[t],
                            0.3,
                            0b111, // full SPO Pearl mask
                            self.seq_len as u16,
                        );
                        self.causal_edges.push(AttentionEdge {
                            layer: layer_idx as u8,
                            head: qh as u8,
                            edge,
                        });
                    }
                }
            }

            // Weighted sum of V[kv_h]
            for t in 0..seq_len {
                let v_off = t * KV_DIM + kv_h * HEAD_DIM;
                let w = scores[t];
                for d in 0..HEAD_DIM {
                    output[q_off + d] += w * self.kv_cache[layer_idx].values[v_off + d];
                }
            }
        }

        // Output projection
        let zero_bias = vec![0.0f32; EMBED_DIM];
        let mut projected = vec![0.0f32; EMBED_DIM];
        layers::matmul_vec(&output, &self.weights.layers[layer_idx].attn_output, &zero_bias, &mut projected, EMBED_DIM, EMBED_DIM);

        projected
    }

    /// Gated MLP: gate(x) = SiLU(W_gate @ x), up(x) = W_up @ x.
    /// output = W_down @ (gate(x) * up(x))
    fn gated_mlp(&self, layer_idx: usize, input: &[f32]) -> Vec<f32> {
        let layer = &self.weights.layers[layer_idx];
        let zero_bias_mlp = vec![0.0f32; MLP_DIM];
        let zero_bias_out = vec![0.0f32; EMBED_DIM];

        // Gate projection + SiLU
        let mut gate = vec![0.0f32; MLP_DIM];
        layers::matmul_vec(input, &layer.ffn_gate, &zero_bias_mlp, &mut gate, EMBED_DIM, MLP_DIM);
        layers::silu(&mut gate);

        // Up projection
        let mut up = vec![0.0f32; MLP_DIM];
        layers::matmul_vec(input, &layer.ffn_up, &zero_bias_mlp, &mut up, EMBED_DIM, MLP_DIM);

        // Element-wise gate * up
        let chunks = MLP_DIM / 16;
        for c in 0..chunks {
            let off = c * 16;
            let vg = F32x16::from_slice(&gate[off..off + 16]);
            let vu = F32x16::from_slice(&up[off..off + 16]);
            let result = vg * vu;
            result.copy_to_slice(&mut gate[off..off + 16]);
        }
        for i in (chunks * 16)..MLP_DIM {
            gate[i] *= up[i];
        }

        // Down projection
        let mut output = vec![0.0f32; EMBED_DIM];
        layers::matmul_vec(&gate, &layer.ffn_down, &zero_bias_out, &mut output, MLP_DIM, EMBED_DIM);

        output
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<GeneratedToken> {
        self.reset();
        let mut generated = Vec::new();

        // Process prompt (fill KV cache)
        let mut last_logits = vec![0.0f32; VOCAB_SIZE];
        for &token in prompt_tokens {
            last_logits = self.forward(token);
        }

        // Generate
        for _ in 0..max_new_tokens {
            if temperature != 1.0 && temperature > 0.0 {
                let inv_temp = 1.0 / temperature;
                for l in &mut last_logits {
                    *l *= inv_temp;
                }
            }

            // Greedy argmax
            let mut best_id = 0u32;
            let mut best_logit = f32::NEG_INFINITY;
            for (i, &l) in last_logits.iter().enumerate() {
                if l > best_logit {
                    best_logit = l;
                    best_id = i as u32;
                }
            }

            // EOS
            if best_id == chat_template::EOS_TOKEN_ID {
                break;
            }

            generated.push(GeneratedToken {
                token_id: best_id,
                logprob: best_logit,
            });

            last_logits = self.forward(best_id);
        }

        generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_weights() -> OpenChatWeights {
        let layer = MistralLayerWeights {
            attn_norm: vec![1.0; EMBED_DIM],
            attn_q: vec![0.0; EMBED_DIM * EMBED_DIM],
            attn_k: vec![0.0; KV_DIM * EMBED_DIM],
            attn_v: vec![0.0; KV_DIM * EMBED_DIM],
            attn_output: vec![0.0; EMBED_DIM * EMBED_DIM],
            ffn_norm: vec![1.0; EMBED_DIM],
            ffn_gate: vec![0.0; MLP_DIM * EMBED_DIM],
            ffn_up: vec![0.0; MLP_DIM * EMBED_DIM],
            ffn_down: vec![0.0; EMBED_DIM * MLP_DIM],
        };
        OpenChatWeights {
            token_embd: vec![0.01; VOCAB_SIZE * EMBED_DIM],
            layers: vec![layer; 1], // 1 layer for testing (32 would OOM in tests)
            output_norm: vec![1.0; EMBED_DIM],
            output: vec![0.01; VOCAB_SIZE * EMBED_DIM],
        }
    }

    #[test]
    fn test_engine_creation() {
        let w = dummy_weights();
        let engine = OpenChatEngine::new(w);
        assert_eq!(engine.seq_len, 0);
        assert!(!engine.emit_causal_edges);
    }

    #[test]
    fn test_engine_reset() {
        let w = dummy_weights();
        let mut engine = OpenChatEngine::new(w);
        engine.seq_len = 5;
        engine.token_history.push(42);
        engine.reset();
        assert_eq!(engine.seq_len, 0);
        assert!(engine.token_history.is_empty());
    }

    #[test]
    fn test_gqa_ratio() {
        assert_eq!(GQA_RATIO, 4, "32Q / 8KV = 4:1 sharing");
    }

    #[test]
    fn test_forward_produces_logits() {
        let w = dummy_weights();
        let mut engine = OpenChatEngine::new(w);
        let logits = engine.forward(0);
        assert_eq!(logits.len(), VOCAB_SIZE);
        // With near-zero weights, logits should be near-zero
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit.is_finite(), "logits should be finite, got {}", max_logit);
    }

    #[test]
    fn test_forward_increments_seq_len() {
        let w = dummy_weights();
        let mut engine = OpenChatEngine::new(w);
        engine.forward(0);
        assert_eq!(engine.seq_len, 1);
        engine.forward(1);
        assert_eq!(engine.seq_len, 2);
    }

    #[test]
    fn test_kv_cache_grows() {
        let w = dummy_weights();
        let mut engine = OpenChatEngine::new(w);
        engine.forward(0);
        // KV cache should have 1 entry per layer
        assert_eq!(engine.kv_cache[0].keys.len(), KV_DIM);
        assert_eq!(engine.kv_cache[0].values.len(), KV_DIM);
        engine.forward(1);
        assert_eq!(engine.kv_cache[0].keys.len(), 2 * KV_DIM);
    }
}
