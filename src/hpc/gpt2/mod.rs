//! GPT-2 inference engine — autoregressive text generation on CPU.
//!
//! Full GPT-2 (124M) running through:
//! - `crate::simd::F32x16` for all transcendental ops
//! - Base17 palette for O(1) embedding lookup
//! - Optional AttentionTable for O(1) attention (when compiled)
//! - CausalEdge64 for causal reasoning on generated tokens
//!
//! # Architecture
//!
//! ```text
//! Input text → BPE tokenize → token IDs
//!   → wte[token_id] (embedding lookup) + wpe[position] (positional)
//!   → 12 transformer layers:
//!       LayerNorm → MultiHeadAttention → Residual
//!       LayerNorm → MLP (GELU) → Residual
//!   → LayerNorm → logits → argmax/sample → next token
//!   → repeat until <|endoftext|> or max_tokens
//! ```
//!
//! # Speed Target
//!
//! GPT-2 small: 768D, 12 layers, 12 heads, 50K vocab.
//! Full matmul path: ~50ms per token on single CPU core.
//! With AttentionTable: ~5ms per token (10× faster).
//! With SIMD exp/sigmoid: ~30% faster transcendentals.

pub mod weights;
pub mod inference;
pub mod api;
