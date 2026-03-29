//! Shared model primitives — used by GPT-2, Stable Diffusion, BERT, Jina.
//!
//! Extracts common patterns so each model crate is thin:
//! - `safetensors`: generic file loader (header parse + tensor extract)
//! - `layers`: SIMD-accelerated ops (LayerNorm, GELU, softmax, matmul)
//! - `api_types`: OpenAI-compatible request/response envelope

pub mod safetensors;
pub mod layers;
pub mod api_types;
