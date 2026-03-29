//! Shared model primitives — used by GPT-2, Stable Diffusion, BERT, Jina, OpenChat.
//!
//! Extracts common patterns so each model crate is thin:
//! - `safetensors`: generic file loader (header parse + tensor extract)
//! - `layers`: SIMD-accelerated ops (LayerNorm, GELU, softmax, matmul)
//! - `api_types`: OpenAI-compatible request/response types (1:1 field match)
//! - `router`: unified dispatch to all models by endpoint + model ID

pub mod safetensors;
pub mod layers;
pub mod api_types;
pub mod router;
