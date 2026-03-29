//! OpenChat 3.5 inference engine — Mistral-7B architecture on CPU.
//!
//! OpenChat 3.5 is a Mistral-7B fine-tune with ChatGPT-3.5-level performance.
//! Same architecture as Mistral: GQA + RoPE + RMSNorm + SiLU.
//!
//! # Architecture differences from GPT-2
//!
//! ```text
//! GPT-2:     LayerNorm → MHA(12 heads) → GELU → 768D
//! OpenChat:  RMSNorm → GQA(32Q/8KV) → SiLU → 4096D
//! ```
//!
//! | Feature         | GPT-2           | OpenChat/Mistral-7B |
//! |-----------------|-----------------|---------------------|
//! | Params          | 124M            | 7B                  |
//! | Embed dim       | 768             | 4096                |
//! | Layers          | 12              | 32                  |
//! | Q heads         | 12 (MHA)        | 32 (GQA)            |
//! | KV heads        | 12              | 8 (4:1 ratio)       |
//! | Activation      | GELU            | SiLU                |
//! | Positional      | Learned         | RoPE (θ=10000)      |
//! | Norm            | Pre-LayerNorm   | RMSNorm             |
//! | Vocab           | 50,257 (BPE)    | 32,000 (SPM)        |
//! | Weight format   | Safetensors     | GGUF (Q4_K_M)       |
//!
//! # Integration
//!
//! All ops via `crate::hpc::models::layers` (shared F32x16 SIMD):
//! - `rms_norm()` — RMSNorm
//! - `rope_apply()` — Rotary Positional Embedding
//! - `silu()` — SiLU/Swish activation
//! - `softmax()`, `matmul_vec()`, `dot_product()` — standard
//!
//! Weight loading via `crate::hpc::gguf` (Q4_K_M dequantization).
//! Codec via `crate::hpc::jina::runtime` (HHTL/CausalEdge64 when available).

pub mod weights;
pub mod inference;
pub mod api;
