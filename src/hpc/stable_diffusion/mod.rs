//! Stable Diffusion inference — text-to-image on CPU.
//!
//! Architecture:
//! ```text
//! Text prompt → CLIP tokenize → CLIP encoder (transformer, shared layers with GPT-2)
//!   → text embeddings [77, 768]
//!   → UNet denoiser (cross-attention + ResBlocks + GroupNorm)
//!     × N diffusion steps (DDPM/DDIM scheduler)
//!   → VAE decoder → RGB pixels [512, 512, 3]
//! ```
//!
//! # Shared with GPT-2 (via `models::layers`)
//!
//! - LayerNorm, GELU, softmax, matmul — identical F32x16 SIMD ops
//! - Safetensors loader — same format
//! - CausalEdge64 — cross-attention patterns → causal edges
//! - AttentionTable — palette-based O(1) approximate attention
//!
//! # SD-specific
//!
//! - GroupNorm (via `models::layers::group_norm`)
//! - SiLU activation (via `models::layers::silu`)
//! - Conv2D (new, not in GPT-2)
//! - Noise scheduler (DDPM/DDIM)
//! - VAE encoder/decoder

pub mod clip;
pub mod unet;
pub mod vae;
pub mod scheduler;
pub mod weights;
pub mod api;
