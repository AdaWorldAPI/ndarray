//! Jina embedding codec — GGUF → Base17 → Palette → CausalEdge64.
//!
//! Extracts token embeddings from Jina GGUF models, compresses through
//! the HHTL cascade, and provides O(1) similarity lookup.
//!
//! # Compression Chain (measured on Jina v4 3.1B)
//!
//! ```text
//! F16 embedding (2048D × 2B = 4096B per token)
//!   → Base17 (17D × 2B = 34B, 120× compression)
//!     → Palette (1B index + 8.5KB codebook, 4096× total)
//!       → CausalEdge64 S/P/O fields (3 × 1B = 3B per triple)
//! ```
//!
//! # Speed
//!
//! ```text
//! Palette lookup:    0.01μs per token
//! Attention table:   0.01μs per pair (256×256 precomputed)
//! NARS revision:     0.01μs per evidence
//! Total per triple:  0.05μs → 20M observations/second
//! ```

pub mod cache;
pub mod codec;
pub mod causal;
