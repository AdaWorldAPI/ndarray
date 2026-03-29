//! OpenAI-compatible API types for GPT-2 inference.
//!
//! Provides request/response structs matching the OpenAI API surface:
//! - `/v1/completions` — text completion
//! - `/v1/embeddings` — token embeddings via wte
//! - `/v1/models` — model listing
//!
//! These types are transport-agnostic — they serialize/deserialize
//! but don't depend on any HTTP framework.

use super::inference::{GeneratedToken, Gpt2Engine};
use super::weights::*;

// ============================================================================
// /v1/completions
// ============================================================================

/// Request body for /v1/completions.
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    /// Model name (ignored — we only have gpt2).
    pub model: String,
    /// Input text prompt (will be tokenized externally).
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (1.0 = greedy effective).
    pub temperature: f32,
    /// Stop token ID (default: 50256 = <|endoftext|>).
    pub stop_token: Option<u32>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: "gpt2".into(),
            prompt_tokens: Vec::new(),
            max_tokens: 128,
            temperature: 1.0,
            stop_token: Some(50256),
        }
    }
}

/// Single completion choice.
#[derive(Clone, Debug)]
pub struct CompletionChoice {
    pub index: usize,
    pub tokens: Vec<GeneratedToken>,
    pub finish_reason: FinishReason,
}

/// Why generation stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Response body for /v1/completions.
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// Token usage statistics.
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// /v1/embeddings
// ============================================================================

/// Request body for /v1/embeddings.
#[derive(Clone, Debug)]
pub struct EmbeddingRequest {
    pub model: String,
    /// Token IDs to embed (one embedding per token).
    pub input_tokens: Vec<u32>,
}

/// Single embedding result.
#[derive(Clone, Debug)]
pub struct EmbeddingData {
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// Response body for /v1/embeddings.
#[derive(Clone, Debug)]
pub struct EmbeddingResponse {
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: Usage,
}

// ============================================================================
// /v1/models
// ============================================================================

/// Model info for /v1/models.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub owned_by: String,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
}

impl ModelInfo {
    /// GPT-2 small (124M) model info.
    pub fn gpt2_small() -> Self {
        Self {
            id: "gpt2".into(),
            owned_by: "adaworldapi".into(),
            vocab_size: VOCAB_SIZE,
            embed_dim: EMBED_DIM,
            num_layers: NUM_LAYERS,
            num_heads: NUM_HEADS,
            max_seq_len: MAX_SEQ_LEN,
        }
    }
}

// ============================================================================
// Engine wrapper — stateless API over stateful engine
// ============================================================================

/// Stateless API wrapper around Gpt2Engine.
/// Handles request→response conversion.
pub struct Gpt2Api {
    engine: Gpt2Engine,
    request_counter: u64,
}

impl Gpt2Api {
    /// Create from pre-loaded weights.
    pub fn new(weights: Gpt2Weights) -> Self {
        Self {
            engine: Gpt2Engine::new(weights),
            request_counter: 0,
        }
    }

    /// /v1/completions handler.
    pub fn complete(&mut self, req: &CompletionRequest) -> CompletionResponse {
        self.request_counter += 1;

        let generated = self.engine.generate(
            &req.prompt_tokens,
            req.max_tokens,
            req.temperature,
        );

        let finish_reason = if generated.len() < req.max_tokens {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let completion_tokens = generated.len();
        let prompt_tokens = req.prompt_tokens.len();

        CompletionResponse {
            id: format!("cmpl-{}", self.request_counter),
            model: "gpt2".into(),
            choices: vec![CompletionChoice {
                index: 0,
                tokens: generated,
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }

    /// /v1/embeddings handler — returns wte embeddings for token IDs.
    pub fn embed(&self, req: &EmbeddingRequest) -> EmbeddingResponse {
        let mut data = Vec::with_capacity(req.input_tokens.len());

        for (idx, &token_id) in req.input_tokens.iter().enumerate() {
            let offset = token_id as usize * EMBED_DIM;
            let embedding = self.engine.weights().wte[offset..offset + EMBED_DIM].to_vec();
            data.push(EmbeddingData {
                index: idx,
                embedding,
            });
        }

        EmbeddingResponse {
            model: "gpt2".into(),
            data,
            usage: Usage {
                prompt_tokens: req.input_tokens.len(),
                completion_tokens: 0,
                total_tokens: req.input_tokens.len(),
            },
        }
    }

    /// /v1/models handler.
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo::gpt2_small()
    }

    /// Access the underlying engine (for advanced usage).
    pub fn engine_mut(&mut self) -> &mut Gpt2Engine {
        &mut self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let info = ModelInfo::gpt2_small();
        assert_eq!(info.vocab_size, 50257);
        assert_eq!(info.embed_dim, 768);
        assert_eq!(info.num_layers, 12);
        assert_eq!(info.num_heads, 12);
        assert_eq!(info.max_seq_len, 1024);
    }

    #[test]
    fn test_completion_request_default() {
        let req = CompletionRequest::default();
        assert_eq!(req.max_tokens, 128);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.stop_token, Some(50256));
    }

    #[test]
    fn test_finish_reason_variants() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::Length);
    }
}
