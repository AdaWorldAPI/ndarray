//! GPT-2 API — wraps the inference engine with OpenAI-compatible types.
//!
//! Endpoints:
//! - `/v1/completions` — text completion
//! - `/v1/embeddings` — token embeddings via wte
//! - `/v1/models` — model info

use crate::hpc::models::api_types::*;
use super::inference::{GeneratedToken, Gpt2Engine};
use super::weights::*;

/// Stateless API wrapper around Gpt2Engine.
pub struct Gpt2Api {
    engine: Gpt2Engine,
    request_counter: u64,
}

impl Gpt2Api {
    pub fn new(weights: Gpt2Weights) -> Self {
        Self { engine: Gpt2Engine::new(weights), request_counter: 0 }
    }

    /// `/v1/completions`
    pub fn complete(&mut self, req: &CompletionRequest) -> CompletionResponse {
        self.request_counter += 1;
        let tokens = req.prompt_tokens.as_deref().unwrap_or(&[]);
        let max = req.max_tokens.unwrap_or(128);
        let temp = req.temperature.unwrap_or(1.0);

        let generated = self.engine.generate(tokens, max, temp);

        let finish_reason = if generated.len() < max {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let text = generated.iter().map(|t| format!("[{}]", t.token_id)).collect::<String>();
        let logprobs: Vec<LogprobInfo> = generated.iter().map(|t| LogprobInfo {
            token: format!("{}", t.token_id),
            token_id: t.token_id,
            logprob: t.logprob,
            bytes: None,
            top_logprobs: Vec::new(),
        }).collect();

        let use_logprobs = req.logprobs.is_some();

        CompletionResponse::new(
            format!("cmpl-{}", self.request_counter),
            "gpt2".into(),
            vec![CompletionChoice {
                index: 0,
                text,
                logprobs: if use_logprobs { Some(logprobs) } else { None },
                finish_reason: Some(finish_reason),
            }],
            Usage {
                prompt_tokens: tokens.len(),
                completion_tokens: generated.len(),
                total_tokens: tokens.len() + generated.len(),
            },
            0,
        )
    }

    /// `/v1/embeddings`
    pub fn embed(&self, req: &EmbeddingRequest) -> EmbeddingResponse {
        let token_ids: Vec<u32> = match &req.input {
            EmbeddingInput::TokenIds(ids) => ids.clone(),
            _ => req.input_tokens.clone().unwrap_or_default(),
        };

        let data: Vec<EmbeddingData> = token_ids.iter().enumerate().map(|(idx, &tid)| {
            let offset = tid as usize * EMBED_DIM;
            let mut emb = self.engine.weights().wte[offset..offset + EMBED_DIM].to_vec();
            if let Some(dim) = req.dimensions {
                emb.truncate(dim);
            }
            EmbeddingData::new(idx, emb)
        }).collect();

        EmbeddingResponse::new(
            "gpt2".into(),
            data,
            Usage { prompt_tokens: token_ids.len(), completion_tokens: 0, total_tokens: token_ids.len() },
        )
    }

    /// `/v1/models/{id}`
    pub fn model_info() -> Model {
        Model::new("gpt2", "adaworldapi", 0)
    }

    pub fn engine_mut(&mut self) -> &mut Gpt2Engine {
        &mut self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let m = Gpt2Api::model_info();
        assert_eq!(m.id, "gpt2");
        assert_eq!(m.object, "model");
    }

    #[test]
    fn test_completion_defaults() {
        let req = CompletionRequest::default();
        assert_eq!(req.model, "gpt2");
        assert_eq!(req.max_tokens, Some(128));
    }

    #[test]
    fn test_completion_response_object() {
        let resp = CompletionResponse::new("x".into(), "gpt2".into(), vec![], Usage::default(), 0);
        assert_eq!(resp.object, "text_completion");
    }
}
