//! Unified model router — single API surface dispatching to all models.
//!
//! Matches OpenAI endpoint semantics:
//! - `complete()` → `/v1/completions` (GPT-2)
//! - `chat_complete()` → `/v1/chat/completions` (OpenChat 3.5, or GPT-2 via adapter)
//! - `embed()` → `/v1/embeddings` (GPT-2 wte, or any model with embeddings)
//! - `generate_image()` → `/v1/images/generations` (Stable Diffusion)
//! - `list_models()` → `/v1/models`
//! - `get_model()` → `/v1/models/{id}`
//!
//! The router owns all loaded model engines. Models are registered at startup.
//! Any consumer (Axum, Actix, gRPC, CLI) can call these methods directly.

use super::api_types::*;
use crate::hpc::gpt2;
use crate::hpc::openchat;

/// Which model backend to route to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelBackend {
    Gpt2,
    OpenChat,
    StableDiffusion,
    Jina,
    Bert,
}

/// Unified model router.
///
/// Holds optional engines for each model. Only loaded models respond.
/// Thread-safe: wrap in `Arc<Mutex<ModelRouter>>` for concurrent access.
pub struct ModelRouter {
    gpt2: Option<gpt2::api::Gpt2Api>,
    openchat: Option<openchat::api::OpenChatApi>,
    // SD and embedding models added when loaded
    request_counter: u64,
}

impl ModelRouter {
    /// Create an empty router (no models loaded).
    pub fn new() -> Self {
        Self { gpt2: None, openchat: None, request_counter: 0 }
    }

    // ── Model registration ─────────────────────────────────────────────

    /// Load GPT-2 weights and register the engine.
    pub fn register_gpt2(&mut self, weights: gpt2::weights::Gpt2Weights) {
        self.gpt2 = Some(gpt2::api::Gpt2Api::new(weights));
    }

    /// Load OpenChat weights and register the engine.
    pub fn register_openchat(&mut self, weights: openchat::weights::OpenChatWeights) {
        self.openchat = Some(openchat::api::OpenChatApi::new(weights));
    }

    /// Check which models are loaded.
    pub fn loaded_models(&self) -> Vec<&'static str> {
        let mut models = Vec::new();
        if self.gpt2.is_some() { models.push("gpt2"); }
        if self.openchat.is_some() { models.push("openchat_3.5"); }
        models
    }

    // ── /v1/models ─────────────────────────────────────────────────────

    /// `GET /v1/models` — list all available models.
    pub fn list_models(&self) -> ModelList {
        let mut data = Vec::new();
        if self.gpt2.is_some() {
            data.push(gpt2::api::Gpt2Api::model_info());
        }
        if self.openchat.is_some() {
            data.push(openchat::api::OpenChatApi::model_info());
        }
        // SD always advertised (scaffold)
        data.push(Model::new("stable-diffusion-v1-5", "stabilityai", 0));
        // Embedding models
        data.push(Model::new("text-embedding-jina-v4", "jinaai", 0));
        data.push(Model::new("text-embedding-bert-base", "google", 0));
        ModelList::new(data)
    }

    /// `GET /v1/models/{id}` — get a specific model.
    pub fn get_model(&self, id: &str) -> Result<Model, ApiError> {
        match id {
            "gpt2" if self.gpt2.is_some() => Ok(gpt2::api::Gpt2Api::model_info()),
            "openchat_3.5" if self.openchat.is_some() => Ok(openchat::api::OpenChatApi::model_info()),
            "stable-diffusion-v1-5" => Ok(Model::new("stable-diffusion-v1-5", "stabilityai", 0)),
            _ => Err(ApiError::model_not_found(id)),
        }
    }

    // ── /v1/completions ────────────────────────────────────────────────

    /// `POST /v1/completions` — text completion.
    ///
    /// Routes to GPT-2. Returns error if GPT-2 is not loaded.
    pub fn complete(&mut self, req: &CompletionRequest) -> Result<CompletionResponse, ApiError> {
        let engine = self.gpt2.as_mut()
            .ok_or_else(|| ApiError::model_not_found(&req.model))?;
        Ok(engine.complete(req))
    }

    // ── /v1/chat/completions ───────────────────────────────────────────

    /// `POST /v1/chat/completions` — chat completion.
    ///
    /// Routes by model name:
    /// - `"openchat_3.5"` / `"openchat"` → OpenChat engine
    /// - `"gpt2"` → GPT-2 via chat adapter (messages → single prompt)
    pub fn chat_complete(&mut self, req: &ChatCompletionRequest) -> Result<ChatCompletionResponse, ApiError> {
        match req.model.as_str() {
            "openchat_3.5" | "openchat" => {
                let engine = self.openchat.as_mut()
                    .ok_or_else(|| ApiError::model_not_found(&req.model))?;
                Ok(engine.chat_complete(req))
            }
            "gpt2" => {
                // Adapter: convert chat messages to a single text prompt for GPT-2
                let engine = self.gpt2.as_mut()
                    .ok_or_else(|| ApiError::model_not_found("gpt2"))?;
                let completion_req = chat_to_completion(req);
                let completion_resp = engine.complete(&completion_req);
                Ok(completion_to_chat(completion_resp))
            }
            other => Err(ApiError::model_not_found(other)),
        }
    }

    // ── /v1/embeddings ─────────────────────────────────────────────────

    /// `POST /v1/embeddings` — generate embeddings.
    ///
    /// Routes to GPT-2 wte (or any model that supports embeddings).
    pub fn embed(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, ApiError> {
        match req.model.as_str() {
            "gpt2" | "text-embedding-gpt2" => {
                let engine = self.gpt2.as_ref()
                    .ok_or_else(|| ApiError::model_not_found(&req.model))?;
                Ok(engine.embed(req))
            }
            other => Err(ApiError::model_not_found(other)),
        }
    }

    // ── /v1/images/generations ─────────────────────────────────────────

    // Note: SD API is stateless per-request (no engine to hold).
    // The router would hold an SD engine when weights are loaded.
    // For now, return model_not_found until SD weights are registered.
}

// ============================================================================
// Adapters: convert between completion and chat formats
// ============================================================================

/// Convert a chat request to a completion request (for GPT-2 chat adapter).
fn chat_to_completion(req: &ChatCompletionRequest) -> CompletionRequest {
    // Concatenate all messages into a single prompt
    let mut prompt = String::new();
    for msg in &req.messages {
        if let Some(c) = &msg.content {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str("System: ");
                    prompt.push_str(c);
                    prompt.push('\n');
                }
                ChatRole::User => {
                    prompt.push_str("User: ");
                    prompt.push_str(c);
                    prompt.push('\n');
                }
                ChatRole::Assistant => {
                    prompt.push_str("Assistant: ");
                    prompt.push_str(c);
                    prompt.push('\n');
                }
                ChatRole::Tool => {
                    prompt.push_str("Tool: ");
                    prompt.push_str(c);
                    prompt.push('\n');
                }
            }
        }
    }
    prompt.push_str("Assistant:");

    CompletionRequest {
        model: "gpt2".into(),
        prompt: Some(prompt),
        prompt_tokens: req.prompt_tokens.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        n: req.n,
        stream: req.stream,
        stop: req.stop.clone(),
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: req.seed,
        ..CompletionRequest::default()
    }
}

/// Convert a completion response to a chat response (for GPT-2 chat adapter).
fn completion_to_chat(resp: CompletionResponse) -> ChatCompletionResponse {
    let choices: Vec<ChatChoice> = resp.choices.into_iter().map(|c| {
        ChatChoice {
            index: c.index,
            message: ChatMessage::assistant(c.text),
            finish_reason: c.finish_reason,
            logprobs: None,
        }
    }).collect();

    ChatCompletionResponse::new(
        resp.id.replace("cmpl-", "chatcmpl-"),
        resp.model,
        choices,
        resp.usage,
        resp.created,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_router() {
        let router = ModelRouter::new();
        assert!(router.loaded_models().is_empty());
    }

    #[test]
    fn test_list_models_always_has_sd() {
        let router = ModelRouter::new();
        let list = router.list_models();
        assert_eq!(list.object, "list");
        assert!(list.data.iter().any(|m| m.id == "stable-diffusion-v1-5"));
    }

    #[test]
    fn test_get_model_not_found() {
        let router = ModelRouter::new();
        let err = router.get_model("nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn test_complete_no_model() {
        let mut router = ModelRouter::new();
        let req = CompletionRequest { model: "gpt2".into(), ..Default::default() };
        let err = router.complete(&req);
        assert!(err.is_err());
    }

    #[test]
    fn test_chat_complete_no_model() {
        let mut router = ModelRouter::new();
        let req = ChatCompletionRequest { model: "openchat_3.5".into(), ..Default::default() };
        let err = router.chat_complete(&req);
        assert!(err.is_err());
    }

    #[test]
    fn test_embed_no_model() {
        let router = ModelRouter::new();
        let req = EmbeddingRequest { model: "gpt2".into(), ..Default::default() };
        let err = router.embed(&req);
        assert!(err.is_err());
    }

    #[test]
    fn test_chat_to_completion_adapter() {
        let req = ChatCompletionRequest {
            model: "gpt2".into(),
            messages: vec![
                ChatMessage::system("Be helpful"),
                ChatMessage::user("Hello"),
            ],
            max_tokens: Some(100),
            temperature: Some(0.5),
            ..Default::default()
        };
        let comp = chat_to_completion(&req);
        assert!(comp.prompt.as_ref().unwrap().contains("System: Be helpful"));
        assert!(comp.prompt.as_ref().unwrap().contains("User: Hello"));
        assert!(comp.prompt.as_ref().unwrap().ends_with("Assistant:"));
        assert_eq!(comp.max_tokens, Some(100));
        assert_eq!(comp.temperature, Some(0.5));
    }

    #[test]
    fn test_completion_to_chat_adapter() {
        let resp = CompletionResponse::new(
            "cmpl-42".into(),
            "gpt2".into(),
            vec![CompletionChoice {
                index: 0,
                text: "Hello world".into(),
                logprobs: None,
                finish_reason: Some(FinishReason::Stop),
            }],
            Usage { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
            0,
        );
        let chat = completion_to_chat(resp);
        assert_eq!(chat.object, "chat.completion");
        assert_eq!(chat.id, "chatcmpl-42");
        assert_eq!(chat.choices[0].message.role, ChatRole::Assistant);
        assert_eq!(chat.choices[0].message.content.as_deref(), Some("Hello world"));
        assert_eq!(chat.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(chat.usage.total_tokens, 7);
    }

    #[test]
    fn test_sd_always_in_model_list() {
        let router = ModelRouter::new();
        let list = router.list_models();
        let sd = list.data.iter().find(|m| m.id == "stable-diffusion-v1-5");
        assert!(sd.is_some());
        assert_eq!(sd.unwrap().object, "model");
    }
}
