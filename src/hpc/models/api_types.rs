//! OpenAI-compatible API types shared across all model endpoints.
//!
//! Transport-agnostic — no HTTP framework dependency.
//! Used by GPT-2 (/v1/completions), Stable Diffusion (/v1/images/generations),
//! BERT/Jina (/v1/embeddings).

/// Token usage statistics (shared by all endpoints).
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Why generation stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit stop token or stop sequence.
    Stop,
    /// Hit max_tokens limit.
    Length,
    /// Content filter triggered.
    ContentFilter,
}

/// Error response envelope.
#[derive(Clone, Debug)]
pub struct ApiError {
    pub message: String,
    pub error_type: String,
    pub code: Option<String>,
}

impl ApiError {
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            error_type: "invalid_request_error".into(),
            code: None,
        }
    }

    pub fn model_not_found(model: &str) -> Self {
        Self {
            message: format!("model '{}' not found", model),
            error_type: "invalid_request_error".into(),
            code: Some("model_not_found".into()),
        }
    }
}

/// Model info for /v1/models listing.
#[derive(Clone, Debug)]
pub struct ModelCard {
    pub id: String,
    pub owned_by: String,
    pub created: u64,
}

/// Embedding data for /v1/embeddings response.
#[derive(Clone, Debug)]
pub struct EmbeddingData {
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// /v1/embeddings response (shared by BERT, Jina, GPT-2 wte).
#[derive(Clone, Debug)]
pub struct EmbeddingResponse {
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: Usage,
}

/// Image data for /v1/images/generations response.
#[derive(Clone, Debug)]
pub struct ImageData {
    /// Base64-encoded PNG, or URL if hosted.
    pub b64_json: Option<String>,
    pub url: Option<String>,
    pub revised_prompt: Option<String>,
}

/// /v1/images/generations response (Stable Diffusion).
#[derive(Clone, Debug)]
pub struct ImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_default() {
        let u = Usage::default();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.total_tokens, 0);
    }

    #[test]
    fn test_api_error_invalid_request() {
        let e = ApiError::invalid_request("bad input");
        assert_eq!(e.error_type, "invalid_request_error");
        assert!(e.code.is_none());
    }

    #[test]
    fn test_api_error_model_not_found() {
        let e = ApiError::model_not_found("gpt-5");
        assert!(e.message.contains("gpt-5"));
        assert_eq!(e.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn test_finish_reason_eq() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::Length);
        assert_ne!(FinishReason::Length, FinishReason::ContentFilter);
    }
}
