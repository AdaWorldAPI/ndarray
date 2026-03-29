//! OpenAI-compatible API types — 1:1 field match with OpenAI REST API.
//!
//! Single source of truth for all model endpoints. Every struct matches
//! the exact JSON field names from the OpenAI API reference.
//!
//! Endpoints covered:
//! - `POST /v1/completions` — text completion (GPT-2)
//! - `POST /v1/chat/completions` — chat completion (OpenChat 3.5)
//! - `POST /v1/embeddings` — embeddings (GPT-2 wte, Jina, BERT)
//! - `POST /v1/images/generations` — image generation (Stable Diffusion)
//! - `GET  /v1/models` — model listing
//! - `GET  /v1/models/{id}` — model detail
//!
//! Transport-agnostic — no HTTP framework dependency.
//! When the `serde` feature is enabled, all types derive Serialize/Deserialize.

// ============================================================================
// Common types
// ============================================================================

/// Token usage statistics. Matches OpenAI `usage` object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Why generation stopped. Matches OpenAI `finish_reason` string values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Model hit a stop token or stop sequence. JSON: `"stop"`
    Stop,
    /// Hit `max_tokens` limit. JSON: `"length"`
    Length,
    /// Content filter triggered. JSON: `"content_filter"`
    ContentFilter,
    /// Tool/function call requested. JSON: `"tool_calls"`
    ToolCalls,
}

impl FinishReason {
    /// OpenAI JSON string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// Error response envelope. Matches OpenAI `error` object.
#[derive(Clone, Debug)]
pub struct ApiError {
    pub message: String,
    /// `"invalid_request_error"`, `"authentication_error"`, `"rate_limit_error"`, etc.
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), r#type: "invalid_request_error".into(), param: None, code: None }
    }
    pub fn model_not_found(model: &str) -> Self {
        Self {
            message: format!("The model '{}' does not exist", model),
            r#type: "invalid_request_error".into(),
            param: Some("model".into()),
            code: Some("model_not_found".into()),
        }
    }
}

/// Wrapper for error responses: `{ "error": { ... } }`.
#[derive(Clone, Debug)]
pub struct ErrorResponse {
    pub error: ApiError,
}

// ============================================================================
// /v1/models
// ============================================================================

/// Model object. Matches OpenAI `Model` response.
#[derive(Clone, Debug)]
pub struct Model {
    /// Unique model identifier (e.g., `"gpt2"`, `"openchat_3.5"`).
    pub id: String,
    /// Always `"model"`.
    pub object: &'static str,
    /// Unix timestamp (seconds) when the model was created.
    pub created: u64,
    /// Organization that owns the model.
    pub owned_by: String,
}

impl Model {
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>, created: u64) -> Self {
        Self { id: id.into(), object: "model", created, owned_by: owned_by.into() }
    }
}

/// Response for `GET /v1/models`. Matches OpenAI list response.
#[derive(Clone, Debug)]
pub struct ModelList {
    pub object: &'static str, // "list"
    pub data: Vec<Model>,
}

impl ModelList {
    pub fn new(models: Vec<Model>) -> Self {
        Self { object: "list", data: models }
    }
}

// ============================================================================
// /v1/completions
// ============================================================================

/// Request body for `POST /v1/completions`.
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    pub model: String,
    /// Text prompt. Mutually exclusive with `prompt_tokens`.
    pub prompt: Option<String>,
    /// Pre-tokenized prompt (extension — not in OpenAI API).
    pub prompt_tokens: Option<Vec<u32>>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub stream: Option<bool>,
    pub logprobs: Option<usize>,
    pub echo: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub best_of: Option<usize>,
    pub user: Option<String>,
    /// Suffix for insertion completions (fill-in-the-middle).
    pub suffix: Option<String>,
    /// Seed for deterministic generation.
    pub seed: Option<u64>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: "gpt2".into(),
            prompt: None,
            prompt_tokens: None,
            max_tokens: Some(128),
            temperature: Some(1.0),
            top_p: None,
            n: Some(1),
            stream: Some(false),
            logprobs: None,
            echo: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            user: None,
            suffix: None,
            seed: None,
        }
    }
}

/// Log probability information for a token.
#[derive(Clone, Debug)]
pub struct LogprobInfo {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    /// Top-N alternative tokens and their logprobs.
    pub top_logprobs: Vec<TopLogprob>,
}

/// An alternative token with its logprob.
#[derive(Clone, Debug)]
pub struct TopLogprob {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
}

/// Single completion choice. Matches OpenAI `Choice` object.
#[derive(Clone, Debug)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub logprobs: Option<Vec<LogprobInfo>>,
    pub finish_reason: Option<FinishReason>,
}

/// Response body for `POST /v1/completions`.
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str, // "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    pub system_fingerprint: Option<String>,
}

impl CompletionResponse {
    pub fn new(id: String, model: String, choices: Vec<CompletionChoice>, usage: Usage, created: u64) -> Self {
        Self { id, object: "text_completion", created, model, choices, usage, system_fingerprint: None }
    }
}

// ============================================================================
// /v1/chat/completions
// ============================================================================

/// Chat message role. Matches OpenAI `role` string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Self::System),
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            "tool" => Some(Self::Tool),
            _ => None,
        }
    }
}

/// A single chat message. Matches OpenAI message object.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Option<String>,
    /// Function/tool call name (when role=assistant and calling a tool).
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (when role=tool, responding to a tool call).
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: Some(content.into()), name: None, tool_calls: None, tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: Some(content.into()), name: None, tool_calls: None, tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: Some(content.into()), name: None, tool_calls: None, tool_call_id: None }
    }
}

/// Tool call in an assistant message.
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // "function"
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Clone, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool definition for /v1/chat/completions request.
#[derive(Clone, Debug)]
pub struct Tool {
    pub r#type: String, // "function"
    pub function: FunctionDef,
}

/// Function definition.
#[derive(Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<String>, // JSON Schema string
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Clone, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<String>,
    pub user: Option<String>,
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,
    /// Pre-tokenized prompt (extension — for direct token input).
    pub prompt_tokens: Option<Vec<u32>>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            max_tokens: Some(512),
            temperature: Some(1.0),
            top_p: None,
            n: Some(1),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            user: None,
            seed: None,
            response_format: None,
            prompt_tokens: None,
        }
    }
}

/// Response format constraint.
#[derive(Clone, Debug)]
pub struct ResponseFormat {
    pub r#type: String, // "text" or "json_object"
}

/// Single chat completion choice.
#[derive(Clone, Debug)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<ChatLogprobs>,
}

/// Logprobs for chat completion.
#[derive(Clone, Debug)]
pub struct ChatLogprobs {
    pub content: Option<Vec<LogprobInfo>>,
}

/// Response body for `POST /v1/chat/completions`.
#[derive(Clone, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: Usage, created: u64) -> Self {
        Self { id, object: "chat.completion", created, model, choices, usage, system_fingerprint: None }
    }
}

/// Streaming chunk for `POST /v1/chat/completions` with `stream: true`.
#[derive(Clone, Debug)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    pub system_fingerprint: Option<String>,
}

/// Single streaming choice delta.
#[derive(Clone, Debug)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<FinishReason>,
}

/// Delta content in a streaming chunk.
#[derive(Clone, Debug, Default)]
pub struct ChatDelta {
    pub role: Option<ChatRole>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ============================================================================
// /v1/embeddings
// ============================================================================

/// Request body for `POST /v1/embeddings`.
#[derive(Clone, Debug)]
pub struct EmbeddingRequest {
    pub model: String,
    /// Input text(s) to embed.
    pub input: EmbeddingInput,
    /// Optional: encoding format (`"float"` or `"base64"`).
    pub encoding_format: Option<String>,
    /// Optional: dimensions to truncate to.
    pub dimensions: Option<usize>,
    pub user: Option<String>,
    /// Pre-tokenized input (extension — for direct token input).
    pub input_tokens: Option<Vec<u32>>,
}

/// Embedding input — string, array of strings, or token IDs.
#[derive(Clone, Debug)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
    TokenIds(Vec<u32>),
    BatchTokenIds(Vec<Vec<u32>>),
}

impl Default for EmbeddingRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            input: EmbeddingInput::Single(String::new()),
            encoding_format: None,
            dimensions: None,
            user: None,
            input_tokens: None,
        }
    }
}

/// Single embedding result.
#[derive(Clone, Debug)]
pub struct EmbeddingData {
    pub object: &'static str, // "embedding"
    pub index: usize,
    pub embedding: Vec<f32>,
}

impl EmbeddingData {
    pub fn new(index: usize, embedding: Vec<f32>) -> Self {
        Self { object: "embedding", index, embedding }
    }
}

/// Response body for `POST /v1/embeddings`.
#[derive(Clone, Debug)]
pub struct EmbeddingResponse {
    pub object: &'static str, // "list"
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: Usage,
}

impl EmbeddingResponse {
    pub fn new(model: String, data: Vec<EmbeddingData>, usage: Usage) -> Self {
        Self { object: "list", model, data, usage }
    }
}

// ============================================================================
// /v1/images/generations
// ============================================================================

/// Request body for `POST /v1/images/generations`.
#[derive(Clone, Debug)]
pub struct ImageGenerationRequest {
    pub model: Option<String>,
    pub prompt: String,
    /// Number of images to generate (1-10).
    pub n: Option<usize>,
    /// `"256x256"`, `"512x512"`, `"1024x1024"`, `"1792x1024"`, `"1024x1792"`.
    pub size: Option<String>,
    /// `"url"` or `"b64_json"`.
    pub response_format: Option<String>,
    /// `"vivid"` or `"natural"`.
    pub style: Option<String>,
    /// `"standard"` or `"hd"`.
    pub quality: Option<String>,
    pub user: Option<String>,
    /// Seed (extension — not in OpenAI API but useful for reproducibility).
    pub seed: Option<u64>,
    /// Pre-tokenized prompt (extension — for direct CLIP token input).
    pub prompt_tokens: Option<Vec<u32>>,
}

impl Default for ImageGenerationRequest {
    fn default() -> Self {
        Self {
            model: Some("stable-diffusion-v1-5".into()),
            prompt: String::new(),
            n: Some(1),
            size: Some("512x512".into()),
            response_format: Some("b64_json".into()),
            style: None,
            quality: None,
            user: None,
            seed: None,
            prompt_tokens: None,
        }
    }
}

impl ImageGenerationRequest {
    /// Parse `size` string into (width, height).
    pub fn dimensions(&self) -> (usize, usize) {
        match self.size.as_deref() {
            Some("256x256") => (256, 256),
            Some("512x512") => (512, 512),
            Some("1024x1024") => (1024, 1024),
            Some("1792x1024") => (1792, 1024),
            Some("1024x1792") => (1024, 1792),
            _ => (512, 512),
        }
    }
}

/// Single image result.
#[derive(Clone, Debug)]
pub struct ImageData {
    pub b64_json: Option<String>,
    pub url: Option<String>,
    pub revised_prompt: Option<String>,
}

/// Response body for `POST /v1/images/generations`.
#[derive(Clone, Debug)]
pub struct ImageResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

// ============================================================================
// Tests
// ============================================================================

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
    fn test_finish_reason_str() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
        assert_eq!(FinishReason::ContentFilter.as_str(), "content_filter");
        assert_eq!(FinishReason::ToolCalls.as_str(), "tool_calls");
    }

    #[test]
    fn test_api_error() {
        let e = ApiError::invalid_request("bad");
        assert_eq!(e.r#type, "invalid_request_error");
        let e = ApiError::model_not_found("gpt-5");
        assert!(e.message.contains("gpt-5"));
        assert_eq!(e.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn test_model_object() {
        let m = Model::new("gpt2", "adaworldapi", 1700000000);
        assert_eq!(m.object, "model");
        assert_eq!(m.id, "gpt2");
    }

    #[test]
    fn test_model_list() {
        let list = ModelList::new(vec![Model::new("a", "x", 0), Model::new("b", "x", 0)]);
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 2);
    }

    #[test]
    fn test_completion_defaults() {
        let req = CompletionRequest::default();
        assert_eq!(req.model, "gpt2");
        assert_eq!(req.max_tokens, Some(128));
        assert_eq!(req.temperature, Some(1.0));
    }

    #[test]
    fn test_completion_response_object() {
        let resp = CompletionResponse::new("cmpl-1".into(), "gpt2".into(), vec![], Usage::default(), 0);
        assert_eq!(resp.object, "text_completion");
    }

    #[test]
    fn test_chat_role_roundtrip() {
        assert_eq!(ChatRole::from_str("system"), Some(ChatRole::System));
        assert_eq!(ChatRole::from_str("user"), Some(ChatRole::User));
        assert_eq!(ChatRole::from_str("assistant"), Some(ChatRole::Assistant));
        assert_eq!(ChatRole::from_str("tool"), Some(ChatRole::Tool));
        assert_eq!(ChatRole::from_str("invalid"), None);
        assert_eq!(ChatRole::System.as_str(), "system");
    }

    #[test]
    fn test_chat_message_constructors() {
        let m = ChatMessage::system("be helpful");
        assert_eq!(m.role, ChatRole::System);
        assert_eq!(m.content.as_deref(), Some("be helpful"));
        let m = ChatMessage::user("hello");
        assert_eq!(m.role, ChatRole::User);
        let m = ChatMessage::assistant("hi");
        assert_eq!(m.role, ChatRole::Assistant);
    }

    #[test]
    fn test_chat_completion_response_object() {
        let resp = ChatCompletionResponse::new("chatcmpl-1".into(), "oc".into(), vec![], Usage::default(), 0);
        assert_eq!(resp.object, "chat.completion");
    }

    #[test]
    fn test_chat_defaults() {
        let req = ChatCompletionRequest::default();
        assert_eq!(req.max_tokens, Some(512));
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn test_embedding_data_object() {
        let d = EmbeddingData::new(0, vec![0.1, 0.2]);
        assert_eq!(d.object, "embedding");
    }

    #[test]
    fn test_embedding_response_object() {
        let r = EmbeddingResponse::new("m".into(), vec![], Usage::default());
        assert_eq!(r.object, "list");
    }

    #[test]
    fn test_image_dimensions() {
        let mut req = ImageGenerationRequest::default();
        assert_eq!(req.dimensions(), (512, 512));
        req.size = Some("1024x1024".into());
        assert_eq!(req.dimensions(), (1024, 1024));
        req.size = Some("1792x1024".into());
        assert_eq!(req.dimensions(), (1792, 1024));
    }

    #[test]
    fn test_streaming_chunk_object() {
        let chunk = ChatCompletionChunk {
            id: "x".into(), object: "chat.completion.chunk", created: 0,
            model: "m".into(), choices: vec![], system_fingerprint: None,
        };
        assert_eq!(chunk.object, "chat.completion.chunk");
    }

    #[test]
    fn test_error_response() {
        let err = ErrorResponse { error: ApiError::invalid_request("test") };
        assert_eq!(err.error.r#type, "invalid_request_error");
    }
}
