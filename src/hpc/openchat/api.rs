//! OpenAI-compatible chat completions API for OpenChat 3.5.
//!
//! Implements `/v1/chat/completions` with the OpenChat template:
//! ```text
//! GPT4 Correct User: {message}<|end_of_turn|>
//! GPT4 Correct Assistant:
//! ```

use crate::hpc::models::api_types::{Usage, FinishReason};
use super::inference::{GeneratedToken, OpenChatEngine};
use super::weights::*;

/// A chat message (role + content).
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Chat role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Request body for /v1/chat/completions.
#[derive(Clone, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Pre-tokenized prompt (built from messages via chat template).
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream: bool,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: "openchat_3.5".into(),
            messages: Vec::new(),
            prompt_tokens: Vec::new(),
            max_tokens: 512,
            temperature: 0.7,
            stream: false,
        }
    }
}

/// A chat completion choice.
#[derive(Clone, Debug)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Response body for /v1/chat/completions.
#[derive(Clone, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// OpenChat API wrapper.
pub struct OpenChatApi {
    engine: OpenChatEngine,
    request_counter: u64,
}

impl OpenChatApi {
    pub fn new(weights: OpenChatWeights) -> Self {
        Self {
            engine: OpenChatEngine::new(weights),
            request_counter: 0,
        }
    }

    /// /v1/chat/completions handler.
    pub fn chat_complete(&mut self, req: &ChatCompletionRequest) -> ChatCompletionResponse {
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

        ChatCompletionResponse {
            id: format!("chatcmpl-{}", self.request_counter),
            model: "openchat_3.5".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: format!("[{} tokens generated]", completion_tokens),
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }

    /// Build prompt token sequence from chat messages using OpenChat template.
    ///
    /// Format:
    /// ```text
    /// <bos>GPT4 Correct User: {user_msg}<|end_of_turn|>GPT4 Correct Assistant:
    /// ```
    ///
    /// Returns a description of the template (actual tokenization requires SentencePiece).
    pub fn format_chat_template(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                ChatRole::User => {
                    prompt.push_str(chat_template::USER_PREFIX);
                    prompt.push_str(&msg.content);
                    prompt.push_str(chat_template::EOT_TOKEN);
                }
                ChatRole::Assistant => {
                    prompt.push_str(chat_template::ASSISTANT_PREFIX);
                    prompt.push(' ');
                    prompt.push_str(&msg.content);
                    prompt.push_str(chat_template::EOT_TOKEN);
                }
            }
        }
        // Always end with assistant prefix to prompt generation
        prompt.push_str(chat_template::ASSISTANT_PREFIX);
        prompt
    }

    /// Access engine for direct manipulation.
    pub fn engine_mut(&mut self) -> &mut OpenChatEngine {
        &mut self.engine
    }

    /// Model info.
    pub fn model_info() -> crate::hpc::models::api_types::ModelCard {
        crate::hpc::models::api_types::ModelCard {
            id: "openchat_3.5".into(),
            owned_by: "openchat".into(),
            created: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_format() {
        let messages = vec![
            ChatMessage { role: ChatRole::User, content: "Hello!".into() },
        ];
        let prompt = OpenChatApi::format_chat_template(&messages);
        assert!(prompt.contains("GPT4 Correct User: Hello!"));
        assert!(prompt.contains("<|end_of_turn|>"));
        assert!(prompt.ends_with("GPT4 Correct Assistant:"));
    }

    #[test]
    fn test_chat_template_multi_turn() {
        let messages = vec![
            ChatMessage { role: ChatRole::User, content: "Hi".into() },
            ChatMessage { role: ChatRole::Assistant, content: "Hello!".into() },
            ChatMessage { role: ChatRole::User, content: "How are you?".into() },
        ];
        let prompt = OpenChatApi::format_chat_template(&messages);
        // Should have two user turns and one assistant turn
        assert_eq!(prompt.matches("GPT4 Correct User:").count(), 2);
        assert!(prompt.contains("Hello!"));
    }

    #[test]
    fn test_chat_template_with_system() {
        let messages = vec![
            ChatMessage { role: ChatRole::System, content: "You are helpful.".into() },
            ChatMessage { role: ChatRole::User, content: "Hi".into() },
        ];
        let prompt = OpenChatApi::format_chat_template(&messages);
        assert!(prompt.starts_with("You are helpful."));
    }

    #[test]
    fn test_default_request() {
        let req = ChatCompletionRequest::default();
        assert_eq!(req.model, "openchat_3.5");
        assert_eq!(req.max_tokens, 512);
        assert!(!req.stream);
    }

    #[test]
    fn test_model_info() {
        let info = OpenChatApi::model_info();
        assert_eq!(info.id, "openchat_3.5");
    }

    #[test]
    fn test_chat_role_eq() {
        assert_eq!(ChatRole::User, ChatRole::User);
        assert_ne!(ChatRole::User, ChatRole::Assistant);
    }
}
