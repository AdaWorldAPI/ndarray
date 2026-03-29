//! OpenChat 3.5 API — wraps the inference engine with OpenAI-compatible types.
//!
//! Endpoint: `/v1/chat/completions`
//!
//! Uses the OpenChat template: `GPT4 Correct User: {msg}<|end_of_turn|>`

use crate::hpc::models::api_types::*;
use super::inference::{GeneratedToken, OpenChatEngine};
use super::weights::*;

/// OpenChat API wrapper.
pub struct OpenChatApi {
    engine: OpenChatEngine,
    request_counter: u64,
}

impl OpenChatApi {
    pub fn new(weights: OpenChatWeights) -> Self {
        Self { engine: OpenChatEngine::new(weights), request_counter: 0 }
    }

    /// `/v1/chat/completions`
    pub fn chat_complete(&mut self, req: &ChatCompletionRequest) -> ChatCompletionResponse {
        self.request_counter += 1;
        let tokens = req.prompt_tokens.as_deref().unwrap_or(&[]);
        let max = req.max_tokens.unwrap_or(512);
        let temp = req.temperature.unwrap_or(0.7);

        let generated = self.engine.generate(tokens, max, temp);

        let finish_reason = if generated.len() < max {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let content: String = generated.iter().map(|t| format!("[{}]", t.token_id)).collect();

        ChatCompletionResponse::new(
            format!("chatcmpl-{}", self.request_counter),
            "openchat_3.5".into(),
            vec![ChatChoice {
                index: 0,
                message: ChatMessage::assistant(content),
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            Usage {
                prompt_tokens: tokens.len(),
                completion_tokens: generated.len(),
                total_tokens: tokens.len() + generated.len(),
            },
            0,
        )
    }

    /// Build prompt string from chat messages using OpenChat template.
    ///
    /// ```text
    /// GPT4 Correct User: {msg}<|end_of_turn|>GPT4 Correct Assistant:
    /// ```
    pub fn format_chat_template(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    if let Some(c) = &msg.content {
                        prompt.push_str(c);
                        prompt.push('\n');
                    }
                }
                ChatRole::User => {
                    prompt.push_str(chat_template::USER_PREFIX);
                    if let Some(c) = &msg.content {
                        prompt.push_str(c);
                    }
                    prompt.push_str(chat_template::EOT_TOKEN);
                }
                ChatRole::Assistant => {
                    prompt.push_str(chat_template::ASSISTANT_PREFIX);
                    prompt.push(' ');
                    if let Some(c) = &msg.content {
                        prompt.push_str(c);
                    }
                    prompt.push_str(chat_template::EOT_TOKEN);
                }
                ChatRole::Tool => {
                    // Tool responses treated as user messages
                    prompt.push_str(chat_template::USER_PREFIX);
                    if let Some(c) = &msg.content {
                        prompt.push_str(c);
                    }
                    prompt.push_str(chat_template::EOT_TOKEN);
                }
            }
        }
        prompt.push_str(chat_template::ASSISTANT_PREFIX);
        prompt
    }

    /// `/v1/models/{id}`
    pub fn model_info() -> Model {
        Model::new("openchat_3.5", "openchat", 0)
    }

    pub fn engine_mut(&mut self) -> &mut OpenChatEngine {
        &mut self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_format() {
        let messages = vec![ChatMessage::user("Hello!")];
        let prompt = OpenChatApi::format_chat_template(&messages);
        assert!(prompt.contains("GPT4 Correct User: Hello!"));
        assert!(prompt.contains("<|end_of_turn|>"));
        assert!(prompt.ends_with("GPT4 Correct Assistant:"));
    }

    #[test]
    fn test_chat_template_multi_turn() {
        let messages = vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let prompt = OpenChatApi::format_chat_template(&messages);
        assert_eq!(prompt.matches("GPT4 Correct User:").count(), 2);
        assert!(prompt.contains("Hello!"));
    }

    #[test]
    fn test_chat_template_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi"),
        ];
        let prompt = OpenChatApi::format_chat_template(&messages);
        assert!(prompt.starts_with("You are helpful."));
    }

    #[test]
    fn test_default_request() {
        let req = ChatCompletionRequest::default();
        assert_eq!(req.max_tokens, Some(512));
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn test_model_info() {
        let m = OpenChatApi::model_info();
        assert_eq!(m.id, "openchat_3.5");
        assert_eq!(m.object, "model");
    }

    #[test]
    fn test_chat_response_object() {
        let resp = ChatCompletionResponse::new("x".into(), "m".into(), vec![], Usage::default(), 0);
        assert_eq!(resp.object, "chat.completion");
    }
}
