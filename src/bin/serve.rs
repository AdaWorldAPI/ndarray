//! OpenAI-compatible REST API server powered by ndarray HPC pipeline.
//!
//! ```bash
//! cargo run --bin serve --features serve --release
//! curl http://localhost:3000/v1/models
//! curl -X POST http://localhost:3000/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello"}]}'
//! ```

#[cfg(feature = "serve")]
mod server {
    use axum::{
        extract::State,
        http::StatusCode,
        response::Json,
        routing::{get, post},
        Router,
    };
    use std::sync::Mutex;

    use ndarray::hpc::models::api_types::*;
    use ndarray::hpc::models::router::ModelRouter;

    type AppState = std::sync::Arc<Mutex<ModelRouter>>;

    async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
        let router = state.lock().unwrap();
        let models = router.list_models();
        Json(serde_json::json!({
            "object": "list",
            "data": models.data.iter().map(|m| serde_json::json!({
                "id": m.id,
                "object": "model",
                "owned_by": m.owned_by,
            })).collect::<Vec<_>>()
        }))
    }

    async fn chat_completions(
        State(state): State<AppState>,
        Json(req): Json<serde_json::Value>,
    ) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
        let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("gpt2");
        let messages = req.get("messages").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let temperature = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;

        let chat_messages: Vec<ChatMessage> = messages.iter().filter_map(|m| {
            let role_str = m.get("role")?.as_str()?;
            let content = m.get("content")?.as_str()?.to_string();
            let role = match role_str {
                "system" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                "tool" => ChatRole::Tool,
                _ => ChatRole::User,
            };
            Some(ChatMessage {
                role,
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            })
        }).collect();

        let chat_req = ChatCompletionRequest {
            model: model.to_string(),
            messages: chat_messages,
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            ..ChatCompletionRequest::default()
        };

        let mut router = state.lock().unwrap();
        match router.chat_complete(&chat_req) {
            Ok(resp) => {
                let choices: Vec<serde_json::Value> = resp.choices.iter().map(|c| {
                    serde_json::json!({
                        "index": c.index,
                        "message": {
                            "role": c.message.role.as_str(),
                            "content": c.message.content.as_deref().unwrap_or("")
                        },
                        "finish_reason": match &c.finish_reason {
                            Some(FinishReason::Stop) => "stop",
                            Some(FinishReason::Length) => "length",
                            Some(FinishReason::ContentFilter) => "content_filter",
                            _ => "stop",
                        }
                    })
                }).collect();

                Ok(Json(serde_json::json!({
                    "id": resp.id,
                    "object": "chat.completion",
                    "model": model,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                })))
            }
            Err(e) => Err((StatusCode::BAD_REQUEST, format!("{:?}", e))),
        }
    }

    async fn completions(
        State(state): State<AppState>,
        Json(req): Json<serde_json::Value>,
    ) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
        let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("gpt2");
        let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let temperature = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;

        let comp_req = CompletionRequest {
            model: model.to_string(),
            prompt: Some(prompt.to_string()),
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            ..CompletionRequest::default()
        };

        let mut router = state.lock().unwrap();
        match router.complete(&comp_req) {
            Ok(resp) => {
                let choices: Vec<serde_json::Value> = resp.choices.iter().map(|c| {
                    serde_json::json!({
                        "text": c.text,
                        "index": c.index,
                        "finish_reason": match &c.finish_reason {
                            Some(FinishReason::Stop) => "stop",
                            Some(FinishReason::Length) => "length",
                            _ => "stop",
                        },
                    })
                }).collect();

                Ok(Json(serde_json::json!({
                    "id": resp.id,
                    "object": "text_completion",
                    "model": model,
                    "choices": choices,
                })))
            }
            Err(e) => Err((StatusCode::BAD_REQUEST, format!("{:?}", e))),
        }
    }

    async fn health() -> &'static str {
        "ok"
    }

    pub async fn run(port: u16) {
        let router = ModelRouter::new();
        let state: AppState = std::sync::Arc::new(Mutex::new(router));

        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .with_state(state);

        let addr = format!("0.0.0.0:{port}");
        eprintln!("ndarray serve listening on {addr}");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }
}

#[cfg(feature = "serve")]
#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    server::run(port).await;
}

#[cfg(not(feature = "serve"))]
fn main() {
    eprintln!("Enable the 'serve' feature: cargo run --bin serve --features serve");
    std::process::exit(1);
}
