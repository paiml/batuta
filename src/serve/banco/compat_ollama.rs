//! Ollama API compatibility layer.
//!
//! Translates Ollama protocol requests (`/api/generate`, `/api/chat`, `/api/tags`)
//! to Banco's OpenAI-compatible endpoints. Enables tools like Open WebUI,
//! Continue.dev, and Aider to work with Banco out of the box.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::BancoState;
use crate::serve::templates::ChatMessage;

// ============================================================================
// Ollama Types
// ============================================================================

/// Ollama /api/chat request.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama message (role + content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

/// Ollama /api/chat response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaMessage,
    pub done: bool,
    pub total_duration: u64,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
}

/// Ollama /api/tags response (model list).
#[derive(Debug, Clone, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

/// Ollama model info.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
}

/// Ollama /api/show request.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaShowRequest {
    pub name: String,
}

/// Ollama /api/show response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/chat — Ollama chat endpoint.
pub async fn ollama_chat_handler(
    State(state): State<BancoState>,
    Json(request): Json<OllamaChatRequest>,
) -> Json<OllamaChatResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());

    // Convert Ollama messages to internal ChatMessage
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => ChatMessage::system(&m.content),
            "assistant" => ChatMessage::assistant(&m.content),
            _ => ChatMessage::user(&m.content),
        })
        .collect();

    // Apply template (validates the pipeline)
    let formatted = state.template_engine.apply(&messages);
    let prompt_tokens = state.context_manager.estimate_tokens(&messages) as u32;

    // Phase 1: echo response
    let content = format!(
        "[banco dry-run] route={:?} | prompt_len={} | formatted_len={}",
        state.router.route(),
        messages.len(),
        formatted.len()
    );
    let eval_count = (content.len() / 4) as u32;

    Json(OllamaChatResponse {
        model,
        created_at: chrono::Utc::now().to_rfc3339(),
        message: OllamaMessage { role: "assistant".to_string(), content },
        done: true,
        total_duration: 0,
        prompt_eval_count: prompt_tokens,
        eval_count,
    })
}

/// GET /api/tags — Ollama model list.
pub async fn ollama_tags_handler(State(state): State<BancoState>) -> Json<OllamaTagsResponse> {
    let backends = state.backend_selector.recommend();
    let models = backends
        .iter()
        .map(|b| {
            let name = format!("{b:?}").to_lowercase();
            OllamaModelInfo { name: name.clone(), model: name, size: 0, digest: String::new() }
        })
        .collect();
    Json(OllamaTagsResponse { models })
}

/// POST /api/show — Ollama model info.
pub async fn ollama_show_handler(
    Json(request): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    Json(OllamaShowResponse {
        modelfile: format!("FROM {}", request.name),
        parameters: "temperature 0.7\ntop_p 1.0".to_string(),
        template: "{{ .System }}\n{{ .Prompt }}".to_string(),
    })
}
