//! Banco API types — OpenAI-compatible request/response structures.

use crate::serve::templates::{ChatMessage, Role};
use serde::{Deserialize, Serialize};

// ============================================================================
// BANCO-TYP-001: Chat Completion Request
// ============================================================================

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatRequest {
    /// Model identifier (optional in Phase 1 — echo mode).
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature (0.0–2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Nucleus sampling probability.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Whether to stream the response via SSE.
    #[serde(default)]
    pub stream: bool,
    /// Optional conversation ID to append messages to.
    #[serde(default)]
    pub conversation_id: Option<String>,
}

fn default_max_tokens() -> u32 {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    1.0
}

// ============================================================================
// BANCO-TYP-002: Chat Completion Response (non-streaming)
// ============================================================================

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics with context window info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Total context window size (tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Percentage of context window used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_used_pct: Option<f32>,
}

// ============================================================================
// BANCO-TYP-006: Tokenize / Detokenize
// ============================================================================

/// Tokenize request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

/// Tokenize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: u32,
}

/// Detokenize request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeRequest {
    pub tokens: Vec<u32>,
}

/// Detokenize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

// ============================================================================
// BANCO-TYP-007: Embeddings
// ============================================================================

/// OpenAI-compatible embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model identifier (ignored in Phase 1 — uses heuristic).
    #[serde(default)]
    pub model: Option<String>,
    /// Input text(s) to embed.
    pub input: EmbeddingsInput,
}

/// Embeddings input — single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingsInput {
    pub fn texts(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Batch(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

/// OpenAI-compatible embeddings response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

/// A single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// Token usage for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// BANCO-TYP-003: SSE Streaming Chunk
// ============================================================================

/// A single Server-Sent Event chunk for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BancoChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// A single choice within a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Delta content within a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ============================================================================
// BANCO-TYP-004: Health / Models / System
// ============================================================================

/// Health endpoint response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub circuit_breaker_state: String,
    pub uptime_secs: u64,
}

/// A single model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub local: bool,
}

/// Models list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// System information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResponse {
    pub privacy_tier: String,
    pub backends: Vec<String>,
    pub gpu_available: bool,
    pub version: String,
    /// Banco never collects telemetry. Always false.
    pub telemetry: bool,
    /// Whether a model is currently loaded.
    pub model_loaded: bool,
    /// ID of the loaded model (None if no model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

// ============================================================================
// BANCO-TYP-005: Error Response
// ============================================================================

/// OpenAI-compatible error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail within an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub code: u16,
}

impl ErrorResponse {
    /// Create a new error response.
    #[must_use]
    pub fn new(message: impl Into<String>, type_: impl Into<String>, code: u16) -> Self {
        Self { error: ErrorDetail { message: message.into(), type_: type_.into(), code } }
    }
}

// ============================================================================
// BANCO-TYP-008: Conversations
// ============================================================================

/// Create conversation request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

/// Conversation list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationsListResponse {
    pub conversations: Vec<super::conversations::ConversationMeta>,
}

/// Single conversation response (with messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationResponse {
    #[serde(flatten)]
    pub conversation: super::conversations::Conversation,
}

/// Conversation created response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationCreatedResponse {
    pub id: String,
    pub title: String,
}

// ============================================================================
// BANCO-TYP-010: Model Management
// ============================================================================

/// Model load request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadRequest {
    /// Path or URI to model file.
    pub model: String,
    /// Which slot to load into (default: "primary").
    #[serde(default = "default_slot")]
    pub slot: String,
}

fn default_slot() -> String {
    "primary".to_string()
}

/// Model status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusResponse {
    pub loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<super::model_slot::ModelSlotInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<u64>,
}

// ============================================================================
// BANCO-TYP-009: Prompt Presets
// ============================================================================

/// Create/update prompt preset request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavePromptRequest {
    pub name: String,
    pub content: String,
}

/// Prompt presets list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsListResponse {
    pub presets: Vec<super::prompts::PromptPreset>,
}
