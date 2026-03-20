//! Banco HTTP endpoint handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::Stream;

use super::state::BancoState;
use super::types::{
    BancoChatChunk, BancoChatRequest, BancoChatResponse, ChatChoice, ChatChunkChoice, ChatDelta,
    ConversationCreatedResponse, ConversationResponse, ConversationsListResponse,
    CreateConversationRequest, DetokenizeRequest, DetokenizeResponse, EmbeddingData,
    EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage, ErrorResponse, PromptsListResponse,
    SavePromptRequest, TokenizeRequest, TokenizeResponse, Usage,
};
use crate::serve::router::RoutingDecision;
use crate::serve::templates::{ChatMessage, Role};

// ============================================================================
// BANCO-HDL-001: Health
// ============================================================================

pub async fn health_handler(State(state): State<BancoState>) -> Json<super::types::HealthResponse> {
    Json(state.health_status())
}

// ============================================================================
// BANCO-HDL-002: Models
// ============================================================================

pub async fn models_handler(State(state): State<BancoState>) -> Json<super::types::ModelsResponse> {
    Json(state.list_models())
}

// ============================================================================
// BANCO-HDL-003: System
// ============================================================================

pub async fn system_handler(State(state): State<BancoState>) -> Json<super::types::SystemResponse> {
    Json(state.system_info())
}

// ============================================================================
// BANCO-HDL-004: Chat Completions
// ============================================================================

pub async fn chat_completions_handler(
    State(state): State<BancoState>,
    Json(request): Json<BancoChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Expand @preset: references in message content
    let mut request = request;
    for msg in &mut request.messages {
        msg.content = state.prompts.expand(&msg.content);
    }

    // Validate messages are not empty
    if request.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("messages must not be empty", "invalid_request", 400)),
        ));
    }

    // Check context window
    if !state.context_manager.fits(&request.messages) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "messages exceed context window",
                "context_length_exceeded",
                400,
            )),
        ));
    }

    // Check circuit breaker (Phase 1: estimate tiny cost)
    if let Err(e) = state.circuit_breaker.check(0.001) {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse::new(e.to_string(), "rate_limit", 429)),
        ));
    }

    // Save user messages to conversation if conversation_id provided
    if let Some(ref conv_id) = request.conversation_id {
        for msg in &request.messages {
            let _ = state.conversations.append(conv_id, msg.clone());
        }
    }

    // Route the request
    let decision = state.router.route();

    if request.stream {
        Ok(stream_response(state, request, decision).into_response())
    } else {
        Ok(sync_response(state, request, decision).into_response())
    }
}

// ============================================================================
// BANCO-HDL-005: Non-streaming response (echo/dry-run)
// ============================================================================

fn sync_response(
    state: BancoState,
    request: BancoChatRequest,
    decision: RoutingDecision,
) -> Json<BancoChatResponse> {
    let model = request.model.clone().unwrap_or_else(|| "banco-echo".to_string());

    // Apply template engine to format the prompt (validates the pipeline)
    let formatted = state.template_engine.apply(&request.messages);

    // Build echo content describing the routing decision
    let content = format!(
        "[banco dry-run] route={decision:?} | model={model} | prompt_len={} | formatted_len={}",
        request.messages.len(),
        formatted.len()
    );

    let prompt_tokens = state.context_manager.estimate_tokens(&request.messages) as u32;
    let completion_tokens = (content.len() / 4) as u32;

    Json(BancoChatResponse {
        id: format!("banco-{}", now_epoch()),
        object: "chat.completion".to_string(),
        created: now_epoch(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant(content),
            finish_reason: "dry_run".to_string(),
        }],
        usage: {
            let total = prompt_tokens + completion_tokens;
            let window = state.context_manager.available_tokens() as u32;
            let pct = if window > 0 { (total as f32 / window as f32) * 100.0 } else { 0.0 };
            Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: total,
                context_window: Some(window),
                context_used_pct: Some(pct),
            }
        },
    })
}

// ============================================================================
// BANCO-HDL-006: SSE streaming response
// ============================================================================

fn stream_response(
    _state: BancoState,
    request: BancoChatRequest,
    decision: RoutingDecision,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let model = request.model.clone().unwrap_or_else(|| "banco-echo".to_string());
    let id = format!("banco-{}", now_epoch());
    let created = now_epoch();

    let tokens: Vec<String> = vec![
        "[banco".to_string(),
        " dry-run]".to_string(),
        format!(" route={decision:?}"),
        " |".to_string(),
        " done".to_string(),
    ];

    let stream = async_stream::stream! {
        // Role chunk
        let role_chunk = BancoChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: Some(Role::Assistant), content: None },
                finish_reason: None,
            }],
        };
        if let Ok(data) = serde_json::to_string(&role_chunk) {
            yield Ok(Event::default().data(data));
        }

        // Content chunks
        for token in &tokens {
            let chunk = BancoChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta { role: None, content: Some(token.clone()) },
                    finish_reason: None,
                }],
            };
            if let Ok(data) = serde_json::to_string(&chunk) {
                yield Ok(Event::default().data(data));
            }
        }

        // Final chunk with finish_reason
        let done_chunk = BancoChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: None, content: None },
                finish_reason: Some("dry_run".to_string()),
            }],
        };
        if let Ok(data) = serde_json::to_string(&done_chunk) {
            yield Ok(Event::default().data(data));
        }

        // [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

// ============================================================================
// BANCO-HDL-006: Embeddings
// ============================================================================

pub async fn embeddings_handler(
    State(state): State<BancoState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Json<EmbeddingsResponse> {
    let model = request.model.unwrap_or_else(|| "banco-heuristic".to_string());
    let texts = request.input.texts();

    // Phase 1: heuristic embeddings (hash-based, 128-dim)
    // Real embeddings require a loaded model (Phase 2)
    let data: Vec<EmbeddingData> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let embedding = heuristic_embedding(text);
            EmbeddingData { object: "embedding".to_string(), index: i as u32, embedding }
        })
        .collect();

    let total_tokens: u32 = texts
        .iter()
        .map(|t| state.context_manager.estimate_tokens(&[ChatMessage::user(*t)]) as u32)
        .sum();

    Json(EmbeddingsResponse {
        object: "list".to_string(),
        data,
        model,
        usage: EmbeddingsUsage { prompt_tokens: total_tokens, total_tokens },
    })
}

/// Phase 1 heuristic: deterministic 128-dim embedding from text hash.
/// Not semantically meaningful — placeholder until a model is loaded.
fn heuristic_embedding(text: &str) -> Vec<f32> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }
    // Generate 128 dimensions from the hash via simple PRNG
    let mut state = hash;
    (0..128)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Map to [-1.0, 1.0] range
            ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        })
        .collect()
}

// ============================================================================
// BANCO-HDL-007: Tokenize
// ============================================================================

pub async fn tokenize_handler(
    State(state): State<BancoState>,
    Json(request): Json<TokenizeRequest>,
) -> Json<TokenizeResponse> {
    // Phase 1: heuristic tokenizer (~4 chars/token)
    let estimated = state.context_manager.estimate_tokens(&[ChatMessage::user(&request.text)]);
    // Generate pseudo-token IDs (sequential, since we don't have a real tokenizer)
    let tokens: Vec<u32> = (0..estimated as u32).collect();
    Json(TokenizeResponse { count: estimated as u32, tokens })
}

// ============================================================================
// BANCO-HDL-008: Detokenize
// ============================================================================

pub async fn detokenize_handler(
    Json(request): Json<DetokenizeRequest>,
) -> Json<DetokenizeResponse> {
    // Phase 1: heuristic (each token ≈ 4 chars, return placeholder)
    let approx_chars = request.tokens.len() * 4;
    let text = format!("[{} tokens ≈ {} chars]", request.tokens.len(), approx_chars);
    Json(DetokenizeResponse { text })
}

// ============================================================================
// BANCO-HDL-009: Conversations
// ============================================================================

pub async fn create_conversation_handler(
    State(state): State<BancoState>,
    Json(request): Json<CreateConversationRequest>,
) -> Json<ConversationCreatedResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());
    let id = state.conversations.create(&model);

    // Apply custom title if provided
    if let Some(title) = request.title {
        if let Some(mut conv) = state.conversations.get(&id) {
            conv.meta.title = title;
        }
    }

    let conv = state.conversations.get(&id);
    let title = conv.map(|c| c.meta.title).unwrap_or_else(|| "New conversation".to_string());
    Json(ConversationCreatedResponse { id, title })
}

pub async fn list_conversations_handler(
    State(state): State<BancoState>,
) -> Json<ConversationsListResponse> {
    Json(ConversationsListResponse { conversations: state.conversations.list() })
}

pub async fn get_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<ConversationResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    state.conversations.get(&id).map(|c| Json(ConversationResponse { conversation: c })).ok_or((
        axum::http::StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
    ))
}

pub async fn delete_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<axum::http::StatusCode, (axum::http::StatusCode, Json<ErrorResponse>)> {
    state.conversations.delete(&id).map(|()| axum::http::StatusCode::NO_CONTENT).map_err(|_| {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
        )
    })
}

// ============================================================================
// BANCO-HDL-010: Prompt Presets
// ============================================================================

pub async fn list_prompts_handler(State(state): State<BancoState>) -> Json<PromptsListResponse> {
    Json(PromptsListResponse { presets: state.prompts.list() })
}

pub async fn get_prompt_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<super::prompts::PromptPreset>, (StatusCode, Json<ErrorResponse>)> {
    state.prompts.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Preset {id} not found"), "not_found", 404)),
    ))
}

pub async fn save_prompt_handler(
    State(state): State<BancoState>,
    Json(request): Json<SavePromptRequest>,
) -> Json<super::prompts::PromptPreset> {
    Json(state.prompts.create(&request.name, &request.content))
}

pub async fn delete_prompt_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    if state.prompts.delete(&id) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Preset {id} not found"), "not_found", 404)),
        ))
    }
}

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
