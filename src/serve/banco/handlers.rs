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
    DetokenizeRequest, DetokenizeResponse, ErrorResponse, TokenizeRequest, TokenizeResponse, Usage,
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

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
