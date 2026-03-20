//! Conversation CRUD handlers.

use axum::{extract::State, http::StatusCode, response::Json};

use super::state::BancoState;
use super::types::{
    ConversationCreatedResponse, ConversationResponse, ConversationsListResponse,
    CreateConversationRequest, ErrorResponse,
};

pub async fn create_conversation_handler(
    State(state): State<BancoState>,
    Json(request): Json<CreateConversationRequest>,
) -> Json<ConversationCreatedResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());
    let id = state.conversations.create(&model);

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
) -> Result<Json<ConversationResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.conversations.get(&id).map(|c| Json(ConversationResponse { conversation: c })).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
    ))
}

pub async fn delete_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.conversations.delete(&id).map(|()| StatusCode::NO_CONTENT).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
        )
    })
}
