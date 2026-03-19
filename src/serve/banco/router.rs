//! Banco route wiring.

use axum::{
    middleware,
    routing::{get, post},
    Router,
};

use super::handlers::{
    chat_completions_handler, detokenize_handler, health_handler, models_handler, system_handler,
    tokenize_handler,
};
use super::middleware::privacy_layer;
use super::state::BancoState;

/// Build the full Banco router with all endpoints and middleware.
///
/// Mounts endpoints at both `/api/v1/` (Banco canonical) and `/v1/` (OpenAI SDK compat).
pub fn create_banco_router(state: BancoState) -> Router {
    let tier = state.privacy_tier;

    Router::new()
        .route("/health", get(health_handler))
        // Banco canonical paths
        .route("/api/v1/models", get(models_handler))
        .route("/api/v1/chat/completions", post(chat_completions_handler))
        .route("/api/v1/system", get(system_handler))
        .route("/api/v1/tokenize", post(tokenize_handler))
        .route("/api/v1/detokenize", post(detokenize_handler))
        // OpenAI SDK compat paths (/v1/ prefix)
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(middleware::from_fn(move |req, next| privacy_layer(tier, req, next)))
        .with_state(state)
}
