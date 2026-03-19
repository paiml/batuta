//! Banco route wiring.

use axum::{
    middleware,
    routing::{get, post},
    Router,
};

use super::audit::{audit_layer, AuditLog};
use super::handlers::{
    chat_completions_handler, detokenize_handler, embeddings_handler, health_handler,
    models_handler, system_handler, tokenize_handler,
};
use super::middleware::privacy_layer;
use super::state::BancoState;

/// Build the full Banco router with all endpoints and middleware.
///
/// Mounts endpoints at both `/api/v1/` (Banco canonical) and `/v1/` (OpenAI SDK compat).
pub fn create_banco_router(state: BancoState) -> Router {
    create_banco_router_with_audit(state, AuditLog::new())
}

/// Build router with an explicit audit log (for testing).
pub fn create_banco_router_with_audit(state: BancoState, audit_log: AuditLog) -> Router {
    let tier = state.privacy_tier;
    let log = audit_log.clone();

    Router::new()
        .route("/health", get(health_handler))
        // Banco canonical paths
        .route("/api/v1/models", get(models_handler))
        .route("/api/v1/chat/completions", post(chat_completions_handler))
        .route("/api/v1/system", get(system_handler))
        .route("/api/v1/tokenize", post(tokenize_handler))
        .route("/api/v1/detokenize", post(detokenize_handler))
        .route("/api/v1/embeddings", post(embeddings_handler))
        // OpenAI SDK compat paths (/v1/ prefix)
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        // Middleware: audit logging (outermost, runs first)
        .layer(middleware::from_fn(move |req, next| audit_layer(log.clone(), req, next)))
        // Middleware: privacy tier header + sovereign gate
        .layer(middleware::from_fn(move |req, next| privacy_layer(tier, req, next)))
        .with_state(state)
}
