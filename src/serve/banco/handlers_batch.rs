//! Batch inference endpoint handlers.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::batch::{BatchItem, BatchItemResult, BatchJob};
use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/batch — submit a batch of prompts for processing.
pub async fn submit_batch_handler(
    State(state): State<BancoState>,
    Json(request): Json<BatchRequest>,
) -> Json<BatchJob> {
    let job = state.batches.run(request.items, |item| {
        // Process each item through the template engine (dry-run without model)
        let formatted = state.template_engine.apply(&item.messages);
        let content = format!(
            "[batch dry-run] id={} | prompt_len={} | formatted_len={}",
            item.id,
            item.messages.len(),
            formatted.len()
        );
        let tokens = (content.len() / 4) as u32;
        BatchItemResult {
            id: item.id.clone(),
            content,
            finish_reason: "dry_run".to_string(),
            tokens,
        }
    });
    Json(job)
}

/// GET /api/v1/batch/:id — get batch job status and results.
pub async fn get_batch_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchJob>, (StatusCode, Json<ErrorResponse>)> {
    state.batches.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Batch {id} not found"), "not_found", 404)),
    ))
}

/// GET /api/v1/batch — list all batch jobs.
pub async fn list_batches_handler(State(state): State<BancoState>) -> Json<BatchListResponse> {
    Json(BatchListResponse { batches: state.batches.list() })
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub items: Vec<BatchItem>,
}

#[derive(Debug, serde::Serialize)]
pub struct BatchListResponse {
    pub batches: Vec<BatchJob>,
}
