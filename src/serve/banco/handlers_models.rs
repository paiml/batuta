//! Model management handlers — load, unload, status.

use axum::{extract::State, http::StatusCode, response::Json};

use super::state::BancoState;
use super::types::{ErrorResponse, ModelLoadRequest, ModelStatusResponse};

/// POST /api/v1/models/load — Load a model from path or URI.
pub async fn model_load_handler(
    State(state): State<BancoState>,
    Json(request): Json<ModelLoadRequest>,
) -> Result<Json<ModelStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let info = state.model.load(&request.model).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(e.to_string(), "model_load_error", 500)),
        )
    })?;

    state.events.emit(&super::events::BancoEvent::ModelLoaded {
        model_id: info.model_id.clone(),
        format: format!("{:?}", info.format).to_lowercase(),
    });

    Ok(Json(ModelStatusResponse { loaded: true, model: Some(info), uptime_secs: Some(0) }))
}

/// POST /api/v1/models/unload — Unload the current model.
pub async fn model_unload_handler(
    State(state): State<BancoState>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state
        .model
        .unload()
        .map(|()| {
            state.events.emit(&super::events::BancoEvent::ModelUnloaded);
            StatusCode::NO_CONTENT
        })
        .map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(e.to_string(), "no_model", 400)))
        })
}

/// GET /api/v1/models/status — Current model status.
pub async fn model_status_handler(State(state): State<BancoState>) -> Json<ModelStatusResponse> {
    let info = state.model.info();
    let loaded = info.is_some();
    Json(ModelStatusResponse {
        loaded,
        model: info,
        uptime_secs: if loaded { Some(state.model.uptime_secs()) } else { None },
    })
}
