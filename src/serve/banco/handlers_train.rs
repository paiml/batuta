//! Training run endpoint handlers.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::state::BancoState;
use super::training::{TrainingConfig, TrainingMethod, TrainingRun, TrainingStatus};
use super::types::ErrorResponse;

/// POST /api/v1/train/start — start a training run.
pub async fn start_training_handler(
    State(state): State<BancoState>,
    Json(request): Json<StartTrainingRequest>,
) -> Json<TrainingRun> {
    let method = request.method.unwrap_or(TrainingMethod::Lora);
    let config = request.config.unwrap_or_default();
    let mut run = state.training.start(&request.dataset_id, method, config);

    // Without ml feature, mark as dry-run complete immediately
    #[cfg(not(feature = "ml"))]
    {
        run.status = TrainingStatus::Complete;
        run.metrics.push(super::training::TrainingMetric {
            step: 0,
            loss: 0.0,
            learning_rate: run.config.learning_rate,
        });
    }

    Json(run)
}

/// GET /api/v1/train/runs — list training runs.
pub async fn list_training_runs_handler(
    State(state): State<BancoState>,
) -> Json<TrainingRunsResponse> {
    Json(TrainingRunsResponse { runs: state.training.list() })
}

/// GET /api/v1/train/runs/:id — get run status.
pub async fn get_training_run_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<TrainingRun>, (StatusCode, Json<ErrorResponse>)> {
    state.training.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Run {id} not found"), "not_found", 404)),
    ))
}

/// POST /api/v1/train/runs/:id/stop — stop a running training.
pub async fn stop_training_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.training.stop(&id).map(|()| StatusCode::OK).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Run {id} not found"), "not_found", 404)),
        )
    })
}

/// DELETE /api/v1/train/runs/:id — delete a run.
pub async fn delete_training_run_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.training.delete(&id).map(|()| StatusCode::NO_CONTENT).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Run {id} not found"), "not_found", 404)),
        )
    })
}

#[derive(Debug, Deserialize)]
pub struct StartTrainingRequest {
    pub dataset_id: String,
    #[serde(default)]
    pub method: Option<TrainingMethod>,
    #[serde(default)]
    pub config: Option<TrainingConfig>,
}

#[derive(Debug, serde::Serialize)]
pub struct TrainingRunsResponse {
    pub runs: Vec<TrainingRun>,
}
