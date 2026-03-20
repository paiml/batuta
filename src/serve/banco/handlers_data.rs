//! Data management handlers — file upload, list, delete.

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
};

use super::state::BancoState;
use super::storage::FileInfo;
use super::types::ErrorResponse;

/// POST /api/v1/data/upload — upload files via multipart form.
pub async fn upload_handler(
    State(state): State<BancoState>,
    mut multipart: Multipart,
) -> Result<Json<Vec<FileInfo>>, (StatusCode, Json<ErrorResponse>)> {
    let mut uploaded = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.file_name().unwrap_or("unnamed").to_string();
        let data = field.bytes().await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("Failed to read field: {e}"), "upload_error", 400)),
            )
        })?;

        if data.is_empty() {
            continue;
        }

        let info = state.files.store(&name, &data);
        uploaded.push(info);
    }

    if uploaded.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("No files uploaded", "no_files", 400)),
        ));
    }

    Ok(Json(uploaded))
}

/// POST /api/v1/data/upload/json — upload a file via JSON body (simpler for testing).
pub async fn upload_json_handler(
    State(state): State<BancoState>,
    Json(request): Json<UploadJsonRequest>,
) -> Json<FileInfo> {
    let info = state.files.store(&request.name, request.content.as_bytes());
    Json(info)
}

/// JSON upload request (alternative to multipart).
#[derive(Debug, serde::Deserialize)]
pub struct UploadJsonRequest {
    pub name: String,
    pub content: String,
}

/// GET /api/v1/data/files — list all uploaded files.
pub async fn list_files_handler(State(state): State<BancoState>) -> Json<FilesListResponse> {
    Json(FilesListResponse { files: state.files.list() })
}

/// DELETE /api/v1/data/files/:id — delete an uploaded file.
pub async fn delete_file_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.files.delete(&id).map(|()| StatusCode::NO_CONTENT).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("File {id} not found"), "not_found", 404)),
        )
    })
}

/// File list response.
#[derive(Debug, serde::Serialize)]
pub struct FilesListResponse {
    pub files: Vec<FileInfo>,
}
