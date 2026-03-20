//! RAG endpoint handlers — index management and status.

use axum::{extract::State, http::StatusCode, response::Json};

use super::rag::RagStatus;
use super::state::BancoState;

/// POST /api/v1/rag/index — force re-index all uploaded documents.
pub async fn rag_index_handler(State(state): State<BancoState>) -> Json<RagIndexResponse> {
    let files = state.files.list();
    let mut indexed = 0;

    for file_info in &files {
        if let Some(content) = state.files.read_content(&file_info.id) {
            let text = String::from_utf8_lossy(&content);
            state.rag.index_document(&file_info.id, &file_info.name, &text);
            indexed += 1;
        }
    }

    Json(RagIndexResponse { indexed_files: indexed, status: state.rag.status() })
}

/// GET /api/v1/rag/status — index stats.
pub async fn rag_status_handler(State(state): State<BancoState>) -> Json<RagStatus> {
    Json(state.rag.status())
}

/// DELETE /api/v1/rag/index — clear the RAG index.
pub async fn rag_clear_handler(State(state): State<BancoState>) -> StatusCode {
    state.rag.clear();
    StatusCode::NO_CONTENT
}

/// Response from index rebuild.
#[derive(Debug, serde::Serialize)]
pub struct RagIndexResponse {
    pub indexed_files: usize,
    pub status: RagStatus,
}
