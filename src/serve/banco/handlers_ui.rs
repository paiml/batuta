//! Browser UI handlers — serve the embedded SPA.

use axum::{
    http::{header, StatusCode},
    response::{Html, IntoResponse},
};

/// GET / — serve the SPA index page.
pub async fn index_handler() -> Html<&'static str> {
    Html(super::ui::INDEX_HTML)
}

/// GET /assets/* — serve static assets (CSS/JS/WASM).
/// Currently returns 404 — assets are inlined in the HTML.
/// This route exists as a scaffold for presentar WASM bundles.
pub async fn assets_handler(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    // Future: serve presentar WASM bundles here
    let _ = path;
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/plain")],
        "Asset not found — UI is self-contained in index.html",
    )
}
