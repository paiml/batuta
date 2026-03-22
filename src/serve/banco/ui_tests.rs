//! Browser UI serving tests.

// ============================================================================
// UI unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_UI_001_index_html_contains_banco() {
    assert!(super::ui::INDEX_HTML.contains("Banco"));
    assert!(super::ui::INDEX_HTML.contains("<!DOCTYPE html>"));
}

#[test]
#[allow(non_snake_case)]
fn test_UI_002_index_html_has_chat_api() {
    assert!(super::ui::INDEX_HTML.contains("/api/v1/chat/completions"));
}

#[test]
#[allow(non_snake_case)]
fn test_UI_003_index_html_has_websocket() {
    assert!(super::ui::INDEX_HTML.contains("/api/v1/ws"));
}

#[test]
#[allow(non_snake_case)]
fn test_UI_004_index_html_has_system_info() {
    assert!(super::ui::INDEX_HTML.contains("/api/v1/system"));
}

// ============================================================================
// UI endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_001_index_returns_html() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response =
        app.oneshot(Request::get("/").body(Body::empty()).expect("req")).await.expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let ct = response.headers().get("content-type").expect("ct").to_str().expect("str");
    assert!(ct.contains("text/html"), "expected HTML content-type, got: {ct}");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("Banco"));
    // Zero-JS UI uses <form> instead of <script>
    assert!(html.contains("<form"), "Should have a form element");
    assert!(!html.contains("<script>"), "Zero-JS: no inline script tags");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_002_assets_returns_404() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/assets/app.js").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}
