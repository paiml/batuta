//! Storage and data endpoint tests.

// ============================================================================
// FileStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_STOR_001_store_and_get() {
    let store = super::storage::FileStore::in_memory();
    let info = store.store("test.txt", b"Hello, world!");
    assert!(info.id.starts_with("file-"));
    assert_eq!(info.name, "test.txt");
    assert_eq!(info.size_bytes, 13);
    assert_eq!(info.content_type, "text/plain");
    assert!(!info.content_hash.is_empty());

    let retrieved = store.get(&info.id).expect("should exist");
    assert_eq!(retrieved.name, "test.txt");
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_002_list_files() {
    let store = super::storage::FileStore::in_memory();
    store.store("a.csv", b"a,b,c");
    store.store("b.json", b"{}");
    store.store("c.pdf", b"%PDF");
    let files = store.list();
    assert_eq!(files.len(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_003_delete() {
    let store = super::storage::FileStore::in_memory();
    let info = store.store("test.txt", b"data");
    assert_eq!(store.len(), 1);
    store.delete(&info.id).expect("delete");
    assert_eq!(store.len(), 0);
    assert!(store.get(&info.id).is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_004_delete_not_found() {
    let store = super::storage::FileStore::in_memory();
    let result = store.delete("nonexistent");
    assert!(result.is_err());
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_005_content_type_detection() {
    let store = super::storage::FileStore::in_memory();
    assert_eq!(store.store("a.pdf", b"x").content_type, "application/pdf");
    assert_eq!(store.store("b.csv", b"x").content_type, "text/csv");
    assert_eq!(store.store("c.json", b"x").content_type, "application/json");
    assert_eq!(store.store("d.jsonl", b"x").content_type, "application/jsonl");
    assert_eq!(store.store("e.txt", b"x").content_type, "text/plain");
    assert_eq!(store.store("f.bin", b"x").content_type, "application/octet-stream");
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_006_content_hash_deterministic() {
    let store = super::storage::FileStore::in_memory();
    let a = store.store("a.txt", b"same content");
    let b = store.store("b.txt", b"same content");
    assert_eq!(a.content_hash, b.content_hash);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007_different_content_different_hash() {
    let store = super::storage::FileStore::in_memory();
    let a = store.store("a.txt", b"content A");
    let b = store.store("b.txt", b"content B");
    assert_ne!(a.content_hash, b.content_hash);
}

// ============================================================================
// Data endpoint tests (via router oneshot)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_001_upload_json() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "test.csv", "content": "a,b,c\n1,2,3"});
    let response = app
        .oneshot(
            Request::post("/api/v1/data/upload/json")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("file-"));
    assert_eq!(json["name"], "test.csv");
    assert_eq!(json["content_type"], "text/csv");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_002_list_files() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.files.store("a.txt", b"hello");
    state.files.store("b.csv", b"1,2,3");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/data/files").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["files"].as_array().expect("files").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_003_delete_file() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let info = state.files.store("test.txt", b"data");

    let app = super::router::create_banco_router(state.clone());
    let response = app
        .oneshot(
            Request::delete(&format!("/api/v1/data/files/{}", info.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
    assert!(state.files.is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_004_delete_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::delete("/api/v1/data/files/nonexistent").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}
