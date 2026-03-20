//! P2 cross-cutting tests: system prompt presets + Ollama API compat.

// ============================================================================
// System Prompt Presets
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_defaults() {
    let store = super::prompts::PromptStore::new();
    let presets = store.list();
    assert!(presets.len() >= 3, "should have built-in presets");
    assert!(store.get("coding").is_some());
    assert!(store.get("concise").is_some());
    assert!(store.get("tutor").is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_create() {
    let store = super::prompts::PromptStore::new();
    let preset = store.create("My Custom", "You are a pirate. Arr!");
    assert!(preset.id.starts_with("preset-"));
    assert_eq!(preset.name, "My Custom");
    assert_eq!(preset.content, "You are a pirate. Arr!");
    assert!(store.get(&preset.id).is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_delete() {
    let store = super::prompts::PromptStore::new();
    assert!(store.delete("coding"));
    assert!(store.get("coding").is_none());
    assert!(!store.delete("nonexistent"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_preset_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("@preset:coding");
    assert!(expanded.contains("expert software engineer"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_no_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("Just a normal message");
    assert_eq!(expanded, "Just a normal message");
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_unknown_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("@preset:nonexistent");
    assert_eq!(expanded, "@preset:nonexistent");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_prompts_list_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/prompts").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let presets = json["presets"].as_array().expect("presets array");
    assert!(presets.len() >= 3);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_prompts_create_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "Test", "content": "Be helpful"});
    let response = app
        .oneshot(
            Request::post("/api/v1/prompts")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("preset-"));
    assert_eq!(json["name"], "Test");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_chat_expands_preset_ref() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [
            {"role": "system", "content": "@preset:coding"},
            {"role": "user", "content": "Hello"}
        ]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    // The formatted output should contain the expanded preset content
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    // prompt_len should reflect expanded content (> 2 tokens for "Hello" alone)
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens");
    assert!(prompt_tokens > 5, "preset should expand, increasing token count");
}

// ============================================================================
// API Key Auth
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P2_auth_store_local_mode() {
    let store = super::auth::AuthStore::local();
    assert!(!store.requires_auth());
    assert!(store.validate("anything")); // local mode accepts all
    assert_eq!(store.mode(), super::auth::AuthMode::Local);
}

#[test]
#[allow(non_snake_case)]
fn test_P2_auth_store_api_key_mode() {
    let (store, key) = super::auth::AuthStore::api_key_mode();
    assert!(store.requires_auth());
    assert!(key.starts_with("bk_"));
    assert!(store.validate(&key));
    assert!(!store.validate("wrong_key"));
    assert_eq!(store.key_count(), 1);
}

#[test]
#[allow(non_snake_case)]
fn test_P2_key_scope_chat() {
    let scope = super::auth::KeyScope::Chat;
    assert!(scope.allows_path("/api/v1/chat/completions"));
    assert!(scope.allows_path("/health"));
    assert!(scope.allows_path("/api/v1/models"));
    assert!(scope.allows_path("/api/v1/embeddings"));
    assert!(scope.allows_path("/api/v1/prompts"));
    assert!(!scope.allows_path("/api/v1/train/start"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_key_scope_admin() {
    let scope = super::auth::KeyScope::Admin;
    assert!(scope.allows_path("/anything"));
    assert!(scope.allows_path("/api/v1/models/load"));
    assert!(scope.allows_path("/api/v1/config"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_auth_local_mode_no_header_needed() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    // Default state uses local mode — no auth needed
    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Ollama API Compat
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_tags() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/tags").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["models"].as_array().expect("models").len() > 0);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_chat() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["done"], true);
    assert_eq!(json["message"]["role"], "assistant");
    assert!(json["message"]["content"].as_str().expect("content").contains("banco"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_show() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "llama3"});
    let response = app
        .oneshot(
            Request::post("/api/show")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["modelfile"].as_str().expect("modelfile").contains("llama3"));
}
