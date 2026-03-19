//! Conversation persistence tests.

use crate::serve::templates::ChatMessage;

// ============================================================================
// ConversationStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONV_001_create_and_get() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("test-model");
    assert!(id.starts_with("conv-"));
    let conv = store.get(&id).expect("conversation should exist");
    assert_eq!(conv.meta.model, "test-model");
    assert_eq!(conv.meta.title, "New conversation");
    assert!(conv.messages.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_002_append_messages() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store.append(&id, ChatMessage::user("Hello")).expect("append");
    store.append(&id, ChatMessage::assistant("Hi!")).expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.messages.len(), 2);
    assert_eq!(conv.meta.message_count, 2);
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_003_auto_title() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store
        .append(&id, ChatMessage::user("What is the meaning of life and everything"))
        .expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.meta.title, "What is the meaning of...");
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_003b_auto_title_short() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store.append(&id, ChatMessage::user("Hello")).expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.meta.title, "Hello");
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_004_list_conversations() {
    let store = super::conversations::ConversationStore::in_memory();
    store.create("a");
    store.create("b");
    store.create("c");
    let list = store.list();
    assert_eq!(list.len(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_005_delete() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    assert_eq!(store.len(), 1);
    store.delete(&id).expect("delete");
    assert_eq!(store.len(), 0);
    assert!(store.get(&id).is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_006_delete_not_found() {
    let store = super::conversations::ConversationStore::in_memory();
    let result = store.delete("nonexistent");
    assert!(result.is_err());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_007_append_not_found() {
    let store = super::conversations::ConversationStore::in_memory();
    let result = store.append("nonexistent", ChatMessage::user("hi"));
    assert!(result.is_err());
}

// ============================================================================
// Conversation HTTP endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_001_create_conversation() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({});
    let response = app
        .oneshot(
            Request::post("/api/v1/conversations")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("conv-"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_002_list_conversations() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.conversations.create("model-a");
    state.conversations.create("model-b");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/conversations").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["conversations"].as_array().expect("array").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_003_get_conversation_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::get("/api/v1/conversations/nonexistent").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_004_chat_appends_to_conversation() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let conv_id = state.conversations.create("echo");

    let app = super::router::create_banco_router(state.clone());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "conversation_id": conv_id
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

    // Verify message was saved
    let conv = state.conversations.get(&conv_id).expect("conversation");
    assert_eq!(conv.messages.len(), 1);
    assert_eq!(conv.messages[0].content, "Hello!");
    assert_eq!(conv.meta.title, "Hello!");
}
