//! Banco L2 Integration Tests — chat completion validation over real TCP.
//!
//! Validates OpenAI-compatible chat completion API using raw HTTP via reqwest.

#![cfg(feature = "banco")]

use std::time::{Duration, Instant};

async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let state = batuta::serve::banco::state::BancoStateInner::with_defaults();
    let app = batuta::serve::banco::router::create_banco_router(state);
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    tokio::time::sleep(Duration::from_millis(100)).await;
    (base, handle)
}

#[tokio::test]
async fn l2_health_check() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_valid_structure() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "Hello Banco!"}],
            "max_tokens": 32
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["id"].is_string(), "Should have id");
    assert!(json["choices"].is_array(), "Should have choices");
    assert!(!json["choices"].as_array().unwrap().is_empty());
    assert!(json["choices"][0]["message"]["content"].is_string());
    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_has_content() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "What is Banco?"}],
            "max_tokens": 64
        }))
        .send()
        .await
        .unwrap();
    let json: serde_json::Value = resp.json().await.unwrap();
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty());
    assert!(content.len() > 5, "Response too short: {content}");
    handle.abort();
}

#[tokio::test]
async fn l2_chat_latency_under_budget() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let start = Instant::now();
    let _resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8
        }))
        .send()
        .await
        .unwrap();
    assert!(start.elapsed() < Duration::from_secs(10), "Latency exceeded 10s");
    handle.abort();
}

#[tokio::test]
async fn l2_chat_streaming_sse() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "max_tokens": 16,
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers().get("content-type").map(|v| v.to_str().unwrap_or(""));
    assert!(ct.unwrap_or("").contains("text/event-stream"));
    let body = resp.text().await.unwrap();
    assert!(body.contains("data:"), "SSE body should contain data: lines");
    handle.abort();
}
