//! Banco L2 Integration Tests — probar::llm against real TCP server.
//!
//! These tests start a real Banco HTTP server, use probar's LlmClient
//! to send actual OpenAI-compatible requests over TCP, and validate
//! responses with LlmAssertion builders.
//!
//! This is the test suite that would have caught "nothing works" in UAT.

#![cfg(feature = "banco")]

use std::time::Duration;

/// Start a Banco server on a random port. Returns (base_url, abort_handle).
async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");

    let state = batuta::serve::banco::state::BancoStateInner::with_defaults();
    let app = batuta::serve::banco::router::create_banco_router(state);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give server time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    (base, handle)
}

// ============================================================================
// L2: Chat completion tests via probar LlmClient
// ============================================================================

#[tokio::test]
async fn l2_health_via_llm_client() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let healthy = client.health_check().await.unwrap();
    assert!(healthy, "Server should be healthy");

    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_valid_structure() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "Hello Banco!".to_string(),
        }],
        temperature: None,
        max_tokens: Some(32),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();

    // Structural assertions via probar
    let assertions = jugar_probar::llm::LlmAssertion::new().assert_response_valid();

    assert!(
        assertions.run_all_pass(&timed),
        "Response structure invalid: {:?}",
        assertions.run(&timed)
    );

    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_has_content() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "What is Banco?".to_string(),
        }],
        temperature: None,
        max_tokens: Some(64),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();
    let content = &timed.response.choices[0].message.content;

    assert!(!content.is_empty(), "Response content is empty");
    // Without a loaded model, should mention "No model loaded" or similar
    // With a model, should have actual generated text
    assert!(content.len() > 5, "Response too short: {content}");

    handle.abort();
}

#[tokio::test]
async fn l2_chat_latency_under_budget() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "hi".to_string(),
        }],
        temperature: None,
        max_tokens: Some(8),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();

    let assertions =
        jugar_probar::llm::LlmAssertion::new().assert_latency_under(Duration::from_secs(10));

    assert!(
        assertions.run_all_pass(&timed),
        "Latency exceeded 10s: {:?}ms",
        timed.latency.as_millis()
    );

    handle.abort();
}

#[tokio::test]
async fn l2_chat_streaming_sse() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "Count to 3".to_string(),
        }],
        temperature: None,
        max_tokens: Some(16),
        stream: Some(true),
    };

    let streamed = client.chat_completion_stream(&request).await.unwrap();

    assert!(!streamed.content.is_empty(), "Stream produced no content");
    assert!(streamed.ttft < Duration::from_secs(10), "TTFT too slow: {:?}", streamed.ttft);

    handle.abort();
}

// ============================================================================
// L2: Non-chat endpoint tests via raw HTTP (probar LlmClient is chat-focused)
// ============================================================================

#[tokio::test]
async fn l2_system_info_returns_json() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/system")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["endpoints"].as_u64().unwrap() > 0);
    assert_eq!(json["telemetry"], false);
    handle.abort();
}

#[tokio::test]
async fn l2_browser_ui_serves_html() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.contains("text/html"));
    let body = resp.text().await.unwrap();
    assert!(body.contains("Banco"));
    assert!(body.contains("/api/v1/chat/completions"));
    handle.abort();
}

#[tokio::test]
async fn l2_tool_calculator() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/tools/execute"))
        .json(&serde_json::json!({
            "id": "l2-test",
            "name": "calculator",
            "arguments": {"expression": "6 * 7"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["content"], "42");
    handle.abort();
}

#[tokio::test]
async fn l2_mcp_initialize() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/mcp"))
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["result"]["serverInfo"]["name"], "banco");
    handle.abort();
}

#[tokio::test]
async fn l2_prometheus_metrics() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("banco_requests_total"));
    assert!(body.contains("banco_uptime_seconds"));
    handle.abort();
}

#[tokio::test]
async fn l2_health_probes() {
    let (base, handle) = start_server().await;
    let live = reqwest::get(format!("{base}/health/live")).await.unwrap();
    assert_eq!(live.status(), 200);
    let ready = reqwest::get(format!("{base}/health/ready")).await.unwrap();
    assert_eq!(ready.status(), 503); // no model loaded
    handle.abort();
}

#[tokio::test]
async fn l2_upload_and_rag_search() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Upload
    let resp = client
        .post(format!("{base}/api/v1/data/upload/json"))
        .json(&serde_json::json!({
            "name": "test.txt",
            "content": "Banco is a sovereign AI workbench"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Search
    let resp =
        reqwest::get(format!("{base}/api/v1/rag/search?q=sovereign+workbench")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(!json["results"].as_array().unwrap().is_empty());

    handle.abort();
}

#[tokio::test]
async fn l2_ollama_tags() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/tags")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["models"].is_array());
    handle.abort();
}

// ============================================================================
// L2: Tokenize, conversations, config, audit endpoints
// ============================================================================

#[tokio::test]
async fn l2_tokenize_returns_tokens() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/tokenize"))
        .json(&serde_json::json!({"text": "Hello world"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["count"].as_u64().unwrap() > 0);
    assert!(json["tokens"].is_array());
    handle.abort();
}

#[tokio::test]
async fn l2_detokenize_returns_text() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/detokenize"))
        .json(&serde_json::json!({"tokens": [1, 2, 3]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["text"].is_string());
    handle.abort();
}

#[tokio::test]
async fn l2_conversations_crud() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // List (initially empty)
    let resp = reqwest::get(format!("{base}/api/v1/conversations")).await.unwrap();
    assert_eq!(resp.status(), 200);

    // Create via chat (conversations are created implicitly)
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Test conversation"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_config_get_put() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // GET config
    let resp = reqwest::get(format!("{base}/api/v1/config")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.is_object());

    // PUT config
    let resp = client
        .put(format!("{base}/api/v1/config"))
        .json(&serde_json::json!({"theme": "dark"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_audit_log() {
    let (base, handle) = start_server().await;

    // Make a request first (to generate audit entry)
    reqwest::get(format!("{base}/api/v1/system")).await.unwrap();

    // Check audit log
    let resp = reqwest::get(format!("{base}/api/v1/audit")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["entries"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_embeddings_endpoint() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "local",
            "input": "Test embedding"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["data"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_models_status_no_model() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/api/v1/models/status")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["loaded"], false);
    // No tokenizer field when no model loaded
    assert!(json["tokenizer"].is_null());

    handle.abort();
}

#[tokio::test]
async fn l2_openai_models_list() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_batch_completions() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/v1/batch"))
        .json(&serde_json::json!({
            "items": [
                {"id": "b1", "messages": [{"role": "user", "content": "Hi"}]},
                {"id": "b2", "messages": [{"role": "user", "content": "Hello"}]}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["results"].is_array());
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);

    handle.abort();
}
