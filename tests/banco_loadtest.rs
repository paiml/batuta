//! Banco Load Test — probar LoadTest against real TCP server.
//!
//! Validates throughput and latency under concurrent load using probar's
//! LoadTest framework. Runs with 2 concurrent workers for 3 seconds to
//! establish baseline performance metrics.

#![cfg(feature = "banco")]

use std::time::Duration;

async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let state = batuta::serve::banco::state::BancoStateInner::with_defaults();
    let app = batuta::serve::banco::router::create_banco_router(state);
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    tokio::time::sleep(Duration::from_millis(200)).await;
    (base, handle)
}

#[tokio::test]
async fn load_test_chat_completions() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    assert!(client.health_check().await.unwrap(), "Server should be healthy");

    let prompts = vec![
        jugar_probar::llm::ChatRequest {
            model: "local".to_string(),
            messages: vec![jugar_probar::llm::ChatMessage {
                role: jugar_probar::llm::Role::User,
                content: "Hello".to_string(),
            }],
            temperature: None,
            max_tokens: Some(16),
            stream: None,
        },
        jugar_probar::llm::ChatRequest {
            model: "local".to_string(),
            messages: vec![jugar_probar::llm::ChatMessage {
                role: jugar_probar::llm::Role::User,
                content: "What is 2+2?".to_string(),
            }],
            temperature: None,
            max_tokens: Some(16),
            stream: None,
        },
    ];

    let config = jugar_probar::llm::LoadTestConfig {
        concurrency: 2,
        duration: Duration::from_secs(3),
        prompts,
        runtime_name: "banco-dry-run".to_string(),
        slo_latency_ms: Some(5000.0),
        ..Default::default()
    };

    let load_test = jugar_probar::llm::LoadTest::new(client, config);
    let result = load_test.run().await.unwrap();

    // Validate results
    assert!(result.total_requests > 0, "Should complete requests");
    assert!(result.successful > 0, "Should have successes");
    assert!(
        result.latency_p50_ms < 5000.0,
        "p50 latency should be under 5s: {:.0}ms",
        result.latency_p50_ms
    );

    eprintln!("[load] {} req in 3s, {:.1} RPS", result.total_requests, result.throughput_rps);
    eprintln!("[load] p50={:.0}ms p99={:.0}ms", result.latency_p50_ms, result.latency_p99_ms);

    handle.abort();
}
