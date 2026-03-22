//! Banco L2 Coverage Gap Tests — closing falsification-identified gaps.
//!
//! Fixes: #54 (experiment compare, audio), #55 (file ops, RAG index, prompts),
//! #56 (Ollama show/pull/delete, tool config, batch by ID).

#![cfg(feature = "banco")]

use std::time::Duration;

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

// ============================================================================
// #54: Experiment compare workflow
// ============================================================================

#[tokio::test]
async fn l2_experiment_add_runs_and_compare() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create experiment
    let resp = client
        .post(format!("{base}/api/v1/experiments"))
        .json(&serde_json::json!({"name": "compare-test", "description": "L2 test"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let exp: serde_json::Value = resp.json().await.unwrap();
    let exp_id = exp["id"].as_str().unwrap().to_string();

    // Start 2 training runs
    let resp = client
        .post(format!("{base}/api/v1/train/start"))
        .json(&serde_json::json!({"dataset_id": "d1", "preset": "quick-lora"}))
        .send()
        .await
        .unwrap();
    let run1: serde_json::Value = resp.json().await.unwrap();
    let run1_id = run1["id"].as_str().unwrap().to_string();

    let resp = client
        .post(format!("{base}/api/v1/train/start"))
        .json(&serde_json::json!({"dataset_id": "d2", "preset": "standard-lora"}))
        .send()
        .await
        .unwrap();
    let run2: serde_json::Value = resp.json().await.unwrap();
    let run2_id = run2["id"].as_str().unwrap().to_string();

    // Add runs to experiment
    let resp = client
        .post(format!("{base}/api/v1/experiments/{exp_id}/runs"))
        .json(&serde_json::json!({"run_id": run1_id}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client
        .post(format!("{base}/api/v1/experiments/{exp_id}/runs"))
        .json(&serde_json::json!({"run_id": run2_id}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Compare
    let resp = reqwest::get(format!("{base}/api/v1/experiments/{exp_id}/compare")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["runs"].is_array() || json["comparison"].is_object());

    handle.abort();
}

#[tokio::test]
async fn l2_audio_transcription_no_file() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // POST audio without actual file — should return structured response
    let resp = client
        .post(format!("{base}/api/v1/audio/transcriptions"))
        .json(&serde_json::json!({"file": "nonexistent.wav", "model": "whisper-1"}))
        .send()
        .await
        .unwrap();
    // Expect 200 with dry-run or 422/400 with error
    assert!(resp.status().is_success() || resp.status().is_client_error());

    handle.abort();
}

// ============================================================================
// #55: File operations, RAG index, prompts by ID
// ============================================================================

#[tokio::test]
async fn l2_file_upload_info_download() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Upload
    let resp = client
        .post(format!("{base}/api/v1/data/upload/json"))
        .json(&serde_json::json!({"name": "test.txt", "content": "File content for download test"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let file: serde_json::Value = resp.json().await.unwrap();
    let fid = file["id"].as_str().unwrap();

    // Get file info
    let resp = reqwest::get(format!("{base}/api/v1/data/files/{fid}/info")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let info: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(info["name"], "test.txt");

    // Delete file (returns 204 No Content)
    let resp = client.delete(format!("{base}/api/v1/data/files/{fid}")).send().await.unwrap();
    assert_eq!(resp.status(), 204);

    handle.abort();
}

#[tokio::test]
async fn l2_rag_index_trigger() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client.post(format!("{base}/api/v1/rag/index")).send().await.unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_prompt_by_id_and_delete() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create prompt
    let resp = client
        .post(format!("{base}/api/v1/prompts"))
        .json(&serde_json::json!({"name": "test-p", "content": "Be helpful."}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let prompt: serde_json::Value = resp.json().await.unwrap();
    let pid = prompt["id"].as_str().unwrap().to_string();

    // Get by ID
    let resp = reqwest::get(format!("{base}/api/v1/prompts/{pid}")).await.unwrap();
    assert_eq!(resp.status(), 200);

    // Delete (returns 204 No Content)
    let resp = client.delete(format!("{base}/api/v1/prompts/{pid}")).send().await.unwrap();
    assert_eq!(resp.status(), 204);

    handle.abort();
}

// ============================================================================
// #56: Ollama show/pull/delete, tool config, batch by ID
// ============================================================================

#[tokio::test]
async fn l2_ollama_show() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/show"))
        .json(&serde_json::json!({"name": "local"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["modelfile"].is_string());
    handle.abort();
}

#[tokio::test]
async fn l2_ollama_pull() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/pull"))
        .json(&serde_json::json!({"name": "llama3:8b"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["status"], "success");
    handle.abort();
}

#[tokio::test]
async fn l2_ollama_delete() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .delete(format!("{base}/api/delete"))
        .json(&serde_json::json!({"name": "local"}))
        .send()
        .await
        .unwrap();
    // Delete returns 200 (acknowledged) even if model doesn't exist
    assert!(resp.status().is_success());
    handle.abort();
}

#[tokio::test]
async fn l2_tool_config() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .put(format!("{base}/api/v1/tools/calculator/config"))
        .json(&serde_json::json!({"enabled": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    handle.abort();
}

#[tokio::test]
async fn l2_batch_get_by_id() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Submit batch first
    let resp = client
        .post(format!("{base}/api/v1/batch"))
        .json(&serde_json::json!({
            "items": [{"id": "x", "messages": [{"role": "user", "content": "Hi"}]}]
        }))
        .send()
        .await
        .unwrap();
    let batch: serde_json::Value = resp.json().await.unwrap();
    let batch_id = batch["batch_id"].as_str().unwrap();

    // Get by ID
    let resp = reqwest::get(format!("{base}/api/v1/batch/{batch_id}")).await.unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_eval_run_by_id() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create eval run
    let resp = client
        .post(format!("{base}/api/v1/eval/perplexity"))
        .json(&serde_json::json!({"text": "Test text for eval"}))
        .send()
        .await
        .unwrap();
    let eval: serde_json::Value = resp.json().await.unwrap();
    let eval_id = eval["eval_id"].as_str().unwrap();

    // Get by ID
    let resp = reqwest::get(format!("{base}/api/v1/eval/runs/{eval_id}")).await.unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}
