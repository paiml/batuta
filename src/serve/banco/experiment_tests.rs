//! Experiment tracking tests.

use super::training::{TrainingConfig, TrainingMethod};

// ============================================================================
// ExperimentStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_EXP_001_create() {
    let store = super::experiment::ExperimentStore::new();
    let exp = store.create("test-exp", "Testing LoRA vs QLoRA");
    assert!(exp.id.starts_with("exp-"));
    assert_eq!(exp.name, "test-exp");
    assert!(exp.run_ids.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_EXP_002_add_run() {
    let store = super::experiment::ExperimentStore::new();
    let exp = store.create("exp", "desc");
    store.add_run(&exp.id, "run-1").expect("add");
    store.add_run(&exp.id, "run-2").expect("add");
    let updated = store.get(&exp.id).expect("get");
    assert_eq!(updated.run_ids.len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_EXP_003_dedup_run() {
    let store = super::experiment::ExperimentStore::new();
    let exp = store.create("exp", "");
    store.add_run(&exp.id, "run-1").expect("add");
    store.add_run(&exp.id, "run-1").expect("add again");
    let updated = store.get(&exp.id).expect("get");
    assert_eq!(updated.run_ids.len(), 1, "should not duplicate");
}

#[test]
#[allow(non_snake_case)]
fn test_EXP_004_compare() {
    let exp_store = super::experiment::ExperimentStore::new();
    let train_store = super::training::TrainingStore::new();

    let run1 = train_store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    let run2 = train_store.start("ds-1", TrainingMethod::Qlora, TrainingConfig::default());

    let exp = exp_store.create("compare", "LoRA vs QLoRA");
    exp_store.add_run(&exp.id, &run1.id).expect("add");
    exp_store.add_run(&exp.id, &run2.id).expect("add");

    let comparison = exp_store.compare(&exp.id, &train_store).expect("compare");
    assert_eq!(comparison.runs.len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_EXP_005_list() {
    let store = super::experiment::ExperimentStore::new();
    store.create("a", "");
    store.create("b", "");
    assert_eq!(store.list().len(), 2);
}

// ============================================================================
// Experiment endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_EXP_HDL_001_create() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "test-exp", "description": "comparing methods"});
    let response = app
        .oneshot(
            Request::post("/api/v1/experiments")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("exp-"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_EXP_HDL_002_compare() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();

    // Create runs
    let run1 = state.training.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    let run2 = state.training.start("ds-1", TrainingMethod::Qlora, TrainingConfig::default());

    // Create experiment and add runs
    let exp = state.experiments.create("compare", "test");
    state.experiments.add_run(&exp.id, &run1.id).expect("add");
    state.experiments.add_run(&exp.id, &run2.id).expect("add");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get(&format!("/api/v1/experiments/{}/compare", exp.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["runs"].as_array().expect("runs").len(), 2);
}
