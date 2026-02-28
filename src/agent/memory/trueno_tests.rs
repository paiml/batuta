use super::*;

fn make_substrate() -> TruenoMemory {
    TruenoMemory::open_in_memory()
        .expect("in-memory open")
}

#[tokio::test]
async fn test_remember_and_recall() {
    let mem = make_substrate();
    let id = mem
        .remember(
            "agent1",
            "Rust is great for systems programming",
            MemorySource::User,
            None,
        )
        .await
        .expect("remember");
    assert!(id.starts_with("trueno-"));

    let results = mem
        .recall("Rust systems", 10, None, None)
        .await
        .expect("recall");
    assert_eq!(results.len(), 1);
    assert!(
        results[0].content.contains("systems programming")
    );
    assert!(results[0].relevance_score > 0.0);
}

#[tokio::test]
async fn test_recall_no_match() {
    let mem = make_substrate();
    mem.remember(
        "a",
        "SIMD vector operations",
        MemorySource::System,
        None,
    )
    .await
    .expect("remember");

    let results = mem
        .recall("cryptocurrency blockchain", 10, None, None)
        .await
        .expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_recall_empty_query() {
    let mem = make_substrate();
    mem.remember(
        "a",
        "some content",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");

    let results =
        mem.recall("", 10, None, None).await.expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_recall_limit() {
    let mem = make_substrate();
    for i in 0..10 {
        mem.remember(
            "a",
            &format!("memory about Rust topic {i}"),
            MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember");
    }

    let results = mem
        .recall("Rust topic", 3, None, None)
        .await
        .expect("recall");
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_recall_bm25_ranking() {
    let mem = make_substrate();
    mem.remember(
        "a",
        "machine learning algorithms",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    mem.remember(
        "a",
        "machine learning machine learning deep learning neural networks",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    mem.remember(
        "a",
        "cooking recipes for dinner",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");

    let results = mem
        .recall("machine learning", 10, None, None)
        .await
        .expect("recall");
    // Should find ML-related memories, not cooking
    assert!(results.len() >= 2);
    assert!(results
        .iter()
        .all(|r| !r.content.contains("cooking")));
}

#[tokio::test]
async fn test_filter_by_agent_id() {
    let mem = make_substrate();
    mem.remember(
        "agent1",
        "secret data about Rust",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    mem.remember(
        "agent2",
        "other data about Rust",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");

    let filter = MemoryFilter {
        agent_id: Some("agent1".into()),
        ..Default::default()
    };
    let results = mem
        .recall("Rust", 10, Some(filter), None)
        .await
        .expect("recall");
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("secret"));
}

#[tokio::test]
async fn test_kv_set_get() {
    let mem = make_substrate();
    mem.set("a", "counter", serde_json::json!(42))
        .await
        .expect("set");

    let val =
        mem.get("a", "counter").await.expect("get");
    assert_eq!(val, Some(serde_json::json!(42)));
}

#[tokio::test]
async fn test_kv_missing_key() {
    let mem = make_substrate();
    let val = mem
        .get("a", "nonexistent")
        .await
        .expect("get");
    assert!(val.is_none());
}

#[tokio::test]
async fn test_kv_overwrite() {
    let mem = make_substrate();
    mem.set("a", "key", serde_json::json!("old"))
        .await
        .expect("set");
    mem.set("a", "key", serde_json::json!("new"))
        .await
        .expect("set");

    let val = mem.get("a", "key").await.expect("get");
    assert_eq!(val, Some(serde_json::json!("new")));
}

#[tokio::test]
async fn test_kv_isolation() {
    let mem = make_substrate();
    mem.set("agent1", "key", serde_json::json!("one"))
        .await
        .expect("set");
    mem.set("agent2", "key", serde_json::json!("two"))
        .await
        .expect("set");

    let v1 =
        mem.get("agent1", "key").await.expect("get");
    let v2 =
        mem.get("agent2", "key").await.expect("get");
    assert_eq!(v1, Some(serde_json::json!("one")));
    assert_eq!(v2, Some(serde_json::json!("two")));
}

#[tokio::test]
async fn test_kv_complex_value() {
    let mem = make_substrate();
    let complex = serde_json::json!({
        "name": "test",
        "items": [1, 2, 3],
        "nested": {"a": true}
    });
    mem.set("a", "config", complex.clone())
        .await
        .expect("set");

    let val =
        mem.get("a", "config").await.expect("get");
    assert_eq!(val, Some(complex));
}

#[tokio::test]
async fn test_forget() {
    let mem = make_substrate();
    let id = mem
        .remember(
            "a",
            "forget me Rust",
            MemorySource::User,
            None,
        )
        .await
        .expect("remember");

    // Forget using the full doc_id pattern
    let doc_id = format!("a:{id}");
    mem.forget(doc_id).await.expect("forget");

    let results = mem
        .recall("forget Rust", 10, None, None)
        .await
        .expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_unique_ids() {
    let mem = make_substrate();
    let id1 = mem
        .remember("a", "one", MemorySource::User, None)
        .await
        .expect("remember");
    let id2 = mem
        .remember("a", "two", MemorySource::User, None)
        .await
        .expect("remember");
    assert_ne!(id1, id2);
}

#[tokio::test]
async fn test_fragment_count() {
    let mem = make_substrate();
    assert_eq!(
        mem.fragment_count().expect("count"),
        0
    );

    mem.remember(
        "a",
        "first memory",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    assert_eq!(
        mem.fragment_count().expect("count"),
        1
    );

    mem.remember(
        "a",
        "second memory",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    assert_eq!(
        mem.fragment_count().expect("count"),
        2
    );
}

#[tokio::test]
async fn test_relevance_normalized() {
    let mem = make_substrate();
    mem.remember(
        "a",
        "Rust programming language",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");

    let results = mem
        .recall("Rust", 10, None, None)
        .await
        .expect("recall");
    assert_eq!(results.len(), 1);
    // Score should be normalized to 0.0-1.0
    assert!(results[0].relevance_score >= 0.0);
    assert!(results[0].relevance_score <= 1.0);
}
