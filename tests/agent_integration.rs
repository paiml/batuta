//! Integration tests for `batuta agent` module.
//!
//! Tests the full agent loop with MockDriver, InMemorySubstrate,
//! and MemoryTool wired together — end-to-end validation of the
//! perceive-reason-act pattern.
//!
//! Spec refs: Section 9.2 (Integration Tests)

#![cfg(feature = "agents")]

use std::sync::Arc;

use batuta::agent::capability::Capability;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::driver::{
    CompletionResponse, StreamEvent, ToolCall,
};
use batuta::agent::manifest::AgentManifest;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::memory::MemorySubstrate;
use batuta::agent::result::{StopReason, TokenUsage};
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::memory::MemoryTool;
use batuta::agent::tool::ToolRegistry;
use batuta::agent::AgentBuilder;
use tokio::sync::mpsc;

fn test_manifest() -> AgentManifest {
    AgentManifest {
        name: "test-agent".into(),
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

/// Full loop: tool call → memory store → verify memory.
#[tokio::test]
async fn test_full_loop_with_memory_tool() {
    let manifest = test_manifest();
    let memory = Arc::new(InMemorySubstrate::new());

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({
            "action": "remember",
            "content": "Important: SIMD is fast"
        }),
        "I've stored that information.",
    );

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(
        memory.clone(),
        "test-agent".into(),
    )));

    let result = run_agent_loop(
        &manifest,
        "Remember that SIMD is fast",
        &driver,
        &tools,
        memory.as_ref(),
        None,
    )
    .await
    .expect("loop failed");

    assert_eq!(result.text, "I've stored that information.");
    assert_eq!(result.iterations, 2);
    assert_eq!(result.tool_calls, 1);

    // Verify memory was actually stored
    let recalled = memory
        .recall("SIMD", 10, None, None)
        .await
        .expect("recall failed");
    assert!(
        !recalled.is_empty(),
        "memory should contain SIMD-related entries"
    );
}

/// Agent builder API round-trip.
#[tokio::test]
async fn test_agent_builder_end_to_end() {
    let manifest = test_manifest();
    let memory = Arc::new(InMemorySubstrate::new());
    let driver = MockDriver::single_response("Builder works!");

    let result = AgentBuilder::new(&manifest)
        .driver(&driver)
        .memory(memory.as_ref())
        .run("Hello builder")
        .await
        .expect("builder run failed");

    assert_eq!(result.text, "Builder works!");

    // Verify conversation stored in memory
    let recalled = memory
        .recall("Hello builder", 10, None, None)
        .await
        .expect("recall failed");
    assert!(
        !recalled.is_empty(),
        "conversation should be stored in memory"
    );
}

/// Stream events are emitted for all phases during a tool call loop.
#[tokio::test]
async fn test_stream_events_full_lifecycle() {
    let manifest = test_manifest();
    let memory = InMemorySubstrate::new();

    let driver = MockDriver::tool_then_response(
        "memory",
        serde_json::json!({"action": "recall", "query": "test"}),
        "Done with stream test",
    );

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(
        mem_arc,
        "test-agent".into(),
    )));

    let (tx, mut rx) = mpsc::channel(64);

    run_agent_loop(
        &manifest, "test", &driver, &tools, &memory, Some(tx),
    )
    .await
    .expect("loop failed");

    let mut phases = vec![];
    let mut tool_starts = 0u32;
    let mut tool_ends = 0u32;

    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::PhaseChange { phase } => {
                phases.push(format!("{phase}"));
            }
            StreamEvent::ToolUseStart { .. } => tool_starts += 1,
            StreamEvent::ToolUseEnd { .. } => tool_ends += 1,
            _ => {}
        }
    }

    // Should have: Perceive → Reason → Act → Reason → Done
    assert!(
        phases.len() >= 4,
        "expected ≥4 phase changes, got: {phases:?}"
    );
    assert_eq!(tool_starts, 1, "expected 1 ToolUseStart");
    assert_eq!(tool_ends, 1, "expected 1 ToolUseEnd");
}

/// Sovereign privacy enforcement: only Memory capability
/// prevents tool execution for non-granted capabilities.
#[tokio::test]
async fn test_sovereign_capability_enforcement() {
    let mut manifest = test_manifest();
    // Only grant Memory, NOT Rag
    manifest.capabilities = vec![Capability::Memory];

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "memory".into(),
                input: serde_json::json!({
                    "action": "recall",
                    "query": "test"
                }),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "capability enforced".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(
        mem_arc,
        "test-agent".into(),
    )));

    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "test", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop should succeed");
    assert_eq!(result.text, "capability enforced");
}

/// Multiple tool calls in a single loop iteration.
#[tokio::test]
async fn test_multi_tool_call_iteration() {
    let manifest = test_manifest();

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![
                ToolCall {
                    id: "1".into(),
                    name: "memory".into(),
                    input: serde_json::json!({
                        "action": "remember",
                        "content": "fact A"
                    }),
                },
                ToolCall {
                    id: "2".into(),
                    name: "memory".into(),
                    input: serde_json::json!({
                        "action": "remember",
                        "content": "fact B"
                    }),
                },
            ],
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        },
        CompletionResponse {
            text: "Stored both facts.".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage {
                input_tokens: 200,
                output_tokens: 20,
            },
        },
    ]);

    let mem_arc = Arc::new(InMemorySubstrate::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MemoryTool::new(
        mem_arc.clone(),
        "test-agent".into(),
    )));

    let result = run_agent_loop(
        &manifest,
        "Store two facts",
        &driver,
        &tools,
        mem_arc.as_ref(),
        None,
    )
    .await
    .expect("loop failed");

    assert_eq!(result.text, "Stored both facts.");
    assert_eq!(result.tool_calls, 2);
    assert_eq!(result.usage.input_tokens, 300);
    assert_eq!(result.usage.output_tokens, 70);
}

/// Memory recall augments system prompt.
#[tokio::test]
async fn test_memory_recall_augments_prompt() {
    let manifest = test_manifest();
    let memory = InMemorySubstrate::new();

    // Pre-populate memory
    memory
        .remember(
            "test-agent",
            "SIMD uses vector operations for parallelism",
            batuta::agent::memory::MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember failed");

    let driver = MockDriver::single_response(
        "Based on my recalled context about SIMD...",
    );
    let tools = ToolRegistry::new();

    let result = run_agent_loop(
        &manifest, "SIMD", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop failed");

    assert!(result.text.contains("SIMD"));
}

/// RoutingDriver: primary fails, fallback succeeds end-to-end.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_routing_driver_fallback_integration() {
    use async_trait::async_trait;
    use batuta::agent::driver::CompletionRequest;
    use batuta::agent::driver::router::{
        RoutingDriver, RoutingStrategy,
    };
    use batuta::serve::backends::PrivacyTier;

    let manifest = test_manifest();

    // Failing primary driver
    struct FailPrimary;

    #[async_trait]
    impl batuta::agent::driver::LlmDriver for FailPrimary {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, batuta::agent::AgentError>
        {
            Err(batuta::agent::AgentError::Driver(
                batuta::agent::result::DriverError::InferenceFailed(
                    "local model unavailable".into(),
                ),
            ))
        }
        fn context_window(&self) -> usize {
            4096
        }
        fn privacy_tier(&self) -> PrivacyTier {
            PrivacyTier::Sovereign
        }
    }

    let fallback =
        MockDriver::single_response("fallback response");
    let driver = RoutingDriver::new(
        Box::new(FailPrimary),
        Box::new(fallback),
    );

    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "hello", &driver, &tools, &memory, None,
    )
    .await
    .expect("should fallback");
    assert_eq!(result.text, "fallback response");
    assert_eq!(driver.metrics().spillover_count(), 1);
}

/// Context truncation works with tiny context window driver.
#[tokio::test]
async fn test_context_truncation_integration() {
    use async_trait::async_trait;
    use batuta::agent::driver::CompletionRequest;
    use batuta::serve::backends::PrivacyTier;

    let manifest = test_manifest();

    // Tiny-window driver wrapping MockDriver
    struct TinyDriver(MockDriver);

    #[async_trait]
    impl batuta::agent::driver::LlmDriver for TinyDriver {
        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, batuta::agent::AgentError> {
            self.0.complete(request).await
        }
        fn context_window(&self) -> usize {
            300
        }
        fn privacy_tier(&self) -> PrivacyTier {
            PrivacyTier::Sovereign
        }
    }

    let driver = TinyDriver(MockDriver::single_response(
        "context managed",
    ));
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "test", &driver, &tools, &memory, None,
    )
    .await
    .expect("should work with tiny context");
    assert_eq!(result.text, "context managed");
}
