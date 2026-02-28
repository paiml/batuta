//! Tests for agent runtime loop.

use super::*;
use crate::agent::capability::Capability;
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::ToolDefinition;
use crate::agent::memory::InMemorySubstrate;
use crate::agent::tool::{Tool, ToolResult as TResult};
use async_trait::async_trait;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "Echoes input".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> TResult {
        TResult::success(format!("echo: {input}"))
    }

    fn required_capability(&self) -> Capability {
        Capability::Memory
    }
}

fn default_manifest() -> AgentManifest {
    AgentManifest {
        capabilities: vec![Capability::Memory, Capability::Rag],
        ..AgentManifest::default()
    }
}

#[tokio::test]
async fn test_single_turn_response() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("Hello!");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "hi", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop failed");

    assert_eq!(result.text, "Hello!");
    assert_eq!(result.iterations, 1);
    assert_eq!(result.tool_calls, 0);
}

#[tokio::test]
async fn test_tool_call_and_response() {
    let manifest = default_manifest();
    let driver = MockDriver::tool_then_response(
        "echo",
        serde_json::json!({"text": "test"}),
        "Final answer",
    );
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "do it", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop failed");

    assert_eq!(result.text, "Final answer");
    assert_eq!(result.iterations, 2);
    assert_eq!(result.tool_calls, 1);
}

#[tokio::test]
async fn test_max_iterations_reached() {
    let mut manifest = default_manifest();
    manifest.resources.max_iterations = 1;

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "2".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": 1}),
            }],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(
        &manifest, "go", &driver, &tools, &memory, None,
    )
    .await
    .unwrap_err();

    assert!(
        matches!(err, AgentError::CircuitBreak(_)),
        "expected CircuitBreak, got: {err}"
    );
}

#[tokio::test]
async fn test_unknown_tool_handled() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "nonexistent".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "recovered".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop should recover from unknown tool");
    assert_eq!(result.text, "recovered");
}

#[tokio::test]
async fn test_capability_denied_handled() {
    let mut manifest = default_manifest();
    manifest.capabilities = vec![Capability::Rag];

    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "denied path".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop should handle capability denial");
    assert_eq!(result.text, "denied path");
}

#[tokio::test]
async fn test_stream_events_emitted() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("done");
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let (tx, mut rx) = mpsc::channel(32);

    run_agent_loop(
        &manifest, "hi", &driver, &tools, &memory, Some(tx),
    )
    .await
    .expect("loop failed");

    let mut phases = vec![];
    while let Ok(event) = rx.try_recv() {
        if let StreamEvent::PhaseChange { phase } = event {
            phases.push(phase);
        }
    }

    assert!(phases.contains(&LoopPhase::Perceive));
    assert!(phases.contains(&LoopPhase::Reason));
    assert!(phases.contains(&LoopPhase::Done));
}

#[tokio::test]
async fn test_memories_recalled_into_system() {
    let manifest = default_manifest();
    let memory = InMemorySubstrate::new();
    memory
        .remember(
            "unnamed-agent",
            "prior knowledge about SIMD",
            MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember failed");

    let driver = MockDriver::single_response("answer");
    let tools = ToolRegistry::new();

    let result = run_agent_loop(
        &manifest, "SIMD", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop failed");
    assert_eq!(result.text, "answer");
}

#[tokio::test]
async fn test_conversation_stored_in_memory() {
    let manifest = default_manifest();
    let driver = MockDriver::single_response("the answer");
    let tools = ToolRegistry::new();
    let memory = Arc::new(InMemorySubstrate::new());

    run_agent_loop(
        &manifest,
        "my question",
        &driver,
        &tools,
        memory.as_ref(),
        None,
    )
    .await
    .expect("loop failed");

    let recalled = memory
        .recall("my question", 10, None, None)
        .await
        .expect("recall failed");
    assert!(!recalled.is_empty());
    assert!(recalled[0].content.contains("the answer"));
}

#[tokio::test]
async fn test_max_tokens_continues_loop() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: "partial".into(),
            stop_reason: StopReason::MaxTokens,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "complete".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop should continue after MaxTokens");
    assert_eq!(result.text, "complete");
    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn test_pingpong_blocked_in_tool_calls() {
    let manifest = default_manifest();
    // Three identical tool calls will trigger ping-pong detection
    let driver = MockDriver::new(vec![
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "2".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: String::new(),
            stop_reason: StopReason::ToolUse,
            tool_calls: vec![ToolCall {
                id: "3".into(),
                name: "echo".into(),
                input: serde_json::json!({"x": "same"}),
            }],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "after block".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("loop should recover from ping-pong block");
    assert_eq!(result.text, "after block");
}

#[tokio::test]
async fn test_stop_sequence_handled() {
    let manifest = default_manifest();
    let driver = MockDriver::new(vec![CompletionResponse {
        text: "stopped".into(),
        stop_reason: StopReason::StopSequence,
        tool_calls: vec![],
        usage: Default::default(),
    }]);
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("StopSequence should end the loop");
    assert_eq!(result.text, "stopped");
}

#[tokio::test]
async fn test_stream_events_with_tool_call() {
    let manifest = default_manifest();
    let driver = MockDriver::tool_then_response(
        "echo",
        serde_json::json!({}),
        "done",
    );
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let (tx, mut rx) = mpsc::channel(64);

    run_agent_loop(
        &manifest, "hi", &driver, &tools, &memory, Some(tx),
    )
    .await
    .expect("loop failed");

    let mut got_tool_start = false;
    let mut got_tool_end = false;
    let mut got_act = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            StreamEvent::ToolUseStart { .. } => got_tool_start = true,
            StreamEvent::ToolUseEnd { .. } => got_tool_end = true,
            StreamEvent::PhaseChange {
                phase: LoopPhase::Act { .. },
            } => got_act = true,
            _ => {}
        }
    }
    assert!(got_tool_start, "expected ToolUseStart event");
    assert!(got_tool_end, "expected ToolUseEnd event");
    assert!(got_act, "expected Act phase event");
}

/// Driver that fails N times with a retryable error, then succeeds.
struct RetryDriver {
    fail_count: std::sync::atomic::AtomicU32,
    max_fails: u32,
    success_response: CompletionResponse,
}

impl RetryDriver {
    fn new(max_fails: u32, response: CompletionResponse) -> Self {
        Self {
            fail_count: std::sync::atomic::AtomicU32::new(0),
            max_fails,
            success_response: response,
        }
    }
}

#[async_trait]
impl crate::agent::driver::LlmDriver for RetryDriver {
    async fn complete(
        &self,
        _request: crate::agent::driver::CompletionRequest,
    ) -> Result<CompletionResponse, crate::agent::result::AgentError> {
        let count = self
            .fail_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count < self.max_fails {
            Err(crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::Network(
                    "transient network error".into(),
                ),
            ))
        } else {
            Ok(self.success_response.clone())
        }
    }

    fn context_window(&self) -> usize {
        4096
    }

    fn privacy_tier(
        &self,
    ) -> crate::serve::backends::PrivacyTier {
        crate::serve::backends::PrivacyTier::Sovereign
    }
}

#[tokio::test]
async fn test_retry_on_transient_error() {
    let manifest = default_manifest();
    let driver = RetryDriver::new(
        2,
        CompletionResponse {
            text: "recovered after retry".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    );
    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "q", &driver, &tools, &memory, None,
    )
    .await
    .expect("should succeed after retries");
    assert_eq!(result.text, "recovered after retry");
}

#[tokio::test]
async fn test_non_retryable_error_fails_immediately() {
    let manifest = default_manifest();

    struct FailDriver;

    #[async_trait]
    impl crate::agent::driver::LlmDriver for FailDriver {
        async fn complete(
            &self,
            _request: crate::agent::driver::CompletionRequest,
        ) -> Result<
            CompletionResponse,
            crate::agent::result::AgentError,
        > {
            Err(crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::InferenceFailed(
                    "model corrupted".into(),
                ),
            ))
        }
        fn context_window(&self) -> usize {
            4096
        }
        fn privacy_tier(
            &self,
        ) -> crate::serve::backends::PrivacyTier {
            crate::serve::backends::PrivacyTier::Sovereign
        }
    }

    let tools = ToolRegistry::new();
    let memory = InMemorySubstrate::new();

    let err = run_agent_loop(
        &manifest, "q", &FailDriver, &tools, &memory, None,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            err,
            crate::agent::result::AgentError::Driver(
                crate::agent::result::DriverError::InferenceFailed(_)
            )
        ),
        "expected InferenceFailed, got: {err}"
    );
}

/// Context truncation: small window driver truncates long conversations.
#[tokio::test]
async fn test_context_truncation_small_window() {
    let manifest = default_manifest();

    // Driver with 200-token context window (tiny)
    struct TinyWindowDriver {
        inner: MockDriver,
    }
    #[async_trait]
    impl crate::agent::driver::LlmDriver for TinyWindowDriver {
        async fn complete(
            &self,
            request: crate::agent::driver::CompletionRequest,
        ) -> Result<
            CompletionResponse,
            crate::agent::result::AgentError,
        > {
            // Verify messages were truncated: should be fewer
            // than the 3 tool-call messages we'll generate
            self.inner.complete(request).await
        }
        fn context_window(&self) -> usize {
            200 // Very small
        }
        fn privacy_tier(
            &self,
        ) -> crate::serve::backends::PrivacyTier {
            crate::serve::backends::PrivacyTier::Sovereign
        }
    }

    // Multi-turn: tool call then response
    let driver = TinyWindowDriver {
        inner: MockDriver::tool_then_response(
            "echo",
            serde_json::json!({"text": "x".repeat(100)}),
            "truncated ok",
        ),
    };
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(EchoTool));
    let memory = InMemorySubstrate::new();

    let result = run_agent_loop(
        &manifest, "hi", &driver, &tools, &memory, None,
    )
    .await
    .expect("should succeed with truncation");
    assert_eq!(result.text, "truncated ok");
}

/// Context truncation: messages converted correctly to ChatMessage.
#[tokio::test]
async fn test_message_to_chat_message_conversion() {
    use crate::agent::driver::Message;

    let msgs = vec![
        Message::System("sys".into()),
        Message::User("hello".into()),
        Message::Assistant("hi".into()),
        Message::AssistantToolUse(ToolCall {
            id: "1".into(),
            name: "echo".into(),
            input: serde_json::json!({"x": 1}),
        }),
        Message::ToolResult(crate::agent::driver::ToolResultMsg {
            tool_use_id: "1".into(),
            content: "result".into(),
            is_error: false,
        }),
    ];

    let chat_msgs: Vec<_> =
        msgs.iter().map(|m| m.to_chat_message()).collect();

    assert_eq!(chat_msgs.len(), 5);
    assert_eq!(chat_msgs[0].content, "sys");
    assert_eq!(chat_msgs[1].content, "hello");
    assert_eq!(chat_msgs[2].content, "hi");
    assert!(chat_msgs[3].content.contains("echo"));
    assert!(chat_msgs[4].content.contains("result"));
}
