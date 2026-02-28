//! Agent runtime — the core perceive-reason-act loop.
//!
//! Orchestrates the agent loop: recall memories (perceive),
//! generate LLM completions (reason), execute tool calls (act),
//! repeat until done or guard triggers (Jidoka).
//!
//! See: arXiv:2512.10350 (loop dynamics), arXiv:2501.09136 (agentic RAG).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use super::capability::capability_matches;
use super::driver::{
    CompletionRequest, CompletionResponse, LlmDriver, Message,
    StreamEvent, ToolCall, ToolResultMsg,
};
use super::guard::{LoopGuard, LoopVerdict};
use super::manifest::AgentManifest;
use super::memory::{MemorySource, MemorySubstrate};
use super::phase::LoopPhase;
use super::result::{AgentError, AgentLoopResult, StopReason};
use super::tool::ToolRegistry;

/// Maximum retry attempts for retryable driver errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (milliseconds).
const RETRY_BASE_MS: u64 = 1000;

/// Run the agent loop to completion.
///
/// Returns the final response text and usage statistics.
/// The loop terminates when the model produces an `EndTurn`
/// response, or when the guard circuit-breaks.
pub async fn run_agent_loop(
    manifest: &AgentManifest,
    query: &str,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    let mut guard = LoopGuard::new(
        manifest.resources.max_iterations,
        manifest.resources.max_tool_calls,
        manifest.resources.max_cost_usd,
    );

    // ═══ PERCEIVE ═══
    emit(&stream_tx, StreamEvent::PhaseChange {
        phase: LoopPhase::Perceive,
    })
    .await;

    let memories = memory
        .recall(query, 5, None, None)
        .await
        .unwrap_or_default();

    let mut system = manifest.model.system_prompt.clone();
    if !memories.is_empty() {
        system.push_str("\n\n## Recalled Context\n");
        for m in &memories {
            system.push_str(&format!("- {}\n", m.content));
        }
    }

    let mut messages = vec![Message::User(query.to_string())];

    loop {
        // Check iteration budget
        match guard.check_iteration() {
            LoopVerdict::CircuitBreak(msg) => {
                return Err(AgentError::CircuitBreak(msg));
            }
            LoopVerdict::Block(msg) => {
                return Err(AgentError::CircuitBreak(msg));
            }
            LoopVerdict::Allow | LoopVerdict::Warn(_) => {}
        }

        // ═══ REASON ═══
        emit(&stream_tx, StreamEvent::PhaseChange {
            phase: LoopPhase::Reason,
        })
        .await;

        let request = CompletionRequest {
            model: String::new(),
            messages: messages.clone(),
            tools: tools.definitions_for(&manifest.capabilities),
            max_tokens: manifest.model.max_tokens,
            temperature: manifest.model.temperature,
            system: Some(system.clone()),
        };

        let response =
            call_with_retry(driver, &request).await?;
        guard.record_usage(&response.usage);

        match response.stop_reason {
            StopReason::EndTurn | StopReason::StopSequence => {
                guard.reset_max_tokens();
                // ═══ REMEMBER ═══
                let _ = memory
                    .remember(
                        &manifest.name,
                        &format!("Q: {query}\nA: {}", response.text),
                        MemorySource::Conversation,
                        None,
                    )
                    .await;

                emit(&stream_tx, StreamEvent::PhaseChange {
                    phase: LoopPhase::Done,
                })
                .await;

                return Ok(AgentLoopResult {
                    text: response.text,
                    usage: guard.usage().clone(),
                    iterations: guard.current_iteration(),
                    tool_calls: guard.total_tool_calls(),
                });
            }

            StopReason::ToolUse => {
                guard.reset_max_tokens();
                handle_tool_calls(
                    &response,
                    &mut messages,
                    &mut guard,
                    manifest,
                    tools,
                    &stream_tx,
                )
                .await?;
            }

            StopReason::MaxTokens => {
                if let LoopVerdict::CircuitBreak(msg) =
                    guard.record_max_tokens()
                {
                    return Err(AgentError::CircuitBreak(msg));
                }
                messages
                    .push(Message::Assistant(response.text));
            }
        }
    }
}

/// Process tool calls from a completion response.
async fn handle_tool_calls(
    response: &CompletionResponse,
    messages: &mut Vec<Message>,
    guard: &mut LoopGuard,
    manifest: &AgentManifest,
    tools: &ToolRegistry,
    stream_tx: &Option<mpsc::Sender<StreamEvent>>,
) -> Result<(), AgentError> {
    for call in &response.tool_calls {
        let tool = match tools.get(&call.name) {
            Some(t) => t,
            None => {
                push_tool_error(
                    messages,
                    call,
                    &format!("unknown tool: {}", call.name),
                );
                continue;
            }
        };

        // Poka-Yoke: capability check
        if !capability_matches(
            &manifest.capabilities,
            &tool.required_capability(),
        ) {
            push_tool_error(
                messages,
                call,
                &format!(
                    "capability denied for tool '{}'",
                    call.name
                ),
            );
            continue;
        }

        // Jidoka: loop guard check
        match guard.check_tool_call(&call.name, &call.input) {
            LoopVerdict::Allow | LoopVerdict::Warn(_) => {}
            LoopVerdict::Block(msg) => {
                push_tool_error(messages, call, &msg);
                continue;
            }
            LoopVerdict::CircuitBreak(msg) => {
                return Err(AgentError::CircuitBreak(msg));
            }
        }

        // ═══ ACT ═══
        emit(stream_tx, StreamEvent::PhaseChange {
            phase: LoopPhase::Act {
                tool_name: call.name.clone(),
            },
        })
        .await;

        emit(stream_tx, StreamEvent::ToolUseStart {
            id: call.id.clone(),
            name: call.name.clone(),
        })
        .await;

        let result = tokio::time::timeout(
            tool.timeout(),
            tool.execute(call.input.clone()),
        )
        .await
        .unwrap_or_else(|_| {
            super::tool::ToolResult::error(format!(
                "tool '{}' timed out",
                call.name
            ))
        });

        emit(stream_tx, StreamEvent::ToolUseEnd {
            id: call.id.clone(),
            name: call.name.clone(),
            result: result.content.clone(),
        })
        .await;

        messages.push(Message::AssistantToolUse(ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        }));
        messages.push(Message::ToolResult(ToolResultMsg {
            tool_use_id: call.id.clone(),
            content: result.content,
            is_error: result.is_error,
        }));
    }
    Ok(())
}

fn push_tool_error(
    messages: &mut Vec<Message>,
    call: &ToolCall,
    error: &str,
) {
    messages.push(Message::AssistantToolUse(ToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        input: call.input.clone(),
    }));
    messages.push(Message::ToolResult(ToolResultMsg {
        tool_use_id: call.id.clone(),
        content: error.to_string(),
        is_error: true,
    }));
}

/// Retry driver.complete() with exponential backoff for retryable errors.
///
/// Policy (spec §4.3): 1s base, 3 max retries for
/// RateLimited/Overloaded/Network. Immediate fail for
/// ModelNotFound/InferenceFailed.
async fn call_with_retry(
    driver: &dyn LlmDriver,
    request: &CompletionRequest,
) -> Result<CompletionResponse, AgentError> {
    let mut last_err = None;
    for attempt in 0..=MAX_RETRIES {
        match driver.complete(request.clone()).await {
            Ok(response) => return Ok(response),
            Err(AgentError::Driver(ref e)) if e.is_retryable() => {
                last_err = Some(AgentError::Driver(e.clone()));
                if attempt < MAX_RETRIES {
                    let delay = RETRY_BASE_MS * 2u64.pow(attempt);
                    tokio::time::sleep(Duration::from_millis(delay))
                        .await;
                }
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.expect("retry loop should have set last_err"))
}

async fn emit(
    tx: &Option<mpsc::Sender<StreamEvent>>,
    event: StreamEvent,
) {
    if let Some(tx) = tx {
        let _ = tx.send(event).await;
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
