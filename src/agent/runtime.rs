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
use crate::serve::context::{
    ContextConfig, ContextManager, ContextWindow, TokenEstimator,
    TruncationStrategy,
};

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
    emit(stream_tx.as_ref(), StreamEvent::PhaseChange {
        phase: LoopPhase::Perceive,
    })
    .await;

    let system =
        build_system_prompt(manifest, query, memory).await;
    let tool_defs = tools.definitions_for(&manifest.capabilities);
    let context =
        build_context(driver, &system, &tool_defs, manifest);

    let mut messages = vec![Message::User(query.to_string())];

    loop {
        check_verdict(guard.check_iteration())?;

        // ═══ REASON ═══
        emit(stream_tx.as_ref(), StreamEvent::PhaseChange {
            phase: LoopPhase::Reason,
        })
        .await;

        let response = reason_step(
            driver, &messages, &tool_defs, manifest,
            &system, &context,
        )
        .await?;
        guard.record_usage(&response.usage);

        // INV-005: Estimate cost and enforce budget (Muda)
        let cost = driver.estimate_cost(&response.usage);
        check_verdict(guard.record_cost(cost))?;

        match response.stop_reason {
            StopReason::EndTurn | StopReason::StopSequence => {
                return finish_loop(
                    &response, &guard, manifest, query,
                    memory, stream_tx.as_ref(),
                )
                .await;
            }
            StopReason::ToolUse => {
                guard.reset_max_tokens();
                handle_tool_calls(
                    &response, &mut messages, &mut guard,
                    manifest, tools, stream_tx.as_ref(),
                )
                .await?;
            }
            StopReason::MaxTokens => {
                check_verdict(guard.record_max_tokens())?;
                messages
                    .push(Message::Assistant(response.text));
            }
        }
    }
}

/// Check a `LoopVerdict` and return error on circuit-break/block.
fn check_verdict(
    verdict: LoopVerdict,
) -> Result<(), AgentError> {
    match verdict {
        LoopVerdict::CircuitBreak(msg)
        | LoopVerdict::Block(msg) => {
            Err(AgentError::CircuitBreak(msg))
        }
        LoopVerdict::Allow | LoopVerdict::Warn(_) => Ok(()),
    }
}

/// Execute a single reason step: truncate, build request, call driver.
async fn reason_step(
    driver: &dyn LlmDriver,
    messages: &[Message],
    tool_defs: &[super::driver::ToolDefinition],
    manifest: &AgentManifest,
    system: &str,
    context: &ContextManager,
) -> Result<CompletionResponse, AgentError> {
    let truncated_messages =
        truncate_messages(messages, context)?;

    let request = CompletionRequest {
        model: String::new(),
        messages: truncated_messages,
        tools: tool_defs.to_vec(),
        max_tokens: manifest.model.max_tokens,
        temperature: manifest.model.temperature,
        system: Some(system.to_string()),
    };

    call_with_retry(driver, &request).await
}

/// Finalize the loop: remember conversation, emit Done, return result.
async fn finish_loop(
    response: &CompletionResponse,
    guard: &LoopGuard,
    manifest: &AgentManifest,
    query: &str,
    memory: &dyn MemorySubstrate,
    stream_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<AgentLoopResult, AgentError> {
    let _ = memory
        .remember(
            &manifest.name,
            &format!("Q: {query}\nA: {}", response.text),
            MemorySource::Conversation,
            None,
        )
        .await;

    emit(stream_tx, StreamEvent::PhaseChange {
        phase: LoopPhase::Done,
    })
    .await;

    Ok(AgentLoopResult {
        text: response.text.clone(),
        usage: guard.usage().clone(),
        iterations: guard.current_iteration(),
        tool_calls: guard.total_tool_calls(),
    })
}

/// Build system prompt with recalled memories.
async fn build_system_prompt(
    manifest: &AgentManifest,
    query: &str,
    memory: &dyn MemorySubstrate,
) -> String {
    let memories = memory
        .recall(query, 5, None, None)
        .await
        .unwrap_or_default();

    let mut system = manifest.model.system_prompt.clone();
    if !memories.is_empty() {
        use std::fmt::Write;
        system.push_str("\n\n## Recalled Context\n");
        for m in &memories {
            let _ = writeln!(system, "- {}", m.content);
        }
    }
    system
}

/// Build context manager with token budget.
fn build_context(
    driver: &dyn LlmDriver,
    system: &str,
    tool_defs: &[super::driver::ToolDefinition],
    manifest: &AgentManifest,
) -> ContextManager {
    let estimator = TokenEstimator::new();
    let system_tokens = estimator.estimate(system);
    let tool_json = serde_json::to_string(tool_defs)
        .unwrap_or_default();
    let tool_tokens = estimator.estimate(&tool_json);
    let context_window = driver.context_window();
    let effective_window = context_window
        .saturating_sub(system_tokens)
        .saturating_sub(tool_tokens);
    ContextManager::new(ContextConfig {
        window: ContextWindow::new(
            effective_window,
            manifest.model.max_tokens as usize,
        ),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 2,
    })
}

/// Process tool calls from a completion response.
async fn handle_tool_calls(
    response: &CompletionResponse,
    messages: &mut Vec<Message>,
    guard: &mut LoopGuard,
    manifest: &AgentManifest,
    tools: &ToolRegistry,
    stream_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<(), AgentError> {
    for call in &response.tool_calls {
        let Some(tool) = tools.get(&call.name) else {
            push_tool_error(
                messages,
                call,
                &format!("unknown tool: {}", call.name),
            );
            continue;
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

/// Truncate agent messages to fit within context window.
///
/// Converts `Message` to `ChatMessage` for the `ContextManager`,
/// then maps truncated results back to the original `Message` list.
/// Returns `ContextOverflow` if the strategy is `Error` and
/// messages exceed the window.
fn truncate_messages(
    messages: &[Message],
    context: &ContextManager,
) -> Result<Vec<Message>, AgentError> {
    let chat_msgs: Vec<_> =
        messages.iter().map(Message::to_chat_message).collect();

    if context.fits(&chat_msgs) {
        return Ok(messages.to_vec());
    }

    let truncated = context.truncate(&chat_msgs).map_err(
        |crate::serve::context::ContextError::ExceedsLimit {
             tokens,
             limit,
         }| AgentError::ContextOverflow {
            required: tokens,
            available: limit,
        },
    )?;

    // Map truncated ChatMessages back to original Messages
    // by matching content. SlidingWindow keeps most recent,
    // so iterate from end of original list.
    let mut result = Vec::with_capacity(truncated.len());
    let mut msg_idx = messages.len();
    for chat_msg in truncated.iter().rev() {
        while msg_idx > 0 {
            msg_idx -= 1;
            if messages[msg_idx].to_chat_message().content
                == chat_msg.content
            {
                result.push(messages[msg_idx].clone());
                break;
            }
        }
    }
    result.reverse();
    Ok(result)
}

/// Retry `driver.complete()` with exponential backoff for retryable errors.
///
/// Policy (spec §4.3): 1s base, 3 max retries for
/// `RateLimited`/`Overloaded`/`Network`. Immediate fail for
/// `ModelNotFound`/`InferenceFailed`.
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
    tx: Option<&mpsc::Sender<StreamEvent>>,
    event: StreamEvent,
) {
    if let Some(tx) = tx {
        let _ = tx.send(event).await;
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
