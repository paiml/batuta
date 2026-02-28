//! SSE streaming parsers for Anthropic and OpenAI APIs.
//!
//! Extracted from `remote.rs` for QA-002 (≤500 lines).
//! Handles `content_block_start/delta/stop` (Anthropic) and
//! `choices[0].delta` (OpenAI) Server-Sent Event formats.

use crate::agent::driver::{
    CompletionResponse, StreamEvent, ToolCall,
};
use crate::agent::result::{StopReason, TokenUsage};

/// Parse Anthropic Messages API response.
pub(super) fn parse_anthropic_response(
    body: &serde_json::Value,
) -> CompletionResponse {
    let stop_reason = match body["stop_reason"]
        .as_str()
        .unwrap_or("end_turn")
    {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = body["content"].as_array() {
        for block in content {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(t) = block["text"].as_str() {
                        text.push_str(t);
                    }
                }
                Some("tool_use") => {
                    tool_calls.push(ToolCall {
                        id: block["id"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string(),
                        name: block["name"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        input: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }
    }

    let usage = TokenUsage {
        input_tokens: body["usage"]["input_tokens"]
            .as_u64()
            .unwrap_or(0),
        output_tokens: body["usage"]["output_tokens"]
            .as_u64()
            .unwrap_or(0),
    };

    CompletionResponse {
        text,
        stop_reason,
        tool_calls,
        usage,
    }
}

/// Parse `OpenAI` Chat Completions response.
pub(super) fn parse_openai_response(
    body: &serde_json::Value,
) -> CompletionResponse {
    let choice = &body["choices"][0];
    let message = &choice["message"];

    let stop_reason = match choice["finish_reason"]
        .as_str()
        .unwrap_or("stop")
    {
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    let text = message["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let mut tool_calls = Vec::new();
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let input: serde_json::Value = call["function"]
                ["arguments"]
                .as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::json!({}));

            tool_calls.push(ToolCall {
                id: call["id"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
                name: call["function"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                input,
            });
        }
    }

    let usage = TokenUsage {
        input_tokens: body["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap_or(0),
        output_tokens: body["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or(0),
    };

    CompletionResponse {
        text,
        stop_reason,
        tool_calls,
        usage,
    }
}

/// Process a single Anthropic SSE event.
///
/// Accumulates text deltas, tool calls (partial JSON), usage,
/// and stop reason from the Anthropic Messages streaming API.
pub(super) async fn process_anthropic_event(
    event: &serde_json::Value,
    full_text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    usage: &mut TokenUsage,
    stop_reason: &mut StopReason,
    current_tool: &mut Option<(String, String, String)>,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) {
    let event_type = event["type"].as_str().unwrap_or("");
    match event_type {
        "content_block_start" => {
            let block = &event["content_block"];
            if block["type"].as_str() == Some("tool_use") {
                let id = block["id"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let name = block["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                *current_tool = Some((id, name, String::new()));
            }
        }
        "content_block_delta" => {
            let delta = &event["delta"];
            if let Some(text) = delta["text"].as_str() {
                full_text.push_str(text);
                let _ = tx
                    .send(StreamEvent::TextDelta {
                        text: text.to_string(),
                    })
                    .await;
            }
            if let Some(json) = delta["partial_json"].as_str() {
                if let Some((_, _, ref mut accum)) = current_tool {
                    accum.push_str(json);
                }
            }
        }
        "content_block_stop" => {
            if let Some((id, name, json_str)) = current_tool.take()
            {
                let input = serde_json::from_str(&json_str)
                    .unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall { id, name, input });
            }
        }
        "message_delta" => {
            if let Some(sr) =
                event["delta"]["stop_reason"].as_str()
            {
                *stop_reason = match sr {
                    "tool_use" => StopReason::ToolUse,
                    "max_tokens" => StopReason::MaxTokens,
                    "stop_sequence" => StopReason::StopSequence,
                    _ => StopReason::EndTurn,
                };
            }
            if let Some(out) =
                event["usage"]["output_tokens"].as_u64()
            {
                usage.output_tokens = out;
            }
        }
        "message_start" => {
            if let Some(inp) = event["message"]["usage"]
                ["input_tokens"]
                .as_u64()
            {
                usage.input_tokens = inp;
            }
        }
        _ => {}
    }
}

/// Process a single OpenAI SSE event.
///
/// Accumulates text deltas, tool calls (indexed partial JSON),
/// usage, and stop reason from the OpenAI Chat Completions
/// streaming API.
pub(super) async fn process_openai_event(
    event: &serde_json::Value,
    full_text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    usage: &mut TokenUsage,
    stop_reason: &mut StopReason,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) {
    let choice = &event["choices"][0];
    let delta = &choice["delta"];

    if let Some(text) = delta["content"].as_str() {
        full_text.push_str(text);
        let _ = tx
            .send(StreamEvent::TextDelta {
                text: text.to_string(),
            })
            .await;
    }

    if let Some(calls) = delta["tool_calls"].as_array() {
        for call in calls {
            let idx =
                call["index"].as_u64().unwrap_or(0) as usize;
            while tool_calls.len() <= idx {
                tool_calls.push(ToolCall {
                    id: String::new(),
                    name: String::new(),
                    input: serde_json::json!({}),
                });
            }
            if let Some(id) = call["id"].as_str() {
                tool_calls[idx].id = id.to_string();
            }
            if let Some(name) =
                call["function"]["name"].as_str()
            {
                tool_calls[idx].name = name.to_string();
            }
            if let Some(args) =
                call["function"]["arguments"].as_str()
            {
                let existing = tool_calls[idx]
                    .input
                    .as_str()
                    .unwrap_or("");
                let combined = format!("{existing}{args}");
                tool_calls[idx].input =
                    serde_json::from_str(&combined)
                        .unwrap_or(serde_json::json!(combined));
            }
        }
    }

    if let Some(fr) = choice["finish_reason"].as_str() {
        *stop_reason = match fr {
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };
    }

    if let Some(u) = event.get("usage") {
        if let Some(inp) = u["prompt_tokens"].as_u64() {
            usage.input_tokens = inp;
        }
        if let Some(out) = u["completion_tokens"].as_u64() {
            usage.output_tokens = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_anthropic_text_delta() {
        let event = serde_json::json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"}
        });

        let (tx, mut rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        process_anthropic_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert_eq!(text, "Hello");
        let evt = rx.try_recv().expect("expected event");
        assert!(matches!(evt, StreamEvent::TextDelta { text } if text == "Hello"));
    }

    #[tokio::test]
    async fn test_anthropic_tool_use_stream() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        // content_block_start with tool_use
        let start = serde_json::json!({
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "tool_abc",
                "name": "rag"
            }
        });
        process_anthropic_event(
            &start,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;
        assert!(current_tool.is_some());

        // partial JSON delta
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "delta": {"partial_json": "{\"query\":\"test\"}"}
        });
        process_anthropic_event(
            &delta,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        // content_block_stop finalizes tool call
        let stop_evt = serde_json::json!({
            "type": "content_block_stop"
        });
        process_anthropic_event(
            &stop_evt,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tool_abc");
        assert_eq!(tool_calls[0].name, "rag");
        assert_eq!(
            tool_calls[0].input,
            serde_json::json!({"query": "test"})
        );
        assert!(current_tool.is_none());
    }

    #[tokio::test]
    async fn test_anthropic_message_start_usage() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        let event = serde_json::json!({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 250}
            }
        });
        process_anthropic_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert_eq!(usage.input_tokens, 250);
    }

    #[tokio::test]
    async fn test_anthropic_message_delta_stop() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        let event = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 4096}
        });
        process_anthropic_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert_eq!(stop, StopReason::MaxTokens);
        assert_eq!(usage.output_tokens, 4096);
    }

    #[tokio::test]
    async fn test_openai_text_delta() {
        let event = serde_json::json!({
            "choices": [{
                "delta": {"content": "World"},
                "index": 0
            }]
        });

        let (tx, mut rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;

        process_openai_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &tx,
        )
        .await;

        assert_eq!(text, "World");
        let evt = rx.try_recv().expect("expected event");
        assert!(matches!(evt, StreamEvent::TextDelta { text } if text == "World"));
    }

    #[tokio::test]
    async fn test_openai_tool_call_stream() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;

        let event = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "rag",
                            "arguments": "{\"query\":\"test\"}"
                        }
                    }]
                },
                "index": 0
            }]
        });

        process_openai_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &tx,
        )
        .await;

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].name, "rag");
    }

    #[tokio::test]
    async fn test_openai_finish_reason() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;

        let event = serde_json::json!({
            "choices": [{
                "delta": {},
                "finish_reason": "length",
                "index": 0
            }]
        });

        process_openai_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &tx,
        )
        .await;

        assert_eq!(stop, StopReason::MaxTokens);
    }

    #[tokio::test]
    async fn test_openai_usage_event() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;

        let event = serde_json::json!({
            "choices": [{"delta": {}, "index": 0}],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 80
            }
        });

        process_openai_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &tx,
        )
        .await;

        assert_eq!(usage.input_tokens, 150);
        assert_eq!(usage.output_tokens, 80);
    }

    #[tokio::test]
    async fn test_anthropic_unknown_event_type_ignored() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        let event = serde_json::json!({
            "type": "ping"
        });
        process_anthropic_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert!(text.is_empty());
        assert!(tool_calls.is_empty());
    }

    #[tokio::test]
    async fn test_anthropic_tool_use_stop_reason() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;
        let mut current_tool = None;

        let event = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 100}
        });
        process_anthropic_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &mut current_tool,
            &tx,
        )
        .await;

        assert_eq!(stop, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn test_openai_tool_calls_stop_reason() {
        let (tx, _rx) = mpsc::channel(16);
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut usage =
            TokenUsage { input_tokens: 0, output_tokens: 0 };
        let mut stop = StopReason::EndTurn;

        let event = serde_json::json!({
            "choices": [{
                "delta": {},
                "finish_reason": "tool_calls",
                "index": 0
            }]
        });
        process_openai_event(
            &event,
            &mut text,
            &mut tool_calls,
            &mut usage,
            &mut stop,
            &tx,
        )
        .await;

        assert_eq!(stop, StopReason::ToolUse);
    }
}
