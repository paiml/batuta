//! Tests for RemoteDriver (Anthropic + OpenAI API).

use super::*;
use crate::agent::driver::{Message, ToolDefinition};

fn test_config(provider: ApiProvider) -> RemoteDriverConfig {
    RemoteDriverConfig {
        base_url: "https://api.example.com".into(),
        api_key: "test-key".into(),
        model: "test-model".into(),
        provider,
        context_window: 4096,
    }
}

fn test_request() -> CompletionRequest {
    CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("Hello".into()),
            Message::Assistant("Hi there".into()),
            Message::User("What is 2+2?".into()),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("You are helpful.".into()),
    }
}

fn request_with_tools() -> CompletionRequest {
    CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("Search for X".into())],
        tools: vec![ToolDefinition {
            name: "rag".into(),
            description: "Search docs".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }],
        max_tokens: 1024,
        temperature: 0.5,
        system: None,
    }
}

#[test]
fn test_anthropic_body_format() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let body = driver.build_anthropic_body(&test_request());

    assert_eq!(body["model"], "test-model");
    assert_eq!(body["max_tokens"], 1024);
    assert_eq!(body["system"], "You are helpful.");
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
}

#[test]
fn test_openai_body_format() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let body = driver.build_openai_body(&test_request());

    assert_eq!(body["model"], "test-model");
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are helpful.");
    assert_eq!(msgs.len(), 4); // system + 3 user/assistant
}

#[test]
fn test_anthropic_body_with_tools() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let body =
        driver.build_anthropic_body(&request_with_tools());

    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "rag");
}

#[test]
fn test_openai_body_with_tools() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let body = driver.build_openai_body(&request_with_tools());

    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "rag");
}

#[test]
fn test_parse_anthropic_text_response() {
    let body = serde_json::json!({
        "content": [
            {"type": "text", "text": "Hello!"}
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.text, "Hello!");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.usage.input_tokens, 100);
    assert_eq!(resp.usage.output_tokens, 50);
}

#[test]
fn test_parse_anthropic_tool_use() {
    let body = serde_json::json!({
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "rag",
                "input": {"query": "SIMD"}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30
        }
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "rag");
    assert_eq!(resp.tool_calls[0].id, "tool_1");
}

#[test]
fn test_parse_openai_text_response() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.text, "Hello!");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.usage.input_tokens, 80);
    assert_eq!(resp.usage.output_tokens, 20);
}

#[test]
fn test_parse_openai_tool_calls() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "rag",
                        "arguments": "{\"query\": \"test\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "rag");
    assert_eq!(
        resp.tool_calls[0].input,
        serde_json::json!({"query": "test"})
    );
}

#[test]
fn test_parse_anthropic_stop_sequence() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "stop_sequence",
        "usage": {"input_tokens": 50, "output_tokens": 20}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::StopSequence);
}

#[test]
fn test_parse_anthropic_mixed_content() {
    let body = serde_json::json!({
        "content": [
            {"type": "text", "text": "I'll search for that."},
            {
                "type": "tool_use",
                "id": "tool_2",
                "name": "rag",
                "input": {"query": "SIMD"}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.text, "I'll search for that.");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
}

#[test]
fn test_parse_anthropic_empty_content() {
    let body = serde_json::json!({
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 0}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn test_parse_openai_null_content() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.text, "");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[test]
fn test_parse_openai_length_finish() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "truncated output"
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 4096
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.stop_reason, StopReason::MaxTokens);
    assert_eq!(resp.text, "truncated output");
}

#[test]
fn test_parse_anthropic_max_tokens() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 50, "output_tokens": 4096}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::MaxTokens);
}

#[test]
fn test_privacy_tier_is_standard() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    assert_eq!(driver.privacy_tier(), PrivacyTier::Standard);
}

#[test]
fn test_context_window() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    assert_eq!(driver.context_window(), 4096);
}

#[test]
fn test_openai_body_tool_use_messages() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("search for X".into()),
            Message::AssistantToolUse(ToolCall {
                id: "call_42".into(),
                name: "rag".into(),
                input: serde_json::json!({"query": "test"}),
            }),
            Message::ToolResult(
                crate::agent::driver::ToolResultMsg {
                    tool_use_id: "call_42".into(),
                    content: "found it".into(),
                    is_error: false,
                },
            ),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("You are helpful.".into()),
    };

    let body = driver.build_openai_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // system + user + assistant(tool_calls) + tool result = 4
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[2]["role"], "assistant");
    assert!(msgs[2]["tool_calls"].is_array());
    assert_eq!(
        msgs[2]["tool_calls"][0]["function"]["name"],
        "rag"
    );
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "call_42");
    assert_eq!(msgs[3]["content"], "found it");
}

#[test]
fn test_openai_body_system_message_in_messages() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::System("Extra system context.".into()),
            Message::User("hello".into()),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("Main system prompt.".into()),
    };

    let body = driver.build_openai_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // request.system (1) + System message (1) + User (1) = 3
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Main system prompt.");
    assert_eq!(msgs[1]["role"], "system");
    assert_eq!(msgs[1]["content"], "Extra system context.");
    assert_eq!(msgs[2]["role"], "user");
}

#[test]
fn test_anthropic_body_system_message_filtered() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::System("filtered out".into()),
            Message::User("hello".into()),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("system".into()),
    };

    let body = driver.build_anthropic_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // System messages are filtered out for Anthropic
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
}

#[test]
fn test_anthropic_body_no_system() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("hi".into())],
        tools: vec![],
        max_tokens: 512,
        temperature: 0.0,
        system: None,
    };

    let body = driver.build_anthropic_body(&request);
    assert!(body.get("system").is_none());
}

#[test]
fn test_build_request_anthropic_url() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let (url, body) = driver.build_request(&test_request());
    assert!(url.ends_with("/v1/messages"));
    assert_eq!(body["model"], "test-model");
}

#[test]
fn test_build_request_openai_url() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let (url, body) = driver.build_request(&test_request());
    assert!(url.ends_with("/v1/chat/completions"));
    assert_eq!(body["model"], "test-model");
}

#[test]
fn test_estimate_cost() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let usage = crate::agent::result::TokenUsage {
        input_tokens: 1_000_000,
        output_tokens: 100_000,
    };
    let cost = driver.estimate_cost(&usage);
    // input: 1M * 3/1M = $3.0
    // output: 100K * 15/1M = $1.5
    let expected = 3.0 + 1.5;
    assert!(
        (cost - expected).abs() < 0.001,
        "cost {cost} != expected {expected}"
    );
}

#[test]
fn test_estimate_cost_zero() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let usage = crate::agent::result::TokenUsage {
        input_tokens: 0,
        output_tokens: 0,
    };
    assert_eq!(driver.estimate_cost(&usage), 0.0);
}

#[test]
fn test_anthropic_tool_result_messages() {
    let driver =
        RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("search".into()),
            Message::AssistantToolUse(ToolCall {
                id: "1".into(),
                name: "rag".into(),
                input: serde_json::json!({"query": "test"}),
            }),
            Message::ToolResult(
                crate::agent::driver::ToolResultMsg {
                    tool_use_id: "1".into(),
                    content: "found it".into(),
                    is_error: false,
                },
            ),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: None,
    };

    let body = driver.build_anthropic_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["role"], "assistant");
    assert!(msgs[1]["content"][0]["type"]
        .as_str()
        .expect("type")
        .contains("tool_use"));
    assert_eq!(msgs[2]["role"], "user");
}
