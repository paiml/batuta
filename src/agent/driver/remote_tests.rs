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

    let resp = RemoteDriver::parse_anthropic_response(&body);
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

    let resp = RemoteDriver::parse_anthropic_response(&body);
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

    let resp = RemoteDriver::parse_openai_response(&body);
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

    let resp = RemoteDriver::parse_openai_response(&body);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "rag");
    assert_eq!(
        resp.tool_calls[0].input,
        serde_json::json!({"query": "test"})
    );
}

#[test]
fn test_parse_anthropic_max_tokens() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 50, "output_tokens": 4096}
    });

    let resp = RemoteDriver::parse_anthropic_response(&body);
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
