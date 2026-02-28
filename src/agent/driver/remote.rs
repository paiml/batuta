//! Remote HTTP driver for external LLM APIs.
//!
//! Supports Anthropic Messages API and OpenAI-compatible Chat
//! Completions API. Privacy tier is `Standard` — blocked under
//! Sovereign manifests (Poka-Yoke).
//!
//! Phase 2: Used by `RoutingDriver` as fallback when local
//! inference fails or is unavailable.

use async_trait::async_trait;

use crate::agent::driver::{
    CompletionRequest, CompletionResponse, LlmDriver, Message,
    ToolCall,
};
use crate::agent::result::{
    AgentError, DriverError, StopReason, TokenUsage,
};
use crate::serve::backends::PrivacyTier;

/// API provider format for request/response translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    /// Anthropic Messages API (tool_use blocks).
    Anthropic,
    /// OpenAI-compatible Chat Completions API.
    OpenAi,
}

/// Remote HTTP LLM driver configuration.
#[derive(Debug, Clone)]
pub struct RemoteDriverConfig {
    /// API base URL (e.g., "https://api.anthropic.com").
    pub base_url: String,
    /// API key for authentication.
    pub api_key: String,
    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,
    /// API provider format.
    pub provider: ApiProvider,
    /// Context window size in tokens.
    pub context_window: usize,
}

/// Remote HTTP driver for external LLM APIs.
///
/// Sends requests to Anthropic/OpenAI-compatible endpoints.
/// Always operates at `PrivacyTier::Standard` — data leaves
/// the machine, so this driver is blocked under Sovereign
/// privacy manifests.
pub struct RemoteDriver {
    config: RemoteDriverConfig,
}

impl RemoteDriver {
    /// Create a new remote driver.
    pub fn new(config: RemoteDriverConfig) -> Self {
        Self { config }
    }

    /// Build request body for Anthropic Messages API.
    fn build_anthropic_body(
        &self,
        request: &CompletionRequest,
    ) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User(text) => Some(serde_json::json!({
                    "role": "user",
                    "content": text
                })),
                Message::Assistant(text) => {
                    Some(serde_json::json!({
                        "role": "assistant",
                        "content": text
                    }))
                }
                Message::AssistantToolUse(call) => {
                    Some(serde_json::json!({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.input
                        }]
                    }))
                }
                Message::ToolResult(result) => {
                    Some(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": result.tool_use_id,
                            "content": result.content,
                            "is_error": result.is_error
                        }]
                    }))
                }
                Message::System(_) => None,
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        });

        if let Some(ref system) = request.system {
            body["system"] = serde_json::json!(system);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        body
    }

    /// Build request body for OpenAI Chat Completions API.
    fn build_openai_body(
        &self,
        request: &CompletionRequest,
    ) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        for m in &request.messages {
            match m {
                Message::System(text) => {
                    messages.push(serde_json::json!({
                        "role": "system",
                        "content": text
                    }));
                }
                Message::User(text) => {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": text
                    }));
                }
                Message::Assistant(text) => {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
                Message::AssistantToolUse(call) => {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.input.to_string()
                            }
                        }]
                    }));
                }
                Message::ToolResult(result) => {
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": result.tool_use_id,
                        "content": result.content
                    }));
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        });

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        body
    }

    /// Parse Anthropic Messages API response.
    fn parse_anthropic_response(
        &self,
        body: &serde_json::Value,
    ) -> Result<CompletionResponse, AgentError> {
        let stop_reason =
            match body["stop_reason"].as_str().unwrap_or("end_turn") {
                "end_turn" => StopReason::EndTurn,
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

        Ok(CompletionResponse {
            text,
            stop_reason,
            tool_calls,
            usage,
        })
    }

    /// Parse OpenAI Chat Completions response.
    fn parse_openai_response(
        &self,
        body: &serde_json::Value,
    ) -> Result<CompletionResponse, AgentError> {
        let choice = &body["choices"][0];
        let message = &choice["message"];

        let stop_reason = match choice["finish_reason"]
            .as_str()
            .unwrap_or("stop")
        {
            "stop" => StopReason::EndTurn,
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

        Ok(CompletionResponse {
            text,
            stop_reason,
            tool_calls,
            usage,
        })
    }
}

#[async_trait]
impl LlmDriver for RemoteDriver {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        let (url, body) = match self.config.provider {
            ApiProvider::Anthropic => {
                let url = format!(
                    "{}/v1/messages",
                    self.config.base_url
                );
                (url, self.build_anthropic_body(&request))
            }
            ApiProvider::OpenAi => {
                let url = format!(
                    "{}/v1/chat/completions",
                    self.config.base_url
                );
                (url, self.build_openai_body(&request))
            }
        };

        let client = reqwest::Client::new();
        let mut req_builder = client.post(&url);

        match self.config.provider {
            ApiProvider::Anthropic => {
                req_builder = req_builder
                    .header("x-api-key", &self.config.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json");
            }
            ApiProvider::OpenAi => {
                req_builder = req_builder
                    .header(
                        "authorization",
                        format!(
                            "Bearer {}",
                            self.config.api_key
                        ),
                    )
                    .header("content-type", "application/json");
            }
        }

        let response = req_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::Driver(DriverError::Network(
                    format!("HTTP request failed: {e}"),
                ))
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            return Err(AgentError::Driver(
                DriverError::RateLimited {
                    retry_after_ms: 1000,
                },
            ));
        }
        if status == 529 || status == 503 {
            return Err(AgentError::Driver(
                DriverError::Overloaded {
                    retry_after_ms: 2000,
                },
            ));
        }
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Driver(DriverError::Network(
                format!("HTTP {status}: {text}"),
            )));
        }

        let resp_body: serde_json::Value =
            response.json().await.map_err(|e| {
                AgentError::Driver(
                    DriverError::InferenceFailed(format!(
                        "JSON parse error: {e}"
                    )),
                )
            })?;

        match self.config.provider {
            ApiProvider::Anthropic => {
                self.parse_anthropic_response(&resp_body)
            }
            ApiProvider::OpenAi => {
                self.parse_openai_response(&resp_body)
            }
        }
    }

    fn context_window(&self) -> usize {
        self.config.context_window
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Standard // data leaves the machine
    }
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
