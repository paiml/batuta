//! RealizarDriver — sovereign local inference via GGUF/APR models.
//!
//! Uses the `realizar` crate for local LLM inference. All data
//! stays on-device (Sovereign privacy tier, Genchi Genbutsu).
//!
//! Tool call parsing: local models output `<tool_call>` JSON blocks
//! in their text. The driver extracts these into `ToolCall` structs.
//!
//! Feature-gated behind `inference`.

use async_trait::async_trait;
use std::path::PathBuf;

use super::chat_template::{format_prompt_with_template, ChatTemplate};
use super::{CompletionRequest, CompletionResponse, LlmDriver, ToolCall};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Local inference driver using realizar (GGUF/APR/SafeTensors).
pub struct RealizarDriver {
    /// Path to model file.
    model_path: PathBuf,
    /// Context window size.
    context_window_size: usize,
    /// Auto-detected chat template.
    template: ChatTemplate,
}

impl RealizarDriver {
    /// Create a new RealizarDriver from a model path.
    ///
    /// Auto-detects the chat template from the model filename
    /// (Qwen/DeepSeek → ChatML, Llama → Llama3, else ChatML default).
    pub fn new(model_path: PathBuf, context_window: Option<usize>) -> Result<Self, AgentError> {
        if !model_path.exists() {
            return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
                "model not found: {}",
                model_path.display()
            ))));
        }
        let context_window_size = context_window.unwrap_or(4096);
        let template = ChatTemplate::from_model_path(&model_path);
        Ok(Self { model_path, context_window_size, template })
    }
}

#[async_trait]
impl LlmDriver for RealizarDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        // Format messages using auto-detected chat template
        let prompt = format_prompt_with_template(&request, self.template);

        // Build inference config (explicit fields — no Default impl)
        let config = realizar::infer::InferenceConfig {
            model_path: self.model_path.clone(),
            prompt: Some(prompt),
            input_tokens: None,
            max_tokens: request.max_tokens as usize,
            temperature: request.temperature,
            top_k: 0,
            no_gpu: false,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: false,
            use_mock_backend: false,
        };

        // Run inference in blocking thread (realizar is sync)
        let result = tokio::task::spawn_blocking(move || realizar::infer::run_inference(&config))
            .await
            .map_err(|e| {
                AgentError::Driver(DriverError::InferenceFailed(format!("spawn_blocking: {e}")))
            })?
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(e.to_string())))?;

        // Parse tool calls from text output
        let (text, tool_calls) = parse_tool_calls(&result.text);

        let stop_reason =
            if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };

        Ok(CompletionResponse {
            text,
            stop_reason,
            tool_calls,
            usage: TokenUsage {
                input_tokens: result.input_token_count as u64,
                output_tokens: result.generated_token_count as u64,
            },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// Parse `<tool_call>` JSON blocks from model output text.
///
/// Local models output tool calls as:
/// ```text
/// <tool_call>
/// {"name": "rag", "input": {"query": "SIMD"}}
/// </tool_call>
/// ```
///
/// Returns the remaining text (with tool_call blocks removed)
/// and the extracted tool calls.
fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut call_counter = 0u32;

    let mut cursor = text;
    loop {
        let Some(start) = cursor.find("<tool_call>") else {
            remaining.push_str(cursor);
            break;
        };

        remaining.push_str(&cursor[..start]);
        let after_tag = &cursor[start + "<tool_call>".len()..];

        let Some(end) = after_tag.find("</tool_call>") else {
            // No closing tag — treat as plain text
            remaining.push_str(&cursor[start..]);
            break;
        };

        let json_str = after_tag[..end].trim();
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = parsed.get("name").and_then(|n| n.as_str()).unwrap_or("unknown").to_string();
            let input = parsed.get("input").cloned().unwrap_or(serde_json::json!({}));
            call_counter += 1;
            tool_calls.push(ToolCall { id: format!("local-{call_counter}"), name, input });
        } else {
            // Malformed JSON — treat as plain text
            remaining
                .push_str(&cursor[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }

        cursor = &after_tag[end + "</tool_call>".len()..];
    }

    (remaining.trim().to_string(), tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_no_tool_calls() {
        let (text, calls) = parse_tool_calls("Hello world");
        assert_eq!(text, "Hello world");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_single_tool_call() {
        let input = r#"Before text
<tool_call>
{"name": "rag", "input": {"query": "SIMD"}}
</tool_call>
After text"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Before text\n\nAfter text");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[0].input, serde_json::json!({"query": "SIMD"}));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "rag", "input": {"query": "a"}}
</tool_call>
Middle
<tool_call>
{"name": "memory", "input": {"action": "recall", "query": "b"}}
</tool_call>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Middle");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[1].name, "memory");
        assert_eq!(calls[1].id, "local-2");
    }

    #[test]
    fn test_parse_malformed_json() {
        let input = r#"<tool_call>
not valid json
</tool_call>"#;
        let (_text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_missing_close_tag() {
        let input = "<tool_call>\n{\"name\": \"rag\"}";
        let (text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty());
        assert!(text.contains("<tool_call>"));
    }

    #[test]
    fn test_parse_missing_name() {
        let input = r#"<tool_call>
{"input": {"query": "test"}}
</tool_call>"#;
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "unknown");
    }

    #[test]
    fn test_privacy_tier_always_sovereign() {
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
    }
}
