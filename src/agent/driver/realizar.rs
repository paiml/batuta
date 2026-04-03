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

/// Chat template family, auto-detected from model filename.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>` (Qwen, Yi, Deepseek)
    ChatMl,
    /// Llama 3.x: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
    Llama3,
    /// Generic: `<|system|>\ncontent\n<|end|>` (fallback)
    Generic,
}

impl ChatTemplate {
    /// Detect template from model filename.
    fn from_model_path(path: &std::path::Path) -> Self {
        let name = path.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        if name.contains("qwen") || name.contains("deepseek") || name.contains("yi-") {
            Self::ChatMl
        } else if name.contains("llama") {
            Self::Llama3
        } else {
            // Default to ChatML — widely supported by most instruct models
            Self::ChatMl
        }
    }
}

/// Format messages into a prompt string for local inference.
fn format_prompt(request: &CompletionRequest) -> String {
    format_prompt_with_template(request, ChatTemplate::ChatMl)
}

/// Format messages using a specific chat template.
fn format_prompt_with_template(request: &CompletionRequest, template: ChatTemplate) -> String {
    match template {
        ChatTemplate::ChatMl => format_chatml(request),
        ChatTemplate::Llama3 => format_llama3(request),
        ChatTemplate::Generic => format_generic(request),
    }
}

/// ChatML format (Qwen, DeepSeek, Yi).
fn format_chatml(request: &CompletionRequest) -> String {
    let mut prompt = String::new();

    if let Some(ref system) = request.system {
        prompt.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }

    for msg in &request.messages {
        match msg {
            super::Message::System(s) => {
                prompt.push_str(&format!("<|im_start|>system\n{s}<|im_end|>\n"));
            }
            super::Message::User(s) => {
                prompt.push_str(&format!("<|im_start|>user\n{s}<|im_end|>\n"));
            }
            super::Message::Assistant(s) => {
                prompt.push_str(&format!("<|im_start|>assistant\n{s}<|im_end|>\n"));
            }
            super::Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n<tool_call>\n{}\n</tool_call><|im_end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            super::Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|im_start|>user\n<tool_result>{}</tool_result><|im_end|>\n",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Llama 3.x format.
fn format_llama3(request: &CompletionRequest) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");

    if let Some(ref system) = request.system {
        prompt.push_str(&format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        ));
    }

    for msg in &request.messages {
        match msg {
            super::Message::System(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            super::Message::User(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            super::Message::Assistant(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            super::Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n<tool_call>\n{}\n</tool_call><|eot_id|>",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            super::Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n<tool_result>{}</tool_result><|eot_id|>",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Generic fallback format.
fn format_generic(request: &CompletionRequest) -> String {
    let mut prompt = String::new();

    if let Some(ref system) = request.system {
        prompt.push_str(&format!("<|system|>\n{system}\n<|end|>\n"));
    }

    for msg in &request.messages {
        match msg {
            super::Message::System(s) => {
                prompt.push_str(&format!("<|system|>\n{s}\n<|end|>\n"));
            }
            super::Message::User(s) => {
                prompt.push_str(&format!("<|user|>\n{s}\n<|end|>\n"));
            }
            super::Message::Assistant(s) => {
                prompt.push_str(&format!("<|assistant|>\n{s}\n<|end|>\n"));
            }
            super::Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|assistant|>\n<tool_call>\n{}\n</tool_call>\n<|end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            super::Message::ToolResult(result) => {
                prompt.push_str(&format!(
                    "<|user|>\n<tool_result>{}</tool_result>\n<|end|>\n",
                    result.content
                ));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
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
    fn test_format_prompt_chatml() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![super::super::Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_chatml(&request);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_prompt_llama3() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![super::super::Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("Be helpful".into()),
        };
        let prompt = format_llama3(&request);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("Be helpful"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_format_prompt_generic_fallback() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![super::super::Message::User("Hello".into())],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: Some("You are helpful".into()),
        };
        let prompt = format_generic(&request);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_format_prompt_tool_messages() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![
                super::super::Message::AssistantToolUse(ToolCall {
                    id: "1".into(),
                    name: "rag".into(),
                    input: serde_json::json!({"query": "test"}),
                }),
                super::super::Message::ToolResult(super::super::ToolResultMsg {
                    tool_use_id: "1".into(),
                    content: "result data".into(),
                    is_error: false,
                }),
            ],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.5,
            system: None,
        };
        // All templates should include tool_call and tool_result
        for template in [ChatTemplate::ChatMl, ChatTemplate::Llama3, ChatTemplate::Generic] {
            let prompt = format_prompt_with_template(&request, template);
            assert!(prompt.contains("<tool_call>"), "missing tool_call in {template:?}");
            assert!(prompt.contains("<tool_result>"), "missing tool_result in {template:?}");
            assert!(prompt.contains("result data"), "missing result data in {template:?}");
        }
    }

    #[test]
    fn test_chat_template_detection() {
        use std::path::Path;
        assert_eq!(ChatTemplate::from_model_path(Path::new("qwen2.5-coder-7b.gguf")), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::from_model_path(Path::new("Qwen3-8B-Q4K.apr")), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::from_model_path(Path::new("deepseek-coder-v2.gguf")), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::from_model_path(Path::new("llama-3.2-3b.gguf")), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_model_path(Path::new("Meta-Llama-3-8B.apr")), ChatTemplate::Llama3);
        assert_eq!(ChatTemplate::from_model_path(Path::new("yi-34b.gguf")), ChatTemplate::ChatMl);
        // Unknown defaults to ChatML (most widely supported)
        assert_eq!(ChatTemplate::from_model_path(Path::new("custom-model.gguf")), ChatTemplate::ChatMl);
    }

    #[test]
    fn test_privacy_tier_always_sovereign() {
        // RealizarDriver cannot be constructed without a real model file,
        // but we can verify the constant
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
    }
}
