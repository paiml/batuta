//! Chat template formatting for local LLM inference.
//!
//! Different model families require different prompt formats.
//! Auto-detects the template from the model filename:
//! - Qwen, DeepSeek, Yi → ChatML
//! - Llama → Llama 3.x
//! - Unknown → ChatML (most widely supported)
//!
//! See: apr-code.md §5.1

use super::{CompletionRequest, Message};

/// Chat template family, auto-detected from model filename.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>` (Qwen, Yi, Deepseek)
    ChatMl,
    /// Llama 3.x: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
    Llama3,
    /// Generic: `<|system|>\ncontent\n<|end|>` (fallback)
    Generic,
}

impl ChatTemplate {
    /// Detect template from model filename.
    pub(crate) fn from_model_path(path: &std::path::Path) -> Self {
        let name = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();

        if name.contains("qwen") || name.contains("deepseek") || name.contains("yi-") {
            Self::ChatMl
        } else if name.contains("llama") {
            Self::Llama3
        } else {
            Self::ChatMl
        }
    }
}

/// Format messages using a specific chat template.
pub(crate) fn format_prompt_with_template(
    request: &CompletionRequest,
    template: ChatTemplate,
) -> String {
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
            Message::System(s) => {
                prompt.push_str(&format!("<|im_start|>system\n{s}<|im_end|>\n"));
            }
            Message::User(s) => {
                prompt.push_str(&format!("<|im_start|>user\n{s}<|im_end|>\n"));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!("<|im_start|>assistant\n{s}<|im_end|>\n"));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|im_start|>assistant\n<tool_call>\n{}\n</tool_call><|im_end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
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
        prompt
            .push_str(&format!("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"));
    }

    for msg in &request.messages {
        match msg {
            Message::System(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::User(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n{s}<|eot_id|>"
                ));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n<tool_call>\n{}\n</tool_call><|eot_id|>",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
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
            Message::System(s) => {
                prompt.push_str(&format!("<|system|>\n{s}\n<|end|>\n"));
            }
            Message::User(s) => {
                prompt.push_str(&format!("<|user|>\n{s}\n<|end|>\n"));
            }
            Message::Assistant(s) => {
                prompt.push_str(&format!("<|assistant|>\n{s}\n<|end|>\n"));
            }
            Message::AssistantToolUse(call) => {
                prompt.push_str(&format!(
                    "<|assistant|>\n<tool_call>\n{}\n</tool_call>\n<|end|>\n",
                    serde_json::json!({"name": call.name, "input": call.input})
                ));
            }
            Message::ToolResult(result) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::driver::ToolCall;

    #[test]
    fn test_format_prompt_chatml() {
        let request = CompletionRequest {
            model: "test".into(),
            messages: vec![Message::User("Hello".into())],
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
            messages: vec![Message::User("Hello".into())],
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
            messages: vec![Message::User("Hello".into())],
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
                Message::AssistantToolUse(ToolCall {
                    id: "1".into(),
                    name: "rag".into(),
                    input: serde_json::json!({"query": "test"}),
                }),
                Message::ToolResult(crate::agent::driver::ToolResultMsg {
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
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("qwen2.5-coder-7b.gguf")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("Qwen3-8B-Q4K.apr")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("deepseek-coder-v2.gguf")),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("llama-3.2-3b.gguf")),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("Meta-Llama-3-8B.apr")),
            ChatTemplate::Llama3
        );
        assert_eq!(ChatTemplate::from_model_path(Path::new("yi-34b.gguf")), ChatTemplate::ChatMl);
        assert_eq!(
            ChatTemplate::from_model_path(Path::new("custom-model.gguf")),
            ChatTemplate::ChatMl
        );
    }
}
