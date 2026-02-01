//! Chat Template Engine
//!
//! Unified prompt templating for different model formats.
//! Implements Toyota Way "Standardized Work" principle.
//!
//! ## Supported Formats
//!
//! - Llama 2: `[INST] {prompt} [/INST]`
//! - Mistral: `[INST] {prompt} [/INST]`
//! - ChatML: `<|im_start|>user\n{prompt}<|im_end|>`
//! - Alpaca: `### Instruction:\n{prompt}\n### Response:`
//! - Vicuna: `USER: {prompt}\nASSISTANT:`
//! - Raw: No formatting (pass-through)

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// SERVE-TPL-001: Chat Template Types
// ============================================================================

/// Chat message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// A chat message with role and content
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Chat template format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TemplateFormat {
    /// Llama 2 format: `[INST] {prompt} [/INST]`
    Llama2,
    /// Mistral format: `[INST] {prompt} [/INST]`
    Mistral,
    /// ChatML format: <|im_start|>role\n{content}<|im_end|>
    ChatML,
    /// Alpaca format: ### Instruction:\n{prompt}\n### Response:
    Alpaca,
    /// Vicuna format: USER: {prompt}\nASSISTANT:
    Vicuna,
    /// Raw pass-through (no formatting)
    #[default]
    Raw,
}

impl TemplateFormat {
    /// Detect template format from model name
    #[must_use]
    pub fn from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("llama-2") || lower.contains("llama2") {
            Self::Llama2
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            Self::Mistral
        } else if lower.contains("chatml") || lower.contains("openhermes") {
            Self::ChatML
        } else if lower.contains("alpaca") {
            Self::Alpaca
        } else if lower.contains("vicuna") {
            Self::Vicuna
        } else {
            Self::Raw
        }
    }
}

// ============================================================================
// SERVE-TPL-002: Chat Template Engine
// ============================================================================

/// Chat template engine for formatting prompts
#[derive(Debug, Clone)]
pub struct ChatTemplateEngine {
    format: TemplateFormat,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplateEngine {
    /// Create a new template engine with the specified format
    #[must_use]
    pub fn new(format: TemplateFormat) -> Self {
        let (bos_token, eos_token) = match format {
            TemplateFormat::Llama2 => (Some("<s>".to_string()), Some("</s>".to_string())),
            TemplateFormat::Mistral => (Some("<s>".to_string()), Some("</s>".to_string())),
            TemplateFormat::ChatML => (None, None),
            TemplateFormat::Alpaca => (None, None),
            TemplateFormat::Vicuna => (None, None),
            TemplateFormat::Raw => (None, None),
        };
        Self {
            format,
            bos_token,
            eos_token,
        }
    }

    /// Create engine from model name (auto-detect format)
    #[must_use]
    pub fn from_model(model_name: &str) -> Self {
        Self::new(TemplateFormat::from_model_name(model_name))
    }

    /// Get the template format
    #[must_use]
    pub fn format(&self) -> TemplateFormat {
        self.format
    }

    /// Apply template to a list of chat messages
    #[must_use]
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self.format {
            TemplateFormat::Llama2 => self.apply_llama2(messages),
            TemplateFormat::Mistral => self.apply_mistral(messages),
            TemplateFormat::ChatML => self.apply_chatml(messages),
            TemplateFormat::Alpaca => self.apply_alpaca(messages),
            TemplateFormat::Vicuna => self.apply_vicuna(messages),
            TemplateFormat::Raw => self.apply_raw(messages),
        }
    }

    /// Apply template to a simple prompt string
    #[must_use]
    pub fn apply_prompt(&self, prompt: &str) -> String {
        self.apply(&[ChatMessage::user(prompt)])
    }

    // Llama 2 format
    fn apply_llama2(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        // Add BOS token
        if let Some(ref bos) = self.bos_token {
            result.push_str(bos);
        }

        let mut system_prompt = None;
        for msg in messages {
            match msg.role {
                Role::System => {
                    system_prompt = Some(&msg.content);
                }
                Role::User => {
                    result.push_str("[INST] ");
                    if let Some(sys) = system_prompt.take() {
                        result.push_str("<<SYS>>\n");
                        result.push_str(sys);
                        result.push_str("\n<</SYS>>\n\n");
                    }
                    result.push_str(&msg.content);
                    result.push_str(" [/INST]");
                }
                Role::Assistant => {
                    result.push(' ');
                    result.push_str(&msg.content);
                    if let Some(ref eos) = self.eos_token {
                        result.push_str(eos);
                    }
                }
            }
        }

        result
    }

    // Mistral format (similar to Llama 2 but without <<SYS>> tags)
    fn apply_mistral(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        if let Some(ref bos) = self.bos_token {
            result.push_str(bos);
        }

        for msg in messages {
            match msg.role {
                Role::System => {
                    // Mistral prepends system to first user message
                    result.push_str("[INST] ");
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                Role::User => {
                    if !result.contains("[INST]") {
                        result.push_str("[INST] ");
                    }
                    result.push_str(&msg.content);
                    result.push_str(" [/INST]");
                }
                Role::Assistant => {
                    result.push_str(&msg.content);
                    if let Some(ref eos) = self.eos_token {
                        result.push_str(eos);
                    }
                }
            }
        }

        result
    }

    // ChatML format
    fn apply_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        for msg in messages {
            result.push_str("<|im_start|>");
            result.push_str(&msg.role.to_string());
            result.push('\n');
            result.push_str(&msg.content);
            result.push_str("<|im_end|>\n");
        }

        // Add assistant prompt
        result.push_str("<|im_start|>assistant\n");

        result
    }

    // Alpaca format
    fn apply_alpaca(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                Role::User => {
                    result.push_str("### Instruction:\n");
                    result.push_str(&msg.content);
                    result.push_str("\n\n### Response:\n");
                }
                Role::Assistant => {
                    result.push_str(&msg.content);
                    result.push('\n');
                }
            }
        }

        result
    }

    // Vicuna format
    fn apply_vicuna(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                Role::User => {
                    result.push_str("USER: ");
                    result.push_str(&msg.content);
                    result.push_str("\nASSISTANT:");
                }
                Role::Assistant => {
                    result.push(' ');
                    result.push_str(&msg.content);
                    result.push('\n');
                }
            }
        }

        result
    }

    // Raw format (no transformation)
    fn apply_raw(&self, messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for ChatTemplateEngine {
    fn default() -> Self {
        Self::new(TemplateFormat::Raw)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // SERVE-TPL-001: Role and ChatMessage Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_001_role_display() {
        assert_eq!(format!("{}", Role::System), "system");
        assert_eq!(format!("{}", Role::User), "user");
        assert_eq!(format!("{}", Role::Assistant), "assistant");
    }

    #[test]
    fn test_SERVE_TPL_001_chat_message_system() {
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "You are a helpful assistant.");
    }

    #[test]
    fn test_SERVE_TPL_001_chat_message_user() {
        let msg = ChatMessage::user("Hello!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn test_SERVE_TPL_001_chat_message_assistant() {
        let msg = ChatMessage::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there!");
    }

    // ========================================================================
    // SERVE-TPL-002: Template Format Detection Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_002_detect_llama2() {
        assert_eq!(
            TemplateFormat::from_model_name("meta-llama/Llama-2-7b"),
            TemplateFormat::Llama2
        );
        assert_eq!(
            TemplateFormat::from_model_name("llama2-13b"),
            TemplateFormat::Llama2
        );
    }

    #[test]
    fn test_SERVE_TPL_002_detect_mistral() {
        assert_eq!(
            TemplateFormat::from_model_name("mistralai/Mistral-7B"),
            TemplateFormat::Mistral
        );
        assert_eq!(
            TemplateFormat::from_model_name("mixtral-8x7b"),
            TemplateFormat::Mistral
        );
    }

    #[test]
    fn test_SERVE_TPL_002_detect_chatml() {
        assert_eq!(
            TemplateFormat::from_model_name("OpenHermes-2.5"),
            TemplateFormat::ChatML
        );
        assert_eq!(
            TemplateFormat::from_model_name("chatml-model"),
            TemplateFormat::ChatML
        );
    }

    #[test]
    fn test_SERVE_TPL_002_detect_alpaca() {
        assert_eq!(
            TemplateFormat::from_model_name("alpaca-7b"),
            TemplateFormat::Alpaca
        );
    }

    #[test]
    fn test_SERVE_TPL_002_detect_vicuna() {
        assert_eq!(
            TemplateFormat::from_model_name("vicuna-13b"),
            TemplateFormat::Vicuna
        );
    }

    #[test]
    fn test_SERVE_TPL_002_detect_raw_fallback() {
        assert_eq!(
            TemplateFormat::from_model_name("unknown-model"),
            TemplateFormat::Raw
        );
    }

    // ========================================================================
    // SERVE-TPL-003: Llama 2 Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_003_llama2_simple() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Llama2);
        let result = engine.apply_prompt("Hello!");
        assert!(result.contains("[INST]"));
        assert!(result.contains("[/INST]"));
        assert!(result.contains("Hello!"));
    }

    #[test]
    fn test_SERVE_TPL_003_llama2_with_system() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Llama2);
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi!"),
        ];
        let result = engine.apply(&messages);
        assert!(result.contains("<<SYS>>"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<</SYS>>"));
        assert!(result.contains("Hi!"));
    }

    #[test]
    fn test_SERVE_TPL_003_llama2_bos_token() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Llama2);
        let result = engine.apply_prompt("Test");
        assert!(result.starts_with("<s>"));
    }

    // ========================================================================
    // SERVE-TPL-004: Mistral Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_004_mistral_simple() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Mistral);
        let result = engine.apply_prompt("Hello!");
        assert!(result.contains("[INST]"));
        assert!(result.contains("[/INST]"));
    }

    #[test]
    fn test_SERVE_TPL_004_mistral_no_sys_tags() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Mistral);
        let messages = vec![ChatMessage::system("Be helpful."), ChatMessage::user("Hi!")];
        let result = engine.apply(&messages);
        // Mistral doesn't use <<SYS>> tags
        assert!(!result.contains("<<SYS>>"));
    }

    // ========================================================================
    // SERVE-TPL-005: ChatML Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_005_chatml_simple() {
        let engine = ChatTemplateEngine::new(TemplateFormat::ChatML);
        let result = engine.apply_prompt("Hello!");
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_SERVE_TPL_005_chatml_with_system() {
        let engine = ChatTemplateEngine::new(TemplateFormat::ChatML);
        let messages = vec![
            ChatMessage::system("You are an AI."),
            ChatMessage::user("Hi!"),
        ];
        let result = engine.apply(&messages);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are an AI."));
    }

    // ========================================================================
    // SERVE-TPL-006: Alpaca Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_006_alpaca_simple() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Alpaca);
        let result = engine.apply_prompt("What is 2+2?");
        assert!(result.contains("### Instruction:"));
        assert!(result.contains("### Response:"));
        assert!(result.contains("What is 2+2?"));
    }

    // ========================================================================
    // SERVE-TPL-007: Vicuna Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_007_vicuna_simple() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Vicuna);
        let result = engine.apply_prompt("Hello!");
        assert!(result.contains("USER:"));
        assert!(result.contains("ASSISTANT:"));
    }

    // ========================================================================
    // SERVE-TPL-008: Raw Template Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_008_raw_passthrough() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Raw);
        let result = engine.apply_prompt("Hello!");
        assert_eq!(result, "Hello!");
    }

    #[test]
    fn test_SERVE_TPL_008_raw_multiple_messages() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Raw);
        let messages = vec![ChatMessage::user("A"), ChatMessage::user("B")];
        let result = engine.apply(&messages);
        assert_eq!(result, "A\nB");
    }

    // ========================================================================
    // SERVE-TPL-009: Engine Factory Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_009_from_model() {
        let engine = ChatTemplateEngine::from_model("meta-llama/Llama-2-7b-chat");
        assert_eq!(engine.format(), TemplateFormat::Llama2);
    }

    #[test]
    fn test_SERVE_TPL_009_default() {
        let engine = ChatTemplateEngine::default();
        assert_eq!(engine.format(), TemplateFormat::Raw);
    }

    // ========================================================================
    // SERVE-TPL-010: Multi-turn Conversation Tests
    // ========================================================================

    #[test]
    fn test_SERVE_TPL_010_llama2_multiturn() {
        let engine = ChatTemplateEngine::new(TemplateFormat::Llama2);
        let messages = vec![
            ChatMessage::user("Hi!"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let result = engine.apply(&messages);
        // Should have multiple [INST] blocks
        assert!(result.matches("[INST]").count() >= 2);
    }

    #[test]
    fn test_SERVE_TPL_010_chatml_multiturn() {
        let engine = ChatTemplateEngine::new(TemplateFormat::ChatML);
        let messages = vec![
            ChatMessage::user("Hi!"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let result = engine.apply(&messages);
        // Should have multiple im_start tags
        assert!(result.matches("<|im_start|>").count() >= 3);
    }
}
