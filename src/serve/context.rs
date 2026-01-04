//! Context Window Management
//!
//! Automatic token counting and context truncation.
//! Prevents silent failures when prompts exceed model context limits.

use crate::serve::templates::ChatMessage;
use serde::{Deserialize, Serialize};

// ============================================================================
// SERVE-CTX-001: Context Configuration
// ============================================================================

/// Known model context window sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Maximum context size in tokens
    pub max_tokens: usize,
    /// Reserved tokens for output
    pub output_reserve: usize,
}

impl ContextWindow {
    /// Create a new context window configuration
    #[must_use]
    pub const fn new(max_tokens: usize, output_reserve: usize) -> Self {
        Self {
            max_tokens,
            output_reserve,
        }
    }

    /// Available tokens for input after reserving output space
    #[must_use]
    pub const fn available_input(&self) -> usize {
        self.max_tokens.saturating_sub(self.output_reserve)
    }

    /// Get context window for known model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let lower = model.to_lowercase();

        // GPT-4 variants
        if lower.contains("gpt-4-turbo") || lower.contains("gpt-4o") {
            Self::new(128_000, 4096)
        } else if lower.contains("gpt-4-32k") {
            Self::new(32_768, 4096)
        } else if lower.contains("gpt-4") {
            Self::new(8_192, 2048)
        }
        // GPT-3.5
        else if lower.contains("gpt-3.5-turbo-16k") {
            Self::new(16_384, 4096)
        } else if lower.contains("gpt-3.5") {
            Self::new(4_096, 1024)
        }
        // Claude
        else if lower.contains("claude-3") || lower.contains("claude-2") {
            Self::new(200_000, 4096)
        } else if lower.contains("claude") {
            Self::new(100_000, 4096)
        }
        // Llama variants
        else if lower.contains("llama-3") {
            Self::new(8_192, 2048)
        } else if lower.contains("llama-2") && lower.contains("32k") {
            Self::new(32_768, 4096)
        } else if lower.contains("llama") {
            Self::new(4_096, 1024)
        }
        // Mistral
        else if lower.contains("mixtral") {
            Self::new(32_768, 4096)
        } else if lower.contains("mistral") {
            Self::new(8_192, 2048)
        }
        // Default conservative estimate
        else {
            Self::new(4_096, 1024)
        }
    }
}

impl Default for ContextWindow {
    fn default() -> Self {
        Self::new(4_096, 1024)
    }
}

// ============================================================================
// SERVE-CTX-002: Token Estimation
// ============================================================================

/// Simple token estimator (approximation without full tokenizer)
///
/// Uses heuristic: ~4 characters per token for English text.
/// For accurate counts, use a proper tokenizer.
pub struct TokenEstimator {
    /// Characters per token (default: 4.0)
    chars_per_token: f64,
}

impl TokenEstimator {
    /// Create with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }

    /// Create with custom chars-per-token ratio
    #[must_use]
    pub fn with_ratio(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }

    /// Estimate token count for a string
    #[must_use]
    pub fn estimate(&self, text: &str) -> usize {
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    /// Estimate tokens for chat messages
    #[must_use]
    pub fn estimate_messages(&self, messages: &[ChatMessage]) -> usize {
        let mut total = 0;
        for msg in messages {
            // Role tokens (approximately 3-4 tokens per message for formatting)
            total += 4;
            total += self.estimate(&msg.content);
        }
        total
    }
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SERVE-CTX-003: Context Manager
// ============================================================================

/// Truncation strategy when context is exceeded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Remove oldest messages first (sliding window)
    #[default]
    SlidingWindow,
    /// Remove from the middle, keep first and last
    MiddleOut,
    /// Fail with error instead of truncating
    Error,
}

/// Context management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Context window settings
    pub window: ContextWindow,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
    /// Always preserve system message
    pub preserve_system: bool,
    /// Minimum messages to keep
    pub min_messages: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            window: ContextWindow::default(),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: true,
            min_messages: 2,
        }
    }
}

impl ContextConfig {
    /// Create config for a specific model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self {
            window: ContextWindow::for_model(model),
            ..Default::default()
        }
    }
}

/// Context manager for handling token limits
pub struct ContextManager {
    config: ContextConfig,
    estimator: TokenEstimator,
}

impl ContextManager {
    /// Create a new context manager
    #[must_use]
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            estimator: TokenEstimator::new(),
        }
    }

    /// Create for a specific model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self::new(ContextConfig::for_model(model))
    }

    /// Check if messages fit within context window
    #[must_use]
    pub fn fits(&self, messages: &[ChatMessage]) -> bool {
        let tokens = self.estimator.estimate_messages(messages);
        tokens <= self.config.window.available_input()
    }

    /// Get estimated token count for messages
    #[must_use]
    pub fn estimate_tokens(&self, messages: &[ChatMessage]) -> usize {
        self.estimator.estimate_messages(messages)
    }

    /// Get available token budget
    #[must_use]
    pub fn available_tokens(&self) -> usize {
        self.config.window.available_input()
    }

    /// Truncate messages to fit within context window
    ///
    /// Returns truncated messages or error if strategy is `Error` and truncation needed.
    pub fn truncate(&self, messages: &[ChatMessage]) -> Result<Vec<ChatMessage>, ContextError> {
        let available = self.config.window.available_input();
        let current = self.estimator.estimate_messages(messages);

        if current <= available {
            return Ok(messages.to_vec());
        }

        match self.config.strategy {
            TruncationStrategy::Error => Err(ContextError::ExceedsLimit {
                tokens: current,
                limit: available,
            }),
            TruncationStrategy::SlidingWindow => {
                Ok(self.truncate_sliding_window(messages, available))
            }
            TruncationStrategy::MiddleOut => Ok(self.truncate_middle_out(messages, available)),
        }
    }

    fn truncate_sliding_window(
        &self,
        messages: &[ChatMessage],
        available: usize,
    ) -> Vec<ChatMessage> {
        let mut result = Vec::new();
        let mut tokens_used = 0;

        // Extract system message if preserving
        let (system_msg, other_msgs): (Vec<_>, Vec<_>) = if self.config.preserve_system {
            messages
                .iter()
                .partition(|m| matches!(m.role, crate::serve::templates::Role::System))
        } else {
            (vec![], messages.iter().collect())
        };

        // Add system message first
        for msg in &system_msg {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                result.push((*msg).clone());
                tokens_used += msg_tokens;
            }
        }

        // Add messages from the end (most recent first)
        let mut recent_msgs: Vec<ChatMessage> = Vec::new();
        for msg in other_msgs.into_iter().rev() {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                recent_msgs.push(msg.clone());
                tokens_used += msg_tokens;
            } else if recent_msgs.len() >= self.config.min_messages {
                break;
            }
        }

        // Reverse to restore chronological order
        recent_msgs.reverse();
        result.extend(recent_msgs);

        result
    }

    fn truncate_middle_out(&self, messages: &[ChatMessage], available: usize) -> Vec<ChatMessage> {
        if messages.len() <= 2 {
            return messages.to_vec();
        }

        let mut result = Vec::new();
        let mut tokens_used = 0;

        // Always keep first message (often system)
        let first = &messages[0];
        let first_tokens = self.estimator.estimate(&first.content) + 4;
        result.push(first.clone());
        tokens_used += first_tokens;

        // Always keep last message
        let last = &messages[messages.len() - 1];
        let last_tokens = self.estimator.estimate(&last.content) + 4;
        tokens_used += last_tokens;

        // Add messages from the end, working backwards
        let middle = &messages[1..messages.len() - 1];
        let mut kept_from_end: Vec<ChatMessage> = Vec::new();

        for msg in middle.iter().rev() {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                kept_from_end.push(msg.clone());
                tokens_used += msg_tokens;
            } else {
                break;
            }
        }

        // Reverse and add
        kept_from_end.reverse();
        result.extend(kept_from_end);
        result.push(last.clone());

        result
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new(ContextConfig::default())
    }
}

/// Context management errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// Context window exceeded and strategy is Error
    ExceedsLimit { tokens: usize, limit: usize },
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExceedsLimit { tokens, limit } => {
                write!(
                    f,
                    "Context exceeds limit: {} tokens, max {} tokens",
                    tokens, limit
                )
            }
        }
    }
}

impl std::error::Error for ContextError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::serve::templates::ChatMessage;

    // ========================================================================
    // SERVE-CTX-001: Context Window Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_001_context_window_new() {
        let window = ContextWindow::new(8192, 2048);
        assert_eq!(window.max_tokens, 8192);
        assert_eq!(window.output_reserve, 2048);
    }

    #[test]
    fn test_SERVE_CTX_001_available_input() {
        let window = ContextWindow::new(8192, 2048);
        assert_eq!(window.available_input(), 6144);
    }

    #[test]
    fn test_SERVE_CTX_001_for_model_gpt4() {
        let window = ContextWindow::for_model("gpt-4-turbo");
        assert_eq!(window.max_tokens, 128_000);
    }

    #[test]
    fn test_SERVE_CTX_001_for_model_claude() {
        let window = ContextWindow::for_model("claude-3-sonnet");
        assert_eq!(window.max_tokens, 200_000);
    }

    #[test]
    fn test_SERVE_CTX_001_for_model_llama() {
        let window = ContextWindow::for_model("llama-2-7b");
        assert_eq!(window.max_tokens, 4_096);
    }

    #[test]
    fn test_SERVE_CTX_001_for_model_mistral() {
        let window = ContextWindow::for_model("mistral-7b");
        assert_eq!(window.max_tokens, 8_192);
    }

    #[test]
    fn test_SERVE_CTX_001_default() {
        let window = ContextWindow::default();
        assert_eq!(window.max_tokens, 4_096);
    }

    // ========================================================================
    // SERVE-CTX-002: Token Estimator Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_002_estimate_simple() {
        let estimator = TokenEstimator::new();
        // 20 chars / 4 = 5 tokens
        assert_eq!(estimator.estimate("Hello, how are you?"), 5);
    }

    #[test]
    fn test_SERVE_CTX_002_estimate_empty() {
        let estimator = TokenEstimator::new();
        assert_eq!(estimator.estimate(""), 0);
    }

    #[test]
    fn test_SERVE_CTX_002_estimate_long_text() {
        let estimator = TokenEstimator::new();
        let text = "a".repeat(1000);
        assert_eq!(estimator.estimate(&text), 250);
    }

    #[test]
    fn test_SERVE_CTX_002_estimate_messages() {
        let estimator = TokenEstimator::new();
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let tokens = estimator.estimate_messages(&messages);
        // Each message: ~4 formatting tokens + content tokens
        assert!(tokens > 0);
    }

    #[test]
    fn test_SERVE_CTX_002_custom_ratio() {
        let estimator = TokenEstimator::with_ratio(3.0);
        // 12 chars / 3 = 4 tokens
        assert_eq!(estimator.estimate("Hello World!"), 4);
    }

    // ========================================================================
    // SERVE-CTX-003: Context Manager Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_003_fits_under_limit() {
        let manager = ContextManager::for_model("gpt-4-turbo");
        let messages = vec![ChatMessage::user("Hello!")];
        assert!(manager.fits(&messages));
    }

    #[test]
    fn test_SERVE_CTX_003_estimate_tokens() {
        let manager = ContextManager::default();
        let messages = vec![ChatMessage::user("Test message")];
        let tokens = manager.estimate_tokens(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_SERVE_CTX_003_truncate_not_needed() {
        let manager = ContextManager::for_model("gpt-4-turbo");
        let messages = vec![ChatMessage::user("Hello!")];
        let result = manager.truncate(&messages).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_SERVE_CTX_003_truncate_error_strategy() {
        let config = ContextConfig {
            window: ContextWindow::new(10, 0), // Very small window
            strategy: TruncationStrategy::Error,
            ..Default::default()
        };
        let manager = ContextManager::new(config);
        let messages = vec![ChatMessage::user(
            "This is a longer message that exceeds limit",
        )];
        let result = manager.truncate(&messages);
        assert!(result.is_err());
    }

    // ========================================================================
    // SERVE-CTX-004: Sliding Window Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_004_sliding_window_truncation() {
        let config = ContextConfig {
            window: ContextWindow::new(100, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First message"),
            ChatMessage::assistant("First response"),
            ChatMessage::user("Second message"),
            ChatMessage::assistant("Second response"),
            ChatMessage::user("Third message - most recent"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // Should keep the most recent messages
        assert!(result.len() < messages.len() || manager.fits(&messages));
    }

    #[test]
    fn test_SERVE_CTX_004_preserves_system() {
        let config = ContextConfig {
            window: ContextWindow::new(200, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: true,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Message 1"),
            ChatMessage::user("Message 2"),
            ChatMessage::user("Message 3"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // First message should be system if it fits
        if !result.is_empty() {
            // System message should be preserved if space allows
            let has_system = result
                .iter()
                .any(|m| matches!(m.role, crate::serve::templates::Role::System));
            // Only check if original had system and result has multiple messages
            if result.len() > 1 {
                assert!(has_system);
            }
        }
    }

    // ========================================================================
    // SERVE-CTX-005: Middle-Out Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_005_middle_out_keeps_first_last() {
        let config = ContextConfig {
            window: ContextWindow::new(150, 0),
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 2,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First"),
            ChatMessage::assistant("Middle 1"),
            ChatMessage::user("Middle 2"),
            ChatMessage::assistant("Last"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // Should keep first and last at minimum
        assert!(result.len() >= 2);
        assert_eq!(result[0].content, "First");
        assert_eq!(result[result.len() - 1].content, "Last");
    }

    // ========================================================================
    // SERVE-CTX-006: Error Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_006_error_display() {
        let err = ContextError::ExceedsLimit {
            tokens: 10000,
            limit: 4096,
        };
        let msg = err.to_string();
        assert!(msg.contains("10000"));
        assert!(msg.contains("4096"));
    }

    // ========================================================================
    // SERVE-CTX-007: Default Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_007_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.strategy, TruncationStrategy::SlidingWindow);
        assert!(config.preserve_system);
    }

    #[test]
    fn test_SERVE_CTX_007_manager_default() {
        let manager = ContextManager::default();
        assert_eq!(manager.available_tokens(), 3072); // 4096 - 1024
    }

    // ========================================================================
    // SERVE-CTX-008: Additional Model Coverage Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_008_for_model_gpt4_32k() {
        let window = ContextWindow::for_model("gpt-4-32k");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_gpt4_base() {
        let window = ContextWindow::for_model("gpt-4");
        assert_eq!(window.max_tokens, 8_192);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_gpt4o() {
        let window = ContextWindow::for_model("gpt-4o-2024");
        assert_eq!(window.max_tokens, 128_000);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_gpt35_16k() {
        let window = ContextWindow::for_model("gpt-3.5-turbo-16k");
        assert_eq!(window.max_tokens, 16_384);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_gpt35() {
        let window = ContextWindow::for_model("gpt-3.5-turbo");
        assert_eq!(window.max_tokens, 4_096);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_claude_v1() {
        let window = ContextWindow::for_model("claude-instant-v1");
        assert_eq!(window.max_tokens, 100_000);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_llama3() {
        let window = ContextWindow::for_model("llama-3-70b");
        assert_eq!(window.max_tokens, 8_192);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_llama2_32k() {
        let window = ContextWindow::for_model("llama-2-32k-instruct");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_mixtral() {
        let window = ContextWindow::for_model("mixtral-8x7b");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_SERVE_CTX_008_for_model_unknown() {
        let window = ContextWindow::for_model("unknown-model-xyz");
        assert_eq!(window.max_tokens, 4_096); // Default
    }

    // ========================================================================
    // SERVE-CTX-009: Token Estimator Default
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_009_estimator_default() {
        let estimator = TokenEstimator::default();
        assert_eq!(estimator.estimate("Test"), 1);
    }

    // ========================================================================
    // SERVE-CTX-010: Edge Cases
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_010_middle_out_short_messages() {
        let config = ContextConfig {
            window: ContextWindow::new(1000, 0),
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        // Only 2 messages - should return both
        let messages = vec![ChatMessage::user("First"), ChatMessage::assistant("Last")];

        let result = manager.truncate(&messages).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_SERVE_CTX_010_middle_out_single_message() {
        let config = ContextConfig {
            window: ContextWindow::new(1000, 0),
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![ChatMessage::user("Only one")];

        let result = manager.truncate(&messages).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_SERVE_CTX_010_sliding_window_no_system() {
        let config = ContextConfig {
            window: ContextWindow::new(50, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First"),
            ChatMessage::user("Second with more content"),
            ChatMessage::user("Third"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // Should keep recent messages
        assert!(!result.is_empty());
    }

    #[test]
    fn test_SERVE_CTX_010_available_input_saturating() {
        // Test saturating_sub doesn't panic
        let window = ContextWindow::new(100, 200);
        assert_eq!(window.available_input(), 0);
    }

    #[test]
    fn test_SERVE_CTX_010_context_error_is_error() {
        // Ensure ContextError implements std::error::Error
        let err = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_SERVE_CTX_010_config_for_model() {
        let config = ContextConfig::for_model("gpt-4-turbo");
        assert_eq!(config.window.max_tokens, 128_000);
        assert!(config.preserve_system);
    }

    // ========================================================================
    // SERVE-CTX-011: Additional Edge Case Coverage
    // ========================================================================

    #[test]
    fn test_SERVE_CTX_011_window_zero_reserve() {
        let window = ContextWindow::new(4096, 0);
        assert_eq!(window.available_input(), 4096);
    }

    #[test]
    fn test_SERVE_CTX_011_window_full_reserve() {
        let window = ContextWindow::new(4096, 4096);
        assert_eq!(window.available_input(), 0);
    }

    #[test]
    fn test_SERVE_CTX_011_truncation_strategy_variants() {
        // Test all variants exist
        let _sliding = TruncationStrategy::SlidingWindow;
        let _middle = TruncationStrategy::MiddleOut;
        let _error = TruncationStrategy::Error;
    }

    #[test]
    fn test_SERVE_CTX_011_all_error_variants() {
        // Test all error variants for display
        let err = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_SERVE_CTX_011_context_manager_no_reserve() {
        let config = ContextConfig {
            window: ContextWindow::new(1000, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);
        assert_eq!(manager.available_tokens(), 1000);
    }

    #[test]
    fn test_SERVE_CTX_011_truncate_single_fits() {
        let config = ContextConfig {
            window: ContextWindow::new(1000, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![ChatMessage::user("Short message")];
        let result = manager.truncate(&messages).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_SERVE_CTX_011_for_model_fallback() {
        // Unknown models get default
        let window = ContextWindow::for_model("some-unknown-model");
        assert_eq!(window.max_tokens, 4_096);
    }

    #[test]
    fn test_SERVE_CTX_011_for_model_mixtral() {
        let window = ContextWindow::for_model("mixtral-8x7b");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_SERVE_CTX_011_for_model_llama() {
        let window = ContextWindow::for_model("llama-some-model");
        assert_eq!(window.max_tokens, 4_096);
    }

    // ========================================================================
    // SERVE-CTX-012: More Model Coverage
    // ========================================================================

    #[test]
    fn test_ctx_cov_001_gpt4_32k() {
        let window = ContextWindow::for_model("gpt-4-32k");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_ctx_cov_002_gpt4_base() {
        let window = ContextWindow::for_model("gpt-4");
        assert_eq!(window.max_tokens, 8_192);
    }

    #[test]
    fn test_ctx_cov_003_gpt4o() {
        let window = ContextWindow::for_model("gpt-4o-mini");
        assert_eq!(window.max_tokens, 128_000);
    }

    #[test]
    fn test_ctx_cov_004_gpt35_16k() {
        let window = ContextWindow::for_model("gpt-3.5-turbo-16k");
        assert_eq!(window.max_tokens, 16_384);
    }

    #[test]
    fn test_ctx_cov_005_gpt35_base() {
        let window = ContextWindow::for_model("gpt-3.5-turbo");
        assert_eq!(window.max_tokens, 4_096);
    }

    #[test]
    fn test_ctx_cov_006_claude2() {
        let window = ContextWindow::for_model("claude-2.1");
        assert_eq!(window.max_tokens, 200_000);
    }

    #[test]
    fn test_ctx_cov_007_claude_base() {
        let window = ContextWindow::for_model("claude-instant");
        assert_eq!(window.max_tokens, 100_000);
    }

    #[test]
    fn test_ctx_cov_008_llama3() {
        let window = ContextWindow::for_model("llama-3-8b");
        assert_eq!(window.max_tokens, 8_192);
    }

    #[test]
    fn test_ctx_cov_009_llama2_32k() {
        let window = ContextWindow::for_model("llama-2-70b-32k");
        assert_eq!(window.max_tokens, 32_768);
    }

    #[test]
    fn test_ctx_cov_010_mistral_base() {
        let window = ContextWindow::for_model("mistral-large");
        assert_eq!(window.max_tokens, 8_192);
    }

    // ========================================================================
    // SERVE-CTX-013: Truncation Edge Cases
    // ========================================================================

    #[test]
    fn test_ctx_cov_011_middle_out_two_messages() {
        let config = ContextConfig {
            window: ContextWindow::new(100, 0),
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![ChatMessage::user("First"), ChatMessage::assistant("Second")];
        let result = manager.truncate(&messages).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ctx_cov_012_middle_out_many_messages() {
        let config = ContextConfig {
            window: ContextWindow::new(150, 0),
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First message"),
            ChatMessage::assistant("Response 1"),
            ChatMessage::user("Question 2"),
            ChatMessage::assistant("Response 2"),
            ChatMessage::user("Final message"),
        ];
        let result = manager.truncate(&messages).unwrap();
        // Should keep first and last at minimum
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_ctx_cov_013_sliding_with_system() {
        let config = ContextConfig {
            window: ContextWindow::new(200, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: true,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::system("You are a helpful assistant"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];
        let result = manager.truncate(&messages).unwrap();
        // System should be preserved
        assert!(result
            .iter()
            .any(|m| matches!(m.role, crate::serve::templates::Role::System)));
    }

    #[test]
    fn test_ctx_cov_014_sliding_no_preserve_system() {
        let config = ContextConfig {
            window: ContextWindow::new(50, 0),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::system("You are a helpful assistant"),
            ChatMessage::user("Hello there! This is a longer message."),
        ];
        let result = manager.truncate(&messages).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_ctx_cov_015_estimator_default() {
        let estimator = TokenEstimator::default();
        assert_eq!(estimator.estimate("test"), 1);
    }

    #[test]
    fn test_ctx_cov_016_context_window_serialize() {
        let window = ContextWindow::new(8192, 2048);
        let json = serde_json::to_string(&window).unwrap();
        let deserialized: ContextWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_tokens, window.max_tokens);
    }

    #[test]
    fn test_ctx_cov_017_truncation_strategy_serialize() {
        let strategy = TruncationStrategy::MiddleOut;
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: TruncationStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, strategy);
    }

    #[test]
    fn test_ctx_cov_018_context_config_serialize() {
        let config = ContextConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ContextConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.min_messages, config.min_messages);
    }

    #[test]
    fn test_ctx_cov_019_context_error_eq() {
        let err1 = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let err2 = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_ctx_cov_020_context_error_clone() {
        let err = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_ctx_cov_021_context_error_debug() {
        let err = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("ExceedsLimit"));
    }

    #[test]
    fn test_ctx_cov_022_window_saturating_sub() {
        // Test case where reserve > max
        let window = ContextWindow::new(100, 200);
        assert_eq!(window.available_input(), 0);
    }

    #[test]
    fn test_ctx_cov_023_empty_messages() {
        let manager = ContextManager::default();
        let messages: Vec<ChatMessage> = vec![];
        let result = manager.truncate(&messages).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_ctx_cov_024_context_manager_default_trait() {
        let manager = ContextManager::default();
        assert!(manager.available_tokens() > 0);
    }

    // ========================================================================
    // SERVE-CTX-014: More Coverage Tests
    // ========================================================================

    #[test]
    fn test_ctx_cov_025_sliding_window_min_messages_break() {
        // Test the `else if recent_msgs.len() >= self.config.min_messages` branch
        let config = ContextConfig {
            window: ContextWindow::new(40, 0), // Very small window
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: false,
            min_messages: 1, // Only need 1 message minimum
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First message that is quite long"),
            ChatMessage::assistant("Response 1 also quite long"),
            ChatMessage::user("Question 2 with more content"),
            ChatMessage::assistant("Response 2 with content"),
            ChatMessage::user("Final"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // Should have at least min_messages
        assert!(result.len() >= 1);
    }

    #[test]
    fn test_ctx_cov_026_middle_out_break_branch() {
        // Test the `else { break; }` branch in truncate_middle_out
        let config = ContextConfig {
            window: ContextWindow::new(60, 0), // Tight window
            strategy: TruncationStrategy::MiddleOut,
            preserve_system: false,
            min_messages: 1,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::user("First"),
            ChatMessage::assistant("Middle message that is long"),
            ChatMessage::user("Another middle msg"),
            ChatMessage::assistant("More middle content"),
            ChatMessage::user("Last"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // Should keep first and last at minimum
        assert!(result.len() >= 2);
        assert_eq!(result[0].content, "First");
        assert_eq!(result.last().unwrap().content, "Last");
    }

    #[test]
    fn test_ctx_cov_027_sliding_window_system_too_large() {
        // Test when system message alone exceeds available tokens
        let config = ContextConfig {
            window: ContextWindow::new(10, 0), // Very small
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: true,
            min_messages: 0,
        };
        let manager = ContextManager::new(config);

        let messages = vec![
            ChatMessage::system("This is a very long system message that exceeds the token limit"),
            ChatMessage::user("Hello"),
        ];

        let result = manager.truncate(&messages).unwrap();
        // System message too large, should not be included
        assert!(result.is_empty() || result.len() <= 1);
    }

    #[test]
    fn test_ctx_cov_028_truncation_strategy_default() {
        let strategy = TruncationStrategy::default();
        assert_eq!(strategy, TruncationStrategy::SlidingWindow);
    }

    #[test]
    fn test_ctx_cov_029_context_window_copy() {
        let window1 = ContextWindow::new(8192, 2048);
        let window2 = window1; // Copy trait
        assert_eq!(window1.max_tokens, window2.max_tokens);
    }

    #[test]
    fn test_ctx_cov_030_truncation_strategy_copy() {
        let strat1 = TruncationStrategy::MiddleOut;
        let strat2 = strat1; // Copy trait
        assert_eq!(strat1, strat2);
    }

    #[test]
    fn test_ctx_cov_031_context_config_clone() {
        let config = ContextConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.min_messages, config.min_messages);
    }

    #[test]
    fn test_ctx_cov_032_context_config_debug() {
        let config = ContextConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ContextConfig"));
    }

    #[test]
    fn test_ctx_cov_033_context_error_std_error() {
        let err = ContextError::ExceedsLimit {
            tokens: 100,
            limit: 50,
        };
        let _: &dyn std::error::Error = &err;
        // Just verify it implements Error trait
    }
}
