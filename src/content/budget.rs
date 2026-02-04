//! Token budgeting (Heijunka)
//!
//! Token budget calculation (spec section 5.4) for load leveling.

use super::ContentError;
use serde::{Deserialize, Serialize};

/// Model context window sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelContext {
    /// Claude Sonnet/Opus (200K) - Default
    #[default]
    Claude200K,
    /// Claude Haiku (200K)
    ClaudeHaiku,
    /// Gemini Pro (1M)
    GeminiPro,
    /// Gemini Flash (1M)
    GeminiFlash,
    /// GPT-4 Turbo (128K)
    Gpt4Turbo,
    /// Custom context window
    Custom(usize),
}

impl ModelContext {
    /// Get the context window size in tokens
    pub fn window_size(&self) -> usize {
        match self {
            ModelContext::Claude200K => 200_000,
            ModelContext::ClaudeHaiku => 200_000,
            ModelContext::GeminiPro => 1_000_000,
            ModelContext::GeminiFlash => 1_000_000,
            ModelContext::Gpt4Turbo => 128_000,
            ModelContext::Custom(size) => *size,
        }
    }

    /// Get the model name
    pub fn name(&self) -> &'static str {
        match self {
            ModelContext::Claude200K => "claude-sonnet",
            ModelContext::ClaudeHaiku => "claude-haiku",
            ModelContext::GeminiPro => "gemini-pro",
            ModelContext::GeminiFlash => "gemini-flash",
            ModelContext::Gpt4Turbo => "gpt-4-turbo",
            ModelContext::Custom(_) => "custom",
        }
    }
}

/// Token budget calculation (spec section 5.4)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Target model context window
    pub context_window: usize,
    /// Reserved for system prompt
    pub system_reserve: usize,
    /// Reserved for source context (--source-context)
    pub source_context: usize,
    /// Reserved for RAG context (--rag-context)
    pub rag_context: usize,
    /// Reserved for few-shot examples
    pub few_shot: usize,
    /// Target output tokens
    pub output_target: usize,
}

impl TokenBudget {
    /// Create a new token budget for a model
    pub fn new(model: ModelContext) -> Self {
        Self {
            context_window: model.window_size(),
            system_reserve: 2_000,
            source_context: 0,
            rag_context: 0,
            few_shot: 1_500,
            output_target: 4_000,
        }
    }

    /// Set source context tokens
    pub fn with_source_context(mut self, tokens: usize) -> Self {
        self.source_context = tokens;
        self
    }

    /// Set RAG context tokens
    pub fn with_rag_context(mut self, tokens: usize) -> Self {
        self.rag_context = tokens;
        self
    }

    /// Set output target tokens
    pub fn with_output_target(mut self, tokens: usize) -> Self {
        self.output_target = tokens;
        self
    }

    /// Calculate total prompt tokens (excluding output)
    pub fn prompt_tokens(&self) -> usize {
        self.system_reserve + self.source_context + self.rag_context + self.few_shot
    }

    /// Calculate available margin
    pub fn available_margin(&self) -> usize {
        let used = self.prompt_tokens() + self.output_target;
        self.context_window.saturating_sub(used)
    }

    /// Validate the budget fits within context window
    pub fn validate(&self) -> Result<(), ContentError> {
        let total = self.prompt_tokens() + self.output_target;
        if total > self.context_window {
            Err(ContentError::TokenBudgetExceeded {
                used: total,
                limit: self.context_window,
            })
        } else {
            Ok(())
        }
    }

    /// Estimate tokens from word count (rough: 1 word ≈ 1.3 tokens)
    pub fn words_to_tokens(words: usize) -> usize {
        (words as f64 * 1.3).ceil() as usize
    }

    /// Estimate words from token count
    pub fn tokens_to_words(tokens: usize) -> usize {
        (tokens as f64 / 1.3).floor() as usize
    }

    /// Format budget as display string
    pub fn format_display(&self, model_name: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Token Budget for {} ({}K context):\n",
            model_name,
            self.context_window / 1000
        ));
        output.push_str(&format!(
            "├── System prompt:     {:>6} tokens\n",
            self.system_reserve
        ));
        output.push_str(&format!(
            "├── Source context:    {:>6} tokens\n",
            self.source_context
        ));
        output.push_str(&format!(
            "├── RAG context:       {:>6} tokens\n",
            self.rag_context
        ));
        output.push_str(&format!(
            "├── Few-shot examples: {:>6} tokens\n",
            self.few_shot
        ));
        output.push_str(&format!(
            "├── Output reserved:   {:>6} tokens (~{} words)\n",
            self.output_target,
            Self::tokens_to_words(self.output_target)
        ));
        let margin = self.available_margin();
        let status = if margin > 0 { "✓" } else { "✗" };
        output.push_str(&format!(
            "└── Available margin:  {:>6} tokens {}\n",
            margin, status
        ));
        output
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::new(ModelContext::Claude200K)
    }
}
