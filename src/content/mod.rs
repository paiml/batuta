#![allow(dead_code)]
//! Content Creation Tooling Module
//!
//! Implements the Content Creation Tooling Specification v1.1.0 for generating
//! structured prompts for educational and technical content.
//!
//! # Content Types
//! - HLO: High-Level Outline
//! - DLO: Detailed Outline
//! - BCH: Book Chapter (mdBook)
//! - BLP: Blog Post
//! - PDM: Presentar Demo
//!
//! # Toyota Way Integration
//! - Jidoka: LLM-as-a-Judge validation
//! - Poka-Yoke: Structural constraints in templates
//! - Genchi Genbutsu: Source context mandate
//! - Heijunka: Token budgeting
//! - Kaizen: Dynamic template composition

use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::PathBuf;
use std::str::FromStr;
use thiserror::Error;

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during content operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ContentError {
    #[error("Invalid content type: {0}")]
    InvalidContentType(String),

    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Template parse error: {0}")]
    TemplateParseFailed(String),

    #[error("Token budget exceeded: {used} tokens exceeds {limit} limit")]
    TokenBudgetExceeded { used: usize, limit: usize },

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Source context error: {0}")]
    SourceContextError(String),

    #[error("RAG context error: {0}")]
    RagContextError(String),
}

// ============================================================================
// CONTENT TYPES
// ============================================================================

/// Content type taxonomy from spec section 2.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// High-Level Outline (HLO) - Course/book structure planning
    HighLevelOutline,
    /// Detailed Outline (DLO) - Section-level content planning
    DetailedOutline,
    /// Book Chapter (BCH) - Technical documentation (mdBook)
    BookChapter,
    /// Blog Post (BLP) - Technical articles
    BlogPost,
    /// Presentar Demo (PDM) - Interactive WASM demos
    PresentarDemo,
}

/// Course level configuration for detailed outlines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CourseLevel {
    /// Short course: 1 week, 2 modules, 3 videos each
    Short,
    /// Standard course: 3 weeks, 3 modules, 5 videos each (default)
    #[default]
    Standard,
    /// Extended course: 6 weeks, 6 modules, 5 videos each
    Extended,
    /// Custom configuration
    Custom {
        weeks: u8,
        modules: u8,
        videos_per_module: u8,
    },
}

impl CourseLevel {
    /// Get duration in weeks
    pub fn weeks(&self) -> u8 {
        match self {
            CourseLevel::Short => 1,
            CourseLevel::Standard => 3,
            CourseLevel::Extended => 6,
            CourseLevel::Custom { weeks, .. } => *weeks,
        }
    }

    /// Get number of modules
    pub fn modules(&self) -> u8 {
        match self {
            CourseLevel::Short => 2,
            CourseLevel::Standard => 3,
            CourseLevel::Extended => 6,
            CourseLevel::Custom { modules, .. } => *modules,
        }
    }

    /// Get videos per module
    pub fn videos_per_module(&self) -> u8 {
        match self {
            CourseLevel::Short => 3,
            CourseLevel::Standard => 5,
            CourseLevel::Extended => 5,
            CourseLevel::Custom {
                videos_per_module, ..
            } => *videos_per_module,
        }
    }
}

impl FromStr for CourseLevel {
    type Err = ContentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "short" | "s" | "1" => Ok(CourseLevel::Short),
            "standard" | "std" | "3" => Ok(CourseLevel::Standard),
            "extended" | "ext" | "6" => Ok(CourseLevel::Extended),
            _ => Err(ContentError::InvalidContentType(format!(
                "Invalid course level: {}. Use: short, standard, extended",
                s
            ))),
        }
    }
}

impl ContentType {
    /// Get the short code for this content type
    pub fn code(&self) -> &'static str {
        match self {
            ContentType::HighLevelOutline => "HLO",
            ContentType::DetailedOutline => "DLO",
            ContentType::BookChapter => "BCH",
            ContentType::BlogPost => "BLP",
            ContentType::PresentarDemo => "PDM",
        }
    }

    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            ContentType::HighLevelOutline => "High-Level Outline",
            ContentType::DetailedOutline => "Detailed Outline",
            ContentType::BookChapter => "Book Chapter",
            ContentType::BlogPost => "Blog Post",
            ContentType::PresentarDemo => "Presentar Demo",
        }
    }

    /// Get the output format
    pub fn output_format(&self) -> &'static str {
        match self {
            ContentType::HighLevelOutline => "YAML/Markdown",
            ContentType::DetailedOutline => "YAML/Markdown",
            ContentType::BookChapter => "Markdown (mdBook)",
            ContentType::BlogPost => "Markdown + TOML",
            ContentType::PresentarDemo => "HTML + YAML",
        }
    }

    /// Get the target length range (in words for text, lines for outlines)
    pub fn target_length(&self) -> Range<usize> {
        match self {
            ContentType::HighLevelOutline => 50..200,  // lines
            ContentType::DetailedOutline => 200..1000, // lines
            ContentType::BookChapter => 2000..8000,    // words
            ContentType::BlogPost => 500..3000,        // words
            ContentType::PresentarDemo => 0..0,        // N/A
        }
    }

    /// Get all content types
    pub fn all() -> Vec<ContentType> {
        vec![
            ContentType::HighLevelOutline,
            ContentType::DetailedOutline,
            ContentType::BookChapter,
            ContentType::BlogPost,
            ContentType::PresentarDemo,
        ]
    }
}

impl FromStr for ContentType {
    type Err = ContentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hlo" | "high-level-outline" | "outline" => Ok(ContentType::HighLevelOutline),
            "dlo" | "detailed-outline" | "detailed" => Ok(ContentType::DetailedOutline),
            "bch" | "book-chapter" | "chapter" => Ok(ContentType::BookChapter),
            "blp" | "blog-post" | "blog" => Ok(ContentType::BlogPost),
            "pdm" | "presentar-demo" | "demo" => Ok(ContentType::PresentarDemo),
            _ => Err(ContentError::InvalidContentType(s.to_string())),
        }
    }
}

// ============================================================================
// TOKEN BUDGETING (Heijunka)
// ============================================================================

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

// ============================================================================
// SOURCE CONTEXT (Genchi Genbutsu)
// ============================================================================

/// Source context for grounding content in reality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceContext {
    /// Paths to source files/directories
    pub paths: Vec<PathBuf>,
    /// Extracted content snippets
    pub snippets: Vec<SourceSnippet>,
    /// Total tokens used
    pub total_tokens: usize,
}

/// A snippet extracted from source material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSnippet {
    /// Source file path
    pub path: PathBuf,
    /// Line range (start, end)
    pub lines: Option<(usize, usize)>,
    /// Content
    pub content: String,
    /// Estimated tokens
    pub tokens: usize,
}

impl SourceContext {
    /// Create new empty source context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path to include
    pub fn add_path(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    /// Add a snippet
    pub fn add_snippet(&mut self, snippet: SourceSnippet) {
        self.total_tokens += snippet.tokens;
        self.snippets.push(snippet);
    }

    /// Format for inclusion in prompt
    pub fn format_for_prompt(&self) -> String {
        if self.snippets.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        output.push_str("## Source Context (Genchi Genbutsu)\n\n");
        output.push_str("The following source material must be referenced in your output:\n\n");

        for snippet in &self.snippets {
            output.push_str(&format!("### {}\n", snippet.path.display()));
            if let Some((start, end)) = snippet.lines {
                output.push_str(&format!("Lines {}-{}:\n", start, end));
            }
            output.push_str("```\n");
            output.push_str(&snippet.content);
            output.push_str("\n```\n\n");
        }

        output.push_str("**Requirement**: Quote or reference specific line numbers from the source material above.\n\n");
        output
    }
}

// ============================================================================
// VALIDATION (Jidoka)
// ============================================================================

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical - must halt
    Critical,
    /// Error - should halt
    Error,
    /// Warning - flag for revision
    Warning,
    /// Info - informational
    Info,
}

/// A single validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Constraint that was violated
    pub constraint: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Location in content (e.g., "paragraph 3", "code block 5")
    pub location: String,
    /// The offending text
    pub text: String,
    /// Suggested fix
    pub suggestion: String,
}

/// Validation result from content validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Quality score (0-100)
    pub score: u8,
    /// List of violations
    pub violations: Vec<ValidationViolation>,
}

impl ValidationResult {
    /// Create a passing result
    pub fn pass(score: u8) -> Self {
        Self {
            passed: true,
            score,
            violations: Vec::new(),
        }
    }

    /// Create a failing result
    pub fn fail(violations: Vec<ValidationViolation>) -> Self {
        let score = Self::calculate_score(&violations);
        Self {
            passed: false,
            score,
            violations,
        }
    }

    /// Add a violation
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
        self.score = Self::calculate_score(&self.violations);
        self.passed = !self.violations.iter().any(|v| {
            matches!(
                v.severity,
                ValidationSeverity::Critical | ValidationSeverity::Error
            )
        });
    }

    /// Calculate score based on violations
    fn calculate_score(violations: &[ValidationViolation]) -> u8 {
        let mut score = 100i32;
        for v in violations {
            match v.severity {
                ValidationSeverity::Critical => score -= 50,
                ValidationSeverity::Error => score -= 25,
                ValidationSeverity::Warning => score -= 10,
                ValidationSeverity::Info => score -= 2,
            }
        }
        score.max(0) as u8
    }

    /// Check if there are critical violations
    pub fn has_critical(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ValidationSeverity::Critical)
    }

    /// Check if there are errors
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ValidationSeverity::Error)
    }

    /// Format as display string
    pub fn format_display(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Quality Score: {}/100\n\n", self.score));

        if self.violations.is_empty() {
            output.push_str("No violations found. ✓\n");
            return output;
        }

        output.push_str(&format!("Violations ({}):\n", self.violations.len()));
        for (i, v) in self.violations.iter().enumerate() {
            let prefix = if i == self.violations.len() - 1 {
                "└──"
            } else {
                "├──"
            };
            let severity = match v.severity {
                ValidationSeverity::Critical => "CRITICAL",
                ValidationSeverity::Error => "ERROR",
                ValidationSeverity::Warning => "WARNING",
                ValidationSeverity::Info => "INFO",
            };
            output.push_str(&format!(
                "{} [{}] {} @ {}\n",
                prefix, severity, v.constraint, v.location
            ));
            output.push_str(&format!("    Text: \"{}\"\n", v.text));
            output.push_str(&format!("    Fix: {}\n", v.suggestion));
        }

        output
    }
}

/// Content validator for Jidoka quality gates
#[derive(Debug, Clone)]
pub struct ContentValidator {
    /// Content type being validated
    content_type: ContentType,
}

impl ContentValidator {
    /// Create a new validator for a content type
    pub fn new(content_type: ContentType) -> Self {
        Self { content_type }
    }

    /// Validate content against all rules
    pub fn validate(&self, content: &str) -> ValidationResult {
        let mut result = ValidationResult::pass(100);

        // Run all validation checks
        self.validate_instructor_voice(content, &mut result);
        self.validate_code_blocks(content, &mut result);
        self.validate_heading_hierarchy(content, &mut result);
        self.validate_meta_commentary(content, &mut result);

        // Content-type specific validation
        match self.content_type {
            ContentType::BookChapter | ContentType::BlogPost => {
                self.validate_frontmatter(content, &mut result);
            }
            _ => {}
        }

        result
    }

    /// Check for meta-commentary (Andon)
    fn validate_meta_commentary(&self, content: &str, result: &mut ValidationResult) {
        let meta_phrases = [
            "in this chapter",
            "in this section",
            "we will learn",
            "we will explore",
            "we will discuss",
            "this chapter covers",
            "this section covers",
            "as mentioned earlier",
            "as we discussed",
        ];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            for phrase in &meta_phrases {
                if lower.contains(phrase) {
                    result.add_violation(ValidationViolation {
                        constraint: "no_meta_commentary".to_string(),
                        severity: ValidationSeverity::Warning,
                        location: format!("line {}", line_num + 1),
                        text: line.trim().chars().take(60).collect::<String>() + "...",
                        suggestion: "Use direct instruction instead of meta-commentary".to_string(),
                    });
                }
            }
        }
    }

    /// Validate instructor voice
    fn validate_instructor_voice(&self, content: &str, result: &mut ValidationResult) {
        // Check for passive voice indicators in instruction contexts
        let passive_indicators = [
            "is being",
            "was being",
            "has been",
            "have been",
            "will be shown",
            "can be seen",
        ];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            // Only check non-code lines
            if !line.trim().starts_with("```") && !line.trim().starts_with("//") {
                for phrase in &passive_indicators {
                    if lower.contains(phrase) {
                        result.add_violation(ValidationViolation {
                            constraint: "instructor_voice".to_string(),
                            severity: ValidationSeverity::Info,
                            location: format!("line {}", line_num + 1),
                            text: line.trim().chars().take(60).collect::<String>(),
                            suggestion: "Consider using active voice for clearer instruction"
                                .to_string(),
                        });
                    }
                }
            }
        }
    }

    /// Validate code blocks have language specifiers
    fn validate_code_blocks(&self, content: &str, result: &mut ValidationResult) {
        let mut in_code_block = false;
        let mut block_start = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().starts_with("```") {
                if !in_code_block {
                    // Starting a code block
                    in_code_block = true;
                    block_start = line_num + 1;
                    let lang = line.trim().trim_start_matches('`');
                    if lang.is_empty() {
                        result.add_violation(ValidationViolation {
                            constraint: "code_block_language".to_string(),
                            severity: ValidationSeverity::Warning,
                            location: format!("line {}", line_num + 1),
                            text: "```".to_string(),
                            suggestion: "Specify language: ```rust, ```python, ```bash, etc."
                                .to_string(),
                        });
                    }
                } else {
                    // Ending a code block
                    in_code_block = false;
                }
            }
        }

        // Check for unclosed code block
        if in_code_block {
            result.add_violation(ValidationViolation {
                constraint: "code_block_closed".to_string(),
                severity: ValidationSeverity::Error,
                location: format!("line {}", block_start),
                text: "Unclosed code block".to_string(),
                suggestion: "Add closing ``` to code block".to_string(),
            });
        }
    }

    /// Validate heading hierarchy (no skipped levels)
    fn validate_heading_hierarchy(&self, content: &str, result: &mut ValidationResult) {
        let mut last_level = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.starts_with('#') {
                let level = line.chars().take_while(|c| *c == '#').count();
                if last_level > 0 && level > last_level + 1 {
                    result.add_violation(ValidationViolation {
                        constraint: "heading_hierarchy".to_string(),
                        severity: ValidationSeverity::Error,
                        location: format!("line {}", line_num + 1),
                        text: line.trim().to_string(),
                        suggestion: format!(
                            "Heading level {} skips from level {}. Use H{}.",
                            level,
                            last_level,
                            last_level + 1
                        ),
                    });
                }
                last_level = level;
            }
        }
    }

    /// Validate frontmatter presence (for BCH and BLP)
    fn validate_frontmatter(&self, content: &str, result: &mut ValidationResult) {
        let _has_yaml_frontmatter = content.starts_with("---");
        let has_toml_frontmatter = content.starts_with("+++");

        if self.content_type == ContentType::BlogPost && !has_toml_frontmatter {
            result.add_violation(ValidationViolation {
                constraint: "frontmatter_present".to_string(),
                severity: ValidationSeverity::Critical,
                location: "beginning".to_string(),
                text: "Missing TOML frontmatter".to_string(),
                suggestion: "Add +++ frontmatter with title, date, description".to_string(),
            });
        }
    }
}

// ============================================================================
// PROMPT EMITTER
// ============================================================================

/// Configuration for prompt emission
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmitConfig {
    /// Content type to generate
    pub content_type: Option<ContentType>,
    /// Title or topic
    pub title: Option<String>,
    /// Target audience
    pub audience: Option<String>,
    /// Word count target
    pub word_count: Option<usize>,
    /// Source context paths
    pub source_context_paths: Vec<PathBuf>,
    /// RAG context directory
    pub rag_context_path: Option<PathBuf>,
    /// RAG token limit
    pub rag_limit: usize,
    /// Model context
    pub model: ModelContext,
    /// Show token budget
    pub show_budget: bool,
    /// Course level (for detailed outlines)
    pub course_level: CourseLevel,
}

impl EmitConfig {
    /// Create a new emit config
    pub fn new(content_type: ContentType) -> Self {
        Self {
            content_type: Some(content_type),
            model: ModelContext::Claude200K,
            rag_limit: 4000,
            ..Default::default()
        }
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set audience
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = Some(audience.into());
        self
    }

    /// Set word count
    pub fn with_word_count(mut self, count: usize) -> Self {
        self.word_count = Some(count);
        self
    }

    /// Add source context path
    pub fn with_source_context(mut self, path: PathBuf) -> Self {
        self.source_context_paths.push(path);
        self
    }

    /// Set RAG context
    pub fn with_rag_context(mut self, path: PathBuf, limit: usize) -> Self {
        self.rag_context_path = Some(path);
        self.rag_limit = limit;
        self
    }

    /// Set course level (for detailed outlines)
    pub fn with_course_level(mut self, level: CourseLevel) -> Self {
        self.course_level = level;
        self
    }
}

/// Prompt emitter for content generation
#[derive(Debug, Clone)]
pub struct PromptEmitter {
    /// Toyota Way constraints (shared across all content types)
    toyota_constraints: String,
    /// Quality gates template
    quality_gates: String,
}

impl PromptEmitter {
    /// Create a new prompt emitter
    pub fn new() -> Self {
        Self {
            toyota_constraints: Self::default_toyota_constraints(),
            quality_gates: Self::default_quality_gates(),
        }
    }

    /// Default Toyota Way constraints
    fn default_toyota_constraints() -> String {
        r#"## Toyota Way Constraints

These principles MUST be followed:

1. **Jidoka (Built-in Quality)**: Every output must pass quality gates
2. **Poka-Yoke (Error Prevention)**: Follow structural constraints exactly
3. **Genchi Genbutsu (Go and See)**: Reference provided source material
4. **Heijunka (Level Loading)**: Stay within target length range
5. **Kaizen (Continuous Improvement)**: Learn from feedback

**Voice Requirements**:
- Use direct instruction, not meta-commentary
- NO: "In this chapter, we will learn about..."
- YES: "Create a new project with cargo new..."
- NO: "This section covers error handling."
- YES: "Rust's Result type provides explicit error handling."
"#
        .to_string()
    }

    /// Default quality gates
    fn default_quality_gates() -> String {
        r#"## Quality Gates (Andon)

Before submitting, verify:
- [ ] No meta-commentary ("In this section, we will...")
- [ ] All code blocks specify language (```rust, ```python)
- [ ] Heading hierarchy is strict (no skipped levels)
- [ ] Content is within target length range
- [ ] Source material is referenced where provided
"#
        .to_string()
    }

    /// Emit a prompt for the given configuration
    pub fn emit(&self, config: &EmitConfig) -> Result<String, ContentError> {
        let content_type = config
            .content_type
            .ok_or_else(|| ContentError::MissingRequiredField("content_type".to_string()))?;

        let mut prompt = String::new();

        // Header
        prompt.push_str(&format!(
            "# Content Generation Request: {}\n\n",
            content_type.name()
        ));

        // Context section
        prompt.push_str("## Context\n\n");
        prompt.push_str(&format!(
            "You are creating a {} ({}).\n\n",
            content_type.name(),
            content_type.code()
        ));

        if let Some(title) = &config.title {
            prompt.push_str(&format!("**Title/Topic**: {}\n", title));
        }
        if let Some(audience) = &config.audience {
            prompt.push_str(&format!("**Target Audience**: {}\n", audience));
        }
        if let Some(word_count) = config.word_count {
            prompt.push_str(&format!("**Target Length**: {} words\n", word_count));
        } else {
            let range = content_type.target_length();
            if range.start > 0 {
                prompt.push_str(&format!(
                    "**Target Length**: {}-{} {}\n",
                    range.start,
                    range.end,
                    if matches!(
                        content_type,
                        ContentType::HighLevelOutline | ContentType::DetailedOutline
                    ) {
                        "lines"
                    } else {
                        "words"
                    }
                ));
            }
        }
        prompt.push_str(&format!(
            "**Output Format**: {}\n\n",
            content_type.output_format()
        ));

        // Toyota Way constraints
        prompt.push_str(&self.toyota_constraints);
        prompt.push('\n');

        // Content-type specific instructions
        prompt.push_str(&self.emit_type_specific(content_type, config));
        prompt.push('\n');

        // Quality gates
        prompt.push_str(&self.quality_gates);
        prompt.push('\n');

        // Token budget if requested
        if config.show_budget {
            let budget = TokenBudget::new(config.model).with_output_target(
                TokenBudget::words_to_tokens(config.word_count.unwrap_or(4000)),
            );
            prompt.push_str("## Token Budget\n\n");
            prompt.push_str(&budget.format_display(config.model.name()));
            prompt.push('\n');
        }

        // Final instruction
        prompt.push_str("---\n\n");
        prompt.push_str("Generate the content now, following all constraints above.\n");

        Ok(prompt)
    }

    /// Emit type-specific instructions
    fn emit_type_specific(&self, content_type: ContentType, config: &EmitConfig) -> String {
        match content_type {
            ContentType::HighLevelOutline => r#"## Structure Requirements (Poka-Yoke)

1. **Parts**: 3-7 major parts/sections
2. **Chapters**: 3-5 chapters per part
3. **Learning Objectives**: 2-4 per chapter (use Bloom's taxonomy verbs)
4. **Balance**: No part exceeds 25% of total content
5. **Progression**: Fundamentals → Intermediate → Advanced

## Output Schema

```yaml
type: high_level_outline
version: "1.0"
metadata:
  title: string
  description: string
  target_audience: string
  prerequisites: [string]
  estimated_duration: string
structure:
  - part: string
    title: string
    chapters:
      - number: int
        title: string
        summary: string
        learning_objectives: [string]
```
"#
            .to_string(),
            ContentType::DetailedOutline => {
                let level = &config.course_level;
                let weeks = level.weeks();
                let modules = level.modules();
                let videos = level.videos_per_module();
                let is_short = matches!(level, CourseLevel::Short);

                let mut output = format!(
                    r#"## Structure Requirements (Poka-Yoke)

1. **Duration**: {} week{} total course length
2. **Modules**: {} module{} ({} per week)
3. **Per Module**: {} video{} + 1 quiz + 1 reading + 1 lab
4. **Video Length**: 5-15 minutes each
5. **Balance**: 60% video instruction, 15% reading, 15% lab, 10% quiz
6. **Transitions**: Each video connects to previous/next
7. **Learning Objectives**: 3 for course{}

## Output Schema

```yaml
type: detailed_outline
version: "1.0"
course:
  title: string
  description: string (2-3 sentences summarizing the course)
  duration_weeks: {}
  total_modules: {}
  learning_objectives:
    - objective: string
    - objective: string
    - objective: string
"#,
                    weeks,
                    if weeks == 1 { "" } else { "s" },
                    modules,
                    if modules == 1 { "" } else { "s" },
                    if weeks > 0 {
                        format!("{:.1}", modules as f32 / weeks as f32)
                    } else {
                        "N/A".to_string()
                    },
                    videos,
                    if videos == 1 { "" } else { "s" },
                    if is_short { "" } else { ", 3 per week" },
                    weeks,
                    modules
                );

                // Add weekly learning objectives for non-short courses
                if !is_short {
                    output.push_str("weeks:\n");
                    for w in 1..=weeks {
                        output.push_str(&format!(
                            r#"  - week: {}
    learning_objectives:
      - objective: string
      - objective: string
      - objective: string
"#,
                            w
                        ));
                    }
                }

                output.push_str("modules:\n");

                // Generate module entries
                for m in 1..=modules {
                    let week = if weeks > 0 {
                        ((m - 1) / (modules / weeks).max(1)) + 1
                    } else {
                        1
                    };
                    output.push_str(&format!(
                        r#"  - id: module_{}
    week: {}
    title: string
    description: string
    learning_objectives:
      - objective: string
    videos:
"#,
                        m,
                        week.min(weeks)
                    ));

                    // Generate video entries
                    for v in 1..=videos {
                        if v == 1 {
                            output.push_str(&format!(
                                r#"      - id: video_{}_{}
        title: string
        duration_minutes: int (5-15)
        key_points:
          - point: string
            code_snippet: optional
"#,
                                m, v
                            ));
                        } else {
                            output.push_str(&format!(
                                r#"      - id: video_{}_{}
        title: string
        duration_minutes: int
"#,
                                m, v
                            ));
                        }
                    }

                    output.push_str(
                        r#"    reading:
      title: string
      duration_minutes: int (15-30)
      content_summary: string
      key_concepts:
        - concept: string
    quiz:
      title: string
      num_questions: int (5-10)
      topics_covered:
        - topic: string
    lab:
      title: string
      duration_minutes: int (30-60)
      objectives:
        - objective: string
      starter_code: optional
"#,
                    );
                }

                output.push_str("```\n");
                output
            }
            ContentType::BookChapter => r#"## Writing Guidelines

1. **Instructor voice**: Direct teaching, show then explain
2. **Code-first**: Show implementation, then explain
3. **Progressive complexity**: Start simple, build up
4. **mdBook format**: Proper heading hierarchy

## Formatting Requirements

- H1 (#) for chapter title ONLY
- H2 (##) for major sections (5-7 per chapter)
- H3 (###) for subsections as needed
- Code blocks with language specifier: ```rust, ```python
- Callouts: > **Note/Warning/Tip**: text

## Example Structure

```markdown
# Chapter Title

Introduction paragraph...

## Section 1: Getting Started

Content...

### Subsection 1.1

```rust
// Code with comments
fn example() {
    println!("Hello");
}
```

> **Note**: Important information here.

## Summary

- Key point 1
- Key point 2

## Exercises

1. Exercise description...
```
"#
            .to_string(),
            ContentType::BlogPost => r#"## Blog Post Guidelines

1. **Hook**: First paragraph must grab attention
2. **Scannable**: Clear headings, short paragraphs
3. **Practical**: Real-world examples and takeaways
4. **SEO-friendly**: Title under 60 chars, description under 160

## Required TOML Frontmatter

```toml
+++
title = "Post Title"
date = 2025-12-05
description = "SEO description under 160 characters"
[taxonomies]
tags = ["tag1", "tag2", "tag3"]
categories = ["category"]
[extra]
author = "Author Name"
reading_time = "X min"
+++
```

## Structure

1. Introduction with hook
2. 3-5 main sections with H2 headings
3. Code examples where relevant
4. Conclusion with call-to-action
"#
            .to_string(),
            ContentType::PresentarDemo => r#"## Presentar Demo Configuration

Generate a complete interactive demo specification.

## Required Output Schema

```yaml
type: presentar_demo
version: "1.0"
metadata:
  title: string
  description: string
  demo_type: shell-autocomplete|ml-inference|wasm-showcase|terminal-repl

wasm_config:
  module_path: string
  model_path: optional
  runner_path: string

ui_config:
  theme: light|dark|high-contrast
  show_performance_metrics: bool
  debounce_ms: int

performance_gates:
  latency_target_ms: 1
  cold_start_target_ms: 100
  bundle_size_kb: 500

instructions:
  setup: string
  interaction_guide: string
  expected_behavior: string
```

## Accessibility Requirements

- WCAG 2.1 AA compliance
- Keyboard navigation
- Screen reader support
- Graceful degradation
"#
            .to_string(),
        }
    }
}

impl Default for PromptEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS (TDD - Written First)
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // ContentType Tests
    // ========================================================================

    #[test]
    fn test_CONTENT_001_content_type_codes() {
        assert_eq!(ContentType::HighLevelOutline.code(), "HLO");
        assert_eq!(ContentType::DetailedOutline.code(), "DLO");
        assert_eq!(ContentType::BookChapter.code(), "BCH");
        assert_eq!(ContentType::BlogPost.code(), "BLP");
        assert_eq!(ContentType::PresentarDemo.code(), "PDM");
    }

    #[test]
    fn test_CONTENT_002_content_type_names() {
        assert_eq!(ContentType::HighLevelOutline.name(), "High-Level Outline");
        assert_eq!(ContentType::DetailedOutline.name(), "Detailed Outline");
        assert_eq!(ContentType::BookChapter.name(), "Book Chapter");
        assert_eq!(ContentType::BlogPost.name(), "Blog Post");
        assert_eq!(ContentType::PresentarDemo.name(), "Presentar Demo");
    }

    #[test]
    fn test_CONTENT_003_content_type_from_str() {
        assert_eq!(
            ContentType::from_str("hlo").unwrap(),
            ContentType::HighLevelOutline
        );
        assert_eq!(
            ContentType::from_str("DLO").unwrap(),
            ContentType::DetailedOutline
        );
        assert_eq!(
            ContentType::from_str("book-chapter").unwrap(),
            ContentType::BookChapter
        );
        assert_eq!(
            ContentType::from_str("blog").unwrap(),
            ContentType::BlogPost
        );
        assert_eq!(
            ContentType::from_str("demo").unwrap(),
            ContentType::PresentarDemo
        );
    }

    #[test]
    fn test_CONTENT_004_content_type_from_str_invalid() {
        let result = ContentType::from_str("invalid");
        assert!(result.is_err());
        assert!(matches!(result, Err(ContentError::InvalidContentType(_))));
    }

    #[test]
    fn test_CONTENT_005_content_type_output_formats() {
        assert_eq!(
            ContentType::HighLevelOutline.output_format(),
            "YAML/Markdown"
        );
        assert_eq!(
            ContentType::BookChapter.output_format(),
            "Markdown (mdBook)"
        );
        assert_eq!(ContentType::BlogPost.output_format(), "Markdown + TOML");
        assert_eq!(ContentType::PresentarDemo.output_format(), "HTML + YAML");
    }

    #[test]
    fn test_CONTENT_006_content_type_all() {
        let all = ContentType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&ContentType::HighLevelOutline));
        assert!(all.contains(&ContentType::PresentarDemo));
    }

    #[test]
    fn test_CONTENT_007_content_type_target_length() {
        assert_eq!(ContentType::HighLevelOutline.target_length(), 50..200);
        assert_eq!(ContentType::BookChapter.target_length(), 2000..8000);
        assert_eq!(ContentType::BlogPost.target_length(), 500..3000);
    }

    // ========================================================================
    // TokenBudget Tests
    // ========================================================================

    #[test]
    fn test_BUDGET_001_token_budget_new() {
        let budget = TokenBudget::new(ModelContext::Claude200K);
        assert_eq!(budget.context_window, 200_000);
        assert_eq!(budget.system_reserve, 2_000);
    }

    #[test]
    fn test_BUDGET_002_token_budget_with_source_context() {
        let budget = TokenBudget::new(ModelContext::Claude200K).with_source_context(10_000);
        assert_eq!(budget.source_context, 10_000);
    }

    #[test]
    fn test_BUDGET_003_token_budget_prompt_tokens() {
        let budget = TokenBudget::new(ModelContext::Claude200K)
            .with_source_context(5_000)
            .with_rag_context(3_000);
        // system_reserve (2000) + source (5000) + rag (3000) + few_shot (1500)
        assert_eq!(budget.prompt_tokens(), 11_500);
    }

    #[test]
    fn test_BUDGET_004_token_budget_available_margin() {
        let budget = TokenBudget::new(ModelContext::Claude200K)
            .with_source_context(5_000)
            .with_output_target(10_000);
        // 200000 - (2000 + 5000 + 0 + 1500 + 10000) = 181500
        assert_eq!(budget.available_margin(), 181_500);
    }

    #[test]
    fn test_BUDGET_005_token_budget_validate_ok() {
        let budget = TokenBudget::new(ModelContext::Claude200K).with_output_target(10_000);
        assert!(budget.validate().is_ok());
    }

    #[test]
    fn test_BUDGET_006_token_budget_validate_exceeded() {
        let budget = TokenBudget {
            context_window: 1000,
            system_reserve: 500,
            source_context: 500,
            rag_context: 500,
            few_shot: 500,
            output_target: 500,
        };
        let result = budget.validate();
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ContentError::TokenBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_BUDGET_007_words_to_tokens() {
        assert_eq!(TokenBudget::words_to_tokens(1000), 1300);
        assert_eq!(TokenBudget::words_to_tokens(4000), 5200);
    }

    #[test]
    fn test_BUDGET_008_tokens_to_words() {
        assert_eq!(TokenBudget::tokens_to_words(1300), 1000);
        assert_eq!(TokenBudget::tokens_to_words(5200), 4000);
    }

    #[test]
    fn test_BUDGET_009_model_context_window_sizes() {
        assert_eq!(ModelContext::Claude200K.window_size(), 200_000);
        assert_eq!(ModelContext::GeminiPro.window_size(), 1_000_000);
        assert_eq!(ModelContext::Gpt4Turbo.window_size(), 128_000);
        assert_eq!(ModelContext::Custom(50_000).window_size(), 50_000);
    }

    #[test]
    fn test_BUDGET_010_format_display() {
        let budget = TokenBudget::new(ModelContext::Claude200K)
            .with_source_context(8_000)
            .with_rag_context(4_000)
            .with_output_target(15_000);
        let display = budget.format_display("claude-sonnet");
        assert!(display.contains("claude-sonnet"));
        assert!(display.contains("200K"));
        assert!(display.contains("8000"));
        assert!(display.contains("✓"));
    }

    // ========================================================================
    // SourceContext Tests
    // ========================================================================

    #[test]
    fn test_SOURCE_001_source_context_new() {
        let ctx = SourceContext::new();
        assert!(ctx.paths.is_empty());
        assert!(ctx.snippets.is_empty());
        assert_eq!(ctx.total_tokens, 0);
    }

    #[test]
    fn test_SOURCE_002_source_context_add_path() {
        let mut ctx = SourceContext::new();
        ctx.add_path(PathBuf::from("/src/lib.rs"));
        assert_eq!(ctx.paths.len(), 1);
    }

    #[test]
    fn test_SOURCE_003_source_context_add_snippet() {
        let mut ctx = SourceContext::new();
        ctx.add_snippet(SourceSnippet {
            path: PathBuf::from("/src/lib.rs"),
            lines: Some((1, 10)),
            content: "fn main() {}".to_string(),
            tokens: 50,
        });
        assert_eq!(ctx.snippets.len(), 1);
        assert_eq!(ctx.total_tokens, 50);
    }

    #[test]
    fn test_SOURCE_004_source_context_format_empty() {
        let ctx = SourceContext::new();
        assert!(ctx.format_for_prompt().is_empty());
    }

    #[test]
    fn test_SOURCE_005_source_context_format_with_snippets() {
        let mut ctx = SourceContext::new();
        ctx.add_snippet(SourceSnippet {
            path: PathBuf::from("/src/lib.rs"),
            lines: Some((1, 10)),
            content: "fn main() {}".to_string(),
            tokens: 50,
        });
        let formatted = ctx.format_for_prompt();
        assert!(formatted.contains("Source Context"));
        assert!(formatted.contains("Genchi Genbutsu"));
        assert!(formatted.contains("lib.rs"));
        assert!(formatted.contains("fn main()"));
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_VALID_001_validation_result_pass() {
        let result = ValidationResult::pass(100);
        assert!(result.passed);
        assert_eq!(result.score, 100);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_VALID_002_validation_result_fail() {
        let violations = vec![ValidationViolation {
            constraint: "test".to_string(),
            severity: ValidationSeverity::Error,
            location: "line 1".to_string(),
            text: "bad text".to_string(),
            suggestion: "fix it".to_string(),
        }];
        let result = ValidationResult::fail(violations);
        assert!(!result.passed);
        assert!(result.score < 100);
    }

    #[test]
    fn test_VALID_003_validation_score_calculation() {
        let mut result = ValidationResult::pass(100);
        result.add_violation(ValidationViolation {
            constraint: "test".to_string(),
            severity: ValidationSeverity::Warning,
            location: "line 1".to_string(),
            text: "text".to_string(),
            suggestion: "fix".to_string(),
        });
        assert_eq!(result.score, 90); // 100 - 10 for warning
    }

    #[test]
    fn test_VALID_004_validation_has_critical() {
        let mut result = ValidationResult::pass(100);
        assert!(!result.has_critical());
        result.add_violation(ValidationViolation {
            constraint: "test".to_string(),
            severity: ValidationSeverity::Critical,
            location: "line 1".to_string(),
            text: "text".to_string(),
            suggestion: "fix".to_string(),
        });
        assert!(result.has_critical());
    }

    #[test]
    fn test_VALID_005_validator_meta_commentary() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Chapter 1\n\nIn this chapter, we will learn about Rust.";
        let result = validator.validate(content);
        assert!(result
            .violations
            .iter()
            .any(|v| v.constraint == "no_meta_commentary"));
    }

    #[test]
    fn test_VALID_006_validator_code_block_no_language() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Chapter 1\n\n```\nfn main() {}\n```";
        let result = validator.validate(content);
        assert!(result
            .violations
            .iter()
            .any(|v| v.constraint == "code_block_language"));
    }

    #[test]
    fn test_VALID_007_validator_code_block_with_language() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Chapter 1\n\n```rust\nfn main() {}\n```";
        let result = validator.validate(content);
        assert!(!result
            .violations
            .iter()
            .any(|v| v.constraint == "code_block_language"));
    }

    #[test]
    fn test_VALID_008_validator_heading_hierarchy_ok() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# H1\n\n## H2\n\n### H3";
        let result = validator.validate(content);
        assert!(!result
            .violations
            .iter()
            .any(|v| v.constraint == "heading_hierarchy"));
    }

    #[test]
    fn test_VALID_009_validator_heading_hierarchy_skipped() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# H1\n\n### H3 skipped H2";
        let result = validator.validate(content);
        assert!(result
            .violations
            .iter()
            .any(|v| v.constraint == "heading_hierarchy"));
    }

    #[test]
    fn test_VALID_010_validator_unclosed_code_block() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Chapter\n\n```rust\nfn main() {}";
        let result = validator.validate(content);
        assert!(result
            .violations
            .iter()
            .any(|v| v.constraint == "code_block_closed"));
    }

    #[test]
    fn test_VALID_011_validator_blog_missing_frontmatter() {
        let validator = ContentValidator::new(ContentType::BlogPost);
        let content = "# Blog Post\n\nContent here.";
        let result = validator.validate(content);
        assert!(result
            .violations
            .iter()
            .any(|v| v.constraint == "frontmatter_present"));
    }

    #[test]
    fn test_VALID_012_format_display() {
        let mut result = ValidationResult::pass(100);
        result.add_violation(ValidationViolation {
            constraint: "test_constraint".to_string(),
            severity: ValidationSeverity::Warning,
            location: "line 5".to_string(),
            text: "some text".to_string(),
            suggestion: "fix this".to_string(),
        });
        let display = result.format_display();
        assert!(display.contains("90/100"));
        assert!(display.contains("WARNING"));
        assert!(display.contains("test_constraint"));
    }

    // ========================================================================
    // PromptEmitter Tests
    // ========================================================================

    #[test]
    fn test_EMIT_001_emitter_new() {
        let emitter = PromptEmitter::new();
        assert!(!emitter.toyota_constraints.is_empty());
        assert!(!emitter.quality_gates.is_empty());
    }

    #[test]
    fn test_EMIT_002_emit_hlo() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::HighLevelOutline)
            .with_title("Rust for Data Engineers")
            .with_audience("Python developers");
        let prompt = emitter.emit(&config).unwrap();
        assert!(prompt.contains("High-Level Outline"));
        assert!(prompt.contains("Rust for Data Engineers"));
        assert!(prompt.contains("Python developers"));
        assert!(prompt.contains("Toyota Way"));
    }

    #[test]
    fn test_EMIT_003_emit_bch() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::BookChapter)
            .with_title("Error Handling")
            .with_word_count(4000);
        let prompt = emitter.emit(&config).unwrap();
        assert!(prompt.contains("Book Chapter"));
        assert!(prompt.contains("4000 words"));
        assert!(prompt.contains("mdBook"));
    }

    #[test]
    fn test_EMIT_004_emit_blp() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::BlogPost).with_title("Rust Performance Tips");
        let prompt = emitter.emit(&config).unwrap();
        assert!(prompt.contains("Blog Post"));
        assert!(prompt.contains("TOML Frontmatter"));
        assert!(prompt.contains("SEO"));
    }

    #[test]
    fn test_EMIT_005_emit_pdm() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::PresentarDemo).with_title("Shell Autocomplete");
        let prompt = emitter.emit(&config).unwrap();
        assert!(prompt.contains("Presentar Demo"));
        assert!(prompt.contains("wasm")); // lowercase in template
        assert!(prompt.contains("Accessibility"));
    }

    #[test]
    fn test_EMIT_006_emit_missing_content_type() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::default();
        let result = emitter.emit(&config);
        assert!(result.is_err());
        assert!(matches!(result, Err(ContentError::MissingRequiredField(_))));
    }

    #[test]
    fn test_EMIT_007_emit_with_budget() {
        let emitter = PromptEmitter::new();
        let mut config = EmitConfig::new(ContentType::BookChapter);
        config.show_budget = true;
        config.word_count = Some(4000);
        let prompt = emitter.emit(&config).unwrap();
        assert!(prompt.contains("Token Budget"));
        assert!(prompt.contains("claude-sonnet"));
    }

    #[test]
    fn test_EMIT_008_toyota_constraints_content() {
        let emitter = PromptEmitter::new();
        assert!(emitter.toyota_constraints.contains("Jidoka"));
        assert!(emitter.toyota_constraints.contains("Poka-Yoke"));
        assert!(emitter.toyota_constraints.contains("Genchi Genbutsu"));
        assert!(emitter.toyota_constraints.contains("Heijunka"));
        assert!(emitter.toyota_constraints.contains("Kaizen"));
    }

    #[test]
    fn test_EMIT_009_quality_gates_content() {
        let emitter = PromptEmitter::new();
        assert!(emitter.quality_gates.contains("Andon"));
        assert!(emitter.quality_gates.contains("meta-commentary"));
        assert!(emitter.quality_gates.contains("code blocks"));
        assert!(emitter.quality_gates.contains("Heading hierarchy"));
    }

    #[test]
    fn test_EMIT_010_emit_config_builder() {
        let config = EmitConfig::new(ContentType::BookChapter)
            .with_title("Test")
            .with_audience("Developers")
            .with_word_count(5000)
            .with_source_context(PathBuf::from("/src"));
        assert_eq!(config.title, Some("Test".to_string()));
        assert_eq!(config.audience, Some("Developers".to_string()));
        assert_eq!(config.word_count, Some(5000));
        assert_eq!(config.source_context_paths.len(), 1);
    }

    // ========================================================================
    // CourseLevel Tests
    // ========================================================================

    #[test]
    fn test_LEVEL_001_course_level_short_config() {
        let level = CourseLevel::Short;
        assert_eq!(level.weeks(), 1);
        assert_eq!(level.modules(), 2);
        assert_eq!(level.videos_per_module(), 3);
    }

    #[test]
    fn test_LEVEL_002_course_level_standard_config() {
        let level = CourseLevel::Standard;
        assert_eq!(level.weeks(), 3);
        assert_eq!(level.modules(), 3);
        assert_eq!(level.videos_per_module(), 5);
    }

    #[test]
    fn test_LEVEL_003_course_level_extended_config() {
        let level = CourseLevel::Extended;
        assert_eq!(level.weeks(), 6);
        assert_eq!(level.modules(), 6);
        assert_eq!(level.videos_per_module(), 5);
    }

    #[test]
    fn test_LEVEL_004_course_level_custom_config() {
        let level = CourseLevel::Custom {
            weeks: 4,
            modules: 8,
            videos_per_module: 4,
        };
        assert_eq!(level.weeks(), 4);
        assert_eq!(level.modules(), 8);
        assert_eq!(level.videos_per_module(), 4);
    }

    #[test]
    fn test_LEVEL_005_course_level_from_str_short() {
        let level: CourseLevel = "short".parse().unwrap();
        assert_eq!(level, CourseLevel::Short);

        let level: CourseLevel = "s".parse().unwrap();
        assert_eq!(level, CourseLevel::Short);

        let level: CourseLevel = "1".parse().unwrap();
        assert_eq!(level, CourseLevel::Short);
    }

    #[test]
    fn test_LEVEL_006_course_level_from_str_standard() {
        let level: CourseLevel = "standard".parse().unwrap();
        assert_eq!(level, CourseLevel::Standard);

        let level: CourseLevel = "std".parse().unwrap();
        assert_eq!(level, CourseLevel::Standard);

        let level: CourseLevel = "3".parse().unwrap();
        assert_eq!(level, CourseLevel::Standard);
    }

    #[test]
    fn test_LEVEL_007_course_level_from_str_extended() {
        let level: CourseLevel = "extended".parse().unwrap();
        assert_eq!(level, CourseLevel::Extended);

        let level: CourseLevel = "ext".parse().unwrap();
        assert_eq!(level, CourseLevel::Extended);

        let level: CourseLevel = "6".parse().unwrap();
        assert_eq!(level, CourseLevel::Extended);
    }

    #[test]
    fn test_LEVEL_008_course_level_from_str_case_insensitive() {
        let level: CourseLevel = "SHORT".parse().unwrap();
        assert_eq!(level, CourseLevel::Short);

        let level: CourseLevel = "Standard".parse().unwrap();
        assert_eq!(level, CourseLevel::Standard);

        let level: CourseLevel = "EXTENDED".parse().unwrap();
        assert_eq!(level, CourseLevel::Extended);
    }

    #[test]
    fn test_LEVEL_009_course_level_from_str_invalid() {
        let result: Result<CourseLevel, _> = "invalid".parse();
        assert!(result.is_err());
        assert!(matches!(result, Err(ContentError::InvalidContentType(_))));
    }

    #[test]
    fn test_LEVEL_010_course_level_default() {
        let level = CourseLevel::default();
        assert_eq!(level, CourseLevel::Standard);
    }

    #[test]
    fn test_LEVEL_011_emit_config_with_course_level() {
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Test Course")
            .with_course_level(CourseLevel::Extended);
        assert_eq!(config.course_level, CourseLevel::Extended);
    }

    #[test]
    fn test_LEVEL_012_emit_dlo_short_no_weekly_objectives() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Quick Start")
            .with_course_level(CourseLevel::Short);
        let prompt = emitter.emit(&config).unwrap();

        // Short courses should NOT have weekly learning objectives section
        assert!(!prompt.contains("weeks:\n"));
        // But should have course-level objectives
        assert!(prompt.contains("learning_objectives:"));
        // Should have correct duration
        assert!(prompt.contains("1 week"));
        assert!(prompt.contains("2 modules"));
    }

    #[test]
    fn test_LEVEL_013_emit_dlo_standard_has_weekly_objectives() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Complete Course")
            .with_course_level(CourseLevel::Standard);
        let prompt = emitter.emit(&config).unwrap();

        // Standard courses SHOULD have weekly learning objectives
        assert!(prompt.contains("weeks:\n"));
        assert!(prompt.contains("- week: 1"));
        assert!(prompt.contains("- week: 2"));
        assert!(prompt.contains("- week: 3"));
        // Should have correct duration
        assert!(prompt.contains("3 weeks"));
        assert!(prompt.contains("3 modules"));
    }

    #[test]
    fn test_LEVEL_014_emit_dlo_extended_has_six_weeks() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Masterclass")
            .with_course_level(CourseLevel::Extended);
        let prompt = emitter.emit(&config).unwrap();

        // Extended courses should have 6 weeks
        assert!(prompt.contains("6 weeks"));
        assert!(prompt.contains("6 modules"));
        assert!(prompt.contains("- week: 1"));
        assert!(prompt.contains("- week: 6"));
    }

    #[test]
    fn test_LEVEL_015_emit_dlo_course_description() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Test Course")
            .with_course_level(CourseLevel::Standard);
        let prompt = emitter.emit(&config).unwrap();

        // All courses should have description and learning objectives
        assert!(prompt.contains("description: string"));
        assert!(prompt.contains("learning_objectives:"));

        // Check the course section has 3 learning objectives before weeks section
        // The format is: course: ... learning_objectives: ... weeks:
        let course_start = prompt.find("course:").expect("Should have course section");
        let weeks_start = prompt.find("\nweeks:").expect("Should have weeks section");
        let course_section = &prompt[course_start..weeks_start];
        let objective_count = course_section.matches("- objective: string").count();
        assert_eq!(
            objective_count, 3,
            "Course should have 3 learning objectives"
        );
    }

    #[test]
    fn test_LEVEL_016_emit_dlo_structure_requirements() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Test")
            .with_course_level(CourseLevel::Standard);
        let prompt = emitter.emit(&config).unwrap();

        // Check structure requirements are documented
        assert!(prompt.contains("**Duration**"));
        assert!(prompt.contains("**Modules**"));
        assert!(prompt.contains("**Per Module**"));
        assert!(prompt.contains("**Learning Objectives**"));
        assert!(prompt.contains("3 for course, 3 per week"));
    }

    #[test]
    fn test_LEVEL_017_emit_dlo_short_learning_objectives_text() {
        let emitter = PromptEmitter::new();
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title("Quick Start")
            .with_course_level(CourseLevel::Short);
        let prompt = emitter.emit(&config).unwrap();

        // Short courses should say "3 for course" without ", 3 per week"
        assert!(prompt.contains("**Learning Objectives**: 3 for course\n"));
    }

    #[test]
    fn test_LEVEL_018_course_level_equality() {
        assert_eq!(CourseLevel::Short, CourseLevel::Short);
        assert_eq!(CourseLevel::Standard, CourseLevel::Standard);
        assert_eq!(CourseLevel::Extended, CourseLevel::Extended);
        assert_ne!(CourseLevel::Short, CourseLevel::Standard);

        let custom1 = CourseLevel::Custom {
            weeks: 4,
            modules: 4,
            videos_per_module: 4,
        };
        let custom2 = CourseLevel::Custom {
            weeks: 4,
            modules: 4,
            videos_per_module: 4,
        };
        assert_eq!(custom1, custom2);
    }

    #[test]
    fn test_LEVEL_019_course_level_debug() {
        let level = CourseLevel::Standard;
        let debug_str = format!("{:?}", level);
        assert!(debug_str.contains("Standard"));
    }

    #[test]
    fn test_LEVEL_020_course_level_clone() {
        let level = CourseLevel::Extended;
        let cloned = level.clone();
        assert_eq!(level, cloned);
    }
}
