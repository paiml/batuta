//! Prompt emitter for content generation
//!
//! This module contains the PromptEmitter and EmitConfig extracted from content/mod.rs.

use super::{ContentError, ContentType, CourseLevel, ModelContext, TokenBudget};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// EMIT CONFIG
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

// ============================================================================
// PROMPT EMITTER
// ============================================================================

/// Prompt emitter for content generation
#[derive(Debug, Clone)]
pub struct PromptEmitter {
    /// Toyota Way constraints (shared across all content types)
    toyota_constraints: String,
    /// Quality gates template
    quality_gates: String,
}

impl PromptEmitter {
    /// Get the Toyota Way constraints
    pub fn toyota_constraints(&self) -> &str {
        &self.toyota_constraints
    }

    /// Get the quality gates
    pub fn quality_gates(&self) -> &str {
        &self.quality_gates
    }

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
