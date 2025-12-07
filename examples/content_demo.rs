//! Content Creation Tooling Demo
//!
//! Demonstrates the Content Creation Tooling module from spec v1.1.0
//! for generating structured prompts for educational content.
//!
//! # Run
//! ```bash
//! cargo run --features native --example content_demo
//! ```

use batuta::content::{
    ContentType, ContentValidator, CourseLevel, EmitConfig, ModelContext, PromptEmitter,
    SourceContext, SourceSnippet, TokenBudget,
};
use std::path::PathBuf;
use std::str::FromStr;

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Content Creation Tooling Demo (Spec v1.1.0)              ║");
    println!("║     Toyota Way Principles for LLM Content Generation         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Content Types Overview
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 1. CONTENT TYPES                                             │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    for ct in ContentType::all() {
        let range = ct.target_length();
        let length_desc = if range.start == 0 && range.end == 0 {
            "N/A".to_string()
        } else {
            format!("{}-{}", range.start, range.end)
        };
        println!(
            "  {} ({}) - {} [{}]",
            ct.name(),
            ct.code(),
            ct.output_format(),
            length_desc
        );
    }

    // Parse from string
    println!("\n  Parsing from string:");
    let parsed = ContentType::from_str("bch")?;
    println!("    'bch' -> {} ({})", parsed.name(), parsed.code());

    // =========================================================================
    // 2. Token Budgeting (Heijunka)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ 2. TOKEN BUDGETING (Heijunka - Level Loading)                │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Create budget for Claude
    let budget = TokenBudget::new(ModelContext::Claude200K)
        .with_source_context(10_000)
        .with_rag_context(5_000)
        .with_output_target(15_000);

    println!("{}", budget.format_display("Claude Sonnet"));

    // Validate budget
    match budget.validate() {
        Ok(()) => println!("  ✓ Budget is valid\n"),
        Err(e) => println!("  ✗ Budget error: {}\n", e),
    }

    // Show different model contexts
    println!("  Model Context Windows:");
    println!(
        "    Claude 200K:  {:>10} tokens",
        ModelContext::Claude200K.window_size()
    );
    println!(
        "    Gemini Pro:   {:>10} tokens",
        ModelContext::GeminiPro.window_size()
    );
    println!(
        "    GPT-4 Turbo:  {:>10} tokens",
        ModelContext::Gpt4Turbo.window_size()
    );

    // =========================================================================
    // 3. Source Context (Genchi Genbutsu)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ 3. SOURCE CONTEXT (Genchi Genbutsu - Go and See)             │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let mut source_ctx = SourceContext::new();
    source_ctx.add_snippet(SourceSnippet {
        path: PathBuf::from("src/lib.rs"),
        lines: Some((1, 15)),
        content: r#"/// Error handling with Result
pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}"#
        .to_string(),
        tokens: 85,
    });

    println!("  Source snippets added: {}", source_ctx.snippets.len());
    println!("  Total tokens used: {}", source_ctx.total_tokens);
    println!("\n  Formatted for prompt:");
    println!("  {}", "-".repeat(50));
    for line in source_ctx.format_for_prompt().lines().take(10) {
        println!("  {}", line);
    }
    println!("  ...");

    // =========================================================================
    // 4. Content Validation (Jidoka)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ 4. CONTENT VALIDATION (Jidoka - Built-in Quality)            │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let validator = ContentValidator::new(ContentType::BookChapter);

    // Good content
    let good_content = r#"# Error Handling in Rust

Rust provides two primary mechanisms for handling errors: `Result<T, E>` and `panic!`.

## The Result Type

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::read_to_string("config.toml")?;
    println!("Config: {}", file);
    Ok(())
}
```

> **Note**: The `?` operator propagates errors automatically.

## Propagating Errors

### Using the ? Operator

The `?` operator unwraps `Ok` values or returns `Err` early.
"#;

    println!("  Testing GOOD content:");
    let result = validator.validate(good_content);
    println!("    Passed: {}", result.passed);
    println!("    Score: {}/100", result.score);
    println!("    Violations: {}", result.violations.len());

    // Bad content with issues
    let bad_content = r#"# Chapter Title

In this chapter, we will learn about error handling.

## Section 1

```
some code without language
```

#### Skipped H3

This section covers the basics.
"#;

    println!("\n  Testing BAD content:");
    let result = validator.validate(bad_content);
    println!("    Passed: {}", result.passed);
    println!("    Score: {}/100", result.score);
    println!("    Violations: {}", result.violations.len());

    for v in &result.violations {
        let severity = match v.severity {
            batuta::content::ValidationSeverity::Critical => "CRIT",
            batuta::content::ValidationSeverity::Error => "ERR ",
            batuta::content::ValidationSeverity::Warning => "WARN",
            batuta::content::ValidationSeverity::Info => "INFO",
        };
        println!("      [{}] {} @ {}", severity, v.constraint, v.location);
    }

    // =========================================================================
    // 5. Prompt Emission
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ 5. PROMPT EMISSION                                           │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let emitter = PromptEmitter::new();

    // Book Chapter prompt
    let config = EmitConfig::new(ContentType::BookChapter)
        .with_title("SIMD Optimization in Rust")
        .with_audience("Systems programmers")
        .with_word_count(4000);

    let prompt = emitter.emit(&config)?;

    println!("  Generated Book Chapter Prompt:");
    println!("  {}", "-".repeat(50));
    for line in prompt.lines().take(20) {
        println!("  {}", line);
    }
    println!("  ... ({} total chars)", prompt.len());

    // Blog Post prompt
    println!("\n  Generated Blog Post Prompt (summary):");
    let blog_config = EmitConfig::new(ContentType::BlogPost)
        .with_title("Why Rust for ML Inference")
        .with_word_count(1500);
    let blog_prompt = emitter.emit(&blog_config)?;
    println!("    Total length: {} chars", blog_prompt.len());
    println!("    Contains TOML: {}", blog_prompt.contains("TOML"));
    println!("    Contains SEO: {}", blog_prompt.contains("SEO"));

    // Presentar Demo prompt
    println!("\n  Generated Presentar Demo Prompt (summary):");
    let demo_config =
        EmitConfig::new(ContentType::PresentarDemo).with_title("Shell Autocomplete WASM");
    let demo_prompt = emitter.emit(&demo_config)?;
    println!("    Total length: {} chars", demo_prompt.len());
    println!("    Contains WASM: {}", demo_prompt.contains("wasm"));
    println!("    Contains WCAG: {}", demo_prompt.contains("WCAG"));

    // =========================================================================
    // 6. Course Levels (Configurable Outlines)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ 6. COURSE LEVELS (Configurable Detailed Outlines)           │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("  Available Course Levels:");
    println!("  {}", "-".repeat(50));

    // Display all course levels with their configurations
    let levels = [
        ("Short", CourseLevel::Short),
        ("Standard", CourseLevel::Standard),
        ("Extended", CourseLevel::Extended),
    ];

    for (name, level) in &levels {
        println!(
            "    {:10} │ {} week(s) │ {} modules │ {} videos/module",
            name,
            level.weeks(),
            level.modules(),
            level.videos_per_module()
        );
    }

    // Parse from string
    println!("\n  Parsing from string:");
    let parsed: CourseLevel = "short".parse()?;
    println!(
        "    'short' -> {} weeks, {} modules",
        parsed.weeks(),
        parsed.modules()
    );
    let parsed: CourseLevel = "extended".parse()?;
    println!(
        "    'extended' -> {} weeks, {} modules",
        parsed.weeks(),
        parsed.modules()
    );

    // Generate detailed outline with different levels
    println!("\n  Generating Detailed Outlines with Course Levels:");
    println!("  {}", "-".repeat(50));

    // Short course
    let short_config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Quick Start Guide")
        .with_course_level(CourseLevel::Short);
    let short_prompt = emitter.emit(&short_config)?;
    println!("\n  SHORT Course (1 week, 2 modules, 3 videos each):");
    println!("    Total prompt length: {} chars", short_prompt.len());
    println!("    Contains '1 week': {}", short_prompt.contains("1 week"));
    println!(
        "    Contains '2 modules': {}",
        short_prompt.contains("2 modules")
    );
    println!(
        "    Has weekly objectives: {} (expected: false)",
        short_prompt.contains("weeks:\n")
    );

    // Standard course (default)
    let standard_config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Complete Course")
        .with_course_level(CourseLevel::Standard);
    let standard_prompt = emitter.emit(&standard_config)?;
    println!("\n  STANDARD Course (3 weeks, 3 modules, 5 videos each):");
    println!("    Total prompt length: {} chars", standard_prompt.len());
    println!(
        "    Contains '3 weeks': {}",
        standard_prompt.contains("3 weeks")
    );
    println!(
        "    Contains '3 modules': {}",
        standard_prompt.contains("3 modules")
    );
    println!(
        "    Has weekly objectives: {} (expected: true)",
        standard_prompt.contains("weeks:\n")
    );

    // Extended course
    let extended_config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Comprehensive Masterclass")
        .with_course_level(CourseLevel::Extended);
    let extended_prompt = emitter.emit(&extended_config)?;
    println!("\n  EXTENDED Course (6 weeks, 6 modules, 5 videos each):");
    println!("    Total prompt length: {} chars", extended_prompt.len());
    println!(
        "    Contains '6 weeks': {}",
        extended_prompt.contains("6 weeks")
    );
    println!(
        "    Contains '6 modules': {}",
        extended_prompt.contains("6 modules")
    );
    println!(
        "    Has weekly objectives: {} (expected: true)",
        extended_prompt.contains("weeks:\n")
    );

    // Show structure requirements from standard course
    println!("\n  Sample Structure Requirements (Standard):");
    for line in standard_prompt.lines() {
        if line.contains("**Duration**")
            || line.contains("**Modules**")
            || line.contains("**Per Module**")
            || line.contains("**Learning Objectives**")
        {
            println!("    {}", line.trim());
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Toyota Way Principles Applied:                               ║");
    println!("║                                                                ║");
    println!("║  • Jidoka     - Quality gates catch errors automatically      ║");
    println!("║  • Poka-Yoke  - Structural constraints prevent mistakes       ║");
    println!("║  • Genchi Genbutsu - Source context grounds content           ║");
    println!("║  • Heijunka   - Token budgeting levels context usage          ║");
    println!("║  • Kaizen     - Dynamic templates enable improvement          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
