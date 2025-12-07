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
    ContentType, ContentValidator, EmitConfig, ModelContext, PromptEmitter, SourceContext,
    SourceSnippet, TokenBudget,
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
