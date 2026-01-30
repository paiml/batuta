//! PARF (Program Analysis for Rust Files) command implementations
//!
//! This module contains the PARF analysis command extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::parf;
use std::path::Path;

/// PARF output format
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ParfOutputFormat {
    /// Plain text output
    Text,
    /// JSON output
    Json,
    /// Markdown output
    Markdown,
}

/// Run PARF analysis
pub fn cmd_parf(
    path: &Path,
    find_symbol: Option<&str>,
    detect_patterns: bool,
    analyze_dependencies: bool,
    find_dead_code: bool,
    format: ParfOutputFormat,
    output_file: Option<&Path>,
) -> anyhow::Result<()> {
    use parf::{CodePattern, ParfAnalyzer, SymbolKind};

    println!("{}", "🔍 PARF Analysis".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Create analyzer
    let mut analyzer = ParfAnalyzer::new();

    // Index codebase
    println!("{}", "Indexing codebase...".dimmed());
    analyzer.index_codebase(path)?;
    println!("{} Indexing complete", "✓".bright_green());
    println!();

    let mut output = String::new();

    // Find references to specific symbol
    if let Some(symbol) = find_symbol {
        println!(
            "{} Finding references to '{}'...",
            "→".bright_blue(),
            symbol.cyan()
        );
        let refs = analyzer.find_references(symbol, SymbolKind::Function);

        match format {
            ParfOutputFormat::Text => {
                output.push_str(&format!("\nReferences to '{}': {}\n", symbol, refs.len()));
                for (i, r) in refs.iter().enumerate() {
                    output.push_str(&format!(
                        "  {}. {}:{} - {}\n",
                        i + 1,
                        r.file.display(),
                        r.line,
                        r.context
                    ));
                }
            }
            ParfOutputFormat::Json => {
                output.push_str(&serde_json::to_string_pretty(&refs)?);
                output.push('\n');
            }
            ParfOutputFormat::Markdown => {
                output.push_str(&format!("## References to '{}'\n\n", symbol));
                output.push_str(&format!("Found {} references:\n\n", refs.len()));
                for (i, r) in refs.iter().enumerate() {
                    output.push_str(&format!(
                        "{}. `{}:{}` - {}\n",
                        i + 1,
                        r.file.display(),
                        r.line,
                        r.context
                    ));
                }
            }
        }
    }

    // Detect patterns
    if detect_patterns {
        println!("{} Detecting code patterns...", "→".bright_blue());
        let patterns = analyzer.detect_patterns();

        let mut tech_debt_count = 0;
        let mut error_handling_count = 0;
        let mut resource_mgmt_count = 0;
        let mut deprecated_count = 0;

        for pattern in &patterns {
            match pattern {
                CodePattern::TechDebt { .. } => tech_debt_count += 1,
                CodePattern::ErrorHandling { .. } => error_handling_count += 1,
                CodePattern::ResourceManagement { .. } => resource_mgmt_count += 1,
                CodePattern::DeprecatedApi { .. } => deprecated_count += 1,
                _ => {}
            }
        }

        match format {
            ParfOutputFormat::Text => {
                output.push_str(&format!("\nCode Patterns Detected: {}\n", patterns.len()));
                output.push_str(&format!(
                    "  Technical Debt (TODO/FIXME): {}\n",
                    tech_debt_count
                ));
                output.push_str(&format!(
                    "  Error Handling Issues: {}\n",
                    error_handling_count
                ));
                output.push_str(&format!("  Resource Management: {}\n", resource_mgmt_count));
                output.push_str(&format!("  Deprecated APIs: {}\n", deprecated_count));
            }
            ParfOutputFormat::Json => {
                output.push_str(&serde_json::to_string_pretty(&patterns)?);
                output.push('\n');
            }
            ParfOutputFormat::Markdown => {
                output.push_str("## Code Patterns\n\n");
                output.push_str(&format!("Total patterns detected: {}\n\n", patterns.len()));
                output.push_str(&format!("- Technical Debt: {}\n", tech_debt_count));
                output.push_str(&format!(
                    "- Error Handling Issues: {}\n",
                    error_handling_count
                ));
                output.push_str(&format!("- Resource Management: {}\n", resource_mgmt_count));
                output.push_str(&format!("- Deprecated APIs: {}\n", deprecated_count));
            }
        }
    }

    // Analyze dependencies
    if analyze_dependencies {
        println!("{} Analyzing dependencies...", "→".bright_blue());
        let deps = analyzer.analyze_dependencies();

        match format {
            ParfOutputFormat::Text => {
                output.push_str(&format!("\nDependencies: {}\n", deps.len()));
                for (i, dep) in deps.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "  {}. {} → {} ({:?})\n",
                        i + 1,
                        dep.from.display(),
                        dep.to.display(),
                        dep.kind
                    ));
                }
                if deps.len() > 10 {
                    output.push_str(&format!("  ... and {} more\n", deps.len() - 10));
                }
            }
            ParfOutputFormat::Json => {
                output.push_str(&serde_json::to_string_pretty(&deps)?);
                output.push('\n');
            }
            ParfOutputFormat::Markdown => {
                output.push_str("## Dependencies\n\n");
                output.push_str(&format!("Total dependencies: {}\n\n", deps.len()));
                for (i, dep) in deps.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "{}. `{}` → `{}` ({:?})\n",
                        i + 1,
                        dep.from.display(),
                        dep.to.display(),
                        dep.kind
                    ));
                }
            }
        }
    }

    // Find dead code
    if find_dead_code {
        println!("{} Finding dead code...", "→".bright_blue());
        let dead_code = analyzer.find_dead_code();

        match format {
            ParfOutputFormat::Text => {
                output.push_str(&format!("\nPotentially Dead Code: {}\n", dead_code.len()));
                for (i, dc) in dead_code.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "  {}. {} ({:?}) in {}:{} - {}\n",
                        i + 1,
                        dc.symbol,
                        dc.kind,
                        dc.file.display(),
                        dc.line,
                        dc.reason
                    ));
                }
                if dead_code.len() > 10 {
                    output.push_str(&format!("  ... and {} more\n", dead_code.len() - 10));
                }
            }
            ParfOutputFormat::Json => {
                output.push_str(&serde_json::to_string_pretty(&dead_code)?);
                output.push('\n');
            }
            ParfOutputFormat::Markdown => {
                output.push_str("## Dead Code\n\n");
                output.push_str(&format!(
                    "Potentially unused symbols: {}\n\n",
                    dead_code.len()
                ));
                for (i, dc) in dead_code.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "{}. `{}` ({:?}) in `{}:{}`\n   - {}\n",
                        i + 1,
                        dc.symbol,
                        dc.kind,
                        dc.file.display(),
                        dc.line,
                        dc.reason
                    ));
                }
            }
        }
    }

    // Generate overall report if no specific analysis requested
    if find_symbol.is_none() && !detect_patterns && !analyze_dependencies && !find_dead_code {
        let report = analyzer.generate_report();
        output.push_str(&report);
    }

    // Output results
    if let Some(out_path) = output_file {
        std::fs::write(out_path, &output)?;
        println!();
        println!(
            "{} Report written to: {}",
            "✓".bright_green(),
            out_path.display().to_string().cyan()
        );
    } else {
        println!();
        println!("{}", output);
    }

    println!();
    println!("{}", "✅ PARF analysis complete!".bright_green().bold());
    println!();

    Ok(())
}
