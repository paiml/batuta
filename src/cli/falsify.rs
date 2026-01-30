//! Falsification command implementations
//!
//! This module contains the Popperian Falsification Checklist command extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use std::path::PathBuf;

/// Falsify output format
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum FalsifyOutputFormat {
    /// Human-readable text output with colors
    Text,
    /// JSON output for CI/CD integration
    Json,
    /// Markdown output for documentation
    Markdown,
    /// GitHub Actions annotations
    GithubActions,
}

/// Run Popperian Falsification Checklist
pub fn cmd_falsify(
    path: PathBuf,
    critical_only: bool,
    format: FalsifyOutputFormat,
    output: Option<PathBuf>,
    min_grade: &str,
    verbose: bool,
) -> anyhow::Result<()> {
    use batuta::falsification::{evaluate_critical_only, evaluate_project, TpsGrade};

    // Parse minimum grade threshold
    let min_grade_threshold = match min_grade.to_lowercase().replace('-', "").as_str() {
        "toyotastandard" | "toyota" => TpsGrade::ToyotaStandard,
        "kaizenrequired" | "kaizen" => TpsGrade::KaizenRequired,
        "andonwarning" | "andon" => TpsGrade::AndonWarning,
        "stoptheline" | "stop" => TpsGrade::StopTheLine,
        _ => {
            anyhow::bail!(
                "Invalid min-grade: '{}'. Valid values: toyota-standard, kaizen-required, andon-warning, stop-the-line",
                min_grade
            );
        }
    };

    // Run evaluation
    let result = if critical_only {
        evaluate_critical_only(&path)
    } else {
        evaluate_project(&path)
    };

    // Format output
    let output_text = match format {
        FalsifyOutputFormat::Text => format_falsify_text(&result, verbose),
        FalsifyOutputFormat::Json => {
            serde_json::to_string_pretty(&result).unwrap_or_else(|e| format!("JSON error: {}", e))
        }
        FalsifyOutputFormat::Markdown => format_falsify_markdown(&result, verbose),
        FalsifyOutputFormat::GithubActions => format_falsify_github_actions(&result),
    };

    // Write to file or stdout
    if let Some(output_path) = output {
        std::fs::write(&output_path, &output_text)?;
        println!(
            "{} Report written to: {}",
            "✓".bright_green(),
            output_path.display()
        );
    } else {
        println!("{}", output_text);
    }

    // Check grade threshold
    let passes_threshold = match min_grade_threshold {
        TpsGrade::ToyotaStandard => result.grade == TpsGrade::ToyotaStandard,
        TpsGrade::KaizenRequired => matches!(
            result.grade,
            TpsGrade::ToyotaStandard | TpsGrade::KaizenRequired
        ),
        TpsGrade::AndonWarning => !matches!(result.grade, TpsGrade::StopTheLine),
        TpsGrade::StopTheLine => true, // Always passes
    };

    if !passes_threshold {
        anyhow::bail!(
            "Grade {} does not meet minimum threshold {}",
            result.grade,
            min_grade_threshold
        );
    }

    Ok(())
}

fn format_falsify_text(result: &batuta::falsification::ChecklistResult, verbose: bool) -> String {
    use batuta::falsification::{CheckStatus, Severity, TpsGrade};

    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "{}\n",
        "╔═══════════════════════════════════════════════════════════════════╗".bright_cyan()
    ));
    output.push_str(&format!(
        "{}\n",
        "║     POPPERIAN FALSIFICATION CHECKLIST - Sovereign AI Protocol    ║".bright_cyan()
    ));
    output.push_str(&format!(
        "{}\n",
        "╚═══════════════════════════════════════════════════════════════════╝".bright_cyan()
    ));
    output.push('\n');

    // Project info
    output.push_str(&format!(
        "Project: {}\n",
        result.project_path.display().to_string().cyan()
    ));
    output.push_str(&format!("Evaluated: {}\n", result.timestamp.dimmed()));
    output.push('\n');

    // Grade display
    let grade_color = match result.grade {
        TpsGrade::ToyotaStandard => "✓ Toyota Standard".bright_green(),
        TpsGrade::KaizenRequired => "◐ Kaizen Required".bright_yellow(),
        TpsGrade::AndonWarning => "⚠ Andon Warning".bright_red(),
        TpsGrade::StopTheLine => "✗ STOP THE LINE".on_red().white().bold(),
    };
    output.push_str(&format!("Grade: {}\n", grade_color));
    output.push_str(&format!("Score: {:.1}%\n", result.score));
    output.push_str(&format!(
        "Items: {}/{} passed, {} failed\n",
        result.passed_items, result.total_items, result.failed_items
    ));
    output.push('\n');

    // Section results
    for (section_name, items) in &result.sections {
        output.push_str(&format!("{}\n", format!("─── {} ───", section_name).bold()));

        for item in items {
            let status_icon = match item.status {
                CheckStatus::Pass => "✓".bright_green(),
                CheckStatus::Partial => "◐".bright_yellow(),
                CheckStatus::Fail => "✗".bright_red(),
                CheckStatus::Skipped => "○".dimmed(),
            };

            let severity_tag = match item.severity {
                Severity::Critical => "[CRITICAL]".on_red().white(),
                Severity::Major => "[MAJOR]".bright_red(),
                Severity::Minor => "[MINOR]".bright_yellow(),
                Severity::Info => "[INFO]".dimmed(),
            };

            output.push_str(&format!(
                "  {} {} {} {}\n",
                status_icon,
                item.id.bold(),
                item.name,
                severity_tag
            ));

            if verbose {
                output.push_str(&format!("    Claim: {}\n", item.claim.dimmed()));
                if !item.tps_principle.is_empty() {
                    output.push_str(&format!("    TPS: {}\n", item.tps_principle.cyan()));
                }
                if let Some(reason) = &item.rejection_reason {
                    output.push_str(&format!("    Reason: {}\n", reason.bright_red()));
                }
                for evidence in &item.evidence {
                    output.push_str(&format!(
                        "    Evidence: {}\n",
                        evidence.description.dimmed()
                    ));
                }
            }
        }
        output.push('\n');
    }

    // Summary
    if result.has_critical_failure {
        output.push_str(&format!(
            "{}\n",
            "⚠️  CRITICAL FAILURE DETECTED - Release blocked!"
                .on_red()
                .white()
                .bold()
        ));
    } else if result.passes() {
        output.push_str(&format!(
            "{}\n",
            "✅ All critical checks passed - Release allowed".bright_green()
        ));
    }

    output
}

fn format_falsify_markdown(
    result: &batuta::falsification::ChecklistResult,
    verbose: bool,
) -> String {
    use batuta::falsification::{CheckStatus, Severity, TpsGrade};

    let mut output = String::new();

    // Header
    output.push_str("# Popperian Falsification Checklist Report\n\n");
    output.push_str(&format!(
        "**Project:** `{}`\n\n",
        result.project_path.display()
    ));
    output.push_str(&format!("**Evaluated:** {}\n\n", result.timestamp));

    // Grade badge
    let grade_badge = match result.grade {
        TpsGrade::ToyotaStandard => {
            "![Grade](https://img.shields.io/badge/Grade-Toyota%20Standard-brightgreen)"
        }
        TpsGrade::KaizenRequired => {
            "![Grade](https://img.shields.io/badge/Grade-Kaizen%20Required-yellow)"
        }
        TpsGrade::AndonWarning => {
            "![Grade](https://img.shields.io/badge/Grade-Andon%20Warning-orange)"
        }
        TpsGrade::StopTheLine => {
            "![Grade](https://img.shields.io/badge/Grade-STOP%20THE%20LINE-red)"
        }
    };
    output.push_str(&format!("{}\n\n", grade_badge));

    // Summary table
    output.push_str("## Summary\n\n");
    output.push_str("| Metric | Value |\n");
    output.push_str("|--------|-------|\n");
    output.push_str(&format!("| Score | {:.1}% |\n", result.score));
    output.push_str(&format!("| Passed | {} |\n", result.passed_items));
    output.push_str(&format!("| Failed | {} |\n", result.failed_items));
    output.push_str(&format!("| Total | {} |\n", result.total_items));
    output.push_str(&format!(
        "| Critical Failure | {} |\n\n",
        if result.has_critical_failure {
            "Yes"
        } else {
            "No"
        }
    ));

    // Section results
    for (section_name, items) in &result.sections {
        output.push_str(&format!("## {}\n\n", section_name));
        output.push_str("| ID | Name | Status | Severity |\n");
        output.push_str("|----|------|--------|----------|\n");

        for item in items {
            let status = match item.status {
                CheckStatus::Pass => "✅ Pass",
                CheckStatus::Partial => "⚠️ Partial",
                CheckStatus::Fail => "❌ Fail",
                CheckStatus::Skipped => "⏭️ Skipped",
            };
            let severity = match item.severity {
                Severity::Critical => "🔴 Critical",
                Severity::Major => "🟠 Major",
                Severity::Minor => "🟡 Minor",
                Severity::Info => "🔵 Info",
            };
            output.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                item.id, item.name, status, severity
            ));
        }

        if verbose {
            output.push_str("\n### Details\n\n");
            for item in items {
                if item.status == CheckStatus::Fail || item.status == CheckStatus::Partial {
                    output.push_str(&format!("#### {} - {}\n\n", item.id, item.name));
                    output.push_str(&format!("**Claim:** {}\n\n", item.claim));
                    if let Some(reason) = &item.rejection_reason {
                        output.push_str(&format!("**Rejection:** {}\n\n", reason));
                    }
                }
            }
        }
        output.push('\n');
    }

    output
}

fn format_falsify_github_actions(result: &batuta::falsification::ChecklistResult) -> String {
    use batuta::falsification::{CheckStatus, Severity};

    let mut output = String::new();

    // Set output variables
    output.push_str(&format!("::set-output name=score::{:.1}\n", result.score));
    output.push_str(&format!("::set-output name=grade::{}\n", result.grade));
    output.push_str(&format!("::set-output name=passes::{}\n", result.passes()));

    // Annotations for failures
    for items in result.sections.values() {
        for item in items {
            if item.status == CheckStatus::Fail {
                let level = match item.severity {
                    Severity::Critical | Severity::Major => "error",
                    Severity::Minor => "warning",
                    Severity::Info => "notice",
                };

                output.push_str(&format!(
                    "::{} title={}::{}{}",
                    level,
                    item.id,
                    item.name,
                    item.rejection_reason
                        .as_ref()
                        .map(|r| format!(" - {}", r))
                        .unwrap_or_default()
                ));
                output.push('\n');
            }
        }
    }

    // Summary annotation
    if result.has_critical_failure {
        output.push_str(
            "::error::Popperian Falsification Check FAILED - Critical failure detected\n",
        );
    } else if !result.passes() {
        output.push_str("::warning::Popperian Falsification Check - Grade below threshold\n");
    } else {
        output.push_str(&format!(
            "::notice::Popperian Falsification Check PASSED - Grade: {} ({:.1}%)\n",
            result.grade, result.score
        ));
    }

    output
}
