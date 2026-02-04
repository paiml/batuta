//! Bug Hunter CLI Module
//!
//! Command-line interface for proactive bug hunting.
//!
//! Implements BH-01 to BH-15 from the Popperian Falsification Checklist.

use crate::ansi_colors::Colorize;
use crate::bug_hunter::{
    hunt, hunt_ensemble, hunt_with_spec, hunt_with_ticket, FindingSeverity, HuntConfig, HuntMode,
    HuntResult, SbflFormula,
};
use clap::{Subcommand, ValueEnum};
use std::path::PathBuf;

/// Bug Hunter subcommands.
#[derive(Subcommand, Debug)]
pub enum BugHunterCommand {
    /// LLM-augmented static analysis (LLIFT pattern)
    ///
    /// Runs clippy and pattern detection, optionally filtering with LLM.
    Analyze {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories to scan
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Enable LLM filtering of false positives
        #[arg(long)]
        llm_filter: bool,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Maximum findings to report
        #[arg(long, default_value = "50")]
        max_findings: usize,
    },

    /// SBFL without failing tests (SBEST pattern)
    ///
    /// Uses stack traces or coverage data for fault localization.
    Hunt {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Stack trace file to analyze
        #[arg(long)]
        stack_trace: Option<PathBuf>,

        /// Coverage file (lcov format)
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// SBFL formula to use
        #[arg(long, value_enum, default_value = "ochiai")]
        formula: SbflFormulaArg,

        /// Number of top suspicious locations
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Mutation-based invariant falsification (FDV pattern)
    ///
    /// Identifies mutation testing targets and weak test coverage.
    Falsify {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target files or directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Minimum mutation kill rate to pass
        #[arg(long, default_value = "80")]
        min_kill_rate: u8,

        /// Mutation timeout in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Targeted unsafe Rust fuzzing (FourFuzz pattern)
    ///
    /// Identifies unsafe blocks and generates fuzzing targets.
    Fuzz {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Focus on unsafe blocks only
        #[arg(long)]
        target_unsafe: bool,

        /// Fuzzing duration in seconds (for actual fuzzing)
        #[arg(long, default_value = "60")]
        duration: u64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Hybrid concolic + SBFL (COTTONTAIL pattern)
    ///
    /// Deep analysis of complex conditionals and path coverage.
    DeepHunt {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Coverage file (lcov format)
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// Enable concolic execution analysis
        #[arg(long)]
        concolic: bool,

        /// Use SBFL ensemble
        #[arg(long)]
        sbfl_ensemble: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Run all modes and combine results (ensemble approach)
    Ensemble {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Spec-driven bug hunting (BH-11)
    ///
    /// Hunt bugs guided by a specification file.
    Spec {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Specification file path
        #[arg(long, short = 's')]
        spec: PathBuf,

        /// Filter to specific section (e.g., "Authentication")
        #[arg(long)]
        section: Option<String>,

        /// Update spec with findings (BH-14: Bidirectional Linking)
        #[arg(long)]
        update_spec: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Quick mode: skip clippy, only do pattern matching
        #[arg(long, short = 'q')]
        quick: bool,
    },

    /// Ticket-scoped bug hunting (BH-12)
    ///
    /// Hunt bugs focused on PMAT work ticket areas.
    Ticket {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// PMAT ticket reference (ID or file path)
        #[arg(long, short = 't')]
        ticket: String,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,
    },
}

/// Output format for bug hunter results.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BugHunterOutputFormat {
    /// Human-readable text
    Text,
    /// JSON output
    Json,
    /// SARIF format (Static Analysis Results Interchange Format)
    Sarif,
    /// Markdown table
    Markdown,
}

/// SBFL formula argument.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SbflFormulaArg {
    Tarantula,
    Ochiai,
    Dstar2,
    Dstar3,
}

impl From<SbflFormulaArg> for SbflFormula {
    fn from(arg: SbflFormulaArg) -> Self {
        match arg {
            SbflFormulaArg::Tarantula => SbflFormula::Tarantula,
            SbflFormulaArg::Ochiai => SbflFormula::Ochiai,
            SbflFormulaArg::Dstar2 => SbflFormula::DStar2,
            SbflFormulaArg::Dstar3 => SbflFormula::DStar3,
        }
    }
}

/// Handle bug hunter command.
pub fn handle_bug_hunter_command(command: BugHunterCommand) -> Result<(), String> {
    match command {
        BugHunterCommand::Analyze {
            path,
            target,
            llm_filter,
            min_suspiciousness,
            format,
            max_findings,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Analyze,
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                max_findings,
                llm_filter,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Hunt {
            path,
            stack_trace: _,
            coverage,
            formula,
            top_n,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Hunt,
                targets: vec![PathBuf::from("src")],
                max_findings: top_n,
                sbfl_formula: formula.into(),
                coverage_path: coverage,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Falsify {
            path,
            target,
            min_kill_rate: _,
            timeout,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Falsify,
                targets: target.into_iter().map(PathBuf::from).collect(),
                mutation_timeout_secs: timeout,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Fuzz {
            path,
            target,
            target_unsafe: _,
            duration,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Fuzz,
                targets: target.into_iter().map(PathBuf::from).collect(),
                fuzz_duration_secs: duration,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::DeepHunt {
            path,
            target,
            coverage,
            concolic: _,
            sbfl_ensemble: _,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::DeepHunt,
                targets: target.into_iter().map(PathBuf::from).collect(),
                coverage_path: coverage,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Ensemble {
            path,
            target,
            min_suspiciousness,
            format,
        } => {
            let config = HuntConfig {
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                ..Default::default()
            };
            let result = hunt_ensemble(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Spec {
            path,
            spec,
            section,
            update_spec,
            format,
            min_suspiciousness,
            quick,
        } => {
            let config = HuntConfig {
                min_suspiciousness,
                // Quick mode does pattern-only scan, no clippy/coverage
                mode: if quick { HuntMode::Quick } else { HuntMode::Analyze },
                ..Default::default()
            };
            let (result, mut parsed_spec) =
                hunt_with_spec(&path, &spec, section.as_deref(), config)?;

            output_result(&result, format);

            // Update spec file if requested (BH-14)
            if update_spec {
                let findings_by_claim: Vec<(String, Vec<_>)> = parsed_spec
                    .claims
                    .iter()
                    .map(|c| (c.id.clone(), Vec::new()))
                    .collect();
                if let Ok(updated_content) = parsed_spec.update_with_findings(&findings_by_claim) {
                    if let Err(e) = parsed_spec.write_updated(&updated_content) {
                        eprintln!("Warning: Failed to update spec: {}", e);
                    } else {
                        println!("\nSpec updated: {}", spec.display());
                    }
                }
            }

            Ok(())
        }

        BugHunterCommand::Ticket {
            path,
            ticket,
            format,
            min_suspiciousness,
        } => {
            let config = HuntConfig {
                min_suspiciousness,
                ..Default::default()
            };
            let result = hunt_with_ticket(&path, &ticket, config)?;
            output_result(&result, format);
            Ok(())
        }
    }
}

/// Output result in the specified format.
fn output_result(result: &HuntResult, format: BugHunterOutputFormat) {
    match format {
        BugHunterOutputFormat::Text => output_text(result),
        BugHunterOutputFormat::Json => output_json(result),
        BugHunterOutputFormat::Sarif => output_sarif(result),
        BugHunterOutputFormat::Markdown => output_markdown(result),
    }
}

/// Output as human-readable text.
fn output_text(result: &HuntResult) {
    println!(
        "\n{}\n",
        "🔍 Bug Hunter Report".bright_cyan().bold()
    );

    println!(
        "Mode: {} | Duration: {}ms | Findings: {}\n",
        format!("{}", result.mode).bright_yellow(),
        result.duration_ms,
        result.findings.len()
    );

    // Statistics summary
    println!("{}", "Statistics:".bold());
    for (severity, count) in &result.stats.by_severity {
        let colored = match severity {
            FindingSeverity::Critical => format!("{:?}: {}", severity, count).bright_red(),
            FindingSeverity::High => format!("{:?}: {}", severity, count).red(),
            FindingSeverity::Medium => format!("{:?}: {}", severity, count).yellow(),
            FindingSeverity::Low => format!("{:?}: {}", severity, count).blue(),
            FindingSeverity::Info => format!("{:?}: {}", severity, count).white(),
        };
        println!("  {}", colored);
    }

    println!();

    // Top findings
    let top = result.top_findings(20);
    if top.is_empty() {
        println!("{}", "No findings discovered.".green());
    } else {
        println!("{}", "Top Findings:".bold());
        println!(
            "{:<12} {:<10} {:<12} {:<6} {}",
            "ID", "Severity", "Category", "Score", "Location"
        );
        println!("{}", "-".repeat(80));

        for finding in top {
            let severity_colored = match finding.severity {
                FindingSeverity::Critical => format!("{:?}", finding.severity).bright_red(),
                FindingSeverity::High => format!("{:?}", finding.severity).red(),
                FindingSeverity::Medium => format!("{:?}", finding.severity).yellow(),
                FindingSeverity::Low => format!("{:?}", finding.severity).blue(),
                FindingSeverity::Info => format!("{:?}", finding.severity).white(),
            };

            println!(
                "{:<12} {:<10} {:<12} {:<6.2} {}",
                finding.id,
                severity_colored,
                format!("{:?}", finding.category).chars().take(12).collect::<String>(),
                finding.suspiciousness,
                finding.location()
            );
            println!("             {}", finding.title.bright_white());
            if !finding.description.is_empty() {
                println!("             {}", finding.description.dimmed());
            }
            println!();
        }
    }

    // Summary
    println!("{}", result.summary().bold());
}

/// Output as JSON.
fn output_json(result: &HuntResult) {
    match serde_json::to_string_pretty(result) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing to JSON: {}", e),
    }
}

/// Output as SARIF (Static Analysis Results Interchange Format).
fn output_sarif(result: &HuntResult) {
    let sarif = build_sarif(result);
    match serde_json::to_string_pretty(&sarif) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing SARIF: {}", e),
    }
}

/// Build SARIF structure.
fn build_sarif(result: &HuntResult) -> serde_json::Value {
    let results: Vec<serde_json::Value> = result
        .findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "ruleId": f.id,
                "level": match f.severity {
                    FindingSeverity::Critical | FindingSeverity::High => "error",
                    FindingSeverity::Medium => "warning",
                    FindingSeverity::Low | FindingSeverity::Info => "note",
                },
                "message": {
                    "text": f.title
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f.file.display().to_string()
                        },
                        "region": {
                            "startLine": f.line,
                            "startColumn": f.column.unwrap_or(1)
                        }
                    }
                }],
                "properties": {
                    "suspiciousness": f.suspiciousness,
                    "category": format!("{:?}", f.category),
                    "discoveredBy": format!("{}", f.discovered_by)
                }
            })
        })
        .collect();

    serde_json::json!({
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "batuta bug-hunter",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/paiml/batuta"
                }
            },
            "results": results
        }]
    })
}

/// Output as Markdown.
fn output_markdown(result: &HuntResult) {
    println!("# Bug Hunter Report\n");
    println!("**Mode:** {} | **Duration:** {}ms | **Findings:** {}\n", result.mode, result.duration_ms, result.findings.len());

    println!("## Statistics\n");
    println!("| Severity | Count |");
    println!("|----------|-------|");
    for (severity, count) in &result.stats.by_severity {
        println!("| {:?} | {} |", severity, count);
    }

    println!("\n## Top Findings\n");
    println!("| ID | Severity | Category | Score | Location |");
    println!("|-----|----------|----------|-------|----------|");

    for finding in result.top_findings(20) {
        println!(
            "| {} | {:?} | {:?} | {:.2} | `{}` |",
            finding.id,
            finding.severity,
            finding.category,
            finding.suspiciousness,
            finding.location()
        );
    }

    println!("\n---\n");
    println!("*{}*", result.summary());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbfl_formula_conversion() {
        assert_eq!(SbflFormula::from(SbflFormulaArg::Tarantula), SbflFormula::Tarantula);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Ochiai), SbflFormula::Ochiai);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Dstar2), SbflFormula::DStar2);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Dstar3), SbflFormula::DStar3);
    }

    #[test]
    fn test_handle_analyze_command() {
        let cmd = BugHunterCommand::Analyze {
            path: PathBuf::from("."),
            target: vec!["src".to_string()],
            llm_filter: false,
            min_suspiciousness: 0.5,
            format: BugHunterOutputFormat::Json,
            max_findings: 10,
        };
        // Should not panic
        let _ = handle_bug_hunter_command(cmd);
    }
}
